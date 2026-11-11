"""RSA DAILY + EMA20 Moomoo DRY-RUN V3 bot. Run 08:45-16:10 America/New_York on trading days.
Dependencies: pip install moomoo-api pandas numpy. OpenD must run on 127.0.0.1:11111.
IMPORTANT: Paper only; no live order pathway. Read README before use.
"""
from __future__ import annotations
import csv, json, logging, math, os, time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from moomoo import OpenQuoteContext, OpenSecTradeContext, TrdMarket, TrdEnv, TrdSide, OrderType, RET_OK

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / 'data' / 'daily'
STATE_DIR = ROOT/'rsa_paper_state'; STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR/'state.json'; LOG_FILE=STATE_DIR/'events.csv'
TZ=ZoneInfo('America/New_York'); HOST='127.0.0.1'; PORT=11111
CAPITAL_ASSUMED=15000.0; DOLLARS_PER_TRADE=500.0; MAX_POSITIONS=30
ATR_MULT=1.5; MAX_HOLD_SESSIONS=60; MIN_PRICE=30; MIN_AVG_VOL=1_000_000; MIN_AVG_DOLLAR=100_000_000
RS_THRESHOLD=0.03; POLL_SECONDS=60; MAX_BUY_LIMIT_PREMIUM=0.003; MAX_SELL_LIMIT_DISCOUNT=0.003
# Prevent unattended orders until you have checked account ID, data freshness and order handling.
ENABLE_PAPER_ORDERS=False  # V2 is DRY-RUN ONLY; broker submission is disabled
PAPER_ACCOUNT_ID=1397202  # Set this to the paper account ID returned by get_acc_list().

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def now(): return datetime.now(TZ)
def log(event,ticker='',detail=''):
    with LOG_FILE.open('a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        if f.tell()==0:w.writerow(['timestamp_et','event','ticker','detail'])
        w.writerow([now().isoformat(),event,ticker,str(detail)])
    logging.info('%s %s %s',event,ticker,detail)

def state_load():
    if not STATE_FILE.exists():return {'orders':{},'entries':{},'dry_positions':{},'last_signal_day':'','trading_date':''}
    s=json.loads(STATE_FILE.read_text())
    s.setdefault('orders',{})
    s.setdefault('entries',{})
    s.setdefault('dry_positions',{})
    s.setdefault('last_signal_day','')
    s.setdefault('trading_date','')
    return s
def state_save(s):
    tmp=STATE_FILE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2,default=str));os.replace(tmp,STATE_FILE)

def load_daily(p):
    d=pd.read_csv(p);d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    dc=next((c for c in ('date','datetime','timestamp','time','time_key') if c in d),None)
    if not dc:return None
    for c in ('open','high','low','close','volume'):
        if c not in d:return None
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['date']=pd.to_datetime(d[dc],utc=True,errors='coerce').dt.tz_localize(None).dt.normalize()
    d=d.dropna(subset=['date','open','high','low','close','volume']).drop_duplicates('date',keep='last').sort_values('date')
    d=d[(d[['open','high','low','close']]>0).all(axis=1)&(d.high>=d[['open','close','low']].max(axis=1))&(d.low<=d[['open','close']].min(axis=1))]
    return d.set_index('date') if len(d)>=75 else None

def add_indicators(d):
    d=d.copy()
    d['ema20']=d.close.ewm(span=20,adjust=False).mean();d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    prev=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    d['atr14']=tr.rolling(14,min_periods=14).mean()
    d['avgvol']=d.volume.rolling(20,min_periods=20).mean()
    d['avgdollar']=(d.close*d.volume).rolling(20,min_periods=20).mean()
    d['ret20']=d.close/d.close.shift(20)-1
    return d

def paths():
    out={}
    for p in sorted(DAILY_DIR.glob('*.csv')):
        t=p.stem.upper().removeprefix('US.').removesuffix('.US')
        for suffix in ('_DAILY','_5M'):
            if t.endswith(suffix):t=t[:-len(suffix)]
        out.setdefault(t,p)
    return out

def candidates_and_indicators():
    pths=paths()
    if 'SPY' not in pths:raise RuntimeError('SPY.csv is required for relative strength')
    spy_raw=load_daily(pths['SPY'])
    if spy_raw is None:raise RuntimeError('Invalid or insufficient SPY daily data')
    spy=add_indicators(spy_raw)
    # Previous session: use SPY last completed date; never use a current-day partial daily candle.
    cutoff=pd.Timestamp(now().date())-pd.Timedelta(days=1)
    spy=spy.loc[:cutoff]
    if len(spy)<75:raise RuntimeError('Insufficient SPY history')
    signal_day=spy.index[-1]
    # Freshness is checked separately for NEW entries; stale data must not disable stop monitoring.
    spyret=spy.close/spy.close.shift(20)-1
    found=[]; indicator_cache={};bad=0
    for t,p in pths.items():
        try:
            raw=load_daily(p)
            if raw is None:continue
            d=add_indicators(raw.loc[:signal_day]);indicator_cache[t]=d
            if signal_day not in d.index or len(d)<75:continue
            x=d.iloc[-1];y=d.iloc[-2]
            if not np.isfinite([x.atr14,x.ema20,x.ema50,x.avgvol,x.avgdollar,x.ret20,spyret.loc[signal_day]]).all():continue
            if x.close<MIN_PRICE or x.avgvol<MIN_AVG_VOL or x.avgdollar<MIN_AVG_DOLLAR or x.atr14<=0:continue
            trend=x.close>x.ema50 and x.ema20>x.ema50 and x.ema50>d.ema50.iloc[-6]
            pullback=y.low<=y.ema20*1.01 and x.close>x.open and x.close>x.ema20 and x.close>y.close and x.low>=x.ema20*.97
            rs=x.ret20-spyret.loc[signal_day]
            if trend and pullback and rs>RS_THRESHOLD:found.append((t,float(rs),float(x.atr14)))
        except Exception as e:
            bad+=1;log('DATA_ERROR',t,str(e)[:180])
    found.sort(key=lambda x:(-x[1],x[0]))
    log('SCAN',detail=f'signal_day={signal_day.date()} candidates={len(found)} errors={bad}')
    return signal_day,found,indicator_cache

def must_ok(result,action):
    rc,df=result
    if rc!=RET_OK:raise RuntimeError(f'{action} failed: {df}')
    return df

def get_positions(trade,account):
    df=must_ok(trade.position_list_query(acc_id=account,trd_env=TrdEnv.SIMULATE),'position_list_query')
    result={}
    for _,r in df.iterrows():
        code=str(r.get('code',''))
        if code.startswith('US.') and float(r.get('qty',0))>0:
            result[code[3:]]=dict(qty=int(float(r.qty)),cost=float(r.get('cost_price',0)))
    return result

def get_pending(trade,account):
    df=must_ok(trade.order_list_query(acc_id=account,trd_env=TrdEnv.SIMULATE),'order_list_query')
    pending=set()
    for _,r in df.iterrows():
        status=str(r.get('order_status','')).upper()
        if any(k in status for k in ('SUBMIT','WAIT','PART','PROCESS','PENDING')):
            code=str(r.get('code',''))
            if code.startswith('US.'):pending.add(code[3:])
    return pending

def snapshot(quote,ticker):
    df=must_ok(quote.get_market_snapshot([f'US.{ticker}']),'get_market_snapshot')
    if df.empty:raise RuntimeError(f'No quote for {ticker}')
    r=df.iloc[0]
    price=float(r.get('last_price',0))
    if not np.isfinite(price) or price<=0:raise RuntimeError(f'Invalid price for {ticker}')
    # Quotes may be delayed. Verify subscription/quote permissions before enabling orders.
    return price


# V3 retains the existing state.json and events.csv paths. Do not run older versions simultaneously.
# A paper position is an assumed snapshot fill; no actual orders are submitted.
# Stop triggers use snapshot LAST price, not an exchange-native stop order.
# Delayed quotes and prices moving between polls can cause missed triggers.
HEARTBEAT_SECONDS=300
MAX_SNAPSHOT_AGE_SECONDS=180

def previous_weekday(day):
    # Weekend-aware, not a full exchange holiday calendar. On holidays, fail closed on new entries.
    d=day-timedelta(days=1)
    while d.weekday()>=5:d-=timedelta(days=1)
    return d

def quote_info(quote,ticker):
    df=must_ok(quote.get_market_snapshot([f'US.{ticker}']),'get_market_snapshot')
    if df.empty:raise RuntimeError('Empty snapshot')
    r=df.iloc[0]
    price=float(r.get('last_price',0))
    if not math.isfinite(price) or price<=0:raise RuntimeError(f'Invalid snapshot price: {price}')
    # Some Moomoo snapshots have update_time (HH:MM:SS) but no date. Do not
    # pretend to validate freshness without a reliable date+time pair.
    update_time=str(r.get('update_time',''))
    update_date=str(r.get('update_date',''))
    if update_date and update_time and update_date.lower()!='nan' and update_time.lower()!='nan':
        try:
            stamp=pd.Timestamp(update_date+' '+update_time)
            if stamp.tzinfo is None:stamp=stamp.tz_localize(TZ)
            else:stamp=stamp.tz_convert(TZ)
            age=(now()-stamp.to_pydatetime()).total_seconds()
            if age>MAX_SNAPSHOT_AGE_SECONDS or age < -120:
                raise RuntimeError(f'Stale/future quote: age_seconds={age:.0f}')
        except (ValueError,TypeError) as e:
            raise RuntimeError(f'Cannot parse quote timestamp: {e}') from e
    return price,update_date,update_time

def position_stop(p):
    return float(p['entry_price'])-ATR_MULT*float(p['atr'])

def emit_exit(s,ticker,p,reason,observed):
    today=now().date().isoformat()
    key=f'{today}:{ticker}:SELL'
    if key in s['orders']:
        log('DUPLICATE_EXIT_BLOCK',ticker,key)
        return
    qty=int(p['qty']);entry=float(p['entry_price']);stop=position_stop(p)
    pnl=(observed-entry)*qty
    # Record exit and remove position in one atomic state update.
    s['orders'][key]={'status':'DRY_RUN_EXIT','reason':reason,
        'time':now().isoformat(),'price':observed,'qty':qty,
        'entry_price':entry,'stop':stop,'estimated_pnl':round(pnl,2)}
    del s['dry_positions'][ticker]
    state_save(s)
    log('DRY_EXIT_SIGNAL',ticker,
        f'{reason} qty={qty} entry={entry:.4f} observed_exit={observed:.4f} '
        f'stop={stop:.4f} estimated_pnl={pnl:.2f}')

def monitor_one(s,quote,ticker,p,cache,clock):
    stop=position_stop(p)
    # Snapshot first: stop monitoring remains active throughout market hours.
    price,update_date,update_time=quote_info(quote,ticker)
    p['last_price']=price
    p['last_checked']=now().isoformat()
    p['stop_price']=stop
    p['quote_update_date']=update_date
    p['quote_update_time']=update_time
    if price<=stop:
        log('DRY_STOP_TRIGGER',ticker,f'last={price:.4f} stop={stop:.4f}')
        emit_exit(s,ticker,p,f'STOP_TRIGGER_{stop:.2f}',price)
        return

    # Evaluate completed DAILY bars once per date, at ANY time after startup.
    # Missing/stale data must not generate a trend/time exit.
    today=now().date().isoformat()
    if p.get('last_daily_exit_check_day')==today:
        return
    d=cache.get(ticker)
    if d is None or len(d)<3:
        log('DAILY_EXIT_DATA_UNAVAILABLE',ticker,'No sufficient cached daily bars')
        p['last_daily_exit_check_day']=today
        return
    last_bar=d.index[-1].date()
    expected=previous_weekday(now().date())
    if last_bar!=expected:
        log('DAILY_EXIT_DATA_STALE',ticker,
            f'last_bar={last_bar} expected={expected}; no EMA/time exit')
        p['last_daily_exit_check_day']=today
        return

    entry_date=pd.Timestamp(p['entry_date'])
    # Never count bars preceding the entry or its opening day.
    completed=d.loc[d.index>entry_date]
    sessions=len(completed)
    below=False
    if sessions>=2:
        last_two=completed.iloc[-2:]
        below=bool((last_two['close']<last_two['ema20']).all())
    else:
        last_two=completed.iloc[-2:]
    details=(f'completed_sessions={sessions} last_bar={last_bar} '
             f'last_two_close_ema20='+','.join(
                 f'{float(r.close):.4f}/{float(r.ema20):.4f}'
                 for _,r in last_two.iterrows())+
             f' below_two={below} max_hold={MAX_HOLD_SESSIONS}')
    log('DAILY_EXIT_CHECK',ticker,details)
    p['last_daily_exit_check_day']=today
    if below:
        emit_exit(s,ticker,p,'EMA20_TWO_CLOSES',price)
    elif sessions>=MAX_HOLD_SESSIONS:
        emit_exit(s,ticker,p,'MAX_HOLD_60',price)

def main():
    if ENABLE_PAPER_ORDERS:
        raise RuntimeError('V3 is dry-run only. Keep ENABLE_PAPER_ORDERS=False.')
    if not DAILY_DIR.exists():raise FileNotFoundError(DAILY_DIR)
    if now().weekday()>=5:raise RuntimeError('Run on a trading weekday')
    s=state_load()
    quote=None
    try:
        # Scan failures must NOT prevent monitoring existing virtual positions.
        candidates=[];cache={};signal_day=None;fresh=False
        try:
            signal_day,candidates,cache=candidates_and_indicators()
            fresh=(signal_day.date()==previous_weekday(now().date()))
            if not fresh:
                log('STALE_SIGNALS_ENTRY_BLOCK',detail=
                    f'signal_day={signal_day.date()} expected={previous_weekday(now().date())}')
            s['last_signal_day']=str(signal_day.date());state_save(s)
        except Exception as e:
            log('SCAN_ERROR_ENTRIES_DISABLED',detail=str(e)[:300])
        quote=OpenQuoteContext(host=HOST,port=PORT)
        log('START',detail=f'DRY_RUN_ONLY candidates={len(candidates)} fresh={fresh} '
            f'carried_positions={len(s["dry_positions"])}')
        last_heartbeat=0.0
        while now().time()<dtime(16,0):
            clock=now().time()
            if clock<dtime(9,31):time.sleep(15);continue
            # Always check existing positions, even if scan data is stale or missing.
            for ticker in list(s['dry_positions']):
                try:
                    monitor_one(s,quote,ticker,s['dry_positions'][ticker],cache,clock)
                except Exception as e:
                    log('POSITION_CHECK_ERROR',ticker,str(e)[:300])
            state_save(s)
            if time.monotonic()-last_heartbeat>=HEARTBEAT_SECONDS:
                for ticker,p in s['dry_positions'].items():
                    log('POSITION_HEARTBEAT',ticker,
                        f'entry={float(p["entry_price"]):.4f} stop={position_stop(p):.4f} '
                        f'last={p.get("last_price","UNKNOWN")} '
                        f'checked={p.get("last_checked","NEVER")}')
                last_heartbeat=time.monotonic()
            # New entries only in the opening window with previous-session data.
            if fresh and dtime(9,31)<=clock<dtime(9,45):
                for ticker,score,atr in candidates:
                    try:
                        if len(s['dry_positions'])>=MAX_POSITIONS:break
                        if ticker in s['dry_positions'] or ticker in s['entries']:continue
                        today=now().date().isoformat();key=f'{today}:{ticker}:BUY'
                        if key in s['orders']:continue
                        price,qd,qt=quote_info(quote,ticker)
                        qty=math.floor(DOLLARS_PER_TRADE/(price*(1+MAX_BUY_LIMIT_PREMIUM)))
                        if qty<1:continue
                        existing=sum(int(v['qty'])*float(v['entry_price'])
                                     for v in s['dry_positions'].values())
                        if existing+qty*price>CAPITAL_ASSUMED:continue
                        # Persist position and buy intent together, before logging.
                        s['orders'][key]={'status':'DRY_RUN','reason':'RSA_DAILY_ENTRY',
                            'time':now().isoformat(),'price':price,'qty':qty}
                        s['dry_positions'][ticker]={'entry_date':today,'entry_price':price,
                            'atr':atr,'qty':qty,'stop_price':price-ATR_MULT*atr}
                        state_save(s)
                        log('DRY_ENTRY_SIGNAL',ticker,
                            f'qty={qty} assumed_fill={price:.4f} atr={atr:.4f} '
                            f'stop={price-ATR_MULT*atr:.4f}')
                    except Exception as e:
                        log('ENTRY_ERROR',ticker,str(e)[:300])
            time.sleep(POLL_SECONDS)
    finally:
        if quote is not None:
            try:quote.close()
            except Exception:pass
        log('STOPPED',detail=f'open_dry_positions={len(s["dry_positions"])}')

if __name__=='__main__':main()
