"""RSA DAILY + EMA20 Moomoo PAPER bot. Run 08:45-16:10 America/New_York on trading days.
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
ENABLE_PAPER_ORDERS=False
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
    if not STATE_FILE.exists():return {'orders':{},'entries':{},'last_signal_day':'','trading_date':''}
    return json.loads(STATE_FILE.read_text())
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
    spy=add_indicators(load_daily(pths['SPY']))
    # Previous session: use SPY last completed date; never use a current-day partial daily candle.
    cutoff=pd.Timestamp(now().date())-pd.Timedelta(days=1)
    spy=spy.loc[:cutoff]
    if len(spy)<75:raise RuntimeError('Insufficient SPY history')
    signal_day=spy.index[-1]
    if (pd.Timestamp(now().date())-signal_day).days>4:raise RuntimeError(f'Stale SPY daily data: {signal_day.date()}')
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

def submit(trade,account,side,ticker,qty,price,reason,s):
    key=f'{now().date()}:{ticker}:{side}'
    if key in s['orders']:
        log('DUPLICATE_BLOCK',ticker,key);return False
    # Write reservation BEFORE network request. If request times out, manually reconcile broker orders.
    s['orders'][key]={'status':'RESERVED','reason':reason,'time':now().isoformat()};state_save(s)
    log('ORDER_INTENT',ticker,f'{side} {qty} @ {price:.2f} {reason}')
    if not ENABLE_PAPER_ORDERS:
        s['orders'][key]['status']='DRY_RUN';state_save(s);return False
    try:
        result=trade.place_order(price=round(price,2),qty=qty,code=f'US.{ticker}',
            trd_side=TrdSide.BUY if side=='BUY' else TrdSide.SELL,
            order_type=OrderType.NORMAL,trd_env=TrdEnv.SIMULATE,acc_id=account)
        df=must_ok(result,'place_order')
        s['orders'][key]['status']='SUBMITTED';s['orders'][key]['broker_result']=str(df.to_dict('records'))[:1200]
        state_save(s);log('ORDER_SUBMITTED',ticker,side)
        return True
    except Exception as e:
        s['orders'][key]['status']='UNKNOWN_RECONCILE';state_save(s)
        log('ORDER_UNKNOWN_RECONCILE',ticker,str(e));return False

def main():
    if not DAILY_DIR.exists():raise FileNotFoundError(DAILY_DIR)
    if now().weekday()>=5:raise RuntimeError('Run on a US trading day (holiday calendar not included)')
    s=state_load()
    signal_day,candidates,cache=candidates_and_indicators()
    if s.get('last_signal_day')!=str(signal_day.date()):
        s['last_signal_day']=str(signal_day.date());state_save(s)
    quote=OpenQuoteContext(host=HOST,port=PORT)
    trade=OpenSecTradeContext(host=HOST,port=PORT,filter_trdmarket=TrdMarket.US)
    try:
        accounts=must_ok(trade.get_acc_list(),'get_acc_list')
        log('ACCOUNTS',detail=str(accounts[['acc_id','trd_env']].to_dict('records')) if {'acc_id','trd_env'}.issubset(accounts.columns) else str(accounts))
        if PAPER_ACCOUNT_ID is None:
            log('SETUP_REQUIRED',detail='Set PAPER_ACCOUNT_ID to your SIMULATE account; no orders will be sent')
            return
        account=PAPER_ACCOUNT_ID
        allowed=accounts[(accounts.acc_id.astype(str)==str(account)) & (accounts.trd_env.astype(str).str.upper().str.contains('SIMULATE'))]
        if allowed.empty:raise RuntimeError('Configured account is not a verified SIMULATE account')
        log('START',detail=f'PAPER_ONLY enable_orders={ENABLE_PAPER_ORDERS}, candidates={len(candidates)}')
        today=str(now().date())
        # Entry and trend-exit actions are intentionally limited to opening window.
        # After the opening window, continue stop monitoring until market close.
        while now().time()<dtime(16,10):
            clock=now().time()
            if clock<dtime(9,31):time.sleep(15);continue
            if clock>=dtime(16,0):break
            try:
                positions=get_positions(trade,account);pending=get_pending(trade,account)
                # Exits first. EMA20 trend signals use two completed prior-session closes.
                for ticker,pos in positions.items():
                    if ticker in pending:continue
                    d=cache.get(ticker)
                    if d is None or len(d)<3:continue
                    ent=s['entries'].get(ticker)
                    if ent is None:
                        log('UNTRACKED_POSITION',ticker,'No locally confirmed entry; skip automated exits');continue
                    atr=float(ent['atr']);entry=float(ent['entry_price'])
                    stop=entry-ATR_MULT*atr
                    close_below=(d.close.iloc[-1]<d.ema20.iloc[-1] and d.close.iloc[-2]<d.ema20.iloc[-2])
                    sessions=int((d.index>=pd.Timestamp(ent['entry_date'])).sum())
                    reason=None
                    if close_below and clock<dtime(9,45):reason='EMA20_TWO_CLOSES'
                    elif sessions>=MAX_HOLD_SESSIONS and clock<dtime(9,45):reason='MAX_HOLD_60'
                    else:
                        last=snapshot(quote,ticker)
                        if last<=stop:reason=f'STOP_TRIGGER_{stop:.2f}'
                    if reason:
                        last=snapshot(quote,ticker)
                        limit=max(.01,round(last*(1-MAX_SELL_LIMIT_DISCOUNT),2))
                        submit(trade,account,'SELL',ticker,pos['qty'],limit,reason,s)
                if dtime(9,31)<=clock<dtime(9,45):
                    for ticker,score,atr in candidates:
                        if len(positions)+len(pending)>=MAX_POSITIONS:break
                        if ticker in positions or ticker in pending or ticker in s['entries']:continue
                        if f'{today}:{ticker}:BUY' in s['orders']:continue
                        last=snapshot(quote,ticker)
                        qty=math.floor(DOLLARS_PER_TRADE/(last*(1+MAX_BUY_LIMIT_PREMIUM)))
                        if qty<1:continue
                        limit=round(last*(1+MAX_BUY_LIMIT_PREMIUM),2)
                        # Capital checks must use actual buying power before enabling orders.
                        # For now, limit aggregate notional to configured paper budget.
                        existing=sum(v['qty']*v['cost'] for v in positions.values())
                        if existing+qty*limit>CAPITAL_ASSUMED:continue
                        if submit(trade,account,'BUY',ticker,qty,limit,'RSA_DAILY_ENTRY',s):
                            pending.add(ticker)
                # Reconcile actual filled buys. Do not assume submitted == filled.
                latest=get_positions(trade,account)
                for ticker,pos in latest.items():
                    if ticker in s['entries']:continue
                    key=f'{today}:{ticker}:BUY'
                    if s['orders'].get(key,{}).get('status')=='SUBMITTED':
                        atr=next((a for t,_,a in candidates if t==ticker),None)
                        if atr:
                            s['entries'][ticker]={'entry_date':today,'entry_price':pos['cost'],'atr':atr}
                            state_save(s);log('ENTRY_CONFIRMED',ticker,str(s['entries'][ticker]))
                # On a fully closed position, remove tracked entry so future signals can re-enter.
                for ticker in list(s['entries']):
                    if ticker not in latest and f'{today}:{ticker}:SELL' in s['orders']:
                        del s['entries'][ticker];state_save(s);log('EXIT_CONFIRMED',ticker)
            except Exception as e:
                log('LOOP_ERROR',detail=str(e))
            time.sleep(POLL_SECONDS)
    finally:
        trade.close();quote.close();log('STOPPED')

if __name__=='__main__':main()
