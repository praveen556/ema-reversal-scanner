"""Standalone, no-order historical RSA / EMA scanner comparison.

Run: python compare_rsa_ema_backtest.py
Dependencies: pip install pandas numpy openpyxl
All paths can be overridden with --root / --output.

No broker connectivity. Source files are read-only. Results are written under
output/rsa_ema_comparison/. Backtest is restricted to sessions for which both
prior-day scanner files and next-day 5-minute bars exist.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(r'C:\Users\prave\PycharmProjects\ema-reversal-scanner')
STRATEGIES = ('A_RSA', 'B_RSA_AND_SCANNER', 'C_SCANNER', 'D_COMBINED_INTRADAY',
              'E_RSA_SPY_ABOVE_EMA20', 'F_RSA_SPY_ABOVE_EMA200',
              'G_COMBINED_SPY_ABOVE_EMA20')
MIN_PRICE, MIN_VOL, MIN_DOLLAR = 30., 1_000_000., 100_000_000.
CAPITAL, ALLOCATION, MAX_POSITIONS = 15_000., 500., 30
ATR_MULT, MAX_HOLD = 1.5, 60
RS_THRESHOLD = .03
# Conservative but adjustable transaction cost assumptions (per side).
SLIPPAGE_BPS, COMMISSION_PER_SHARE = 5., 0.
TZ = ZoneInfo('America/New_York')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--end', default='2026-11-05', help='Last execution session (YYYY-MM-DD)')
    p.add_argument('--start', default='2026-09-28', help='First execution session')
    p.add_argument('--slippage-bps', type=float, default=SLIPPAGE_BPS)
    p.add_argument('--allocation', type=float, default=ALLOCATION)
    p.add_argument('--capital', type=float, default=CAPITAL)
    p.add_argument('--max-positions', type=int, default=MAX_POSITIONS)
    p.add_argument('--skip-intraday', action='store_true', help='Disable strategy D')
    return p.parse_args()


def sym_from_path(p):
    t=p.stem.upper().removeprefix('US.').removesuffix('.US')
    for suffix in ('_DAILY','_5M','_5MIN','_HOURLY','_1H'):
        if t.endswith(suffix): t=t[:-len(suffix)]
    return t


def index_files(folder):
    out={}
    for p in sorted(folder.glob('*.csv')):
        out.setdefault(sym_from_path(p),p)
    return out


def load_ohlcv(path):
    df=pd.read_csv(path,low_memory=False)
    df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
    datecol=next((c for c in ('datetime','timestamp','date','time','time_key') if c in df),None)
    if datecol is None or any(c not in df for c in ('open','high','low','close','volume')):
        raise ValueError(f'Missing OHLCV/date columns: {path.name}')
    for c in ('open','high','low','close','volume'):
        df[c]=pd.to_numeric(df[c],errors='coerce')
    # Date-only daily data must not be shifted across calendar dates by UTC parsing.
    s=df[datecol].astype(str).str.strip()
    if s.str.match(r'^\d{4}-\d{2}-\d{2}(?:[ T]00:00:00)?$').all():
        dates=pd.to_datetime(s,errors='coerce')
    else:
        parsed=pd.to_datetime(s,utc=True,errors='coerce')
        dates=parsed.dt.tz_convert(TZ)
    df['ts']=dates
    df=df.dropna(subset=['ts','open','high','low','close','volume'])
    df=df[(df[['open','high','low','close']]>0).all(axis=1)]
    df=df[(df.high>=df[['open','close','low']].max(axis=1)) & (df.low<=df[['open','close']].min(axis=1))]
    return df.drop_duplicates('ts',keep='last').sort_values('ts').set_index('ts')[['open','high','low','close','volume']]


def daily_indicators(df):
    d=df.copy()
    # Use trading dates, never assume a calendar-day gap is a missing candle.
    if getattr(d.index,'tz',None) is not None:
        d.index=d.index.tz_localize(None).normalize()
    else: d.index=d.index.normalize()
    d=d[~d.index.duplicated(keep='last')]
    d['ema20']=d.close.ewm(span=20,adjust=False).mean()
    d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    d['ema200']=d.close.ewm(span=200,adjust=False).mean()
    prev=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    d['atr14']=tr.rolling(14,min_periods=14).mean()
    d['avgvol']=d.volume.rolling(20,min_periods=20).mean()
    d['avgdollar']=(d.close*d.volume).rolling(20,min_periods=20).mean()
    d['ret20']=d.close/d.close.shift(20)-1
    return d


def scanner_files(output, prefix):
    out={}
    for p in output.glob(prefix+'*.csv'):
        m=re.search(r'(\d{4}-\d{2}-\d{2})',p.stem)
        if m: out[pd.Timestamp(m.group(1))]=p
    return out


def scanner_candidates(p):
    df=pd.read_csv(p,low_memory=False)
    df.columns=[c.strip() for c in df.columns]
    if 'Ticker' not in df: raise ValueError(f'No Ticker column: {p}')
    if 'Chart Review' in df:
        df=df[df['Chart Review'].astype(str).str.strip().str.upper().isin(('YES','TRUE','1'))]
    df=df.dropna(subset=['Ticker']).drop_duplicates('Ticker')
    out={}
    for _,r in df.iterrows():
        ticker=str(r['Ticker']).strip().upper().removeprefix('US.')
        if not ticker or ticker=='NAN': continue
        bucket=str(r.get('Bucket','')).strip()
        stage=str(r.get('Latest Entry Stage','')).strip()
        persistence=pd.to_numeric(r.get('Persistence %',np.nan),errors='coerce')
        out[ticker]={'bucket':bucket,'stage':stage,
                     'persistence':float(persistence) if pd.notna(persistence) else None}
    return out


def scanner_rank(meta):
    bucket=meta.get('bucket','').upper()
    stage=meta.get('stage','').upper()
    priority=3 if 'CONFIRM' in bucket else 2 if 'IMPROV' in bucket else 1 if 'DEVELOP' in bucket else 0
    priority+=2 if 'CONFIRM' in stage else 0
    return (priority,meta.get('persistence') or 0)



def spy_regime(spy, signal_date):
    """Market conditions known at signal-day close, before next-day entry."""
    prior=spy.loc[:signal_date]
    if len(prior)<201:
        return {'regime':'INSUFFICIENT_HISTORY','above_ema20':False,
                'above_ema200':False,'bullish_20_50':False,'spy_close':None,
                'spy_ema20':None,'spy_ema50':None,'spy_ema200':None,
                'spy_ret20_pct':None,'spy_atr_pct':None}
    r=prior.iloc[-1]
    prev=prior.iloc[-6]
    above20=bool(r.close>r.ema20)
    above200=bool(r.close>r.ema200)
    bullish=bool(above20 and r.close>r.ema50 and r.ema20>prev.ema20
                 and r.ema50>prev.ema50)
    regime=('BULLISH_TREND' if bullish else
            'BELOW_EMA200' if not above200 else
            'BELOW_EMA20' if not above20 else 'MIXED_ABOVE_EMA20')
    return {'regime':regime,'above_ema20':above20,'above_ema200':above200,
            'bullish_20_50':bullish,'spy_close':round(float(r.close),4),
            'spy_ema20':round(float(r.ema20),4),
            'spy_ema50':round(float(r.ema50),4),
            'spy_ema200':round(float(r.ema200),4),
            'spy_ret20_pct':round(float(r.ret20)*100,3),
            'spy_atr_pct':round(float(r.atr14/r.close)*100,3)}


def signal_for(d, spy, signal_date):
    if signal_date not in d.index or signal_date not in spy.index:return None
    a=d.loc[:signal_date]
    if len(a)<75 or len(spy.loc[:signal_date])<75:return None
    x,y=a.iloc[-1],a.iloc[-2]
    benchmark=spy.loc[signal_date,'ret20']
    nums=[x.atr14,x.ema20,x.ema50,x.avgvol,x.avgdollar,x.ret20,benchmark]
    if not np.isfinite(nums).all():return None
    liquid=bool(x.close>=MIN_PRICE and x.avgvol>=MIN_VOL and x.avgdollar>=MIN_DOLLAR and x.atr14>0)
    trend=bool(x.close>x.ema50 and x.ema20>x.ema50 and x.ema50>a.ema50.iloc[-6])
    pullback=bool(y.low<=y.ema20*1.01 and x.close>x.open and x.close>x.ema20
                  and x.close>y.close and x.low>=x.ema20*.97)
    rs=float(x.ret20-benchmark)
    return {'liquid':liquid,'rsa':bool(liquid and trend and pullback and rs>RS_THRESHOLD),
            'rs':rs,'atr':float(x.atr14),'close':float(x.close),
            'ema20':float(x.ema20),'trend':trend,'pullback':pullback,
            'avgvol':float(x.avgvol),'avgdollar':float(x.avgdollar)}


class FiveMinute:
    def __init__(self,paths): self.paths=paths;self.cache={};self.failures=collections.Counter()
    def get(self,ticker):
        if ticker not in self.cache:
            p=self.paths.get(ticker)
            if p is None:self.failures['missing_file']+=1;self.cache[ticker]=None
            else:
                try:
                    d=load_ohlcv(p)
                    if getattr(d.index,'tz',None) is None:
                        d.index=d.index.tz_localize(TZ)
                    self.cache[ticker]=d.between_time('09:30','15:55')
                except Exception as exc:
                    self.failures[f'bad_file:{type(exc).__name__}']+=1
                    self.cache[ticker]=None
        return self.cache[ticker]
    def day(self,ticker,date):
        d=self.get(ticker)
        if d is None:return None
        day=d.loc[d.index.date==date.date()]
        return day if len(day) else None


def entry_bar(day,intraday=False):
    # 9:30 bar contains trades before 9:31. Earliest defensible bar-open is 9:35.
    if not intraday:
        x=day.between_time('09:35','09:35')
        return x.iloc[0] if len(x) else None, x.index[0] if len(x) else None
    # Opening range: 09:30..09:55 bars; confirmation 10:00..10:55 close
    # above OR high; execute NEXT 5-minute bar OPEN, no same-bar lookahead.
    opening=day.between_time('09:30','09:55')
    if len(opening)<6:return None,None
    or_high=float(opening.high.max())
    for ts,r in day.between_time('10:00','10:55').iterrows():
        if r.close>or_high:
            later=day.loc[day.index>ts]
            if len(later):return later.iloc[0],later.index[0]
            break
    return None,None


def cost_fill(price,side,bps):
    return float(price)*(1+bps/10000 if side=='BUY' else 1-bps/10000)


def exit_position(book,ticker,ts,raw_price,reason,bps,trades):
    p=book['positions'].pop(ticker)
    fill=cost_fill(raw_price,'SELL',bps)
    proceeds=p['qty']*fill-p['qty']*COMMISSION_PER_SHARE
    pnl=proceeds-p['cost']
    book['cash']+=proceeds
    trades.append({'strategy':book['name'],'ticker':ticker,'signal_date':str(p['signal_date'].date()),
        'entry_time':str(p['entry_ts']),'exit_time':str(ts),'entry_price':round(p['entry'],4),
        'exit_price':round(fill,4),'qty':p['qty'],'pnl':round(pnl,2),
        'return_pct':round(100*pnl/p['cost'],3),'holding_sessions':p['sessions'],
        'exit_reason':reason,'bucket':p['bucket'],'stage':p['stage'],
        'rs':round(p['rs'],4),'atr':round(p['atr'],4),
        'spy_regime_at_entry':p.get('spy_regime',''),
        'spy_above_ema20_at_entry':p.get('spy_above_ema20'),
        'spy_above_ema200_at_entry':p.get('spy_above_ema200'),
        'spy_bullish_20_50_at_entry':p.get('spy_bullish_20_50'),
        'mfe_pct':round(p['mfe']*100,3),
        'mae_pct':round(p['mae']*100,3)})


def equity(book,prices):
    return book['cash']+sum(p['qty']*prices.get(t,p.get('last_mark',p['entry'])) for t,p in book['positions'].items())


def run():
    args=parse_args()
    root=args.root; output=root/'output'; daily_dir=root/'data'/'daily'; five_dir=root/'data'/'5min'
    dest=args.output or output/'rsa_ema_comparison';dest.mkdir(parents=True,exist_ok=True)
    for folder in (daily_dir,five_dir,output):
        if not folder.is_dir():raise FileNotFoundError(f'Missing directory: {folder}')
    daily_paths=index_files(daily_dir);five_paths=index_files(five_dir)
    if 'SPY' not in daily_paths:raise RuntimeError('SPY daily CSV required')
    spy=daily_indicators(load_ohlcv(daily_paths['SPY']))
    sessions=spy.index[(spy.index>=pd.Timestamp(args.start)) & (spy.index<=pd.Timestamp(args.end))]
    if not len(sessions):raise RuntimeError('No SPY sessions in requested range')
    charts=scanner_files(output,'Chart_Review_')
    history=scanner_files(output,'Setup_History_')
    daily={};invalid=[]
    for ticker,path in daily_paths.items():
        try:daily[ticker]=daily_indicators(load_ohlcv(path))
        except Exception as exc:invalid.append((ticker,str(exc)[:150]))
    bars=FiveMinute(five_paths)
    books={name:{'name':name,'cash':args.capital,'positions':{}} for name in STRATEGIES
           if not(args.skip_intraday and name=='D_COMBINED_INTRADAY')}
    trades=[];candidates=[];skips=[];equities=[];market_rows=[]
    # All exit decisions based on PREVIOUS completed session only.
    all_sessions=list(spy.index)
    for session in sessions:
        idx=all_sessions.index(session)
        if idx<1:continue
        prior=all_sessions[idx-1]
        if prior not in charts:
            skips.append({'session':str(session.date()),'ticker':'','reason':'NO_PRIOR_CHART_REVIEW','detail':str(prior.date())})
            continue
        if prior not in history:
            skips.append({'session':str(session.date()),'ticker':'','reason':'NO_PRIOR_SETUP_HISTORY','detail':str(prior.date())})
            continue
        try:scan=scanner_candidates(charts[prior])
        except Exception as exc:
            skips.append({'session':str(session.date()),'ticker':'','reason':'BAD_SCANNER_FILE','detail':str(exc)[:180]})
            continue
        market=spy_regime(spy,prior)
        market_rows.append({'session':str(session.date()),'signal_date':str(prior.date()),**market})
        # Precompute signals from prior completed day, including existing positions.
        eligible={}
        for ticker,d in daily.items():
            sig=signal_for(d,spy,prior)
            if sig is not None:eligible[ticker]=sig
        for name,book in books.items():
            positions=book['positions']
            # Manage existing positions, using 5m bars to monitor intraday stop.
            for ticker in list(positions):
                p=positions[ticker]
                day=bars.day(ticker,session)
                if day is None or day.empty:
                    skips.append({'session':str(session.date()),'ticker':ticker,'reason':'MISSING_5MIN_FOR_OPEN_POSITION','detail':name})
                    continue
                p['sessions']+=1
                p['last_mark']=float(day.close.iloc[-1])
                d=daily.get(ticker)
                # Trend and max hold exits at first eligible 09:35 open.
                reason=None
                if d is not None and prior in d.index:
                    old=d.loc[:prior]
                    if len(old)>=2 and old.close.iloc[-1]<old.ema20.iloc[-1] and old.close.iloc[-2]<old.ema20.iloc[-2]:
                        reason='EMA20_TWO_CLOSES'
                if reason is None and p['sessions']>=MAX_HOLD:reason='MAX_HOLD_60'
                early,early_ts=entry_bar(day)
                if early is None:
                    skips.append({'session':str(session.date()),'ticker':ticker,'reason':'NO_0935_BAR','detail':name});continue
                if reason:
                    exit_position(book,ticker,early_ts,float(early.open),reason,args.slippage_bps,trades)
                    continue
                # For stop gaps, fill at next observed bar open (worse than stop).
                # If intrabar low crosses stop, assume stop filled at stop level.
                for ts,r in day.iterrows():
                    if ts.time()<dt.time(9,35):continue
                    p['mfe']=max(p['mfe'],float(r.high)/p['entry']-1)
                    p['mae']=min(p['mae'],float(r.low)/p['entry']-1)
                    if r.open<=p['stop']:
                        exit_position(book,ticker,ts,float(r.open),'STOP_GAP',args.slippage_bps,trades);break
                    if r.low<=p['stop']:
                        exit_position(book,ticker,ts,p['stop'],'ATR_STOP',args.slippage_bps,trades);break
            selected=[]
            for ticker,sig in eligible.items():
                meta=scan.get(ticker)
                if name in ('A_RSA','E_RSA_SPY_ABOVE_EMA20','F_RSA_SPY_ABOVE_EMA200'):
                    ok=sig['rsa']
                elif name in ('B_RSA_AND_SCANNER','D_COMBINED_INTRADAY','G_COMBINED_SPY_ABOVE_EMA20'):
                    ok=sig['rsa'] and meta is not None
                else:
                    ok=sig['liquid'] and meta is not None
                if ok:
                    selected.append((ticker,sig,meta or {}))
            selected.sort(key=lambda t:(-t[1]['rs'],t[0]) if name in ('A_RSA','B_RSA_AND_SCANNER','E_RSA_SPY_ABOVE_EMA20','F_RSA_SPY_ABOVE_EMA200','G_COMBINED_SPY_ABOVE_EMA20')
                          else (-scanner_rank(t[2])[0],-scanner_rank(t[2])[1],-t[1]['rs'],t[0]))
            for rank,(ticker,sig,meta) in enumerate(selected,1):
                row={'session':str(session.date()),'signal_date':str(prior.date()),'strategy':name,
                     'rank':rank,'ticker':ticker,'relative_strength_pct':round(sig['rs']*100,3),
                     'atr14':round(sig['atr'],4),'bucket':meta.get('bucket',''),
                     'stage':meta.get('stage',''),'persistence_pct':meta.get('persistence'),
                     'spy_regime':market['regime'],
                     'spy_above_ema20':market['above_ema20'],
                     'spy_above_ema200':market['above_ema200'],
                     'spy_bullish_20_50':market['bullish_20_50'],
                     'selected':False,'reason':''}
                if name in ('E_RSA_SPY_ABOVE_EMA20','G_COMBINED_SPY_ABOVE_EMA20') and not market['above_ema20']:
                    row['reason']='SPY_BELOW_EMA20'
                elif name=='F_RSA_SPY_ABOVE_EMA200' and not market['above_ema200']:
                    row['reason']='SPY_BELOW_EMA200'
                elif ticker in positions:row['reason']='ALREADY_OPEN'
                elif len(positions)>=args.max_positions:row['reason']='MAX_POSITIONS'
                else:
                    day=bars.day(ticker,session)
                    if day is None:row['reason']='NO_5MIN_DATA'
                    else:
                        bar,ts=entry_bar(day,name=='D_COMBINED_INTRADAY')
                        if bar is None:row['reason']='NO_ENTRY_BAR_OR_CONFIRMATION'
                        else:
                            fill=cost_fill(bar.open,'BUY',args.slippage_bps)
                            qty=math.floor(args.allocation/(fill+COMMISSION_PER_SHARE))
                            cost=qty*(fill+COMMISSION_PER_SHARE)
                            if qty<1:row['reason']='PRICE_EXCEEDS_ALLOCATION'
                            elif cost>book['cash']:row['reason']='INSUFFICIENT_CASH'
                            else:
                                book['cash']-=cost
                                p={'entry':fill,'entry_ts':ts,'signal_date':prior,'qty':qty,'cost':cost,
                                   'atr':sig['atr'],'stop':fill-ATR_MULT*sig['atr'],'sessions':0,
                                   'bucket':meta.get('bucket',''),'stage':meta.get('stage',''),
                                   'rs':sig['rs'],'mfe':0.,'mae':0.,'last_mark':float(day.close.iloc[-1]),
                                   'spy_regime':market['regime'],
                                   'spy_above_ema20':market['above_ema20'],
                                   'spy_above_ema200':market['above_ema200'],
                                   'spy_bullish_20_50':market['bullish_20_50']}
                                positions[ticker]=p;row['selected']=True;row['reason']='ENTERED'
                                # Check stop from entry candle onward. Intrabar order unknown:
                                # use stop conservatively if entry candle low crosses it.
                                after=day.loc[day.index>=ts]
                                for bt,r in after.iterrows():
                                    p['mfe']=max(p['mfe'],float(r.high)/fill-1)
                                    p['mae']=min(p['mae'],float(r.low)/fill-1)
                                    if r.open<=p['stop']:
                                        exit_position(book,ticker,bt,float(r.open),'STOP_GAP',args.slippage_bps,trades);break
                                    if r.low<=p['stop']:
                                        exit_position(book,ticker,bt,p['stop'],'ATR_STOP',args.slippage_bps,trades);break
                candidates.append(row)
            marks={}
            for ticker in positions:
                day=bars.day(ticker,session)
                if day is not None and len(day):marks[ticker]=float(day.close.iloc[-1])
            equities.append({'session':str(session.date()),'strategy':name,
                'equity':round(equity(book,marks),2),'cash':round(book['cash'],2),'open_positions':len(positions)})
        print(f'{session.date()}: candidates={sum(c["session"]==str(session.date()) for c in candidates)}',flush=True)
    open_rows=[]
    for name,book in books.items():
        for ticker,p in book['positions'].items():
            lastday=bars.day(ticker,sessions[-1]);mark=float(lastday.close.iloc[-1]) if lastday is not None and len(lastday) else p.get('last_mark',p['entry'])
            open_rows.append({'strategy':name,'ticker':ticker,'entry_time':str(p['entry_ts']),
                'entry_price':round(p['entry'],4),'qty':p['qty'],'mark_price':round(mark,4),
                'unrealized_pnl':round(p['qty']*(mark-p['entry'])-p['qty']*COMMISSION_PER_SHARE,2),
                'stop':round(p['stop'],4),'stage':p['stage'],'bucket':p['bucket'],
                'spy_regime_at_entry':p.get('spy_regime','')})
    tdf=pd.DataFrame(trades);edf=pd.DataFrame(equities);summary=[]
    for name,book in books.items():
        sub=tdf[tdf.strategy==name] if len(tdf) else pd.DataFrame()
        curve=edf[edf.strategy==name].equity if len(edf) else pd.Series(dtype=float)
        gains=sub[sub.pnl>0].pnl.sum() if len(sub) else 0.
        losses=-sub[sub.pnl<0].pnl.sum() if len(sub) else 0.
        drawdown=(curve/curve.cummax()-1).min()*100 if len(curve) else 0.
        end_equity=float(curve.iloc[-1]) if len(curve) else args.capital
        summary.append({'strategy':name,'closed_trades':len(sub),'wins':int((sub.pnl>0).sum()) if len(sub) else 0,
            'win_rate_pct':round(100*(sub.pnl>0).mean(),2) if len(sub) else None,
            'realized_pnl':round(float(sub.pnl.sum()),2) if len(sub) else 0.,
            'avg_trade_pnl':round(float(sub.pnl.mean()),2) if len(sub) else None,
            'profit_factor':round(float(gains/losses),3) if losses>0 else None,
            'open_positions':len(book['positions']),'ending_equity_marked':round(end_equity,2),
            'return_pct':round(100*(end_equity/args.capital-1),3),
            'max_drawdown_pct':round(float(drawdown),3)})
    stage=[]
    if len(tdf):
        for (strategy,bucket),g in tdf.groupby(['strategy','bucket'],dropna=False):
            stage.append({'strategy':strategy,'bucket':bucket,'trades':len(g),
                'win_rate_pct':round(100*(g.pnl>0).mean(),2),'total_pnl':round(g.pnl.sum(),2),
                'avg_pnl':round(g.pnl.mean(),2)})
    # Regime attribution is based on the market state at ENTRY, not exit.
    regime_results=[]
    if len(tdf):
        for (strategy,regime),g in tdf.groupby(['strategy','spy_regime_at_entry'],dropna=False):
            regime_results.append({'strategy':strategy,'spy_regime_at_entry':regime,
                'closed_trades':len(g),'wins':int((g.pnl>0).sum()),
                'win_rate_pct':round(100*(g.pnl>0).mean(),2),
                'realized_pnl':round(float(g.pnl.sum()),2),
                'avg_trade_pnl':round(float(g.pnl.mean()),2),
                'atr_stop_or_gap_count':int(g.exit_reason.isin(['ATR_STOP','STOP_GAP']).sum())})
    regime_counts=[]
    if candidates:
        cdf=pd.DataFrame(candidates)
        for (strategy,regime),g in cdf.groupby(['strategy','spy_regime']):
            regime_counts.append({'strategy':strategy,'spy_regime':regime,
                'qualifying_candidates':len(g),'entries':int(g.selected.sum()),
                'filtered_by_spy':int(g.reason.isin(['SPY_BELOW_EMA20','SPY_BELOW_EMA200']).sum())})
    reports={'Summary':pd.DataFrame(summary),'Trades':tdf,'Open_Positions':pd.DataFrame(open_rows),
             'Daily_Candidates':pd.DataFrame(candidates),'Equity_Curve':edf,
             'Stage_Results':pd.DataFrame(stage),'Data_Issues':pd.DataFrame(skips),
             'Invalid_Daily':pd.DataFrame(invalid,columns=['ticker','error']),
             'SPY_Market_Regime':pd.DataFrame(market_rows),
             'Regime_Trade_Results':pd.DataFrame(regime_results),
             'Regime_Candidate_Counts':pd.DataFrame(regime_counts)}
    for name,df in reports.items():df.to_csv(dest/(name+'.csv'),index=False)
    # Excel workbook is convenience output. CSV files are always generated even if
    # Excel writer dependencies are missing.
    try:
        with pd.ExcelWriter(dest/'RSA_EMA_Backtest_Comparison.xlsx',engine='openpyxl') as writer:
            for name,df in reports.items():
                df.to_excel(writer,sheet_name=name,index=False)
                ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
                from openpyxl.styles import Font,PatternFill,Alignment
                for cell in ws[1]:
                    cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='17365D')
                    cell.alignment=Alignment(wrap_text=True)
                for col in ws.columns:
                    letter=col[0].column_letter
                    width=min(45,max(12,max(len(str(c.value or '')) for c in list(col)[:min(150,len(col))])+2))
                    ws.column_dimensions[letter].width=width
    except (ImportError,ModuleNotFoundError) as exc:
        print('Excel skipped (install openpyxl):',exc)
    settings={'root':str(root),'start':args.start,'end':args.end,'capital':args.capital,
        'allocation':args.allocation,'slippage_bps_per_side':args.slippage_bps,
        'commission_per_share':COMMISSION_PER_SHARE,'entry':'09:35 next-day bar open; strategy D first-hour breakout next bar open',
        'stop':'entry - 1.5*ATR14; check 5-minute lows, gaps at bar open',
        'exit':'two completed daily closes below EMA20, or 60 held sessions, at next 09:35 bar open',
        'note':'Historical simulation only; 5m bars cannot resolve intrabar ordering; incomplete ticker coverage biases comparisons.',
        'missing_5min_counts':dict(bars.failures),
        'spy_filter_rules':{'E':'RSA baseline, enter only if prior close SPY > EMA20',
            'F':'RSA baseline, enter only if prior close SPY > EMA200',
            'G':'RSA+scanner, enter only if prior close SPY > EMA20'},
        'spy_regime_rule':'BULLISH_TREND: SPY close > EMA20 and EMA50, both EMA20 and EMA50 rising over 5 sessions; BELOW_EMA200; BELOW_EMA20; otherwise MIXED_ABOVE_EMA20',
        'spy_filter_applies_to':'new entries only; existing positions retain original exit rules'}
    (dest/'Run_Settings.json').write_text(json.dumps(settings,indent=2))
    print('\nDONE. Reports:',dest)
    print(pd.DataFrame(summary).to_string(index=False))
    print('Data issues:',len(skips),'Invalid daily:',len(invalid))

if __name__=='__main__':run()
