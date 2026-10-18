"""Compare EMA reversal confirmation streaks with local daily OHLC data.
Run: python test_confirmation_sequence.py
Dependencies: pip install pandas openpyxl xlsxwriter
"""
from pathlib import Path
import re, time
import pandas as pd

# Place script in project root alongside output/ and data/.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'output'
DATA = ROOT / 'data/daily'
REPORT = OUTPUT / 'EMA_Confirmation_Sequence_Test.xlsx'
TARGET, STOP, HOLD = .04, .03, 5
BUCKET = 'STABLE CONFIRMED'
STAGE = 'CONFIRMED NEAR ENTRY'
STATUS = 'SETUP'
# Use False to count all scanner signals, True to suppress overlapping positions per ticker.
NO_OVERLAPPING_TRADES = False


def key(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')


def col(df, name):
    return next((c for c in df.columns if key(c) == key(name)), None)


def get(df, name, default=''):
    c = col(df, name)
    return df[c] if c is not None else pd.Series(default, index=df.index)


def date_from_name(name):
    m = re.search(r'20\d{2}[-_]\d{2}[-_]\d{2}|20\d{6}', name)
    return pd.to_datetime(m.group().replace('_', '-'), errors='coerce') if m else pd.NaT


def streak(sequence):
    # Exact stage tokens separated by arrows, NOT substring matching.
    tokens = [s.strip().upper() for s in re.split(r'\s*(?:->|→)\s*', str(sequence)) if s.strip()]
    n = 0
    for token in reversed(tokens):
        if token == STAGE:
            n += 1
        else:
            break
    return n


def read_signals():
    frames = []
    files = sorted(p for p in OUTPUT.rglob('Chart_Review*') if p.suffix.lower() in ('.csv', '.xlsx'))
    print(f'Found {len(files)} Chart_Review files', flush=True)
    for p in files:
        try:
            df = pd.read_csv(p, encoding='utf-8-sig', low_memory=False) if p.suffix.lower()=='.csv' else pd.read_excel(p)
            if col(df,'Ticker') is None:
                print('Skipped missing ticker:',p.name); continue
            dt = date_from_name(p.name)
            if pd.isna(dt):
                print('Skipped missing filename date:',p.name); continue
            df = df.copy()
            df['signal_date'] = dt
            df['symbol'] = get(df,'Ticker').astype(str).str.strip().str.upper()
            df['bucket_normalized'] = get(df,'Bucket').astype(str).str.strip().str.upper()
            df['stage_normalized'] = get(df,'Latest Entry Stage').astype(str).str.strip().str.upper()
            df['status_normalized'] = get(df,'Latest Status').astype(str).str.strip().str.upper()
            df['sequence'] = get(df,'Stage Sequence').astype(str)
            df['confirmation_streak'] = df['sequence'].map(streak)
            df['source_file'] = p.name
            frames.append(df)
        except Exception as exc:
            print('Skipped',p.name,exc)
    if not frames: raise RuntimeError('No readable Chart_Review files under output/')
    df = pd.concat(frames,ignore_index=True)
    df = df[(df.bucket_normalized==BUCKET)&(df.stage_normalized==STAGE)&(df.status_normalized==STATUS)].copy()
    df = df.drop_duplicates(['symbol','signal_date']).sort_values(['signal_date','symbol']).reset_index(drop=True)
    print(f'Filtered {len(df)} signals from {df.symbol.nunique()} tickers',flush=True)
    return df


def load_quotes(symbols):
    quotes = {}
    for i,sym in enumerate(sorted(symbols),1):
        paths = [p for p in DATA.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv','.xlsx') and p.stem.upper()==sym]
        for p in paths:
            try:
                q = pd.read_csv(p,low_memory=False) if p.suffix.lower()=='.csv' else pd.read_excel(p)
                needed = {c:col(q,c) for c in ['Date','Open','High','Low','Close']}
                if any(v is None for v in needed.values()): continue
                q = q.rename(columns={v:k.lower() for k,v in needed.items()})
                q['date']=pd.to_datetime(q['date'],errors='coerce').dt.normalize()
                for c in ['open','high','low','close']:
                    q[c]=pd.to_numeric(q[c],errors='coerce')
                q=q.dropna(subset=['date','open','high','low','close'])
                q=q[(q[['open','high','low','close']]>0).all(axis=1)]
                q=q.sort_values('date').drop_duplicates('date',keep='last')
                if not q.empty: quotes[sym]=q.reset_index(drop=True)
                break
            except Exception as exc:
                print('Quote error',p.name,exc)
        if i%50==0 or i==len(symbols): print(f'Quote lookup {i}/{len(symbols)}',flush=True)
    return quotes


def simulate(row, quotes):
    out={'entry_date':pd.NaT,'entry_price':float('nan'),'exit_date':pd.NaT,'exit_price':float('nan'),
         'result':'NO_QUOTES','exit_reason':'','return_pct':float('nan'),'bars_held':float('nan'),
         'ambiguous_bar':False}
    q=quotes.get(row.symbol)
    if q is None:
        out['exit_reason']='No ticker quote file';return out
    future=q.loc[q.date>row.signal_date]
    if future.empty:
        out.update(result='PENDING',exit_reason='No next-day quotes');return out
    first=future.iloc[0]
    if (first.date-row.signal_date).days>5:
        out.update(result='INSUFFICIENT_DATA',exit_reason='Next trading day likely missing');return out
    entry=float(first.open)
    out.update(entry_date=first.date,entry_price=entry)
    target=entry*(1+TARGET);stop=entry*(1-STOP)
    bars=future.head(HOLD)
    for n,bar in enumerate(bars.itertuples(index=False),1):
        exitprice=None;reason=None
        if n>1 and bar.open<=stop: exitprice=float(bar.open);reason='GAP_STOP'
        elif n>1 and bar.open>=target: exitprice=float(bar.open);reason='GAP_TARGET'
        elif bar.low<=stop:
            exitprice=stop;reason='STOP_FIRST_AMBIGUOUS' if bar.high>=target else 'STOP'
            out['ambiguous_bar']=(bar.high>=target)
        elif bar.high>=target: exitprice=target;reason='TARGET'
        if reason:
            out.update(exit_date=bar.date,exit_price=exitprice,exit_reason=reason,bars_held=n,
                       result='SUCCESS' if reason in ('TARGET','GAP_TARGET') else 'FAILED')
            break
    if out['exit_reason']=='':
        if len(bars)<HOLD:
            out.update(result='PENDING',exit_reason='Less than five future bars')
        else:
            bar=bars.iloc[-1]
            price=float(bar.close)
            out.update(exit_date=bar.date,exit_price=price,exit_reason='TIME_EXIT',bars_held=HOLD,
                       result='TIME_EXIT_GAIN' if price>entry else ('TIME_EXIT_LOSS' if price<entry else 'TIME_EXIT_FLAT'))
    if pd.notna(out['exit_price']): out['return_pct']=round((out['exit_price']/entry-1)*100,4)
    return out


RESOLVED={'SUCCESS','FAILED','TIME_EXIT_GAIN','TIME_EXIT_LOSS','TIME_EXIT_FLAT'}
def summarize(df, label):
    d=df[df.result.isin(RESOLVED)]
    n=len(d)
    return {'Group':label,'Signals':len(df),'Resolved':n,'Pending / no quotes':len(df)-n,
            'Profitable':int(d.result.isin(['SUCCESS','TIME_EXIT_GAIN']).sum()),
            'Profitable %':round(100*d.result.isin(['SUCCESS','TIME_EXIT_GAIN']).mean(),2) if n else None,
            'Stops':int((d.result=='FAILED').sum()),
            'Stop %':round(100*(d.result=='FAILED').mean(),2) if n else None,
            'Target hits':int((d.result=='SUCCESS').sum()),
            'Target %':round(100*(d.result=='SUCCESS').mean(),2) if n else None,
            'Avg return %':round(d.return_pct.mean(),3) if n else None,
            'Median return %':round(d.return_pct.median(),3) if n else None}


def main():
    t=time.perf_counter()
    if not OUTPUT.is_dir() or not DATA.is_dir():
        raise RuntimeError('Place script in project root with output/ and data/ folders')
    signals=read_signals()
    if signals.empty: raise RuntimeError('No signals match selected Bucket + Stage + SETUP filters')
    quotes=load_quotes(set(signals.symbol))
    print(f'Found quotes for {len(quotes)}/{signals.symbol.nunique()} tickers',flush=True)
    evaluations=pd.DataFrame([simulate(r,quotes) for r in signals.itertuples(index=False)])
    df=pd.concat([signals.reset_index(drop=True),evaluations],axis=1)
    if NO_OVERLAPPING_TRADES:
        kept=[];active={}
        for idx,r in df.sort_values(['signal_date','symbol']).iterrows():
            # Pending signals reserve the full 5-session window if possible; conservative.
            last=active.get(r.symbol)
            if last is not None and r.signal_date<=last: continue
            kept.append(idx)
            if pd.notna(r.entry_date):
                q=quotes.get(r.symbol)
                future=q[q.date>=r.entry_date] if q is not None else None
                expiry=future.iloc[min(HOLD,len(future))-1].date if future is not None and len(future) else r.entry_date
                active[r.symbol]=r.exit_date if pd.notna(r.exit_date) else expiry
        df=df.loc[kept].copy()
    groups=[summarize(df,'All matching SETUP signals')]
    for n in (1,2,3,4,5):
        groups.append(summarize(df[df.confirmation_streak>=n],f'At least {n} consecutive confirmations'))
    for n in sorted(df.confirmation_streak.unique()):
        groups.append(summarize(df[df.confirmation_streak==n],f'Exactly {n} consecutive confirmations'))
    matrix=pd.DataFrame(groups)
    details=df[['signal_date','symbol','bucket_normalized','stage_normalized','status_normalized','sequence',
                'confirmation_streak','source_file','entry_date','entry_price','exit_date','exit_price',
                'result','exit_reason','return_pct','bars_held','ambiguous_bar']].copy()
    pending=details[~details.result.isin(RESOLVED)]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(REPORT,engine='xlsxwriter',datetime_format='yyyy-mm-dd') as writer:
        for name,frame in [('Sequence Comparison',matrix),('All Filtered Signals',details),
                           ('Resolved Signals',details[details.result.isin(RESOLVED)]),
                           ('Pending Missing',pending)]:
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name];sheet.freeze_panes(1,0)
            sheet.autofilter(0,0,len(frame),len(frame.columns)-1)
            for i,c in enumerate(frame.columns):sheet.set_column(i,i,min(55,max(14,len(str(c))+3)))
    print('\n'+matrix.to_string(index=False))
    print(f'\nSaved: {REPORT} | Runtime: {time.perf_counter()-t:.1f}s')
    if len(pending):print(f'Warning: {len(pending)} signals are pending/missing; not counted as wins or losses.')

if __name__=='__main__':main()
