"""Compare exits on identical 3-confirmation SETUP signals.
Place alongside test_confirmation_sequence.py in the project root.
Run: python compare_trailing_stops.py
Requires pandas and xlsxwriter (pip install pandas xlsxwriter).
Uses regular-session daily OHLC; ambiguous intraday events are treated conservatively.
"""
from pathlib import Path
import pandas as pd
import test_confirmation_sequence_gap_aware as base

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'output' / 'EMA_Trailing_Stop_Comparison.xlsx'
MIN_STREAK = 3
NO_OVERLAPPING_TRADES = True
HOLD = 5
STOP = 0.03
TARGET = 0.04

# TRAIL_3: trail stop 3% below highest observed price, never lower than initial stop.
# STEP: at +2% move stop to breakeven; at +4% move to +1%; at +6% to +3%; at +8% to +5%.
STRATEGIES = ['FIXED_4_3', 'FIXED_STOP_NO_TARGET', 'TRAIL_3', 'STEP']
RESOLVED = {'TARGET','STOP','GAP_STOP','GAP_TARGET','TIME_EXIT'}


def adjusted_stop(mode, entry, highwater, prior_stop):
    if mode == 'TRAIL_3':
        return max(prior_stop, highwater * 0.97)
    if mode == 'STEP':
        gain = highwater / entry - 1
        if gain >= 0.08: return max(prior_stop, entry * 1.05)
        if gain >= 0.06: return max(prior_stop, entry * 1.03)
        if gain >= 0.04: return max(prior_stop, entry * 1.01)
        if gain >= 0.02: return max(prior_stop, entry)
    return prior_stop


def simulate(q, signal_date, mode):
    out = dict(result='PENDING', exit_reason='NO_NEXT_BAR', entry_date=pd.NaT,
               entry_price=float('nan'), exit_date=pd.NaT, exit_price=float('nan'),
               return_pct=float('nan'), bars_held=float('nan'), ambiguous=False,
               stop_at_exit=float('nan'))
    if q is None or q.empty:
        out.update(result='NO_QUOTES', exit_reason='NO_QUOTES'); return out
    future = q[q.date > signal_date]
    if future.empty: return out
    if (future.iloc[0].date - signal_date).days > 5:
        out.update(result='INSUFFICIENT_DATA',exit_reason='MISSING_ENTRY_SESSION');return out
    entry = float(future.iloc[0].open)
    stop = entry * (1-STOP)
    target = entry * (1+TARGET)
    highwater = entry
    out.update(entry_date=future.iloc[0].date, entry_price=entry)
    bars = future.head(HOLD)
    for n, bar in enumerate(bars.itertuples(index=False),1):
        op, hi, lo = float(bar.open), float(bar.high), float(bar.low)
        # Gap exits use the OPEN: never assume the stop price fills through a gap.
        if n > 1 and op <= stop:
            out.update(result='GAP_STOP',exit_reason='GAP_STOP',exit_date=bar.date,exit_price=op,bars_held=n,stop_at_exit=stop)
            break
        if mode == 'FIXED_4_3' and n > 1 and op >= target:
            out.update(result='GAP_TARGET',exit_reason='GAP_TARGET',exit_date=bar.date,exit_price=op,bars_held=n,stop_at_exit=stop)
            break
        # The stop active at the OPEN is checked before any same-bar high is used.
        old_stop = stop
        if lo <= old_stop:
            ambiguous = mode == 'FIXED_4_3' and hi >= target
            out.update(result='STOP',exit_reason='STOP_FIRST_AMBIGUOUS' if ambiguous else 'STOP',
                       ambiguous=ambiguous,exit_date=bar.date,exit_price=old_stop,bars_held=n,stop_at_exit=old_stop)
            break
        if mode == 'FIXED_4_3' and hi >= target:
            out.update(result='TARGET',exit_reason='TARGET',exit_date=bar.date,exit_price=target,bars_held=n,stop_at_exit=old_stop)
            break
        # New trailing level may have been touched after the high in this bar.
        # Conservative daily OHLC assumption: if the new stop is inside the day's
        # range, assume the low happened BEFORE the high, so DO NOT claim a fill.
        # Activate the raised stop on the NEXT bar only. This is deliberately
        # conservative and avoids using unknown intraday high/low ordering.
        highwater = max(highwater, hi)
        stop = adjusted_stop(mode, entry, highwater, stop)
    if pd.isna(out['exit_date']):
        if len(bars) < HOLD:
            out.update(result='PENDING',exit_reason='INSUFFICIENT_FUTURE_BARS')
        else:
            last=bars.iloc[-1]
            out.update(result='TIME_EXIT',exit_reason='TIME_EXIT',exit_date=last.date,
                       exit_price=float(last.close),bars_held=HOLD,stop_at_exit=stop)
    if pd.notna(out['exit_price']):
        out['return_pct'] = round((out['exit_price']/entry-1)*100,4)
    return out


def select_nonoverlapping(signals, quotes):
    """Choose entries in signal-date order; block same ticker until its 5th session.
    Uses a COMMON selection for all strategies, avoiding strategy-dependent samples.
    """
    selected=[]; blocked={}
    for row in signals.sort_values(['signal_date','symbol']).itertuples(index=False):
        q=quotes.get(row.symbol)
        if q is None: selected.append(row);continue
        future=q[q.date>row.signal_date]
        if future.empty: selected.append(row);continue
        entry_date=future.iloc[0].date
        if row.symbol in blocked and entry_date <= blocked[row.symbol]: continue
        selected.append(row)
        if len(future)>=HOLD: blocked[row.symbol]=future.iloc[HOLD-1].date
        else: blocked[row.symbol]=future.iloc[-1].date
    return selected


def main():
    signals=base.read_signals()
    signals=signals[signals.confirmation_streak>=MIN_STREAK].copy()
    quotes=base.load_quotes(set(signals.symbol))
    rows=list(signals.itertuples(index=False))
    if NO_OVERLAPPING_TRADES: rows=select_nonoverlapping(signals,quotes)
    records=[]
    for r in rows:
        for mode in STRATEGIES:
            result=simulate(quotes.get(r.symbol),r.signal_date,mode)
            records.append(dict(symbol=r.symbol,signal_date=r.signal_date,
                                confirmation_streak=r.confirmation_streak,strategy=mode,**result))
    df=pd.DataFrame(records)
    summary=[]
    for mode, group in df.groupby('strategy',sort=False):
        resolved=group[group.result.isin(RESOLVED)]
        returns=resolved.return_pct
        summary.append(dict(strategy=mode,signals=len(group),resolved=len(resolved),
            pending_missing=len(group)-len(resolved),profitable_exit_pct=round(100*(returns>0).mean(),2) if len(resolved) else None,
            avg_return_pct=round(returns.mean(),3) if len(resolved) else None,
            median_return_pct=round(returns.median(),3) if len(resolved) else None,
            stop_exit_pct=round(100*resolved.result.isin(['STOP','GAP_STOP']).mean(),2) if len(resolved) else None,
            worst_return_pct=round(returns.min(),3) if len(resolved) else None,
            ambiguous_bars=int(group.ambiguous.sum())))
    summary=pd.DataFrame(summary)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(REPORT,engine='xlsxwriter',datetime_format='yyyy-mm-dd') as writer:
        for name, frame in [('Comparison',summary),('Trade Details',df),('Unresolved',df[~df.result.isin(RESOLVED)])]:
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name];sheet.freeze_panes(1,0)
            sheet.autofilter(0,0,max(len(frame),1),max(len(frame.columns)-1,0))
            for i,c in enumerate(frame.columns):sheet.set_column(i,i,min(38,max(14,len(str(c))+2)))
    print(summary.to_string(index=False))
    print(f'\nSaved {REPORT}')
    print('NOTE: daily OHLC cannot establish same-bar ordering; trailing stop updates take effect NEXT session.')

if __name__=='__main__': main()
