"""Score my chart-review picks against the tickers I passed on.

Log each pick in picks/my_picks.csv, one row per ticker:
    date,ticker,note
    2026-11-16,UNP,tight base under 305
`date` is the scan date of the Chart_Review file the pick came from. Every
other ticker in output/Chart_Review_<date>.csv counts as a pass for that day.
Log a date with ticker NONE if you reviewed the list and picked nothing.

Run: python src/evaluate_my_picks.py
Outcome for both groups: buy at the next session's open, then
  - 4% target / 3% stop / 5 sessions (stop assumed first on a tie), and
  - the plain 10-session return.
Trades without a full window yet are counted as pending and left out.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY, OUTPUT = ROOT / 'data' / 'daily', ROOT / 'output'
TARGET, STOP, HOLD, LONG_HOLD, COST = 0.04, 0.03, 5, 10, 0.001


def outcome(ticker, date):
    p = DAILY / f'{ticker}.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d.columns = [c.lower() for c in d.columns]
    d['date'] = pd.to_datetime(d[d.columns[0]].astype(str).str[:10])
    d = d[d.date > date].reset_index(drop=True)
    if len(d) < LONG_HOLD:
        return None
    entry = d.open[0]
    stop, target = entry * (1 - STOP), entry * (1 + TARGET)
    bracket = d.close[HOLD - 1] / entry - 1
    for k in range(HOLD):
        if k and d.open[k] <= stop:
            bracket = d.open[k] / entry - 1; break
        if d.low[k] <= stop:
            bracket = -STOP; break
        if k and d.open[k] >= target:
            bracket = d.open[k] / entry - 1; break
        if d.high[k] >= target:
            bracket = TARGET; break
    return bracket - COST, d.close[LONG_HOLD - 1] / entry - 1 - COST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--picks', type=Path, default=ROOT / 'picks' / 'my_picks.csv')
    a = ap.parse_args()
    P = pd.read_csv(a.picks, parse_dates=['date'])
    P['ticker'] = P.ticker.astype(str).str.strip().str.upper()
    if P.empty:
        print(f'No picks logged yet in {a.picks}'); return
    rows, off_list, pending = [], [], 0
    for date, g in P.groupby('date'):
        f = OUTPUT / f'Chart_Review_{date:%Y-%m-%d}.csv'
        if not f.exists():
            print(f'skipped {date:%Y-%m-%d}: no Chart_Review file'); continue
        listed = pd.read_csv(f, encoding='utf-8-sig')['Ticker'].astype(str).str.upper().tolist()
        picked = set(g.ticker) - {'NONE'}
        off_list += [(date.date(), t) for t in picked - set(listed)]
        for t in listed:
            o = outcome(t, date)
            if o is None:
                pending += 1; continue
            rows.append(dict(date=date, ticker=t, picked=t in picked, bracket=o[0], ret10=o[1]))
    if off_list:
        print('picks not on that day\'s list (ignored):', off_list)
    R = pd.DataFrame(rows)
    if R.empty:
        print(f'Nothing resolved yet ({pending} pending).'); return
    print(f'{R.date.nunique()} review days, {int(R.picked.sum())} picks and {int((~R.picked).sum())} passes resolved, {pending} pending\n')
    out = []
    for name, g in (('Picked', R[R.picked]), ('Passed', R[~R.picked])):
        if len(g):
            out.append({'group': name, 'trades': len(g), '4/3/5 win %': round((g.bracket > 0).mean() * 100, 1),
                        '4/3/5 avg %': round(g.bracket.mean() * 100, 2), '10-day win %': round((g.ret10 > 0).mean() * 100, 1),
                        '10-day avg %': round(g.ret10.mean() * 100, 2)})
    print(pd.DataFrame(out).to_string(index=False))
    # Same-day comparison removes the effect of good and bad market days.
    day = R.groupby(['date', 'picked'])[['bracket', 'ret10']].mean().unstack('picked').dropna()
    if len(day) >= 3:
        for col, label in (('bracket', '4/3/5'), ('ret10', '10-day')):
            diff = day[(col, True)] - day[(col, False)]
            t = diff.mean() / diff.std() * np.sqrt(len(diff))
            print(f'\n{label}: picks beat same-day passes by {diff.mean() * 100:+.2f}% on average over {len(diff)} days '
                  f'(t = {t:.2f}; above about 2 is unlikely to be luck)')
    n = int(R.picked.sum())
    if n < 100 or R.date.nunique() < 30:
        print(f'\nToo early to conclude: {n} picks over {R.date.nunique()} days. Aim for 100+ picks across 30+ days.')


if __name__ == '__main__':
    main()
