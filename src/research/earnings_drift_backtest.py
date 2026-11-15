"""Post-earnings drift: do stocks keep moving after a strong earnings reaction?

For every earnings announcement:
  reaction  = stock return over the announcement day and the next session,
              minus SPY over the same two days (robust to before/after-market timing)
  entry     = open of the following session
  exit      = close 20 or 60 sessions later
Events are grouped by EPS surprise and by reaction size; buckets were fixed
in advance. A 20-slot portfolio buys "beat and jumped" events (positive
surprise, reaction above +5%) and holds 60 sessions.

Universes:
  SP500  point-in-time index members, 2013 onwards
  BROAD  today's liquid tickers, 2018 onwards (survivorship-biased)

Run: python src/research/earnings_drift_backtest.py
Needs data/earnings (download_earnings_dates.py) and the price folders.
Costs: 5 bps per side. Prices exclude dividends.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy_families_backtest import slot_sim, perf  # noqa: E402

EARN = ROOT / 'data' / 'earnings'
OUT = ROOT / 'output' / 'research'
COST = 0.0005
SUR_BINS = [-np.inf, -10, 0, 5, 15, np.inf]
SUR_LAB = ['miss >10%', 'miss 0-10%', 'beat 0-5%', 'beat 5-15%', 'beat >15%']
EAR_BINS = [-np.inf, -5, -2, 2, 5, np.inf]
EAR_LAB = ['fell >5%', 'fell 2-5%', 'flat', 'rose 2-5%', 'rose >5%']


def load_px(p):
    d = pd.read_csv(p, parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
    return d[(d[['Open', 'Close']] > 0).all(axis=1)]


def events(folder, start, member=None):
    spy = load_px(folder / 'SPY.csv')
    rows, closes = [], {}
    for p in sorted(folder.glob('*.csv')):
        t = p.stem
        e = EARN / f'{t}.csv'
        if t == 'SPY' or not e.exists():
            continue
        d = load_px(p)
        if len(d) < 300:
            continue
        s = spy.reindex(d.index)
        O, C, V = d['Open'].to_numpy(), d['Close'].to_numpy(), d['Volume'].to_numpy()
        SO, SC = s['Open'].to_numpy(), s['Close'].to_numpy()
        avgvol = pd.Series(V).rolling(20).mean().to_numpy()
        E = pd.read_csv(e, parse_dates=['date'])
        E = E[(E.date >= start) & (E.date <= d.index[-1])]
        idx = d.index.searchsorted(E.date.to_numpy())
        closes[t] = d['Close']
        for i, sur, when in zip(idx, E.surprise_pct.to_numpy(), E.date):
            if i < 21 or i + 2 >= len(d) or (d.index[i] - when).days > 5:
                continue  # no price bar within a few days of the announcement
            day = d.index[i]
            if member is not None:
                if not member(t, day):
                    continue
            elif not (C[i - 1] > 10 and avgvol[i - 1] > 1_000_000):
                continue
            ear = (C[i + 1] / C[i - 1] - 1) - (SC[i + 1] / SC[i - 1] - 1)
            rec = dict(ticker=t, earn_date=day, entry_date=d.index[i + 2], entry=O[i + 2], surprise=sur, ear=ear * 100)
            for h in (20, 60):
                j = i + 2 + h - 1
                if j < len(d):
                    rec[f'ret{h}'] = C[j] / O[i + 2] - 1 - 2 * COST
                    rec[f'x{h}'] = rec[f'ret{h}'] - (SC[j] / SO[i + 2] - 1)
                    rec[f'exit_date{h}'], rec[f'exit{h}'] = d.index[j], C[j]
            rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(closes).sort_index(), spy


def bucket_table(T, col, bins, labels, split):
    T = T.copy()
    T['b'] = pd.cut(T[col], bins, labels=labels)
    rows = []
    for b, g in T.groupby('b', observed=True):
        r = dict(bucket=b, events=len(g))
        for lab, h in (('all', g), ('first half', g[g.entry_date < split]), ('second half', g[g.entry_date >= split])):
            for hz in (20, 60):
                x = h[f'x{hz}'].dropna()
                r[f'{lab} x{hz}'] = round(x.mean() * 100, 2)
        m = g.dropna(subset=['x60']).groupby(g.entry_date.dt.to_period('M')).x60.mean()
        r['t (monthly) x60'] = round(m.mean() / m.std() * np.sqrt(len(m)), 2)
        r['win60 %'] = round((g.ret60.dropna() > 0).mean() * 100, 1)
        rows.append(r)
    return pd.DataFrame(rows).to_string(index=False)


def portfolio(T, C, spy, name, mask):
    P = T[mask].dropna(subset=['exit60']).rename(columns={'exit_date60': 'exit_date', 'exit60': 'exit'})
    r, held = slot_sim(P, C, 20, 'ear', False)
    s = spy['Close'].pct_change().reindex(r.index).fillna(0)
    yr = pd.DataFrame({'strategy': r, 'spy': s})
    y = (1 + yr).groupby(yr.index.year).prod() - 1
    a, b = perf(r), perf(s)
    print(f"{name}: {len(P)} candidate events, avg {held:.0f} of 20 slots used | strategy {a['cagr']}%/yr, maxdd {a['maxdd']}%, "
          f"sharpe {a['sharpe']} | SPY {b['cagr']}%/yr, maxdd {b['maxdd']}% | beat SPY in {int((y.strategy > y.spy).sum())}/{len(y)} years")
    print((y * 100).round(1).T.to_string())


def main():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    M = pd.read_csv(OUT / 'sp500_membership.csv', parse_dates=['start', 'end'])
    iv = {t: list(zip(g.start, g.end.fillna(pd.Timestamp('2100-01-01')))) for t, g in M.groupby('ticker')}
    member = lambda t, day: any(s <= day < e for s, e in iv.get(t, ()))
    for name, folder, start, mem, split in (('SP500 members', ROOT / 'data' / 'daily_sp500', '2013-01-01', member, '2020-01-01'),
                                             ('BROAD liquid (biased list)', ROOT / 'data' / 'daily_long', '2018-01-01', None, '2022-06-01')):
        T, C, spy = events(folder, pd.Timestamp(start), mem)
        T.to_pickle(OUT / f"earnings_events_{name.split()[0]}.pkl")
        print(f'\n================ {name}: {len(T)} events, {T.ticker.nunique()} tickers, '
              f'{T.earn_date.min().date()} to {T.earn_date.max().date()} (halves split at {split})')
        print('columns: average return minus SPY, %, after costs; x20 = 20 sessions, x60 = 60 sessions')
        a = T.dropna(subset=['x60'])
        print(f"ALL EVENTS: x20 {a.x20.mean() * 100:.2f}  x60 {a.x60.mean() * 100:.2f}  win60 {(a.ret60 > 0).mean() * 100:.1f}%")
        print('\nBy EPS surprise\n' + bucket_table(T.dropna(subset=['surprise']), 'surprise', SUR_BINS, SUR_LAB, split))
        print('\nBy two-day reaction\n' + bucket_table(T, 'ear', EAR_BINS, EAR_LAB, split))
        both = (T.surprise > 0) & (T.ear > 5)
        print('\nBeat and jumped (surprise > 0, reaction > +5%)\n' + bucket_table(T[both], 'ear', [-np.inf, np.inf], ['beat+jump'], split))
        print()
        portfolio(T, C, spy, 'Portfolio: beat and jumped, 60 sessions', both)
        portfolio(T, C, spy, 'Portfolio: any reaction > +5%, 60 sessions', T.ear > 5)


if __name__ == '__main__':
    main()
