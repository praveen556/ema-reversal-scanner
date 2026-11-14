"""Textbook strategy families on long daily history. No orders, no tuning.

Parameters are the standard published ones, fixed before looking at results:
  MOM     12-1 month momentum, top decile, monthly rebalance
  MOM_T   same, in cash while SPY closed below its 200-day average
  REV     weekly: bottom decile of 5-day return among stocks above 200-day avg
  RSI2    close above 200-day avg and RSI(2) < 10; buy next open; sell at the
          first close above the 5-day avg, or after 10 sessions
  BRK     new 252-day closing high on volume > 1.5x average; buy next open;
          sell at the close 20 sessions later
Benchmarks: SPY (price only), SPY with 200-day timing, equal-weight universe.

Run: python src/research/strategy_families_backtest.py
Reads data/daily_long, writes output/research/families_*.csv

Limits: today's tickers only (survivorship: compare with the equal-weight
universe, not just SPY); portfolio weights take effect one full day after the
signal; 5 bps cost per side; no dividends anywhere.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / 'data' / 'daily_long'
OUT = ROOT / 'output' / 'research'
COST = 0.0005
SPLIT = '2022-01-01'


def perf(r):
    r = r.dropna()
    if len(r) < 50:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return dict(cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), vol=round(r.std() * np.sqrt(252) * 100, 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2), maxdd=round((eq / eq.cummax() - 1).min() * 100, 1))


def report(name, r, ew):
    row = {'strategy': name}
    for lab, s in (('all', r), ('2017-21', r[:SPLIT]), ('2022-26', r[SPLIT:])):
        p = perf(s)
        b = perf(ew.reindex(s.index))
        row.update({f'{lab} cagr': p['cagr'], f'{lab} vs univ': round(p['cagr'] - b['cagr'], 1)})
        if lab == 'all':
            row.update({'sharpe': p['sharpe'], 'maxdd': p['maxdd']})
    return row


def rebalance_returns(W, ret):
    """W has target weights on rebalance rows (NaN rows elsewhere)."""
    W = W.ffill().fillna(0)
    turn = W.diff().abs().sum(axis=1)
    return (W.shift(2) * ret).sum(axis=1) - turn.shift(2).fillna(0) * COST


def slot_sim(trades, px, slots, rank_col, ascending):
    T = trades.sort_values(['entry_date', rank_col], ascending=[True, ascending])
    by = {k: g for k, g in T.groupby('entry_date')}
    cash, held, out = 1.0, {}, []
    for day in px.index[px.index >= T.entry_date.min()]:
        row = px.loc[day]
        mark = lambda: cash + sum(v[0] * (row[k] if row[k] == row[k] else v[3]) for k, v in held.items())
        equity = mark()
        for r in by.get(day, T.iloc[0:0]).itertuples():
            if len(held) >= slots:
                break
            if r.ticker in held:
                continue
            size = min(equity / slots, cash)
            if size <= 0:
                break
            held[r.ticker] = (size / (r.entry * (1 + COST)), r.exit, r.exit_date, r.entry)
            cash -= size
        for k in [k for k, v in held.items() if v[2] <= day]:
            v = held.pop(k)
            cash += v[0] * v[1] * (1 - COST)
        out.append((day, mark(), len(held)))
    c = pd.DataFrame(out, columns=['date', 'equity', 'n']).set_index('date')
    return c.equity.pct_change().fillna(0), c.n.mean()


def trade_stats(T, name):
    rows = []
    for lab, g in (('all', T), ('2017-21', T[T.entry_date < SPLIT]), ('2022-26', T[T.entry_date >= SPLIT])):
        d = g.groupby('entry_date')
        dx, dr = d.x.mean(), d.ret.mean()
        t = lambda s: round(s.mean() / s.std() * np.sqrt(len(s)), 2)
        w, l = g.ret[g.ret > 0], g.ret[g.ret <= 0]
        rows.append(dict(strategy=name, period=lab, trades=len(g), win=round((g.ret > 0).mean() * 100, 1),
                         avg=round(g.ret.mean() * 100, 3), avg_win=round(w.mean() * 100, 2), avg_loss=round(l.mean() * 100, 2),
                         pf=round(w.sum() / -l.sum(), 2), sessions=round(g.sessions.mean(), 1), t=t(dr),
                         vs_univ=round(g.x.mean() * 100, 3), t_vs_univ=t(dx)))
    return rows


def main():
    t0 = time.time()
    f = {k: {} for k in ('open', 'high', 'low', 'close', 'volume')}
    for p in sorted(DAILY.glob('*.csv')):
        d = pd.read_csv(p, parse_dates=['Date']).drop_duplicates('Date').set_index('Date')
        if len(d) < 300:
            continue
        for k in f:
            f[k][p.stem] = d[k.capitalize()]
    O, C, V = (pd.DataFrame(f[k]).sort_index() for k in ('open', 'close', 'volume'))
    spy = C.pop('SPY'); O.pop('SPY'); V.pop('SPY')
    print('panel', C.shape, f'{time.time() - t0:.0f}s', flush=True)
    ret = C.pct_change(fill_method=None)
    liquid = (C > 10) & (V.rolling(20).mean() > 1_000_000)
    sma200, sma5 = C.rolling(200).mean(), C.rolling(5).mean()
    start = C.index[260]
    ew = ret.where(liquid.shift(1, fill_value=False)).mean(axis=1)[start:]
    ew_idx = (1 + ew.fillna(0)).cumprod()
    spy_r = spy.pct_change()[start:]
    spy_on = (spy > spy.rolling(200).mean()).shift(2, fill_value=False)
    rows = [report('SPY buy and hold', spy_r, ew), report('SPY with 200-day timing', spy_r.where(spy_on[start:], 0), ew),
            report('Equal-weight universe', ew, ew)]

    # ---- MOM / MOM_T
    mom = (C.shift(21) / C.shift(252) - 1).where(liquid)
    month_ends = C.index.to_series().groupby(C.index.to_period('M')).last()
    Wt = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    Wb = Wt.copy()
    for d in month_ends:
        m = mom.loc[d].dropna()
        if len(m) < 100:
            continue
        top, bot = m[m >= m.quantile(0.9)].index, m[m <= m.quantile(0.1)].index
        Wt.loc[d] = 0.0; Wt.loc[d, top] = 1 / len(top)
        Wb.loc[d] = 0.0; Wb.loc[d, bot] = 1 / len(bot)
    mom_r = rebalance_returns(Wt, ret)[start:]
    rows += [report('MOM top decile', mom_r, ew), report('MOM bottom decile (check)', rebalance_returns(Wb, ret)[start:], ew),
             report('MOM_T top decile + SPY filter', mom_r.where(spy_on[start:], 0), ew)]

    # ---- REV
    r5 = (C / C.shift(5) - 1).where(liquid & (C > sma200))
    Wr = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    for d in C.index[260::5]:
        m = r5.loc[d].dropna()
        if len(m) < 100:
            continue
        bot = m[m <= m.quantile(0.1)].index
        Wr.loc[d] = 0.0; Wr.loc[d, bot] = 1 / len(bot)
    rows.append(report('REV weekly losers in uptrend', rebalance_returns(Wr, ret)[start:], ew))

    # ---- RSI2 and BRK (event trades)
    delta = C.diff()
    g = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = 100 - 100 / (1 + g / l.replace(0, np.nan))
    sig_rsi = (liquid & (C > sma200) & (rsi2 < 10)).to_numpy()
    hi252 = C.rolling(252).max()
    at_high = C >= hi252
    sig_brk = (liquid & at_high & ~at_high.shift(1, fill_value=False) & (V > 1.5 * V.rolling(20).mean()) & hi252.notna()).to_numpy()
    Cn, On, S5, R2 = C.to_numpy(), O.to_numpy(), sma5.to_numpy(), rsi2.to_numpy()
    dates, ewv = C.index, ew_idx.reindex(C.index).to_numpy()
    i0 = C.index.get_loc(start)
    tr = {'RSI2': [], 'BRK': []}
    for ci, tkr in enumerate(C.columns):
        c, o, s5 = Cn[:, ci], On[:, ci], S5[:, ci]
        n = len(c)
        busy = 0
        for t in np.flatnonzero(sig_rsi[i0:, ci]) + i0:
            if t < busy or t + 1 >= n or not o[t + 1] > 0:
                continue
            j = t + 1
            while j < min(t + 10, n - 1) and not c[j] > s5[j]:
                j += 1
            if not c[j] > 0:
                continue
            busy = j + 1
            tr['RSI2'].append((tkr, dates[t + 1], dates[j], o[t + 1], c[j], j - t, R2[t, ci], ewv[j] / ewv[t] - 1))
        for t in np.flatnonzero(sig_brk[i0:, ci]) + i0:
            j = t + 20
            if j >= n or not (o[t + 1] > 0 and c[j] > 0):
                continue
            tr['BRK'].append((tkr, dates[t + 1], dates[j], o[t + 1], c[j], 20, 0.0, ewv[j] / ewv[t] - 1))
    stats = []
    for name, lst in tr.items():
        T = pd.DataFrame(lst, columns=['ticker', 'entry_date', 'exit_date', 'entry', 'exit', 'sessions', 'rank', 'ew_ret'])
        T['ret'] = T.exit / T.entry - 1 - 2 * COST
        T['x'] = T.ret - T.ew_ret
        T.to_csv(OUT / f'families_{name}_trades.csv', index=False)
        stats += trade_stats(T, name)
        pr, npos = slot_sim(T, C, 20, 'rank', True)
        rows.append(report(f'{name} portfolio, 20 slots (avg {npos:.0f} held)', pr[start:], ew))
        yearly = T.groupby(T.entry_date.dt.year).agg(trades=('ret', 'size'), win=('ret', lambda s: round((s > 0).mean() * 100, 1)),
                                                     avg=('ret', lambda s: round(s.mean() * 100, 2)), vs_univ=('x', lambda s: round(s.mean() * 100, 2)))
        print(f'\n{name} by year\n{yearly.to_string()}')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'families_summary.csv', index=False)
    print('\nPORTFOLIO-LEVEL (% per year; "vs univ" = minus equal-weight universe)\n' + R.to_string(index=False))
    print('\nTRADE-LEVEL\n' + pd.DataFrame(stats).to_string(index=False))
    yr = pd.DataFrame({'SPY': spy_r, 'universe': ew, 'MOM': mom_r, 'MOM_T': mom_r.where(spy_on[start:], 0)})
    print('\nBY YEAR %\n' + ((1 + yr).groupby(yr.index.year).prod() - 1).mul(100).round(1).to_string())
    print(f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
