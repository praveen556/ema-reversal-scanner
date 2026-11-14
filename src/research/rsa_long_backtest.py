"""RSA (relative strength + EMA20 pullback) on long daily history. No orders.

Rules mirror src/rsa_moomoo_paper_v1.py:
  filters  close >= 30, 20d avg volume >= 1M, 20d avg dollar volume >= 100M
  trend    close > EMA50, EMA20 > EMA50, EMA50 rising over 5 sessions
  pullback prior low <= prior EMA20 * 1.01, close > open, close > EMA20,
           close > prior close, low >= EMA20 * 0.97
  strength 20d return minus SPY 20d return > 3 points
  entry    next session open
  exit     stop at entry - 1.5 * ATR14 (gap-aware), or two closes below EMA20
           (next open), or 60 sessions
Portfolio: 30 slots, equal weight of current equity, ranked by strength.

Run: python src/research/rsa_long_backtest.py
Reads data/daily_long, writes output/research/rsa_long_*.

Limits: today's tickers only (survivorship), daily bars (stop assumed hit
before any same-day recovery), next-open fills, no earnings filter.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / 'data' / 'daily_long'
OUT = ROOT / 'output' / 'research'
MIN_PRICE, MIN_VOL, MIN_DOLLAR = 30.0, 1_000_000, 100_000_000
ATR_MULT, MAX_HOLD, RS_MIN = 1.5, 60, 0.03
SLOTS, COST_SIDE = 30, 0.0005
CONTROL_EVERY = 10  # sample 1 in N liquid stock-days as a no-signal control


def load(p):
    d = pd.read_csv(p)
    d.columns = [c.lower() for c in d.columns]
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)
    return d if len(d) >= 120 else None


def lag(a, k):
    out = np.full(len(a), np.nan)
    out[k:] = a[:len(a) - k]
    return out


def run_exit(t, O, L, C, e20, atr):
    """Return (exit_index, exit_price, reason) for a signal on day t, or None."""
    n, e = len(O), t + 1
    if e >= n:
        return None
    stop = O[e] - ATR_MULT * atr[t]
    below = 0
    for j in range(e, min(e + MAX_HOLD, n)):
        if j > e and O[j] <= stop:
            return j, O[j], 'GAP_STOP'
        if L[j] <= stop:
            return j, stop, 'STOP'
        below = below + 1 if C[j] < e20[j] else 0
        if below >= 2 or j == e + MAX_HOLD - 1:
            if j + 1 < n:
                return j + 1, O[j + 1], 'EMA20' if below >= 2 else 'MAX_HOLD'
            return j, C[j], 'OPEN_AT_END'
    return n - 1, C[n - 1], 'OPEN_AT_END'


def main():
    t0 = time.time()
    spy = load(DAILY / 'SPY.csv').set_index('date')
    spy_ret20 = spy.close / spy.close.shift(20) - 1
    s20 = spy.close.ewm(span=20, adjust=False).mean()
    s200 = spy.close.ewm(span=200, adjust=False).mean()
    trades, closes, liquid_ret = [], {}, []
    for i, p in enumerate(sorted(DAILY.glob('*.csv')), 1):
        tkr = p.stem
        d = load(p)
        if d is None or tkr == 'SPY':
            continue
        O, H, L, C, V = (d[c].to_numpy(float) for c in ('open', 'high', 'low', 'close', 'volume'))
        e20 = d.close.ewm(span=20, adjust=False).mean().to_numpy()
        e50 = d.close.ewm(span=50, adjust=False).mean().to_numpy()
        pc = lag(C, 1)
        tr = np.nanmax(np.vstack([H - L, np.abs(H - pc), np.abs(L - pc)]), axis=0)
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        avgvol = pd.Series(V).rolling(20).mean().to_numpy()
        avgdol = pd.Series(C * V).rolling(20).mean().to_numpy()
        ret20 = C / lag(C, 20) - 1
        rs = ret20 - spy_ret20.reindex(d.date).to_numpy()
        idx = np.arange(len(d))
        liquid = (C >= MIN_PRICE) & (avgvol >= MIN_VOL) & (avgdol >= MIN_DOLLAR) & (atr > 0) & (idx >= 75)
        trend = (C > e50) & (e20 > e50) & (e50 > lag(e50, 5))
        pull = (lag(L, 1) <= lag(e20, 1) * 1.01) & (C > O) & (C > e20) & (C > pc) & (L >= e20 * 0.97)
        sig = liquid & trend & pull & (rs > RS_MIN)
        ctrl = liquid & ~sig & ((idx + i) % CONTROL_EVERY == 0)
        for kind, mask in (('RSA', sig), ('CONTROL', ctrl)):
            for t in np.flatnonzero(mask):
                x = run_exit(t, O, L, C, e20, atr)
                if x is None:
                    continue
                j, px, why = x
                trades.append((kind, tkr, d.date[t], d.date[t + 1], d.date[j], O[t + 1], px, why,
                               j - t, rs[t], atr[t] / C[t], bool(trend[t]), bool(pull[t]), rs[t] > RS_MIN))
        closes[tkr] = pd.Series(C, index=d.date)
        # Membership is decided on the PRIOR day, so the benchmark has no look-ahead.
        liquid_ret.append(pd.Series(np.where(np.r_[False, liquid[:-1]], C / pc - 1, np.nan), index=d.date))
        if i % 500 == 0:
            print(i, f'{time.time() - t0:.0f}s', flush=True)
    T = pd.DataFrame(trades, columns=['kind', 'ticker', 'signal_date', 'entry_date', 'exit_date', 'entry', 'exit',
                                      'reason', 'sessions', 'rs', 'atr_pct', 'trend', 'pull', 'strong'])
    T['ret'] = T.exit / T.entry - 1 - 2 * COST_SIDE
    ew = pd.concat(liquid_ret, axis=1).mean(axis=1).fillna(0)       # equal-weight liquid universe, daily
    ew_idx = (1 + ew).cumprod()
    spy_c = spy.close
    # Benchmarks over each trade's own holding window (signal-day close to exit-day close).
    T['spy_ret'] = spy_c.reindex(T.exit_date).to_numpy() / spy_c.reindex(T.signal_date).to_numpy() - 1
    T['ew_ret'] = ew_idx.reindex(T.exit_date).to_numpy() / ew_idx.reindex(T.signal_date).to_numpy() - 1
    T['spy_above20'] = (spy_c > s20).reindex(T.signal_date).to_numpy()
    T['spy_above200'] = (spy_c > s200).reindex(T.signal_date).to_numpy()
    T.to_pickle(OUT / 'rsa_long_trades.pkl')

    # ---- portfolio simulation
    R = T[(T.kind == 'RSA') & (T.reason != 'OPEN_AT_END')].sort_values(['entry_date', 'rs'], ascending=[True, False])
    by_entry = {k: g for k, g in R.groupby('entry_date')}
    px = pd.DataFrame(closes).sort_index()
    dates = px.index[px.index >= R.entry_date.min()]
    cash, held, curve, taken = 1.0, {}, [], []
    for day in dates:
        row = px.loc[day]
        equity = cash + sum(v['shares'] * (row[k] if row[k] == row[k] else v['entry']) for k, v in held.items())
        for _, r in (by_entry.get(day, R.iloc[0:0])).iterrows():
            if len(held) >= SLOTS:
                break
            if r.ticker in held:
                continue
            size = min(equity / SLOTS, cash)
            if size <= 0:
                break
            held[r.ticker] = dict(shares=size / (r.entry * (1 + COST_SIDE)), exit=r.exit, exit_date=r.exit_date, entry=r.entry)
            cash -= size
            taken.append(r.name)
        # Close everything due today, including stops hit on the entry day.
        for tkr in [k for k, v in held.items() if v['exit_date'] <= day]:
            v = held.pop(tkr)
            cash += v['shares'] * v['exit'] * (1 - COST_SIDE)
        equity = cash + sum(v['shares'] * (row[k] if row[k] == row[k] else v['entry']) for k, v in held.items())
        curve.append((day, equity, len(held)))
    curve = pd.DataFrame(curve, columns=['date', 'equity', 'positions']).set_index('date')
    curve['spy'] = spy_c.reindex(curve.index).ffill() / spy_c.reindex(curve.index).ffill().iloc[0]
    curve['ew_universe'] = ew_idx.reindex(curve.index) / ew_idx.reindex(curve.index).iloc[0]
    curve.to_csv(OUT / 'rsa_long_equity.csv')
    T.loc[taken].to_csv(OUT / 'rsa_long_portfolio_trades.csv', index=False)
    print('trades', len(T), 'RSA', (T.kind == 'RSA').sum(), 'portfolio trades', len(taken), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
