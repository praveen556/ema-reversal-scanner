"""Momentum inside the S&P 500 using point-in-time membership. No orders.

At each month-end, only stocks that were index members on that date are
ranked. Standard rule, fixed in advance: rank by return from 12 months ago to
1 month ago, hold the top slice equal-weight for a month. A 6-month lookback
and a top-20% slice are reported alongside as robustness checks.
Weights take effect one full day after the signal; 5 bps cost per side.

Run: python src/research/sp500_membership.py   (once)
     python src/research/download_schwab_daily_history.py --years 20 --out-dir data/daily_sp500 --tickers ...
     python src/research/sp500_momentum_backtest.py

Limits: former members that no longer trade have no Schwab history and are
missing (coverage is printed per year); prices exclude dividends throughout,
including for SPY; membership before 2012 is unreliable, so the test starts
in 2013.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / 'data' / 'daily_sp500'
OUT = ROOT / 'output' / 'research'
COST, START, SPLIT = 0.0005, '2013-01-01', '2020-01-01'


def perf(r):
    eq = (1 + r).cumprod()
    return dict(cagr=round((eq.iloc[-1] ** (252 / len(r)) - 1) * 100, 1), vol=round(r.std() * np.sqrt(252) * 100, 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2), maxdd=round((eq / eq.cummax() - 1).min() * 100, 1))


def run(W, ret):
    W = W.ffill().fillna(0)
    turn = W.diff().abs().sum(axis=1)
    return ((W.shift(2) * ret).sum(axis=1) - turn.shift(2).fillna(0) * COST)[START:]


def main():
    closes = {}
    for p in sorted(DAILY.glob('*.csv')):
        d = pd.read_csv(p, parse_dates=['Date']).drop_duplicates('Date').set_index('Date')
        if len(d) > 60:
            closes[p.stem] = d['Close']
    C = pd.DataFrame(closes).sort_index()
    spy = C.pop('SPY')
    ret = C.pct_change(fill_method=None)
    M = pd.read_csv(OUT / 'sp500_membership.csv', parse_dates=['start', 'end'])
    member = pd.DataFrame(False, index=C.index, columns=C.columns)
    missing = {}
    for r in M.itertuples():
        end = r.end if pd.notna(r.end) else C.index[-1] + pd.Timedelta(days=1)
        if r.ticker in member:
            member.loc[(C.index >= r.start) & (C.index < end), r.ticker] = True
            if pd.notna(r.end) and not (M[M.ticker == r.ticker].end.isna()).any():
                # After a stock leaves for good its ticker may be reused: ignore later prices.
                last = M[M.ticker == r.ticker].end.max() + pd.Timedelta(days=31)
                ret.loc[ret.index > last, r.ticker] = np.nan
                C.loc[C.index > last, r.ticker] = np.nan
        elif end > pd.Timestamp(START):
            missing[r.ticker] = (max(r.start, pd.Timestamp(START)), end)
    ends = C.index.to_series().groupby(C.index.to_period('M')).last()
    ends = ends[ends >= pd.Timestamp(START) - pd.Timedelta(days=40)]

    cov = []
    for y in range(2013, C.index[-1].year + 1):
        d = pd.Timestamp(f'{y}-06-28')
        have = int((member.loc[:d].iloc[-1] & C.loc[:d].iloc[-1].notna()).sum())
        lost = sum(1 for s, e in missing.values() if s <= d < e)
        cov.append(dict(year=y, members_with_prices=have, members_missing=lost, coverage_pct=round(have / (have + lost) * 100, 1)))
    print(pd.DataFrame(cov).to_string(index=False))
    print('missing examples:', sorted(missing)[:25])

    def slice_weights(lookback, lo, hi):
        mom = C.shift(21) / C.shift(21 * lookback) - 1
        W = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
        for d in ends:
            m = mom.loc[d][member.loc[d]].dropna()
            if len(m) < 100:
                continue
            pick = m[(m >= m.quantile(lo)) & (m <= m.quantile(hi))].index
            W.loc[d] = 0.0
            W.loc[d, pick] = 1 / len(pick)
        return W

    Wall = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    for d in ends:
        pick = member.loc[d][member.loc[d] & C.loc[d].notna()].index
        Wall.loc[d] = 0.0
        Wall.loc[d, pick] = 1 / len(pick)
    S = {'SPY (price only)': spy.pct_change()[START:], 'All members, equal weight': run(Wall, ret),
         'MOM 12-1, top 10%': run(slice_weights(12, 0.9, 1.0), ret),
         'MOM 12-1, top 20%': run(slice_weights(12, 0.8, 1.0), ret),
         'MOM 6-1, top 10%': run(slice_weights(6, 0.9, 1.0), ret),
         'MOM 6-1, top 20%': run(slice_weights(6, 0.8, 1.0), ret),
         'MOM 12-1, bottom 10% (check)': run(slice_weights(12, 0.0, 0.1), ret)}
    R = pd.DataFrame(S).dropna()
    yearly = (1 + R).groupby(R.index.year).prod() - 1
    roll3 = (1 + R).rolling(756).apply(np.prod, raw=True) - 1
    base, ew = 'SPY (price only)', 'All members, equal weight'
    rows = []
    for name in R:
        a, b, c = perf(R[name]), perf(R[name][:SPLIT]), perf(R[name][SPLIT:])
        rows.append({'strategy': name, 'cagr': a['cagr'], 'vs SPY': round(a['cagr'] - perf(R[base])['cagr'], 1),
                     'vs members': round(a['cagr'] - perf(R[ew])['cagr'], 1), 'sharpe': a['sharpe'], 'maxdd': a['maxdd'],
                     '2013-19': b['cagr'], '2020-26': c['cagr'],
                     'years beat SPY': f"{int((yearly[name] > yearly[base]).sum())}/{len(yearly)}",
                     '3y windows beat SPY %': round((roll3[name] > roll3[base]).loc[roll3[name].notna()].mean() * 100)})
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'sp500_momentum_summary.csv', index=False)
    R.to_csv(OUT / 'sp500_momentum_daily_returns.csv')
    print(R.index[0].date(), R.index[-1].date())
    print(T.to_string(index=False))
    print('\nBY YEAR %\n' + yearly.mul(100).round(1).to_string())


if __name__ == '__main__':
    main()
