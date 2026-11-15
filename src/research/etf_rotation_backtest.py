"""ETF momentum rotation versus SPY, on dividend-adjusted prices. No orders.

ETFs have no delisting bias, so this is a clean test. Rules are the standard
published ones, fixed in advance; all three lookbacks are reported.
  SECTOR  top 3 SPDR sector funds by N-month total return, monthly
  ASSET   top 3 of 12 asset-class funds by N-month total return, monthly
  GEM     dual momentum: SPY or EFA, whichever has the higher 12-month
          return, but bonds (AGG) if that return is below T-bills (BIL)
  60/40   60% SPY, 40% IEF, monthly rebalance (reference)
Weights take effect one full day after the month-end signal; 5 bps per side.

Run: python src/research/etf_rotation_backtest.py
Prices come from Yahoo (cached in data/etf/prices.csv).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data' / 'etf' / 'prices.csv'
OUT = ROOT / 'output' / 'research'
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'DBC', 'LQD', 'HYG']
OTHER = ['AGG', 'BIL', 'SHY']
COST, START, SPLIT = 0.0005, '2008-01-01', '2017-01-01'


def prices():
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    import yfinance as yf
    tick = sorted(set(SECTORS + ASSETS + OTHER))
    p = yf.download(tick, start='2004-01-01', auto_adjust=True, progress=False)['Close']
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(CACHE)
    return p


def perf(r):
    eq = (1 + r).cumprod()
    return dict(cagr=round((eq.iloc[-1] ** (252 / len(r)) - 1) * 100, 1), vol=round(r.std() * np.sqrt(252) * 100, 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2), maxdd=round((eq / eq.cummax() - 1).min() * 100, 1))


def run(W, ret):
    W = W.ffill().fillna(0)
    turn = W.diff().abs().sum(axis=1)
    return ((W.shift(2) * ret).sum(axis=1) - turn.shift(2).fillna(0) * COST)[START:]


def top_n(P, names, months, n=3):
    mom = P[names] / P[names].shift(21 * months) - 1
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for d in P.index.to_series().groupby(P.index.to_period('M')).last():
        m = mom.loc[d].dropna()
        if len(m) < n + 2:
            continue
        W.loc[d] = 0.0
        W.loc[d, m.nlargest(n).index] = 1 / n
    return W


def main():
    P = prices().dropna(how='all')
    ret = P.pct_change(fill_method=None)
    ends = P.index.to_series().groupby(P.index.to_period('M')).last()
    S = {'SPY buy and hold': ret['SPY'][START:]}
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for d in ends:
        W.loc[d] = 0.0; W.loc[d, 'SPY'] = 0.6; W.loc[d, 'IEF'] = 0.4
    S['60/40 SPY/IEF'] = run(W, ret)
    for m in (3, 6, 12):
        S[f'SECTOR top 3, {m}-month'] = run(top_n(P, SECTORS, m), ret)
    for m in (3, 6, 12):
        S[f'ASSET top 3, {m}-month'] = run(top_n(P, ASSETS, m), ret)
    r12 = P / P.shift(252) - 1
    cash = r12['BIL'].fillna(r12['SHY'])
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for d in ends:
        if pd.isna(r12.loc[d, 'SPY']) or pd.isna(r12.loc[d, 'EFA']):
            continue
        best = 'SPY' if r12.loc[d, 'SPY'] >= r12.loc[d, 'EFA'] else 'EFA'
        W.loc[d] = 0.0
        W.loc[d, best if r12.loc[d, best] > cash.loc[d] else 'AGG'] = 1.0
    S['GEM dual momentum'] = run(W, ret)

    R = pd.DataFrame(S).dropna()
    yearly = (1 + R).groupby(R.index.year).prod() - 1
    full_years = yearly.iloc[:-1] if R.index[-1].month < 12 else yearly
    roll3 = (1 + R).rolling(756).apply(np.prod, raw=True) - 1
    rows = []
    for name in R:
        a, b, c = perf(R[name]), perf(R[name][:SPLIT]), perf(R[name][SPLIT:])
        rows.append({'strategy': name, 'cagr': a['cagr'], 'vs SPY': round(a['cagr'] - perf(R['SPY buy and hold'])['cagr'], 1),
                     'sharpe': a['sharpe'], 'maxdd': a['maxdd'], '2008-16': b['cagr'], '2017-26': c['cagr'],
                     'years beat SPY': f"{int((yearly[name] > yearly['SPY buy and hold']).sum())}/{len(yearly)}",
                     '3y windows beat SPY %': round((roll3[name] > roll3['SPY buy and hold']).loc[roll3[name].notna()].mean() * 100)})
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'etf_rotation_summary.csv', index=False)
    print(R.index[0].date(), R.index[-1].date())
    print(T.to_string(index=False))
    print('\nBY YEAR %\n' + yearly.mul(100).round(1).to_string())


if __name__ == '__main__':
    main()
