"""Repair Schwab daily files that contain impossible prices.

Schwab's history back-adjusts some special dividends by subtraction, which
produces zero or negative prices (TDG, TPL, KDP ...), and has occasional bad
ticks. Any file with a non-positive close or a one-day move beyond +60% / -45%
is re-fetched from Yahoo (split-adjusted, not dividend-adjusted, to match the
rest). If Yahoo has no comparable history the bad rows are dropped instead.

Run: python src/research/repair_price_history.py --dir data/daily_sp500
"""
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


def suspicious(d):
    r = d['Close'].pct_change()
    return bool((d[['Open', 'High', 'Low', 'Close']] <= 0).any().any() or (r > 0.6).any() or (r < -0.45).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True)
    a = ap.parse_args()
    folder = a.dir if a.dir.is_absolute() else ROOT / a.dir
    flagged = []
    for p in sorted(folder.glob('*.csv')):
        d = pd.read_csv(p)
        if suspicious(d):
            flagged.append((p, d))
    print(f'{len(flagged)} suspicious files in {folder}')
    log = []
    for p, d in flagged:
        start = d['Date'].iloc[0]
        try:
            y = yf.download(p.stem, start=start, auto_adjust=False, progress=False)
            if isinstance(y.columns, pd.MultiIndex):
                y.columns = y.columns.get_level_values(0)
            y = y[COLS].dropna()
        except Exception:
            y = pd.DataFrame()
        if len(y) >= 0.8 * len(d) and not suspicious(y.reset_index()) or (len(y) >= 0.8 * len(d) and (y[COLS[:4]] > 0).all().all()):
            y.index = pd.to_datetime(y.index).strftime('%Y-%m-%d')
            y.index.name = 'Date'
            y.to_csv(p)
            log.append((p.stem, 'replaced_from_yahoo', len(d), len(y)))
        else:
            ok = (d[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
            d[ok].to_csv(p, index=False)
            log.append((p.stem, 'dropped_bad_rows', len(d), int(ok.sum())))
    L = pd.DataFrame(log, columns=['ticker', 'action', 'rows_before', 'rows_after'])
    L.to_csv(folder / '_repaired.txt', index=False)  # not .csv: the folder is globbed for price files
    print(L.action.value_counts().to_string())
    print(L[L.action == 'dropped_bad_rows'].to_string(index=False))


if __name__ == '__main__':
    main()
