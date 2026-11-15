"""Download historical earnings dates and EPS surprises from Yahoo.

Writes data/earnings/<TICKER>.csv (date, time_et, eps_estimate, eps_reported,
surprise_pct). Covers every ticker that has a price file in data/daily_sp500
or data/daily_long. Existing files are skipped, so the run can be resumed.

Run: python src/research/download_earnings_dates.py
"""
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'earnings'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    first = sorted(p.stem for p in (ROOT / 'data' / 'daily_sp500').glob('*.csv'))
    rest = sorted(p.stem for p in (ROOT / 'data' / 'daily_long').glob('*.csv'))
    tickers = [t for t in dict.fromkeys(first + rest) if t != 'SPY']
    ok = failed = skipped = 0
    bad = []
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        path = OUT / f'{t}.csv'
        if path.exists():
            skipped += 1
            continue
        d = None
        for attempt in range(3):
            try:
                d = yf.Ticker(t).get_earnings_dates(limit=100)
                break
            except Exception as exc:
                if 'Too Many' in str(exc) or '429' in str(exc):
                    time.sleep(30 * (attempt + 1))
                else:
                    break
        if d is None or d.empty:
            failed += 1
            bad.append(t)
        else:
            idx = pd.to_datetime(d.index, utc=True).tz_convert('America/New_York')
            out = pd.DataFrame({'date': idx.strftime('%Y-%m-%d'), 'time_et': idx.strftime('%H:%M'),
                                'eps_estimate': d['EPS Estimate'].to_numpy(), 'eps_reported': d['Reported EPS'].to_numpy(),
                                'surprise_pct': d['Surprise(%)'].to_numpy()})
            out.drop_duplicates('date').sort_values('date').to_csv(path, index=False)
            ok += 1
        if i % 100 == 0:
            print(f'{i}/{len(tickers)} ok={ok} skipped={skipped} failed={failed} {time.time() - t0:.0f}s', flush=True)
        time.sleep(0.25)
    (OUT / '_failed.txt').write_text('\n'.join(bad))
    print(f'DONE ok={ok} skipped={skipped} failed={failed}')


if __name__ == '__main__':
    main()
