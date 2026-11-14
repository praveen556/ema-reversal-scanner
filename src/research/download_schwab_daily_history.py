"""Download long daily history from Schwab for research backtests. Never trades.

Writes data/daily_long/<TICKER>.csv (Date,Open,High,Low,Close,Volume). The
scanner's own data/daily folder is not touched. Existing files are skipped,
so the run can be resumed.

Run: python src/research/download_schwab_daily_history.py --years 10
Needs SCHWAB_APP_KEY / SCHWAB_APP_SECRET and a valid .schwab_auth/token.json.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from schwab_spy_daily_sync import get_client  # noqa: E402

OUT = ROOT / 'data' / 'daily_long'
ET = ZoneInfo('America/New_York')


def default_tickers():
    """Tickers that ever produced a liquid scanner signal, plus SPY."""
    ev = pd.read_pickle(ROOT / 'output' / 'research' / 'pit_events.pkl')
    return ['SPY'] + sorted(set(ev['ticker']) - {'SPY'})


def fetch(client, symbol, start, end):
    for attempt in range(4):
        r = client.get_price_history_every_day(
            symbol, start_datetime=start, end_datetime=end, need_extended_hours_data=False)
        if r.status_code == 200:
            return r.json().get('candles', [])
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code in (401, 403):
            raise RuntimeError(f'Auth failure HTTP {r.status_code}: {r.text[:200]}')
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=float, default=10)
    ap.add_argument('--tickers', nargs='*')
    ap.add_argument('--pause', type=float, default=0.55)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = a.tickers or default_tickers()
    client = get_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(a.years * 365.25))
    today = datetime.now(ET).date()
    ok = skipped = failed = 0
    bad = []
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        path = OUT / f'{t}.csv'
        if path.exists():
            skipped += 1
            continue
        try:
            candles = fetch(client, t.replace('-', '/'), start, end)
        except RuntimeError:
            raise
        except Exception as exc:
            candles = None
            print('ERR', t, str(exc)[:120], flush=True)
        if not candles:
            failed += 1
            bad.append(t)
        else:
            d = pd.DataFrame(candles)
            dt = pd.to_datetime(d['datetime'], unit='ms', utc=True).dt.tz_convert(ET)
            d['Date'] = dt.dt.strftime('%Y-%m-%d')
            d = d[dt.dt.date < today]  # completed sessions only
            d = d.rename(columns=str.capitalize)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            d.drop_duplicates('Date', keep='last').to_csv(path, index=False)
            ok += 1
        if i % 100 == 0:
            print(f'{i}/{len(tickers)} ok={ok} skipped={skipped} failed={failed} {time.time() - t0:.0f}s', flush=True)
        time.sleep(a.pause)
    (OUT / '_failed.txt').write_text('\n'.join(bad))
    print(f'DONE ok={ok} skipped={skipped} failed={failed} -> {OUT}')


if __name__ == '__main__':
    main()
