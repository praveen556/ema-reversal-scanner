"""Incrementally update Schwab regular-session 5-minute bars for Top20 Combined.

Place at: <project>/src/intraday_volatility/download_schwab_5min.py
Run directly from PyCharm. No command-line arguments.

Requires: pip install schwab-py pandas openpyxl
PyCharm environment variables: SCHWAB_APP_KEY, SCHWAB_APP_SECRET
Your OAuth token is reused from <project>/.schwab_auth/token.json.
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as clock_time
from multiprocessing import freeze_support
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from schwab.auth import easy_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANKING_FILE = PROJECT_ROOT / 'output' / 'intraday_volatility' / 'Intraday_Volatility_Rankings.xlsx'
RANKING_SHEET = 'Top20 Combined'
OUTPUT_DIR = PROJECT_ROOT / 'data' / '5min'
TOKEN_PATH = PROJECT_ROOT / '.schwab_auth' / 'token.json'
CALLBACK_URL = 'https://127.0.0.1:8182'  # Must exactly match Schwab app registration

TOP_N = 7040
WORKERS = 3
BATCH_SIZE = 100
# Shared request pacing across workers; adjust only after monitoring HTTP 429s.
MIN_SECONDS_BETWEEN_REQUESTS = 0.7
_rate_lock = threading.Lock()
_next_request_at = 0.0
_stop = threading.Event()

INITIAL_LOOKBACK_DAYS = 270   # Calendar days; change to 240 if desired
CHUNK_DAYS = 270               # Keep requests in modest date windows
OVERLAP_DAYS = 2              # Refresh latest existing candles on repeat runs
SKIP_UP_TO_DATE = True        # Skip files current through the latest expected completed 5-min bar
MAX_ATTEMPTS = 4
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']


def read_tickers():
    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f'Ranking workbook not found: {RANKING_FILE}\n'
            'Run rank_volatility.py first, or change RANKING_FILE to its actual path.'
        )
    df = pd.read_excel(RANKING_FILE, sheet_name=RANKING_SHEET)
    if 'Ticker' not in df.columns:
        raise ValueError(f'Column "Ticker" missing from {RANKING_SHEET}')
    symbols = df['Ticker'].dropna().astype(str).str.strip().str.upper()
    return list(dict.fromkeys(s for s in symbols if s and s != 'NAN'))[:TOP_N]


def authenticate():
    key = os.getenv('SCHWAB_APP_KEY')
    secret = os.getenv('SCHWAB_APP_SECRET')
    if not key or not secret:
        raise RuntimeError(
            'Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET in the PyCharm Run Configuration '
            'for THIS downloader (or set them as system environment variables).'
        )
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    return easy_client(
        api_key=key,
        app_secret=secret,
        callback_url=CALLBACK_URL,
        token_path=str(TOKEN_PATH),
    )


def load_existing(path):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    if 'datetime' not in df.columns:
        raise ValueError(f'Existing file missing datetime column: {path}')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df = df.dropna(subset=['datetime'])
    return df


def normalize_candles(candles):
    if not candles:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(candles)
    if not {'datetime', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
        raise ValueError('Schwab response missing required OHLCV columns')
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
    local = df['datetime'].dt.tz_convert(ET)
    session_time = local.dt.time
    df = df[(session_time >= clock_time(9, 30)) & (session_time < clock_time(16, 0))]
    return df[COLUMNS].copy()


def throttle():
    global _next_request_at
    with _rate_lock:
        delay = max(0, _next_request_at - time.monotonic())
        if delay:
            time.sleep(delay)
        _next_request_at = time.monotonic() + MIN_SECONDS_BETWEEN_REQUESTS


def fetch_window(client, ticker, start, end):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if _stop.is_set():
            raise RuntimeError("Stopped after authentication failure")
        try:
            throttle()
            response = client.get_price_history_every_five_minutes(
                ticker,
                start_datetime=start,
                end_datetime=end,
                need_extended_hours_data=False,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < MAX_ATTEMPTS:
                    delay = min(30, 2 ** attempt)
                    print(f'  {ticker}: HTTP {response.status_code}, retrying in {delay}s', flush=True)
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            body = response.json()
            if body.get('error'):
                raise RuntimeError(str(body['error']))
            return normalize_candles(body.get('candles', []))
        except Exception as exc:
            if is_auth_error(exc) or attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(30, 2 ** attempt))
    return pd.DataFrame(columns=COLUMNS)


def latest_expected_bar(now):
    """Latest fully completed regular-session 5-min candle start (ET).

    Uses weekday schedule only; holidays may result in a harmless extra API check.
    """
    day = now.date()
    if now.weekday() >= 5 or now.time() < clock_time(9, 35):
        day -= timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return datetime.combine(day, clock_time(15, 55), ET)
    if now.time() >= clock_time(16, 0):
        return datetime.combine(day, clock_time(15, 55), ET)
    # The 9:30 candle completes at 9:35. At 9:37, latest complete starts at 9:30.
    minutes = now.hour * 60 + now.minute
    start_minutes = ((minutes - 5) // 5) * 5
    return datetime.combine(day, clock_time(start_minutes // 60, start_minutes % 60), ET)


def download_ticker(client, ticker, now):
    file_path = OUTPUT_DIR / f'{ticker}.csv'
    existing = load_existing(file_path)
    if not existing.empty:
        existing = existing.sort_values('datetime').drop_duplicates('datetime', keep='last')
        last_utc = existing['datetime'].max()
        expected_utc = pd.Timestamp(latest_expected_bar(now)).tz_convert('UTC')
        if SKIP_UP_TO_DATE and last_utc >= expected_utc:
            return len(existing), 0, 'SKIPPED (up to date)'
        start = last_utc.to_pydatetime().astimezone(ET) - timedelta(days=OVERLAP_DAYS)
    else:
        start = now - timedelta(days=INITIAL_LOOKBACK_DAYS)

    if start >= now:
        return len(existing), 0, 'SKIPPED (no new period)'

    pieces = [existing]
    fetched = 0
    cursor = start
    while cursor < now:
        end = min(cursor + timedelta(days=CHUNK_DAYS), now)
        batch = fetch_window(client, ticker, cursor, end)
        if not batch.empty:
            pieces.append(batch)
            fetched += len(batch)
        cursor = end

    combined = pd.concat(pieces, ignore_index=True)
    if combined.empty:
        return 0, 0, 'NO DATA'
    combined['datetime'] = pd.to_datetime(combined['datetime'], utc=True, errors='coerce')
    combined = combined.dropna(subset=['datetime'])
    combined = combined.sort_values('datetime').drop_duplicates('datetime', keep='last')
    # Avoid rewriting unchanged files; preserve original data on an empty API response.
    if fetched == 0 and not existing.empty:
        return len(existing), 0, 'UNCHANGED (API returned no candles)'
    combined['datetime'] = combined['datetime'].dt.tz_convert(ET).astype(str)
    temp_path = file_path.with_suffix('.csv.tmp')
    combined.to_csv(temp_path, index=False)
    temp_path.replace(file_path)  # Atomic replacement on same drive
    return len(combined), fetched, 'UPDATED' if not existing.empty else 'CREATED'


def is_auth_error(exc):
    """Identify Schwab OAuth failures without mistaking data errors for auth."""
    message = str(exc).lower()
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    return status == 401 or any(term in message for term in (
        'token_invalid', 'invalid_token', 'token_expired',
        'invalid_grant', 'unauthorized', '401 client error',
    ))


def run_one(client, ticker, now):
    if _stop.is_set():
        return ticker, 0, 0, 'CANCELLED', None
    try:
        total, fetched, status = download_ticker(client, ticker, now)
        return ticker, total, fetched, status, None
    except Exception as exc:
        if is_auth_error(exc):
            _stop.set()
        return ticker, 0, 0, 'ERROR', exc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers()
    if not tickers:
        raise RuntimeError('No tickers found in Top20 Combined sheet')
    print(f'{len(tickers)} tickers; {WORKERS} workers; batches of {BATCH_SIZE}', flush=True)
    print('Using shared Schwab client and one shared API rate limiter.', flush=True)
    client = authenticate()
    now = datetime.now(ET)
    successes = skipped = 0
    failures = []
    for offset in range(0, len(tickers), BATCH_SIZE):
        if _stop.is_set():
            break
        batch = tickers[offset:offset+BATCH_SIZE]
        print(f'Batch {offset//BATCH_SIZE + 1}: tickers {offset+1}-{offset+len(batch)}', flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_one, client, ticker, now): ticker for ticker in batch}
            for future in as_completed(futures):
                ticker, total, fetched, status, exc = future.result()
                if exc is not None:
                    failures.append(ticker)
                    print(f'  ERROR {ticker}: {exc}', flush=True)
                    if is_auth_error(exc):
                        print('  AUTH ERROR: stopping. Existing CSVs preserved. Check token authorization.', flush=True)
                elif status == 'CANCELLED':
                    pass
                else:
                    print(f'  {ticker}: {status}, {total:,} stored, {fetched:,} fetched', flush=True)
                    if status.startswith('SKIPPED'):
                        skipped += 1
                    else:
                        successes += 1
        print(f'Progress: {min(offset+BATCH_SIZE,len(tickers))}/{len(tickers)}; '
              f'processed={successes}, skipped={skipped}, failed={len(failures)}', flush=True)
    print(f'Finished: processed={successes}, skipped={skipped}, failed={len(failures)}. '
          f'Output: {OUTPUT_DIR}', flush=True)
    if failures:
        print('Failed tickers: ' + ', '.join(failures), flush=True)
    if _stop.is_set():
        print('Authentication failure detected. Run your Schwab auth test/reauthorization '
              'and restart; existing files will be skipped if up to date.', flush=True)


if __name__ == '__main__':
    freeze_support()  # Required for schwab-py OAuth on Windows
    main()
