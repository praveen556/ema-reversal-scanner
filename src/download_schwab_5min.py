"""Incrementally update Schwab regular-session 5-minute bars for Top20 Combined.

Place at: <project>/src/intraday_volatility/download_schwab_5min.py
Run directly from PyCharm. No command-line arguments.

Requires: pip install schwab-py pandas openpyxl
PyCharm environment variables: SCHWAB_APP_KEY, SCHWAB_APP_SECRET
Your OAuth token is reused from <project>/.schwab_auth/token.json.
"""

import os
import time
from datetime import datetime, timedelta, time as clock_time
from multiprocessing import freeze_support
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from schwab.auth import easy_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKING_FILE = PROJECT_ROOT / 'output' / 'intraday_volatility' / 'Intraday_Volatility_Rankings.xlsx'
RANKING_SHEET = 'Top20 Combined'
OUTPUT_DIR = PROJECT_ROOT / 'data' / '5min'
TOKEN_PATH = PROJECT_ROOT / '.schwab_auth' / 'token.json'
CALLBACK_URL = 'https://127.0.0.1:8182'  # Must exactly match Schwab app registration

TOP_N = 7500
INITIAL_LOOKBACK_DAYS = 300   # Calendar days; change to 240 if desired
CHUNK_DAYS = 100               # Keep requests in modest date windows
OVERLAP_DAYS = 2              # Refresh latest existing candles on repeat runs
SKIP_UP_TO_DATE = True        # Skip files current through the latest expected completed 5-min bar
REQUEST_DELAY_SECONDS = 0.7
MAX_ATTEMPTS = 4
AUTH_CHECK_INTERVAL_SECONDS = 25 * 60  # Connection health check, not browser OAuth
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


class SchwabAuthenticationError(RuntimeError):
    """An OAuth failure requiring user reauthorization, not a retry."""


def is_auth_failure(exc):
    message = str(exc).lower()
    return any(term in message for term in (
        'invalid_grant', 'unsupported_token_type', 'refresh token is invalid',
        'refresh token is expired', 'refresh token is revoked',
        'invalid_token', 'token_expired',
    ))


def raise_if_auth_failure(exc):
    if is_auth_failure(exc):
        raise SchwabAuthenticationError(
            'Schwab OAuth authorization expired or was revoked. '
            'Stop downloads and reauthorize .schwab_auth/token.json. '
            'Existing CSV files have not been deleted.'
        ) from exc


def auth_health_check(client):
    """Use the existing OAuth client; schwab-py refreshes access tokens as needed.

    A revoked refresh token cannot be fixed without interactive reauthorization.
    """
    print('  AUTH CHECK: verifying Schwab session...', flush=True)
    try:
        response = client.get_price_history_every_day('SPY')
        if response.status_code in (401, 403):
            raise SchwabAuthenticationError(
                f'HTTP {response.status_code}: Schwab authorization/permission failure'
            )
        response.raise_for_status()
    except SchwabAuthenticationError:
        raise
    except Exception as exc:
        raise_if_auth_failure(exc)
        raise
    print('  AUTH CHECK: OK (automatic access-token refresh remains enabled)', flush=True)


def fetch_window(client, ticker, start, end):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.get_price_history_every_five_minutes(
                ticker,
                start_datetime=start,
                end_datetime=end,
                need_extended_hours_data=True,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < MAX_ATTEMPTS:
                    delay = min(30, 2 ** attempt)
                    print(f'  {ticker}: HTTP {response.status_code}, retrying in {delay}s', flush=True)
                    time.sleep(delay)
                    continue
            if response.status_code in (401, 403):
                raise SchwabAuthenticationError(
                    f'Schwab HTTP {response.status_code}: authentication/permission failure; '
                    'check authorization and API access.'
                )
            response.raise_for_status()
            body = response.json()
            if body.get('error'):
                raise RuntimeError(str(body['error']))
            return normalize_candles(body.get('candles', []))
        except SchwabAuthenticationError:
            raise
        except Exception as exc:
            raise_if_auth_failure(exc)
            if attempt >= MAX_ATTEMPTS:
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


def download_ticker(client, ticker, now, auth_clock=None):
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
        if auth_clock is not None and time.monotonic() - auth_clock[0] >= AUTH_CHECK_INTERVAL_SECONDS:
            auth_health_check(client)
            auth_clock[0] = time.monotonic()
        batch = fetch_window(client, ticker, cursor, end)
        if not batch.empty:
            pieces.append(batch)
            fetched += len(batch)
        cursor = end
        time.sleep(REQUEST_DELAY_SECONDS)

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers()
    if not tickers:
        raise RuntimeError('No tickers found in Top20 Combined sheet')
    print(f'Found {len(tickers)} tickers: {", ".join(tickers)}', flush=True)
    print('Connecting to Schwab using existing OAuth token...', flush=True)
    try:
        client = authenticate()
    except Exception as exc:
        raise_if_auth_failure(exc)
        raise
    now = datetime.now(ET)
    successes = 0
    skipped = 0
    failures = []
    auth_clock = [time.monotonic()]
    for i, ticker in enumerate(tickers, 1):
        print(f'[{i}/{len(tickers)}] {ticker}...', flush=True)
        try:
            total, fetched, status = download_ticker(client, ticker, now, auth_clock=auth_clock)
            print(f'  {status}: {total:,} stored candles ({fetched:,} downloaded this run)', flush=True)
            if status.startswith("SKIPPED"):
                skipped += 1
            else:
                successes += 1
        except SchwabAuthenticationError as exc:
            print(f'  AUTH ERROR {ticker}: {exc}', flush=True)
            raise SystemExit(2) from exc
        except Exception as exc:
            raise_if_auth_failure(exc)
            print(f'  ERROR {ticker}: {exc}', flush=True)
            failures.append(ticker)
    print(f'Finished: {successes} processed, {skipped} skipped, {len(failures)} failed. Files: {OUTPUT_DIR}', flush=True)
    if failures:
        print('Failed tickers (retry later): ' + ', '.join(failures), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    freeze_support()  # Required for schwab-py OAuth on Windows
    main()
