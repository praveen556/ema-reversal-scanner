"""Incrementally update SPY daily candles with schwab-py; never place trades.

Run by run_daily_scanner_updated.py or directly from PyCharm. Credentials are read
from SCHWAB_APP_KEY and SCHWAB_APP_SECRET environment variables.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, time as clock_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from schwab import auth

ROOT = Path(__file__).resolve().parent
TOKEN = ROOT / '.schwab_auth' / 'token.json'
OUTPUT = ROOT / 'data' / 'daily' / 'SPY.csv'
CALLBACK = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1:8182')
ET = ZoneInfo('America/New_York')
COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume']
INITIAL_DAYS = 730
OVERLAP_DAYS = 7
CHUNK_DAYS = 180


def get_client():
    key, secret = os.getenv('SCHWAB_APP_KEY'), os.getenv('SCHWAB_APP_SECRET')
    if not key or not secret:
        raise RuntimeError('Missing SCHWAB_APP_KEY / SCHWAB_APP_SECRET in this PyCharm run configuration.')
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    try:
        return auth.easy_client(api_key=key, app_secret=secret,
                                callback_url=CALLBACK, token_path=str(TOKEN))
    except Exception as exc:
        raise RuntimeError(f'Schwab authorization failed: {exc}. Run schwab_reauthorize.py if needed.') from exc


def read_existing():
    if not OUTPUT.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(OUTPUT)
    if not set(COLS).issubset(df.columns):
        raise ValueError(f'Unexpected SPY CSV columns in {OUTPUT}; existing file preserved.')
    return df[COLS].copy()


def normalize(candles):
    if not candles:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(candles)
    if not set(COLS).issubset(df.columns):
        raise ValueError('Schwab response is missing required OHLCV columns.')
    dt = pd.to_datetime(df['datetime'], unit='ms', utc=True).dt.tz_convert(ET)
    # Do not include the still-forming daily candle, or weekend/holiday placeholders.
    now = datetime.now(ET)
    if now.time() < clock_time(16, 15):
        df = df.loc[dt.dt.date < now.date()].copy()
        dt = dt.loc[df.index]
    df['datetime'] = dt.dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[COLS].dropna().copy()


def sync():
    client = get_client()
    existing = read_existing()
    now = datetime.now(timezone.utc)
    if existing.empty:
        start = now - timedelta(days=INITIAL_DAYS)
    else:
        last = pd.to_datetime(existing['datetime'], utc=True, errors='coerce').max()
        if pd.isna(last):
            raise ValueError('Cannot parse existing SPY timestamps; refusing to overwrite.')
        start = last.to_pydatetime() - timedelta(days=OVERLAP_DAYS)
    cursor = start
    chunks = []
    print(f'Schwab SPY daily sync: {start.date()} to {now.date()}', flush=True)
    while cursor < now:
        end = min(cursor + timedelta(days=CHUNK_DAYS), now)
        try:
            response = client.get_price_history_every_day('SPY', start_datetime=cursor,
                    end_datetime=end, need_extended_hours_data=False)
            if response.status_code != 200:
                body = response.text[:400]
                if response.status_code in (400, 401, 403) or 'invalid_grant' in body:
                    raise RuntimeError(f'Authentication/permission failure HTTP {response.status_code}: {body}. Reauthorize using schwab_reauthorize.py.')
                raise RuntimeError(f'Schwab HTTP {response.status_code}: {body}')
            chunks.append(normalize(response.json().get('candles', [])))
        except Exception as exc:
            raise RuntimeError(f'SPY daily download FAILED for {cursor.date()}–{end.date()}; no CSV changes saved: {exc}') from exc
        cursor = end
        time.sleep(0.25)
    incoming = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=COLS)
    if incoming.empty:
        raise RuntimeError('Schwab returned no SPY daily candles; existing CSV preserved.')
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined['_dt'] = pd.to_datetime(combined['datetime'], utc=True, errors='coerce')
    if combined['_dt'].isna().any():
        raise RuntimeError('Invalid timestamp in SPY data; existing CSV preserved.')
    combined = combined.sort_values('_dt').drop_duplicates('_dt', keep='last').drop(columns='_dt')
    latest = pd.to_datetime(combined['datetime'], utc=True).max().tz_convert(ET).date()
    today = datetime.now(ET)
    # For ordinary weekday sessions, refuse stale data. Holiday/early-close cases
    # require an exchange calendar; default is to stop rather than guess.
    if today.weekday() < 5 and today.time() >= clock_time(16, 30):
        expected = today.date()
    else:
        expected = (today - timedelta(days=1)).date()
        while expected.weekday() >= 5:
            expected -= timedelta(days=1)
    if latest < expected:
        raise RuntimeError(f'SPY latest session {latest}; expected {expected}. CSV not saved. If today is an exchange holiday, rerun on the next trading day.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix('.csv.tmp')
    combined[COLS].to_csv(tmp, index=False)
    tmp.replace(OUTPUT)
    print(f'SPY daily OK: latest={latest}, {len(combined):,} rows -> {OUTPUT}', flush=True)


if __name__ == '__main__':
    try:
        sync()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
