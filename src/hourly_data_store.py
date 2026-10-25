"""Persistent regular-session 1-hour Yahoo OHLCV cache for EMA scanner.

Place beside scanner_v2.py in src/. The scanner downloads candidate tickers only.
Data is stored in <project>/data/hourly/<SYMBOL>.csv, with timezone-aware
America/New_York timestamps. Older history is never discarded on refresh.
"""
from __future__ import annotations

from pathlib import Path
import re
import time
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
HOURLY_DIR = ROOT / 'data' / 'hourly'
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
TZ = 'America/New_York'


def _file(symbol: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9.^=_-]+', symbol):
        raise ValueError(f'Invalid ticker: {symbol!r}')
    return HOURLY_DIR / f'{symbol.upper()}.csv'


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name='Datetime', tz=TZ))
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if 'Datetime' in out.columns:
        out = out.set_index('Datetime')
    elif 'Date' in out.columns:
        out = out.set_index('Date')
    out.index = pd.to_datetime(out.index, utc=True, errors='coerce').tz_convert(TZ)
    out = out.loc[~out.index.isna()]
    missing = set(COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f'Missing hourly columns: {sorted(missing)}')
    out = out[COLUMNS].copy()
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['Open', 'High', 'Low', 'Close'])
    out = out.between_time('09:30', '15:59')
    out = out.loc[~out.index.duplicated(keep='last')].sort_index()
    out.index.name = 'Datetime'
    return out


def load_hourly(symbol: str) -> pd.DataFrame:
    path = _file(symbol)
    if not path.exists():
        return normalize(pd.DataFrame())
    try:
        df = pd.read_csv(path)
        # Saved timestamps include UTC offset; utc=True safely handles DST.
        return normalize(df)
    except Exception as exc:
        raise RuntimeError(f'Cannot read cached hourly file {path}: {exc}') from exc


def update_hourly(symbol: str, *, initial_period: str = '60d', refresh_period: str = '7d',
                  retries: int = 2) -> tuple[pd.DataFrame, str]:
    """Fetch latest available data, merge by timestamp, and atomically save.

    Returns (entire local history, status). On download failure, existing cache
    remains usable but status clearly says STALE. No backfilled synthetic bars.
    """
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_hourly(symbol)
    # Use a longer overlap after missed runs, within the provider's 1h retention.
    if existing.empty:
        period = initial_period
    else:
        age_days = (pd.Timestamp.now(tz=TZ) - existing.index.max()).total_seconds() / 86400
        period = initial_period if age_days > 5 else refresh_period
        if age_days > 55:
            print(f'WARNING {symbol}: last cached hour is {age_days:.0f} days old; provider may no longer offer the missing interval')
    last_error = None
    fetched = None
    for attempt in range(retries):
        try:
            fetched = yf.Ticker(symbol).history(
                period=period, interval='1h', prepost=False, auto_adjust=False,
                actions=False, raise_errors=True,
            )
            if fetched is None or fetched.empty:
                raise ValueError('Provider returned no hourly candles')
            fetched = normalize(fetched)
            if fetched.empty:
                raise ValueError('No valid regular-session candles returned')
            break
        except Exception as exc:
            last_error = exc
            fetched = None
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    if fetched is None:
        status = f'STALE (download failed: {last_error})' if not existing.empty else f'MISSING ({last_error})'
        return existing, status
    merged = pd.concat([existing, fetched])
    merged = merged.loc[~merged.index.duplicated(keep='last')].sort_index()
    path = _file(symbol)
    temp = path.with_suffix('.tmp')
    merged.to_csv(temp, date_format='%Y-%m-%dT%H:%M:%S%z')
    temp.replace(path)
    added = len(merged) - len(existing)
    return merged, f'UPDATED (+{added} bars, {len(merged)} total)'


def update_candidates(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Refresh each candidate and return local DataFrames for scanner use."""
    result = {}
    unique = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    print(f'Hourly cache: updating {len(unique)} daily candidates -> {HOURLY_DIR}')
    failed = []
    for i, ticker in enumerate(unique, 1):
        try:
            frame, status = update_hourly(ticker)
            if not frame.empty:
                result[ticker] = frame
            if status.startswith(('STALE', 'MISSING')):
                failed.append(ticker)
            if i <= 10 or i % 25 == 0 or status.startswith(('STALE', 'MISSING')):
                print(f'  [{i}/{len(unique)}] {ticker}: {status}')
        except Exception as exc:
            failed.append(ticker)
            print(f'  [{i}/{len(unique)}] {ticker}: ERROR {exc}')
    print(f'Hourly cache complete: {len(result)} available, {len(failed)} download/read issues')
    if failed:
        print('Hourly issues:', ', '.join(failed[:30]))
    return result
