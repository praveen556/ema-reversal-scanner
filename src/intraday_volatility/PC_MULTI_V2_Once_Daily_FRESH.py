"""Scenario PC_MULTI_V2_ONCE_DAILY: previous-close extremes; one position per ticker; TP/SL only.
Run in PyCharm without arguments. Research on shares, not options.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = ROOT / 'data' / 'daily'
FIVE_DIR = ROOT / 'data' / '5min'
BASE_OUT = ROOT / 'output' / 'intraday_volatility'
OUT = BASE_OUT / 'PC_MULTI_V2_ONCE_DAILY'
QUALITY = BASE_OUT / '5min_Data_Quality_Report.csv'
if not QUALITY.exists():
    QUALITY = ROOT / 'output' / 'intraday_volatility' / '5min_Data_Quality_Report.csv'


LOOKBACK = 20
BUFFER_POINTS = 1.0
TARGET_PCT = 1.5
STOP_PCT = 1.0
ROUND_TRIP_BPS = 10.0
MAX_MISSING_SLOTS = 9
# None = all available 5-min sessions. Entries require a complete 78-bar session.
# Exits are evaluated on every available candle, including incomplete sessions.


def filtered_mean(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    s = s[s >= 0]
    if s.empty:
        return np.nan
    cutoff = 2 * s.mean()
    kept = s[s <= cutoff]
    return float(kept.mean()) if len(kept) else np.nan


def load_daily(path):
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = cols.get('date', cols.get('datetime'))
    if date_col is None:
        date_col = next((c for c in df.columns if str(c).startswith('Unnamed')), None)
    if date_col is None or not all(x in cols for x in ('open', 'high', 'low', 'close')):
        raise ValueError('Daily data requires date, open, high, low, close')
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()})
    for c in ('open', 'high', 'low', 'close'):
        out[c] = pd.to_numeric(df[cols[c]], errors='coerce')
    out = out.dropna().sort_values('date').drop_duplicates('date', keep='last')
    out = out[(out.open > 0) & (out.low > 0) & (out.high >= out[['open', 'close']].max(axis=1)) & (out.low <= out[['open', 'close']].min(axis=1))]
    out['prior_close'] = out['close'].shift(1)
    # Directional excursions: clamp negative excursions to zero.
    out['up_pct'] = ((out.high / out.prior_close - 1) * 100).clip(lower=0)
    out['down_pct'] = ((1 - out.low / out.prior_close) * 100).clip(lower=0)
    return out.set_index('date')


def load_five(path):
    df = pd.read_csv(path)
    required = {'datetime', 'open', 'high', 'low', 'close'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing 5-min columns: {required - set(df.columns)}')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True).dt.tz_convert('America/New_York')
    for c in ('open', 'high', 'low', 'close'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['datetime', 'open', 'high', 'low', 'close']).sort_values('datetime')
    df = df.drop_duplicates('datetime', keep='last')
    t = df.datetime.dt.strftime('%H:%M')
    df = df[(t >= '09:30') & (t < '16:00')].copy()
    df['date'] = df.datetime.dt.tz_localize(None).dt.normalize()
    return df


def complete_session(g):
    if len(g) != 78:
        return False
    times = g.datetime.dt.strftime('%H:%M').tolist()
    expected = pd.date_range('2000-01-01 09:30', periods=78, freq='5min').strftime('%H:%M').tolist()
    return times == expected


def exit_fill(position, bar):
    """Check gap at bar open first; if both intrabar levels hit, stop first."""
    o, h, l = float(bar.open), float(bar.high), float(bar.low)
    tp, sl = position['Target'], position['Stop']
    if position['Direction'] == 'long':
        if o <= sl: return o, 'STOP_GAP'
        if o >= tp: return o, 'TARGET_GAP'
        if l <= sl: return sl, 'STOP'
        if h >= tp: return tp, 'TARGET'
    else:
        if o >= sl: return o, 'STOP_GAP'
        if o <= tp: return o, 'TARGET_GAP'
        if h >= sl: return sl, 'STOP'
        if l <= tp: return tp, 'TARGET'
    return None


def new_position(ticker, direction, price, when, entry_type, trigger, avg, reference):
    tp = price * (1 + TARGET_PCT / 100 if direction == 'long' else 1 - TARGET_PCT / 100)
    sl = price * (1 - STOP_PCT / 100 if direction == 'long' else 1 + STOP_PCT / 100)
    return {'Ticker': ticker, 'Direction': direction, 'Entry Type': entry_type,
            'Entry Time': str(when), 'Entry Date': str(when.date()), 'Entry': price,
            'Target': tp, 'Stop': sl, 'Trigger Level': trigger,
            'Avg Excursion 20D %': avg, 'Previous Close': reference,
            'Bars Held': 0, 'Sessions Held': 0, 'Entry Datetime': when,
            'Missing Sessions While Open': 0}


def test_ticker(ticker, daily_path, five_path):
    day = load_daily(daily_path)
    bars = load_five(five_path)
    closed, open_positions = [], []
    position = None
    incomplete = 0
    last_bar = None
    entered_dates = set()  # At most one new entry per ticker on any calendar date
    for date, session in bars.groupby('date', sort=True):
        g = session.reset_index(drop=True)
        full = complete_session(g)
        if not full:
            incomplete += 1
        if position is not None:
            position['Sessions Held'] += 1
            if not full:
                position['Missing Sessions While Open'] += 1
        history = day.loc[day.index < date].tail(LOOKBACK)
        can_enter = full and len(history) == LOOKBACK and history['prior_close'].notna().all()
        if can_enter:
            up = filtered_mean(history['up_pct'])
            down = filtered_mean(history['down_pct'])
            can_enter = np.isfinite(up) and np.isfinite(down)
            if can_enter:
                ref = float(history.close.iloc[-1])
                long_level = ref * (1 - (down + BUFFER_POINTS) / 100)
                short_level = ref * (1 + (up + BUFFER_POINTS) / 100)
        for i, bar in enumerate(g.itertuples(index=False)):
            last_bar = bar
            if position is not None:
                position['Bars Held'] += 1
                result = exit_fill(position, bar)
                if result is not None:
                    exit_price, reason = result
                    sign = 1 if position['Direction'] == 'long' else -1
                    gross = sign * (exit_price / position['Entry'] - 1) * 100
                    record = {k: v for k, v in position.items() if k != 'Entry Datetime'}
                    record.update({'Exit Time': str(bar.datetime), 'Exit': exit_price,
                                   'Exit Reason': reason, 'Gross Return %': gross,
                                   'Net Return %': gross - ROUND_TRIP_BPS / 100,
                                   'Calendar Days Held': (bar.datetime.date() - position['Entry Datetime'].date()).days,
                                   'Status': 'CLOSED'})
                    closed.append(record)
                    position = None
                # Do not re-enter on the same candle after a position was active.
                continue
            if not can_enter or date in entered_dates:
                continue
            # Opening gap: enter at first 09:30 bar open, and evaluate exits in that same bar.
            if i == 0:
                opening = float(bar.open)
                if opening <= long_level:
                    direction, level, avg = 'long', long_level, down
                elif opening >= short_level:
                    direction, level, avg = 'short', short_level, up
                else:
                    direction = None
                if direction is not None:
                    position = new_position(ticker, direction, opening, bar.datetime, 'OPEN_GAP', level, avg, ref)
                    entered_dates.add(date)
                    position['Sessions Held'] = 1
                    position['Bars Held'] = 1
                    result = exit_fill(position, bar)
                    if result is not None:
                        exit_price, reason = result
                        sign = 1 if direction == 'long' else -1
                        gross = sign * (exit_price / opening - 1) * 100
                        rec = {k: v for k, v in position.items() if k != 'Entry Datetime'}
                        rec.update({'Exit Time': str(bar.datetime), 'Exit': exit_price, 'Exit Reason': reason,
                                    'Gross Return %': gross, 'Net Return %': gross - ROUND_TRIP_BPS / 100,
                                    'Calendar Days Held': 0, 'Status': 'CLOSED'})
                        closed.append(rec)
                        position = None
                    continue
            # Intraday threshold is recognized only after the candle finishes.
            # Entry is next candle's open, so the signal bar cannot be used for fills.
            if i >= len(g) - 1:
                continue
            if float(bar.low) <= long_level:
                direction, level, avg = 'long', long_level, down
            elif float(bar.high) >= short_level:
                direction, level, avg = 'short', short_level, up
            else:
                continue
            nextbar = g.iloc[i + 1]
            entry = float(nextbar['open'])
            position = new_position(ticker, direction, entry, nextbar['datetime'], 'INTRADAY', level, avg, ref)
            entered_dates.add(date)
            position['Sessions Held'] = 1
            # Immediately evaluate the entry candle, and continue after it.
            # We need to avoid processing the next candle twice: handled below by entry time guard.
            position['Bars Held'] = 0
        # end session
    if position is not None:
        rec = {k: v for k, v in position.items() if k != 'Entry Datetime'}
        rec.update({'Status': 'OPEN', 'Last Observed Time': str(last_bar.datetime) if last_bar else '',
                    'Last Observed Close': float(last_bar.close) if last_bar else np.nan,
                    'Unrealized Return %': ((float(last_bar.close) / position['Entry'] - 1) *
                                             (1 if position['Direction'] == 'long' else -1) * 100) if last_bar else np.nan})
        open_positions.append(rec)
    return closed, open_positions, incomplete


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not QUALITY.exists():
        raise FileNotFoundError(f'Quality report missing: {QUALITY}')
    q = pd.read_csv(QUALITY)
    col = 'Missing Interior 5-min Slots'
    q[col] = pd.to_numeric(q[col], errors='coerce')
    tickers = sorted(set(q.loc[q[col].between(0, MAX_MISSING_SLOTS), 'Ticker'].astype(str).str.upper()))
    print(f'PC_MULTI_V2: {len(tickers)} eligible tickers (<10 missing slots)', flush=True)
    all_closed, all_open, skipped = [], [], []
    start = time.monotonic()
    for n, ticker in enumerate(tickers, 1):
        daily, five = DAILY_DIR / f'{ticker}.csv', FIVE_DIR / f'{ticker}.csv'
        if not daily.exists() or not five.exists():
            skipped.append({'Ticker': ticker, 'Reason': 'Missing daily or 5min CSV'})
            continue
        try:
            closed, opened, incomplete = test_ticker(ticker, daily, five)
            all_closed.extend(closed)
            all_open.extend(opened)
        except Exception as exc:
            skipped.append({'Ticker': ticker, 'Reason': str(exc)})
        if n % 25 == 0 or n == len(tickers):
            elapsed = time.monotonic() - start
            print(f'{n}/{len(tickers)} | closed={len(all_closed)} | open={len(all_open)} | errors={len(skipped)} | elapsed={elapsed/60:.1f} min', flush=True)
    pd.DataFrame(all_closed).to_csv(OUT / 'PC_MULTI_V2_Closed_Trades.csv', index=False)
    pd.DataFrame(all_open).to_csv(OUT / 'PC_MULTI_V2_Open_Positions.csv', index=False)
    pd.DataFrame(skipped, columns=['Ticker', 'Reason']).to_csv(OUT / 'PC_MULTI_V2_Skipped_Tickers.csv', index=False)
    if all_closed:
        d = pd.DataFrame(all_closed)
        def summary_frame(group_cols):
            result = d.groupby(group_cols, dropna=False).agg(
                Trades=('Net Return %', 'size'),
                **{'Win Rate %': ('Net Return %', lambda x: (x > 0).mean() * 100),
                   'Average Net Return %': ('Net Return %', 'mean'),
                   'Median Net Return %': ('Net Return %', 'median'),
                   'Avg Calendar Days Held': ('Calendar Days Held', 'mean'),
                   'Unique Tickers': ('Ticker', 'nunique')}
            ).reset_index().round(4)
            return result
        summary_frame(['Direction', 'Entry Type']).to_csv(OUT / 'PC_MULTI_V2_Strategy_Summary.csv', index=False)
        summary_frame(['Ticker', 'Direction', 'Entry Type']).to_csv(OUT / 'PC_MULTI_V2_Ticker_Summary.csv', index=False)
    print(f'Finished. Scenario outputs: {OUT}', flush=True)
    print('Caution: assumes fills at official opening prices for gaps; no borrow, slippage or options pricing.', flush=True)


if __name__ == '__main__':
    main()
