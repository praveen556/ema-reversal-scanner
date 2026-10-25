"""Five-minute intraday reversal research backtest (shares, NOT options).
Run directly from PyCharm. No CLI arguments.
Historical daily thresholds use prior 20 completed days only.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # src/intraday_volatility/ -> project root
DAILY_DIR = ROOT / 'data' / 'daily'
FIVE_MIN_DIR = ROOT / 'data' / '5min'
OUT_DIR = ROOT / 'output' / 'intraday_volatility'
QUALITY_REPORT = OUT_DIR / '5min_Data_Quality_Report.csv'
# Alternative if you saved the validator reports in output/ directly:
if not QUALITY_REPORT.exists():
    QUALITY_REPORT = ROOT / 'output' / '5min_Data_Quality_Report.csv'

MAX_MISSING_SLOTS = 9  # strictly fewer than 10
LOOKBACK = 20
BUFFER_PCT_POINTS = 1.0
TARGET_PCT = 1.5
STOP_PCT = 1.0
ROUND_TRIP_COST_BPS = 10.0
LATEST_SIGNAL_TIME = '15:00'


def filtered_mean(values):
    """Remove values > 2x original average, as in volatility ranking."""
    s = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    s = s[s >= 0]
    if s.empty:
        return np.nan
    raw_mean = s.mean()
    kept = s[s <= 2 * raw_mean]
    return float(kept.mean()) if not kept.empty else np.nan


def load_daily(path):
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = cols.get('date') or cols.get('datetime')
    if date_col is None:
        date_col = next((c for c in df.columns if str(c).startswith('Unnamed')), None)
    if date_col is None or any(x not in cols for x in ('open', 'high', 'low', 'close')):
        raise ValueError('Daily file lacks date/open/high/low/close columns')
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    for k in ('open', 'high', 'low', 'close'):
        out[k] = pd.to_numeric(df[cols[k]], errors='coerce')
    out = out.dropna().sort_values('date').drop_duplicates('date', keep='last')
    out = out[(out['open'] > 0) & (out['low'] > 0) & (out['high'] >= out[['open','close']].max(axis=1)) & (out['low'] <= out[['open','close']].min(axis=1))]
    return out.set_index('date')


def load_intraday(path):
    df = pd.read_csv(path)
    required = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing 5-minute columns: {required - set(df.columns)}')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True).dt.tz_convert('America/New_York')
    for k in ('open', 'high', 'low', 'close', 'volume'):
        df[k] = pd.to_numeric(df[k], errors='coerce')
    df = df.dropna().sort_values('datetime')
    df['date'] = df['datetime'].dt.tz_localize(None).dt.normalize()
    return df


def full_session(g):
    """Only normal 9:30-15:55 ET sessions with all 78 distinct five-minute slots."""
    if len(g) != 78 or g['datetime'].duplicated().any():
        return False
    local = g['datetime'].dt.tz_localize(None)
    expected = pd.date_range(local.iloc[0].normalize() + pd.Timedelta(hours=9, minutes=30), periods=78, freq='5min')
    return bool(local.reset_index(drop=True).equals(pd.Series(expected)))


def backtest_symbol(ticker, intraday_path, daily_path):
    daily = load_daily(daily_path)
    raw = load_intraday(intraday_path)
    trades = []
    skipped_sessions = 0
    for date, session in raw.groupby('date', sort=True):
        g = session.reset_index(drop=True)
        if not full_session(g):
            skipped_sessions += 1
            continue
        history = daily.loc[daily.index < date].tail(LOOKBACK)
        if len(history) < LOOKBACK:
            continue
        up = filtered_mean((history['high'] - history['open']) / history['open'] * 100)
        down = filtered_mean((history['open'] - history['low']) / history['open'] * 100)
        if not np.isfinite(up) or not np.isfinite(down):
            continue
        prev_close = float(history['close'].iloc[-1])
        today_open = float(g['open'].iloc[0])
        for method, ref in [('open', today_open), ('previous_close', prev_close)]:
            for direction, avg in [('long', down), ('short', up)]:
                trigger_pct = avg + BUFFER_PCT_POINTS
                level = ref * (1 - trigger_pct / 100 if direction == 'long' else 1 + trigger_pct / 100)
                signal = None
                # Require a completed signal bar, then enter at NEXT bar open.
                for i in range(3, len(g) - 1):
                    if g.loc[i, 'datetime'].strftime('%H:%M') > LATEST_SIGNAL_TIME:
                        break
                    if (direction == 'long' and g.loc[i, 'low'] <= level) or (direction == 'short' and g.loc[i, 'high'] >= level):
                        signal = i
                        break
                if signal is None:
                    continue
                entry_i = signal + 1
                entry = float(g.loc[entry_i, 'open'])
                tp = entry * (1 + TARGET_PCT / 100 if direction == 'long' else 1 - TARGET_PCT / 100)
                sl = entry * (1 - STOP_PCT / 100 if direction == 'long' else 1 + STOP_PCT / 100)
                exit_price = float(g.iloc[-1]['close'])
                exit_i = len(g) - 1
                reason = 'EOD'
                # First bar begins at entry open; gaps handled using actual bar opens.
                for j in range(entry_i, len(g)):
                    bar = g.iloc[j]
                    o, h, l = float(bar['open']), float(bar['high']), float(bar['low'])
                    if direction == 'long':
                        if o <= sl: exit_price, reason = o, 'stop_open'
                        elif o >= tp: exit_price, reason = o, 'target_open'
                        elif l <= sl: exit_price, reason = sl, 'stop'
                        elif h >= tp: exit_price, reason = tp, 'target'
                        else: continue
                    else:
                        if o >= sl: exit_price, reason = o, 'stop_open'
                        elif o <= tp: exit_price, reason = o, 'target_open'
                        elif h >= sl: exit_price, reason = sl, 'stop'
                        elif l <= tp: exit_price, reason = tp, 'target'
                        else: continue
                    exit_i = j
                    break
                gross = (exit_price / entry - 1) * (1 if direction == 'long' else -1) * 100
                trades.append({
                    'Date': date.date(), 'Ticker': ticker, 'Method': method, 'Direction': direction,
                    'Avg Excursion 20D %': round(avg, 4), 'Trigger %': round(trigger_pct, 4),
                    'Trigger Level': round(level, 4), 'Signal Time': str(g.loc[signal, 'datetime']),
                    'Entry Time': str(g.loc[entry_i, 'datetime']), 'Exit Time': str(g.loc[exit_i, 'datetime']),
                    'Entry': entry, 'Exit': exit_price, 'Exit Reason': reason,
                    'Gross Return %': round(gross, 5),
                    'Net Return %': round(gross - ROUND_TRIP_COST_BPS / 100, 5),
                    'Bars Held': exit_i - entry_i + 1
                })
    return trades, skipped_sessions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not QUALITY_REPORT.exists():
        raise FileNotFoundError(f'Quality report not found: {QUALITY_REPORT}. Move your CSV there or update QUALITY_REPORT.')
    if not DAILY_DIR.is_dir() or not FIVE_MIN_DIR.is_dir():
        raise FileNotFoundError(f'Expected daily CSVs in {DAILY_DIR} and five-minute CSVs in {FIVE_MIN_DIR}')
    quality = pd.read_csv(QUALITY_REPORT)
    missing_col = 'Missing Interior 5-min Slots'
    if not {'Ticker', missing_col}.issubset(quality.columns):
        raise ValueError(f'Quality report needs Ticker and {missing_col} columns')
    quality[missing_col] = pd.to_numeric(quality[missing_col], errors='coerce')
    eligible = sorted(set(quality.loc[quality[missing_col].between(0, MAX_MISSING_SLOTS), 'Ticker'].astype(str).str.upper()))
    print(f'Eligible tickers (<10 missing slots): {len(eligible)}', flush=True)
    trades = []
    errors = []
    skipped_sessions = 0
    for idx, ticker in enumerate(eligible, 1):
        five = FIVE_MIN_DIR / f'{ticker}.csv'
        daily = DAILY_DIR / f'{ticker}.csv'
        if not five.exists() or not daily.exists():
            errors.append({'Ticker': ticker, 'Reason': 'Missing 5min or daily CSV'})
            continue
        try:
            new_trades, skipped = backtest_symbol(ticker, five, daily)
            trades.extend(new_trades)
            skipped_sessions += skipped
        except Exception as exc:
            errors.append({'Ticker': ticker, 'Reason': str(exc)})
        if idx % 25 == 0 or idx == len(eligible):
            print(f'Processed {idx}/{len(eligible)} | trades={len(trades)} | errors={len(errors)}', flush=True)
    pd.DataFrame(errors, columns=['Ticker', 'Reason']).to_csv(OUT_DIR / 'Backtest_Skipped_Tickers.csv', index=False)
    if not trades:
        print('No trades generated. Check the skipped ticker report and historical coverage.')
        return
    df = pd.DataFrame(trades).sort_values(['Date', 'Ticker', 'Method', 'Direction'])
    df.to_csv(OUT_DIR / 'Trades_Baseline.csv', index=False)
    summary = df.groupby(['Method', 'Direction']).agg(
        Trades=('Net Return %', 'size'),
        Win_Rate_Pct=('Net Return %', lambda x: round((x > 0).mean() * 100, 2)),
        Avg_Net_Return_Pct=('Net Return %', 'mean'),
        Median_Net_Return_Pct=('Net Return %', 'median'),
        Sum_Net_Return_Pct=('Net Return %', 'sum'),
        Unique_Tickers=('Ticker', 'nunique'),
    ).reset_index().round(4)
    summary.to_csv(OUT_DIR / 'Strategy_Comparison_Baseline.csv', index=False)
    by_ticker = df.groupby(['Ticker', 'Method', 'Direction']).agg(
        Trades=('Net Return %', 'size'),
        Win_Rate_Pct=('Net Return %', lambda x: (x > 0).mean() * 100),
        Avg_Net_Return_Pct=('Net Return %', 'mean'),
        Sum_Net_Return_Pct=('Net Return %', 'sum'),
    ).reset_index().round(4)
    by_ticker.to_csv(OUT_DIR / 'Ticker_Strategy_Results.csv', index=False)
    print('\n' + summary.to_string(index=False))
    print(f'\nSkipped incomplete sessions: {skipped_sessions}; errors: {len(errors)}')
    print(f'Reports saved in: {OUT_DIR}')
    print('RESEARCH ONLY: four overlapping hypothetical strategy variants; not a combined portfolio return.')
    print('No option premiums, slippage modeling, borrow constraints, or historical rolling universe selection.')


if __name__ == '__main__':
    main()
