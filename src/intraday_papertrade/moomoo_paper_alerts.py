"""Moomoo US stocks: alerts + guarded SIMULATE-only paper execution.

Run from PyCharm while OpenD is running. Requires: pip install moomoo-api pandas
Config CSV next to this script: watchlist.csv (ticker,decline_pct,enabled).

Important: decline_pct is a manually supplied threshold (not recalculated).
For safety ENABLE_PAPER_ORDERS defaults False until strategy parity is verified.
Duplicate prevention: one eligible buy alert/entry attempt per ticker per ET day, persisted before action. This prototype checks threshold on completed 5m bars, and places at the next
5m boundary when polling sees a new bar. Not guaranteed to fill at bar open.
"""
import csv
import json
import logging
import math
import os
import time
import tempfile
from datetime import datetime, time as dtime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from moomoo import (OpenQuoteContext, OpenSecTradeContext, TrdMarket, TrdEnv,
                    SubType, KLType, AuType, TrdSide, OrderType, RET_OK)

BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / 'output' / 'intraday_volatility'
WATCHLIST = OUTPUT / 'watchlist.csv'
STATE_FILE = OUTPUT / 'paper_state.json'
EVENTS_FILE = OUTPUT / 'paper_events.csv'
ORDERS_FILE = OUTPUT / 'paper_order_updates.csv'
SIGNALS_FILE = OUTPUT / 'paper_signals.csv'
DIAGNOSTICS_FILE = OUTPUT / 'paper_candle_diagnostics.csv'
EARNINGS_FILE = OUTPUT / 'earnings_calendar.csv'
EARNINGS_BLACKOUT_DAYS = 4  # Earnings day + four calendar days on each side
HOST, PORT = '127.0.0.1', 11111
SIM_ACCOUNT_ID = 1397202  # ID from your verified SIMULATE account; never use REAL
ENABLE_PAPER_ORDERS = True  # SAFETY: verify corrected timing before SIMULATE orders.
USD_PER_TRADE = 1000
MAX_OPEN_POSITIONS = 15
TRAIL_ACTIVATE_GAIN = 0.01  # Activate only at +1% over actual position cost
TRAIL_DISTANCE = 0.01       # Trail 1% below highest observed price
STOP_LOSS = 0.01            # Original hard stop stays active
POLL_SECONDS = 15
ET = ZoneInfo('America/New_York')
LOG = logging.getLogger('moomoo_paper')


def log_event(kind, ticker='', details=''):
    first = not EVENTS_FILE.exists()
    with EVENTS_FILE.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if first:
            w.writerow(['time_et', 'event', 'ticker', 'details'])
        w.writerow([datetime.now(ET).isoformat(), kind, ticker, str(details)])
    LOG.info('%s %s %s', kind, ticker, details)


def append_csv(path, fields, record):
    first = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if first:
            writer.writeheader()
        writer.writerow(record)


def signal_record(ticker, bar_time, low, threshold, decline, status, qty='', price=''):
    append_csv(SIGNALS_FILE,
               ['time_et', 'ticker', 'signal_bar_et', 'bar_low', 'trigger_price',
                'decline_pct', 'decision', 'qty', 'reference_price'],
               dict(time_et=datetime.now(ET).isoformat(), ticker=ticker,
                    signal_bar_et=bar_time, bar_low=low, trigger_price=threshold,
                    decline_pct=decline, decision=status, qty=qty,
                    reference_price=price))


def order_updates(trade, state):
    ret, df = trade.order_list_query(trd_env=TrdEnv.SIMULATE, acc_id=SIM_ACCOUNT_ID)
    if ret != RET_OK:
        raise RuntimeError(f'Paper order reconciliation failed: {df}')
    if df.empty:
        return df
    for _, r in df.iterrows():
        oid = str(r.get('order_id', ''))
        if not oid or oid == 'nan':
            continue
        status = str(r.get('order_status', ''))
        filled = str(r.get('dealt_qty', ''))
        avg = str(r.get('dealt_avg_price', ''))
        fingerprint = '|'.join((status, filled, avg))
        if state['last_orders'].get(oid) == fingerprint:
            continue
        state['last_orders'][oid] = fingerprint
        append_csv(ORDERS_FILE,
                   ['observed_at_et', 'order_id', 'code', 'side', 'status',
                    'order_qty', 'order_price', 'filled_qty', 'avg_fill_price'],
                   dict(observed_at_et=datetime.now(ET).isoformat(), order_id=oid,
                        code=r.get('code', ''), side=r.get('trd_side', ''),
                        status=status, order_qty=r.get('qty', ''),
                        order_price=r.get('price', ''), filled_qty=filled,
                        avg_fill_price=avg))
        log_event('ORDER_STATUS', str(r.get('code', '')), f'id={oid} status={status} filled={filled} avg={avg}')
    return df


def load_watchlist():
    if not WATCHLIST.exists():
        raise FileNotFoundError(f'Missing {WATCHLIST}. Use the included template.')
    out = {}
    with WATCHLIST.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if str(row.get('enabled', '')).strip().lower() not in ('1', 'true', 'yes'):
                continue
            ticker = str(row['ticker']).strip().upper()
            if ticker.startswith('US.'):
                ticker = ticker[3:]
            if ticker.endswith('.US'):
                ticker = ticker[:-3]
            decline = float(row['decline_pct'])
            if not ticker.isalnum() or not (0 < decline < 30):
                raise ValueError(f'Invalid ticker/decline_pct: {row}')
            if ticker in out:
                raise ValueError(f'Duplicate ticker: {ticker}')
            out[ticker] = decline
    if len(out) > 62:
        raise ValueError('Limit this initial test to 62 tickers')
    if not out:
        raise ValueError('No enabled tickers in watchlist.csv')
    return out


def load_earnings_calendar():
    """Read CSV produced by update_earnings_calendar.py (ticker,earnings_date).

    Fail closed when file is missing, empty, malformed or contains invalid dates.
    A symbol absent from a valid calendar is assumed to have no earnings in the
    downloaded date range; ensure the updater covers today's +/- 4 days.
    """
    if not EARNINGS_FILE.is_file():
        raise FileNotFoundError(
            f'Missing {EARNINGS_FILE}. Run update_earnings_calendar.py first; '
            'earnings protection cannot be verified.'
        )
    result = {}
    with EARNINGS_FILE.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {'ticker', 'earnings_date'}.issubset(reader.fieldnames):
            raise ValueError('Earnings CSV must have ticker,earnings_date columns')
        count = 0
        for line_number, row in enumerate(reader, 2):
            symbol = (row.get('ticker') or '').strip().upper().removeprefix('US.')
            day = (row.get('earnings_date') or '').strip()
            if not symbol or not day:
                raise ValueError(f'Incomplete earnings CSV row {line_number}: {row}')
            try:
                report_date = date.fromisoformat(day)
            except ValueError as exc:
                raise ValueError(f'Invalid earnings date on CSV row {line_number}: {day}') from exc
            result.setdefault(symbol, set()).add(report_date)
            count += 1
    if count == 0:
        raise ValueError('Earnings CSV has no records; refusing to issue alerts')
    return result, count


def earnings_blackout(ticker, trading_day, calendar):
    """Return earnings date when within inclusive 9-day blackout, else None."""
    for earnings_day in sorted(calendar.get(ticker, ())):
        if abs((trading_day - earnings_day).days) <= EARNINGS_BLACKOUT_DAYS:
            return earnings_day
    return None


def load_state():
    if not STATE_FILE.exists():
        return {'last_bar': {}, 'pending': {}, 'entered_on': {}, 'last_orders': {}, 'alerted_on': {}, 'entry_attempts': {}, 'trailing_positions': {}, 'exit_attempts': {}}
    state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    for key in ('last_bar', 'pending', 'entered_on', 'last_orders', 'alerted_on', 'entry_attempts', 'trailing_positions', 'exit_attempts'):
        state.setdefault(key, {})
    return state


def save_state(state):
    # Unique temporary filename prevents collisions with other runs or old .tmp files.
    # On Windows, antivirus/sync tools or an editor can briefly lock the destination.
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='paper_state_', suffix='.tmp', dir=STATE_FILE.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp_name, STATE_FILE)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except PermissionError:
                pass


def checked_sim_account(trade):
    ret, accounts = trade.get_acc_list()
    if ret != RET_OK:
        raise RuntimeError(f'Account lookup failed: {accounts}')
    rows = accounts[accounts['acc_id'].astype(str) == str(SIM_ACCOUNT_ID)]
    if len(rows) != 1 or str(rows.iloc[0]['trd_env']).upper() != 'SIMULATE':
        raise RuntimeError('FAIL CLOSED: configured account is not uniquely SIMULATE')
    if str(rows.iloc[0]['acc_status']).upper() != 'ACTIVE':
        raise RuntimeError('FAIL CLOSED: simulated account not ACTIVE')
    if 'US' not in str(rows.iloc[0]['trdmarket_auth']):
        raise RuntimeError('FAIL CLOSED: simulated account lacks US market authorization')
    return SIM_ACCOUNT_ID


def positions(trade):
    ret, df = trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=SIM_ACCOUNT_ID)
    if ret != RET_OK:
        raise RuntimeError(f'Cannot reconcile paper positions: {df}')
    return df


def pending_orders(trade):
    ret, df = trade.order_list_query(trd_env=TrdEnv.SIMULATE, acc_id=SIM_ACCOUNT_ID)
    if ret != RET_OK:
        raise RuntimeError(f'Cannot reconcile paper orders: {df}')
    return df


def paper_order(trade, code, qty, side, price):
    if not ENABLE_PAPER_ORDERS:
        log_event('ALERT_ONLY_ORDER_DISABLED', code, f'{side} qty={qty} price={price}')
        return False
    checked_sim_account(trade)  # Re-validate before EVERY order
    # LIMIT order avoids uncertain support for simulated MARKET orders.
    # Price is a limit; fills are not guaranteed. No REAL environment exists here.
    ret, result = trade.place_order(
        price=round(price, 2), qty=int(qty), code=code, trd_side=side,
        order_type=OrderType.NORMAL, trd_env=TrdEnv.SIMULATE,
        acc_id=SIM_ACCOUNT_ID,
    )
    if ret != RET_OK:
        log_event('ORDER_REJECTED', code, result)
        return False
    log_event('PAPER_ORDER_SUBMITTED', code, result)
    return True


def regular_hours(now):
    return now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    watch = load_watchlist()
    state = load_state()
    earnings_calendar, earnings_count = load_earnings_calendar()
    LOG.info('Loaded %s earnings records from %s', earnings_count, EARNINGS_FILE)
    quote = OpenQuoteContext(host=HOST, port=PORT)
    trade = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=HOST, port=PORT)
    codes = ['US.' + ticker for ticker in watch]
    try:
        checked_sim_account(trade)
        ret, result = quote.subscribe(codes, [SubType.K_5M], subscribe_push=False)
        if ret != RET_OK:
            raise RuntimeError(f'Quote subscription failed: {result}')
        log_event('START', details=f'{len(codes)} tickers; paper_orders={ENABLE_PAPER_ORDERS}; earnings_rows={earnings_count}')
        while True:
            now = datetime.now(ET)
            if not regular_hours(now):
                time.sleep(30)
                continue
            # Fail closed on broker reconciliation failure.
            pos = positions(trade)
            orders = order_updates(trade, state)
            # Do not submit orders when unknown pending orders exist.
            active_statuses = {'SUBMITTING', 'SUBMITTED', 'WAITING_SUBMIT', 'PART_FILLED', 'PARTIALLY_FILLED', 'SUBMITTING_TO_EXCHANGE', 'WAITING_SUBMIT_TO_EXCHANGE'}
            active_codes = set()
            if not orders.empty and 'order_status' in orders:
                active_codes = set(orders.loc[
                    orders['order_status'].astype(str).str.upper().isin(active_statuses), 'code'
                ].astype(str))
            owned = {}
            if not pos.empty:
                for _, p in pos.iterrows():
                    qty = int(float(p.get('qty', 0)))
                    if qty > 0:
                        owned[str(p['code'])] = (qty, float(p.get('cost_price', 0)))
            # Exit manager: use actual SIMULATE positions, not alert reference prices.
            # State is saved before any sell request, so a crash cannot cause blind retries.
            ret, snap = quote.get_market_snapshot(codes)
            if ret != RET_OK:
                log_event('SNAPSHOT_ERROR', details=snap)
                time.sleep(POLL_SECONDS)
                continue
            prices = {str(r['code']): float(r['last_price']) for _, r in snap.iterrows()}
            session_date = now.date().isoformat()
            # Drop obsolete peak tracking only after the position is no longer held.
            # Retain exit_attempts for reconciliation / manual review.
            for stale_code in list(state['trailing_positions']):
                if stale_code not in owned:
                    del state['trailing_positions'][stale_code]
            for code, (qty, cost) in owned.items():
                px = prices.get(code)
                if px is None or not math.isfinite(px) or px <= 0 or cost <= 0:
                    continue
                tracking = state['trailing_positions'].get(code)
                # A position can change quantity or average cost due to fills.
                # Reset tracking rather than reuse a peak from a different cost basis.
                if (not isinstance(tracking, dict)
                        or tracking.get('qty') != qty
                        or abs(float(tracking.get('cost', 0)) - cost) > 0.005):
                    tracking = {'qty': qty, 'cost': cost, 'high': cost,
                                'active': False, 'date': session_date}
                    state['trailing_positions'][code] = tracking
                    log_event('TRAIL_TRACK_START', code, f'cost={cost:.4f} qty={qty}')
                # Keep peak tracking across sessions; no end-of-day forced exit.
                # Only prices observed after tracking starts contribute to the peak.
                # Snapshot polling can miss intrapoll spikes; execution is not guaranteed.
                if px > tracking['high']:
                    tracking['high'] = px
                if not tracking['active'] and tracking['high'] >= cost * (1 + TRAIL_ACTIVATE_GAIN):
                    tracking['active'] = True
                    log_event('TRAIL_ACTIVATED', code,
                              f'cost={cost:.4f} high={tracking["high"]:.4f}')
                hard_stop = cost * (1 - STOP_LOSS)
                active_stop = (max(cost, tracking['high'] * (1 - TRAIL_DISTANCE))
                               if tracking['active'] else hard_stop)
                log_event('EXIT_MONITOR', code,
                          f'last={px:.4f} high={tracking["high"]:.4f} '
                          f'trail_active={tracking["active"]} stop={active_stop:.4f}')
                reason = None
                if px <= active_stop:
                    reason = 'TRAIL' if tracking['active'] else 'HARD_STOP'
                if reason is None:
                    continue
                if code in active_codes:
                    log_event('SKIP_EXIT_PENDING_ORDER', code, f'reason={reason}')
                    continue
                # Persist an exit intent BEFORE placing an order. If an order is
                # rejected/uncertain, inspect the broker account before retrying.
                prior = state['exit_attempts'].get(code)
                if prior and prior.get('date') == session_date and prior.get('qty') == qty:
                    log_event('SKIP_EXIT_ALREADY_ATTEMPTED', code, f'reason={reason}')
                    continue
                log_event('EXIT_ALERT_' + reason, code,
                          f'last={px:.4f} stop={active_stop:.4f} qty={qty}')
                if ENABLE_PAPER_ORDERS:
                    state['exit_attempts'][code] = {
                        'date': session_date, 'qty': qty, 'cost': cost,
                        'reason': reason, 'status': 'SUBMISSION_UNKNOWN'}
                    save_state(state)
                    # Aggressive limit is not a stop order; it may remain unfilled.
                    ok = paper_order(trade, code, qty, TrdSide.SELL, max(0.01, px * 0.997))
                    if ok:
                        state['exit_attempts'][code]['status'] = 'SUBMITTED'
                        active_codes.add(code)
                else:
                    log_event('ALERT_ONLY_EXIT', code,
                              f'reason={reason} last={px:.4f} stop={active_stop:.4f}')
            save_state(state)
            for ticker, decline in watch.items():
                code = 'US.' + ticker
                ret, bars = quote.get_cur_kline(code, 5, KLType.K_5M, AuType.QFQ)
                if ret != RET_OK or len(bars) < 3:
                    log_event('BAR_ERROR', ticker, bars if ret != RET_OK else 'insufficient bars')
                    continue
                bars = bars.sort_values('time_key')
                # Moomoo can expose a future-dated bar in the last row.
                # Select a bar only when its five-minute interval has actually ended.
                observed_now = datetime.now(ET)
                bar_starts = pd.to_datetime(bars['time_key'], errors='coerce')
                if bar_starts.isna().any():
                    log_event('BAR_TIME_ERROR', ticker, 'Unparseable candle timestamp')
                    continue
                elapsed = (pd.Timestamp(observed_now.replace(tzinfo=None)) - bar_starts).dt.total_seconds()
                eligible = bars.loc[elapsed >= 300]
                if eligible.empty:
                    log_event('WAIT_FOR_COMPLETED_BAR', ticker, 'No five-minute candle closed yet')
                    continue
                completed = eligible.iloc[-1]
                bar_time = str(completed['time_key'])
                bar_age_seconds = (observed_now - pd.Timestamp(bar_time).tz_localize(ET).to_pydatetime()).total_seconds()
                # Allow up to one bar of delay after close; do not replay older signals.
                if not (300 <= bar_age_seconds < 600):
                    log_event('SKIP_STALE_COMPLETED_BAR', ticker,
                              f'completed={bar_time} age_seconds={bar_age_seconds:.0f}')
                    continue
                current_time = observed_now.strftime('%Y-%m-%d %H:%M:%S')
                age_seconds = bar_age_seconds - 300
                last_seen = state['last_bar'].get(ticker)
                if last_seen == bar_time:
                    continue
                state['last_bar'][ticker] = bar_time
                # Previous daily close from the snapshot's prev_close_price field.
                snap_row = snap.loc[snap['code'] == code]
                if snap_row.empty or 'prev_close_price' not in snap_row:
                    log_event('NO_PREV_CLOSE', ticker)
                    continue
                prev_close = float(snap_row.iloc[0]['prev_close_price'])
                if not math.isfinite(prev_close) or prev_close <= 0:
                    continue
                trigger = prev_close * (1 - decline / 100)
                bar_low = float(completed['low'])
                session_date = now.date().isoformat()
                already_entered = state['entered_on'].get(ticker) == session_date
                already_alerted = state['alerted_on'].get(ticker) == session_date
                already_attempted = state['entry_attempts'].get(ticker, {}).get('date') == session_date
                append_csv(DIAGNOSTICS_FILE,
                           ['observed_at_et', 'ticker', 'completed_bar_et', 'current_bar_et',
                            'prev_close', 'decline_pct', 'threshold', 'completed_low',
                            'threshold_hit', 'already_entered', 'held', 'pending', 'age_seconds'],
                           dict(observed_at_et=datetime.now(ET).isoformat(), ticker=ticker,
                                completed_bar_et=bar_time, current_bar_et=current_time,
                                prev_close=prev_close, decline_pct=decline,
                                threshold=round(trigger, 4), completed_low=bar_low,
                                threshold_hit=bar_low <= trigger, already_entered=already_entered,
                                held=code in owned, pending=code in active_codes,
                                age_seconds=round(age_seconds, 1)))
                if bar_low <= trigger:
                    blackout_date = earnings_blackout(ticker, now.date(), earnings_calendar)
                    if blackout_date is not None:
                        log_event('SKIP_EARNINGS_FILTER', ticker,
                                  f'earnings={blackout_date} session={session_date} bar={bar_time}')
                        signal_record(ticker, bar_time, bar_low, trigger, decline,
                                      'SKIP_EARNINGS_FILTER')
                    elif already_entered or already_alerted or already_attempted:
                        log_event('SKIP_DUPLICATE_DAILY_SIGNAL', ticker, f'bar={bar_time}')
                    elif code in owned or code in active_codes:
                        log_event('SKIP_ALREADY_HELD_OR_PENDING', ticker)
                        signal_record(ticker, bar_time, bar_low, trigger, decline, 'SKIP_HELD_OR_PENDING')
                    elif len(set(owned) | active_codes) >= MAX_OPEN_POSITIONS:
                        log_event('SKIP_POSITION_LIMIT', ticker)
                        signal_record(ticker, bar_time, bar_low, trigger, decline, 'SKIP_POSITION_LIMIT')
                    else:
                        px = prices.get(code)
                        if px and math.isfinite(px) and px > 0:
                            qty = math.floor(USD_PER_TRADE / (px * 1.003))
                            if qty >= 1:
                                # Persist the day's alert BEFORE logging or submitting an order.
                                # A restart will not repeat the alert or an uncertain order attempt.
                                state['alerted_on'][ticker] = session_date
                                save_state(state)
                                signal_record(ticker, bar_time, round(bar_low, 4), round(trigger, 4), decline, 'TRIGGERED')
                                log_event('LONG_SIGNAL', ticker,
                                          f'bar={bar_time} low={bar_low:.2f} threshold={trigger:.2f} decline={decline:.3f}%')
                                log_event('BUY_ALERT', ticker, f'qty={qty} approx_price={px:.2f} current_bar={current_time}')
                                signal_record(ticker, bar_time, bar_low, trigger, decline,
                                              'BUY_ATTEMPT' if ENABLE_PAPER_ORDERS else 'ALERT_ONLY', qty, px)
                                if ENABLE_PAPER_ORDERS:
                                    # Fail closed: write intent before broker call; do not automatically
                                    # retry after a crash/timeout because submission might have succeeded.
                                    state['entry_attempts'][ticker] = {'date': session_date, 'bar': bar_time,
                                                                       'qty': qty, 'status': 'SUBMISSION_UNKNOWN'}
                                    save_state(state)
                                if paper_order(trade, code, qty, TrdSide.BUY, px * 1.003):
                                    state['entered_on'][ticker] = session_date
                                    state['entry_attempts'][ticker]['status'] = 'SUBMITTED'
                                    active_codes.add(code)
                                    signal_record(ticker, bar_time, bar_low, trigger, decline, 'ORDER_SUBMITTED', qty, px)
                save_state(state)
            save_state(state)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log_event('STOP', details='Stopped by user')
    finally:
        save_state(state)
        quote.close()
        trade.close()


if __name__ == '__main__':
    run()
