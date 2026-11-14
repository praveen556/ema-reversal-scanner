"""Point-in-time rebuild of the V2 scanner signals on local daily data.

For every ticker and every historical day, recomputes the scanner's signal,
entry stage and diagnostics using only data available on that day, then
attaches forward outcomes (entry at the NEXT session's open).

Run: python src/research/point_in_time_backtest.py
Writes output/research/pit_events.pkl and output/research/pit_universe.pkl.
No broker connectivity. Read-only on data/daily.

Known limits: today's ticker list only (delisted names missing, which
flatters results); daily bars cannot resolve intrabar order (stop assumed
first); no earnings filter; costs are applied in the analysis step.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / 'data' / 'daily'
OUT = ROOT / 'output' / 'research'
OUT.mkdir(parents=True, exist_ok=True)

TOL, LOOKBACK = 1.5, 5
MIN_PRICE, MIN_VOL = 10.0, 1_000_000
NEAR_MAX, EXT_MIN = 3.0, 8.0


def load(path):
    d = pd.read_csv(path)
    d.columns = [str(c).strip().lower() for c in d.columns]
    dc = 'date' if 'date' in d else ('datetime' if 'datetime' in d else None)
    if dc is None or not {'open', 'high', 'low', 'close', 'volume'} <= set(d.columns):
        return None
    d['date'] = pd.to_datetime(d[dc].astype(str).str[:10], errors='coerce')
    for c in ('open', 'high', 'low', 'close', 'volume'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
    d = d[(d[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    d = d.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
    return d if len(d) >= 230 else None


def sh(a, k):
    """a[t+k] with NaN padding."""
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[:len(a) - k] = a[k:]
    return out


def lag(a, k):
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:len(a) - k]
    return out


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def bracket(O, H, L, C, tp, sl, hold):
    """Entry next open; stop-first; gap-aware; time exit at close of day `hold`."""
    entry = sh(O, 1)
    stop, tgt = entry * (1 - sl), entry * (1 + tp)
    ret = np.full(len(O), np.nan)
    done = np.zeros(len(O), bool)
    for k in range(1, hold + 1):
        o, h, l = sh(O, k), sh(H, k), sh(L, k)
        live = ~done & ~np.isnan(o)
        for cond, val in (
            ((o <= stop) & (k > 1), o / entry - 1),
            (l <= stop, stop / entry - 1),
            ((o >= tgt) & (k > 1), o / entry - 1),
            (h >= tgt, tgt / entry - 1),
        ):
            hit = live & ~done & cond
            ret[hit] = val[hit] if isinstance(val, np.ndarray) else val
            done |= hit
    last = sh(C, hold)
    rest = ~done
    ret[rest] = last[rest] / entry[rest] - 1
    ret[np.isnan(last)] = np.nan  # require the full window to exist
    return ret


def process(ticker, d, spy_ret20):
    O, H, L, C, V = (d[c].to_numpy(float) for c in ('open', 'high', 'low', 'close', 'volume'))
    close = d['close']
    e20, e50, e89, e200 = (ema(close, n).to_numpy() for n in (20, 50, 89, 200))
    r = rsi(close).to_numpy()
    avgvol = d['volume'].rolling(20).mean().to_numpy()
    prev = lag(C, 1)
    tr = np.nanmax(np.vstack([H - L, np.abs(H - prev), np.abs(L - prev)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    n = len(d)
    idx = np.arange(n)

    def last_touch(e):
        touch = (np.abs(L - e) / e * 100 <= TOL) & (C > e)
        lt = pd.Series(np.where(touch, idx, np.nan)).ffill().to_numpy()
        ago = idx - lt
        ago[~(ago <= LOOKBACK - 1)] = np.nan
        return ago

    ago89, ago200 = last_touch(e89), last_touch(e200)
    liquid = (C > MIN_PRICE) & (avgvol > MIN_VOL) & (idx >= 200)
    s89 = ~np.isnan(ago89) & (C > e89) & liquid
    s200 = ~np.isnan(ago200) & (C > e200) & liquid
    sig = s89 | s200
    ago = np.where(s89 & s200, np.fmin(ago89, ago200), np.where(s89, ago89, ago200))
    ago = np.where(sig, ago, np.nan)

    rev = np.where(sig, idx - np.nan_to_num(ago), 0).astype(int)
    rsi_chg = r - r[rev]
    px_chg = (C / C[rev] - 1) * 100
    stage = np.full(n, '', dtype=object)
    stage[sig & (ago == 0)] = 'EARLY REVERSAL'
    conf = sig & (ago > 0) & (rsi_chg > 0) & (px_chg > 0)
    stage[conf] = 'CONFIRMED'
    stage[conf & (px_chg <= NEAR_MAX)] = 'CONFIRMED NEAR ENTRY'
    stage[conf & (px_chg >= EXT_MIN)] = 'EXTENDED'
    stage[sig & (ago > 0) & (rsi_chg < 0) & (px_chg < 0)] = 'LOSING MOMENTUM'
    stage[sig & (stage == '')] = 'MIXED'
    cne = stage == 'CONFIRMED NEAR ENTRY'
    grp = np.cumsum(~cne)
    streak = pd.Series(cne.astype(int)).groupby(grp).cumsum().to_numpy()

    hi20 = d['high'].rolling(20).max().to_numpy()
    lo20 = d['low'].rolling(20).min().to_numpy()
    range20 = (hi20 - lo20) / lo20 * 100
    emas = np.vstack([e20, e50, e89, e200])
    compression = (emas.max(0) - emas.min(0)) / C * 100
    a89, a200 = pd.Series(C > e89), pd.Series(C > e200)
    crosses = ((a89 != a89.shift()).rolling(19).sum() + (a200 != a200.shift()).rolling(19).sum()).to_numpy()
    choppy = (compression < 3.0) & (crosses >= 8)
    hi120 = d['high'].rolling(120).max().to_numpy()

    entry = sh(O, 1)
    fwd = {f'r{k}': sh(C, k) / entry - 1 for k in (3, 5, 10, 20)}
    mfe10 = np.nanmax(np.vstack([sh(H, k) for k in range(1, 11)]), axis=0) / entry - 1
    mae10 = np.nanmin(np.vstack([sh(L, k) for k in range(1, 11)]), axis=0) / entry - 1
    atrp = atr / C

    uni = pd.DataFrame({'date': d['date'], **fwd})[liquid]

    if not sig.any():
        return None, uni

    # Trend-following exit (RSA-style) only for signal rows.
    te_ret = np.full(n, np.nan)
    te_days = np.full(n, np.nan)
    for t in np.flatnonzero(sig):
        e = t + 1
        if e >= n or np.isnan(atr[t]):
            continue
        ent = O[e]
        stop = ent - 1.5 * atr[t]
        below = 0
        out = None
        for j in range(e, min(e + 60, n)):
            if j > e and O[j] <= stop:
                out = (O[j], j); break
            if L[j] <= stop:
                out = (stop, j); break
            below = below + 1 if C[j] < e20[j] else 0
            if below >= 2 or j == e + 59:
                if j + 1 < n:
                    out = (O[j + 1], j + 1)
                break
        if out:
            te_ret[t] = out[0] / ent - 1
            te_days[t] = out[1] - e + 1

    ret20 = C / lag(C, 20) - 1
    ev = pd.DataFrame({
        'date': d['date'], 'ticker': ticker,
        'signal': np.where(s89 & s200, 'BOTH', np.where(s89, 'EMA89', 'EMA200')),
        'days_ago': ago, 'stage': stage, 'cne_streak': streak,
        'first_day': sig & ~np.r_[False, sig[:-1]],
        'price': C, 'rsi': r, 'rsi_rev': r[rev], 'rsi_chg': rsi_chg, 'px_chg': px_chg,
        'range20': range20, 'compression': compression, 'crosses': crosses, 'choppy': choppy,
        'above20': C > e20, 'above50': C > e50, 'above89': C > e89, 'above200': C > e200,
        'e50_gt_200': e50 > e200, 'e20_gt_50': e20 > e50,
        'slope50': (e50 / lag(e50, 5) - 1) * 100, 'slope200': (e200 / lag(e200, 5) - 1) * 100,
        'dist89': (C / e89 - 1) * 100, 'dist200': (C / e200 - 1) * 100,
        'volratio': V / avgvol, 'rev_volratio': V[rev] / avgvol[rev],
        'close_pos': np.where(H > L, (C - L) / (H - L) * 100, 50.0),
        'bull_candle': C > O, 'day_ret': (C / prev - 1) * 100,
        'atr_pct': atrp * 100, 'ret20': ret20 * 100, 'ret60': (C / lag(C, 60) - 1) * 100,
        'rs20': (ret20 - spy_ret20.reindex(d['date']).to_numpy()) * 100,
        'off_high120': (C / hi120 - 1) * 100,
        'dollar_vol': C * avgvol,
        'gap_next': (entry / C - 1) * 100,
        **fwd, 'mfe10': mfe10, 'mae10': mae10,
        'b_4_3_5': bracket(O, H, L, C, 0.04, 0.03, 5),
        'b_6_3_10': bracket(O, H, L, C, 0.06, 0.03, 10),
        'b_8_4_10': bracket(O, H, L, C, 0.08, 0.04, 10),
        'b_atr_2_1_10': bracket(O, H, L, C, 2.0 * atrp, 1.0 * atrp, 10),
        'b_atr_3_1p5_10': bracket(O, H, L, C, 3.0 * atrp, 1.5 * atrp, 10),
        'b_atr_3_1p5_20': bracket(O, H, L, C, 3.0 * atrp, 1.5 * atrp, 20),
        'trend_ret': te_ret, 'trend_days': te_days,
    })[sig]
    return ev, uni


def main():
    global DAILY
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily-dir', type=Path, default=DAILY)
    ap.add_argument('--tag', default='pit', help='output file prefix')
    a = ap.parse_args()
    DAILY = a.daily_dir if a.daily_dir.is_absolute() else ROOT / a.daily_dir
    t0 = time.time()
    spy = load(DAILY / 'SPY.csv')
    if spy is None:
        sys.exit('SPY.csv missing or too short')
    sc = spy.set_index('date')['close']
    spy_ret20 = sc / sc.shift(20) - 1
    e = {n: ema(sc, n) for n in (20, 50, 200)}
    so = spy.set_index('date')['open']
    market = pd.DataFrame({
        'spy_above20': sc > e[20], 'spy_above50': sc > e[50], 'spy_above200': sc > e[200],
        'spy_e20_rising': e[20] > e[20].shift(5), 'spy_ret5': (sc / sc.shift(5) - 1) * 100,
        'spy_ret20': spy_ret20 * 100,
        **{f'spy_r{k}': sc.shift(-k) / so.shift(-1) - 1 for k in (3, 5, 10, 20)},
    })
    events, universe = [], []
    files = sorted(DAILY.glob('*.csv'))
    for i, p in enumerate(files, 1):
        try:
            d = load(p)
            if d is None:
                continue
            ev, uni = process(p.stem.upper(), d, spy_ret20)
            if ev is not None:
                events.append(ev)
            if len(uni):
                universe.append(uni)
        except Exception as exc:  # keep going; report
            print('ERR', p.name, str(exc)[:120])
        if i % 500 == 0:
            print(f'{i}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    ev = pd.concat(events, ignore_index=True)
    uni = pd.concat(universe).groupby('date').mean()
    uni.columns = ['uni_' + c for c in uni.columns]
    ev = ev.join(market, on='date').join(uni, on='date')
    ev.to_pickle(OUT / f'{a.tag}_events.pkl')
    uni.to_pickle(OUT / f'{a.tag}_universe.pkl')
    print('events', len(ev), 'tickers', ev.ticker.nunique(), 'days', ev.date.nunique(),
          ev.date.min().date(), ev.date.max().date(), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
