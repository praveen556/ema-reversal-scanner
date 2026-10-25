"""Rank local daily OHLCV files, without modifying the EMA project data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
 'date':['date','datetime','timestamp','time'],
 'open':['open','o'], 'high':['high','h'], 'low':['low','l'],
 'close':['close','adjclose','adjustedclose','c'],
 'volume':['volume','vol','v']
}

def locate_column(df, field):
    normalized={''.join(c for c in str(col).lower() if c.isalnum()):col for col in df.columns}
    for alias in ALIASES[field]:
        if alias in normalized:return normalized[alias]
    return None

def read_daily(path:Path):
    df=pd.read_csv(path)
    # Supports CSVs with an unnamed date index, yfinance column headings, and mixed-case columns.
    if len(df)<20:return None
    date_col=locate_column(df,'date')
    if date_col is None:
        unnamed=[c for c in df.columns if str(c).startswith('Unnamed')]
        date_col=unnamed[0] if unnamed else None
    cols={k:locate_column(df,k) for k in ('open','high','low','close','volume')}
    if not date_col or any(cols[k] is None for k in ('open','high','low','close')):return None
    d=pd.DataFrame({'date':pd.to_datetime(df[date_col],errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()})
    for k,c in cols.items():
        d[k]=pd.to_numeric(df[c],errors='coerce') if c else np.nan
    d=d.dropna(subset=['date','open','high','low','close']).sort_values('date').drop_duplicates('date',keep='last')
    d=d[(d.open>0)&(d.high>=d.low)&(d.low>0)&(d.high>=d[['open','close']].max(axis=1))&(d.low<=d[['open','close']].min(axis=1))]
    return d if len(d)>=20 else None

def filtered_mean(values, multiplier=2.0):
    """One-pass upper-only filter: exclude values > multiplier * raw mean."""
    values = pd.to_numeric(pd.Series(values), errors='coerce')
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values >= 0]
    if values.empty:
        return np.nan, np.nan, 0, 0
    raw = float(values.mean())
    kept = values[values <= raw * multiplier]
    # With nonnegative values and multiplier >= 1, at least one value is retained.
    return float(kept.mean()), raw, int(len(values) - len(kept)), int(len(kept))

def score_ranks(df, rank_col):
    n=len(df)
    return 100*(n-df[rank_col])/(n-1) if n>1 else pd.Series([100.0]*n,index=df.index)

def build_rankings(data_dir:Path,out_dir:Path,lookback=20,min_price=30,
                   min_avg_volume=1_000_000,min_dollar_volume=100_000_000,
                   buffer=1.0,asof=None):
    out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; skipped=[]
    files=sorted(data_dir.rglob('*.csv'))
    cutoff=pd.Timestamp(asof).normalize() if asof else None
    for index, path in enumerate(files, start=1):
        if index % 500 == 0:
            print(f'Processed {index:,}/{len(files):,} files', flush=True)
        # Only accept files directly in daily data folders; do not consume scanner reports.
        if any(x in path.parts for x in ('5min','hourly','1hour','4hour','output')):continue
        ticker=path.stem.upper()
        if not ticker.replace('.','').replace('-','').isalnum():continue
        try:
            d=read_daily(path)
            if d is None:skipped.append((ticker,'invalid schema or fewer than 20 valid days'));continue
            if cutoff is not None:d=d[d.date<cutoff] # strict: previous completed sessions only
            if len(d)<lookback:skipped.append((ticker,'insufficient history'));continue
            recent=d.tail(lookback).copy()
            close=float(d.close.iloc[-1])
            avg_volume=float(recent.volume.mean()) if recent.volume.notna().all() else np.nan
            dollar=float((recent.close * recent.volume).mean()) if recent.volume.notna().all() else np.nan
            reasons=[]
            if close < min_price:
                reasons.append(f'Latest close ${close:.2f} < ${min_price:.2f}')
            if not np.isfinite(avg_volume):
                reasons.append('Missing/invalid 20D share volume')
            elif avg_volume < min_avg_volume:
                reasons.append(f'20D avg shares {avg_volume:,.0f} < {min_avg_volume:,.0f}')
            if not np.isfinite(dollar):
                reasons.append('Missing/invalid 20D dollar volume')
            elif dollar < min_dollar_volume:
                reasons.append(f'20D avg dollar volume ${dollar:,.0f} < ${min_dollar_volume:,.0f}')
            if reasons:
                skipped.append((ticker, '; '.join(reasons)))
                continue
            # All three measurements are INTRADAY, referenced to the same day's open.
            d['range_pct'] = (d.high - d.low) / d.open * 100
            d['up_pct'] = (d.high - d.open) / d.open * 100
            d['down_pct'] = (d.open - d.low) / d.open * 100
            recent = d.tail(lookback)

            metrics = {}
            for period, frame in [('20D', recent), ('History', d)]:
                for label, col in [('Range', 'range_pct'), ('Up', 'up_pct'), ('Down', 'down_pct')]:
                    avg, raw, excluded, valid = filtered_mean(frame[col])
                    metrics[f'Avg {label} {period} %'] = avg
                    metrics[f'Raw Avg {label} {period} %'] = raw
                    metrics[f'Excluded {label} Days {period}'] = excluded
                    metrics[f'Valid {label} Days {period}'] = valid

            avg20 = metrics['Avg Range 20D %']
            avgh = metrics['Avg Range History %']
            rows.append({
              'Ticker': ticker, 'Latest Close': close,
              **metrics,
              'Median Range 20D %': float(recent.range_pct.median()),
              'Median Range History %': float(d.range_pct.median()),
              'Max Daily Range %': float(d.range_pct.max()),
              'Volatility Change %': (avg20 / avgh - 1) * 100 if avgh else np.nan,
              'Entry Threshold 20D %': avg20 + buffer,
              'Entry Threshold History %': avgh + buffer,
              'Avg Share Volume 20D': avg_volume,
              'Avg Dollar Volume 20D': float(dollar), 'History Days': len(d),
              'First Date': str(d.date.iloc[0].date()),
              'Last Date': str(d.date.iloc[-1].date()),
              'Source File': str(path)
            })
        except Exception as exc:skipped.append((ticker,str(exc)[:180]))
    if not rows:
        pd.DataFrame(skipped, columns=['Ticker','Reason']).to_csv(out_dir/'Excluded_Tickers.csv',index=False)
        raise RuntimeError(f'No eligible daily OHLCV files found in {data_dir}; see Excluded_Tickers.csv. Files scanned: {len(files)}')
    r=pd.DataFrame(rows).sort_values('Ticker').drop_duplicates('Ticker',keep='first')
    r['Rank 20D']=r['Avg Range 20D %'].rank(method='min',ascending=False).astype(int)
    r['Rank History']=r['Avg Range History %'].rank(method='min',ascending=False).astype(int)
    r['Recent Score']=score_ranks(r,'Rank 20D')
    r['History Score']=score_ranks(r,'Rank History')
    r['Combined Score']=0.5*r['Recent Score']+0.5*r['History Score']
    r['Combined Rank']=r['Combined Score'].rank(method='min',ascending=False).astype(int)
    r=r.sort_values(['Combined Rank','Ticker']).reset_index(drop=True)
    r.to_csv(out_dir/'Intraday_Volatility_All_Tickers.csv',index=False)
    r.nsmallest(20,'Rank 20D').to_csv(out_dir/'Intraday_Volatility_Top20_Recent.csv',index=False)
    r.nsmallest(20,'Rank History').to_csv(out_dir/'Intraday_Volatility_Top20_History.csv',index=False)
    r.head(20).to_csv(out_dir/'Intraday_Volatility_Top20_Combined.csv',index=False)
    excluded=pd.DataFrame(skipped,columns=['Ticker','Reason'])
    excluded.to_csv(out_dir/'Excluded_Tickers.csv',index=False)
    excluded.to_csv(out_dir/'Skipped_Files.csv',index=False)
    with pd.ExcelWriter(out_dir/'Intraday_Volatility_Rankings.xlsx',engine='openpyxl') as x:
        for name,frame in [('All Tickers',r),('Top20 Combined',r.head(20)),('Top20 Recent',r.nsmallest(20,'Rank 20D')),('Top20 History',r.nsmallest(20,'Rank History')),('Excluded Tickers',excluded)]:
            frame.to_excel(x,sheet_name=name,index=False)
            ws=x.sheets[name];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
            from openpyxl.styles import Font,PatternFill
            for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D')
            for col in ws.columns:
                letter=col[0].column_letter
                ws.column_dimensions[letter].width=min(36,max(12,max(len(str(c.value or '')) for c in list(col)[:100])+2))
    print(f'Eligible tickers: {len(r)}; skipped: {len(skipped)}; output: {out_dir}')
    print(r[['Ticker','Rank 20D','Rank History','Combined Rank','Avg Range 20D %','Avg Range History %']].head(20).to_string(index=False))
    return r

if __name__ == '__main__':
    # Script location: <project>/src/intraday_volatility/rank_volatility.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / 'data' / 'daily'
    OUTPUT_DIR = PROJECT_ROOT / 'output' / 'intraday_volatility'

    build_rankings(
        DATA_DIR,
        OUTPUT_DIR,
        lookback=20,
        min_price=30,
        min_avg_volume=1_000_000,
        min_dollar_volume=100_000_000,
        buffer=1.0,
        asof=None,
    )
