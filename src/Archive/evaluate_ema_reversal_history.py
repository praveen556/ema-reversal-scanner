"""EMA reversal historical signal evaluator.

Install: pip install pandas openpyxl xlsxwriter
Run: python evaluate_ema_reversal_history.py
Default folders relative to this script: output/ (scanner files and report), data/ (quotes).

Assumptions: long-only signals, next-session open entry, 5% target, 3% stop,
5 trading sessions including entry session; stop-first for daily OHLC ambiguity.
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SIGNALS_DIR = BASE_DIR / 'output'
QUOTES_DIR = BASE_DIR / 'data'
REPORT_FILE = SIGNALS_DIR / 'EMA_Reversal_Backtest_Results.xlsx'

TARGET = .04
STOP = .03
HOLD_DAYS = 5
SUPPORTED = {'.csv', '.xlsx', '.xls'}
DATE_ALIASES = ['signal_date', 'scan_date', 'date', 'as_of_date', 'asofdate', 'run_date']
SYMBOL_ALIASES = ['symbol', 'ticker', 'stock', 'stock_symbol']
CLASS_ALIASES = ['latest_status', 'classification', 'category', 'signal', 'setup_class', 'priority', 'status', 'bucket', 'setup_type']
PRICE_ALIASES = {
    'date': ['date', 'datetime', 'timestamp', 'time'],
    'open': ['open', 'o'], 'high': ['high', 'h'],
    'low': ['low', 'l'], 'close': ['close', 'adj_close', 'adjusted_close', 'c'],
    'symbol': SYMBOL_ALIASES,
}

def norm(x):
    return re.sub(r'[^a-z0-9]+', '_', str(x).strip().lower()).strip('_')

def pick(df, names):
    mapping = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in mapping:
            return mapping[norm(name)]
    return None

def read_tables(path):
    if path.suffix.lower() == '.csv':
        try:
            return [('CSV', pd.read_csv(path, low_memory=False))]
        except UnicodeDecodeError:
            return [('CSV', pd.read_csv(path, encoding='latin1', low_memory=False))]
    return list(pd.read_excel(path, sheet_name=None).items())

def parse_date_series(values):
    # Handles ISO strings, timestamps, and most Excel date cells.
    return pd.to_datetime(values, errors='coerce').dt.normalize()

def date_from_filename(path):
    name = path.stem
    for pattern, fmt in [(r'(?<!\d)(20\d{2}[-_]\d{2}[-_]\d{2})(?!\d)', '%Y-%m-%d'),
                         (r'(?<!\d)(\d{2}[-_]\d{2}[-_]20\d{2})(?!\d)', '%m-%d-%Y'),
                         (r'(?<!\d)(20\d{6})(?!\d)', '%Y%m%d'),
                         (r'(?<!\d)(\d{8})(?!\d)', '%m%d%Y')]:
        match = re.search(pattern, name)
        if match:
            try:
                return pd.Timestamp(pd.to_datetime(match.group(1).replace('_','-'), format=fmt))
            except (ValueError, TypeError):
                pass
    return pd.NaT

def all_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith('~$'))

def load_signals(folder, warnings):
    chunks = []
    for path in (p for p in all_files(folder) if p.name.lower().startswith("chart_review")):
        if path.resolve() == REPORT_FILE.resolve():
            continue  # Never re-read our own generated report as scanner input.
        try:
            tables = read_tables(path)
        except Exception as e:
            warnings.append(f'SKIP scanner file {path.name}: {e}')
            continue
        for sheet, df in tables:
            symbol_col = pick(df, SYMBOL_ALIASES)
            if symbol_col is None:
                continue
            date_col = pick(df, DATE_ALIASES)
            fallback_date = date_from_filename(path)
            if date_col is None and pd.isna(fallback_date):
                warnings.append(f'SKIP {path.name}/{sheet}: no date column and no date in filename')
                continue
            class_col = pick(df, CLASS_ALIASES)
            out = pd.DataFrame({
                'symbol': df[symbol_col].astype('string').str.strip().str.upper(),
                'signal_date': parse_date_series(df[date_col]) if date_col else fallback_date,
                'classification': df[class_col].astype('string').fillna('UNCLASSIFIED') if class_col else 'UNCLASSIFIED',
                'source_file': path.name,
                'source_sheet': sheet,
                'bucket': df[pick(df, ['bucket'])].astype('string') if pick(df, ['bucket']) else 'UNKNOWN',
                'entry_stage': df[pick(df, ['latest_entry_stage'])].astype('string') if pick(df, ['latest_entry_stage']) else 'UNKNOWN',
            })
            out = out.dropna(subset=['symbol', 'signal_date'])
            out = out[out.symbol.str.fullmatch(r'[A-Z0-9.^=-]{1,20}', na=False)]
            if not out.empty:
                chunks.append(out)
    if not chunks:
        raise RuntimeError('No scanner signals found. Check columns and filename dates; see README.')
    signals = pd.concat(chunks, ignore_index=True)
    signals['classification'] = signals['classification'].astype(str).str.strip().str.upper()
    # A stock may appear in several scanner sheets on the same day; keep distinct classifications.
    signals = signals.drop_duplicates(subset=['signal_date','symbol','classification']).sort_values(['signal_date','symbol'])
    return signals.reset_index(drop=True)

def load_quotes(folder, warnings, needed_symbols):
    """Load only tickers referenced in scanner signals, with progress reporting."""
    quotes = {}
    needed = set(needed_symbols)
    paths = [p for p in all_files(folder) if p.stem.upper() in needed]
    print(f"Quote files needed: {len(paths)} for {len(needed)} unique tickers", flush=True)
    for index, path in enumerate(paths, 1):
        try:
            tables = read_tables(path)
        except Exception as e:
            warnings.append(f'SKIP quote file {path.name}: {e}')
            continue
        for sheet, df in tables:
            cols = {field: pick(df, aliases) for field, aliases in PRICE_ALIASES.items()}
            if any(cols[c] is None for c in ('date','open','high','low','close')):
                continue
            symbol_col = cols['symbol']
            sym = (df[symbol_col].astype('string').str.upper().str.strip()
                   if symbol_col else pd.Series(path.stem.upper(), index=df.index, dtype='string'))
            out = pd.DataFrame({'symbol': sym, 'date': parse_date_series(df[cols['date']])})
            for field in ('open','high','low','close'):
                out[field] = pd.to_numeric(df[cols[field]], errors='coerce')
            out = out.dropna(subset=['symbol','date','open','high','low','close'])
            out = out[out.symbol.isin(needed)]
            out = out[(out[['open','high','low','close']] > 0).all(axis=1)]
            out = out[(out.high >= out[['open','close','low']].max(axis=1)) & (out.low <= out[['open','close']].min(axis=1))]
            for symbol, part in out.groupby('symbol'):
                quotes.setdefault(str(symbol), []).append(part)
        if index % 50 == 0 or index == len(paths):
            print(f"Loaded {index}/{len(paths)} quote files", flush=True)
    if not quotes:
        raise RuntimeError('No valid matching daily OHLC quote files found. Check ticker filenames and quote column names.')
    return {symbol: pd.concat(parts, ignore_index=True)
            .drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
            for symbol, parts in quotes.items()}

def evaluate(row, quotes, latest_quote_date):
    base = row.to_dict()
    base.update(entry_date=pd.NaT, entry_price=None, target_price=None, stop_price=None,
                exit_date=pd.NaT, exit_price=None, exit_reason=None, result=None,
                return_pct=None, bars_held=None, mfe_pct=None, mae_pct=None, ambiguous_intraday=False)
    data = quotes.get(row.symbol)
    if data is None or data.empty:
        base.update(result='NO_QUOTES', exit_reason='No quote data for symbol')
        return base
    future = data[data.date > row.signal_date]
    if future.empty:
        base.update(result='PENDING', exit_reason='No quote bar after signal date')
        return base
    # Reject stale quote series rather than interpreting old data as a complete five-day test.
    if data.date.max() < row.signal_date:
        base.update(result='INSUFFICIENT_DATA', exit_reason='Quote history ends before signal date')
        return base
    entry_bar = future.iloc[0]
    # Gap in the source data could skip the real next trading session.
    if (entry_bar.date - row.signal_date).days > 5:
        base.update(result='INSUFFICIENT_DATA', exit_reason='Missing likely next-session quote')
        return base
    entry = float(entry_bar.open)
    target = entry * (1 + TARGET)
    stop = entry * (1 - STOP)
    base.update(entry_date=entry_bar.date, entry_price=entry, target_price=target, stop_price=stop)
    bars = future.head(HOLD_DAYS)
    for n, bar in enumerate(bars.itertuples(index=False), 1):
        # Entry bar: entry is at the open. Subsequent gaps are filled at open, not stop/target.
        if n > 1 and bar.open <= stop:
            base.update(exit_date=bar.date, exit_price=float(bar.open), exit_reason='GAP_STOP', result='FAILED', bars_held=n)
            break
        if n > 1 and bar.open >= target:
            base.update(exit_date=bar.date, exit_price=float(bar.open), exit_reason='GAP_TARGET', result='SUCCESS', bars_held=n)
            break
        hit_stop = bar.low <= stop
        hit_target = bar.high >= target
        if hit_stop and hit_target:
            base.update(ambiguous_intraday=True)
        if hit_stop:
            base.update(exit_date=bar.date, exit_price=stop, exit_reason='STOP_FIRST_AMBIGUOUS' if hit_target else 'STOP', result='FAILED', bars_held=n)
            break
        if hit_target:
            base.update(exit_date=bar.date, exit_price=target, exit_reason='TARGET', result='SUCCESS', bars_held=n)
            break
    observed = bars.head(base['bars_held']) if base['bars_held'] is not None else bars
    base.update(mfe_pct=round((float(observed.high.max()) / entry - 1) * 100, 3),
                mae_pct=round((float(observed.low.min()) / entry - 1) * 100, 3))
    if base['result'] is None:
        if len(bars) < HOLD_DAYS:
            base.update(result='PENDING', exit_reason='Fewer than five future quote bars')
        else:
            last = bars.iloc[-1]
            exit_price = float(last.close)
            base.update(exit_date=last.date, exit_price=exit_price, exit_reason='TIME_EXIT',
                        result='TIME_EXIT_GAIN' if exit_price > entry else ('TIME_EXIT_LOSS' if exit_price < entry else 'TIME_EXIT_FLAT'), bars_held=HOLD_DAYS)
    if base['exit_price'] is not None:
        base['return_pct'] = round((base['exit_price'] / entry - 1) * 100, 3)
    return base

def write_report(df, output, warnings):
    resolved = df[df.result.isin(['SUCCESS','FAILED','TIME_EXIT_GAIN','TIME_EXIT_LOSS','TIME_EXIT_FLAT'])]
    summary = pd.DataFrame([
        ['Total signals', len(df)],
        ['Target reached', int((df.result == 'SUCCESS').sum())],
        ['Stop hit', int((df.result == 'FAILED').sum())],
        ['Time exit gain', int((df.result == 'TIME_EXIT_GAIN').sum())],
        ['Time exit loss/flat', int(df.result.isin(['TIME_EXIT_LOSS','TIME_EXIT_FLAT']).sum())],
        ['Pending / insufficient / no quotes', int((~df.result.isin(['SUCCESS','FAILED','TIME_EXIT_GAIN','TIME_EXIT_LOSS','TIME_EXIT_FLAT'])).sum())],
        ['Target hit rate (resolved)', round(100*(resolved.result == 'SUCCESS').mean(),2) if len(resolved) else None],
        ['Profitable exit rate (resolved)', round(100*resolved.result.isin(['SUCCESS','TIME_EXIT_GAIN']).mean(),2) if len(resolved) else None],
        ['Average realized return %', round(resolved.return_pct.mean(),3) if len(resolved) else None],
        ['Ambiguous same-bar stop/target', int(df.ambiguous_intraday.sum())],
    ], columns=['Metric','Value'])
    classification = (df.groupby('classification', dropna=False)
        .agg(signals=('symbol','size'), targets=('result',lambda s:(s=='SUCCESS').sum()),
             stops=('result',lambda s:(s=='FAILED').sum()),
             profitable_exits=('result',lambda s:s.isin(['SUCCESS','TIME_EXIT_GAIN']).sum()),
             resolved=('result',lambda s:s.isin(['SUCCESS','FAILED','TIME_EXIT_GAIN','TIME_EXIT_LOSS','TIME_EXIT_FLAT']).sum()),
             avg_return_pct=('return_pct','mean')).reset_index())
    classification['target_hit_rate_pct'] = (100*classification.targets/classification.resolved.replace(0,float('nan'))).round(2)
    classification['profitable_exit_rate_pct'] = (100*classification.profitable_exits/classification.resolved.replace(0,float('nan'))).round(2)
    repeated = df.groupby('symbol').agg(appearances=('symbol','size'), first_signal=('signal_date','min'), last_signal=('signal_date','max'), targets=('result',lambda s:(s=='SUCCESS').sum())).reset_index()
    repeated = repeated[repeated.appearances>1].sort_values('appearances',ascending=False)
    sheets = {
        'Summary': summary, 'All Signals': df,
        'Successful': df[df.result=='SUCCESS'],
        'Failed': df[df.result=='FAILED'],
        'Time Exits': df[df.result.astype(str).str.startswith('TIME_EXIT')],
        'Pending & Missing': df[df.result.isin(['PENDING','NO_QUOTES','INSUFFICIENT_DATA'])],
        'By Classification': classification,
        'Repeated Signals': repeated,
        'Warnings': pd.DataFrame({'Warning': warnings or ['None']}),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name]
            ws.freeze_panes(1,0)
            ws.autofilter(0,0,max(len(frame),1),max(len(frame.columns)-1,0))
            for i,col in enumerate(frame.columns):
                ws.set_column(i,i,min(42,max(13,len(str(col))+2)))
            header=writer.book.add_format({'bold':True,'bg_color':'#17324D','font_color':'white','border':0})
            for i,col in enumerate(frame.columns):
                ws.write(0,i,col,header)
    print(f'Wrote {output} | {len(df)} signals | {len(resolved)} resolved | {len(warnings)} warnings')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signals',type=Path,default=SIGNALS_DIR,help='Scanner exports folder (default: ./output)')
    parser.add_argument('--quotes',type=Path,default=QUOTES_DIR,help='Quote files folder (default: ./data)')
    parser.add_argument('--output',type=Path,default=REPORT_FILE,help='Output workbook (default: ./output/EMA_Reversal_Backtest_Results.xlsx)')
    args=parser.parse_args()
    for folder in (args.signals,args.quotes):
        if not folder.is_dir(): parser.error(f'Folder does not exist: {folder}')
    warnings=[]
    started=time.perf_counter()
    print("Reading Chart_Review files...", flush=True)
    signals=load_signals(args.signals,warnings)
    print(f"Read {len(signals)} signals / {signals.symbol.nunique()} tickers in {time.perf_counter()-started:.1f}s", flush=True)
    quotes=load_quotes(args.quotes,warnings,set(signals.symbol))
    print(f"Quotes ready in {time.perf_counter()-started:.1f}s", flush=True)
    latest_quote_date=max(q.date.max() for q in quotes.values())
    results=pd.DataFrame([evaluate(row,quotes,latest_quote_date) for _, row in signals.iterrows()])
    print(f"Evaluated {len(results)} signals in {time.perf_counter()-started:.1f}s", flush=True)
    write_report(results,args.output,warnings)
    print(f"Total runtime: {time.perf_counter()-started:.1f}s", flush=True)

if __name__=='__main__': main()
