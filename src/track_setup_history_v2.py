"""Track EMA scanner history using ALL STOCKS (not the moving SETUP sheet).

Run: python track_setup_history.py --output-dir output
Requires Python 3.9+; no third-party packages. Writes Excel-readable CSV files.
"""
import argparse
import csv
import datetime as dt
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}
PREFIX = re.compile(r'^EMA_Reversal_V2_US_All_Stocks_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})\.xlsx$', re.I)
SCORES = {'CONFIRMED': 4, 'CONFIRMED NEAR ENTRY': 3, 'EARLY REVERSAL': 1,
          'MIXED': 0, 'LOSING MOMENTUM': -2, 'EXTENDED': 0}



def read_all_stocks(path):
    """Read XLSX with stdlib; resolve sheet by name, not sheet index."""
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read('xl/workbook.xml'))
        sheet = next((s for s in book.findall('m:sheets/m:sheet', NS)
                      if s.get('name', '').strip().upper().replace('_', ' ') in ('ALL STOCKS', 'ALL TICKERS')), None)
        if sheet is None:
            raise ValueError('ALL STOCKS / ALL_TICKERS sheet missing')
        rel_id = sheet.get('{%s}id' % NS['r'])
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rel = next(r for r in rels if r.get('Id') == rel_id)
        target = rel.get('Target').lstrip('/')
        if not target.startswith('xl/'):
            target = 'xl/' + target
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared = [''.join(t.text or '' for t in si.iter('{%s}t' % NS['m']))
                      for si in root.findall('m:si', NS)]
        rows = []
        # iterparse keeps memory low even for thousands of tickers.
        with z.open(target) as stream:
            for _, elem in ET.iterparse(stream, events=('end',)):
                if elem.tag != '{%s}row' % NS['m']:
                    continue
                cells = {}
                for c in elem.findall('m:c', NS):
                    col = re.match(r'[A-Z]+', c.get('r', ''))
                    if not col:
                        continue
                    value = c.find('m:v', NS)
                    inline = c.find('m:is', NS)
                    if c.get('t') == 's' and value is not None:
                        val = shared[int(value.text)]
                    elif inline is not None:
                        val = ''.join(t.text or '' for t in inline.iter('{%s}t' % NS['m']))
                    else:
                        val = value.text if value is not None else ''
                    cells[col.group()] = val or ''
                if cells:
                    rows.append(cells)
                elem.clear()
    if not rows:
        return []
    headers = {col: str(value).strip() for col, value in rows[0].items()}
    return [{headers[c]: v for c, v in row.items() if c in headers} for row in rows[1:]]


def field(row, name):
    return row.get(name, '').strip()


def numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def truth(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y')


def stage(row):
    value = field(row, 'Entry Stage').upper().replace('_', ' ')
    if value:
        return value
    # Historical sheets lacking Entry Stage should be marked unknown,
    # not reconstructed from Status (different concepts).
    return 'UNKNOWN'


def write_csv(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--output-dir',
        default=None,
        type=Path,
        help='Folder containing scanner Excel files'
    )

    parser.add_argument(
        '--days',
        default=5,
        type=int,
        help='Recent sessions for classification, default 5'
    )

    parser.add_argument(
        '--min-price',
        default=10.0,
        type=float
    )

    args = parser.parse_args()

    if args.days < 2:
        parser.error('--days must be at least 2')

    # ============================================================
    # PROJECT PATHS
    # ============================================================

    # Current file:
    # ema-reversal-scanner/src/track_setup_history_v2.py
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Project root:
    # ema-reversal-scanner/
    PROJECT_ROOT = SCRIPT_DIR.parent

    # Default output:
    # ema-reversal-scanner/output/
    if args.output_dir is None:
        directory = PROJECT_ROOT / "output"
    else:
        directory = args.output_dir.resolve()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output folder: {directory}")
    by_date = defaultdict(list)
    for p in directory.iterdir():
        m = PREFIX.match(p.name)
        if m and not p.name.startswith('~$'):
            by_date[m.group(1)].append((m.group(2), p))
    if not by_date:
        sys.exit('No matching scanner .xlsx workbooks in %s' % directory)
    history = defaultdict(dict)
    selected = {}
    for date in sorted(by_date):
        # Duplicate same-day files: prefer latest run by filename timestamp;
        # fall back to an earlier valid file if latest cannot be opened.
        for _, path in sorted(by_date[date], reverse=True):
            try:
                data = read_all_stocks(path)
                if not data or 'Ticker' not in data[0]:
                    raise ValueError('Empty sheet or missing Ticker column')
                if 'Entry Stage' not in data[0]:
                    raise ValueError('Missing Entry Stage column')
                seen = set()
                for row in data:
                    ticker = field(row, 'Ticker').upper()
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    history[ticker][date] = row
                selected[date] = path.name
                print('%s: %s (%d unique tickers)' % (date, path.name, len(seen)))
                break
            except (ValueError, KeyError, zipfile.BadZipFile, ET.ParseError, OSError) as exc:
                print('WARNING: %s: %s' % (path.name, exc), file=sys.stderr)
    dates = sorted(selected)
    if len(dates) < 2:
        sys.exit('Need at least two valid daily workbooks')
    latest = dates[-1]
    window = dates[-args.days:]
    all_rows = []
    for ticker, records in history.items():
        if latest not in records:
            continue  # Keep historical records, but current review requires a current row.
        current = records[latest]
        price = numeric(field(current, 'Price'))
        states = [stage(records[d]) if d in records else 'MISSING' for d in window]
        valid = [s for s in states if s not in ('MISSING', 'UNKNOWN')]
        history_count = len(valid)
        prev = next((s for s in reversed(states[:-1]) if s not in ('MISSING', 'UNKNOWN')), None)
        now = states[-1]
        confirmed = {'CONFIRMED', 'CONFIRMED NEAR ENTRY'}
        prior_confirmed = any(s in confirmed for s in states[:-1])
        prior_developing = any(s in ('EARLY REVERSAL', 'CONFIRMED NEAR ENTRY') for s in states[:-1])
        if history_count < 2:
            bucket = 'INSUFFICIENT HISTORY'
        elif now == 'EXTENDED':
            bucket = 'EXTENDED'
        elif now == 'LOSING MOMENTUM' or (prior_confirmed and now in ('MIXED', 'EARLY REVERSAL')):
            bucket = 'DETERIORATING'
        elif now == 'CONFIRMED' and prev != 'CONFIRMED' and prior_developing:
            bucket = 'NEW CONFIRMATION'
        elif now in confirmed and prev in confirmed:
            bucket = 'STABLE CONFIRMED'
        elif now == 'CONFIRMED':
            bucket = 'NEW CONFIRMATION'
        elif now == 'CONFIRMED NEAR ENTRY' and prev == 'EARLY REVERSAL':
            bucket = 'IMPROVING'
        elif now in ('EARLY REVERSAL', 'CONFIRMED NEAR ENTRY'):
            bucket = 'DEVELOPING'
        else:
            bucket = 'OTHER'
        # Weighted recent history; missing sessions contribute no points and
        # reduce coverage rather than silently counting as bearish.
        weighted = sum((i + 1) * SCORES.get(s, 0) for i, s in enumerate(states))
        possible = sum((i + 1) * 4 for i, s in enumerate(states) if s not in ('MISSING', 'UNKNOWN'))
        # A single known session is not enough to call the signal persistent.
        persistence = round(100 * weighted / possible, 1) if possible and history_count >= 2 else ''
        bullish = truth(field(current, '4H Bullish'))
        choppy = truth(field(current, 'Choppy Warning'))
        range_ok = truth(field(current, '20D Range OK'))
        distance = numeric(field(current, 'Distance %'))
        volume = numeric(field(current, 'Current Volume Ratio'))
        # These are screening heuristics, not backtested success probabilities.
        flags = []
        if not bullish: flags.append('4H not bullish')
        if choppy: flags.append('Choppy')
        if not range_ok: flags.append('20D range fails')
        if price is None or price < args.min_price: flags.append('Price below threshold')
        if distance is not None and abs(distance) > 8: flags.append('Far from reference EMA')
        if volume is not None and volume < 0.8: flags.append('Low volume ratio')
        if bucket in ('EXTENDED', 'DETERIORATING'): flags.append(bucket)
        order = {'NEW CONFIRMATION': 1, 'IMPROVING': 2, 'STABLE CONFIRMED': 3,
                 'DEVELOPING': 4, 'EXTENDED': 5, 'DETERIORATING': 6,
                 'INSUFFICIENT HISTORY': 7, 'OTHER': 8}
        record = {'Ticker': ticker, 'Bucket': bucket, 'Bucket Order': order[bucket],
                  'Latest Status': field(current, 'Status'), 'Latest Entry Stage': now,
                  'Previous Stage': prev or '', 'Persistence %': persistence,
                  'History Coverage': '%d/%d' % (len(valid), len(window)),
                  'Price': field(current, 'Price'), 'Distance %': field(current, 'Distance %'),
                  'RSI14': field(current, 'RSI14'), 'Current Volume Ratio': field(current, 'Current Volume Ratio'),
                  '4H Bullish': bullish, '20D Range OK': range_ok, 'Choppy': choppy,
                  'EMA89': field(current, 'EMA89'), 'EMA200': field(current, 'EMA200'),
                  'Flags': '; '.join(flags), 'Chart Review': 'YES' if bucket in
                  ('NEW CONFIRMATION', 'IMPROVING', 'STABLE CONFIRMED', 'DEVELOPING') and not flags else 'CHECK FLAGS',
                  'Stage Sequence': ' -> '.join(states)}
        for d, s in zip(window, states):
            record[d] = s
        all_rows.append(record)
    all_rows.sort(key=lambda r: (r['Bucket Order'], r['Chart Review'] != 'YES',
                                 -(r['Persistence %'] if isinstance(r['Persistence %'], float) else -100), r['Ticker']))
    headers = ['Ticker', 'Bucket', 'Latest Status', 'Latest Entry Stage', 'Previous Stage',
               *window, 'Stage Sequence', 'Persistence %', 'History Coverage', 'Price',
               'Distance %', 'RSI14', 'Current Volume Ratio', '4H Bullish', '20D Range OK',
               'Choppy', 'EMA89', 'EMA200', 'Flags', 'Chart Review']
    full_path = directory / ('Setup_History_%s.csv' % latest)
    review_path = directory / ('Chart_Review_%s.csv' % latest)
    write_csv(full_path, headers, all_rows)
    write_csv(review_path, headers, [r for r in all_rows if r['Chart Review'] == 'YES'])
    print('Saved %d current tickers to %s' % (len(all_rows), full_path))
    print('Saved %d unflagged chart-review candidates to %s' %
          (sum(r['Chart Review'] == 'YES' for r in all_rows), review_path))
    print('NOTE: Classifications are scanner states, not validated future returns.')


if __name__ == '__main__':
    main()
