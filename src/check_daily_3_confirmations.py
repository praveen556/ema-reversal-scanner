"""Daily EMA scanner post-filter. Run without arguments from any working directory.
Requires: pip install pandas
Reads the latest output/Chart_Review_YYYY-MM-DD.csv and writes
output/Three_Confirmation_YYYY-MM-DD.csv (including an empty header-only file if none qualify).
"""
from pathlib import Path
from datetime import datetime
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
BUCKET = 'STABLE CONFIRMED'
ENTRY_STAGE = 'CONFIRMED NEAR ENTRY'
CLASSIFICATION = 'SETUP'
MIN_CONFIRMATIONS = 3


def normalized(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def find_column(frame, *names):
    lookup = {normalized(c): c for c in frame.columns}
    for name in names:
        if normalized(name) in lookup:
            return lookup[normalized(name)]
    return None


def trailing_confirmations(value):
    # Only count consecutive stages at the RIGHT end of the sequence.
    tokens = [s.strip().upper() for s in re.split(r'\s*(?:->|→|⇒)\s*', str(value)) if s.strip()]
    count = 0
    for token in reversed(tokens):
        if token == ENTRY_STAGE:
            count += 1
        else:
            break
    return count


def main():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f'Missing output directory: {OUTPUT_DIR}')

    candidates = []
    for path in OUTPUT_DIR.glob('Chart_Review_*.csv'):
        match = re.fullmatch(r'Chart_Review_(\d{4}-\d{2}-\d{2})\.csv', path.name, flags=re.I)
        if match:
            try:
                day = datetime.strptime(match.group(1), '%Y-%m-%d').date()
                candidates.append((day, path))
            except ValueError:
                pass
    if not candidates:
        raise FileNotFoundError(f'No Chart_Review_YYYY-MM-DD.csv found in {OUTPUT_DIR}')
    signal_day, source = max(candidates, key=lambda pair: pair[0])
    print(f'Using latest scanner file: {source.name}')
    frame = pd.read_csv(source, encoding='utf-8-sig', low_memory=False)

    bucket_col = find_column(frame, 'Bucket')
    stage_col = find_column(frame, 'Latest Entry Stage')
    seq_col = find_column(frame, 'Stage Sequence')
    ticker_col = find_column(frame, 'Ticker', 'Symbol')
    # Scanner's actual SETUP field is Latest Status. If a distinct
    # Classification field exists, prefer that as explicitly requested.
    class_col = find_column(frame, 'Classification', 'Latest Status')
    required = {'Ticker': ticker_col, 'Bucket': bucket_col,
                'Latest Entry Stage': stage_col, 'Classification/Latest Status': class_col,
                'Stage Sequence': seq_col}
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f'Missing required columns: {missing}. Available: {list(frame.columns)}')

    streaks = frame[seq_col].map(trailing_confirmations)
    mask = (
        frame[bucket_col].fillna('').astype(str).str.strip().str.upper().eq(BUCKET)
        & frame[stage_col].fillna('').astype(str).str.strip().str.upper().eq(ENTRY_STAGE)
        & frame[class_col].fillna('').astype(str).str.strip().str.upper().eq(CLASSIFICATION)
        & streaks.ge(MIN_CONFIRMATIONS)
    )
    result = frame.loc[mask].copy()
    result.insert(0, 'Signal Date', signal_day.isoformat())
    result.insert(1, 'Consecutive Confirmations', streaks.loc[mask].astype(int).to_numpy())
    result = result.sort_values(['Consecutive Confirmations', ticker_col], ascending=[False, True])
    destination = OUTPUT_DIR / f'Three_Confirmation_{signal_day.isoformat()}.csv'
    result.to_csv(destination, index=False, encoding='utf-8-sig')
    print(f'Checked {len(frame)} scanner rows; found {len(result)} qualifying signals.')
    if not result.empty:
        print(result[[ticker_col, 'Consecutive Confirmations']].to_string(index=False))
    print(f'Saved: {destination}')
    print('Research shortlist only: no next-day price outcome is assumed.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
