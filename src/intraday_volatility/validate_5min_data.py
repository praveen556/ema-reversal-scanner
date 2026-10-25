"""Validate Schwab 5-minute CSV files. Run directly from PyCharm (no arguments).

Expected layout:
  ema-reversal-scanner/
    src/intraday_volatility/validate_5min_data.py
    data/5min/ABNB.csv ...
    output/intraday_volatility/ (reports are created here)

Requires: pandas
This script READS source CSVs; it never changes them.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import traceback
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "5min"
REPORT_DIR = PROJECT_ROOT / "output" / "intraday_volatility"
TIMEZONE = "America/New_York"
REQUIRED = ["datetime", "open", "high", "low", "close", "volume"]
EXPECTED_FULL_SESSION_BARS = 78
# Sessions with fewer than 78 bars are flagged for review, not automatically failed:
# US equity markets have scheduled early closes, halts, and other exceptions.


def check_file(path: Path):
    problems = []
    sessions = []
    record = {"Ticker": path.stem.upper(), "File": str(path), "Status": "PASS"}
    try:
        df = pd.read_csv(path, low_memory=False)
        missing_cols = [c for c in REQUIRED if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
        record["Rows"] = len(df)
        if df.empty:
            raise ValueError("Empty CSV")

        timestamps = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        bad_timestamps = int(timestamps.isna().sum())
        duplicate_timestamps = int(timestamps.dropna().duplicated().sum())
        record["Invalid Timestamps"] = bad_timestamps
        record["Duplicate Timestamps"] = duplicate_timestamps
        if bad_timestamps:
            problems.append(f"{bad_timestamps} invalid timestamps")
        if duplicate_timestamps:
            problems.append(f"{duplicate_timestamps} duplicate timestamps")

        for c in REQUIRED[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        missing_numeric = int(df[REQUIRED[1:]].isna().sum().sum())
        record["Missing/Non-numeric OHLCV Cells"] = missing_numeric
        if missing_numeric:
            problems.append(f"{missing_numeric} missing/non-numeric OHLCV cells")

        o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
        invalid_ohlc = ((h < l) | (h < o) | (h < c) | (l > o) | (l > c) |
                        (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).fillna(False)
        bad_ohlc = int(invalid_ohlc.sum())
        bad_volume = int((v < 0).fillna(False).sum())
        record["Invalid OHLC Rows"] = bad_ohlc
        record["Negative Volume Rows"] = bad_volume
        if bad_ohlc:
            problems.append(f"{bad_ohlc} invalid OHLC rows")
        if bad_volume:
            problems.append(f"{bad_volume} negative-volume rows")

        local = timestamps.dt.tz_convert(TIMEZONE)
        valid_time = local.notna()
        mins = local.dt.hour * 60 + local.dt.minute
        outside = valid_time & ((mins < 570) | (mins >= 960))
        misaligned = valid_time & ((mins % 5 != 0) | (local.dt.second != 0))
        record["Outside Regular Hours Rows"] = int(outside.sum())
        record["Misaligned 5-min Rows"] = int(misaligned.sum())
        if outside.any():
            problems.append(f"{int(outside.sum())} outside 09:30–16:00 ET")
        if misaligned.any():
            problems.append(f"{int(misaligned.sum())} non-5-minute-aligned timestamps")

        valid = pd.DataFrame({"ts": local[valid_time]})
        valid["date"] = valid["ts"].dt.date
        record["First Date"] = str(valid["date"].min()) if not valid.empty else ""
        record["Last Date"] = str(valid["date"].max()) if not valid.empty else ""
        record["Trading Sessions"] = int(valid["date"].nunique())
        record["Full 78-bar Sessions"] = 0
        record["Short Sessions"] = 0
        record["Overfull Sessions"] = 0
        record["Missing Interior 5-min Slots"] = 0

        for date, group in valid.groupby("date", sort=True):
            times = group["ts"].sort_values().drop_duplicates()
            count = len(times)
            # Missing bars between first and last recorded candle. Does not assume
            # the day should extend to 16:00 (e.g., half days).
            if count > 1:
                intervals = times.diff().dropna().dt.total_seconds().div(300)
                interior_missing = int(((intervals - 1).clip(lower=0)).round().sum())
            else:
                interior_missing = 0
            regular = times[(times.dt.hour * 60 + times.dt.minute >= 570) &
                            (times.dt.hour * 60 + times.dt.minute < 960)]
            regular_count = len(regular)
            if regular_count == EXPECTED_FULL_SESSION_BARS:
                record["Full 78-bar Sessions"] += 1
            elif regular_count < EXPECTED_FULL_SESSION_BARS:
                record["Short Sessions"] += 1
            else:
                record["Overfull Sessions"] += 1
            record["Missing Interior 5-min Slots"] += interior_missing
            if regular_count != EXPECTED_FULL_SESSION_BARS or interior_missing:
                sessions.append({"Ticker": record["Ticker"], "Date": str(date),
                                 "Regular Bars": regular_count, "Unique Bars": count,
                                 "Missing Interior Slots": interior_missing,
                                 "First Candle ET": str(times.iloc[0]),
                                 "Last Candle ET": str(times.iloc[-1]),
                                 "Note": "Review: could be early close, halt, or missing data"})

        if record["Short Sessions"] or record["Overfull Sessions"] or record["Missing Interior 5-min Slots"]:
            problems.append(f"{record['Short Sessions']} short sessions, "
                            f"{record['Overfull Sessions']} overfull sessions, "
                            f"{record['Missing Interior 5-min Slots']} interior missing slots (review)")
        severe = (bad_timestamps + duplicate_timestamps + missing_numeric + bad_ohlc +
                  bad_volume + int(outside.sum()) + int(misaligned.sum())) > 0
        record["Status"] = "FAIL" if severe else ("REVIEW" if problems else "PASS")
    except Exception as exc:
        record["Status"] = "ERROR"
        problems.append(str(exc))
    record["Issues"] = "; ".join(problems)
    return record, sessions


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading: {DATA_DIR}")
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Folder not found: {DATA_DIR}\nPlace this script under src/intraday_volatility/.")
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records, session_issues = [], []
    for i, path in enumerate(files, 1):
        record, sessions = check_file(path)
        records.append(record)
        session_issues.extend(sessions)
        print(f"[{i}/{len(files)}] {path.stem}: {record['Status']} — {record.get('Rows', 0)} rows")

    report_path = REPORT_DIR / "5min_Data_Quality_Report.csv"
    session_path = REPORT_DIR / "5min_Session_Issues.csv"
    pd.DataFrame(records).to_csv(report_path, index=False)
    pd.DataFrame(session_issues, columns=["Ticker", "Date", "Regular Bars", "Unique Bars",
        "Missing Interior Slots", "First Candle ET", "Last Candle ET", "Note"]).to_csv(session_path, index=False)
    counts = Counter(r["Status"] for r in records)
    print("\nFINISHED")
    print(f"Files checked: {len(files)} | PASS: {counts['PASS']} | REVIEW: {counts['REVIEW']} | FAIL: {counts['FAIL']} | ERROR: {counts['ERROR']}")
    print(f"Summary: {report_path}")
    print(f"Session details: {session_path}")
    print("REVIEW does not necessarily mean bad data; inspect early closes and trading halts.")


if __name__ == "__main__":
    main()
