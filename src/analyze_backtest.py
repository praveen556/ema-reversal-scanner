from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

# Leave as None to automatically use the newest EMA_Backtest_SP500_*.xlsx
# from the project's output folder.
INPUT_FILE = None

FORWARD_HORIZONS = [1, 2, 3, 5, 10]
MFE_MAE_WINDOW = 10


# =========================================================
# HELPERS
# =========================================================

def safe_qcut(series, q, labels):
    """Quantile buckets; gracefully handle duplicate boundaries."""
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(numeric, q=q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series([None] * len(series), index=series.index)


def add_fixed_buckets(df):
    out = df.copy()

    out["RSI Bucket"] = pd.cut(
        out["RSI14"],
        bins=[-np.inf, 40, 45, 50, 55, 60, 65, 70, np.inf],
        labels=["<40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70+"],
        right=False,
    )

    out["Volume Ratio Bucket"] = pd.cut(
        out["Current Volume Ratio"],
        bins=[-np.inf, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf],
        labels=["<0.75x", "0.75-1.0x", "1.0-1.25x", "1.25-1.5x",
                "1.5-2.0x", "2.0-3.0x", "3.0x+"],
        right=False,
    )

    out["Close Position Bucket"] = pd.cut(
        out["Daily Close Position %"],
        bins=[-np.inf, 20, 40, 60, 70, 80, 90, np.inf],
        labels=["<20%", "20-40%", "40-60%", "60-70%", "70-80%", "80-90%", "90%+"],
        right=False,
    )

    out["EMA Compression Bucket"] = pd.cut(
        out["EMA Compression %"],
        bins=[-np.inf, 2, 3, 5, 8, 12, 20, np.inf],
        labels=["<2%", "2-3%", "3-5%", "5-8%", "8-12%", "12-20%", "20%+"],
        right=False,
    )

    out["EMA Cross Bucket"] = pd.cut(
        out["20D EMA Cross Count"],
        bins=[-np.inf, 1, 3, 5, 8, 11, np.inf],
        labels=["0", "1-2", "3-4", "5-7", "8-10", "11+"],
        right=False,
    )

    out["Distance Bucket"] = pd.cut(
        out["Distance %"],
        bins=[-np.inf, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-8%", "8%+"],
        right=False,
    )

    out["20D Range Bucket"] = pd.cut(
        out["20D Range %"],
        bins=[-np.inf, 5, 8, 12, 16, 22, 30, np.inf],
        labels=["<5%", "5-8%", "8-12%", "12-16%", "16-22%", "22-30%", "30%+"],
        right=False,
    )

    return out


def summarize(df, group_name, group_col=None):
    rows = []

    if group_col is None:
        groups = [("ALL DAY-0 EVENTS", df)]
    else:
        groups = df.groupby(group_col, dropna=False, observed=False)

    for value, part in groups:
        if len(part) == 0:
            continue

        row = {
            "Analysis": group_name,
            "Group": str(value),
            "Events": len(part),
        }

        for h in FORWARD_HORIZONS:
            col = f"Return +{h}D %"
            vals = pd.to_numeric(part[col], errors="coerce").dropna()
            if len(vals):
                row[f"Avg +{h}D %"] = vals.mean()
                row[f"Median +{h}D %"] = vals.median()
                row[f"Positive +{h}D %"] = (vals > 0).mean() * 100

        mfe_col = f"MFE Next {MFE_MAE_WINDOW}D %"
        mae_col = f"MAE Next {MFE_MAE_WINDOW}D %"

        mfe = pd.to_numeric(part[mfe_col], errors="coerce").dropna()
        mae = pd.to_numeric(part[mae_col], errors="coerce").dropna()

        if len(mfe):
            row["Avg 10D MFE %"] = mfe.mean()
            row["Median 10D MFE %"] = mfe.median()
        if len(mae):
            row["Avg 10D MAE %"] = mae.mean()
            row["Median 10D MAE %"] = mae.median()

        rows.append(row)

    return pd.DataFrame(rows)


def build_combination_summary(day0):
    """
    Small set of interpretable combinations.
    These are diagnostics only, not proposed scanner rules.
    """
    tests = {
        "4H Bullish + Structure OK":
            day0["4H Bullish"].astype(bool)
            & day0["Daily Structure OK"].astype(bool),

        "4H Bullish + Not Choppy":
            day0["4H Bullish"].astype(bool)
            & ~day0["Choppy Warning"].astype(bool),

        "Bullish Candle + Close >=70%":
            day0["Daily Bullish Candle"].astype(bool)
            & (day0["Daily Close Position %"] >= 70),

        "Bullish Candle + Close >=70% + 4H Bullish":
            day0["Daily Bullish Candle"].astype(bool)
            & (day0["Daily Close Position %"] >= 70)
            & day0["4H Bullish"].astype(bool),

        "RSI 45-60":
            day0["RSI14"].between(45, 60, inclusive="both"),

        "RSI 45-60 + Not Choppy":
            day0["RSI14"].between(45, 60, inclusive="both")
            & ~day0["Choppy Warning"].astype(bool),

        "Close >=70% + Not Choppy":
            (day0["Daily Close Position %"] >= 70)
            & ~day0["Choppy Warning"].astype(bool),

        "EMA89+200 + Not Choppy":
            (day0["Signal"] == "EMA89 + EMA200")
            & ~day0["Choppy Warning"].astype(bool),
    }

    frames = []
    for name, mask in tests.items():
        selected = day0.loc[mask].copy()
        if selected.empty:
            continue
        summary = summarize(selected, "COMBINATION")
        summary["Group"] = name
        frames.append(summary)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# =========================================================
# MAIN
# =========================================================

def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"

    if INPUT_FILE:
        input_path = Path(INPUT_FILE)
        if not input_path.is_absolute():
            input_path = output_dir / input_path
    else:
        candidates = sorted(
            output_dir.glob("EMA_Backtest_SP500_*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No EMA_Backtest_SP500_*.xlsx file found in the output folder."
            )
        input_path = candidates[0]

    print("=" * 72)
    print("EMA BACKTEST - UNIQUE REVERSAL EVENT ANALYSIS")
    print("=" * 72)
    print(f"Input: {input_path}")

    signals = pd.read_excel(input_path, sheet_name="COMPLETE SIGNALS")

    required = [
        "Ticker", "Signal Date", "Signal", "Days Ago",
        "Status", "4H Bullish", "Daily Structure OK",
        "Choppy Warning", "RSI14", "Current Volume Ratio",
        "Daily Bullish Candle", "Daily Close Position %",
        "EMA Compression %", "20D EMA Cross Count",
        "Distance %", "20D Range %",
        "Return +1D %", "Return +2D %", "Return +3D %",
        "Return +5D %", "Return +10D %",
        f"MFE Next {MFE_MAE_WINDOW}D %",
        f"MAE Next {MFE_MAE_WINDOW}D %",
    ]

    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    signals["Signal Date"] = pd.to_datetime(signals["Signal Date"])

    # =====================================================
    # UNIQUE EVENT DEFINITION
    # =====================================================
    # The live scanner keeps a reversal visible for up to 5 days.
    # Therefore the same reversal can appear multiple times in the
    # original backtest.  For the first clean analysis we use only
    # Days Ago == 0.  Each row is therefore an actual reversal-day
    # observation rather than a repeated observation of an older touch.
    # =====================================================
    day0 = signals.loc[signals["Days Ago"] == 0].copy()

    day0["Event ID"] = (
        day0["Ticker"].astype(str)
        + "_"
        + day0["Signal Date"].dt.strftime("%Y-%m-%d")
    )

    duplicate_events = day0["Event ID"].duplicated().sum()
    if duplicate_events:
        print(f"WARNING: {duplicate_events} duplicate Day-0 Event IDs found.")
        day0 = day0.drop_duplicates("Event ID", keep="first").copy()

    day0 = add_fixed_buckets(day0)

    # Extra quantile diagnostics.  These let the data choose equally
    # populated groups instead of relying only on our hand-picked bins.
    day0["RSI Quintile"] = safe_qcut(
        day0["RSI14"], 5, ["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
    )
    day0["Compression Quintile"] = safe_qcut(
        day0["EMA Compression %"], 5,
        ["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
    )
    day0["Range Quintile"] = safe_qcut(
        day0["20D Range %"], 5,
        ["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
    )

    summaries = [
        summarize(day0, "ALL"),
        summarize(day0, "STATUS", "Status"),
        summarize(day0, "SIGNAL", "Signal"),
        summarize(day0, "4H", "4H Bullish"),
        summarize(day0, "STRUCTURE", "Daily Structure OK"),
        summarize(day0, "CHOPPY", "Choppy Warning"),
        summarize(day0, "CANDLE COLOR", "Daily Bullish Candle"),
        summarize(day0, "RSI BUCKET", "RSI Bucket"),
        summarize(day0, "VOLUME BUCKET", "Volume Ratio Bucket"),
        summarize(day0, "CLOSE POSITION", "Close Position Bucket"),
        summarize(day0, "COMPRESSION", "EMA Compression Bucket"),
        summarize(day0, "EMA CROSSES", "EMA Cross Bucket"),
        summarize(day0, "DISTANCE", "Distance Bucket"),
        summarize(day0, "20D RANGE", "20D Range Bucket"),
        summarize(day0, "RSI QUINTILE", "RSI Quintile"),
        summarize(day0, "COMPRESSION QUINTILE", "Compression Quintile"),
        summarize(day0, "RANGE QUINTILE", "Range Quintile"),
    ]

    summary = pd.concat(
        [x for x in summaries if not x.empty],
        ignore_index=True,
    )

    combinations = build_combination_summary(day0)

    # Compare original repeated-row backtest with unique Day-0 events.
    comparison = pd.DataFrame([
        {
            "Dataset": "Original complete signal rows",
            "Rows": len(signals),
            "Unique Tickers": signals["Ticker"].nunique(),
            "Start": signals["Signal Date"].min(),
            "End": signals["Signal Date"].max(),
        },
        {
            "Dataset": "Unique Day-0 reversal events",
            "Rows": len(day0),
            "Unique Tickers": day0["Ticker"].nunique(),
            "Start": day0["Signal Date"].min(),
            "End": day0["Signal Date"].max(),
        },
    ])

    output_path = output_dir / "EMA_Backtest_Unique_Event_Analysis.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="EVENT CHECK", index=False)
        day0.to_excel(writer, sheet_name="DAY0 EVENTS", index=False)
        summary.to_excel(writer, sheet_name="FACTOR SUMMARY", index=False)
        combinations.to_excel(writer, sheet_name="COMBINATIONS", index=False)

        for sheet_name in [
            "EVENT CHECK",
            "DAY0 EVENTS",
            "FACTOR SUMMARY",
            "COMBINATIONS",
        ]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for column_cells in ws.columns:
                max_length = max(
                    (
                        len(str(cell.value))
                        for cell in column_cells
                        if cell.value is not None
                    ),
                    default=0,
                )
                ws.column_dimensions[column_cells[0].column_letter].width = min(
                    max_length + 3,
                    28,
                )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Original complete rows: {len(signals):,}")
    print(f"Unique Day-0 events:    {len(day0):,}")
    print(f"Tickers represented:    {day0['Ticker'].nunique():,}")
    print(f"Event period:           {day0['Signal Date'].min().date()} "
          f"through {day0['Signal Date'].max().date()}")
    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
