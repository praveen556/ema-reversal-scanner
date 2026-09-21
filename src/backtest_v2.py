from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from config_v2 import (
    SCAN_MODE,
    TEST_TICKERS,
    MIN_PRICE,
    MIN_AVG_VOLUME,
    EMA_20,
    EMA_50,
    EMA_FAST,
    EMA_SLOW,
    TOUCH_TOLERANCE_PCT,
    LOOKBACK_DAYS,
    CHOPPY_MAX_COMPRESSION_PCT,
    CHOPPY_MIN_CROSS_COUNT,
    RSI_LENGTH,
    V2_HIGH_CONF_RSI_MAX,
    V2_HIGH_CONF_RANGE_MAX_PCT,
    V2_HIGH_CONF_COMPRESSION_MAX_PCT,
    V2_SETUP_RSI_MAX,
    V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF,
)

# =========================================================
# V2 BACKTEST SETTINGS
# =========================================================
DAILY_PERIOD = "5y"
FORWARD_DAYS = 10
MIN_WARMUP_BARS = 260
PROGRESS_EVERY = 25


def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_stock_universe():
    if SCAN_MODE == "TEST":
        print("Backtest mode: TEST")
        return TEST_TICKERS.copy()

    if SCAN_MODE == "SP500":
        print("Backtest mode: current S&P 500 universe")
        url = (
            "https://raw.githubusercontent.com/"
            "datasets/s-and-p-500-companies/"
            "main/data/constituents.csv"
        )
        table = pd.read_csv(url)
        tickers = (
            table["Symbol"]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        return list(dict.fromkeys(tickers))

    raise ValueError(f"Unknown SCAN_MODE: {SCAN_MODE}")


def get_ticker_frame(batch, ticker):
    try:
        df = batch[ticker].copy()
    except (KeyError, TypeError):
        return None

    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=needed).copy()
    if df.empty:
        return None
    return df


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = calculate_ema(df["Close"], EMA_20)
    df["EMA50"] = calculate_ema(df["Close"], EMA_50)
    df["EMA89"] = calculate_ema(df["Close"], EMA_FAST)
    df["EMA200"] = calculate_ema(df["Close"], EMA_SLOW)
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = calculate_rsi(df["Close"], RSI_LENGTH)

    df["DistanceLow89"] = (
        (df["Low"] - df["EMA89"]).abs() / df["EMA89"] * 100
    )
    df["DistanceLow200"] = (
        (df["Low"] - df["EMA200"]).abs() / df["EMA200"] * 100
    )

    df["Touch89"] = (
        (df["DistanceLow89"] <= TOUCH_TOLERANCE_PCT)
        & (df["Close"] > df["EMA89"])
    )
    df["Touch200"] = (
        (df["DistanceLow200"] <= TOUCH_TOLERANCE_PCT)
        & (df["Close"] > df["EMA200"])
    )

    return df


def classify_v2(row):
    if not (row["Price OK"] and row["Volume OK"]):
        return "NO SETUP"

    high_conf = (
        row["RSI14"] < V2_HIGH_CONF_RSI_MAX
        and row["20D Range %"] < V2_HIGH_CONF_RANGE_MAX_PCT
        and row["EMA Compression %"] < V2_HIGH_CONF_COMPRESSION_MAX_PCT
        and (
            not V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF
            or not row["Choppy Warning"]
        )
    )

    if high_conf:
        return "HIGH CONFIDENCE"

    if row["RSI14"] < V2_SETUP_RSI_MAX:
        return "SETUP"

    return "OBSERVE"


def build_day0_event(ticker, df, i):
    """
    Create ONE unique Day-0 event for ticker/date.

    This deliberately tests the actual reversal day only, rather than
    repeating the same touch through the scanner's 5-day lookback.
    """
    current = df.iloc[i]
    price = float(current["Close"])
    avg_volume = float(current["AvgVolume20"])

    touch89 = bool(current["Touch89"])
    touch200 = bool(current["Touch200"])

    price_ok = price > MIN_PRICE
    volume_ok = pd.notna(avg_volume) and avg_volume > MIN_AVG_VOLUME

    if not (touch89 or touch200):
        return None

    if touch89 and touch200:
        signal = "EMA89 + EMA200"
        distance_pct = min(
            (price - float(current["EMA89"])) / float(current["EMA89"]) * 100,
            (price - float(current["EMA200"])) / float(current["EMA200"]) * 100,
        )
    elif touch89:
        signal = "EMA89"
        distance_pct = (
            (price - float(current["EMA89"])) / float(current["EMA89"]) * 100
        )
    else:
        signal = "EMA200"
        distance_pct = (
            (price - float(current["EMA200"])) / float(current["EMA200"]) * 100
        )

    recent20 = df.iloc[i - 19:i + 1]
    high20 = float(recent20["High"].max())
    low20 = float(recent20["Low"].min())
    range20 = ((high20 - low20) / low20 * 100) if low20 > 0 else 0.0
    range_position = (
        (price - low20) / (high20 - low20) * 100
        if high20 > low20 else 50.0
    )

    ema_values = [
        float(current["EMA20"]),
        float(current["EMA50"]),
        float(current["EMA89"]),
        float(current["EMA200"]),
    ]
    compression = (max(ema_values) - min(ema_values)) / price * 100

    above89 = recent20["Close"] > recent20["EMA89"]
    above200 = recent20["Close"] > recent20["EMA200"]
    crosses89 = int((above89 != above89.shift(1)).iloc[1:].sum())
    crosses200 = int((above200 != above200.shift(1)).iloc[1:].sum())
    cross_count = crosses89 + crosses200

    choppy = (
        compression < CHOPPY_MAX_COMPRESSION_PCT
        and cross_count >= CHOPPY_MIN_CROSS_COUNT
    )

    open_ = float(current["Open"])
    high = float(current["High"])
    low = float(current["Low"])
    daily_range = high - low

    bullish_candle = price > open_
    body_pct = abs(price - open_) / open_ * 100 if open_ > 0 else None
    close_position = (
        (price - low) / daily_range * 100 if daily_range > 0 else 50.0
    )

    current_volume = float(current["Volume"])
    volume_ratio = (
        current_volume / avg_volume
        if pd.notna(avg_volume) and avg_volume > 0
        else None
    )

    def slope_pct(column):
        old = float(df.iloc[i - 5][column])
        new = float(current[column])
        return ((new - old) / old) * 100 if old else None

    above20_now = price > float(current["EMA20"])
    above50_now = price > float(current["EMA50"])
    above89_now = price > float(current["EMA89"])

    structure89 = above20_now and above50_now
    structure200 = above89_now
    if signal == "EMA89 + EMA200":
        structure_ok = structure89 or structure200
    elif signal == "EMA89":
        structure_ok = structure89
    else:
        structure_ok = structure200

    event = {
        "Event ID": f"{ticker}_{df.index[i].date()}",
        "Ticker": ticker,
        "Signal Date": df.index[i].date(),
        "Status": None,
        "Signal": signal,
        "Days Ago": 0,
        "Distance %": round(distance_pct, 4),
        "Price": round(price, 4),

        "RSI14": round(float(current["RSI14"]), 4),
        "Current Volume Ratio": round(volume_ratio, 4)
        if volume_ratio is not None else None,

        "Daily Bullish Candle": bullish_candle,
        "Daily Body %": round(body_pct, 4) if body_pct is not None else None,
        "Daily Close Position %": round(close_position, 4),

        "EMA20": round(float(current["EMA20"]), 4),
        "EMA50": round(float(current["EMA50"]), 4),
        "EMA89": round(float(current["EMA89"]), 4),
        "EMA200": round(float(current["EMA200"]), 4),

        "Above EMA20": above20_now,
        "Above EMA50": above50_now,
        "Above EMA89": above89_now,
        "Daily Structure OK": structure_ok,

        "EMA20 Slope %": round(slope_pct("EMA20"), 4),
        "EMA50 Slope %": round(slope_pct("EMA50"), 4),
        "EMA89 Slope %": round(slope_pct("EMA89"), 4),
        "EMA200 Slope %": round(slope_pct("EMA200"), 4),
        "EMA Compression %": round(compression, 4),

        "20D High": round(high20, 4),
        "20D Low": round(low20, 4),
        "20D Range %": round(range20, 4),
        "20D Range Position %": round(range_position, 4),

        "20D EMA89 Crosses": crosses89,
        "20D EMA200 Crosses": crosses200,
        "20D EMA Cross Count": cross_count,
        "Choppy Warning": choppy,

        "Price OK": price_ok,
        "Volume OK": volume_ok,
        "20D Avg Volume": int(avg_volume) if pd.notna(avg_volume) else None,
    }

    event["Status"] = classify_v2(event)
    return event


def add_forward_outcomes(event, df, i):
    signal_close = float(df.iloc[i]["Close"])

    horizons = [1, 2, 3, 5, 10]
    for n in horizons:
        future_close = float(df.iloc[i + n]["Close"])
        event[f"+{n}D Return %"] = (
            (future_close - signal_close) / signal_close * 100
        )

    future10 = df.iloc[i + 1:i + 11]
    future_high = float(future10["High"].max())
    future_low = float(future10["Low"].min())

    event["10D MFE %"] = (
        (future_high - signal_close) / signal_close * 100
    )
    event["10D MAE %"] = (
        (future_low - signal_close) / signal_close * 100
    )

    return event


def summarize_group(df, group_name, group_value):
    if df.empty:
        return None

    row = {
        "Group": group_name,
        "Value": group_value,
        "Events": len(df),
    }

    for n in [1, 2, 3, 5, 10]:
        col = f"+{n}D Return %"
        row[f"Avg +{n}D %"] = df[col].mean()
        row[f"Median +{n}D %"] = df[col].median()
        row[f"Positive +{n}D %"] = (df[col] > 0).mean() * 100

    row["Avg 10D MFE %"] = df["10D MFE %"].mean()
    row["Median 10D MFE %"] = df["10D MFE %"].median()
    row["Avg 10D MAE %"] = df["10D MAE %"].mean()
    row["Median 10D MAE %"] = df["10D MAE %"].median()

    return row


def build_summary(events):
    rows = []

    overall = summarize_group(events, "ALL", "ALL")
    if overall:
        rows.append(overall)

    for column in [
        "Status",
        "Signal",
        "Choppy Warning",
        "Daily Structure OK",
        "Daily Bullish Candle",
    ]:
        for value, group in events.groupby(column, dropna=False):
            summary = summarize_group(group, column, value)
            if summary:
                rows.append(summary)

    # Fixed research buckets. These do not alter V2 classification.
    bucket_specs = {
        "RSI Bucket": pd.cut(
            events["RSI14"],
            bins=[-float("inf"), 40, 45, 50, 55, 60, 65, 70, float("inf")],
            labels=["<40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70+"],
            right=False,
        ),
        "Distance Bucket": pd.cut(
            events["Distance %"],
            bins=[-float("inf"), .5, 1, 1.5, 2, 3, 5, float("inf")],
            labels=["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", "5+"],
            right=False,
        ),
        "20D Range Bucket": pd.cut(
            events["20D Range %"],
            bins=[-float("inf"), 5, 8, 12, 16, 22, 30, float("inf")],
            labels=["<5", "5-8", "8-12", "12-16", "16-22", "22-30", "30+"],
            right=False,
        ),
        "Compression Bucket": pd.cut(
            events["EMA Compression %"],
            bins=[-float("inf"), 2, 3, 5, 8, 12, float("inf")],
            labels=["<2", "2-3", "3-5", "5-8", "8-12", "12+"],
            right=False,
        ),
    }

    for bucket_name, series in bucket_specs.items():
        temp = events.copy()
        temp[bucket_name] = series
        for value, group in temp.groupby(bucket_name, observed=True):
            summary = summarize_group(group, bucket_name, value)
            if summary:
                rows.append(summary)

    return pd.DataFrame(rows)


def build_high_confidence_diagnostics(events):
    hc = events[events["Status"] == "HIGH CONFIDENCE"].copy()
    if hc.empty:
        return pd.DataFrame()

    tests = {
        "ALL HIGH CONFIDENCE": pd.Series(True, index=hc.index),
        "EMA89 only": hc["Signal"] == "EMA89",
        "EMA200 only": hc["Signal"] == "EMA200",
        "EMA89 + EMA200": hc["Signal"] == "EMA89 + EMA200",
        "Distance <0.5%": hc["Distance %"] < 0.5,
        "Distance <1.0%": hc["Distance %"] < 1.0,
        "RSI <40": hc["RSI14"] < 40,
        "RSI 40-45": (hc["RSI14"] >= 40) & (hc["RSI14"] < 45),
        "RSI 45-50": (hc["RSI14"] >= 45) & (hc["RSI14"] < 50),
        "Compression <3%": hc["EMA Compression %"] < 3,
        "Range <4%": hc["20D Range %"] < 4,
        "Range 4-5%": (hc["20D Range %"] >= 4) & (hc["20D Range %"] < 5),
        "Close position <40%": hc["Daily Close Position %"] < 40,
        "Close position >=70%": hc["Daily Close Position %"] >= 70,
        "Bullish candle": hc["Daily Bullish Candle"] == True,
        "Bearish candle": hc["Daily Bullish Candle"] == False,
        "Structure OK": hc["Daily Structure OK"] == True,
        "Structure NOT OK": hc["Daily Structure OK"] == False,
        "Cross count <=2": hc["20D EMA Cross Count"] <= 2,
        "Cross count >=3": hc["20D EMA Cross Count"] >= 3,
        "Range position <40%": hc["20D Range Position %"] < 40,
        "Range position 40-70%": (
            (hc["20D Range Position %"] >= 40)
            & (hc["20D Range Position %"] < 70)
        ),
        "Range position >=70%": hc["20D Range Position %"] >= 70,
    }

    rows = []
    for name, mask in tests.items():
        group = hc[mask]
        summary = summarize_group(group, "HIGH CONF TEST", name)
        if summary:
            rows.append(summary)

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("EMA REVERSAL SCANNER - V2 HISTORICAL BACKTEST")
    print("=" * 72)
    print("IMPORTANT: This tests unique Day-0 reversal observations.")
    print("It uses the CURRENT S&P 500 universe, so survivorship bias remains.")
    print()

    tickers = get_stock_universe()
    print(f"Stocks requested: {len(tickers)}")
    print("Downloading Daily history...")

    batch = yf.download(
        tickers=tickers,
        period=DAILY_PERIOD,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if batch.empty:
        raise RuntimeError("No Daily history downloaded.")

    events = []
    skipped = []

    for number, ticker in enumerate(tickers, start=1):
        try:
            df = get_ticker_frame(batch, ticker)

            if df is None or len(df) < MIN_WARMUP_BARS + FORWARD_DAYS:
                skipped.append((ticker, "Insufficient Daily history"))
                continue

            df = add_indicators(df)

            first_i = max(MIN_WARMUP_BARS, 20, 5)
            last_i = len(df) - FORWARD_DAYS - 1

            for i in range(first_i, last_i + 1):
                event = build_day0_event(ticker, df, i)
                if event is None:
                    continue

                event = add_forward_outcomes(event, df, i)
                events.append(event)

        except Exception as exc:
            skipped.append((ticker, str(exc)))

        if number % PROGRESS_EVERY == 0 or number == len(tickers):
            print(f"Processed {number} / {len(tickers)} stocks")

    if not events:
        print("No historical V2 events found.")
        return

    events_df = pd.DataFrame(events)

    # Safety dedupe: one Day-0 event per ticker/date.
    before = len(events_df)
    events_df = events_df.drop_duplicates(
        subset=["Event ID"],
        keep="first",
    ).copy()
    duplicates_removed = before - len(events_df)

    events_df = events_df.sort_values(
        ["Signal Date", "Ticker"]
    ).reset_index(drop=True)

    high_conf_df = events_df[
        events_df["Status"] == "HIGH CONFIDENCE"
    ].copy()
    setup_df = events_df[
        events_df["Status"] == "SETUP"
    ].copy()
    observe_df = events_df[
        events_df["Status"] == "OBSERVE"
    ].copy()

    summary_df = build_summary(events_df)
    hc_diag_df = build_high_confidence_diagnostics(events_df)

    run_info = pd.DataFrame(
        [
            ["Backtest Version", "V2 Day-0"],
            ["Universe", "Current S&P 500 constituents"],
            ["Universe Warning", "Survivorship bias: historical membership is not reconstructed"],
            ["Daily Download Period", DAILY_PERIOD],
            ["First Signal Date", str(events_df["Signal Date"].min())],
            ["Last Signal Date", str(events_df["Signal Date"].max())],
            ["Stocks Requested", len(tickers)],
            ["Stocks Skipped", len(skipped)],
            ["Unique Day-0 Events", len(events_df)],
            ["Duplicates Removed", duplicates_removed],
            ["High Confidence Events", len(high_conf_df)],
            ["Setup Events", len(setup_df)],
            ["Observe Events", len(observe_df)],
            ["V2 High Confidence RSI Max", V2_HIGH_CONF_RSI_MAX],
            ["V2 High Confidence 20D Range Max %", V2_HIGH_CONF_RANGE_MAX_PCT],
            ["V2 High Confidence Compression Max %", V2_HIGH_CONF_COMPRESSION_MAX_PCT],
            ["V2 Setup RSI Max", V2_SETUP_RSI_MAX],
            ["Forward Outcome", "Close-to-close; +1/+2/+3/+5/+10 trading days"],
            ["MFE/MAE", "Next 10 trading days using future High/Low vs signal close"],
            ["4H Note", "V2 does not require 4H confirmation; this backtest intentionally avoids 1H dependency"],
        ],
        columns=["Setting", "Value"],
    )

    skipped_df = pd.DataFrame(
        skipped,
        columns=["Ticker", "Reason"],
    )

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = "SP500" if SCAN_MODE == "SP500" else "Test"
    out_file = output_dir / f"EMA_Backtest_V2_{label}_{timestamp}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        run_info.to_excel(writer, sheet_name="RUN INFO", index=False)
        events_df.to_excel(writer, sheet_name="ALL DAY0 EVENTS", index=False)
        high_conf_df.to_excel(writer, sheet_name="HIGH CONFIDENCE", index=False)
        setup_df.to_excel(writer, sheet_name="SETUP", index=False)
        observe_df.to_excel(writer, sheet_name="OBSERVE", index=False)
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
        hc_diag_df.to_excel(writer, sheet_name="HC DIAGNOSTICS", index=False)
        skipped_df.to_excel(writer, sheet_name="SKIPPED", index=False)

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.columns:
                max_len = max(
                    (len(str(c.value)) for c in cells if c.value is not None),
                    default=0,
                )
                worksheet.column_dimensions[cells[0].column_letter].width = min(
                    max_len + 3, 28
                )

    print()
    print("=" * 72)
    print("V2 BACKTEST COMPLETE")
    print("=" * 72)
    print(f"Unique Day-0 events: {len(events_df):,}")
    print(f"High Confidence:     {len(high_conf_df):,}")
    print(f"Setup:               {len(setup_df):,}")
    print(f"Observe:             {len(observe_df):,}")
    print(f"Skipped stocks:      {len(skipped_df):,}")
    print()
    print(f"Excel saved to: {out_file}")


if __name__ == "__main__":
    main()
