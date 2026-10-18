from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import yfinance as yf

from config import (
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
    MIN_20D_RANGE_PCT,
    CHOPPY_MAX_COMPRESSION_PCT,
    CHOPPY_MIN_CROSS_COUNT,
    RSI_LENGTH,
    EMA_4H_FAST,
    EMA_4H_SLOW,
)

from backtest_config import (
    DAILY_PERIOD,
    HOURLY_PERIOD,
    HOURLY_BATCH_SIZE,
    MIN_4H_BARS,
    FORWARD_HORIZONS,
    MAX_FORWARD_DAYS,
    MFE_MAE_WINDOW,
    INCLUDE_WATCH,
    PROGRESS_EVERY,
)


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
        print("Backtest universe: TEST")
        return TEST_TICKERS.copy()

    if SCAN_MODE == "SP500":
        print("Backtest universe: current S&P 500 constituents")
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


def download_daily_batch(tickers):
    print(f"Downloading {DAILY_PERIOD} of Daily data for {len(tickers)} stocks...")
    data = yf.download(
        tickers=tickers,
        period=DAILY_PERIOD,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("No Daily data downloaded.")
    print("Daily download completed.")
    return data


def extract_ticker_frame(batch, ticker):
    try:
        df = batch[ticker].copy()
    except (KeyError, TypeError):
        # yfinance returns a flat frame for a single ticker.
        if isinstance(batch.columns, pd.MultiIndex):
            return None
        df = batch.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return None
    return df


def prepare_daily_frame(batch, ticker):
    df = extract_ticker_frame(batch, ticker)
    if df is None:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < EMA_SLOW + MAX_FORWARD_DAYS + 20:
        return None

    df["EMA20"] = calculate_ema(df["Close"], EMA_20)
    df["EMA50"] = calculate_ema(df["Close"], EMA_50)
    df["EMA89"] = calculate_ema(df["Close"], EMA_FAST)
    df["EMA200"] = calculate_ema(df["Close"], EMA_SLOW)
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = calculate_rsi(df["Close"], RSI_LENGTH)

    df["DistanceLow89"] = (df["Low"] - df["EMA89"]).abs() / df["EMA89"] * 100
    df["DistanceLow200"] = (df["Low"] - df["EMA200"]).abs() / df["EMA200"] * 100
    df["Touch89"] = (
        (df["DistanceLow89"] <= TOUCH_TOLERANCE_PCT)
        & (df["Close"] > df["EMA89"])
    )
    df["Touch200"] = (
        (df["DistanceLow200"] <= TOUCH_TOLERANCE_PCT)
        & (df["Close"] > df["EMA200"])
    )
    return df


def find_days_ago(window, column):
    positions = [i for i in range(len(window)) if bool(window[column].iloc[i])]
    if not positions:
        return None
    return len(window) - 1 - positions[-1]


def daily_signal_asof(ticker, df, pos):
    # Need 200 bars for EMA, 20 for diagnostics, 6 for five-day slope,
    # and a complete +10D outcome window.
    if pos < max(EMA_SLOW - 1, 20, 5):
        return None
    if pos + MAX_FORWARD_DAYS >= len(df):
        return None

    hist = df.iloc[: pos + 1]
    latest = hist.iloc[-1]
    five_days_ago = hist.iloc[-6]

    price = float(latest["Close"])
    avg_volume = float(latest["AvgVolume20"])
    current_volume = float(latest["Volume"])
    current_rsi = float(latest["RSI14"])

    if pd.isna(avg_volume) or pd.isna(current_rsi):
        return None

    price_ok = price > MIN_PRICE
    volume_ok = avg_volume > MIN_AVG_VOLUME

    recent = hist.iloc[-LOOKBACK_DAYS:]
    days_ago_89 = find_days_ago(recent, "Touch89")
    days_ago_200 = find_days_ago(recent, "Touch200")

    recent89 = days_ago_89 is not None and price > float(latest["EMA89"])
    recent200 = days_ago_200 is not None and price > float(latest["EMA200"])

    # Preserve live-scanner behavior: basic filters are part of signal89/200.
    signal89 = recent89 and price_ok and volume_ok
    signal200 = recent200 and price_ok and volume_ok

    if not signal89 and not signal200:
        return None

    above_ema20 = price > float(latest["EMA20"])
    above_ema50 = price > float(latest["EMA50"])
    above_ema89 = price > float(latest["EMA89"])
    structure89_ok = above_ema20 and above_ema50
    structure200_ok = above_ema89

    if signal89 and signal200:
        signal = "EMA89 + EMA200"
        days_ago = min(days_ago_89, days_ago_200)
        distance_pct = min(
            (price - float(latest["EMA89"])) / float(latest["EMA89"]) * 100,
            (price - float(latest["EMA200"])) / float(latest["EMA200"]) * 100,
        )
        daily_structure_ok = structure89_ok or structure200_ok
    elif signal89:
        signal = "EMA89"
        days_ago = days_ago_89
        distance_pct = (price - float(latest["EMA89"])) / float(latest["EMA89"]) * 100
        daily_structure_ok = structure89_ok
    else:
        signal = "EMA200"
        days_ago = days_ago_200
        distance_pct = (price - float(latest["EMA200"])) / float(latest["EMA200"]) * 100
        daily_structure_ok = structure200_ok

    ema_values = [
        float(latest["EMA20"]), float(latest["EMA50"]),
        float(latest["EMA89"]), float(latest["EMA200"]),
    ]
    ema_compression_pct = (max(ema_values) - min(ema_values)) / price * 100

    recent_20 = hist.iloc[-20:]
    high_20 = float(recent_20["High"].max())
    low_20 = float(recent_20["Low"].min())
    range_20_pct = ((high_20 - low_20) / low_20 * 100) if low_20 > 0 else 0.0
    range_20_ok = range_20_pct >= MIN_20D_RANGE_PCT
    range_position_20 = (
        (price - low_20) / (high_20 - low_20) * 100 if high_20 > low_20 else 50.0
    )

    close_above_ema89 = recent_20["Close"] > recent_20["EMA89"]
    close_above_ema200 = recent_20["Close"] > recent_20["EMA200"]
    ema89_crosses = int((close_above_ema89 != close_above_ema89.shift(1)).iloc[1:].sum())
    ema200_crosses = int((close_above_ema200 != close_above_ema200.shift(1)).iloc[1:].sum())
    cross_count = ema89_crosses + ema200_crosses
    choppy_warning = (
        ema_compression_pct < CHOPPY_MAX_COMPRESSION_PCT
        and cross_count >= CHOPPY_MIN_CROSS_COUNT
    )

    daily_open = float(latest["Open"])
    daily_high = float(latest["High"])
    daily_low = float(latest["Low"])
    daily_bullish = price > daily_open
    daily_body_pct = abs(price - daily_open) / daily_open * 100 if daily_open > 0 else None
    candle_range = daily_high - daily_low
    close_position = (price - daily_low) / candle_range * 100 if candle_range > 0 else 50.0

    current_volume_ratio = current_volume / avg_volume if avg_volume > 0 else None

    def slope_pct(column):
        old = float(five_days_ago[column])
        new = float(latest[column])
        return (new - old) / old * 100

    reversal_row = hist.iloc[-(days_ago + 1)]
    reversal_avg_volume = float(reversal_row["AvgVolume20"])
    reversal_volume_ratio = (
        float(reversal_row["Volume"]) / reversal_avg_volume
        if pd.notna(reversal_avg_volume) and reversal_avg_volume > 0 else None
    )
    reversal_rsi = float(reversal_row["RSI14"])
    reversal_price = float(reversal_row["Close"])
    rsi_change = current_rsi - reversal_rsi
    price_change = (price - reversal_price) / reversal_price * 100 if reversal_price > 0 else None

    # Forward outcomes use closes for horizon returns and intraday H/L for MFE/MAE.
    outcomes = {}
    for horizon in FORWARD_HORIZONS:
        future_close = float(df.iloc[pos + horizon]["Close"])
        outcomes[f"Return +{horizon}D %"] = (future_close - price) / price * 100

    future_window = df.iloc[pos + 1 : pos + 1 + MFE_MAE_WINDOW]
    max_high = float(future_window["High"].max())
    min_low = float(future_window["Low"].min())
    mfe_pct = (max_high - price) / price * 100
    mae_pct = (min_low - price) / price * 100

    return {
        "Signal ID": f"{ticker}_{hist.index[-1].date()}_{signal.replace(' + ', '_')}",
        "Ticker": ticker,
        "Signal Date": hist.index[-1].date(),
        "Signal": signal,
        "Days Ago": days_ago,
        "Price": price,
        "Distance %": distance_pct,
        "Daily Structure OK": daily_structure_ok,
        "20D Range OK": range_20_ok,
        "Choppy Warning": choppy_warning,
        "Price OK": price_ok,
        "Volume OK": volume_ok,
        "Daily Bullish Candle": daily_bullish,
        "Daily Body %": daily_body_pct,
        "Daily Close Position %": close_position,
        "EMA20": float(latest["EMA20"]),
        "EMA50": float(latest["EMA50"]),
        "EMA89": float(latest["EMA89"]),
        "EMA200": float(latest["EMA200"]),
        "Above EMA20": above_ema20,
        "Above EMA50": above_ema50,
        "Above EMA89": above_ema89,
        "EMA20 Slope %": slope_pct("EMA20"),
        "EMA50 Slope %": slope_pct("EMA50"),
        "EMA89 Slope %": slope_pct("EMA89"),
        "EMA200 Slope %": slope_pct("EMA200"),
        "EMA Compression %": ema_compression_pct,
        "20D High": high_20,
        "20D Low": low_20,
        "20D Range %": range_20_pct,
        "20D Range Position %": range_position_20,
        "20D EMA89 Crosses": ema89_crosses,
        "20D EMA200 Crosses": ema200_crosses,
        "20D EMA Cross Count": cross_count,
        "RSI14": current_rsi,
        "Reversal RSI14": reversal_rsi,
        "RSI Change Since Reversal": rsi_change,
        "Price Change Since Reversal %": price_change,
        "Current Volume Ratio": current_volume_ratio,
        "Reversal Volume Ratio": reversal_volume_ratio,
        "20D Avg Volume": avg_volume,
        **outcomes,
        f"MFE Next {MFE_MAE_WINDOW}D %": mfe_pct,
        f"MAE Next {MFE_MAE_WINDOW}D %": mae_pct,
    }


def build_4h_frame(hourly):
    hourly = hourly.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if hourly.empty:
        return None

    if hourly.index.tz is None:
        # Yahoo normally supplies tz-aware intraday data; localize UTC only as fallback.
        hourly.index = hourly.index.tz_localize("UTC")
    hourly.index = hourly.index.tz_convert("America/New_York")
    hourly = hourly.between_time("09:30", "16:00")

    bars = []
    for _, day in hourly.groupby(hourly.index.date):
        day = day.sort_index()
        for block in (day.iloc[0:4], day.iloc[4:]):
            if block.empty:
                continue
            bars.append({
                "Datetime": block.index[0],
                "TradingDate": block.index[0].date(),
                "Open": float(block["Open"].iloc[0]),
                "High": float(block["High"].max()),
                "Low": float(block["Low"].min()),
                "Close": float(block["Close"].iloc[-1]),
                "Volume": float(block["Volume"].sum()),
            })

    if not bars:
        return None

    four = pd.DataFrame(bars).set_index("Datetime")
    four["EMA5"] = calculate_ema(four["Close"], EMA_4H_FAST)
    four["EMA13"] = calculate_ema(four["Close"], EMA_4H_SLOW)
    four["BarNumber"] = range(1, len(four) + 1)
    return four


def hourly_for_candidates(candidate_tickers):
    frames = {}
    tickers = sorted(set(candidate_tickers))
    print(f"\nDownloading {HOURLY_PERIOD} of 1H data for {len(tickers)} candidate tickers...")

    for start in range(0, len(tickers), HOURLY_BATCH_SIZE):
        chunk = tickers[start : start + HOURLY_BATCH_SIZE]
        print(f"Hourly batch {start + 1}-{min(start + len(chunk), len(tickers))} / {len(tickers)}")
        try:
            batch = yf.download(
                tickers=chunk,
                period=HOURLY_PERIOD,
                interval="1h",
                auto_adjust=False,
                prepost=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"WARNING: hourly batch failed: {exc}")
            continue

        for ticker in chunk:
            df = extract_ticker_frame(batch, ticker)
            if df is not None and not df.empty:
                frames[ticker] = df

        # Small pause is friendly to Yahoo when many batches are requested.
        time.sleep(0.4)

    print(f"Hourly data received for {len(frames)} / {len(tickers)} candidate tickers.")
    return frames


def attach_4h_and_classify(signals_df, hourly_frames):
    signals_df = signals_df.copy()
    signals_df["4H Data Complete"] = False
    signals_df["4H Bullish"] = False
    signals_df["4H Close"] = None
    signals_df["4H EMA5"] = None
    signals_df["4H EMA13"] = None
    signals_df["Status"] = "INCOMPLETE"

    four_cache = {}
    for ticker, hourly in hourly_frames.items():
        four_cache[ticker] = build_4h_frame(hourly)

    for idx, row in signals_df.iterrows():
        ticker = row["Ticker"]
        signal_date = row["Signal Date"]
        four = four_cache.get(ticker)
        if four is None or four.empty:
            continue

        eligible = four[four["TradingDate"] <= signal_date]
        if eligible.empty:
            continue

        latest = eligible.iloc[-1]
        # Require that the last 4H bar actually belongs to the signal date.
        if latest["TradingDate"] != signal_date:
            continue
        if int(latest["BarNumber"]) < MIN_4H_BARS:
            continue

        close_4h = float(latest["Close"])
        ema5 = float(latest["EMA5"])
        ema13 = float(latest["EMA13"])
        bullish = close_4h > ema13 and ema5 > ema13

        signals_df.at[idx, "4H Data Complete"] = True
        signals_df.at[idx, "4H Bullish"] = bullish
        signals_df.at[idx, "4H Close"] = close_4h
        signals_df.at[idx, "4H EMA5"] = ema5
        signals_df.at[idx, "4H EMA13"] = ema13

        if (
            bool(row["Daily Structure OK"])
            and bool(row["20D Range OK"])
            and not bool(row["Choppy Warning"])
            and bullish
        ):
            status = "QUALIFIED"
        else:
            status = "WATCH"
        signals_df.at[idx, "Status"] = status

    return signals_df


def build_summary(complete_df):
    if complete_df.empty:
        return pd.DataFrame()

    groups = []
    grouping_specs = [
        ("ALL", None),
        ("STATUS", "Status"),
        ("SIGNAL", "Signal"),
        ("4H", "4H Bullish"),
        ("STRUCTURE", "Daily Structure OK"),
        ("CHOPPY", "Choppy Warning"),
        ("BULLISH CANDLE", "Daily Bullish Candle"),
    ]

    for group_name, column in grouping_specs:
        if column is None:
            subsets = [("ALL", complete_df)]
        else:
            subsets = [(str(value), part) for value, part in complete_df.groupby(column, dropna=False)]

        for value, part in subsets:
            item = {
                "Group": group_name,
                "Value": value,
                "Signals": len(part),
            }
            for h in FORWARD_HORIZONS:
                col = f"Return +{h}D %"
                item[f"Avg +{h}D %"] = part[col].mean()
                item[f"Median +{h}D %"] = part[col].median()
                item[f"Positive +{h}D %"] = (part[col] > 0).mean() * 100
            item[f"Avg MFE {MFE_MAE_WINDOW}D %"] = part[f"MFE Next {MFE_MAE_WINDOW}D %"].mean()
            item[f"Avg MAE {MFE_MAE_WINDOW}D %"] = part[f"MAE Next {MFE_MAE_WINDOW}D %"].mean()
            groups.append(item)

    return pd.DataFrame(groups)


def round_numeric_columns(df):
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if "Volume" in col and "Ratio" not in col:
            continue
        df[col] = df[col].round(2)
    return df


def autosize_workbook(writer, sheet_names):
    for sheet_name in sheet_names:
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cells in ws.columns:
            max_len = max((len(str(c.value)) for c in cells if c.value is not None), default=0)
            ws.column_dimensions[cells[0].column_letter].width = min(max_len + 3, 24)


def main():
    print("=" * 72)
    print("EMA REVERSAL SIGNAL-ONLY BACKTEST")
    print("=" * 72)
    print("No trade simulation. No position sizing. No stop/target assumptions.")

    tickers = get_stock_universe()
    print(f"Stocks in universe: {len(tickers)}")
    daily_batch = download_daily_batch(tickers)

    all_daily_signals = []
    prepared = {}

    print("\nBuilding historical Daily signals...")
    for n, ticker in enumerate(tickers, start=1):
        try:
            df = prepare_daily_frame(daily_batch, ticker)
            if df is None:
                continue
            prepared[ticker] = df

            first_pos = max(EMA_SLOW - 1, 20, 5)
            last_pos = len(df) - MAX_FORWARD_DAYS - 1
            for pos in range(first_pos, last_pos + 1):
                row = daily_signal_asof(ticker, df, pos)
                if row is not None:
                    all_daily_signals.append(row)
        except Exception as exc:
            print(f"ERROR Daily backtest {ticker}: {exc}")

        if n % PROGRESS_EVERY == 0 or n == len(tickers):
            print(f"Daily backtest: {n} / {len(tickers)} completed")

    if not all_daily_signals:
        print("No historical Daily signals found.")
        return

    signals_df = pd.DataFrame(all_daily_signals)
    candidate_tickers = signals_df["Ticker"].unique().tolist()
    print(f"\nHistorical Daily signal rows: {len(signals_df)}")
    print(f"Tickers with at least one Daily signal: {len(candidate_tickers)}")

    hourly_frames = hourly_for_candidates(candidate_tickers)
    signals_df = attach_4h_and_classify(signals_df, hourly_frames)

    complete_df = signals_df[signals_df["4H Data Complete"]].copy()
    incomplete_df = signals_df[~signals_df["4H Data Complete"]].copy()

    if not INCLUDE_WATCH:
        complete_df = complete_df[complete_df["Status"] == "QUALIFIED"].copy()

    complete_df = complete_df.sort_values(["Signal Date", "Ticker"]).reset_index(drop=True)
    incomplete_df = incomplete_df.sort_values(["Signal Date", "Ticker"]).reset_index(drop=True)

    summary_df = build_summary(complete_df)

    # Metadata makes completeness/auditability explicit.
    if complete_df.empty:
        actual_start = None
        actual_end = None
    else:
        actual_start = complete_df["Signal Date"].min()
        actual_end = complete_df["Signal Date"].max()

    metadata = pd.DataFrame([
        ["Universe", "Current S&P 500" if SCAN_MODE == "SP500" else "TEST"],
        ["Daily download period", DAILY_PERIOD],
        ["Hourly download period", HOURLY_PERIOD],
        ["Actual complete signal start", actual_start],
        ["Actual complete signal end", actual_end],
        ["Stocks in universe", len(tickers)],
        ["Stocks with Daily history", len(prepared)],
        ["Daily signal rows before 4H validation", len(signals_df)],
        ["Complete signals", len(complete_df)],
        ["Incomplete/skipped 4H signals", len(incomplete_df)],
        ["Qualified complete signals", int((complete_df["Status"] == "QUALIFIED").sum()) if not complete_df.empty else 0],
        ["Watch complete signals", int((complete_df["Status"] == "WATCH").sum()) if not complete_df.empty else 0],
        ["Forward horizons", ", ".join(map(str, FORWARD_HORIZONS)) + " trading days"],
        ["MFE/MAE window", f"{MFE_MAE_WINDOW} trading days"],
        ["4H minimum warm-up bars", MIN_4H_BARS],
        ["Universe bias note", "Uses today's S&P 500 constituents; historical membership is not reconstructed."],
        ["4H construction note", "Matches live scanner approximation: first 4 regular-session hourly observations + remaining observations."],
    ], columns=["Item", "Value"])

    complete_df = round_numeric_columns(complete_df)
    incomplete_df = round_numeric_columns(incomplete_df)
    summary_df = round_numeric_columns(summary_df)

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = "SP500" if SCAN_MODE == "SP500" else "Test"
    excel_file = output_dir / f"EMA_Backtest_{label}_{timestamp}.xlsx"
    csv_file = output_dir / f"EMA_Backtest_COMPLETE_{label}_{timestamp}.csv"

    complete_df.to_csv(csv_file, index=False)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="RUN INFO", index=False)
        complete_df.to_excel(writer, sheet_name="COMPLETE SIGNALS", index=False)
        qualified = complete_df[complete_df["Status"] == "QUALIFIED"].copy()
        watch = complete_df[complete_df["Status"] == "WATCH"].copy()
        qualified.to_excel(writer, sheet_name="QUALIFIED", index=False)
        watch.to_excel(writer, sheet_name="WATCH", index=False)
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
        incomplete_df.to_excel(writer, sheet_name="INCOMPLETE", index=False)
        autosize_workbook(
            writer,
            ["RUN INFO", "COMPLETE SIGNALS", "QUALIFIED", "WATCH", "SUMMARY", "INCOMPLETE"],
        )

    print("\n" + "=" * 72)
    print("BACKTEST COMPLETE")
    print("=" * 72)
    print(f"Complete signals:       {len(complete_df)}")
    print(f"Qualified:              {(complete_df['Status'] == 'QUALIFIED').sum() if not complete_df.empty else 0}")
    print(f"Watch:                  {(complete_df['Status'] == 'WATCH').sum() if not complete_df.empty else 0}")
    print(f"Incomplete 4H signals:  {len(incomplete_df)}")
    print(f"Complete signal period: {actual_start} through {actual_end}")
    print(f"Excel saved to:         {excel_file}")
    print(f"CSV saved to:           {csv_file}")
    print("\nSend me the Excel workbook after the run. We'll analyze the signal quality before changing any scanner rules.")


if __name__ == "__main__":
    main()
