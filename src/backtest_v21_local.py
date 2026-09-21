from pathlib import Path
from datetime import datetime

import pandas as pd

from config_v2 import (
    MIN_PRICE,
    MIN_AVG_VOLUME,
    EMA_20,
    EMA_50,
    EMA_FAST,
    EMA_SLOW,
    TOUCH_TOLERANCE_PCT,
    CHOPPY_MAX_COMPRESSION_PCT,
    CHOPPY_MIN_CROSS_COUNT,
    RSI_LENGTH,
    V2_HIGH_CONF_RSI_MAX,
    V2_HIGH_CONF_RANGE_MAX_PCT,
    V2_HIGH_CONF_COMPRESSION_MAX_PCT,
    V2_SETUP_RSI_MAX,
    V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF,
    V21_PRIORITY_RSI_MIN,
    V21_PRIORITY_RSI_MAX,
)


# ============================================================
# PATHS
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent

DAILY_DATA_FOLDER = (
    PROJECT_FOLDER / "data" / "daily"
)

OUTPUT_FOLDER = (
    PROJECT_FOLDER / "output"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BACKTEST SETTINGS
# ============================================================

# Enough prior history for EMA200 to stabilize before
# we start accepting signals.
MIN_WARMUP_BARS = 260

# Need at least 10 future trading days for outcome testing.
FORWARD_DAYS = 10

# Progress display.
PROGRESS_EVERY = 250


# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(series, length):
    return series.ewm(
        span=length,
        adjust=False,
    ).mean()


def calculate_rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# READ LOCAL STOCK
# ============================================================

def read_local_stock(file_path):

    try:

        df = pd.read_csv(
            file_path,
            index_col=0,
            parse_dates=True,
        )

    except Exception:
        return None

    if df.empty:
        return None

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return None

    # --------------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # --------------------------------------------------------

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if df.empty:
        return None

    df.index = pd.to_datetime(
        df.index
    )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ============================================================
# ADD INDICATORS
# ============================================================

def prepare_stock(df):

    df = df.copy()

    df["EMA20"] = calculate_ema(
        df["Close"],
        EMA_20,
    )

    df["EMA50"] = calculate_ema(
        df["Close"],
        EMA_50,
    )

    df["EMA89"] = calculate_ema(
        df["Close"],
        EMA_FAST,
    )

    df["EMA200"] = calculate_ema(
        df["Close"],
        EMA_SLOW,
    )

    df["AvgVolume20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_LENGTH,
    )

    # --------------------------------------------------------
    # EXACT DAY-0 EMA TOUCH/RECLAIM
    # --------------------------------------------------------

    df["DistanceLow89"] = (
        (df["Low"] - df["EMA89"]).abs()
        / df["EMA89"]
        * 100
    )

    df["DistanceLow200"] = (
        (df["Low"] - df["EMA200"]).abs()
        / df["EMA200"]
        * 100
    )

    df["Touch89"] = (
        (
            df["DistanceLow89"]
            <= TOUCH_TOLERANCE_PCT
        )
        & (
            df["Close"]
            > df["EMA89"]
        )
    )

    df["Touch200"] = (
        (
            df["DistanceLow200"]
            <= TOUCH_TOLERANCE_PCT
        )
        & (
            df["Close"]
            > df["EMA200"]
        )
    )

    return df


# ============================================================
# CLASSIFY HISTORICAL DAY-0 EVENT
# ============================================================

def classify_event(
    signal,
    rsi,
    range_20_pct,
    compression_pct,
    choppy,
):

    high_confidence = (
        pd.notna(rsi)
        and pd.notna(range_20_pct)
        and pd.notna(compression_pct)
        and rsi < V2_HIGH_CONF_RSI_MAX
        and range_20_pct
        < V2_HIGH_CONF_RANGE_MAX_PCT
        and compression_pct
        < V2_HIGH_CONF_COMPRESSION_MAX_PCT
        and (
            not V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF
            or not choppy
        )
    )

    priority = (
        high_confidence
        and signal == "EMA200"
        and rsi >= V21_PRIORITY_RSI_MIN
        and rsi < V21_PRIORITY_RSI_MAX
    )

    if priority:
        return "PRIORITY"

    if high_confidence:
        return "HIGH CONFIDENCE"

    if (
        pd.notna(rsi)
        and rsi < V2_SETUP_RSI_MAX
    ):
        return "SETUP"

    return "OBSERVE"


# ============================================================
# FORWARD RETURN
# ============================================================

def forward_return(
    df,
    position,
    days,
):

    future_position = (
        position + days
    )

    if future_position >= len(df):
        return None

    current_close = float(
        df["Close"].iloc[position]
    )

    future_close = float(
        df["Close"].iloc[
            future_position
        ]
    )

    if current_close <= 0:
        return None

    return (
        (
            future_close
            - current_close
        )
        / current_close
        * 100
    )


# ============================================================
# MFE / MAE
# ============================================================

def calculate_mfe_mae(
    df,
    position,
):

    end_position = (
        position + FORWARD_DAYS
    )

    if end_position >= len(df):
        return None, None

    entry_price = float(
        df["Close"].iloc[position]
    )

    if entry_price <= 0:
        return None, None

    future = df.iloc[
        position + 1:
        end_position + 1
    ]

    if future.empty:
        return None, None

    highest = float(
        future["High"].max()
    )

    lowest = float(
        future["Low"].min()
    )

    mfe = (
        (highest - entry_price)
        / entry_price
        * 100
    )

    mae = (
        (lowest - entry_price)
        / entry_price
        * 100
    )

    return mfe, mae


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(
    ticker,
    df,
):

    events = []

    if len(df) < (
        MIN_WARMUP_BARS
        + FORWARD_DAYS
        + 1
    ):
        return events

    df = prepare_stock(
        df
    )

    # --------------------------------------------------------
    # Historical scan
    #
    # Start after warmup.
    # Stop early enough that every event has a complete
    # +10 trading-day outcome.
    # --------------------------------------------------------

    for position in range(
        MIN_WARMUP_BARS,
        len(df) - FORWARD_DAYS,
    ):

        row = df.iloc[position]

        price = float(
            row["Close"]
        )

        avg_volume = float(
            row["AvgVolume20"]
        )

        rsi = float(
            row["RSI14"]
        )

        # ----------------------------------------------------
        # BASIC FILTERS
        # ----------------------------------------------------

        if price <= MIN_PRICE:
            continue

        if (
            pd.isna(avg_volume)
            or avg_volume
            <= MIN_AVG_VOLUME
        ):
            continue

        # ----------------------------------------------------
        # DAY-0 TOUCH SIGNAL
        # ----------------------------------------------------

        touch89 = bool(
            row["Touch89"]
        )

        touch200 = bool(
            row["Touch200"]
        )

        if not touch89 and not touch200:
            continue

        if touch89 and touch200:
            signal = "EMA89 + EMA200"

        elif touch89:
            signal = "EMA89"

        else:
            signal = "EMA200"

        # ----------------------------------------------------
        # DISTANCE FROM RELEVANT EMA
        # ----------------------------------------------------

        distance_89 = (
            (
                price
                - float(row["EMA89"])
            )
            / float(row["EMA89"])
            * 100
        )

        distance_200 = (
            (
                price
                - float(row["EMA200"])
            )
            / float(row["EMA200"])
            * 100
        )

        if signal == "EMA89":

            distance_pct = (
                distance_89
            )

        elif signal == "EMA200":

            distance_pct = (
                distance_200
            )

        else:

            distance_pct = min(
                distance_89,
                distance_200,
            )

        # ----------------------------------------------------
        # DAILY STRUCTURE
        # Diagnostic only in V2.
        # ----------------------------------------------------

        above_ema20 = (
            price
            > float(row["EMA20"])
        )

        above_ema50 = (
            price
            > float(row["EMA50"])
        )

        above_ema89 = (
            price
            > float(row["EMA89"])
        )

        structure89_ok = (
            above_ema20
            and above_ema50
        )

        structure200_ok = (
            above_ema89
        )

        if signal == "EMA89":

            daily_structure_ok = (
                structure89_ok
            )

        elif signal == "EMA200":

            daily_structure_ok = (
                structure200_ok
            )

        else:

            daily_structure_ok = (
                structure89_ok
                or structure200_ok
            )

        # ----------------------------------------------------
        # EMA COMPRESSION
        # ----------------------------------------------------

        ema_values = [
            float(row["EMA20"]),
            float(row["EMA50"]),
            float(row["EMA89"]),
            float(row["EMA200"]),
        ]

        compression_pct = (
            (
                max(ema_values)
                - min(ema_values)
            )
            / price
            * 100
        )

        # ----------------------------------------------------
        # 20-DAY RANGE
        # ----------------------------------------------------

        recent_20 = df.iloc[
            position - 19:
            position + 1
        ]

        if len(recent_20) < 20:
            continue

        high_20 = float(
            recent_20["High"].max()
        )

        low_20 = float(
            recent_20["Low"].min()
        )

        if low_20 <= 0:
            continue

        range_20_pct = (
            (high_20 - low_20)
            / low_20
            * 100
        )

        # ----------------------------------------------------
        # EMA CROSS COUNT / CHOPPINESS
        # ----------------------------------------------------

        close_above_ema89 = (
            recent_20["Close"]
            > recent_20["EMA89"]
        )

        close_above_ema200 = (
            recent_20["Close"]
            > recent_20["EMA200"]
        )

        ema89_cross_count = int(
            (
                close_above_ema89
                != close_above_ema89.shift(1)
            )
            .iloc[1:]
            .sum()
        )

        ema200_cross_count = int(
            (
                close_above_ema200
                != close_above_ema200.shift(1)
            )
            .iloc[1:]
            .sum()
        )

        ema_cross_count_20d = (
            ema89_cross_count
            + ema200_cross_count
        )

        choppy = (
            compression_pct
            < CHOPPY_MAX_COMPRESSION_PCT
            and ema_cross_count_20d
            >= CHOPPY_MIN_CROSS_COUNT
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        status = classify_event(
            signal=signal,
            rsi=rsi,
            range_20_pct=(
                range_20_pct
            ),
            compression_pct=(
                compression_pct
            ),
            choppy=choppy,
        )

        # ----------------------------------------------------
        # CURRENT VOLUME RATIO
        # ----------------------------------------------------

        current_volume = float(
            row["Volume"]
        )

        volume_ratio = (
            current_volume
            / avg_volume
            if avg_volume > 0
            else None
        )

        # ----------------------------------------------------
        # EMA200 SLOPE
        #
        # Same 5-trading-day style diagnostic used by scanner.
        # ----------------------------------------------------

        ema200_slope_pct = None

        if position >= 5:

            old_ema200 = float(
                df["EMA200"].iloc[
                    position - 5
                ]
            )

            current_ema200 = float(
                row["EMA200"]
            )

            if old_ema200 > 0:

                ema200_slope_pct = (
                    (
                        current_ema200
                        - old_ema200
                    )
                    / old_ema200
                    * 100
                )

        # ----------------------------------------------------
        # DAILY CANDLE
        # ----------------------------------------------------

        daily_open = float(
            row["Open"]
        )

        daily_high = float(
            row["High"]
        )

        daily_low = float(
            row["Low"]
        )

        bullish_candle = (
            price > daily_open
        )

        candle_range = (
            daily_high
            - daily_low
        )

        close_position_pct = (
            (
                price - daily_low
            )
            / candle_range
            * 100
            if candle_range > 0
            else 50.0
        )

        # ----------------------------------------------------
        # FORWARD OUTCOMES
        # ----------------------------------------------------

        return_1d = forward_return(
            df,
            position,
            1,
        )

        return_2d = forward_return(
            df,
            position,
            2,
        )

        return_3d = forward_return(
            df,
            position,
            3,
        )

        return_5d = forward_return(
            df,
            position,
            5,
        )

        return_10d = forward_return(
            df,
            position,
            10,
        )

        mfe_10d, mae_10d = (
            calculate_mfe_mae(
                df,
                position,
            )
        )

        # ----------------------------------------------------
        # SAVE EVENT
        # ----------------------------------------------------

        events.append(
            {
                "Ticker": ticker,
                "Signal Date": (
                    df.index[position]
                ),
                "Signal": signal,
                "Status": status,

                "Price": round(
                    price,
                    4,
                ),

                "RSI14": round(
                    rsi,
                    2,
                ),

                "Distance %": round(
                    distance_pct,
                    2,
                ),

                "20D Range %": round(
                    range_20_pct,
                    2,
                ),

                "EMA Compression %": round(
                    compression_pct,
                    2,
                ),

                "EMA Cross Count 20D": (
                    ema_cross_count_20d
                ),

                "Choppy Warning": (
                    choppy
                ),

                "Daily Structure OK": (
                    daily_structure_ok
                ),

                "Volume Ratio": (
                    round(
                        volume_ratio,
                        2,
                    )
                    if volume_ratio
                    is not None
                    else None
                ),

                "EMA200 Slope %": (
                    round(
                        ema200_slope_pct,
                        3,
                    )
                    if ema200_slope_pct
                    is not None
                    else None
                ),

                "Bullish Candle": (
                    bullish_candle
                ),

                "Close Position %": (
                    round(
                        close_position_pct,
                        2,
                    )
                ),

                "Return +1D %": (
                    round(
                        return_1d,
                        3,
                    )
                ),

                "Return +2D %": (
                    round(
                        return_2d,
                        3,
                    )
                ),

                "Return +3D %": (
                    round(
                        return_3d,
                        3,
                    )
                ),

                "Return +5D %": (
                    round(
                        return_5d,
                        3,
                    )
                ),

                "Return +10D %": (
                    round(
                        return_10d,
                        3,
                    )
                ),

                "MFE 10D %": round(
                    mfe_10d,
                    3,
                ),

                "MAE 10D %": round(
                    mae_10d,
                    3,
                ),
            }
        )

    return events


# ============================================================
# SUMMARY
# ============================================================

def make_summary(events_df):

    rows = []

    status_order = [
        "PRIORITY",
        "HIGH CONFIDENCE",
        "SETUP",
        "OBSERVE",
    ]

    for status in status_order:

        subset = events_df[
            events_df["Status"]
            == status
        ]

        if subset.empty:
            continue

        row = {
            "Status": status,
            "Events": len(subset),
        }

        for days in [
            1,
            2,
            3,
            5,
            10,
        ]:

            column = (
                f"Return +{days}D %"
            )

            row[
                f"Avg +{days}D %"
            ] = round(
                subset[column].mean(),
                3,
            )

            row[
                f"Median +{days}D %"
            ] = round(
                subset[column].median(),
                3,
            )

            row[
                f"Positive +{days}D %"
            ] = round(
                (
                    subset[column] > 0
                ).mean()
                * 100,
                2,
            )

        row["Avg MFE 10D %"] = round(
            subset["MFE 10D %"].mean(),
            3,
        )

        row["Avg MAE 10D %"] = round(
            subset["MAE 10D %"].mean(),
            3,
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRIORITY BREAKDOWN
# ============================================================

def make_priority_breakdown(
    events_df,
):

    priority = events_df[
        events_df["Status"]
        == "PRIORITY"
    ].copy()

    if priority.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # RSI SUB-BUCKET
    # --------------------------------------------------------

    priority["RSI Bucket"] = pd.cut(
        priority["RSI14"],
        bins=[
            40,
            42.5,
            45,
        ],
        right=False,
        labels=[
            "40-42.5",
            "42.5-45",
        ],
    )

    # --------------------------------------------------------
    # DISTANCE BUCKET
    # --------------------------------------------------------

    priority["Distance Bucket"] = (
        pd.cut(
            priority["Distance %"],
            bins=[
                -100,
                0.5,
                1.0,
                1.5,
                100,
            ],
            labels=[
                "<0.5%",
                "0.5-1.0%",
                "1.0-1.5%",
                ">1.5%",
            ],
        )
    )

    # --------------------------------------------------------
    # EMA200 SLOPE
    # --------------------------------------------------------

    priority["EMA200 Slope Group"] = (
        priority[
            "EMA200 Slope %"
        ].apply(
            lambda value:
            "RISING"
            if pd.notna(value)
            and value > 0
            else "FLAT/FALLING"
        )
    )

    breakdown_rows = []

    breakdown_specs = [
        (
            "RSI Bucket",
            "RSI Bucket",
        ),
        (
            "Distance",
            "Distance Bucket",
        ),
        (
            "EMA200 Slope",
            "EMA200 Slope Group",
        ),
        (
            "Daily Structure",
            "Daily Structure OK",
        ),
        (
            "Choppy",
            "Choppy Warning",
        ),
    ]

    for category_name, column in (
        breakdown_specs
    ):

        grouped = priority.groupby(
            column,
            observed=True,
            dropna=False,
        )

        for group_name, subset in grouped:

            breakdown_rows.append(
                {
                    "Category": (
                        category_name
                    ),

                    "Group": str(
                        group_name
                    ),

                    "Events": len(
                        subset
                    ),

                    "Positive +3D %": round(
                        (
                            subset[
                                "Return +3D %"
                            ]
                            > 0
                        ).mean()
                        * 100,
                        2,
                    ),

                    "Positive +5D %": round(
                        (
                            subset[
                                "Return +5D %"
                            ]
                            > 0
                        ).mean()
                        * 100,
                        2,
                    ),

                    "Positive +10D %": round(
                        (
                            subset[
                                "Return +10D %"
                            ]
                            > 0
                        ).mean()
                        * 100,
                        2,
                    ),

                    "Avg +10D %": round(
                        subset[
                            "Return +10D %"
                        ].mean(),
                        3,
                    ),

                    "Median +10D %": round(
                        subset[
                            "Return +10D %"
                        ].median(),
                        3,
                    ),

                    "Avg MFE 10D %": round(
                        subset[
                            "MFE 10D %"
                        ].mean(),
                        3,
                    ),

                    "Avg MAE 10D %": round(
                        subset[
                            "MAE 10D %"
                        ].mean(),
                        3,
                    ),
                }
            )

    return pd.DataFrame(
        breakdown_rows
    )


# ============================================================
# YEARLY PRIORITY
# ============================================================

def make_priority_yearly(
    events_df,
):

    priority = events_df[
        events_df["Status"]
        == "PRIORITY"
    ].copy()

    if priority.empty:
        return pd.DataFrame()

    priority["Year"] = (
        pd.to_datetime(
            priority["Signal Date"]
        ).dt.year
    )

    rows = []

    for year, subset in (
        priority.groupby("Year")
    ):

        rows.append(
            {
                "Year": year,

                "Events": len(
                    subset
                ),

                "Positive +3D %": round(
                    (
                        subset[
                            "Return +3D %"
                        ]
                        > 0
                    ).mean()
                    * 100,
                    2,
                ),

                "Positive +5D %": round(
                    (
                        subset[
                            "Return +5D %"
                        ]
                        > 0
                    ).mean()
                    * 100,
                    2,
                ),

                "Positive +10D %": round(
                    (
                        subset[
                            "Return +10D %"
                        ]
                        > 0
                    ).mean()
                    * 100,
                    2,
                ),

                "Avg +10D %": round(
                    subset[
                        "Return +10D %"
                    ].mean(),
                    3,
                ),

                "Median +10D %": round(
                    subset[
                        "Return +10D %"
                    ].median(),
                    3,
                ),

                "Avg MFE 10D %": round(
                    subset[
                        "MFE 10D %"
                    ].mean(),
                    3,
                ),

                "Avg MAE 10D %": round(
                    subset[
                        "MAE 10D %"
                    ].mean(),
                    3,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V2.1 LOCAL DATABASE BACKTEST")
    print("=" * 70)

    print(
        f"Daily data folder: "
        f"{DAILY_DATA_FOLDER}"
    )

    csv_files = sorted(
        DAILY_DATA_FOLDER.glob(
            "*.csv"
        )
    )

    # Ignore special files if any appear later.
    csv_files = [
        file_path
        for file_path in csv_files
        if not file_path.name.startswith(
            "_"
        )
    ]

    print(
        f"Local stock files found: "
        f"{len(csv_files)}"
    )

    all_events = []

    usable_stocks = 0
    skipped_stocks = 0

    for number, file_path in enumerate(
        csv_files,
        start=1,
    ):

        ticker = file_path.stem

        df = read_local_stock(
            file_path
        )

        if df is None:

            skipped_stocks += 1
            continue

        if len(df) < (
            MIN_WARMUP_BARS
            + FORWARD_DAYS
            + 1
        ):

            skipped_stocks += 1
            continue

        usable_stocks += 1

        events = analyze_stock(
            ticker,
            df,
        )

        all_events.extend(
            events
        )

        if (
            number % PROGRESS_EVERY == 0
            or number == len(csv_files)
        ):

            print(
                f"Processed "
                f"{number}/"
                f"{len(csv_files)} "
                f"files | "
                f"Usable: "
                f"{usable_stocks} | "
                f"Events: "
                f"{len(all_events)}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not all_events:

        print()
        print(
            "No historical events "
            "were found."
        )

        return

    events_df = pd.DataFrame(
        all_events
    )

    events_df["Signal Date"] = (
        pd.to_datetime(
            events_df["Signal Date"]
        )
    )

    events_df = (
        events_df
        .sort_values(
            [
                "Signal Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SUBSETS
    # ========================================================

    priority_df = events_df[
        events_df["Status"]
        == "PRIORITY"
    ].copy()

    high_conf_df = events_df[
        events_df["Status"]
        == "HIGH CONFIDENCE"
    ].copy()

    setup_df = events_df[
        events_df["Status"]
        == "SETUP"
    ].copy()

    observe_df = events_df[
        events_df["Status"]
        == "OBSERVE"
    ].copy()

    # ========================================================
    # SUMMARY TABLES
    # ========================================================

    summary_df = make_summary(
        events_df
    )

    priority_breakdown_df = (
        make_priority_breakdown(
            events_df
        )
    )

    priority_yearly_df = (
        make_priority_yearly(
            events_df
        )
    )

    # ========================================================
    # RUN INFO
    # ========================================================

    run_info_df = pd.DataFrame(
        [
            {
                "Item": (
                    "Backtest Run"
                ),
                "Value": (
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ),
            },
            {
                "Item": (
                    "Local CSV Files"
                ),
                "Value": len(
                    csv_files
                ),
            },
            {
                "Item": (
                    "Usable Stocks"
                ),
                "Value": (
                    usable_stocks
                ),
            },
            {
                "Item": (
                    "Skipped Stocks"
                ),
                "Value": (
                    skipped_stocks
                ),
            },
            {
                "Item": (
                    "Historical Events"
                ),
                "Value": len(
                    events_df
                ),
            },
            {
                "Item": (
                    "Priority Events"
                ),
                "Value": len(
                    priority_df
                ),
            },
            {
                "Item": (
                    "Minimum Warmup Bars"
                ),
                "Value": (
                    MIN_WARMUP_BARS
                ),
            },
            {
                "Item": (
                    "Forward Outcome Days"
                ),
                "Value": (
                    FORWARD_DAYS
                ),
            },
            {
                "Item": (
                    "Priority Rule"
                ),
                "Value": (
                    "V2 High Confidence "
                    "+ EMA200-only "
                    "+ RSI 40-45"
                ),
            },
            {
                "Item": (
                    "4H Used"
                ),
                "Value": "No",
            },
        ]
    )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_file = (
        OUTPUT_FOLDER
        / (
            "V21_Local_Backtest_"
            f"{timestamp}.xlsx"
        )
    )

    print()
    print(
        "Writing Excel workbook..."
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        run_info_df.to_excel(
            writer,
            sheet_name="RUN INFO",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="SUMMARY",
            index=False,
        )

        priority_df.to_excel(
            writer,
            sheet_name="PRIORITY",
            index=False,
        )

        high_conf_df.to_excel(
            writer,
            sheet_name="HIGH CONFIDENCE",
            index=False,
        )

        setup_df.to_excel(
            writer,
            sheet_name="SETUP",
            index=False,
        )

        observe_df.to_excel(
            writer,
            sheet_name="OBSERVE",
            index=False,
        )

        priority_breakdown_df.to_excel(
            writer,
            sheet_name=(
                "PRIORITY BREAKDOWN"
            ),
            index=False,
        )

        priority_yearly_df.to_excel(
            writer,
            sheet_name=(
                "PRIORITY YEARLY"
            ),
            index=False,
        )

        events_df.to_excel(
            writer,
            sheet_name="ALL EVENTS",
            index=False,
        )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        f"Local files:       "
        f"{len(csv_files)}"
    )

    print(
        f"Usable stocks:     "
        f"{usable_stocks}"
    )

    print(
        f"Skipped stocks:    "
        f"{skipped_stocks}"
    )

    print(
        f"Total events:      "
        f"{len(events_df)}"
    )

    print(
        f"Priority events:   "
        f"{len(priority_df)}"
    )

    print(
        f"High Confidence:   "
        f"{len(high_conf_df)}"
    )

    print(
        f"Setup:             "
        f"{len(setup_df)}"
    )

    print(
        f"Observe:           "
        f"{len(observe_df)}"
    )

    if not priority_df.empty:

        print()
        print(
            "PRIORITY RESULTS"
        )

        print(
            "Positive +3D:  "
            f"{(
                priority_df['Return +3D %']
                > 0
            ).mean() * 100:.2f}%"
        )

        print(
            "Positive +5D:  "
            f"{(
                priority_df['Return +5D %']
                > 0
            ).mean() * 100:.2f}%"
        )

        print(
            "Positive +10D: "
            f"{(
                priority_df['Return +10D %']
                > 0
            ).mean() * 100:.2f}%"
        )

        print(
            "Avg +10D:      "
            f"{priority_df[
                'Return +10D %'
            ].mean():.2f}%"
        )

        print(
            "Median +10D:   "
            f"{priority_df[
                'Return +10D %'
            ].median():.2f}%"
        )

        print(
            "Avg MFE 10D:   "
            f"{priority_df[
                'MFE 10D %'
            ].mean():.2f}%"
        )

        print(
            "Avg MAE 10D:   "
            f"{priority_df[
                'MAE 10D %'
            ].mean():.2f}%"
        )

    print()
    print(
        f"Output file:"
    )

    print(
        output_file
    )

    print("=" * 70)


if __name__ == "__main__":
    main()