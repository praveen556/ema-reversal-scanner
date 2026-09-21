from datetime import datetime

import pandas as pd
import yfinance as yf
from pathlib import Path

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
    MIN_20D_RANGE_PCT,
    CHOPPY_MAX_COMPRESSION_PCT,
    CHOPPY_MIN_CROSS_COUNT,
    RSI_LENGTH,
    EMA_4H_FAST,
    EMA_4H_SLOW,
    PROGRESS_EVERY,
    NEAR_ENTRY_MAX_PCT,
    EXTENDED_MIN_PCT,
    V2_HIGH_CONF_RSI_MAX,
    V2_HIGH_CONF_RANGE_MAX_PCT,
    V2_HIGH_CONF_COMPRESSION_MAX_PCT,
    V2_SETUP_RSI_MAX,
    V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF,
    V21_PRIORITY_RSI_MIN,
    V21_PRIORITY_RSI_MAX,
    INCLUDE_SP500,
    INCLUDE_NASDAQ100,
    INCLUDE_DOW30,
    INCLUDE_RUSSELL2000,
)

# ============================================================
# LOCAL DAILY DATA
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent

DAILY_DATA_FOLDER = (
    PROJECT_FOLDER / "data" / "daily"
)


def calculate_ema(series, length):
    """Calculate an exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()


def calculate_rsi(series, length=14):
    """Calculate RSI using Wilder-style exponential smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def clean_ticker(ticker):
    """
    Convert symbols to Yahoo Finance format.
    Example:
        BRK.B -> BRK-B
        BF.B  -> BF-B
    """
    ticker = str(ticker).strip().upper()

    if not ticker or ticker == "NAN":
        return None

    ticker = ticker.replace(".", "-")

    return ticker


def get_sp500_tickers():
    print("Loading S&P 500...")

    url = (
        "https://raw.githubusercontent.com/"
        "datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )

    df = pd.read_csv(url)

    tickers = [
        clean_ticker(ticker)
        for ticker in df["Symbol"].tolist()
    ]

    tickers = [t for t in tickers if t]

    print(f"S&P 500 loaded: {len(tickers)}")

    return tickers


def get_nasdaq100_tickers():
    print("Loading Nasdaq-100...")

    url = (
        "https://raw.githubusercontent.com/"
        "Gary-Strauss/NASDAQ100_Constituents/"
        "master/data/nasdaq100_constituents.csv"
    )

    try:
        df = pd.read_csv(url)

        ticker_column = None

        for column in df.columns:
            if column.lower() in (
                "ticker",
                "symbol",
            ):
                ticker_column = column
                break

        if ticker_column is None:
            raise ValueError(
                "Could not identify Nasdaq ticker column."
            )

        tickers = [
            clean_ticker(ticker)
            for ticker in df[ticker_column].tolist()
        ]

        tickers = [t for t in tickers if t]

        print(f"Nasdaq-100 loaded: {len(tickers)}")

        return tickers

    except Exception as exc:
        print(f"Nasdaq-100 loading failed: {exc}")
        return []

def get_dow30_tickers():
    print("Loading Dow 30...")

    url = (
        "https://yfiua.github.io/"
        "index-constituents/"
        "constituents-dowjones.csv"
    )

    try:
        df = pd.read_csv(url)

        ticker_column = None

        for column in df.columns:
            if column.lower() in (
                "ticker",
                "symbol",
            ):
                ticker_column = column
                break

        if ticker_column is None:
            raise ValueError(
                "Could not identify Dow ticker column."
            )

        tickers = [
            clean_ticker(ticker)
            for ticker in df[ticker_column].tolist()
        ]

        tickers = [t for t in tickers if t]

        print(f"Dow 30 loaded: {len(tickers)}")

        return tickers

    except Exception as exc:
        print(f"Dow 30 loading failed: {exc}")
        return []

def get_russell2000_tickers():
    """
    Load a current Russell 2000 constituent list.
    The source is maintained from IWM holdings.
    """

    print("Loading Russell 2000...")

    url = (
        "https://raw.githubusercontent.com/"
        "benjaminpo/finance-dataset/"
        "main/config/listings/"
        "russell2000-constituents.csv"
    )

    try:
        df = pd.read_csv(url)

        ticker_column = None

        for column in df.columns:
            if column.strip().lower() in (
                "symbol",
                "ticker",
            ):
                ticker_column = column
                break

        if ticker_column is None:
            raise ValueError(
                f"Ticker column not found. "
                f"Columns: {list(df.columns)}"
            )

        tickers = []

        for raw_ticker in df[ticker_column]:
            ticker = clean_ticker(raw_ticker)

            if ticker is None:
                continue

            if len(ticker) > 6:
                continue

            tickers.append(ticker)

        tickers = list(dict.fromkeys(tickers))

        if len(tickers) < 1500:
            raise ValueError(
                f"Only {len(tickers)} Russell 2000 "
                f"tickers were loaded."
            )

        print(
            f"Russell 2000 loaded: {len(tickers)}"
        )

        return tickers

    except Exception as exc:
        print(
            f"Russell 2000 loading failed: {exc}"
        )

        return []
def get_nasdaq_listed_stocks():
    """
    Load Nasdaq-listed non-ETF securities from
    the official Nasdaq Trader Symbol Directory.
    """

    print("Loading Nasdaq-listed stocks...")

    url = (
        "https://www.nasdaqtrader.com/"
        "dynamic/SymDir/nasdaqlisted.txt"
    )

    try:
        df = pd.read_csv(
            url,
            sep="|",
            dtype=str,
        )

        # Remove Nasdaq's file-creation footer
        df = df[
            ~df["Symbol"]
            .fillna("")
            .str.startswith("File Creation Time")
        ].copy()

        # Remove test issues
        if "Test Issue" in df.columns:
            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]

        # Remove ETFs when the field is available
        if "ETF" in df.columns:
            df = df[
                df["ETF"].fillna("N") != "Y"
            ]

        tickers = []

        for raw_ticker in df["Symbol"]:
            ticker = clean_ticker(raw_ticker)

            if ticker is None:
                continue

            tickers.append(ticker)

        tickers = list(dict.fromkeys(tickers))

        print(
            f"Nasdaq-listed stocks loaded: "
            f"{len(tickers)}"
        )

        return tickers

    except Exception as exc:
        print(
            f"Nasdaq stock list loading failed: {exc}"
        )
        return []


def get_other_exchange_stocks():
    """
    Load NYSE and NYSE American non-ETF securities
    from the official Nasdaq Trader Symbol Directory.
    """

    print("Loading NYSE / NYSE American stocks...")

    url = (
        "https://www.nasdaqtrader.com/"
        "dynamic/SymDir/otherlisted.txt"
    )

    try:
        df = pd.read_csv(
            url,
            sep="|",
            dtype=str,
        )

        # Remove Nasdaq's file-creation footer
        df = df[
            ~df["ACT Symbol"]
            .fillna("")
            .str.startswith("File Creation Time")
        ].copy()

        # Keep:
        # N = NYSE
        # A = NYSE American / NYSE MKT
        df = df[
            df["Exchange"].isin(["N", "A"])
        ].copy()

        # Remove ETFs
        if "ETF" in df.columns:
            df = df[
                df["ETF"].fillna("N") != "Y"
            ]

        # Remove test issues
        if "Test Issue" in df.columns:
            df = df[
                df["Test Issue"].fillna("N") != "Y"
            ]

        tickers = []

        for raw_ticker in df["ACT Symbol"]:
            ticker = clean_ticker(raw_ticker)

            if ticker is None:
                continue

            tickers.append(ticker)

        tickers = list(dict.fromkeys(tickers))

        print(
            f"NYSE / NYSE American stocks loaded: "
            f"{len(tickers)}"
        )

        return tickers

    except Exception as exc:
        print(
            f"NYSE stock list loading failed: {exc}"
        )
        return []


def get_stock_universe():

    if SCAN_MODE == "TEST":
        print("Scanner mode: TEST")

        ticker_sources = {
            clean_ticker(ticker): ["TEST"]
            for ticker in TEST_TICKERS
        }

        return list(ticker_sources.keys()), ticker_sources


    if SCAN_MODE == "SP500":
        print("Scanner mode: S&P 500")

        sp500 = get_sp500_tickers()

        ticker_sources = {
            ticker: ["S&P 500"]
            for ticker in sp500
        }

        return sp500, ticker_sources


    if SCAN_MODE == "US_MULTI_INDEX":

        print("=" * 60)
        print("Scanner mode: US MULTI-INDEX")
        print("=" * 60)

        ticker_sources = {}


        def add_tickers(tickers, universe_name):

            for ticker in tickers:

                if ticker not in ticker_sources:
                    ticker_sources[ticker] = []

                if universe_name not in ticker_sources[ticker]:
                    ticker_sources[ticker].append(
                        universe_name
                    )


        if INCLUDE_SP500:
            add_tickers(
                get_sp500_tickers(),
                "S&P 500",
            )


        if INCLUDE_NASDAQ100:
            add_tickers(
                get_nasdaq100_tickers(),
                "Nasdaq-100",
            )


        if INCLUDE_DOW30:
            add_tickers(
                get_dow30_tickers(),
                "Dow 30",
            )


        if INCLUDE_RUSSELL2000:
            add_tickers(
                get_russell2000_tickers(),
                "Russell 2000",
            )


        tickers = list(ticker_sources.keys())

        print()
        print("=" * 60)
        print(
            f"TOTAL UNIQUE STOCKS: "
            f"{len(tickers)}"
        )
        print("=" * 60)

        return tickers, ticker_sources

    if SCAN_MODE == "US_ALL_STOCKS":

        print("=" * 60)
        print("Scanner mode: US ALL STOCKS")
        print("=" * 60)

        ticker_sources = {}

        def add_tickers(tickers, universe_name):
            for ticker in tickers:

                if ticker not in ticker_sources:
                    ticker_sources[ticker] = []

                if universe_name not in ticker_sources[ticker]:
                    ticker_sources[ticker].append(
                        universe_name
                    )

        # Nasdaq-listed stocks
        add_tickers(
            get_nasdaq_listed_stocks(),
            "NASDAQ",
        )

        # NYSE + NYSE American
        add_tickers(
            get_other_exchange_stocks(),
            "NYSE / NYSE American",
        )

        tickers = list(
            ticker_sources.keys()
        )

        print()
        print("=" * 60)
        print(
            f"TOTAL UNIQUE US STOCKS: "
            f"{len(tickers)}"
        )
        print("=" * 60)

        return tickers, ticker_sources

    raise ValueError(
        f"Unknown SCAN_MODE: {SCAN_MODE}"
    )

def load_daily_data_from_local(tickers):
    """
    Load locally cached Daily stock data.

    Expected files:
        data/daily/AAPL.csv
        data/daily/FN.csv
        data/daily/NVDA.csv
        etc.

    Returns one combined DataFrame with the same
    ticker -> OHLCV structure expected by the scanner.
    """

    print()
    print("=" * 70)
    print("LOADING DAILY DATA FROM LOCAL DATABASE")
    print("=" * 70)

    print(
        f"Requested stocks: "
        f"{len(tickers)}"
    )

    print(
        f"Data folder: "
        f"{DAILY_DATA_FOLDER}"
    )

    ticker_frames = {}
    missing_files = []
    invalid_files = []
    insufficient_history = []

    for number, ticker in enumerate(
        tickers,
        start=1,
    ):

        file_path = (
            DAILY_DATA_FOLDER /
            f"{ticker}.csv"
        )

        # ----------------------------------------------------
        # FILE DOES NOT EXIST
        # ----------------------------------------------------

        if not file_path.exists():
            missing_files.append(ticker)
            continue

        try:

            # ------------------------------------------------
            # READ LOCAL CSV
            # ------------------------------------------------

            df = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=True,
            )

            # ------------------------------------------------
            # BASIC VALIDATION
            # ------------------------------------------------

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                invalid_files.append(ticker)
                continue

            # ------------------------------------------------
            # REMOVE EMPTY PRICE ROWS
            # ------------------------------------------------

            df = df.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ],
                how="all",
            )

            if df.empty:
                invalid_files.append(ticker)
                continue

            # ------------------------------------------------
            # SORT / REMOVE DUPLICATE DATES
            # ------------------------------------------------

            df = df.sort_index()

            df = df[
                ~df.index.duplicated(
                    keep="last"
                )
            ]

            # ------------------------------------------------
            # TRACK STOCKS WITH < EMA200 HISTORY
            #
            # Keep them out of the combined scanner data
            # because EMA200 cannot yet be calculated
            # reliably.
            # ------------------------------------------------

            if len(df) < EMA_SLOW:
                insufficient_history.append(
                    ticker
                )
                continue

            # ------------------------------------------------
            # CONVERT TO SAME STRUCTURE AS OLD YAHOO BATCH
            #
            # Before:
            # batch_data["FN"]
            #
            # After:
            # batch_data["FN"]
            #
            # So scan_ticker_from_batch() does not need
            # to know whether data came from Yahoo or disk.
            # ------------------------------------------------

            df.columns = (
                pd.MultiIndex.from_product(
                    [[ticker], df.columns]
                )
            )

            ticker_frames[ticker] = df

        except Exception as exc:

            print(
                f"ERROR reading {ticker}: "
                f"{exc}"
            )

            invalid_files.append(ticker)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            number % 500 == 0
            or number == len(tickers)
        ):
            print(
                f"Checked "
                f"{number}/{len(tickers)} "
                f"stocks..."
            )

    # ========================================================
    # COMBINE LOCAL STOCK DATA
    # ========================================================

    if not ticker_frames:
        raise RuntimeError(
            "No usable local Daily data was found."
        )

    print()
    print("Combining local Daily data...")

    ordered_frames = [
        ticker_frames[ticker]
        for ticker in tickers
        if ticker in ticker_frames
    ]

    combined_data = pd.concat(
        ordered_frames,
        axis=1,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("LOCAL DAILY DATA LOADED")
    print("=" * 70)

    print(
        f"Universe stocks:         "
        f"{len(tickers)}"
    )

    print(
        f"Usable local stocks:     "
        f"{len(ticker_frames)}"
    )

    print(
        f"Missing local files:     "
        f"{len(missing_files)}"
    )

    print(
        f"Invalid local files:     "
        f"{len(invalid_files)}"
    )

    print(
        f"Insufficient history:    "
        f"{len(insufficient_history)}"
    )

    # --------------------------------------------------------
    # SAVE DIAGNOSTIC LISTS
    # --------------------------------------------------------

    diagnostic_folder = (
        PROJECT_FOLDER /
        "output"
    )

    diagnostic_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        diagnostic_folder /
        "missing_local_daily.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in missing_files:
            file.write(
                f"{ticker}\n"
            )

    with open(
        diagnostic_folder /
        "invalid_local_daily.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in invalid_files:
            file.write(
                f"{ticker}\n"
            )

    with open(
        diagnostic_folder /
        "insufficient_history.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in insufficient_history:
            file.write(
                f"{ticker}\n"
            )

    print()
    print(
        "Diagnostic lists saved "
        "in output folder."
    )

    print("=" * 70)

    return combined_data


def download_hourly_batch(tickers):
    """Download 1H data only for Daily reversal candidates."""
    if not tickers:
        return None

    print()
    print(f"Downloading 1H data for {len(tickers)} Daily candidates...")

    data = yf.download(
        tickers=tickers,
        period="60d",
        interval="1h",
        auto_adjust=False,
        progress=False,
        prepost=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        print("WARNING: No 1H data downloaded.")
        return None

    print("1H batch download completed.")
    return data


def get_4h_momentum_from_batch(ticker, hourly_batch):
    """
    Build approximate regular-session 4H bars from Yahoo 1H data.

    Each trading day is kept separate:
    - first block: first four hourly observations
    - second block: remaining regular-session observations
    """
    if hourly_batch is None:
        return False, None, None, None

    try:
        hourly = hourly_batch[ticker].copy()
    except KeyError:
        print(f"WARNING: No 1H batch data for {ticker}")
        return False, None, None, None

    hourly = hourly.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    ).copy()

    if hourly.empty:
        return False, None, None, None

    if hourly.index.tz is not None:
        hourly.index = hourly.index.tz_convert("America/New_York")

    hourly = hourly.between_time("09:30", "16:00")

    four_hour_bars = []

    for _, day in hourly.groupby(hourly.index.date):
        day = day.sort_index()
        if day.empty:
            continue

        for block in (day.iloc[0:4], day.iloc[4:]):
            if block.empty:
                continue

            four_hour_bars.append(
                {
                    "Datetime": block.index[0],
                    "Open": block["Open"].iloc[0],
                    "High": block["High"].max(),
                    "Low": block["Low"].min(),
                    "Close": block["Close"].iloc[-1],
                    "Volume": block["Volume"].sum(),
                }
            )

    if not four_hour_bars:
        return False, None, None, None

    four_hour = pd.DataFrame(four_hour_bars).set_index("Datetime")
    four_hour["EMA5"] = calculate_ema(four_hour["Close"], EMA_4H_FAST)
    four_hour["EMA13"] = calculate_ema(four_hour["Close"], EMA_4H_SLOW)

    latest = four_hour.iloc[-1]
    close_4h = float(latest["Close"])
    ema5 = float(latest["EMA5"])
    ema13 = float(latest["EMA13"])

    bullish = close_4h > ema13 and ema5 > ema13
    return bullish, close_4h, ema5, ema13


def scan_ticker_from_batch(ticker, batch_data):
    """Run the Daily EMA reversal analysis using pre-downloaded data."""
    try:
        df = batch_data[ticker].copy()
    except KeyError:
        print(f"WARNING: No batch data for {ticker}")
        return None

    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()

    if len(df) < EMA_SLOW:
        print(f"WARNING: Not enough history for {ticker}")
        return None

    # =========================================================
    # DAILY INDICATORS
    # =========================================================
    df["EMA20"] = calculate_ema(df["Close"], EMA_20)
    df["EMA50"] = calculate_ema(df["Close"], EMA_50)
    df["EMA89"] = calculate_ema(df["Close"], EMA_FAST)
    df["EMA200"] = calculate_ema(df["Close"], EMA_SLOW)
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = calculate_rsi(df["Close"], RSI_LENGTH)

    # EMA touch/reclaim.
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

    latest = df.iloc[-1]
    five_days_ago = df.iloc[-6]

    price = float(latest["Close"])
    avg_volume = float(latest["AvgVolume20"])
    current_volume = float(latest["Volume"])
    current_rsi = float(latest["RSI14"])

    # =========================================================
    # DAILY CANDLE STRENGTH
    # =========================================================

    daily_open = float(latest["Open"])
    daily_high = float(latest["High"])
    daily_low = float(latest["Low"])
    daily_close = price

    daily_bullish_candle = (
        daily_close > daily_open
    )

    # Body size as % of opening price
    daily_body_pct = (
        abs(daily_close - daily_open)
        / daily_open
        * 100
        if daily_open > 0
        else None
    )

    # Where did price close inside today's entire candle?
    # 0% = at the low
    # 50% = middle
    # 100% = at the high
    daily_range = daily_high - daily_low

    daily_close_position_pct = (
        (daily_close - daily_low)
        / daily_range
        * 100
        if daily_range > 0
        else 50.0
    )

    price_ok = price > MIN_PRICE
    volume_ok = avg_volume > MIN_AVG_VOLUME

    current_volume_ratio = (
        current_volume / avg_volume
        if pd.notna(avg_volume) and avg_volume > 0
        else None
    )

    # =========================================================
    # EMA SLOPES
    # =========================================================
    def slope_pct(column):
        old_value = float(five_days_ago[column])
        new_value = float(latest[column])
        return ((new_value - old_value) / old_value) * 100

    ema20_slope_pct = slope_pct("EMA20")
    ema50_slope_pct = slope_pct("EMA50")
    ema89_slope_pct = slope_pct("EMA89")
    ema200_slope_pct = slope_pct("EMA200")

    # =========================================================
    # MOST RECENT EMA TOUCH WITHIN LOOKBACK
    # =========================================================
    recent = df.iloc[-LOOKBACK_DAYS:]

    touch89_positions = [
        i for i in range(len(recent)) if bool(recent["Touch89"].iloc[i])
    ]
    touch200_positions = [
        i for i in range(len(recent)) if bool(recent["Touch200"].iloc[i])
    ]

    days_ago_89 = (
        len(recent) - 1 - touch89_positions[-1]
        if touch89_positions
        else None
    )
    days_ago_200 = (
        len(recent) - 1 - touch200_positions[-1]
        if touch200_positions
        else None
    )

    recent89 = (
        days_ago_89 is not None
        and price > float(latest["EMA89"])
    )
    recent200 = (
        days_ago_200 is not None
        and price > float(latest["EMA200"])
    )

    signal89 = recent89 and price_ok and volume_ok
    signal200 = recent200 and price_ok and volume_ok

    # =========================================================
    # DAILY TREND STRUCTURE
    # =========================================================
    above_ema20 = price > float(latest["EMA20"])
    above_ema50 = price > float(latest["EMA50"])
    above_ema89 = price > float(latest["EMA89"])

    structure89_ok = above_ema20 and above_ema50
    structure200_ok = above_ema89

    # =========================================================
    # EMA COMPRESSION
    # =========================================================
    ema_values = [
        float(latest["EMA20"]),
        float(latest["EMA50"]),
        float(latest["EMA89"]),
        float(latest["EMA200"]),
    ]
    ema_compression_pct = (
        (max(ema_values) - min(ema_values)) / price * 100
    )

    # =========================================================
    # 20-DAY RANGE + EMA CROSSING DIAGNOSTICS
    # =========================================================
    recent_20 = df.iloc[-20:]

    high_20 = float(recent_20["High"].max())
    low_20 = float(recent_20["Low"].min())

    range_20_pct = (
        ((high_20 - low_20) / low_20) * 100
        if low_20 > 0
        else 0.0
    )
    range_20_ok = range_20_pct >= MIN_20D_RANGE_PCT

    range_position_20 = (
        ((price - low_20) / (high_20 - low_20)) * 100
        if high_20 > low_20
        else 50.0
    )

    close_above_ema89 = recent_20["Close"] > recent_20["EMA89"]
    close_above_ema200 = recent_20["Close"] > recent_20["EMA200"]

    ema89_cross_count = int(
        (close_above_ema89 != close_above_ema89.shift(1))
        .iloc[1:]
        .sum()
    )
    ema200_cross_count = int(
        (close_above_ema200 != close_above_ema200.shift(1))
        .iloc[1:]
        .sum()
    )
    ema_cross_count_20d = ema89_cross_count + ema200_cross_count

    choppy_warning = (
        ema_compression_pct < CHOPPY_MAX_COMPRESSION_PCT
        and ema_cross_count_20d >= CHOPPY_MIN_CROSS_COUNT
    )

    # =========================================================
    # DETERMINE DAILY SIGNAL + STRUCTURE
    # =========================================================
    if signal89 and signal200:
        signal = "EMA89 + EMA200"
        days_ago = min(days_ago_89, days_ago_200)

        distance_pct = min(
            (price - float(latest["EMA89"]))
            / float(latest["EMA89"])
            * 100,
            (price - float(latest["EMA200"]))
            / float(latest["EMA200"])
            * 100,
        )

        # Keep OR intentionally: early reversals can be valid before
        # both EMA structure tests become true.
        daily_structure_ok = structure89_ok or structure200_ok

    elif signal89:
        signal = "EMA89"
        days_ago = days_ago_89
        distance_pct = (
            (price - float(latest["EMA89"]))
            / float(latest["EMA89"])
            * 100
        )
        daily_structure_ok = structure89_ok

    elif signal200:
        signal = "EMA200"
        days_ago = days_ago_200
        distance_pct = (
            (price - float(latest["EMA200"]))
            / float(latest["EMA200"])
            * 100
        )
        daily_structure_ok = structure200_ok

    else:
        signal = "NO SIGNAL"
        days_ago = None
        distance_pct = None
        daily_structure_ok = False

    # =========================================================
    # REVERSAL-DAY MOMENTUM / VOLUME
    # =========================================================
    reversal_rsi = None
    reversal_volume_ratio = None
    rsi_change_since_reversal = None
    price_change_since_reversal_pct = None
    momentum_confirmation = None
    entry_stage = None

    if days_ago is not None:
        reversal_row = df.iloc[-(days_ago + 1)]

        reversal_volume = float(reversal_row["Volume"])
        reversal_avg_volume = float(reversal_row["AvgVolume20"])
        reversal_rsi = float(reversal_row["RSI14"])
        reversal_price = float(reversal_row["Close"])

        if pd.notna(reversal_avg_volume) and reversal_avg_volume > 0:
            reversal_volume_ratio = (
                reversal_volume / reversal_avg_volume
            )

        rsi_change_since_reversal = current_rsi - reversal_rsi

        if reversal_price > 0:
            price_change_since_reversal_pct = (
                (price - reversal_price) / reversal_price * 100
            )

        # Diagnostic only. It does NOT affect QUALIFIED/WATCH.
        if days_ago == 0:
            momentum_confirmation = "EARLY"
        elif (
            rsi_change_since_reversal > 0
            and price_change_since_reversal_pct > 0
        ):
            momentum_confirmation = "CONFIRMED"
        elif (
            rsi_change_since_reversal < 0
            and price_change_since_reversal_pct < 0
        ):
            momentum_confirmation = "WEAKENING"
        else:
            momentum_confirmation = "MIXED"

        # =====================================================
        # ENTRY STAGE
        # =====================================================

        if momentum_confirmation == "EARLY":
            entry_stage = "EARLY REVERSAL"

        elif momentum_confirmation == "WEAKENING":
            entry_stage = "LOSING MOMENTUM"

        elif momentum_confirmation == "MIXED":
            entry_stage = "MIXED"

        elif momentum_confirmation == "CONFIRMED":

            if (
                price_change_since_reversal_pct
                <= NEAR_ENTRY_MAX_PCT
            ):
                entry_stage = "CONFIRMED NEAR ENTRY"

            elif (
                price_change_since_reversal_pct
                >= EXTENDED_MIN_PCT
            ):
                entry_stage = "EXTENDED"

            else:
                entry_stage = "CONFIRMED"

    return {
        "Ticker": ticker,
        "Signal": signal,
        "Daily Structure OK": daily_structure_ok,
        "Days Ago": days_ago,
        "Distance %": round(distance_pct, 2) if distance_pct is not None else None,
        "Price": round(price, 2),

        "Daily Bullish Candle": daily_bullish_candle,

        "Daily Body %": (
            round(daily_body_pct, 2)
            if daily_body_pct is not None
            else None
        ),

        "Daily Close Position %": round(
            daily_close_position_pct,
            2
        ),

        "EMA20": round(float(latest["EMA20"]), 2),
        "EMA50": round(float(latest["EMA50"]), 2),
        "EMA89": round(float(latest["EMA89"]), 2),
        "EMA200": round(float(latest["EMA200"]), 2),

        "Above EMA20": above_ema20,
        "Above EMA50": above_ema50,
        "Above EMA89": above_ema89,

        "EMA20 Slope %": round(ema20_slope_pct, 2),
        "EMA50 Slope %": round(ema50_slope_pct, 2),
        "EMA89 Slope %": round(ema89_slope_pct, 2),
        "EMA200 Slope %": round(ema200_slope_pct, 2),
        "EMA Compression %": round(ema_compression_pct, 2),

        "20D High": round(high_20, 2),
        "20D Low": round(low_20, 2),
        "20D Range %": round(range_20_pct, 2),
        "20D Range OK": range_20_ok,
        "20D Range Position %": round(range_position_20, 2),

        "20D EMA89 Crosses": ema89_cross_count,
        "20D EMA200 Crosses": ema200_cross_count,
        "20D EMA Cross Count": ema_cross_count_20d,
        "Choppy Warning": choppy_warning,

        "RSI14": round(current_rsi, 2),
        "Reversal RSI14": (
            round(reversal_rsi, 2)
            if reversal_rsi is not None
            else None
        ),
        "RSI Change Since Reversal": (
            round(rsi_change_since_reversal, 2)
            if rsi_change_since_reversal is not None
            else None
        ),
        "Price Change Since Reversal %": (
            round(price_change_since_reversal_pct, 2)
            if price_change_since_reversal_pct is not None
            else None
        ),
        "Momentum Confirmation": momentum_confirmation,
        "Entry Stage": entry_stage,
        "Current Volume Ratio": (
            round(current_volume_ratio, 2)
            if current_volume_ratio is not None
            else None
        ),
        "Reversal Volume Ratio": (
            round(reversal_volume_ratio, 2)
            if reversal_volume_ratio is not None
            else None
        ),

        "20D Avg Volume": int(avg_volume),
        "Price OK": price_ok,
        "Volume OK": volume_ok,
    }


def classify_stock(row):
    has_daily_setup = row["Signal"] != "NO SIGNAL"
    basic_filters_ok = bool(row["Price OK"] and row["Volume OK"])

    if not (has_daily_setup and basic_filters_ok):
        return "NO SETUP"

    rsi = row["RSI14"]
    range_20 = row["20D Range %"]
    compression = row["EMA Compression %"]
    choppy = bool(row["Choppy Warning"])
    signal = row["Signal"]

    # -----------------------------------------------------
    # EXISTING V2 HIGH CONFIDENCE LOGIC
    # -----------------------------------------------------
    high_confidence = (
        pd.notna(rsi)
        and pd.notna(range_20)
        and pd.notna(compression)
        and rsi < V2_HIGH_CONF_RSI_MAX
        and range_20 < V2_HIGH_CONF_RANGE_MAX_PCT
        and compression < V2_HIGH_CONF_COMPRESSION_MAX_PCT
        and (
            not V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF
            or not choppy
        )
    )

    # -----------------------------------------------------
    # V2.1 PRIORITY
    #
    # Must FIRST satisfy all existing HIGH CONFIDENCE rules.
    # Then:
    #   1. EMA200-only signal
    #   2. RSI >= 40
    #   3. RSI < 45
    # -----------------------------------------------------
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

    if pd.notna(rsi) and rsi < V2_SETUP_RSI_MAX:
        return "SETUP"

    return "OBSERVE"


def main():
    print("=" * 60)
    print("EMA REVERSAL SCANNER - V2")
    print("=" * 60)

    tickers, ticker_sources = get_stock_universe()

    total_tickers = len(tickers)

    # =========================================================
    # STEP 1: DAILY BATCH DOWNLOAD + ANALYSIS
    # =========================================================
    batch_data = load_daily_data_from_local(
        tickers
    )
    results = []

    print()
    print(f"Analyzing Daily data for {total_tickers} stocks...")

    for number, ticker in enumerate(tickers, start=1):
        try:
            result = scan_ticker_from_batch(ticker, batch_data)

            if result is not None:
                result["Universe"] = " | ".join(
                    ticker_sources.get(ticker, ["Unknown"])
                )

                results.append(result)

        except Exception as exc:
            print(f"ERROR analyzing {ticker}: {exc}")

        if number % PROGRESS_EVERY == 0 or number == total_tickers:
            print(
                f"Daily analysis: "
                f"{number} / {total_tickers} completed"
            )

    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)

    # =========================================================
    # STEP 2: DAILY CANDIDATES
    # =========================================================
    daily_candidates = results_df[
        (results_df["Signal"] != "NO SIGNAL")
        & results_df["Price OK"]
        & results_df["Volume OK"]
    ].copy()

    candidate_tickers = daily_candidates["Ticker"].tolist()

    print()
    print(f"Daily reversal candidates: {len(candidate_tickers)}")

    # =========================================================
    # STEP 3: 4H COLUMNS + CANDIDATE-ONLY 4H ANALYSIS
    # =========================================================
    results_df["4H Bullish"] = False
    results_df["4H Close"] = None
    results_df["4H EMA5"] = None
    results_df["4H EMA13"] = None

    if candidate_tickers:
        hourly_batch = download_hourly_batch(candidate_tickers)

        print()
        print(
            f"Calculating 4H momentum for "
            f"{len(candidate_tickers)} candidates..."
        )

        for number, ticker in enumerate(candidate_tickers, start=1):
            try:
                bullish, close_4h, ema5_4h, ema13_4h = (
                    get_4h_momentum_from_batch(ticker, hourly_batch)
                )

                mask = results_df["Ticker"] == ticker

                results_df.loc[mask, "4H Bullish"] = bullish
                results_df.loc[mask, "4H Close"] = (
                    round(close_4h, 2)
                    if close_4h is not None
                    else None
                )
                results_df.loc[mask, "4H EMA5"] = (
                    round(ema5_4h, 2)
                    if ema5_4h is not None
                    else None
                )
                results_df.loc[mask, "4H EMA13"] = (
                    round(ema13_4h, 2)
                    if ema13_4h is not None
                    else None
                )

            except Exception as exc:
                print(f"ERROR calculating 4H for {ticker}: {exc}")

            if (
                number % PROGRESS_EVERY == 0
                or number == len(candidate_tickers)
            ):
                print(
                    f"4H analysis: "
                    f"{number} / {len(candidate_tickers)} completed"
                )

        print("4H confirmation completed.")

    # =========================================================
    # STEP 4: CLASSIFICATION
    # =========================================================
    results_df["Status"] = results_df.apply(
        classify_stock,
        axis=1,
    )

    # =========================================================
    # STEP 5: OUTPUT COLUMN ORDER
    # =========================================================
    column_order = [
        "Ticker",
        "Universe",
        "Status",
        "Signal",
        "Daily Structure OK",
        "Days Ago",
        "Distance %",
        "Price",

        "Daily Bullish Candle",
        "Daily Body %",
        "Daily Close Position %",

        "EMA20",
        "EMA50",
        "EMA89",
        "EMA200",
        "Above EMA20",
        "Above EMA50",
        "Above EMA89",
        "EMA20 Slope %",
        "EMA50 Slope %",
        "EMA89 Slope %",
        "EMA200 Slope %",
        "EMA Compression %",
        "20D High",
        "20D Low",
        "20D Range %",
        "20D Range OK",
        "20D Range Position %",
        "20D EMA89 Crosses",
        "20D EMA200 Crosses",
        "20D EMA Cross Count",
        "Choppy Warning",
        "RSI14",
        "Reversal RSI14",
        "RSI Change Since Reversal",
        "Price Change Since Reversal %",
        "Momentum Confirmation",
        "Entry Stage",
        "Current Volume Ratio",
        "Reversal Volume Ratio",
        "4H Bullish",
        "4H Close",
        "4H EMA5",
        "4H EMA13",
        "20D Avg Volume",
        "Price OK",
        "Volume OK",
    ]

    results_df = results_df[column_order]

    # =========================================================
    # STEP 6: SORT + GROUP
    # =========================================================
    status_order = {
        "PRIORITY": 1,
        "HIGH CONFIDENCE": 2,
        "SETUP": 3,
        "OBSERVE": 4,
        "NO SETUP": 5,
    }

    results_df["SortOrder"] = results_df["Status"].map(status_order)

    results_df = (
        results_df
        .sort_values(
            by=["SortOrder", "Days Ago", "Distance %", "Ticker"],
            na_position="last",
        )
        .drop(columns=["SortOrder"])
    )

    priority_df = results_df[
        results_df["Status"] == "PRIORITY"
        ].copy()

    high_confidence_df = results_df[
        results_df["Status"] == "HIGH CONFIDENCE"
    ].copy()
    setup_df = results_df[
        results_df["Status"] == "SETUP"
    ].copy()
    observe_df = results_df[
        results_df["Status"] == "OBSERVE"
    ].copy()
    all_df = results_df.copy()

    # =========================================================
    # STEP 7: SAVE CSV + EXCEL
    # =========================================================
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if SCAN_MODE == "SP500":
        file_label = "SP500"

    elif SCAN_MODE == "US_MULTI_INDEX":
        file_label = "US_Multi_Index"

    elif SCAN_MODE == "US_ALL_STOCKS":
        file_label = "US_All_Stocks"

    elif SCAN_MODE == "TEST":
        file_label = "Test"

    else:
        file_label = "Unknown"

    csv_file = (
        output_dir
        / f"EMA_Reversal_V2_{file_label}_{timestamp}.csv"
    )
    excel_file = (
        output_dir
        / f"EMA_Reversal_V2_{file_label}_{timestamp}.xlsx"
    )

    results_df.to_csv(csv_file, index=False)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        priority_df.to_excel(
            writer,
            sheet_name="PRIORITY",
            index=False,
        )
        high_confidence_df.to_excel(
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
        all_df.to_excel(
            writer,
            sheet_name="ALL STOCKS",
            index=False,
        )

        for sheet_name in (
                "PRIORITY",
                "HIGH CONFIDENCE",
                "SETUP",
                "OBSERVE",
                "ALL STOCKS",
        ):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    (
                        len(str(cell.value))
                        for cell in column_cells
                        if cell.value is not None
                    ),
                    default=0,
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(
                    max_length + 3,
                    24,
                )


    # =========================================================
    # STEP 8: SUMMARY
    # =========================================================
    no_setup_count = len(results_df) - (
        len(high_confidence_df)
        + len(setup_df)
        + len(observe_df)
    )

    print()
    print("=" * 70)
    print("V2 SCAN SUMMARY")
    print("=" * 70)
    print(f"Stocks scanned:     {len(results_df)}")
    print(f"High Confidence:    {len(high_confidence_df)}")
    print(f"Setup:              {len(setup_df)}")
    print(f"Observe:            {len(observe_df)}")
    print(f"No Setup:           {no_setup_count}")

    display_columns = [
        "Ticker",
        "Signal",
        "Days Ago",
        "Distance %",
        "RSI14",
        "20D Range %",
        "EMA Compression %",
        "Choppy Warning",
        "Price",
        "4H Bullish",
    ]

    print()
    print("=" * 70)
    print("HIGH CONFIDENCE")
    print("=" * 70)

    if high_confidence_df.empty:
        print("No High Confidence stocks.")
    else:
        print(
            high_confidence_df[display_columns].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("SETUP")
    print("=" * 70)

    if setup_df.empty:
        print("No Setup candidates.")
    else:
        print(
            setup_df[display_columns].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("OBSERVE")
    print("=" * 70)

    if observe_df.empty:
        print("No Observe candidates.")
    else:
        print(
            observe_df[display_columns].to_string(index=False)
        )

    print(f"CSV saved to:   {csv_file}")
    print(f"Excel saved to: {excel_file}")


if __name__ == "__main__":
    main()
