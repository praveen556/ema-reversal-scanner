import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config_v2 import (
    INITIAL_DAILY_PERIOD,
    DAILY_BATCH_SIZE,
    DAILY_RETRY_BATCH_SIZE,
    DAILY_BATCH_PAUSE_SECONDS,
    DAILY_RETRY_PAUSE_SECONDS,
    DAILY_MAX_RETRY_ROUNDS,
    UPDATE_LOOKBACK_DAYS,
)

from scanner_v2 import get_stock_universe


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent

DAILY_DATA_FOLDER = (
    PROJECT_FOLDER / "data" / "daily"
)

DAILY_DATA_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NORMALIZE YAHOO OUTPUT
# ============================================================

def normalize_download(data, requested_tickers):
    """
    Make Yahoo output consistently use:

        ticker -> OHLCV field

    This handles both single-ticker and
    multi-ticker downloads.
    """

    if data is None or data.empty:
        return data

    # --------------------------------------------------------
    # SINGLE-LEVEL COLUMNS
    # --------------------------------------------------------

    if not isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if len(requested_tickers) == 1:

            ticker = requested_tickers[0]

            data = data.copy()

            data.columns = (
                pd.MultiIndex.from_product(
                    [[ticker], data.columns]
                )
            )

        return data

    # --------------------------------------------------------
    # MULTI-INDEX COLUMNS
    # --------------------------------------------------------

    level_0 = set(
        str(value)
        for value
        in data.columns.get_level_values(0)
    )

    level_1 = set(
        str(value)
        for value
        in data.columns.get_level_values(1)
    )

    requested_set = set(
        requested_tickers
    )

    # Already ticker -> field
    if requested_set.intersection(level_0):
        return data

    # Yahoo returned field -> ticker
    if requested_set.intersection(level_1):

        data = data.swaplevel(
            0,
            1,
            axis=1,
        )

        return data

    return data


# ============================================================
# EXTRACT ONE TICKER FROM YAHOO BATCH
# ============================================================

def extract_ticker_data(
    batch_data,
    ticker,
):

    if (
        batch_data is None
        or batch_data.empty
    ):
        return None

    try:
        ticker_data = (
            batch_data[ticker].copy()
        )

    except Exception:
        return None

    if isinstance(
        ticker_data,
        pd.Series,
    ):
        return None

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required_columns:

        if column not in ticker_data.columns:
            return None

    # Remove rows where price data is completely empty
    ticker_data = ticker_data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
        how="all",
    )

    if ticker_data.empty:
        return None

    ticker_data.index = pd.to_datetime(
        ticker_data.index
    )

    ticker_data.index.name = "Date"

    ticker_data = (
        ticker_data
        .sort_index()
    )

    ticker_data = ticker_data[
        ~ticker_data.index.duplicated(
            keep="last"
        )
    ]

    return ticker_data


# ============================================================
# LOCAL FILE PATH
# ============================================================

def get_local_file(ticker):

    return (
        DAILY_DATA_FOLDER /
        f"{ticker}.csv"
    )


# ============================================================
# READ EXISTING LOCAL FILE
# ============================================================

def read_local_data(ticker):

    file_path = get_local_file(
        ticker
    )

    if not file_path.exists():
        return None

    try:

        df = pd.read_csv(
            file_path,
            index_col=0,
            parse_dates=True,
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

    except Exception as exc:

        print(
            f"Could not read local "
            f"{ticker}: {exc}"
        )

        return None


# ============================================================
# SAVE LOCAL DATA
# ============================================================

def save_local_data(
    ticker,
    data,
):

    if data is None or data.empty:
        return

    file_path = get_local_file(
        ticker
    )

    data = data.copy()

    data.index = pd.to_datetime(
        data.index
    )

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]

    data.index.name = "Date"

    data.to_csv(
        file_path,
        index=True,
    )


# ============================================================
# MERGE OLD + NEW DATA
# ============================================================

def merge_local_and_new(
    old_data,
    new_data,
):

    if old_data is None:
        return new_data

    if new_data is None:
        return old_data

    combined = pd.concat(
        [
            old_data,
            new_data,
        ]
    )

    combined.index = pd.to_datetime(
        combined.index
    )

    combined = combined.sort_index()

    # New Yahoo data is second in concat,
    # so keep="last" replaces overlapping
    # old rows with the newest Yahoo values.
    combined = combined[
        ~combined.index.duplicated(
            keep="last"
        )
    ]

    return combined


# ============================================================
# DOWNLOAD FULL HISTORY
# ============================================================

def download_full_batch(tickers):
    """
    Used ONLY for stocks without a local file.
    """

    if not tickers:
        return None

    try:

        data = yf.download(
            tickers=tickers,
            period=INITIAL_DAILY_PERIOD,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        return normalize_download(
            data,
            tickers,
        )

    except Exception as exc:

        print(
            f"Full-history download "
            f"error: {exc}"
        )

        return None


# ============================================================
# DOWNLOAD RECENT DATA
# ============================================================

def download_recent_batch(tickers):
    """
    Used for stocks that already have local history.

    We intentionally request a small overlapping
    recent window. Overlapping dates are replaced
    during the merge.
    """

    if not tickers:
        return None

    # yfinance accepts period values such as
    # "10d", "30d", etc.
    recent_period = (
        f"{UPDATE_LOOKBACK_DAYS}d"
    )

    try:

        data = yf.download(
            tickers=tickers,
            period=recent_period,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        return normalize_download(
            data,
            tickers,
        )

    except Exception as exc:

        print(
            f"Recent-data download "
            f"error: {exc}"
        )

        return None


# ============================================================
# UPDATE EXISTING STOCKS
# ============================================================

def update_existing_batch(
    tickers,
    successful,
):

    if not tickers:
        return []

    yahoo_data = download_recent_batch(
        tickers
    )

    failed = []

    for ticker in tickers:

        new_data = extract_ticker_data(
            yahoo_data,
            ticker,
        )

        if new_data is None:
            failed.append(ticker)
            continue

        old_data = read_local_data(
            ticker
        )

        # If the file disappeared or became unreadable,
        # don't overwrite it with only a few recent days.
        if old_data is None:
            failed.append(ticker)
            continue

        combined = merge_local_and_new(
            old_data,
            new_data,
        )

        save_local_data(
            ticker,
            combined,
        )

        successful.add(ticker)

    return failed


# ============================================================
# CREATE MISSING STOCKS
# ============================================================

def create_missing_batch(
    tickers,
    successful,
):

    if not tickers:
        return []

    yahoo_data = download_full_batch(
        tickers
    )

    failed = []

    for ticker in tickers:

        ticker_data = extract_ticker_data(
            yahoo_data,
            ticker,
        )

        if ticker_data is None:
            failed.append(ticker)
            continue

        save_local_data(
            ticker,
            ticker_data,
        )

        successful.add(ticker)

    return failed


# ============================================================
# PROCESS IN BATCHES
# ============================================================

def process_in_batches(
    tickers,
    batch_size,
    process_function,
    successful,
    pause_seconds,
    description,
):

    failed = []

    if not tickers:
        return failed

    total_batches = (
        len(tickers)
        + batch_size
        - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(
            0,
            len(tickers),
            batch_size,
        ),
        start=1,
    ):

        batch = tickers[
            start:
            start + batch_size
        ]

        print(
            f"{description} "
            f"{batch_number}/"
            f"{total_batches} "
            f"- {len(batch)} stocks"
        )

        batch_failed = (
            process_function(
                batch,
                successful,
            )
        )

        failed.extend(
            batch_failed
        )

        print(
            f"Successful so far: "
            f"{len(successful)}"
        )

        if batch_failed:

            print(
                f"Failed in this batch: "
                f"{len(batch_failed)}"
            )

        if batch_number < total_batches:

            time.sleep(
                pause_seconds
            )

    return list(
        dict.fromkeys(failed)
    )


# ============================================================
# MAIN INCREMENTAL UPDATE
# ============================================================

def update_database():

    print()
    print("=" * 70)
    print("LOCAL DAILY DATABASE UPDATE")
    print("=" * 70)

    tickers, ticker_sources = (
        get_stock_universe()
    )

    print()
    print(
        f"Current universe: "
        f"{len(tickers)} stocks"
    )

    print(
        f"Local data folder: "
        f"{DAILY_DATA_FOLDER}"
    )

    # ========================================================
    # DIVIDE INTO EXISTING VS MISSING
    # ========================================================

    existing_tickers = []
    missing_tickers = []

    for ticker in tickers:

        file_path = get_local_file(
            ticker
        )

        if file_path.exists():
            existing_tickers.append(
                ticker
            )
        else:
            missing_tickers.append(
                ticker
            )

    print()
    print("=" * 70)
    print("DATABASE STATUS")
    print("=" * 70)

    print(
        f"Existing local files: "
        f"{len(existing_tickers)}"
    )

    print(
        f"Missing local files:  "
        f"{len(missing_tickers)}"
    )

    print("=" * 70)

    successful = set()

    # Keep separate failure lists because an existing
    # ticker should NEVER accidentally be replaced by
    # only a short recent-history download.

    existing_failed = []
    missing_failed = []

    # ========================================================
    # STEP 1
    # UPDATE EXISTING FILES
    # ========================================================

    if existing_tickers:

        print()
        print("=" * 70)
        print(
            "UPDATING EXISTING STOCKS "
            f"({UPDATE_LOOKBACK_DAYS}-DAY OVERLAP)"
        )
        print("=" * 70)

        existing_failed = (
            process_in_batches(
                tickers=existing_tickers,
                batch_size=DAILY_BATCH_SIZE,
                process_function=(
                    update_existing_batch
                ),
                successful=successful,
                pause_seconds=(
                    DAILY_BATCH_PAUSE_SECONDS
                ),
                description="Update batch",
            )
        )

    # ========================================================
    # STEP 2
    # CREATE MISSING FILES
    # ========================================================

    if missing_tickers:

        print()
        print("=" * 70)
        print(
            "DOWNLOADING FULL HISTORY "
            "FOR MISSING STOCKS"
        )
        print("=" * 70)

        missing_failed = (
            process_in_batches(
                tickers=missing_tickers,
                batch_size=(
                    DAILY_RETRY_BATCH_SIZE
                ),
                process_function=(
                    create_missing_batch
                ),
                successful=successful,
                pause_seconds=(
                    DAILY_RETRY_PAUSE_SECONDS
                ),
                description=(
                    "Missing-stock batch"
                ),
            )
        )

    # ========================================================
    # ONE RETRY ROUND
    # ========================================================

    if DAILY_MAX_RETRY_ROUNDS > 0:

        # ----------------------------------------------------
        # RETRY EXISTING STOCK UPDATES
        # ----------------------------------------------------

        if existing_failed:

            print()
            print("=" * 70)
            print(
                "RETRYING FAILED "
                "EXISTING-STOCK UPDATES"
            )
            print("=" * 70)

            time.sleep(
                DAILY_RETRY_PAUSE_SECONDS
            )

            existing_failed = (
                process_in_batches(
                    tickers=existing_failed,
                    batch_size=(
                        DAILY_RETRY_BATCH_SIZE
                    ),
                    process_function=(
                        update_existing_batch
                    ),
                    successful=successful,
                    pause_seconds=(
                        DAILY_RETRY_PAUSE_SECONDS
                    ),
                    description=(
                        "Existing retry"
                    ),
                )
            )

        # ----------------------------------------------------
        # RETRY MISSING FULL-HISTORY STOCKS
        # ----------------------------------------------------

        if missing_failed:

            print()
            print("=" * 70)
            print(
                "RETRYING FAILED "
                "MISSING STOCKS"
            )
            print("=" * 70)

            time.sleep(
                DAILY_RETRY_PAUSE_SECONDS
            )

            missing_failed = (
                process_in_batches(
                    tickers=missing_failed,
                    batch_size=(
                        DAILY_RETRY_BATCH_SIZE
                    ),
                    process_function=(
                        create_missing_batch
                    ),
                    successful=successful,
                    pause_seconds=(
                        DAILY_RETRY_PAUSE_SECONDS
                    ),
                    description=(
                        "Missing retry"
                    ),
                )
            )

    # ========================================================
    # FINAL FAILURE LIST
    # ========================================================

    final_failed = list(
        dict.fromkeys(
            existing_failed
            + missing_failed
        )
    )

    # ========================================================
    # COUNT ACTUAL LOCAL FILES
    # ========================================================

    local_count = 0

    for ticker in tickers:

        if get_local_file(
            ticker
        ).exists():

            local_count += 1

    # ========================================================
    # SAVE FAILURE REPORT
    # ========================================================

    failed_file = (
        DAILY_DATA_FOLDER /
        "_failed_tickers.txt"
    )

    with open(
        failed_file,
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in final_failed:

            file.write(
                f"{ticker}\n"
            )

    # ========================================================
    # FN VERIFICATION
    # ========================================================

    fn_file = get_local_file(
        "FN"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("DATABASE UPDATE COMPLETE")
    print("=" * 70)

    print(
        f"Universe stocks:          "
        f"{len(tickers)}"
    )

    print(
        f"Local files now:          "
        f"{local_count}"
    )

    print(
        f"Updated/created this run: "
        f"{len(successful)}"
    )

    print(
        f"Final failed this run:    "
        f"{len(final_failed)}"
    )

    print(
        f"FN local file exists:     "
        f"{fn_file.exists()}"
    )

    if fn_file.exists():

        try:

            fn_data = read_local_data(
                "FN"
            )

            if fn_data is not None:

                print(
                    f"FN rows:                  "
                    f"{len(fn_data)}"
                )

                print(
                    f"FN latest date:           "
                    f"{fn_data.index.max().date()}"
                )

        except Exception as exc:

            print(
                f"FN verification error: "
                f"{exc}"
            )

    print()
    print(
        f"Failed ticker report:"
    )

    print(
        failed_file
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    update_database()