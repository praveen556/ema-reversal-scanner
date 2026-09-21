# =========================================================
# SCANNER MODE
# =========================================================

# Use:
# "TEST"  = validation stocks
# "SP500" = current S&P 500 stocks
# "US_MULTI_INDEX" = All the main indexes

SCAN_MODE = "US_ALL_STOCKS"

INCLUDE_SP500 = True
INCLUDE_NASDAQ100 = True
INCLUDE_DOW30 = True
INCLUDE_RUSSELL2000 = True


# =========================================================
# TEST STOCKS
# =========================================================

TEST_TICKERS = [
    "GOOG",
    "AMZN",
    "NVDA",
    "AMD",
    "MSFT",
    "AXON",
    "FN",
    "RDDT",
]

# ============================================================
# LOCAL MARKET DATA SETTINGS
# ============================================================

DATA_FOLDER = "../data/daily"

# Initial history needed for EMA200 and scanner research
INITIAL_DAILY_PERIOD = "2y"

# Yahoo download settings
DAILY_BATCH_SIZE = 250
DAILY_RETRY_BATCH_SIZE = 20

# Pauses help reduce Yahoo rate limiting
DAILY_BATCH_PAUSE_SECONDS = 2
DAILY_RETRY_PAUSE_SECONDS = 10

# Number of retry rounds for symbols Yahoo temporarily misses
DAILY_MAX_RETRY_ROUNDS = 1

# Future incremental updates will request only recent data
UPDATE_LOOKBACK_DAYS = 10


# =========================================================
# BASIC FILTERS
# =========================================================

MIN_PRICE = 10.0
MIN_AVG_VOLUME = 1_000_000


# =========================================================
# DAILY EMA SETTINGS
# =========================================================

EMA_20 = 20
EMA_50 = 50
EMA_FAST = 89
EMA_SLOW = 200


# =========================================================
# REVERSAL SETTINGS
# =========================================================

TOUCH_TOLERANCE_PCT = 1.5
LOOKBACK_DAYS = 5
MIN_20D_RANGE_PCT = 5.0


# =========================================================
# CHOPPINESS DIAGNOSTIC
# =========================================================

CHOPPY_MAX_COMPRESSION_PCT = 3.0
CHOPPY_MIN_CROSS_COUNT = 8


# =========================================================
# RSI SETTINGS
# =========================================================

RSI_LENGTH = 14


# =========================================================
# 4H MOMENTUM SETTINGS
# =========================================================

EMA_4H_FAST = 5
EMA_4H_SLOW = 13


# =========================================================
# ENTRY STAGE SETTINGS
# =========================================================

NEAR_ENTRY_MAX_PCT = 3.0
EXTENDED_MIN_PCT = 8.0


# =========================================================
# OUTPUT / PROGRESS
# =========================================================

PROGRESS_EVERY = 25

# =========================================================
# V2 EVIDENCE-BASED CLASSIFICATION
# =========================================================
# V2 does NOT require:
# - 4H Bullish
# - Daily Structure OK
# - bullish Daily candle
# - elevated Daily volume ratio
#
# Those remain diagnostics in the scanner output.

# Strongest robust high-confidence profile found in the unique Day-0
# event analysis.  This is intentionally selective.
V2_HIGH_CONF_RSI_MAX = 50.0
V2_HIGH_CONF_RANGE_MAX_PCT = 5.0
V2_HIGH_CONF_COMPRESSION_MAX_PCT = 5.0

# Guardrail against already-overextended reversal candidates.
V2_SETUP_RSI_MAX = 65.0

# Preserve the existing choppiness diagnostic as a setup-quality guardrail.
V2_EXCLUDE_CHOPPY_FROM_HIGH_CONF = True

# =========================================================
# V2.1 PRIORITY SIGNAL
# Historical research candidate:
# Existing HIGH CONFIDENCE + EMA200-only + RSI 40-45
# =========================================================

V21_PRIORITY_RSI_MIN = 40.0
V21_PRIORITY_RSI_MAX = 45.0