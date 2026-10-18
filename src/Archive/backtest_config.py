# =========================================================
# SIGNAL-ONLY BACKTEST SETTINGS
# =========================================================

# Daily history includes indicator warm-up + future outcome bars.
DAILY_PERIOD = "3y"

# Yahoo 1H history is limited. 729d stays inside the usual 730-day limit.
HOURLY_PERIOD = "729d"

# Download hourly data in smaller batches to reduce failures/rate pressure.
HOURLY_BATCH_SIZE = 40

# A 4H EMA13 should have a reasonable warm-up before it is trusted.
MIN_4H_BARS = 30

# Forward trading-day horizons to measure after each as-of scan date.
FORWARD_HORIZONS = (1, 2, 3, 5, 10)
MAX_FORWARD_DAYS = max(FORWARD_HORIZONS)

# Measure MFE/MAE over this many future trading sessions.
MFE_MAE_WINDOW = 10

# Keep WATCH signals too so we can test whether QUALIFIED filters add value.
INCLUDE_WATCH = True

# Progress display.
PROGRESS_EVERY = 25
