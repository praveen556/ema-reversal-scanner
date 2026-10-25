# EMA project: intraday volatility module

Copy `src/intraday_volatility/` into your EXISTING EMA project's `src/` folder. Do not replace any existing files.

## Layout

```text
EMA_Reversal/
  data/                 # existing daily ticker OHLCV files
  data/5min/            # NEW: Schwab 5-minute files
  output/intraday_volatility/  # NEW reports
  src/intraday_volatility/
    rank_volatility.py
    download_schwab_5min.py
    backtest_reversals.py
```

Install: `pip install pandas numpy openpyxl requests`

Run from your EMA project root (Windows PowerShell):

```powershell
python src/intraday_volatility/rank_volatility.py --data-dir data --out-dir output/intraday_volatility
```

If your daily files are in `data/daily`, set `--data-dir data/daily`. The ranker recursively scans CSVs but ignores `5min`, `hourly`, `1hour`, `4hour`, and `output` directories. It requires OHLCV columns and a date column; see skipped-files report for files not included. A `--asof YYYY-MM-DD` argument ranks using only sessions before the chosen date.

Optional download: first obtain an OAuth access token via your own existing Schwab login flow, then set an environment variable locally (never commit it):

```powershell
$env:SCHWAB_ACCESS_TOKEN="<your locally obtained token>"
python src/intraday_volatility/download_schwab_5min.py --ranking output/intraday_volatility/Intraday_Volatility_Top20_Combined.csv --out-dir data/5min --start 2026-09-01 --end 2026-10-24
```

**Schwab historical data availability and app entitlements vary.** This downloader uses an existing access token, not automatic OAuth refresh; if the token expires, refresh it through your own existing OAuth implementation. It requests a single date window; shorten the date range if your app rejects a long interval.

Diagnostic baseline backtest:

```powershell
python src/intraday_volatility/backtest_reversals.py --daily-dir data --five-min-dir data/5min --out-dir output/intraday_volatility
```

## Methodology and important limitations

- 20D average range = mean of `(High-Low)/Open * 100` over the 20 prior completed daily bars.
- Historical average range = same metric over all locally available daily bars.
- Rank descending separately. Score each rank on a 0–100 scale: `100*(N-rank)/(N-1)`; combine 50% recent and 50% historical. Rank 1 is most volatile.
- `Entry Threshold 20D %` = recent average + 1 **percentage point**. Full-history threshold is also retained.
- Liquidity filter defaults: latest close >= $10 and last-20-day average `Close*Volume` >= $50m.
- The rankings use the latest rows in each file; make sure your files are updated to the same completed market date before running. Data may be unadjusted or split-adjusted depending on the source. Verify split handling before backtesting.
- The backtest is an **initial diagnostic**, not a final validated strategy. It tests threshold-only entry on the next 5-minute bar open with target 1.5%, stop 1%, EOD exit, and 10 bps estimated round-trip costs. For bearish signals, it models underlying short-price returns as a **put-direction proxy**, not actual put-option prices. No options P&L is calculated.
- A fully unbiased portfolio backtest must **recalculate the top 20 for each historical session** and must download 5-minute history for the historical selected symbols, not only today's top 20. The supplied baseline does not do that yet. Do not treat its win rate as an unbiased top-20 strategy result.
- No reversal-confirmation version yet; that requires a separately specified and tested signal rule.
- On 5-minute bars, intrabar ordering is unknown. If stop and target are both touched, stop is assumed first. A stop triggered by an opening gap is filled at the bar's open.
- No live orders or brokerage trading operations are performed.
