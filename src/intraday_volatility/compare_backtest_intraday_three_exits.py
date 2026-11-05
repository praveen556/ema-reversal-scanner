"""
Compare THREE exit rules on IDENTICAL historical long entries, side-by-side.
Run directly in PyCharm. Requires pandas, openpyxl.
No broker connection or orders.

Inputs:
  output/intraday_volatility/INTRA_V2_ONCE_DAILY/INTRA_V2_Closed_Trades.csv
  data/5min/TICKER.csv
Optional:
  output/intraday_volatility/earnings_calendar.csv

Outputs in output/intraday_volatility/THREE_EXIT_COMPARISON/:
  Three_Exit_Strategies.xlsx, Three_Exit_Trades.csv, Three_Exit_Summary.csv,
  Three_Exit_Missing_Data.csv

NOTE: Uses historical entry timestamps from an existing backtest. It does NOT
reconstruct live moomoo fills, simultaneous-position capacity, or order slippage.
5-minute OHLC cannot establish intrabar sequence. Conservative approach:
previous stop is tested before a candle's high updates the trailing stop.
A stop raised by the current candle's high becomes effective NEXT candle.
"""
from pathlib import Path
from datetime import timedelta
import pandas as pd
import math

ROOT = Path(__file__).resolve().parents[2]  # src/intraday_papertrade/ -> root
OUT = ROOT / "output" / "intraday_volatility"
SIGNALS = OUT / "INTRA_V2_ONCE_DAILY" / "INTRA_V2_Closed_Trades.csv"
if not SIGNALS.exists():
    SIGNALS = OUT / "INTRA_V2_Closed_Trades.csv"
FIVE_MIN = ROOT / "data" / "5min"
EARNINGS = OUT / "earnings_calendar.csv"
RESULTS = OUT / "THREE_EXIT_COMPARISON"

# Limit to recent historical sessions for a quick first run. Set to None for all.
LAST_N_TRADING_DAYS = 30
REQUIRE_PREVIOUS_CLOSE_METHOD = True
REQUIRE_EARNINGS_CALENDAR = False  # True = stop if missing
SKIP_EARNINGS_WINDOW = True        # four days before/after, including report day
INITIAL_STOP_PCT = 1.0
FIXED_TARGET_PCT = 1.5
TRAIL_DISTANCE_PCT = 1.0
TRAIL_ACTIVATIONS = {"B": 1.0, "C": 1.5}
ROUND_TRIP_COST_BPS = 10.0
USD_PER_TRADE = 1000.0
EOD_EXIT_BAR = "15:55"  # exit at close of the 15:55-16:00 bar
MAX_EXCEL_ROWS = 50000

def read_earnings():
    if not SKIP_EARNINGS_WINDOW:
        return {}, "disabled"
    if not EARNINGS.exists():
        if REQUIRE_EARNINGS_CALENDAR:
            raise FileNotFoundError(f"Earnings calendar missing: {EARNINGS}")
        return {}, "missing (no earnings exclusions applied)"
    df = pd.read_csv(EARNINGS)
    if not {"ticker", "earnings_date"}.issubset(df.columns):
        raise ValueError("Earnings CSV requires ticker,earnings_date columns")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("US.", "", regex=False)
    d = {}
    for ticker, g in df.dropna(subset=["earnings_date"]).groupby("ticker"):
        d[ticker] = set(g["earnings_date"])
    return d, f"loaded {sum(map(len, d.values()))} dates"

def read_5min(ticker):
    path = FIVE_MIN / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {"datetime","open","high","low","close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path.name}: missing {needed-set(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime","open","high","low","close"])
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    df = df[(df.high >= df[["open","close","low"]].max(axis=1)) & (df.low <= df[["open","close"]].min(axis=1))]
    df = df.sort_values("datetime").drop_duplicates("datetime",keep="last")
    return df

def simulate(bars, entry, activation=None):
    """Same entry for A/B/C; no lookahead: new high updates stop only for next bar."""
    hard = entry * (1 - INITIAL_STOP_PCT/100)
    stop = hard
    peak = entry
    activated = False
    exit_price = None
    exit_reason = "EOD"
    exit_time = None
    mfe = 0.0
    mae = 0.0
    for i, b in enumerate(bars.itertuples(index=False)):
        op, hi, lo, cl = float(b.open),float(b.high),float(b.low),float(b.close)
        when = b.datetime
        # First bar opens at entry. Subsequent bars can gap through stop.
        if i > 0 and op <= stop:
            exit_price, exit_reason, exit_time = op, ("GAP_TRAIL" if activated else "GAP_HARD_STOP"), when
            break
        if activation is None:
            target = entry*(1+FIXED_TARGET_PCT/100)
            if i > 0 and op >= target:
                exit_price, exit_reason, exit_time = op, "GAP_TARGET", when
                break
        if lo <= stop:
            exit_price, exit_reason, exit_time = stop, ("TRAIL_STOP" if activated else "HARD_STOP"), when
            break
        if activation is None:
            if hi >= target:
                exit_price, exit_reason, exit_time = target, "FIXED_TARGET", when
                break
        # Conservative: after existing stop checks, process current high.
        # Current bar's new trail cannot retroactively stop at its low.
        peak = max(peak, hi)
        mfe = max(mfe, (hi/entry-1)*100)
        mae = min(mae, (lo/entry-1)*100)
        if activation is not None and peak >= entry*(1+activation/100):
            activated = True
            stop = max(stop, entry, peak*(1-TRAIL_DISTANCE_PCT/100))
        if when.strftime("%H:%M") >= EOD_EXIT_BAR:
            exit_price, exit_reason, exit_time = cl, "EOD", when
            break
    if exit_price is None:
        return None
    gross = (exit_price/entry-1)*100
    net = gross-ROUND_TRIP_COST_BPS/100
    qty = math.floor(USD_PER_TRADE/entry)
    return dict(exit_price=round(exit_price,5),exit_time=str(exit_time),
                reason=exit_reason, gross_pct=round(gross,4),
                net_pct=round(net,4), pnl_usd=round(qty*entry*net/100,2),
                peak_pct=round(mfe,4), worst_pct=round(mae,4),
                activated=activated, qty=qty)

def build_workbook(trades, summary, by_ticker, missing, note):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    sheets = [
        ("Summary", summary), ("Trade Comparison", trades),
        ("By Ticker", by_ticker), ("Missing Data", missing),
        ("Methodology", pd.DataFrame({"Item": [
            "Source entries", "Shared entry prices", "Strategy A", "Strategy B", "Strategy C",
            "Intrabar ambiguity", "Price gaps", "Costs", "Earnings filtering",
            "Position limit", "Market data", "Data status"],
            "Description": [
            "Existing INTRA_V2 closed-trade file, long previous-close entries",
            "Entry is OPEN of the recorded entry 5-minute candle for all strategies",
            "Fixed +1.5% target / -1% hard stop",
            "+1% activation; 1% trailing from high; break-even floor",
            "+1.5% activation; 1% trailing from high; break-even floor",
            "Stop tested before current high; new trail becomes active next candle",
            "Subsequent candle open crossing stop fills at open, not threshold",
            f"{ROUND_TRIP_COST_BPS} bps round-trip cost deducted",
            note, "Not enforced: identical signal cohort across all strategies",
            "Historical 5-minute OHLC; not tick-level; no live execution",
            "Missing/partial sessions excluded; inspect Missing Data sheet"]}))
    ]
    for name, frame in sheets:
        ws = wb.create_sheet(name)
        ws.append(list(frame.columns))
        for row in frame.head(MAX_EXCEL_ROWS).itertuples(index=False, name=None):
            ws.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 26
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="17324D")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(vertical="center")
        for idx, column in enumerate(ws.columns, 1):
            values = [str(cell.value) for cell in list(column)[:250] if cell.value is not None]
            width = min(44, max(14, max((len(v) for v in values), default=12) + 2))
            ws.column_dimensions[get_column_letter(idx)].width = width
        if name == "Methodology":
            ws.column_dimensions["A"].width = 28
            ws.column_dimensions["B"].width = 85
    path = RESULTS / "Three_Exit_Strategies.xlsx"
    wb.save(path)
    return path

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    if not SIGNALS.exists(): raise FileNotFoundError(f"Missing historical signals: {SIGNALS}")
    if not FIVE_MIN.exists(): raise FileNotFoundError(f"Missing 5-minute folder: {FIVE_MIN}")
    earnings, earnings_note=read_earnings()
    source=pd.read_csv(SIGNALS,low_memory=False)
    required={"Date","Ticker","Entry Time","Direction","Method"}
    if not required.issubset(source.columns): raise ValueError(f"Missing columns {required-set(source.columns)}")
    source["Date"]=pd.to_datetime(source["Date"],errors="coerce").dt.date
    source["Ticker"]=source["Ticker"].astype(str).str.upper().str.replace("US.","",regex=False)
    source=source[source.Direction.astype(str).str.lower().eq("long")]
    if REQUIRE_PREVIOUS_CLOSE_METHOD:
        source=source[source.Method.astype(str).str.lower().eq("previous_close")]
    source=source.dropna(subset=["Date","Entry Time"]).sort_values(["Date","Ticker","Entry Time"])
    source=source.drop_duplicates(["Date","Ticker"],keep="first")
    if LAST_N_TRADING_DAYS is not None:
        dates=sorted(source["Date"].unique())
        source=source[source["Date"].isin(dates[-LAST_N_TRADING_DAYS:])]
    rows=[]; missing=[]; excluded=0
    for ticker,g in source.groupby("Ticker"):
        try: candles=read_5min(ticker)
        except Exception as exc:
            candles=None
            missing.append({"Ticker":ticker,"Date":"","Reason":str(exc)})
        if candles is None:
            missing.extend({"Ticker":ticker,"Date":str(d),"Reason":"Missing 5min data"} for d in g["Date"])
            continue
        for _,r in g.iterrows():
            day=r["Date"]
            if any(abs((day-e).days)<=4 for e in earnings.get(ticker,set())):
                excluded+=1
                continue
            entry_at=pd.to_datetime(r["Entry Time"],errors="coerce",utc=True)
            if pd.isna(entry_at):
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Invalid entry time"});continue
            entry_at=entry_at.tz_convert("America/New_York")
            bars=candles[(candles["datetime"]>=entry_at)&(candles["datetime"].dt.date==day)]
            if bars.empty or bars.iloc[0]["datetime"]!=entry_at:
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Missing exact entry candle"});continue
            if bars.iloc[-1]["datetime"].strftime("%H:%M")!="15:55":
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Missing closing 15:55 candle"});continue
            entry=float(bars.iloc[0]["open"])
            outcomes={"A":simulate(bars,entry,None),"B":simulate(bars,entry,TRAIL_ACTIVATIONS["B"]),
                      "C":simulate(bars,entry,TRAIL_ACTIVATIONS["C"])}
            if any(x is None for x in outcomes.values()):
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"No completed exit"});continue
            out={"Date":str(day),"Ticker":ticker,"Entry Time":str(entry_at),"Entry Price":round(entry,5)}
            for key,v in outcomes.items():
                out.update({f"{key} Exit Price":v["exit_price"],f"{key} Exit Time":v["exit_time"],
                            f"{key} Reason":v["reason"],f"{key} Net %":v["net_pct"],
                            f"{key} P&L $":v["pnl_usd"],f"{key} Peak %":v["peak_pct"],
                            f"{key} Trail Activated":v["activated"]})
            out["B minus A %"]=round(out["B Net %"]-out["A Net %"],4)
            out["C minus A %"]=round(out["C Net %"]-out["A Net %"],4)
            out["B minus C %"]=round(out["B Net %"]-out["C Net %"],4)
            rows.append(out)
        print(f"Processed {ticker}: {len(g)} historical entries",flush=True)
    if not rows:
        raise RuntimeError("No matched entries. See source signals and data/5min; check dates and ticker filenames.")
    trades=pd.DataFrame(rows).sort_values(["Date","Entry Time","Ticker"])
    missing_df=pd.DataFrame(missing,columns=["Ticker","Date","Reason"])
    summaries=[]
    by_ticker=[]
    for label in ("A","B","C"):
        vals=trades[f"{label} Net %"]
        summaries.append({"Strategy":label,"Trades":len(vals),"Win Rate %":round((vals>0).mean()*100,2),
                          "Average Net %":round(vals.mean(),4),"Median Net %":round(vals.median(),4),
                          "Total P&L $":round(trades[f"{label} P&L $"].sum(),2),
                          "Average P&L $":round(trades[f"{label} P&L $"].mean(),2),
                          "Stop Exits":int(trades[f"{label} Reason"].str.contains("STOP|TRAIL").sum()),
                          "EOD Exits":int(trades[f"{label} Reason"].eq("EOD").sum())})
        for ticker,g in trades.groupby("Ticker"):
            by_ticker.append({"Ticker":ticker,"Strategy":label,"Trades":len(g),
                              "Win Rate %":round((g[f"{label} Net %"]>0).mean()*100,2),
                              "Avg Net %":round(g[f"{label} Net %"].mean(),4),
                              "Total P&L $":round(g[f"{label} P&L $"].sum(),2)})
    summary=pd.DataFrame(summaries)
    ticker_df=pd.DataFrame(by_ticker)
    trades.to_csv(RESULTS/"Three_Exit_Trades.csv",index=False)
    summary.to_csv(RESULTS/"Three_Exit_Summary.csv",index=False)
    missing_df.to_csv(RESULTS/"Three_Exit_Missing_Data.csv",index=False)
    note=f"{earnings_note}; {excluded} signal entries excluded"
    path=build_workbook(trades,summary,ticker_df,missing_df,note)
    print(f"Completed {len(trades)} matched entries, {excluded} earnings exclusions, {len(missing)} missing.",flush=True)
    print(summary.to_string(index=False))
    print(f"Excel: {path}")

if __name__=="__main__":
    main()
