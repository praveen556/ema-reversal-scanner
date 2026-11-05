"""
Intraday 5-minute ENTRY FILTER research, compared with exits A/B/C.
Place beside compare_backtest_intraday_three_exits_FIXED.py in
src/intraday_volatility/, then run this script. No broker orders.

pip install pandas openpyxl numpy

Important: This is a research backtest. Entry confirmation is calculated
from candles COMPLETED BEFORE the historical entry candle. It does NOT
recreate the original trigger detection or tick-level order fills.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "compare_backtest_intraday_three_exits.py"
if not ORIGINAL.exists():
    ORIGINAL = HERE / "compare_intraday_three_exits.py"
if not ORIGINAL.exists():
    raise FileNotFoundError("Place the existing three-exit backtest .py beside this file.")
spec = importlib.util.spec_from_file_location("original_three_exits", ORIGINAL)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROOT = HERE.parents[1]   # .../ema-reversal-scanner
OUT = ROOT / "output" / "intraday_volatility"
SIGNALS = OUT / "INTRA_V2_ONCE_DAILY" / "INTRA_V2_Closed_Trades.csv"
if not SIGNALS.exists():
    SIGNALS = OUT / "INTRA_V2_Closed_Trades.csv"
FIVE_MIN = ROOT / "data" / "5min"
RESULTS = OUT / "ENTRY_FILTER_COMPARISON"
EARNINGS = OUT / "earnings_calendar.csv"
LAST_N_TRADING_DAYS = None
MAX_SIMULTANEOUS_POSITIONS = 5
CAPACITY_MODE = True
ROUND_TRIP_COST_BPS = 10.0
USD_PER_TRADE = 1000.0
# Train/test split by DATE, not random trades, to reduce lookahead.
TRAIN_FRACTION = 0.70
# Require 5-minute files with full regular session and volume.
MIN_HISTORY_BARS = 2
FILTERS = [
    "BASELINE", "BULLISH_CANDLE", "VWAP_RECLAIM",
    "EMA9_RECLAIM_HIGHER_LOW", "BULLISH_VWAP_VOLUME",
    "BULLISH_HIGHER_LOW", "AFTER_0950_BULLISH",
]
EXITS = {"A": None, "B": 1.0, "C": 1.5}

def load_earnings():
    if not EARNINGS.exists():
        return {}, "MISSING: earnings exclusions not applied"
    df = pd.read_csv(EARNINGS)
    if not {"ticker","earnings_date"}.issubset(df.columns):
        raise ValueError("earnings_calendar.csv requires ticker,earnings_date")
    df["ticker"] = df.ticker.astype(str).str.upper().str.replace("US.","",regex=False)
    df["earnings_date"] = pd.to_datetime(df.earnings_date,errors="coerce").dt.date
    return {k:set(g.earnings_date.dropna()) for k,g in df.groupby("ticker")}, "loaded"

def load_candles(ticker):
    file = FIVE_MIN / f"{ticker}.csv"
    if not file.exists():
        return None
    d = pd.read_csv(file)
    required={"datetime","open","high","low","close","volume"}
    if not required.issubset(d.columns):
        raise ValueError(f"{file.name} lacks {required-set(d.columns)}")
    # Input convention matches original script: UTC timestamp strings.
    d["datetime"] = pd.to_datetime(d.datetime,errors="coerce",utc=True).dt.tz_convert("America/New_York")
    for c in ("open","high","low","close","volume"):
        d[c] = pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=list(required)).sort_values("datetime").drop_duplicates("datetime")
    d=d[(d.open>0)&(d.high>0)&(d.low>0)&(d.close>0)&(d.volume>=0)]
    d=d[(d.high>=d[["open","low","close"]].max(axis=1))&
        (d.low<=d[["open","close"]].min(axis=1))]
    d=d.set_index("datetime",drop=False)
    return d

def features(day):
    """All features on bar i use bars <= i; entry is at bar i+1 OPEN."""
    d=day.copy()
    tp=(d.high+d.low+d.close)/3
    pv=tp*d.volume
    d["vwap"]=pv.cumsum()/d.volume.cumsum().replace(0,np.nan)
    d["ema9"]=d.close.ewm(span=9,adjust=False).mean()
    d["ema20"]=d.close.ewm(span=20,adjust=False).mean()
    d["prev_high"]=d.high.shift(1)
    d["prev_low"]=d.low.shift(1)
    d["prev_close"]=d.close.shift(1)
    d["prev_vwap"]=d.vwap.shift(1)
    # Exclude current bar from volume benchmark, min 5 prior candles.
    d["prior_avg_vol"]=d.volume.shift(1).rolling(10,min_periods=5).mean()
    d["bullish"]=(d.close>d.open)&(d.close>d.prev_close)
    d["higher_low"]=(d.low>d.prev_low)
    d["vwap_reclaim"]=(d.close>d.vwap)&(d.prev_close<=d.prev_vwap)
    d["ema9_reclaim"]=(d.close>d.ema9)&(d.prev_close<=d.ema9.shift(1))
    d["volume_confirm"]=(d.volume>=1.2*d.prior_avg_vol)
    return d

def allowed(name, f, entry_time):
    if name=="BASELINE": return True
    if name=="BULLISH_CANDLE": return bool(f.bullish)
    if name=="VWAP_RECLAIM": return bool(f.vwap_reclaim)
    if name=="EMA9_RECLAIM_HIGHER_LOW": return bool(f.ema9_reclaim and f.higher_low)
    if name=="BULLISH_VWAP_VOLUME":
        return bool(f.bullish and f.close>f.vwap and f.volume_confirm)
    if name=="BULLISH_HIGHER_LOW": return bool(f.bullish and f.higher_low)
    if name=="AFTER_0950_BULLISH":
        return bool(entry_time.strftime("%H:%M")>"09:50" and f.bullish)
    return False

def run_exit(bars, entry, activation):
    # Reuse the exact previous simulator, so A/B/C rules remain consistent.
    return base.simulate(bars,entry,activation)

def metrics(d, name, exit_code, split):
    v=d["Net %"]
    wins=v[v>0]; losses=v[v<0]
    pf=(wins.sum()/(-losses.sum())) if len(losses) and losses.sum()<0 else float("nan")
    return {"Filter":name,"Exit":exit_code,"Sample":split,
            "Trades":len(d),"Win Rate %":round(100*(v>0).mean(),2) if len(v) else np.nan,
            "Avg Net %":round(v.mean(),4) if len(v) else np.nan,
            "Median Net %":round(v.median(),4) if len(v) else np.nan,
            "Profit Factor":round(pf,3) if np.isfinite(pf) else np.nan,
            "Total P&L $":round(d["P&L $"].sum(),2),
            "Trading Days":d.Date.nunique() if len(d) else 0}

def enforce_capacity(d):
    if not CAPACITY_MODE or d.empty:
        return d
    selected=[]
    active=[]  # exit timestamps
    for _,r in d.sort_values(["Entry Time","Ticker"]).iterrows():
        at=pd.Timestamp(r["Entry Time"])
        active=[x for x in active if x>at]
        if len(active)>=MAX_SIMULTANEOUS_POSITIONS:
            continue
        selected.append(r)
        active.append(pd.Timestamp(r["Exit Time"]))
    return pd.DataFrame(selected,columns=d.columns)

def excel_export(tables,path):
    wb=Workbook();wb.remove(wb.active)
    for name,df in tables.items():
        ws=wb.create_sheet(name)
        ws.append(list(df.columns))
        for tup in df.itertuples(index=False,name=None):
            ws.append([None if pd.isna(x) else (x.item() if hasattr(x,"item") else x) for x in tup])
        ws.freeze_panes="A2"
        ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:
            cell.fill=PatternFill("solid",fgColor="17324D")
            cell.font=Font(color="FFFFFF",bold=True)
        for col in ws.columns:
            letter=get_column_letter(col[0].column)
            width=max((len(str(c.value)) for c in list(col)[:120] if c.value is not None),default=10)
            ws.column_dimensions[letter].width=min(40,max(14,width+2))
    wb.save(path)

def main():
    if not SIGNALS.exists(): raise FileNotFoundError(f"Missing signals: {SIGNALS}")
    if not FIVE_MIN.exists(): raise FileNotFoundError(f"Missing 5-minute directory: {FIVE_MIN}")
    RESULTS.mkdir(parents=True,exist_ok=True)
    src=pd.read_csv(SIGNALS,low_memory=False)
    required={"Date","Ticker","Entry Time","Direction","Method"}
    if not required.issubset(src.columns):
        raise ValueError(f"Missing signal columns: {required-set(src.columns)}")
    src["Date"]=pd.to_datetime(src.Date,errors="coerce").dt.date
    src["Ticker"]=src.Ticker.astype(str).str.upper().str.replace("US.","",regex=False)
    src=src[src.Direction.astype(str).str.lower().eq("long")]
    src=src[src.Method.astype(str).str.lower().eq("previous_close")]
    src=src.dropna(subset=["Date","Entry Time"]).sort_values(["Date","Ticker","Entry Time"])
    src=src.drop_duplicates(["Date","Ticker"])
    if LAST_N_TRADING_DAYS:
        dates=sorted(src.Date.unique())[-LAST_N_TRADING_DAYS:]
        src=src[src.Date.isin(dates)]
    earnings, earnings_status=load_earnings()
    rows=[]; missing=[]; excluded=0
    for ticker,g in src.groupby("Ticker"):
        try: allbars=load_candles(ticker)
        except Exception as e:
            allbars=None
            missing.append({"Ticker":ticker,"Date":"","Reason":str(e)})
        if allbars is None:
            missing.extend({"Ticker":ticker,"Date":str(x),"Reason":"No valid 5-minute volume CSV"} for x in g.Date)
            continue
        for _,r in g.iterrows():
            day=r.Date
            if any(abs((day-e).days)<=4 for e in earnings.get(ticker,set())):
                excluded+=1;continue
            at=pd.to_datetime(r["Entry Time"],utc=True,errors="coerce")
            if pd.isna(at):
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Invalid entry timestamp"});continue
            at=at.tz_convert("America/New_York")
            bars=allbars[allbars.datetime.dt.date==day].copy()
            bars=bars[(bars.datetime.dt.strftime("%H:%M")>="09:30")&
                      (bars.datetime.dt.strftime("%H:%M")<="15:59")]
            if bars.empty or at not in bars.index:
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Entry candle absent"});continue
            if bars.iloc[-1].datetime.strftime("%H:%M")!="15:55":
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"Missing 15:55 candle"});continue
            loc=bars.index.get_loc(at)
            if not isinstance(loc,(int,np.integer)) or loc<1:
                missing.append({"Ticker":ticker,"Date":str(day),"Reason":"No completed candle before entry"});continue
            f=features(bars)
            prev=f.iloc[loc-1]
            # All filters inspect ONLY previous completed bar.
            following=bars.iloc[loc:].reset_index(drop=True)
            entry=float(following.iloc[0].open)
            for name in FILTERS:
                if name!="BASELINE" and (loc<MIN_HISTORY_BARS or not allowed(name,prev,at)):
                    continue
                for label,activation in EXITS.items():
                    result=run_exit(following,entry,activation)
                    if result is None:continue
                    rows.append({"Date":str(day),"Ticker":ticker,"Filter":name,"Exit":label,
                                 "Entry Time":str(at),"Entry Price":entry,
                                 "Exit Time":result["exit_time"],"Exit Price":result["exit_price"],
                                 "Exit Reason":result["reason"],"Net %":result["net_pct"],
                                 "P&L $":result["pnl_usd"],"Trailing Activated":result["activated"],
                                 "Previous Candle Close":round(float(prev.close),5),
                                 "VWAP":round(float(prev.vwap),5) if pd.notna(prev.vwap) else np.nan,
                                 "EMA9":round(float(prev.ema9),5),
                                 "Relative Volume":round(float(prev.volume/prev.prior_avg_vol),3)
                                    if pd.notna(prev.prior_avg_vol) and prev.prior_avg_vol>0 else np.nan})
        print(f"Processed {ticker}: {len(g)} signals",flush=True)
    if not rows:
        raise RuntimeError("No evaluated entries. Check timestamps and data/5min files.")
    alltrades=pd.DataFrame(rows)
    dates=sorted(alltrades.Date.unique())
    cut=max(1,min(len(dates)-1,int(len(dates)*TRAIN_FRACTION))) if len(dates)>1 else len(dates)
    test_dates=set(dates[cut:])
    alltrades["Sample"]=np.where(alltrades.Date.isin(test_dates),"TEST","TRAIN")
    selected=[]
    for (name,exitcode),part in alltrades.groupby(["Filter","Exit"]):
        selected.append(enforce_capacity(part))
    trades=pd.concat(selected,ignore_index=True)
    summary=[]
    for (name,exitcode),part in trades.groupby(["Filter","Exit"]):
        for split in ("TRAIN","TEST","ALL"):
            subset=part if split=="ALL" else part[part.Sample==split]
            summary.append(metrics(subset,name,exitcode,split))
    summary=pd.DataFrame(summary)
    # Show test results first and preserve baseline comparison.
    summary["Order"]=summary.Sample.map({"TEST":0,"TRAIN":1,"ALL":2})
    summary=summary.sort_values(["Order","Win Rate %"],ascending=[True,False]).drop(columns="Order")
    missing_df=pd.DataFrame(missing,columns=["Ticker","Date","Reason"])
    notes=pd.DataFrame({"Setting":["Signal file","5-minute directory","Entries","Confirmation timing",
                                  "Position capacity","Costs","Training/test","Earnings",
                                  "Intrabar assumption","Interpretation"],
                        "Value":[str(SIGNALS),str(FIVE_MIN),
                                 "Original previous_close signal, at original entry candle OPEN",
                                 "Filters inspect only the preceding completed candle",
                                 f"{MAX_SIMULTANEOUS_POSITIONS} concurrent trades per filter+exit; chronological ticker tie-break" if CAPACITY_MODE else "Unlimited",
                                 f"{ROUND_TRIP_COST_BPS} bps round-trip (from original simulator)",
                                 f"First {TRAIN_FRACTION:.0%} trading dates TRAIN; remainder TEST",
                                 f"{earnings_status}; {excluded} excluded signals",
                                 "Original simulator: current bar new trailing stop active next bar",
                                 "Selection bias possible; historical signals, no tick fills; no promise of 60%"]})
    trades=trades.sort_values(["Date","Entry Time","Filter","Exit","Ticker"])
    summary.to_csv(RESULTS/"Entry_Filter_Summary.csv",index=False)
    trades.to_csv(RESULTS/"Entry_Filter_Trades.csv",index=False)
    missing_df.to_csv(RESULTS/"Entry_Filter_Missing.csv",index=False)
    excel_export({"Summary":summary,"Trades":trades,"Missing Data":missing_df,"Methodology":notes},
                 RESULTS/"Entry_Filter_Comparison.xlsx")
    print(f"\nDone: {len(trades)} selected strategy trades; {len(test_dates)} test dates")
    print(summary[summary.Sample=="TEST"].to_string(index=False))
    print("Workbook:",RESULTS/"Entry_Filter_Comparison.xlsx")

if __name__=="__main__":
    main()
