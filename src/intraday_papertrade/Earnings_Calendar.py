
import csv
import os
from pathlib import Path
from datetime import datetime
import requests

# ------------------------------------
# CONFIGURATION
# ------------------------------------

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

FROM_DATE = "2026-10-23"
TO_DATE = "2026-11-30"

BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "output" / "intraday_volatility"
OUTPUT.mkdir(parents=True, exist_ok=True)

EARNINGS_FILE = OUTPUT / "earnings_calendar.csv"

# ------------------------------------
# FETCH EARNINGS FROM FINNHUB
# ------------------------------------

def update_earnings_calendar():
    if not FINNHUB_API_KEY:
        raise ValueError(
            "FINNHUB_API_KEY is missing. "
            "Set it in your environment variables."
        )

    print("Fetching earnings calendar from Finnhub...")

    response = requests.get(
        "https://finnhub.io/api/v1/calendar/earnings",
        params={
            "from": FROM_DATE,
            "to": TO_DATE,
            "token": FINNHUB_API_KEY
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()
    earnings = data.get("earningsCalendar")

    if not isinstance(earnings, list):
        raise ValueError(
            "Finnhub did not return a valid earnings calendar."
        )

    if not earnings:
        raise ValueError(
            "Finnhub returned no earnings records. "
            "Existing CSV was not overwritten."
        )

    # ------------------------------------
    # SAVE EARNINGS TO CSV
    # ------------------------------------

    temp_file = EARNINGS_FILE.with_suffix(".tmp")

    try:
        with open(
            temp_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=["ticker", "earnings_date"]
            )

            writer.writeheader()

            count = 0

            for item in earnings:
                ticker = item.get("symbol")
                earnings_date = item.get("date")

                if ticker and earnings_date:
                    writer.writerow({
                        "ticker": ticker,
                        "earnings_date": earnings_date
                    })
                    count += 1

        if count == 0:
            raise ValueError("No valid earnings records to save.")

        os.replace(temp_file, EARNINGS_FILE)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    print(f"Successfully saved {count} earnings records.")
    print(f"File location: {EARNINGS_FILE}")


if __name__ == "__main__":
    update_earnings_calendar()
