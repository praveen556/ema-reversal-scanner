
import os
from pathlib import Path
from multiprocessing import freeze_support

from schwab.auth import easy_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKEN_DIR = PROJECT_ROOT / ".schwab_auth"
TOKEN_PATH = TOKEN_DIR / "token.json"

CALLBACK_URL = "https://127.0.0.1:8182"


def main():
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    client_id = os.getenv("SCHWAB_APP_KEY")
    client_secret = os.getenv("SCHWAB_APP_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Schwab credentials. "
            "Check SCHWAB_APP_KEY and SCHWAB_APP_SECRET "
            "in PyCharm environment variables."
        )

    print("Starting Schwab authorization...", flush=True)

    client = easy_client(
        api_key=client_id,
        app_secret=client_secret,
        callback_url=CALLBACK_URL,
        token_path=str(TOKEN_PATH),
    )

    print("Schwab authorization successful!", flush=True)

    print("Downloading AAPL 5-minute data...", flush=True)

    response = client.get_price_history_every_five_minutes(
        "AAPL",
        need_extended_hours_data=False,
    )

    response.raise_for_status()

    candles = response.json().get("candles", [])

    print(f"Downloaded {len(candles):,} candles.", flush=True)

    if candles:
        import pandas as pd

        df = pd.DataFrame(candles)

        df["datetime"] = (
            pd.to_datetime(df["datetime"], unit="ms", utc=True)
            .dt.tz_convert("America/New_York")
        )

        output_dir = PROJECT_ROOT / "data" / "5min"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "AAPL.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved to: {output_path}", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
