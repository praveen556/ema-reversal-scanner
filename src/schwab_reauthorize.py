"""Explicit one-time Schwab OAuth renewal. Never prints credentials or token values."""
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import freeze_support
from schwab import auth

ROOT = Path(__file__).resolve().parent
TOKEN = ROOT / '.schwab_auth' / 'token.json'
CALLBACK = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1:8182')


def main():
    key, secret = os.getenv('SCHWAB_APP_KEY'), os.getenv('SCHWAB_APP_SECRET')
    if not key or not secret:
        raise RuntimeError('Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET environment variables.')
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN.exists():
        backup = TOKEN.with_name('token.backup.' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        TOKEN.replace(backup)
        print('Existing token backed up locally. Do not share the backup file.')
    print('Complete the Schwab browser OAuth authorization when prompted.')
    client = auth.easy_client(api_key=key, app_secret=secret,
                              callback_url=CALLBACK, token_path=str(TOKEN))
    response = client.get_price_history_every_day('SPY')
    response.raise_for_status()
    print('Schwab authorization successful; SPY daily API check passed.')
    print('Token stored at:', TOKEN)


if __name__ == '__main__':
    freeze_support()
    main()
