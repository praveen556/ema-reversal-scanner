
from moomoo import (
    OpenQuoteContext,
    OpenSecTradeContext,
    TrdMarket,
    RET_OK,
)

HOST = "127.0.0.1"
PORT = 11111

quote_ctx = None
trade_ctx = None

try:
    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)

    ret, data = quote_ctx.get_global_state()
    if ret == RET_OK:
        print("QUOTE CONNECTION: SUCCESS")
        print(data)
    else:
        print("QUOTE CONNECTION FAILED:", data)

    trade_ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.US,
        host=HOST,
        port=PORT,
    )

    ret, accounts = trade_ctx.get_acc_list()

    if ret == RET_OK:
        print("\nTRADE CONNECTION: SUCCESS")
        print(accounts.to_string(index=False))
    else:
        print("TRADE CONNECTION FAILED:", accounts)

finally:
    if quote_ctx is not None:
        quote_ctx.close()
    if trade_ctx is not None:
        trade_ctx.close()
