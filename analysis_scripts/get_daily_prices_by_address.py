# %%
import json
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

result = json.load(open("coingecko_gambling_token_addr_results.json"))
valid_addresses = []
token_ids = []
for addr, v in result.items():
    if v is not None:
        valid_addresses.append(addr)
        token_ids.append(v["id"])
print(valid_addresses)  # just 14, printing not saving
print(token_ids)
# %%
from datetime import date, timedelta
import os
import time
os.makedirs("./market_raw", exist_ok=True)

for addr in valid_addresses:
    today = date.today()
    coin_id = result[addr]["id"]

    market_history_data_list = []
    while True:
        # time.sleep(2)
        today = today - timedelta(days=1)
        market_data = cg.get_coin_history_by_id(coin_id, today.strftime("%d-%m-%Y"))
        if market_data.get("market_data") is None:
            break
        else:
            market_history_data_list.append(market_data)

    with open(os.path.join("./market_raw", addr + ".json"), "w") as f:
        json.dump(market_history_data_list, f)
