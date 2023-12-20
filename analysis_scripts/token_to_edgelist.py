from concurrent.futures import ThreadPoolExecutor, wait
from decimal import Decimal

from pymongo import MongoClient
import orjson as json # support Decimal
from threading import Lock, Thread
import argparse
import traceback
import os

# Transfer (address indexed from, address indexed to, uint256 value)
TOKEN_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


class Worker:
    def __init__(self, input_file, output_folder):
        self.contracts = input_file.read().lower().splitlines() # lower all addrs

        os.makedirs(output_folder, exist_ok=True)
        self.files = {contract_addr: open( os.path.join(output_folder, contract_addr+".edgelist") , "w") for contract_addr in self.contracts}
        # self.lock = Lock()

        client = MongoClient('mongodb://172.24.1.2:27017/')
        db = client.ethrpc
        coll = db.receiptColl
        self.coll = coll
    
    def start_mt(self):
        # with ThreadPoolExecutor(max_workers=128) as executor:
        #     fus = set()
        #     for contract in self.contracts:
        #         fu = executor.submit(self.get_token_tfs, contract)
        #         fus.add(fu)
        #     wait(fus)
        threads = [Thread(target=self.get_token_tfs, args=(contract,)) for contract in self.contracts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def get_token_tfs(self, token_contract):
        try:
            for receipt in self.coll.find({"logs.address": token_contract, "logs.topics.0": TOKEN_TRANSFER}):
                block_number = int(receipt["blockNumber"], 16)
                gas_used = int(receipt["gasUsed"], 16)
                effective_gas_price = int(receipt["effectiveGasPrice"], 16)
                for log in receipt["logs"]:
                    if len(log["topics"]) > 0 and log["topics"][0] == TOKEN_TRANSFER:
                        if len(log["topics"]) == 4:
                            from_addr = log["topics"][1].replace("0x000000000000000000000000", "0x")
                            to_addr = log["topics"][2].replace("0x000000000000000000000000", "0x")
                            transfer_value = int(log["topics"][3], 16)
                        elif len(log["topics"]) == 3:
                            from_addr = log["topics"][1].replace("0x000000000000000000000000", "0x")
                            to_addr = log["topics"][2].replace("0x000000000000000000000000", "0x")
                            transfer_value = 0 if log["data"] == "0x" else int(log["data"], 16)
                        elif len(log["topics"]) == 2:
                            data = log["data"].removeprefix("0x")
                            print(f"length == 2, data length == {len(data)}")
                            from_addr = log["topics"][1].replace("0x000000000000000000000000", "0x")
                            to_addr = "0x" + data[0: 40]
                            transfer_value = 0 if data[40:] == "" else int(data[40:], 16)
                        else: # if len(log["topics"]) == 1:
                            data = log["data"].removeprefix("0x")
                            print(f"length == 1, data length == {len(data)}") # 192
                            from_addr = "0x" + data[0: 64].removeprefix("000000000000000000000000")
                            to_addr = "0x" + data[64: 128].removeprefix("000000000000000000000000")
                            transfer_value = 0 if data[128:] == "" else int(data[128:], 16)

                        edge_attrs = {
                            "block_number": block_number,
                            "transfer_value": float(Decimal(transfer_value) / Decimal(1000000000000000000)),
                            "gas_used": gas_used,
                            "effective_gas_price": float(Decimal(effective_gas_price) / Decimal(1000000000000000000)),
                        }
                        # with self.lock:
                        self.files[token_contract].write(f"{from_addr} {to_addr} {json.dumps(edge_attrs).decode('utf-8')}\n")
        except Exception as e:
            print(e)
            traceback.print_exc()
            import os
            os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="increase output verbosity")
    parser.add_argument("-o", "--output", default="output", help="increase output verbosity")
    args = parser.parse_args()

    with open(args.input) as i:
        worker = Worker(i, args.output)
        worker.start_mt()
