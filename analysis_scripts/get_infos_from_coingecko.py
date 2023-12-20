# %%
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

# %%
# platforms = cg.get_asset_platforms()
# print(platforms)
# [{'id': 'factom', 'chain_identifier': None, 'name': 'Factom', 'shortname': ''}, {'id': 'openledger', 'chain_identifier': None, 'name': 'OpenLedger', 'shortname': ''}, {'id': 'cosmos', 'chain_identifier': None, 'name': 'Cosmos', 'shortname': ''}, {'id': 'tezos', 'chain_identifier': None, 'name': 'Tezos', 'shortname': ''}, {'id': 'metaverse-etp', 'chain_identifier': None, 'name': 'Metaverse ETP', 'shortname': ''}, {'id': 'nem', 'chain_identifier': None, 'name': 'NEM', 'shortname': ''}, {'id': 'findora', 'chain_identifier': None, 'name': 'Findora', 'shortname': ''}, {'id': 'icon', 'chain_identifier': None, 'name': 'ICON', 'shortname': ''}, {'id': 'waves', 'chain_identifier': None, 'name': 'Waves', 'shortname': ''}, {'id': 'kujira', 'chain_identifier': None, 'name': 'Kujira', 'shortname': ''}, {'id': 'stratis', 'chain_identifier': None, 'name': 'Stratis', 'shortname': ''}, {'id': 'theta', 'chain_identifier': 361, 'name': 'Theta', 'shortname': ''}, {'id': 'nuls', 'chain_identifier': None, 'name': 'Nuls', 'shortname': ''}, {'id': 'qtum', 'chain_identifier': None, 'name': 'Qtum', 'shortname': ''}, {'id': 'stellar', 'chain_identifier': None, 'name': 'Stellar', 'shortname': ''}, {'id': 'nxt', 'chain_identifier': None, 'name': 'NXT', 'shortname': ''}, {'id': 'ardor', 'chain_identifier': None, 'name': 'Ardor', 'shortname': ''}, {'id': 'ontology', 'chain_identifier': None, 'name': 'Ontology', 'shortname': ''}, {'id': 'eos', 'chain_identifier': None, 'name': 'EOS', 'shortname': ''}, {'id': 'vechain', 'chain_identifier': None, 'name': 'VeChain', 'shortname': ''}, {'id': 'omni', 'chain_identifier': None, 'name': 'Omni', 'shortname': ''}, {'id': 'counterparty', 'chain_identifier': None, 'name': 'Counterparty', 'shortname': ''}, {'id': 'chiliz', 'chain_identifier': None, 'name': 'Chiliz', 'shortname': ''}, {'id': 'bitshares', 'chain_identifier': None, 'name': 'BitShares', 'shortname': ''}, {'id': 'neo', 'chain_identifier': None, 'name': 'NEO', 'shortname': ''}, {'id': 'super-zero', 'chain_identifier': None, 'name': 'Sero', 'shortname': ''}, {'id': 'tron', 'chain_identifier': None, 'name': 'TRON', 'shortname': ''}, {'id': '', 'chain_identifier': None, 'name': 'Radix', 'shortname': ''}, {'id': 'komodo', 'chain_identifier': None, 'name': 'Komodo', 'shortname': ''}, {'id': 'libre', 'chain_identifier': None, 'name': 'Libre', 'shortname': ''}, {'id': 'achain', 'chain_identifier': None, 'name': 'Achain', 'shortname': ''}, {'id': 'vite', 'chain_identifier': None, 'name': 'Vite', 'shortname': ''}, {'id': 'gochain', 'chain_identifier': None, 'name': 'GoChain', 'shortname': ''}, {'id': 'bittorrent', 'chain_identifier': 199, 'name': 'BitTorrent', 'shortname': ''}, {'id': 'enq-enecuum', 'chain_identifier': None, 'name': 'Enecuum', 'shortname': ''}, {'id': 'mdex', 'chain_identifier': None, 'name': 'Mdex', 'shortname': ''}, {'id': 'iotex', 'chain_identifier': None, 'name': 'IoTeX', 'shortname': 'iotex'}, {'id': 'bitkub-chain', 'chain_identifier': 96, 'name': 'Bitkub Chain', 'shortname': ''}, {'id': 'kusama', 'chain_identifier': None, 'name': 'Kusama', 'shortname': ''}, {'id': 'binancecoin', 'chain_identifier': None, 'name': 'BNB Beacon Chain', 'shortname': 'BEP2'}, {'id': 'bitcoin-cash', 'chain_identifier': None, 'name': 'Simple Ledger Protocol (Bitcoin Cash)', 'shortname': 'SLP'}, {'id': 'wanchain', 'chain_identifier': None, 'name': 'Wanchain', 'shortname': ''}, {'id': 'huobi-token', 'chain_identifier': 128, 'name': 'Huobi ECO Chain Mainnet', 'shortname': 'HECO'}, {'id': 'zilliqa', 'chain_identifier': None, 'name': 'Zilliqa', 'shortname': ''}, {'id': 'klay-token', 'chain_identifier': None, 'name': 'Klaytn', 'shortname': ''}, {'id': 'ethereum-classic', 'chain_identifier': None, 'name': 'Ethereum Classic', 'shortname': ''}, {'id': 'polis-chain', 'chain_identifier': 333999, 'name': 'Polis Chain', 'shortname': ''}, {'id': 'defichain', 'chain_identifier': None, 'name': 'DeFiChain', 'shortname': ''}, {'id': 'fusion-network', 'chain_identifier': None, 'name': 'Fusion Network', 'shortname': 'fusion-network'}, {'id': 'celer-network', 'chain_identifier': None, 'name': 'Celer Network', 'shortname': 'Celer'}, {'id': 'proof-of-memes', 'chain_identifier': None, 'name': 'Proof of Memes', 'shortname': ''}, {'id': 'telos', 'chain_identifier': None, 'name': 'Telos', 'shortname': ''}, {'id': 'hoo', 'chain_identifier': None, 'name': 'Hoo', 'shortname': 'Hoo'}, {'id': 'Bitcichain', 'chain_identifier': None, 'name': 'Bitcichain', 'shortname': 'Bitcichain'}, {'id': 'algorand', 'chain_identifier': None, 'name': 'Algorand', 'shortname': ''}, {'id': 'yocoin', 'chain_identifier': None, 'name': 'Yocoin', 'shortname': 'yocoin'}, {'id': 'near-protocol', 'chain_identifier': None, 'name': 'Near Protocol', 'shortname': 'near-protocol'}, {'id': 'mixin-network', 'chain_identifier': None, 'name': 'Mixin Network', 'shortname': ''}, {'id': 'xrp', 'chain_identifier': None, 'name': 'XRP Ledger', 'shortname': 'xrp'}, {'id': 'polkadot', 'chain_identifier': None, 'name': 'Polkadot', 'shortname': ''}, {'id': 'cardano', 'chain_identifier': None, 'name': 'Cardano', 'shortname': ''}, {'id': 'secret', 'chain_identifier': None, 'name': 'Secret', 'shortname': ''}, {'id': 'xdai', 'chain_identifier': 100, 'name': 'Gnosis Chain', 'shortname': ''}, {'id': 'rollux', 'chain_identifier': None, 'name': 'Rollux', 'shortname': ''}, {'id': 'ronin', 'chain_identifier': None, 'name': 'Ronin', 'shortname': 'ron'}, {'id': 'neon-evm', 'chain_identifier': None, 'name': 'Neon EVM', 'shortname': ''}, {'id': 'terra', 'chain_identifier': None, 'name': 'Terra Classic', 'shortname': ''}, {'id': 'fantom', 'chain_identifier': 250, 'name': 'Fantom', 'shortname': ''}, {'id': 'exosama', 'chain_identifier': None, 'name': 'Exosama', 'shortname': ''}, {'id': 'osmosis', 'chain_identifier': None, 'name': 'Osmosis', 'shortname': 'Osmo'}, {'id': 'optimistic-ethereum', 'chain_identifier': 10, 'name': 'Optimism', 'shortname': 'Optimism'}, {'id': 'sora', 'chain_identifier': None, 'name': 'Sora', 'shortname': ''}, {'id': 'polygon-pos', 'chain_identifier': 137, 'name': 'Polygon POS', 'shortname': 'MATIC'}, {'id': 'bitgert', 'chain_identifier': None, 'name': 'Bitgert Chain', 'shortname': 'Bitgert Brise'}, {'id': 'thorchain', 'chain_identifier': None, 'name': 'Thorchain', 'shortname': ''}, {'id': 'elrond', 'chain_identifier': None, 'name': 'Elrond', 'shortname': 'elrond'}, {'id': 'wemix-network', 'chain_identifier': None, 'name': 'Wemix Network', 'shortname': ''}, {'id': 'moonriver', 'chain_identifier': 1285, 'name': 'Moonriver', 'shortname': 'moonriver'}, {'id': 'cronos', 'chain_identifier': 25, 'name': 'Cronos', 'shortname': 'CRO'}, {'id': 'smartbch', 'chain_identifier': 10000, 'name': 'SmartBCH', 'shortname': ''}, {'id': 'aurora', 'chain_identifier': 1313161554, 'name': 'Aurora', 'shortname': 'aurora'}, {'id': 'tomochain', 'chain_identifier': 88, 'name': 'TomoChain', 'shortname': ''}, {'id': 'avalanche', 'chain_identifier': 43114, 'name': 'Avalanche', 'shortname': 'AVAX'}, {'id': 'metis-andromeda', 'chain_identifier': 1088, 'name': 'Metis Andromeda', 'shortname': ''}, {'id': 'ethereum', 'chain_identifier': 1, 'name': 'Ethereum', 'shortname': ''}, {'id': 'acala', 'chain_identifier': None, 'name': 'Acala', 'shortname': ''}, {'id': 'harmony-shard-0', 'chain_identifier': 1666600000, 'name': 'Harmony Shard 0', 'shortname': 'Harmony Shard 0'}, {'id': 'defi-kingdoms-blockchain', 'chain_identifier': None, 'name': 'DFK Chain', 'shortname': 'DFK Chain'}, {'id': 'evmos', 'chain_identifier': 9001, 'name': 'Evmos', 'shortname': 'evmos'}, {'id': 'karura', 'chain_identifier': None, 'name': 'Karura', 'shortname': ''}, {'id': 'everscale', 'chain_identifier': None, 'name': 'Everscale', 'shortname': ''}, {'id': 'boba', 'chain_identifier': 288, 'name': 'Boba Network', 'shortname': ''}, {'id': 'sx-network', 'chain_identifier': None, 'name': 'SX Network', 'shortname': 'sxn'}, {'id': 'cube', 'chain_identifier': None, 'name': 'Cube', 'shortname': ''}, {'id': '', 'chain_identifier': None, 'name': 'Matrix', 'shortname': ''}, {'id': 'elastos', 'chain_identifier': None, 'name': 'Elastos Smart Contract Chain', 'shortname': 'Elastos'}, {'id': 'celo', 'chain_identifier': 42220, 'name': 'Celo', 'shortname': 'celo'}, {'id': 'echelon', 'chain_identifier': None, 'name': 'Echelon', 'shortname': ''}, {'id': 'hydra', 'chain_identifier': None, 'name': 'Hydra', 'shortname': ''}, {'id': 'pulsechain', 'chain_identifier': None, 'name': 'Pulsechain', 'shortname': ''}, {'id': 'coinex-smart-chain', 'chain_identifier': 52, 'name': 'CoinEx Smart Chain', 'shortname': 'CSC'}, {'id': 'hedera-hashgraph', 'chain_identifier': None, 'name': 'Hedera Hashgraph', 'shortname': 'hashgraph'}, {'id': 'linea', 'chain_identifier': None, 'name': 'Linea', 'shortname': ''}, {'id': 'songbird', 'chain_identifier': None, 'name': 'Songbird', 'shortname': ''}, {'id': 'ethereumpow', 'chain_identifier': None, 'name': 'EthereumPoW', 'shortname': ''}, {'id': 'astar', 'chain_identifier': None, 'name': 'Astar', 'shortname': ''}, {'id': 'moonbeam', 'chain_identifier': 1284, 'name': 'Moonbeam', 'shortname': ''}, {'id': 'hoo-smart-chain', 'chain_identifier': 70, 'name': 'Hoo Smart Chain', 'shortname': ''}, {'id': 'dogechain', 'chain_identifier': None, 'name': 'Dogechain', 'shortname': ''}, {'id': 'oasis', 'chain_identifier': 42262, 'name': 'Oasis', 'shortname': 'oasis'}, {'id': 'skale', 'chain_identifier': None, 'name': 'Skale', 'shortname': ''}, {'id': 'flare-network', 'chain_identifier': None, 'name': 'Flare Network', 'shortname': ''}, {'id': 'stacks', 'chain_identifier': None, 'name': 'Stacks', 'shortname': ''}, {'id': 'ShibChain', 'chain_identifier': None, 'name': 'ShibChain', 'shortname': ''}, {'id': 'xdc-network', 'chain_identifier': 50, 'name': 'XDC Network', 'shortname': 'xdc xinfin'}, {'id': 'kadena', 'chain_identifier': None, 'name': 'Kadena', 'shortname': ''}, {'id': 'rootstock', 'chain_identifier': 30, 'name': 'Rootstock RSK', 'shortname': ''}, {'id': 'callisto', 'chain_identifier': None, 'name': 'Callisto', 'shortname': ''}, {'id': 'function-x', 'chain_identifier': 530, 'name': 'Function X', 'shortname': ''}, {'id': 'redlight-chain', 'chain_identifier': 2611, 'name': 'Redlight Chain', 'shortname': ''}, {'id': 'shiden network', 'chain_identifier': 8545, 'name': 'Shiden Network', 'shortname': ''}, {'id': 'sui', 'chain_identifier': None, 'name': 'Sui', 'shortname': ''}, {'id': 'oasys', 'chain_identifier': None, 'name': 'Oasys', 'shortname': ''}, {'id': 'energi', 'chain_identifier': None, 'name': 'Energi', 'shortname': ''}, {'id': 'meter', 'chain_identifier': 82, 'name': 'Meter', 'shortname': ''}, {'id': 'syscoin', 'chain_identifier': 57, 'name': 'Syscoin NEVM', 'shortname': 'syscoin'}, {'id': 'velas', 'chain_identifier': 106, 'name': 'Velas', 'shortname': 'velas'}, {'id': 'okex-chain', 'chain_identifier': 66, 'name': 'OKExChain', 'shortname': 'OKEx'}, {'id': 'onus', 'chain_identifier': None, 'name': 'ONUS', 'shortname': ''}, {'id': 'empire', 'chain_identifier': 3693, 'name': 'Empire', 'shortname': ''}, {'id': 'canto', 'chain_identifier': 7700, 'name': 'Canto', 'shortname': ''}, {'id': 'fuse', 'chain_identifier': 122, 'name': 'Fuse', 'shortname': ''}, {'id': 'tombchain', 'chain_identifier': None, 'name': 'Tombchain', 'shortname': ''}, {'id': 'core', 'chain_identifier': None, 'name': 'Core', 'shortname': ''}, {'id': 'terra-2', 'chain_identifier': None, 'name': 'Terra', 'shortname': ''}, {'id': 'arbitrum-one', 'chain_identifier': 42161, 'name': 'Arbitrum One', 'shortname': 'Arbitrum'}, {'id': 'zksync', 'chain_identifier': 324, 'name': 'zkSync', 'shortname': ''}, {'id': 'polygon-zkevm', 'chain_identifier': 1101, 'name': 'Polygon zkEVM', 'shortname': ''}, {'id': 'arbitrum-nova', 'chain_identifier': 42170, 'name': 'Arbitrum Nova', 'shortname': ''}, {'id': 'base', 'chain_identifier': None, 'name': 'Base', 'shortname': ''}, {'id': 'ultron', 'chain_identifier': 1231, 'name': 'Ultron', 'shortname': ''}, {'id': 'solana', 'chain_identifier': None, 'name': 'Solana', 'shortname': ''}, {'id': 'kava', 'chain_identifier': None, 'name': 'Kava', 'shortname': ''}, {'id': 'kardiachain', 'chain_identifier': None, 'name': 'KardiaChain', 'shortname': 'kardiachain'}, {'id': 'conflux', 'chain_identifier': None, 'name': 'Conflux', 'shortname': 'conflux'}, {'id': 'kucoin-community-chain', 'chain_identifier': 321, 'name': 'Kucoin Community Chain', 'shortname': 'KCC'}, {'id': 'clover', 'chain_identifier': None, 'name': 'Clover', 'shortname': ''}, {'id': 'step-network', 'chain_identifier': None, 'name': 'Step Network', 'shortname': ''}, {'id': 'godwoken', 'chain_identifier': None, 'name': 'Godwoken', 'shortname': ''}, {'id': 'eos-evm', 'chain_identifier': 17777, 'name': 'EOS EVM', 'shortname': ''}, {'id': 'trustless-computer', 'chain_identifier': None, 'name': 'Trustless Computer', 'shortname': ''}, {'id': 'neutron', 'chain_identifier': None, 'name': 'Neutron', 'shortname': ''}, {'id': 'injective', 'chain_identifier': None, 'name': 'Injective', 'shortname': ''}, {'id': 'aptos', 'chain_identifier': None, 'name': 'Aptos', 'shortname': ''}, {'id': 'binance-smart-chain', 'chain_identifier': 56, 'name': 'BNB Smart Chain', 'shortname': 'BSC'}, {'id': 'tenet', 'chain_identifier': 1559, 'name': 'Tenet', 'shortname': ''}, {'id': 'thundercore', 'chain_identifier': 108, 'name': 'ThunderCore', 'shortname': ''}, {'id': 'ordinals', 'chain_identifier': None, 'name': 'Ordinals', 'shortname': ''}, {'id': 'mantle', 'chain_identifier': 5000, 'name': 'Mantle', 'shortname': ''}, {'id': 'milkomeda-cardano', 'chain_identifier': 2001, 'name': 'Milkomeda (Cardano)', 'shortname': ''}]
# %%
import os
import time
import json
results = {}
for file in os.listdir("./pos"):
    addr = file.split(".")[0]
    try:
        info = cg.get_coin_info_from_contract_address_by_id("ethereum", addr)
    except:
        info = None
    time.sleep(2)
    results[addr] = info
with open("coingecko_gambling_token_addr_results.json", "w") as f:
    json.dump(results, f)

# %%

