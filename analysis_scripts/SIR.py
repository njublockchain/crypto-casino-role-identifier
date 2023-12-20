import igraph as ig
import csv
import multiprocessing
import random

# 读取 GraphML 文件
def read_graph(file_path):
    return ig.Graph.Read_GraphML(file_path)

# 获取节点的索引
def get_node_indices(G):
    return range(G.vcount())

# # 获取高阶邻居
# def get_high_order_neighbors(G, u, order, neighbor_type='out'):
#     if order == 0:
#         return set([u])
#     elif order == 1:
#         if neighbor_type == 'in':
#             return set(G.neighbors(u, mode="in"))
#         else:
#             return set(G.neighbors(u, mode="out"))
    
#     neighbors = set()
#     if order > 1:
#         if neighbor_type == 'in':
#             for v in get_high_order_neighbors(G, u, order - 1, neighbor_type):
#                 neighbors.update(G.neighbors(v, mode="in"))
#         else:
#             for v in get_high_order_neighbors(G, u, order - 1, neighbor_type):
#                 neighbors.update(G.neighbors(v, mode="out"))
#         neighbors.difference_update(get_high_order_neighbors(G, u, order - 2, neighbor_type))
#     neighbors.discard(u)
#     return neighbors

# # 计算全局感染概率
# def calculate_global_infection_probability(G):
#     total_weight = 0
#     total_weight_squared = 0

#     for edge in G.es:
#         weight = edge['amount']  # 假设边的权重属性名为 'weight'
#         total_weight += weight
#         total_weight_squared += weight**2

#     return total_weight / total_weight_squared if total_weight_squared > 0 else 0

# # 计算 H-index
# def h_index(scores):
#     scores.sort(reverse=True)
#     h = 0
#     for i, score in enumerate(scores):
#         if score >= i + 1:
#             h = i + 1
#         else:
#             break
#     return h

# def calculate_h_indices(G):
#     h_indices = {}
#     for u in get_node_indices(G):
#         degrees = [G.degree(v) for v in G.neighbors(u, mode="ALL")]
#         h_indices[u] = h_index(degrees)
#     return h_indices

# # 计算感染概率得分
# def calculate_infected_probability_score(G, u, alpha, i):
#     if i == 0:
#         return 1
#     score = 1
#     for q in get_high_order_neighbors(G, u, i - 1):
#         score *= (1 - calculate_infected_probability_score(G, q, alpha, i - 1) * alpha)
#     return 1 - score

# # 计算单个节点的 Rank_out 和 Rank_in
# def calculate_rank_for_node(G, u, alpha):
#     rank_out = sum(calculate_infected_probability_score(G, v, alpha, i)
#                    for i in range(1, 2)
#                    for v in get_high_order_neighbors(G, u, i, 'out'))

#     rank_in = 0
#     for v in G.neighbors(u, mode="in"):
#         for i in range(1, 2):
#             for w in get_high_order_neighbors(G, v, i, 'out'):
#                 rank_in += calculate_infected_probability_score(G, w, alpha, i)

#     return u, rank_out, rank_in

# # 使用多进程计算所有节点的 Rank_out 和 Rank_in
# def calculate_ranks_multiprocessing(G, alpha):
#     ranks_out = {}
#     ranks_in = {}

#     pool = multiprocessing.Pool(processes=min(20, multiprocessing.cpu_count()))
#     results = [pool.apply_async(calculate_rank_for_node, args=(G, u, alpha)) for u in get_node_indices(G)]
#     pool.close()
#     pool.join()

#     for result in results:
#         u, rank_out, rank_in = result.get()
#         ranks_out[u] = rank_out
#         ranks_in[u] = rank_in

#     return ranks_out, ranks_in

# # 以下部分是新增的，用于计算额外的中心性和权重度数指标
# def calculate_additional_metrics(G):
#     G.es['amount'] = G.es['amount'] if 'amount' in G.es.attribute_names() else [1] * G.ecount()  # 确保所有边都有权重
#     in_degree = G.degree(mode='in')
#     out_degree = G.degree(mode='out')
#     weighted_in_degree = G.strength(mode='in', weights='amount')
#     weighted_out_degree = G.strength(mode='out', weights='amount')
#     closeness = G.closeness()
#     betweenness = G.betweenness()
#     pagerank = G.pagerank(weights='amount')
#     authority, hub = G.authority_score(weights='amount'), G.hub_score(weights='amount')
#     clustering_coefficients = G.transitivity_local_undirected(vertices=None, mode="zero")

#     return in_degree, out_degree, weighted_in_degree, weighted_out_degree, closeness, betweenness, pagerank, authority, hub, clustering_coefficients

# SIR模型模拟
def simulate_sir(G, initial_infected, beta, gamma, max_steps):
    status = {v.index: 'S' for v in G.vs}  # 所有节点初始为易感
    status[initial_infected] = 'I'  # 设置初始感染节点

    recovered_count = 0
    for _ in range(max_steps):
        new_status = status.copy()
        for v in G.vs:
            if status[v.index] == 'I':
                if random.random() < gamma:  # 恢复概率
                    new_status[v.index] = 'R'
                    recovered_count += 1
                else:
                    for neighbor in v.neighbors():
                        if status[neighbor.index] == 'S' and random.random() < beta:
                            new_status[neighbor.index] = 'I'
        status = new_status

    return recovered_count / G.vcount()  # 返回恢复的节点比例

# 计算每个节点的 SIR F(tc) 平均值
def calculate_sir_ftc(G, beta, gamma, max_steps, simulations):
    ftc_values = {v.index: 0 for v in G.vs}
    for v in G.vs:
        total_ftc = 0
        for _ in range(simulations):
            total_ftc += simulate_sir(G, v.index, beta, gamma, max_steps)
        ftc_values[v.index] = total_ftc / simulations
    return ftc_values

# 读取图
G = read_graph('/home/ta/gambling/4nd-Macao-online-casino/graph_0xc4a482146c2b493066aa7427d23bea4f66e5279c_2.graphml')

# # 计算全局感染概率
# alpha = calculate_global_infection_probability(G)

# # 计算 Rank_out 和 Rank_in
# ranks_out, ranks_in = calculate_ranks_multiprocessing(G, alpha)

# # 计算额外的指标
# in_degree, out_degree, weighted_in_degree, weighted_out_degree, closeness, betweenness, pagerank, authority, hub, clustering_coefficients = calculate_additional_metrics(G)

# # 计算H-index
# h_indices = calculate_h_indices(G)

# 运行SIR模型并计算F(tc)
sir_ftc = calculate_sir_ftc(G, beta=0.3, gamma=0.1, max_steps=10, simulations=100)

# 将结果写入 CSV
with open('node_metrics_arb_onlysir.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['Node ID', 'Rank_out', 'Rank_in', 'H-index', 'In-Degree', 'Out-Degree', 'Weighted In-Degree', 'Weighted Out-Degree', 'Closeness', 'Betweenness', 'Pagerank', 'Authoritativeness', 'Hubness', 'Clustering Coefficients', 'SIR F(tc)'])
    # writer.writerow(['Node ID', 'SIR F(tc)'])
    for u in get_node_indices(G):
        node_id = G.vs[u]['id']
        #writer.writerow([node_id, ranks_out[u], ranks_in[u], h_indices[u], in_degree[u], out_degree[u], weighted_in_degree[u], weighted_out_degree[u], closeness[u], betweenness[u], pagerank[u], authority[u], hub[u], clustering_coefficients[u], sir_ftc[u]])
        writer.writerow([node_id, sir_ftc[u]])
print("Metrics including SIR F(tc) have been successfully calculated and saved to 'node_metrics.csv'.")