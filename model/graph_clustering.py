import networkx as nx
import json
from collections import defaultdict
import random
import pandas as pd


def graph_clustering(edge_file_path, parquet_path, output_path="final_clusters.json"):
    # Load cạnh
    with open(edge_file_path, "r") as f:
        inter_edges = json.load(f)

    df = pd.read_parquet(parquet_path)
    df["domain1"] = df["domain1"].astype(str)
    df["domain2"] = df["domain2"].astype(str)
    df = df.dropna(subset=["domain1", "domain2", "similarity"])
    domain_groups = df.groupby("domain1")["domain2"].apply(set).to_dict()

    # Tạo graph và các thành phần liên thông
    G = nx.Graph()
    G.add_edges_from(inter_edges)
    components = list(nx.connected_components(G))
    print(f"🔹 Số lượng cụm: {len(components)}")
    print(f"🔹 Kích thước cụm lớn nhất: {max(len(c) for c in components)}")

    # Gom domain2 theo cụm domain1
    final_clusters = []
    for domain1_set in components:
        domain2_set = []
        for d in domain1_set:
            domain2_set.extend(domain_groups.get(d, []))
        final_clusters.append((list(domain1_set), domain2_set))

    # Đếm số lần xuất hiện
    domain_to_cluster_count = defaultdict(lambda: defaultdict(int))
    for i, (d1_list, d2_list) in enumerate(final_clusters):
        for domain in d1_list + d2_list:
            domain_to_cluster_count[domain][i] += 1

    # Gán domain vào cụm xuất hiện nhiều nhất
    domain_final_cluster = {}
    for domain, cluster_counts in domain_to_cluster_count.items():
        max_count = max(cluster_counts.values())
        best_clusters = [idx for idx, count in cluster_counts.items() if count == max_count]
        chosen_cluster = random.choice(best_clusters)
        domain_final_cluster[domain] = chosen_cluster

    # Gom lại thành danh sách
    new_clusters = defaultdict(list)
    for domain, cluster_idx in domain_final_cluster.items():
        new_clusters[cluster_idx].append(domain)

    final_clusters_unique = list(new_clusters.values())

    with open(output_path, "w") as f:
        json.dump(final_clusters_unique, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(final_clusters_unique)} cụm domain vào {output_path}")
    return final_clusters_unique
