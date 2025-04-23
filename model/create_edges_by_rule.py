# create_edges_by_rule.py

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import json
import pandas as pd


def process_pairs(args):
    """Tính toán các cặp (i, j) mà i thuộc khoảng [start_i, end_i]"""
    start_i, end_i, domain_list, domain_groups, threshold, worker_id = args
    print(f"🔹 CPU {worker_id} xử lý từ index {start_i} đến {end_i - 1}")
    inter_edges = []

    for i in range(start_i, end_i):
        d1 = domain_list[i]
        for j in range(i + 1, len(domain_list)):
            d2 = domain_list[j]

            intersection = len(domain_groups[d1] & domain_groups[d2])
            min_size = min(len(domain_groups[d1]), len(domain_groups[d2]))

            if min_size > 0 and (intersection / min_size) >= threshold:
                inter_edges.append([d1, d2])

    return inter_edges


def create_edges_from_similarity_file(
    input_file: str,
    output_file: str,
    threshold: float = 0.6,
    num_workers: int = 4
):
    """
    Tạo danh sách cạnh giữa các domain dựa trên độ trùng lặp domain2.

    Parameters:
        input_file (str): Đường dẫn tới file Parquet chứa các cặp domain và similarity.
        output_file (str): Đường dẫn file JSON để lưu kết quả.
        threshold (float): Ngưỡng trùng lặp tối thiểu (0.0 - 1.0).
        num_workers (int): Số lượng tiến trình xử lý song song.
    """
    print(f"📥 Đọc file: {input_file}")
    df = pd.read_parquet(input_file)
    df["domain1"] = df["domain1"].astype(str)
    df["domain2"] = df["domain2"].astype(str)
    df = df.dropna(subset=["domain1", "domain2", "similarity"])

    domain_groups = df.groupby("domain1")["domain2"].apply(set).to_dict()
    domain_list = list(domain_groups.keys())
    N = len(domain_list)

    print(f"🧠 Tổng số domain cần xử lý: {N} (dùng {num_workers} workers)")
    chunk_size = N // num_workers
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else N,
         domain_list, domain_groups, threshold, i)
        for i in range(num_workers)
    ]

    for start, end, *_ , worker_id in ranges:
        print(f"🟢 Worker {worker_id}: Xử lý từ {start} đến {end - 1}")

    inter_edges = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_pairs, ranges))

    for res in results:
        inter_edges.extend(res)

    print(f"✅ Tổng số cạnh tạo được: {len(inter_edges)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(inter_edges, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu vào: {output_file}")
