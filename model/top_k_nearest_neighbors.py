# top_k_nearest_neighbor.py

import os
import glob
import re
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

def extract_number(filename):
    """Lấy số đầu tiên trong tên file để sắp xếp."""
    name = os.path.basename(filename)
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')

def extract_top_k_neighbors(folder: str, output_folder: str, top_k: int = 10):
    """
    Hàm đọc các file similarity matrix từ SBERT, trích xuất top K domain tương tự nhất và lưu kết quả ra file Parquet.

    Parameters:
        folder (str): Đường dẫn tới thư mục chứa các file Parquet.
        output_folder (str): Thư mục để lưu kết quả.
        top_k (int): Số lượng domain gần nhất cần lấy cho mỗi domain.
    """
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")), key=extract_number)

    all_nearest_neighbors = []

    for file in tqdm(files, desc="🔄 Đang xử lý các file"):
        print(f"🔍 Đang xử lý: {file}")
        df = pl.read_parquet(file)
        domain_ = df["domain_"]
        df_data = df.drop(["domain_", "__null_dask_index__"])
        columns = df_data.columns
        similarity_matrix = df_data.to_numpy()

        for i, domain in enumerate(domain_):
            nearest_idx = np.argsort(similarity_matrix[i])[::-1]
            nearest_idx = nearest_idx[nearest_idx != i]  # Bỏ chính domain

            for j in nearest_idx[:top_k]:
                all_nearest_neighbors.append((domain, columns[j], similarity_matrix[i][j]))

    df_nearest = pd.DataFrame(all_nearest_neighbors, columns=["domain1", "domain2", "similarity"])
    output_file = os.path.join(output_folder, "top_k_nearest_neighbors.parquet")
    df_nearest.to_parquet(output_file, index=False)
    print(f"✅ Đã lưu kết quả vào {output_file}")
