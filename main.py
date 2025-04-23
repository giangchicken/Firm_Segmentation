from model.create_edges_by_rule import *
from model.top_k_nearest_neighbors import *
from model.graph_clustering import *
from src.embedding.embedding import *
from src.similarity_calculation.similarity_matrix import *

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import polars as pl


def main():
    # ==== 0. LOAD CONFIG ====
    load_dotenv("./config/config.env")

    HTML_DIR = Path(os.getenv("HTML_DIR"))
    MODEL_NAME = os.getenv("MODEL_NAME")
    EMBEDDING_OUTPUT_DIR = Path(os.getenv("EMBEDDING_OUTPUT_DIR"))
    EMBEDDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    similarity_output_path = os.getenv("SIMILARITY_PARQUET_OUTPUT")
    memory_limit = os.getenv("MEMORY_LIMIT")
    n_workers = int(os.getenv("N_WORKERS", 1))
    threads_per_worker = int(os.getenv("THREADS_PER_WORKER", 1))
    tmp_dir = os.getenv("TEMPORARY_DIRECTORY", "/tmp")

    top_k_output_path = os.getenv("TOP_K_NEIGHBOR_OUTPUT")
    edge_output_path = os.getenv("EDGE_OUTPUT_PATH")
    cluster_output_path = os.getenv("CLUSTER_OUTPUT_PATH")

    # ==== 1. NHÚNG EMBEDDING ====
    print("🚀 Đang nhúng embedding từ file HTML...")
    json_file_path = Path(HTML_DIR) / "mapping.json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        firm_info_list = json.load(f)
    file_paths = [str(item["html_path"]) for item in firm_info_list]

    if not file_paths:
        print("❌ Không tìm thấy file HTML.")
        exit()

    processor = EmbeddingProcessor(
        file_paths=file_paths,
        output_dir="embeddings",
        model_name=MODEL_NAME,
        type="dev",
        batch_size=8
    )

    info = processor.process_embedding()
    print(f"✅ Đã lưu embedding tại {info['storage_folder']}, tổng {info['num_files']} files")

    # ==== 2. LƯU EMBEDDING & DOMAIN ====
    print("📥 Load embeddings và metadata...")
    embeddings_path = Path(info["storage_folder"]) / "X_dev.npy"
    metadata_path = Path(info["storage_folder"]) / "split_info_dev.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    firms = [str(item["firm"]) for item in firm_info_list if item["html_path"] in metadata["dev"]]

    all_vectors = np.load(embeddings_path)

    vectors_list = all_vectors.tolist()

    # Tạo DataFrame với firm + vector
    df = pl.DataFrame({
        "firm_": firms,
        "feature_vector": vectors_list
    })

    # Lưu ra file Parquet
    df.write_parquet((EMBEDDING_OUTPUT_DIR / "dev_vectors.parquet").as_posix())
    print(f"✅ Đã lưu vector vào: {EMBEDDING_OUTPUT_DIR / 'dev_vectors.parquet'}")

    # ==== 3. TÍNH SIMILARITY MATRIX BẰNG DASK ====
    print("🔹 Bước 3: Tính similarity matrix")

    smc = SimilarityMatrixDask(
        data_path=(EMBEDDING_OUTPUT_DIR / "dev_vectors.parquet").as_posix(),
        output_path=similarity_output_path,
        memory_limit=memory_limit,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        error_api_url=None
    )
    smc.setup_client({"./temporary_directory": tmp_dir})
    smc.compute_and_save_similarity_matrix()

    # ==== 4. TOP-K NEAREST NEIGHBORS ====
    print("🔹 Bước 4: Lấy top-k nearest neighbors")
    extract_top_k_neighbors(folder=similarity_output_path, top_k=3, output_folder=top_k_output_path)

    # ==== 5. TẠO EDGE THEO LUẬT ====
    print("🔹 Bước 5: Tạo danh sách cạnh đồ thị")
    create_edges_from_similarity_file(
        input_file=top_k_output_path,
        output_file=edge_output_path,
        threshold=0.6,
        num_workers=6
    )

    # ==== 6. GRAPH CLUSTERING ====
    print("🔹 Bước 6: Gom nhóm domain theo graph clustering")
    graph_clustering(
        edge_file_path=edge_output_path,
        parquet_path=top_k_output_path,
        output_path=cluster_output_path
    )

    print("✅ Hoàn thành toàn bộ pipeline!")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    main()
