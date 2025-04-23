import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .preprocessing import Preprocessor
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger('uvicorn.error')

class EmbeddingProcessor:
    """
    Class Embedding:
        - Nhúng văn bản từ file HTML.
        - Lưu embeddings vào thư mục.

    Args:
        - file_paths: Danh sách đường dẫn file HTML.
        - labels: Danh sách nhãn tương ứng.
        - output_dir: Thư mục lưu embeddings.
        - model_name: Tên mô hình embedding.
        - get_text_lib: Thư viện đọc văn bản từ file HTML (newspaper, inscriptis).
        - type: storage folder 
    
    Output:
        - Tạo file embeddings .npy và metadata .json.
        - Trả về thư mục lưu trữ, số lượng file và tổng dung lượng thư mục.
    """

    MODELS = {
        "SBERT": "keepitreal/vietnamese-sbert",
        "Bi-Encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    }

    def __init__(self, file_paths, output_dir="embedding", model_name="SBERT", type="train", batch_size=8):
        self.file_paths = file_paths
        self.output_dir = Path(output_dir) / model_name
        self.type = type
        self.batch_size = batch_size
        # self.error_files = []
        self.preprocessor = Preprocessor()

        if model_name not in self.MODELS:
            raise ValueError(f"Model {model_name} không được hỗ trợ.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"🔍 Using device: {self.device}")
        
        self.encoder = SentenceTransformer(self.MODELS[model_name]).to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)



    def process_embedding(self):
        """
        Nhúng embedding và xử lý dữ liệu.
        """
        
        descriptions, file_names = self.preprocessor.process_files(self.file_paths)

        # Nhúng văn bản bằng batch
        embeddings = self.batch_encode(descriptions)

        # Lưu dữ liệu
        self.save_embedding(embeddings, file_names, self.type)
        
        # Trả về thông tin thư mục lưu trữ
        return self.get_storage_info()

    def batch_encode(self, texts):
        """Nhúng văn bản theo batch để tiết kiệm RAM."""
        batched_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding batch"):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.encoder.encode(batch)
            # batch_embeddings = batch_embeddings.cpu().numpy()
            batched_embeddings.append(batch_embeddings)
        return np.vstack(batched_embeddings)

    def save_embedding(self, embeddings, file_names, type):
        """Lưu tập dữ liệu embeddings."""
        np.save(self.output_dir / f"X_{type}.npy", embeddings)

        split_info = {f"{type}": file_names}
        with open(self.output_dir / f"split_info_{type}.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=4)

        logger.debug(f"✅ Saved embeddings in {self.output_dir}")

    def get_storage_info(self):
        """Trả về thư mục lưu trữ, số lượng file đã xử lý và tổng dung lượng thư mục."""
        total_size = sum(f.stat().st_size for f in self.output_dir.glob("**/*") if f.is_file())
        num_files = len(list(self.output_dir.glob("**/*")))
        
        return {
            "storage_folder": str(self.output_dir),
            "num_files": num_files,
            "total_size_bytes": total_size
        }
