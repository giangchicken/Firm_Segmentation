import os
from inscriptis import get_text

class Preprocessor:
    """
    Class Preprocessor:
        - Nhận danh sách đường dẫn file HTML.
        - Tiền xử lý và loại bỏ ký tự xuống dòng.
    
    Output:
        - Danh sách extracted text
    """

    def read_html(self, html_file):
        """Đọc nội dung trang web từ file HTML và loại bỏ ký tự xuống dòng."""
        try:
            with open(html_file, "r", encoding="utf-8") as file:
                html_content = file.read()
                extracted_text = get_text(html_content)
            return extracted_text.replace("\n", " ").strip() if extracted_text else ""
        except Exception as e:
            print(f"Error reading file {html_file}: {str(e)}")
            return ""

    def process_files(self, file_paths, labels=None):
        """Tiền xử lý văn bản và nhúng embedding."""
        descriptions, file_names = [], []

        for file_path in file_paths:
            extracted_text = self.read_html(file_path)
            if extracted_text:
                descriptions.append(extracted_text)
                file_names.append(file_path)

        return descriptions, file_names


