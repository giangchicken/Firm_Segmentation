# Phân Nhóm Doanh Nghiệp Dựa Trên Văn Bản Website

## Tổng Quan

Dự án này nhằm mục tiêu **phân nhóm (segment) các doanh nghiệp** dựa trên nội dung văn bản được thu thập từ **website chính thức** của họ. Bằng cách ứng dụng kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và phân cụm đồ thị, chúng tôi xây dựng một hệ thống có thể mở rộng và dễ diễn giải để hiểu mối liên hệ giữa các doanh nghiệp và phân loại chúng theo lĩnh vực hoạt động.

## Mục Tiêu

- Thu thập và xử lý nội dung từ website doanh nghiệp.
- Biểu diễn doanh nghiệp thông qua embedding sinh từ mô hình ngôn ngữ hiện đại.
- Xây dựng ma trận tương đồng và biểu diễn dưới dạng đồ thị.
- Phân cụm doanh nghiệp để tìm ra các nhóm đặc trưng.

## Phương Pháp

1. **Thu thập dữ liệu**: Crawl nội dung HTML từ các website chính thức của doanh nghiệp.
2. **Tiền xử lý**: Làm sạch và trích xuất văn bản (giới thiệu, dịch vụ...).
3. **Embedding**: Mã hóa văn bản bằng Sentence-BERT (SBERT) hoặc các mô hình tương đương.
4. **Tính tương đồng**: Dùng cosine similarity để tính mức độ tương đồng giữa các doanh nghiệp.
5. **Xây dựng đồ thị**: Nối các doanh nghiệp lại với nhau dựa trên k lân cận gần nhất (top-k).
6. **Phân cụm**: Sử dụng thuật toán Louvain hoặc các phương pháp phát hiện cộng đồng khác để chia nhóm.
7. **Xuất kết quả**: Trả ra danh sách các nhóm doanh nghiệp theo cấu trúc đã phân tích.

---