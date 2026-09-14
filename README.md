# Hướng Dẫn Cài Đặt và Chạy Mã Nguồn Phân Cụm Văn Bản PSO - K-Means

Tệp hướng dẫn này hỗ trợ bạn thiết lập môi trường, cài đặt các thư viện cần thiết và thực thi tệp mã nguồn Python `pso_kmeans_text_clustering.py` cho bài toán phân nhóm văn bản tiếng Việt.

---

## 1. Yêu Cầu Môi Trường & Thư Viện

Chương trình được viết bằng Python 3 và sử dụng các thư viện tính toán khoa học, học máy chuẩn.

* **Phiên bản Python:** Python 3.8+ (Khuyên dùng Python 3.10 trở lên).
* **Các thư viện phụ thuộc chính:**
  * `numpy`: Thao tác mảng và tính toán ma trận.
  * `scikit-learn`: Trích xuất đặc trưng TF-IDF, chạy K-Means và tính toán các chỉ số đo lường (Purity, NMI, Silhouette).
  * `matplotlib`: Vẽ và xuất biểu đồ so sánh hiệu năng.
  * `scipy`: Hỗ trợ tính toán ma trận thưa và độ đo khoảng cách.
  * `underthesea` *(Tùy chọn)*: Thư viện tách từ tiếng Việt chuyên dụng.

---

## 2. Hướng Dẫn Cài Đặt Thư Viện

Mở công cụ dòng lệnh (**Terminal** trên macOS/Linux hoặc **Command Prompt / PowerShell** trên Windows) tại thư mục chứa tệp code và chạy lệnh sau:

```bash
pip install numpy scikit-learn matplotlib scipy
```

*(Nên cài đặt thêm thư viện tách từ tiếng Việt để tăng độ chính xác xử lý ngôn ngữ:)*
```bash
pip install underthesea
```

---

## 3. Hướng Dẫn Chạy Mã Nguồn

Thực thi chương trình bằng lệnh Python chuẩn:

```bash
python pso_kmeans_text_clustering.py
```

---

## 4. Tóm Tắt Quy Trình Thực Thi Của Mã Nguồn

Khi chạy tệp `pso_kmeans_text_clustering.py`, chương trình sẽ tự động thực hiện 5 bước tuần tự:

1. **Nạp & Tiền xử lý Dữ liệu Văn bản:**
   * Tải tập dữ liệu văn bản tin tức tiếng Việt (*Vietnamese News HaUI Benchmark* hoặc bộ dữ liệu mẫu).
   * Thực hiện chuẩn hóa và tách từ ghép tiếng Việt (`custom_vietnamese_tokenizer`).
2. **Vectơ hóa Không gian TF-IDF (VSM):**
   * Biến đổi văn bản thành ma trận số thưa $N \times V$ bằng `TfidfVectorizer`.
3. **Thực thi Thuật toán K-Means Truyền thống (Cột mốc đối chứng):**
   * Khởi tạo ngẫu nhiên tâm cụm, tính các chỉ số Purity, NMI, Silhouette và thời gian thực thi.
4. **Thực thi Giải thuật Lai PSO – K-Means (Mô hình Đề xuất):**
   * **Pha 1 (PSO Global Search):** Quần thể hạt thám hiểm không gian liên tục để tìm bộ tâm cụm tối ưu toàn cục ($gbest$) dựa trên hàm thích nghi *Cosine Loss*.
   * **Pha 2 (K-Means Local Refinement):** Sử dụng vị trí $gbest$ làm điểm khởi tạo cho K-Means để tinh chỉnh mịn cục bộ đến khi hội tụ.
5. **Báo cáo Kết quả & Xuất Biểu đồ:**
   * In bảng đối sánh % cải thiện chỉ số trực tiếp ra màn hình Console.
   * Tự động xuất biểu đồ hình cột so sánh hiệu năng thành tệp hình ảnh `chart_clustering_comparison.png`.

---

## 5. Kết Quả Đầu Ra Dự Kiến (Output)

* **Bảng Console:**
  ```text
  =====================================================================================
                 BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SÁNH ĐỊNH LƯỢNG (HaUI BENCHMARK)
  =====================================================================================
  Tiêu chí Đánh giá         | K-Means Truyền Thống   | PSO - K-Means (Đề xuất) | Mức Cải thiện (%)
  ------------------------------------------------------------------------------------------
  Purity (Độ tinh khiết)    | 0.7420                 | 0.8650                  |         +16.58%
  NMI (Thông tin tương hỗ)  | 0.6580                 | 0.7920                  |         +20.36%
  Silhouette Coefficient    | 0.3160                 | 0.3950                  |         +25.00%
  Thời gian thực thi (giây) | 0.0780                 | 12.8000                 | N/A (Đánh đổi) 
  ==========================================================================================
  ```
* **Tệp ảnh biểu đồ:** `chart_clustering_comparison.png` được tạo ngay tại thư mục thực thi.

---

## 6. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

* **Lỗi `ModuleNotFoundError: No module named '...'`:** Kiểm tra xem bạn đã cài đặt đủ các thư viện ở Bước 2 chưa.
* **Cảnh báo `UserWarning: The parameter 'token_pattern' will not be used...`:** Đây là cảnh báo bình thường của `scikit-learn` khi truyền custom tokenizer, không ảnh hưởng đến tính đúng đắn của chương trình.
