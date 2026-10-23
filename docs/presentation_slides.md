# SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN NGHIÊN CỨU

## ĐỀ TÀI: UFM-REC — HỆ KHUYẾN NGHỊ SẢN PHẨM COLD-START ĐA PHƯƠNG THỨC DỰA TRÊN ĐỘ TIN CẬY HỌC ĐƯỢC
*(Multimodal Cold-Start Recommender with Learned Reliability Fusion)*

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tên đề tài**: UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được
- **Học phần**: Học máy ứng dụng
- **Lĩnh vực**: Hệ thống gợi ý (Recommender Systems), Học sâu đa phương thức (Multimodal Deep Learning), Ước lượng độ tin cậy (Reliability Proxy).
- **Tập dữ liệu**: Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, 11,57 triệu tương tác) & All Beauty (86.014 sản phẩm).
- **Nhóm thực hiện**: Trần An Kỳ, Nguyễn Thành Phong, Hải, Hân
- **Giảng viên hướng dẫn**: ThS. Nguyễn Thái Anh
- **Thời gian**: Tháng 10 / 2026

> **Lời thoại mở đầu (Speaker Notes)**:
> *"Kính thưa Thầy Cô trong Hội đồng và các bạn, hôm nay nhóm em xin phép trình bày đề tài UFM-Rec — nghiên cứu hệ khuyến nghị sản phẩm kết hợp đặc trưng nền tảng thị giác - ngôn ngữ CLIP đóng băng với biểu diễn tuần tự nhân quả và cơ chế cổng độ tin cậy học được."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & ĐỘNG LỰC NGHIÊN CỨU
- **Vấn đề thực tế**:
  + Hàng trăm nghìn sản phẩm mới liên tục gia nhập sàn TMĐT với rất ít hoặc chưa có lịch sử tương tác.
- **Phân loại phân tầng tần suất sản phẩm trong tập Train (Regimes)**:
  1. `Zero-shot`: Sản phẩm chưa có tương tác trong tập train ($count_{train} = 0$).
  2. `Extreme Cold`: Sản phẩm cực hiếm ($1 \le count_{train} \le 5$).
  3. `Cold`: Sản phẩm ít tương tác ($6 \le count_{train} \le 20$).
  4. `Warm`: Sản phẩm phổ biến ($count_{train} > 20$).
- **Thách thức của các tiếp cận hiện tại**:
  + **Collaborative Filtering (BPR-MF, SVD) & Sequential thuần ID**: Không có embedding đã học cho item zero-shot, suy thoái nặng ở nhóm ít tương tác.
  + **Content-based (TF-IDF)**: Bắt từ khóa văn bản tốt ở nhóm lạnh nhưng bỏ qua ngữ cảnh hình ảnh và kém hiệu quả ở nhóm Warm.
  + **Ghép nối đơn giản (Concat/MLP)**: Nhánh ID dễ lấn át nhánh nội dung hoặc gây nhiễu khi sản phẩm chỉ có vài lượt tương tác.

---

### SLIDE 3: GIẢI PHÁP ĐỀ XUẤT — UFM-REC
**Ý tưởng cốt lõi**:
1. **Đa phương thức đóng băng (Frozen Multimodal)**: Sử dụng **CLIP ViT-B/32** (khóa cố định commit revision) để trích xuất biểu diễn ngữ nghĩa của văn bản và hình ảnh sản phẩm.
2. **Chuỗi hành vi nhân quả (Causal Sequential)**: Sử dụng mạng **Transformer Encoder** mô hình hóa chuỗi lịch sử tương tác (tối đa 20 sản phẩm gần nhất).
3. **Cổng độ tin cậy học được (Learned Reliability Gate)**: Ước lượng đại lượng proxy độ bất định của từng nhánh để điều tiết trọng số giữa nội dung CLIP và lịch sử ID.
4. **Căn chỉnh liên phương thức (Cross-Modal Alignment)**: Đồng bộ hóa không gian biểu diễn khi cả hai nhánh cùng khả dụng.

---

### SLIDE 4: KIẾN TRÚC TỔNG THỂ HỆ THỐNG
*(Trình chiếu sơ đồ kiến trúc UFM-Rec)*

- **Nhánh Ngữ nghĩa (Semantic Branch)**:
  + Input: Text Embedding (512d) + Image Embedding (512d) + Availability Mask nhị phân $m \in \{0,1\}^2$.
  + Phương thức bị thiếu được zero-fill trước khi qua Adapter 2 tầng tuyến tính + GELU + LayerNorm chiếu về $d = 128$.
- **Nhánh Cộng tác (Collaborative Branch)**:
  + Input: Chuỗi ID tương tác (độ dài 20, right-padded) + Causal Masking.
  + **Unknown Item Masking**: Các ID có $count_{train} = 0$ được gán về ID 0 (Padding) **trước khi lookup**, ngăn chặn rò rỉ trọng số hoặc sinh gradient lỗi.
- **Tương tác Cặp (Pair Interaction)**:
  + Ghép nối $[user, item, user \odot item]$ cùng đặc trưng độ dài lịch sử / tần suất $\log(1 + count_{train})$.

---

### SLIDE 5: CƠ CHẾ TOÁN HỌC CỦA CỔNG DUNG HỢP
#### 1. Ước lượng đại lượng Proxy độ bất định (Learned Reliability Proxy)
$$u_{cf} = \text{Softplus}(\text{MLP}_{cf}(h_{cf})), \quad u_{sem} = \text{Softplus}(\text{MLP}_{sem}(h_{sem}))$$
*(Huấn luyện khớp phần dư bình phương có điều kiện $(\sigma(s_k)_{\text{detached}} - y)^2$ trên tập ứng viên)*

#### 2. Trọng số độ tin cậy (Reliability Weights)
$$w_k = \frac{\exp(-u_k)}{\sum_{j \in \{cf, sem\}} \exp(-u_j)} \cdot \mathbb{I}(\text{branch } k \text{ available})$$
- Khi item là **Zero-shot** ($count_{train} = 0$): $\mathbb{I}_{cf} = 0 \implies w_{cf} = 0,0$ và $w_{sem} = 1,0$.
- Khi item đã có tương tác trong train: Mạng tự động phân bổ trọng số giữa hai nhánh.

#### 3. Căn chỉnh liên phương thức (Cross-Modal Alignment)
$$g_{align} = \sigma(\text{MLP}_{gate}([h_{cf}, h_{sem}])) \odot \text{Linear}([h_{cf}, h_{sem}]) \cdot \mathbb{I}_{cf \wedge sem}$$
$$h_{fused} = w_{cf} h_{cf} + w_{sem} h_{sem} + g_{align}$$

---

### SLIDE 6: THIẾT LẬP THỰC NGHIỆM CHUẨN MỰC
- **Tập dữ liệu chính**: Amazon Toys & Games (**767.045 sản phẩm**, **758.692 ảnh hợp lệ — 98,91%**, **11.572.689 tương tác**).
- **Chiến lược phân chia**: **Global Temporal Split 80/10/10** theo mốc thời gian thực tế, ngăn chặn rò rỉ dữ liệu tương lai.
- **Giao thức đánh giá Validation**:
  + **40.000 ca kiểm thử** (1 sản phẩm đích + 99 sản phẩm âm tính/ca; đánh giá trên tập ứng viên lấy mẫu, không phải full-catalog).
  + Lấy mẫu âm tính theo tần suất xuất hiện trong tập train (**Popularity Negative Sampling**).
  + Phân tầng cân bằng: 10.000 ca cho mỗi nhóm Zero-shot, Extreme Cold, Cold, Warm.

---

### SLIDE 7: KẾT QUẢ THỰC NGHIỆM TRÊN TẬP VALIDATION (`seed=42`)

| Phân Tầng (Regime) | TF-IDF | BPR-MF | SVD | **UFM-Rec (`seed=42`)** | Nhận xét thực nghiệm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Zero-shot (0)** | 0,1865 | 0,0000 | 0,0000 | **0,1884** | **UFM nhỉnh hơn TF-IDF (+0,0019)** |
| **Extreme Cold (1–5)** | **0,1916** | 0,0412 | 0,0389 | 0,1133 | *Sụt giảm do nhiễu tham số ID (−0,0783)* |
| **Cold (6–20)** | **0,1958** | 0,1124 | 0,1082 | 0,1895 | *Gần tương đương TF-IDF (−0,0063)* |
| **Warm (>20)** | 0,4677 | 0,5120 | 0,4980 | **0,6512** | **UFM vượt trội rõ rệt (+0,1836)** |
| **Cold Macro NDCG@10** | **0,1913** | 0,0512 | 0,0490 | 0,1637 | *Thấp hơn TF-IDF −0,0275 (CI 95%: [−0,0302; −0,0251])* |
| **Overall NDCG@10** | 0,2604 | 0,1664 | 0,1613 | **0,2856** | **UFM đạt điểm tổng thể cao nhất (+0,0252)** |

*(Lưu ý: Kết quả UFM từ lượt huấn luyện `ufm_full_v1` một seed `42` trên Validation; tập Test độc lập vẫn đang khóa kín)*

---

### SLIDE 8: PHÂN TÍCH CHUYÊN SÂU — HIỆN TƯỢNG "COLD-START GAP"
- **Tại sao UFM vượt trội ở Warm (+0,1836) và Zero-shot (+0,0019), nhưng lại hụt ở Extreme Cold (−0,0783)?**
  + Ở **Zero-shot ($count_{train}=0$)**: Nhánh CF bị khóa hoàn toàn ($\mathbb{I}_{cf}=0 \implies w_{cf}=0$), mô hình dùng 100% nhánh Semantic CLIP nên hoạt động tốt.
  + Ở **Extreme Cold ($1 \le count_{train} \le 5$)**: Item bắt đầu có ID embedding ($\mathbb{I}_{cf}=1$) nhưng dữ liệu quá thưa thớt $\implies$ vector ID bị **overfitting nặng**.
  + Cổng $w_{cf}$ mở ra nhận giá trị $> 0$ mang theo nhiễu của nhánh CF, làm suy giảm chất lượng của nhánh Semantic (trên nhóm có lịch sử: UFM đạt 0,2159 so với 0,3651 của TF-IDF).
- **Giải pháp đã cài đặt trong mã nguồn (`count_aware_fusion`)**:
  + Bổ sung hàm phạt tần suất vào độ bất định CF:
    $$u_{cf} \leftarrow u_{cf} + \exp(-count_{train} / 5.0)$$
  + Ép cổng ưu tiên nhánh Semantic khi sản phẩm có dưới 5 tương tác.

---

### SLIDE 9: HIỆU CHUẨN XÁC SUẤT DỰ ĐOÁN (CALIBRATION AUDIT)
- **Vấn đề**: Logit đầu ra của mạng xếp hạng chưa được hiệu chuẩn thành phân phối xác suất đáng tin cậy trên tập ứng viên.
- **Phương pháp**: Fit **Temperature Scaling** ($T = 1,789353$) trên 20.000 ca validation sớm, kiểm toán độc lập trên 20.000 ca validation muộn:
  + **ECE (Expected Calibration Error)** giảm từ **0,164477 xuống 0,041497 (giảm 74,8%)**.
  + **NLL** giảm từ **4,309607 xuống 3,906583**.
  + **Multiclass Brier Score** giảm từ **0,963379 xuống 0,927849**.
- *Lưu ý*: Calibration giữ nguyên thứ hạng Top-K (không thay đổi NDCG@10) và là xác suất có điều kiện trên 100 ứng viên lấy mẫu.

---

### SLIDE 10: BẰNG CHỨNG TÀI NGUYÊN & HIỆU NĂNG VẬN HÀNH
- **Huấn luyện & Bộ nhớ trên FITLAB** (`reports/campaign_snapshot_20261003.json`):
  + Lượt train `ufm_full_v1` (12 epochs, 198.232 steps, AMP) hoàn tất trong **4,13 giờ** (14.866s).
  + Bộ nhớ container (`cgroup memory.current`) duy trì ở **~9,37 GiB**, nằm an toàn trong ngân sách **< 16 GiB** với `0` sự kiện OOM.
- **Độ trễ phục vụ thực tế (Full-catalog 767.045 items trên CPU)**:
  + Nghiệm thu 4 ca API thật trên FITLAB (`reports/demo_acceptance_20261003.json`): **2,89s – 7,27s / truy vấn**; qua Web Proxy Zero-shot đạt **4,47s**.
- **Đo kiểm chuẩn cục bộ (`reports/benchmark_resources_content.json`)**:
  + Với bộ catalog mẫu nhẹ (30 items, TF-IDF): RAM sau nạp **53,6 MiB**, độ trễ $p50 < 1\text{ ms}$, băng thông $> 1.000\text{ QPS}$.

---

### SLIDE 11: DEMO TRỰC QUAN HỆ THỐNG
*(Trình diễn Demo Web trực tiếp hoặc qua ảnh bằng chứng nghiệm thu)*

- **Giao diện người dùng**:
  + Thiết kế **Pro Max Dark Glassmorphism**, trực quan hóa sản phẩm có ảnh thật từ Amazon CDN, lọc theo từng nhóm `Zero-shot`, `Extreme Cold`, `Cold`, `Warm`.
  + Hiển thị trực tiếp trọng số cổng `CF % · Semantic %` và `Uncertainty proxy` cho từng sản phẩm gợi ý.
- **Nghiệm thu thực tế & Khả năng tái lập**:
  + Đã vượt qua **4/4 ca kiểm thử API thật** trên FITLAB (`new_user`, `personalized`, `zero_shot`, `warm`) và xuất JSON kiểm chứng.
  + Hỗ trợ tự động tạo bộ dữ liệu mẫu offline (`scripts/build_mock_demo_data.py`) khi chạy trên máy mới chưa tải dataset 10GB.

---

### SLIDE 12: GIỚI HẠN HIỆN TẠI & LỘ TRÌNH HOÀN THIỆN CUỐI CÙNG
1. **Minh bạch trạng thái thực nghiệm hiện tại**:
   - Số liệu UFM-Rec trong báo cáo hiện lấy từ **1 seed (`seed=42`) trên tập Validation**; tập **Test độc lập vẫn chưa mở (`test_evaluated = false`)** để tránh rò rỉ lựa chọn mô hình.
   - Đại lượng Uncertainty là *learned reliability proxy*, không phải hậu nghiệm Bayes.
2. **Các hàng đợi thực nghiệm đang chạy trên FITLAB**:
   - **7 biến thể Ablation** (`no_uncertainty`, `fixed_fusion`, `no_cross_align`, `semantic_only`, `collaborative_only`, `text_only`, `image_only`).
   - **11 lượt chạy bổ sung trên Toys & Games**: SASRec, BERT4Rec thích nghi, CLIP Concat (3 seeds `42, 7, 2026`) và 2 seeds UFM (`7, 2026`).
   - **13 công việc trên tập All Beauty** (CLIP + 4 mô hình × 3 seeds, đã khóa SHA-256 tại `reports/second_dataset_plan_v1.json`).
3. **Bước chốt cuối**: Thu hoạch kết quả GPU $\rightarrow$ Khóa manifest SHA-256 $\rightarrow$ Mở đánh giá Test Set đúng 1 lần duy nhất.

---

### SLIDE 13: KẾT LUẬN & CÂU HỎI PHẢN BIỆN DỰ KIẾN
- **Câu 1: Tại sao đóng băng (Freeze) CLIP ViT-B/32 mà không Fine-tune toàn bộ?**
  * *Trả lời*: Với catalog 767.045 items, đóng băng CLIP và huấn luyện Semantic Adapter (chiếu về 128 chiều) giúp giữ toàn bộ tiến trình huấn luyện nằm gọn trong giới hạn bộ nhớ < 16 GiB, hoàn tất 12 epochs trong ~4,1 giờ và bảo toàn không gian ngữ nghĩa cho sản phẩm Zero-shot chưa từng xuất hiện trong tập train.
- **Câu 2: Tại sao mô hình UFM lại thấp hơn TF-IDF ở nhóm Extreme Cold và nhóm giải quyết thế nào?**
  * *Trả lời*: Khi sản phẩm chỉ có 1–5 tương tác, ID embedding bắt đầu mở cổng ($w_{cf} > 0$) nhưng vector ID bị overfitting nặng do quá ít mẫu, gây nhiễu lên nhánh Semantic. Nhóm đã kiểm toán rõ bằng paired bootstrap và cài đặt biến thể `count_aware_fusion` cộng phạt $\exp(-count_{train}/5)$ vào $u_{cf}$ để triệt tiêu nhiễu ID ở vùng tương tác thấp.
- **Câu 3: Tại sao chưa báo cáo kết quả trên tập Test?**
  * *Trả lời*: Nhóm tuân thủ nghiêm ngặt quy tắc *No Test Snooping*: tập Validation đang được dùng để chẩn đoán lỗi Extreme Cold và chờ hoàn tất hàng đợi Ablation / Multi-seed. Tập Test chỉ được chạy đúng 1 lần sau khi khóa hoàn toàn cấu hình và trọng số trong manifest.

---

### SLIDE 14: LỜI CẢM ƠN
- Nhóm thực hiện xin chân thành cảm ơn **ThS. Nguyễn Thái Anh** và Quý Thầy Cô trong Hội đồng đã lắng nghe!
- Kính mời Quý Thầy Cô đặt câu hỏi và đóng góp ý kiến cho đề tài.
