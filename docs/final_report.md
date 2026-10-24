# BÁO CÁO TỔNG KẾT ĐỒ ÁN NGHIÊN CỨU & PHÁT TRIỂN

## UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được

- **Học phần**: Học máy ứng dụng
- **Giảng viên hướng dẫn**: ThS. Nguyễn Thái Anh
- **Nhóm thực hiện**: Trần An Kỳ, Nguyễn Thành Phong, Hải, Hân
- **Thời gian cập nhật**: Tháng 10 / 2026

---

### TÓM TẮT ĐỀ TÀI (ABSTRACT)

Bài toán khởi đầu lạnh (Cold-Start) và đặc biệt là cực lạnh (Extreme Cold-Start) là một trong những thách thức cốt lõi và dai dẳng nhất của các hệ thống gợi ý hiện đại (Recommender Systems). Khi một sản phẩm mới xuất hiện trên sàn thương mại điện tử hoặc một người dùng mới gia nhập nền tảng, các phương pháp lọc cộng tác truyền thống (Collaborative Filtering - CF) và mô hình chuỗi hành vi (Sequential Recommenders như SASRec, BERT4Rec) hoàn toàn mất tác dụng do thiếu hụt ma trận tương tác.

Để giải quyết bài toán này, đề tài xây dựng và thực nghiệm kiến trúc **UFM-Rec (Uncertainty-aware Fusion Model Recommender)**. Mô hình kết hợp biểu diễn đa phương thức đóng băng (Frozen Multimodal Features) từ **CLIP ViT-B/32** (bao gồm văn bản tiêu đề, mô tả và hình ảnh trực quan sản phẩm) với biểu diễn tuần tự (Causal Sequential Transformer) của lịch sử tương tác. Đóng góp thiết kế của UFM-Rec là việc tích hợp **Cơ chế Ước Lượng Độ Tin Cậy Học Được (Learned Reliability Proxy)** và **Cổng Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate)**, cho phép mạng tự động điều tiết trọng số giữa ngữ nghĩa nội dung (Semantic Branch) và hành vi cộng tác (Collaborative Branch) tùy theo mức độ khả dụng và sai số ước lượng của dữ liệu.

Nghiên cứu được triển khai thực nghiệm trên tập dữ liệu chuẩn quy mô lớn **Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, 11,57 triệu tương tác)** theo chiến lược phân chia thời gian thực tế **80/10/10 Global Temporal Split**. Trên tập Validation 40.000 ca kiểm thử (`seed=42`, chưa mở tập Test), UFM-Rec đạt **Overall NDCG@10 = 0,285611** (vượt TF-IDF 0,260367 nhờ ưu thế lớn ở nhóm Warm đạt 0,651231), nhưng **Cold Macro NDCG@10 đạt 0,163738** (thấp hơn TF-IDF 0,191270 do hiện tượng nhiễu tham số ID ở phân tầng Extreme Cold 1–5 tương tác). Báo cáo phân tích sâu nguyên nhân toán học của hiện tượng này và kết quả hiệu chuẩn xác suất với Temperature Scaling (ECE từ 0,164477 xuống 0,041497). Trên All Beauty hiện có kết quả TF-IDF validation sơ bộ, chưa có kết quả UFM. UFM trên Toys vẫn là một seed và validation-only; các kết quả ablation, đa seed, benchmark All Beauty, đánh giá test và đo tài nguyên trên catalog đầy đủ chưa có trong bộ bằng chứng hiện lưu.

---

### MỤC LỤC

1. [CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ](#chương-1-giới-thiệu--đặt-vấn-đề)
2. [CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH UFM-REC](#chương-2-kiến-trúc-mô-hình-ufm-rec)
3. [CHƯƠNG 3: THIẾT LẬP THỰC NGHIỆM & GIAO THỨC ĐÁNH GIÁ](#chương-3-thiết-lập-thực-nghiệm--giao-thức-đánh-giá)
4. [CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM, TÀI NGUYÊN & GIỚI HẠN NGHIÊN CỨU](#chương-4-kết-quả-thực-nghiệm-tài-nguyên--giới-hạn-nghiên-cứu)
5. [CHƯƠNG 5: CÁC ĐỀ XUẤT TỐI ƯU & HƯỚNG MỞ RỘNG](#chương-5-các-đề-xuất-tối-ưu--hướng-mở-rộng)
6. [CHƯƠNG 6: KẾT LUẬN](#chương-6-kết-luận)

---

### CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ

#### 1.1. Thách thức Cold-Start trong Thương mại Điện tử
Trên các nền tảng thương mại điện tử quy mô lớn như Amazon, hàng triệu sản phẩm mới được niêm yết mỗi tháng. Các sản phẩm này rơi vào các phân tầng lịch sử (Regimes) khác nhau tính theo số lượt xuất hiện trong tập huấn luyện ($count_{train}$):
- **Zero-shot**: Sản phẩm chưa từng có bất kỳ tương tác nào trong tập huấn luyện ($count_{train} = 0$).
- **Extreme Cold**: Sản phẩm chỉ có từ 1 đến 5 tương tác ($1 \le count_{train} \le 5$).
- **Cold**: Sản phẩm có từ 6 đến 20 tương tác ($6 \le count_{train} \le 20$).
- **Warm**: Sản phẩm phổ biến, có trên 20 tương tác ($count_{train} > 20$).

Các mô hình CF cổ điển (Matrix Factorization, BPR-MF, SVD) không có embedding đã học cho item zero-shot và rất dễ bị nhiễu trên các item ít tương tác, dẫn đến sai lệch khuyến nghị nghiêm trọng.

#### 1.2. Hạn chế của các tiếp cận hiện tại
1. **Tiếp cận Content-based thuần túy (TF-IDF, BM25)**: Bắt được từ khóa tốt nhưng bỏ qua ngữ cảnh hình ảnh, không học được chuỗi sở thích người dùng theo thời gian và kém linh hoạt ở nhóm warm.
2. **Tiếp cận Ghép nối đơn giản (Late Fusion / Concat)**: Ghép vector ID với vector CLIP rồi đưa qua MLP thường khiến nhánh ID chiếm ưu thế (dominance), làm mô hình bị suy thoái khi gặp item hiếm.
3. **Thiếu khả năng tự lượng giá độ tin cậy theo từng nhánh (Branch Reliability Awareness)**: Mô hình không phân biệt được khi nào nhánh ID đáng tin cậy và khi nào cần dựa hoàn toàn vào nhánh nội dung đa phương thức.

---

### CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH UFM-REC

Mô hình UFM-Rec được thiết kế gồm 4 khối thành phần chính:

```
[Candidate Item]                 [User Interaction History]
  │       │                                │          │
Text    Image                            Text       Image
  │       │                                │          │
  └───┬───┘                                └───┬──────┘
      │ (CLIP ViT-B/32 Frozen)                 │
      ▼                                        ▼
[Semantic Adapter]                      [Semantic Adapter]
      │                                        │
      │ ─── Item Semantic Vector (cs)          │ ─── User Semantic Profile (sem_user)
      │                                        │
      │                                [Causal Transformer]
      │                                        │
      │ (Train Count Lookup)                   ▼
      ▼                                [User Sequential Vector] (cf_user)
[Collaborative Embedding]                      │
  Item CF Vector (cc)                          │
      │                                        │
      ├───────────────────┬────────────────────┤
      ▼                   ▼                    ▼
[CF Interaction]   [Semantic Inter.]   [Cross-Modal Alignment Gate]
    (cf_pair)          (sem_pair)                   │
      │                   │                         │
  Uncertainty         Uncertainty                   │
    $u_{cf}$            $u_{sem}$                   │
      │                   │                         │
      └─────────┬─────────┘                         │
                ▼                                   │
      [Reliability Weights]                         │
       $w_{cf}$   $w_{sem}$                         │
                │                                   │
                ▼                                   │
      [Confidence-Weighted Fusion] ◄────────────────┘
                │
                ▼
      [Recommendation Head] ──► Score $\hat{y}$
```

#### 2.1. Nhánh Biểu Diễn Ngữ Nghĩa (Semantic Branch)
- Đặc trưng trích xuất từ **CLIP ViT-B/32** (khóa cố định commit revision): vector văn bản $e_{text} \in \mathbb{R}^{512}$ và hình ảnh $e_{img} \in \mathbb{R}^{512}$ được chuẩn hóa $\ell_2$.
- Khi một phương thức bị thiếu (ví dụ sản phẩm không có ảnh), vector đó được zero-fill trước khi qua adapter và được đánh dấu bằng cờ nhị phân $m \in \{0, 1\}^2$ thay vì chèn nhiễu ngẫu nhiên.
- Bộ điều hợp Semantic Adapter đưa đặc trưng về không gian tiềm ẩn $d = 128$:
  $$v_{sem} = \text{LayerNorm}(\text{Linear}(\text{GELU}(\text{Linear}([e_{text} \odot m_{text}, e_{img} \odot m_{img}, m]))))$$

#### 2.2. Nhánh Lịch Sử Chuỗi Hành Vi (Collaborative Branch)
- Chuỗi $H = [i_1, i_2, \dots, i_T]$ (với $T = 20$) được đưa qua mạng **Causal Transformer Encoder** 2 lớp, 4 attention heads với mặt nạ nhân quả (causal mask) và đệm phải (right padding).
- Điểm mấu chốt: Các item ID chưa từng xuất hiện trong tập Train ($count_{train} = 0$) được gán trực tiếp về chỉ số Padding ($idx=0$) **trước khi lookup** để loại trừ hoàn toàn nguy cơ rò rỉ trọng số hoặc phát sinh gradient bất định.

#### 2.3. Cổng Ước Lượng Độ Tin Cậy (Uncertainty-aware Fusion Gate)
Thay vì dùng cổng tĩnh hoặc softmax tự do, UFM-Rec định nghĩa đại lượng proxy độ bất định học được qua hàm kích hoạt dương:
$$u_{cf} = \text{Softplus}(\text{MLP}_{cf}(h_{cf})), \quad u_{sem} = \text{Softplus}(\text{MLP}_{sem}(h_{sem}))$$

Trong đó mỗi nhánh học khớp phần dư bình phương có điều kiện trên tập ứng viên lấy mẫu ($\text{MSE}(u_k, (\sigma(s_k)_{\text{detached}} - y)^2)$). Trọng số độ tin cậy được tính theo công thức:
$$w_k = \frac{\exp(-u_k)}{\sum_{j \in \{cf, sem\}} \exp(-u_j)} \cdot \mathbb{I}(\text{branch } k \text{ available})$$

Khi candidate là item zero-shot ($count_{train}=0$), nhánh CF không khả dụng ($\mathbb{I}_{cf}=0$), cổng tự động triệt tiêu $w_{cf} = 0$ và chuyển toàn bộ 100% trọng số sang $w_{sem} = 1,0$.

#### 2.4. Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate)
Khi cả hai nhánh đều khả dụng, thành phần Alignment đóng vai trò cầu nối liên phương thức:
$$g_{align} = \sigma(\text{MLP}_{gate}([h_{cf}, h_{sem}])) \odot \text{Linear}([h_{cf}, h_{sem}]) \cdot \mathbb{I}_{cf \wedge sem}$$
Vector tổng hợp cuối cùng:
$$h_{fused} = w_{cf} h_{cf} + w_{sem} h_{sem} + g_{align}$$
$$\hat{y} = \text{MLP}_{out}(h_{fused})$$

---

### CHƯƠNG 3: THIẾT LẬP THỰC NGHIỆM & GIAO THỨC ĐÁNH GIÁ

#### 3.1. Tập Dữ Liệu
Nghiên cứu sử dụng tập dữ liệu thực tế quy mô lớn từ Amazon Reviews 2023:
- Danh mục chính: **Toys & Games** ([`data/processed/toys_games_full_temporal/split_manifest.json`](../data/processed/toys_games_full_temporal/split_manifest.json))
- Tổng số sản phẩm (Items): **767.045**
- Số sản phẩm có ảnh hợp lệ trích xuất thành công: **758.692** (98,91%)
- Tổng số tương tác sau làm sạch: **11.572.689**
- Chiến lược phân chia: **Global Temporal Split (80% Train, 10% Validation, 10% Test)** theo mốc thời gian thực để ngăn chặn rò rỉ tương lai (Data Leakage).
- Danh mục thứ hai (chuẩn bị cho kiểm chứng đa miền): **All Beauty** gồm **86.014 sản phẩm, 445.239 tương tác** ([`reports/second_dataset_preparation.json`](../reports/second_dataset_preparation.json)).

#### 3.2. Giao thức Đánh giá Phân tầng
Để phản ánh năng lực xếp hạng theo từng mức độ phổ biến của sản phẩm, tập Validation gồm **40.000 trường hợp** kiểm thử độc lập (mỗi ca gồm 1 sản phẩm Ground Truth + 99 sản phẩm âm tính được lấy mẫu theo phân phối tần suất xuất hiện trong tập train — *Popularity Negative Sampling*, không phải đánh giá full-catalog).
Các trường hợp được phân bổ cân bằng 10.000 ca cho mỗi nhóm:
1. `zero_shot`: Sản phẩm đích có $count_{train} = 0$.
2. `extreme_cold`: Sản phẩm đích có $1 \le count_{train} \le 5$.
3. `cold`: Sản phẩm đích có $6 \le count_{train} \le 20$.
4. `warm`: Sản phẩm đích có $count_{train} > 20$.

Metric chính: **NDCG@10**, **Recall@10**, **MRR@10** và **Cold Macro NDCG@10** (trung bình cộng của Zero-shot, Extreme Cold và Cold).

---

### CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM, TÀI NGUYÊN & GIỚI HẠN NGHIÊN CỨU

#### 4.1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Validation Set — Single Seed `seed=42`)

> **Lưu ý phương pháp luận**: Số liệu UFM-Rec dưới đây được trích xuất từ báo cáo kiểm toán [`reports/ufm_validation_audit_v2/report.md`](../reports/ufm_validation_audit_v2/report.md) trên tập Validation (40.000 ca, 100 ứng viên/ca, `seed=42`, best epoch 9/12, `test_evaluated = false`).

| Nhóm Phân Tầng (Regime) | Số Mẫu | TF-IDF Content | BPR-MF (3 seeds) | Collaborative SVD | **UFM-Rec (`seed=42`)** | Chênh lệch (UFM vs TF-IDF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | 10.000 | 0,186491 | 0,000000 | 0,000000 | **0,188403** | **+0,001912** |
| *· Zero-shot (có lịch sử)* | 5.129 | 0,363602 | 0,000000 | 0,000000 | **0,367329** | **+0,003727** |
| *· Zero-shot (không lịch sử)* | 4.871 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 |
| **Extreme Cold (1–5)** | 10.000 | **0,191555** | 0,041200 | 0,038900 | 0,113300 | −0,078255 |
| *· Extreme Cold (có lịch sử)* | 5.247 | **0,365075** | 0,078500 | 0,074100 | 0,215933 | −0,149142 |
| **Cold (6–20)** | 10.000 | **0,195763** | 0,112400 | 0,108200 | 0,189511 | −0,006252 |
| **Warm (>20)** | 10.000 | 0,467661 | 0,512000 | 0,498000 | **0,651231** | **+0,183570** |
| **Cold Macro NDCG@10** | 30.000 | **0,191270** | 0,051200 | 0,049033 | 0,163738 | **−0,027532** |
| **Overall NDCG@10** | 40.000 | 0,260367 | 0,166400 | 0,161275 | **0,285611** | **+0,025244** |

#### 4.2. Phân Tích Hiện Tượng "Cold-Start Gap"
1. **Ưu thế rõ rệt ở nhóm Warm**: UFM-Rec vượt xa các mô hình cơ sở ở nhóm Warm (NDCG@10 đạt **0,651231** so với 0,467661 của TF-IDF). Điều này cho thấy sự kết hợp giữa Transformer tuần tự và CLIP biểu diễn ngữ nghĩa khai thác rất hiệu quả lịch sử tương tác khi sản phẩm đã có đủ dữ liệu huấn luyện.
2. **UFM chưa đạt mục tiêu vượt TF-IDF trên Cold Macro (hiện tượng sụt giảm ở Extreme Cold)**:
   - Khoảng chênh lệch Cold Macro giữa UFM và TF-IDF là $−0,027532$ (khoảng tin cậy paired bootstrap 95% theo user trên 1.000 lần lấy mẫu lại là $[−0,030221; −0,025082]$ — xem [`reports/ufm_validation_audit_v2/report.md`](../reports/ufm_validation_audit_v2/report.md)).
   - *Nguyên nhân cốt lõi*: Ở nhóm Zero-shot ($count_{train}=0$), nhánh CF bị khóa hoàn toàn ($w_{cf}=0, w_{sem}=1$) nên UFM đạt kết quả nhỉnh hơn TF-IDF (+0,001912 overall và +0,003727 trên nhóm có lịch sử). Tuy nhiên, khi một item có từ 1 đến 5 lượt xuất hiện (Extreme Cold), mạng Collaborative Embedding bắt đầu kích hoạt ($\mathbb{I}_{cf}=1$), dẫn đến cổng $w_{cf}$ nhận giá trị lớn hơn 0. Do số mẫu quá ít, vector ID bị overfitting nặng (nhiễu tham số), làm suy giảm điểm số dung hợp so với nhánh Semantic thuần túy (0,215933 so với 0,365075 của TF-IDF trên tập có lịch sử).

#### 4.3. Cân Chỉnh Xác Suất Độ Tin Cậy (Confidence Calibration)
Nhằm kiểm tra chất lượng hiệu chuẩn xác suất Top-1 trên tập 100 ứng viên lấy mẫu, phương pháp **Temperature Scaling** được fit trên 20.000 ca validation sớm và kiểm toán độc lập trên 20.000 ca validation muộn ([`reports/ufm_validation_audit_v2/report.json`](../reports/ufm_validation_audit_v2/report.json)):
- Nhiệt độ tối ưu học được: $T = 1,789353$
- Kết quả kiểm chứng trên 20.000 ca muộn:
  * **ECE (Expected Calibration Error)** giảm mạnh từ **0,164477 xuống 0,041497** (giảm 74,8%).
  * **NLL** giảm từ **4,309607 xuống 3,906583**.
  * **Multiclass Brier Score** giảm từ **0,963379 xuống 0,927849**.
- *Lưu ý*: Phép chia nhiệt độ $T > 0$ là phép biến đổi đơn điệu nên giữ nguyên thứ tự xếp hạng (không làm thay đổi NDCG@10) và xác suất này là xác suất có điều kiện trên tập 100 ứng viên lấy mẫu, không phải xác suất mua hàng ngoài đời thực.

#### 4.4. Đánh Giá Tài Nguyên & Hiệu Năng Vận Hành
Các bằng chứng vận hành thực tế đã được ghi nhận và lưu trữ trong kho báo cáo:
1. **Tài nguyên huấn luyện & vận hành trên máy chủ FITLAB** ([`reports/campaign_snapshot_20261003.json`](../reports/campaign_snapshot_20261003.json)):
   - Lượt huấn luyện `ufm_full_v1` (12 epochs, 198.232 steps với AMP mixed precision) hoàn tất trong **14.866,28 giây (~4,13 giờ)**.
   - Bộ nhớ hệ thống của container (`cgroup memory.current`) ghi nhận ở mức **10.059.272.192 bytes (~9,37 GiB)**, đảm bảo nằm dưới ngưỡng giới hạn **16 GiB** với **0 sự kiện tràn bộ nhớ** (`oom: 0`, `oom_kill: 0`).
2. **Độ trễ suy luận thực tế trên toàn bộ Catalog 767.045 sản phẩm** ([`reports/demo_acceptance_20261003.json`](../reports/demo_acceptance_20261003.json) & [`reports/demo_ui_export_20261003.json`](../reports/demo_ui_export_20261003.json)):
   - Khi phục vụ trên CPU (4 threads, duyệt theo từng chunk 1.024 items qua toàn bộ 767.045 sản phẩm mà không dùng chỉ mục xấp xỉ), thời gian phản hồi thực tế qua 4 ca kiểm thử nghiệm thu API đạt từ **2,89 giây đến 7,27 giây/truy vấn** (`new_user`: 2.892,0 ms; `zero_shot`: 3.014,4 ms; `warm`: 3.002,6 ms; `personalized`: 7.272,2 ms) và **4,47 giây** (4.468,4 ms) khi truy vấn lọc Zero-shot từ lịch sử LEGO qua giao diện Web Proxy.
3. **Đo kiểm chuẩn kịch bản với [`scripts/benchmark_resources.py`](../scripts/benchmark_resources.py)**:
   - Công cụ đo tự động (`RAM RSS`, `VRAM peak`, `latency p50/p95/p99` qua 30 lượt lặp và `throughput QPS`) đã được xây dựng tại [`scripts/benchmark_resources.py`](../scripts/benchmark_resources.py) và chạy kiểm chứng trên bộ dữ liệu mẫu cục bộ ([`reports/benchmark_resources_content.json`](../reports/benchmark_resources_content.json)). Bộ bằng chứng hiện chưa có lượt đo 30 chu kỳ trên catalog Toys 767.045 sản phẩm; số đo bộ mẫu không đại diện cho mức dùng tài nguyên hoặc độ trễ production.

#### 4.5. Kết quả baseline sơ bộ trên All Beauty
Tập thứ hai đã được chuẩn bị độc lập theo split thời gian 80/10/10. Manifest ghi nhận **445.239 tương tác dương**, trong đó 356.402 train, 44.396 validation và 44.441 test; catalog có **86.014 sản phẩm**. Test chưa được đánh giá. Kết quả dưới đây là baseline TF-IDF và popularity trên validation, không phải kết quả UFM hoặc đánh giá chuyển miền của mô hình đề xuất.

| Nhóm item | Số ca | Popularity NDCG@10 | TF-IDF NDCG@10 | Popularity Recall@10 | TF-IDF Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 10.000 | 0,000000 | 0,027645 | 0,000000 | 0,039500 |
| Extreme cold | 10.000 | 0,001102 | 0,020921 | 0,003700 | 0,030300 |
| Cold | 7.381 | 0,246404 | 0,246271 | 0,673621 | 0,652486 |
| Warm | 10.000 | 0,770103 | 0,742460 | 1,000000 | 0,967500 |
| **Overall** | **37.381** | **0,254963** | **0,260239** | **0,401514** | **0,406329** |

TF-IDF tăng overall NDCG@10 thêm 0,005276 so với popularity; mức tăng tập trung ở zero-shot và extreme-cold, trong khi cold gần như ngang nhau và warm thấp hơn popularity. Chỉ **2.627/37.381** ca validation có history, còn 34.754 ca dùng history rỗng; trên nhóm có history, TF-IDF đạt NDCG@10 0,278014 so với popularity 0,202934. Đây là baseline tham chiếu hữu ích cho các lượt chạy tiếp theo, nhưng chưa đủ để kết luận UFM khái quát sang miền All Beauty.

Giao thức lấy mẫu All Beauty có 100 ứng viên/ca, lấy negative uniform không hoàn lại; nhóm cold gồm toàn bộ 7.381 ca sẵn có, còn các nhóm khác lấy tối đa 10.000 ca. Giao thức này khác Toys (40.000 ca cân bằng và popularity negative sampling), vì vậy không so sánh trực tiếp metric giữa hai dataset. Nguồn: [`sampling_report.json`](../data/processed/all_beauty_full_temporal/evaluation/sampling_report.json), [`metrics.json`](../data/processed/all_beauty_full_temporal/models/content_baseline/metrics.json), [`split_manifest.json`](../data/processed/all_beauty_full_temporal/split_manifest.json).

#### 4.6. Giới Hạn Nghiên Cứu & Trạng Thái Thực Nghiệm Hiện Tại (Limitations & Threats to Validity)
Để đảm bảo tính trung thực và minh bạch học thuật, báo cáo công bố rõ các giới hạn hiện tại của thực nghiệm tính đến thời điểm lập báo cáo:
1. **Kết quả UFM hiện tại dựa trên một hạt giống (`seed=42`) và tập Validation**:
   - Bảng so sánh chính của UFM-Rec hiện được trích từ lượt chạy `ufm_full_v1` (`seed=42`) trên 40.000 ca validation (nơi tập validation cũng được sử dụng để chọn best epoch 9/12 và fit nhiệt độ $T$). Khoảng tin cậy bootstrap 95% theo user phản ánh độ biến thiên lấy mẫu người dùng trên tập validation, không thay thế cho độ lệch chuẩn đa hạt giống (`mean ± std` qua nhiều random seeds).
2. **Tập Test độc lập chưa được mở (`test_evaluated = false`)**:
   - Theo đúng giao thức chống rò rỉ kết quả kiểm thử trong [`docs/followup.md`](../docs/followup.md), tập Test (10% cuối theo thời gian) của UFM-Rec vẫn được khóa hoàn toàn và chưa chạy đánh giá. Tập Test chỉ được mở đúng một lần duy nhất sau khi khóa manifest (gồm SHA-256 của checkpoint, cấu hình và nhiệt độ hiệu chuẩn $T$).
3. **Kết quả ablation, đa hạt giống và benchmark UFM trên All Beauty chưa có trong bộ bằng chứng hiện lưu**:
   - Snapshot FITLAB gần nhất có trong repo là ngày 03/10/2026. Tại snapshot đó, **7 biến thể Ablation** (`no_uncertainty`, `fixed_fusion`, `no_cross_align`, `semantic_only`, `collaborative_only`, `text_only`, `image_only`), **11 lượt chạy bổ sung trên Toys & Games** (SASRec, BERT4Rec thích nghi, CLIP Concat với 3 seeds `42, 7, 2026` và 2 seeds UFM bổ sung `7, 2026`), cùng **13 công việc All Beauty** (trích xuất CLIP + 4 mô hình × 3 seeds; kế hoạch tại [`reports/second_dataset_plan_v1.json`](../reports/second_dataset_plan_v1.json)) đã được gửi vào queue. Repo không chứa snapshot FITLAB mới hơn hoặc thư mục `runs`; vì vậy không thể suy ra các job hiện còn chạy, đã hoàn tất hay thất bại. Chỉ cập nhật trạng thái sau khi đọc heartbeat và marker trên FITLAB; riêng All Beauty hiện có TF-IDF baseline validation ở mục 4.5, chưa có CLIP/UFM comparison.
4. **Phạm vi của đại lượng Uncertainty**:
   - Đại lượng $u_{cf}, u_{sem}$ là *learned reliability proxy* phục vụ điều tiết cổng dung hợp, không phải ước lượng hậu nghiệm Bayes (Bayesian posterior) hay tách biệt bất định nhận thức/ngẫu nhiên (epistemic/aleatoric).

---

### CHƯƠNG 5: CÁC ĐỀ XUẤT TỐI ƯU & HƯỚNG MỞ RỘNG

1. **Khắc phục Extreme Cold với `count_aware_fusion` và `adaptive_dropout` (Đã triển khai trong [`src/ufm_model.py`](../src/ufm_model.py))**:
   - Thay vì để mạng tự do học $u_{cf}$ trên các item hiếm, áp dụng hàm phạt cưỡng bức theo tần suất xuất hiện trong tập train:
     $$u_{cf} \leftarrow u_{cf} + \exp\left(-\frac{count_{train}}{5.0}\right)$$
   - Kết hợp cùng tần suất dropout thích nghi $p_{\text{drop}}(i) = p_0 \cdot \exp(-count_{train}(i)/10.0)$, cơ chế này ép mô hình giảm trọng số Collaborative và ưu tiên nhánh Semantic CLIP khi sản phẩm có từ 1–5 tương tác. Biến thể này đã được cài đặt và kiểm thử đơn vị trong `UFMRec` (`variant="count_aware_fusion"`), sẵn sàng đưa vào chạy thực nghiệm đối chứng trên GPU.
2. **Tăng tốc độ truy vấn phục vụ Production với FAISS Vector Index (Two-stage Retrieval)**:
   - Hiện tại Demo UFM quét toàn bộ danh mục 767.045 sản phẩm bằng ma trận dày trên CPU mất 2,89s – 7,27s. Trong phiên bản triển khai quy mô lớn, kiến trúc được tách thành 2 giai đoạn:
     * *Giai đoạn 1 (Candidate Retrieval)*: Dùng chỉ mục xấp xỉ FAISS HNSW / IVF trên vector người dùng và sản phẩm để trích xuất top 100–200 ứng viên tiềm năng trong $< 20\text{ ms}$.
     * *Giai đoạn 2 (UFM Reranking)*: Đưa 100–200 ứng viên qua cổng UFM-Rec Head để chấm điểm xếp hạng tinh vi và trích xuất trọng số giải thích $w_{cf}, w_{sem}$.
3. **Mở rộng và Kiểm chứng Đa miền (Cross-domain Generalization)**:
   - Hoàn tất chuỗi thực nghiệm chuyển giao sang tập dữ liệu thứ hai **All Beauty (86.014 items, 37.381 ca validation)** theo đúng kế hoạch đã đóng băng SHA-256 tại [`reports/second_dataset_plan_v1.json`](../reports/second_dataset_plan_v1.json).

---

### CHƯƠNG 6: KẾT LUẬN

Đề tài đã hoàn thành việc thiết kế, xây dựng pipeline dữ liệu thời gian thực quy mô lớn, huấn luyện mô hình **UFM-Rec** và kiểm toán kết quả chặt chẽ trên tập dữ liệu Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, 11,57 triệu tương tác). Những kết quả và đóng góp chính bao gồm:
1. Xây dựng hoàn chỉnh kiến trúc dung hợp đa phương thức kết hợp biểu diễn đóng băng **CLIP ViT-B/32** (98,91% độ phủ ảnh hợp lệ) với **Causal Sequential Transformer**, mang lại bước tiến lớn ở phân khúc Warm (**0,651231 NDCG@10**, +0,183570 so với TF-IDF) và nâng Overall NDCG@10 lên **0,285611**.
2. Kiểm toán minh bạch và lý giải bằng thực nghiệm nguyên nhân khiến Cold Macro của UFM (`0,163738`) còn thấp hơn TF-IDF (`0,191270`): sự suy giảm tập trung ở nhóm Extreme Cold (1–5 tương tác) khi nhánh ID bắt đầu mở cổng nhưng embedding bị nhiễu do thiếu mẫu, từ đó thiết kế giải pháp `count_aware_fusion` trong mã nguồn.
3. Cân chỉnh thành công xác suất dự đoán Top-1 trên tập ứng viên với Temperature Scaling ($T = 1,789353$), giúp giảm sai số **ECE từ 0,164477 xuống 0,041497** trên 20.000 ca kiểm định độc lập theo thời gian.
4. Xây dựng hệ thống phần mềm nghiên cứu có khả năng tái lập cao: kiểm tra toàn vẹn mã nguồn và dữ liệu bằng SHA-256, cơ chế tự phục hồi hàng đợi huấn luyện, giao diện Demo Web Pro Max Dark Glassmorphism vượt qua 4/4 ca kiểm thử API thực tế, cùng bộ kiểm thử tự động 14/14 test suites đạt chuẩn.
