# BÁO CÁO TỔNG KẾT ĐỒ ÁN NGHIÊN CỨU & PHÁT TRIỂN

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](../reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->

> **Phạm vi đã chốt với giảng viên theo xác nhận của nhóm:** hệ khuyến nghị sản phẩm dựa trên hành vi, hình ảnh và tên/mô tả. Cold-start sản phẩm là phân tích trọng tâm, không phải lời hứa cá nhân hóa người dùng chưa có lịch sử. Trong v3, full − semantic-only NDCG@10 theo train count là −0,002424 (0), −0,024101 (1–5), +0,002586 (6–20), +0,014865 (21+), mỗi nhóm 10.000 ca. Nguồn: [phân tích dự đoán v3](../reports/continuation_20261028/count_effect_v3.json). Đây là một seed lịch sử; [15 run đã khóa](../configs/multimodal_followup_20261028.json) chưa có kết quả train mới. Bản Word/PDF trước đây vẫn chưa đồng bộ.


Đối chiếu ngày **28/10/2026**: bộ sửa 1–15 trước đây đạt Linux **87 pass** / Windows **84 pass, 3 skip**. Mã bổ sung calibration/source loader đạt Linux **94 pass** / Windows **91 pass, 3 skip**; các bổ sung restore sau đó có kiểm thử riêng. Xem báo cáo sửa mục 1–15 và log để phân biệt bản mã hiện tại với bản mã đã tạo checkpoint.

## UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được

- **Học phần**: Học máy ứng dụng
- **Giảng viên hướng dẫn**: ThS. Nguyễn Thái Anh
- **Nhóm thực hiện**: Trần An Kỳ, Nguyễn Thành Phong, Hải, Hân
- **Thời gian cập nhật**: 28/10/2026 — bản phục vụ trao đổi tiến độ

---

### TÓM TẮT ĐỀ TÀI (ABSTRACT)

Bài toán khởi đầu lạnh (Cold-Start) và đặc biệt là cực lạnh (Extreme Cold-Start) là một trong những thách thức cốt lõi và dai dẳng nhất của các hệ thống gợi ý hiện đại (Recommender Systems). Khi một sản phẩm mới xuất hiện trên sàn thương mại điện tử hoặc một người dùng mới gia nhập nền tảng, các phương pháp lọc cộng tác truyền thống (Collaborative Filtering - CF) và mô hình chuỗi hành vi (Sequential Recommenders như SASRec, BERT4Rec) gặp hạn chế do thiếu tương tác để học biểu diễn định danh cho người dùng hoặc sản phẩm mới.

Để giải quyết bài toán này, đề tài xây dựng và thực nghiệm kiến trúc **UFM-Rec (Uncertainty-aware Fusion Model Recommender)**. Mô hình kết hợp biểu diễn đa phương thức đóng băng (Frozen Multimodal Features) từ **CLIP ViT-B/32** (bao gồm văn bản tiêu đề, mô tả và hình ảnh trực quan sản phẩm) với biểu diễn tuần tự (Causal Sequential Transformer) của lịch sử tương tác. Đóng góp thiết kế của UFM-Rec là việc tích hợp **Cơ chế Ước Lượng Độ Tin Cậy Học Được (Learned Reliability Proxy)** và **Cổng Căn Chỉnh Đa Phương Thức (Cross-Modal Alignment Gate)**, cho phép mạng tự động điều tiết trọng số giữa ngữ nghĩa nội dung (Semantic Branch) và hành vi cộng tác (Collaborative Branch) tùy theo mức độ khả dụng và sai số ước lượng của dữ liệu.

Nghiên cứu thực nghiệm trên Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, khoảng 11,57 triệu tương tác), chia thời gian 80/10/10. Vòng đánh giá lại ngày 28/10/2026 dùng validation v3 cố định: 40.000 ca, 1 positive + 99 negatives, loại positive đã biết trước target. Negative vẫn lấy theo regime của target và thống kê cuối split nên đánh giá có tính hồi cứu. Trên cùng v3, Cold macro NDCG@10 của TF-IDF là **0,195682**, semantic-only dùng CLIP **0,190036**, UFM full **0,182056**, popularity **0,042527**. Tám checkpoint UFM được đối chiếu cùng nguồn đặc trưng và candidates; chưa chứng minh fusion/uncertainty cải thiện chỉ số chính. Các checkpoint có một seed (42), chọn từ validation v1; đánh giá v3 không tạo holdout độc lập. Hiệu chỉnh thăm dò trên hai phần validation theo thời gian giảm ECE của full UFM từ **0,082061** xuống **0,037478**; đây là confidence có điều kiện trên 100 ứng viên, không phải xác suất mua. Test v2 đã được đánh giá trong lịch sử; vòng mới không mở thêm test. Chưa có kết quả đa seed UFM hoặc UFM trên All Beauty được xác minh. Số liệu lịch sử v1/v2 được giữ riêng trong Chương 4; nguồn hiện tại là [bộ bằng chứng ngày 28/10](../reports/continuation_20261028/README.md).

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
$$w_k = \frac{\mathbb{I}(k\in A)\exp(-u_k)}{\sum_{j\in A}\exp(-u_j)},\quad A=\{j:\text{branch }j\text{ available}\},\ A\ne\varnothing$$

Nếu cả hai nhánh không khả dụng, mô hình dùng fallback popularity train-only. Khi nhánh Semantic khả dụng và candidate là item zero-shot ($count_{train}=0$), nhánh CF không khả dụng ($\mathbb{I}_{cf}=0$), cổng tự động triệt tiêu $w_{cf} = 0$ và chuyển toàn bộ 100% trọng số sang $w_{sem} = 1,0$.

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
- Chiến lược phân chia: **Global Temporal Split (80% Train, 10% Validation, 10% Test)** theo mốc thời gian. Chia split theo thời gian không tự bảo đảm mọi bước lấy mẫu và metadata chỉ dùng thông tin có sẵn tại thời điểm dự đoán; giới hạn được nêu ở mục 3.2.
- Danh mục thứ hai (chuẩn bị cho kiểm chứng đa miền): **All Beauty** gồm **86.014 sản phẩm, 445.239 tương tác** ([`reports/second_dataset_preparation.json`](../reports/second_dataset_preparation.json)).

#### 3.2. Giao thức Đánh giá Phân tầng
Mỗi bộ mẫu Toys có 40.000 ca, mỗi ca gồm 1 target ở cột 0 và 99 negative duy nhất, được chia thành 10.000 ca cho mỗi regime theo **số tương tác train của target**: zero-shot = 0; extreme-cold = 1–5; cold = 6–20; warm >20. Các ca có thể thuộc cùng người dùng, vì vậy không gọi chúng là các quan sát độc lập. Cold macro là trung bình NDCG@10 của ba nhóm lạnh; Overall là trung bình trên bộ mẫu cân bằng, không đại diện trực tiếp cho phân bố lưu lượng tự nhiên.

**Cần phân biệt hai bộ ứng viên**:
- **Bộ lịch sử** dùng trong audit `ufm_validation_audit_v2`: hash mẫu validation `8efa116508fddd6b5227b69d6c65e3b635de634dd542046463fda84e78b47162`. Tên “audit v2” là phiên bản báo cáo kiểm toán, không có nghĩa nó dùng bộ mẫu mới `evaluation_v2`. Bảng 4.1 giữ các số liệu lịch sử cùng nguồn; chưa replay bộ lịch sử trong lần chỉnh tài liệu này.
- **Bộ mẫu `evaluation_v2`** nhập local ngày 27/10: hash validation `c71906e65ce4067b405a94c13e22e8a25a17f46cbee5b1315f07d01d4d1e59df`, test `a4422c2cfe420e7ffc072915b16ff88a218d640054fb107c937d867dca20b109`. Candidate audit báo replay và tính nhất quán regime đạt. Code hiện tại lấy 99 negative **cùng regime với target**, loại target và các item trong history được ghi nhận, lấy trọng số popularity bằng `train_count + validation_count` khi đánh giá validation, và cộng thêm `test_count` khi đánh giá test. Đây là benchmark hồi cứu có dùng thống kê đến cuối split để dựng tập ứng viên, **không phải negative sampling chỉ từ train** và không phải đánh giá full-catalog.

Việc dùng thống kê cuối split và biết regime của target để chọn negative giới hạn khả năng diễn giải như đánh giá online. History được ghi nhận có cửa sổ hữu hạn; audit cũ ghi 3 trường hợp negative trùng positive train ở validation và 7 ở test. Audit validation mới trên FITLAB-01 xét mọi sự kiện trước target, bao gồm cùng split, ghi 7 negative trong 6 ca của v2. Validation v3 tạo riêng đã loại các known positive và audit đạt; tám checkpoint UFM cùng popularity/TF-IDF đã được đánh giá trên đúng hash v3. Không diễn giải `passed=true` thành “không có false negative”. Mỗi bảng so sánh phải dùng cùng bộ ứng viên.

Metric báo cáo: NDCG@10, Recall@10, MRR@10; **Cold macro NDCG@10 là chỉ số chính**. Nguồn: [`src/build_recommender_samples.py`](../src/build_recommender_samples.py), [artifact và hash v2](../reports/ufm_evaluation_v2_20261027/README.md).

---

### CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM, TÀI NGUYÊN & GIỚI HẠN NGHIÊN CỨU

#### 4.1. Kết quả validation lịch sử và nguồn baseline

Bảng này ghi **NDCG@10 lịch sử**, không phải kết quả trên `evaluation_v2` mới. UFM: audit `ufm_validation_audit_v2/report.json`, seed 42, epoch 9/12. TF-IDF: `models/content_baseline/metrics.json`. BPR-MF: `models/bpr_mf/epoch_001.json`, seed 42, best epoch 1 theo `completed.json`; SVD: `models/collaborative_svd/report.json`, mục validation. Các giá trị BPR-MF/SVD trước đây không khớp artifact đã được thay bằng số liệu từ các nguồn này. Chưa kiểm tra lại đồng nhất bộ mẫu của mọi baseline bằng replay; bảng là đối chiếu artifact lưu, không phải benchmark mới chạy.

| Nhóm | Số ca | TF-IDF | BPR-MF (`seed=42`, epoch 1) | SVD | UFM (`seed=42`, epoch 9) | UFM − TF-IDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot (0) | 10.000 | 0,186491 | 0,000000 | 0,000000 | 0,188403 | 0,001912 |
| Extreme cold (1–5) | 10.000 | 0,191555 | 0,013367 | 0,036952 | 0,113300 | -0,078255 |
| Cold (6–20) | 10.000 | 0,195763 | 0,027292 | 0,055349 | 0,189511 | -0,006252 |
| Warm (>20) | 10.000 | 0,467661 | 0,501546 | 0,391065 | 0,651231 | 0,183570 |
| **Cold macro** | 30.000 | **0,191270** | 0,013553 | 0,030767 | 0,163738 | −0,027532 |
| **Overall** | 40.000 | 0,260367 | 0,135551 | 0,120842 | **0,285611** | +0,025244 |

BPR-MF ba seed `42, 7, 2026` được báo riêng, không trộn vào cột seed 42: Overall **0,135465 ± 0,000231**, Cold macro **0,013602 ± 0,000233** (mean ± sample std, `ddof=1`), nguồn [`BPR_MF_multiseed_2026_09_30.json`](../reports/BPR_MF_multiseed_2026_09_30.json). UFM chưa có độ lệch chuẩn qua nhiều seed.

#### 4.2. Phân Tích Hiện Tượng "Cold-Start Gap"
1. **Ưu thế rõ rệt ở nhóm Warm**: UFM-Rec vượt xa các mô hình cơ sở ở nhóm Warm (NDCG@10 đạt **0,651231** so với 0,467661 của TF-IDF). Đây là quan sát trên bộ mẫu lịch sử; đã có ablation một seed; chưa quy phần tăng điểm cho riêng từng thành phần hoặc suy rộng giữa các giao thức, và không suy rộng sang giao thức v2.
2. **UFM chưa đạt mục tiêu vượt TF-IDF trên Cold Macro (hiện tượng sụt giảm ở Extreme Cold)**:
   - Khoảng chênh lệch Cold Macro giữa UFM và TF-IDF là $−0,027532$ (khoảng tin cậy paired bootstrap 95% theo user trên 2.000 lần lấy mẫu lại là $[−0,030221; −0,025082]$ — xem [`reports/ufm_validation_audit_v2/report.md`](../reports/ufm_validation_audit_v2/report.md)).
   - *Giả thuyết giải thích (chưa được chứng minh bằng thực nghiệm loại trừ)*: Ở nhóm Zero-shot ($count_{train}=0$), nhánh CF bị khóa hoàn toàn ($w_{cf}=0, w_{sem}=1$) nên UFM đạt kết quả nhỉnh hơn TF-IDF (+0,001912 overall và +0,003727 trên nhóm có lịch sử). Tuy nhiên, khi một item có từ 1 đến 5 lượt xuất hiện (Extreme Cold), mạng Collaborative Embedding bắt đầu kích hoạt ($\mathbb{I}_{cf}=1$), dẫn đến cổng $w_{cf}$ nhận giá trị lớn hơn 0. Giả thuyết được đặt ra là vector ID với số mẫu rất ít trở thành nguồn nhiễu tham số, làm suy giảm điểm số dung hợp so với TF-IDF (0,215933 so với 0,365075 trên nhóm extreme-cold có lịch sử); TF-IDF không phải nhánh Semantic CLIP của UFM.
   - *Phép kiểm chứng còn thiếu*: (i) ablation loại trừ nhánh CF (biến thể `semantic_only`) trên chính nhóm Extreme Cold; (ii) phân tích phân phối trọng số $w_{cf}$ trên checkpoint thật `ufm_full_v1`; (iii) đối chiếu chất lượng vector ID giữa item Extreme Cold và item Warm. Cho đến khi có ba phép thử này, đề tài **không** kết luận đây là nguyên nhân đã được chứng minh.

#### 4.3. Cân Chỉnh Xác Suất Độ Tin Cậy (Confidence Calibration)
Nhằm kiểm tra chất lượng hiệu chuẩn xác suất Top-1 trên tập 100 ứng viên lấy mẫu, phương pháp **Temperature Scaling** được fit trên 20.000 ca validation sớm và kiểm toán trên 20.000 ca validation muộn tách biệt với phần fit nhiệt độ ([`reports/ufm_validation_audit_v2/report.json`](../reports/ufm_validation_audit_v2/report.json)):
- Nhiệt độ tối ưu học được: $T = 1,789353$
- Kết quả kiểm chứng trên 20.000 ca muộn:
  * **ECE (Expected Calibration Error)** giảm mạnh từ **0,164477 xuống 0,041497** (giảm 74,8%).
  * **NLL** giảm từ **4,309607 xuống 3,906583**.
  * **Multiclass Brier Score** giảm từ **0,963379 xuống 0,927849**.
- *Phạm vi*: checkpoint đã được chọn bằng toàn bộ validation, nên đây là calibration **thăm dò**, không phải holdout độc lập với lựa chọn mô hình. Nhiệt độ và ECE này chỉ thuộc bộ ứng viên lịch sử; chưa áp dụng để diễn giải kết quả v2.
- *Lưu ý*: Phép chia nhiệt độ $T > 0$ là phép biến đổi đơn điệu nên giữ nguyên thứ tự xếp hạng (không làm thay đổi NDCG@10) và xác suất này là xác suất có điều kiện trên tập 100 ứng viên lấy mẫu, không phải xác suất mua hàng ngoài đời thực.

#### 4.4. Đánh Giá Tài Nguyên & Hiệu Năng Vận Hành
Các bằng chứng vận hành thực tế đã được ghi nhận và lưu trữ trong kho báo cáo:
1. **Tài nguyên huấn luyện & vận hành trên máy chủ FITLAB** ([`reports/campaign_snapshot_20261003.json`](../reports/campaign_snapshot_20261003.json)):
   - Lượt huấn luyện `ufm_full_v1` (12 epochs, 198.232 steps với AMP mixed precision) hoàn tất trong **14.866,28 giây (~4,13 giờ)**.
   - Bộ nhớ hệ thống của container (`cgroup memory.current`) ghi nhận ở mức **10.059.272.192 bytes (~9,37 GiB)**, là một mẫu quan sát, không phải peak RAM hoặc peak VRAM của lượt train; snapshot có **0 sự kiện tràn bộ nhớ** (`oom: 0`, `oom_kill: 0`).
2. **Độ trễ suy luận thực tế trên toàn bộ Catalog 767.045 sản phẩm** ([`reports/demo_acceptance_20261003.json`](../reports/demo_acceptance_20261003.json) & [`reports/demo_ui_export_20261003.json`](../reports/demo_ui_export_20261003.json)):
   - Khi phục vụ trên CPU (4 threads, duyệt theo từng chunk 1.024 items qua toàn bộ 767.045 sản phẩm mà không dùng chỉ mục xấp xỉ), thời gian phản hồi qua 4 ca kiểm thử nghiệm thu API đạt từ **2,89 giây đến 7,27 giây/truy vấn** (`warm`: 2.886,5 ms; `zero_shot`: 3.770,5 ms; `personalized`: 7.047,7 ms; `new_user`: 7.268,8 ms) và **4,47 giây** (4.468,4 ms) khi truy vấn lọc Zero-shot từ lịch sử LEGO qua giao diện Web Proxy.
   - Phạm vi của số đo này được ghi rõ trong artifact: mỗi ca đo **một request duy nhất**, chỉ tính thời gian inference ở server, **không phải** benchmark thông lượng; do đó không dùng để suy ra QPS.
3. **Đo kiểm chuẩn kịch bản với [`scripts/benchmark_resources.py`](../scripts/benchmark_resources.py)**:
   - Công cụ đo tự động (`RAM RSS`, `VRAM peak`, `latency p50/p95/p99` qua 30 lượt lặp và `throughput QPS`) đã được xây dựng tại [`scripts/benchmark_resources.py`](../scripts/benchmark_resources.py) và chạy kiểm chứng trên bộ dữ liệu mẫu cục bộ ([`reports/benchmark_resources_content.json`](../reports/benchmark_resources_content.json)): catalog **mẫu 30 sản phẩm**, backend content, thiết bị CPU — thông lượng **2.167,96 QPS** (4.336 truy vấn / 2,0 s, history 5, top-10), độ trễ **p50 0,80 ms / p95 1,13 ms** ở history 5 (khoảng 0,27–0,80 ms p50 và 0,38–1,17 ms p95 theo độ dài lịch sử), RAM sau nạp catalog **53,5 MiB**.
   - Phạm vi số đo: kết quả trên **chỉ áp dụng cho catalog mẫu 30 sản phẩm**, không đại diện cho mức dùng tài nguyên hay độ trễ production trên catalog đầy đủ. Bộ bằng chứng hiện chưa có lượt đo 30 chu kỳ trên catalog Toys 767.045 sản phẩm, và cũng chưa có bản ghi đỉnh VRAM (peak VRAM) cho lượt huấn luyện — mẫu tức thời 3.490 MiB trong [`reports/FITLAB_snapshot_2026_09_28_2223.json`](../reports/FITLAB_snapshot_2026_09_28_2223.json) được ghi rõ là không đại diện cho đỉnh.

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

#### 4.6. Giới hạn và trạng thái thực nghiệm đối chiếu ngày 27/10/2026
1. **UFM hiện có một seed**: bootstrap theo user trên validation lịch sử không thay thế cho biến thiên giữa các seed. Calibration lịch sử mang tính thăm dò vì checkpoint đã được chọn bằng validation.
2. **Test UFM đã xuất hiện trong artifact v2**: report `reeval_ufm_full_v1_v2.json` có cả `splits.validation` và `splits.test`. Cờ `test_evaluated=false` trong audit lịch sử chỉ mô tả lượt chạy lúc đó; không còn là trạng thái toàn dự án. Chưa có bằng chứng manifest khóa đầy đủ trước lượt test v2, nên không gọi đây là nghiệm thu test cuối độc lập. Các thực nghiệm tiếp theo cần công bố test đã được xem và tránh chọn cấu hình theo điểm test này.
3. **Trạng thái máy chủ**: ngày 27/10 đã kiểm tra trực tiếp FITLAB-01. Checkpoint epoch 9, mã nguồn tạo checkpoint và cache CLIP khớp fingerprint; forward CPU hai ca validation hữu hạn và metric tính lại từ predictions lưu khớp marker. Các snapshot 03/10 và 24/10 chỉ là lịch sử. Nguồn: [kết quả sửa mục 1–15](../reports/fix_1_15_20261027/README.md).
4. **Thực nghiệm còn thiếu**: đã có popularity và TF-IDF validation v2. Còn cần multi-seed UFM, mở rộng ablation qua nhiều seed, kết quả hai biến thể mới, UFM All Beauty và benchmark tài nguyên theo phạm vi giảng viên chốt. Không mô tả các job lịch sử như queue đang chạy.
5. **Kiểm thử phần mềm sau sửa**: local có **84 passed, 3 skipped, 10 subtests passed**; ba skip là kiểm tra Linux flock/procfs trên Windows. Đã sửa fixture, test vô điều kiện, sample builder, audit và ledger. Bản mã hiện tại được kiểm thử riêng trên FITLAB-01; mã tạo checkpoint được giữ nguyên. Chưa chạy GitHub Actions. Nguồn: [kết quả sửa mục 1–15](../reports/fix_1_15_20261027/README.md).
6. **Demo và độ tin cậy**: `Start_Demo.cmd` mở backend content và sinh catalog mẫu nếu thiếu dữ liệu; bằng chứng 4/4 API UFM là lịch sử. Uncertainty là learned reliability proxy; xác suất calibration có điều kiện trên sampled candidates, không phải xác suất mua.

#### 4.7. Kết quả UFM trên bộ ứng viên evaluation_v2
Nguồn: [`reeval_ufm_full_v1_v2.json`](../reports/ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json), checkpoint epoch 9, SHA-256 `5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8`. Đã kiểm tra hash mẫu/metadata và xác minh trực tiếp checkpoint, nguồn huấn luyện, cache trên FITLAB-01. Không chạy lại toàn bộ suy luận UFM trong lần cập nhật này.

| Nhóm | Số ca mỗi split | Validation NDCG@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: |
| Zero-shot (0) | 10.000 | 0,156641 | 0,157281 |
| Extreme cold (1–5) | 10.000 | 0,191567 | 0,195261 |
| Cold (6–20) | 10.000 | 0,197946 | 0,198373 |
| Warm (>20) | 10.000 | 0,110375 | 0,113859 |
| **Cold macro** | 30.000 | **0,182051** | **0,183638** |
| **Overall** | 40.000 | **0,164132** | **0,166193** |

V2 có giao thức và hash bộ mẫu khác bảng 4.1, vì vậy **không so trực tiếp điểm v2 với TF-IDF/BPR-MF/SVD lịch sử**, không kết luận extreme-cold đã được sửa, và không gán mức tăng/giảm giữa hai bộ mẫu cho thay đổi mô hình. Baseline mới trên cùng validation v2 đã có: **TF-IDF Overall/Cold macro 0,181498 / 0,195672; popularity 0,040904 / 0,042527**. UFM tương ứng **0,164132 / 0,182051**, thấp hơn TF-IDF. TF-IDF dùng 10 history item, UFM dùng 20; đây chưa phải ablation kiểm soát chỉ một thành phần. Nguồn: [kết quả sửa mục 1–15](../reports/fix_1_15_20261027/README.md). Test đã được đánh giá; không tiếp tục dùng cách diễn đạt “chưa mở test”.

---

### CHƯƠNG 5: CÁC ĐỀ XUẤT TỐI ƯU & HƯỚNG MỞ RỘNG

1. **Khắc phục Extreme Cold với `count_aware_fusion` và `adaptive_dropout` (Đã triển khai trong [`src/ufm_model.py`](../src/ufm_model.py))**:
   - Thay vì để mạng tự do học $u_{cf}$ trên các item hiếm, áp dụng hàm phạt cưỡng bức theo tần suất xuất hiện trong tập train:
     $$u_{cf} \leftarrow u_{cf} + \exp\left(-\frac{count_{train}}{5.0}\right)$$
   - Kết hợp cùng tần suất dropout thích nghi $p_{\text{drop}}(i) = p_0 \cdot \exp(-count_{train}(i)/10.0)$, cơ chế này ép mô hình giảm trọng số Collaborative và ưu tiên nhánh Semantic CLIP khi sản phẩm có từ 1–5 tương tác. Biến thể này đã được cài đặt và kiểm thử đơn vị trong `UFMRec` (`variant="count_aware_fusion"`), sẵn sàng đưa vào chạy thực nghiệm đối chứng trên GPU.
2. **Tăng tốc độ truy vấn phục vụ Production với FAISS Vector Index (Two-stage Retrieval)**:
   - Bằng chứng demo UFM lịch sử ghi nhận quét toàn bộ danh mục 767.045 sản phẩm bằng ma trận dày trên CPU mất 2,89s – 7,27s. Trong phiên bản triển khai quy mô lớn, kiến trúc được tách thành 2 giai đoạn:
     * *Giai đoạn 1 (Candidate Retrieval)*: Dùng chỉ mục xấp xỉ FAISS HNSW / IVF trên vector người dùng và sản phẩm để trích xuất top 100–200 ứng viên tiềm năng; độ trễ mục tiêu cần đo, chưa có benchmark xác nhận.
     * *Giai đoạn 2 (UFM Reranking)*: Đưa 100–200 ứng viên qua cổng UFM-Rec Head để chấm điểm xếp hạng tinh vi và trích xuất trọng số giải thích $w_{cf}, w_{sem}$.
3. **Mở rộng và Kiểm chứng Đa miền (Cross-domain Generalization)**:
   - Hoàn tất chuỗi thực nghiệm chuyển giao sang tập dữ liệu thứ hai **All Beauty (86.014 items, 37.381 ca validation)** theo đúng kế hoạch đã đóng băng SHA-256 tại [`reports/second_dataset_plan_v1.json`](../reports/second_dataset_plan_v1.json).

---

### CHƯƠNG 6: KẾT LUẬN

Nhóm đã triển khai pipeline chia thời gian, kiến trúc UFM-Rec kết hợp CLIP đóng băng với Transformer và cổng độ tin cậy, cùng demo và công cụ kiểm toán. Phần thực nghiệm vẫn đang hoàn thiện.

- Trên bộ validation lịch sử, UFM có Overall NDCG@10 **0,285611** và Cold macro **0,163738**; TF-IDF tương ứng **0,260367 / 0,191270**. UFM chưa đạt mục tiêu cold-start trên bộ này. Nhiễu ID là giả thuyết chưa được ablation xác nhận.
- Trên `evaluation_v2`, artifact UFM ghi validation Overall/Cold macro **0,164132 / 0,182051**, test **0,166193 / 0,183638**. Đã có baseline cùng validation v2: TF-IDF Overall/Cold macro 0,181498 / 0,195672, cao hơn UFM. Cần ablation để kết luận nguyên nhân; đã có điểm validation v3 tại phần cập nhật đầu tài liệu.
- Temperature scaling lịch sử giảm ECE **0,164477 → 0,041497** trên phần validation muộn; đây là kiểm toán thăm dò có điều kiện trên sampled candidates.
- Cần hoàn thiện tính nhất quán giao thức, baseline v2, kiểm thử phần mềm, multi-seed/ablation và xác nhận tài nguyên FITLAB trước khi chốt báo cáo cuối. Các con số demo và tài nguyên lịch sử không xác nhận khả năng vận hành hiện tại.
