# SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN NGHIÊN CỨU

<!-- current-evidence-20261028 -->
> Cập nhật 28/10/2026: validation v3 đã hoàn tất (40.000 ca, 100 ứng viên/ca, seed 42). Cold macro NDCG@10: **TF-IDF 0,195682; semantic-only 0,190036; UFM full 0,182056; popularity 0,042527**. Tám checkpoint UFM dùng cùng candidate hash. Đây là đánh giá lại checkpoint chọn bằng validation v1, chưa phải holdout độc lập. Test v2 đã được xem; vòng v3 này không đánh giá test. Chi tiết và giới hạn: [bằng chứng mới](../reports/continuation_20261028/README.md). Các số liệu ở phần lịch sử phải đọc theo phiên bản giao thức ghi kèm. Word/PDF cũ chưa đồng bộ cập nhật này.
<!-- /current-evidence-20261028 -->

> **Cập nhật phạm vi:** giảng viên đã chốt hệ khuyến nghị sản phẩm dùng hành vi, hình ảnh và tên/mô tả theo xác nhận của nhóm. Cold-start sản phẩm là nhóm đánh giá trọng tâm. [Phân tích v3 theo train count](../reports/continuation_20261028/count_effect_v3.json) cho thấy full − semantic-only NDCG@10 = −0,024101 ở nhóm 1–5 và +0,014865 ở nhóm 21+ (mỗi nhóm 10.000 ca, một seed). [Giao thức 15 run mới](../configs/multimodal_followup_20261028.json) đã khóa; chưa có kết quả huấn luyện mới. Không dùng slide này để tuyên bố cải tiến đã thắng.


Đối chiếu ngày **28/10/2026**: bộ sửa 1–15 trước đây đạt Linux **87 pass** / Windows **84 pass, 3 skip**. Mã bổ sung calibration/source loader đạt Linux **94 pass** / Windows **91 pass, 3 skip**; các bổ sung restore sau đó có kiểm thử riêng. Xem báo cáo sửa mục 1–15 và log để phân biệt bản mã hiện tại với bản mã đã tạo checkpoint.

## ĐỀ TÀI: UFM-REC — HỆ KHUYẾN NGHỊ SẢN PHẨM COLD-START ĐA PHƯƠNG THỨC DỰA TRÊN ĐỘ TIN CẬY HỌC ĐƯỢC
*(Multimodal Cold-Start Recommender with Learned Reliability Fusion)*

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tên đề tài**: UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được
- **Học phần**: Học máy ứng dụng
- **Lĩnh vực**: Hệ thống gợi ý (Recommender Systems), Học sâu đa phương thức (Multimodal Deep Learning), Ước lượng độ tin cậy (Reliability Proxy).
- **Tập dữ liệu thực nghiệm chính**: Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, 11,57 triệu tương tác). All Beauty đã chuẩn bị dữ liệu; chưa có kết quả UFM đã xác minh trên tập thứ hai.
- **Nhóm thực hiện**: Trần An Kỳ, Nguyễn Thành Phong, Hải, Hân
- **Giảng viên hướng dẫn**: ThS. Nguyễn Thái Anh
- **Thời gian**: 27/10/2026 — trao đổi tiến độ

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
$$w_k=\frac{\mathbb{I}(k\in A)\exp(-u_k)}{\sum_{j\in A}\exp(-u_j)},\quad A=\{j:\text{branch }j\text{ available}\},\ A\ne\varnothing$$
- Nếu không nhánh nào khả dụng, dùng popularity train-only. Khi Semantic khả dụng và item là **Zero-shot** ($count_{train} = 0$): $\mathbb{I}_{cf} = 0 \implies w_{cf} = 0,0$ và $w_{sem} = 1,0$.
- Khi item đã có tương tác trong train: Mạng tự động phân bổ trọng số giữa hai nhánh.

#### 3. Căn chỉnh liên phương thức (Cross-Modal Alignment)
$$g_{align} = \sigma(\text{MLP}_{gate}([h_{cf}, h_{sem}])) \odot \text{Linear}([h_{cf}, h_{sem}]) \cdot \mathbb{I}_{cf \wedge sem}$$
$$h_{fused} = w_{cf} h_{cf} + w_{sem} h_{sem} + g_{align}$$

---

### SLIDE 6: THIẾT LẬP THỰC NGHIỆM CHUẨN MỰC
- **Tập dữ liệu chính**: Amazon Toys & Games (**767.045 sản phẩm**, **758.692 ảnh hợp lệ — 98,91%**, **11.572.689 tương tác**).
- **Chiến lược phân chia**: **Global Temporal Split 80/10/10** theo mốc thời gian; lấy mẫu ứng viên vẫn có giới hạn thông tin cuối split.
- **Giao thức đánh giá Validation**:
  + **40.000 ca kiểm thử** (1 sản phẩm đích + 99 sản phẩm âm tính/ca; đánh giá trên tập ứng viên lấy mẫu, không phải full-catalog).
  + Code/v2: negative cùng regime với target; trọng số popularity dùng `train + validation` cho validation, thêm `test` cho test; loại target và history được ghi nhận. Đây là lấy mẫu hồi cứu, không phải train-only.
  + Validation v3 mới loại mọi positive đã biết trước target, kể cả ngoài cửa sổ history. Audit v1 và đánh giá lại tám UFM checkpoint cùng baseline trên v3 đã hoàn tất. Mỗi phiên bản có hash riêng.
  + Phân tầng cân bằng: 10.000 ca cho mỗi nhóm Zero-shot, Extreme Cold, Cold, Warm.

---

### SLIDE 7: VALIDATION V3 — KẾT QUẢ CHÍNH NGÀY 28/10

- Cùng 40.000 ca và candidate SHA `9387e66e…`; 1 positive + 99 negatives, seed 42.
- **Cold macro NDCG@10:** TF-IDF **0,195682**; semantic-only CLIP **0,190036**; UFM full **0,182056**; popularity **0,042527**.
- Bảy ablation UFM đã train được kiểm toán và đánh giá lại cùng v3. Full thấp hơn semantic-only: chưa xác nhận lợi ích fusion/uncertainty ở chỉ số chính.
- Checkpoint chọn trên validation v1; đây là đánh giá lại, chưa phải holdout độc lập. Chưa có mean/std nhiều seed UFM.
- Nguồn: [kết quả và hash từng mô hình](../reports/continuation_20261028/README.md).

#### Phụ lục của slide 7: kết quả lịch sử v1, giữ riêng với v3

| Nhóm | Số ca | TF-IDF | BPR-MF (`seed=42`, epoch 1) | SVD | UFM (`seed=42`, epoch 9) | UFM − TF-IDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot (0) | 10.000 | 0,186491 | 0,000000 | 0,000000 | 0,188403 | 0,001912 |
| Extreme cold (1–5) | 10.000 | 0,191555 | 0,013367 | 0,036952 | 0,113300 | -0,078255 |
| Cold (6–20) | 10.000 | 0,195763 | 0,027292 | 0,055349 | 0,189511 | -0,006252 |
| Warm (>20) | 10.000 | 0,467661 | 0,501546 | 0,391065 | 0,651231 | 0,183570 |
| **Cold macro** | 30.000 | **0,191270** | 0,013553 | 0,030767 | 0,163738 | −0,027532 |
| **Overall** | 40.000 | 0,260367 | 0,135551 | 0,120842 | **0,285611** | +0,025244 |

Nguồn: UFM/TF-IDF từ audit lịch sử; BPR-MF seed 42 best epoch 1; SVD từ report validation. BPR-MF ba seed báo riêng: Overall **0,135465 ± 0,000231**, Cold macro **0,013602 ± 0,000233**. Chưa có mean ± std của UFM. Các số liệu này không thuộc bộ `evaluation_v2`.

---

### SLIDE 8: PHÂN TÍCH CHUYÊN SÂU — HIỆN TƯỢNG "COLD-START GAP"
- **Tại sao UFM vượt trội ở Warm (+0,1836) và Zero-shot (+0,0019), nhưng lại hụt ở Extreme Cold (−0,0783)?**
  + Ở **Zero-shot ($count_{train}=0$)**: Nhánh CF bị khóa hoàn toàn ($\mathbb{I}_{cf}=0 \implies w_{cf}=0$), mô hình dùng 100% nhánh Semantic CLIP nên hoạt động tốt.
  + Ở **Extreme Cold ($1 \le count_{train} \le 5$)**: Item bắt đầu có ID embedding ($\mathbb{I}_{cf}=1$); cổng $w_{cf}$ mở ra nhận giá trị $> 0$ nhưng chưa đủ bằng chứng rằng cổng gây giảm điểm. Trên nhóm extreme-cold có lịch sử của bộ cũ, UFM đạt 0,2159 so với TF-IDF 0,3651; TF-IDF không phải nhánh Semantic CLIP của UFM.
  + **Giả thuyết** (chưa có thực nghiệm loại trừ): vector ID với 1–5 lượt tương tác bị **overfitting** thành nguồn nhiễu tham số. Cần kiểm chứng bằng ablation `semantic_only` trên nhóm Extreme Cold, phân tích phân phối $w_{cf}$ trên checkpoint thật, và đối chiếu chất lượng vector ID giữa item Extreme Cold và Warm.
- **Giải pháp đã cài đặt trong mã nguồn (`count_aware_fusion`, chưa có số đo định lượng)**:
  + Bổ sung hàm phạt tần suất vào độ bất định CF:
    $$u_{cf} \leftarrow u_{cf} + \exp(-count_{train} / 5.0)$$
  + Ép cổng ưu tiên nhánh Semantic khi sản phẩm có dưới 5 tương tác.

---

### SLIDE 9: HIỆU CHỈNH CONFIDENCE TRÊN VALIDATION V3
- **Vấn đề**: Logit đầu ra của mạng xếp hạng chưa được hiệu chuẩn thành phân phối xác suất đáng tin cậy trên tập ứng viên.
- **Phương pháp**: Fit **Temperature Scaling** ($T = 2,012405$) trên 20.000 ca validation v3 sớm, audit 20.000 ca muộn, giữ timestamp bằng nhau cùng phía:
  + **ECE** từ **0,082061 xuống 0,037478**.
  + **NLL** từ **4,497667 xuống 4,258086**.
  + Đây là thăm dò vì checkpoint từng chọn bằng validation; các phần có thể chung user. Confidence có điều kiện trên 100 ứng viên, không phải xác suất mua; temperature không cải thiện thứ hạng.
  + **Multiclass Brier Score** giảm từ **0,963379 xuống 0,927849**.
- *Phạm vi*: số liệu calibration thuộc bộ lịch sử; validation đã được dùng chọn checkpoint nên đây là kết quả thăm dò, không phải holdout độc lập với chọn mô hình.
- *Lưu ý*: Calibration giữ nguyên thứ hạng Top-K (không thay đổi NDCG@10) và là xác suất có điều kiện trên 100 ứng viên lấy mẫu.

---

### SLIDE 10: BẰNG CHỨNG TÀI NGUYÊN & HIỆU NĂNG VẬN HÀNH
- **Huấn luyện & Bộ nhớ trên FITLAB** (`reports/campaign_snapshot_20261003.json`):
  + Lượt train `ufm_full_v1` (12 epochs, 198.232 steps, AMP) hoàn tất trong **4,13 giờ** (14.866s).
  + Snapshot ghi bộ nhớ container (`cgroup memory.current`) **~9,37 GiB**, `0` sự kiện OOM; đây không phải peak RAM/VRAM và không chứng minh toàn lượt train dưới 16 GiB.
- **Độ trễ phục vụ thực tế (Full-catalog 767.045 items trên CPU)**:
  + Nghiệm thu 4 ca API thật trên FITLAB (`reports/demo_acceptance_20261003.json`): **2,89s – 7,27s / truy vấn**; qua Web Proxy Zero-shot đạt **4,47s**. Mỗi ca đo **1 request**, chỉ tính inference server — không dùng để suy ra QPS.
- **Đo kiểm chuẩn cục bộ (`reports/benchmark_resources_content.json`)**:
  + Với bộ catalog mẫu nhẹ (30 items, backend content, CPU): RAM sau nạp **53,6 MiB**, độ trễ $p50 = 0,80\text{ ms}$ / $p95 = 1,13\text{ ms}$ (history 5), băng thông **2.167,96 QPS**. Số đo này **chỉ áp dụng cho catalog mẫu 30 items**, chưa phải catalog đầy đủ.

---

### SLIDE 11: DEMO TRỰC QUAN HỆ THỐNG
*(Trình diễn Demo Web trực tiếp hoặc qua ảnh bằng chứng nghiệm thu)*

- **Giao diện người dùng**:
  + Thiết kế **Pro Max Dark Glassmorphism**, trực quan hóa sản phẩm có ảnh thật từ Amazon CDN, lọc theo từng nhóm `Zero-shot`, `Extreme Cold`, `Cold`, `Warm`.
  + Hiển thị trực tiếp trọng số cổng `CF % · Semantic %` và `Uncertainty proxy` cho từng sản phẩm gợi ý.
- **Nghiệm thu thực tế & Khả năng tái lập**:
  + Đã vượt qua **4/4 ca kiểm thử API thật** trên FITLAB (`new_user`, `personalized`, `zero_shot`, `warm`) và xuất JSON kiểm chứng.
  + `Start_Demo.cmd` chạy backend **content**, không tự nạp checkpoint UFM. Ảnh/API UFM trên là bằng chứng lịch sử.
  + Hỗ trợ tự động tạo bộ dữ liệu mẫu offline (`scripts/build_mock_demo_data.py`) khi chạy trên máy mới chưa tải dataset 10GB.

---

### SLIDE 12: KẾT QUẢ V2 VÀ TRẠNG THÁI ĐỐI CHIẾU 27/10

| Nhóm | Số ca mỗi split | Validation NDCG@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: |
| Zero-shot (0) | 10.000 | 0,156641 | 0,157281 |
| Extreme cold (1–5) | 10.000 | 0,191567 | 0,195261 |
| Cold (6–20) | 10.000 | 0,197946 | 0,198373 |
| Warm (>20) | 10.000 | 0,110375 | 0,113859 |
| **Cold macro** | 30.000 | **0,182051** | **0,183638** |
| **Overall** | 40.000 | **0,164132** | **0,166193** |

- Artifact nhập từ `tmp/transfer/evaluation_v2`; đã đối chiếu hash mẫu và metadata, chưa chạy lại GPU. Checkpoint report là epoch 9; tên run liên hệ lượt seed 42 lịch sử, JSON v2 không ghi seed riêng. Ngày 27/10 là ngày đối chiếu local, không phải thời điểm chạy đã được xác nhận.
- **Test đã được đánh giá trong v2**; không gọi toàn dự án là validation-only. Chưa xác nhận manifest khóa đầy đủ trước test, nên chưa gọi đây là nghiệm thu test cuối.
- Hash mẫu v2 khác audit lịch sử. Baseline mới dùng đúng v2: TF-IDF Cold macro **0,195672**, popularity **0,042527**, UFM **0,182051**. Không so với baseline lịch sử ở slide 7; UFM chưa vượt TF-IDF trên v2.
- 03/10: bằng chứng train/queue lịch sử. 24/10: bản ghi khôi phục FITLAB thiếu cache/checkpoint. 27/10: đã xác minh trực tiếp checkpoint/cache/source trên FITLAB-01, CPU forward hữu hạn và metric predictions lưu khớp marker.
- Còn cần multi-seed UFM, mở rộng ablation qua nhiều seed, đánh giá biến thể mới và UFM All Beauty theo phạm vi. Test local sau sửa: **84 pass, 3 skip** (Linux-only). V3 đã tạo/audit, đã có metric tám checkpoint UFM và baseline v3. [Bằng chứng](../reports/fix_1_15_20261027/README.md).

---

### SLIDE 13: KẾT LUẬN VÀ NỘI DUNG XIN GÓP Ý
- **Kết quả quan sát**: UFM tăng Overall nhưng thua Cold macro trên bộ lịch sử. V2 đã có baseline cùng bộ mẫu: UFM thua TF-IDF về Overall và Cold macro; chưa chứng minh cải thiện.
- **Đóng băng CLIP**: cho phép cache đặc trưng và giảm phần encoder cần huấn luyện; chưa có đối chứng fine-tuning và peak VRAM để định lượng mức tiết kiệm.
- **Giả thuyết extreme-cold**: nhánh ID có thể chưa học đủ tốt; cần semantic-only, fixed-fusion và phân tích gate trên checkpoint thật. Test mô hình khởi tạo ngẫu nhiên chưa chứng minh nguyên nhân.
- **Test đã được xem**: công bố lượt đánh giá v2, đối chiếu manifest và chốt giao thức cho thực nghiệm sau; không chọn cấu hình theo điểm test đã biết.
- **Xin ý kiến thầy**: ưu tiên chuẩn hóa một dataset hay mở rộng All Beauty; baseline/ablation tối thiểu nào cần có; thiết kế candidate sampling và holdout tiếp theo thế nào cho phù hợp.

---

### SLIDE 14: LỜI CẢM ƠN
- Nhóm thực hiện xin chân thành cảm ơn **ThS. Nguyễn Thái Anh** và Quý Thầy Cô trong Hội đồng đã lắng nghe!
- Kính mời Quý Thầy Cô đặt câu hỏi và đóng góp ý kiến cho đề tài.
