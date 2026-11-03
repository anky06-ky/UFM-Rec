# SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN NGHIÊN CỨU

> Nguồn nội dung của [PPTX bảo vệ 14 slide](../output/presentation/UFM_Rec_Bao_Ve_20261103.pptx), cùng số liệu với [báo cáo nộp](final_report.md). Số liệu chính là validation v3 (40.000 ca, 100 ứng viên/ca, seed 42); nhóm chốt đồ án với kết quả hiện có, vòng 15 lượt đa seed là hướng phát triển. Bản 29/10 nằm ở [lịch sử](../output/history/presentation/UFM_Rec_Bao_Ve_Hien_Trang_20261029.pptx).

## ĐỀ TÀI: UFM-REC — HỆ KHUYẾN NGHỊ SẢN PHẨM COLD-START ĐA PHƯƠNG THỨC DỰA TRÊN ĐỘ TIN CẬY HỌC ĐƯỢC
*(Multimodal Cold-Start Recommender with Learned Reliability Fusion)*

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tên đề tài**: UFM-Rec: Hệ Khuyến Nghị Sản Phẩm Cold-Start Đa Phương Thức Dựa Trên Độ Tin Cậy Học Được
- **Học phần**: Học máy ứng dụng
- **Lĩnh vực**: Hệ thống gợi ý (Recommender Systems), Học sâu đa phương thức (Multimodal Deep Learning), Ước lượng độ tin cậy (Reliability Proxy).
- **Tập dữ liệu thực nghiệm chính**: Amazon Reviews 2023 — Toys & Games (767.045 sản phẩm, 11,57 triệu tương tác). All Beauty đã chuẩn bị dữ liệu; chưa có kết quả UFM đã xác minh trên tập thứ hai.
- **Nhóm thực hiện**: Nguyễn Thành Phong, Trần An Kỳ, Nguyễn Hoàng Hải, Hân
- **Giảng viên hướng dẫn**: Phan Hồ Viết Trường
- **Thời gian**: 11/2026 — bản bảo vệ, chốt với kết quả hiện có

> **Lời thoại mở đầu (Speaker Notes)**:
> *"Kính thưa Thầy Cô trong Hội đồng và các bạn, hôm nay nhóm em xin phép trình bày đề tài UFM-Rec — nghiên cứu hệ khuyến nghị sản phẩm kết hợp đặc trưng nền tảng thị giác - ngôn ngữ CLIP đóng băng với biểu diễn tuần tự nhân quả và cơ chế cổng độ tin cậy học được."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & ĐỘNG LỰC NGHIÊN CỨU
- **Vấn đề thực tế**:
  + Sản phẩm ít hoặc chưa có tương tác train thiếu dữ liệu để học biểu diễn định danh.
- **Phân loại phân tầng tần suất sản phẩm trong tập Train (Regimes)**:
  1. `Zero-shot`: Sản phẩm chưa có tương tác trong tập train ($count_{train} = 0$).
  2. `Extreme Cold`: Sản phẩm cực hiếm ($1 \le count_{train} \le 5$).
  3. `Cold`: Sản phẩm ít tương tác ($6 \le count_{train} \le 20$).
  4. `Warm`: Sản phẩm phổ biến ($count_{train} > 20$).
- **Thách thức của các tiếp cận hiện tại**:
  + **Collaborative Filtering (BPR-MF, SVD) & Sequential thuần ID**: Không có embedding đã học cho item zero-shot, suy thoái nặng ở nhóm ít tương tác.
  + **Content-based (TF-IDF)**: Bắt từ khóa văn bản tốt ở nhóm lạnh nhưng bỏ qua ngữ cảnh hình ảnh và không dùng ảnh; chất lượng ở warm cần đo theo đúng protocol.
  + **Ghép nối đơn giản (Concat/MLP)**: Cần kiểm chứng việc thêm nhánh ID giúp hay làm giảm chất lượng ở các item ít tương tác; chưa coi dominance là nguyên nhân đã xác nhận.

---

### SLIDE 3: GIẢI PHÁP ĐỀ XUẤT — UFM-REC
**Ý tưởng cốt lõi**:
1. **Đa phương thức đóng băng (Frozen Multimodal)**: Sử dụng **CLIP ViT-B/32** (khóa cố định commit revision) để trích xuất biểu diễn ngữ nghĩa của văn bản và hình ảnh sản phẩm.
2. **Chuỗi hành vi nhân quả (Causal Sequential)**: Sử dụng mạng **Transformer Encoder** mô hình hóa chuỗi lịch sử tương tác (tối đa 20 sản phẩm gần nhất).
3. **Cổng độ tin cậy học được (Learned Reliability Gate)**: Ước lượng đại lượng proxy độ bất định của từng nhánh để điều tiết trọng số giữa nội dung CLIP và lịch sử ID.
4. **Căn chỉnh liên phương thức (Cross-Modal Alignment)**: Đồng bộ hóa không gian biểu diễn khi cả hai nhánh cùng khả dụng.

---

### SLIDE 4: KIẾN TRÚC TỔNG THỂ HỆ THỐNG
Nhánh semantic dùng trung bình vector history khả dụng; nhánh ID dùng Transformer causal.

- **Nhánh Ngữ nghĩa (Semantic Branch)**:
  + Input: Text Embedding (512d) + Image Embedding (512d) + Availability Mask nhị phân $m \in \{0,1\}^2$.
  + Phương thức bị thiếu được zero-fill trước khi qua Adapter 2 tầng tuyến tính + GELU + LayerNorm chiếu về $d = 128$.
- **Nhánh Cộng tác (Collaborative Branch)**:
  + Input: Chuỗi ID tương tác (độ dài 20, right-padded) + Causal Masking.
  + **Unknown Item Masking**: Các ID có $count_{train} = 0$ được gán về ID 0 (Padding) **trước khi lookup**, tránh dùng embedding ID chưa được học.
- **Tương tác Cặp (Pair Interaction)**:
  + CF pair ghép user, item, tích từng phần, độ dài history và train count. Semantic pair ghép user, item, tích từng phần và modality masks.

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

### SLIDE 6: THIẾT LẬP VÀ GIỚI HẠN GIAO THỨC
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

Ghi chú nói: TF-IDF dùng 10 history item, UFM dùng 20; đây là so sánh hệ thống. SASRec/BERT4Rec đã có bằng chứng v1, chưa có bảng v3 xác minh. Bảng v1/v2 giữ ở báo cáo và nguồn lịch sử, không ghép vào v3.

Hình: `output/figures/current_v3_20261029/cold_macro.png`. Bảng đủ 10 mô hình: `reports/nontraining_20261029/ranking_table.md`.

---

### SLIDE 8: PHÂN TÍCH V3 THEO TRAIN COUNT

| Train count | Full trừ semantic-only NDCG@10 |
| --- | ---: |
| 0 | −0,002424 |
| 1–5 | −0,024101 |
| 6–20 | +0,002586 |
| Trên 20 | +0,014865 |

- Mỗi nhóm 10.000 ca, cùng candidates v3, một seed 42.
- Trong 5.247 ca extreme-cold có history, chênh lệch là −0,045933; các ca không history cho cùng điểm vì fallback.
- Đây là quan sát, chưa chứng minh ID/gate là nguyên nhân. Full cao hơn semantic-only ở warm không có nghĩa cao hơn TF-IDF ở warm v3.
- `count_aware_fusion` chỉ thêm penalty theo count; `adaptive_dropout` là can thiệp riêng. Cả hai chưa có số đo sau train.

Ghi chú nói: Bảy ablation một seed đã có. Full cold macro 0,182056 thấp hơn no-uncertainty 0,182253 và no-cross-align 0,183393 ở seed này; chưa xác nhận ích lợi cold macro của các thành phần. Nguồn: `count_effect_v3.json`, `validation_v3_results.json`. Hình: `output/figures/current_v3_20261029/count_effect.png`.

---

### SLIDE 9: CALIBRATION TRÊN VALIDATION V3

Temperature T = 2,012405; fit 20.000 ca sớm, audit 20.000 ca muộn, giữ timestamp bằng nhau cùng phía.

| Chỉ số audit | Trước | Sau |
| --- | ---: | ---: |
| ECE | 0,082061 | 0,037478 |
| NLL | 4,497667 | 4,258086 |
| Multiclass Brier | 0,969342 | 0,960605 |

- Confidence có điều kiện trên 100 ứng viên, không phải xác suất mua.
- Kết quả thăm dò: validation đã tham gia chọn checkpoint và hai phần có thể chung user.
- Temperature giữ thứ hạng; ECE giảm không tự chứng minh uncertainty head cải thiện ranking.

Nguồn: `reports/continuation_20261028/summary.json`; [Guo và cộng sự 2017](https://proceedings.mlr.press/v70/guo17a.html). Brier ở đây thuộc v3, đã bỏ số audit v1 bị trộn trong bản cũ.

---

### SLIDE 10: HIỆU NĂNG UFM TRÊN WINDOWS

- Catalog 767.045 item, checkpoint full epoch 9, backend UFM CPU.
- Hai warm-up + 10 request, history một item và K=5: **p50 21,25 giây; p95 26,35 giây**.
- RSS cao nhất quan sát **2.551,5 MiB**; QPS tuần tự khoảng **0,0457**.
- Hai request đồng thời trả một 200 và một 429; chưa phải hệ phục vụ nhiều suy luận đồng thời.
- Chưa có peak VRAM train đủ để nghiệm thu mục tiêu GPU 16 GiB.

Ghi chú nói: 10 request chưa mô tả p99 ổn định. Số đo FITLAB 03/10 là request đơn; số catalog content 30 item là demo mẫu, không đặt cạnh số UFM đầy đủ như cùng benchmark. Nguồn: `windows_ufm_benchmark.json`; không suy thời gian Colab từ lượt train lịch sử 4,13 giờ.

---

### SLIDE 11: DEMO UFM THẬT

- Tìm sản phẩm, chọn history, lọc regime và nhận Top-K; xem backend, train count, điểm, trọng số nhánh và uncertainty.
- UFM đã có sáu ca API và UI zero-shot trên catalog 767.045 item; counts thật và checkpoint được kiểm SHA.
- `Start_Demo.cmd` tự chọn UFM khi có checkpoint hoàn tất hoặc content khi chưa có; kiểm tra backend hiển thị trước khi trình diễn.
- Ảnh JPEG trên UI để hiển thị; suy luận dùng CLIP cache, ảnh UI lỗi không tự đổi điểm.
- API đạt 6/6; UI Chrome headless đạt 4/4, gồm export JSON, ảnh lỗi và mất mạng. [Receipt UI](../reports/nontraining_20261029/demo_ui_end_to_end_20261029.json) và [ảnh zero-shot](../reports/nontraining_20261029/demo_ui_zero_shot_20261029.png) đã lưu; cần thử lại trên máy bảo vệ.

Ghi chú nói: Mở bản sao UFM đã kiểm SHA. Chuẩn bị ảnh/video đúng backend và báo trước thời gian chờ. Người dùng không history nhận fallback, không gọi đó là cá nhân hóa người dùng mới. Nguồn: `docs/demo.md` và các receipt Windows.

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
- Còn cần multi-seed UFM, mở rộng ablation qua nhiều seed, đánh giá biến thể mới và UFM All Beauty theo phạm vi. Test local sau sửa ngày 27/10 là 84 pass, 3 skip; bản local sau sửa notebook ngày 29/10 đạt **116 pass, 3 skip, 10 subtests**. V3 đã tạo/audit, đã có metric tám checkpoint UFM và baseline v3. [Bằng chứng](../reports/fix_1_15_20261027/README.md).

---

### SLIDE 13: KẾT LUẬN VÀ PHẦN CÒN THIẾU

- Hệ thống ba nguồn tín hiệu đã có code, checkpoint, đối chứng một seed, audit và demo UFM thật.
- Kết quả chính v3: full chưa vượt TF-IDF/semantic-only ở cold macro; nhóm 1–5 tương tác là hướng kiểm chứng tiếp.
- CLIP đóng băng cho phép cache đặc trưng; chưa có đối chứng fine-tuning để định lượng mức tiết kiệm tài nguyên.
- Kế hoạch 15 lượt gồm năm biến thể × ba seed; phần train hiện được để lại. Chưa có kết quả hai cải tiến hoặc UFM All Beauty.
- Hướng tiếp theo: chạy 15 lượt đa seed đã khóa và đánh giá trên một holdout độc lập được thống nhất trước.

Ghi chú nói: Không chọn seed hoặc cấu hình theo test đã biết; không coi bootstrap user là thay nhiều seed. Nội dung và số liệu cùng bản `docs/final_report.md`. Tài liệu gốc của CLIP, SASRec, BERT4Rec, BPR, calibration và dataset nằm ở phần tài liệu tham khảo của báo cáo.

---

### SLIDE 14: LỜI CẢM ƠN
- Nhóm thực hiện xin chân thành cảm ơn giảng viên hướng dẫn **Phan Hồ Viết Trường** và Quý Thầy Cô trong Hội đồng đã lắng nghe!
- Kính mời Quý Thầy Cô đặt câu hỏi và đóng góp ý kiến cho đề tài.
