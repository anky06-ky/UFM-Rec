# BÁO CÁO TIẾN TRÌNH UFM-REC
## Snapshot ngày 30/09/2026

Hệ khuyến nghị sản phẩm cold-start trên Amazon Reviews 2023 - Toys and Games.
Mốc FITLAB: 11:07 ngày 30/09/2026 (UTC+07). Kiểm tra demo local: 11:10 cùng ngày.

### 1. Kết luận hiện tại

Đã có pipeline dữ liệu đầy đủ, các baseline thực nghiệm và demo local hoạt động trên 767.045 sản phẩm. Đợt này hoàn tất thêm BPR-MF với 3 seed, calibration TF-IDF theo thời gian và giao diện demo có lọc cold-start. Toàn bộ 36 kiểm thử local PASS.

UFM đầy đủ chưa train xong. CLIP đang trích đặc trưng; queue UFM và 7 ablation chờ điều kiện đầu vào/GPU. Chưa có metric GPU UFM, chưa đánh giá test cho BPR-MF hoặc chiến dịch UFM mới. Không thể kết luận dự án đã hoàn tất proposal.

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Temporal data + TF-IDF | Hoàn thành | 11.572.689 tương tác; catalog 767.045 |
| Popularity / content / SVD / RRF | Có kết quả test trước đây | Cùng 40.000 sampled cases |
| BPR-MF | Hoàn thành 3 seed | Mỗi seed 5 epoch, chọn bằng validation |
| Calibration TF-IDF | Hoàn thành bước thăm dò | 20.000 fit + 20.000 audit theo thời gian |
| CLIP text + image | Đang chạy | 522.880/767.045 dòng, 68.17% |
| UFM + 7 ablation | Đã có code/queue; đang chờ | Chưa có production checkpoint |
| Demo local | Hoạt động với TF-IDF thật | 4 API scenarios và luồng UI đã kiểm tra |

### 2. Dữ liệu và phạm vi đánh giá

- 16.260.406 review gốc; giữ 11.572.689 tương tác verified, rating từ 4 trở lên.
- Temporal 80/10/10: train 9.258.647; validation 1.159.827; test 1.154.215.
- 584.846 sản phẩm có tương tác train; 182.199 sản phẩm zero-shot.
- Validation/test sample: 40.000 trường hợp mỗi split, 10.000 mỗi regime, 1 positive + 99 negative cố định.
- Phần trăm CLIP chỉ phản ánh tiến độ extraction, không phải phần trăm hoàn thành toàn dự án.

<!-- pagebreak -->
## Kết quả baseline đã kiểm chứng

### 3. Test của các baseline cổ điển (kết quả đã có)

| Mô hình | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | 0.2669 | 0.1691 | 0.1393 |
| TF-IDF content | 0.4013 | 0.2658 | 0.2243 |
| TruncatedSVD | 0.2193 | 0.1310 | 0.1043 |
| Hybrid RRF | 0.4231 | 0.2657 | 0.2172 |

RRF đạt Recall@10 cao nhất trong nhóm trên; TF-IDF có NDCG@10 cao hơn rất ít. Chênh lệch chưa được kiểm chứng thống kê. Các giá trị test này không so trực tiếp với bảng validation bên dưới.

### 4. BPR-MF: kết quả mới trên validation

ID-only implicit BPR, 32 factors, learning rate 0,05, regularization 0,0001, batch 4.096, 5 epoch/seed. Graph train giữ 1.622.880 người dùng có ít nhất 2 sản phẩm khác nhau và 5.832.475 cặp user-item. Negative chỉ lấy từ item train và loại toàn bộ positive đã biết của user.

Checkpoint mỗi seed được chọn theo NDCG@10 macro của zero-shot, extreme cold và cold; warm/overall được báo đồng thời. Các lượt có cùng hash source, code và cấu hình ngoài seed.

| Seed / epoch chọn | Overall NDCG@10 | Cold macro NDCG@10 | Warm NDCG@10 |
|---|---:|---:|---:|
| 42 / 1 | 0.135551 | 0.013553 | 0.501546 |
| 7 / 1 | 0.135203 | 0.013397 | 0.500620 |
| 2026 / 1 | 0.135641 | 0.013856 | 0.500998 |

Trung bình ± độ lệch chuẩn mẫu (3 seed): overall 0.135465 ± 0.000231; cold macro 0.013602 ± 0.000233; warm 0.501055 ± 0.000466.

Độ lệch chuẩn seed không phải khoảng tin cậy trên người dùng. BPR-MF không có nội dung để biểu diễn item zero-shot; Recall@10 nhóm này bằng 0 trong cả ba seed. Kết quả cho thấy baseline thiên về warm, chưa phải bằng chứng UFM cải thiện cold-start.

Đối chứng TF-IDF + Transformer GPU cũ đã hoàn tất 8 epoch, best epoch 5: validation overall NDCG@10 0,277327; known-user 0,380043. Đây là kiến trúc đối chứng, không phải checkpoint UFM đầy đủ.

<!-- pagebreak -->
## Calibration theo thời gian

### 5. Thiết lập và kết quả mới

Tái tạo score TF-IDF của 40.000 validation cases; toàn bộ target ranks khớp artifacts baseline. Chia theo timestamp: 20.000 mẫu sớm để fit temperature, 20.000 mẫu muộn để audit; các timestamp trùng không bị tách qua ranh giới. Temperature chọn bằng NLL trên fold fit, trong [0,01; 100], thu được T = 1.972659.

| Metric audit | Trước (T = 1) | Sau temperature scaling |
|---|---:|---:|
| ECE top-1, 15 bins | 0.168601 | 0.116684 |
| NLL | 4.750213 | 4.513757 |
| Brier multiclass | 0.999966 | 0.983381 |

![Reliability diagram](../output/figures/calibration_validation_20260930.svg)

Temperature dương không đổi thứ tự ranking; nó đổi phân phối softmax trên 100 candidates. Các trường hợp đồng hạng top-1 dùng xác suất đúng kỳ vọng với tie-break đều để không thiên vị cột positive.

### Giới hạn cần giữ khi viết báo cáo cuối kỳ

- Đây là calibration thăm dò của baseline TF-IDF, chưa phải kiểm chứng uncertainty của UFM.
- Baseline đã được đánh giá trên full validation trước đó; audit này không phải holdout mới cho việc chọn mô hình.
- Fold thời gian có thể chung người dùng. Chưa có bootstrap interval hay nhiều seed UFM.
- ECE/NLL/Brier chỉ có ý nghĩa trong candidate protocol đã lấy mẫu, không phải xác suất mua trong toàn catalog.
- Phương pháp tham khảo: Guo et al. (2017), On Calibration of Modern Neural Networks, PMLR 70:1321-1330; proceedings.mlr.press/v70/guo17a.html.

<!-- pagebreak -->
## Demo có thể trình bày ngay

### 6. Giao diện và luồng sử dụng

Mở Start_Demo.cmd tại thư mục dự án hoặc truy cập http://127.0.0.1:8765 khi server đang chạy. Demo dùng TF-IDF trên dữ liệu thật; backend và giới hạn được ghi ngay trên giao diện.

![Demo TF-IDF, Top 5 zero-shot từ lịch sử LEGO](../output/demo/UFM_Rec_demo_2026_09_30.png)

Tìm LEGO → thêm sản phẩm → chọn Top K → tìm gợi ý → lọc zero-shot → xuất JSON. Có xóa lịch sử, chống thêm trùng, đánh dấu kết quả cũ khi thay lựa chọn, fallback khi ảnh lỗi và popularity khi lịch sử rỗng. File JSON tải qua UI đã được kiểm tra: đúng 5 kết quả zero-shot, không chứa ASIN lịch sử.

| Kịch bản API trên catalog thật | Kiểm tra | Thời gian server |
|---|---|---|
| new_user | PASS, Top 5 | 250.7 ms |
| personalized | PASS, Top 10 | 696.7 ms |
| zero_shot | PASS, Top 5 | 390.2 ms |
| warm | PASS, Top 5 | 299.9 ms |

Mỗi kịch bản đo một lần; chưa phải benchmark throughput, p95 hoặc so sánh phần cứng. Backend UFM có đường dẫn features thay thế với kiểm tra hash; cần checkpoint full hoàn tất trước khi phục vụ. Hướng dẫn và kịch bản 3 phút: docs/DEMO_GUIDE.md.

<!-- pagebreak -->
## Phần còn lại và điều kiện hoàn tất

### 7. Trạng thái FITLAB

CLIP đã phục hồi sau SIGKILL tại 497.856 dòng ngày 29/09; snapshot mới ghi 522.880 dòng và 520.317 ảnh thành công. Cursor queue đọc ở chu kỳ trước là 522.752, nên lệch 128 dòng so với log mới hơn là bình thường. GPU utilization 100%, free 6.528 MiB; PID extraction 443209.

UFM đang waiting_for_complete_features_and_idle_gpu; ablation đang waiting_for_full_ufm_and_idle_gpu. Không restart job đang chạy. Queue tự tiếp tục trong cửa sổ đã cấu hình khi đủ features và GPU rảnh; không bảo đảm tự phục hồi nếu container bị tạo lại hoặc tiến trình bị kill.

| Mốc tiếp theo | Việc cần thực hiện | Điều kiện nghiệm thu |
|---|---|---|
| 1. Full CLIP | Chờ extraction, audit coverage/hash | complete.json đủ 767.045 item, vector hữu hạn |
| 2. UFM full | GPU smoke/resume rồi train/validation | Production completed.json, best checkpoint và metric |
| 3. Ablation | 7 biến thể train độc lập | Cùng budget/seed/protocol, bảng cold/warm và gate |
| 4. Baseline còn thiếu | SASRec, BERT4Rec, concat hybrid | Tái hiện đúng phương pháp, cùng dữ liệu/candidates |
| 5. Kiểm chứng mở rộng | MovieLens-1M; nhiều seed UFM; calibration UFM | Tách rõ sanity check không ảnh và thí nghiệm đa phương thức |
| 6. Hiệu năng + test | VRAM 16 GiB, throughput; khóa config trước test | Bằng chứng đo thực tế, test một đợt có kiểm soát |
| 7. Bàn giao cuối | Demo UFM checkpoint thật, báo cáo và slide | Kết quả thực nghiệm đầy đủ, kết luận có nguồn |

Chưa thể đánh dấu các mốc 2-7 hoàn thành. SASRec/BERT4Rec, concat hybrid, dataset thứ hai và slide bảo vệ cuối cùng chưa được làm xong trong snapshot này. Các bảng mới không thay thế những hạng mục đó.

### 8. Bằng chứng và tái lập

- BPR: models/bpr_mf, bpr_mf_seed7, bpr_mf_seed2026 dưới data/processed/toys_games_full_temporal; tổng hợp reports/BPR_MF_multiseed_2026_09_30.json.
- Calibration: models/content_calibration_v1/report.json và scores_validation.npz; chạy src/calibrate_validation.py với output mới.
- FITLAB: reports/FITLAB_snapshot_2026_09_30.json; báo cáo phục hồi cùng ngày; các log/queue trên server.
- Demo: reports/demo_acceptance_2026_09_30.json; output/demo/ufm-rec-demo.json; ảnh chụp giao diện; Start_Demo.cmd.
- Kiểm thử: python -m unittest discover -s tests -v; 36 tests PASS (lần cuối 8,959 giây). Smoke/unit tests không phải bằng chứng chất lượng recommendation.
- Mã train UFM và fingerprints của chiến dịch đang chạy được giữ nguyên trong đợt này. Thay đổi local chưa commit/push.
