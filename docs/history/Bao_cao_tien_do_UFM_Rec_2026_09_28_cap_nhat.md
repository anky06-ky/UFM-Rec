# BÁO CÁO TIẾN ĐỘ UFM-REC

## Cập nhật triển khai và chiến dịch thực nghiệm dài ngày

Ngày 28/09/2026 · Snapshot FITLAB: 17:55:11, múi giờ UTC+07:00.

### 1. Kết luận tiến độ hiện tại

Đã khôi phục thành công lượt GPU cũ: mô hình two-tower TF-IDF + Transformer hoàn tất 8 epoch, 132.216 optimizer steps; checkpoint tốt nhất ở epoch 5. Phạm vi đồ án vẫn giữ đầy đủ Foundation Model, uncertainty và UGAF. Mã UFM và lịch thực nghiệm đã triển khai, nhưng chưa có kết quả huấn luyện UFM đầy đủ để kết luận hiệu quả.

GPU Quadro RTX 6000 đang trích CLIP trên catalog. Đã commit đặc trưng cho 101.888/767.045 sản phẩm, tương đương 13,28%. Hàng đợi tự chuyển sang GPU smoke/resume, full UFM rồi 7 ablation sau khi đặc trưng hoàn tất và GPU rảnh. Demo CPU đã hoạt động trong lúc GPU làm việc.

| Hạng mục | Trạng thái tại snapshot | Bằng chứng |
|---|---|---|
| Dữ liệu lớn, split theo thời gian | Hoàn tất | split_manifest.json |
| Khôi phục đối chứng GPU | Hoàn tất 8 epoch | completed.json, history.jsonl |
| CLIP text + image toàn catalog | Đang trích, 13,28% | progress.json |
| Graph positive train-only | Hoàn tất 4.230.848 mẫu | complete.json |
| UFM đầy đủ | Mã sẵn sàng; GPU chưa tối ưu | Queue đang chờ features |
| 7 ablation, seed 42 | Runner đã bật; chưa có kết quả | Suite đang chờ full UFM |
| Demo và kiểm thử | CPU demo chạy; 32 test PASS | Kiểm thử local và FITLAB |

### 2. Dữ liệu và tính độc lập của các tập

Nguồn Amazon Reviews 2023, Toys and Games: 16.260.406 bản ghi raw; giữ verified purchase và rating từ 4, loại 121.726 bản ghi trùng chính xác. Còn 11.572.689 tương tác, 6.158.534 người dùng và catalog 767.045 sản phẩm.

| Tập | Số tương tác | Tỷ lệ thực tế | Ranh giới thời gian UTC |
|---|---:|---:|---|
| Train | 9.258.647 | 80,004% | Trước 11/08/2021 |
| Validation | 1.159.827 | 10,022% | 11/08/2021 đến trước 16/07/2022 |
| Test | 1.154.215 | 9,974% | Từ 16/07/2022 |

Split là temporal 80/10/10 gần đúng tại ranh giới ngày, không phải item-disjoint. Tương tác được chia theo thời gian; cùng sản phẩm/ảnh có thể xuất hiện ở nhiều tập. Zero-shot được xác định riêng bằng 0 tương tác train. Lịch sử chỉ dùng sự kiện trước mục tiêu; sự kiện cùng timestamp không thấy nhau. Không dùng nhãn validation/test để tạo gradient hay lọc negative train.

<!-- pagebreak -->

## Kết quả đã xác nhận và mô hình đầy đủ

### 3. Đối chứng GPU sau khôi phục

Two-tower content-aware dùng TF-IDF và lịch sử Transformer, không phải UFM có CLIP/uncertainty/UGAF. Checkpoint epoch 5 được chọn bằng NDCG@10 trên nhóm có lịch sử; dừng sớm sau epoch 8. Validation cố định 40.000 mẫu, 1 positive + 99 negatives/mẫu.

| Nhóm validation | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Toàn bộ, 40.000 mẫu | 0,412500 | 0,277327 | 0,235743 |
| Có lịch sử, 20.601 mẫu | 0,556575 | 0,380043 | 0,325622 |
| Lịch sử rỗng, 19.399 mẫu | 0,259498 | 0,168247 | 0,140295 |
| Zero-shot | 0,292800 | 0,201335 | 0,173013 |
| Extreme cold, 1–5 tương tác train | 0,304600 | 0,212453 | 0,183888 |
| Cold, 6–20 tương tác train | 0,333800 | 0,219677 | 0,185317 |
| Warm, trên 20 tương tác train | 0,718800 | 0,475845 | 0,400753 |

Mỗi regime có 10.000 mẫu. Đây là sampled ranking, không phải xếp hạng toàn catalog. Không đưa train_loss epoch 5 vào bảng vì log sau resume chỉ tính phần batch còn lại. Không coi thời lượng trong completed.json là tổng thời gian hai phiên train.

Các baseline cổ điển đã có kết quả test ở README trước chiến dịch này. Riêng transformer GPU và UFM hiện chưa đánh giá test. Không dùng bảng test lịch sử để tuning UFM; cần chốt checkpoint bằng validation và so sánh lại theo cùng criterion trước kết luận mô hình tốt hơn.

### 4. Kiến trúc UFM giữ đầy đủ proposal

- Foundation Model: frozen CLIP ViT-B/32, text và image 512 chiều; học adapter. Không tuyên bố đã dùng LoRA hoặc fine-tune backbone.
- Collaborative: embedding ID train-visible và Transformer lịch sử causal, tối đa 20 sản phẩm; mask ID cold/padding và ID dropout 0,5.
- Semantic: tổng hợp đặc trưng văn bản/ảnh, mask modality thiếu; không điền embedding ảnh giả để vượt kiểm tra.
- Uncertainty: proxy độ tin cậy học từ residual; không phải posterior Bayesian hay khoảng tin cậy đã được chứng minh calibration.
- UGAF: alignment giữa hai nhánh và fusion có trọng số độ tin cậy. Loss BPR + 0,01 alignment + 0,01 uncertainty + 0,1 sampled BCE.

Cấu hình đã chốt: dim 128, 2 lớp/4 heads, batch 256, 8 negatives/mẫu, AdamW lr 0,0003, weight decay 0,0001, AMP, seed 42; tối đa 20 epoch, patience 3. Chọn best bằng trung bình NDCG@10 của ba nhóm zero-shot/extreme-cold/cold trên validation, không bằng warm hoặc test.

<!-- pagebreak -->

## Chiến dịch chạy nền và sản phẩm đã làm thêm

### 5. Thứ tự chạy đã bật trên FITLAB

CLIP đầy đủ → kiểm tra nguồn/cache/coverage → GPU smoke 5 steps có ngắt ở step 2 và resume → full UFM → 7 ablation độc lập. Tại snapshot, GPU smoke UFM và full UFM chưa hoàn tất; smoke encoder CLIP 128 text/128 image đã PASS trước đó.

| Tiến trình | Trạng thái | Giới hạn đã đặt |
|---|---|---|
| CLIP, PID 115890 | Đang trích toàn catalog | Không dùng cache partial để train |
| UFM queue, wrapper PID 146362 | Chờ features đầy đủ và GPU rảnh | Chờ tối đa 168 giờ |
| Ablation suite, wrapper PID 146364 | Chờ full UFM và GPU rảnh | Cửa sổ khởi chạy 7 ngày |

PID chỉ có giá trị trong phiên container hiện tại. Bảy biến thể: no uncertainty, fixed fusion, no cross align, semantic only, collaborative only, text only, image only. Mỗi lượt học từ đầu, không mượn best.pt full model; dùng cùng candidates, budget, seed, criterion và provenance. No cross align bỏ alignment trong fusion; cosine consistency loss vẫn giữ theo cấu hình chung.

Suite hết cửa sổ sẽ không khởi chạy lượt mới; lượt đang chạy được để hoàn tất. Một seed chưa chứng minh độ ổn định thống kê. Không ép chạy đủ ngày khi validation đã dừng sớm. Không cam kết tất cả lượt hoàn tất trong 7 ngày vì GPU trường dùng chung và tải ảnh phụ thuộc mạng.

### 6. Kiểm soát an toàn, khôi phục và chất lượng

- Khóa flock ngăn runner trùng; chỉ chuyển job khi không còn extractor/trainer và GPU rảnh hai lần kiểm tra cách nhau 30 giây, tối thiểu 8 GiB VRAM trống. Không tắt job của người khác.
- Checkpoint lưu model, optimizer, scaler, RNG, batch cursor và tổng loss mỗi 1.000 steps/mỗi epoch; kiểm tra SHA/config/version trước resume. Không sửa mã trainer giữa chiến dịch.
- Graph train-only có 4.230.848 mẫu, 1.626.421 người dùng đủ lịch sử, 5.835.970 cạnh positive. Negative loại toàn bộ positive train của user liên quan, không chỉ 20 sản phẩm lịch sử.
- 101.424/101.888 ảnh đã xử lý thành công; 456 URLError, 4 thiếu URL, 4 HTTP 404. Tỷ lệ thành công hiện 99,54%, chưa phải coverage của catalog hoàn tất. Full-cache checks yêu cầu đủ số dòng và ngưỡng phủ trước tối ưu UFM.

Đóng tab hoặc tắt máy cá nhân không trực tiếp dừng job đã tách nền. Tuy nhiên trường thu hồi/tắt container sẽ dừng tiến trình; cache/checkpoint trong iDragonCloud được giữ để khôi phục, còn Python isolated ở /tmp cần dựng lại. Runner chưa tự khởi động sau reboot. Metadata là snapshot hiện tại và text CLIP bị giới hạn token; cần trình bày hạn chế này, không tuyên bố mô phỏng metadata lịch sử hoàn hảo.

### 7. Demo và kiểm thử đã hoàn thành

Demo CPU tại http://127.0.0.1:8765 dùng TF-IDF train-only thật trên 767.045 sản phẩm, không tranh GPU và không giả là UFM đã train. Đã kiểm tra tìm ASIN B006GBITXC, chọn lịch sử và Top 5: trả gợi ý thật, không gợi ý lại sản phẩm lịch sử. Backend UFM đã viết, chỉ nhận checkpoint production hoàn tất cùng cache/SHA hợp lệ; chưa chạy với full UFM.

32 kiểm thử local PASS; 32 kiểm thử FITLAB PASS trong 10,468 giây, Python 3.12.3/PyTorch 2.12.1+cu130. Kiểm tra bao gồm negative exclusion, mask/cold IDs, partial/tampered cache, resume, ablation protocol, demo HTTP và chống input/origin sai. Đây là correctness tests, không phải metric chất lượng hay GPU benchmark.

<!-- pagebreak -->

## Phần còn lại và tiêu chí hoàn thành đồ án

### 8. Kế hoạch theo mốc kết quả

| Mốc tiếp theo | Công việc cần làm | Điều kiện nghiệm thu |
|---|---|---|
| Sau CLIP | Audit cache, GPU smoke/resume và full UFM | Full marker hợp lệ; loss/gradient hữu hạn; best checkpoint và validation |
| Sau UFM | 7 ablation, đối chứng MF/SASRec/BERT4Rec chuẩn | Cùng split/candidates/criterion; bảng theo bốn regime và lịch sử rỗng |
| Kiểm tra độ tin cậy | Nhiều seed, calibration trên validation riêng, uncertainty analysis | Mean/std; ECE/NLL/Brier và reliability plots, nêu rõ sampled probabilities |
| Hoàn thiện phạm vi proposal | Dataset thứ hai và kiểm tra ngân sách VRAM 16GB | Kết quả tái lập, cấu hình/peak memory/latency có bằng chứng |
| Chốt mô hình | Khóa config bằng validation rồi đánh giá test cuối | Không tuning tiếp trên test; phân tích lỗi và giới hạn |
| Sản phẩm bảo vệ | Nối demo UFM, biểu đồ, báo cáo cuối kỳ, slide | Demo checkpoint thật; tài liệu khớp số liệu và repository |

Không gán tỷ lệ hoàn thành tổng thể tùy ý: phần dữ liệu, baseline và nền tảng phần mềm đã có; bằng chứng thực nghiệm UFM đầy đủ, multi-seed, dữ liệu thứ hai và test cuối còn thiếu. Cần trao đổi với giảng viên về diễn giải uncertainty/UGAF và các tiêu chí thực nghiệm, nhưng không tự cắt bỏ Foundation Model hoặc hai thành phần này.

### 9. Theo dõi, bàn giao và truy vết

Trong thư mục dự án FITLAB, chạy python src/monitor_ufm_campaign.py để đọc trạng thái. Monitor chỉ đọc, không cấp phát GPU, không mở test. Khi gặp stopped_with_error, xem log rồi sửa nguyên nhân; không bỏ audit hoặc tạo kết quả thay thế. Xem docs/LONG_RUNNING_EXPERIMENTS.md để phục hồi an toàn và vận hành demo.

Mã nguồn được quản lý trong repository private github.com/anky06-ky/UFM-Rec. Chỉ đưa code, tests, notebook, tài liệu và báo cáo nhỏ lên GitHub; không đưa dữ liệu Amazon raw, embeddings, checkpoints, môi trường ảo hoặc thông tin đăng nhập. Bản báo cáo này là snapshot, không tự cập nhật khi lượt train sau đó tiến lên.

Các nguồn bằng chứng trong dự án:

- data/processed/toys_games_full_temporal/split_manifest.json và foundation_catalog_v1/catalog_report.json: nguồn, lọc, split và mapping.
- runs/content_transformer_v1/completed.json, history.jsonl, validation_epoch_005.json: đối chứng GPU 8 epoch và metric validation epoch 5.
- foundation_clip_b32_v1/progress.json và ufm_positive_graph_h20_v1/complete.json: tiến độ CLIP và graph train-only.
- runs/ufm_training_queue_v1.json, ufm_ablation_suite_v1.json và các log tương ứng: heartbeat, lượt chờ, giới hạn chạy.
- runs/ufm_cpu_checks_campaign_20260928.json, tests/ và src/: bằng chứng kiểm thử và mã triển khai. Dữ liệu runs chỉ nằm trên FITLAB, không trong GitHub.

Báo cáo cũ ngày 28/09/2026 được giữ nguyên để lưu lịch sử. Bản cập nhật này thay mô tả “4 epoch, dừng giữa epoch 5” bằng kết quả khôi phục thực tế, đồng thời tách rõ “đã viết mã”, “đang trích đặc trưng”, “đang chờ huấn luyện” và “đã có kết quả”.
