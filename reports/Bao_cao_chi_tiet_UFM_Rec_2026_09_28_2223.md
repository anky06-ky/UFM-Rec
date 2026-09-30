# BÁO CÁO TIẾN ĐỘ CHI TIẾT

## UFM-Rec: hệ khuyến nghị sản phẩm cold-start

Snapshot FITLAB: 22:23:24 ngày 28/09/2026, UTC+07:00. Báo cáo có sơ đồ pipeline, kiến trúc mô hình, số liệu vận hành, kết quả đã xác nhận và kế hoạch hoàn thành. Đây là bản mới; các báo cáo trước được giữ nguyên.

### 1. Kết luận nhanh

CLIP vẫn đang trích đặc trưng thật trên FITLAB, đạt 187.648/767.045 sản phẩm (24,46%). Hai hàng đợi UFM và ablation còn hoạt động, có heartbeat mới. Không phát hiện trạng thái stopped_with_error trong ba status đang đọc. Chưa có checkpoint, epoch hoặc kết quả GPU UFM; không được coi tiến độ CLIP là tiến độ huấn luyện UFM.

| Hạng mục | Trạng thái tại snapshot | Số liệu/bằng chứng |
|---|---|---|
| Dữ liệu temporal lớn | Hoàn tất | 11.572.689 tương tác; 767.045 sản phẩm |
| Đối chứng GPU TF-IDF + Transformer | Hoàn tất 8 epoch | 132.216 steps; best ở epoch 5 |
| CLIP frozen text/image | Đang chạy | 187.648 dòng đã commit, còn 579.397 |
| Graph positive train-only | Hoàn tất | 4.230.848 mẫu; 5.835.970 cạnh |
| GPU smoke/resume UFM | Đang chờ CLIP đầy đủ | 0 epoch; latest.pt chưa tạo |
| Full UFM | Chưa tối ưu GPU | Queue đang chờ features và GPU |
| 7 ablation độc lập | Runner sống, chưa train biến thể | completed_variants rỗng |
| Demo nghiên cứu | Đã có demo CPU TF-IDF | UFM backend chờ checkpoint thật |

### 2. Tiến độ tăng từ lần kiểm tra trước

| Snapshot | Sản phẩm đã xử lý | Tỷ lệ catalog | Ghi chú |
|---|---:|---:|---|
| 17:55, báo cáo trước | 101.888 | 13,28% | Snapshot PDF cũ |
| 18:02, kiểm tra cuối lượt trước | 104.128 | 13,58% | Mốc đã thông báo người dùng |
| 22:23, báo cáo này | 187.648 | 24,46% | progress.json thật trên FITLAB |

Tăng 85.760 sản phẩm so với báo cáo 17:55, hoặc 83.520 so với mốc 104.128. Phần trăm trên chỉ là mức hoàn thành extraction catalog, không phải tỷ lệ hoàn thành đồ án hoặc số epoch UFM.

### 3. Thời gian và tài nguyên

Log ghi elapsed_s = 33.809,9 giây (khoảng 9,39 giờ). Cửa sổ 100 batch gần nhất kéo dài 1.224,9 giây, tốc độ khoảng 5,225 sản phẩm/giây. Nếu tốc độ này giữ nguyên, 579.397 sản phẩm còn lại cần khoảng 30,8 giờ. Đây chỉ là ước lượng có điều kiện cho CLIP, không gồm thời gian full UFM/ablation và không phải cam kết hoàn tất.

GPU Quadro RTX 6000: tổng 24.576 MiB, tại một lần lấy mẫu dùng 3.490 MiB và utilization 0%. Đây là số tức thời trên GPU dùng chung, không phải peak VRAM. 0% tại một lần lấy mẫu không chứng minh job dừng: log tiếp tục tăng, extractor còn sống và quá trình còn tải/xử lý ảnh. Chưa chẩn đoán hay tối ưu throughput trong lượt kiểm tra này.

<!-- pagebreak -->

## Dữ liệu và pipeline đầu-cuối

### 4. Sơ đồ pipeline thực tế

![Sơ đồ pipeline UFM-Rec theo trạng thái 22:23](../output/figures/UFM_pipeline_2026_09_28_2223.png)

Hình 1. Xanh: đã hoàn tất; vàng: đang chạy; xám: đang chờ hoặc chưa thực nghiệm. Sơ đồ thể hiện dữ liệu và các điều kiện chuyển giai đoạn, không tuyên bố tất cả khối đã hoàn tất. Demo TF-IDF là sản phẩm riêng đã kiểm tra trước đó; demo UFM chờ checkpoint production.

### 5. Chính sách dữ liệu và chống leakage

Nguồn Amazon Reviews 2023, Toys and Games: 16.260.406 bản ghi raw. Giữ verified purchase và rating >= 4; loại 121.726 bản ghi trùng chính xác. Còn 11.572.689 tương tác, 6.158.534 người dùng.

| Tập | Số tương tác | Tỷ lệ | Thời gian UTC |
|---|---:|---:|---|
| Train | 9.258.647 | 80,004% | Trước 11/08/2021 |
| Validation | 1.159.827 | 10,022% | 11/08/2021 đến trước 16/07/2022 |
| Test | 1.154.215 | 9,974% | Từ 16/07/2022 |

Split temporal gần đúng 80/10/10 tại ranh giới ngày, không phải item-disjoint. Cùng sản phẩm/ảnh có thể xuất hiện ở nhiều tập. Zero-shot là 0 tương tác train; extreme cold là 1-5, cold 6-20, warm >20. Không dùng count tương lai hoặc metadata rating_number thay train_count.

Lịch sử positive chỉ trước mục tiêu; cùng timestamp không thấy nhau. Pipeline giữ 50 sự kiện, UFM lấy 20 gần nhất. Negative loại toàn bộ positive train của user, kể cả ngoài history 20; không dùng nhãn validation/test để lọc. Validation dùng lịch sử đã xảy ra theo online-style, không tạo gradient.

<!-- pagebreak -->

## Kiến trúc mô hình giữ đầy đủ phạm vi

### 6. Semantic + Collaborative + Dual uncertainty + UGAF

![Kiến trúc UFM-Rec đã triển khai trong mã](../output/figures/UFM_architecture_2026_09_28_2223.png)

Hình 2. Kiến trúc trong src/ufm_model.py, không phải mô hình đã train xong. Foundation encoder đóng băng; adapter, nhánh tuần tự, uncertainty, alignment và scoring được học bằng train. Missing-modality và cold-ID masks quyết định nhánh nào khả dụng.

### 7. Diễn giải từng thành phần

- Foundation: CLIP ViT-B/32 text và image, ghim revision; mỗi modality 512 chiều. Chuẩn hóa vector, lưu float16 với padding row 0. Text CLIP giới hạn 77 token; không có LoRA hoặc fine-tuning backbone trong cấu hình này.
- Semantic: nối text/image với hai modality flags, qua adapter về 128 chiều; tổng hợp lịch sử semantic và tạo pair vector theo candidate. Không thay ảnh thiếu bằng embedding giả.
- Collaborative: ID embedding chỉ cho item đã xuất hiện trong train, positional embedding và causal Transformer 2 lớp/4 heads. Unknown ID được mask trước lookup, tránh dùng trọng số chưa học; ID dropout 0,5.
- Uncertainty: hai MLP + Softplus tạo u_CF và u_sem; khớp residual sigmoid bình phương đã detach trên sampled labels. Đây là learned reliability proxy, không tách epistemic/aleatoric hoặc posterior Bayesian.
- UGAF: chuẩn hóa exp(-u) trên nhánh khả dụng; kết hợp weighted CF/semantic và gated cross-alignment. Scoring head tạo điểm xếp hạng; nếu không còn tín hiệu, fallback popularity train-only.

Loss: BPR + 0,01 cosine consistency + 0,01 uncertainty residual regularization + 0,1 sampled BCE. Hệ số là cấu hình thực nghiệm đã chốt, chưa được chứng minh tối ưu. BCE/ECE/NLL/Brier trên candidate set không phải xác suất mua hàng thực tế. Calibration validation-only và phân tích reliability vẫn còn cần làm.

<!-- pagebreak -->

## Chất lượng đặc trưng và kết quả đã có

### 8. Catalog đa phương thức và lỗi ảnh

Catalog giữ đúng mapping 767.045 sản phẩm, khớp metadata sau khi quét 890.874 dòng. Có text: 767.044; có URL ảnh: 767.007; có cả hai: 767.006. Có URL không bảo đảm ảnh tải/giải mã thành công. Chỉ dùng ảnh sản phẩm, không dùng ảnh review.

| Kết quả ảnh ở 187.648 dòng đã commit | Số lượng | Diễn giải |
|---|---:|---|
| Thành công | 186.978 | 99,64% phần đã xử lý |
| URLError | 653 | Nhóm lỗi URL/mạng, chưa phân tích sâu |
| Không có URL | 7 | Modality ảnh được mask |
| HTTP 404 | 10 | Tài nguyên không tồn tại tại lúc tải |
| Tổng thiếu/lỗi ảnh | 670 | 0,36% phần đã xử lý |

Độ phủ này chưa đại diện catalog hoàn tất hoặc từng regime. Trước train, auditor phải xác nhận đủ 767.045 item, limit=0, nguồn/hash/mapping/dimensions đúng, feature hữu hạn và normalized/masked; ngưỡng text >=99%, image >=90%. Cache partial hoặc smoke không đủ điều kiện. Metadata là snapshot có thể chứa chỉnh sửa sau thời điểm tương tác; chưa có bằng chứng metadata lịch sử hoàn hảo.

### 9. Đối chứng GPU hoàn tất, best epoch 5

Two-tower TF-IDF + Transformer hoàn tất 8 epoch; best chọn bằng NDCG@10 của nhóm có lịch sử. Validation cố định 40.000 mẫu, mỗi mẫu 1 positive + 99 negatives, mỗi regime 10.000 mẫu.

| Nhóm validation | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Toàn bộ (40.000) | 0,412500 | 0,277327 | 0,235743 |
| Có lịch sử (20.601) | 0,556575 | 0,380043 | 0,325622 |
| Lịch sử rỗng (19.399) | 0,259498 | 0,168247 | 0,140295 |
| Zero-shot | 0,292800 | 0,201335 | 0,173013 |
| Extreme cold | 0,304600 | 0,212453 | 0,183888 |
| Cold | 0,333800 | 0,219677 | 0,185317 |
| Warm | 0,718800 | 0,475845 | 0,400753 |

Đây là sampled validation, không phải test/full-catalog ranking và không phải kết quả UFM. Warm tốt hơn cold trong đối chứng này; chưa có bằng chứng uncertainty/UGAF cải thiện. Criterion của UFM là cold-macro NDCG nên cần tái so sánh công bằng, không chọn winner chỉ bằng hai metric khác tiêu chí.

Baseline cổ điển đã có test trong README trước chiến dịch GPU. Riêng transformer GPU và UFM chưa có test_metrics; lượt này không đánh giá test hoặc tuning từ kết quả test cũ. Không báo train_loss epoch 5 như mean toàn epoch vì phiên resume chỉ log phần batch còn lại; không lấy elapsed_seconds của phiên khôi phục làm tổng thời gian hai phiên.

<!-- pagebreak -->

## Cấu hình thực nghiệm, ablation và vận hành

### 10. Full UFM và quy tắc chọn mô hình

4.230.848 mẫu train có lịch sử; graph có 1.626.421 user liên quan, 5.835.970 cạnh positive và 767.045 item trong mapping. Batch 256, dim 128, history 20, 8 negatives/mẫu; AdamW lr 0,0003, weight decay 0,0001; dropout 0,1, ID dropout 0,5; AMP; seed 42; tối đa 20 epoch, patience 3.

Chọn best bằng trung bình NDCG@10 của zero-shot/extreme-cold/cold trên toàn bộ validation sample 40.000. Báo warm, overall và empty-history riêng. Trước full run: GPU smoke 5 optimizer steps, ngắt có kiểm soát ở step 2, resume và audit. Smoke không phải kết quả đồ án; tại snapshot smoke UFM chưa bắt đầu.

### 11. Bảy ablation đã xếp hàng

| Biến thể | Thay đổi so với full | Mục đích |
|---|---|---|
| no_uncertainty | Learned gate thay reliability; bỏ mục tiêu uncertainty | Kiểm tra giá trị của uncertainty proxy |
| fixed_fusion | CF/semantic có trọng số 0,5/0,5 khi cả hai khả dụng | So sánh fusion cố định với gate bất định |
| no_cross_align | Bỏ gated alignment trong fusion | Kiểm tra thành phần alignment trực tiếp |
| semantic_only | Nhánh CF không khả dụng | Đo đóng góp semantic Foundation |
| collaborative_only | Nhánh semantic không khả dụng | Đo đóng góp ID/tuần tự |
| text_only | Mask modality image | Đo giá trị bổ sung của ảnh |
| image_only | Mask modality text | Đo giá trị bổ sung của văn bản |

Mỗi biến thể học từ đầu, không dùng best.pt full model để gọi đó là ablation training. Inherit cùng split/candidates/config/budget/seed/selection/provenance. No cross align vẫn giữ cosine consistency loss theo cấu hình chung: đây không phải phép bỏ mọi alignment loss. Fixed fusion dùng weight=1 khi chỉ một nhánh khả dụng; fallback được giữ nhất quán.

### 12. Trạng thái runner và phục hồi

Extractor PID 115890 vẫn sống. UFM wrapper/child 146362/146363 chờ đủ features và GPU; heartbeat 22:23:22. Ablation wrapper/child 146364/146365 chờ full UFM và GPU; heartbeat 22:23:21. Queue UFM dùng max-wait-hours=168; suite max-days=7. PID chỉ có giá trị trong phiên container này. Status Foundation có timestamp lúc launch 12:58, không phải heartbeat định kỳ; log/cursor mới là bằng chứng tiến độ extraction.

flock ngăn runner trùng; chuyển job sau hai lần GPU-rảnh cách nhau 30 giây, ít nhất 8 GiB VRAM trống và không còn extractor/trainer. GPU utilization=0% một thời điểm chưa đủ để bỏ qua active PID. Hết cửa sổ suite sẽ không khởi động lượt mới; lượt đang chạy được để hoàn tất, không bị ép kill. Một seed chưa thay multi-seed và không cam kết xong toàn bộ trong 7 ngày.

Checkpoint mỗi 1.000 steps/mỗi epoch có model, optimizer, scaler, RNG, batch cursor và loss tích lũy; resume kiểm SHA/config/version. Không sửa trainer/model giữa chiến dịch. Sleep máy cá nhân không trực tiếp dừng job tách nền; trường tắt/thu hồi container vẫn dừng job. Python isolated ở /tmp cần dựng lại nếu container mất; runner chưa tự khởi động sau reboot.

<!-- pagebreak -->

## Sản phẩm bàn giao và phần còn thiếu

### 13. Những sản phẩm đã có

- Pipeline lọc/split/history, manifest và candidate set dùng chung; TF-IDF fit trên text train-visible rồi transform catalog với vocabulary/IDF đóng băng.
- Popularity/content/SVD/hybrid cổ điển; đối chứng TF-IDF Transformer GPU; lõi UFM đầy đủ, extractor và queue an toàn.
- Demo CPU TF-IDF đã thử với catalog thật: tìm ASIN, chọn lịch sử, Top K, tên/ảnh, train_count và regime; loại item trong history. Backend UFM đã viết nhưng chưa phục vụ checkpoint full vì checkpoint chưa tạo. Phiên kiểm tra này chưa chạy lại demo trên máy cá nhân.
- 32 correctness tests trước đó PASS cả local và FITLAB; record được đọc lại, không rerun trong lượt này. Đây không phải model-quality metric, full extraction audit hoặc GPU benchmark.
- Code đã push trong commit b2e1f18 ở repository private UFM-Rec. Báo cáo lần này gồm hai sơ đồ, Markdown và snapshot JSON; chưa push bản cập nhật báo cáo trong lượt kiểm tra này.

### 14. Kế hoạch hoàn thành giữ đầy đủ phạm vi

| Mốc | Công việc còn lại | Bằng chứng cần đạt |
|---|---|---|
| Đặc trưng đầy đủ | Hoàn tất CLIP, audit cache/mask/coverage | Marker full và hash hợp lệ; độ phủ theo regime |
| Mô hình chính | GPU smoke/resume và full UFM | Steps/epochs, best.pt/latest.pt, losses và validation |
| So sánh | 7 ablation, MF/BPR, SASRec/BERT4Rec chuẩn, fixed/concat hybrid | Cùng protocol; bảng breakdown cold/warm/empty-history |
| Độ tin cậy | Nhiều seed, calibration-validation riêng, uncertainty/gate analysis | Mean/std, ECE/NLL/Brier, reliability plots và phân tích lỗi |
| Phạm vi proposal | Dataset thứ hai, adapter/frozen-feature đối chứng, chi phí mô hình | Tái lập kết quả; mức kiểm chứng H5 và giới hạn modality |
| Tài nguyên | Đo peak VRAM, params, throughput/latency; xác minh 16 GiB | Số đo thật; 24 GiB chạy được không đủ chứng minh 16 GiB |
| Chốt & bảo vệ | Khóa config theo validation, test cuối, demo UFM, biểu đồ, báo cáo và slide | Không tune lại trên test; checkpoint/demo/số liệu thống nhất |

Chưa có hạn nộp hoặc bằng chứng giảng viên duyệt diễn giải triển khai. Cần xác nhận uncertainty proxy, UGAF và tiêu chí đánh giá với giảng viên; không tự cắt Foundation Model/uncertainty/UGAF. Không gán phần trăm hoàn thành tổng thể tùy ý và chưa tuyên bố mô hình tốt nhất hay cải thiện 3%-8% theo kỳ vọng proposal.

### 15. Theo dõi và nguồn kiểm chứng

Terminal 3 (tail) trong Chrome FITLAB theo dõi runs/foundation_queue_v1.log; Ctrl+C tại terminal tail chỉ dừng xem log. python src/monitor_ufm_campaign.py xem toàn chiến dịch. Gặp stopped_with_error: đọc log trước khi phục hồi; không chạy job trùng, bỏ audit hoặc tạo cache giả.

Nguồn số liệu: split_manifest.json; content/content_report.json; foundation_catalog_v1/catalog_report.json; foundation_clip_b32_v1/progress.json; ufm_positive_graph_h20_v1/complete.json; validation_epoch_005.json/history.jsonl/completed.json của content_transformer_v1; ba queue status và log; ufm_cpu_checks_campaign_20260928.json. Snapshot gọn kèm báo cáo lưu trong reports/FITLAB_snapshot_2026_09_28_2223.json.

SHA256 trainer/model không đổi; full hash trong snapshot JSON. Không restart job, sửa source/config GPU hoặc đánh giá test trong lượt này. Báo cáo/GitHub không chứa dữ liệu lớn, venv hay thông tin đăng nhập.
