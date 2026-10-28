# Sửa và xác minh mục 1–15 — FITLAB-01, ngày 27/10/2026

Đã làm trên `/home/coder/iDragonCloud/DA_AI` của FITLAB-01. Bằng chứng thực thi ghi ngày 27/10; hoàn tất đối chiếu và bàn giao ngày 28/10/2026. Kiểm tra checkpoint, cache và baseline dùng CPU. Không huấn luyện lại mô hình hoặc chạy suy luận trên test trong đợt này. Các JSON v2 nhập trước đó được giữ nguyên byte.

## Kết quả quan trọng

- Checkpoint `ufm_full_v1/best.pt` epoch 9 tồn tại và khớp SHA-256 `5630013d…`. Code huấn luyện, marker/features, mapping và train counts đều khớp; forward CPU trên hai ca validation hữu hạn. Metric cold macro lịch sử được tính lại từ predictions đã lưu và khớp marker. Đây không phải chạy lại toàn bộ dự đoán UFM. Nguồn: [inventory](fitlab01_run1/inventory.json).
- Trên **cùng 40.000 ca validation v2**, Cold macro NDCG@10: **UFM 0,182051; TF-IDF 0,195672; popularity 0,042527**. Overall tương ứng **0,164132; 0,181498; 0,040904**. UFM chưa vượt TF-IDF; không dùng kết quả này để khẳng định cải tiến đã thành công. Nguồn: [baseline mới](fitlab01_run1/baseline_v2/metrics.json), [UFM đã có](../ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json).
- Hai bên dùng cùng candidate hash `c71906e65ce4067b405a94c13e22e8a25a17f46cbee5b1315f07d01d4d1e59df`, cùng metric/tie-breaking. TF-IDF dùng 10 item history gần nhất, UFM dùng tối đa 20; đây là so sánh hệ thống, chưa phải ablation chỉ thay một thành phần.
- Audit v2 mới phát hiện **7 negative trùng positive trước target trong 6/40.000 ca**. Audit cũ chỉ xét positive train và ghi 3; kiểm tra mới xét cả sự kiện sớm hơn trong cùng split. History/trace/replay đạt. Nguồn: [audit v2](fitlab01_run1/candidate_audit.json).
- Đã tạo validation v3 mới và audit đạt: **0/3.960.000 negative trùng positive trước target**, gồm ngoài history window. Candidate hash v3: `9387e66e9f1905c44d02b65c3809f5fa8b6c022f6abf0c1a37d52cb57767a60e`. V3 vẫn là lấy mẫu hồi cứu cùng regime; chưa phải protocol online. Chưa có metric UFM/baseline v3, không ghép điểm v2 vào v3. Nguồn: [audit v3](fitlab01_finish1/candidate_audit_v3.json), [sampling v3](fitlab01_finish1/evaluation_v3/sampling_report.json).
- Cache thực có **767.045 item, 767.044 text, 758.692 ảnh**; thiếu 1 text và 8.353 ảnh, không item nào thiếu cả hai. Đã kiểm tra ASIN/row giữa items, text, catalog và mask, thống kê theo regime. Lý do lỗi ảnh của extractor chỉ có tổng toàn tập, không đủ log để gán từng lý do cho từng regime. Nguồn: [coverage](fitlab01_finish1/modality_coverage.json).
- Các ảnh thiếu gồm 8.123 `URLError`, 149 timeout, 43 HTTP 404 và 38 không có URL. Đây là trạng thái lúc tạo cache, không phải kiểm tra lại URL hiện tại.
- **Bản mã hiện tại trên Linux: 87 passed, 0 skipped, 10 subtests passed.** Toàn bộ 111 tệp của snapshot kiểm thử khớp SHA-256 với local. Nguồn: [log Linux](fitlab01_finish1/current_source_tests_v2.log), [môi trường và kết luận](fitlab01_finish1/verification_summary.json), [manifest mã](source_manifest.json).

## Trạng thái từng mục

1. **P0-01 — Đã kiểm kê FITLAB-01.** Inventory ghi thời điểm, tệp, GPU, cgroup RAM, disk và tiến trình liên quan. Dung lượng trống của mount là số đo filesystem dùng chung, không phải cam kết quota cá nhân.
2. **P0-02 — Đã xác minh checkpoint/cache hiện có.** Không cần tạo cache hoặc train lại để khôi phục. Bản mã huấn luyện được lưu trong [trained_snapshot](fitlab01_finish1/trained_snapshot/).
3. **P0-03 — Đã sửa fixture.** Test cold-start thực sự chạy vào logic mô hình.
4. **P0-04 — Đã thay test thất bại vô điều kiện.** Kiểm tra masking, finite output, chuẩn hóa gate, count penalty và dropout; không suy ra NDCG từ mô hình ngẫu nhiên.
5. **P0-05 — Đã sửa sample builder.** Default output giải quyết lúc gọi, từ chối ghi đè, kiểm tra tập rỗng/pool thiếu, dọn output dở dang; sửa cả custom reservoir cap.
6. **P0-06 — Đã sửa audit.** Đúng history slice, timestamp, `--data`, replay dưới cap/lọc lịch sử; bổ sung fixture tính tay và audit lại dữ liệu thật.
7. **P0-07 — Đã thêm ledger test.** Import bằng chứng v2, khóa khi ghi, lưu lượt thất bại; đổi checkpoint/output không xóa lịch sử. Thời điểm chạy cũ chưa xác minh giữ null.
8. **P0-08 — Đã sửa discovery, kiểm thử Windows/Linux và cấu hình CI; chờ GitHub Actions.** Windows đạt **84 passed, 3 skipped, 10 subtests passed**; Linux **87 passed, 0 skipped, 10 subtests passed**. Ba skip trên Windows do Linux flock/procfs. Cấu hình CI chạy Python 3.12 CPU trên Ubuntu/Windows; chưa push hoặc chạy GitHub Actions trong phiên này. Xem [log local](local_tests.log), [môi trường](local_environment.json) và [log Linux](fitlab01_finish1/current_source_tests_v2.log).
9. **P1-01 — Đã soạn phạm vi, chờ giảng viên xác nhận.** Ưu tiên cold-start sản phẩm trên Toys, Cold macro là metric chính; no-history là fallback. Không thể thay nhóm/giảng viên duyệt phạm vi. Xem [protocol làm việc](../../docs/protocol_current.md).
10. **P1-02 — Đã cố định protocol kỹ thuật.** Config v2 ghi pool, weights, exclusion, seeds, hash và giới hạn hồi cứu; v3 là phiên bản riêng.
11. **P1-03 — Đã sửa lọc positive và kiểm tra thời gian.** V3 chỉ dùng sự kiện trước target; cùng timestamp/tương lai không bị dùng làm nhãn lọc. Validation audit không tự mở nhãn test.
12. **P1-04 — Đã kiểm tra mapping/mask và coverage cache thực.** Công bố missing theo regime, tổng lý do lỗi ảnh, giới hạn metadata snapshot và 77 token.
13. **P1-05 — Đã phân biệt demo/benchmark.** Demo manifest ghi synthetic counts, API/status/recommend/export có provenance, UI có cảnh báo; builder/baseline từ chối dataset có demo manifest. Ảnh trong backend TF-IDF chỉ dùng hiển thị.
14. **P1-06 — Đã khóa manifest so sánh validation và ghi test đã xem.** Manifest lưu dataset, mapping, features, candidates, checkpoint, code và selection rule; không giả vờ đây là manifest được khóa trước lần test cũ. Nguồn: [manifest](fitlab01_finish1/comparison_manifest.json).
15. **P2-01 — Đã chạy popularity và TF-IDF trên đúng validation v2.** Có ranks NPZ trên máy chủ, metrics và hash đối chiếu với UFM. BPR-MF/SVD v2 nằm ngoài hai baseline tối thiểu đã chạy, cần xác nhận phạm vi tiếp theo.

## Hai bản mã trên FITLAB-01

Workspace gốc chứa mã đã tạo checkpoint; local có thêm `count_aware_fusion`, `adaptive_dropout` và thay đổi vận hành. Vì vậy không ghi đè ba tệp được khóa theo checkpoint bằng bản local.

Bản mã sửa hiện tại và ảnh demo được đặt riêng tại `/home/coder/iDragonCloud/DA_AI/audit_v2/fix_1_15_workspace_20261027`, kiểm tra SHA-256 từng tệp trước chạy test. Chạy test bản này bằng Python `.venv_ufm_runtime/bin/python` của workspace gốc. Test chỉ dùng dữ liệu nhỏ tự tạo; không chạy training benchmark.

Lần thử trên workspace gốc ghi nhận 3 fail: hai biến thể chưa có trong mã huấn luyện cũ và một khác biệt ngưỡng GPU của queue cũ. Log được giữ, không chuyển các lỗi này thành skip. Lần đầu chuyển snapshot thiếu `status.py`, đã bổ sung rồi chạy lại đầy đủ: 87 pass. Thao tác sao chép snapshot cũng đã sửa tên `train_gpu_recommender.py`; log thử ban đầu được giữ để truy vết. Chỉ kết quả của đúng bản mã hiện tại được dùng nghiệm thu phần sửa.

## Cách nói với giảng viên

“Dạ thầy, nhóm đã xác minh checkpoint và đặc trưng đa phương thức, đồng thời chạy popularity và TF-IDF trên đúng bộ validation của UFM. Cold macro hiện UFM đạt 0,182051, thấp hơn TF-IDF 0,195672 nên nhóm chưa kết luận mô hình cải thiện. Nhóm đã sửa audit và tạo bộ mẫu mới loại positive đã biết; bộ mới cần đánh giá mô hình lại. Test v2 đã được xem, nhóm có ghi rõ lịch sử. Nhờ thầy góp ý phạm vi nghiệm thu, các đối chứng/ablation tối thiểu và cách đánh giá tiếp theo.”

Word/PDF cũ chưa được xuất lại trong phạm vi mục 1–15. Chưa push GitHub.
