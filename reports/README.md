# Mục lục bằng chứng UFM Rec

**Trạng thái hiện tại:** [hồ sơ hoàn thiện 03/11](final_20261103/README.md) (bộ nộp, dọn workspace, việc còn lại). Mốc trước: [phần ngoài train 29/10](nontraining_20261029/status.md), [checklist 29/10](project_readiness_20261029.md), [nhật ký dọn 29/10](workspace_cleanup_20261029.md). Các snapshot bên dưới không xác nhận job/dịch vụ đang sống.

## Bằng chứng đang dùng

- [Validation v3 ngày 28/10](continuation_20261028/README.md): 40.000 ca, 100 candidates, seed 42; checkpoint lịch sử, phân tích thăm dò.
- [Bảng bằng chứng đã chuẩn hóa](nontraining_20261029/evidence.json): cùng candidates/source, đủ 10 mô hình; các biểu đồ và báo cáo hiện tại dùng nguồn này.
- [Ghép UFM + TF-IDF ngày 03/11](hybrid_20261103/hybrid_v3.json): không huấn luyện lại; điểm tính lại khớp số v3 đã công bố, α chọn trên 20.000 ca sớm, kết quả trên 20.000 ca muộn. Dựng bằng `scripts/evaluate_hybrid_v3.py`.
- [API Windows 6/6 ngày 29/10](nontraining_20261029/demo_api_verification.json) và [history toàn unknown](nontraining_20261029/demo_unknown_history.json).
- [Kiểm thử local](nontraining_20261029/pytest.txt) · [kiểm tra PDF](nontraining_20261029/pdf_verification.json).
- [Bàn giao backup](continuation_20261028/backup_handoff.md) · [receipt dữ liệu Colab local](continuation_20261028/colab_data_transfer_verification_20261029.json). Receipt local không thay nghiệm thu trên GPU.
- [Ledger test](test_evaluation_ledger.json): test v2 đã được xem; không gọi v3 là test độc lập.

## Hồ sơ lịch sử giữ riêng

- [Evaluation v2 ngày 27/10](ufm_evaluation_v2_20261027/README.md): UFM đã có validation và test, candidates khác v3.
- [Audit validation cũ](ufm_validation_audit_v2/report.md): regime/history, bootstrap theo user và calibration; tên audit v2 không đồng nghĩa bộ candidates evaluation_v2.
- [BPR-MF nhiều seed](BPR_MF_multiseed_2026_09_30.json) và [báo cáo baseline](BPR_MF_validation_2026_09_30.md).
- [All Beauty](second_dataset_preparation.json): dữ liệu và baseline tham chiếu, chưa có UFM xác minh.
- [Demo UFM 03/10](demo_acceptance_20261003.json), [snapshot FITLAB 03/10](campaign_snapshot_20261003.json), [demo baseline 30/09](demo_acceptance_2026_09_30.json).
- [Tiến độ tổng hợp trước đó](progress.md), [diễn biến 01–02/10](../docs/history/Progress_UFM_2026-10-02.md), [baseline 30/09](../docs/history/Bao_cao_tien_trinh_UFM_Rec_2026_09_30.md).
- [PDF hiện trạng](../output/history/pdf/UFM_Rec_Bao_Cao_Hien_Trang_20261029.pdf) và [mục lục các bản xuất](../output/README.md).

Giữ snapshot mã nguồn, manifest và report dù một số file có SHA trùng nhau: chúng ghi nhận nguồn đã chạy ở từng thời điểm và từng seed. Không ghép số liệu khác giao thức vào một bảng hoặc xóa bản snapshot chỉ vì giống code hiện tại.
