# Hiện trạng đồ án UFM Rec ngày 29/10/2026

Rà từ commit `4b72a2de05e16a14a068b26389bdf3d45e2b8264`. Theo yêu cầu của nhóm, chưa chạy train mới và chưa tạo commit/push. Tài liệu này tách rõ bằng chứng đã có khỏi việc phải hoàn thành trước khi nộp.

## Đã hoàn thành

1. **Số liệu hiện trạng:** [JSON bằng chứng](evidence.json) khóa cùng candidate hash, 40.000 ca validation v3, bốn regime mỗi nhóm 10.000 ca; [bảng 10 mô hình](ranking_table.md) và hai [hình](../../output/figures/current_v3_20261029/) lấy từ dữ liệu thật. Không gộp v1, v2 và v3.
2. **Báo cáo hiện trạng:** [Markdown](../../docs/final_report.md), [Word 12 trang](../../output/history/docx/UFM_Rec_Bao_Cao_Hien_Trang_20261029.docx) và [PDF 11 trang](../../output/history/pdf/UFM_Rec_Bao_Cao_Hien_Trang_20261029.pdf) đã đồng bộ kết quả chính, giới hạn và tài liệu tham khảo. Word có công thức sửa được; cả Word và PDF đã render, rà từng trang. Đây là bản đọc/rà soát, còn cần mẫu trường và kết quả cuối.
3. **Slide bảo vệ hiện trạng:** [PPTX 14 slide](../../output/history/presentation/UFM_Rec_Bao_Ve_Hien_Trang_20261029.pptx) có bảng và biểu đồ sửa được; package integrity và layout đều đạt, từng slide đã được xem. [Receipt Word/PDF/PPTX](submission_artifacts_20261029.json) ghi kích thước, SHA và số trang/slide đã rà. [Nguồn nội dung](../../docs/presentation_slides.md), [script xuất](../../scripts/build_current_slides.mjs). Slide chưa phản ánh train mới.
4. **Demo backend thật:** [6/6 ca API](demo_api_verification.json) đạt trên checkpoint epoch 9 và catalog 767.045 sản phẩm. [History toàn unknown](demo_unknown_history.json) trả HTTP 400 như thiết kế, không phải gợi ý thành công.
5. **Demo UI thật:** [4/4 ca Chrome headless](demo_ui_end_to_end_20261029.json) đạt: chọn item và zero-shot, tải [JSON](demo_ui_downloads/ufm-rec-demo.json) khớp request/model/Top-5, ảnh lỗi vẫn có placeholder, mất mạng hiện lỗi. [Ảnh zero-shot](demo_ui_zero_shot_20261029.png) và [ảnh lỗi mạng](demo_ui_offline_20261029.png) dùng dự phòng. Đây là nghiệm thu tự động trên máy Windows hiện tại; chưa thay thử trên máy bảo vệ.
6. **Hướng dẫn bảo vệ:** [kịch bản 8 phút và demo 3 phút](../../docs/defense_guide.md), câu hỏi/đáp và đường dẫn chứng cứ đã có. Nhóm chưa phân công và diễn tập có bấm giờ.
7. **Kiểm tra và dọn workspace:** `116 passed, 3 skipped, 10 subtests passed` trong [receipt pytest](pytest_receipt.json). Đã chuyển 3.945 file tạm/cache và 4 PDF cũ vào nơi lưu ngoài repo/lịch sử; xem [nhật ký dọn](../workspace_cleanup_20261029.md). Lệnh xóa vĩnh viễn bị kiểm duyệt tự động chặn, nên các file tạm được lưu ngoài repo, không tính là giải phóng dung lượng.

## Còn phải làm để chốt kết quả đồ án

### Quyết định và train đã chủ động để lại

1. **Chốt với giảng viên** metric, tập đánh giá cuối sau khi test v2 đã được xem, số seed/đối chứng tối thiểu và All Beauty có bắt buộc hay không. Phạm vi hành vi, ảnh, tên/mô tả đã được chốt; không cần hỏi lại.
2. **Nghiệm thu Colab/GPU:** kiểm SHA dữ liệu và CLIP cache trên Drive, chạy smoke CUDA, đo throughput/peak VRAM, thử resume checkpoint gồm optimizer/RNG/cursor. Receipt Windows không thay receipt GPU.
3. **Chạy đủ 15 job đã khóa:** 9 lượt `full`, `semantic_only`, `fixed_fusion` và 6 lượt `count_aware_fusion`, `adaptive_dropout` trên seed 42/7/2026 theo [manifest](../../configs/multimodal_followup_20261028.json). Theo dõi loss/early stopping; sao lưu best/latest checkpoint, config, log, prediction và kiểm restore sau từng lượt.
4. **Tổng hợp kết quả:** từng seed và mean ± sample std, NDCG/Recall/MRR@10 theo bốn regime và history/no-history; kiểm cùng candidates, source và số ca. Báo cả trường hợp cải tiến không thắng.
5. **Kiểm chứng nguyên nhân:** phân tích gate/uncertainty, lỗi từng nhóm, nhất là 1–5 tương tác; thêm đối chứng tuần tự cùng protocol nếu dùng trong kết luận. Chênh lệch count hiện có chỉ là quan sát, không chứng minh ID/gate gây ra.
6. **Calibration và độ ổn định:** chỉ cập nhật cho checkpoint cuối nếu vẫn giữ tuyên bố đó; thêm seed cho ablation nếu muốn nói lợi ích uncertainty/alignment ổn định. Không chuyển nhiệt độ cũ sang model mới.

### Báo cáo, demo và nộp cuối

7. **Word theo mẫu trường:** bản Word hiện trạng đã có và đã rà 12 trang; còn phải khôi phục hoặc xin đúng mẫu/rubric UFM. `Mau_Bao_Cao_Nhom_4.docx` cũ không có; mẫu Văn Lang tìm được trên Drive không phù hợp. Chốt tên đầy đủ, MSSV, khoa, giảng viên và chuẩn trang/trích dẫn; sau đó áp mẫu, cập nhật số liệu cuối và rà lại từng trang. Bản hiện trạng chưa thay bản nộp theo mẫu.
8. **Cập nhật bộ nộp theo kết quả cuối:** thay số liệu và giới hạn trong Markdown, PDF, Word, PPTX; thống nhất README, START, proposal/progress/checklist còn dùng. Rà từng trang/slide, link, bảng, hình, chú thích, tài liệu tham khảo.
9. **Diễn tập thật:** thử launcher/restore và 4 ca UI trên máy bảo vệ, đo thời gian chờ, phân công ai trình bày/demo, diễn tập có bấm giờ. Ảnh dự phòng đã có; quay video nếu nhóm cần phương án trình diễn khi không có mạng.
10. **Tài nguyên thực tế:** đo train time/peak VRAM và số tham số trainable/frozen; benchmark inference trên cấu hình bảo vệ với history/K/tải đại diện. CPU cũ chỉ có 10 request, p50 21,25 s và p95 26,35 s; chưa đủ cho p99 ổn định hoặc khẳng định train dưới 16 GiB.
11. **Bàn giao và commit cuối:** thử môi trường sạch, kiểm manifest/SHA và lệnh tái lập; giữ dataset/checkpoint lớn ngoài Git; sau khi nhóm duyệt nội dung mới commit/push và chạy CI tại commit nộp. Chưa commit theo lựa chọn của người dùng.

### Chỉ làm nếu nằm trong yêu cầu nghiệm thu

- **All Beauty:** hiện mới có dataset/baseline; chưa có UFM xác minh. Nếu bắt buộc, cần CLIP coverage, đối chứng, seed và đánh giá riêng.
- **Tối ưu retrieval/ANN:** đo recall retrieval, chất lượng cuối và latency; không mặc nhiên thay scoring candidate dependent.
- **Robustness, coverage, diversity, novelty, popularity bias** và các encoder/LoRA mới: chỉ mở rộng khi có câu hỏi thực nghiệm và thời gian rõ. Kiểm thử chức năng thiếu modality hiện có không thay thí nghiệm chất lượng.

[Checklist chi tiết kèm tiêu chí nghiệm thu](../project_readiness_20261029.md).
