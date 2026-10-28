# Đối chiếu kết quả UFM evaluation_v2 ngày 27/10/2026

Ba JSON trong thư mục này được sao chép nguyên byte từ `tmp/transfer/evaluation_v2` để giữ bằng chứng nhỏ trong repo. Các file mẫu NPZ/CSV gzip lớn vẫn ở local và bị Git bỏ qua. Ngày 27/10 là ngày đối chiếu artifact local; report không ghi thời điểm hoàn tất lượt đánh giá, nên không gán ngày đó cho thời điểm chạy mô hình.

Sau đợt dọn thư mục 28/10, thư mục chuyển tạm được lưu ngoài repo tại `D:\Nam3_ki1\UFM-Rec_Backups\workspace_cleanup_20261028\tmp\transfer\evaluation_v2`. Mỗi tệp chuyển ra đã được đối chiếu SHA; ba JSON bằng chứng tại đây được giữ nguyên.

## Kết quả và phạm vi

- Checkpoint theo report: `runs/ufm_full_v1/best.pt`, epoch 9, variant `full`, seed không được ghi riêng trong JSON v2; tên run gắn với lượt seed 42 lịch sử. Đã xác minh checkpoint/config/source/cache trên FITLAB-01 trong lượt sửa 1–15; xem [bằng chứng mới](../fix_1_15_20261027/README.md).
- SHA-256 checkpoint: `5630013d0cbb6db633c4f4cf5bcd840fb703dbfc64a13c0352278624e03e11c8`.
- Validation: 40.000 ca, Overall NDCG@10 **0,164132**, Cold macro **0,182051**.
- Test: 40.000 ca, Overall NDCG@10 **0,166193**, Cold macro **0,183638**.
- Nguồn kết quả: [reeval_ufm_full_v1_v2.json](reeval_ufm_full_v1_v2.json). Có kết quả test trong artifact, vì vậy không tiếp tục mô tả toàn dự án là “chưa mở test”. Không có bằng chứng đầy đủ về manifest khóa trước test trong bộ artifact này; đây chưa được xác nhận là lượt nghiệm thu test cuối độc lập.
- Đã bổ sung [baseline trên cùng validation v2](../fix_1_15_20261027/fitlab01_run1/baseline_v2/metrics.json): Cold macro TF-IDF 0,195672, popularity 0,042527, so với UFM 0,182051. Không so trực tiếp các số này với TF-IDF/BPR-MF/SVD lịch sử; cũng không áp dụng temperature calibration lịch sử cho v2.

## Đối chiếu tính toàn vẹn

Đã tính SHA-256 từ cả hai NPZ và hai metadata CSV gzip local; kết quả khớp report re-evaluation và candidate audit:

- Validation NPZ: `c71906e65ce4067b405a94c13e22e8a25a17f46cbee5b1315f07d01d4d1e59df`.
- Test NPZ: `a4422c2cfe420e7ffc072915b16ff88a218d640054fb107c937d867dca20b109`.
- Validation metadata: `d529b0aeb68e4b026556756601efc4380b179dc43f96e05a76eccef0843a051c`.
- Test metadata: `273501122f37028c607fbe3c4a699d5c66e073f1f66d1210c4244de831178521`.

Bộ validation lịch sử trong [audit UFM cũ](../ufm_validation_audit_v2/report.json) có hash `8efa116508fddd6b5227b69d6c65e3b635de634dd542046463fda84e78b47162`, khác v2. Tên `ufm_validation_audit_v2` chỉ phiên bản audit, không phải tên bộ mẫu `evaluation_v2`.

Hash giúp xác nhận file mẫu tương ứng với report; việc đối chiếu này không tái tạo prediction và không tự xác minh checkpoint/features có thật trên máy chủ hiện tại. Không chạy thêm train hoặc test recommendation khi sửa tài liệu.

## Giao thức và giới hạn

[sampling_report.json](sampling_report.json) ghi một positive + 99 negative, lấy theo popularity đến cuối split, loại target và history. Mô tả này chưa ghi rõ phạm vi pool. [candidate_validity_v2.json](candidate_validity_v2.json) ghi negative cùng regime với target và deterministic replay đạt; điều này khớp code `src/build_recommender_samples.py` hiện tại. Giữ nguyên JSON nguồn để không sửa bằng chứng hồi cứu.

- Validation dùng `train_count + validation_count`; test cộng thêm `test_count` làm trọng số. Regime luôn được xác định từ train count. Giao thức có dùng thống kê cuối split và regime target để dựng candidate set; không gọi là train-only hoặc full-catalog.
- Mỗi regime có 10.000 ca; Overall mô tả bộ mẫu cân bằng, không phải phân bố tương tác tự nhiên.
- Audit báo structure/exclusions/regimes/replay đạt, nhưng có 3 negative trùng positive train ngoài cửa sổ history ở validation và 7 ở test. Audit cũng đếm 46 future-positive negative ở validation và 45 ở test. Đây là chỉ số disclosure, không phải kiểm định chứng minh mọi negative là không liên quan.
- `passed=true` được giữ theo report gốc; chưa kiểm tra lại logic audit hoặc tái chạy replay trên toàn dữ liệu trong lần chỉnh tài liệu. Cần đối chiếu logic và dữ liệu nguồn trước khi dùng làm nghiệm thu cuối.
- Bản ghi 24/10 về FITLAB là trạng thái khôi phục tại thời điểm đó. Artifact nhập về sau không xác nhận trạng thái realtime của queue, GPU hoặc vị trí checkpoint.

## Việc cần xin ý kiến giảng viên

Ưu tiên chuẩn hóa candidate protocol và chạy baseline trên cùng validation v2 trước khi mở rộng dataset. Công bố rằng test v2 đã được xem; thống nhất cách khóa cấu hình và đánh giá cho các lượt tiếp theo, tránh lựa chọn mô hình theo điểm test đã biết. Nhiễu ID vẫn là giả thuyết, cần ablation và phân tích checkpoint thật.

## Kiểm tra trực tiếp sau đó trên FITLAB-01

[Bản sửa 1–15](../fix_1_15_20261027/README.md) giữ nguyên ba JSON gốc ở đây, bổ sung audit validation xét cả positive sớm hơn trong cùng split (7 negative trong 6 ca), baseline cùng hash và validation v3 riêng. Test v2 không được suy luận lại.
