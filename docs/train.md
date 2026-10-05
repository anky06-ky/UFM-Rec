# UFM-Rec: lượt train đầy đủ trên FITLAB (28/09/2026)

## Phạm vi và trạng thái khi triển khai

Giữ proposal đầy đủ: Foundation Model + hai nhánh semantic/collaborative +
uncertainty + UGAF. Không thay bằng đối chứng TF-IDF. Đối chứng cũ hoàn tất 8 epoch;
checkpoint validation tốt nhất ở epoch 5. CLIP GPU smoke 128/128 text và ảnh đạt;
trích full-catalog đang chạy, chưa có kết quả UFM toàn bộ.

Đã triển khai sáu tệp mới sau khi đối chiếu SHA256; 25 kiểm thử CPU PASS trên
FITLAB trong 9,692 giây, PyTorch 2.12.1+cu130. Queue được khởi động với wrapper
PID 117817 và xác nhận có child chuẩn bị graph PID 117822. Lúc kiểm tra, chưa
có optimizer step UFM GPU; việc trích CLIP không được gọi là đã train UFM.

`run_ufm_training_when_ready.py` chuẩn bị đồ thị positive bằng CPU ngay, sau đó chờ
CLIP đủ 767.045 sản phẩm và GPU rảnh hai lần kiểm tra. Không giết job đang chạy.
Lượt GPU UFM bắt đầu bằng smoke 5 bước: chủ động dừng ở bước 2, resume, kiểm tra
checkpoint, validation, loss và tính toàn vẹn. Smoke không được dùng làm kết quả
đồ án. Sau đó tự train `runs/ufm_full_v1`, tối đa 20 epoch, patience 3.

## Cấu hình cố định trước khi chạy

- Dữ liệu full temporal 80/10/10: 9.258.647 tương tác train; dùng 4.230.848
  mẫu train có lịch sử. Không dùng validation/test để tạo gradient.
- Frozen `openai/clip-vit-base-patch32`, revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, hai bảng 512 chiều float16.
  Adapter được học; không tuyên bố đã triển khai LoRA.
- UFM dim 128, Transformer 2 lớp/4 heads, history 20, batch 256,
  8 negatives/mẫu, ID dropout 0,5; AdamW lr 0,0003, decay 0,0001; AMP.
- `BPR + 0,01 alignment + 0,01 uncertainty + 0,1 sampled BCE calibration`.
  Uncertainty là proxy độ tin cậy học từ residual, không phải posterior Bayesian.
- Negative loại **toàn bộ positive của người đó trong train**, cả tương tác đầu
  tiên và tương tác train không nằm trong history 20. Không nhìn nhãn tương lai
  validation/test để lọc negative.
- Chọn best bằng trung bình NDCG@10 của zero-shot, extreme-cold và cold trên
  validation cố định (40.000 mẫu, mỗi nhóm 10.000). Báo cáo warm/empty-history riêng.
  Đối chứng cũ chọn bằng known-user NDCG nên cần tái so sánh bằng cùng giao thức.
- Xếp hạng sampled 1 positive + 99 negatives, không phải full-catalog ranking.
  ECE/NLL/Brier là xác suất có điều kiện trên bộ candidates; chưa temperature-fit
  và không phải xác suất mua hàng ngoài thực tế.
- Checkpoint gồm optimizer/scaler/RNG/cursor và tổng loss tích lũy. Resume từ
  batch đúng, không lặp batch cuối; kiểm tra SHA/config/code/version trước resume.
  Checkpoint định kỳ 1.000 optimizer steps và mỗi epoch; dữ liệu/checkpoint cũ giữ nguyên.

## Theo dõi

Trong terminal FITLAB, tại `/home/coder/iDragonCloud/DA_AI`:

```bash
/home/coder/iDragonCloud/.venv/bin/python ops/monitor_ufm_training.py
```

Status `runs/ufm_training_queue_v1.json`, log `runs/ufm_training_queue_v1.log`.
`waiting_for_complete_features_and_idle_gpu` nghĩa là **chưa train UFM GPU**;
`full_ufm_training_running` mới là đang train sản phẩm đầy đủ.
`stopped_with_error` cần chẩn đoán log; không tự tạo cache giả hoặc bỏ nhánh ảnh.
Giới hạn chờ ban đầu 72 giờ; đã gia hạn queue lên 168 giờ lúc khoảng 17:54 ngày
28/09/2026, khi queue chỉ đang chờ và chưa tối ưu GPU. Chỉ dừng wrapper/child
đúng argv/PGID của queue chờ cũ; không dừng extractor CLIP. Wrapper mới PID
146362; ablation suite 7 ngày wrapper PID 146364. Nếu quá giới hạn, dữ liệu và
checkpoint được giữ. Xem `ops.md` cho trạng thái chiến dịch mới.

Môi trường isolated hiện ở `/tmp/ufm_venv_20260928_cdcp1xhw/bin/python`.
Nếu container bị tạo lại, cần khôi phục môi trường trước khi chạy lại queue;
code, log, model cache, features và checkpoint ở thư mục dự án vẫn bền vững.
Không chạy trùng nhiều queue. Queue được bảo vệ bằng `flock` độc quyền.

## Kiểm thử và công việc còn lại

Kiểm thử CPU synthetic bao gồm negative exclusion, phát hiện cache partial/tamper,
không đọc test, không ghi đè dữ liệu cũ, resume cho trọng số/loss giống lượt chạy
liền mạch và biên cuối batch. Không coi các phép thử synthetic là kết quả GPU thật.
GPU smoke/resume của UFM vẫn chờ features full; khác với smoke encoder CLIP đã đạt.

Riêng transformer GPU và UFM chưa chạy test; baseline cổ điển đã có test trong
README trước chiến dịch này. Sau khi validation chốt mô hình: chạy ablation, đối chứng chuẩn
SASRec/BERT4Rec, calibration trên validation riêng, kiểm tra ngân sách VRAM 16GB,
thử tập dữ liệu thứ hai theo proposal, rồi đánh giá test một lần, demo khuyến nghị,
biểu đồ, báo cáo cuối kỳ và slide. Xem `scope.md` để đối chiếu phạm vi.

Bổ sung ngày 28/09/2026: 32 kiểm thử CPU của chiến dịch PASS trên FITLAB trong
10,468 giây (bao gồm suite/demo), không thay thế GPU smoke UFM. Demo CPU TF-IDF
đã chạy trên máy cá nhân; backend UFM còn chờ checkpoint production.
