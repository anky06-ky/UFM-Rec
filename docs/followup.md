# Thí nghiệm tiếp theo — 03/10/2026

Giữ nguyên trainer/model UFM đã khóa hash. Ablation hiện tại tiếp tục bằng
checkpoint của chính variant; không dùng trọng số full để thay thế.

## Kế hoạch cố định trước benchmark

`src/run_followup_suite.py` đợi đủ 7 ablation, kiểm tra protocol, rồi chạy:

1. SASRec, BERT4Rec thích nghi, concat CLIP: seed 42.
2. Ba baseline trên: seed 7; UFM full: seed 7.
3. Ba baseline trên: seed 2026; UFM full: seed 2026.

Tổng cộng 11 run production bổ sung. Ba baseline có smoke GPU 5 bước riêng trước
lượt production đầu tiên. Cùng split, candidate set, toàn bộ 40.000 validation
cases; kế thừa history/dim/layers/batch/lr/epoch/patience từ full UFM. Chọn best
bằng cold macro NDCG@10. Không chạy test. Không thay số seed theo kết quả.

BERT4Rec dùng Transformer hai chiều, Cloze masking 15%, thay thế 80% bằng MASK,
10% item train ngẫu nhiên, 10% giữ nguyên; 10% sequence được ép mask item cuối.
Trong catalog 767k item, dùng softmax trên positive và các negative sampled,
loại mọi positive train của user. Đây là **bản thích nghi sampled-softmax**,
không là reproduction nguyên bản/full-softmax của
[Sun et al., BERT4Rec](https://arxiv.org/abs/1904.06690).
Khi inference, thêm MASK cuối history. Chỉ đọc prefix train khi tạo bài Cloze.

Concat ghép user/item/tích từng chiều của hai nhánh sequential ID và semantic
CLIP qua MLP, cùng adapter 1026→256→128. CLIP frozen; không uncertainty gate,
không cross-alignment, không auxiliary loss. Objective BCE sampled giống SASRec
trong repo. Đây là đối chứng kiến trúc; không đồng nhất objective với UFM BPR.

SASRec vẫn là bản thích nghi next-item theo từng prefix. Item ID chưa train bị
mask trước lookup ở history/candidate. Concat vẫn dùng nội dung của item unseen.

## Chạy và phục hồi

Trên FITLAB, sau khi deploy và kiểm thử:

```bash
.venv_ufm_runtime/bin/python src/run_followup_suite.py --dry-run
nohup setsid .venv_ufm_runtime/bin/python -u src/run_followup_suite.py > runs/followup_suite_v1.log 2>&1 < /dev/null &
python3 status.py
```

Lock riêng được child kế thừa; hai probe GPU rảnh, tối thiểu 8192 MiB, cách 30s.
Chạy một trainer mỗi lần. Cửa sổ khởi chạy tối đa 7 ngày; không kill run đang chạy.
SIGKILL/OOM retry tối đa 3 lần, SIGTERM/lỗi invariant dừng để kiểm tra.
Container bị thu hồi: kiểm tra process, log và checkpoint rồi chạy lại cùng lệnh;
không tự cam kết process tồn tại qua reboot. Followup chưa nối vào autostart cũ.

## Phân tích không cần GPU

```bash
.venv_ufm_runtime/bin/python src/audit_ufm_validation.py --output reports/ufm_validation_audit_new
```

Đối chiếu saved predictions với cold macro của completed marker; báo từng regime
và có/không history; bootstrap paired theo user 2000 lần, percentile 95%.
Temperature fit nửa thời gian validation trước và audit nửa sau, không tách các
timestamp bằng nhau. Vì checkpoint đã chọn trên toàn validation, calibration/CI
là exploratory, không phải holdout độc lập hay độ biến thiên theo seed.

Mỗi lần chạy chọn thư mục output mới. Kết quả ngày 03/10 đang dùng là
`reports/ufm_validation_audit_v2`; v1 bị thay thế do sửa độ chính xác phép tính NDCG.

## Điều kiện trước test cuối

Hoàn tất benchmark/ablation; xem xét lỗi cold-start trên validation; khóa config,
checkpoint SHA256, candidate protocol, seed và temperature vào manifest trước khi
đọc test UFM. Sau đó chạy test một đợt cho các cấu hình đã chốt. Test baseline cổ
điển đã được xem trong lịch sử. Dataset thứ hai và giới hạn GPU 16 GiB vẫn phải
được kiểm chứng riêng; không coi queue trên là đã hoàn tất hai hạng mục đó.

## Bộ dữ liệu thứ hai

All_Beauty từ cùng nguồn Amazon Reviews 2023 đã chuẩn bị CPU ngày 03/10:
701.528 review gốc → 445.239 tương tác positive verified không trùng;
356.402/44.396/44.441 train/validation/test; 86.014 item, 19.638 prefix train
có history. Metadata khớp 86.014 item; 86.012 có text và tất cả có URL ảnh
(chưa đồng nghĩa tải ảnh thành công). Đã có TF-IDF validation và catalog CLIP.

All Beauty có 37.381 validation cases (zero/extreme/warm mỗi nhóm 10.000; cold
7.381), cùng 100 candidates/case. TF-IDF overall NDCG@10 0,260239 và cold macro
0,098279. Chỉ 2.627 case có history, 34.754 case rỗng: phải báo cáo riêng hai nhóm,
không diễn giải overall thành chất lượng cá nhân hóa. Trên dataset này, cap=0 là
dùng đủ 37.381 case sẵn có, không phải nhân bản để đủ 40.000.

`scripts/prepare_second_dataset.py` tải hai gzip từ link chính thức, lưu SHA256,
tái sử dụng pipeline temporal/content/candidate/cache hiện có vào đường dẫn riêng.
Các stage partial dừng để kiểm tra, không ghi đè kết quả hoàn tất. Tạo split/test
candidates không phải đánh giá test; baseline chỉ chạy validation.

`src/run_second_dataset_suite.py` đợi Toys followup hoàn tất rồi giữ lock GPU
dùng chung: CLIP toàn catalog, UFM seed42, sau đó SASRec/BERT4Rec/concat và UFM
đủ seeds 42,7,2026. Tổng 1 extraction + 12 run production. Các marker và hash
được kiểm tra; chạy một job, đợi GPU rảnh. SIGTERM/lỗi worker dừng để kiểm tra,
chạy lại cùng lệnh sẽ resume checkpoint chưa hoàn tất. Theo dõi `DATASET2` trong
`status.py`. Cửa sổ khởi chạy 14 ngày; chưa nối vào autostart cũ.

```bash
nohup setsid .venv_ufm_runtime/bin/python -u src/run_second_dataset_suite.py > runs/second_dataset_suite_v1.log 2>&1 < /dev/null &
```
