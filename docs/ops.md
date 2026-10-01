# Các lượt huấn luyện dài ngày và demo

Pipeline GPU chạy nền, không phụ thuộc tab trình duyệt hoặc trạng thái ngủ của
máy cá nhân. Tuy nhiên container FITLAB vẫn phải được trường giữ hoạt động;
không có cam kết job sống qua việc trường thu hồi hoặc tạo lại container.
Checkpoint và cache ở iDragonCloud, Python isolated hiện ở `/tmp` nên môi trường
phải được dựng lại nếu container mất. Không cài lại Torch CUDA và không tắt job
khác để lấy GPU.

## Thứ tự thực nghiệm

1. CLIP đóng băng trích toàn bộ 767.045 sản phẩm. Full marker, nguồn dữ liệu,
   SHA256, padding/mask/norm và độ phủ phải đạt trước khi train UFM.
2. Queue UFM chuẩn bị graph positive train-only, kiểm tra GPU smoke 5 bước gồm
   ngắt/resume, rồi full UFM tối đa 20 epoch, patience 3, seed 42.
3. Suite chạy nối tiếp bảy biến thể: no uncertainty, fixed fusion, no cross align,
   semantic only, collaborative only, text only và image only. Mỗi biến thể học
   từ đầu, cùng budget/config/candidates và criterion cold macro NDCG@10.
   Không dùng trọng số full UFM để gọi đó là ablation training.
4. Cửa sổ khởi chạy suite tối đa 7 ngày. Nếu hết cửa sổ, không khởi động lượt
   mới; lượt đang chạy được để hoàn tất và không xóa kết quả. Suite dừng có log
   nếu nguồn/config sai, GPU không rảnh hoặc lượt con thất bại; không ép bỏ lỗi.

Chỉ một job GPU của pipeline chạy tại một thời điểm. Hai lần kiểm tra rảnh cách
30 giây, không còn process trích features/train và còn ít nhất 8.192 MiB VRAM.
GPU trường dùng chung có thể khiến queue phải chờ. Dừng sớm là kết quả validation
hợp lệ; không ép train nhiều ngày chỉ để tăng thời lượng.

Suite đầu tiên chỉ có một seed, chưa thay thế thử nhiều seed/độ biến thiên.
No cross align loại thành phần alignment trong fusion; cosine consistency loss
vẫn giữ như cấu hình chung. Các nhánh không khả dụng dùng mask và train-popularity
fallback. Chưa gọi lệnh test, chưa temperature-fit và chưa chứng minh tốt hơn
baseline. SASRec/BERT4Rec chuẩn, dữ liệu thứ hai và ngân sách VRAM 16GB còn cần làm.

## Theo dõi và phục hồi

```bash
python src/monitor_ufm_campaign.py
```

### Khôi phục ngày 01/10/2026

Lượt CLIP cũ bị `SIGKILL` khi đã commit 594.304/767.045 item. `memory.events`
của container tại lúc kiểm tra báo `oom_kill=0`; chưa xác định được nguồn gửi
SIGKILL. Cache/`progress.json` còn đủ và đúng batch 64, dữ liệu nguồn, revision.
Supervisor `scripts/resume_campaign.py` đang chạy nền trên FITLAB. Nó mở từng
queue theo thứ tự, dùng `flock` riêng và tối đa ba lượt khi queue dừng do SIGKILL.
Nếu gặp lỗi cấu hình, checksum, hoặc lỗi khác thì dừng và ghi status; không lặp
vô hạn. GPU đang bận thì queue đợi, không chiếm GPU của job khác.
Ngày 01/10, GPU đã bận trở lại sau hai lần kiểm tra rảnh và queue dừng trước khi
trích CLIP. Supervisor đã được cập nhật để coi đây là tình huống chờ lại, trong
cửa sổ tối đa 168 giờ; không tính vào ba lượt SIGKILL.

```bash
python status.py
cat runs/campaign_recovery_20261001.json
tail -n 20 runs/campaign_recovery_20261001.log
```

Chỉ khi đã xác nhận supervisor và mọi queue cũ đều dừng mới khởi động lại bằng
Python isolated `/tmp/ufm_venv_20260928_cdcp1xhw/bin/python` và `nohup setsid`.
PID, venv `/tmp` và tiến trình không sống qua việc container bị xóa; các file
cache/checkpoint trên iDragonCloud vẫn cần được kiểm tra trước khi resume.

Status/log: `runs/foundation_queue_v1.*`, `runs/ufm_training_queue_v1.*`,
`runs/ufm_ablation_suite_v1.*`; từng ablation có log và `best.pt`, `latest.pt`,
`config.json`, `epoch_*.json`, `completed.json` riêng. Không chạy queue trùng.
Khi khôi phục, kiểm tra tất cả process đang sống và lỗi log trước; dùng cùng
external `flock`, nguồn/config/code/version không đổi. Trainer tự resume latest
chỉ khi output unfinished. Full/smoke hoàn thành không được ghi đè.

Lệnh suite khi đã xác nhận không còn runner suite cũ:

```bash
flock -n runs/ufm_ablation_suite_v1.lock /path/to/ufm-python -u src/run_ufm_ablation_suite.py --max-days 7
```

Thay `/path/to/ufm-python` bằng Python isolated thật, không sao chép placeholder.
Mã không tự khởi động sau reboot; xem hướng dẫn dựng lại venv trong
`foundation.md`. Nguồn dữ liệu/checkpoint lớn không đưa lên GitHub.

Queue UFM đã gia hạn lên 168 giờ khi vẫn chỉ chờ features. `.gitattributes`
giữ nguyên byte Python giữa Windows và Linux để Git không tự đổi newline làm
khác SHA256 của mã. Không sửa trainer/model trong khi chiến dịch đã được cấu hình.

## Demo CPU trong khi GPU đang làm việc

```bash
python src/demo_recommender.py --backend content --port 8765
```

Mở `http://127.0.0.1:8765`. Demo hiện dùng TF-IDF content đã fit trên train,
không được gọi là UFM đã train. Tìm sản phẩm, chọn lịch sử cũ đến mới, xem Top K,
tên/ảnh, số tương tác train và regime. Candidate là toàn catalog và loại sản
phẩm trong lịch sử; điểm không phải xác suất mua hàng. Không đọc nhãn test.

Sau khi full UFM hoàn thành và cache/checkpoint đã kiểm chứng:

```bash
python src/demo_recommender.py --backend ufm --run runs/ufm_full_v1 --port 8765
```

Backend UFM nạp `best.pt` validation-selected, kiểm tra code/features SHA,
full cache/mapping rồi phục vụ bằng CPU theo chunk để không tranh GPU.
Hiển thị trọng số CF/semantic, uncertainty proxy và mask modality. Không nhận
cache/checkpoint smoke để giả làm mô hình production. Demo chỉ bind loopback,
không tự publish công khai, không lưu lịch sử nhập vào log.
