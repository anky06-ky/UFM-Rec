# Chạy liên tục và khôi phục chiến dịch FITLAB

Ngày 01/10/2026, CLIP đang chạy từ checkpoint. `ops/watch_campaign.py` theo dõi
supervisor, queue và worker trong container hiện tại. Nếu supervisor hoặc worker
bị SIGKILL, watchdog chờ mọi process cũ kết thúc rồi chạy lại supervisor. Queue
chỉ dùng `--resume` khi có progress/checkpoint chưa hoàn tất. Các stage chạy tuần tự:
CLIP, UFM full, sau đó bảy ablation. Các lỗi checksum, cấu hình và invariant vẫn
dừng và hiện `needs_attention` để tránh lặp một thử nghiệm sai.

```bash
python status.py
cat runs/campaign_watchdog_v1.json
tail -n 30 runs/campaign_watchdog_v1.log
tail -n 30 runs/campaign_recovery_20261001.log
```

Python của campaign nằm tại `.venv_ufm_runtime/bin/python` trên iDragonCloud.
Watchdog dùng `/usr/bin/python3`, nên không phụ thuộc venv tạm trong `/tmp`.
Nếu container FITLAB được tạo lại, vào thư mục dự án và chạy:

```bash
bash scripts/start_campaign.sh
python status.py
```

Không chạy thêm queue hay trainer thủ công khi watchdog còn hoạt động. File lock
và kiểm tra process ngăn hai lượt GPU cùng chạy. Cache, checkpoint, status và log
được giữ dưới `data/` và `runs/` trên iDragonCloud; không đưa chúng lên GitHub.

## Tự khởi động khi mở lại workspace

Chạy một lần trên FITLAB:

```bash
python3 ops/install_autostart.py
```

Installer giữ các settings/task hiện có, lưu bản sao trong `runs/autostart_backups/`,
thêm task `UFM: keep campaign running`. Code-server hiện yêu cầu lựa chọn Allow
Automatic Tasks cho tất cả workspace đã tin cậy. Người dùng đã đồng ý bật tùy chọn
này trên FITLAB ngày 01/10/2026. Khi cài mới cần chọn Allow trong hộp thoại.
Khi code-server mở lại thư mục dự án, task gọi `start_campaign.sh`. Khóa watchdog
ngăn việc mở lại tab tạo thêm GPU job. Cơ chế `runOn: folderOpen` được mô tả trong
[tài liệu VS Code](https://code.visualstudio.com/docs/debugtest/tasks#_run-behavior).

Lúc 20:36 ngày 01/10, worker CLIP bị SIGKILL ở cursor 764.032. Supervisor đã lên
lịch retry sau 120 giây; heartbeat watchdog còn tới 20:37:38. PID 1 của container
mới bắt đầu lúc 20:38:14, trước khi retry diễn ra. Toàn bộ process cũ mất theo
container. Cache và venv trên iDragonCloud còn nguyên, đã resume thành công.

FITLAB không có cron hoặc user systemd đang hoạt động. Task tự bật khi workspace
được mở/kết nối lại; nếu container bị thu hồi và workspace chưa được mở thì chưa
có process nào chạy. Có thể dùng lệnh khởi động thủ công ở trên. Watchdog mới ghi
thêm định danh container và bộ đếm memory/oom vào log để chẩn đoán lần ngắt sau.

## UFM dừng do gradient AMP ngày 01/10

CLIP đã hoàn tất 767.045 item. Full UFM dừng sau bước 6.000 vì gradient không
hữu hạn trong khi AMP scale là 524.288; checkpoint bước 6.000 có trọng số hữu hạn.
Trainer hiện để GradScaler giảm scale và bỏ qua batch bị tràn. Trên FITLAB đã
chạy `scripts/migrate_ufm_amp_checkpoint.py` một lần để cập nhật SHA mã trong
checkpoint/config, giữ bản gốc `latest.pre_amp_fix.pt` và
`config.pre_amp_fix.json`. Watchdog được mở lại sau khi ghi `repair_applied` vào
recovery status. UFM queue đã gọi trainer với `--resume` và log xác nhận bước
6.000. Lần tràn tiếp theo ở batch 6.072 được xử lý: scale giảm, training vẫn
tiến tới bước 7.600. Kiểm tra log và `python status.py` để theo dõi tiếp.

## FITLAB mất mount và UFM demo thoát bằng Bus error ngày 02/10

Lúc 17:56 UTC+07, `/api/status` của demo trả `UFM Rec` và catalog 767.045 sản phẩm.
Sau request inference tiếp theo, shell ghi `[1]+ Lỗi bus` cho PID 72211. Cùng thời
điểm, `stat /home/coder/iDragonCloud` và thư mục dự án trả `EIO`; code-server báo
workspace không tồn tại/mất kết nối. Sau khi tải lại code-server, danh sách cổng
không còn tiến trình 8766. Chưa đọc được log demo để xác định nguyên nhân.

Khi FITLAB kết nối lại, kiểm tra mount, dung lượng và khả năng đọc checkpoint
trước khi khởi động bất kỳ trainer hoặc demo nào. Chạy `python3 status.py` để xem
queue/watchdog; chỉ tiếp tục campaign nếu status, log và checkpoint đọc được.
Không xóa hoặc tạo lại thư mục mount để xử lý `EIO`, vì dữ liệu lớn và checkpoint
đang nằm ở đó. Khởi động demo bằng origin/base path như trong `docs/demo.md`, rồi
chạy một request inference có kiểm soát và theo dõi log trước khi nghiệm thu proxy.
# Khôi phục lượt GPU ngày 28 tháng 9 năm 2026

## Kiểm tra trước khi khôi phục

Dự án remote: `/home/coder/iDragonCloud/DA_AI`.
Lượt chạy: `runs/content_transformer_v1`.

- Không có tiến trình train trước lúc kiểm tra; `completed.json` và `test_metrics.json` chưa tồn tại.
- GPU Quadro RTX 6000 còn khoảng 20,9 GiB VRAM khả dụng tại lúc kiểm tra.
- Bộ đếm `memory.events` hiện tại báo `oom=0`, `oom_kill=0`. Đây không phải bằng chứng xác định nguyên nhân dừng lần trước.
- Đọc được cả `latest.pt` và `best.pt` bằng `torch.load(..., weights_only=True)`. Trọng số mô hình hữu hạn, không NaN/Inf.
- Fingerprint của hai checkpoint khớp marker cache: `8b3974ab69146d46e069957f0016e5d1fd13d3bbf626d96bb02984889283163c`.
- SHA256 mã train local và remote giống nhau: `e6ca4b9b1e2e620e689272cb2d8c29094153d0ffbbb6465eb56324bd637b3546`.

Trạng thái `latest.pt`: epoch nội bộ `4` (đang chạy epoch 5), `next_batch=14892`, `global_step=81000`, `best=0.372614`, `stale=0`. Log cũ kết thúc bước 81.100, nên resume sẽ lặp lại khoảng 100 batch chưa được lưu, không huấn luyện lại từ đầu.

`best.pt` thuộc epoch 4. AdamW có một nhóm tham số và 37 trạng thái tham số trong checkpoint. Resume của script nạp model, optimizer, GradScaler, RNG CPU và CUDA; kiểm tra cấu hình và tính lại fingerprint dữ liệu trước khi train.

## Bảo toàn trạng thái

Thư mục backup trên FITLAB: `runs/recovery_20260928_1153`.
Bản sao gồm `best.pt`, `latest.pt`, `config.json`, `history.jsonl`, log và mã train. Không xóa dữ liệu hay sửa tham số mô hình.

SHA256 checkpoint gốc và bản sao khớp từng cặp:

- `best.pt`: `eca9cfbb9b183ddffee34c64948a7a2afb6937e485d18b4c8aa153954ebffd42`.
- `latest.pt`: `aa4aa92adbeb9a722cde35e880f216aff278b48a5cf58e891b8686e316468b7b`.

Lệnh resume dùng Python GPU tuyệt đối:

```bash
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py \
  --output runs/content_transformer_v1 --epochs 20 --resume
```

Không chạy lại lệnh này khi đã có tiến trình train cùng lượt. Lượt chạy nền dùng khóa `flock`, nối tiếp log cũ và ghi exit code khi tiến trình kết thúc bình thường hoặc báo lỗi. Không thể ghi exit code nếu toàn bộ container bị tắt hoặc tiến trình giám sát bị SIGKILL.

## Theo dõi và giới hạn

```bash
tail -n 12 runs/content_transformer_v1.log
pgrep -af '[t]rain_gpu_recommender.py'
```

Notebook `notebooks/02_baseline.ipynb` chỉ đọc log và các metric, có thể chạy lại để cập nhật. Đóng tab không dừng job nền; dừng container FITLAB vẫn có thể làm mất tiến trình. Checkpoint ở iDragonCloud dùng để khôi phục.

Không chạy test trong bước khôi phục. Cấu hình vẫn giới hạn 20 epoch, early stopping sau ba epoch không cải thiện NDCG@10 của nhóm có lịch sử trên validation. Khi epoch 5 được hoàn thành sau resume, `train_loss` do mã hiện tại ghi là trung bình phần batch chạy sau resume, không phải trung bình đầy đủ epoch 5.

## Kết quả khởi động lại đã xác nhận

- Bắt đầu resume lúc 11:55:20 ngày 28/09/2026, múi giờ +07:00.
- PID Python lúc khởi động: `106228`; PID wrapper khóa: `106224` (chỉ có giá trị trong phiên container này).
- Self-test CUDA PASS và log mới báo `TRAIN START` với đúng 4.230.848 mẫu.
- Đã tiến qua bước `81300`, epoch 5, loss gần nhất `4.00709`. Bước `81100` chạy lại cho cùng loss `4.02751` như log trước khi ngắt, phù hợp với khôi phục RNG và batch order.
- Job chưa hoàn thành toàn bộ train; đây là xác nhận resume đang hoạt động, không phải kết quả test hay cam kết job sẽ không bị ngắt tiếp.

Phạm vi Foundation Model, uncertainty và UGAF phải được chốt riêng; khôi phục lượt này không đồng nghĩa đã hoàn thành UFM-Rec trong proposal gốc.
