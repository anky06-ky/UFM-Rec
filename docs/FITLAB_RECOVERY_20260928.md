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

Notebook `notebooks/FITLAB_Training_Monitor.ipynb` chỉ đọc log và các metric, có thể chạy lại để cập nhật. Đóng tab không dừng job nền; dừng container FITLAB vẫn có thể làm mất tiến trình. Checkpoint ở iDragonCloud dùng để khôi phục.

Không chạy test trong bước khôi phục. Cấu hình vẫn giới hạn 20 epoch, early stopping sau ba epoch không cải thiện NDCG@10 của nhóm có lịch sử trên validation. Khi epoch 5 được hoàn thành sau resume, `train_loss` do mã hiện tại ghi là trung bình phần batch chạy sau resume, không phải trung bình đầy đủ epoch 5.

## Kết quả khởi động lại đã xác nhận

- Bắt đầu resume lúc 11:55:20 ngày 28/09/2026, múi giờ +07:00.
- PID Python lúc khởi động: `106228`; PID wrapper khóa: `106224` (chỉ có giá trị trong phiên container này).
- Self-test CUDA PASS và log mới báo `TRAIN START` với đúng 4.230.848 mẫu.
- Đã tiến qua bước `81300`, epoch 5, loss gần nhất `4.00709`. Bước `81100` chạy lại cho cùng loss `4.02751` như log trước khi ngắt, phù hợp với khôi phục RNG và batch order.
- Job chưa hoàn thành toàn bộ train; đây là xác nhận resume đang hoạt động, không phải kết quả test hay cam kết job sẽ không bị ngắt tiếp.

Phạm vi Foundation Model, uncertainty và UGAF phải được chốt riêng; khôi phục lượt này không đồng nghĩa đã hoàn thành UFM-Rec trong proposal gốc.
