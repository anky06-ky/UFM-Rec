# FITLAB: triển khai bước Foundation Model

Snapshot kiểm tra: 28/09/2026 12:43, Asia/Bangkok. Không phải báo cáo đã hoàn
thành huấn luyện UFM hay kết quả test.

**Cập nhật sau snapshot:** baseline đã hoàn tất 8 epoch, CLIP GPU smoke đạt
128/128 văn bản và ảnh, full extraction đang chạy. Trainer UFM đã triển khai;
25 kiểm thử CPU PASS trên FITLAB và queue train nền đã bật để chuẩn bị graph,
chờ features hoàn chỉnh rồi chạy GPU smoke/resume và full train.
Các trạng thái 12:43 bên dưới được giữ làm lịch sử, không phải trạng thái mới nhất.
Xem [hướng dẫn train UFM](train.md) và dùng
`src/monitor_ufm_training.py` để đọc trạng thái hiện tại.

## Kết quả đã xác nhận

- Root remote: `/home/coder/iDragonCloud/DA_AI`.
- Triển khai 8 file mới: 4 module trong `src`, 2 file kiểm thử, requirements và
  tài liệu phạm vi. Không ghi đè mã train cũ hay checkpoint.
- 17 kiểm thử CPU PASS trên FITLAB trong 26,308 giây. Đây là kiểm thử kỹ thuật,
  không phải bằng chứng recommendation tốt hơn đối chứng.
- Python 3.12.3, PyTorch **2.12.1+cu130** vẫn được đọc từ venv cũ;
  `torch.cuda.is_available()` True. Transformers 4.57.6, Hub 0.36.2.
- Nhập catalog local 767.045 sản phẩm; kiểm tra hash catalog và đối chiếu
  mapping, văn bản, split trên remote đều PASS.
- Queue đã chạy nền, wrapper `flock` PID **113819**. Trạng thái xác nhận:
  `waiting_for_baseline_and_idle_gpu`, baseline PID 106228, GPU sử dụng 49%,
  còn 18.600 MiB lúc snapshot. Smoke/full features **chưa bắt đầu**.
- Đối chứng đã hoàn thành epoch 1–7 và đang train epoch 8 lúc kiểm tra. Best
  validation nhóm có lịch sử là epoch 5, NDCG@10 = 0,380043. Epoch 7 = 0,375546.
  Đây là sampled ranking 1 positive + 99 negatives, không phải full-catalog.
- Không gọi trainer UFM hoặc lệnh đánh giá test; `test_metrics.json` đối chứng
  chưa tồn tại. Giữ split temporal 80/10/10 của bộ lớn.

## Catalog và nguồn

FITLAB thiếu `data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz`. Lần dựng
catalog remote đầu tiên dừng vì thiếu file; log giữ lại tại
`runs/foundation_catalog_build_v1.log`. Không bỏ qua lỗi hoặc tự tạo metadata.
Thay vào đó chuyển catalog đã dựng và kiểm kê từ metadata thật trên local.

- Catalog: `data/processed/toys_games_full_temporal/foundation_catalog_v1`.
- Đã quét local 890.874 metadata, khớp 767.045 item.
- Có text: 767.044; có URL ảnh: 767.007; có cả hai: 767.006.
- Có URL chưa bảo đảm tải/giải mã ảnh thành công. Coverage thật lấy sau extraction.
- Metadata gốc vẫn trên máy local; report lưu SHA256 để giữ nguồn gốc. Catalog
  đã nhập có đủ URL cho extractor, không cần tải lại metadata 658 MB lên FITLAB.

Hash nguồn đã đối chiếu trên remote:

| Nguồn | SHA256 |
|---|---|
| items | `e67e6310a6076f1e66ff14b8ce68e3d28a1aba204498175ae77eb0b04d0050fc` |
| text | `5e134cfd3ac4e94e78b3bdac9bbd40927fb57fbfa45874e57fceae7abe5aca13` |
| split | `585f102f14e12bfae332f53d28bb1f853acde1586eb102555c4b118ccee75763` |
| catalog | `65d63e7b58babb133db78f3369b3224db6a451d80c4f616a4b8c0023f73b9003` |

Gói mã: `docs/ufm_foundation_fitlab_v1.tar`, SHA256
`1671af8f168941c5b36077eb4a67354aa7fdcc91bc836aa112fca406a97a7fd3`.
Gói catalog: `docs/ufm_catalog_fitlab_v1.tar`, SHA256
`d02f05d14a80f8c9117e4d411b8d1b928ed9159fa188c2d66b7e243e22d24b11`.
Mã baseline giữ nguyên SHA256
`e6ca4b9b1e2e620e689272cb2d8c29094153d0ffbbb6465eb56324bd637b3546`.

## Môi trường riêng và lưu trữ

Python chạy queue/extractor hiện tại:

```text
/tmp/ufm_venv_20260928_cdcp1xhw/bin/python
```

Tạo venv trong iDragonCloud thất bại khi tạo symlink `lib64` (I/O error). Folder
`.venv_ufm` remote là lần tạo chưa hoàn thành, **không phải môi trường dùng được**;
không xóa dữ liệu hay sửa cấu hình filesystem để vượt lỗi.

Môi trường mới trên đĩa local `/tmp` có packages UFM riêng và file `.pth` đọc
site-packages của GPU venv cũ. Không nâng/hạ PyTorch CUDA của baseline. Môi trường
tạm mất khi container bị tạo lại; mã, catalog, features, model cache và logs nằm
trên iDragonCloud để giữ qua các lần tạo lại container.

Nếu môi trường tạm đã mất, chạy recipe sau từ terminal Linux trên FITLAB để
tạo **môi trường mới**, rồi ghi lại đường dẫn in ra:

```bash
/home/coder/iDragonCloud/.venv/bin/python - <<'PY'
from pathlib import Path
import site, subprocess, tempfile, venv
root = Path('/home/coder/iDragonCloud/DA_AI')
env = Path(tempfile.mkdtemp(prefix='ufm_venv_'))
base = [p for p in site.getsitepackages() if Path(p).is_dir()]
venv.EnvBuilder(with_pip=True, symlinks=False).create(env)
(env / 'lib/python3.12/site-packages/fitlab_torch_readonly.pth').write_text('\n'.join(base) + '\n')
python = env / 'bin/python'
subprocess.run([str(python), '-m', 'pip', 'install', '--index-url',
                'https://pypi.org/simple', '-r', str(root / 'requirements-ufm.txt')], check=True)
print('NEW_UFM_PYTHON:', python)
PY
```

Kiểm tra Torch/CUDA và chạy lại tests trước khi phục hồi queue. Không dùng lại
đường dẫn `/tmp` trong tài liệu nếu file Python đó không còn tồn tại.

## Theo dõi và thứ tự chạy

Lệnh độc lập Jupyter (chỉ đọc) trong terminal Linux:

```bash
cd /home/coder/iDragonCloud/DA_AI
/home/coder/iDragonCloud/.venv/bin/python src/monitor_ufm_features.py
```

Mở `notebooks/03_clip.ipynb` trên FITLAB, chọn kernel venv sẵn
có và **Run All**. Notebook chỉ đọc trạng thái và log, không tạo job GPU mới.
Trong lần kiểm tra này, kernel được báo đã khởi động nhưng các cell vẫn pending
qua một lần hủy/thử lại. Toàn bộ mã cell đã chạy PASS trực tiếp bằng Python trong
terminal; **chưa xác nhận Run All thành công trong giao diện notebook**. Dùng CLI
ở trên nếu gặp pending, không restart kernel của notebook train hay tạo job GPU.
Lỗi metadata trong log dựng catalog đầu tiên không có nghĩa catalog nhập bị lỗi;
marker `foundation_catalog_v1/catalog_report.json` và hash nguồn mới là bằng chứng.

Queue đọc trạng thái mỗi 30 giây, chờ tối đa 24 giờ. Điều kiện khởi động:

1. Baseline có `completed.json` full-run và `best.pt`.
2. Không còn process `train_gpu_recommender.py`.
3. GPU còn ít nhất 8.192 MiB và utilization không quá 20% qua 2 lần liên tiếp.
4. Chạy CLIP đóng băng 128 item, kiểm tra hash/shape/norm/mask; tối thiểu 120 text
   và 116 ảnh thành công. Nếu lỗi/coverage thấp, dừng, không chạy full.
5. Kiểm tra GPU lại, rồi trích toàn catalog nếu smoke đạt.

Các file theo dõi:

| File/folder | Ý nghĩa |
|---|---|
| `runs/foundation_queue_v1.json` | Stage hiện tại / lỗi nếu queue dừng |
| `runs/foundation_queue_v1.log` | Log queue và extractor |
| `runs/foundation_queue_v1.lock` | Khóa tránh chạy queue trùng |
| `runs/foundation_smoke_128_v1` | Cache thử GPU 128 sản phẩm |
| `data/processed/toys_games_full_temporal/foundation_clip_b32_v1` | Cache full dự kiến |
| `tmp/hf_ufm_models` | Cache weights CLIP pinned commit |

Đóng tab không dừng process đã chạy nền, nhưng FITLAB tắt/tạo lại container sẽ
dừng process. Không bảo đảm queue hoạt động xuyên qua việc trường thu hồi máy.

Nếu queue đã dừng, đọc lỗi và kiểm tra process trước; không khởi động thêm queue
trong lúc queue hiện tại còn sống. Sau khi xử lý nguyên nhân, dùng cùng external
`flock` và Python UFM còn hợp lệ để phục hồi. Runner tự thêm `--resume` cho cache
chưa hoàn thành và cùng cấu hình; không ghi đè cache đã hoàn thành. Nếu đổi batch,
revision hoặc policy ảnh thì dùng output mới, không ép resume sai fingerprint.

## Bước tiếp sau khi features hoàn thành

1. Đối chiếu đủ 767.045 item, hash các bảng, padding/mask/norm và ảnh tải thất bại
   theo bốn regime. Chưa làm bước này vì full extraction chưa bắt đầu.
2. Viết trainer UFM có train-only sampling, validation cold macro, resume đủ
   optimizer/scaler/RNG/cursor; smoke backward và mô phỏng resume trước full train.
3. Train full UFM và các ablation từ đầu với cùng protocol; bổ sung baseline
   SASRec/BERT4Rec chuẩn, bộ dữ liệu thứ hai, calibration và đo chi phí.
4. Chốt config bằng validation, mới mở test. Sau đó demo, biểu đồ, báo cáo và slides.

Foundation features là đầu vào cho encoder đóng băng + adapter; không đồng nghĩa
đã fine-tune LoRA CLIP. Uncertainty vẫn là learned reliability proxy cần chứng minh
bằng ablation và reliability/calibration plots, không phải posterior Bayes.
