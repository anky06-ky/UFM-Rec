# Chạy vòng UFM đa seed trên FITLAB và mang kết quả về máy

Vòng này chỉ dùng Toys & Games. Phạm vi sản phẩm kết hợp hành vi, ảnh và tên/mô tả đã được nhóm xác nhận với giảng viên. [Manifest](../configs/multimodal_followup_20261028.json) khóa **15 run**: full, semantic-only, fixed-fusion, count-aware fusion, adaptive dropout × seed 42, 7, 2026. Chưa có kết quả huấn luyện mới. Test v2 đã được xem; không dùng nó để chọn mô hình.

Ngày 28/10, checkout bên dưới đã có trên filesystem dùng chung và dry-run 9/9 đầu vào SHA đạt cả FITLAB-01 lẫn FITLAB-02. [Receipt cổng 01](../reports/continuation_20261028/fitlab01_reopen_20261028.json), [cổng 02](../reports/continuation_20261028/fitlab02_check_20261028.json). Hai cổng dùng GPU khác nhau nhưng cả hai đang bận. Trên FITLAB-02, `python` mặc định thiếu PyTorch; dùng `/home/coder/iDragonCloud/DA_AI/.venv_ufm_runtime/bin/python` đã xác nhận PyTorch `2.12.1+cu130` và `torch.cuda.is_available() == True`. Thay `python` trong các lệnh huấn luyện/đánh giá dưới đây bằng đường dẫn này khi thao tác từ FITLAB-02. Kiểm tra lại GPU trống ngay trước lúc chạy và sao lưu từng run về máy.

Colab T4 đã qua [smoke CUDA 5/5 biến thể](../reports/continuation_20261028/colab_cuda_smoke_20261028.json) trên dữ liệu giả lập. Để chạy giao thức đầy đủ trên Colab phải chuyển đúng 9 đầu vào đã khóa cùng cache huấn luyện, giữ đủ dung lượng RAM/VRAM và cơ chế sao lưu checkpoint trước khi phiên ngắt. Chưa thực hiện chuyển dữ liệu lớn, nên không coi Colab là nơi có kết quả train mới.

## 1. Dùng source mới trong workspace riêng

Trên FITLAB-01, giữ nguyên dữ liệu/checkpoint cũ tại `/home/coder/iDragonCloud/DA_AI`. Tạo checkout mới của nhánh `followup/multimodal-20261028` ở một thư mục khác, ví dụ `.../DA_AI/audit_v2/followup_multimodal_20261028`. Không sửa archived training source của checkpoint cũ. Chạy bằng môi trường Python UFM đã có trên FITLAB.

```bash
cd /home/coder/iDragonCloud/DA_AI/audit_v2
git clone --branch followup/multimodal-20261028 --single-branch \
  https://github.com/anky06-ky/UFM-Rec.git followup_multimodal_20261028
cd followup_multimodal_20261028
```

Trong checkout mới, dùng đúng Python có PyTorch CUDA và các thư viện trong `requirements.txt`:

```bash
python scripts/run_locked_ufm_followup.py \
  --workspace /home/coder/iDragonCloud/DA_AI
```

Dry-run phải báo `preflight: passed`, **9 locked inputs**, **15 jobs**. Nó chỉ đọc file, không huấn luyện. Nếu hash khác, dừng và đối chiếu đúng tệp; không bỏ guard hoặc ghi đè bản cũ.

## 2. Chạy từng run và sao lưu ngay

Một `best.pt` lịch sử khoảng 397 MB, `latest.pt` khoảng 1,19 GB. Giữ toàn bộ 15 run cùng lúc cần khoảng **32 GiB trống** theo guard của runner. Với `--max-new-jobs 1`, runner cần ít nhất **4 GiB** trước một run và nên có thêm khoảng **2 GiB** cho ZIP sao lưu trên cùng ổ. Sau khi xác minh bản local, chỉ `latest.pt` của run đã hoàn tất mới có thể dọn để giảm dung lượng; giữ `best.pt`, `completed.json`, config, log và report trên FITLAB để runner nhận biết run đã hoàn tất. Kiểm tra GPU không có job khác; runner sẽ từ chối nếu bận và không dừng tiến trình của ai.

```bash
python scripts/run_locked_ufm_followup.py \
  --workspace /home/coder/iDragonCloud/DA_AI \
  --max-new-jobs 1 --execute
```

Lệnh này huấn luyện tối đa một run mới, rồi đánh giá checkpoint vừa chọn trên **validation v3** và lưu scores, report, config, log. Chạy lại cùng lệnh để chuyển sang run kế tiếp. Nếu quá trình dừng giữa chừng, `latest.pt` được dùng để resume chỉ khi fingerprint/config khớp. Output nằm trong `runs/ufm_followup_20261028/` và `reports/ufm_followup_20261028/` của checkout mới.

Sau **mỗi run**, tạo gói riêng chỉ chứa artifact của run đó. Đổi `LABEL` theo thứ tự runner in ra; ví dụ run đầu là `full_s42`. Dùng một thư mục output **mới** cho mỗi lần tạo gói:

```bash
LABEL=full_s42
python scripts/artifact_backup.py create \
  --root "$PWD" \
  --output "/home/coder/iDragonCloud/DA_AI/audit_v2/followup_backup_${LABEL}" \
  --include "runs/ufm_followup_20261028/${LABEL}" \
  --include "runs/ufm_followup_20261028/${LABEL}.log" \
  --include "reports/ufm_followup_20261028/${LABEL}_v3.json" \
  --include "reports/ufm_followup_20261028/${LABEL}_v3.log" \
  --include "reports/ufm_followup_20261028/${LABEL}_v3_scores" \
  --include "configs/multimodal_followup_20261028.json"
```

Tải toàn bộ `bundle_*.zip` và `manifest.json` về `D:\Nam3_ki1\UFM-Rec_Backups\followup_backup_full_s42` (đổi nhãn tương ứng). Trên Windows, chạy:

```powershell
python scripts/artifact_backup.py restore `
  --backup 'D:\Nam3_ki1\UFM-Rec_Backups\followup_backup_full_s42' `
  --output 'D:\Nam3_ki1\UFM-Rec_Backups\restored_followup_full_s42'
```

Chỉ coi run đã an toàn khi `restore_receipt.json` ghi `all_files_sha256_verified=true` và `missing_files=[]`. Giữ cả ZIP gốc lẫn thư mục đã khôi phục trên máy. Sau bước này có thể dọn **riêng** `runs/ufm_followup_20261028/<LABEL>/latest.pt` và thư mục ZIP batch tương ứng trên FITLAB nếu thiếu dung lượng; `best.pt` và report ở lại. Không dọn khi chưa đủ gói hoặc receipt chưa hợp lệ. Chạy lại runner với `--max-new-jobs 1 --execute` để tiếp tục run sau.

## 3. Tổng kết đúng giao thức

Khi đủ 15 report v3:

```bash
python scripts/summarize_locked_ufm_followup.py \
  --reports reports/ufm_followup_20261028 \
  --output reports/ufm_followup_20261028/summary_all_seeds.json
```

Script sẽ từ chối nếu thiếu một seed, seed trong checkpoint không khớp, sai candidate/source/feature/count SHA, sai variant hoặc có test. Báo từng seed, mean và sample standard deviation; tách zero-shot, extreme-cold, cold, warm và history. Validation v3 dùng lại quần thể từng tham gia chọn checkpoint, nên kết quả vẫn là **thăm dò**, không phải holdout độc lập. Chỉ cập nhật Word/PDF/slide bằng điểm mới sau khi đủ bằng chứng và đối chiếu receipt local.
