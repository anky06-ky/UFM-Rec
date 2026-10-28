# Bản sửa 1–15 trên FITLAB-01

Mã đang sửa và đã đạt **87 test trên Linux** nằm ở:

`/home/coder/iDragonCloud/DA_AI/audit_v2/fix_1_15_workspace_20261027`

[Mở workspace trong code-server FITLAB-01](https://coder-2474802010460.fitlab-01.is-tech.vn/?folder=/home/coder/iDragonCloud/DA_AI/audit_v2/fix_1_15_workspace_20261027).

Chạy kiểm thử từ workspace này:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/coder/iDragonCloud/DA_AI/.venv_ufm_runtime/bin/python \
  -m pytest -q -p no:cacheprovider --tb=short -rs
```

Workspace gốc `/home/coder/iDragonCloud/DA_AI` giữ dataset, checkpoint và ba tệp mã huấn luyện khớp hash checkpoint. Bản local có hai biến thể mới nên không chép đè các tệp này vào bản đã huấn luyện để ép nạp checkpoint.

- Bằng chứng: `reports/fix_1_15_20261027/` trong workspace gốc và bản mã mới.
- Validation v2: `data/processed/toys_games_full_temporal/evaluation_v2/` trong workspace gốc.
- Validation v3: `reports/fix_1_15_20261027/fitlab01_finish1/evaluation_v3/` trong workspace gốc.
- Checkpoint đã xác minh: `runs/ufm_full_v1/best.pt` trong workspace gốc.
- Các lệnh audit/baseline của bản mã mới nhận `--data` và `--evaluation`; truyền đường dẫn tuyệt đối tới dữ liệu cần dùng. Không giả định thư mục dữ liệu mặc định của bản mã mới đã chứa benchmark.

Xem [báo cáo sửa](../reports/fix_1_15_20261027/README.md) và [nội dung trao đổi với giảng viên](trao_doi_gvhd_2026_10_27.md). Chưa có kết quả mô hình trên validation v3; chưa push GitHub.
