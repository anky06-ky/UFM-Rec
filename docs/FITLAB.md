# Chạy đồ án trên FITLAB

Thư mục dự án: `/home/coder/iDragonCloud/DA_AI`.

Môi trường GPU đã có sẵn: `/home/coder/iDragonCloud/.venv/bin/python`.
Python mặc định `/home/coder/.venv/bin/python` là môi trường khác và chưa có PyTorch tại thời điểm kiểm tra.

Trong terminal của code-server:

```bash
cd /home/coder/iDragonCloud/DA_AI
/home/coder/iDragonCloud/.venv/bin/python -u src/check_fitlab.py
```

Trong notebook, chọn kernel có đường dẫn `/home/coder/iDragonCloud/.venv/bin/python`.

Gọi Python bằng đường dẫn đầy đủ: script `activate` của venv đã được chuyển thư mục có thể vẫn trỏ tới môi trường cũ `/home/coder/.venv`.

Dùng dữ liệu `data/processed/toys_games_full_temporal`: chia theo thời gian 80/10/10, tổng 11.572.689 tương tác. Train có 9.258.647 tương tác và 4.230.848 mẫu có lịch sử. Bộ 70/15/15 cũ không dùng cho lần chạy này.

Các baseline hiện có trong `src/` dùng scikit-learn và CPU. GPU smoke test chỉ xác nhận môi trường; không có nghĩa mô hình đã được huấn luyện. Giữ bộ validation để chọn mô hình và tham số; chỉ dùng test để đánh giá mô hình đã chọn. Lưu checkpoint và kết quả vào thư mục dự án trên iDragonCloud để giữ qua lần tạo lại container.

Không cài lại toàn bộ PyTorch khi môi trường hiện tại đã vượt qua kiểm tra GPU. Dung lượng/RAM của máy chủ có thể dùng chung; không coi toàn bộ số báo bởi `free` hoặc `df` là hạn mức riêng.
