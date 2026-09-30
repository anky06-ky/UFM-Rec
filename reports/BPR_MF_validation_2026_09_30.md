# Đối chứng BPR-MF trên Amazon Toys and Games

Cập nhật 30/09/2026. Đây là **validation**, trên cùng 40.000 mẫu cố định (10.000 mẫu mỗi regime, một positive và 99 negative) với các baseline cũ. Chưa đánh giá test.

## Thiết lập

- Dùng đúng train temporal 9.258.647 tương tác. Sau khử trùng user–item, giữ 5.832.475 cặp thuộc 1.622.880 người dùng có ít nhất hai sản phẩm train khác nhau. 3.404.800 người dùng singleton không có vector MF; khi đánh giá dùng popularity train-only nếu người dùng không có factor hoặc lịch sử rỗng.
- BPR-MF chỉ học embedding ID người dùng/sản phẩm 32 chiều; negative được lấy từ các item có tương tác train và phải nằm ngoài toàn bộ positive train của chính người dùng. Không sử dụng validation/test để lấy gradient, tạo negative train hoặc học factor cho item zero-shot.
- Mini-batch 4.096, SGD 0,05, regularization 0,0001, seed 42, năm epoch. Mỗi epoch đánh giá toàn bộ validation; chọn checkpoint bằng trung bình NDCG@10 của zero-shot, extreme-cold, cold. File `best.npz` là epoch 1, `latest.npz` là epoch 5. Kết quả smoke 1.000 người dùng trong `tmp/` không được đưa vào bảng.
- BPR-MF là mô hình collaborative chỉ dùng ID; nó không xử lý sản phẩm zero-shot bằng nội dung. Người dùng chỉ xuất hiện sau train và item chưa có tương tác train dùng train-popularity fallback. Đây là giới hạn thực nghiệm của baseline, không phải lỗi của bộ dữ liệu.

## Kết quả validation tại checkpoint được chọn

| Mô hình | Overall NDCG@10 | Cold macro NDCG@10 | Warm NDCG@10 |
|---|---:|---:|---:|
| TF-IDF content profile | 0,260367 | 0,191270 | 0,467661 |
| Collaborative TruncatedSVD | 0,120842 | 0,030767 | 0,391065 |
| **BPR-MF (epoch 1)** | **0,135551** | **0,013553** | **0,501546** |

BPR-MF đạt Recall@10 overall **0,216825**; zero-shot **0**, extreme-cold **0,0290**, cold **0,0767**, warm **0,7616**. Warm tốt hơn SVD và TF-IDF theo NDCG@10, nhưng cold macro kém cả hai. Epoch 5 đạt warm NDCG@10 0,532274, song cold macro giảm còn 0,012797; theo tiêu chí đã chốt phải dùng checkpoint epoch 1. Đây là sampled ranking, không phải xếp hạng toàn catalog hay kết quả test.

## Tái lập và nguồn kiểm tra

```bash
python src/train_evaluate_bpr_mf.py --epochs 5 --factors 32 --batch-size 4096
```

Mã từ chối ghi đè output đã hoàn thành. `config.json` lưu hash nguồn và mã; `graph_report.json` ghi quy mô graph; `epoch_*.json` và `completed.json` ghi validation. Checkpoint factor/graph là dữ liệu lớn, được `.gitignore` loại khỏi Git. Kiểm thử nhỏ `tests/test_bpr_mf.py` xác nhận negative không trùng positive, BPR tăng margin và pipeline validation không mở test.

Nguồn so sánh: `data/processed/toys_games_full_temporal/models/content_baseline/metrics.json`, `models/collaborative_svd/report.json` và `models/bpr_mf/epoch_001.json`. Cần chạy SASRec/BERT4Rec chuẩn và UFM trên cùng giao thức trước khi kết luận về kiến trúc chính.
