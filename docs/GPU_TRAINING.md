# Huấn luyện GPU trên FITLAB

Mô hình thử nghiệm: content-aware two-tower, item encoder dùng 64 trọng số TF-IDF cao nhất và embedding ID chỉ cho item có trong train; user encoder dùng Transformer 2 lớp trên tối đa 20 sản phẩm trước đó. Đây là mô hình tự triển khai, không phải bản tái hiện chính thức của SASRec hay mô hình được bảo đảm tốt nhất.

Vocabulary và IDF đã được fit từ train. Gradient chỉ sử dụng `train_with_history.csv.gz`. ID dropout 50% giúp nhánh nội dung học khi ID bị ẩn. In-batch negatives loại item trùng target, item trong lịch sử và target của cùng người dùng trong batch. Các tương tác dương khác ngoài lịch sử vẫn có thể bị coi là negative, một hạn chế của implicit feedback.

Tham khảo thiết kế: https://www.tensorflow.org/recommenders/examples/basic_retrieval
Checkpoint: https://docs.pytorch.org/tutorials/beginner/saving_loading_models

## Lệnh chạy

Chạy trong `/home/coder/iDragonCloud/DA_AI`, dùng đúng Python GPU:

```bash
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py --mode self-test
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py --output runs/smoke_gpu_v1 --epochs 1 --max-steps 30 --validation-cap 400
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py --output runs/content_transformer_v1 --epochs 20
```

20 epoch là giới hạn tối đa. Dừng sớm sau 3 epoch không cải thiện NDCG@10 của nhóm có lịch sử trên validation. Tất cả 40.000 candidate sets validation cố định được đánh giá mỗi epoch của lượt đầy đủ. Các metric được báo theo cả nhóm cold-start và người dùng chưa có lịch sử (dùng popularity tính từ train).

Mỗi 1.000 bước lưu `latest.pt`; checkpoint tốt nhất theo validation là `best.pt`. Checkpoint được ghi file tạm rồi đổi tên. Khi container ngắt, có thể tiếp tục từ checkpoint gần nhất:

```bash
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py --output runs/content_transformer_v1 --epochs 20 --resume
```

Theo dõi `runs/content_transformer_v1.log` và `runs/content_transformer_v1/history.jsonl`. Nếu cache chuẩn bị dở chưa có `complete.json`, chọn tên cache mới bằng `--cache`; không dùng file chuẩn bị dở.

Chỉ sau khi đã chốt mô hình bằng validation, đánh giá test trong một lệnh riêng:

```bash
/home/coder/iDragonCloud/.venv/bin/python -u src/train_gpu_recommender.py --mode test --output runs/content_transformer_v1
```

Đánh giá là sampled ranking: 1 positive + 99 negative, không phải xếp hạng toàn bộ 767.045 sản phẩm. Metadata snapshot có thể chứa nội dung cập nhật sau thời điểm tương tác; chia theo thời gian không loại bỏ được hạn chế lịch sử của metadata. Mô hình cần được so sánh với baseline trên cùng candidate sets trước khi kết luận tốt hơn.

GPU job chạy bằng `nohup` có thể tiếp tục khi đóng tab trình duyệt, nhưng không thể tiếp tục khi trường dừng/xóa container. Checkpoint trên iDragonCloud dùng để resume.
