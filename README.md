# UFM-Rec

Hệ khuyến nghị sản phẩm mới trên thương mại điện tử, được xây dựng trên Amazon Reviews 2023 - Toys and Games.

Phiên bản hiện tại đã có pipeline tiền xử lý theo thời gian, đặc trưng nội dung TF-IDF cho toàn bộ sản phẩm, content-based baseline, collaborative TruncatedSVD và hybrid reciprocal-rank fusion. Bộ mẫu đánh giá cố định được dùng chung giữa các mô hình để so sánh công bằng.

## Dữ liệu và quy mô

- Nguồn: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), danh mục Toys and Games.
- 16.260.406 review gốc; giữ lại 11.572.689 tương tác mua hàng đã xác minh và có rating từ 4 trở lên.
- Chia theo thời gian 80/10/10: 9.258.647 train, 1.159.827 validation, 1.154.215 test.
- 767.045 sản phẩm có vector TF-IDF 20.000 chiều; vocabulary chỉ fit trên sản phẩm đã xuất hiện trong train.
- Bốn chế độ cold-start: `zero_shot`, `extreme_cold`, `cold`, `warm`.

File dữ liệu và model lớn không được commit. Xem [data/README.md](data/README.md) để biết cấu trúc thư mục và nguồn tải.

## Cài đặt

Yêu cầu Python 3.11+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy pipeline quy mô lớn

Chạy lần lượt từ thư mục gốc:

```bash
python src/prepare_full_temporal.py
python src/build_full_content_features.py
python src/build_recommender_samples.py
python src/evaluate_content_baseline.py
python src/train_evaluate_collaborative_svd.py
python src/train_evaluate_hybrid_rrf.py
```

Pipeline tự kiểm tra input, ghi file theo cách an toàn qua file tạm và từ chối ghi đè output đã hoàn thành. Các manifest, tham số và metric nhỏ được lưu trong `data/` để có thể kiểm tra lại thí nghiệm.

## Kết quả

Đánh giá trên test sample cân bằng 40.000 trường hợp, mỗi trường hợp có 1 positive và 99 negative cố định:

| Mô hình | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | 0.2669 | 0.1691 | 0.1393 |
| TF-IDF content profile | 0.4013 | **0.2658** | **0.2243** |
| Collaborative TruncatedSVD | 0.2193 | 0.1310 | 0.1043 |
| Candidate-aware hybrid RRF | **0.4231** | 0.2657 | 0.2172 |

Theo Recall@10 trên từng chế độ:

| Mô hình | Zero-shot | Extreme cold | Cold | Warm |
|---|---:|---:|---:|---:|
| TF-IDF content profile | 0.2816 | 0.2874 | 0.3296 | **0.7065** |
| Collaborative TruncatedSVD | 0.0000 | 0.0775 | 0.1498 | 0.6497 |
| Candidate-aware hybrid RRF | 0.2731 | 0.2575 | 0.3146 | **0.8473** |

Hybrid dùng RRF với trọng số theo train-frequency regime của từng candidate. Trọng số được chọn chỉ trên validation; test không tham gia tuning. Cổng phối hợp đã học được:

| Candidate regime | Popularity | Content | Collaborative |
|---|---:|---:|---:|
| Zero-shot | 0.0 | 1.0 | 0.0 |
| Extreme cold | 0.1 | 0.9 | 0.0 |
| Cold | 0.1 | 0.9 | 0.0 |
| Warm | 0.7 | 0.3 | 0.0 |

Hybrid tăng Recall@10 thêm 5,4% tương đối so với content baseline và cải thiện mạnh nhóm warm; NDCG@10 gần như ngang nhau. Collaborative SVD được giữ làm baseline, nhưng validation đã chọn trọng số 0 cho tín hiệu này vì chưa mang lại giá trị bổ sung so với popularity và content.

## Cấu trúc chính

```text
src/
  prepare_full_temporal.py               # lọc, khử trùng và chia theo thời gian
  build_full_content_features.py         # văn bản sản phẩm + TF-IDF
  build_recommender_samples.py           # candidate set dùng chung
  evaluate_content_baseline.py           # popularity + content baseline
  train_evaluate_collaborative_svd.py    # collaborative SVD baseline
  train_evaluate_hybrid_rrf.py            # tune trên validation + test hybrid
docs/
  Proposal.pdf
data/
  README.md
  raw/**/source_manifest.json
  processed/**/{manifest,report,metrics}.json
```

## Tái lập kết quả

Seed của split/sampling là `42`. Phiên bản thư viện được ghim trong `requirements.txt`; checksum dữ liệu nguồn và các quy tắc lọc/split được ghi trong các JSON manifest.
