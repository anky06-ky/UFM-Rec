# UFM-Rec

**Bắt đầu:** [START.md](START.md) · [Việc tiếp theo](docs/next.md) · `python status.py`

**Cập nhật 02/10/2026:** CLIP hoàn tất 767.045 sản phẩm; UFM full hoàn tất 12
epoch. Best validation checkpoint là epoch 9: overall NDCG@10 **0,285611**,
cold macro **0,163738**. UFM cao hơn TF-IDF ở overall/warm nhưng thấp hơn ở cold
macro; test chưa chạy. Bảy ablation đang chờ GPU chia sẻ rảnh, supervisor còn sống.
[Báo cáo mới nhất](reports/Tien_do_UFM_2026-10-02.md) và [việc tiếp theo](docs/next.md).

## Bàn giao baseline: 30/09/2026

- **Mở demo:** chạy [Start_Demo.cmd](Start_Demo.cmd), hoặc vào
  [localhost:8765](http://127.0.0.1:8765) khi server đang chạy.
  Demo local dùng TF-IDF; demo UFM đang được nghiệm thu qua FITLAB.
  [Hướng dẫn + kịch bản trình bày](docs/demo.md).
- **Tiến trình:** [PDF 5 trang](output/pdf/Bao_cao_tien_trinh_UFM_Rec_2026_09_30.pdf)
  và [bản Markdown](reports/Bao_cao_tien_trinh_UFM_Rec_2026_09_30.md).
- BPR-MF đã hoàn thành 3 seed (42, 7, 2026), 5 epoch/seed. Validation overall
  NDCG@10 trung bình **0,135465 ± 0,000231** (độ lệch chuẩn mẫu).
  Cold macro **0,013602 ± 0,000233**. Chưa đánh giá test.
- Calibration TF-IDF: fit 20.000 validation sớm, audit 20.000 validation muộn;
  ECE top-1 từ **0,168601** xuống **0,116684**. Đây là kiểm chứng thăm dò cho
  baseline trong sampled candidate set, chưa phải calibration UFM.
- **36 kiểm thử local PASS**; 4 kịch bản API demo trên catalog thật PASS.
- Snapshot FITLAB 11:07 UTC+07: CLIP **522.880/767.045 (68,17%)**;
  UFM và 7 ablation còn chờ. Các phần SASRec/BERT4Rec, concat hybrid,
  dataset thứ hai, nhiều seed/calibration UFM, test cuối và slide vẫn chưa hoàn tất.

Các kết quả benchmark bên dưới là mốc baseline trước UFM; dùng báo cáo mới nhất ở
trên cho trạng thái dự án hiện tại.

Phạm vi đã chốt: giữ đầy đủ Foundation Model văn bản/ảnh, dual uncertainty và
UGAF theo proposal. Xem [kế hoạch triển khai đầy đủ](docs/scope.md).
Lõi mô hình mới và kiểm thử nằm ở `src/ufm_model.py`, `tests/test_ufm_core.py`;
pipeline Foundation Model mới chưa phải một lượt UFM đã huấn luyện hoàn chỉnh.

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

## Thử nghiệm GPU trên FITLAB

Mã `src/train_gpu_recommender.py` bổ sung một mô hình content-aware two-tower:
nhánh sản phẩm dùng TF-IDF và embedding ID chỉ dành cho sản phẩm có trong train;
nhánh người dùng dùng Transformer 2 lớp trên lịch sử. Đây là thử nghiệm mới,
chưa được xác nhận tốt hơn các baseline ở bảng trên.

Xem [hướng dẫn GPU](docs/gpu.md). Notebook
`notebooks/02_baseline.ipynb` chỉ đọc log và checkpoint của job đang chạy.
Lượt train đầy đủ dùng 4.230.848 mẫu train có lịch sử, chọn checkpoint bằng validation,
lưu định kỳ để resume và không tự động mở test.

## Tái lập kết quả GPU và baseline

Seed của split/sampling là `42`. Phiên bản thư viện được ghim trong `requirements.txt`; checksum dữ liệu nguồn và các quy tắc lọc/split được ghi trong các JSON manifest.

Thử nghiệm GPU dùng PyTorch có sẵn trên FITLAB; phiên bản runtime, cấu hình và fingerprint dữ liệu được lưu riêng trong `runs/<tên_lượt>/config.json`.

## UFM đầy đủ và chiến dịch chạy nền (28/09/2026)

Giữ đầy đủ Foundation Model, uncertainty và UGAF; không gọi mô hình TF-IDF cũ
là UFM. Đối chứng GPU đã hoàn tất 8 epoch, best epoch 5: validation toàn bộ
NDCG@10 **0.277327**, known-user NDCG@10 **0.380043**. Đây không phải test metric.

CLIP ViT-B/32 đóng băng đang trích text/image toàn catalog. Queue UFM chờ tối đa
168 giờ, kiểm tra GPU smoke/resume rồi train full; suite nối tiếp 7 ablation có
cửa sổ khởi chạy 7 ngày, không buộc train đủ ngày khi early stopping. Tại snapshot
17:55 ngày 28/09, CLIP đạt 101.888/767.045 dòng; **chưa có kết quả UFM GPU**.
32 correctness tests local và FITLAB PASS; không phải bằng chứng chất lượng mô hình.

```bash
python src/monitor_ufm_campaign.py
python src/demo_recommender.py --backend content --port 8765
```

Demo CPU tại `http://127.0.0.1:8765` dùng TF-IDF thật, loại sản phẩm đã xem và
xếp hạng toàn catalog. Backend UFM chỉ nhận production checkpoint đã hoàn tất;
chưa được chạy với mô hình full hiện còn trong queue. Khác với bảng test baseline
cổ điển phía trên, chiến dịch GPU mới không mở test để tuning.

Xem [vận hành dài ngày và demo](docs/ops.md),
[phạm vi đầy đủ](docs/scope.md),
[báo cáo tiến độ mới](reports/Bao_cao_tien_do_UFM_Rec_2026_09_28_cap_nhat.md)
và [bản PDF mới nhất](output/pdf/Bao_cao_tien_trinh_UFM_Rec_2026_09_30.pdf).
Job tách nền không phụ thuộc máy cá nhân, nhưng container FITLAB phải còn sống;
không tự khởi động sau reboot. Dữ liệu/model/venv và thông tin đăng nhập không
đưa lên GitHub. MF/SASRec/BERT4Rec chuẩn, nhiều seed, calibration, dataset thứ hai,
kiểm tra VRAM 16GB và đánh giá test cuối vẫn còn trong kế hoạch hoàn thành proposal.

## Cập nhật FITLAB (30/09/2026)

CLIP bị dừng bởi `SIGKILL` ở 497.856/767.045 sản phẩm ngày 29/09. Ngày 30/09,
ba runner đã được phục hồi từ checkpoint; cursor CLIP đã tăng lên 497.984/767.045.
UFM và ablation đang chờ đầu vào, chưa có kết quả GPU UFM. Xem
[biên bản phục hồi](reports/FITLAB_recovery_2026_09_30.md) để kiểm tra trạng thái
và điều kiện tiếp tục.

## Bổ sung baseline BPR-MF (30/09/2026)

Đã train BPR-MF chỉ dùng ID trên train temporal, chọn checkpoint epoch 1 bằng
validation cold macro NDCG@10. Kết quả validation: overall NDCG@10 0,135551;
cold macro 0,013553; warm 0,501546. Nhóm zero-shot có Recall@10 bằng 0,
phù hợp giới hạn của baseline không dùng nội dung. Chưa đánh giá test. Xem
[báo cáo BPR-MF](reports/BPR_MF_validation_2026_09_30.md) và chạy
`python src/train_evaluate_bpr_mf.py` để tái lập với dữ liệu đầy đủ.
