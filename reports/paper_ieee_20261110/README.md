# Báo cáo UFM-Rec theo mẫu IEEE (10/11/2026)

- `ufm_rec_ieee.tex`: nguồn LaTeX (IEEEtran, conference, tiếng Việt). Sơ đồ kiến trúc và biểu đồ vẽ bằng TikZ/pgfplots ngay trong file, không cần ảnh ngoài.
- `ufm_rec_ieee.pdf`: bản đã biên dịch (5 trang).

## Biên dịch

**Overleaf (dễ nhất):** tạo project mới, tải `ufm_rec_ieee.tex` lên, chọn compiler **pdfLaTeX**. Overleaf có sẵn IEEEtran và vntex (font tiếng Việt T5).

**Máy cá nhân:** cần TeX Live có `texlive-publishers` (IEEEtran) và `texlive-lang-other` (vntex):

```
pdflatex ufm_rec_ieee.tex
pdflatex ufm_rec_ieee.tex
```

## Nội dung và nguồn số liệu

| Mục | Nguồn số liệu |
|---|---|
| UFM v2 (Bảng I) | `reports/rebuild_20261108/runs/ufm_v2/`, `reports/SUMMARY_20261109.md` |
| Kết hợp + xu hướng (Bảng II, Hình 3) | `reports/improve_20261108/trend_blend_v1run.json`, `trend_blend_v2run.json` |
| Truy xuất RAG (Bảng III) | `reports/rag_20261109/rerank_ablation.json`, `expansion_ablation_Qwen2.5-1.5B-Instruct.json` |
| Chưng cất (Bảng IV) | `reports/rag_20261109/distill_eval.json`, `student/completed.json` |

Danh sách tác giả, đơn vị và lời cảm ơn cần nhóm kiểm tra lại trước khi nộp.
