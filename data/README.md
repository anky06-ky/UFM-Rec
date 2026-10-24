# Data layout

The raw dataset and generated matrices are intentionally excluded from Git because they are reproducible and several files are hundreds of MB to multiple GB. Small JSON manifests and evaluation reports are versioned.

Expected inputs:

```text
data/raw/toys_games_full/Toys_and_Games.jsonl.gz
data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz
```

The official Amazon Reviews 2023 Toys & Games product metadata also has a pinned
Parquet mirror in `data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games/`. Restore
and verify its five shards with:

```bash
python scripts/download_toys_metadata.py
```

The downloader resumes `.part` files and verifies the pinned size and SHA-256
before renaming them. Its small `download_manifest.json` records the source
revision and hashes. Parquet files are ignored by Git. To build the image URL
catalog, the prepared `content/items.csv.gz`, `content/products_text.csv.gz`,
and `split_manifest.json` must also be present; `text` is used only for the
existing row alignment and does not supply image URLs. Install
`requirements-catalog-parquet.txt` before passing the ordered Parquet shards to
`src/prepare_ufm_catalog.py`.

The interaction file is the public [Amazon Reviews 2023 Toys and Games dataset](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Toys_and_Games.jsonl.gz). Its expected SHA-256 and record count are stored in `raw/toys_games_full/source_manifest.json`.

Run the pipeline from the repository root. Each stage writes under `data/processed/` and refuses to overwrite a completed output directory. Remove or archive that stage's output before intentionally rebuilding it.
