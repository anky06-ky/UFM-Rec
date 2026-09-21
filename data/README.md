# Data layout

The raw dataset and generated matrices are intentionally excluded from Git because they are reproducible and several files are hundreds of MB to multiple GB. Small JSON manifests and evaluation reports are versioned.

Expected inputs:

```text
data/raw/toys_games_full/Toys_and_Games.jsonl.gz
data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz
```

The interaction file is the public [Amazon Reviews 2023 Toys and Games dataset](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Toys_and_Games.jsonl.gz). Its expected SHA-256 and record count are stored in `raw/toys_games_full/source_manifest.json`.

Run the pipeline from the repository root. Each stage writes under `data/processed/` and refuses to overwrite a completed output directory. Remove or archive that stage's output before intentionally rebuilding it.
