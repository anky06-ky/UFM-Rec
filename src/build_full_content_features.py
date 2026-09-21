"""Build leakage-safe text and TF-IDF features for every temporal-split item."""

from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from build_product_text import clean_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
METADATA = ROOT / "data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz"
OUTPUT = DATA / "content"
PRODUCTS = OUTPUT / "products_text.csv.gz"
SOURCE_FIELDS = ("title", "features", "description", "store", "categories")
TEXT_COLUMNS = (
    "parent_asin",
    "title",
    "text",
    "train_count",
    "validation_count",
    "test_count",
    "cold_start_regime",
)
PARAMETERS = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "lowercase": True,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}
BATCH_SIZE = 5_000


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_item_statistics() -> dict[str, dict[str, str]]:
    path = DATA / "item_statistics.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        rows = {row["parent_asin"]: row for row in csv.DictReader(source)}
    if not rows:
        raise ValueError("No item statistics found")
    return rows


def build_product_text(items: dict[str, dict[str, str]]) -> dict:
    print("Stage 1/3: extracting product text from metadata...", flush=True)
    temporary = PRODUCTS.with_suffix(PRODUCTS.suffix + ".tmp")
    seen = set()
    stats = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TEXT_COLUMNS)
        writer.writeheader()
        with gzip.open(METADATA, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                record = json.loads(line)
                item = record.get("parent_asin")
                item_stats = items.get(item)
                if item_stats is not None and item not in seen:
                    cleaned = {field: clean_text(record.get(field)) for field in SOURCE_FIELDS}
                    text = " ".join(value for value in cleaned.values() if value)
                    writer.writerow(
                        {
                            "parent_asin": item,
                            "title": cleaned["title"],
                            "text": text,
                            **{column: item_stats[column] for column in TEXT_COLUMNS[3:]},
                        }
                    )
                    seen.add(item)
                    stats["products"] += 1
                    stats["empty_text"] += int(not text)
                    stats["train_products_with_text"] += int(
                        int(item_stats["train_count"]) > 0 and bool(text)
                    )
                    stats["total_characters"] += len(text)
                    stats["max_characters"] = max(stats["max_characters"], len(text))
                if line_number % 100_000 == 0:
                    print(
                        f"  metadata {line_number:,}; matched {len(seen):,}/{len(items):,}",
                        flush=True,
                    )

        missing = sorted(items.keys() - seen)
        for item in missing:
            item_stats = items[item]
            writer.writerow(
                {
                    "parent_asin": item,
                    "title": "",
                    "text": "",
                    **{column: item_stats[column] for column in TEXT_COLUMNS[3:]},
                }
            )
        stats["products"] += len(missing)
        stats["empty_text"] += len(missing)
        stats["missing_metadata"] = len(missing)

    if stats["products"] != len(items):
        raise AssertionError("Product text row count mismatch")
    temporary.replace(PRODUCTS)
    stats["mean_characters"] = round(
        stats["total_characters"] / stats["products"], 2
    )
    return dict(stats)


def training_texts():
    yielded = 0
    with gzip.open(PRODUCTS, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["train_count"]) > 0 and row["text"]:
                yielded += 1
                if yielded % 50_000 == 0:
                    print(f"  fit texts {yielded:,}", flush=True)
                yield row["text"]


def transform_batch(model, rows, row_offset, mapping_writer, zero_by_regime):
    matrix = model.transform(row["text"] for row in rows).tocsr()
    matrix.sort_indices()
    has_vector = np.diff(matrix.indptr) > 0
    norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    np.testing.assert_allclose(norms[has_vector], 1.0, atol=2e-5)
    for index, (row, present) in enumerate(zip(rows, has_vector)):
        mapping_writer.writerow(
            {
                "row_index": row_offset + index,
                "parent_asin": row["parent_asin"],
                "train_count": row["train_count"],
                "cold_start_regime": row["cold_start_regime"],
                "has_text_vector": int(present),
            }
        )
        if not present:
            zero_by_regime[row["cold_start_regime"]] += 1
    return matrix


def build_tfidf(expected_products: int, fit_products_with_text: int) -> dict:
    print("Stage 2/3: fitting TF-IDF on train-visible product text only...", flush=True)
    model = TfidfVectorizer(**PARAMETERS)
    model.fit(training_texts())
    if len(model.vocabulary_) != PARAMETERS["max_features"]:
        print(f"  vocabulary contains {len(model.vocabulary_):,} terms", flush=True)

    model_path = OUTPUT / "vectorizer.joblib"
    model_temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, model_temporary, compress=3)
    model = joblib.load(model_temporary)

    print("Stage 3/3: transforming every product with frozen train vocabulary...", flush=True)
    mapping_path = OUTPUT / "items.csv.gz"
    mapping_temporary = mapping_path.with_suffix(".gz.tmp")
    matrices = []
    batch = []
    row_offset = 0
    zero_by_regime = Counter()
    with gzip.open(mapping_temporary, "wt", encoding="utf-8", newline="") as mapping_file:
        mapping_writer = csv.DictWriter(
            mapping_file,
            fieldnames=(
                "row_index",
                "parent_asin",
                "train_count",
                "cold_start_regime",
                "has_text_vector",
            ),
        )
        mapping_writer.writeheader()
        with gzip.open(PRODUCTS, "rt", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                batch.append(row)
                if len(batch) == BATCH_SIZE:
                    matrices.append(
                        transform_batch(
                            model, batch, row_offset, mapping_writer, zero_by_regime
                        )
                    )
                    row_offset += len(batch)
                    batch = []
                    if row_offset % 50_000 == 0:
                        print(f"  transformed {row_offset:,}/{expected_products:,}", flush=True)
            if batch:
                matrices.append(
                    transform_batch(model, batch, row_offset, mapping_writer, zero_by_regime)
                )
                row_offset += len(batch)

    if row_offset != expected_products:
        raise AssertionError("TF-IDF mapping row count mismatch")
    matrix = sparse.vstack(matrices, format="csr", dtype=np.float32)
    matrix.sort_indices()
    del matrices
    gc.collect()
    if matrix.shape != (expected_products, len(model.vocabulary_)):
        raise AssertionError("TF-IDF matrix shape mismatch")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise AssertionError("TF-IDF matrix contains invalid values")

    matrix_path = OUTPUT / "tfidf_all.npz"
    matrix_temporary = matrix_path.with_suffix(".npz.tmp")
    with matrix_temporary.open("wb") as handle:
        sparse.save_npz(handle, matrix, compressed=True)
    shape = list(matrix.shape)
    nonzero = int(matrix.nnz)
    sparse_mib = round(
        (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2,
        2,
    )
    del matrix
    gc.collect()

    with matrix_temporary.open("rb") as handle:
        restored = sparse.load_npz(handle)
    if list(restored.shape) != shape or restored.dtype != np.float32:
        raise AssertionError("Saved TF-IDF matrix failed reload verification")
    del restored
    gc.collect()

    model_temporary.replace(model_path)
    mapping_temporary.replace(mapping_path)
    matrix_temporary.replace(matrix_path)
    return {
        "fit_products_with_text": fit_products_with_text,
        "vocabulary_size": len(model.vocabulary_),
        "shape": shape,
        "nonzero_values": nonzero,
        "sparse_memory_mib": sparse_mib,
        "zero_vectors_by_regime": dict(zero_by_regime),
        "matrix_file_bytes": matrix_path.stat().st_size,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items = read_item_statistics()
    text_report = build_product_text(items)
    tfidf_report = build_tfidf(len(items), text_report["train_products_with_text"])
    report = {
        "source_fields": list(SOURCE_FIELDS),
        "text_cleaning": "Remove HTML/script/style, decode entities, normalize NFC, remove selected invisible characters and collapse whitespace.",
        "fit_policy": "Fit TF-IDF only on products with train_count > 0; transform every item with frozen vocabulary and IDF.",
        "parameters": {**PARAMETERS, "dtype": "float32"},
        "versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "product_text": text_report,
        "tfidf": tfidf_report,
        "products_sha256": file_hash(PRODUCTS),
    }
    temporary = OUTPUT / "content_report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "content_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"PASS: full content features created at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
