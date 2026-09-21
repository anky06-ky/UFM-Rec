"""Fit TF-IDF on train product text; transform and verify every split."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


DATA = Path(__file__).resolve().parents[1] / "data/processed/toys_games_70_15_15"
OUTPUT = DATA / "tfidf"
SPLITS = ("train", "validation", "test")
PARAMETERS = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "lowercase": True,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_products(split, expected_ids):
    frame = pd.read_csv(
        DATA / f"products_text_{split}.csv.gz",
        usecols=["parent_asin", "text"],
        dtype=str,
        keep_default_na=False,
    )
    assert frame["parent_asin"].is_unique
    assert set(frame["parent_asin"]) == expected_ids
    return frame


def self_check():
    model = TfidfVectorizer(dtype=np.float32)
    model.fit(["red truck", "blue truck", ""])
    vocabulary = model.vocabulary_.copy()
    idf = model.idf_.copy()
    vectors = model.transform(["holdoutonlytoken", "", "red truck"])
    assert vectors[:2].nnz == 0
    assert vectors[2].nnz > 0
    assert "holdoutonlytoken" not in model.vocabulary_
    assert model.vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.idf_, idf)


def save_and_verify(split, products, matrix, model, expected_ids):
    matrix = matrix.tocsr()
    matrix.sort_indices()
    assert matrix.shape == (len(products), len(model.vocabulary_))
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
    has_vector = np.diff(matrix.indptr) > 0
    norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    np.testing.assert_allclose(norms[has_vector], 1.0, atol=2e-5)
    assert not has_vector[products["text"].eq("").to_numpy()].any()

    matrix_path = OUTPUT / f"tfidf_{split}.npz"
    temporary = matrix_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        sparse.save_npz(handle, matrix, compressed=True)
    with temporary.open("rb") as handle:
        restored = sparse.load_npz(handle)
    assert restored.shape == matrix.shape and restored.dtype == matrix.dtype
    assert (restored != matrix).nnz == 0

    # Transform samples again using the model loaded from disk, and compare
    # against their row positions in the full matrix.
    sample_rows = np.linspace(0, len(products) - 1, min(64, len(products)), dtype=int)
    difference = restored[sample_rows] - model.transform(products.iloc[sample_rows]["text"])
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-6
    del restored

    mapping_path = OUTPUT / f"items_{split}.csv.gz"
    mapping_temp = mapping_path.with_suffix(".gz.tmp")
    mapping = pd.DataFrame({
        "row_index": np.arange(len(products)),
        "parent_asin": products["parent_asin"].to_numpy(),
        "has_text_vector": has_vector.astype(np.int8),
    })
    mapping.to_csv(mapping_temp, index=False, compression="gzip")
    reread = pd.read_csv(mapping_temp, compression="gzip", dtype={"parent_asin": str})
    pd.testing.assert_frame_equal(mapping, reread, check_dtype=False)
    assert set(reread["parent_asin"]) == expected_ids

    temporary.replace(matrix_path)
    mapping_temp.replace(mapping_path)
    return {
        "shape": list(matrix.shape),
        "nonzero_values": int(matrix.nnz),
        "zero_vector_products": int((~has_vector).sum()),
        "zero_vector_ids": products.loc[~has_vector, "parent_asin"].tolist(),
        "sparse_memory_mib": round((matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2, 2),
        "matrix_file_bytes": matrix_path.stat().st_size,
        "source_sha256": file_hash(DATA / f"products_text_{split}.csv.gz"),
        "sample_rows_verified": len(sample_rows),
    }


def main():
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(DATA / "item_split.csv.gz", dtype=str)
    expected = {split: set(mapping.loc[mapping["split"].eq(split), "parent_asin"]) for split in SPLITS}
    report = {
        "fit_split": "train",
        "fit_fields": ["text"],
        "parameters": {**PARAMETERS, "dtype": "float32"},
        "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "zero_vector_policy": "Retain row and mark has_text_vector=0. Downstream ranking needs a fallback; no fabricated text.",
        "verification": "Exact sparse serialization; all item IDs and norms; sampled retransformation; frozen vocabulary and IDF on validation/test.",
        "splits": {},
    }
    started = perf_counter()
    print("Reading train product text...", flush=True)
    products = read_products("train", expected["train"])
    model = TfidfVectorizer(**PARAMETERS)
    print("Fitting vocabulary and IDF on TRAIN ONLY...", flush=True)
    vectors = model.fit_transform(products["text"])
    vocabulary = model.vocabulary_.copy()
    idf = model.idf_.copy()
    model_temp = OUTPUT / "vectorizer.joblib.tmp"
    joblib.dump(model, model_temp, compress=3)
    model = joblib.load(model_temp)
    assert model.vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.idf_, idf)

    for split in SPLITS:
        if split != "train":
            products = read_products(split, expected[split])
            print(f"Transforming {split} with frozen train vocabulary/IDF...", flush=True)
            vectors = model.transform(products["text"])
        print(f"Saving and verifying {split}: {vectors.shape}...", flush=True)
        report["splits"][split] = save_and_verify(split, products, vectors, model, expected[split])
        assert model.vocabulary_ == vocabulary
        np.testing.assert_array_equal(model.idf_, idf)
        del products, vectors

    model_temp.replace(OUTPUT / "vectorizer.joblib")
    report["elapsed_seconds"] = round(perf_counter() - started, 2)
    report_temp = OUTPUT / "tfidf_report.json.tmp"
    report_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_temp.replace(OUTPUT / "tfidf_report.json")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print("PASS: TF-IDF features saved and verified.")


if __name__ == "__main__":
    main()
