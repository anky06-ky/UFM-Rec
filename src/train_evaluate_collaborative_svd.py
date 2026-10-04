"""Train and evaluate a CPU-friendly implicit collaborative SVD baseline."""

from __future__ import annotations

import csv
import gzip
import json
from array import array
from itertools import groupby
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from evaluate_content_baseline import (
    load_train_counts,
    rank_of_target,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
CONTENT = DATA / "content"
EVALUATION = DATA / "evaluation"
OUTPUT = DATA / "models/collaborative_svd"
SPLITS = ("validation", "test")
FACTORS = 64
ITERATIONS = 20
PROFILE_ITEMS = 10
SEED = 42


def load_item_mapping():
    global_ids = []
    train_global_rows = []
    with gzip.open(CONTENT / "items.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            global_row = int(row["row_index"])
            if global_row != len(global_ids):
                raise AssertionError("Item mapping is not contiguous")
            global_ids.append(row["parent_asin"])
            if int(row["train_count"]) > 0:
                train_global_rows.append(global_row)
    global_to_collaborative = np.full(len(global_ids), -1, dtype=np.int32)
    global_to_collaborative[train_global_rows] = np.arange(
        len(train_global_rows), dtype=np.int32
    )
    id_to_global = {item: index for index, item in enumerate(global_ids)}
    return global_ids, id_to_global, np.asarray(train_global_rows), global_to_collaborative


def build_user_item_matrix(id_to_global, global_to_collaborative):
    print("Stage 1/3: building binary user-item matrix...", flush=True)
    indices = array("i")
    indptr = array("q", [0])
    kept_users = skipped_singleton_users = source_rows = duplicate_user_items = 0
    next_report = 1_000_000
    path = DATA / "interactions_train.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for _, user_rows in groupby(reader, key=lambda row: row["user_id"]):
            rows = list(user_rows)
            source_rows += len(rows)
            unique_items = {
                int(global_to_collaborative[id_to_global[row["parent_asin"]]])
                for row in rows
            }
            if -1 in unique_items:
                raise AssertionError("Train interaction references a non-train item")
            duplicate_user_items += len(rows) - len(unique_items)
            if len(unique_items) < 2:
                skipped_singleton_users += 1
            else:
                indices.extend(sorted(unique_items))
                indptr.append(len(indices))
                kept_users += 1
            if source_rows >= next_report:
                print(
                    f"  source rows {source_rows:,}; users with >=2 items {kept_users:,}",
                    flush=True,
                )
                next_report += 1_000_000

    index_values = np.frombuffer(indices, dtype=np.int32)
    pointer_values = np.frombuffer(indptr, dtype=np.int64)
    data = np.ones(len(index_values), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, index_values, pointer_values),
        shape=(kept_users, int(global_to_collaborative.max()) + 1),
    )
    matrix.sort_indices()
    return matrix, {
        "source_rows": source_rows,
        "users_with_at_least_two_unique_items": kept_users,
        "skipped_singleton_users": skipped_singleton_users,
        "duplicate_user_item_interactions": duplicate_user_items,
        "unique_user_item_pairs": int(matrix.nnz),
        "shape": list(matrix.shape),
    }


def train_factors(matrix, global_rows, catalog_size):
    print("Stage 2/3: TF-IDF weighting collaborative matrix...", flush=True)
    weighted = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(matrix)
    del matrix
    print(
        f"Stage 3/3: fitting TruncatedSVD({FACTORS}) on {weighted.shape[0]:,} users...",
        flush=True,
    )
    model = TruncatedSVD(
        n_components=FACTORS,
        n_iter=ITERATIONS,
        random_state=SEED,
    )
    model.fit(weighted)
    collaborative_factors = model.components_.T.astype(np.float32, copy=True)
    norms = np.linalg.norm(collaborative_factors, axis=1)
    nonzero = norms > 0
    collaborative_factors[nonzero] /= norms[nonzero, None]
    global_factors = np.zeros((catalog_size, FACTORS), dtype=np.float32)
    global_factors[global_rows] = collaborative_factors

    factors_path = OUTPUT / "item_factors.npy.tmp"
    with factors_path.open("wb") as output:
        np.save(output, global_factors, allow_pickle=False)
    factors_path.replace(OUTPUT / "item_factors.npy")
    model_path = OUTPUT / "svd.joblib.tmp"
    joblib.dump(model, model_path, compress=3)
    model_path.replace(OUTPUT / "svd.joblib")
    return global_factors, {
        "factors": FACTORS,
        "iterations": ITERATIONS,
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        "train_items_with_factor": int(nonzero.sum()),
        "factor_file_bytes": (OUTPUT / "item_factors.npy").stat().st_size,
    }


def evaluate_split(split, factors, train_counts):
    samples = np.load(EVALUATION / f"samples_{split}.npz")
    candidates = samples["candidates"]
    history_indices = samples["history_indices"]
    history_indptr = samples["history_indptr"]
    regime_codes = samples["regime_codes"]
    history_lengths = np.diff(history_indptr)
    has_factor = np.linalg.norm(factors, axis=1) > 0
    ranks = np.empty(len(candidates), dtype=np.uint8)

    for index, candidate_rows in enumerate(candidates):
        popularity = np.log1p(train_counts[candidate_rows])
        start, end = history_indptr[index : index + 2]
        history = history_indices[max(start, end - PROFILE_ITEMS) : end]
        if len(history):
            history = history[has_factor[history]]
        if len(history):
            profile = factors[history].sum(axis=0)
            scores = factors[candidate_rows] @ profile
            scale = float(popularity.max()) or 1.0
            scores = scores + popularity / scale * 1e-7
        else:
            scores = popularity
        ranks[index] = rank_of_target(scores, candidate_rows)
        if (index + 1) % 10_000 == 0:
            print(f"  {split}: {index + 1:,}/{len(candidates):,}", flush=True)

    ranks_path = OUTPUT / f"ranks_{split}.npz.tmp"
    with ranks_path.open("wb") as output:
        np.savez_compressed(
            output,
            collaborative_ranks=ranks,
            regime_codes=regime_codes,
            history_lengths=history_lengths,
        )
    ranks_path.replace(OUTPUT / f"ranks_{split}.npz")
    return summarize(ranks, regime_codes, history_lengths)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    global_ids, id_to_global, train_global_rows, global_to_collaborative = (
        load_item_mapping()
    )
    matrix, matrix_report = build_user_item_matrix(
        id_to_global, global_to_collaborative
    )
    factors, training_report = train_factors(
        matrix, train_global_rows, len(global_ids)
    )
    train_counts = load_train_counts()
    report = {
        "model": "Implicit collaborative TruncatedSVD over TF-IDF-weighted binary user-item histories.",
        "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "matrix": matrix_report,
        "training": training_report,
        "profile_items": PROFILE_ITEMS,
        "candidate_protocol": "Same fixed one-positive plus 99-negative samples as content baseline.",
        "metrics": {},
    }
    for split in SPLITS:
        print(f"Evaluating collaborative model on {split}...", flush=True)
        report["metrics"][split] = evaluate_split(split, factors, train_counts)
    temporary = OUTPUT / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: collaborative SVD trained and evaluated at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
