"""Create deterministic sampled-ranking benchmarks shared by all recommenders."""

from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
CONTENT = DATA / "content"
OUTPUT = DATA / "evaluation"
SPLITS = ("validation", "test")
REGIMES = ("zero_shot", "extreme_cold", "cold", "warm")
REGIME_CODE = {name: index for index, name in enumerate(REGIMES)}
SAMPLES_PER_REGIME = 10_000
NEGATIVES = 99
SEED = 42


def load_catalog():
    ids = []
    train_counts = []
    with gzip.open(CONTENT / "items.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["row_index"]) != len(ids):
                raise AssertionError("Content item mapping is not contiguous")
            ids.append(row["parent_asin"])
            train_counts.append(int(row["train_count"]))
    id_to_row = {item: index for index, item in enumerate(ids)}
    validation_counts = np.zeros(len(ids), dtype=np.int32)
    test_counts = np.zeros(len(ids), dtype=np.int32)
    with gzip.open(DATA / "item_statistics.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            index = id_to_row[row["parent_asin"]]
            validation_counts[index] = int(row["validation_count"])
            test_counts[index] = int(row["test_count"])
    return (
        ids,
        id_to_row,
        np.asarray(train_counts, dtype=np.int32),
        validation_counts,
        test_counts,
    )


def reservoir_sample(split: str, min_interactions: int = 0):
    reservoirs = {regime: [] for regime in REGIMES}
    seen = Counter()
    rng = random.Random(SEED + SPLITS.index(split))
    path = DATA / f"{split}_with_history.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if min_interactions > 0:
                if int(row["history_length"]) + 1 < min_interactions:
                    continue
            regime = row["cold_start_regime"]
            seen[regime] += 1
            reservoir = reservoirs[regime]
            if len(reservoir) < SAMPLES_PER_REGIME:
                reservoir.append(row)
            else:
                replacement = rng.randrange(seen[regime])
                if replacement < SAMPLES_PER_REGIME:
                    reservoir[replacement] = row
    samples = []
    for regime in REGIMES:
        samples.extend(reservoirs[regime])
    samples.sort(
        key=lambda row: (
            REGIME_CODE[row["cold_start_regime"]],
            row["user_id"],
            int(row["timestamp"]),
            row["parent_asin"],
        )
    )
    return samples, seen


def choose_negatives(rng, pool, cdf, excluded, count):
    selected = []
    selected_set = set()
    while len(selected) < count:
        draws = pool[np.searchsorted(cdf, rng.random(max(128, count * 2)) * cdf[-1])]
        for candidate in draws:
            value = int(candidate)
            if value not in excluded and value not in selected_set:
                selected.append(value)
                selected_set.add(value)
                if len(selected) == count:
                    break
    return selected


def build_split(
    split,
    ids,
    id_to_row,
    train_counts,
    validation_counts,
    test_counts,
    min_interactions=0,
):
    print(f"Sampling {split} targets...", flush=True)
    samples, population = reservoir_sample(split, min_interactions)
    if split == "validation":
        counts = train_counts + validation_counts
    else:
        counts = train_counts + validation_counts + test_counts
    item_regime = np.zeros_like(train_counts, dtype=np.uint8)
    item_regime[(train_counts >= 1) & (train_counts <= 5)] = 1
    item_regime[(train_counts >= 6) & (train_counts <= 20)] = 2
    item_regime[train_counts > 20] = 3

    pools = {k: np.flatnonzero((item_regime == k) & (counts > 0)) for k in range(4)}
    cdfs = {k: np.cumsum(counts[pools[k]]) for k in range(4)}

    sample_count = len(samples)
    candidates = np.empty((sample_count, NEGATIVES + 1), dtype=np.int32)
    history_indptr = np.zeros(sample_count + 1, dtype=np.int64)
    history_values = []
    regime_codes = np.empty(sample_count, dtype=np.uint8)
    repeat_purchases = np.empty(sample_count, dtype=np.uint8)
    rng = np.random.default_rng(SEED + 100 + SPLITS.index(split))
    trace_path = OUTPUT / f"samples_{split}.csv.gz.tmp"
    with gzip.open(trace_path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "sample_index",
                "user_id",
                "parent_asin",
                "rating",
                "timestamp",
                "history",
                "history_length",
                "train_item_count",
                "cold_start_regime",
                "is_repeat_purchase",
            ),
        )
        writer.writeheader()
        for sample_index, row in enumerate(samples):
            target = id_to_row[row["parent_asin"]]
            history = [id_to_row[item] for item in row["history"].split()]
            excluded = set(history)
            excluded.add(target)
            candidates[sample_index, 0] = target
            regime_code = REGIME_CODE[row["cold_start_regime"]]
            regime_codes[sample_index] = regime_code
            candidates[sample_index, 1:] = choose_negatives(
                rng, pools[regime_code], cdfs[regime_code], excluded, NEGATIVES
            )
            history_values.extend(history)
            history_indptr[sample_index + 1] = len(history_values)
            is_repeat = 1 if target in set(history) else 0
            repeat_purchases[sample_index] = is_repeat
            writer.writerow({"sample_index": sample_index, "is_repeat_purchase": is_repeat, **row})
            if (sample_index + 1) % 10_000 == 0:
                print(f"  built {sample_index + 1:,}/{sample_count:,}", flush=True)

    arrays_path = OUTPUT / f"samples_{split}.npz.tmp"
    with arrays_path.open("wb") as output:
        np.savez_compressed(
            output,
            candidates=candidates,
            history_indices=np.asarray(history_values, dtype=np.int32),
            history_indptr=history_indptr,
            regime_codes=regime_codes,
            repeat_purchases=repeat_purchases,
        )
    trace_path.replace(OUTPUT / f"samples_{split}.csv.gz")
    arrays_path.replace(OUTPUT / f"samples_{split}.npz")
    return {
        "population_by_regime": {name: population[name] for name in REGIMES},
        "samples_by_regime": {
            name: int((regime_codes == code).sum()) for name, code in REGIME_CODE.items()
        },
        "empty_history_samples": int((np.diff(history_indptr) == 0).sum()),
        "repeat_purchase_samples": int(sum(
            1 for i in range(sample_count)
            if candidates[i, 0] in set(history_values[history_indptr[i]:history_indptr[i+1]])
        )),
        "eligible_candidate_items": len(eligible),
        "candidates_per_sample": NEGATIVES + 1,
        "negative_sampling": "Proportional to popularity among items observed by the end of the split; exclude target and history items.",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-interactions', type=int, default=0,
                        help='Minimum total interactions (train + history + 1) for a user to be evaluated')
    args = parser.parse_args()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ids, id_to_row, train_counts, validation_counts, test_counts = load_catalog()
    report = {
        "seed": SEED,
        "regimes": list(REGIMES),
        "samples_per_regime_cap": SAMPLES_PER_REGIME,
        "min_interactions_filter": args.min_interactions,
        "splits": {},
    }
    for split in SPLITS:
        report["splits"][split] = build_split(
            split,
            ids,
            id_to_row,
            train_counts,
            validation_counts,
            test_counts,
            args.min_interactions,
        )
    temporary = OUTPUT / "sampling_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "sampling_report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: shared evaluation samples created at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
