"""Create deterministic sampled-ranking benchmarks shared by all recommenders."""

from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from temporal_positives import positive_events, known_before, prior_paths


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


def load_catalog(data=None):
    data = DATA if data is None else Path(data)
    content = CONTENT if data == DATA else data / 'content'
    ids = []
    train_counts = []
    with gzip.open(content / "items.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["row_index"]) != len(ids):
                raise AssertionError("Content item mapping is not contiguous")
            ids.append(row["parent_asin"])
            train_counts.append(int(row["train_count"]))
    id_to_row = {item: index for index, item in enumerate(ids)}
    validation_counts = np.zeros(len(ids), dtype=np.int32)
    test_counts = np.zeros(len(ids), dtype=np.int32)
    with gzip.open(data / "item_statistics.csv.gz", "rt", encoding="utf-8", newline="") as source:
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


def reservoir_sample(split: str, min_interactions: int = 0, *, data=None, cap=None, seed=None):
    data = DATA if data is None else Path(data)
    cap = SAMPLES_PER_REGIME if cap is None else cap
    seed = SEED if seed is None else seed
    if cap < 1 or min_interactions < 0:
        raise ValueError('Positive sample cap and nonnegative history filter required.')
    reservoirs = {regime: [] for regime in REGIMES}
    seen = Counter()
    rng = random.Random(seed + SPLITS.index(split))
    path = data / f"{split}_with_history.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if min_interactions > 0:
                if int(row["history_length"]) + 1 < min_interactions:
                    continue
            regime = row["cold_start_regime"]
            seen[regime] += 1
            reservoir = reservoirs[regime]
            if len(reservoir) < cap:
                reservoir.append(row)
            else:
                replacement = rng.randrange(seen[regime])
                if replacement < cap:
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


def choose_negatives(rng, pool, cdf, excluded, count, *, validate=True):
    pool, cdf = np.asarray(pool), np.asarray(cdf)
    if count == 0:
        return []
    if count < 0 or pool.ndim != 1 or cdf.shape != pool.shape or not len(pool):
        raise ValueError('Nonempty matching candidate pool/CDF and nonnegative count required.')
    if validate and (not np.isfinite(cdf).all() or cdf[0] <= 0 or (np.diff(cdf) <= 0).any()):
        raise ValueError('CDF must have finite, strictly positive increments.')
    if len(pool) - len(excluded) < count and np.count_nonzero(~np.isin(pool, list(excluded))) < count:
        raise RuntimeError(f'Cannot draw {count} unique negatives after exclusions.')
    selected = []
    selected_set = set()
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError(
                f"Cannot draw {count} unique negatives from a pool of {len(pool)} items "
                f"({len(selected)} unique after {attempts} attempts)."
            )
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
    output=None,
    *, data=None, cap=None, negatives=None, seed=None, exclude_known_positives=False,
):
    output = OUTPUT if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Sampling {split} targets...", flush=True)
    for name in (f'samples_{split}.npz', f'samples_{split}.csv.gz'):
        if (output / name).exists():
            raise FileExistsError(f'Refusing to overwrite fixed samples: {output / name}')
    negatives = NEGATIVES if negatives is None else negatives
    seed = SEED if seed is None else seed
    samples, population = reservoir_sample(split, min_interactions, data=data, cap=cap, seed=seed)
    if not samples:
        raise ValueError('No targets remain after filtering; refusing to write an empty benchmark.')
    known = (positive_events(prior_paths(DATA if data is None else data, split),
                             id_to_row, {row['user_id'] for row in samples})
             if exclude_known_positives else {})
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
    eligible = np.flatnonzero(counts > 0)

    sample_count = len(samples)
    candidates = np.empty((sample_count, negatives + 1), dtype=np.int32)
    history_indptr = np.zeros(sample_count + 1, dtype=np.int64)
    history_values = []
    regime_codes = np.empty(sample_count, dtype=np.uint8)
    repeat_purchases = np.empty(sample_count, dtype=np.uint8)
    rng = np.random.default_rng(seed + 100 + SPLITS.index(split))
    trace_path = output / f"samples_{split}.csv.gz.tmp"
    arrays_path = output / f"samples_{split}.npz.tmp"
    published = []
    try:
        with gzip.open(trace_path, "wt", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
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
                if exclude_known_positives:
                    excluded.update(known_before(known, row['user_id'], int(row['timestamp'])))
                candidates[sample_index, 0] = target
                regime_code = REGIME_CODE[row["cold_start_regime"]]
                regime_codes[sample_index] = regime_code
                candidates[sample_index, 1:] = choose_negatives(
                    rng, pools[regime_code], cdfs[regime_code], excluded, negatives, validate=False
                )
                history_values.extend(history)
                history_indptr[sample_index + 1] = len(history_values)
                is_repeat = 1 if target in set(history) else 0
                repeat_purchases[sample_index] = is_repeat
                writer.writerow({"sample_index": sample_index, "is_repeat_purchase": is_repeat, **row})
                if (sample_index + 1) % 10_000 == 0:
                    print(f"  built {sample_index + 1:,}/{sample_count:,}", flush=True)

        with arrays_path.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                candidates=candidates,
                history_indices=np.asarray(history_values, dtype=np.int32),
                history_indptr=history_indptr,
                regime_codes=regime_codes,
                repeat_purchases=repeat_purchases,
            )
        for temporary, final in ((trace_path, output / f"samples_{split}.csv.gz"),
                                 (arrays_path, output / f"samples_{split}.npz")):
            temporary.replace(final)
            published.append(final)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        trace_path.unlink(missing_ok=True)
        arrays_path.unlink(missing_ok=True)
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
        "candidates_per_sample": negatives + 1,
        "exclude_known_positives": exclude_known_positives,
        "negative_sampling": f"Per-regime sampling: {negatives} negatives weighted by popularity at the end of the split; target and recorded history excluded. " + ("All positives strictly before the target are also excluded." if exclude_known_positives else "Legacy history-window exclusion only."),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-interactions', type=int, default=0,
                        help='Minimum recorded history length plus target (not lifetime user interactions).')
    parser.add_argument('--output', type=Path, default=OUTPUT,
                        help='Directory for the generated samples (must be empty or new).')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['validation'])
    parser.add_argument('--exclude-known-positives', action=argparse.BooleanOptionalAction, default=True,
                        help='Exclude positives strictly before target; disable only for legacy v2 replay.')
    args = parser.parse_args()
    if (args.data / 'demo_manifest.json').exists():
        raise ValueError('Synthetic demo counts cannot define a research benchmark.')
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ids, id_to_row, train_counts, validation_counts, test_counts = load_catalog(args.data)
    report = {
        "seed": SEED,
        "regimes": list(REGIMES),
        "samples_per_regime_cap": SAMPLES_PER_REGIME,
        "min_interactions_filter": args.min_interactions,
        "protocol_version": 3 if args.exclude_known_positives else 2,
        "exclude_known_positives": args.exclude_known_positives,
        "negative_pool": "target_regime",
        "popularity_scope": "end_of_split_retrospective",
        "splits": {},
    }
    for split in args.splits:
        report["splits"][split] = build_split(
            split,
            ids,
            id_to_row,
            train_counts,
            validation_counts,
            test_counts,
            args.min_interactions,
            output,
            data=args.data, exclude_known_positives=args.exclude_known_positives,
        )
    temporary = output / "sampling_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output / "sampling_report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: shared evaluation samples created at {output}", flush=True)


if __name__ == "__main__":
    main()
