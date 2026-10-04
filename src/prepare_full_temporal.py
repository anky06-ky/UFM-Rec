"""Prepare the full Amazon Toys & Games data for temporal recommendation.

The pipeline is intentionally disk-backed: users are partitioned into stable
buckets, then each bucket is sorted independently. This keeps peak memory
bounded on machines that cannot hold all interactions in RAM.
"""

from __future__ import annotations

import csv
import gzip
import json
import zlib
from collections import Counter, deque
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/toys_games_full/Toys_and_Games.jsonl.gz"
METADATA = ROOT / "data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz"
OUTPUT = ROOT / "data/processed/toys_games_full_temporal"
WORK = ROOT / "tmp/prepare_toys_games_full_temporal"

BUCKET_COUNT = 128
DAY_MS = 86_400_000
MAX_HISTORY = 50
POSITIVE_RATING = 4.0
FIELDS = ("user_id", "parent_asin", "rating", "timestamp")
SPLITS = ("train", "validation", "test")


def split_name(timestamp: int, validation_start: int, test_start: int) -> str:
    if timestamp < validation_start:
        return "train"
    if timestamp < test_start:
        return "validation"
    return "test"


def regime(train_count: int) -> str:
    if train_count == 0:
        return "zero_shot"
    if train_count <= 5:
        return "extreme_cold"
    if train_count <= 20:
        return "cold"
    return "warm"


def day_boundary(day_counts: Counter, fraction: float) -> int:
    target = sum(day_counts.values()) * fraction
    cumulative = 0
    for day, count in sorted(day_counts.items()):
        cumulative += count
        if cumulative >= target:
            return (day + 1) * DAY_MS
    raise ValueError("Cannot choose a time boundary from empty data")


def history_rows(rows, train_counts, validation_start, test_start):
    for user_id, user_events in groupby(rows, key=lambda row: row["user_id"]):
        history = deque(maxlen=MAX_HISTORY)
        for timestamp_text, simultaneous in groupby(
            user_events, key=lambda row: row["timestamp"]
        ):
            batch = list(simultaneous)
            timestamp = int(timestamp_text)
            history_text = " ".join(history)
            for row in batch:
                item = row["parent_asin"]
                count = train_counts.get(item, 0)
                yield split_name(timestamp, validation_start, test_start), {
                    **row,
                    "history": history_text,
                    "history_length": len(history),
                    "train_item_count": count,
                    "cold_start_regime": regime(count),
                }
            history.extend(row["parent_asin"] for row in batch)


def self_check() -> None:
    assert [regime(value) for value in (0, 1, 5, 6, 20, 21)] == [
        "zero_shot",
        "extreme_cold",
        "extreme_cold",
        "cold",
        "cold",
        "warm",
    ]
    boundaries = Counter({1: 8, 2: 1, 3: 1})
    assert day_boundary(boundaries, 0.8) == 2 * DAY_MS
    rows = [
        {"user_id": "u", "parent_asin": "a", "rating": "5", "timestamp": "1"},
        {"user_id": "u", "parent_asin": "b", "rating": "5", "timestamp": "2"},
        {"user_id": "u", "parent_asin": "c", "rating": "4", "timestamp": "2"},
        {"user_id": "u", "parent_asin": "d", "rating": "5", "timestamp": "3"},
    ]
    built = list(history_rows(rows, Counter({"a": 1}), 3, 4))
    assert built[0][1]["history"] == ""
    assert built[1][1]["history"] == built[2][1]["history"] == "a"
    assert built[3][1]["history"] == "a b c"


def require_clean_targets() -> None:
    missing = [path for path in (RAW, METADATA) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(map(str, missing)))
    existing = [OUTPUT / name for name in ("split_manifest.json", "interactions_train.csv.gz")]
    if any(path.exists() for path in existing):
        raise FileExistsError(f"Refusing to overwrite completed output: {OUTPUT}")
    if WORK.exists() and any(WORK.iterdir()):
        raise FileExistsError(f"Working directory is not empty: {WORK}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)


def partition_raw() -> dict:
    print("Stage 1/4: filtering and partitioning raw interactions...", flush=True)
    paths = [WORK / f"bucket_{index:03d}.csv" for index in range(BUCKET_COUNT)]
    handles = [path.open("w", encoding="utf-8", newline="") for path in paths]
    writers = [csv.DictWriter(handle, fieldnames=FIELDS) for handle in handles]
    for writer in writers:
        writer.writeheader()

    stats = {
        "raw_records": 0,
        "positive_verified_records": 0,
        "unverified_records": 0,
        "below_positive_rating_records": 0,
        "invalid_records": 0,
        "rating_counts": Counter(),
    }
    try:
        with gzip.open(RAW, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stats["raw_records"] += 1
                try:
                    record = json.loads(line)
                    user_id = record["user_id"]
                    item = record["parent_asin"]
                    rating = float(record["rating"])
                    timestamp = int(record["timestamp"])
                    verified = record["verified_purchase"] is True
                    if not isinstance(user_id, str) or not user_id:
                        raise ValueError("invalid user_id")
                    if not isinstance(item, str) or not item:
                        raise ValueError("invalid parent_asin")
                    if not 1 <= rating <= 5 or timestamp <= 0:
                        raise ValueError("invalid rating/timestamp")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    stats["invalid_records"] += 1
                    continue

                stats["rating_counts"][str(rating)] += 1
                if not verified:
                    stats["unverified_records"] += 1
                    continue
                if rating < POSITIVE_RATING:
                    stats["below_positive_rating_records"] += 1
                    continue

                bucket = zlib.crc32(user_id.encode("utf-8")) % BUCKET_COUNT
                writers[bucket].writerow(
                    {
                        "user_id": user_id,
                        "parent_asin": item,
                        "rating": rating,
                        "timestamp": timestamp,
                    }
                )
                stats["positive_verified_records"] += 1
                if line_number % 1_000_000 == 0:
                    print(
                        f"  raw {line_number:,}; kept {stats['positive_verified_records']:,}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.close()

    stats["rating_counts"] = dict(sorted(stats["rating_counts"].items()))
    return stats


def normalize_buckets() -> tuple[Counter, int, int, int]:
    print("Stage 2/4: sorting buckets, removing exact duplicates...", flush=True)
    day_counts = Counter()
    duplicate_records = 0
    unique_records = 0
    unique_users = 0

    for index in range(BUCKET_COUNT):
        raw_path = WORK / f"bucket_{index:03d}.csv"
        normalized_path = WORK / f"bucket_{index:03d}.csv.gz"
        records = set()
        raw_count = 0
        with raw_path.open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                records.add(
                    (
                        row["user_id"],
                        row["parent_asin"],
                        row["rating"],
                        int(row["timestamp"]),
                    )
                )
                raw_count += 1
        duplicate_records += raw_count - len(records)
        ordered = sorted(records, key=lambda row: (row[0], row[3], row[1]))

        with gzip.open(normalized_path, "wt", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(FIELDS)
            writer.writerows(ordered)

        previous_user = None
        for user_id, _, _, timestamp in ordered:
            day_counts[timestamp // DAY_MS] += 1
            unique_records += 1
            if user_id != previous_user:
                unique_users += 1
                previous_user = user_id
        raw_path.unlink()
        if (index + 1) % 16 == 0 or index + 1 == BUCKET_COUNT:
            print(
                f"  buckets {index + 1}/{BUCKET_COUNT}; unique rows {unique_records:,}",
                flush=True,
            )
    return day_counts, duplicate_records, unique_records, unique_users


def write_splits(validation_start: int, test_start: int):
    print("Stage 3/4: writing temporal splits and item statistics...", flush=True)
    paths = {split: OUTPUT / f"interactions_{split}.csv.gz.tmp" for split in SPLITS}
    handles = {
        split: gzip.open(path, "wt", encoding="utf-8", newline="")
        for split, path in paths.items()
    }
    writers = {split: csv.DictWriter(handles[split], fieldnames=FIELDS) for split in SPLITS}
    for writer in writers.values():
        writer.writeheader()

    split_counts = Counter()
    item_counts = {split: Counter() for split in SPLITS}
    try:
        for index in range(BUCKET_COUNT):
            path = WORK / f"bucket_{index:03d}.csv.gz"
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    split = split_name(int(row["timestamp"]), validation_start, test_start)
                    writers[split].writerow(row)
                    split_counts[split] += 1
                    item_counts[split][row["parent_asin"]] += 1
            if (index + 1) % 16 == 0 or index + 1 == BUCKET_COUNT:
                print(f"  buckets {index + 1}/{BUCKET_COUNT}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    item_path = OUTPUT / "item_statistics.csv.gz.tmp"
    all_items = set().union(*(counts.keys() for counts in item_counts.values()))
    with gzip.open(item_path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "parent_asin",
                "train_count",
                "validation_count",
                "test_count",
                "cold_start_regime",
            ]
        )
        for item in sorted(all_items):
            train_count = item_counts["train"][item]
            writer.writerow(
                [
                    item,
                    train_count,
                    item_counts["validation"][item],
                    item_counts["test"][item],
                    regime(train_count),
                ]
            )
    return split_counts, item_counts


def write_histories(validation_start: int, test_start: int, train_counts: Counter):
    print("Stage 4/4: building leakage-safe user histories...", flush=True)
    fields = [*FIELDS, "history", "history_length", "train_item_count", "cold_start_regime"]
    paths = {split: OUTPUT / f"{split}_with_history.csv.gz.tmp" for split in SPLITS}
    handles = {
        split: gzip.open(path, "wt", encoding="utf-8", newline="")
        for split, path in paths.items()
    }
    writers = {split: csv.DictWriter(handles[split], fieldnames=fields) for split in SPLITS}
    for writer in writers.values():
        writer.writeheader()

    written = Counter()
    empty = Counter()
    regime_counts = {split: Counter() for split in SPLITS}
    user_counts = Counter()
    try:
        for index in range(BUCKET_COUNT):
            path = WORK / f"bucket_{index:03d}.csv.gz"
            users_by_split = {split: set() for split in SPLITS}
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                rows = csv.DictReader(source)
                for split, row in history_rows(rows, train_counts, validation_start, test_start):
                    users_by_split[split].add(row["user_id"])
                    if split == "train" and row["history_length"] == 0:
                        continue
                    writers[split].writerow(row)
                    written[split] += 1
                    empty[split] += int(row["history_length"] == 0)
                    regime_counts[split][row["cold_start_regime"]] += 1
            for split in SPLITS:
                user_counts[split] += len(users_by_split[split])
            if (index + 1) % 16 == 0 or index + 1 == BUCKET_COUNT:
                print(f"  buckets {index + 1}/{BUCKET_COUNT}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return written, empty, regime_counts, user_counts


def iso_time(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def promote_outputs() -> None:
    for temporary in OUTPUT.glob("*.tmp"):
        temporary.replace(temporary.with_suffix(""))


def main() -> None:
    self_check()
    require_clean_targets()
    raw_stats = partition_raw()
    day_counts, duplicate_records, unique_records, unique_users = normalize_buckets()
    validation_start = day_boundary(day_counts, 0.8)
    test_start = day_boundary(day_counts, 0.9)
    if validation_start >= test_start:
        raise AssertionError("Temporal boundaries are not increasing")
    print(f"Validation starts: {iso_time(validation_start)}", flush=True)
    print(f"Test starts:       {iso_time(test_start)}", flush=True)

    split_counts, item_counts = write_splits(validation_start, test_start)
    if sum(split_counts.values()) != unique_records:
        raise AssertionError("Temporal split lost interactions")
    written, empty, regime_counts, user_counts = write_histories(
        validation_start, test_start, item_counts["train"]
    )
    if written["validation"] != split_counts["validation"]:
        raise AssertionError("Validation history row count mismatch")
    if written["test"] != split_counts["test"]:
        raise AssertionError("Test history row count mismatch")

    manifest = {
        "source": RAW.relative_to(ROOT).as_posix(),
        "metadata": METADATA.relative_to(ROOT).as_posix(),
        "policy": {
            "positive_interaction": "verified_purchase is true and rating >= 4",
            "duplicates": "Remove exact duplicates of user, parent item, rating and timestamp.",
            "split": "Global chronological 80/10/10 split at UTC day boundaries.",
            "history": "Prior positive interactions only; same-timestamp events cannot see each other; keep latest 50.",
            "test_history": "Online-style history includes earlier train/validation events.",
            "cold_start_regimes": {
                "zero_shot": "0 train interactions",
                "extreme_cold": "1-5 train interactions",
                "cold": "6-20 train interactions",
                "warm": ">20 train interactions",
            },
        },
        "raw": raw_stats,
        "exact_duplicate_records": duplicate_records,
        "retained_unique_records": unique_records,
        "retained_unique_users": unique_users,
        "validation_start_utc": iso_time(validation_start),
        "test_start_utc": iso_time(test_start),
        "interactions": dict(split_counts),
        "interaction_ratios": {
            split: split_counts[split] / unique_records for split in SPLITS
        },
        "unique_items": {
            split: len(item_counts[split]) for split in SPLITS
        },
        "history_rows": dict(written),
        "empty_histories": dict(empty),
        "users_by_split": dict(user_counts),
        "regime_rows": {
            split: dict(regime_counts[split]) for split in SPLITS
        },
    }
    manifest_path = OUTPUT / "split_manifest.json.tmp"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    promote_outputs()

    for path in WORK.glob("bucket_*.csv.gz"):
        path.unlink()
    try:
        WORK.rmdir()
    except OSError:
        pass

    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"PASS: temporal data prepared at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
