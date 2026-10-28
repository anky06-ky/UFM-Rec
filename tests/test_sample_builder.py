"""Regression tests for the shared evaluation-sample builder.

The builder regressed silently once: commit 077818f removed the local
`eligible` array but left `len(eligible)` in the report, so every run died
with NameError *after* the sample files were already written. These tests
pin both the report contract and the negative-sampling loop.
"""
from pathlib import Path
import csv
import gzip
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_recommender_samples as builder


def _write_csv(path: Path, rows, fieldnames):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def toy_dataset(tmp_path, monkeypatch):
    """Minimal catalog + interaction history that exercises every regime."""
    data = tmp_path / "data"
    content = data / "content"
    content.mkdir(parents=True)

    # 40 items: 10 per regime. Every regime pool must stay larger than the
    # negative request once the target item is excluded.
    train_counts = [0] * 10 + [3] * 10 + [10] * 10 + [50] * 10
    regimes = (
        ["zero_shot"] * 10 + ["extreme_cold"] * 10 + ["cold"] * 10 + ["warm"] * 10
    )
    items = [
        {"row_index": index, "parent_asin": f"item{index}", "train_count": count}
        for index, count in enumerate(train_counts)
    ]
    _write_csv(content / "items.csv.gz", items, ("row_index", "parent_asin", "train_count"))
    _write_csv(
        data / "item_statistics.csv.gz",
        [{"parent_asin": f"item{i}", "validation_count": 2, "test_count": 1} for i in range(len(items))],
        ("parent_asin", "validation_count", "test_count"),
    )

    history_rows = [
        {
            "user_id": f"u{index}",
            "parent_asin": f"item{index}",
            "timestamp": "1000",
            "history": "",
            "history_length": "0",
            "train_item_count": str(train_counts[index]),
            "cold_start_regime": regimes[index],
        }
        for index in range(len(items))
    ]
    _write_csv(
        data / "validation_with_history.csv.gz",
        history_rows,
        ("user_id", "parent_asin", "timestamp", "history", "history_length",
         "train_item_count", "cold_start_regime"),
    )
    _write_csv(
        data / "test_with_history.csv.gz",
        history_rows,
        ("user_id", "parent_asin", "timestamp", "history", "history_length",
         "train_item_count", "cold_start_regime"),
    )

    monkeypatch.setattr(builder, "DATA", data)
    monkeypatch.setattr(builder, "CONTENT", content)
    output = tmp_path / "evaluation"
    output.mkdir()
    monkeypatch.setattr(builder, "OUTPUT", output)
    monkeypatch.setattr(builder, "SAMPLES_PER_REGIME", 10)
    monkeypatch.setattr(builder, "NEGATIVES", 3)
    return data


def test_build_split_report_counts_eligible_items(toy_dataset):
    """The report must expose the item pool size, not raise NameError."""
    ids, id_to_row, train_counts, validation_counts, test_counts = builder.load_catalog()

    report = builder.build_split(
        "validation",
        ids,
        id_to_row,
        train_counts,
        validation_counts,
        test_counts,
        output=toy_dataset.parent / "evaluation",
    )

    assert "eligible_candidate_items" in report
    # Every item appears in validation or test statistics, so all are eligible.
    assert report["eligible_candidate_items"] == 40
    assert report["candidates_per_sample"] == builder.NEGATIVES + 1


def test_build_split_candidates_exclude_target_and_history(toy_dataset):
    ids, id_to_row, train_counts, validation_counts, test_counts = builder.load_catalog()

    builder.build_split(
        "validation",
        ids,
        id_to_row,
        train_counts,
        validation_counts,
        test_counts,
        output=toy_dataset.parent / "evaluation",
    )

    candidates = np.load(toy_dataset.parent / "evaluation" / "samples_validation.npz")["candidates"]
    for row in candidates:
        target, negatives = row[0], row[1:]
        assert len(set(negatives.tolist())) == len(negatives)
        assert target not in negatives.tolist()


def test_choose_negatives_draws_unique_items():
    rng = np.random.default_rng(0)
    pool = np.arange(50, dtype=np.int32)
    cdf = np.cumsum(np.ones(50, dtype=np.float64))

    selected = builder.choose_negatives(rng, pool, cdf, excluded={0, 1}, count=10)

    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert 0 not in selected and 1 not in selected


def test_choose_negatives_raises_instead_of_looping_forever():
    """A pool too small to satisfy the request must fail loudly, not hang."""
    rng = np.random.default_rng(0)
    pool = np.arange(2, dtype=np.int32)
    cdf = np.cumsum(np.ones(2, dtype=np.float64))

    with pytest.raises(RuntimeError):
        builder.choose_negatives(rng, pool, cdf, excluded=set(), count=50)


def test_custom_reservoir_cap_is_respected(toy_dataset):
    first, population = builder.reservoir_sample('validation', cap=2)
    second, _ = builder.reservoir_sample('validation', cap=2)
    assert first == second
    assert len(first) == 8
    assert all(sum(row['cold_start_regime'] == regime for row in first) == 2
               for regime in builder.REGIMES)
    assert all(population[regime] == 10 for regime in builder.REGIMES)


def test_empty_filtered_split_leaves_no_artifact(toy_dataset, tmp_path):
    catalog = builder.load_catalog()
    output = tmp_path / 'empty'
    with pytest.raises(ValueError, match='No targets'):
        builder.build_split('validation', *catalog, min_interactions=2, output=output)
    assert not list(output.iterdir())


def test_impossible_negative_request_cleans_partial_output(toy_dataset, tmp_path):
    output = tmp_path / 'incomplete'
    with pytest.raises(RuntimeError, match='unique negatives'):
        builder.build_split('validation', *builder.load_catalog(), output=output, negatives=99)
    assert not list(output.iterdir())
