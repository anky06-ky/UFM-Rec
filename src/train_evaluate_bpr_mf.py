"""Train-only implicit BPR matrix factorization; select on fixed validation samples.

This ID-only baseline cannot represent unseen items. Test evaluation is deliberately
separate from this command and is not performed while UFM selection is ongoing.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.special import expit

from common import sha256, write_json
from evaluate_content_baseline import REGIMES, rank_of_target, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"


def user_key(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little", signed=True)


def save_factors(path, users, items, epoch, metric):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez(output, users=users, items=items, epoch=epoch, metric=metric)
    temporary.replace(path)


def build_graph(data, max_users=0):
    ids, counts = {}, []
    with gzip.open(data / "content/items.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["row_index"]) != len(ids) or row["parent_asin"] in ids:
                raise ValueError("Duplicate or non-contiguous item mapping.")
            ids[row["parent_asin"]] = len(ids)
            counts.append(int(row["train_count"]))
    counts = np.asarray(counts, dtype=np.int32)
    if not len(counts) or np.any(counts < 0):
        raise ValueError("Invalid train counts.")
    indices, indptr, keys = array("i"), array("q", [0]), array("q")
    observed = np.zeros(len(counts), dtype=np.int32)
    source_rows = skipped = duplicates = 0
    with gzip.open(data / "interactions_train.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for name, group in groupby(csv.DictReader(source), key=lambda row: row["user_id"]):
            rows = list(group)
            source_rows += len(rows)
            mapped = [ids[row["parent_asin"]] for row in rows]
            for item in mapped:
                observed[item] += 1
            positive = sorted(set(mapped))
            duplicates += len(rows) - len(positive)
            if len(positive) < 2:
                skipped += 1
                continue
            keys.append(user_key(name))
            indices.extend(positive)
            indptr.append(len(indices))
            if max_users and len(keys) >= max_users:
                break
    keys = np.frombuffer(keys, dtype=np.int64).copy()
    if not len(keys):
        raise ValueError("No users with at least two train positives.")
    graph = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.uint8), np.frombuffer(indices, dtype=np.int32),
         np.frombuffer(indptr, dtype=np.int64)), shape=(len(keys), len(counts)))
    order = np.argsort(keys)
    keys = keys[order]
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError("Non-contiguous user rows or user hash collision.")
    graph = graph[order].tocsr()
    if not max_users:
        manifest = json.loads((data / "split_manifest.json").read_text(encoding="utf-8"))
        if source_rows != manifest["interactions"]["train"] or not np.array_equal(observed, counts):
            raise ValueError("Train graph/source count mismatch.")
    return graph, keys, counts, dict(source_rows=source_rows, users=len(keys),
                                      pairs=graph.nnz, skipped_singleton_users=skipped,
                                      duplicate_interactions=duplicates,
                                      truncated=bool(max_users))


def sample_negatives(graph, users, eligible, rng):
    negative = rng.choice(eligible, size=len(users)).astype(np.int32)
    for _ in range(100):
        collision = graph[users, negative].A1.astype(bool)
        if not collision.any():
            return negative
        negative[collision] = rng.choice(eligible, size=int(collision.sum()))
    raise ValueError("Could not draw train-unobserved negatives for this batch.")


def update_batch(user_factors, item_factors, users, positive, negative, lr, regularization):
    p = user_factors[users].copy()
    q_pos = item_factors[positive].copy()
    q_neg = item_factors[negative].copy()
    margin = np.einsum("bd,bd->b", p, q_pos - q_neg)
    weight = expit(-margin)[:, None]
    np.add.at(user_factors, users, lr * (weight * (q_pos - q_neg) - regularization * p))
    np.add.at(item_factors, positive, lr * (weight * p - regularization * q_pos))
    np.add.at(item_factors, negative, lr * (-weight * p - regularization * q_neg))
    return float(np.logaddexp(0, -margin).mean())


def validation_user_rows(data, keys, samples):
    path = data / "evaluation/samples_validation.csv.gz"
    hashes = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            if int(row["sample_index"]) != index:
                raise ValueError("Validation trace order mismatch.")
            hashes.append(user_key(row["user_id"]))
    if len(hashes) != len(samples["candidates"]):
        raise ValueError("Validation trace/sample count mismatch.")
    hashes = np.asarray(hashes, dtype=np.int64)
    positions = np.searchsorted(keys, hashes)
    known = positions < len(keys)
    known[known] &= keys[positions[known]] == hashes[known]
    return positions, known


def evaluate(samples, user_rows, known, user_factors, item_factors, counts):
    candidates = samples["candidates"]
    lengths = np.diff(samples["history_indptr"])
    ranks = np.empty(len(candidates), dtype=np.uint8)
    for start in range(0, len(candidates), 256):
        end = min(start + 256, len(candidates))
        candidate = candidates[start:end]
        popularity = np.log1p(counts[candidate]).astype(np.float32)
        active = known[start:end] & (lengths[start:end] > 0)
        scores = popularity.copy()
        if active.any():
            ids = np.flatnonzero(active)
            values = np.einsum("bd,bkd->bk", user_factors[user_rows[start:end][ids]],
                               item_factors[candidate[ids]])
            from common import push_unseen_last
            seen = counts[candidate[ids]] > 0
            values = push_unseen_last(values, seen)
            scale = np.maximum(popularity[ids].max(axis=1), 1)
            scores[ids] = values + popularity[ids] / scale[:, None] * 1e-7
        for offset, row in enumerate(candidate):
            ranks[start + offset] = rank_of_target(scores[offset], row)
    report = summarize(ranks, samples["regime_codes"], lengths, samples.get("repeat_purchases"))
    report["cold_macro_ndcg@10"] = float(np.mean(
        [report["by_regime"][name]["ndcg@10"] for name in REGIMES[:3]]))
    return report


def train(args):
    data, output = args.data, args.output
    sources = {name: sha256(data / name) for name in (
        "split_manifest.json", "content/items.csv.gz", "interactions_train.csv.gz",
        "evaluation/samples_validation.npz", "evaluation/samples_validation.csv.gz")}
    config = dict(version=2, sources=sources, code_sha256=sha256(Path(__file__)),
                  epochs=args.epochs, factors=args.factors, batch_size=args.batch_size, lr=args.lr,
                  regularization=args.regularization, seed=args.seed,
                  max_users=args.max_users, selection_metric=args.selection_metric,
                  selection=f"Validation {args.selection_metric} NDCG@10; no test.")
    if args.resume:
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume code/source/hyperparameter mismatch.")
        graph = sparse.load_npz(output / "graph.npz")
        keys, counts = np.load(output / "user_keys.npy"), np.load(output / "train_counts.npy")
        with np.load(output / "latest.npz") as state:
            user_factors, item_factors = state["users"].copy(), state["items"].copy()
            start_epoch = int(state["epoch"])
        best = -1.0
        if (output / "best.npz").exists():
            with np.load(output / "best.npz") as state:
                best = float(state["metric"])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a new output or use --resume for this exact run.")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
        graph, keys, counts, graph_report = build_graph(data, args.max_users)
        sparse.save_npz(output / "graph.npz", graph)
        np.save(output / "user_keys.npy", keys)
        np.save(output / "train_counts.npy", counts)
        write_json(output / "graph_report.json", graph_report)
        rng = np.random.default_rng(args.seed)
        user_factors = rng.normal(0, 0.05, (len(keys), args.factors)).astype(np.float32)
        item_factors = rng.normal(0, 0.05, (len(counts), args.factors)).astype(np.float32)
        start_epoch, best = 0, -1.0
        save_factors(output / "latest.npz", user_factors, item_factors, 0, best)
    if (output / "completed.json").exists():
        raise FileExistsError("Run already completed; no overwrite.")
    if graph.shape != (len(keys), len(counts)) or user_factors.shape != (len(keys), args.factors) or item_factors.shape != (len(counts), args.factors):
        raise ValueError("Graph/factor shapes mismatch.")
    eligible = np.flatnonzero(counts > 0).astype(np.int32)
    if not len(eligible):
        raise ValueError("No train-positive candidate items.")
    with np.load(data / "evaluation/samples_validation.npz") as source:
        samples = {name: source[name] for name in source.files}
    user_rows, known = validation_user_rows(data, keys, samples)
    pair_users = np.repeat(np.arange(len(keys), dtype=np.int32), np.diff(graph.indptr))
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed + 1000 + epoch)
        order = rng.permutation(graph.nnz)
        losses = []
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            users = pair_users[selected]
            positive = graph.indices[selected]
            negative = sample_negatives(graph, users, eligible, rng)
            losses.append(update_batch(user_factors, item_factors, users, positive, negative,
                                       args.lr, args.regularization))
        if not np.isfinite(user_factors).all() or not np.isfinite(item_factors).all():
            raise FloatingPointError("Non-finite BPR factors.")
        report = evaluate(samples, user_rows, known, user_factors, item_factors, counts)
        if args.selection_metric == "overall":
            metric = report["overall"]["ndcg@10"]
        else:
            metric = report["cold_macro_ndcg@10"]
        if metric > best + 1e-6:
            best = metric
            save_factors(output / "best.npz", user_factors, item_factors, epoch + 1, best)
        save_factors(output / "latest.npz", user_factors, item_factors, epoch + 1, best)
        record = dict(epoch=epoch + 1, bpr_loss=float(np.mean(losses)), validation=report,
                      selection_metric=args.selection_metric,
                      best_selection_value=best, elapsed_seconds=time.monotonic() - started)
        write_json(output / f"epoch_{epoch+1:03d}.json", record)
        print(json.dumps(record), flush=True)
    write_json(output / "completed.json", dict(epochs=args.epochs,
               selection_metric=args.selection_metric, best_selection_value=best,
               smoke_run=bool(args.max_users), test_evaluated=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DATA / "models/bpr_mf")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0, help="Bounded smoke only; results not comparable.")
    parser.add_argument("--selection-metric", choices=("overall", "cold_macro"), default="overall",
                        help="ID-only models should select on overall; content-aware on cold_macro.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.factors, args.batch_size) < 1 or args.max_users < 0 or args.lr <= 0 or args.regularization < 0:
        parser.error("Positive sizes/rate and non-negative regularization/max-users required.")
    train(args)


if __name__ == "__main__":
    main()
