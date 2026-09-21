"""Check all history rows and compare sampled histories to original train events."""

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data/processed/toys_games_70_15_15"
manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
mapping = pd.read_csv(DATA / "item_split.csv.gz")
item_split = dict(zip(mapping["parent_asin"], mapping["split"]))
train_items = set(mapping.loc[mapping["split"].eq("train"), "parent_asin"])
samples = []
counts = {}

for split in ("train", "validation", "test"):
    total = empty = 0
    for chunk in pd.read_csv(
        DATA / f"{split}_with_history.csv.gz",
        chunksize=100_000,
        keep_default_na=False,
    ):
        assert chunk["parent_asin"].map(item_split).eq(split).all()
        assert chunk["rating"].between(1, 5).all()
        for row in chunk.itertuples(index=False):
            history = row.history.split()
            assert len(history) <= 50
            assert all(item in train_items for item in history)
            if split != "train":
                assert row.parent_asin not in history
            empty += int(not history)
        total += len(chunk)
        samples.extend(chunk.sample(n=min(10, len(chunk)), random_state=42).to_dict("records"))
    if split == "train":
        assert empty == 0
        assert 0 < total <= manifest["interaction_rows"][split]
    else:
        assert total == manifest["interaction_rows"][split]
    counts[split] = {"rows": total, "empty_history": empty}
    print(split, counts[split], flush=True)

# Independently recompute sampled histories from raw train rows by filtering
# timestamps. This checks same-user membership, ordering, and future leakage.
sample_users = {row["user_id"] for row in samples}
parts = []
for chunk in pd.read_csv(DATA / "interactions_train.csv.gz", chunksize=100_000):
    parts.append(chunk.loc[chunk["user_id"].isin(sample_users)])
events = pd.concat(parts).sort_values(["user_id", "timestamp", "parent_asin"])
lookup = {user: group for user, group in events.groupby("user_id", sort=False)}
for row in samples:
    user_events = lookup.get(row["user_id"])
    expected = [] if user_events is None else user_events.loc[
        user_events["timestamp"] < row["timestamp"], "parent_asin"
    ].tail(50).tolist()
    assert row["history"].split() == expected, f"History mismatch: {row}"

print(f"PASS: all history rows checked; {len(samples)} temporal samples matched train.")
