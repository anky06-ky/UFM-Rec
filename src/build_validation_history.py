import csv
import gzip
from bisect import bisect_left
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_70_15_15"

print("Đang đọc lịch sử từ train...", flush=True)

# Dùng train gốc để giữ cả tương tác đầu tiên của mỗi người.
train = pd.read_csv(
    DATA / "interactions_train.csv.gz",
    dtype={"user_id": "string", "parent_asin": "string"},
)
train = train.sort_values(["user_id", "timestamp", "parent_asin"])

lookup = {}
for user_id, events in train.groupby("user_id", sort=False):
    lookup[user_id] = (
        events["timestamp"].tolist(),
        events["parent_asin"].tolist(),
    )

del train

validation = pd.read_csv(
    DATA / "interactions_validation.csv.gz",
    dtype={"user_id": "string", "parent_asin": "string"},
)

output = DATA / "validation_with_history.csv.gz"
written = 0
empty = 0

print("Đang tạo lịch sử validation...", flush=True)

with gzip.open(output, "wt", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["user_id", "parent_asin", "rating", "timestamp", "history"]
    )

    for row in validation.itertuples(index=False):
        times, items = lookup.get(row.user_id, ([], []))

        # Loại cả tương tác cùng thời điểm và tương tác tương lai.
        end = bisect_left(times, row.timestamp)
        history = items[max(0, end - 50):end]

        # Sản phẩm mục tiêu không được xuất hiện trong lịch sử.
        assert row.parent_asin not in history

        writer.writerow([
            row.user_id,
            row.parent_asin,
            row.rating,
            row.timestamp,
            " ".join(history),
        ])

        written += 1
        empty += int(not history)

assert written == len(validation)

print(f"Đã ghi: {written:,} dòng")
print(f"Trong đó lịch sử rỗng: {empty:,} dòng")
print("File kết quả:", output)