import csv
import gzip
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_70_15_15"

print("Đang đọc và sắp xếp tập train...")

df = pd.read_csv(
    DATA / "interactions_train.csv.gz",
    dtype={"user_id": "string", "parent_asin": "string"},
)
df = df.sort_values(
    ["user_id", "timestamp", "parent_asin"],
    kind="stable",
)

output = DATA / "train_with_history.csv.gz"
written = 0
skipped = 0

with gzip.open(output, "wt", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["user_id", "parent_asin", "rating", "timestamp", "history"]
    )

    for user_id, events in df.groupby("user_id", sort=False):
        history = deque(maxlen=50)

        # Các tương tác cùng thời điểm không được nhìn thấy nhau.
        for timestamp, batch in events.groupby("timestamp", sort=False):
            history_text = " ".join(history)

            for row in batch.itertuples(index=False):
                if history:
                    writer.writerow([
                        user_id,
                        row.parent_asin,
                        row.rating,
                        timestamp,
                        history_text,
                    ])
                    written += 1
                else:
                    skipped += 1

            history.extend(batch["parent_asin"].tolist())

assert written + skipped == len(df)

print(f"Đã ghi: {written:,} dòng")
print(f"Bỏ qua vì chưa có lịch sử: {skipped:,} dòng")
print("File kết quả:", output)