from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/processed/toys_games_70_15_15/interactions_train.csv.gz"

# Lấy người dùng đầu tiên để minh họa.
user_id = pd.read_csv(path, nrows=1).iloc[0]["user_id"]

# Đọc từng phần để tiết kiệm RAM.
parts = []
for chunk in pd.read_csv(path, chunksize=200_000):
    parts.append(chunk.loc[chunk["user_id"] == user_id])

events = pd.concat(parts).sort_values("timestamp")

print("Người dùng:", user_id)

for row in events.itertuples(index=False):
    # Chỉ lấy tương tác xảy ra trước thời điểm đang dự đoán.
    history = events.loc[
        events["timestamp"] < row.timestamp, "parent_asin"
    ].tolist()

    print("\nSản phẩm mục tiêu:", row.parent_asin)
    print("Lịch sử trước đó:", history)