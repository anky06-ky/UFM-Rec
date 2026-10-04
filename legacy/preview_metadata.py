import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / (
    "data/processed/toys_games_70_15_15/metadata_train.jsonl.gz"
)

with gzip.open(path, "rt", encoding="utf-8") as file:
    for index, line in enumerate(file):
        product = json.loads(line)

        print("\nSẢN PHẨM", index + 1)
        print("ID:", product.get("parent_asin"))
        print("Tên:", product.get("title"))
        print("Đặc điểm:", product.get("features"))
        print("Mô tả:", product.get("description"))
        print("Ảnh MAIN:", product.get("images"))

        if index == 2:
            break
