"""Audit real text/image coverage and index URLs in the EXISTING item order.

No image downloads, model fitting, review reading or test evaluation. The output
contains train-only counts, not metadata rating_number or future interaction counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"
METADATA = ROOT / "data/raw/toys_games_5core/meta_Toys_and_Games.jsonl.gz"


def allowed_image_url(url):
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname == "m.media-amazon.com"
                and not parsed.username and not parsed.password and parsed.port in (None, 443)
                and parsed.path.startswith("/images/I/"))
    except ValueError:
        return False


def primary_image(record):
    images = record.get("images") or []
    if not isinstance(images, list):
        return ""
    images = [x for x in images if isinstance(x, dict)]
    ordered = sorted(images, key=lambda x: x.get("variant") != "MAIN")
    # Prefer medium resolution to bound bandwidth. Higher resolution is unnecessary
    # for the initial 224px CLIP encoder. Never silently use review images.
    for image in ordered:
        for field in ("large", "hi_res", "thumb"):
            url = image.get(field)
            if allowed_image_url(url):
                return url
    return ""


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def build_catalog(data: Path, metadata: Path, output: Path):
    paths = {"items": data / "content/items.csv.gz",
             "text": data / "content/products_text.csv.gz",
             "split": data / "split_manifest.json", "metadata": metadata}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a NEW output directory; existing data is never overwritten.")
    wanted = {}
    with gzip.open(paths["items"], "rt", encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            if int(row["row_index"]) != index or row["parent_asin"] in wanted or int(row["train_count"]) < 0:
                raise ValueError("Non-contiguous/duplicate item mapping or invalid train count.")
            wanted[row["parent_asin"]] = None
    if not wanted:
        raise ValueError("Empty item mapping.")
    stats = Counter()
    print(f"Reading metadata for {len(wanted):,} mapped products...", flush=True)
    with gzip.open(metadata, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            asin = record.get("parent_asin")
            stats["metadata_rows_scanned"] += 1
            if asin in wanted:
                if wanted[asin] is None:
                    wanted[asin] = primary_image(record)
                    stats["matched_metadata"] += 1
                else:
                    stats["duplicate_metadata_ignored"] += 1
            if stats["metadata_rows_scanned"] % 200_000 == 0:
                print(f"metadata={stats['metadata_rows_scanned']:,} matched={stats['matched_metadata']:,}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "catalog.csv.gz.tmp"
    regimes = {}
    with gzip.open(paths["items"], "rt", encoding="utf-8", newline="") as item_file, \
         gzip.open(paths["text"], "rt", encoding="utf-8", newline="") as text_file, \
         gzip.open(temporary, "wt", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["row_index", "parent_asin", "train_count",
                                                   "cold_start_regime", "has_text", "image_url"])
        writer.writeheader()
        for item, text in zip_longest(csv.DictReader(item_file), csv.DictReader(text_file)):
            if item is None or text is None or item["parent_asin"] != text["parent_asin"]:
                raise ValueError("Text/mapping row alignment mismatch.")
            asin = item["parent_asin"]
            has_text = bool(text["text"].strip())
            image_url = wanted[asin] or ""
            group = regimes.setdefault(item["cold_start_regime"], Counter())
            flags = {"items": 1, "has_text": int(has_text), "has_image_url": int(bool(image_url)),
                     "both": int(has_text and bool(image_url)),
                     "neither": int(not has_text and not image_url),
                     "missing_metadata": int(wanted[asin] is None)}
            group.update(flags)
            stats.update(flags)
            writer.writerow({"row_index": item["row_index"], "parent_asin": asin,
                             "train_count": item["train_count"], "cold_start_regime": item["cold_start_regime"],
                             "has_text": int(has_text), "image_url": image_url})
    if stats["items"] != len(wanted):
        raise ValueError("Output item count mismatch.")
    report = {"version": 1, "counts": dict(stats), "regimes": {k: dict(v) for k, v in regimes.items()},
              "sources_sha256": {k: sha256(v) for k, v in paths.items()},
              "catalog_sha256": sha256(temporary),
              "policy": "Existing row_index; one MAIN-preferred product image URL; no review images; train_count only; no learning or test evaluation.",
              "limitations": ["Image URL availability is NOT successful download/decode coverage.",
                              "Snapshot metadata may include later edits; no historical product-launch/image timestamp."]}
    temporary.replace(output / "catalog.csv.gz")
    marker = output / "catalog_report.json.tmp"
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.replace(output / "catalog_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output", type=Path, default=DATA / "foundation_catalog_v1")
    args = parser.parse_args()
    build_catalog(args.data, args.metadata, args.output)


if __name__ == "__main__":
    main()
