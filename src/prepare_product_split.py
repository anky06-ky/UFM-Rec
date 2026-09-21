from __future__ import annotations

import gzip
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "toys_games_5core"
OUTPUT_DIR = ROOT / "data" / "processed" / "toys_games_70_15_15"

SOURCE_FILES = [
    RAW_DIR / "Toys_and_Games.train.csv.gz",
    RAW_DIR / "Toys_and_Games.valid.csv.gz",
    RAW_DIR / "Toys_and_Games.test.csv.gz",
]
METADATA_FILE = RAW_DIR / "meta_Toys_and_Games.jsonl.gz"

SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 42
CHUNK_SIZE = 200_000

INTERACTION_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
INTERACTION_DTYPES = {
    "user_id": "string",
    "parent_asin": "string",
    "rating": "float32",
    "timestamp": "int64",
}

# Keep only fields that are available without aggregating evaluation interactions.
SAFE_METADATA_FIELDS = (
    "main_category",
    "title",
    "features",
    "description",
    "price",
    "store",
    "categories",
    "details",
    "parent_asin",
)

IMAGE_ID_PATTERN = re.compile(r"/images/I/([^./?]+)")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def require_sources() -> None:
    missing = [path for path in [*SOURCE_FILES, METADATA_FILE] if not path.is_file()]
    if missing:
        names = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing source files:\n{names}")


def main_image(record: dict) -> dict:
    images = record.get("images") or []
    if isinstance(images, dict):
        images = [images]
    if not isinstance(images, list):
        return {}
    valid_images = [image for image in images if isinstance(image, dict)]
    if not valid_images:
        return {}
    return next(
        (image for image in valid_images if str(image.get("variant", "")).upper() == "MAIN"),
        valid_images[0],
    )


def image_ids(image: dict):
    for field in ("thumb", "large", "hi_res"):
        values = image.get(field)
        if not isinstance(values, list):
            values = [values]
        for url in values:
            if not isinstance(url, str) or not url:
                continue
            match = IMAGE_ID_PATTERN.search(url)
            yield match.group(1) if match else url.split("?", 1)[0]


def scan_interactions() -> tuple[Counter, int, int, int]:
    item_counts: Counter = Counter()
    users: set[str] = set()
    event_hashes: set[int] = set()
    total_rows = 0
    duplicate_rows = 0

    for path in SOURCE_FILES:
        print(f"Scanning interactions: {path.name}")
        for chunk in pd.read_csv(
            path,
            usecols=INTERACTION_COLUMNS,
            dtype=INTERACTION_DTYPES,
            chunksize=CHUNK_SIZE,
        ):
            if chunk[["user_id", "parent_asin"]].isna().any().any():
                raise ValueError(f"Null user/item ID found in {path.name}")
            if not chunk["rating"].between(1, 5).all():
                raise ValueError(f"Rating outside 1..5 found in {path.name}")

            hashes = set(
                int(value)
                for value in pd.util.hash_pandas_object(
                    chunk[INTERACTION_COLUMNS], index=False
                ).to_numpy()
            )
            duplicate_rows += len(chunk) - len(hashes)
            duplicate_rows += len(hashes & event_hashes)
            event_hashes.update(hashes)

            item_counts.update(chunk["parent_asin"].value_counts().to_dict())
            users.update(chunk["user_id"].dropna().astype(str).unique())
            total_rows += len(chunk)

    return item_counts, total_rows, len(users), duplicate_rows


def connect_shared_main_images(
    item_index: dict[str, int], union_find: UnionFind
) -> tuple[set[int], int, int]:
    image_owner: dict[str, int] = {}
    metadata_items: set[int] = set()
    shared_links = 0

    print("Scanning metadata and grouping products that share a MAIN image...")
    with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in metadata line {line_number}: {error}"
                ) from error

            item = record.get("parent_asin")
            index = item_index.get(item)
            if index is None:
                continue

            metadata_items.add(index)
            for image_id in set(image_ids(main_image(record))):
                owner = image_owner.get(image_id)
                if owner is None:
                    image_owner[image_id] = index
                elif owner != index:
                    union_find.union(owner, index)
                    shared_links += 1

    return metadata_items, len(image_owner), shared_links


def assign_components(
    items: list[str], item_counts: Counter, union_find: UnionFind
) -> tuple[dict[str, str], dict[str, int], int, int]:
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        components[union_find.find(index)].append(index)

    weighted_components = [
        (indices, sum(item_counts[items[index]] for index in indices))
        for indices in components.values()
    ]
    random.Random(SEED).shuffle(weighted_components)
    weighted_components.sort(key=lambda component: component[1], reverse=True)

    total_rows = sum(item_counts.values())
    targets = {split: total_rows * TARGET_RATIOS[split] for split in SPLITS}
    assigned_rows = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}

    for indices, weight in weighted_components:
        split = min(
            SPLITS,
            key=lambda name: assigned_rows[name] / targets[name],
        )
        assigned_rows[split] += weight
        for index in indices:
            assignment[items[index]] = split

    largest_component = max(len(indices) for indices, _ in weighted_components)
    return assignment, assigned_rows, len(weighted_components), largest_component


def write_interactions(assignment: dict[str, str]) -> tuple[dict[str, Path], Counter]:
    paths = {
        split: OUTPUT_DIR / f"interactions_{split}.csv.gz" for split in SPLITS
    }
    temporary = {split: path.with_suffix(path.suffix + ".tmp") for split, path in paths.items()}
    row_counts: Counter = Counter()
    wrote_header = {split: False for split in SPLITS}

    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(
                gzip.open(path, "wt", encoding="utf-8", newline="")
            )
            for split, path in temporary.items()
        }
        for source_path in SOURCE_FILES:
            print(f"Writing split interactions from: {source_path.name}")
            for chunk in pd.read_csv(
                source_path,
                usecols=INTERACTION_COLUMNS,
                dtype=INTERACTION_DTYPES,
                chunksize=CHUNK_SIZE,
            ):
                assigned = chunk["parent_asin"].map(assignment)
                if assigned.isna().any():
                    raise ValueError("An interaction product has no split assignment")

                for split in SPLITS:
                    output = chunk.loc[assigned.eq(split), INTERACTION_COLUMNS]
                    if output.empty:
                        continue
                    output.to_csv(
                        writers[split],
                        index=False,
                        header=not wrote_header[split],
                    )
                    wrote_header[split] = True
                    row_counts[split] += len(output)

    return {split: temporary[split] for split in SPLITS}, row_counts


def write_metadata(
    assignment: dict[str, str]
) -> tuple[dict[str, Path], Counter, set[str], int]:
    paths = {split: OUTPUT_DIR / f"metadata_{split}.jsonl.gz" for split in SPLITS}
    temporary = {split: path.with_suffix(path.suffix + ".tmp") for split, path in paths.items()}
    metadata_counts: Counter = Counter()
    metadata_items: set[str] = set()
    image_split: dict[str, str] = {}
    duplicate_metadata = 0

    print("Writing leakage-safe metadata splits...")
    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(
                gzip.open(path, "wt", encoding="utf-8", newline="")
            )
            for split, path in temporary.items()
        }
        with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                item = record.get("parent_asin")
                split = assignment.get(item)
                if split is None:
                    continue
                if item in metadata_items:
                    duplicate_metadata += 1
                    continue

                selected_image = main_image(record)
                for image_id in set(image_ids(selected_image)):
                    previous_split = image_split.setdefault(image_id, split)
                    if previous_split != split:
                        raise AssertionError(
                            f"Image {image_id} appears in {previous_split} and {split}"
                        )

                safe_record = {
                    field: record.get(field) for field in SAFE_METADATA_FIELDS
                }
                safe_record["images"] = [selected_image] if selected_image else []
                writers[split].write(
                    json.dumps(safe_record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                metadata_items.add(item)
                metadata_counts[split] += 1

    return temporary, metadata_counts, metadata_items, duplicate_metadata


def write_item_mapping(
    assignment: dict[str, str], item_counts: Counter
) -> Path:
    path = OUTPUT_DIR / "item_split.csv.gz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    mapping = pd.DataFrame(
        {
            "parent_asin": list(assignment),
            "split": [assignment[item] for item in assignment],
            "interaction_count": [item_counts[item] for item in assignment],
        }
    ).sort_values("parent_asin")
    mapping.to_csv(temporary, index=False, compression="gzip")
    return temporary


def validate(
    assignment: dict[str, str],
    total_rows: int,
    interaction_counts: Counter,
    metadata_items: set[str],
) -> dict:
    split_items = {
        split: {item for item, assigned in assignment.items() if assigned == split}
        for split in SPLITS
    }
    overlaps = {
        "train_validation": len(split_items["train"] & split_items["validation"]),
        "train_test": len(split_items["train"] & split_items["test"]),
        "validation_test": len(split_items["validation"] & split_items["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Product overlap detected: {overlaps}")
    if sum(interaction_counts.values()) != total_rows:
        raise AssertionError("Interaction rows were lost while splitting")

    ratios = {
        split: interaction_counts[split] / total_rows for split in SPLITS
    }
    if any(abs(ratios[split] - TARGET_RATIOS[split]) > 0.005 for split in SPLITS):
        raise AssertionError(f"Interaction ratios are outside tolerance: {ratios}")

    missing_metadata = set(assignment) - metadata_items
    return {
        "seed": SEED,
        "method": "Group products sharing a normalized MAIN-image ID, then greedily balance groups by interaction count.",
        "history_policy": "Original history was removed; rebuild from train-only earlier events.",
        "target_ratios": TARGET_RATIOS,
        "interaction_rows": dict(interaction_counts),
        "interaction_ratios": ratios,
        "product_counts": {
            split: len(split_items[split]) for split in SPLITS
        },
        "product_overlap": overlaps,
        "metadata_counts": {},
        "missing_metadata_products": len(missing_metadata),
    }


def promote(temporary_paths: list[Path]) -> None:
    for temporary in temporary_paths:
        final = temporary.with_suffix("")
        temporary.replace(final)


def main() -> None:
    require_sources()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    item_counts, total_rows, user_count, duplicate_rows = scan_interactions()
    if duplicate_rows:
        raise ValueError(f"Duplicate interaction rows detected: {duplicate_rows}")

    items = sorted(item_counts)
    item_index = {item: index for index, item in enumerate(items)}
    union_find = UnionFind(len(items))
    scanned_metadata_items, image_count, shared_links = connect_shared_main_images(
        item_index, union_find
    )
    missing_on_scan = len(items) - len(scanned_metadata_items)
    if missing_on_scan:
        raise ValueError(f"Products missing metadata: {missing_on_scan}")

    assignment, expected_rows, component_count, largest_component = assign_components(
        items, item_counts, union_find
    )
    del item_index, union_find, scanned_metadata_items

    interaction_temp, interaction_counts = write_interactions(assignment)
    if dict(interaction_counts) != expected_rows:
        raise AssertionError("Written row counts do not match assigned row counts")

    metadata_temp, metadata_counts, metadata_items, duplicate_metadata = write_metadata(
        assignment
    )
    mapping_temp = write_item_mapping(assignment, item_counts)
    manifest = validate(
        assignment, total_rows, interaction_counts, metadata_items
    )
    manifest.update(
        {
            "source_rows": total_rows,
            "source_users": user_count,
            "source_products": len(items),
            "duplicate_interactions": duplicate_rows,
            "duplicate_metadata_records": duplicate_metadata,
            "main_image_ids": image_count,
            "shared_main_image_links": shared_links,
            "image_connected_components": component_count,
            "largest_image_component_products": largest_component,
            "metadata_counts": dict(metadata_counts),
            "excluded_metadata_fields": [
                "average_rating",
                "rating_number",
                "bought_together",
            ],
        }
    )

    manifest_temp = OUTPUT_DIR / "split_manifest.json.tmp"
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    promote(
        [
            *[interaction_temp[split] for split in SPLITS],
            *[metadata_temp[split] for split in SPLITS],
            mapping_temp,
            manifest_temp,
        ]
    )

    print("\nSplit completed and validated")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
