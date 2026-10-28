"""Check real catalog text-only and image-only histories against a live UFM demo."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np


def get(url: str):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8771")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    url = args.url.rstrip("/")
    status = get(url + "/api/status")
    if status["backend"] != "UFM Rec" or status["items"] != 767045:
        raise ValueError("Full UFM demo required")
    masks = np.load(args.data / "foundation_clip_b32_v1/modalities.npy", mmap_mode="r")[1:]
    kinds = {"text_only": masks[:, 0] & ~masks[:, 1],
             "image_only": ~masks[:, 0] & masks[:, 1],
             "neither": ~masks.any(axis=1)}
    indices = {name: np.flatnonzero(mask) for name, mask in kinds.items()}
    if not len(indices["text_only"]) or not len(indices["image_only"]):
        raise ValueError("Missing modality fixtures are absent")
    selected = {int(indices["text_only"][0]), int(indices["image_only"][0])}
    asins = {}
    with gzip.open(args.data / "content/items.csv.gz", "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            i = int(row["row_index"])
            if i in selected:
                asins[i] = row["parent_asin"]
            if len(asins) == len(selected):
                break
    if len(asins) != len(selected):
        raise ValueError("Catalog mapping incomplete")
    cases = []
    for name in ("text_only", "image_only"):
        i = int(indices[name][0])
        asin = asins[i]
        matches = get(url + "/api/search?q=" + quote(asin))
        if not any(item["asin"] == asin for item in matches):
            raise ValueError("Search did not find missing-modality item")
        payload = {"history": [asin], "k": 3, "regime": "all"}
        req = Request(url + "/api/recommend", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
        with urlopen(req, timeout=900) as response:
            result = json.load(response)
        if len(result["results"]) != 3 or any(item["asin"] == asin for item in result["results"]):
            raise ValueError("Invalid recommendation/exclusion for missing modality")
        cases.append({"kind": name, "row_index": i, "asin": asin,
                      "feature_mask_text_image": masks[i].tolist(),
                      "result_asins": [item["asin"] for item in result["results"]],
                      "model_sha256": result["model"]["checkpoint_sha256"]})
    try:
        get(url + "/api/product-image?asin=" + quote(cases[0]["asin"]))
        raise ValueError("Expected no offline JPEG for this full-catalog item")
    except HTTPError as error:
        if error.code != 404:
            raise
    report = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "catalog_items": status["items"], "checkpoint_sha256": status["model"]["checkpoint_sha256"],
              "availability_counts": {name: int(len(values)) for name, values in indices.items()},
              "cases": cases, "missing_local_jpeg_status": 404,
              "limitations": ["No real item lacks both text and image in this feature cache.",
                              "A missing display JPEG does not invalidate the saved CLIP embedding.",
                              "This is API behavior, not a browser image-error visual check."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "availability_counts": report["availability_counts"]}))


if __name__ == "__main__":
    main()
