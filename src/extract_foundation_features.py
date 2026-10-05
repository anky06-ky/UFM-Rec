"""Frozen CLIP text/image cache, bounded downloads and chunk-boundary resume.

No recommender training or test evaluation. By default only 128 products are
processed as a smoke cache. Full extraction must explicitly specify --limit 0
and use a DIFFERENT output folder. Don't run this on a busy training GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from io import BytesIO
from itertools import zip_longest
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from prepare_ufm_catalog import DATA, ROOT, allowed_image_url
from common import sha256
from ufm_model import FrozenCLIPEncoder

CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never follow a metadata URL to a different/private host.


def download_image(url, timeout=10):
    """Single bounded attempt. Failed URLs stay missing, with a reason recorded."""
    if not url:
        return None, "no_url"
    if not allowed_image_url(url):
        return None, "rejected_url"
    try:
        request = Request(url, headers={"User-Agent": "UFM-Rec-academic-feature-check/1.0"})
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_IMAGE_BYTES:
                return None, "too_large"
            content = response.read(MAX_IMAGE_BYTES + 1)
        if len(content) > MAX_IMAGE_BYTES:
            return None, "too_large"
        with Image.open(BytesIO(content)) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                return None, "too_many_pixels"
            source.load()
            image = source.convert("RGB")
        return image, "ok"
    except HTTPError as error:
        return None, f"http_{error.code}"
    except (URLError, TimeoutError, OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        return None, type(error).__name__


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu for a SMALL smoke check.")
    torch.set_num_threads(args.threads)
    catalog = args.catalog / "catalog.csv.gz"
    report_path = args.catalog / "catalog_report.json"
    text_path = args.data / "content/products_text.csv.gz"
    for path in (catalog, report_path, text_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256(catalog) != report["catalog_sha256"] or sha256(text_path) != report["sources_sha256"]["text"]:
        raise ValueError("Catalog or product text hash mismatch.")
    count = report["counts"]["items"]
    count = min(count, args.limit) if args.limit else count
    if not isinstance(count, int) or count < 1:
        raise ValueError("A non-empty catalog is required.")
    config = {"version": 1, "rows": count, "feature_dim": 512, "padding_row": 0,
              "index_policy": "feature[item row_index + 1]; row zero reserved for padding",
              "model": "openai/clip-vit-base-patch32", "revision": args.revision,
              "catalog_sha256": report["catalog_sha256"], "text_sha256": report["sources_sha256"]["text"],
              "limit": args.limit, "batch_size": args.batch_size, "device": args.device,
              "text_max_tokens": 77, "dtype": "float16", "image_timeout": args.timeout,
              "image_policy": "one product image; allowed HTTPS host only; no redirects; max 8MiB/16MP; one attempt",
              "learning": "Frozen pretrained encoder; no recommender gradient; no test evaluation"}
    progress_file = args.output / "progress.json"
    if (args.output / "complete.json").exists():
        raise FileExistsError("Cache already complete; choose a new output.")
    if args.resume:
        saved = json.loads(progress_file.read_text(encoding="utf-8"))
        if saved["config"] != config:
            raise ValueError("Resume config/source mismatch.")
        next_row, failures = saved["next_row"], saved["image_status_counts"]
        if not isinstance(next_row, int) or not 0 <= next_row <= count or sum(failures.values()) != next_row:
            raise ValueError("Invalid resume cursor/status counters.")
        mode = "r+"
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Output nonempty. Use --resume or choose a new folder.")
        args.output.mkdir(parents=True, exist_ok=True)
        next_row, failures, mode = 0, {}, "w+"
    text_table = np.lib.format.open_memmap(args.output / "text.npy", mode=mode, dtype=np.float16, shape=(count + 1, 512))
    image_table = np.lib.format.open_memmap(args.output / "image.npy", mode=mode, dtype=np.float16, shape=(count + 1, 512))
    flags = np.lib.format.open_memmap(args.output / "modalities.npy", mode=mode, dtype=np.bool_, shape=(count + 1, 2))
    if text_table.shape != (count + 1, 512) or image_table.shape != text_table.shape or flags.shape != (count + 1, 2) or text_table.dtype != np.float16 or image_table.dtype != np.float16 or flags.dtype != np.bool_:
        raise ValueError("Cached array shape mismatch.")
    if not args.resume:
        text_table[:] = 0
        image_table[:] = 0
        flags[:] = False
        text_table.flush(); image_table.flush(); flags.flush()
        atomic_json(progress_file, {"config": config, "next_row": 0, "image_status_counts": {}})
    encoder = FrozenCLIPEncoder(args.revision, args.device, cache_dir=args.model_cache)
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Foundation encoder must be fully frozen.")
    started = time.monotonic()
    print(f"Frozen CLIP loaded; extracting {count:,} products from row {next_row:,} on {args.device}.", flush=True)
    with gzip.open(catalog, "rt", encoding="utf-8", newline="") as cf, \
         gzip.open(text_path, "rt", encoding="utf-8", newline="") as tf, \
         ThreadPoolExecutor(max_workers=args.workers) as pool, \
         (args.output / "image_errors.jsonl").open("a", encoding="utf-8") as error_log:
        rows = zip_longest(csv.DictReader(cf), csv.DictReader(tf))
        batch = []
        for index, (catalog_row, text_row) in enumerate(rows):
            if index >= count:
                break
            if catalog_row is None or text_row is None or int(catalog_row["row_index"]) != index or catalog_row["parent_asin"] != text_row["parent_asin"]:
                raise ValueError("Catalog/text row alignment mismatch.")
            if index < next_row:
                continue
            batch.append((index, catalog_row, text_row))
            if len(batch) < args.batch_size and index + 1 < count:
                continue
            ids = np.asarray([x[0] + 1 for x in batch])
            # Erase any uncommitted rows written before a previous interruption.
            text_table[ids] = 0; image_table[ids] = 0; flags[ids] = False
            positions = [i for i, (_, _, row) in enumerate(batch) if row["text"].strip()]
            if positions:
                texts = [batch[i][2]["text"] for i in positions]  # Already title-first in source.
                encoded = encoder.encode_text(texts).cpu().numpy()
                if encoded.shape != (len(positions), 512) or not np.isfinite(encoded).all():
                    raise ValueError("Invalid text features.")
                text_table[ids[positions]] = encoded.astype(np.float16)
                flags[ids[positions], 0] = True
            downloaded = list(pool.map(lambda row: download_image(row[1]["image_url"], args.timeout), batch))
            positions = [i for i, (image, status) in enumerate(downloaded) if status == "ok"]
            if positions:
                encoded = encoder.encode_images([downloaded[i][0] for i in positions]).cpu().numpy()
                if encoded.shape != (len(positions), 512) or not np.isfinite(encoded).all():
                    raise ValueError("Invalid image features.")
                image_table[ids[positions]] = encoded.astype(np.float16)
                flags[ids[positions], 1] = True
            for (_, row, _), (image, status) in zip(batch, downloaded):
                failures[status] = failures.get(status, 0) + 1
                if status != "ok":
                    error_log.write(json.dumps({"row_index": row["row_index"], "parent_asin": row["parent_asin"], "status": status}) + "\n")
                if image is not None:
                    image.close()
            # Tables reach stable storage BEFORE advancing the atomic cursor.
            text_table.flush(); image_table.flush(); flags.flush(); error_log.flush()
            next_row = index + 1
            atomic_json(progress_file, {"config": config, "next_row": next_row, "image_status_counts": failures})
            print(f"features={next_row:,}/{count:,} image_ok={failures.get('ok', 0):,} elapsed_s={time.monotonic() - started:.1f}", flush=True)
            batch = []
    if next_row != count:
        raise ValueError("Input ended before expected rows.")
    from importlib.metadata import version, PackageNotFoundError
    def safe_version(name):
        try: return version(name)
        except PackageNotFoundError: return "unknown"
    completed = {**config, "has_text": int(flags[1:, 0].sum()), "has_image": int(flags[1:, 1].sum()),
                 "image_status_counts": failures, "elapsed_seconds": time.monotonic() - started,
                 "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                 "transformers_version": safe_version("transformers"), "pillow_version": safe_version("pillow"),
                 "encoder_frozen": True,
                 "features_sha256": {name: sha256(args.output / name) for name in ("text.npy", "image.npy", "modalities.npy")},
                 "limitations": ["CLIP 77-token text truncation; metadata snapshot may include later edits.",
                                 "Smoke limit cache cannot be used for full-catalog training.",
                                 "Interrupted chunk can duplicate diagnostic error lines; cursor/counts are authoritative."]}
    atomic_json(args.output / "complete.json", completed)
    print(json.dumps(completed, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--catalog", type=Path, default=DATA / "foundation_catalog_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/foundation_smoke_128_v1")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "tmp/hf_ufm_models")
    parser.add_argument("--revision", default=CLIP_REVISION)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or min(args.batch_size, args.workers, args.threads, args.timeout) <= 0 or args.workers > 16:
        parser.error("Positive sizes/timeout and 1-16 workers required; limit zero explicitly enables full extraction.")
    extract(args)


if __name__ == "__main__":
    main()
