"""Fetch the pinned public Toys & Games metadata shards and verify every file."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


DATASET = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
SUBSET = "raw_meta_Toys_and_Games"
FILES = (
    (218253388, "de03dc00dd34e5f3a37d9fcb85c16d3352d768d88d3b25df655723596e697d08"),
    (209914262, "fbb68c7938ee7cd233755e96b65735fceff5ac52c00e28923f751336c9aa2237"),
    (199494366, "e6b616e21b127deabe9c53f01129cf4213dd7a9161dd45d80392250b4f661ff4"),
    (187869599, "5afbcb1084071ccea033960236131feac73cd9f1771990dde13e53b34447a460"),
    (157135401, "1beb755c24eb9b33505b69924809d1290d824ba1e053d7b559b3519faa4bb668"),
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/raw/amazon_reviews_2023" / SUBSET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, size: int, expected_hash: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == expected_hash


def fetch(index: int, output: Path, curl: str) -> dict:
    name = f"full-{index:05d}-of-00005.parquet"
    expected_size, expected_hash = FILES[index]
    final = output / name
    partial = output / f"{name}.part"

    if final.exists():
        if verify(final, expected_size, expected_hash):
            print(f"Verified existing {name}", flush=True)
            return {"name": name, "size_bytes": expected_size, "sha256": expected_hash}
        raise FileExistsError(f"Existing file does not match the pinned source: {final}")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError(f"Partial file is larger than expected; inspect it manually: {partial}")
    if not verify(partial, expected_size, expected_hash):
        url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
               f"{SUBSET}/{name}")
        print(f"Downloading/resuming {name} ({partial.stat().st_size if partial.exists() else 0:,} bytes)...",
              flush=True)
        result = subprocess.run(
            [curl, "--location", "--fail", "--silent", "--show-error", "--retry", "5",
             "--retry-delay", "2", "--continue-at", "-", "--output", str(partial), url],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"curl failed for {name} with exit code {result.returncode}; partial kept")
    if not verify(partial, expected_size, expected_hash):
        raise ValueError(f"Downloaded file failed size/SHA-256 verification: {partial}")
    partial.replace(final)
    print(f"Verified {name}", flush=True)
    return {"name": name, "size_bytes": expected_size, "sha256": expected_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination folder; incomplete downloads use .part files.")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel shard downloads (default: 3).")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > len(FILES):
        parser.error("--workers must be between 1 and 5")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required to follow Hugging Face's resumable file redirects")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = [None] * len(FILES)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, i, output, curl): i for i in range(len(FILES))}
        for future in as_completed(futures):
            record = future.result()
            records[futures[future]] = record

    manifest = {"dataset": DATASET, "revision": REVISION, "subset": SUBSET,
                "files": records}
    temporary = output / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "download_manifest.json")
    print(f"All {len(FILES)} shards verified. Manifest: {output / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
