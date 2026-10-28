"""Create and verify portable, content-deduplicated backups (standard library only).

No credentials, virtual environments, Git internals or temporary caches are
included. Restore into a NEW directory; never overwrite a working checkout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

FOLDERS = ("data", "runs", "reports", "output", "src", "tests", "scripts",
           "ops", "configs", "demo", "docs", "notebooks", ".github")
EXCLUDED = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "node_modules",
            ".git", ".venv", ".venv_ufm", ".venv_ufm_runtime"}
ROOT_FILES = {"README.md", "START.md", "Start_Demo.cmd", "status.py", "pytest.ini",
              ".gitignore", ".gitattributes", "LICENSE"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or not name.parts or any(p in {"..", "."} or ":" in p or "\\" in p for p in name.parts):
        raise ValueError(f"Unsafe backup path: {relative}")
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Backup path escapes destination: {relative}")
    return target


def allowed(path, root):
    parts = path.relative_to(root).parts
    return not any(part in EXCLUDED or part.startswith(".env") for part in parts) and path.suffix.lower() not in {
        ".pyc", ".pyo", ".pem", ".key", ".partial", ".tmp"}


def source_files(root, includes=None):
    if includes:
        files = []
        for relative in includes:
            start = safe_path(root, relative)
            if not start.exists() or start.is_symlink() or not allowed(start, root):
                raise ValueError(f"Missing or excluded backup input: {relative}")
            candidates = start.rglob("*") if start.is_dir() else [start]
            files.extend(p for p in candidates if p.is_file() and not p.is_symlink()
                         and allowed(p, root) and not any(parent.is_symlink() for parent in p.parents if parent != root))
        if not files:
            raise ValueError("No eligible files in selected backup inputs")
        return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())
    files = []
    for folder in FOLDERS:
        start = root / folder
        if start.exists() and not start.is_symlink():
            files.extend(p for p in start.rglob("*") if p.is_file() and not p.is_symlink()
                         and allowed(p, root) and not any(parent.is_symlink() for parent in p.parents if parent != root))
    files.extend(p for p in root.iterdir() if p.is_file() and not p.is_symlink() and
                 (p.name in ROOT_FILES or p.name.startswith("requirements") or p.suffix.lower() in {".ipynb", ".docx", ".pdf"}))
    snapshot = root / "audit_v2/fix_1_15_workspace_20261027"
    if snapshot.exists():
        files.extend(p for p in snapshot.rglob("*") if p.is_file() and not p.is_symlink()
                     and allowed(p, root) and not any(parent.is_symlink() for parent in p.parents if parent != root))
    # Small provenance first, then full inference checkpoint, data and other runs.
    def priority(p):
        rel = p.relative_to(root).as_posix()
        return (0 if p.stat().st_size < 1024 * 1024 else
                1 if rel == "runs/ufm_full_v1/best.pt" else
                2 if rel.startswith("data/") else 3, rel)
    return sorted(set(files), key=priority)


def save_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def create(root, output, bundle_bytes=768 * 1024**2, includes=None):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or list(output.glob("bundle_*.zip*")):
        raise FileExistsError("Refusing to overwrite a previous backup")
    manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_root=str(root), phase="hashing", files=[], bundles=[],
                    excluded="Virtual environments, Git internals, secrets, temporary caches; no source files deleted.")
    seen = {}
    for p in source_files(root, includes):
        before = p.stat()
        sha = digest(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during snapshot: {p}")
        relative = p.relative_to(root).as_posix()
        canonical = seen.setdefault((sha, before.st_size), relative)
        manifest["files"].append(dict(path=relative, bytes=before.st_size, sha256=sha,
                                      mtime_ns=before.st_mtime_ns, canonical=canonical))
        if len(manifest["files"]) % 25 == 0:
            save_json(output / "progress.json", dict(phase="hashing", files=len(manifest["files"])))
    manifest["logical_bytes"] = sum(f["bytes"] for f in manifest["files"])
    unique = [f for f in manifest["files"] if f["canonical"] == f["path"]]
    manifest["unique_bytes"] = sum(f["bytes"] for f in unique)
    groups, current, size = [], [], 0
    for entry in unique:
        if current and size + entry["bytes"] > bundle_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry["bytes"]
    if current:
        groups.append(current)
    manifest["phase"] = "packing"
    save_json(output / "manifest.pending.json", manifest)
    for index, group in enumerate(groups):
        name = f"bundle_{index:03d}.zip"
        partial = output / (name + ".partial")
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for entry in group:
                p = safe_path(root, entry["path"])
                if (p.stat().st_size, p.stat().st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                    raise RuntimeError(f"Source changed before packing: {p}")
                archive.write(p, entry["path"])
                if (p.stat().st_size, p.stat().st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                    raise RuntimeError(f"Source changed while packing: {p}")
        partial.rename(output / name)
        manifest["bundles"].append(dict(path=name, bytes=(output / name).stat().st_size,
                                         sha256=digest(output / name), files=[e["path"] for e in group]))
        save_json(output / "manifest.pending.json", manifest)
        save_json(output / "progress.json", dict(phase="packing", completed=index + 1, total=len(groups)))
        print(f"BUNDLE_READY {name} {(output / name).stat().st_size}", flush=True)
    manifest["phase"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "manifest.json", manifest)
    save_json(output / "progress.json", dict(phase="complete", bundles=len(groups), files=len(manifest["files"])))
    print(f"BACKUP_COMPLETE files={len(manifest['files'])} logical_bytes={manifest['logical_bytes']} unique_bytes={manifest['unique_bytes']}", flush=True)
    return manifest


def restore(backup, output, available_only=False):
    backup, output = backup.resolve(), output.resolve()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("phase") != "complete":
        raise ValueError("Backup manifest is incomplete or unsupported")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Restore destination must be new or empty")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("Duplicate manifest paths")
    for entry in entries.values():
        safe_path(output, entry["path"])
        safe_path(output, entry["canonical"])
        canonical = entries.get(entry["canonical"])
        if canonical is None or canonical["canonical"] != canonical["path"] or (canonical["sha256"], canonical["bytes"]) != (entry["sha256"], entry["bytes"]):
            raise ValueError("Invalid content-deduplication reference")
    # Validate every archive before writing the first output byte.
    covered = set()
    selected_bundles = []
    missing_bundles = []
    for bundle in manifest["bundles"]:
        p = safe_path(backup, bundle["path"])
        if available_only and not p.exists():
            missing_bundles.append(bundle["path"])
            continue
        if p.stat().st_size != bundle["bytes"] or digest(p) != bundle["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {p.name}")
        selected_bundles.append(bundle)
        with zipfile.ZipFile(p) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(bundle["files"]):
                raise ValueError("Archive contents differ from manifest")
            for info in archive.infolist():
                if info.filename not in entries or entries[info.filename]["canonical"] != info.filename or info.filename in covered:
                    raise ValueError("Unexpected or duplicated archive entry")
                if info.file_size != entries[info.filename]["bytes"] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Invalid archive entry size or symlink")
                covered.add(info.filename)
    if not available_only and covered != {e["path"] for e in entries.values() if e["canonical"] == e["path"]}:
        raise ValueError("Backup omits canonical source files")
    restored_entries = {name: entry for name, entry in entries.items() if entry['canonical'] in covered}
    if not restored_entries:
        raise ValueError('No verified files available to restore')
    # Linux snapshots may contain both docs/FITLAB.md and docs/fitlab.md.
    # Preserve both on case-insensitive disks; record the reversible relocation.
    physical_paths = {}
    occupied = set()
    for name in restored_entries:
        destination = name
        if destination.casefold() in occupied:
            destination = '.restore_case_conflicts/' + hashlib.sha256(name.encode()).hexdigest() + '/' + name
        if destination.casefold() in occupied:
            raise ValueError('Conflicting restore paths')
        occupied.add(destination.casefold())
        physical_paths[name] = destination
    output.mkdir(parents=True, exist_ok=True)
    for bundle in selected_bundles:
        with zipfile.ZipFile(backup / bundle["path"]) as archive:
            for name in bundle["files"]:
                p = safe_path(output, physical_paths[name])
                p.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, p.open("xb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024**2)
                if digest(p) != entries[name]["sha256"]:
                    raise ValueError(f"Restored file checksum mismatch: {name}")
    for name, entry in restored_entries.items():
        if entry["canonical"] != name:
            p = safe_path(output, physical_paths[name])
            p.parent.mkdir(parents=True, exist_ok=True)
            with safe_path(output, physical_paths[entry["canonical"]]).open("rb") as source, p.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024**2)
    # Verify copied aliases as well; keep only fully checked copies in the receipt.
    for name, entry in restored_entries.items():
        p = safe_path(output, physical_paths[name])
        if p.stat().st_size != entry["bytes"] or digest(p) != entry["sha256"]:
            raise ValueError(f"Final restore verification failed: {name}")
    receipt = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), files=len(restored_entries),
                   logical_bytes=sum(e['bytes'] for e in restored_entries.values()),
                   manifest_sha256=digest(backup / "manifest.json"),
                   all_files_sha256_verified=len(restored_entries)==len(entries),
                   restored_files_sha256_verified=True, missing_bundles=missing_bundles,
                   relocated_case_conflicts={name:path for name,path in physical_paths.items() if name!=path},
                   missing_files=sorted(set(entries)-set(restored_entries)), destination=str(output))
    save_json(backup / ("partial_restore_receipt.json" if missing_bundles else "restore_receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle-mib", type=int, default=768)
    p.add_argument("--include", action="append", default=[],
                   help="Relative file or directory to back up; repeat to make a small batch instead of the full workspace.")
    p = sub.add_parser("restore")
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--available-only", action="store_true", help="Restore verified downloaded bundles; receipt explicitly lists missing files.")
    args = parser.parse_args()
    if args.command == "create":
        if args.bundle_mib < 1:
            parser.error("--bundle-mib must be positive")
        create(args.root, args.output, args.bundle_mib * 1024**2, args.include)
    else:
        restore(args.backup, args.output, available_only=args.available_only)
