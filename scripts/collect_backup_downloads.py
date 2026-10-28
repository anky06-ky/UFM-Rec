"""Recover completed backup downloads by SHA-256, even if browser left random names.

Only files whose size AND SHA-256 match a known bundle are moved. Unknown files
are untouched. Duplicate downloads remain until an explicit cleanup step.
"""
import argparse
import json
from pathlib import Path
import shutil

from artifact_backup import digest, save_json


def collect(backup, inbox):
    backup=backup.resolve()
    manifest_path=backup/'manifest.json'
    if not manifest_path.exists():
        manifest_path=backup/'manifest.pending.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    sizes={b['bytes'] for b in manifest['bundles']}
    expected={b['sha256']:b for b in manifest['bundles']}
    moved,duplicates=[],[]
    for folder in inbox:
        for p in folder.resolve().iterdir():
            if not p.is_file() or p.is_symlink() or p.suffix.lower() not in {'.tmp','.zip','.crdownload'}:
                continue
            before=p.stat()
            if before.st_size not in sizes:
                continue
            hashed=digest(p)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                continue
            entry=expected.get(hashed)
            if entry is None:
                continue
            target=backup/entry['path']
            if target.exists():
                if digest(target)!=hashed:
                    raise ValueError('Existing backup bundle has wrong checksum: '+str(target))
                if p.resolve()!=target.resolve():
                    duplicates.append(str(p))
                continue
            shutil.move(str(p),str(target))
            moved.append(dict(source=str(p),destination=str(target),bytes=entry['bytes'],sha256=hashed))
    verified=[]
    missing=[]
    for entry in manifest['bundles']:
        p=backup/entry['path']
        if p.exists() and p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']:
            verified.append(entry['path'])
        else:
            missing.append(entry['path'])
    result=dict(manifest_phase=manifest['phase'],verified=verified,missing=missing,
                verified_bytes=sum(e['bytes'] for e in manifest['bundles'] if e['path'] in verified),
                moved=moved,duplicate_downloads=duplicates)
    save_json(backup/'download_receipt.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--inbox',type=Path,nargs='+',required=True)
    args=p.parse_args()
    print(json.dumps(collect(args.backup,args.inbox),indent=2))
