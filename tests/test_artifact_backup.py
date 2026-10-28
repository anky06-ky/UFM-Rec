"""Backup must survive loss of host and reject corruption or unsafe restores."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("artifact_backup", Path(__file__).parents[1] / "scripts/artifact_backup.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def fixture_backup(tmp_path):
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "data/products.csv").write_bytes(b"asin,title\nA,LEGO\n")
    (root / "runs/checkpoint.pt").write_bytes(b"same trained state")
    (root / "runs/duplicate.pt").write_bytes(b"same trained state")
    (root / "data/.env").write_bytes(b"secret")
    (root / "data/__pycache__").mkdir()
    (root / "data/__pycache__/cache.pyc").write_bytes(b"cache")
    destination = tmp_path / "backup"
    manifest = backup.create(root, destination, bundle_bytes=20)
    return root, destination, manifest


def test_restore_preserves_deduplicated_artifacts_excludes_secrets_and_caches(tmp_path):
    root, packed, manifest = fixture_backup(tmp_path)
    assert len(manifest["files"]) == 3
    assert manifest["unique_bytes"] < manifest["logical_bytes"]
    restored = tmp_path / "restored"
    receipt = backup.restore(packed, restored)
    assert receipt["all_files_sha256_verified"]
    for entry in manifest["files"]:
        assert (root / entry["path"]).read_bytes() == (restored / entry["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        backup.restore(packed, restored)


def test_corrupt_archive_rejected_before_creating_restore(tmp_path):
    _, packed, manifest = fixture_backup(tmp_path)
    (packed / manifest["bundles"][0]["path"]).write_bytes(b"broken download")
    restored = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        backup.restore(packed, restored)
    assert not restored.exists()


def test_manifest_path_traversal_rejected(tmp_path):
    _, packed, manifest = fixture_backup(tmp_path)
    manifest["files"][0]["path"] = "../outside.pt"
    (packed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        backup.restore(packed, tmp_path / "restored")
    assert not (tmp_path / "outside.pt").exists()


def test_partial_restore_identifies_missing_files_and_keeps_deduplicated_aliases(tmp_path):
    root, packed, manifest = fixture_backup(tmp_path)
    missing = manifest['bundles'][0]
    (packed / missing['path']).unlink()
    restored = tmp_path / 'partial'
    receipt = backup.restore(packed, restored, available_only=True)
    assert not receipt['all_files_sha256_verified']
    assert receipt['restored_files_sha256_verified']
    assert receipt['missing_bundles'] == [missing['path']]
    assert set(receipt['missing_files']) == set(missing['files'])
    assert not (packed / 'restore_receipt.json').exists()
    for entry in manifest['files']:
        if entry['path'] not in receipt['missing_files']:
            assert (root / entry['path']).read_bytes() == (restored / entry['path']).read_bytes()


def test_case_collisions_preserved_with_recorded_relocation(tmp_path):
    _, packed, manifest = fixture_backup(tmp_path)
    original = next(e for e in manifest['files'] if e['path'] == 'data/products.csv')
    alias = dict(original, path='data/PRODUCTS.csv')
    manifest['files'].append(alias)
    manifest['logical_bytes'] += alias['bytes']
    (packed / 'manifest.json').write_text(json.dumps(manifest))
    restored = tmp_path / 'portable'
    receipt = backup.restore(packed, restored)
    relocated = receipt['relocated_case_conflicts'][alias['path']]
    assert (restored / relocated).read_bytes() == (restored / original['path']).read_bytes()
    assert receipt['all_files_sha256_verified']
