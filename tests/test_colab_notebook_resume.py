"""Exercise notebook resume paths without mounting Drive or starting GPU work."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["UFM_Colab_Train.ipynb", "UFM_Colab_Browser_Train.ipynb"]


def cell(name, index):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][index]["source"])


def load_function(name, function_name, namespace):
    tree = ast.parse(cell(name, 7))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), namespace)
    return namespace[function_name]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_all_code_cells_parse(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def backed_up_reports(tmp_path):
    remote = tmp_path / "drive" / "reports"
    local = tmp_path / "local_reports"
    remote.mkdir(parents=True)
    local.mkdir()
    entries = []
    for relative, payload in [("full_s42_v3.json", b'{"validated": true}'),
                              ("full_s42_v3_scores/scores_validation.npz", b"saved scores")]:
        path = remote / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append({"path": relative, "bytes": len(payload), "sha256": digest(path)})
    record = {"source_commit": "locked-commit", "files": entries}
    receipt = remote / "sync_receipt.json"
    receipt.write_text(json.dumps(record), encoding="utf-8")

    def copy_verified(source, destination, expected):
        assert digest(source) == expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return remote, local, record, receipt, {
        "Path": Path, "json": json, "DRIVE_ROOT": remote.parent,
        "REPORTS": local, "SOURCE_COMMIT": "locked-commit",
        "sha": digest, "copy_verified": copy_verified,
    }


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_new_session_restores_reports_and_predictions(name, backed_up_reports):
    remote, local, record, _, namespace = backed_up_reports
    restore = load_function(name, "restore_reports", namespace)
    restore()
    for entry in record["files"]:
        assert (local / entry["path"]).read_bytes() == (remote / entry["path"]).read_bytes()
    # Re-running startup is safe and preserves the verified reports.
    restore()


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_restored_complete_evaluation_is_not_run_again(name, backed_up_reports):
    _, _, _, _, namespace = backed_up_reports
    load_function(name, "restore_reports", namespace)()
    checks = []

    def unexpected_evaluation(*args):
        pytest.fail("A complete, SHA-verified evaluation must not be run again")

    namespace.update({
        "RUNS": namespace["REPORTS"].parent / "runs",
        "PROTOCOL": {},
        "locked": SimpleNamespace(verify_completed=lambda *args: checks.append(args)),
        "run_logged": unexpected_evaluation,
        "sync_folder": lambda *args: None,
        "sync_run": lambda *args: None,
    })
    load_function(name, "evaluate_complete", namespace)("full", 42)
    assert len(checks) == 1


@pytest.mark.parametrize("name", NOTEBOOKS)
@pytest.mark.parametrize("failure", ["corrupt_scores", "wrong_commit", "unsafe_path", "missing_receipt"])
def test_invalid_backup_does_not_restore_partial_reports(name, failure, backed_up_reports):
    remote, local, record, receipt, namespace = backed_up_reports
    if failure == "corrupt_scores":
        (remote / record["files"][1]["path"]).write_bytes(b"changed")
    elif failure == "wrong_commit":
        record["source_commit"] = "another-commit"
        receipt.write_text(json.dumps(record), encoding="utf-8")
    elif failure == "unsafe_path":
        record["files"][1]["path"] = "../outside.npz"
        receipt.write_text(json.dumps(record), encoding="utf-8")
    else:
        receipt.unlink()
    restore = load_function(name, "restore_reports", namespace)
    with pytest.raises((RuntimeError, ValueError)):
        restore()
    assert list(local.rglob("*")) == []


@pytest.mark.parametrize("name", NOTEBOOKS)
@pytest.mark.parametrize("existing,expected_stop", [(0, 100), (100, 300), (175, 375)])
def test_missing_throughput_can_be_measured_after_checkpoint(name, existing, expected_stop, tmp_path):
    calls = []

    def train_slice(variant, seed, stop):
        calls.append((variant, seed, stop))
        return [{"at": 10, "step": stop - 99, "batches": 1000, "vram_gib": 1.5},
                {"at": 109, "step": stop, "batches": 1000, "vram_gib": 2.0}]

    def save_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    namespace = {"first": tmp_path / "run", "existing": existing, "PROBE_STEPS": 100,
                 "benchmark_path": tmp_path / "throughput.json", "train_slice": train_slice,
                 "PROTOCOL": {"training": {"epochs": 20, "batch_size": 256}},
                 "JOBS": list(range(15)), "gpu_line": "T4", "SOURCE_COMMIT": "locked-commit",
                 "save_json": save_json, "json": json}
    tree = ast.parse(cell(name, 9))
    measurement = next(node for node in tree.body if isinstance(node, ast.If))
    code = compile(ast.Module(body=[measurement], type_ignores=[]), name, "exec")
    exec(code, namespace)
    assert calls == [("full", 42, expected_stop)]
    assert namespace["benchmark"]["seconds_per_step"] == 1
    assert namespace["benchmark"]["measured_steps"] == 99
    # A saved measurement must be reused without starting another training slice.
    exec(code, namespace)
    assert len(calls) == 1
