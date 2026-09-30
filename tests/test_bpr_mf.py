"""Small end-to-end and ranking checks for the train-only BPR baseline."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_evaluate_bpr_mf as bpr


class BPRTests(unittest.TestCase):
    def test_negative_exclusion_and_update_improves_pair_margin(self):
        from scipy import sparse
        graph = sparse.csr_matrix(([1, 1, 1, 1], ([0, 0, 1, 1], [0, 1, 1, 2])), shape=(2, 4))
        users = np.repeat(np.arange(2), 500)
        negatives = bpr.sample_negatives(graph, users, np.arange(4), np.random.default_rng(7))
        self.assertFalse(graph[users, negatives].A1.any())
        p = np.asarray([[0.1, 0.2]], dtype=np.float32)
        q = np.asarray([[0.2, 0.1], [0.0, 0.1]], dtype=np.float32)
        before = float(p[0] @ (q[0] - q[1]))
        for _ in range(20):
            bpr.update_batch(p, q, np.array([0]), np.array([0]), np.array([1]), 0.05, 0)
        self.assertGreater(float(p[0] @ (q[0] - q[1])), before)

    def test_tiny_train_uses_validation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            output = Path(directory) / "run"
            (data / "content").mkdir(parents=True)
            (data / "evaluation").mkdir()
            with gzip.open(data / "content/items.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row_index", "parent_asin", "train_count"])
                writer.writerows([[i, f"I{i}", count] for i, count in enumerate([1, 2, 1, 1, 0])])
            with gzip.open(data / "interactions_train.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["user_id", "parent_asin"])
                writer.writerows([["A", "I0"], ["A", "I1"], ["B", "I1"], ["B", "I2"], ["C", "I3"]])
            (data / "split_manifest.json").write_text(json.dumps({"interactions": {"train": 5}}), encoding="utf-8")
            with (data / "evaluation/samples_validation.npz").open("wb") as f:
                np.savez(f, candidates=np.asarray([[4, 0, 1], [1, 0, 3], [2, 1, 3], [3, 1, 2]], dtype=np.int32),
                         regime_codes=np.arange(4, dtype=np.uint8),
                         history_indices=np.asarray([0, 1, 1], dtype=np.int32),
                         history_indptr=np.asarray([0, 1, 2, 3, 3], dtype=np.int64))
            with gzip.open(data / "evaluation/samples_validation.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_index", "user_id"])
                writer.writerows([[0, "A"], [1, "A"], [2, "B"], [3, "Z"]])
            bpr.main(["--data", str(data), "--output", str(output), "--epochs", "2", "--factors", "4", "--batch-size", "2"])
            result = json.loads((output / "completed.json").read_text())
            self.assertEqual(result["epochs"], 2)
            self.assertFalse(result["smoke_run"])
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(json.loads((output / "graph_report.json").read_text())["skipped_singleton_users"], 1)
            self.assertTrue((output / "best.npz").is_file())
            with self.assertRaises(FileExistsError):
                bpr.main(["--data", str(data), "--output", str(output), "--epochs", "2", "--factors", "4", "--batch-size", "2"])


if __name__ == "__main__":
    unittest.main()
