"""CPU checks with synthetic data, never real train/validation/test metrics.

Run: python -m unittest discover -s tests -p test_ufm_core.py -v
"""
from dataclasses import replace
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm_model import UFMConfig, UFMRec, FrozenCLIPEncoder, reliability_weights, ufm_loss, candidate_calibration
from prepare_ufm_catalog import allowed_image_url, primary_image, build_catalog
import extract_foundation_features as extraction


class UFMCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = UFMConfig(items=8, feature_dim=12, dim=16, history_size=3,
                                heads=4, layers=2, dropout=0, id_dropout=0)
        self.model = UFMRec(self.config).eval()
        self.text = torch.randn(9, 12, requires_grad=True)
        self.image = torch.randn(9, 12, requires_grad=True)
        self.modalities = torch.ones(9, 2, dtype=torch.bool)
        self.modalities[0] = False
        self.modalities[8] = False
        self.counts = torch.tensor([0, 40, 3, 2, 1, 1, 0, 0, 0])
        self.histories = torch.tensor([[1, 2, 0], [2, 3, 4], [0, 0, 0]])
        self.candidates = torch.tensor([[3, 6, 8], [5, 6, 7], [3, 6, 8]])

    def predict(self, model=None):
        return (model or self.model)(self.histories, self.candidates, self.text,
                                    self.image, self.modalities, self.counts)

    def test_gate_matches_equations_and_remains_stable(self):
        u = torch.tensor([[0.2, 1.3], [1000., 1001.], [4., 3.], [5., 2.]])
        available = torch.tensor([[True, True], [True, True], [False, True], [False, False]])
        weights = reliability_weights(u, available)
        expected = torch.exp(-u[0]) / torch.exp(-u[0]).sum()
        torch.testing.assert_close(weights[0], expected)
        torch.testing.assert_close(weights[1], torch.softmax(-u[1], -1))
        torch.testing.assert_close(weights[2], torch.tensor([0., 1.]))
        torch.testing.assert_close(weights[3], torch.zeros(2))
        self.assertTrue(torch.isfinite(weights).all())
        # Increasing CF uncertainty must reduce CF weight, not increase it.
        self.assertGreater(float(weights[0, 0]), float(reliability_weights(u[:1] + torch.tensor([[5., 0.]]), available[:1])[0, 0]))

    def test_unknown_ids_do_not_affect_predictions_or_receive_gradients(self):
        before = self.predict()["scores"].detach().clone()
        with torch.no_grad():
            self.model.item_ids.weight[6:].fill_(float("nan"))
        torch.testing.assert_close(before, self.predict()["scores"])
        loss = ufm_loss(self.predict(), torch.ones(3, 2, dtype=torch.bool))["total"]
        loss.backward()
        self.assertTrue(torch.equal(self.model.item_ids.weight.grad[6:], torch.zeros(3, 16)))
        self.assertTrue(torch.equal(self.model.item_ids.weight.grad[0], torch.zeros(16)))

    def test_missing_modalities_and_empty_history(self):
        before = self.predict()
        self.assertTrue(torch.isfinite(before["scores"]).all())
        torch.testing.assert_close(before["weights"][0, 1], torch.tensor([0., 1.]))
        self.assertTrue(torch.equal(before["weights"][2], torch.zeros(3, 2)))
        torch.testing.assert_close(before["scores"][2], self.counts[self.candidates[2]].float().log1p())
        with torch.no_grad():
            self.text[0] = float("nan")
            self.image[0] = float("nan")
            self.text[8] = float("nan")
            self.image[8] = float("nan")
        torch.testing.assert_close(before["scores"], self.predict()["scores"])

    def test_causal_history_cannot_read_later_positions(self):
        x = torch.randn(1, 3, 16)
        causal = torch.ones(3, 3, dtype=torch.bool).triu(1)
        before = self.model.sequential(x, mask=causal)
        x[:, 2] += 999
        after = self.model.sequential(x, mask=causal)
        torch.testing.assert_close(before[:, :2], after[:, :2])

    def test_backward_adapter_uncertainty_alignment_and_frozen_tables(self):
        self.model.train()
        output = self.predict()
        valid = torch.tensor([[True, False], [True, True], [False, False]])
        result = ufm_loss(output, valid)
        self.assertTrue(all(torch.isfinite(v) for v in result.values()))
        result["total"].backward()
        self.assertIsNone(self.text.grad)
        self.assertIsNone(self.image.grad)
        for block in (self.model.adapter, self.model.cf_uncertainty, self.model.sem_uncertainty, self.model.align_projection):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer.step()
        self.assertTrue(torch.isfinite(self.predict()["scores"]).all())

    def test_ablation_paths_and_modality_invariance(self):
        for variant in ("full", "no_uncertainty", "fixed_fusion", "no_cross_align", "semantic_only", "collaborative_only", "text_only", "image_only"):
            with self.subTest(variant=variant):
                model = UFMRec(replace(self.config, variant=variant)).eval()
                out = self.predict(model)
                self.assertTrue(torch.isfinite(out["scores"]).all())
                if variant == "semantic_only":
                    self.assertTrue(torch.equal(out["weights"][..., 0], torch.zeros(3, 3)))
                if variant == "collaborative_only":
                    self.assertTrue(torch.equal(out["weights"][..., 1], torch.zeros(3, 3)))
                if variant == "no_cross_align":
                    self.assertTrue(torch.equal(out["alignment"], torch.zeros(3, 3, 16)))
                if variant in ("text_only", "image_only"):
                    target = self.image if variant == "text_only" else self.text
                    previous = target.detach().clone()
                    flags = self.modalities.clone()
                    with torch.no_grad():
                        target.fill_(float("nan"))
                    self.modalities[:, 1 if variant == "text_only" else 0] = False
                    torch.testing.assert_close(out["scores"], self.predict(model)["scores"])
                    with torch.no_grad():
                        target.copy_(previous)
                    self.modalities = flags

    def test_fixed_fusion_extremes_keep_only_available_branch(self):
        for alpha in (0., 1.):
            model = UFMRec(replace(self.config, variant="fixed_fusion", fixed_alpha=alpha)).eval()
            torch.testing.assert_close(self.predict(model)["weights"][0, 1], torch.tensor([0., 1.]))

    def test_no_uncertainty_ablation_does_not_train_uncertainty_heads(self):
        model = UFMRec(replace(self.config, variant="no_uncertainty")).train()
        result = ufm_loss(self.predict(model), torch.ones(3, 2, dtype=torch.bool))
        self.assertEqual(float(result["unc"]), 0.)
        result["total"].backward()
        self.assertTrue(all(p.grad is None for block in (model.cf_uncertainty, model.sem_uncertainty) for p in block.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.learned_gate.parameters()))

    def test_boundaries_and_no_valid_negatives(self):
        with self.assertRaises(ValueError):
            UFMConfig(items=8, dim=15)
        with self.assertRaises(ValueError):
            FrozenCLIPEncoder("main")  # Must reject before downloading any weights.
        with self.assertRaises(ValueError):
            ufm_loss(self.predict(), torch.zeros(3, 2, dtype=torch.bool))
        self.histories[0] = torch.tensor([1, 0, 2])
        with self.assertRaises(ValueError):
            self.predict()

    def test_calibration_has_exact_hand_computable_results(self):
        # Uniform two-class predictions, one correct and one incorrect top-1.
        result = candidate_calibration(torch.zeros(2, 2), torch.tensor([0, 1]), bins=2)
        self.assertAlmostEqual(result["ece"], 0.)
        self.assertAlmostEqual(result["nll"], float(torch.tensor(2.).log()), places=6)
        self.assertAlmostEqual(result["brier_multiclass"], .5)
        self.assertEqual(sum(x["n"] for x in result["bins"]), 2)
        # Even if every positive is column zero, ties must not produce fake 100% accuracy.
        result = candidate_calibration(torch.zeros(2, 2), bins=2)
        self.assertAlmostEqual(result["ece"], 0.)
        with self.assertRaises(ValueError):
            candidate_calibration(torch.zeros(2, 2), temperature=0)


class CatalogTests(unittest.TestCase):
    def test_main_url_selection_and_untrusted_hosts(self):
        url = "https://m.media-amazon.com/images/I/example.jpg"
        self.assertTrue(allowed_image_url(url))
        for invalid in ("http://m.media-amazon.com/images/I/a.jpg", "https://127.0.0.1/images/I/a.jpg",
                        "https://m.media-amazon.com.evil.com/images/I/a.jpg", "https://user@m.media-amazon.com/images/I/a.jpg"):
            self.assertFalse(allowed_image_url(invalid))
        self.assertEqual(primary_image({"images": [{"variant": "PT01", "large": url + "?other"}, {"variant": "MAIN", "large": url}]}), url)

    def test_tiny_catalog_alignment_coverage_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "content").mkdir(parents=True)
            for name, fields, rows in (
                ("items", ["row_index", "parent_asin", "train_count", "cold_start_regime"],
                 [[0, "A", 3, "extreme_cold"], [1, "B", 0, "zero_shot"]]),
                ("products_text", ["parent_asin", "text"], [["A", "Toy"], ["B", ""]]),
            ):
                with gzip.open(data / "content" / f"{name}.csv.gz", "wt", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerows(rows)
            (data / "split_manifest.json").write_text("{}", encoding="utf-8")
            metadata = root / "meta.gz"
            with gzip.open(metadata, "wt", encoding="utf-8") as f:
                f.write(json.dumps({"parent_asin": "A", "images": [{"large": "https://m.media-amazon.com/images/I/a.jpg"}]} ) + "\n")
            output = root / "out"
            result = build_catalog(data, metadata, output)
            self.assertEqual(result["counts"]["items"], 2)
            self.assertEqual(result["counts"]["both"], 1)
            self.assertEqual(result["counts"]["missing_metadata"], 1)
            with self.assertRaises(FileExistsError):
                build_catalog(data, metadata, output)


class ExtractionTests(unittest.TestCase):
    def test_resume_reproduces_uninterrupted_features_and_counts(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def encode_text(self, texts):
                return torch.full((len(texts), 512), 1 / 512 ** .5)

            def encode_images(self, images):
                return torch.full((len(images), 512), 1 / 512 ** .5)

        def fake_download(url, timeout):
            return (Image.new("RGB", (8, 8)), "ok") if url else (None, "no_url")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "content").mkdir(parents=True)
            fields = ["row_index", "parent_asin", "train_count", "cold_start_regime"]
            with gzip.open(data / "content/items.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f); writer.writerow(fields)
                writer.writerows([[i, f"I{i}", 1, "extreme_cold"] for i in range(3)])
            with gzip.open(data / "content/products_text.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f); writer.writerow(["parent_asin", "text"])
                writer.writerows([[f"I{i}", "Toy"] for i in range(3)])
            (data / "split_manifest.json").write_text("{}", encoding="utf-8")
            metadata = root / "metadata.gz"
            with gzip.open(metadata, "wt", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps({"parent_asin": f"I{i}", "images": [{"large": "https://m.media-amazon.com/images/I/a.jpg"}] if i != 2 else []}) + "\n")
            catalog = root / "catalog"
            build_catalog(data, metadata, catalog)
            args = SimpleNamespace(data=data, catalog=catalog, output=root / "interrupted", model_cache=root / "models",
                                   revision=extraction.CLIP_REVISION, device="cpu", limit=3, batch_size=2,
                                   workers=1, threads=2, timeout=1, resume=False)
            original_write = extraction.atomic_json

            def interrupt_after_commit(path, value):
                original_write(path, value)
                if path.name == "progress.json" and value["next_row"] == 2:
                    raise InterruptedError("Synthetic interruption after chunk commit")

            with patch.object(extraction, "FrozenCLIPEncoder", TinyEncoder), patch.object(extraction, "download_image", fake_download):
                with patch.object(extraction, "atomic_json", interrupt_after_commit), self.assertRaises(InterruptedError):
                    extraction.extract(args)
                self.assertFalse((args.output / "complete.json").exists())
                args.resume = True
                extraction.extract(args)
                first_output = args.output
                with self.assertRaises(FileExistsError):
                    extraction.extract(args)
                args.output = root / "fresh"
                args.resume = False
                extraction.extract(args)
            for name in ("text.npy", "image.npy", "modalities.npy"):
                np.testing.assert_array_equal(np.load(first_output / name), np.load(args.output / name))
            first = json.loads((first_output / "complete.json").read_text())
            second = json.loads((args.output / "complete.json").read_text())
            self.assertEqual(first["image_status_counts"], second["image_status_counts"])
            self.assertEqual(first["features_sha256"], second["features_sha256"])


if __name__ == "__main__":
    unittest.main()
