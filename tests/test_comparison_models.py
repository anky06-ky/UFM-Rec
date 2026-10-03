from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train_sasrec import SASRec
from comparison_models import BERT4Rec, ConcatHybrid, cloze_batch
from audit_ufm_validation import paired_user_ci, ndcg


class ComparisonTests(unittest.TestCase):
    def test_bidirectional_context_and_masked_targets(self):
        torch.manual_seed(4)
        model = BERT4Rec(8, 4, 8, 2, 1, dropout=0).eval()
        first = model.encode_tokens(torch.tensor([[1, 9, 2, 0]]))
        changed = model.encode_tokens(torch.tensor([[1, 9, 3, 0]]))
        self.assertFalse(torch.allclose(first[:, 1], changed[:, 1]))
        tokens, rows, cols, labels = cloze_batch(np.array([[1, 2, 0, 0], [3, 4, 5, 0]]),
                                                np.array([3, 6]), 9, np.arange(1, 9),
                                                np.random.default_rng(5))
        original = np.array([[1, 2, 3, 0], [3, 4, 5, 6]])
        np.testing.assert_array_equal(labels, original[rows, cols])
        self.assertEqual(set(rows), {0, 1})
        self.assertTrue(np.all(labels > 0))
        logits = model.logits(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[3, 4]]))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.items.weight.grad)

    def test_concat_masks_unknown_ids_and_missing_modalities(self):
        torch.manual_seed(3)
        text = np.random.default_rng(1).normal(size=(7, 512)).astype(np.float16)
        masks = np.ones((7, 2), dtype=bool); masks[0] = False; masks[6] = False
        text[0] = 0; text[6] = 0
        model = ConcatHybrid(SASRec(6, 3, 8, 2, 1, dropout=0),
                             (text, text.copy(), masks), np.array([0, 1, 1, 1, 0, 1, 0]), 8).eval()
        history, candidates = torch.tensor([[1, 4, 2]]), torch.tensor([[4, 5, 6]])
        before = model.logits(history, candidates)
        with torch.no_grad():
            model.sequential.items.weight[4].fill_(float('nan'))
        after = model.logits(history, candidates)
        torch.testing.assert_close(before, after)
        self.assertTrue(torch.isfinite(after).all())
        self.assertEqual(after[0, 2].item(), 0.)
        after.sum().backward()
        self.assertTrue(torch.isfinite(model.adapter[0].weight.grad).all())
        self.assertFalse(model.text.requires_grad)

    def test_clustered_macro_interval_and_sign(self):
        np.testing.assert_array_equal(ndcg(np.arange(1, 100, dtype=np.uint8)),
                                      ndcg(np.arange(1, 100, dtype=np.int64)))
        regimes = np.tile(np.arange(3), 8)
        users = np.repeat(np.arange(8), 3)
        report = paired_user_ci(np.full(24, .1), users, regimes, 100)
        np.testing.assert_allclose(report['ci95'], [.1, .1])
        same = paired_user_ci(np.zeros(24), users, regimes, 100)
        self.assertEqual(same['ci95'], [0., 0.])
        with self.assertRaises(ValueError):
            paired_user_ci(np.zeros(3), [1, 2, 3], [0, 0, 0], 100)


if __name__ == '__main__':
    unittest.main()
