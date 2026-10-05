from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('sasrec', SOURCE / 'train_sasrec.py')
sasrec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sasrec)


class SASRecTests(unittest.TestCase):
    def test_padding_unknown_and_negative_exclusion(self):
        torch.manual_seed(4)
        model = sasrec.SASRec(6, 3, dim=8, heads=2, layers=1, dropout=0)
        model.eval()
        history = torch.tensor([[1, 2, 0]])
        with torch.no_grad():
            before = model.encode(history).clone()
            model.items.weight[0].fill_(100)
            after = model.encode(history)
        torch.testing.assert_close(before, after)
        with torch.no_grad():
            earlier = model.encode_tokens(torch.tensor([[1, 2, 3]]))[0][:, :2]
            later_changed = model.encode_tokens(torch.tensor([[1, 2, 4]]))[0][:, :2]
        torch.testing.assert_close(earlier, later_changed)
        negatives = sasrec.sample_negatives(
            np.arange(1, 7), np.array([10, 20]), np.array([0, 2, 4]),
            np.array([1, 2, 2, 3]), np.array([10, 20]), np.array([2, 3]),
            np.random.default_rng(9), 2)
        self.assertTrue(set(negatives[0]).isdisjoint({1, 2}))
        self.assertTrue(set(negatives[1]).isdisjoint({2, 3}))
        with self.assertRaises(ValueError):
            sasrec.sample_negatives(np.arange(1, 4), np.array([10]),
                                    np.array([0, 2]), np.array([1, 2]),
                                    np.array([10]), np.array([2]),
                                    np.random.default_rng(1), 2)

    def test_cpu_smoke_uses_fixed_validation_and_does_not_open_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, legacy, graph, output = [root / name for name in
                                           ('data', 'legacy', 'graph', 'run')]
            for folder in (data / 'evaluation', legacy, graph):
                folder.mkdir(parents=True, exist_ok=True)
            arrays = {
                'histories': [[1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 0, 0]],
                'targets': [2, 1, 3, 3], 'users': [10, 10, 20, 20],
                'train_counts': [0, 1, 2, 1, 1, 1, 1],
            }
            for name, value in arrays.items():
                np.save(legacy / (name + '.npy'),
                        np.asarray(value, dtype=np.int64 if name == 'users' else np.int32))
            source = {'legacy_source': 'fixture'}
            for name in arrays:
                source[name + '.npy'] = sasrec.sha256(legacy / (name + '.npy'))
            (legacy / 'complete.json').write_text(json.dumps(dict(
                history_size=3, rows=4, items=6, fingerprint='fixture')))
            graph_arrays = {'user_keys': [10, 20], 'positive_indptr': [0, 2, 4],
                            'positive_items': [1, 2, 2, 3]}
            for name, value in graph_arrays.items():
                np.save(graph / (name + '.npy'), np.asarray(value, dtype=np.int64))
            graph_hash = {name + '.npy': sasrec.sha256(graph / (name + '.npy'))
                          for name in graph_arrays}
            (graph / 'complete.json').write_text(json.dumps(dict(
                history_size=3, rows=4, items=6, sources=source,
                graph_sha256=graph_hash)))
            np.savez(data / 'evaluation/samples_validation.npz',
                     candidates=np.array([[1, 3, 4], [0, 3, 4], [2, 3, 4], [2, 3, 5]]),
                     history_indptr=np.array([0, 1, 2, 3, 4]),
                     history_indices=np.array([0, 1, 1, 2]),
                     regime_codes=np.array([0, 1, 2, 3]))
            args = Namespace(data=data, legacy_cache=legacy, graph=graph,
                             output=output, device='cpu', history=3, dim=8,
                             heads=2, layers=1, batch_size=2, negatives=1,
                             epochs=2, lr=0.001, patience=2, checkpoint_every=1,
                             eval_batch=2, validation_cap=0, max_steps=2,
                             threads=1, seed=42, resume=False)
            done = sasrec.train(args)
            self.assertEqual(done['steps'], 2)
            self.assertTrue(done['smoke_run'])
            self.assertFalse(done['test_evaluated'])
            self.assertTrue((output / 'best.pt').is_file())
            self.assertFalse((output / 'test_metrics.json').exists())
            with self.assertRaises(ValueError):
                sasrec.train(Namespace(**{**vars(args), 'resume': True}))
            (output / 'completed.json').unlink()
            resumed = sasrec.train(Namespace(**{**vars(args), 'resume': True}))
            self.assertEqual(resumed['steps'], done['steps'])
            self.assertEqual(resumed['best_cold_macro_ndcg_at_10'],
                             done['best_cold_macro_ndcg_at_10'])
            bert = sasrec.train(Namespace(**{**vars(args), 'model': 'bert4rec',
                                            'output': root / 'bert'}))
            self.assertEqual(bert['steps'], 2)
            self.assertFalse(bert['test_evaluated'])
            self.assertTrue((root / 'bert/best_validation_predictions.npz').is_file())


if __name__ == '__main__':
    unittest.main()
