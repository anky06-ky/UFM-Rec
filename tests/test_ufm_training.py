"""Synthetic CPU integration checks; NOT experiment quality measurements."""
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import train_ufm_recommender as trainer
from prepare_ufm_catalog import sha256
from train_gpu_recommender import fingerprint


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ufm_training_test_')
        self.root = Path(self.temporary.name)
        self.data, self.legacy, self.features = (self.root / n for n in ['data', 'legacy', 'features'])
        for p in [self.data / 'content', self.data / 'evaluation', self.data / 'foundation_catalog_v1', self.legacy, self.features]:
            p.mkdir(parents=True)
        raw = [('alice', i) for i in [1, 2, 3] + [7]*6 + [8]*21] + [('bob', i) for i in [4,5,6,9,10]]
        counts = np.bincount([i for _, i in raw], minlength=13).astype(np.int64)
        def gz_csv(path, fields, rows):
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
                writer = csv.writer(f); writer.writerow(fields); writer.writerows(rows)
        gz_csv(self.data / 'interactions_train.csv.gz', ['user_id','parent_asin','rating','timestamp'],
               [(u, 'ASIN'+str(i), 5, 1000+j) for j, (u,i) in enumerate(raw)])
        gz_csv(self.data / 'content/items.csv.gz', ['row_index','parent_asin','train_count'],
               [(i-1, 'ASIN'+str(i), counts[i]) for i in range(1,13)])
        gz_csv(self.data / 'content/products_text.csv.gz', ['row_index','text'], [(i, 'synthetic '+str(i)) for i in range(12)])
        (self.data / 'content/tfidf_all.npz').write_bytes(b'synthetic fingerprint placeholder')
        (self.data / 'train_with_history.csv.gz').write_bytes(b'synthetic fingerprint placeholder')
        h = np.array([[1,0,0],[1,2,0],[1,2,3],[7,8,0],[4,0,0],[4,5,0],[4,5,6],[9,10,0]], dtype=np.int32)
        t = np.array([2,3,7,8,5,6,9,10], dtype=np.int32)
        u = np.array([trainer.user_key('alice')]*4 + [trainer.user_key('bob')]*4, dtype=np.int64)
        for name, values in [('histories',h),('targets',t),('users',u),('train_counts',counts)]:
            np.save(self.legacy / (name+'.npy'), values)
        manifest = {'history_rows':{'train':8}, 'interactions':{'train':len(raw)}, 'validation_start_utc':'2021-01-01T00:00:00+00:00'}
        (self.data / 'split_manifest.json').write_text(json.dumps(manifest))
        targets = [10,0,6,7]
        candidates = np.array([[i] + [j for j in range(12) if j != i] for i in targets], dtype=np.int32)
        np.savez(self.data / 'evaluation/samples_validation.npz', candidates=candidates,
                 history_indices=np.array([0,1,0,1,0,1,0,1], dtype=np.int32),
                 history_indptr=np.arange(0,9,2, dtype=np.int64), regime_codes=np.arange(4,dtype=np.int8))
        # No test file exists: integration must work without reading one.
        (self.legacy / 'complete.json').write_text(json.dumps({'fingerprint':fingerprint(self.data), 'history_size':3, 'rows':8, 'items':12}))
        catalog = {'catalog_sha256':'synthetic_catalog', 'sources_sha256':{'items':sha256(self.data / 'content/items.csv.gz')}}
        (self.data / 'foundation_catalog_v1/catalog_report.json').write_text(json.dumps(catalog))
        rng = np.random.default_rng(7)
        for name in ['text','image']:
            x = rng.normal(size=(13,512)).astype(np.float32)
            x /= np.linalg.norm(x,axis=1,keepdims=True); x[0] = 0
            np.save(self.features / (name+'.npy'), x.astype(np.float16))
        flags = np.ones((13,2), dtype=np.bool_); flags[0] = False
        np.save(self.features / 'modalities.npy', flags)
        marker = {'rows':12, 'limit':0, 'encoder_frozen':True, 'revision':trainer.REVISION,
                  'model':'openai/clip-vit-base-patch32', 'feature_dim':512, 'padding_row':0,
                  'catalog_sha256':'synthetic_catalog', 'text_sha256':sha256(self.data / 'content/products_text.csv.gz'),
                  'has_text':12, 'has_image':12, 'features_sha256':{n:sha256(self.features / n) for n in ['text.npy','image.npy','modalities.npy']}}
        (self.features / 'complete.json').write_text(json.dumps(marker))

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, output='run', *extra):
        return trainer.parse_args(['--data',str(self.data),'--legacy-cache',str(self.legacy),
            '--cache',str(self.root / 'graph'),'--features',str(self.features),'--output',str(self.root / output),
            '--device','cpu','--max-steps','3','--batch-size','2','--history','3','--dim','16','--heads','4',
            '--layers','1','--negatives','2','--threads','1','--epochs','2','--eval-batch','2', *extra])

    def run_train(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(args)

    def test_graph_and_negative_exclusion(self):
        before = {p.name:sha256(p) for p in self.legacy.iterdir()}
        args = self.args(); meta = trainer.prepare(args)
        self.assertEqual(meta['edges'], 10)
        keys,ptr,items = (np.load(args.cache / n) for n in ['user_keys.npy','positive_indptr.npy','positive_items.npy'])
        neg,valid = trainer.sample_negatives(np.arange(1,11), keys,ptr,items,
            np.array([trainer.user_key('alice')]), np.array([2]), np.array([[1,0,0]]), np.random.default_rng(2), 8)
        self.assertEqual(int(valid.sum()),5)
        self.assertTrue(set(neg[valid]) == {4,5,6,9,10})
        self.assertTrue(np.all(neg[~valid] == 2))
        self.assertEqual(before, {p.name:sha256(p) for p in self.legacy.iterdir()})

    def test_exact_cpu_resume(self):
        interrupted = self.args('resumed','--interrupt-after-step','2')
        self.run_train(interrupted)
        self.assertFalse((interrupted.output / 'completed.json').exists())
        self.run_train(self.args('resumed','--resume'))
        self.run_train(self.args('fresh'))
        resumed = torch.load(self.root/'resumed/latest.pt', weights_only=True)
        fresh = torch.load(self.root/'fresh/latest.pt', weights_only=True)
        self.assertEqual(resumed['global_step'],3)
        for key in fresh['model']:
            torch.testing.assert_close(fresh['model'][key], resumed['model'][key], rtol=0, atol=0)
        a,b = (json.loads((self.root/n/'epoch_001.json').read_text()) for n in ['resumed','fresh'])
        self.assertEqual(a,b)
        self.assertEqual(a['loss_stats']['examples'],6)
        self.assertTrue(a['partial_epoch'])
        self.assertEqual(len((self.root/'resumed/history.jsonl').read_text().splitlines()),1)
        self.assertFalse(json.loads((self.root/'resumed/completed.json').read_text())['test_evaluated'])

    def test_nonfinite_gradient_skips_batch_without_corrupting_run(self):
        clip = torch.nn.utils.clip_grad_norm_
        calls = 0

        def one_overflow(*args, **kwargs):
            nonlocal calls
            calls += 1
            return torch.tensor(float('inf')) if calls == 1 else clip(*args, **kwargs)

        with patch.object(trainer.torch.nn.utils, 'clip_grad_norm_', side_effect=one_overflow):
            self.run_train(self.args('overflow'))
        record = json.loads((self.root/'overflow/epoch_001.json').read_text())
        self.assertEqual(record['step'], 3)
        self.assertEqual(record['loss_stats']['skipped'], 2)
        self.assertTrue((self.root/'overflow/completed.json').exists())

    def test_resume_at_last_batch_and_max_step(self):
        self.run_train(self.args('boundary','--max-steps','4','--interrupt-after-step','4'))
        self.run_train(self.args('boundary','--max-steps','4','--resume'))
        result = json.loads((self.root/'boundary/epoch_001.json').read_text())
        self.assertEqual(result['step'],4)
        self.assertFalse(result['partial_epoch'])
        self.assertEqual(result['loss_stats']['examples'],8)

    def test_reject_feature_tamper_and_partial(self):
        args = self.args()
        markerpath = self.features/'complete.json'
        marker = json.loads(markerpath.read_text()); marker['limit'] = 12
        markerpath.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError,'Full, frozen'):
            trainer.feature_audit(args,12)
        marker['limit'] = 0; markerpath.write_text(json.dumps(marker))
        with (self.features/'text.npy').open('ab') as f: f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'checksum'):
            trainer.feature_audit(args,12)

    def test_reject_resume_change_and_overwrite(self):
        self.run_train(self.args('run','--interrupt-after-step','2'))
        with self.assertRaisesRegex(ValueError,'Resume config'):
            self.run_train(self.args('run','--resume','--lr','0.001'))
        with self.assertRaises(FileExistsError):
            self.run_train(self.args('run'))


if __name__ == '__main__':
    unittest.main()
