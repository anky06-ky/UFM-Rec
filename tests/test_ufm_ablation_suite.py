import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_ufm_ablation_suite as suite


class AblationTests(unittest.TestCase):
    def base(self):
        base = {k:1 for k in suite.PARAMETERS}
        base.update(variant='full',max_steps=0,amp=True,device='cuda',cache={},
                    feature_marker_sha256='features',code_sha256={},selection='cold macro',versions={})
        return base

    def test_each_variant_fresh_same_budget_then_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for variant in suite.VARIANTS:
                cmd = suite.experiment_command(self.base(),variant,out)
                self.assertEqual(cmd[cmd.index('--variant')+1],variant)
                self.assertNotIn('--resume',cmd)
                self.assertNotIn('best.pt',' '.join(cmd))
                self.assertNotIn('--max-steps',cmd)
                self.assertIn('--amp',cmd)
            (out/'latest.pt').touch()
            self.assertIn('--resume',suite.experiment_command(self.base(),suite.VARIANTS[0],out))

    def test_reject_smoke_parent(self):
        base = self.base(); base['max_steps'] = 5
        with self.assertRaises(ValueError): suite.experiment_command(base,suite.VARIANTS[0],Path('unused'))

    def test_result_protocol_must_match_and_test_not_opened(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); base = self.base(); variant = suite.VARIANTS[0]
            cfg = dict(base,variant=variant)
            marker = dict(smoke_run=False,test_evaluated=False,steps=10)
            marker['best_cold_macro_ndcg@10'] = 0.1
            (out/'completed.json').write_text(json.dumps(marker))
            (out/'config.json').write_text(json.dumps(cfg))
            for name in ['best.pt','latest.pt']: (out/name).touch()
            self.assertEqual(suite.validate_result(out,variant,base)['steps'],10)
            cfg['seed'] = 2; (out/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'seed'): suite.validate_result(out,variant,base)


if __name__ == '__main__': unittest.main()
