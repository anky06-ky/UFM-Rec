from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ops'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from run_followup_suite import plan
from run_ufm_ablation_suite import PARAMETERS
from run_second_dataset_suite import second_plan


class PlanTests(unittest.TestCase):
    def test_unique_runs_fixed_protocol_and_no_test(self):
        base = {key: 1 for key in PARAMETERS}
        base.update(seed=42, amp=True, validation_cap=0, history=20)
        jobs = plan(base)
        self.assertEqual(len(jobs), 11)
        self.assertEqual(len({job['output'] for job in jobs}), 11)
        self.assertEqual([j['seed'] for j in jobs if j['model']=='ufm'], [7, 2026])
        for job in jobs:
            cmd = job['command']
            self.assertEqual(cmd[cmd.index('--validation-cap')+1], '0')
            self.assertEqual(cmd[cmd.index('--history')+1], '20')
            self.assertNotIn('--resume', cmd)
            self.assertFalse(any('samples_test' in word for word in cmd))

    def test_second_dataset_isolated_paths_and_three_seeds(self):
        base = {key:1 for key in PARAMETERS}
        base.update(seed=42,amp=True,validation_cap=0,history=20)
        jobs = second_plan(base)
        self.assertEqual(len(jobs),13)
        self.assertEqual(len({j['output'] for j in jobs}),13)
        for job in jobs:
            self.assertIn('all_beauty',job['output'])
            self.assertFalse(any('toys_games' in word for word in job['command']))
            if job['model']!='clip':
                for flag in ('--data','--legacy-cache','--features'):
                    self.assertIn('all_beauty',job['command'][job['command'].index(flag)+1])
        self.assertEqual({j['seed'] for j in jobs if j['model']=='ufm'},{42,7,2026})


if __name__ == '__main__':
    unittest.main()
