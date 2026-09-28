import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_ufm_training_when_ready as queue


class QueueTests(unittest.TestCase):
    def test_exact_argv_ignores_wrapper_and_cpu_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            proc = Path(d)
            commands = {1:b'python\0src/extract_foundation_features.py\0',
                        2:b'python\0src/train_ufm_recommender.py\0--mode\0prepare\0',
                        3:b'bash\0-c\0python src/train_ufm_recommender.py\0',
                        4:b'python\0src/train_ufm_recommender.py\0--resume\0'}
            for pid,value in commands.items():
                (proc/str(pid)).mkdir(); (proc/str(pid)/'cmdline').write_bytes(value)
            self.assertEqual(sorted(queue.active_ufm_gpu_pids(proc)),[1,4])

    def test_smoke_command_resume_preserves_config(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            first = queue.command(output,smoke=True,interrupt=True)
            self.assertIn('--interrupt-after-step',first)
            self.assertNotIn('--resume',first)
            (output/'latest.pt').touch()
            second = queue.command(output,smoke=True)
            self.assertIn('--resume',second)
            self.assertNotIn('--interrupt-after-step',second)
            self.assertEqual(first[:-2],second[:-1])

    def test_verify_rejects_smoke_as_production(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            (output/'completed.json').write_text(json.dumps({'smoke_run':True,'steps':5,'test_evaluated':False}))
            (output/'feature_audit.json').write_text(json.dumps({'finite_normalized_masked':True,'items':767045}))
            (output/'config.json').write_text(json.dumps({'variant':'full','device':'cuda'}))
            for name in ['best.pt','latest.pt']: (output/name).touch()
            self.assertEqual(queue.verify_run(output,True)['steps'],5)
            with self.assertRaises(ValueError): queue.verify_run(output,False)


if __name__ == '__main__': unittest.main()
