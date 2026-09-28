"""Read-only queue decisions; no waiting, subprocess launching or GPU needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_foundation_when_idle as queue


class QueueTests(unittest.TestCase):
    def test_process_match_uses_argv_not_shell_wrapper_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, command in [('11', b'/env/bin/python\0-u\0src/train_gpu_recommender.py\0'),
                                 ('12', b'bash\0-c\0python -u src/train_gpu_recommender.py\0'),
                                 ('13', b'python\0-u\0src/extract_foundation_features.py\0')]:
                (root / pid).mkdir()
                (root / pid / 'cmdline').write_bytes(command)
            self.assertEqual(queue.active_baseline_pids(root), [11])

    def test_full_baseline_completion_requires_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'runs/content_transformer_v1'
            path.mkdir(parents=True)
            self.assertFalse(queue.baseline_finished(root))
            (path / 'completed.json').write_text(json.dumps({'smoke_run': True}))
            with self.assertRaises(ValueError):
                queue.baseline_finished(root)
            (path / 'completed.json').write_text(json.dumps({'smoke_run': False}))
            with self.assertRaises(ValueError):
                queue.baseline_finished(root)
            (path / 'best.pt').write_bytes(b'test-only')
            self.assertTrue(queue.baseline_finished(root))

    def test_only_matching_unfinished_extraction_is_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = queue.extraction_command(output, 0, 64)
            self.assertNotIn('--resume', command)
            (output / 'progress.json').write_text('{}')
            self.assertIn('--resume', queue.extraction_command(output, 0, 64))
            (output / 'complete.json').write_text('{}')
            self.assertNotIn('--resume', queue.extraction_command(output, 0, 64))

    def test_gpu_thresholds_reject_busy_or_low_memory_device(self):
        with patch.object(queue.subprocess, 'run') as runner:
            runner.return_value.stdout = '18000, 0\n'
            self.assertTrue(queue.gpu_idle()[0])
            runner.return_value.stdout = '18000, 75\n'
            self.assertFalse(queue.gpu_idle()[0])
            runner.return_value.stdout = '4000, 0\n'
            self.assertFalse(queue.gpu_idle()[0])


if __name__ == '__main__':
    unittest.main()
