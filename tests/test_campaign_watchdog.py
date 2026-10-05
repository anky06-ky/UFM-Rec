"""Checkpoint handoff and duplicate-worker guards for FITLAB recovery."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))


@unittest.skipUnless(os.name == 'posix', 'FITLAB recovery uses Linux flock and procfs')
class RecoveryTests(unittest.TestCase):
    def test_interruption_retries_but_corrupt_source_stops(self):
        from resume_campaign import retry_reason
        self.assertEqual(retry_reason({'error': 'CalledProcessError: died with SIGKILL'}, 1),
                         'worker_interrupted')
        self.assertEqual(retry_reason({'error': 'full_ufm_training_running exited -9'}, 1),
                         'worker_interrupted')
        self.assertEqual(retry_reason({}, -9), 'worker_interrupted')
        self.assertEqual(retry_reason({'error': 'GPU became busy; smoke preserved'}, 1),
                         'gpu_contention')
        self.assertEqual(retry_reason({'error': 'TimeoutError: Launch window ended'}, 1),
                         'wait_window_expired')
        self.assertIsNone(retry_reason({'error': 'ValueError: checksum mismatch'}, 1))

    def test_completed_clip_repairs_queue_handoff(self):
        import resume_campaign as campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / 'runs'
            runs.mkdir()
            cache = root / campaign.STAGES[0][3]
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(dict(rows=767045, limit=0, device='cuda',
                                             encoder_frozen=True, has_text=760000,
                                             has_image=740000)))
            for name in ('text.npy', 'image.npy', 'modalities.npy'):
                (cache.parent / name).touch()
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'RUNS', runs):
                self.assertTrue(campaign.reconcile_foundation())
                status = json.loads((runs / 'foundation_queue_v1.json').read_text())
                self.assertEqual(status['stage'], 'extraction_complete')
                self.assertTrue(status['recovered_from_complete_marker'])
                (cache.parent / 'image.npy').unlink()
                self.assertFalse(campaign.expected_success('foundation', campaign.STAGES[0][3]))

    def test_proc_scan_matches_only_this_project(self):
        import watch_campaign as watcher
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            worker = proc / '999991'
            worker.mkdir()
            (worker / 'cmdline').write_bytes(b'python\0-u\0src/extract_foundation_features.py\0')
            (worker / 'cwd').symlink_to(watcher.ROOT, target_is_directory=True)
            other = proc / '999992'
            other.mkdir()
            (other / 'cmdline').write_bytes(b'python\0-u\0src/extract_foundation_features.py\0')
            (other / 'cwd').symlink_to(proc, target_is_directory=True)
            self.assertEqual(watcher.live_jobs(proc), [{'pid': 999991, 'role': 'feature_worker'}])


if __name__ == '__main__':
    unittest.main()
