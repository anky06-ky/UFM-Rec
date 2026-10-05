import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, str(ROOT))
import install_autostart
import status


class StartupTests(unittest.TestCase):
    def test_install_preserves_settings_tasks_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv_ufm_runtime/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            config = root / '.vscode'
            config.mkdir()
            original = {'editor.fontSize': 17, 'files.exclude': {'tmp': True}}
            (config / 'settings.json').write_text(json.dumps(original))
            (config / 'tasks.json').write_text(json.dumps({'version': '2.0.0', 'tasks': [
                {'label': 'existing task', 'type': 'shell', 'command': 'echo hello'}]}))
            with patch.object(install_autostart.sys, 'platform', 'linux'):
                install_autostart.install(root)
                install_autostart.install(root)
            settings = json.loads((config / 'settings.json').read_text())
            tasks = json.loads((config / 'tasks.json').read_text())['tasks']
            self.assertEqual(settings['editor.fontSize'], 17)
            self.assertTrue(settings['files.exclude']['tmp'])
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]['label'], 'existing task')
            self.assertEqual(tasks[1]['runOptions']['runOn'], 'folderOpen')
            backups = list((root / 'runs/autostart_backups').glob('*/settings.json'))
            self.assertTrue(any(json.loads(p.read_text()) == original for p in backups))

    def test_status_does_not_treat_reused_pid_as_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            child = proc / '123'
            child.mkdir()
            (child / 'cmdline').write_bytes(b'python\0unrelated.py\0')
            self.assertFalse(status.process_alive(123, 'ops/resume_campaign.py', ROOT, proc))
            (child / 'cmdline').write_bytes(b'python\0-u\0ops/resume_campaign.py\0')
            self.assertTrue(status.process_alive(123, 'ops/resume_campaign.py', ROOT, proc))


if __name__ == '__main__':
    unittest.main()
