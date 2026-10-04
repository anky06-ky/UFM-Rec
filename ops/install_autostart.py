"""Enable this FITLAB workspace to start its watchdog when code-server opens it."""
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'UFM: keep campaign running'


def install(root=ROOT):
    if sys.platform != 'linux' or not (root / '.venv_ufm_runtime/bin/python').is_file():
        raise RuntimeError('Run this installer inside the FITLAB project with its persistent Python.')
    config = root / '.vscode'
    config.mkdir(exist_ok=True)
    paths = [config / 'settings.json', config / 'tasks.json']
    # Read both before making changes; preserve existing settings and tasks.
    settings, tasks = [json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
                       for p in paths]
    task = dict(label=LABEL, type='process', command='/usr/bin/bash',
                args=[str(root / 'scripts/start_campaign.sh')],
                options={'cwd': str(root)}, problemMatcher=[],
                runOptions={'runOn': 'folderOpen', 'instanceLimit': 1},
                presentation={'reveal': 'silent', 'panel': 'dedicated', 'clear': False})
    tasks['version'] = '2.0.0'
    tasks['tasks'] = [t for t in tasks.get('tasks', []) if t.get('label') != LABEL] + [task]
    settings['python.defaultInterpreterPath'] = str(root / '.venv_ufm_runtime/bin/python')
    settings.setdefault('files.exclude', {})['.venv_ufm_runtime'] = True
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = root / 'runs' / 'autostart_backups' / stamp
    backup.mkdir(parents=True)
    for path, value in zip(paths, (settings, tasks)):
        if path.exists():
            shutil.copyfile(path, backup / path.name)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    print('AUTOSTART task installed; settings backup:', backup)
    print('Reopen this workspace and approve automatic tasks in code-server when prompted.')


if __name__ == '__main__':
    install()
