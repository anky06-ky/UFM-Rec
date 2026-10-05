"""Keep one FITLAB campaign supervisor alive inside the current container.

Run detached with system Python. The watchdog waits for orphaned GPU workers
before restarting a supervisor, so a recovery never launches duplicate jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resume_campaign import ROOT, RUNS, STATUS, read, reconcile_foundation, retry_reason

WATCH_STATUS = RUNS / 'campaign_watchdog_v1.json'
SCRIPTS = {
    ROOT / 'ops/resume_campaign.py': 'supervisor',
    ROOT / 'src/run_foundation_when_idle.py': 'foundation_queue',
    ROOT / 'src/extract_foundation_features.py': 'feature_worker',
    ROOT / 'src/run_ufm_training_when_ready.py': 'ufm_queue',
    ROOT / 'src/run_ufm_ablation_suite.py': 'ablation_queue',
    ROOT / 'src/train_ufm_recommender.py': 'trainer',
}


def report(stage, **fields):
    value = dict(stage=stage, time_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), runtime=runtime_health(), **fields)
    temporary = WATCH_STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(WATCH_STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def runtime_health():
    """Persist the container identity and memory counters before another restart."""
    result = {}
    try:
        result['pid1_start_ticks'] = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19]
        for name in ('memory.current', 'memory.max', 'memory.events'):
            path = Path('/sys/fs/cgroup') / name
            if path.is_file():
                result[name] = path.read_text().strip()
    except OSError as error:
        result['read_error'] = str(error)
    return result


def live_jobs(proc_root=Path('/proc')):
    jobs = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            cwd = (entry / 'cwd').resolve()
        except (OSError, RuntimeError):
            continue
        for argument in argv:
            if not argument.endswith(b'.py'):
                continue
            path = Path(os.fsdecode(argument))
            resolved = (path if path.is_absolute() else cwd / path).resolve()
            if resolved in SCRIPTS:
                jobs.append(dict(pid=int(entry.name), role=SCRIPTS[resolved]))
                break
    return jobs


def python_for_campaign(explicit):
    candidates = [explicit] if explicit else [ROOT / '.venv_ufm_runtime/bin/python',
                                               Path('/tmp/ufm_venv_20260928_cdcp1xhw/bin/python')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def fatal_stop():
    state = read(STATUS)
    if state.get('stage') != 'stopped_with_error':
        return False
    queue = state.get('queue_state', {})
    return retry_reason(queue, state.get('exit_code')) is None


def main(argv=None):
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--python', type=Path)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error('poll-seconds must be at least 5')
    RUNS.mkdir(exist_ok=True)
    with (RUNS / 'campaign_watchdog_v1.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Watchdog already running; no duplicate started.', flush=True)
            return
        while True:
            jobs = live_jobs()
            roles = {job['role'] for job in jobs}
            if 'foundation_queue' not in roles and 'feature_worker' not in roles:
                reconcile_foundation()
            if read(STATUS).get('stage') == 'campaign_complete':
                report('campaign_complete')
                return
            if 'supervisor' in roles:
                report('supervisor_active', jobs=jobs)
            elif jobs:
                report('waiting_for_orphaned_worker', jobs=jobs)
            elif fatal_stop():
                report('needs_attention', recovery=read(STATUS))
            else:
                python = python_for_campaign(args.python)
                if python is None:
                    report('python_missing', searched=['.venv_ufm_runtime/bin/python',
                                                       '/tmp/ufm_venv_20260928_cdcp1xhw/bin/python'])
                else:
                    command = [python, '-u', str(ROOT / 'ops/resume_campaign.py')]
                    with (RUNS / 'campaign_recovery_20261001.log').open('ab') as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                    report('supervisor_started', child_pid=child.pid, command=command)
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
