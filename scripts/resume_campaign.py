"""Resume the FITLAB campaign in order, using the existing queue programs.

Run with the isolated UFM Python from the project root. The supervisor keeps
one flock per queue and retries only a SIGKILL on an unfinished stage. All
model/config/source validation remains inside the queue and trainer programs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
STATUS = RUNS / 'campaign_recovery_20261001.json'
STAGES = (
    ('foundation', 'run_foundation_when_idle.py', 'foundation_queue_v1',
     'data/processed/toys_games_full_temporal/foundation_clip_b32_v1/complete.json',
     ['--max-wait-hours', '168']),
    ('ufm', 'run_ufm_training_when_ready.py', 'ufm_training_queue_v1',
     'runs/ufm_full_v1/completed.json', ['--max-wait-hours', '168']),
    ('ablations', 'run_ufm_ablation_suite.py', 'ufm_ablation_suite_v1',
     None, ['--max-days', '7']),
)


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def report(stage, **fields):
    value = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                 stage=stage, **fields)
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def expected_success(name, marker):
    if marker is not None and not (ROOT / marker).is_file():
        return False
    if name == 'foundation':
        source = read(ROOT / marker)
        return source.get('rows') == 767045 and source.get('limit') == 0
    if name == 'ufm':
        source = read(ROOT / marker)
        return source.get('smoke_run') is False and source.get('test_evaluated') is False
    return read(RUNS / 'ufm_ablation_suite_v1.json').get('stage') == 'suite_complete'


def may_retry(state, marker_complete, attempt, limit):
    return (attempt < limit and not marker_complete
            and state.get('stage') == 'stopped_with_error'
            and 'SIGKILL' in state.get('error', ''))


def gpu_race(state):
    error = state.get('error', '') if state.get('stage') == 'stopped_with_error' else ''
    return ('GPU became busy; smoke kept, full extraction not started.' in error
            or 'GPU became busy; smoke preserved, full run not started.' in error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-sigkill-attempts', type=int, default=3)
    parser.add_argument('--retry-delay-seconds', type=int, default=120)
    parser.add_argument('--max-transient-hours', type=float, default=168)
    args = parser.parse_args(argv)
    if (args.max_sigkill_attempts < 1 or args.retry_delay_seconds < 0
            or args.max_transient_hours <= 0):
        parser.error('Attempts and transient window must be positive; delay nonnegative.')
    if not (ROOT / 'src/run_foundation_when_idle.py').is_file():
        raise FileNotFoundError(ROOT)
    if not Path(sys.executable).resolve().is_file():
        raise FileNotFoundError(sys.executable)
    RUNS.mkdir(exist_ok=True)
    with (RUNS / 'campaign_recovery_20261001.lock').open('a+b') as master:
        fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, program, stem, marker, extra in STAGES:
            if expected_success(name, marker):
                report(name + '_already_complete')
                continue
            deadline = time.monotonic() + args.max_transient_hours * 3600
            sigkills = attempt = 0
            while True:
                attempt += 1
                with (RUNS / (stem + '.lock')).open('a+b') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    command = [sys.executable, '-u', str(ROOT / 'src' / program), *extra]
                    report(name + '_running', attempt=attempt, pid=os.getpid(), command=command)
                    with (RUNS / (stem + '.log')).open('ab') as log:
                        log.write((f'\n=== Recovery {name}, attempt {attempt}, '
                                   f'{datetime.now(timezone.utc).isoformat()} ===\n').encode())
                        log.flush()
                        result = subprocess.run(command, cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and expected_success(name, marker):
                    report(name + '_complete', attempt=attempt)
                    break
                state = read(RUNS / (stem + '.json'))
                complete = expected_success(name, marker)
                if gpu_race(state) and time.monotonic() < deadline:
                    report(name + '_gpu_busy_retry_pending', attempt=attempt,
                           exit_code=result.returncode, delay_seconds=args.retry_delay_seconds)
                    time.sleep(args.retry_delay_seconds)
                    continue
                if may_retry(state, complete, sigkills + 1, args.max_sigkill_attempts):
                    sigkills += 1
                    report(name + '_sigkill_retry_pending', attempt=attempt,
                           sigkills=sigkills, exit_code=result.returncode,
                           delay_seconds=args.retry_delay_seconds)
                    time.sleep(args.retry_delay_seconds)
                    continue
                report('stopped_with_error', failed_stage=name, attempt=attempt,
                       exit_code=result.returncode, queue_state=state)
                raise RuntimeError(f'{name} queue stopped; see {stem}.log')
        report('campaign_complete', test_evaluated=False)


if __name__ == '__main__':
    main()
