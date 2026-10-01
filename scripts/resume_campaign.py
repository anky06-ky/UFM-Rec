"""Run the FITLAB queues in order and resume recoverable interruptions.

The queue programs enforce source, configuration, checkpoint and data checks.
This supervisor retries worker interruption and GPU contention; a validation
or invariant error remains visible for investigation.
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
                 stage=stage, pid=os.getpid(), **fields)
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def expected_success(name, marker):
    if marker is not None and not (ROOT / marker).is_file():
        return False
    if name == 'foundation':
        source = read(ROOT / marker)
        feature_dir = (ROOT / marker).parent
        return (source.get('rows') == 767045 and source.get('limit') == 0
                and source.get('device') == 'cuda'
                and source.get('encoder_frozen') is True
                and all((feature_dir / filename).is_file() for filename in
                        ('text.npy', 'image.npy', 'modalities.npy')))
    if name == 'ufm':
        source = read(ROOT / marker)
        output = (ROOT / marker).parent
        return (source.get('smoke_run') is False and source.get('test_evaluated') is False
                and (output / 'best.pt').is_file() and (output / 'latest.pt').is_file())
    return read(RUNS / 'ufm_ablation_suite_v1.json').get('stage') == 'suite_complete'


def reconcile_foundation():
    """Repair the handoff if CLIP completed but its queue died before reporting."""
    name, _, stem, marker, _ = STAGES[0]
    if not expected_success(name, marker):
        return False
    state_path = RUNS / (stem + '.json')
    if read(state_path).get('stage') == 'extraction_complete':
        return True
    source = read(ROOT / marker)
    feature_dir = (ROOT / marker).parent
    if not all((feature_dir / filename).is_file() for filename in
               ('text.npy', 'image.npy', 'modalities.npy')):
        return False
    value = dict(stage='extraction_complete',
                 time_utc=datetime.now(timezone.utc).isoformat(),
                 output=str(feature_dir), text=source.get('has_text'),
                 image=source.get('has_image'), recommender_trained=False,
                 test_evaluated=False, recovered_from_complete_marker=True)
    temporary = state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(state_path)
    return True


def retry_reason(state, exit_code):
    """Return a reason only when rerunning the same queue is checkpoint safe."""
    error = str(state.get('error', '')).lower()
    if exit_code in (-9, 137) or any(word in error for word in
                                         ('sigkill', 'exited -9', 'exited 137',
                                          'killed by signal 9')):
        return 'worker_interrupted'
    if 'gpu became busy' in error or 'cuda out of memory' in error:
        return 'gpu_contention'
    if any(word in error for word in ('not ready within wait limit',
                                     'launch window ended', 'baseline/gpu not ready')):
        return 'wait_window_expired'
    if 'nvidia-smi' in error and ('calledprocesserror' in error or 'filenotfounderror' in error):
        return 'gpu_probe_unavailable'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-sigkill-attempts', type=int, default=0,
                        help='0 keeps resuming interrupted workers')
    parser.add_argument('--retry-delay-seconds', type=int, default=120)
    parser.add_argument('--max-transient-hours', type=float, default=0,
                        help='0 keeps waiting for a shared GPU')
    args = parser.parse_args(argv)
    if (args.max_sigkill_attempts < 0 or args.retry_delay_seconds < 0
            or args.max_transient_hours < 0):
        parser.error('Limits and delay cannot be negative.')
    if not (ROOT / 'src/run_foundation_when_idle.py').is_file():
        raise FileNotFoundError(ROOT)
    if not Path(sys.executable).resolve().is_file():
        raise FileNotFoundError(sys.executable)
    RUNS.mkdir(exist_ok=True)
    with (RUNS / 'campaign_recovery_20261001.lock').open('a+b') as master:
        fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, program, stem, marker, extra in STAGES:
            if name == 'foundation':
                reconcile_foundation()
            if expected_success(name, marker):
                report(name + '_already_complete')
                continue
            deadline = (time.monotonic() + args.max_transient_hours * 3600
                        if args.max_transient_hours else None)
            interruptions = attempt = 0
            while True:
                attempt += 1
                with (RUNS / (stem + '.lock')).open('a+b') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    command = [sys.executable, '-u', str(ROOT / 'src' / program), *extra]
                    report(name + '_running', attempt=attempt, command=command)
                    with (RUNS / (stem + '.log')).open('ab') as log:
                        log.write((f'\n=== Recovery {name}, attempt {attempt}, '
                                   f'{datetime.now(timezone.utc).isoformat()} ===\n').encode())
                        log.flush()
                        result = subprocess.run(command, cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT, check=False,
                                                pass_fds=(lock.fileno(),))
                if name == 'foundation':
                    reconcile_foundation()
                if expected_success(name, marker):
                    report(name + '_complete', attempt=attempt)
                    break
                state = read(RUNS / (stem + '.json'))
                reason = retry_reason(state, result.returncode)
                if reason == 'worker_interrupted':
                    interruptions += 1
                    if args.max_sigkill_attempts and interruptions >= args.max_sigkill_attempts:
                        reason = None
                if deadline is not None and time.monotonic() >= deadline:
                    reason = None
                if reason:
                    delay = min(1800, args.retry_delay_seconds * min(attempt, 15))
                    report(name + '_retry_pending', attempt=attempt, reason=reason,
                           interruptions=interruptions, exit_code=result.returncode,
                           delay_seconds=delay, queue_state=state)
                    time.sleep(delay)
                    continue
                report('stopped_with_error', failed_stage=name, attempt=attempt,
                       exit_code=result.returncode, queue_state=state)
                raise RuntimeError(f'{name} queue stopped; see {stem}.log')
        report('campaign_complete', test_evaluated=False)


if __name__ == '__main__':
    main()
