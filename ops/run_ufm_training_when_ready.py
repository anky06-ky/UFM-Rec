"""Prepare CPU graph now; GPU smoke/resume/full UFM after COMPLETE CLIP extraction.

Never kills another job, never consumes a partial feature cache, never tests.
Run under an external flock. Durable output is outside the ephemeral Python venv.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_foundation_when_idle import active_baseline_pids, gpu_idle, atomic_status

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'


def active_ufm_gpu_pids(proc_root=Path('/proc')):
    expected = {str(ROOT/'src'/name).encode() for name in ['extract_foundation_features.py','train_ufm_recommender.py']}
    expected |= {('src/'+name).encode() for name in ['extract_foundation_features.py','train_ufm_recommender.py']}
    result = []
    for p in proc_root.iterdir():
        if p.name.isdigit():
            try:
                argv = (p/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
            if any(value in expected for value in argv) and b'prepare' not in argv:
                result.append(int(p.name))
    return result


def command(output, smoke=False, interrupt=False):
    cmd = [sys.executable,'-u',str(ROOT/'src/train_ufm_recommender.py'), '--output',str(output)]
    if smoke:
        cmd += ['--max-steps','5','--batch-size','32','--validation-cap','400','--checkpoint-every','2']
        if interrupt:
            cmd += ['--interrupt-after-step','2']
    if (output/'latest.pt').exists() and not (output/'completed.json').exists():
        cmd.append('--resume')
    return cmd


def verify_run(output, smoke):
    marker = json.loads((output/'completed.json').read_text())
    audit = json.loads((output/'feature_audit.json').read_text())
    cfg = json.loads((output/'config.json').read_text())
    if (marker['smoke_run'] is not smoke or marker['test_evaluated'] is not False
            or not audit['finite_normalized_masked'] or cfg['variant'] != 'full'
            or cfg['device'] != 'cuda' or audit['items'] != 767045
            or not (output/'best.pt').is_file() or not (output/'latest.pt').is_file()):
        raise ValueError('Run completion/audit/checkpoint invariant failed.')
    if smoke and marker['steps'] != 5:
        raise ValueError('GPU smoke must complete exactly five optimizer steps.')
    if not smoke and marker['steps'] < 1:
        raise ValueError('Production run has no optimizer steps.')
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--max-wait-hours',type=float,default=72)
    parser.add_argument('--status',type=Path,default=ROOT/'runs/ufm_training_queue_v1.json')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_wait_hours <= 0:
        parser.error('Poll 1-60 seconds; positive wait limit required.')
    started = time.monotonic()

    def report(stage, **fields):
        value = dict(stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),
                     test_evaluated=False, **fields)
        atomic_status(args.status,value); print(json.dumps(value),flush=True)

    def run(stage, cmd):
        child = subprocess.Popen(cmd,cwd=ROOT)
        report(stage,child_pid=child.pid,command=cmd)
        while child.poll() is None:
            time.sleep(args.poll_seconds)
            if child.poll() is None:
                report(stage,child_pid=child.pid,alive=True)
        if child.returncode:
            raise RuntimeError(f'{stage} exited {child.returncode}; see persistent queue log.')

    def ready():
        pids = active_baseline_pids() + active_ufm_gpu_pids()
        idle,gpu = gpu_idle()
        return not pids and idle, pids, gpu

    try:
        run('preparing_train_positive_graph_cpu', [sys.executable,'-u',str(ROOT/'src/train_ufm_recommender.py'),'--mode','prepare'])
        checks = 0
        full = DATA/'foundation_clip_b32_v1'
        while True:
            source = json.loads((ROOT/'runs/foundation_queue_v1.json').read_text())
            if source['stage'] == 'stopped_with_error':
                raise RuntimeError('Foundation extraction stopped: '+source.get('error','unknown'))
            complete = (full/'complete.json').is_file() and source['stage'] == 'extraction_complete'
            idle,pids,gpu = ready()
            checks = checks+1 if complete and idle else 0
            progress = {}
            if (full/'progress.json').exists():
                values = json.loads((full/'progress.json').read_text())
                progress = {'committed_rows':values['next_row'],'expected_rows':values['config']['rows']}
            report('waiting_for_complete_features_and_idle_gpu',features_complete=complete,
                   source_stage=source['stage'],active_gpu_job_pids=pids,gpu=gpu,idle_checks=checks, **progress)
            if checks >= 2:
                break
            if time.monotonic()-started >= args.max_wait_hours*3600:
                raise TimeoutError('Features/GPU not ready within wait limit; no UFM GPU job started.')
            time.sleep(args.poll_seconds)
        smoke = ROOT/'runs/ufm_gpu_smoke_v1'
        if not (smoke/'completed.json').exists():
            if not (smoke/'latest.pt').exists():
                run('gpu_smoke_controlled_interrupt', command(smoke,smoke=True,interrupt=True))
                if (smoke/'completed.json').exists() or not (smoke/'latest.pt').exists():
                    raise ValueError('Smoke interrupt did not leave an unfinished checkpoint.')
            run('gpu_smoke_resume',command(smoke,smoke=True))
        verified = verify_run(smoke,True)
        report('gpu_smoke_and_resume_passed',steps=verified['steps'],output=str(smoke))
        if not ready()[0]:
            raise RuntimeError('GPU became busy; smoke preserved, full run not started.')
        output = ROOT/'runs/ufm_full_v1'
        if not (output/'completed.json').exists():
            run('full_ufm_training_running',command(output))
        verified = verify_run(output,False)
        report('full_ufm_training_complete',steps=verified['steps'],output=str(output),
               best_cold_macro_ndcg_at_10=verified['best_cold_macro_ndcg@10'])
    except Exception as error:
        report('stopped_with_error',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
