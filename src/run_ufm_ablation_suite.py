"""Seven fresh ablation runs after full UFM, single GPU, validation only.

Bounded 7-day launch window; ongoing work finishes without forced termination.
Existing extraction/full queue is left alone. No checkpoint is borrowed from
the full model. All hyperparameters and selection protocol are inherited.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_foundation_when_idle import ROOT, active_baseline_pids, gpu_idle, atomic_status
from run_ufm_training_when_ready import active_ufm_gpu_pids, verify_run

VARIANTS = ('no_uncertainty','fixed_fusion','no_cross_align','semantic_only',
            'collaborative_only','text_only','image_only')
PARAMETERS = ('data','legacy_cache','features','epochs','batch_size','dim','history','heads','layers',
              'negatives','patience','checkpoint_every','eval_batch','validation_cap','threads','seed',
              'lr','weight_decay','dropout','id_dropout','fixed_alpha','lambda_align','lambda_unc','lambda_cal')


def experiment_command(base,variant,output):
    if variant not in VARIANTS or base['variant'] != 'full' or base['max_steps'] != 0:
        raise ValueError('Require a full production parent and a predefined ablation.')
    cmd = [sys.executable,'-u',str(ROOT/'src/train_ufm_recommender.py'),
           '--variant',variant,'--output',str(output),'--cache',str(ROOT/'data/processed/ufm_positive_graph_h20_v1')]
    for key in PARAMETERS:
        cmd += ['--'+key.replace('_','-'),str(base[key])]
    cmd += ['--amp' if base['amp'] else '--no-amp']
    if (output/'latest.pt').exists() and not (output/'completed.json').exists():
        cmd += ['--resume']
    return cmd


def validate_result(output,variant,base):
    done = json.loads((output/'completed.json').read_text())
    config = json.loads((output/'config.json').read_text())
    if (done['smoke_run'] is not False or done['test_evaluated'] is not False or done['steps'] < 1
            or config['variant'] != variant or config['device'] != 'cuda' or config['max_steps'] != 0
            or not (output/'best.pt').is_file() or not (output/'latest.pt').is_file()):
        raise ValueError('Invalid ablation completion/checkpoint.')
    for key in (*PARAMETERS,'amp','cache','feature_marker_sha256','code_sha256','selection','versions'):
        if config[key] != base[key]:
            raise ValueError('Ablation protocol/source mismatch: '+key)
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poll-seconds',type=float,default=30)
    p.add_argument('--max-days',type=float,default=7)
    args = p.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_days <= 0:
        p.error('Positive day limit; polling 1-60 seconds.')
    status = ROOT/'runs/ufm_ablation_suite_v1.json'
    full = ROOT/'runs/ufm_full_v1'
    deadline = time.monotonic()+args.max_days*86400
    completed = []

    def report(stage,**fields):
        state = dict(stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),
                     completed_variants=completed,test_evaluated=False,**fields)
        atomic_status(status,state); print(json.dumps(state),flush=True)

    def wait_ready(stage,need_full=False):
        checks = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Launch window ended; no running job killed or results deleted.')
            parent_complete = (full/'completed.json').exists()
            queue = ROOT/'runs/ufm_training_queue_v1.json'
            if need_full and queue.exists() and not parent_complete:
                parent = json.loads(queue.read_text())
                if parent['stage'] == 'stopped_with_error':
                    raise RuntimeError('Parent UFM pipeline stopped: '+parent.get('error','unknown'))
            pids = active_baseline_pids()+active_ufm_gpu_pids()
            idle,gpu = gpu_idle()
            ready = idle and not pids and (parent_complete or not need_full)
            checks = checks+1 if ready else 0
            report(stage,parent_complete=parent_complete,active_gpu_job_pids=pids,gpu=gpu,idle_checks=checks)
            if checks >= 2: return
            time.sleep(args.poll_seconds)

    try:
        wait_ready('waiting_for_full_ufm_and_idle_gpu',True)
        verify_run(full,False)
        base = json.loads((full/'config.json').read_text())
        if base['seed'] != 42 or base['validation_cap'] != 0:
            raise ValueError('Expected predeclared seed42/all-validation main run.')
        for variant in VARIANTS:
            output = ROOT/f'runs/ufm_ablation_{variant}_s42_v1'
            if not (output/'completed.json').exists():
                wait_ready('waiting_for_idle_gpu_before_ablation')
                cmd = experiment_command(base,variant,output)
                report('ablation_launching',variant=variant,output=str(output),command=cmd)
                # One per-run append log; checkpoints refuse mismatched overwrites.
                with (ROOT/f'runs/ufm_ablation_{variant}_s42_v1.log').open('ab') as log:
                    child = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        report('ablation_training',variant=variant,child_pid=child.pid,output=str(output))
                        time.sleep(args.poll_seconds)
                    if child.returncode:
                        raise RuntimeError(f'{variant} exited {child.returncode}; investigate its log.')
            result = validate_result(output,variant,base)
            completed.append(variant)
            report('ablation_complete',variant=variant,best_cold_macro_ndcg_at_10=result['best_cold_macro_ndcg@10'])
        report('suite_complete',seeds=[42],multiseed_complete=False)
    except Exception as e:
        report('stopped_with_error',error=f'{type(e).__name__}: {e}')
        raise


if __name__ == '__main__': main()
