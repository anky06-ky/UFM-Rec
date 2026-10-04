"""Sequential validation-only baselines and extra UFM seeds after seven ablations.

One inherited flock, two idle GPU probes, bounded launch window. SIGKILL/OOM
can retry up to three times; SIGTERM and invariant failures require inspection.
No test evaluation or automatic changes to the selected research hypothesis.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_foundation_when_idle import ROOT, atomic_status, gpu_idle, active_baseline_pids
from run_ufm_training_when_ready import active_ufm_gpu_pids, verify_run
from run_ufm_ablation_suite import PARAMETERS, VARIANTS, validate_result


def plan(base):
    jobs = []
    for seed in (42, 7, 2026):
        for model in ('sasrec', 'bert4rec', 'concat'):
            output = ROOT / f'runs/{model}_s{seed}_v2'
            cmd = [sys.executable, '-u', str(ROOT/'src/train_sasrec.py'), '--model', model,
                   '--output', str(output), '--graph', str(ROOT/'data/processed/ufm_positive_graph_h20_v1')]
            for key in ('data','legacy_cache','features','history','dim','heads','layers','batch_size',
                        'negatives','epochs','lr','patience','checkpoint_every','eval_batch','validation_cap','threads'):
                cmd += ['--'+key.replace('_','-'), str(base[key])]
            metric = 'cold_macro' if model == 'concat' else 'overall'
            cmd += ['--seed', str(seed), '--selection-metric', metric]
            jobs.append(dict(model=model, seed=seed, output=str(output), command=cmd))
        if seed != 42:
            output = ROOT/f'runs/ufm_full_s{seed}_v1'
            cmd = [sys.executable,'-u',str(ROOT/'src/train_ufm_recommender.py'),
                   '--variant','full','--output',str(output),
                   '--cache',str(ROOT/'data/processed/ufm_positive_graph_h20_v1')]
            for key in PARAMETERS:
                cmd += ['--'+key.replace('_','-'), str(seed if key=='seed' else base[key])]
            cmd += ['--amp' if base['amp'] else '--no-amp']
            jobs.append(dict(model='ufm',seed=seed,output=str(output),command=cmd))
    return jobs


def additional_trainers():
    expected = {b'src/train_sasrec.py', str(ROOT/'src/train_sasrec.py').encode()}
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if any(x in expected for x in (p/'cmdline').read_bytes().split(b'\0')):
                    result.append(int(p.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    return result


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-days', type=float, default=7)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.max_days <= 0:
        p.error('Positive launch window required.')
    base = read(ROOT/'runs/ufm_full_v1/config.json')
    jobs = plan(base)
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    import fcntl
    from common import sha256
    deadline = time.monotonic()+args.max_days*86400
    status = ROOT/'runs/followup_suite_v1.json'
    completed = []
    def report(stage, **fields):
        value = dict(stage=stage, time_utc=datetime.now(timezone.utc).isoformat(),
                     completed=completed, test_evaluated=False, **fields)
        atomic_status(status, value); print(json.dumps(value), flush=True)
    def wait_ready(require_ablations=True):
        checks = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Launch window expired; running jobs are never killed.')
            ablations = read(ROOT/'runs/ufm_ablation_suite_v1.json')
            ready_parent = ablations.get('stage') == 'suite_complete'
            if require_ablations and not ready_parent:
                report('waiting_for_ablations', ablation_stage=ablations.get('stage'))
                time.sleep(30); continue
            workers = active_baseline_pids()+active_ufm_gpu_pids()+additional_trainers()
            idle, gpu = gpu_idle()
            checks = checks+1 if idle and not workers else 0
            report('waiting_for_idle_gpu', gpu=gpu, workers=workers, idle_checks=checks)
            if checks >= 2:
                return
            time.sleep(30)
    def execute(command, output, label, lock):
        for attempt in range(1, 4):
            wait_ready()
            actual = list(command)
            if (output/'latest.pt').is_file() and not (output/'completed.json').is_file():
                actual += ['--resume']
            logfile = output.with_suffix('.log')
            with logfile.open('ab') as log:
                child = subprocess.Popen(actual, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                while child.poll() is None:
                    report('training', job=label, child_pid=child.pid, command=actual, attempt=attempt)
                    time.sleep(30)
            if child.returncode == 0:
                return
            with logfile.open('rb') as log:
                log.seek(max(0, logfile.stat().st_size-8192))
                tail = log.read().decode(errors='replace')
            retry = child.returncode in (-9,137) or 'CUDA out of memory' in tail
            if not retry or attempt == 3:
                raise RuntimeError(f'{label} exited {child.returncode}; see {logfile}')
            report('retry_pending', job=label, exit_code=child.returncode)
            time.sleep(30)
    with (ROOT/'runs/followup_suite_v1.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            wait_ready()
            verify_run(ROOT/'runs/ufm_full_v1', False)
            for variant in VARIANTS:
                validate_result(ROOT/f'runs/ufm_ablation_{variant}_s42_v1', variant, base)
            manifest = ROOT/'runs/followup_plan_v1.json'
            code = {name: sha256(ROOT/'src'/name) for name in
                    ('train_sasrec.py','comparison_models.py','train_ufm_recommender.py','ufm_model.py')}
            declared = dict(jobs=jobs, code_sha256=code, parent_config=sha256(ROOT/'runs/ufm_full_v1/config.json'))
            if manifest.exists() and read(manifest) != declared:
                raise ValueError('Followup plan changed since launch.')
            atomic_status(manifest, declared)
            for job in jobs:
                output = Path(job['output'])
                label = f"{job['model']}_s{job['seed']}"
                if job['seed']==42:
                    smoke = ROOT/f"runs/{job['model']}_gpu_smoke_v2"
                    if not (smoke/'completed.json').exists():
                        cmd = list(job['command'])
                        cmd[cmd.index('--output')+1] = str(smoke)
                        cmd += ['--max-steps','5','--validation-cap','400','--checkpoint-every','2']
                        execute(cmd, smoke, label+'_smoke', lock)
                    if read(smoke/'completed.json').get('smoke_run') is not True:
                        raise ValueError('Expected technical smoke marker.')
                if not (output/'completed.json').exists():
                    execute(job['command'], output, label, lock)
                done, config = read(output/'completed.json'), read(output/'config.json')
                if (done.get('smoke_run') is not False or done.get('test_evaluated') is not False
                        or config.get('seed') != job['seed'] or config.get('validation_cap') != 0
                        or not (output/'best.pt').is_file()):
                    raise ValueError('Invalid completed followup run: '+label)
                completed.append(dict(job=label, output=str(output), result=done))
                report('job_complete', job=label)
            report('suite_complete', remaining='dataset2, error analysis/tuning decisions, frozen test manifest and final report')
        except Exception as error:
            report('stopped_with_error', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
