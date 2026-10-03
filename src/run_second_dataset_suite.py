"""All_Beauty CLIP and three-seed UFM/baselines, after the Toys followup suite."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_followup_suite import ROOT, PARAMETERS, plan, read, additional_trainers
from run_foundation_when_idle import atomic_status, gpu_idle, active_baseline_pids
from run_ufm_training_when_ready import active_ufm_gpu_pids


def second_plan(base):
    data = ROOT/'data/processed/all_beauty_full_temporal'
    legacy = ROOT/'data/processed/all_beauty_gpu_cache_h20_t64_v1'
    graph = ROOT/'data/processed/all_beauty_positive_graph_h20_v1'
    features = data/'foundation_clip_b32_v1'
    config = {**base, 'data':str(data), 'legacy_cache':str(legacy), 'features':str(features)}
    full = ROOT/'runs/all_beauty_ufm_full_s42_v1'
    command = [sys.executable,'-u',str(ROOT/'src/train_ufm_recommender.py'),
               '--variant','full','--output',str(full),'--cache',str(graph)]
    for key in PARAMETERS:
        command += ['--'+key.replace('_','-'),str(42 if key=='seed' else config[key])]
    command += ['--amp' if config['amp'] else '--no-amp']
    jobs = [dict(model='clip', seed=None, output=str(features), command=[sys.executable,'-u',
        str(ROOT/'src/extract_foundation_features.py'),'--device','cuda','--limit','0',
        '--batch-size','64','--catalog',str(data/'foundation_catalog_v1'),'--output',str(features),
        '--model-cache',str(ROOT/'tmp/hf_ufm_models')]),
        dict(model='ufm',seed=42,output=str(full),command=command)]
    for job in plan(config):
        output = ROOT/'runs'/('all_beauty_'+Path(job['output']).name)
        cmd = [str(graph) if word==str(ROOT/'data/processed/ufm_positive_graph_h20_v1') else word
               for word in job['command']]
        cmd[cmd.index('--output')+1] = str(output)
        jobs.append({**job,'output':str(output),'command':cmd})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-days', type=float, default=14)
    args = p.parse_args()
    if args.max_days <= 0:
        p.error('Positive launch window required.')
    jobs = second_plan(read(ROOT/'runs/ufm_full_v1/config.json'))
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    import fcntl
    from train_sasrec import sha256
    status = ROOT/'runs/second_dataset_suite_v1.json'
    deadline = time.monotonic()+args.max_days*86400
    completed = []
    def report(stage, **fields):
        state = dict(stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),
                     completed=completed,test_evaluated=False,**fields)
        atomic_status(status,state); print(json.dumps(state),flush=True)
    with (ROOT/'runs/second_dataset_suite_v1.lock').open('a+b') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            manifest = ROOT/'runs/second_dataset_plan_v1.json'
            code = {name:sha256(ROOT/'src'/name) for name in ('train_sasrec.py','comparison_models.py',
                    'train_ufm_recommender.py','ufm_model.py','extract_foundation_features.py')}
            declared = dict(jobs=jobs,code_sha256=code,
                            preparation_sha256=sha256(ROOT/'reports/second_dataset_preparation.json'))
            if manifest.exists() and read(manifest) != declared:
                raise ValueError('Second dataset plan changed.')
            atomic_status(manifest,declared)
            while read(ROOT/'runs/followup_suite_v1.json').get('stage')!='suite_complete':
                if time.monotonic()>=deadline:
                    raise TimeoutError('Waiting window expired.')
                report('waiting_for_toys_followup'); time.sleep(30)
            with (ROOT/'runs/followup_suite_v1.lock').open('a+b') as gpu_lock:
                fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                for job in jobs:
                    if any(sha256(ROOT/'src'/name)!=value for name,value in code.items()):
                        raise ValueError('Source changed during second dataset suite.')
                    output = Path(job['output'])
                    marker = output/('complete.json' if job['model']=='clip' else 'completed.json')
                    if not marker.exists():
                        checks = 0
                        while checks<2:
                            if time.monotonic()>=deadline:
                                raise TimeoutError('Launch window expired; no active job killed.')
                            idle,gpu = gpu_idle()
                            workers = active_baseline_pids()+active_ufm_gpu_pids()+additional_trainers()
                            checks = checks+1 if idle and not workers else 0
                            report('waiting_for_idle_gpu',gpu=gpu,workers=workers,idle_checks=checks)
                            time.sleep(30)
                        cmd = list(job['command'])
                        resume = 'progress.json' if job['model']=='clip' else 'latest.pt'
                        if (output/resume).exists():
                            cmd.append('--resume')
                        logpath = ROOT/'runs'/('all_beauty_clip.log' if job['model']=='clip' else output.name+'.log')
                        with logpath.open('ab') as log:
                            child = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                pass_fds=(own_lock.fileno(),gpu_lock.fileno()))
                            while child.poll() is None:
                                report('running',job=job,child_pid=child.pid); time.sleep(30)
                        if child.returncode:
                            raise RuntimeError(f'{output.name} exited {child.returncode}; inspect {logpath}')
                    result = read(marker)
                    if job['model']=='clip':
                        if result.get('limit')!=0 or result.get('encoder_frozen') is not True:
                            raise ValueError('Invalid full CLIP completion.')
                    elif result.get('smoke_run') is not False or result.get('test_evaluated') is not False:
                        raise ValueError('Invalid validation-only production marker.')
                    completed.append(str(output)); report('job_complete',job=job)
            report('suite_complete')
        except Exception as error:
            report('stopped_with_error',error=f'{type(error).__name__}: {error}'); raise


if __name__=='__main__':
    main()
