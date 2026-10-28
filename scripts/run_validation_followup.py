"""Run a fixed, validation-only comparison on v3 and record durable progress.

Uses existing selected checkpoints. This is re-evaluation, not multi-seed training
or independent final-test evidence. All output paths must be new.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import sha256

V3_SHA='9387e66e9f1905c44d02b65c3809f5fa8b6c022f6abf0c1a37d52cb57767a60e'
RUNS=['ufm_full_v1','ufm_ablation_semantic_only_s42_v1',
      'ufm_ablation_fixed_fusion_s42_v1','ufm_ablation_no_uncertainty_s42_v1',
      'ufm_ablation_no_cross_align_s42_v1','ufm_ablation_text_only_s42_v1',
      'ufm_ablation_image_only_s42_v1','ufm_ablation_collaborative_only_s42_v1']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=args.artifacts.resolve()
    source=args.source.resolve()
    output=args.output.resolve()
    evaluation=r/'reports/fix_1_15_20261027/fitlab01_finish1/evaluation_v3'
    if sha256(evaluation/'samples_validation.npz')!=V3_SHA:
        raise ValueError('Validation v3 candidate hash differs from the declared plan.')
    if output.exists():
        raise FileExistsError('Choose new follow-up output; no overwrite/resume guessing.')
    output.mkdir(parents=True)
    data=r/'data/processed/toys_games_full_temporal'
    training_source=r/'reports/fix_1_15_20261027/fitlab01_finish1/trained_snapshot'
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),phase='prepared',
        evaluation_sha256=V3_SHA,split='validation',test_evaluated=False,device='cpu',amp=False,
        declared_runs=RUNS,seeds=[42],training_started=False,
        selection='Existing checkpoints selected using historical validation v1; no selection by these v3 scores.',
        limitations=['V3 is retrospective, same-regime negative sampling.',
                     'Same validation population as prior selection; not an independent holdout.',
                     'No claim of training-seed variability or successful improvements.'],stages=[])
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    def save():
        tmp=output/'progress.json.partial'
        tmp.write_text(json.dumps(plan,indent=2))
        tmp.replace(output/'progress.json')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    def stage(name,command):
        record=dict(name=name,started_at_utc=datetime.now(timezone.utc).isoformat(),phase='running',command=command)
        plan['stages'].append(record);plan['phase']='running';save()
        print('STAGE_STARTED',name,flush=True)
        with (output/(name+'.log')).open('xb') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        record.update(returncode=result.returncode,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      phase='complete' if result.returncode==0 else 'failed')
        save()
        if result.returncode:
            raise RuntimeError('Follow-up stage failed: '+name)
        print('STAGE_COMPLETE',name,flush=True)
    try:
        stage('baseline',[sys.executable,str(source/'src/evaluate_content_baseline.py'),
            '--data',str(data),'--evaluation',str(evaluation),'--output',str(output/'baseline'),'--splits','validation'])
        for name in RUNS:
            stage(name,[sys.executable,str(source/'src/reevaluate_ufm_checkpoints.py'),
                '--checkpoint',str(r/'runs'/name/'best.pt'),'--evaluation',str(evaluation),
                '--features',str(data/'foundation_clip_b32_v1'),
                '--counts',str(r/'data/processed/gpu_cache_h20_t64_v1/train_counts.npy'),
                '--training-source',str(training_source),'--device','cpu','--batch','64',
                '--output',str(output/(name+'.json')),'--save-predictions',str(output/(name+'_scores'))])
            stage(name+'_calibration',[sys.executable,str(source/'src/fit_temperature_v2.py'),
                '--predictions',str(output/(name+'_scores')/'scores_validation.npz'),
                '--evaluation',str(evaluation),'--output',str(output/(name+'_calibration.json'))])
        plan['phase']='complete';save()
    except BaseException as error:
        plan['phase']='failed';plan['error']=str(error);save();raise


if __name__=='__main__':
    main()
