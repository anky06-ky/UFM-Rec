"""CPU-only repair verification against existing FITLAB artifacts; no test inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import sha256,write_json
from evaluation_ledger import TestEvaluationLedger,import_known_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new verification output directory.')
    output.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/toys_games_full_temporal'
    evaluation=data/'evaluation_v2'
    evidence=ROOT/'reports/ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json'
    expected=json.loads(evidence.read_text())['sources']['samples_validation']
    actual=sha256(evaluation/'samples_validation.npz')
    if actual!=expected:
        raise ValueError('Validation v2 candidates do not match the imported UFM evidence.')
    ledger=TestEvaluationLedger(ROOT/'reports/test_evaluation_ledger.json')
    import_known_evidence(ROOT,ledger)
    progress=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),
                  python=sys.executable,validation_candidates_sha256=actual,
                  new_test_inference=False,training_started=False,phases={})
    commands={
        'inventory':[sys.executable,'-u','scripts/audit_ufm_workspace.py',
                     '--verify-checkpoint','--output',str(output/'inventory.json')],
        'audit_v2':[sys.executable,'-u','src/test_candidate_validity.py','--data',str(data),
                    '--evaluation',str(evaluation),'--splits','validation','--output',str(output/'candidate_audit.json')],
        'baseline_v2':[sys.executable,'-u','src/evaluate_content_baseline.py','--data',str(data),
                       '--evaluation',str(evaluation),'--splits','validation','--output',str(output/'baseline_v2')],
        'cpu_tests':[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--tb=short'],
    }
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for name,command in commands.items():
        progress['phases'][name]=dict(status='running',started_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'progress.json',progress)
        print('PHASE_START '+name,flush=True)
        with (output/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        progress['phases'][name].update(status='completed' if result.returncode==0 else 'failed',
               returncode=result.returncode,finished_at_utc=datetime.now(timezone.utc).isoformat())
        if name=='inventory' and (output/'inventory.json').exists():
            progress['phases'][name]['passed']=json.loads((output/'inventory.json').read_text())['passed']
        write_json(output/'progress.json',progress)
        print('PHASE_DONE '+name+' '+str(result.returncode),flush=True)
    progress['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    progress['status']='completed' if all(p['returncode']==0 and p.get('passed',True)
                                          for p in progress['phases'].values()) else 'completed_with_failures'
    write_json(output/'progress.json',progress)
    print('FIX_CHECKS_DONE '+progress['status'],flush=True)


if __name__=='__main__':main()
