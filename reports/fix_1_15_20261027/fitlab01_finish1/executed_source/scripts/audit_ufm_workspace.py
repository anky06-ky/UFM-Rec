"""Read-only inventory and CPU verification of an existing UFM workspace."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-checkpoint', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new inventory output.')
    data = ROOT/'data/processed/toys_games_full_temporal'
    run = ROOT/'runs/ufm_full_v1'
    features = data/'foundation_clip_b32_v1'
    names = ['best.pt','config.json','completed.json','best_validation_predictions.npz']
    paths = [run/name for name in names] + [data/name for name in
        ['content/items.csv.gz','content/products_text.csv.gz','content/tfidf_all.npz',
         'item_statistics.csv.gz','interactions_train.csv.gz','validation_with_history.csv.gz',
         'evaluation_v2/samples_validation.npz','foundation_clip_b32_v1/complete.json']]
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                  python=sys.executable, disk_free_bytes=shutil.disk_usage(ROOT).free,
                  files={str(p.relative_to(ROOT)):dict(exists=p.is_file(),
                         bytes=p.stat().st_size if p.is_file() else None) for p in paths},
                  gpu_scope='Read-only nvidia-smi; no GPU tensor or training allocation.', checks={})
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                                 '--format=csv,noheader'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError) as error:
        report['gpu'] = str(error)
    for name in ['memory.current','memory.max','memory.events']:
        path=Path('/sys/fs/cgroup')/name
        if path.exists(): report[name]=path.read_text().strip()
    report['active_project_jobs']=[]
    for process in Path('/proc').glob('[0-9]*') if Path('/proc').exists() else []:
        try:
            argv=(process/'cmdline').read_bytes().split(b'\0')
            scripts=[x.decode('utf-8','replace') for x in argv if x.endswith(b'.py')]
            relevant=[x for x in scripts if any(token in Path(x).name for token in
                ['train_','run_ufm','extract_foundation','run_followup','run_second'])]
            if relevant: report['active_project_jobs'].append(dict(pid=int(process.name),scripts=relevant))
        except OSError: pass
    if args.verify_checkpoint:
        try:
            import numpy as np
            import torch
            from train_ufm_recommender import feature_audit
            from ufm_model import UFMConfig,UFMRec
            torch.set_num_threads(2)
            report['versions']=dict(numpy=np.__version__,torch=torch.__version__)
            config=json.loads((run/'config.json').read_text())
            report['source_hashes']={name:dict(expected=expected,actual=sha256(ROOT/'src'/name))
                for name,expected in config['code_sha256'].items()}
            source_match=all(v['actual']==v['expected'] for v in report['source_hashes'].values())
            report['checks']['training_code_matches']=source_match
            report['checkpoint_sha256']=sha256(run/'best.pt')
            report['checks']['feature_marker_matches']=sha256(features/'complete.json')==config['feature_marker_sha256']
            counts=[0]
            with gzip.open(data/'content/items.csv.gz','rt',encoding='utf-8') as source:
                for i,row in enumerate(csv.DictReader(source)):
                    if int(row['row_index'])!=i: raise ValueError('Noncontiguous item mapping')
                    counts.append(int(row['train_count']))
            marker=feature_audit(SimpleNamespace(data=data,features=features),len(counts)-1)
            report['checks']['feature_hashes_shapes_masks_norms']=True
            flags=np.load(features/'modalities.npy',mmap_mode='r')
            report['modality_coverage']=dict(items=len(counts)-1,text=int(flags[1:,0].sum()),image=int(flags[1:,1].sum()))
            count_paths=list((ROOT/'data/processed').glob('**/train_counts.npy'))
            matching=[p for p in count_paths if np.load(p,mmap_mode='r').shape==(len(counts),)
                      and np.array_equal(np.load(p,mmap_mode='r'),np.array(counts))]
            report['checks']['train_counts_match_mapping']=bool(matching)
            report['matching_counts']=[str(p.relative_to(ROOT)) for p in matching]
            state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            report['checks']['checkpoint_config_matches']=state['config']==config
            report['checkpoint_epoch']=state['epoch']
            if source_match and matching and report['checks']['feature_marker_matches']:
                model=UFMRec(UFMConfig(**state['model_config'])).eval()
                model.load_state_dict(state['model'])
                with np.load(data/'evaluation_v2/samples_validation.npz') as source:
                    candidates=source['candidates'][:2].astype(np.int64)+1
                    indptr=source['history_indptr'][:3]
                    values=source['history_indices'][:int(indptr[-1])]
                history=np.zeros((2,model.config.history_size),dtype=np.int64)
                for i in range(2):
                    a,b=indptr[i:i+2]; past=values[max(a,b-model.config.history_size):b]+1
                    history[i,:len(past)]=past
                table=[torch.from_numpy(np.load(features/name,mmap_mode='r'))
                       for name in ['text.npy','image.npy','modalities.npy']]
                with torch.no_grad():
                    output=model(torch.from_numpy(history),torch.from_numpy(candidates),*table,
                                 torch.tensor(counts,dtype=torch.long))
                report['checks']['cpu_validation_forward_finite']=bool(torch.isfinite(output['scores']).all())
            else:
                report['checks']['cpu_validation_forward_finite']=False
            historical=data/'evaluation/samples_validation.npz'
            if historical.exists():
                from evaluate_content_baseline import tie_noise
                with np.load(run/'best_validation_predictions.npz') as source:
                    scores=source['scores'].astype(np.float64); idx=source['sample_indices']
                with np.load(historical) as source:
                    candidates=source['candidates'][idx]; regimes=source['regime_codes'][idx]
                adjusted=scores+tie_noise(candidates)*1e-10
                ranks=1+(adjusted[:,1:]>adjusted[:,:1]).sum(1)
                cold=float(np.mean([np.where(ranks[regimes==k]<=10,
                    1/np.log2(ranks[regimes==k]+1),0).mean() for k in range(3)]))
                done=json.loads((run/'completed.json').read_text())
                report['validation_metric_reproduced']=cold
                report['checks']['saved_validation_metric_matches']=abs(cold-done['best_cold_macro_ndcg@10'])<2e-6
            else:
                report['checks']['saved_validation_metric_matches']=False
                report['missing_historical_candidates']=str(historical)
        except Exception as error:
            report['verification_error']=repr(error)
    report['passed']=bool(report['checks']) and all(report['checks'].values()) and 'verification_error' not in report
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps(dict(output=str(args.output),passed=report['passed'],checks=report['checks'],
                          error=report.get('verification_error')),ensure_ascii=True),flush=True)


if __name__=='__main__': main()
