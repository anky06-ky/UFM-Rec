"""Freeze comparison inputs without opening test arrays or evaluating a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Manifest is immutable; choose a new output.')
    inventory=json.loads(args.inventory.read_text())
    baseline=json.loads(args.baseline.read_text())
    if not inventory.get('passed'):
        raise ValueError('A successful checkpoint/cache inventory is required.')
    samples=sha256(args.evaluation/'samples_validation.npz')
    if baseline['sources']['samples_validation']!=samples:
        raise ValueError('Baseline and candidate set differ.')
    ufm_path=ROOT/'reports/ufm_evaluation_v2_20261027/reeval_ufm_full_v1_v2.json'
    ufm=json.loads(ufm_path.read_text())
    if ufm['sources']['samples_validation']!=samples or ufm['checkpoint_sha256']!=inventory['checkpoint_sha256']:
        raise ValueError('UFM evidence and verified comparison inputs differ.')
    protocol=ROOT/'configs/evaluation_protocol_v2.json'
    fields=dict(dataset_manifest=args.data/'split_manifest.json',
                items=args.data/'content/items.csv.gz',
                validation_candidates=args.evaluation/'samples_validation.npz',
                validation_metadata=args.evaluation/'samples_validation.csv.gz',
                baseline_report=args.baseline,inventory_report=args.inventory,
                ufm_report=ufm_path,
                feature_marker=args.data/'foundation_clip_b32_v1/complete.json',
                run_config=ROOT/'runs/ufm_full_v1/config.json')
    if protocol.exists():fields['protocol_config']=protocol
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Validation comparison on preserved v2 candidates; not a pre-test lock.',
        primary_metric='cold_macro_ndcg@10',test_already_observed=True,
        instructor_approval='not_recorded',
        inputs={name:dict(path=str(path),sha256=sha256(path)) for name,path in fields.items()},
        checkpoint_sha256=inventory.get('checkpoint_sha256'),
        checkpoint_verification=inventory.get('checks'),
        training_code_hashes=inventory.get('source_hashes'),
        feature_hashes=json.loads((args.data/'foundation_clip_b32_v1/complete.json').read_text())['features_sha256'],
        selection_rule='Best historical validation cold macro NDCG@10; this v2 comparison does not reselect the checkpoint.',
        calibration='No new temperature fitting or test inference in this comparison; imported UFM report retains original scoring metadata.',
        scorer_sha256=sha256(ROOT/'src/evaluate_content_baseline.py'),
        limitations=['Candidate sampling is retrospective and regime-conditioned.',
                     'Old UFM and new baselines require the same candidate hash.',
                     'This manifest cannot retroactively establish test independence.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,manifest)
    print('MANIFEST_LOCKED '+str(args.output),flush=True)


if __name__=='__main__':main()
