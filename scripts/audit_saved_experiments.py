"""Audit completed saved validation runs without training, GPU use or test labels.

Historical evaluation/ is kept separate from evaluation_v2/v3. A saved metric is
accepted only when predictions, candidate association and completion agree.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import sha256
from evaluate_content_baseline import summarize, tie_noise, REGIMES
from train_gpu_recommender import fingerprint
from audit_ufm_validation import paired_user_ci, ndcg
from calibrate_validation import temporal_folds, fit_temperature, calibration_report

MATCH_KEYS = ['seed','epochs','batch_size','dim','history','heads','layers','negatives',
              'patience','lr','weight_decay','dropout','id_dropout','fixed_alpha',
              'lambda_align','lambda_unc','lambda_cal','amp','validation_cap','max_steps']


def validate_predictions(path, samples):
    with np.load(path) as data:
        scores, indices = data['scores'], data['sample_indices']
        if (not np.array_equal(indices, np.arange(len(samples['candidates'])))
            or scores.shape != samples['candidates'].shape or not np.isfinite(scores).all()):
            raise ValueError('Predictions do not cover the full ordered candidate set.')
        if 'candidates' in data and not np.array_equal(data['candidates'],samples['candidates']):
            raise ValueError('Saved candidate IDs differ.')
    adjusted = scores.astype(np.float64) + tie_noise(samples['candidates'])*1e-10
    ranks = 1 + (adjusted[:,1:] > adjusted[:,:1]).sum(1)
    return scores,ranks


def audit(root, output, repeats=1000):
    if output.exists():
        raise FileExistsError('Choose a new output directory.')
    data = root/'data/processed/toys_games_full_temporal'
    evaluation = data/'evaluation'
    candidates_sha = sha256(evaluation/'samples_validation.npz')
    with np.load(evaluation/'samples_validation.npz') as archive:
        samples = {key:archive[key] for key in archive.files}
    with gzip.open(evaluation/'samples_validation.csv.gz','rt',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows)!=len(samples['candidates']) or any(int(row['sample_index'])!=i for i,row in enumerate(rows)):
        raise ValueError('Metadata order/length mismatch.')
    users = np.asarray([row['user_id'] for row in rows])
    timestamps = np.asarray([int(row['timestamp']) for row in rows],dtype=np.int64)
    lengths = np.diff(samples['history_indptr'])
    regime = samples['regime_codes']
    train_fingerprint = fingerprint(data)
    feature_marker = sha256(data/'foundation_clip_b32_v1/complete.json')
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        split='validation', protocol_version='historical_evaluation_v1', test_evaluated=False,
        candidate_sha256=candidates_sha, samples=len(rows), candidates=samples['candidates'].shape[1],
        models={},excluded_runs={}, paired_comparisons={},
        limitations=['Historical candidates; do not mix these scores with v2/v3.',
                     'One training seed per completed run; bootstrap over users is not seed variability.',
                     'Calibration uses validation already used for checkpoint selection and is exploratory.',
                     'SASRec/BERT4Rec are adapted implementations, not verified exact reproductions.',
                     'Saved scores are verified against the legacy source fingerprint and selected metric; no new model forward here.'])
    rank_sets = {}
    full_config = json.loads((root/'runs/ufm_full_v1/config.json').read_text())
    for config_path in sorted((root/'runs').glob('*/config.json')):
        run = config_path.parent
        name = run.name
        try:
            config = json.loads(config_path.read_text())
            done = json.loads((run/'completed.json').read_text())
            if done.get('smoke_run') is not False or config.get('validation_cap')!=0 or config.get('max_steps',0):
                raise ValueError('Smoke or capped/incomplete validation run.')
            is_ufm = 'variant' in config
            if is_ufm:
                if config['cache']['sources']['legacy_source'] != train_fingerprint:
                    raise ValueError('Legacy data/candidate fingerprint mismatch.')
                if config['feature_marker_sha256'] != feature_marker:
                    raise ValueError('CLIP feature marker mismatch.')
            elif config.get('sources',{}).get('validation') != candidates_sha:
                raise ValueError('Recorded validation candidate SHA mismatch.')
            scores,ranks = validate_predictions(run/'best_validation_predictions.npz', samples)
            metrics = summarize(ranks,regime,lengths,samples.get('repeat_purchases'))
            cold = float(np.mean([ndcg(ranks[regime==code]).mean() for code in range(3)]))
            expected = done.get('best_cold_macro_ndcg@10',done.get('best_cold_macro_ndcg_at_10'))
            if expected is None or abs(cold-float(expected))>2e-6:
                raise ValueError('Saved scores do not reproduce selected checkpoint metric.')
            groups = {}
            for code,label in enumerate(REGIMES):
                groups[label] = {}
                for group,mask in [('all',regime==code),('history',(regime==code)&(lengths>0)),
                                   ('no_history',(regime==code)&(lengths==0))]:
                    selected = ranks[mask]
                    groups[label][group] = dict(samples=len(selected),
                        ndcg10=float(ndcg(selected).mean()) if len(selected) else None,
                        recall10=float((selected<=10).mean()) if len(selected) else None,
                        mrr10=float(np.where(selected<=10,1/selected,0).mean()) if len(selected) else None)
            fit,holdout,boundary = temporal_folds(timestamps)
            temperature = fit_temperature(scores[fit])
            record = dict(model=config.get('variant',config.get('model',name)),seed=config['seed'],
                cold_macro_ndcg10=cold, metrics=metrics, groups=groups, metric_reproduced=True,
                budget={key:config.get(key) for key in MATCH_KEYS},steps=done.get('steps'),
                parameter_count=done.get('trainable_parameters'), peak_vram_bytes=done.get('peak_vram_bytes'),
                training_seconds=done.get('elapsed_seconds',done.get('elapsed_seconds_this_process')),
                sources={key:sha256(run/file) for key,file in [('config','config.json'),
                    ('completion','completed.json'),('predictions','best_validation_predictions.npz')]},
                objective=config.get('objective','UFM BPR + alignment + uncertainty + calibration losses'),
                calibration=dict(temperature=temperature, fit_samples=len(fit),audit_samples=len(holdout),
                    boundary_timestamp_ms=boundary,
                    before=calibration_report(scores[holdout],1.,regime[holdout]),
                    after=calibration_report(scores[holdout],temperature,regime[holdout])))
            if is_ufm:
                record['controlled_changes_from_full']={key:[full_config.get(key),config.get(key)]
                    for key in MATCH_KEYS if full_config.get(key)!=config.get(key)}
                record['same_training_code_as_full']=config.get('code_sha256')==full_config.get('code_sha256')
                record['same_train_graph_as_full']=config.get('cache')==full_config.get('cache')
            rank_sets[name]=ranks
            report['models'][name]=record
            print('AUDITED',name,cold,flush=True)
        except (ValueError,KeyError,FileNotFoundError) as error:
            report['excluded_runs'][name]=str(error)
            print('EXCLUDED',name,str(error),flush=True)
    if 'ufm_full_v1' not in rank_sets:
        raise ValueError('Full UFM did not pass provenance/metric checks.')
    full_ndcg=ndcg(rank_sets['ufm_full_v1'])
    for name,ranks in rank_sets.items():
        if name!='ufm_full_v1':
            report['paired_comparisons']['ufm_full_v1_minus_'+name]=paired_user_ci(
                full_ndcg-ndcg(ranks),users,regime,repeats=repeats)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(output/'verified_ranks_validation.npz',**rank_sets)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.bootstrap<100:
        p.error('At least 100 user-bootstrap repetitions required.')
    import torch
    torch.set_num_threads(2)
    audit(args.root,args.output,args.bootstrap)
