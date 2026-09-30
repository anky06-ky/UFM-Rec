"""Temperature scaling on earlier validation, audit on later validation only.

This is exploratory sampled-candidate calibration for the fixed TF-IDF baseline.
It does not read test labels, change ranking, or estimate purchase probabilities.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

from evaluate_content_baseline import PROFILE_ITEMS, REGIMES, rank_of_target
from train_evaluate_bpr_mf import DATA, sha256, write_json
from ufm_model import candidate_calibration


def temporal_folds(timestamps):
    timestamps = np.asarray(timestamps,dtype=np.int64)
    if timestamps.ndim!=1 or len(timestamps)<4:
        raise ValueError('At least four validation timestamps required.')
    boundary = int(np.sort(timestamps)[len(timestamps)//2])
    fit, audit = np.flatnonzero(timestamps<boundary),np.flatnonzero(timestamps>=boundary)
    if not len(fit) or not len(audit):
        raise ValueError('Cannot separate validation chronologically without splitting timestamp ties.')
    return fit,audit,boundary


def fit_temperature(scores):
    scores = np.asarray(scores,dtype=np.float64)
    if scores.ndim!=2 or not len(scores) or scores.shape[1]<2 or not np.isfinite(scores).all():
        raise ValueError('Finite N x C logits required; positive must be column zero.')
    def nll(log_temperature):
        logits = scores/np.exp(log_temperature)
        return float((logsumexp(logits,axis=1)-logits[:,0]).mean())
    bounds = (np.log(.01),np.log(100.))
    result = minimize_scalar(nll,bounds=bounds,method='bounded',options={'xatol':1e-6})
    if not result.success: raise RuntimeError('Temperature optimization did not converge.')
    best = min([0.,bounds[0],bounds[1],float(result.x)],key=nll)
    return float(np.exp(best))


def calibration_report(scores,temperature,regimes):
    scores = torch.as_tensor(np.asarray(scores,dtype=np.float64))
    report = dict(overall=candidate_calibration(scores,temperature=temperature),by_regime={})
    for code,name in enumerate(REGIMES):
        selected = regimes==code
        report['by_regime'][name] = (candidate_calibration(scores[selected],temperature=temperature)
                                     if selected.any() else {'n':0})
    return report


def collect_scores(data):
    with np.load(data/'evaluation/samples_validation.npz') as f:
        samples = {k:f[k] for k in f.files}
    timestamps = []
    with gzip.open(data/'evaluation/samples_validation.csv.gz','rt',encoding='utf-8') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if int(row['sample_index'])!=index: raise ValueError('Validation sample order mismatch.')
            timestamps.append(int(row['timestamp']))
    if len(timestamps)!=len(samples['candidates']): raise ValueError('Validation sample length mismatch.')
    counts = []
    with gzip.open(data/'content/items.csv.gz','rt',encoding='utf-8') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if int(row['row_index'])!=i: raise ValueError('Item mapping order mismatch.')
            counts.append(int(row['train_count']))
    counts = np.asarray(counts,dtype=np.float32)
    matrix = sparse.load_npz(data/'content/tfidf_all.npz').tocsr()
    if matrix.shape[0]!=len(counts) or not np.isfinite(matrix.data).all():
        raise ValueError('TF-IDF mapping/finite check failed.')
    available = np.diff(matrix.indptr)>0
    scores = np.empty(samples['candidates'].shape,dtype=np.float64)
    ranks = np.empty(len(scores),dtype=np.uint8)
    for i,candidates in enumerate(samples['candidates']):
        popularity = np.log1p(counts[candidates])
        start,end = samples['history_indptr'][i:i+2]
        history = samples['history_indices'][max(start,end-PROFILE_ITEMS):end]
        history = history[available[history]]
        if len(history):
            profile = np.asarray(matrix[history].sum(0)).ravel()
            value = np.asarray(matrix[candidates].dot(profile)).ravel()
            value = value+popularity/(float(popularity.max()) or 1.)*1e-7
        else: value = popularity
        scores[i] = value
        ranks[i] = rank_of_target(value,candidates)
        if (i+1)%10000==0: print(f'validation_scores={i+1}/{len(scores)}',flush=True)
    with np.load(data/'models/content_baseline/ranks_validation.npz') as expected:
        if not np.array_equal(ranks,expected['content_ranks']):
            raise ValueError('Recomputed ranks differ from the published TF-IDF baseline.')
    return scores,np.asarray(timestamps,dtype=np.int64),samples['regime_codes']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',type=Path,default=DATA/'models/content_calibration_v1')
    args = p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory; no overwrite.')
    torch.set_num_threads(4)
    sources = {name:sha256(args.data/name) for name in (
        'evaluation/samples_validation.npz','evaluation/samples_validation.csv.gz',
        'content/items.csv.gz','content/tfidf_all.npz','models/content_baseline/ranks_validation.npz')}
    scores,timestamps,regimes = collect_scores(args.data)
    fit,audit,boundary = temporal_folds(timestamps)
    temperature = fit_temperature(scores[fit])
    report = dict(model='TF-IDF content baseline',sources=sources,
        code_sha256={Path(__file__).name:sha256(Path(__file__)),
                     'ufm_model.py':sha256(Path(__file__).with_name('ufm_model.py'))},
        protocol='One positive (column 0) + 99 fixed negatives; exploratory validation-only calibration.',
        scope='conditional_on_sampled_candidate_set_not_purchase_probability',
        split_rule='earlier timestamps fit; later timestamps audit; ties remain together',
        boundary_timestamp_ms=boundary,fit_samples=len(fit),audit_samples=len(audit),
        fit_max_timestamp_ms=int(timestamps[fit].max()),audit_min_timestamp_ms=int(timestamps[audit].min()),
        temperature=temperature,temperature_bounds=[.01,100.],
        rank_reproduction_passed=True,test_evaluated=False,
        limitations=['Fixed baseline already evaluated on full validation previously; this is not a fresh model-selection holdout.',
                    'Only TF-IDF calibration measured; no evidence about UFM uncertainty heads.',
                    'Temporal folds may contain overlapping users.'],
        before=calibration_report(scores[audit],1.,regimes[audit]),
        after=calibration_report(scores[audit],temperature,regimes[audit]))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'scores_validation.npz',scores=scores,timestamps=timestamps,
                        regime_codes=regimes,fit_indices=fit,audit_indices=audit)
    write_json(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('temperature','fit_samples','audit_samples')}),flush=True)
    print(json.dumps({name:{k:report[name]['overall'][k] for k in ('ece','nll','brier_multiclass')}
                      for name in ('before','after')}),flush=True)


if __name__=='__main__': main()
