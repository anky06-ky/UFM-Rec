"""Audit the fixed sampled-ranking evaluation sets for candidate validity.

Checks per split (validation/test), against the stored npz arrays:
1. structure: 1 positive (column 0) + 99 negatives per row, all unique and in range;
2. exclusions: negatives are never in the recorded history or the target itself;
3. regime purity: the sample regime equals the train-only-count regime of the
   target, and every negative belongs to the same-regime pool with a positive
   count by the end of the split;
4. replay: recomputing the declared deterministic protocol (reservoir sampling
   plus seeded negative draws) reproduces the stored arrays exactly;
5. disclosure metrics (for the report): negatives the same user interacts with
   later in the same split or in a later split (future positives), and negatives
   the user interacted with before but outside the recorded history window
   (accidental past positives; requires --prior-interactions).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from build_recommender_samples import (
    DATA, NEGATIVES, REGIMES, REGIME_CODE, SEED, SPLITS, SAMPLES_PER_REGIME,
    choose_negatives, load_catalog, reservoir_sample,
)
from common import sha256, write_json
from temporal_positives import prior_paths, positive_events, known_before

POSITIVE_RATING = 4.0


def structure_check(candidates, regime_codes, history_indices, history_indptr,
                    repeat_purchases, n_items, negatives=None):
    problems = []
    expected = (len(candidates), (NEGATIVES if negatives is None else negatives) + 1)
    if candidates.ndim != 2 or candidates.shape != expected:
        problems.append(f'candidates shape {candidates.shape}, expected {expected}')
        return {'passed': False, 'problems': problems}
    for name, values in [('history_indptr', history_indptr), ('history_indices', history_indices),
                         ('regime_codes', regime_codes)]:
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            problems.append(f'{name} must be a one-dimensional integer array')
    if problems:
        return {'passed': False, 'problems': problems}
    if len(regime_codes) != len(candidates):
        problems.append(f'regime_codes length {len(regime_codes)} != {len(candidates)}')
    if not len(candidates) or n_items < 1:
        problems.append('empty benchmark or catalog')
    if regime_codes.ndim != 1 or ((regime_codes < 0) | (regime_codes >= len(REGIMES))).any():
        problems.append('invalid regime codes')
    if candidates.dtype != np.int32:
        problems.append(f'candidates dtype {candidates.dtype}, expected int32')
    if (candidates < 0).any() or (candidates >= n_items).any():
        problems.append('candidate index out of item range')
    ordered = np.sort(candidates, axis=1)
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        problems.append('duplicate candidate within a row')
    if len(history_indptr) != len(candidates) + 1 or history_indptr[0] != 0:
        problems.append('history_indptr malformed')
    elif int(history_indptr[-1]) != len(history_indices):
        problems.append('history_indices length mismatch')
    elif (np.diff(history_indptr) < 0).any():
        problems.append('history_indptr not monotonic')
    if (history_indices < 0).any() or (history_indices >= n_items).any():
        problems.append('history index out of item range')
    if problems:
        return {'passed': False, 'problems': problems, 'samples': len(candidates)}
    if repeat_purchases is not None:
        expected_repeats = np.zeros(len(candidates), dtype=np.uint8)
        for i, row in enumerate(candidates):
            start, end = history_indptr[i:i + 2]
            expected_repeats[i] = 1 if int(row[0]) in set(history_indices[start:end].tolist()) else 0
        if not np.array_equal(repeat_purchases, expected_repeats):
            problems.append('repeat_purchases does not match target-in-history')
    return {'passed': not problems, 'problems': problems, 'samples': len(candidates)}


def exclusion_check(candidates, history_indices, history_indptr):
    violated = 0
    for i, row in enumerate(candidates):
        start, end = history_indptr[i:i + 2]
        excluded = set(history_indices[start:end].tolist())
        excluded.add(int(row[0]))
        if any(int(value) in excluded for value in row[1:]):
            violated += 1
    return {'passed': violated == 0, 'violations': violated, 'samples': len(candidates)}


def regime_check(candidates, regime_codes, train_counts, split_counts, n_items):
    item_regime = np.zeros(n_items, dtype=np.uint8)
    item_regime[(train_counts >= 1) & (train_counts <= 5)] = 1
    item_regime[(train_counts >= 6) & (train_counts <= 20)] = 2
    item_regime[train_counts > 20] = 3
    target_ok = bool(np.array_equal(item_regime[candidates[:, 0]], regime_codes))
    same_pool = bool((item_regime[candidates[:, 1:]] == regime_codes[:, None]).all())
    nonzero = bool((split_counts[candidates[:, 1:]] > 0).all())
    return {'passed': target_ok and same_pool and nonzero,
            'target_regime_matches': target_ok,
            'negatives_same_regime_pool': same_pool,
            'negatives_have_split_count': nonzero,
            'positive_train_counts': {name: int((train_counts[candidates[:, 0]][regime_codes == code]).sum())
                                      for code, name in enumerate(REGIMES)}}


def replay(split, ids, id_to_row, train_counts, validation_counts, test_counts,
           *, data=None, min_interactions=0, cap=None, negatives=None, seed=None,
           exclude_known_positives=False):
    from temporal_positives import positive_events, known_before
    negatives = NEGATIVES if negatives is None else negatives
    seed = SEED if seed is None else seed
    samples, seen = reservoir_sample(split, min_interactions, data=data, cap=cap, seed=seed)
    known = (positive_events(prior_paths(DATA if data is None else data, split),
                             id_to_row, {row['user_id'] for row in samples})
             if exclude_known_positives else {})
    samples.sort(key=lambda row: (
        REGIME_CODE[row['cold_start_regime']], row['user_id'],
        int(row['timestamp']), row['parent_asin']))
    counts = (train_counts + validation_counts if split == 'validation'
              else train_counts + validation_counts + test_counts)
    item_regime = np.zeros(len(train_counts), dtype=np.uint8)
    item_regime[(train_counts >= 1) & (train_counts <= 5)] = 1
    item_regime[(train_counts >= 6) & (train_counts <= 20)] = 2
    item_regime[train_counts > 20] = 3
    pools = {k: np.flatnonzero((item_regime == k) & (counts > 0)) for k in range(4)}
    cdfs = {k: np.cumsum(counts[pools[k]]) for k in range(4)}
    rng = np.random.default_rng(seed + 100 + SPLITS.index(split))
    candidates = np.empty((len(samples), negatives + 1), dtype=np.int32)
    history_values = []
    history_indptr = np.zeros(len(samples) + 1, dtype=np.int64)
    regime_codes = np.empty(len(samples), dtype=np.uint8)
    for i, row in enumerate(samples):
        target = id_to_row[row['parent_asin']]
        history = [id_to_row[item] for item in row['history'].split()]
        excluded = set(history)
        excluded.add(target)
        if exclude_known_positives:
            excluded.update(known_before(known, row['user_id'], int(row['timestamp'])))
        candidates[i, 0] = target
        regime_codes[i] = REGIME_CODE[row['cold_start_regime']]
        candidates[i, 1:] = choose_negatives(
            rng, pools[regime_codes[i]], cdfs[regime_codes[i]], excluded, negatives, validate=False)
        history_values.extend(history)
        history_indptr[i + 1] = len(history_values)
    return candidates, np.asarray(history_values, dtype=np.int32), history_indptr, regime_codes


def replay_check(split, arrays, ids, id_to_row, train_counts, validation_counts, test_counts, **options):
    try:
        candidates, history, indptr, regimes = replay(
            split, ids, id_to_row, train_counts, validation_counts, test_counts, **options)
    except Exception as error:
        return {'passed': False, 'error': f'{type(error).__name__}: {error}'}
    equal = {
        'candidates': bool(np.array_equal(candidates, arrays['candidates'])),
        'history_indices': bool(np.array_equal(history, arrays['history_indices'])),
        'history_indptr': bool(np.array_equal(indptr, arrays['history_indptr'])),
        'regime_codes': bool(np.array_equal(regimes, arrays['regime_codes'])),
    }
    return {'passed': all(equal.values()), 'equal': equal}


def events_by_user(path, id_to_row, users=None):
    events = {}
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            if users is not None and row['user_id'] not in users:
                continue
            events.setdefault(row['user_id'], []).append(
                (int(row['timestamp']), id_to_row[row['parent_asin']]))
    return events


def future_positive_check(split, metadata_path, arrays, id_to_row, extra_paths=(), *, data=None):
    with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    users = {row['user_id'] for row in rows}
    events = events_by_user((DATA if data is None else Path(data)) / f'{split}_with_history.csv.gz', id_to_row, users)
    for path in extra_paths:
        for user, extra in events_by_user(path, id_to_row, users).items():
            events.setdefault(user, []).extend(extra)
    candidates = arrays['candidates']
    regimes = arrays['regime_codes']
    future = 0
    simultaneous = 0
    samples_with_future = 0
    regime_future = {name: 0 for name in REGIMES}
    regime_negatives = {name: 0 for name in REGIMES}
    with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(candidates) or any(int(r['sample_index']) != i for i, r in enumerate(rows)):
        raise ValueError(f'{split}: metadata order differs from samples.')
    for i, row in enumerate(rows):
        timestamp = int(row['timestamp'])
        user_events = events.get(row['user_id'], [])
        later = {item for ts, item in user_events if ts > timestamp}
        same = {item for ts, item in user_events if ts == timestamp}
        negatives = candidates[i, 1:].tolist()
        hit = sum(1 for value in negatives if value in later)
        if hit:
            future += hit
            samples_with_future += 1
        simultaneous += sum(1 for value in negatives if value in same)
        regime_future[REGIMES[regimes[i]]] += hit
        regime_negatives[REGIMES[regimes[i]]] += candidates.shape[1] - 1
    total_negatives = len(candidates) * (candidates.shape[1] - 1)
    return {
        'passed': True,
        'samples': len(candidates),
        'negatives_total': total_negatives,
        'future_positive_negatives': future,
        'future_positive_fraction': round(future / total_negatives, 8) if total_negatives else 0.0,
        'samples_with_future_positive': samples_with_future,
        'samples_with_future_positive_fraction': round(samples_with_future / len(candidates), 8) if len(candidates) else 0.0,
        'same_timestamp_negatives': simultaneous,
        'by_regime': {name: {'future': regime_future[name],
                             'fraction': round(regime_future[name] / regime_negatives[name], 8)
                             if regime_negatives[name] else None}
                      for name in REGIMES},
    }


def past_positive_check(metadata_path, arrays, id_to_row, prior_paths, events=None):
    """Report negative candidates that are earlier positives outside recorded history.

    The evaluation keeps negatives tied to the recorded history window. A prior
    positive is therefore an accidental negative only when it occurred before
    the sampled target timestamp and is absent from that stored history window.
    """
    candidates = arrays['candidates']
    regimes = arrays['regime_codes']
    history_indptr = arrays['history_indptr']
    history_indices = arrays['history_indices']
    rows = []
    with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(candidates) or any(int(row['sample_index']) != i for i, row in enumerate(rows)):
        raise ValueError('Metadata order differs from candidate arrays.')
    users = [row['user_id'] for row in rows]
    target_timestamps = [int(row['timestamp']) for row in rows]
    prior = positive_events(prior_paths, id_to_row, set(users)) if events is None else events
    accidental = 0
    samples_with_accidental = 0
    regime_accidental = {name: 0 for name in REGIMES}
    for i, (user_id, target_timestamp) in enumerate(zip(users, target_timestamps)):
        recorded = set(history_indices[history_indptr[i]:history_indptr[i + 1]].tolist())
        earlier_known = {item for item, timestamp in prior[user_id].items()
                         if timestamp < target_timestamp and item not in recorded}
        if not earlier_known:
            continue
        hit = sum(1 for value in candidates[i, 1:].tolist() if value in earlier_known)
        if hit:
            accidental += hit
            samples_with_accidental += 1
        regime_accidental[REGIMES[regimes[i]]] += hit
    total_negatives = len(candidates) * (candidates.shape[1] - 1)
    return {
        'passed': True,
        'samples': len(candidates),
        'negatives_total': total_negatives,
        'accidental_past_positive_negatives': accidental,
        'fraction': round(accidental / total_negatives, 8) if total_negatives else 0.0,
        'samples_with_accidental': samples_with_accidental,
        'samples_with_accidental_fraction': round(samples_with_accidental / len(candidates), 8) if len(candidates) else 0.0,
        'by_regime': {name: regime_accidental[name] for name in REGIMES},
    }


def history_check(metadata_path, arrays, id_to_row, events):
    errors = dict(target_mismatch=0, trace_mismatch=0, nonpast_history=0)
    with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(arrays['candidates']):
        return {'passed': False, 'error': 'Metadata/sample length mismatch'}
    for i, row in enumerate(rows):
        a, b = arrays['history_indptr'][i:i+2]
        history = arrays['history_indices'][a:b].tolist()
        errors['target_mismatch'] += int(id_to_row[row['parent_asin']] != arrays['candidates'][i,0])
        trace = [id_to_row[item] for item in row.get('history','').split()]
        errors['trace_mismatch'] += int(trace != history or int(row['history_length']) != len(history))
        past = known_before(events, row['user_id'], int(row['timestamp']))
        errors['nonpast_history'] += int(any(item not in past for item in history))
    return dict(passed=not any(errors.values()), samples=len(rows), **errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--evaluation', type=Path, default=None,
                        help='Directory holding samples_{split}.npz/.csv.gz.')
    parser.add_argument('--prior-interactions', type=Path, nargs='+', default=[],
                        help='interactions CSVs (train; train+validation) used for the '
                             'beyond-history-window past-positive check.')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'reports/candidate_validity_v1.json')
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['validation'])
    args = parser.parse_args()
    evaluation = args.evaluation or args.data / 'evaluation'
    if args.output.exists():
        raise FileExistsError('Choose a new audit output; prior evidence is immutable.')
    ids, id_to_row, train_counts, validation_counts, test_counts = load_catalog(args.data)
    sampling_path = evaluation / 'sampling_report.json'
    sampling = json.loads(sampling_path.read_text(encoding='utf-8')) if sampling_path.exists() else {}

    n_items = len(ids)
    sources = {
        'items': args.data / 'content/items.csv.gz',
        'item_statistics': args.data / 'item_statistics.csv.gz',
        'manifest': args.data / 'split_manifest.json',
    }
    report = {'data': str(args.data), 'n_items': n_items,
              'sources': {name: sha256(path) for name, path in sources.items()},
              'checks': {}}
    for split in args.splits:
        arrays_path = evaluation / f'samples_{split}.npz'
        metadata_path = evaluation / f'samples_{split}.csv.gz'
        if not arrays_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f'Missing {split} samples or metadata under {evaluation}')
        report['sources'][f'samples_{split}'] = sha256(arrays_path)
        report['sources'][f'metadata_{split}'] = sha256(metadata_path)
        with np.load(arrays_path) as source:
            arrays = {name: source[name] for name in source.files}
        split_config = sampling.get('splits', {}).get(split, {})
        negative_count = split_config.get('candidates_per_sample', NEGATIVES + 1) - 1
        strict_past = split_config.get('exclude_known_positives', sampling.get('exclude_known_positives', False))
        split_counts = (train_counts + validation_counts if split == 'validation'
                        else train_counts + validation_counts + test_counts)
        report['checks'][f'{split}_structure'] = structure_check(
            arrays['candidates'], arrays['regime_codes'], arrays['history_indices'],
            arrays['history_indptr'], arrays.get('repeat_purchases'), n_items, negative_count)
        if not report['checks'][f'{split}_structure']['passed']:
            continue
        report['checks'][f'{split}_exclusions'] = exclusion_check(
            arrays['candidates'], arrays['history_indices'], arrays['history_indptr'])
        report['checks'][f'{split}_regimes'] = regime_check(
            arrays['candidates'], arrays['regime_codes'], train_counts, split_counts, n_items)
        report['checks'][f'{split}_replay'] = replay_check(
            split, arrays, ids, id_to_row, train_counts, validation_counts, test_counts,
            data=args.data, min_interactions=sampling.get('min_interactions_filter', 0),
            cap=sampling.get('samples_per_regime_cap', SAMPLES_PER_REGIME),
            negatives=negative_count, seed=sampling.get('seed', SEED),
            exclude_known_positives=strict_past)
        report['checks'][f'{split}_future_positives'] = future_positive_check(
            split, metadata_path, arrays, id_to_row,
            extra_paths=[], data=args.data)
        # Future-positive disclosure is confined to the selected split; do not open test labels during validation audit.
        past_paths = list(dict.fromkeys([*args.prior_interactions, *prior_paths(args.data, split)]))
        if all(path.exists() for path in past_paths):
            for path in past_paths:
                report['sources']['prior_' + path.name] = sha256(path)
            with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as source:
                users = {row['user_id'] for row in csv.DictReader(source)}
            events = positive_events(past_paths, id_to_row, users)
            check = past_positive_check(metadata_path, arrays, id_to_row, past_paths, events=events)
            report['checks'][f'{split}_history_temporal'] = history_check(metadata_path, arrays, id_to_row, events)
            check['policy'] = 'must_be_zero' if strict_past else 'legacy_disclosure'
            if strict_past:
                check['passed'] = check['accidental_past_positive_negatives'] == 0
            report['checks'][f'{split}_past_positives'] = check
        else:
            report['checks'][f'{split}_past_positives'] = {
                'passed': False if strict_past else None, 'skipped': 'Missing prior interactions',
                'missing': [str(path) for path in past_paths if not path.exists()]}
    failed = [name for name, check in report['checks'].items() if check.get('passed') is False]
    report['complete'] = all(check.get('passed') is not None for check in report['checks'].values())
    report['passed'] = not failed and report['complete']
    report['failed_checks'] = failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'passed': report['passed'], 'failed': failed,
                      'output': str(args.output)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
