"""Temporal audit regressions with hand-calculated positive timestamps."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_candidate_validity as audit
import build_recommender_samples as builder
from test_sample_builder import toy_dataset


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_past_positives_excludes_recorded_history_and_future_labels(tmp_path):
    metadata = tmp_path/'samples.csv.gz'
    write_rows(metadata, [dict(sample_index=0,user_id='u',timestamp=10)])
    prior = tmp_path/'prior.csv.gz'
    write_rows(prior, [dict(user_id='u',parent_asin=k,rating=5,timestamp=t)
                       for k,t in [('a',1),('b',2),('c',11),('d',10)]])
    arrays = dict(candidates=np.array([[4,0,1,2,3]]),regime_codes=np.array([3]),
                  history_indptr=np.array([0,1]),history_indices=np.array([0]))
    result = audit.past_positive_check(metadata,arrays,dict(a=0,b=1,c=2,d=3),[prior])
    assert result['accidental_past_positive_negatives'] == 1
    assert result['negatives_total'] == 4
    assert result['fraction'] == .25


def test_replay_accepts_population_below_cap_and_history_filter(toy_dataset, tmp_path):
    ids, lookup, train, val, test = builder.load_catalog()
    output=tmp_path/'samples'
    builder.build_split('validation',ids,lookup,train,val,test,output=output,cap=100,negatives=3)
    with np.load(output/'samples_validation.npz') as source:
        arrays={k:source[k] for k in source.files}
    result = audit.replay_check('validation',arrays,ids,lookup,train,val,test,
                               data=toy_dataset,cap=100,negatives=3)
    assert result['passed']
    empty = audit.replay('validation',ids,lookup,train,val,test,
                         data=toy_dataset,cap=100,negatives=3,min_interactions=2)
    assert empty[0].shape == (0,4)


def test_structure_reports_bad_indptr_instead_of_crashing():
    result=audit.structure_check(np.array([[0,1]],dtype=np.int32),np.array([0]),
                                  np.array([0]),np.array([0,99]),np.array([0]),2,1)
    assert result['passed'] is False


def test_strict_sampler_excludes_old_positives_but_not_future_ones(toy_dataset,tmp_path):
    prior=toy_dataset/'interactions_train.csv.gz'
    write_rows(prior,[dict(user_id='u0',parent_asin='item1',rating=5,timestamp=1),
                      dict(user_id='u0',parent_asin='item2',rating=5,timestamp=2000)])
    ids,lookup,train,val,test=builder.load_catalog()
    output=tmp_path/'strict'
    builder.build_split('validation',ids,lookup,train,val,test,output=output,
                        data=toy_dataset,cap=10,negatives=7,exclude_known_positives=True)
    with np.load(output/'samples_validation.npz') as source:
        row=source['candidates'][0]
    assert 1 not in row[1:]
    assert len(row)==8


def test_cli_uses_requested_dataset_without_mutating_module_globals(toy_dataset,tmp_path):
    write_rows(toy_dataset/'interactions_train.csv.gz', [dict(user_id='other',parent_asin='item0',rating=5,timestamp=1)])
    ids,lookup,train,val,test=builder.load_catalog()
    output=tmp_path/'cli_samples'
    split=builder.build_split('validation',ids,lookup,train,val,test,output=output,negatives=3)
    (output/'sampling_report.json').write_text(json.dumps(dict(seed=42,samples_per_regime_cap=10,
        splits={'validation':split})),encoding='utf-8')
    (toy_dataset/'split_manifest.json').write_text('{}')
    report=tmp_path/'audit.json'
    result=subprocess.run([sys.executable,str(Path(audit.__file__)),'--data',str(toy_dataset),
          '--evaluation',str(output),'--output',str(report)],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    result=json.loads(report.read_text())
    assert result['checks']['validation_replay']['passed']
    assert result['data']==str(toy_dataset)
    assert not any(k.startswith('test_') for k in result['checks'])


def test_history_rejects_simultaneous_and_future_events(tmp_path):
    metadata=tmp_path/'trace.csv.gz'
    write_rows(metadata,[dict(sample_index=0,user_id='u',parent_asin='a',timestamp=10,
                              history='b c',history_length=2)])
    arrays=dict(candidates=np.array([[0,3]]),history_indptr=np.array([0,2]),history_indices=np.array([1,2]))
    events={'u':{1:10,2:11}}
    result=audit.history_check(metadata,arrays,dict(a=0,b=1,c=2),events)
    assert result['nonpast_history']==1
    assert result['passed'] is False
