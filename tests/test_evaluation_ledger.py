import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation_ledger import TestEvaluationLedger


def test_new_output_and_checkpoint_do_not_erase_seen_dataset(tmp_path):
    ledger=TestEvaluationLedger(tmp_path/'ledger.json')
    identity=dict(checkpoint_sha256='a',samples_test_sha256='b',dataset_sha256='dataset')
    with ledger.evaluation(identity) as first:
        assert first['first_evaluation'] is None
    for changed in [dict(identity,output='new.json'),dict(identity,checkpoint_sha256='new'),
                    dict(identity,samples_test_sha256='other_candidates')]:
        with pytest.raises(ValueError,match='already'):
            with ledger.evaluation(changed): pass
    with ledger.evaluation(identity,allow_seen=True,purpose='diagnostic') as seen:
        assert seen['first_evaluation'] is False


def test_failure_and_concurrent_lock_preserve_history(tmp_path):
    ledger=TestEvaluationLedger(tmp_path/'ledger.json')
    identity=dict(samples_test_sha256='b')
    with pytest.raises(RuntimeError):
        with ledger.evaluation(identity): raise RuntimeError('interrupted')
    with pytest.raises(ValueError):
        with ledger.evaluation(identity): pass
    with ledger.edit():
        with pytest.raises(RuntimeError,match='locked'):
            with ledger.edit(): pass
    assert not ledger.path.with_name('ledger.json.lock').exists()


def test_import_is_idempotent_and_does_not_invent_execution_time(tmp_path):
    report=tmp_path/'report.json'
    report.write_text(json.dumps(dict(checkpoint_sha256='model',sources={'samples_test':'test'},
                                     splits={'test':{'overall':{'ndcg@10':.1}}})))
    ledger=TestEvaluationLedger(tmp_path/'ledger.json')
    ledger.import_report(report);ledger.import_report(report)
    entries=json.loads(ledger.path.read_text())['entries']
    assert len(entries)==1
    assert entries[0]['executed_at_utc'] is None
    assert entries[0]['first_evaluation'] is None
    with pytest.raises(ValueError):
        with ledger.evaluation(dict(samples_test_sha256='test')): pass


def test_legacy_entries_without_ids_survive_new_evaluation(tmp_path):
    path = tmp_path / 'ledger.json'
    old = {'samples_test_sha256': 'legacy', 'status': 'imported'}
    path.write_text(json.dumps([old]))
    ledger = TestEvaluationLedger(path)
    with ledger.evaluation(dict(samples_test_sha256='new')):
        pass
    entries = json.loads(path.read_text())['entries']
    assert entries[0] == old
    assert entries[1]['status'] == 'completed'
    with pytest.raises(ValueError, match='already'):
        with ledger.evaluation(dict(samples_test_sha256='legacy')):
            pass
