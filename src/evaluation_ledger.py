"""Durable, locked history of held-out test access across output filenames."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from common import sha256, write_json


class TestEvaluationLedger:
    __test__ = False

    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def edit(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + '.lock')
        try:
            handle = lock.open('x', encoding='utf-8')
        except FileExistsError as error:
            raise RuntimeError('Evaluation ledger is locked; inspect the active run before retrying.') from error
        try:
            with handle:
                handle.write(datetime.now(timezone.utc).isoformat())
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding='utf-8'))
                payload = {'version': 1, 'entries': payload} if isinstance(payload, list) else payload
            else:
                payload = {'version': 1, 'entries': []}
            if payload.get('version') != 1 or not isinstance(payload.get('entries'), list):
                raise ValueError('Unknown evaluation ledger schema; refusing to discard history.')
            yield payload['entries']
            write_json(self.path, payload)
        finally:
            lock.unlink(missing_ok=True)

    def import_report(self, report_path, audit_path=None):
        report_path = Path(report_path)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if 'test' not in report.get('splits', {}):
            raise ValueError('Imported report has no test evaluation.')
        audit = json.loads(Path(audit_path).read_text(encoding='utf-8')) if audit_path else {}
        fingerprint = sha256(report_path)
        with self.edit() as entries:
            if any(e.get('evidence_sha256') == fingerprint for e in entries):
                return
            entries.append({
                'id': uuid.uuid4().hex, 'status': 'imported',
                'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                'executed_at_utc': None, 'first_evaluation': None,
                'checkpoint_sha256': report['checkpoint_sha256'],
                'samples_test_sha256': report['sources']['samples_test'],
                'dataset_sha256': audit.get('sources', {}).get('manifest'),
                'training_code_sha256': report.get('recorded_code_sha256'),
                'scorer_sha256': None,
                'evidence': str(report_path), 'evidence_sha256': fingerprint,
                'test_metrics': report['splits']['test']['overall'],
                'purpose': 'Imported historical evidence; execution time and prior history unverified.',
            })

    @contextmanager
    def evaluation(self, identity, *, allow_seen=False, purpose=None):
        with self.edit() as entries:
            prior = [e for e in entries if
                     e.get('samples_test_sha256') == identity['samples_test_sha256'] or
                     (identity.get('dataset_sha256') and
                      e.get('dataset_sha256') == identity['dataset_sha256'])]
            if prior and not allow_seen:
                raise ValueError('This test split already has recorded access; changing checkpoint/output does not make it unseen.')
            if allow_seen and not purpose:
                raise ValueError('A purpose is required for an exploratory seen-test evaluation.')
            record = dict(identity, id=uuid.uuid4().hex, status='started',
                          observed_at_utc=datetime.now(timezone.utc).isoformat(),
                          first_evaluation=False if prior else None,
                          prior_access_count=len(prior),
                          purpose=purpose or 'Evaluation with no prior record found; independence not established.')
            entries.append(record)
        try:
            yield record
        except BaseException:
            self._finish(record, 'failed_after_reservation')
            raise
        else:
            self._finish(record, 'completed')

    def _finish(self, record, status):
        record['status'] = status
        record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        with self.edit() as entries:
            for index, entry in enumerate(entries):
                if entry.get('id') == record['id']:
                    entries[index] = record
                    break
            else:
                raise ValueError('Reserved evaluation record disappeared.')


def import_known_evidence(root, ledger):
    folder = Path(root) / 'reports/ufm_evaluation_v2_20261027'
    report = folder / 'reeval_ufm_full_v1_v2.json'
    audit = folder / 'candidate_validity_v2.json'
    if report.exists():
        ledger.import_report(report, audit if audit.exists() else None)
