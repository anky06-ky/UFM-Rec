"""A multi-seed summary must not silently reuse a checkpoint from another seed."""
import json
from pathlib import Path
import sys

import pytest

from scripts import summarize_locked_ufm_followup as summary_script


def test_summary_rejects_checkpoint_seed_mismatch(tmp_path, monkeypatch):
    protocol = json.loads(summary_script.MANIFEST.read_text(encoding='utf-8'))
    sha = protocol['sources_sha256']
    source_hashes = {Path(name).name: value for name, value in sha.items() if name.startswith('src/')}
    report_dir = tmp_path / 'reports'
    report_dir.mkdir()
    for stage in protocol['stages']:
        for variant in stage['variants']:
            for seed in protocol['seeds']:
                report = {
                    'test_evaluated': False,
                    'checkpoint_variant': variant,
                    'checkpoint_seed': seed,
                    'checkpoint_sha256': f'{variant}-{seed}',
                    'model_config': {'variant': variant},
                    'recorded_code_sha256': source_hashes,
                    'sources': {
                        'samples_validation': sha['reports/fix_1_15_20261027/fitlab01_finish1/evaluation_v3/samples_validation.npz'],
                        'features_complete': sha['data/processed/toys_games_full_temporal/foundation_clip_b32_v1/complete.json'],
                        'train_counts': sha['data/processed/gpu_cache_h20_t64_v1/train_counts.npy'],
                    },
                    'splits': {'validation': {
                        'overall': {'samples': 40000, 'ndcg@10': 0.2},
                        'cold_macro_ndcg@10': 0.2,
                        'by_regime': {regime: {'samples': 10000, 'ndcg@10': 0.2}
                                      for regime in summary_script.REGIMES},
                        'known_user': {'ndcg@10': 0.2},
                        'empty_history': {'ndcg@10': 0.2},
                    }},
                }
                (report_dir / f'{variant}_s{seed}_v3.json').write_text(json.dumps(report), encoding='utf-8')
    output = tmp_path / 'valid.json'
    monkeypatch.setattr(sys, 'argv', ['summary', '--reports', str(report_dir), '--output', str(output)])
    summary_script.main()
    assert len(json.loads(output.read_text())['variants']) == 5

    wrong = report_dir / 'full_s7_v3.json'
    report = json.loads(wrong.read_text(encoding='utf-8'))
    report['checkpoint_seed'] = 42
    wrong.write_text(json.dumps(report), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['summary', '--reports', str(report_dir),
                                      '--output', str(tmp_path / 'invalid.json')])
    with pytest.raises(ValueError, match='Protocol mismatch'):
        summary_script.main()
