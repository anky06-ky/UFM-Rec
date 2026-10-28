"""Regression: calibration must keep timestamp ties together and never read test."""
import csv
import gzip
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from calibrate_sampled_validation import main


def setup_validation(root):
    root.mkdir()
    candidates = np.arange(24).reshape(8, 3)
    scores = np.array([[3., 1., 0.], [1., 2., 0.]] * 4)
    np.savez(root/'samples_validation.npz', candidates=candidates,
             regime_codes=np.array([0, 1, 2, 3] * 2))
    with gzip.open(root/'samples_validation.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'timestamp'])
        writer.writerows(enumerate([1, 1, 2, 2, 3, 3, 4, 4]))
    p = root/'predictions.npz'
    np.savez(p, candidates=candidates, scores=scores, sample_indices=np.arange(8))
    # If this file is opened accidentally, NPZ loading must fail.
    (root/'samples_test.npz').write_bytes(b'forbidden test fixture')
    return p, candidates, scores


def test_validation_only_calibration_with_exact_candidate_binding(tmp_path):
    evaluation = tmp_path/'evaluation'
    predictions, _, _ = setup_validation(evaluation)
    output = tmp_path/'calibration.json'
    report = main(['--predictions', str(predictions), '--evaluation', str(evaluation), '--output', str(output)])
    assert report['test_evaluated'] is False
    assert report['fit_samples'] == report['audit_samples'] == 4
    assert report['fit_max_timestamp_ms'] < report['audit_min_timestamp_ms']
    assert report['before']['overall']['n'] == 4
    with pytest.raises(FileExistsError):
        main(['--predictions', str(predictions), '--evaluation', str(evaluation), '--output', str(output)])


def test_scores_for_different_candidates_cannot_be_reused(tmp_path):
    evaluation = tmp_path/'evaluation'
    predictions, candidates, scores = setup_validation(evaluation)
    np.savez(predictions, candidates=candidates[:, ::-1], scores=scores, sample_indices=np.arange(8))
    with pytest.raises(ValueError, match='exact validation candidates'):
        main(['--predictions', str(predictions), '--evaluation', str(evaluation),
              '--output', str(tmp_path/'calibration.json')])
