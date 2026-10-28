"""Archived sources remain usable after relocation; hash guards stay mandatory."""
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from common import sha256
from trained_ufm import model_classes

SRC = Path(__file__).parents[1] / 'src'


def test_verified_archived_model_loads_without_overwriting_current_code(tmp_path):
    names = ['ufm_model.py', 'train_ufm_recommender.py', 'train_gpu_recommender.py']
    for name in names:
        shutil.copy2(SRC / name, tmp_path / name)
    config = dict(code_sha256={name:sha256(SRC / name) for name in names})
    Config, Model = model_classes(config, tmp_path)
    model = Model(Config(items=3, feature_dim=4, dim=4, heads=1, layers=1))
    assert model.config.items == 3
    (tmp_path / 'train_gpu_recommender.py').write_text('# changed after training')
    with pytest.raises(ValueError, match='does not match'):
        model_classes(config, tmp_path)


def test_source_guard_cannot_be_avoided_by_missing_or_traversal_fingerprint(tmp_path):
    with pytest.raises(ValueError, match='fingerprint'):
        model_classes({}, tmp_path)
    with pytest.raises(ValueError, match='filename'):
        model_classes({'code_sha256':{'../ufm_model.py':'bad','ufm_model.py':'bad'}}, tmp_path)
