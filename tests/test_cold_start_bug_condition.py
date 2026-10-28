"""Structural cold-start regressions, not evidence of recommendation quality."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ufm_model import UFMConfig, UFMRec


@pytest.fixture
def setup_model():
    torch.manual_seed(42)
    config = UFMConfig(items=20, feature_dim=32, dim=32, history_size=5,
                       heads=4, layers=2, dropout=0, id_dropout=0)
    model = UFMRec(config).eval()
    text, image = torch.randn(21, 32), torch.randn(21, 32)
    masks = torch.ones(21, 2, dtype=torch.bool)
    counts = torch.zeros(21, dtype=torch.long)
    counts[4], counts[5] = 50, 100
    return config, model, text, image, masks, counts


@pytest.mark.parametrize('count', [0, 1, 5, 6, 20, 21])
def test_regime_boundaries_mask_only_unseen_cf(setup_model, count):
    _, model, text, image, masks, counts = setup_model
    counts[1] = count
    output = model(torch.tensor([[4, 5, 0, 0, 0]]), torch.tensor([[1]]),
                   text, image, masks, counts)
    assert torch.isfinite(output['scores']).all()
    assert bool(output['available'][0, 0, 0]) == (count > 0)
    assert bool(output['available'][0, 0, 1])
    assert output['weights'][0, 0].sum().item() == pytest.approx(1)
    if count == 0:
        assert output['weights'][0, 0].tolist() == [0, 1]


def test_count_penalty_attenuates_cf_without_changing_branch_vectors(setup_model):
    config, full, text, image, masks, counts = setup_model
    counts[1], counts[2], counts[3] = 1, 5, 100
    penalized = UFMRec(replace(config, variant='count_aware_fusion')).eval()
    penalized.load_state_dict(full.state_dict())
    args = (torch.tensor([[4, 5, 0, 0, 0]]), torch.tensor([[1, 2, 3]]),
            text, image, masks, counts)
    with torch.no_grad():
        before, after = full(*args), penalized(*args)
    torch.testing.assert_close(before['cf'], after['cf'])
    torch.testing.assert_close(before['semantic'], after['semantic'])
    assert (after['weights'][0, :2, 0] < before['weights'][0, :2, 0]).all()
    change = after['uncertainty'][0, :, 0] - before['uncertainty'][0, :, 0]
    assert change[0] > change[1] > change[2]


def test_adaptive_dropout_favors_frequent_ids_but_never_unseen(setup_model):
    config, _, _, _, _, counts = setup_model
    counts[1], counts[2] = 1, 100
    model = UFMRec(replace(config, variant='adaptive_dropout', id_dropout=.8)).train()
    ids = torch.tensor([[1, 2, 3]]).repeat(4000, 1)
    _, retained = model.collaborative_items(ids, counts)
    rates = retained.float().mean(0)
    assert rates[1] > rates[0] + .4
    assert not retained[:, 2].any()
    model.eval()
    _, retained = model.collaborative_items(ids[:1], counts)
    assert retained.tolist() == [[True, True, False]]


def test_count_aware_fusion_keeps_full_models_fixed_id_dropout(setup_model):
    config, _, _, _, _, counts = setup_model
    counts[1], counts[2] = 1, 100
    ids = torch.tensor([[1, 2, 3]]).repeat(512, 1)
    masks = {}
    for variant in ('full', 'count_aware_fusion', 'adaptive_dropout'):
        model = UFMRec(replace(config, variant=variant, id_dropout=.8)).train()
        torch.manual_seed(73)
        _, masks[variant] = model.collaborative_items(ids, counts)
    torch.testing.assert_close(masks['full'], masks['count_aware_fusion'])
    assert masks['adaptive_dropout'][:, 1].sum() > masks['full'][:, 1].sum() + 200
    assert not masks['count_aware_fusion'][:, 2].any()


@pytest.mark.parametrize('count', [1, 5])
def test_no_history_uses_popularity_not_personalized_cf(setup_model, count):
    _, model, text, image, masks, counts = setup_model
    counts[1] = count
    result = model(torch.zeros((1, 5), dtype=torch.long), torch.tensor([[1, 4]]),
                   text, image, masks, counts)
    assert not result['available'].any()
    torch.testing.assert_close(result['scores'], counts[torch.tensor([[1, 4]])].float().log1p())
