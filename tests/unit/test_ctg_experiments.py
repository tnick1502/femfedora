import numpy as np
import pandas as pd
import pytest

from src.training.ctg_experiments import HeadCalibrator, decisions, heads_to_prob, grouped_folds


def test_hierarchy_probabilities_use_product_not_hard_gates():
    h = np.array([[.6, .25], [.2, .8]])
    p = heads_to_prob(h, 'D_hierarchy')
    np.testing.assert_allclose(p, [[.4, .45, .15], [.8, .04, .16]])
    np.testing.assert_allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize('family,width', [('C_ovr', 3), ('D_hierarchy', 2), ('A_multiclass', 3)])
def test_calibrator_returns_probability_simplex(family, width):
    rng = np.random.default_rng(6)
    y = np.tile([0, 1, 2], 20)
    h = rng.uniform(.01, .99, size=(60, width))
    p = HeadCalibrator(family).fit(h, y).predict_proba(h[:5])
    assert (p > 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_pathological_threshold_overrides_only_above_cutoff():
    p = np.array([[.4, .25, .35], [.1, .6, .3]])
    np.testing.assert_array_equal(decisions(p, .32), [2, 1])


def test_grouped_folds_do_not_share_records():
    data = pd.DataFrame({'NSP': np.tile([1, 2, 3], 20), 'source_file': np.repeat(np.arange(30), 2),
                         'feature_duplicate_group': np.repeat(np.arange(30), 2)})
    folds = grouped_folds(data, 3, 7)
    assert sum(len(v) for _, v in folds) == len(data)
    for fit, hold in folds:
        assert set(data.iloc[fit].source_file).isdisjoint(data.iloc[hold].source_file)
