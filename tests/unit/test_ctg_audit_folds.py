import numpy as np
import pandas as pd
import pytest

from src.training.ctg_audit_folds import balanced_group_folds, fold_inventory


def sample_frame():
    rows = []
    # The positive outlier recording cannot be divided; other recordings supply
    # independent positive support rather than merely matching row prevalence.
    for group in range(45):
        label = group % 3 + 1
        for _ in range(20 if group == 2 else 2):
            rows.append({'source_file': f'f{group}', 'NSP': label,
                         'feature_duplicate_group': f'd{group}'})
    return pd.DataFrame(rows)


def test_deterministic_group_folds_balance_recording_support():
    frame = sample_frame()
    folds = balanced_group_folds(frame, 5, 42)
    again = balanced_group_folds(frame, 5, 42)
    np.testing.assert_array_equal(np.sort(np.concatenate([hold for _, hold in folds])), np.arange(len(frame)))
    for (fit, hold), (fit2, hold2) in zip(folds, again):
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(hold, hold2)
        assert set(frame.iloc[fit].source_file).isdisjoint(frame.iloc[hold].source_file)
        assert set(frame.iloc[hold].NSP) == {1, 2, 3}
    inventory = fold_inventory(frame, folds)
    assert inventory.path_files.min() >= 2
    assert inventory.path_files.max() - inventory.path_files.min() <= 1
    assert inventory.n_hold.sum() == len(frame)


def test_split_does_not_consult_predictors_or_dataframe_index():
    frame = sample_frame()
    changed = frame.assign(LB=np.arange(len(frame)), junk='arbitrary')
    changed.index = np.arange(1000, 1000 + len(frame))
    for first, second in zip(balanced_group_folds(frame, 3, 7), balanced_group_folds(changed, 3, 7)):
        np.testing.assert_array_equal(first[1], second[1])


def test_rejects_insufficient_independent_class_support_and_cross_file_duplicates():
    frame = sample_frame()
    with pytest.raises(ValueError, match='distinct recordings'):
        balanced_group_folds(frame[frame.source_file != 'f2'], 15, 1)
    frame.loc[0, 'feature_duplicate_group'] = frame.loc[2, 'feature_duplicate_group']
    with pytest.raises(ValueError, match='duplicates span'):
        balanced_group_folds(frame, 3, 1)
