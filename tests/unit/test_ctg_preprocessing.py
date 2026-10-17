"""Behavior that matters when reusing the saved UCI CTG scaler pipeline."""

import pandas as pd
import pytest

from src.features.ctg_preprocessing import CTG_FEATURES, CTGInputGuard


def valid_ctg_row() -> pd.DataFrame:
    values = {name: 1.0 for name in CTG_FEATURES}
    values.update({"Min": 100.0, "Max": 150.0, "Width": 50.0, "Tendency": 0})
    return pd.DataFrame([values])


def test_guard_accepts_complete_values_and_orders_columns() -> None:
    row = valid_ctg_row()
    result = CTGInputGuard().fit_transform(row[list(reversed(CTG_FEATURES))])
    assert result.columns.tolist() == list(CTG_FEATURES)
    assert result.loc[0, "Width"] == 50.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"FM": None}, "missing values"),
        ({"Tendency": 2}, "Tendency"),
        ({"Width": 51}, "Width"),
        ({"ASTV": 101}, "percentages"),
    ],
)
def test_guard_rejects_unsupported_inputs(change: dict, message: str) -> None:
    row = valid_ctg_row()
    for column, value in change.items():
        row.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        CTGInputGuard().fit_transform(row)


def test_guard_rejects_target_as_feature() -> None:
    row = valid_ctg_row()
    row["NSP"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        CTGInputGuard().fit_transform(row)
