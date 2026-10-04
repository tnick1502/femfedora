"""The NCHS race flag must join to the exact scored birth, not a row order."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation import preeclampsia_cdc_race as race


def test_source_row_mapping_and_imputation_codes():
    chunk = pd.DataFrame({
        "restatus": ["1", "4", "2", "3"], "dplural": ["1"] * 4,
        "rf_phype": ["N", "N", "Y", "N"], "rf_ghype": ["Y", "N", "N", "N"],
        "rf_ehype": ["N"] * 4, "mrace6": ["2", "1", "4", "3"],
        "mraceimp": ["1", "", "", "2"],
    }).astype("string")
    selected = race._selected_race_rows(chunk, 2024, 100)
    assert selected.source_row.tolist() == [100, 103]
    assert selected.mrace6.tolist() == [2, 3]
    assert selected.race_imputed.tolist() == [1, 1]


def test_race_sensitivity_checks_keys_and_published_race(tmp_path):
    path = tmp_path / "race.parquet"
    pd.DataFrame({"source_year": [2023, 2024, 2024], "source_row": [8, 1, 2],
                  "mrace6": [2, 2, 2], "race_imputed": [0, 1, 0]}).to_parquet(path)
    scores = pd.DataFrame({"source_year": [2024, 2023, 2024], "source_row": [2, 8, 1],
                           "mrace6": [2, 2, 2], "label": [1, 0, 0],
                           "hist_gb": [.4, .1, .2]})
    with patch.object(race, "RACE_ROWS", path):
        result = race.race_imputation_sensitivity(scores, ["hist_gb"])
        subset = result.loc[result.year.eq(2024) & result.subset.eq("non_imputed")].iloc[0]
        assert subset.n == subset.events == 1
        assert np.isclose(subset.mean_calibration_gap_pp, 60)
        with pytest.raises(ValueError, match="does not match"):
            race.race_imputation_sensitivity(scores.assign(mrace6=1), ["hist_gb"])
