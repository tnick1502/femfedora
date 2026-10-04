import unittest

import numpy as np
import pandas as pd

from src.data.preterm_cdc import BASE_FEATURES, FEATURES, _uniform_from_source_row, prepare_chunk
from src.training.preterm_cdc import score, temperature_scale


class PretermCdcContractTest(unittest.TestCase):
    def test_labels_and_prior_birth_missingness(self):
        raw = pd.DataFrame({
            "source_row": [10, 11, 12, 13], "mager": [25, 30, 35, 40],
            "bmi": [21.0, np.nan, 28.0, 30.0],
            "priorlive": [0, 1, np.nan, 2], "priordead": [0, 1, 0, 0],
            "priorterm": [0, 1, 0, 1], "rf_ppterm": [0, 1, 0, np.nan],
            "rf_phype": [0, 0, 1, 0], "rf_pdiab": [0, 0, 0, 1],
            "rf_cesar": [0, 1, 0, 1], "cig_0": [0, 0, 5, 0],
            "oegest_comb": [31, 33, 36, 39], "mrace6": [1, 2, 3, 4],
            "meduc": [3, 4, 5, 6],
        })
        frame = prepare_chunk(raw, 2023)
        self.assertEqual(frame.risk_band.tolist(), [0, 1, 2, 3])
        self.assertEqual(frame.previous_live_births.iloc[1], 2)
        self.assertTrue(np.isnan(frame.previous_live_births.iloc[2]))
        self.assertEqual(frame.previous_live_births_missing.iloc[2], 1)
        self.assertEqual(frame.prepregnancy_bmi_missing.iloc[1], 1)
        self.assertTrue(set(frame.partition.unique()) <= {"train", "validation", "calibration"})
        self.assertNotIn("gestational_weeks_at_birth", FEATURES)
        self.assertNotIn("reported_race_code_audit_only", FEATURES)

    def test_partition_is_stable_by_source_row(self):
        ids = np.array([1, 2, 3, 1000], dtype=np.int64)
        self.assertTrue(np.array_equal(_uniform_from_source_row(ids),
                                       _uniform_from_source_row(ids[::-1])[::-1]))

    def test_multiclass_probabilities_remain_nested_after_temperature(self):
        y = np.array([0, 1, 2, 3], dtype=np.int8)
        p = np.array([[.6, .2, .1, .1], [.1, .6, .2, .1],
                      [.1, .2, .6, .1], [.05, .05, .1, .8]])
        scaled = temperature_scale(p, 1.25)
        self.assertTrue(np.allclose(scaled.sum(axis=1), 1))
        self.assertTrue(np.all(scaled[:, 0] <= scaled[:, :2].sum(axis=1)))
        self.assertTrue(np.all(scaled[:, :2].sum(axis=1) <= scaled[:, :3].sum(axis=1)))
        result = score(y, scaled)
        self.assertTrue(np.isfinite(result["multiclass_log_loss"]))
        self.assertEqual(result["n"], 4)


if __name__ == "__main__":
    unittest.main()
