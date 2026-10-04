import unittest
from unittest.mock import patch

import numpy as np

from src.evaluation.preterm_woman_inputs import _serving_eligible
from src.inference.preterm_woman import WEEK_BAND, WEEKS, predict_woman, snapshot_features


class FixedModel:
    def predict_proba(self, frame):
        return np.array([[.05, .03, .07, .85]])


class InvalidModel:
    def predict_proba(self, frame):
        return np.array([[.4, .6]])


class PretermWomanTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "gestational_week": 28, "gestational_day": 0, "age_at_due_date_years": 30,
            "singleton_confirmed": True, "pregnancy_ongoing": True, "dating_confirmed": True,
            "prepregnancy_bmi": 24, "previous_live_births": 1, "previous_terminations": 0,
            "previous_preterm_birth": False, "chronic_hypertension": False,
            "preexisting_diabetes": False, "previous_cesarean": False,
            "prepregnancy_cigarettes_per_day": 0,
        }
        weights = np.zeros((10, len(WEEKS)), dtype=float)
        for band in range(3):
            weights[:, WEEK_BAND == band] = 1 / (WEEK_BAND == band).sum()
        self.profile = {"temperature": 1, "score_decile_boundaries": list(np.linspace(.01, .09, 9)),
                        "version": "test-profile"}
        self.weights = weights

    def test_unknown_field_and_out_of_source_range_are_rejected(self):
        unknown = predict_woman({**self.snapshot, "unsupported_field": 1})
        self.assertEqual(unknown["status"], "OUT_OF_SCOPE")
        self.assertNotIn("probabilities", unknown)
        out_of_range = predict_woman({**self.snapshot, "prepregnancy_bmi": 12})
        self.assertEqual(out_of_range["status"], "OUT_OF_SCOPE")

    def test_missing_prior_history_is_not_false(self):
        feature = snapshot_features({**self.snapshot, "previous_preterm_birth": None})
        self.assertTrue(np.isnan(feature.previous_preterm_birth.iat[0]))
        self.assertEqual(feature.previous_preterm_birth_missing.iat[0], 1)
        result = predict_woman({**self.snapshot, "previous_preterm_birth": None})
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertNotIn("probabilities", result)

    def test_unfilled_fields_do_not_become_source_missingness_risk(self):
        for absent in (("prepregnancy_bmi",),
                       ("prepregnancy_bmi", "previous_live_births", "previous_terminations",
                        "previous_preterm_birth", "chronic_hypertension", "preexisting_diabetes",
                        "previous_cesarean", "prepregnancy_cigarettes_per_day")):
            result = predict_woman({**self.snapshot, **{name: None for name in absent}})
            self.assertEqual(result["status"], "INSUFFICIENT_DATA")
            self.assertNotIn("probabilities", result)

    def test_inconsistent_or_unsupported_history_abstains(self):
        cases = (
            {"previous_live_births": 0, "previous_preterm_birth": True},
            {"previous_live_births": 23},
            {"previous_live_births": 60},
            {"age_at_due_date_years": 12},
            {"age_at_due_date_years": 50},
            {"previous_terminations": 30},
        )
        for changes in cases:
            result = predict_woman({**self.snapshot, **changes})
            self.assertEqual(result["status"], "OOD", changes)
            self.assertNotIn("probabilities", result)

    def test_age_is_integer_at_expected_delivery_date(self):
        for age in (30.0, 39.5, 39.55):
            result = predict_woman({**self.snapshot, "age_at_due_date_years": age})
            self.assertEqual(result["status"], "OUT_OF_SCOPE")
        self.assertEqual(predict_woman({**self.snapshot, "maternal_age_years": 30})["status"],
                         "OUT_OF_SCOPE")

    def test_evaluation_subset_matches_individual_input_rules(self):
        cases = ({}, {"prepregnancy_bmi": None}, {"previous_live_births": 23},
                 {"age_at_due_date_years": 12},
                 {"previous_live_births": 0, "previous_preterm_birth": True})
        for changes in cases:
            values = {**self.snapshot, **changes}
            eligible = bool(_serving_eligible(snapshot_features(values))[0])
            predicted = predict_woman(values)["status"] == "OK"
            self.assertEqual(eligible, predicted, changes)

    def test_conditioned_week_forecast_is_coherent(self):
        with patch("src.inference.preterm_woman._model_and_profile",
                   return_value=(FixedModel(), self.profile, self.weights)):
            result = predict_woman(self.snapshot)
            later = predict_woman({**self.snapshot, "gestational_day": 4})
            at_34 = predict_woman({**self.snapshot, "gestational_week": 34})
        self.assertEqual(result["status"], "OK")
        risk = result["probabilities"]
        self.assertLessEqual(risk["before_32_weeks"], risk["before_34_weeks"])
        self.assertLessEqual(risk["before_34_weeks"], risk["before_37_weeks"])
        self.assertAlmostEqual(sum(row["probability"] for row in risk["week_of_birth"]) +
                               risk["at_or_after_37_weeks"], 1)
        self.assertLess(later["probabilities"]["before_37_weeks"], risk["before_37_weeks"])
        self.assertLessEqual(risk["preterm_within_next_4_weeks"], risk["before_37_weeks"])
        self.assertEqual(result["missing_masks"]["previous_preterm_birth_missing"], 0)
        self.assertEqual(at_34["probabilities"]["before_32_weeks"], 0)
        self.assertEqual(at_34["probabilities"]["before_34_weeks"], 0)

    def test_out_of_scope_multiple_or_unconfirmed_dating(self):
        self.assertEqual(predict_woman({**self.snapshot, "singleton_confirmed": False})["status"],
                         "OUT_OF_SCOPE")
        self.assertEqual(predict_woman({**self.snapshot, "dating_confirmed": False})["status"],
                         "INSUFFICIENT_DATA")

    def test_missing_model_bundle_fails_closed(self):
        with patch("src.inference.preterm_woman._model_and_profile", side_effect=FileNotFoundError("model")):
            result = predict_woman(self.snapshot)
        self.assertEqual(result["status"], "MODEL_UNAVAILABLE")
        self.assertNotIn("probabilities", result)

    def test_wrong_model_output_is_rejected(self):
        with patch("src.inference.preterm_woman._model_and_profile",
                   return_value=(InvalidModel(), self.profile, self.weights)):
            result = predict_woman(self.snapshot)
        self.assertEqual(result["status"], "MODEL_ERROR")
        self.assertNotIn("probabilities", result)


if __name__ == "__main__":
    unittest.main()
