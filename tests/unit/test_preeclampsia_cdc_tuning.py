"""Small end-to-end checks for 2023 tuning and temporal refit."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.training import preeclampsia_cdc_tuning as tuning
from src.training import preeclampsia_cdc as training
from src.data.preeclampsia_cdc import FEATURES


class CdcHdpTuningTests(unittest.TestCase):
    def test_runtime_version_change_invalidates_tuning_cache(self):
        with patch.object(tuning, "run_fingerprint", return_value="same-data"), \
             patch.object(tuning, "runtime_versions", return_value={"scikit-learn": "1.8"}):
            older = tuning.tuning_fingerprint()
        with patch.object(tuning, "run_fingerprint", return_value="same-data"), \
             patch.object(tuning, "runtime_versions", return_value={"scikit-learn": "1.9"}):
            newer = tuning.tuning_fingerprint()
        self.assertNotEqual(older, newer)

    def test_real_estimators_apply_candidate_parameters(self):
        rng = np.random.default_rng(42)
        n = 80
        frame = pd.DataFrame({
            "mager": rng.integers(18, 40, n).astype(float),
            "bmi": rng.uniform(19, 36, n),
            "prior_live_birth_count_cat": rng.integers(0, 4, n).astype(float),
            "priorterm": rng.integers(0, 3, n).astype(float),
            "rf_pdiab": rng.integers(0, 2, n),
            "rf_ppterm": rng.integers(0, 2, n),
            "rf_cesar": rng.integers(0, 2, n),
            "bmi_missing": np.zeros(n),
            "prior_live_birth_count_missing": np.zeros(n),
            "priorterm_missing": np.zeros(n),
            "target_hdp_proxy": np.tile([0, 1], n // 2),
        })
        self.assertTrue(set(FEATURES).issubset(frame))
        candidates = {
            "logistic": {"C": 0.1},
            "logistic_spline": {"C": 0.3, "n_knots": 4},
            "hist_gb": {"max_leaf_nodes": 7, "l2_regularization": 1.0,
                        "min_samples_leaf": 5},
            "catboost": {"depth": 4, "l2_leaf_reg": 3.0,
                         "iterations": 8, "thread_count": 1},
            "lightgbm": {"num_leaves": 7, "min_child_samples": 5,
                         "n_estimators": 8, "n_jobs": 1},
        }
        with patch.object(training, "HGB_STAGES", (2, 4)):
            for name, params in candidates.items():
                with self.subTest(model=name):
                    model = training._fit_model(name, frame, "smoke", params)
                    predicted = training._predict_model(name, model, frame)
                    self.assertEqual(len(predicted), n)
                    self.assertTrue(np.isfinite(predicted).all())
                    self.assertTrue(((0 <= predicted) & (predicted <= 1)).all())
                    if name == "logistic":
                        self.assertEqual(model[-1].C, 0.1)
                    elif name == "logistic_spline":
                        self.assertEqual(model[-1].C, 0.3)
                        self.assertEqual(model[0].named_transformers_["splines"]["basis"].n_knots, 4)
                    elif name == "hist_gb":
                        self.assertEqual(model.max_leaf_nodes, 7)
                        self.assertEqual(model.l2_regularization, 1.0)
                    else:
                        self.assertEqual(model.get_params()["depth" if name == "catboost" else "num_leaves"],
                                         4 if name == "catboost" else 7)

    def test_search_uses_2023_only_then_refits_and_rejects_stale_scores(self):
        rows = [{"source_year": year, "source_row": row, "fold": row % 5,
                 "target_hdp_proxy": row // 5}
                for year in (2023, 2024) for row in range(10)]
        data = pd.DataFrame(rows)
        grid = {name: [tuning.BASELINE_PARAMS[name]] for name in tuning.MODEL_NAMES}
        grid["logistic"] = [{"C": 0.1}, {"C": 1.0}]
        fitted_years = []
        fit_labels = []
        predicted_years = []
        stop_temporal_after_two = [True]

        def fake_fit(name, train, label, overrides):
            if label.startswith("Selected 2023") and stop_temporal_after_two[0] and sum(
                item.startswith("Selected 2023") for item in fit_labels) == 2:
                raise RuntimeError("simulated interrupted temporal fit")
            fitted_years.append(set(train.source_year.unique()))
            fit_labels.append(label)
            return {"model": name, "params": overrides}

        def fake_predict(name, model, frame):
            predicted_years.append(set(frame.source_year.unique()))
            return np.full(len(frame), .3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(tuning, "TUNING_DIR", root / "tuning"), \
                 patch.object(tuning, "MODEL_DIR", root / "models"), \
                 patch.object(tuning, "GRID", grid), \
                 patch.object(tuning, "_input_data", return_value=data), \
                 patch.object(tuning, "tuning_fingerprint", return_value="test"), \
                 patch.object(tuning, "_fit_model", side_effect=fake_fit), \
                 patch.object(tuning, "_predict_model", side_effect=fake_predict):
                folds, summary, best = tuning.run_grid_search()
                self.assertEqual(len(folds), 5 * sum(map(len, grid.values())))
                self.assertEqual(len(summary), sum(map(len, grid.values())))
                self.assertTrue(folds.source_year.eq(2023).all())
                self.assertTrue(all(years == {2023} for years in fitted_years))
                self.assertTrue(all(years == {2023} for years in predicted_years))
                self.assertEqual(set(best["best_by_model"]), set(grid))
                self.assertEqual(len(tuning.selected_oof_predictions()), 10)

                n_fits = len(fitted_years)
                tuning.run_grid_search(resume=True)
                self.assertEqual(len(fitted_years), n_fits)

                with self.assertRaisesRegex(RuntimeError, "interrupted temporal fit"):
                    tuning.score_selected_on_2024()
                stop_temporal_after_two[0] = False
                temporal = tuning.score_selected_on_2024()
                self.assertEqual(len(temporal), len(grid))
                self.assertEqual(sum(label.startswith("Selected 2023")
                                     for label in fit_labels), len(grid) - 1)
                self.assertTrue(all(years == {2023} for years in fitted_years[n_fits:]))
                self.assertTrue(all(years == {2024} for years in predicted_years[n_fits:]))

                summary_path = tuning.TUNING_DIR / "candidate_summary.csv"
                original_summary = summary_path.read_bytes()
                summary_path.write_bytes(original_summary + b"\n")
                with self.assertRaisesRegex(ValueError, "saved grid summary"):
                    tuning.load_best_configurations()
                summary_path.write_bytes(original_summary)

                score_path, _ = tuning._paths("logistic",
                    tuning.candidate_id("logistic", {"C": 1.0}), 0)
                stale = pd.read_parquet(score_path)
                stale.loc[0, "source_row"] = 999
                stale.to_parquet(score_path, index=False)
                with self.assertRaisesRegex(ValueError, "Corrupt or stale tuning score hash"):
                    tuning.run_grid_search(resume=True)
                _, sidecar = tuning._paths("logistic",
                    tuning.candidate_id("logistic", {"C": 1.0}), 0)
                metadata = json.loads(sidecar.read_text())
                metadata["score_sha256"] = tuning._file_sha256(score_path)
                sidecar.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "Corrupt or stale tuning scores"):
                    tuning.run_grid_search(resume=True)


if __name__ == "__main__":
    unittest.main()
