"""Contract tests for the CDC HDP-proxy phenotype, leakage and evaluation."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.prepare_open_datasets import CDC_USECOLS, prepare_cdc_chunk
from src.data.preeclampsia_cdc import (AUDIT, DEMOGRAPHIC_FEATURES, EARLY_FEATURES,
    FEATURES, MISSING_MASKS, OUTCOMES, RAW_INPUTS, clean_frame, cohort_query, split_bucket)
from src.evaluation.preeclampsia_cdc import (calibration_parameters,
    apply_intercept_update, art_reclassification, compare_guideline_at_budget,
    fixed_threshold_from_development, observed_guideline_rule,
    secondary_outcome_metrics, secondary_trained_summary, temporal_subgroup_audit, temporal_summary,
    top_tenth_operating_point)
from src.evaluation.preeclampsia_cdc_features import heldout_permutation_importance
from src.training.preeclampsia_cdc import (logistic_pipeline, predict_logistic,
    validate_core_input, validate_early_input, validate_prediction_artifact,
    MODEL_NAMES, TEMPORAL_SENSITIVITIES)


class CdcHdpContractTests(unittest.TestCase):
    def test_saved_temporal_labels_match_exact_source_rows(self):
        from src.evaluation.preeclampsia_cdc import (
            _validate_probability_columns, _validate_saved_outcomes)
        source = pd.DataFrame({"source_year": [2024] * 4,
            "source_row": [0, 1, 2, 3], "fold": [0, 1, 2, 3],
            "target_hdp_proxy": [0, 1, 0, 1],
            "target_hdp_preterm37": [0, 1, 0, 0],
            "target_hdp_preterm34": [0, 0, 0, 0]})
        scored = source.rename(columns={"target_hdp_proxy": "label"}).copy()
        _validate_saved_outcomes(scored, source, 2024)
        scored.loc[[0, 1], "label"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "target_hdp_proxy differs"):
            _validate_saved_outcomes(scored, source, 2024)
        scored["model"] = [.1, .2, 1.1, .4]
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            _validate_probability_columns(scored, ["model"], "test")

    def test_evaluation_rechecks_fold_scores_and_saved_metrics(self):
        from src.evaluation import preeclampsia_cdc as evaluation
        from src.training.preeclampsia_cdc import metrics
        rows = [{"source_year": year, "source_row": index, "fold": index % 5,
                 "target_hdp_proxy": index % 2, "target_hdp_preterm37": 0.,
                 "target_hdp_preterm34": 0.}
                for year in (2023, 2024) for index in range(10)]
        source = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fold_run.json").write_text(json.dumps({"fingerprint": "test"}))
            (root / "audit.json").write_text(json.dumps({"year_checks": {
                str(year): {"eligible_singleton_rows": 10} for year in (2023, 2024)}}))
            source.groupby(["source_year", "fold"]).target_hdp_proxy.agg(
                rows="size", events="sum").reset_index().rename(
                columns={"source_year": "year"}).to_csv(root / "folds_all.csv", index=False)
            for fold in range(5):
                part = source.loc[source.fold.eq(fold)].rename(
                    columns={"target_hdp_proxy": "label"}).copy()
                for i, name in enumerate(MODEL_NAMES):
                    part[name] = np.float32(.2 + .04 * i + .2 * part.label)
                part.to_parquet(root / f"oof_fold_{fold}.parquet", index=False)
                y = part.label.to_numpy(dtype=np.uint8)
                pd.DataFrame([{"fold": fold, "model": name, "n": len(part),
                               "events": int(y.sum()), **metrics(y, part[name].to_numpy())}
                              for name in MODEL_NAMES]).to_csv(
                    root / f"fold_metrics_{fold}.csv", index=False)
            with patch.object(evaluation, "OUT", root), \
                 patch.object(evaluation, "run_fingerprint", return_value="test"), \
                 patch.object(evaluation, "_verify_and_load", return_value=source):
                result, oof = evaluation.validate_and_summarize()
                self.assertEqual(len(result), 5 * len(MODEL_NAMES))
                self.assertEqual(len(oof), len(source))
                metric_path = root / "fold_metrics_0.csv"
                corrupted = pd.read_csv(metric_path)
                corrupted.loc[0, "auroc"] = .01
                corrupted.to_csv(metric_path, index=False)
                with self.assertRaisesRegex(ValueError, "saved metrics disagree"):
                    evaluation.validate_and_summarize()

    def test_cached_scores_must_match_source_rows_labels_and_reporting_route(self):
        expected = pd.DataFrame({"source_year": [2024, 2024], "source_row": [10, 11],
            "fold": [0, 1], "target_hdp_proxy": [0, 1],
            "target_hdp_preterm37": [0., np.nan],
            "target_hdp_preterm34": [0., np.nan],
            "oegest_comb": [39., np.nan],
            "art_source_reported": [0, 1]})
        scored = expected.drop(columns="art_source_reported").rename(
            columns={"target_hdp_proxy": "label"})
        scored["hist_gb"] = [.1, .8]
        scored["hist_gb_early"] = [.1, .7]
        scored["early_core_fallback"] = [1, 0]
        names = ["hist_gb", "hist_gb_early"]
        validate_prediction_artifact(scored, expected, names, name="test")
        for column, value in (("source_row", 12), ("label", 1),
                              ("oegest_comb", 38.),
                              ("hist_gb", np.nan), ("hist_gb_early", -0.1),
                              ("early_core_fallback", 0)):
            corrupted = scored.copy()
            corrupted.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_prediction_artifact(corrupted, expected, names, name="test")

    def test_early_input_rejects_unreported_or_inconsistent_art(self):
        frame = pd.DataFrame({"rf_inftr": [1., np.nan], "rf_artec": [1., np.nan],
            "f_rf_inft": [1, 0], "f_rf_inf_art": [1, 0],
            "art_source_reported": [1, 0], "art_status_unknown_reported": [0, 0]})
        validate_early_input(frame)
        invalid = frame.copy()
        invalid.loc[1, "rf_artec"] = 0
        with self.assertRaisesRegex(ValueError, "Unreported ART"):
            validate_early_input(invalid)
        invalid = frame.copy()
        invalid.loc[0, "rf_inftr"] = 0
        with self.assertRaisesRegex(ValueError, "requires recorded infertility"):
            validate_early_input(invalid)

    def test_feature_contract_forbids_current_outcomes_and_chronic_coding(self):
        forbidden = {"rf_ghype", "rf_ehype", "rf_gdiab", "oegest_comb", "dbwt",
                     "dplural", "rf_phype", "cig_0", *OUTCOMES}
        for group in (FEATURES, EARLY_FEATURES, DEMOGRAPHIC_FEATURES):
            self.assertFalse(forbidden.intersection(group))
        self.assertIn("oegest_comb", AUDIT)
        self.assertNotIn("rf_artec", FEATURES)
        self.assertIn("rf_artec", EARLY_FEATURES)
        self.assertNotIn("mrace6", FEATURES)
        self.assertIn("mrace6", DEMOGRAPHIC_FEATURES)

    def test_source_codes_and_reporting_flags(self):
        base = {key: "N" for key in CDC_USECOLS}
        base.update({"restatus": "1", "rf_phype": "N", "rf_ghype": "N",
                     "rf_ehype": "N", "illb_r": "888", "ilop_r": "999",
                     "ilp_r": "888", "rf_inftr": "N", "rf_artec": "X",
                     "rf_fedrg": "X", "f_rf_inft": "1", "f_rf_inf_art": "1"})
        not_reported = {**base, "f_rf_inft": "0", "f_rf_inf_art": "0",
                        "rf_inftr": "Y", "rf_artec": "Y", "rf_fedrg": "Y"}
        reported_yes = {**base, "rf_inftr": "Y", "rf_artec": "Y", "rf_fedrg": "N"}
        unknown_treatment = {**base, "rf_inftr": "U"}
        prepared = prepare_cdc_chunk(pd.DataFrame([base, not_reported, reported_yes,
                                                    unknown_treatment]), 2024, 0)
        self.assertEqual(prepared.illb_r.iloc[0], 888)
        self.assertTrue(pd.isna(prepared.ilop_r.iloc[0]))
        self.assertEqual(prepared.rf_artec.iloc[0], 0)
        self.assertTrue(pd.isna(prepared.rf_artec.iloc[1]))
        self.assertTrue(pd.isna(prepared.rf_inftr.iloc[1]))
        self.assertEqual(prepared.rf_artec.iloc[2], 1)
        self.assertTrue(pd.isna(prepared.rf_artec.iloc[3]))

    def test_cohort_excludes_chronic_hypertension(self):
        self.assertIn("rf_phype=0", cohort_query(2023))
        self.assertRaises(ValueError, cohort_query, 2022)

    def test_secondary_outcomes_are_composites_with_unknown_gestation(self):
        row = {key: [0, 0, 0] for key in set(RAW_INPUTS + AUDIT)}
        row.update({"mager": [32, 32, 32], "bmi": [25, 25, 25],
                    "priorlive": [0, 0, 0], "priordead": [0, 0, 0],
                    "priorterm": [0, 0, 0], "rf_phype": [0, 0, 0],
                    "f_rf_inft": [1, 1, 1], "f_rf_inf_art": [1, 1, 1],
                    "lbo_rec": [1, 1, 1],
                    "oegest_comb": [36, 33, np.nan],
                    "target_hdp_proxy": [1, 0, 1]})
        result = clean_frame(pd.DataFrame(row))
        self.assertEqual(result.target_hdp_preterm37.iloc[0], 1)
        self.assertEqual(result.target_hdp_preterm34.iloc[0], 0)
        self.assertEqual(result.target_hdp_preterm34.iloc[1], 0)
        self.assertTrue(np.isnan(result.target_hdp_preterm37.iloc[2]))
        self.assertEqual(result.prior_live_birth_count_cat.iloc[0], 0)
        validate_core_input(result)
        self.assertRaises(ValueError, clean_frame, pd.DataFrame(row).assign(rf_phype=1))

    def test_missing_masks_keep_unknown_separate_from_zero(self):
        row = {key: [0, 0] for key in set(RAW_INPUTS + AUDIT)}
        row.update({"mager": [30, 30], "bmi": [np.nan, 25],
                    "priorlive": [np.nan, 0], "priordead": [0, 0],
                    "priorterm": [np.nan, 0], "lbo_rec": [np.nan, 1],
                    "rf_cesar": [1, 0], "f_rf_inft": [0, 1],
                    "f_rf_inf_art": [0, 1], "rf_inftr": [np.nan, 0],
                    "rf_artec": [np.nan, 0], "target_hdp_proxy": [0, 1]})
        data = clean_frame(pd.DataFrame(row))
        self.assertTrue(data.loc[0, MISSING_MASKS].eq(1).all())
        self.assertTrue(data.loc[1, MISSING_MASKS].eq(0).all())
        self.assertTrue(np.isnan(data.prior_live_birth_count_cat.iloc[0]))
        self.assertEqual(data.rf_cesar.iloc[0], 1)
        self.assertEqual(data.art_source_reported.tolist(), [0, 1])
        validate_core_input(data)
        with self.assertRaisesRegex(ValueError, "Inconsistent value/mask"):
            validate_core_input(data.assign(bmi_missing=0))

    def test_hash_folds_are_stable_and_year_sensitive(self):
        rows = np.arange(100_000, dtype=np.uint64)
        first = split_bucket(rows)
        np.testing.assert_array_equal(first, split_bucket(rows.copy()))
        self.assertTrue((np.bincount(first % 5, minlength=5) > 19_000).all())
        self.assertFalse(np.array_equal(first,
            split_bucket((np.uint64(2024) << np.uint64(32)) | rows)))

    def test_partial_guideline_rule_preserves_uncertainty(self):
        frame = pd.DataFrame({"rf_pdiab": [1, 0, 0, 0, 0],
                              "mager": [30, 36, 30, np.nan, 30],
                              "bmi": [24, 31, 24, 31, 24],
                              "priorlive": [1, 1, 1, 1, 1],
                              "priordead": [0, 0, 0, 0, 0],
                              "rf_artec": [0, 0, 0, 0, 1]})
        np.testing.assert_array_equal(observed_guideline_rule(frame)[:3], [1, 1, 0])
        self.assertTrue(np.isnan(observed_guideline_rule(frame)[3]))
        self.assertEqual(observed_guideline_rule(frame, include_ivf=True)[4], 0)
        frame.loc[4, "mager"] = 36
        self.assertEqual(observed_guideline_rule(frame, include_ivf=True)[4], 1)

    def test_logistic_linear_and_spline_score_consistently(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({name: rng.normal(size=150) for name in FEATURES})
        for name in ("rf_pdiab", "rf_ppterm", "rf_cesar"):
            frame[name] = rng.integers(0, 2, size=150).astype(float)
        frame["prior_live_birth_count_cat"] = rng.integers(0, 4, size=150).astype(float)
        frame[MISSING_MASKS] = 0
        frame.loc[0, "bmi"] = np.nan
        frame.loc[0, "bmi_missing"] = 1
        frame.loc[1, "priorterm"] = np.nan
        frame.loc[1, "priorterm_missing"] = 1
        target = rng.integers(0, 2, size=150)
        for spline in (False, True):
            model = logistic_pipeline(spline=spline).fit(frame, target)
            np.testing.assert_allclose(model.predict_proba(frame)[:, 1],
                                       predict_logistic(model, frame), rtol=1e-12)

    def test_calibration_offset_is_separate_from_joint_intercept(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(.05, .45, size=2000)
        y = rng.binomial(1, p).astype(np.uint8)
        result = calibration_parameters(y, p)
        from scipy.special import expit, logit
        self.assertAlmostEqual(expit(logit(p) + result["calibration_in_the_large"]).mean(), y.mean(), places=10)
        self.assertIn("joint_calibration_intercept", result)
        self.assertIn("calibration_slope", result)
        np.testing.assert_allclose(apply_intercept_update(p, result["calibration_in_the_large"]).mean(),
                                   y.mean(), atol=1e-10)

    def test_fixed_threshold_uses_development_scores_only(self):
        dev = np.array([.1, .2, .3, .4, .5])
        self.assertAlmostEqual(fixed_threshold_from_development(dev, .2), .42)
        point = top_tenth_operating_point(np.array([0, 1]*50), np.linspace(0,1,100))
        self.assertEqual(point["true_positive"] + point["false_positive"], 10)

    def test_absolute_guideline_counts_use_same_known_denominator(self):
        frame = pd.DataFrame({"rf_pdiab": [0] * 4, "mager": [35, 30, 35, 30],
                              "bmi": [31, 25, 25, 25], "priorlive": [0] * 4,
                              "priordead": [0] * 4, "label": [1, 1, 0, 0],
                              "hist_gb": [.8, .9, .1, .2]})
        comparison = compare_guideline_at_budget(frame).set_index("model")
        self.assertEqual(comparison.loc["observed_guideline_subset", "cases_per_1000"], 500)
        self.assertEqual(comparison.loc["observed_guideline_subset", "captured_per_1000"], 250)
        self.assertEqual(comparison.loc["hist_gb", "captured_per_1000"], 500)
        self.assertEqual(comparison.loc["hist_gb", "alert_budget"], .5)

    def test_cross_target_scores_report_discrimination_not_brier(self):
        frame = pd.DataFrame({"target_hdp_preterm37": [0, 1, 0, 1],
                              "target_hdp_preterm34": [0, 0, 0, 1],
                              "hist_gb": [.1, .4, .2, .8]})
        rows = secondary_outcome_metrics(frame, ["hist_gb"])
        self.assertEqual(len(rows), 2)
        self.assertIn("auroc", rows)
        self.assertNotIn("brier", rows)

    def test_art_subgroup_reports_calibration_and_fixed_budget(self):
        count = 200
        dev = pd.DataFrame({"source_year": 2023, "source_row": np.arange(count),
                            "hist_gb": np.linspace(.02, .30, count),
                            "hist_gb_early": np.linspace(.03, .31, count)})
        test = pd.DataFrame({"source_year": 2024, "source_row": np.arange(count),
                             "label": np.arange(count) % 4 == 0,
                             "hist_gb": np.linspace(.02, .30, count),
                             "hist_gb_early": np.linspace(.03, .31, count),
                             "rf_pdiab": 0, "rf_inftr": np.arange(count) % 2,
                             "rf_artec": np.arange(count) % 2,
                             "art_source_reported": 1,
                             "art_status_unknown_reported": 0,
                             "bmi_missing": 0, "prior_live_birth_count_missing": 0,
                             "priorterm_missing": 0, "mrace6": np.arange(count) % 2 + 1,
                             "mhisp_r": 0})
        test.loc[:9, "art_source_reported"] = 0
        test.loc[:9, "rf_artec"] = np.nan
        test.loc[:9, "hist_gb_early"] = test.loc[:9, "hist_gb"]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.evaluation.preeclampsia_cdc.OUT", Path(temporary)):
                groups = temporal_subgroup_audit(dev, test,
                    models=["hist_gb", "hist_gb_early"])
                moved = art_reclassification(dev, test)
        row = groups.loc[groups.group.eq("race_code_1") & groups.model.eq("hist_gb")].iloc[0]
        self.assertIn("mean_calibration_gap_pp", row)
        self.assertIn("fixed_sensitivity", row)
        fallback = moved.loc[moved.group.eq("ART_not_reported_core_fallback")].iloc[0]
        self.assertEqual(fallback.changed_individual_scores, 0)
        self.assertEqual(fallback.n, 10)

    def test_heldout_permutation_importance_detects_signal(self):
        class FixedSignalModel:
            def predict_proba(self, frame):
                p = 1/(1+np.exp(-3*frame.signal.to_numpy()))
                return np.column_stack([1-p,p])
        rng = np.random.default_rng(3)
        signal = rng.normal(size=500)
        frame = pd.DataFrame({"fold": 0, "target_hdp_proxy": (signal>0).astype(int),
                              "signal": signal, "noise": rng.normal(size=500)})
        result = heldout_permutation_importance(FixedSignalModel(), frame,
            ["signal", "noise"], "hist_gb", 0, repeats=2,
            groups={"signal_group": ["signal"]})
        means = result.groupby("feature").delta_auprc.mean()
        self.assertGreater(means["signal"], means["noise"]+.2)
        self.assertGreater(means["signal_group"], .2)

    def test_importance_moves_a_value_with_its_missingness_mask(self):
        class MaskModel:
            def predict_proba(self, frame):
                score = frame.bmi.fillna(25).to_numpy() / 50 + .1 * frame.bmi_missing.to_numpy()
                return np.column_stack([1-score, score])
        rng = np.random.default_rng(13)
        bmi = rng.uniform(18, 38, size=300)
        bmi[::10] = np.nan
        frame = pd.DataFrame({"fold": 0, "target_hdp_proxy": rng.integers(0, 2, 300),
                              "bmi": bmi, "bmi_missing": np.isnan(bmi).astype(int)})
        result = heldout_permutation_importance(MaskModel(), frame,
            ["bmi", "bmi_missing"], "hist_gb", 0, repeats=1)
        self.assertEqual(result.feature.tolist(), ["bmi"])
        self.assertEqual(result.members.iloc[0], "bmi,bmi_missing")

    def test_pooled_extension_reuses_validated_legacy_scores(self):
        import joblib
        from src.training.preeclampsia_cdc import (LEGACY_MODEL_NAMES,
            _prediction_frame, fit_folds, metrics)
        rows = []
        for year in (2023, 2024):
            for index in range(10):
                rows.append({"source_year": year, "source_row": index, "fold": index % 5,
                    "target_hdp_proxy": index // 5, "target_hdp_preterm37": 0,
                    "target_hdp_preterm34": 0, "rf_phype": 0, "mager": 30,
                    "bmi": 25, "priorlive": 0, "priordead": 0, "priorterm": 0,
                    "rf_pdiab": 0, "rf_ppterm": 0, "rf_cesar": 0,
                    "rf_inftr": 0, "rf_artec": 0, "f_rf_inft": 1,
                    "f_rf_inf_art": 1, "mrace6": 1, "mhisp_r": 0,
                    "meduc": 4, "prior_live_birth_count_cat": 0,
                    "bmi_missing": 0, "prior_live_birth_count_missing": 0,
                    "priorterm_missing": 0, "core_scoring_ready": 1,
                    "art_source_reported": 1, "art_status_unknown_reported": 0})
        data = pd.DataFrame(rows)
        trained = []
        interrupt_after_first_fold = [True]

        def fake_fit(name, train, label):
            if interrupt_after_first_fold[0] and len(trained) == 2:
                raise RuntimeError("simulated interrupted extension")
            trained.append(name)
            return {"name": name}

        def fake_predict(name, model, frame):
            value = .22 if name == "catboost" else .24 if name == "lightgbm" else .25
            return np.full(len(frame), value,
                           dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models"
            model_dir.mkdir()
            (root / "audit.json").write_text(json.dumps({"prepared_file": {"sha256": "prepared"}}))
            (root / "fold_run.json").write_text(json.dumps({
                "fingerprint": "previous", "models": LEGACY_MODEL_NAMES,
                "features": FEATURES, "design": "pooled 2023+2024, 5 folds"}))
            for fold in range(5):
                valid = data.loc[data.fold.eq(fold)]
                old = _prediction_frame(valid)
                for name in LEGACY_MODEL_NAMES:
                    old[name] = np.full(len(valid), .25, dtype=np.float32)
                old.to_parquet(root / f"oof_fold_{fold}.parquet", index=False)
                y = valid.target_hdp_proxy.to_numpy(dtype=np.uint8)
                pd.DataFrame([{"fold": fold, "model": name, "n": len(valid),
                    "events": int(y.sum()), **metrics(y, old[name].to_numpy())}
                    for name in LEGACY_MODEL_NAMES]).to_csv(
                        root / f"fold_metrics_{fold}.csv", index=False)
                for name in LEGACY_MODEL_NAMES[1:]:
                    joblib.dump({"name": name}, model_dir / f"fold_{fold}_{name}.joblib")
            with patch("src.training.preeclampsia_cdc.OUT", root), \
                 patch("src.training.preeclampsia_cdc.MODEL_DIR", model_dir), \
                 patch("src.training.preeclampsia_cdc._verify_and_load", return_value=data), \
                 patch("src.training.preeclampsia_cdc.LEGACY_POOLED_FINGERPRINT", "previous"), \
                 patch("src.training.preeclampsia_cdc.LEGACY_PREPARED_SHA256", "prepared"), \
                 patch("src.training.preeclampsia_cdc._fit_model", side_effect=fake_fit), \
                 patch("src.training.preeclampsia_cdc._predict_model", side_effect=fake_predict), \
                 patch("src.training.preeclampsia_cdc.run_fingerprint", return_value="new"):
                (root / "audit.json").write_text(json.dumps({"prepared_file": {"sha256": "changed"}}))
                with self.assertRaisesRegex(ValueError, "archive them before rerun"):
                    fit_folds(resume=True)
                (root / "audit.json").write_text(json.dumps({"prepared_file": {"sha256": "prepared"}}))
                with self.assertRaisesRegex(ValueError, "archive them before rerun"):
                    fit_folds(resume=False)
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted extension"):
                    fit_folds(resume=True)
                self.assertEqual(len(trained), 2)
                interrupt_after_first_fold[0] = False
                result = fit_folds(resume=True)
                self.assertEqual(len(result), 5 * len(MODEL_NAMES))
                self.assertEqual(trained, [name for _ in range(5)
                                           for name in ("catboost", "lightgbm")])
                self.assertEqual(json.loads((root / "fold_run.json").read_text())
                                 ["fingerprint"], "new")
                for fold in range(5):
                    scored = pd.read_parquet(root / f"oof_fold_{fold}.parquet")
                    self.assertTrue(scored["hist_gb"].eq(.25).all())
                    self.assertTrue(scored["catboost"].eq(np.float32(.22)).all())
                fit_folds(resume=True)
                self.assertEqual(len(trained), 10)

    def test_temporal_workflow_fits_only_2023_and_tests_all_2024_rows(self):
        from src.training.preeclampsia_cdc import fit_temporal
        rows = []
        for year in (2023, 2024):
            for index in range(20):
                rows.append({"source_year": year, "source_row": index, "fold": index % 5,
                             "target_hdp_proxy": index % 2, "target_hdp_preterm37": index % 4 == 1,
                             "target_hdp_preterm34": index % 8 == 1,
                             "rf_phype": 0, "mager": 30, "bmi": 25,
                             "priorlive": 0, "priordead": 0, "priorterm": 0,
                             "rf_pdiab": 0, "rf_ppterm": 0, "rf_cesar": 0,
                             "rf_inftr": 0, "rf_artec": 0, "f_rf_inft": 1,
                             "f_rf_inf_art": 1, "mrace6": 1,
                             "mhisp_r": 0, "meduc": 4,
                             "prior_live_birth_count_cat": 0,
                             "bmi_missing": 0, "prior_live_birth_count_missing": 0,
                             "priorterm_missing": 0, "core_scoring_ready": 1,
                             "art_source_reported": 0 if year == 2024 and index == 0 else 1,
                             "art_status_unknown_reported": 0})
        data = pd.DataFrame(rows)
        unreported = data.source_year.eq(2024) & data.source_row.eq(0)
        data.loc[unreported, ["rf_inftr", "rf_artec"]] = np.nan
        data.loc[unreported, ["f_rf_inft", "f_rf_inf_art"]] = 0
        trained_years = []

        def fake_fit(name, train, label):
            trained_years.append(set(train.source_year.unique()))
            return {"name": name}

        def fake_predict(name, model, frame):
            return np.full(len(frame), .2 if name == "hist_gb" else .3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "audit.json").write_text(json.dumps({"year_checks": {
                str(year): {"eligible_singleton_rows": 20, "eligible_singleton_events": 10}
                for year in (2023, 2024)}}))
            with patch("src.training.preeclampsia_cdc.OUT", root), \
                 patch("src.training.preeclampsia_cdc.MODEL_DIR", root / "models"), \
                 patch("src.training.preeclampsia_cdc._verify_and_load", return_value=data), \
                 patch("src.training.preeclampsia_cdc._fit_model", side_effect=fake_fit), \
                 patch("src.training.preeclampsia_cdc._predict_model", side_effect=fake_predict), \
                 patch("src.training.preeclampsia_cdc.run_fingerprint", return_value="test"):
                result = fit_temporal(resume=False)
            with patch("src.evaluation.preeclampsia_cdc.OUT", root), \
                 patch("src.evaluation.preeclampsia_cdc._verify_and_load", return_value=data), \
                 patch("src.evaluation.preeclampsia_cdc.run_fingerprint", return_value="test"):
                evaluated, bootstrap = temporal_summary(bootstrap_rounds=2)
            scored = pd.read_parquet(root / "temporal_test_2024.parquet")
            developed = pd.read_parquet(root / "temporal_dev_oof.parquet")
            self.assertEqual(len(scored), 20)
            self.assertEqual(len(developed), 20)
            self.assertTrue(scored.source_year.eq(2024).all())
            self.assertTrue(developed.source_year.eq(2023).all())
            self.assertTrue(all(years == {2023} for years in trained_years))
            self.assertEqual(scored.early_core_fallback.sum(), 1)
            self.assertEqual(scored.loc[scored.early_core_fallback.eq(1), "hist_gb_early"].iloc[0],
                             scored.loc[scored.early_core_fallback.eq(1), "hist_gb"].iloc[0])
            model_count = len(MODEL_NAMES) + len(TEMPORAL_SENSITIVITIES)
            self.assertEqual(len(result), model_count)
            self.assertEqual(len(evaluated), model_count)
            self.assertEqual(len(bootstrap), 2 * (model_count - 1))
            self.assertTrue(np.isnan(evaluated.loc[evaluated.model.eq("prevalence"),
                                                     "development_threshold_10pct"].iloc[0]))

    def test_secondary_models_fit_only_known_2023_composite_labels(self):
        from sklearn.dummy import DummyClassifier
        from src.training.preeclampsia_cdc import fit_secondary_temporal
        rows = []
        for year in (2023, 2024):
            for i in range(100):
                hdp = int(i % 2 == 0)
                gest = np.nan if i % 25 == 0 else (33 if i % 8 == 0 or i % 14 == 1
                    else 36 if i % 4 == 0 or i % 3 == 1 else 39)
                rows.append({"source_year": year, "source_row": i, "fold": i % 5,
                    "rf_phype": 0, "oegest_comb": gest, "target_hdp_proxy": hdp,
                    "target_hdp_preterm37": np.nan if np.isnan(gest) else int(hdp and gest < 37),
                    "target_hdp_preterm34": np.nan if np.isnan(gest) else int(hdp and gest < 34),
                    "mager": 30, "bmi": 25, "prior_live_birth_count_cat": 0, "priorterm": 0,
                    "rf_pdiab": 0, "rf_ppterm": 0, "rf_cesar": 0,
                    "bmi_missing": 0, "prior_live_birth_count_missing": 0,
                    "priorterm_missing": 0, "core_scoring_ready": 1,
                    "art_source_reported": 1, "art_status_unknown_reported": 0})
        data = pd.DataFrame(rows)
        fitted = []
        def fake_hgb(train, features, description, target):
            fitted.append((set(train.source_year), target, train[target].isna().sum()))
            return DummyClassifier(strategy="prior").fit(train[features], train[target])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "audit.json").write_text(json.dumps({"year_checks": {
                "2023": {"eligible_singleton_rows": 100}, "2024": {"eligible_singleton_rows": 100}}}))
            with patch("src.training.preeclampsia_cdc.OUT", root), \
                 patch("src.training.preeclampsia_cdc.MODEL_DIR", root / "models"), \
                 patch("src.training.preeclampsia_cdc._verify_and_load", return_value=data), \
                 patch("src.training.preeclampsia_cdc._fit_hgb", side_effect=fake_hgb), \
                 patch("src.training.preeclampsia_cdc.run_fingerprint", return_value="test"):
                result = fit_secondary_temporal(resume=False)
            main = data.loc[data.source_year.eq(2024),
                            ["source_year", "source_row", "target_hdp_proxy"]].rename(
                            columns={"target_hdp_proxy": "label"})
            main["hist_gb"] = .2
            with patch("src.evaluation.preeclampsia_cdc.OUT", root), \
                 patch("src.evaluation.preeclampsia_cdc._verify_and_load", return_value=data), \
                 patch("src.evaluation.preeclampsia_cdc.run_fingerprint", return_value="test"):
                comparison, folds = secondary_trained_summary(main)
            self.assertEqual(len(result), 2)
            self.assertEqual(len(fitted), 12)
            self.assertTrue(all(years == {2023} and not unknown for years, _, unknown in fitted))
            self.assertEqual(len(folds), 10)
            self.assertEqual(len(comparison), 12)
            self.assertEqual(comparison.brier_if_matching_target.notna().sum(), 2)
            self.assertEqual(set(comparison.subset),
                {"all_known_gestation", "recorded_hdp_negative_only"})


if __name__ == "__main__":
    unittest.main()
