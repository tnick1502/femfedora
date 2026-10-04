"""Fit and audit a coarse week profile using only 2023 calibration records.

No model family or smoothing parameter is chosen with 2024 outcomes. The week
profile is a conditional distribution within CatBoost outcome bands and score
deciles; it is an approximation, not a longitudinal pregnancy survival model.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.data.preterm_cdc import BASE_FEATURES, FEATURES, OUT, SEED, load_year
from src.inference.preterm_woman import BUNDLE_DIR, INPUT_CONTRACT_VERSION, WEEK_BAND, WEEKS
from src.training.preterm_cdc import CHECKPOINTS, predict_probabilities, run_temporal, temperature_scale


PSEUDO_COUNT_PER_BAND = 100.0
LANDMARK_WEEKS = (20, 24, 28, 32, 34, 36)
PROFILE_PATH = OUT / "woman_week_profile.json"


def build_week_profile(force: bool = False) -> dict:
    if PROFILE_PATH.exists() and not force:
        return json.loads(PROFILE_PATH.read_text())
    _, metadata = run_temporal()
    model_path = CHECKPOINTS / "temporal_selected.joblib"
    model = joblib.load(model_path)
    calibration = pd.read_parquet(
        OUT / "cohort_2023.parquet",
        columns=["partition", "gestational_weeks_at_birth", *FEATURES],
    )
    calibration = calibration.loc[calibration.partition.eq("calibration")].reset_index(drop=True)
    probabilities = temperature_scale(
        predict_probabilities(model, calibration, list(FEATURES)), metadata["temperature"]
    )
    score = probabilities[:, :3].sum(axis=1)
    boundaries = np.quantile(score, np.arange(1, 10) / 10)
    if len(np.unique(boundaries)) != 9:
        raise RuntimeError("Risk score has insufficient resolution for deciles")
    bins = np.searchsorted(boundaries, score, side="right")
    gestation = calibration.gestational_weeks_at_birth.to_numpy(dtype=np.int8)
    weights = np.zeros((10, len(WEEKS)), dtype=np.float64)
    support = np.zeros((10, 4), dtype=np.int64)
    global_counts = np.array([(gestation == w).sum() for w in WEEKS], dtype=np.float64)
    for score_bin in range(10):
        selected = bins == score_bin
        for band in range(4):
            band_selected = selected & ((gestation < 32) if band == 0 else
                                        ((gestation < 34) & (gestation >= 32)) if band == 1 else
                                        ((gestation < 37) & (gestation >= 34)) if band == 2 else
                                        (gestation >= 37))
            support[score_bin, band] = int(band_selected.sum())
        for band in range(3):
            indices = np.flatnonzero(WEEK_BAND == band)
            observed = np.array([(gestation[selected] == WEEKS[i]).sum() for i in indices], dtype=np.float64)
            global_shape = global_counts[indices] / global_counts[indices].sum()
            weights[score_bin, indices] = (observed + PSEUDO_COUNT_PER_BAND * global_shape) / (
                observed.sum() + PSEUDO_COUNT_PER_BAND
            )
    if not np.allclose(weights.sum(axis=1), 3):
        raise AssertionError("Expected three normalized preterm bands")
    profile = {
        "version": "cdc-ptb-woman-week-2023-v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "temperature": metadata["temperature"],
        "profile_source": "2023 calibration only; singleton live births",
        "calibration_rows": len(calibration),
        "score_decile_boundaries": boundaries.tolist(),
        "within_band_week_probabilities": weights.tolist(),
        "support_by_decile_and_band": support.tolist(),
        "pseudo_count_per_band": PSEUDO_COUNT_PER_BAND,
        "week_values": WEEKS.tolist(),
        "claim_limit": "Research-only conditional distribution among singleton live births, not a clinically validated pregnancy risk",
    }
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    return profile


def _landmark_probabilities(p: np.ndarray, bins: np.ndarray, weights: np.ndarray,
                            landmark: int, cutoff: int) -> np.ndarray:
    at_risk = np.stack([weights[:, (WEEKS >= landmark) & (WEEKS < limit) & (WEEK_BAND == band)].sum(axis=1)
                        for band, limit in ((0, 37), (1, 37), (2, 37))], axis=1)
    tail = np.sum(p[:, :3] * at_risk[bins], axis=1)
    event = np.stack([weights[:, (WEEKS >= landmark) & (WEEKS < cutoff) & (WEEK_BAND == band)].sum(axis=1)
                      for band in range(3)], axis=1)
    numerator = np.sum(p[:, :3] * event[bins], axis=1)
    return numerator / (tail + p[:, 3])


def _weekly_quality(p: np.ndarray, bins: np.ndarray, weights: np.ndarray,
                    gestation: np.ndarray, reference_gestation: np.ndarray,
                    landmark: int) -> tuple[dict, list[dict]]:
    ongoing = gestation >= landmark
    po = p[ongoing]
    bo = bins[ongoing]
    go = gestation[ongoing]
    tail_by_band = np.stack([weights[:, (WEEKS >= landmark) & (WEEK_BAND == band)].sum(axis=1)
                             for band in range(3)], axis=1)
    denominator = np.sum(po[:, :3] * tail_by_band[bo], axis=1) + po[:, 3]
    actual_probability = np.where(go >= 37, po[:, 3] / denominator, 0.0)
    squared_sum = actual_probability * 0  # one array reused as a bounded accumulator
    squared_sum += (po[:, 3] / denominator) ** 2
    calibration = []
    for index, week in enumerate(WEEKS):
        if week < landmark:
            continue
        weekly = po[:, WEEK_BAND[index]] * weights[bo, index] / denominator
        squared_sum += weekly ** 2
        selected = go == week
        actual_probability[selected] = weekly[selected]
        calibration.append({"landmark_week": landmark, "birth_week": int(week),
                            "n_ongoing_live_births": len(go),
                            "observed_rate": float(selected.mean()),
                            "mean_predicted": float(weekly.mean())})
    term_selected = go >= 37
    calibration.append({"landmark_week": landmark, "birth_week": "37+",
                        "n_ongoing_live_births": len(go),
                        "observed_rate": float(term_selected.mean()),
                        "mean_predicted": float((po[:, 3] / denominator).mean())})
    reference = reference_gestation[reference_gestation >= landmark]
    ref_counts = np.array([(reference == week).sum() for week in WEEKS if week >= landmark] +
                          [(reference >= 37).sum()], dtype=np.float64)
    ref_probs = ref_counts / ref_counts.sum()
    ref_actual = np.where(term_selected, ref_probs[-1], 0.0)
    for index, week in enumerate(WEEKS):
        if week >= landmark:
            ref_actual[go == week] = ref_probs[index - (landmark - int(WEEKS[0]))]
    return {
        "weekly_log_loss": float(-np.log(np.clip(actual_probability, 1e-12, 1)).mean()),
        "weekly_brier": float(np.mean(1 + squared_sum - 2 * actual_probability)),
        "frequency_weekly_log_loss": float(-np.log(np.clip(ref_actual, 1e-12, 1)).mean()),
        "frequency_weekly_brier": float(np.mean(1 + np.sum(ref_probs ** 2) - 2 * ref_actual)),
    }, calibration


def evaluate_week_profile(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = OUT / "woman_landmark_metrics_2024.csv"
    calibration_path = OUT / "woman_landmark_calibration_2024.csv"
    week_path = OUT / "woman_week_calibration_2024.csv"
    if path.exists() and calibration_path.exists() and week_path.exists() and not force:
        return pd.read_csv(path), pd.read_csv(calibration_path)
    profile = build_week_profile()
    weights = np.asarray(profile["within_band_week_probabilities"], dtype=np.float64)
    predictions = pd.read_parquet(OUT / "temporal_predictions_2024.parquet")
    outcomes = pd.read_parquet(OUT / "cohort_2024.parquet",
                               columns=["source_row", "gestational_weeks_at_birth"])
    reference = pd.read_parquet(OUT / "cohort_2023.parquet",
                                columns=["partition", "gestational_weeks_at_birth"])
    reference_gestation = reference.loc[reference.partition.eq("calibration"),
                                       "gestational_weeks_at_birth"].to_numpy(dtype=np.int8)
    test = predictions.merge(outcomes, on="source_row", validate="one_to_one")
    if len(test) < 3_000_000:
        raise AssertionError("Unexpected size of 2024 temporal test")
    p = test[[f"class_{i}" for i in range(4)]].to_numpy(dtype=np.float64)
    gestation = test.gestational_weeks_at_birth.to_numpy(dtype=np.int8)
    bins = np.searchsorted(profile["score_decile_boundaries"], p[:, :3].sum(axis=1), side="right")
    rows, calibration_rows, week_rows = [], [], []
    for landmark in LANDMARK_WEEKS:
        ongoing = gestation >= landmark
        if not ongoing.any():
            raise AssertionError("Empty landmark set")
        weekly_metrics, weekly_calibration = _weekly_quality(p, bins, weights, gestation,
                                                               reference_gestation, landmark)
        week_rows.extend(weekly_calibration)
        for horizon, cutoff in (("before_37", 37), ("next_4_weeks_preterm", min(landmark + 4, 37))):
            predicted = _landmark_probabilities(p, bins, weights, landmark, cutoff)[ongoing]
            observed = (gestation[ongoing] < cutoff).astype(np.int8)
            prevalence = float(observed.mean())
            reference_at_risk = reference_gestation[reference_gestation >= landmark]
            reference_rate = float((reference_at_risk < cutoff).mean())
            brier = float(np.mean((predicted - observed) ** 2))
            rows.append({"landmark_week": landmark, "horizon": horizon,
                         "n_ongoing_live_births": int(ongoing.sum()), "events": int(observed.sum()),
                         "observed_rate": prevalence, "mean_predicted": float(predicted.mean()),
                         "absolute_calibration_gap": float(abs(predicted.mean() - prevalence)),
                         "brier": brier, "frequency_rate_2023": reference_rate,
                         "frequency_brier": float(np.mean((reference_rate - observed) ** 2)),
                         "auprc": float(average_precision_score(observed, predicted)),
                         "auroc": float(roc_auc_score(observed, predicted)),
                         **weekly_metrics})
            for score_bin in range(10):
                selected = bins[ongoing] == score_bin
                calibration_rows.append({"landmark_week": landmark, "horizon": horizon,
                                         "score_decile": score_bin + 1, "n": int(selected.sum()),
                                         "observed_rate": float(observed[selected].mean()),
                                         "mean_predicted": float(predicted[selected].mean())})
    metrics = pd.DataFrame(rows)
    calibration = pd.DataFrame(calibration_rows)
    metrics.to_csv(path, index=False)
    calibration.to_csv(calibration_path, index=False)
    pd.DataFrame(week_rows).to_csv(week_path, index=False)
    return metrics, calibration


def run_woman_feature_importance(force: bool = False, repeats: int = 3) -> pd.DataFrame:
    """Permute each final-model value on calibration rows eligible for serving."""
    path = OUT / "woman_feature_importance_serving_2023.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    _, metadata = run_temporal()
    model = joblib.load(CHECKPOINTS / "temporal_selected.joblib")
    from src.evaluation.preterm_woman_inputs import _serving_eligible

    validation = load_year(2023, columns=["risk_band", "partition", *FEATURES])
    validation = validation.loc[validation.partition.eq("calibration")].drop(columns="partition")
    validation = validation.loc[_serving_eligible(validation)]
    validation = validation.sample(n=min(50_000, len(validation)), random_state=SEED + 8).reset_index(drop=True)
    y = validation.risk_band.to_numpy(dtype=np.int8)
    temperature = metadata["temperature"]
    baseline = log_loss(y, temperature_scale(predict_probabilities(model, validation, list(FEATURES)),
                                             temperature), labels=[0, 1, 2, 3])
    rng = np.random.default_rng(SEED + 8)
    rows = []
    for feature in BASE_FEATURES:
        columns = [feature] + ([f"{feature}_missing"] if f"{feature}_missing" in FEATURES else [])
        for repeat in range(repeats):
            changed = validation.copy()
            order = rng.permutation(len(changed))
            changed.loc[:, columns] = validation.loc[order, columns].to_numpy()
            p = temperature_scale(predict_probabilities(model, changed, list(FEATURES)), temperature)
            rows.append({"feature": feature, "repeat": repeat, "n": len(validation),
                         "delta_log_loss": float(log_loss(y, p, labels=[0, 1, 2, 3]) - baseline)})
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def export_week_bundle() -> dict:
    """Copy model, weekly profile and grouped feature importance together."""
    profile = build_week_profile()
    model_path = CHECKPOINTS / "temporal_selected.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != profile["model_sha256"]:
        raise RuntimeError("Model and week profile mismatch")
    model = joblib.load(model_path)
    if list(model.feature_names_) != list(FEATURES):
        raise RuntimeError("Model feature order differs from the preparation contract")
    native = np.asarray(model.get_feature_importance(type="FeatureImportance"), dtype=np.float64)
    if native.shape != (len(FEATURES),):
        raise RuntimeError("Unexpected CatBoost importance vector")
    permutation = run_woman_feature_importance()
    perm_grouped = permutation.groupby("feature").delta_log_loss.mean().to_dict()
    importance = {
        "model_sha256": profile["model_sha256"],
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "method_native": "CatBoost PredictionValuesChange on the original training distribution; value and missing mask grouped",
        "method_permutation": "2023 calibration sample eligible for individual input; value and constant-zero mask permuted together; final calibrated model",
        "features": sorted([
            {"feature": name,
             "native_importance_percent": float(native[FEATURES.index(name)] +
                 (native[FEATURES.index(f"{name}_missing")] if f"{name}_missing" in FEATURES else 0)),
             "permutation_delta_log_loss": float(perm_grouped[name])}
            for name in BASE_FEATURES
        ], key=lambda row: row["permutation_delta_log_loss"], reverse=True),
    }
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model = BUNDLE_DIR / "model.building.joblib"
    temporary_profile = BUNDLE_DIR / "profile.building.json"
    shutil.copy2(model_path, temporary_model)
    shutil.copy2(PROFILE_PATH, temporary_profile)
    temporary_model.replace(BUNDLE_DIR / "model.joblib")
    temporary_profile.replace(BUNDLE_DIR / "profile.json")
    (BUNDLE_DIR / "feature_importance.json").write_text(json.dumps(importance, ensure_ascii=False, indent=2))
    manifest = {"version": profile["version"], "input_contract_version": INPUT_CONTRACT_VERSION,
                "status": "MODEL_EVALUATED_NOT_CLINICALLY_VALIDATED",
                "model_sha256": profile["model_sha256"],
                "model_file": "model.joblib", "profile_file": "profile.json",
                "feature_importance_file": "feature_importance.json", "feature_order": list(FEATURES),
                "source": "CDC Natality 2023", "evaluation": "CDC Natality 2024 live-birth records"}
    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
