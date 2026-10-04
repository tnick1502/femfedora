"""Audit inference input mapping, missing masks and source support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.preterm_cdc import BASE_FEATURES, FEATURES, OUT, SEED, load_year
from src.inference.preterm_woman import INPUT_CONTRACT_VERSION, OPTIONAL_SOURCE_FIELDS, SERVING_RANGES, snapshot_features


BOOLEAN_MODEL_FIELDS = ("previous_preterm_birth", "chronic_hypertension",
                        "preexisting_diabetes", "previous_cesarean")
MODEL_TO_INPUT = {"maternal_age_at_birth": "age_at_due_date_years",
                  **{name: name for name in BASE_FEATURES[1:]}}
REGISTRY_AUDIT = Path(__file__).resolve().parents[2] / "models/registry/preterm-woman-input-audit-2023-2024.json"
REGISTRY_SERVING_EVAL = Path(__file__).resolve().parents[2] / "models/registry/preterm-woman-serving-evaluation-2024.json"


def _snapshot_from_row(row: pd.Series) -> dict:
    snapshot = {"gestational_week": 28, "gestational_day": 0,
                "singleton_confirmed": True, "pregnancy_ongoing": True, "dating_confirmed": True}
    for feature, input_name in MODEL_TO_INPUT.items():
        value = row[feature]
        if pd.isna(value):
            snapshot[input_name] = None
        elif feature in BOOLEAN_MODEL_FIELDS:
            snapshot[input_name] = bool(value)
        elif feature in ("maternal_age_at_birth", "previous_live_births", "previous_terminations",
                         "prepregnancy_cigarettes_per_day"):
            snapshot[input_name] = int(value)
        else:
            snapshot[input_name] = float(value)
    return snapshot


def audit_input_contract(force: bool = False) -> dict:
    path = OUT / "woman_input_audit.json"
    if path.exists() and not force:
        result = json.loads(path.read_text())
        if result.get("audit_version") == "preterm-woman-input-audit-v2":
            if not REGISTRY_AUDIT.exists():
                REGISTRY_AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return result
    frames = {year: load_year(year, columns=["source_row", *FEATURES]) for year in (2023, 2024)}
    support = {}
    missing = {}
    mismatch = {}
    for year, frame in frames.items():
        support[str(year)] = {name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
                              for name in BASE_FEATURES}
        missing[str(year)] = {name: int(frame[name].isna().sum()) for name in BASE_FEATURES}
        mismatch[str(year)] = {name: int(np.count_nonzero(
            frame[name].isna().to_numpy() != frame[f"{name}_missing"].to_numpy(dtype=bool)))
                               for name in BASE_FEATURES[1:]}
        for name in BOOLEAN_MODEL_FIELDS:
            if not frame[name].dropna().isin([0, 1]).all():
                raise AssertionError(f"Non-Boolean values in {year} {name}")
    if any(value for yearly in mismatch.values() for value in yearly.values()):
        raise AssertionError(f"Training/temporal missing-mask mismatch: {mismatch}")
    outside_2023_support = {}
    for name in BASE_FEATURES:
        lower, upper = support["2023"][name]["min"], support["2023"][name]["max"]
        outside_2023_support[name] = int((frames[2024][name].notna() &
                                         ~frames[2024][name].between(lower, upper)).sum())

    sample_rows = []
    for year, frame in frames.items():
        selected = frame.sample(n=min(250, len(frame)), random_state=SEED + year)
        for name in BASE_FEATURES[1:]:
            absent = frame.loc[frame[name].isna()]
            if not absent.empty:
                selected = pd.concat([selected, absent.sample(n=min(5, len(absent)),
                                                             random_state=SEED + year + BASE_FEATURES.index(name))])
        sample_rows.extend([(year, row) for _, row in selected.drop_duplicates("source_row").iterrows()])
    parity_rows = 0
    for year, row in sample_rows:
        from_input = snapshot_features(_snapshot_from_row(row)).iloc[0]
        for feature in FEATURES:
            actual, expected = from_input[feature], row[feature]
            if not ((pd.isna(actual) and pd.isna(expected)) or np.isclose(actual, expected, atol=1e-6)):
                raise AssertionError(f"{year} row {row.source_row}: input mapping mismatch for {feature}")
        parity_rows += 1
    result = {"audit_version": "preterm-woman-input-audit-v2",
              "source_rows": {"2023": len(frames[2023]), "2024": len(frames[2024])},
              "feature_order": list(FEATURES), "base_features": list(BASE_FEATURES),
              "missing_mask_mismatches": mismatch,
              "missing_counts": missing,
              "observed_support": support,
              "2024_outside_2023_observed_support": outside_2023_support,
              "prediction_input_parity_rows": parity_rows,
              "prediction_input_parity_mismatches": 0,
              "age_mapping_caveat": "Source age is integer age at birth; serving requires integer age at expected delivery, an approximation",
              "population_caveat": "Singleton live-birth records are not all ongoing pregnancies"}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    REGISTRY_AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _serving_eligible(frame: pd.DataFrame) -> np.ndarray:
    eligible = frame[list(OPTIONAL_SOURCE_FIELDS)].notna().all(axis=1).to_numpy()
    for input_name, (lower, upper) in SERVING_RANGES.items():
        feature = "maternal_age_at_birth" if input_name == "age_at_due_date_years" else input_name
        # BMI in the Parquet files is float32; permit its representation of 69.8.
        tolerance = 1e-4 if feature == "prepregnancy_bmi" else 0
        eligible &= frame[feature].between(lower, upper + tolerance).to_numpy()
    eligible &= ~((frame.previous_live_births.eq(0) & frame.previous_preterm_birth.eq(1)).to_numpy())
    return eligible


def evaluate_serving_scope(force: bool = False) -> dict:
    """Re-score the 2024 risk set after the individual-input abstention rules."""
    from src.training.preterm_woman import _landmark_probabilities, _weekly_quality, build_week_profile

    path = OUT / "woman_serving_scope_2024.json"
    profile = build_week_profile()
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if (cached.get("model_sha256") == profile["model_sha256"] and
                cached.get("input_contract_version") == INPUT_CONTRACT_VERSION and
                cached.get("evaluation_version") == "serving-v2"):
            if not REGISTRY_SERVING_EVAL.exists():
                REGISTRY_SERVING_EVAL.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
            return cached
    columns = ["source_row", "gestational_weeks_at_birth", "partition", "maternal_age_at_birth",
               *OPTIONAL_SOURCE_FIELDS]
    reference = load_year(2023, columns=columns)
    test = load_year(2024, columns=columns)
    predictions = pd.read_parquet(OUT / "temporal_predictions_2024.parquet")
    if not test.source_row.equals(predictions.source_row):
        raise RuntimeError("2024 predictions and outcomes are not aligned")
    reference_selected = _serving_eligible(reference) & reference.partition.eq("calibration").to_numpy()
    test_selected = _serving_eligible(test)
    landmark = 28
    reference_ongoing = reference_selected & reference.gestational_weeks_at_birth.ge(landmark).to_numpy()
    test_ongoing = test_selected & test.gestational_weeks_at_birth.ge(landmark).to_numpy()
    if not reference_ongoing.any() or not test_ongoing.any():
        raise RuntimeError("Empty eligible landmark cohort")
    probability = predictions.loc[test_ongoing, [f"class_{i}" for i in range(4)]].to_numpy(dtype=np.float64)
    boundaries = np.asarray(profile["score_decile_boundaries"], dtype=np.float64)
    weights = np.asarray(profile["within_band_week_probabilities"], dtype=np.float64)
    bins = np.searchsorted(boundaries, probability[:, :3].sum(axis=1), side="right")
    outcomes = test.loc[test_ongoing, "gestational_weeks_at_birth"].to_numpy(dtype=np.int8)
    reference_outcomes = reference.loc[reference_ongoing, "gestational_weeks_at_birth"].to_numpy(dtype=np.int8)
    checklist = (test.loc[test_ongoing, "previous_preterm_birth"].eq(1) |
                 test.loc[test_ongoing, "chronic_hypertension"].eq(1) |
                 test.loc[test_ongoing, "preexisting_diabetes"].eq(1)).to_numpy()
    selected_count = int(checklist.sum())
    weekly_metrics, weekly_calibration = _weekly_quality(probability, bins, weights,
                                                          outcomes, reference_outcomes, landmark)
    result = {"evaluation_version": "serving-v2", "model_sha256": profile["model_sha256"],
              "input_contract_version": INPUT_CONTRACT_VERSION, "landmark_week": landmark,
              "eligible_2024_rows": int(test_selected.sum()),
              "total_2024_rows": len(test), "eligible_2024_landmark_rows": int(test_ongoing.sum()),
              "checklist_fields": ["previous_preterm_birth", "chronic_hypertension", "preexisting_diabetes"],
              "checklist_selected_fraction": float(checklist.mean()),
              "age_proxy_caveat": "Evaluation has age at actual birth; serving receives integer age at expected delivery",
              "weekly_metrics": weekly_metrics, "weekly_calibration": weekly_calibration,
              "horizons": {}}
    for name, cutoff in (("before_32", 32), ("before_37", 37)):
        risk = _landmark_probabilities(probability, bins, weights, landmark, cutoff)
        y = outcomes < cutoff
        reference_rate = float((reference_outcomes < cutoff).mean())
        brier = float(np.mean((risk - y) ** 2))
        frequency_brier = float(np.mean((reference_rate - y) ** 2))
        ranked = np.argpartition(risk, -selected_count)[-selected_count:]
        model_selected = np.zeros(len(y), dtype=bool)
        model_selected[ranked] = True
        nulliparous = test.loc[test_ongoing, "previous_live_births"].eq(0).to_numpy()
        result["horizons"][name] = {
            "events": int(y.sum()), "observed_rate": float(y.mean()), "mean_predicted": float(risk.mean()),
            "auroc": float(roc_auc_score(y, risk)),
            "auprc": float(average_precision_score(y, risk)),
            "brier": brier, "frequency_brier": frequency_brier,
            "relative_brier_improvement": float(1 - brier / frequency_brier),
            "nulliparous_rows": int(nulliparous.sum()),
            "nulliparous_auroc": float(roc_auc_score(y[nulliparous], risk[nulliparous])),
            "checklist": {"selected": selected_count, "ppv": float(y[checklist].mean()),
                          "sensitivity": float(y[checklist].sum() / y.sum())},
            "model_at_same_selection": {"selected": selected_count,
                                        "ppv": float(y[model_selected].mean()),
                                        "sensitivity": float(y[model_selected].sum() / y.sum())},
        }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    REGISTRY_SERVING_EVAL.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
