"""Individual preterm-birth research model with gestational-age conditioning.

The source consists of singleton live-birth records, not prospectively
observed pregnancies. Outputs are model forecasts for the source population.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from src.data.preterm_cdc import BASE_FEATURES, FEATURES
from src.training.preterm_cdc import temperature_scale


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "models/exported/preterm_woman"
PROFILE_PATH = BUNDLE_DIR / "profile.json"
INPUT_CONTRACT_VERSION = "preterm-woman-input-v2"
WEEKS = np.arange(17, 37, dtype=np.int8)
WEEK_BAND = np.select([WEEKS < 32, WEEKS < 34], [0, 1], default=2)
NUMERIC_FIELDS = {
    "age_at_due_date_years": "maternal_age_at_birth",
    "prepregnancy_bmi": "prepregnancy_bmi",
    "previous_live_births": "previous_live_births",
    "previous_terminations": "previous_terminations",
    "prepregnancy_cigarettes_per_day": "prepregnancy_cigarettes_per_day",
}
BOOLEAN_FIELDS = (
    "previous_preterm_birth", "chronic_hypertension", "preexisting_diabetes", "previous_cesarean",
)
OPTIONAL_SOURCE_FIELDS = tuple(NUMERIC_FIELDS)[1:] + BOOLEAN_FIELDS
# Limits observed in the 2023 development cohort. The endpoint age codes 12
# and 50 are collapsed CDC categories, not exact patient ages.
SERVING_RANGES = {
    "age_at_due_date_years": (13, 49),
    "prepregnancy_bmi": (13, 69.8),
    "previous_live_births": (0, 22),
    "previous_terminations": (0, 28),
    "prepregnancy_cigarettes_per_day": (0, 96),
}


@dataclass(frozen=True)
class PregnancySnapshot:
    gestational_week: int
    gestational_day: int
    age_at_due_date_years: int
    singleton_confirmed: bool
    pregnancy_ongoing: bool
    dating_confirmed: bool
    prepregnancy_bmi: float | None = None
    previous_live_births: int | None = None
    previous_terminations: int | None = None
    previous_preterm_birth: bool | None = None
    chronic_hypertension: bool | None = None
    preexisting_diabetes: bool | None = None
    previous_cesarean: bool | None = None
    prepregnancy_cigarettes_per_day: int | None = None


def _snapshot_values(snapshot: PregnancySnapshot | Mapping) -> dict:
    if isinstance(snapshot, PregnancySnapshot):
        return asdict(snapshot)
    if not isinstance(snapshot, Mapping):
        raise TypeError("Expected PregnancySnapshot or JSON-like mapping")
    return dict(snapshot)


def _check_input(values: dict, *, serving: bool = False) -> tuple[str | None, list[str]]:
    expected = set(PregnancySnapshot.__dataclass_fields__)
    unknown = set(values) - expected
    if unknown:
        return "OUT_OF_SCOPE", [f"Unknown input field: {name}" for name in sorted(unknown)]
    required = ("gestational_week", "gestational_day", "age_at_due_date_years",
                "singleton_confirmed", "pregnancy_ongoing", "dating_confirmed")
    missing = [name for name in required if values.get(name) is None]
    if missing:
        return "INSUFFICIENT_DATA", [f"Missing required field: {name}" for name in missing]
    if values["singleton_confirmed"] is not True:
        return "OUT_OF_SCOPE", ["Confirmed singleton pregnancy is required"]
    if values["pregnancy_ongoing"] is not True or values["dating_confirmed"] is not True:
        return "INSUFFICIENT_DATA", ["Ongoing pregnancy and confirmed gestational dating are required"]
    week, day = values["gestational_week"], values["gestational_day"]
    if type(week) is not int or type(day) is not int or not (20 <= week <= 36 and 0 <= day <= 6):
        return "OUT_OF_SCOPE", ["Supported gestational age is 20+0 through 36+6"]
    age = values["age_at_due_date_years"]
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not np.isfinite(age) or not (10 <= age <= 55):
        return "OUT_OF_SCOPE", ["Age at the expected delivery date must be between 10 and 55 years"]
    for name, hi in (("prepregnancy_bmi", 69.9), ("previous_live_births", 60),
                     ("previous_terminations", 30), ("prepregnancy_cigarettes_per_day", 97)):
        val = values.get(name)
        if val is None:
            continue
        lo = 13 if name == "prepregnancy_bmi" else 0
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or not lo <= val <= hi:
            return "OUT_OF_SCOPE", [f"Invalid value: {name}"]
        if name != "prepregnancy_bmi" and int(val) != val:
            return "OUT_OF_SCOPE", [f"Expected a whole number: {name}"]
    for name in BOOLEAN_FIELDS:
        if values.get(name) is not None and type(values[name]) is not bool:
            return "OUT_OF_SCOPE", [f"Expected true, false or null: {name}"]
    if serving:
        absent = [name for name in OPTIONAL_SOURCE_FIELDS if values.get(name) is None]
        if absent:
            return "INSUFFICIENT_DATA", [f"Missing model input: {name}" for name in absent]
        if type(age) is not int:
            return "OUT_OF_SCOPE", ["age_at_due_date_years must be a whole number"]
        outside = [name for name, (lo, hi) in SERVING_RANGES.items()
                   if not lo <= values[name] <= hi + (1e-4 if name == "prepregnancy_bmi" else 0)]
        if outside:
            return "OOD", [f"Outside 2023 model support: {name}" for name in outside]
        if values["previous_live_births"] == 0 and values["previous_preterm_birth"]:
            return "OOD", ["Previous preterm live birth requires at least one previous live birth"]
    return None, []


def snapshot_features(snapshot: PregnancySnapshot | Mapping) -> pd.DataFrame:
    values = _snapshot_values(snapshot)
    status, reasons = _check_input(values)
    if status:
        raise ValueError(f"{status}: {'; '.join(reasons)}")
    feature_values = {target: values.get(source) for source, target in NUMERIC_FIELDS.items()}
    feature_values.update({name: (None if values.get(name) is None else int(values[name]))
                           for name in BOOLEAN_FIELDS})
    row = {name: (np.nan if feature_values[name] is None else float(feature_values[name]))
           for name in BASE_FEATURES}
    row.update({f"{name}_missing": int(np.isnan(row[name])) for name in BASE_FEATURES[1:]})
    return pd.DataFrame([row], columns=FEATURES).astype("float32")


@lru_cache(maxsize=2)
def _model_and_profile(profile_path: str):
    bundle_dir = Path(profile_path).parent
    model_path = bundle_dir / "model.joblib"
    profile = json.loads(Path(profile_path).read_text())
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    if (manifest.get("input_contract_version") != INPUT_CONTRACT_VERSION or
            manifest.get("version") != profile.get("version") or
            manifest.get("model_sha256") != profile.get("model_sha256") or
            manifest.get("feature_order") != list(FEATURES)):
        raise RuntimeError("Model bundle does not match the input contract")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != profile["model_sha256"]:
        raise RuntimeError("Week profile was built for a different model")
    if profile.get("week_values") != WEEKS.tolist():
        raise RuntimeError("Week grid differs from preparation contract")
    if not np.isfinite(profile.get("temperature", np.nan)) or profile["temperature"] <= 0:
        raise RuntimeError("Invalid probability calibration temperature")
    weights = np.asarray(profile["within_band_week_probabilities"], dtype=np.float64)
    boundaries = np.asarray(profile["score_decile_boundaries"], dtype=np.float64)
    if (boundaries.shape != (9,) or not np.isfinite(boundaries).all() or
            not np.all(np.diff(boundaries) > 0)):
        raise RuntimeError("Invalid score decile boundaries")
    if weights.shape != (10, len(WEEKS)) or not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError("Invalid week profile")
    for band in range(3):
        if not np.allclose(weights[:, WEEK_BAND == band].sum(axis=1), 1, atol=1e-6):
            raise RuntimeError("Week profile does not normalize within a risk band")
    model = joblib.load(model_path)
    if list(model.feature_names_) != list(FEATURES):
        raise RuntimeError("Model feature order differs from preparation contract")
    return model, profile, weights


def _mass_for_score(probabilities: np.ndarray, profile: dict, weights: np.ndarray) -> tuple[np.ndarray, int]:
    risk_lt37 = float(np.sum(probabilities[:3]))
    bin_index = int(np.searchsorted(profile["score_decile_boundaries"], risk_lt37, side="right"))
    mass = probabilities[WEEK_BAND] * weights[bin_index]
    if not np.isclose(mass.sum() + probabilities[3], 1, atol=1e-6):
        raise RuntimeError("Week probabilities do not sum to one")
    return mass, bin_index


def _condition_on_ongoing(mass: np.ndarray, term_mass: float, week: int, day: int) -> tuple[np.ndarray, float]:
    remaining = mass.copy()
    remaining[WEEKS < week] = 0
    remaining[WEEKS == week] *= (7 - day) / 7
    denominator = float(remaining.sum() + term_mass)
    if denominator <= 1e-12:
        raise RuntimeError("Insufficient model support at gestational age")
    return remaining / denominator, float(term_mass / denominator)


def predict_woman(snapshot: PregnancySnapshot | Mapping, *, profile_path: Path = PROFILE_PATH) -> dict:
    """Return a JSON-ready forecast for one ongoing singleton pregnancy."""
    values = _snapshot_values(snapshot)
    status, reasons = _check_input(values, serving=True)
    base = {"status": status or "OK", "reasons": reasons,
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "gestational_age": {"week": values.get("gestational_week"),
                                "day": values.get("gestational_day")}}
    if status:
        return base
    try:
        model, profile, weights = _model_and_profile(str(profile_path))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        base.update(status="MODEL_UNAVAILABLE", reasons=[str(exc)])
        return base
    frame = snapshot_features(values)
    raw = np.asarray(model.predict_proba(frame), dtype=np.float64)
    if raw.shape != (1, 4) or not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw.sum(axis=1), 1):
        base.update(status="MODEL_ERROR", reasons=["Invalid four-class probability output"])
        return base
    probabilities = temperature_scale(raw, float(profile["temperature"]))[0]
    try:
        mass, _ = _mass_for_score(probabilities, profile, weights)
    except RuntimeError as exc:
        base.update(status="MODEL_ERROR", reasons=[str(exc)])
        return base
    week, day = values["gestational_week"], values["gestational_day"]
    conditional, term_probability = _condition_on_ongoing(mass, probabilities[3], week, day)
    thresholds = {f"before_{limit}_weeks": float(conditional[WEEKS < limit].sum())
                  for limit in (32, 34, 37)}
    # The next 28 days intersect at most five completed-week bins. With CDC's
    # week-only outcome, the first and last portions use a uniform-day proxy.
    horizon_weights = np.zeros(len(WEEKS), dtype=np.float64)
    horizon_weights[(WEEKS >= week) & (WEEKS < week + 4)] = 1
    horizon_weights[WEEKS == week + 4] = day / 7
    next_four = float(np.dot(conditional, horizon_weights))
    missing = [name for name in BASE_FEATURES[1:] if frame[f"{name}_missing"].iat[0] == 1]
    base.update({
        "model_version": profile["version"],
        "probabilities": {**thresholds, "preterm_within_next_4_weeks": next_four,
                          "week_of_birth": [{"completed_week": int(w), "probability": float(p)}
                                            for w, p in zip(WEEKS, conditional) if w >= week],
                          "at_or_after_37_weeks": term_probability},
        "missing_model_inputs": missing,
        "missing_masks": {f"{name}_missing": int(frame[f"{name}_missing"].iat[0])
                          for name in BASE_FEATURES[1:]},
    })
    return base


def get_feature_importance(*, bundle_dir: Path = BUNDLE_DIR) -> list[dict]:
    """Return global diagnostic importance with each value grouped with its mask."""
    profile = json.loads((bundle_dir / "profile.json").read_text())
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    result = json.loads((bundle_dir / "feature_importance.json").read_text())
    if (result.get("model_sha256") != profile.get("model_sha256") or
            manifest.get("model_sha256") != profile.get("model_sha256") or
            result.get("input_contract_version") != INPUT_CONTRACT_VERSION or
            manifest.get("input_contract_version") != INPUT_CONTRACT_VERSION):
        raise RuntimeError("Feature importance belongs to a different model")
    if {row["feature"] for row in result["features"]} != set(BASE_FEATURES):
        raise RuntimeError("Feature importance does not cover the model inputs")
    return result["features"]
