"""Single-holdout, multiclass CDC preterm-birth research baselines.

No cross-validation folds are used. The 2024 outcomes are never consulted for
model or hyperparameter choice. The classes describe gestational age at *live
birth*, including medically indicated deliveries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preterm_cdc import BASE_FEATURES, FEATURES, OUT, SEED, load_year


CHECKPOINTS = Path(__file__).resolve().parents[2] / "models/checkpoints/preterm_cdc"
TUNE_TRAIN_MAX = 500_000
TUNE_VALIDATION_MAX = 200_000
IMPORTANCE_MAX = 50_000
N_CLASSES = 4

# Bounded, clinically appropriate search: linear risk versus two nonlinear
# tabular learners. Each configuration sees the same rows and predictors.
CANDIDATES = (
    {"id": "logreg_c_0p05", "family": "logreg", "C": .05},
    {"id": "logreg_c_0p5", "family": "logreg", "C": .5},
    {"id": "logreg_c_5", "family": "logreg", "C": 5.0},
    {"id": "lightgbm_small", "family": "lightgbm", "num_leaves": 7, "min_child_samples": 500, "reg_lambda": 10.0},
    {"id": "lightgbm_medium", "family": "lightgbm", "num_leaves": 15, "min_child_samples": 500, "reg_lambda": 3.0},
    {"id": "lightgbm_flexible", "family": "lightgbm", "num_leaves": 31, "min_child_samples": 200, "reg_lambda": 10.0},
    {"id": "catboost_shallow", "family": "catboost", "depth": 4, "l2_leaf_reg": 8.0},
    {"id": "catboost_medium", "family": "catboost", "depth": 5, "l2_leaf_reg": 5.0},
    {"id": "catboost_deeper", "family": "catboost", "depth": 6, "l2_leaf_reg": 8.0},
)

ABLATIONS = {
    "without_previous_ptb": ("previous_preterm_birth",),
    "without_obstetric_history": ("previous_live_births", "previous_terminations", "previous_preterm_birth", "previous_cesarean"),
    "without_bmi": ("prepregnancy_bmi",),
    "without_chronic_conditions": ("chronic_hypertension", "preexisting_diabetes"),
    "without_smoking": ("prepregnancy_cigarettes_per_day",),
}


def _signature() -> str:
    manifest = (OUT / "manifest.json").read_bytes()
    payload = (manifest + Path(__file__).read_bytes() + json.dumps(CANDIDATES, sort_keys=True).encode()
               + str(TUNE_TRAIN_MAX).encode() + str(TUNE_VALIDATION_MAX).encode())
    return hashlib.sha256(payload).hexdigest()


def _check_cached(signature: str) -> None:
    path = OUT / "run_signature.json"
    if path.exists() and json.loads(path.read_text()).get("signature") != signature:
        raise RuntimeError("Prepared data or tuning protocol changed. Remove stale preterm_cdc results before rerunning.")


def load_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_year(2023, columns=["source_row", "risk_band", "partition", *FEATURES])
    groups = tuple(frame.loc[frame.partition.eq(name)].drop(columns="partition").reset_index(drop=True)
                   for name in ("train", "validation", "calibration"))
    if sum(len(part) for part in groups) != len(frame):
        raise AssertionError("Unexpected partition in 2023 cohort")
    if not all(set(part.risk_band.unique()) == set(range(N_CLASSES)) for part in groups):
        raise AssertionError("One risk band is absent from a 2023 partition")
    return groups


def development_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation, _ = load_development()
    train = train.sample(n=min(TUNE_TRAIN_MAX, len(train)), random_state=SEED).reset_index(drop=True)
    validation = validation.sample(n=min(TUNE_VALIDATION_MAX, len(validation)), random_state=SEED + 1).reset_index(drop=True)
    return train, validation


def make_model(candidate: dict, iterations: int | None = None):
    family = candidate["family"]
    if family == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=candidate["C"], max_iter=250, solver="lbfgs", tol=1e-4),
        )
    if family == "lightgbm":
        return LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, n_estimators=iterations or 350,
            learning_rate=.05, num_leaves=candidate["num_leaves"],
            min_child_samples=candidate["min_child_samples"],
            reg_lambda=candidate["reg_lambda"], max_bin=127,
            n_jobs=4, verbosity=-1, random_state=SEED,
            deterministic=True, force_col_wise=True,
        )
    if family == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass", eval_metric="MultiClass",
            iterations=iterations or 600, learning_rate=.05,
            depth=candidate["depth"], l2_leaf_reg=candidate["l2_leaf_reg"],
            thread_count=4, random_seed=SEED, verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(family)


def fit_model(candidate: dict, train: pd.DataFrame, validation: pd.DataFrame | None,
              features: list[str], iterations: int | None = None):
    model = make_model(candidate, iterations=iterations)
    x_train = train[features]
    y_train = train.risk_band.to_numpy(dtype=np.int8)
    if validation is None or candidate["family"] == "logreg" or iterations is not None:
        model.fit(x_train, y_train)
    elif candidate["family"] == "lightgbm":
        model.fit(x_train, y_train, eval_set=[(validation[features], validation.risk_band)],
                  eval_metric="multi_logloss", callbacks=[early_stopping(30, verbose=False)])
    else:
        model.fit(x_train, y_train, eval_set=(validation[features], validation.risk_band),
                  early_stopping_rounds=30, use_best_model=True)
    return model


def selected_iterations(candidate: dict, model) -> int | None:
    if candidate["family"] == "lightgbm":
        return int(model.best_iteration_ or model.n_estimators)
    if candidate["family"] == "catboost":
        return int(model.tree_count_)
    return None


def predict_probabilities(model, frame: pd.DataFrame, features: list[str], batch_size: int = 250_000) -> np.ndarray:
    pieces = []
    for start in range(0, len(frame), batch_size):
        p = np.asarray(model.predict_proba(frame.iloc[start:start + batch_size][features]), dtype=np.float64)
        if p.shape[1] != N_CLASSES:
            raise AssertionError("Model output lacks a risk band")
        pieces.append(p)
    probabilities = np.concatenate(pieces)
    if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("Invalid multiclass probabilities")
    return probabilities


def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    result = {"n": len(y), "multiclass_log_loss": log_loss(y, p, labels=list(range(N_CLASSES))),
              "multiclass_brier": float(np.mean(np.sum((p - np.eye(N_CLASSES)[y]) ** 2, axis=1)))}
    for threshold, upper in ((32, 1), (34, 2), (37, 3)):
        binary = (y < upper).astype(np.int8)
        risk = p[:, :upper].sum(axis=1)
        result[f"prevalence_lt{threshold}"] = float(binary.mean())
        result[f"predicted_lt{threshold}"] = float(risk.mean())
        result[f"brier_lt{threshold}"] = float(np.mean((risk - binary) ** 2))
        result[f"auroc_lt{threshold}"] = float(roc_auc_score(binary, risk))
        result[f"auprc_lt{threshold}"] = float(average_precision_score(binary, risk))
    return result


def _prediction_frame(frame: pd.DataFrame, p: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"source_row": frame.source_row.to_numpy(), "risk_band": frame.risk_band.to_numpy(),
                         **{f"{prefix}_class_{i}": p[:, i].astype("float32") for i in range(N_CLASSES)}})


def run_search(force: bool = False) -> tuple[pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    signature = _signature()
    if not force:
        _check_cached(signature)
    result_path = OUT / "tuning_candidates.csv"
    selected_path = OUT / "selected_models.json"
    if result_path.exists() and selected_path.exists() and not force:
        return pd.read_csv(result_path), json.loads(selected_path.read_text())
    train, validation = development_samples()
    y_valid = validation.risk_band.to_numpy(dtype=np.int8)
    prevalence = np.bincount(train.risk_band, minlength=N_CLASSES) / len(train)
    rows = [{"id": "prevalence", "family": "prevalence", "iterations": 0,
             **score(y_valid, np.tile(prevalence, (len(validation), 1)))}]
    best_by_family: dict[str, dict] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    for candidate in CANDIDATES:
        print(f"Fitting {candidate['id']}", flush=True)
        model = fit_model(candidate, train, validation, list(FEATURES))
        p = predict_probabilities(model, validation, list(FEATURES))
        iterations = selected_iterations(candidate, model)
        metrics = score(y_valid, p)
        rows.append({"id": candidate["id"], "family": candidate["family"],
                     "iterations": iterations or 0, **metrics})
        previous = best_by_family.get(candidate["family"])
        if previous is None or metrics["multiclass_log_loss"] < previous["validation_log_loss"]:
            best_by_family[candidate["family"]] = {
                "candidate": candidate, "iterations": iterations,
                "validation_log_loss": metrics["multiclass_log_loss"],
            }
            joblib.dump(model, CHECKPOINTS / f"development_{candidate['family']}.joblib", compress=3)
            prediction_frames[candidate["family"]] = _prediction_frame(validation, p, candidate["family"])
    result = pd.DataFrame(rows).sort_values("multiclass_log_loss").reset_index(drop=True)
    champion = min(best_by_family, key=lambda family: best_by_family[family]["validation_log_loss"])
    selected = {"signature": signature, "champion_family": champion,
                "best_by_family": best_by_family,
                "selection_metric": "2023 single-holdout multiclass log loss",
                "tuning_train_rows": len(train), "validation_rows": len(validation)}
    result.to_csv(result_path, index=False)
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    (OUT / "run_signature.json").write_text(json.dumps({"signature": signature}, indent=2))
    merged = _prediction_frame(validation, np.tile(prevalence, (len(validation), 1)), "prevalence")
    for family, part in prediction_frames.items():
        merged = merged.merge(part.drop(columns="risk_band"), on="source_row", validate="one_to_one")
    merged.to_parquet(OUT / "validation_predictions.parquet", index=False)
    return result, selected


def run_ablations(force: bool = False) -> pd.DataFrame:
    path = OUT / "ablations.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    _, selected = run_search()
    train, validation = development_samples()
    family = selected["champion_family"]
    choice = selected["best_by_family"][family]
    baseline = joblib.load(CHECKPOINTS / f"development_{family}.joblib")
    y = validation.risk_band.to_numpy(dtype=np.int8)
    base_metrics = score(y, predict_probabilities(baseline, validation, list(FEATURES)))
    rows = [{"ablation": "full", "removed": "", "family": family, **base_metrics}]
    for name, removed in ABLATIONS.items():
        features = [feature for feature in FEATURES if feature not in removed and
                    not any(feature == f"{base}_missing" for base in removed)]
        print(f"Ablation {name}", flush=True)
        model = fit_model(choice["candidate"], train, None, features, iterations=choice["iterations"])
        p = predict_probabilities(model, validation, features)
        rows.append({"ablation": name, "removed": ", ".join(removed), "family": family, **score(y, p)})
    result = pd.DataFrame(rows)
    result["delta_log_loss_vs_full"] = result.multiclass_log_loss - base_metrics["multiclass_log_loss"]
    result["delta_brier_vs_full"] = result.multiclass_brier - base_metrics["multiclass_brier"]
    result.to_csv(path, index=False)
    return result


def run_importance(force: bool = False, repeats: int = 3) -> pd.DataFrame:
    path = OUT / "permutation_importance.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    _, selected = run_search()
    family = selected["champion_family"]
    model = joblib.load(CHECKPOINTS / f"development_{family}.joblib")
    _, validation = development_samples()
    validation = validation.sample(n=min(IMPORTANCE_MAX, len(validation)), random_state=SEED + 2).reset_index(drop=True)
    y = validation.risk_band.to_numpy(dtype=np.int8)
    baseline = score(y, predict_probabilities(model, validation, list(FEATURES)))
    rng = np.random.default_rng(SEED)
    rows = []
    for feature in BASE_FEATURES:
        columns = [feature] + ([f"{feature}_missing"] if f"{feature}_missing" in FEATURES else [])
        for repeat in range(repeats):
            changed = validation.copy()
            order = rng.permutation(len(changed))
            changed.loc[:, columns] = validation.loc[order, columns].to_numpy()
            after = score(y, predict_probabilities(model, changed, list(FEATURES)))
            rows.append({"feature": feature, "repeat": repeat, "family": family,
                         "delta_log_loss": after["multiclass_log_loss"] - baseline["multiclass_log_loss"],
                         "delta_brier": after["multiclass_brier"] - baseline["multiclass_brier"],
                         "delta_auprc_lt32": after["auprc_lt32"] - baseline["auprc_lt32"],
                         "n": len(validation)})
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1))
    return softmax(logits / temperature, axis=1)


def run_temporal(force: bool = False) -> tuple[pd.DataFrame, dict]:
    metrics_path = OUT / "temporal_metrics.csv"
    metadata_path = OUT / "temporal_model.json"
    if metrics_path.exists() and metadata_path.exists() and not force:
        return pd.read_csv(metrics_path), json.loads(metadata_path.read_text())
    _, selected = run_search()
    family = selected["champion_family"]
    choice = selected["best_by_family"][family]
    train, validation, calibration = load_development()
    development = pd.concat([train, validation], ignore_index=True)
    print(f"Refitting {family} on {len(development):,} 2023 records", flush=True)
    model = fit_model(choice["candidate"], development, None, list(FEATURES), iterations=choice["iterations"])
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHECKPOINTS / "temporal_selected.joblib", compress=3)
    p_cal = predict_probabilities(model, calibration, list(FEATURES))
    y_cal = calibration.risk_band.to_numpy(dtype=np.int8)
    optimization = minimize_scalar(
        lambda t: log_loss(y_cal, temperature_scale(p_cal, t), labels=list(range(N_CLASSES))),
        bounds=(.5, 3.0), method="bounded", options={"xatol": 1e-3},
    )
    if not optimization.success:
        raise RuntimeError("Temperature calibration failed")
    temperature = float(optimization.x)
    test = load_year(2024, columns=["source_row", "risk_band", *FEATURES])
    print(f"Scoring {len(test):,} 2024 records", flush=True)
    p_raw = predict_probabilities(model, test, list(FEATURES))
    p_calibrated = temperature_scale(p_raw, temperature)
    y_test = test.risk_band.to_numpy(dtype=np.int8)
    prevalence = np.bincount(development.risk_band, minlength=N_CLASSES) / len(development)
    scores = pd.DataFrame([
        {"model": "prevalence", "dataset": "2024 temporal", **score(y_test, np.tile(prevalence, (len(test), 1)))},
        {"model": f"{family}_raw", "dataset": "2024 temporal", **score(y_test, p_raw)},
        {"model": f"{family}_temperature", "dataset": "2024 temporal", **score(y_test, p_calibrated)},
        {"model": f"{family}_raw", "dataset": "2023 calibration", **score(y_cal, p_cal)},
        {"model": f"{family}_temperature", "dataset": "2023 calibration", **score(y_cal, temperature_scale(p_cal, temperature))},
    ])
    scores.to_csv(metrics_path, index=False)
    output = pd.DataFrame({"source_row": test.source_row.to_numpy(), "risk_band": y_test,
                           **{f"class_{i}": p_calibrated[:, i].astype("float32") for i in range(N_CLASSES)}})
    output.to_parquet(OUT / "temporal_predictions_2024.parquet", index=False)
    metadata = {"family": family, "candidate": choice["candidate"], "iterations": choice["iterations"],
                "temperature": temperature, "calibration_rows": len(calibration),
                "development_rows": len(development), "temporal_test_rows": len(test),
                "selection_source": "2023 validation only", "calibration_source": "2023 calibration only",
                "test_source": "2024 labels used only for final reported metrics",
                "claim_limit": "Research-only static live-birth record model; no week-specific or spontaneous PTB risk."}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return scores, metadata
