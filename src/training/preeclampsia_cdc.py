"""Prespecified CDC HDP-proxy baselines; all fits happen only when notebooks run."""
from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.data.preeclampsia_cdc import (BINARY, DEMOGRAPHIC_FEATURES, EARLY_FEATURES,
    FEATURES, MISSING_MASKS, NUMERIC, OUT, OUTCOMES, QUALITY_MASKS, ROOT, SEED, fingerprint_config)
from src.progress import tqdm

MODEL_DIR = ROOT / "models/exported/preeclampsia_cdc"
MODEL_NAMES = ["prevalence", "logistic", "logistic_spline", "hist_gb", "catboost", "lightgbm"]
LEGACY_MODEL_NAMES = ["prevalence", "logistic", "logistic_spline", "hist_gb"]
# The only four-model run eligible for one-time extension in this workspace.
LEGACY_POOLED_FINGERPRINT = "5439eae800e3e3ed8f03f86cdf920fe4f2eea2e3669f4d29cdd3da388d13177b"
LEGACY_PREPARED_SHA256 = "c7b2da1f00bf7fd954e5b5096b902dc8cc461f26e61939cc492e462176d33a89"
TEMPORAL_SENSITIVITIES = ["hist_gb_no_masks", "hist_gb_early", "hist_gb_demographic"]
SECONDARY_TARGETS = {"hist_gb_hdp_preterm37": "target_hdp_preterm37",
                     "hist_gb_hdp_preterm34": "target_hdp_preterm34"}
CORE_NO_MASKS = NUMERIC + BINARY
SPLINE_FEATURES = FEATURES
LOGISTIC_PARAMS = dict(C=1.0, max_iter=300, solver="lbfgs", random_state=SEED)
HGB_PARAMS = dict(max_leaf_nodes=15, learning_rate=0.07, min_samples_leaf=200,
                  l2_regularization=10.0, max_bins=63, early_stopping=False,
                  warm_start=True, random_state=SEED)
HGB_STAGES = (20, 40, 60, 80, 100)
CATBOOST_PARAMS = dict(iterations=300, depth=5, learning_rate=0.05,
                       l2_leaf_reg=8.0, loss_function="Logloss", random_seed=SEED,
                       thread_count=-1, allow_writing_files=False, task_type="CPU")
LIGHTGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=200, reg_lambda=10.0, max_bin=63,
                       objective="binary", random_state=SEED, n_jobs=-1, verbosity=-1)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8)
    y = np.asarray(y)
    if len(y) == 0 or np.unique(y).size != 2:
        raise ValueError("Metrics require both outcome classes")
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
                mean_prediction=float(p.mean()), prevalence=float(np.mean(y)))


def validate_core_input(frame: pd.DataFrame) -> None:
    """Fail closed on a changed source schema or inconsistent missing-value masks."""
    absent = set(FEATURES + QUALITY_MASKS).difference(frame.columns)
    if absent:
        raise ValueError(f"Missing model-input columns: {sorted(absent)}")
    for column in ["mager", *BINARY, *MISSING_MASKS]:
        if frame[column].isna().any():
            raise ValueError(f"Required model input is missing: {column}")
    for column in BINARY + MISSING_MASKS + QUALITY_MASKS:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"Nonbinary input: {column}")
    if not frame.core_scoring_ready.eq(1).all():
        raise ValueError("Core scoring is not available for every row")
    for value, mask in (("bmi", "bmi_missing"),
                        ("prior_live_birth_count_cat", "prior_live_birth_count_missing"),
                        ("priorterm", "priorterm_missing")):
        if not np.array_equal(frame[value].isna().to_numpy(), frame[mask].eq(1).to_numpy()):
            raise ValueError(f"Inconsistent value/mask pair: {value}")
    if not frame.mager.between(10, 55).all() or not frame.bmi.dropna().between(13, 69.9).all():
        raise ValueError("Age or prepregnancy BMI outside prespecified source range")
    if not frame.prior_live_birth_count_cat.dropna().isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid previous live-birth category")


def validate_early_input(frame: pd.DataFrame) -> None:
    """ART values are usable only when both NCHS source flags report them."""
    fields = {"rf_inftr", "rf_artec", "f_rf_inft", "f_rf_inf_art",
              "art_source_reported", "art_status_unknown_reported"}
    if fields.difference(frame.columns):
        raise ValueError(f"Missing early-input fields: {sorted(fields.difference(frame.columns))}")
    for field in ("f_rf_inft", "f_rf_inf_art"):
        if not frame[field].isin([0, 1]).all():
            raise ValueError(f"Invalid reporting flag: {field}")
    reported = frame.f_rf_inft.eq(1) & frame.f_rf_inf_art.eq(1)
    unknown = reported & (frame.rf_inftr.isna() | frame.rf_artec.isna())
    if not np.array_equal(frame.art_source_reported.to_numpy(dtype=bool), reported.to_numpy()):
        raise ValueError("ART source-reporting mask differs from NCHS flags")
    if not np.array_equal(frame.art_status_unknown_reported.to_numpy(dtype=bool), unknown.to_numpy()):
        raise ValueError("Individual ART-unknown mask differs from reported values")
    if frame.loc[~reported, ["rf_inftr", "rf_artec"]].notna().any().any():
        raise ValueError("Unreported ART values were interpreted")
    for field in ("rf_inftr", "rf_artec"):
        if not frame[field].dropna().isin([0, 1]).all():
            raise ValueError(f"Invalid ART value: {field}")
    if frame.loc[frame.rf_artec.eq(1), "rf_inftr"].ne(1).any():
        raise ValueError("Recorded ART requires recorded infertility treatment")


def logistic_pipeline(features: list[str] = FEATURES, *, spline: bool = False,
                      C: float | None = None, n_knots: int = 5) -> Pipeline:
    if spline:
        transformer = ColumnTransformer([
            ("splines", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                  ("basis", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False))]),
             ["mager", "bmi"]),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                  ("scale", StandardScaler())]), ["priorterm"]),
            ("parity", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True)),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             ["prior_live_birth_count_cat"]),
            ("binary", "passthrough", BINARY + MISSING_MASKS),
        ])
    else:
        transformer = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                  ("scale", StandardScaler())]), [x for x in NUMERIC if x in features]),
            ("binary", "passthrough", [x for x in BINARY + MISSING_MASKS if x in features]),
        ])
    params = {**LOGISTIC_PARAMS, **({"C": C} if C is not None else {})}
    return Pipeline([("transform", transformer),
                     ("model", LogisticRegression(**params))])


def predict_logistic(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    transformed = model[:-1].transform(frame)
    return expit(np.asarray(transformed @ model[-1].coef_[0]).reshape(-1) + model[-1].intercept_[0])


def run_fingerprint() -> str:
    sources = [__file__, str(ROOT / "src/data/preeclampsia_cdc.py"),
               str(ROOT / "src/data/prepare_open_datasets.py")]
    hashes = {Path(source).name: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources}
    audit = json.loads((OUT / "audit.json").read_text())
    spec = {"data": fingerprint_config(), "models": MODEL_NAMES,
            "temporal_sensitivities": TEMPORAL_SENSITIVITIES,
            "secondary_targets": SECONDARY_TARGETS, "logistic": LOGISTIC_PARAMS,
            "hgb": HGB_PARAMS, "stages": HGB_STAGES,
            "catboost": CATBOOST_PARAMS, "lightgbm": LIGHTGBM_PARAMS,
            "code_hashes": hashes,
            "prepared_sha256": audit["prepared_file"]["sha256"]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _verify_and_load(columns: list[str] | None = None) -> pd.DataFrame:
    audit = json.loads((OUT / "audit.json").read_text())
    path = OUT / "eligible_all_2023_2024.parquet"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != audit["prepared_file"]["sha256"]:
        raise ValueError("Prepared Parquet SHA-256 differs from the audited file")
    frame = pd.read_parquet(path, columns=columns)
    expected = sum(x["eligible_singleton_rows"] for x in audit["year_checks"].values())
    if len(frame) != expected or frame.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Pooled cohort count or source identifiers invalid")
    if "rf_phype" in frame and not frame.rf_phype.eq(0).all():
        raise ValueError("Chronic hypertension leaked into the primary cohort")
    return frame


def _fit_hgb(train: pd.DataFrame, features: list[str], description: str,
             target: str = "target_hdp_proxy", overrides: dict | None = None):
    categorical = [x for x in ("mrace6", "mhisp_r", "meduc") if x in features]
    model = HistGradientBoostingClassifier(**{**HGB_PARAMS, **(overrides or {})},
        categorical_features=categorical if categorical else None)
    with tqdm(total=HGB_STAGES[-1], desc=description, unit="tree", leave=False) as bar:
        previous = 0
        for stage in HGB_STAGES:
            model.set_params(max_iter=stage)
            model.fit(train[features], train[target].to_numpy(dtype=np.uint8))
            bar.update(stage - previous)
            previous = stage
    return model


class _CatBoostProgress:
    """Translate CatBoost's logged tree numbers to the notebook progress bar."""
    def __init__(self, bar):
        self.bar = bar
        self.pending = ""

    def write(self, chunk: str) -> None:
        self.pending += chunk
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = re.match(r"^(\d+):", line.strip())
            if match:
                completed = min(int(match.group(1)) + 1, self.bar.total)
                self.bar.update(max(0, completed - self.bar.n))

    def flush(self) -> None:
        pass


def _fit_catboost(train: pd.DataFrame, description: str, overrides: dict | None = None):
    params = {**CATBOOST_PARAMS, **(overrides or {})}
    model = CatBoostClassifier(**params)
    with tqdm(total=params["iterations"], desc=description,
              unit="tree", leave=False) as bar:
        model.fit(train[FEATURES], train.target_hdp_proxy.to_numpy(dtype=np.uint8),
                  verbose=25, log_cout=_CatBoostProgress(bar))
        bar.update(bar.total - bar.n)
    return model


def _fit_lightgbm(train: pd.DataFrame, description: str, overrides: dict | None = None):
    params = {**LIGHTGBM_PARAMS, **(overrides or {})}
    model = LGBMClassifier(**params)
    with tqdm(total=params["n_estimators"], desc=description,
              unit="tree", leave=False) as bar:
        def progress(env):
            bar.update(max(0, env.iteration + 1 - bar.n))
        progress.order = 0
        model.fit(train[FEATURES], train.target_hdp_proxy.to_numpy(dtype=np.uint8),
                  callbacks=[progress])
    return model


def _fit_model(name: str, train: pd.DataFrame, label: str,
               overrides: dict | None = None):
    y = train.target_hdp_proxy.to_numpy(dtype=np.uint8)
    overrides = overrides or {}
    if name == "logistic":
        return logistic_pipeline(C=overrides.get("C")).fit(train[FEATURES], y)
    if name == "logistic_spline":
        return logistic_pipeline(spline=True, C=overrides.get("C"),
                                 n_knots=overrides.get("n_knots", 5)).fit(train, y)
    if name == "catboost":
        return _fit_catboost(train, label, overrides)
    if name == "lightgbm":
        return _fit_lightgbm(train, label, overrides)
    feature_map = {"hist_gb": FEATURES, "hist_gb_no_masks": CORE_NO_MASKS,
                   "hist_gb_early": EARLY_FEATURES,
                   "hist_gb_demographic": DEMOGRAPHIC_FEATURES}
    return _fit_hgb(train, feature_map[name], label, overrides=overrides)


def _predict_model(name: str, model, frame: pd.DataFrame) -> np.ndarray:
    feature_map = {"logistic": FEATURES, "logistic_spline": SPLINE_FEATURES,
                   "catboost": FEATURES, "lightgbm": FEATURES,
                   "hist_gb": FEATURES, "hist_gb_no_masks": CORE_NO_MASKS,
                   "hist_gb_early": EARLY_FEATURES,
                   "hist_gb_demographic": DEMOGRAPHIC_FEATURES}
    X = frame[feature_map[name]]
    if name.startswith("logistic"):
        return predict_logistic(model, X).astype(np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def _prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keep = ["source_year", "source_row", "fold", "target_hdp_proxy",
            "target_hdp_preterm37", "target_hdp_preterm34", "mager", "bmi",
            "priorlive", "priordead", "rf_pdiab", "rf_inftr", "rf_artec", "f_rf_inf_art",
            "mrace6", "mhisp_r", "meduc"]
    keep += MISSING_MASKS + QUALITY_MASKS
    return frame[keep].rename(columns={"target_hdp_proxy": "label"}).copy()


def validate_prediction_artifact(scored: pd.DataFrame, expected: pd.DataFrame,
                                 score_names: list[str], *, name: str) -> None:
    """Check cached scores against the exact source rows before using them."""
    required = {"source_year", "source_row", "fold", *OUTCOMES, *score_names}
    label_column = "label" if "label" in scored else "target_hdp_proxy"
    required.remove("target_hdp_proxy")
    required.add(label_column)
    missing = required.difference(scored.columns)
    if missing:
        raise ValueError(f"{name}: missing prediction columns {sorted(missing)}")
    if len(scored) != len(expected):
        raise ValueError(f"{name}: prediction row count differs from source")
    for field in ("source_year", "source_row", "fold"):
        if not np.array_equal(scored[field].to_numpy(), expected[field].to_numpy()):
            raise ValueError(f"{name}: {field} differs from source or row order changed")
    for field in OUTCOMES:
        actual = scored[label_column] if field == "target_hdp_proxy" else scored[field]
        reference = expected[field]
        if not np.array_equal(actual.to_numpy(dtype=float),
                              reference.to_numpy(dtype=float), equal_nan=True):
            raise ValueError(f"{name}: {field} differs from source")
    checked = {"source_year", "source_row", "fold", label_column,
               *OUTCOMES, *score_names, "early_core_fallback"}
    for field in scored.columns:
        if field not in checked and field in expected.columns and not np.array_equal(
                scored[field].to_numpy(dtype=float), expected[field].to_numpy(dtype=float),
                equal_nan=True):
            raise ValueError(f"{name}: audit field {field} differs from source")
    for field in score_names:
        probabilities = scored[field].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"{name}: invalid probabilities in {field}")
    if "early_core_fallback" in scored:
        fallback = expected.art_source_reported.eq(0).to_numpy()
        if not np.array_equal(scored.early_core_fallback.to_numpy(dtype=bool), fallback):
            raise ValueError(f"{name}: early-model reporting route differs from source")
        if not np.array_equal(scored.loc[fallback, "hist_gb_early"].to_numpy(),
                              scored.loc[fallback, "hist_gb"].to_numpy()):
            raise ValueError(f"{name}: unreported ART rows lack core-model fallback")


def _early_with_reporting_fallback(model, frame: pd.DataFrame, core_scores: np.ndarray) -> np.ndarray:
    """Use ART only where both upstream reporting flags permit interpretation."""
    if "art_source_reported" not in frame:
        raise ValueError("ART reporting status is required for early scoring")
    reportable = frame.art_source_reported.eq(1).to_numpy()
    result = np.asarray(core_scores, dtype=np.float32).copy()
    if reportable.any():
        result[reportable] = _predict_model("hist_gb_early", model, frame.loc[reportable])
    return result


def _validated_fold_cache(prediction_path: Path, metric_path: Path,
                          expected: pd.DataFrame, score_names: list[str], fold: int):
    cached = pd.read_parquet(prediction_path)
    validate_prediction_artifact(cached, expected, score_names,
                                 name=f"pooled fold {fold}")
    saved_metrics = pd.read_csv(metric_path)
    if (len(saved_metrics) != len(score_names) or
            set(saved_metrics.model) != set(score_names) or
            not saved_metrics.fold.eq(fold).all()):
        raise ValueError(f"Pooled fold {fold}: saved metric rows are incomplete")
    y = cached.label.to_numpy(dtype=np.uint8)
    for row in saved_metrics.itertuples():
        calculated = metrics(y, cached[row.model].to_numpy())
        if row.n != len(cached) or row.events != int(y.sum()) or any(
            not np.isclose(getattr(row, metric), value, rtol=1e-8, atol=1e-8)
            for metric, value in calculated.items()
        ):
            raise ValueError(f"Pooled fold {fold}: saved metrics disagree with predictions")
    return cached, saved_metrics


def _atomic_model_dump(model, path: Path) -> None:
    """Do not expose a partially serialized model after an interrupted fit."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_model_artifacts(scored: pd.DataFrame, expected: pd.DataFrame,
                              model_paths: dict[str, Path], *, name: str) -> None:
    """A saved score table must still agree with its corresponding model files."""
    if not len(expected) or len(scored) != len(expected):
        raise ValueError(f"{name}: model-check rows differ from source")
    positions = np.linspace(0, len(expected) - 1, min(2048, len(expected)), dtype=int)
    sample = expected.iloc[positions]
    saved = scored.iloc[positions]
    for model_name, path in model_paths.items():
        if not path.exists():
            raise ValueError(f"{name}: missing model artifact: {path}")
        model = joblib.load(path)
        if model_name == "hist_gb_early":
            p = _early_with_reporting_fallback(model, sample,
                saved["hist_gb"].to_numpy(dtype=np.float32))
        else:
            p = _predict_model(model_name, model, sample)
        if not np.allclose(p, saved[model_name].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"{name}: model file disagrees with scores: {model_name}")
        del model


def _validate_secondary_model_artifacts(scored: pd.DataFrame, expected: pd.DataFrame,
                                         model_paths: dict[str, Path]) -> None:
    """Reconcile saved composite model files with their scored 2024 rows."""
    if not len(expected) or len(scored) != len(expected):
        raise ValueError("Secondary model-check rows differ from source")
    positions = np.linspace(0, len(expected) - 1, min(2048, len(expected)), dtype=int)
    sample = expected.iloc[positions]
    saved = scored.iloc[positions]
    for name, path in model_paths.items():
        if not path.exists():
            raise ValueError(f"Missing secondary model artifact: {path}")
        model = joblib.load(path)
        p = model.predict_proba(sample[FEATURES])[:, 1].astype(np.float32)
        if not np.allclose(p, saved[name].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"Secondary model file disagrees with scores: {name}")
        del model


def fit_folds(resume: bool = True) -> pd.DataFrame:
    """Pooled five-fold OOF fit over every 2023 and 2024 eligible record."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "fold_run.json"
    fingerprint = run_fingerprint()
    design = "pooled 2023+2024, 5 folds"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    legacy_extension = False
    if manifest and manifest.get("fingerprint") != fingerprint:
        prepared_sha256 = json.loads((OUT / "audit.json").read_text())["prepared_file"]["sha256"]
        legacy_extension = (resume and manifest.get("models") == LEGACY_MODEL_NAMES and
                            manifest.get("features") == FEATURES and
                            manifest.get("design") == design and
                            manifest.get("fingerprint") == LEGACY_POOLED_FINGERPRINT and
                            prepared_sha256 == LEGACY_PREPARED_SHA256)
        if not legacy_extension:
            raise ValueError("Old fold artifacts have another configuration; archive them before rerun")
    if not legacy_extension:
        manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "models": MODEL_NAMES,
                                             "features": FEATURES, "design": design}, indent=2))
    columns = list(dict.fromkeys(["source_year", "source_row", "fold", "rf_phype"]
                                  + OUTCOMES + FEATURES + QUALITY_MASKS + ["priorlive", "priordead",
                                  "rf_inftr", "rf_artec", "f_rf_inft", "f_rf_inf_art", "mrace6", "mhisp_r", "meduc"]))
    tqdm.write("Loading audited pooled cohort...")
    all_data = _verify_and_load(columns)
    validate_core_input(all_data)
    validate_early_input(all_data)
    if set(all_data.fold.unique()) != set(range(5)):
        raise ValueError("Fold codes must be 0..4")
    rows = []
    for fold in tqdm(range(5), desc="Pooled CV", unit="fold"):
        prediction_path = OUT / f"oof_fold_{fold}.parquet"
        metric_path = OUT / f"fold_metrics_{fold}.csv"
        model_paths = [MODEL_DIR / f"fold_{fold}_{name}.joblib" for name in MODEL_NAMES if name != "prevalence"]
        if (resume and prediction_path.exists() and metric_path.exists() and
                all(x.exists() for x in model_paths) and
                set(MODEL_NAMES).issubset(pq.read_schema(prediction_path).names) and
                set(pd.read_csv(metric_path).model) == set(MODEL_NAMES)):
            expected = all_data.loc[all_data.fold.eq(fold)]
            cached, saved_metrics = _validated_fold_cache(
                prediction_path, metric_path, expected, MODEL_NAMES, fold)
            _validate_model_artifacts(cached, expected,
                {name: MODEL_DIR / f"fold_{fold}_{name}.joblib" for name in MODEL_NAMES[1:]},
                name=f"pooled fold {fold}")
            rows.append(saved_metrics)
            continue
        train = all_data.loc[all_data.fold.ne(fold)]
        valid = all_data.loc[all_data.fold.eq(fold)]
        y_valid = valid.target_hdp_proxy.to_numpy(dtype=np.uint8)
        if legacy_extension:
            legacy_paths = [MODEL_DIR / f"fold_{fold}_{name}.joblib" for name in LEGACY_MODEL_NAMES[1:]]
            if not prediction_path.exists() or not metric_path.exists() or not all(
                path.exists() for path in legacy_paths
            ):
                raise ValueError(f"Pooled fold {fold}: incomplete legacy files; cannot safely extend")
            old, _ = _validated_fold_cache(prediction_path, metric_path, valid,
                                           LEGACY_MODEL_NAMES, fold)
            _validate_model_artifacts(old, valid,
                {name: MODEL_DIR / f"fold_{fold}_{name}.joblib"
                 for name in LEGACY_MODEL_NAMES[1:]}, name=f"legacy pooled fold {fold}")
            predictions = {name: old[name].to_numpy(dtype=np.float32)
                           for name in LEGACY_MODEL_NAMES}
            fit_names = [name for name in MODEL_NAMES[1:] if name not in LEGACY_MODEL_NAMES]
            del old
        else:
            predictions = {"prevalence": np.full(len(valid),
                          train.target_hdp_proxy.mean(), dtype=np.float32)}
            fit_names = MODEL_NAMES[1:]
        for name in tqdm(fit_names, desc=f"Fold {fold}: fit models", unit="model", leave=False):
            model = _fit_model(name, train, f"Fold {fold}: {name}")
            predictions[name] = _predict_model(name, model, valid)
            _atomic_model_dump(model, MODEL_DIR / f"fold_{fold}_{name}.joblib")
            del model
        oof = _prediction_frame(valid)
        for name, p in predictions.items():
            oof[name] = p
        validate_prediction_artifact(oof, valid, MODEL_NAMES,
                                     name=f"pooled fold {fold}")
        fold_metrics = pd.DataFrame([{"fold": fold, "model": name, "n": len(valid),
                                      "events": int(y_valid.sum()), **metrics(y_valid, p)}
                                     for name, p in predictions.items()])
        prediction_tmp = prediction_path.with_suffix(".parquet.tmp")
        metric_tmp = metric_path.with_suffix(".csv.tmp")
        oof.to_parquet(prediction_tmp, index=False)
        fold_metrics.to_csv(metric_tmp, index=False)
        prediction_tmp.replace(prediction_path)
        metric_tmp.replace(metric_path)
        rows.append(fold_metrics)
        del train, valid, oof
        gc.collect()
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT / "fold_metrics_all.csv", index=False)
    if legacy_extension:
        manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "models": MODEL_NAMES,
                                             "features": FEATURES, "design": design,
                                             "extended_from": manifest["fingerprint"]}, indent=2))
    return result


def fit_temporal(resume: bool = True) -> pd.DataFrame:
    """Five 2023 development folds, then one frozen 2023-to-2024 model test.

    Sensitivity models with conception mode or demographics are fitted only here.
    Do not inspect 2024 labels to select a winning model or tune parameters.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "temporal_run.json"
    fingerprint = run_fingerprint()
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Old temporal artifacts have another configuration; archive them before rerun")
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "development": 2023,
                                         "test": 2024, "models": MODEL_NAMES + TEMPORAL_SENSITIVITIES}, indent=2))
    names = MODEL_NAMES + TEMPORAL_SENSITIVITIES
    columns = list(dict.fromkeys(["source_year", "source_row", "fold", "rf_phype"] + OUTCOMES
                                  + FEATURES + EARLY_FEATURES + DEMOGRAPHIC_FEATURES + QUALITY_MASKS
                                  + ["priorlive", "priordead", "f_rf_inft", "f_rf_inf_art"]))
    all_data = _verify_and_load(columns)
    validate_core_input(all_data)
    validate_early_input(all_data)
    dev = all_data.loc[all_data.source_year.eq(2023)].copy()
    test = all_data.loc[all_data.source_year.eq(2024)].copy()
    del all_data
    if set(dev.fold.unique()) != set(range(5)):
        raise ValueError("2023 development folds must cover 0..4")
    dev_parts = []
    for fold in tqdm(range(5), desc="2023 development CV", unit="fold"):
        path = OUT / f"temporal_dev_fold_{fold}.parquet"
        if resume and path.exists():
            part = pd.read_parquet(path)
            validate_prediction_artifact(part, dev.loc[dev.fold.eq(fold)], names,
                                         name=f"temporal development fold {fold}")
            dev_parts.append(part)
            continue
        train = dev.loc[dev.fold.ne(fold)]
        valid = dev.loc[dev.fold.eq(fold)]
        predictions = {"prevalence": np.full(len(valid), train.target_hdp_proxy.mean(), dtype=np.float32)}
        for name in tqdm(names[1:], desc=f"2023 fold {fold}: models", unit="model", leave=False):
            model = _fit_model(name, train, f"2023 fold {fold}: {name}")
            predictions[name] = (_early_with_reporting_fallback(model, valid, predictions["hist_gb"])
                                 if name == "hist_gb_early" else _predict_model(name, model, valid))
            del model
        part = _prediction_frame(valid)
        part["early_core_fallback"] = part.art_source_reported.eq(0).astype(np.int8)
        for name, p in predictions.items():
            part[name] = p
        validate_prediction_artifact(part, valid, names,
                                     name=f"temporal development fold {fold}")
        temporary = path.with_suffix(".parquet.tmp")
        part.to_parquet(temporary, index=False)
        temporary.replace(path)
        dev_parts.append(part)
        del train, valid, part
        gc.collect()
    dev_oof = pd.concat(dev_parts, ignore_index=True)
    if len(dev_oof) != len(dev) or dev_oof.duplicated(["source_year", "source_row"]).any():
        raise ValueError("2023 development OOF predictions do not cover the cohort exactly once")
    dev_path = OUT / "temporal_dev_oof.parquet"
    temporary = dev_path.with_suffix(".parquet.tmp")
    dev_oof.to_parquet(temporary, index=False)
    temporary.replace(dev_path)
    test_path = OUT / "temporal_test_2024.parquet"
    model_paths = [MODEL_DIR / f"temporal_2023_{name}.joblib" for name in names[1:]]
    if resume and test_path.exists() and all(path.exists() for path in model_paths):
        scored = pd.read_parquet(test_path)
    else:
        scored = _prediction_frame(test)
        scored["early_core_fallback"] = scored.art_source_reported.eq(0).astype(np.int8)
        scored["prevalence"] = np.float32(dev.target_hdp_proxy.mean())
        for name in tqdm(names[1:], desc="Final 2023 fits", unit="model"):
            model = _fit_model(name, dev, f"Final 2023: {name}")
            scored[name] = (_early_with_reporting_fallback(model, test, scored["hist_gb"].to_numpy())
                            if name == "hist_gb_early" else _predict_model(name, model, test))
            _atomic_model_dump(model, MODEL_DIR / f"temporal_2023_{name}.joblib")
            del model
        temporary = test_path.with_suffix(".parquet.tmp")
        scored.to_parquet(temporary, index=False)
        temporary.replace(test_path)
    validate_prediction_artifact(scored, test, names, name="temporal 2024 test")
    _validate_model_artifacts(scored, test,
        {name: MODEL_DIR / f"temporal_2023_{name}.joblib" for name in names[1:]},
        name="temporal 2024 test")
    if len(scored) != len(test) or not scored.source_year.eq(2024).all():
        raise ValueError("Temporal test does not cover 2024 exactly")
    result = pd.DataFrame([{"model": name, "n": len(scored),
                            "events": int(scored.label.sum()),
                            **metrics(scored.label.to_numpy(), scored[name].to_numpy())}
                           for name in names])
    result.to_csv(OUT / "temporal_metrics_2024.csv", index=False)
    return result


def fit_secondary_temporal(resume: bool = True) -> pd.DataFrame:
    """Train separate HGBs for the two *composite* endpoints on 2023 only.

    All scoring rows, including unknown gestation, are retained. A missing
    composite label is excluded from fitting and from endpoint evaluation.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "secondary_temporal_run.json"
    fingerprint = run_fingerprint()
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Old secondary temporal artifacts have another configuration")
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "development": 2023,
        "test": 2024, "targets": SECONDARY_TARGETS, "features": FEATURES,
        "interpretation": "recorded HDP proxy AND preterm live birth, not confirmed preterm preeclampsia"}, indent=2))
    columns = list(dict.fromkeys(["source_year", "source_row", "fold", "rf_phype",
        "oegest_comb"] + OUTCOMES + FEATURES + QUALITY_MASKS))
    all_data = _verify_and_load(columns)
    validate_core_input(all_data)
    dev = all_data.loc[all_data.source_year.eq(2023)].copy()
    test = all_data.loc[all_data.source_year.eq(2024)].copy()
    del all_data
    if set(dev.fold.unique()) != set(range(5)):
        raise ValueError("2023 secondary development folds must cover 0..4")
    identity = ["source_year", "source_row", "fold", *OUTCOMES, "oegest_comb"]
    parts = []
    for fold in tqdm(range(5), desc="2023 composite development CV", unit="fold"):
        path = OUT / f"secondary_dev_fold_{fold}.parquet"
        if resume and path.exists():
            part = pd.read_parquet(path)
        else:
            train = dev.loc[dev.fold.ne(fold)]
            valid = dev.loc[dev.fold.eq(fold)]
            part = valid[identity].copy()
            for model_name, target in tqdm(SECONDARY_TARGETS.items(),
                                           desc=f"2023 fold {fold}: composites", unit="model", leave=False):
                known_train = train.loc[train[target].notna()]
                model = _fit_hgb(known_train, FEATURES, f"2023 fold {fold}: {model_name}", target)
                part[model_name] = model.predict_proba(valid[FEATURES])[:, 1].astype(np.float32)
                del model
            temporary = path.with_suffix(".parquet.tmp")
            part.to_parquet(temporary, index=False)
            temporary.replace(path)
        validate_prediction_artifact(part, dev.loc[dev.fold.eq(fold)],
                                     list(SECONDARY_TARGETS),
                                     name=f"secondary development fold {fold}")
        parts.append(part)
        gc.collect()
    dev_oof = pd.concat(parts, ignore_index=True)
    if len(dev_oof) != len(dev) or dev_oof.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Secondary development OOF does not cover 2023 exactly once")
    dev_path = OUT / "secondary_dev_oof.parquet"
    temporary = dev_path.with_suffix(".parquet.tmp")
    dev_oof.to_parquet(temporary, index=False)
    temporary.replace(dev_path)
    del dev_oof, parts
    gc.collect()
    test_path = OUT / "secondary_test_2024.parquet"
    model_paths = [MODEL_DIR / f"secondary_2023_{name}.joblib" for name in SECONDARY_TARGETS]
    if resume and test_path.exists() and all(path.exists() for path in model_paths):
        scored = pd.read_parquet(test_path)
    else:
        scored = test[identity].copy()
        for model_name, target in tqdm(SECONDARY_TARGETS.items(),
                                       desc="Final 2023 composite fits", unit="model"):
            known_dev = dev.loc[dev[target].notna()]
            model = _fit_hgb(known_dev, FEATURES, f"Final 2023: {model_name}", target)
            scored[model_name] = model.predict_proba(test[FEATURES])[:, 1].astype(np.float32)
            _atomic_model_dump(model, MODEL_DIR / f"secondary_2023_{model_name}.joblib")
            del model
        temporary = test_path.with_suffix(".parquet.tmp")
        scored.to_parquet(temporary, index=False)
        temporary.replace(test_path)
    validate_prediction_artifact(scored, test, list(SECONDARY_TARGETS),
                                 name="secondary 2024 test")
    _validate_secondary_model_artifacts(scored, test,
        {name: MODEL_DIR / f"secondary_2023_{name}.joblib" for name in SECONDARY_TARGETS})
    if len(scored) != len(test) or not scored.source_year.eq(2024).all() or \
            scored.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Secondary test does not cover 2024 exactly once")
    rows = []
    for model_name, target in SECONDARY_TARGETS.items():
        known = scored[target].notna()
        y = scored.loc[known, target].to_numpy(dtype=np.uint8)
        p = scored.loc[known, model_name].to_numpy()
        rows.append({"model": model_name, "target": target, "n": int(known.sum()),
                     "events": int(y.sum()), **metrics(y, p)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "secondary_temporal_metrics_2024.csv", index=False)
    return result
