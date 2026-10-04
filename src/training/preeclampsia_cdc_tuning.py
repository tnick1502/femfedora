"""Auditable 2023-only grid search and temporal scoring for CDC HDP-proxy."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import ParameterGrid

from src.data.preeclampsia_cdc import FEATURES, OUT, QUALITY_MASKS, SEED
from src.progress import tqdm
from src.training.preeclampsia_cdc import (MODEL_DIR, MODEL_NAMES, _fit_model,
    _predict_model, _verify_and_load, metrics, run_fingerprint, validate_core_input)

TUNING_DIR = OUT / "tuning_2023"
SELECTION_METRIC = "log_loss"
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "catboost",
                    "lightgbm", "pyarrow", "joblib")
GRID = {
    "prevalence": [{}],
    "logistic": list(ParameterGrid({"C": [0.1, 1.0, 10.0]})),
    "logistic_spline": list(ParameterGrid({"C": [0.3, 1.0, 3.0],
                                            "n_knots": [4, 5]})),
    "hist_gb": list(ParameterGrid({"max_leaf_nodes": [7, 15],
                                    "l2_regularization": [1.0, 10.0]})),
    "catboost": list(ParameterGrid({"depth": [4, 5],
                                     "l2_leaf_reg": [3.0, 8.0]})),
    "lightgbm": list(ParameterGrid({"num_leaves": [7, 15],
                                     "min_child_samples": [200, 500]})),
}
assert list(GRID) == MODEL_NAMES
BASELINE_PARAMS = {
    "prevalence": {}, "logistic": {"C": 1.0},
    "logistic_spline": {"C": 1.0, "n_knots": 5},
    "hist_gb": {"max_leaf_nodes": 15, "l2_regularization": 10.0},
    "catboost": {"depth": 5, "l2_leaf_reg": 8.0},
    "lightgbm": {"num_leaves": 15, "min_child_samples": 200},
}
assert all(BASELINE_PARAMS[name] in candidates for name, candidates in GRID.items())


def candidate_id(model: str, params: dict) -> str:
    payload = json.dumps({"model": model, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def runtime_versions() -> dict[str, str]:
    """Prevent a resumed grid from silently mixing fitted library versions."""
    return {"python": platform.python_version(),
            **{package: version(package) for package in RUNTIME_PACKAGES}}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(best: dict, name: str, stage: str, n: int, events: int) -> dict:
    return {"fingerprint": best["fingerprint"], "stage": stage, "model": name,
            "candidate_id": best["best_by_model"][name]["candidate_id"],
            "params": best["best_by_model"][name]["params"],
            "training_n": n, "training_events": events}


def _checked_model_checkpoint(path: Path, identity: dict) -> bool:
    """Reuse a completed fit only when its sidecar and file both match."""
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        return False
    saved = json.loads(sidecar.read_text())
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Saved model checkpoint belongs to another fit: {path}")
    if saved.get("sha256") != _file_sha256(path):
        raise ValueError(f"Saved model checkpoint hash disagrees with file: {path}")
    return True


def _save_model_checkpoint(model, path: Path, identity: dict) -> None:
    temporary = path.with_suffix(".joblib.tmp")
    sidecar = path.with_suffix(".json")
    sidecar_tmp = sidecar.with_suffix(".json.tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        temporary.replace(path)
        sidecar_tmp.write_text(json.dumps({**identity, "sha256": _file_sha256(path)}, indent=2))
        sidecar_tmp.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)


def tuning_fingerprint() -> str:
    spec = {"baseline_fingerprint": run_fingerprint(), "grid": GRID,
            "selection_metric": SELECTION_METRIC, "development_year": 2023,
            "seed": SEED, "runtime_versions": runtime_versions(),
            "tuning_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _input_data() -> pd.DataFrame:
    columns = list(dict.fromkeys(["source_year", "source_row", "fold", "rf_phype",
                                   "target_hdp_proxy"] + FEATURES + QUALITY_MASKS))
    data = _verify_and_load(columns)
    validate_core_input(data)
    if set(data.source_year.unique()) != {2023, 2024} or set(data.fold.unique()) != set(range(5)):
        raise ValueError("Tuning needs both prepared years and all five original folds")
    return data


def _paths(model: str, key: str, fold: int) -> tuple[Path, Path]:
    stem = f"{model}_{key}_fold_{fold}"
    return TUNING_DIR / f"{stem}.parquet", TUNING_DIR / f"{stem}.json"


def _prediction_table(valid: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    score = np.asarray(p, dtype=np.float32)
    if len(score) != len(valid) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Tuning produced invalid probabilities")
    return pd.DataFrame({"source_year": valid.source_year.to_numpy(dtype=np.int16),
                         "source_row": valid.source_row.to_numpy(),
                         "fold": valid.fold.to_numpy(dtype=np.int8),
                         "label": valid.target_hdp_proxy.to_numpy(dtype=np.uint8),
                         "prediction": score})


def _read_cached(path: Path, metric_path: Path, valid: pd.DataFrame,
                 model: str, params: dict, key: str, fold: int) -> dict | None:
    if not path.exists() or not metric_path.exists():
        return None
    saved = json.loads(metric_path.read_text())
    if saved.get("score_sha256") != _file_sha256(path):
        raise ValueError(f"Corrupt or stale tuning score hash: {path}")
    scored = pd.read_parquet(path)
    required = ["source_year", "source_row", "fold", "label", "prediction"]
    if list(scored.columns) != required:
        raise ValueError(f"Corrupt or stale tuning columns: {path}")
    expected = _prediction_table(valid, scored.prediction.to_numpy())
    if not scored.equals(expected):
        raise ValueError(f"Corrupt or stale tuning scores: {path}")
    if (saved.get("model") != model or saved.get("candidate_id") != key or
            saved.get("params") != params or saved.get("fold") != fold or
            saved.get("fingerprint") != tuning_fingerprint() or
            saved.get("n") != len(valid) or
            saved.get("events") != int(valid.target_hdp_proxy.sum())):
        raise ValueError(f"Corrupt or stale tuning metadata: {metric_path}")
    calculated = metrics(scored.label.to_numpy(), scored.prediction.to_numpy())
    if any(not np.isclose(saved[name], value, rtol=1e-8, atol=1e-8)
           for name, value in calculated.items()):
        raise ValueError(f"Tuning metrics disagree with scores: {metric_path}")
    return saved


def _save_result(path: Path, metric_path: Path, scored: pd.DataFrame, row: dict) -> dict:
    temporary_scores = path.with_suffix(".parquet.tmp")
    temporary_metrics = metric_path.with_suffix(".json.tmp")
    try:
        scored.to_parquet(temporary_scores, index=False)
        saved = {**row, "score_sha256": _file_sha256(temporary_scores)}
        temporary_metrics.write_text(json.dumps(saved, indent=2, sort_keys=True))
        temporary_scores.replace(path)
        temporary_metrics.replace(metric_path)
        return saved
    finally:
        temporary_scores.unlink(missing_ok=True)
        temporary_metrics.unlink(missing_ok=True)


def summarize_tuning(fold_results: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate; weighted log loss is exact for pooled 2023 OOF."""
    expected = 5 * sum(len(candidates) for candidates in GRID.values())
    if len(fold_results) != expected or fold_results.duplicated(
            ["model", "candidate_id", "fold"]).any():
        raise ValueError("Tuning results do not cover every candidate and fold once")
    expected_keys = {(name, candidate_id(name, params))
                     for name, candidates in GRID.items() for params in candidates}
    if set(zip(fold_results.model, fold_results.candidate_id)) != expected_keys:
        raise ValueError("Tuning candidate identifiers disagree with the fixed grid")
    rows = []
    for (name, key), part in fold_results.groupby(["model", "candidate_id"], sort=False):
        if set(part.fold) != set(range(5)) or not part.source_year.eq(2023).all():
            raise ValueError(f"Incomplete 2023 development folds for {name}/{key}")
        weights = part.n.to_numpy()
        rows.append({"model": name, "candidate_id": key,
                     "params_json": part.params_json.iloc[0],
                     "n_oof": int(part.n.sum()), "events_oof": int(part.events.sum()),
                     "cv_log_loss": float(np.average(part.log_loss, weights=weights)),
                     "cv_brier": float(np.average(part.brier, weights=weights)),
                     "auroc_fold_mean": float(part.auroc.mean()),
                     "auroc_fold_sd": float(part.auroc.std(ddof=1)),
                     "auprc_fold_mean": float(part.auprc.mean()),
                     "auprc_fold_sd": float(part.auprc.std(ddof=1)),
                     "log_loss_fold_sd": float(part.log_loss.std(ddof=1)),
                     "fit_seconds_total": float(part.fit_seconds.sum()),
                     "rss_gb_max": float(part.rss_gb.max())})
    return pd.DataFrame(rows).sort_values(["model", "cv_log_loss", "candidate_id"]).reset_index(drop=True)


def ablation_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Paired fold changes versus each model's existing fixed baseline."""
    rows = []
    for name in GRID:
        base_key = candidate_id(name, BASELINE_PARAMS[name])
        baseline = fold_results.loc[(fold_results.model == name) &
                                    (fold_results.candidate_id == base_key)].set_index("fold")
        if len(baseline) != 5:
            raise ValueError(f"Missing baseline folds for {name}")
        for candidate in GRID[name]:
            key = candidate_id(name, candidate)
            current = fold_results.loc[(fold_results.model == name) &
                                       (fold_results.candidate_id == key)].set_index("fold")
            if set(current.index) != set(baseline.index):
                raise ValueError(f"Unpaired candidate folds for {name}/{key}")
            for fold in range(5):
                rows.append({"model": name, "candidate_id": key, "fold": fold,
                             "params_json": json.dumps(candidate, sort_keys=True),
                             "is_fixed_baseline": key == base_key,
                             "delta_log_loss": float(current.loc[fold, "log_loss"] -
                                                     baseline.loc[fold, "log_loss"]),
                             "delta_brier": float(current.loc[fold, "brier"] -
                                                  baseline.loc[fold, "brier"]),
                             "delta_auroc": float(current.loc[fold, "auroc"] -
                                                  baseline.loc[fold, "auroc"]),
                             "delta_auprc": float(current.loc[fold, "auprc"] -
                                                  baseline.loc[fold, "auprc"])})
    return pd.DataFrame(rows)


def select_best(summary: pd.DataFrame, fingerprint: str) -> dict:
    """Select one configuration per model, then one global log-loss winner."""
    if set(summary.model) != set(GRID):
        raise ValueError("Cannot select from an incomplete model grid")
    if summary.n_oof.nunique() != 1 or summary.events_oof.nunique() != 1:
        raise ValueError("Candidates were not evaluated on the same development rows")
    selected = {}
    for name in GRID:
        row = summary.loc[summary.model.eq(name)].sort_values(
            ["cv_log_loss", "candidate_id"]).iloc[0]
        selected[name] = {"candidate_id": row.candidate_id,
                          "params": json.loads(row.params_json),
                          "cv_log_loss": float(row.cv_log_loss),
                          "cv_brier": float(row.cv_brier),
                          "auroc_fold_mean": float(row.auroc_fold_mean),
                          "auprc_fold_mean": float(row.auprc_fold_mean)}
    winner = min(GRID, key=lambda name: (selected[name]["cv_log_loss"], name))
    return {"fingerprint": fingerprint, "development_year": 2023,
            "development_n": int(summary.n_oof.iloc[0]),
            "folds": list(range(5)), "selection_metric": SELECTION_METRIC,
            "overall_winner": winner, "best_by_model": selected,
            "interpretation": "Selected on 2023 OOF only; selection scores are optimistic, and 2024 was previously inspected."}


def run_grid_search(resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every fixed candidate on the same five 2023 development folds."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = tuning_fingerprint()
    manifest_path = TUNING_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Tuning configuration changed; archive the previous search directory")
    else:
        manifest_path.write_text(json.dumps({"fingerprint": fingerprint,
            "grid": GRID, "development_year": 2023, "selection_metric": SELECTION_METRIC,
            "n_fits": 5 * sum(len(items) for items in GRID.values())}, indent=2))
    data = _input_data()
    dev = data.loc[data.source_year.eq(2023)].copy()
    del data
    if set(dev.fold.unique()) != set(range(5)):
        raise ValueError("2023 development folds must cover 0..4")
    rows = []
    total = 5 * sum(len(items) for items in GRID.values())
    with tqdm(total=total, desc="2023 hyperparameter grid", unit="fit") as bar:
        for fold in range(5):
            train = dev.loc[dev.fold.ne(fold)]
            valid = dev.loc[dev.fold.eq(fold)]
            y_valid = valid.target_hdp_proxy.to_numpy(dtype=np.uint8)
            for name, candidates in GRID.items():
                for params in candidates:
                    key = candidate_id(name, params)
                    bar.set_postfix_str(f"fold={fold} {name} {key[:6]}", refresh=False)
                    path, metric_path = _paths(name, key, fold)
                    cached = (_read_cached(path, metric_path, valid, name, params, key, fold)
                              if resume else None)
                    if cached is not None:
                        rows.append(cached)
                        bar.update()
                        continue
                    started = perf_counter()
                    if name == "prevalence":
                        prediction = np.full(len(valid), train.target_hdp_proxy.mean(),
                                             dtype=np.float32)
                    else:
                        model = _fit_model(name, train, f"2023 fold {fold}: {name} {key[:6]}",
                                           params)
                        prediction = _predict_model(name, model, valid)
                        del model
                    elapsed = perf_counter() - started
                    scored = _prediction_table(valid, prediction)
                    row = {"fingerprint": fingerprint, "source_year": 2023,
                           "fold": fold, "model": name,
                           "candidate_id": key, "params": params,
                           "params_json": json.dumps(params, sort_keys=True),
                           "n": len(valid), "events": int(y_valid.sum()),
                           "fit_seconds": elapsed,
                           "rss_gb": psutil.Process().memory_info().rss / 1e9,
                           **metrics(y_valid, scored.prediction.to_numpy())}
                    saved = _save_result(path, metric_path, scored, row)
                    rows.append(saved)
                    bar.update()
                    del scored, prediction
            del train, valid
            gc.collect()
    fold_results = pd.DataFrame(rows)
    summary = summarize_tuning(fold_results)
    ablations = ablation_results(fold_results)
    best = select_best(summary, fingerprint)
    fold_results.to_csv(TUNING_DIR / "candidate_fold_metrics.csv", index=False)
    summary_path = TUNING_DIR / "candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    ablations.to_csv(TUNING_DIR / "ablations_vs_fixed_baseline.csv", index=False)
    best["candidate_summary_sha256"] = _file_sha256(summary_path)
    (TUNING_DIR / "best_configurations.json").write_text(json.dumps(best, indent=2))
    return fold_results, summary, best


def load_best_configurations() -> dict:
    path = TUNING_DIR / "best_configurations.json"
    best = json.loads(path.read_text())
    if best.get("fingerprint") != tuning_fingerprint() or set(best.get("best_by_model", {})) != set(GRID):
        raise ValueError("Selected configurations do not match the current grid and data")
    for name, chosen in best["best_by_model"].items():
        if (chosen["params"] not in GRID[name] or
                chosen["candidate_id"] != candidate_id(name, chosen["params"])):
            raise ValueError(f"Selected candidate is absent from the fixed grid: {name}")
    if best.get("overall_winner") not in GRID:
        raise ValueError("Selected overall winner is absent from the fixed grid")
    summary_path = TUNING_DIR / "candidate_summary.csv"
    if (not summary_path.exists() or
            best.get("candidate_summary_sha256") != _file_sha256(summary_path)):
        raise ValueError("Selected configurations disagree with the saved grid summary")
    summary = pd.read_csv(summary_path, dtype={"model": str, "candidate_id": str,
                                                "params_json": str})
    expected_keys = {(name, candidate_id(name, params))
                     for name, candidates in GRID.items() for params in candidates}
    if len(summary) != len(expected_keys) or set(zip(summary.model, summary.candidate_id)) != expected_keys:
        raise ValueError("Saved grid summary does not cover every candidate")
    reconstructed = select_best(summary, best["fingerprint"])
    if (best["overall_winner"] != reconstructed["overall_winner"] or
            any(best["best_by_model"][name]["candidate_id"] !=
                reconstructed["best_by_model"][name]["candidate_id"] for name in GRID)):
        raise ValueError("Saved winner does not minimize the declared selection metric")
    return best


def selected_oof_predictions() -> pd.DataFrame:
    """Read only the five selected candidates for calibration and rank plots."""
    best = load_best_configurations()
    source = _input_data().loc[lambda frame: frame.source_year.eq(2023)]
    parts = []
    for fold in range(5):
        valid = source.loc[source.fold.eq(fold)]
        frame = None
        identity = None
        for name, chosen in best["best_by_model"].items():
            path, metadata_path = _paths(name, chosen["candidate_id"], fold)
            if _read_cached(path, metadata_path, valid, name, chosen["params"],
                            chosen["candidate_id"], fold) is None:
                raise ValueError(f"Selected candidate score is missing: {path}")
            scored = pd.read_parquet(path)
            current_identity = scored.drop(columns="prediction")
            if identity is None:
                identity = current_identity
                frame = identity.copy()
            elif not current_identity.equals(identity):
                raise ValueError(f"Selected scores disagree on source rows in fold {fold}")
            frame[name] = scored.prediction.to_numpy()
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["source_year", "source_row"]).any() or not result.source_year.eq(2023).all():
        raise ValueError("Selected OOF scores are incomplete or duplicated")
    if len(result) != best["development_n"]:
        raise ValueError("Selected OOF scores do not cover all 2023 development rows")
    return result


def score_selected_on_2024(resume: bool = True) -> pd.DataFrame:
    """Exploratory temporal check; 2024 labels never enter tuning or fitting."""
    best = load_best_configurations()
    path = TUNING_DIR / "selected_temporal_2024.parquet"
    manifest_path = TUNING_DIR / "selected_temporal_2024.json"
    data = _input_data()
    dev = data.loc[data.source_year.eq(2023)].copy()
    test = data.loc[data.source_year.eq(2024)].copy()
    del data
    identity = test[["source_year", "source_row", "target_hdp_proxy"]].rename(
        columns={"target_hdp_proxy": "label"}).reset_index(drop=True)
    dev_n, dev_events = len(dev), int(dev.target_hdp_proxy.sum())
    model_paths = {name: MODEL_DIR / f"tuned_2023_{name}_{chosen['candidate_id']}.joblib"
                   for name, chosen in best["best_by_model"].items() if name != "prevalence"}
    reuse_score = resume and path.exists() and manifest_path.exists() and all(
        _checked_model_checkpoint(model_path,
            _checkpoint_identity(best, name, "temporal_2023", dev_n, dev_events))
        for name, model_path in model_paths.items())
    if reuse_score:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != best["fingerprint"]:
            raise ValueError("Saved temporal tuned test belongs to another search")
        if manifest.get("score_sha256") != _file_sha256(path):
            raise ValueError("Saved temporal tuned scores changed after fitting")
        scored = pd.read_parquet(path)
    else:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        scored = identity.copy()
        scored["prevalence"] = np.float32(dev.target_hdp_proxy.mean())
        for name in tqdm(MODEL_NAMES[1:], desc="Selected 2023 fits", unit="model"):
            chosen = best["best_by_model"][name]
            model_path = model_paths[name]
            identity_spec = _checkpoint_identity(best, name, "temporal_2023",
                dev_n, dev_events)
            if resume and _checked_model_checkpoint(model_path, identity_spec):
                model = joblib.load(model_path)
            else:
                model = _fit_model(name, dev, f"Selected 2023: {name}", chosen["params"])
                _save_model_checkpoint(model, model_path, identity_spec)
            scored[name] = _predict_model(name, model, test)
            del model
        temporary = path.with_suffix(".parquet.tmp")
        scored.to_parquet(temporary, index=False)
        temporary.replace(path)
        manifest_path.write_text(json.dumps({"fingerprint": best["fingerprint"],
            "development_year": 2023, "test_year": 2024,
            "score_sha256": _file_sha256(path),
            "warning": "2024 was previously inspected; exploratory temporal check"}, indent=2))
    if (not scored[list(identity)].equals(identity) or
            set(scored.columns) != (set(identity.columns) | set(MODEL_NAMES))):
        raise ValueError("Temporal selected scores fail identity or coverage check")
    for name in MODEL_NAMES:
        p = scored[name].to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid selected temporal probabilities: {name}")
    positions = np.linspace(0, len(test) - 1, min(2048, len(test)), dtype=int)
    for name, model_path in model_paths.items():
        model = joblib.load(model_path)
        predicted = _predict_model(name, model, test.iloc[positions])
        if not np.allclose(predicted, scored[name].to_numpy()[positions], rtol=0, atol=1e-6):
            raise ValueError(f"Selected temporal model disagrees with scores: {name}")
        del model
    y = scored.label.to_numpy(dtype=np.uint8)
    result = pd.DataFrame([{"model": name, "n": len(scored), "events": int(y.sum()),
                            **metrics(y, scored[name].to_numpy())} for name in MODEL_NAMES])
    result.to_csv(TUNING_DIR / "selected_temporal_metrics_2024.csv", index=False)
    return result
