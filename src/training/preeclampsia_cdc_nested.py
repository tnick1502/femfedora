"""Nested pooled CV for tuned CDC HDP-proxy models.

The existing five fold assignments serve as outer folds. Within each outer
training set, its four remaining fold groups serve as inner validation folds.
No outer label is passed to hyperparameter selection or fitting.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil

from src.data.preeclampsia_cdc import FEATURES, OUT, OUTCOMES, QUALITY_MASKS
from src.progress import tqdm
from src.training.preeclampsia_cdc import (
    MODEL_NAMES, _fit_model, _predict_model, _prediction_frame, _verify_and_load,
    metrics, run_fingerprint, validate_core_input, validate_prediction_artifact,
)
from src.training.preeclampsia_cdc_tuning import (
    GRID, SELECTION_METRIC, _checked_model_checkpoint, _file_sha256,
    _prediction_table, _save_model_checkpoint, candidate_id, runtime_versions,
)

NESTED_DIR = OUT / "nested_cv_2023_2024"
OUTER_FOLDS = tuple(range(5))


def nested_fingerprint() -> str:
    specification = {
        "prepared_and_models": run_fingerprint(),
        "nested_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tuning_code_sha256": hashlib.sha256(
            Path(__file__).with_name("preeclampsia_cdc_tuning.py").read_bytes()).hexdigest(),
        "grid": GRID,
        "selection_metric": SELECTION_METRIC,
        "runtime_versions": runtime_versions(),
        "outer_folds": OUTER_FOLDS,
        "inner_rule": "remaining four original folds, each held out once",
    }
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def _nested_input_data() -> pd.DataFrame:
    """Load the same input and audit fields as the fixed pooled baseline."""
    columns = list(dict.fromkeys(
        ["source_year", "source_row", "fold", "rf_phype"] + OUTCOMES +
        FEATURES + QUALITY_MASKS + ["priorlive", "priordead", "rf_inftr",
            "rf_artec", "f_rf_inf_art", "mrace6", "mhisp_r", "meduc"]
    ))
    data = _verify_and_load(columns)
    validate_core_input(data)
    if set(data.source_year.unique()) != {2023, 2024} or set(data.fold.unique()) != set(OUTER_FOLDS):
        raise ValueError("Nested CV requires both years and five prepared folds")
    return data


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _score_paths(outer: int, inner: int | None, name: str, key: str) -> tuple[Path, Path]:
    stage = f"outer_{outer}_inner_{inner}" if inner is not None else f"outer_{outer}_refit"
    path = NESTED_DIR / stage / f"{name}_{key}.parquet"
    return path, path.with_suffix(".json")


def _cached_fit(path: Path, metadata_path: Path, validation: pd.DataFrame,
                identity: dict) -> tuple[pd.DataFrame, dict] | None:
    if not path.exists() or not metadata_path.exists():
        return None
    saved = json.loads(metadata_path.read_text())
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Nested checkpoint belongs to another fit: {metadata_path}")
    if saved.get("score_sha256") != _file_sha256(path):
        raise ValueError(f"Nested checkpoint score hash differs: {path}")
    scored = pd.read_parquet(path)
    if list(scored.columns) != ["source_year", "source_row", "fold", "label", "prediction"]:
        raise ValueError(f"Nested checkpoint score schema differs: {path}")
    expected = _prediction_table(validation, scored.prediction.to_numpy())
    if not scored.equals(expected):
        raise ValueError(f"Nested checkpoint source rows or labels differ: {path}")
    calculated = metrics(scored.label.to_numpy(), scored.prediction.to_numpy())
    if any(not np.isclose(saved.get(key, np.nan), value, atol=1e-8, rtol=1e-8)
           for key, value in calculated.items()):
        raise ValueError(f"Nested checkpoint metrics differ: {metadata_path}")
    return scored, saved


def _save_fit(scored: pd.DataFrame, path: Path, metadata_path: Path, row: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(scored, path)
    saved = {**row, "score_sha256": _file_sha256(path)}
    _atomic_json(saved, metadata_path)
    return saved


def select_inner_winners(inner_metrics: pd.DataFrame, outer_fold: int, fingerprint: str) -> tuple[pd.DataFrame, dict]:
    """Choose configurations using only the four inner validation folds."""
    inner_folds = set(OUTER_FOLDS) - {outer_fold}
    expected = {(name, candidate_id(name, params), fold)
                for name, candidates in GRID.items()
                for params in candidates for fold in inner_folds}
    observed = list(zip(inner_metrics.model, inner_metrics.candidate_id, inner_metrics.inner_fold))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f"Outer {outer_fold}: incomplete or duplicated inner grid")
    if not inner_metrics.outer_fold.eq(outer_fold).all():
        raise ValueError("Inner scores contain another outer fold")
    if not inner_metrics.groupby("inner_fold")["n"].nunique().eq(1).all():
        raise ValueError("Candidates saw different validation row counts")
    if not inner_metrics.groupby("inner_fold")["events"].nunique().eq(1).all():
        raise ValueError("Candidates saw different validation labels")
    summary_rows = []
    for (name, key), part in inner_metrics.groupby(["model", "candidate_id"], sort=False):
        params = next(p for p in GRID[name] if candidate_id(name, p) == key)
        weights = part.n.to_numpy(dtype=float)
        summary_rows.append({
            "outer_fold": outer_fold, "model": name, "candidate_id": key,
            "params_json": json.dumps(params, sort_keys=True),
            "inner_n": int(part.n.sum()), "inner_events": int(part.events.sum()),
            "inner_log_loss": float(np.average(part.log_loss, weights=weights)),
            "inner_brier": float(np.average(part.brier, weights=weights)),
            "inner_auroc_fold_mean": float(part.auroc.mean()),
            "inner_auprc_fold_mean": float(part.auprc.mean()),
            "fit_seconds_total": float(part.fit_seconds.sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "inner_log_loss", "candidate_id"]).reset_index(drop=True)
    selected = {}
    for name in MODEL_NAMES:
        row = summary.loc[summary.model.eq(name)].iloc[0]
        selected[name] = {
            "candidate_id": row.candidate_id,
            "params": json.loads(row.params_json),
            "inner_log_loss": float(row.inner_log_loss),
        }
    winner = min(MODEL_NAMES, key=lambda name: (selected[name]["inner_log_loss"], name))
    choice = {
        "fingerprint": fingerprint, "outer_fold": outer_fold,
        "inner_folds": sorted(inner_folds), "selection_metric": SELECTION_METRIC,
        "best_by_model": selected, "overall_winner": winner,
    }
    return summary, choice


def _inner_grid(data: pd.DataFrame, outer: int, fingerprint: str,
                progress, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    development = data.loc[data.fold.ne(outer)]
    rows = []
    for inner in OUTER_FOLDS:
        if inner == outer:
            continue
        train = development.loc[development.fold.ne(inner)]
        validation = development.loc[development.fold.eq(inner)]
        if train.empty or validation.empty or train.fold.eq(outer).any():
            raise ValueError("Nested inner split is incomplete or includes outer holdout")
        for name, candidates in GRID.items():
            for params in candidates:
                key = candidate_id(name, params)
                identity = {
                    "fingerprint": fingerprint, "outer_fold": outer, "inner_fold": inner,
                    "model": name, "candidate_id": key, "params": params,
                    "train_n": len(train), "train_events": int(train.target_hdp_proxy.sum()),
                    "n": len(validation), "events": int(validation.target_hdp_proxy.sum()),
                }
                path, metadata_path = _score_paths(outer, inner, name, key)
                cached = _cached_fit(path, metadata_path, validation, identity) if resume else None
                if cached is None:
                    started = perf_counter()
                    if name == "prevalence":
                        predicted = np.full(len(validation), train.target_hdp_proxy.mean(),
                                            dtype=np.float32)
                    else:
                        model = _fit_model(name, train,
                            f"Outer {outer} / inner {inner}: {name} {key[:6]}", params)
                        predicted = _predict_model(name, model, validation)
                        del model
                    scored = _prediction_table(validation, predicted)
                    row = {**identity, "params_json": json.dumps(params, sort_keys=True),
                           "fit_seconds": perf_counter() - started,
                           "rss_gb": psutil.Process().memory_info().rss / 1e9,
                           **metrics(scored.label.to_numpy(), scored.prediction.to_numpy())}
                    saved = _save_fit(scored, path, metadata_path, row)
                    del scored, predicted
                else:
                    _, saved = cached
                rows.append({key: value for key, value in saved.items() if key != "score_sha256"})
                progress.set_postfix_str(f"outer={outer} inner={inner} {name} {key[:6]}", refresh=False)
                progress.update()
        del train, validation
        gc.collect()
    inner_metrics = pd.DataFrame(rows)
    summary, choice = select_inner_winners(inner_metrics, outer, fingerprint)
    root = NESTED_DIR / f"outer_{outer}"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(inner_metrics, root / "inner_candidate_metrics.csv")
    _atomic_csv(summary, root / "inner_candidate_summary.csv")
    choice["inner_summary_sha256"] = _file_sha256(root / "inner_candidate_summary.csv")
    _atomic_json(choice, root / "selection.json")
    return inner_metrics, summary, choice


def _outer_refit(data: pd.DataFrame, outer: int, choice: dict,
                 fingerprint: str, progress, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.fold.ne(outer)]
    validation = data.loc[data.fold.eq(outer)]
    if train.empty or validation.empty:
        raise ValueError("Nested outer split is empty")
    base = _prediction_table(validation, np.zeros(len(validation), dtype=np.float32))
    base = base.drop(columns="prediction")
    metric_rows = []
    for name in MODEL_NAMES:
        selected = choice["best_by_model"][name]
        key = selected["candidate_id"]
        path, metadata_path = _score_paths(outer, None, name, key)
        identity = {
            "fingerprint": fingerprint, "outer_fold": outer, "inner_fold": None,
            "model": name, "candidate_id": key, "params": selected["params"],
            "train_n": len(train), "train_events": int(train.target_hdp_proxy.sum()),
            "n": len(validation), "events": int(validation.target_hdp_proxy.sum()),
        }
        model_path = NESTED_DIR / f"outer_{outer}" / f"{name}_{key}.joblib"
        model_identity = {**identity, "stage": "nested_outer_refit"}
        checkpoint_ok = (name == "prevalence" or
                         (resume and _checked_model_checkpoint(model_path, model_identity)))
        cached = (_cached_fit(path, metadata_path, validation, identity)
                  if resume and checkpoint_ok else None)
        if cached is None:
            started = perf_counter()
            if name == "prevalence":
                predicted = np.full(len(validation), train.target_hdp_proxy.mean(),
                                    dtype=np.float32)
            else:
                if checkpoint_ok:
                    model = joblib.load(model_path)
                else:
                    model = _fit_model(name, train, f"Outer {outer} selected: {name}",
                                       selected["params"])
                    _save_model_checkpoint(model, model_path, model_identity)
                predicted = _predict_model(name, model, validation)
                del model
            scored = _prediction_table(validation, predicted)
            row = {**identity, "params_json": json.dumps(selected["params"], sort_keys=True),
                   "fit_seconds": perf_counter() - started,
                   "rss_gb": psutil.Process().memory_info().rss / 1e9,
                   **metrics(scored.label.to_numpy(), scored.prediction.to_numpy())}
            saved = _save_fit(scored, path, metadata_path, row)
        else:
            scored, saved = cached
        if name != "prevalence":
            # A matching hash is necessary; a prediction spot-check links the
            # checkpoint model to the saved outer score table as well.
            model = joblib.load(model_path)
            indices = np.linspace(0, len(validation)-1, min(256, len(validation)), dtype=int)
            check = _predict_model(name, model, validation.iloc[indices])
            if not np.allclose(check, scored.prediction.to_numpy()[indices], atol=1e-6, rtol=0):
                raise ValueError(f"Outer {outer}: selected model and scores disagree: {name}")
            del model
        base[name] = scored.prediction.to_numpy(dtype=np.float32)
        metric_rows.append({key: value for key, value in saved.items() if key != "score_sha256"})
        progress.set_postfix_str(f"outer={outer} {name} {key[:6]}", refresh=False)
        progress.update()
        del scored
        gc.collect()
    winner = choice["overall_winner"]
    base["selected"] = base[winner].to_numpy(dtype=np.float32)
    base["selection_model"] = winner
    y = base.label.to_numpy(dtype=np.uint8)
    metric_rows.append({"fingerprint": fingerprint, "outer_fold": outer,
        "inner_fold": None, "model": "selected", "candidate_id": choice["best_by_model"][winner]["candidate_id"],
        "params": choice["best_by_model"][winner]["params"],
        "params_json": json.dumps(choice["best_by_model"][winner]["params"], sort_keys=True),
        "train_n": len(train), "train_events": int(train.target_hdp_proxy.sum()),
        "n": len(validation), "events": int(y.sum()), "fit_seconds": np.nan,
        "rss_gb": np.nan, **metrics(y, base.selected.to_numpy())})
    root = NESTED_DIR / f"outer_{outer}"
    _atomic_parquet(base, root / "outer_predictions.parquet")
    _atomic_csv(pd.DataFrame(metric_rows), root / "outer_metrics.csv")
    return base, pd.DataFrame(metric_rows)


def run_nested_cv(resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune inside each outer fold, then evaluate each selected model once outside it.

    Returns inner fit rows, outer metric rows, outer choices, and all-row OOF
    probabilities. Does not select a final model from outer test performance.
    """
    data = _nested_input_data()
    if data.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Nested CV source row identifiers are duplicated")
    NESTED_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = nested_fingerprint()
    manifest_path = NESTED_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Existing nested CV artifacts use another data, code, or grid; archive them before rerun")
    else:
        _atomic_json({"fingerprint": fingerprint, "design": "5 outer x 4 inner original folds",
            "n_inner_fits": 5 * 4 * sum(map(len, GRID.values())),
            "n_outer_models": 5 * len(MODEL_NAMES), "selection_metric": SELECTION_METRIC,
            "warning": "Previously inspected outcomes mean this is not a pristine prospective test."},
            manifest_path)
    inner_parts, outer_parts, score_parts, choices = [], [], [], []
    total_inner = 5 * 4 * sum(map(len, GRID.values()))
    with tqdm(total=total_inner, desc="Nested CV: inner grid", unit="fit") as inner_progress, \
         tqdm(total=5 * len(MODEL_NAMES), desc="Nested CV: outer refits", unit="model") as outer_progress:
        for outer in OUTER_FOLDS:
            inner, _, choice = _inner_grid(data, outer, fingerprint, inner_progress, resume)
            predictions, outer_metrics = _outer_refit(data, outer, choice, fingerprint,
                                                       outer_progress, resume)
            inner_parts.append(inner)
            outer_parts.append(outer_metrics)
            score_parts.append(predictions)
            choices.append({"outer_fold": outer, "overall_winner": choice["overall_winner"],
                **{f"{name}_candidate_id": choice["best_by_model"][name]["candidate_id"]
                   for name in MODEL_NAMES}})
            gc.collect()
    inner_results = pd.concat(inner_parts, ignore_index=True)
    outer_results = pd.concat(outer_parts, ignore_index=True)
    oof = pd.concat(score_parts, ignore_index=True)
    choices = pd.DataFrame(choices)
    if len(oof) != len(data) or oof.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Nested OOF does not cover every source row exactly once")
    identity = ["source_year", "source_row", "fold"]
    expected = data[identity + ["target_hdp_proxy"]].sort_values(identity).reset_index(drop=True)
    actual = oof[identity + ["label"]].sort_values(identity).reset_index(drop=True)
    if any(not np.array_equal(actual[field].to_numpy(), expected[field].to_numpy())
           for field in identity) or not np.array_equal(
            actual.label.to_numpy(), expected.target_hdp_proxy.to_numpy()):
        raise ValueError("Nested OOF row identity or labels differ from prepared source")
    if len(outer_results) != 5 * (len(MODEL_NAMES) + 1):
        raise ValueError("Nested outer metrics are incomplete")
    if set(outer_results.model) != set(MODEL_NAMES + ["selected"]):
        raise ValueError("Nested outer models are incomplete")
    for name in MODEL_NAMES + ["selected"]:
        scores = oof[name].to_numpy()
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Nested OOF probabilities invalid: {name}")
    # Align by the already-verified source keys before adding audit columns.
    # Positional assignment avoids a second large many-column merge.
    oof = oof.sort_values(identity).reset_index(drop=True)
    data = data.sort_values(identity).reset_index(drop=True)
    audit = _prediction_frame(data)
    oof = pd.concat([oof, audit.drop(columns=identity + ["label"])], axis=1)
    validate_prediction_artifact(oof, data, MODEL_NAMES + ["selected"],
                                 name="nested pooled OOF")
    _atomic_csv(inner_results, NESTED_DIR / "inner_candidate_metrics_all.csv")
    _atomic_csv(outer_results, NESTED_DIR / "outer_fold_metrics.csv")
    _atomic_csv(choices, NESTED_DIR / "outer_choices.csv")
    _atomic_parquet(oof, NESTED_DIR / "nested_oof.parquet")
    pooled = pd.DataFrame([{"model": name, "n": len(oof),
        "events": int(oof.label.sum()), **metrics(oof.label.to_numpy(), oof[name].to_numpy())}
        for name in MODEL_NAMES + ["selected"]])
    _atomic_csv(pooled, NESTED_DIR / "pooled_metrics.csv")
    return inner_results, outer_results, choices, oof


def load_nested_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and audit completed outer predictions; never fit or tune a model."""
    manifest = json.loads((NESTED_DIR / "manifest.json").read_text())
    fingerprint = nested_fingerprint()
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("Nested evaluation files belong to another data, code, or grid")
    data = _nested_input_data()
    inner = pd.read_csv(NESTED_DIR / "inner_candidate_metrics_all.csv",
                        dtype={"candidate_id": str})
    outer = pd.read_csv(NESTED_DIR / "outer_fold_metrics.csv",
                        dtype={"candidate_id": str})
    choices = pd.read_csv(NESTED_DIR / "outer_choices.csv", dtype=str)
    choices["outer_fold"] = choices.outer_fold.astype(int)
    oof = pd.read_parquet(NESTED_DIR / "nested_oof.parquet")
    if (len(inner) != 5 * 4 * sum(map(len, GRID.values())) or
            len(outer) != 5 * (len(MODEL_NAMES) + 1) or
            len(choices) != 5 or len(oof) != len(data)):
        raise ValueError("Nested evaluation files have incomplete coverage")
    if (set(choices.outer_fold) != set(OUTER_FOLDS) or
            choices.outer_fold.duplicated().any() or
            oof.duplicated(["source_year", "source_row"]).any()):
        raise ValueError("Nested evaluation fold or source identity is duplicated")
    identity = ["source_year", "source_row", "fold"]
    expected = data.sort_values(identity).reset_index(drop=True)
    scored = oof.sort_values(identity).reset_index(drop=True)
    validate_prediction_artifact(scored, expected, MODEL_NAMES + ["selected"],
                                 name="saved nested pooled OOF")
    for fold in OUTER_FOLDS:
        part = scored.loc[scored.fold.eq(fold)]
        fold_metrics = outer.loc[outer.outer_fold.eq(fold)]
        if (len(fold_metrics) != len(MODEL_NAMES) + 1 or
                set(fold_metrics.model) != set(MODEL_NAMES + ["selected"])):
            raise ValueError(f"Nested outer fold {fold} has missing models")
        _, choice = select_inner_winners(inner.loc[inner.outer_fold.eq(fold)],
                                         fold, fingerprint)
        selected = choices.loc[choices.outer_fold.eq(fold)].iloc[0]
        if selected.overall_winner != choice["overall_winner"]:
            raise ValueError(f"Nested outer fold {fold} selection changed")
        for name in MODEL_NAMES:
            if selected[f"{name}_candidate_id"] != choice["best_by_model"][name]["candidate_id"]:
                raise ValueError(f"Nested outer fold {fold} selected candidate changed: {name}")
        if (not part.selection_model.eq(choice["overall_winner"]).all() or
                not np.array_equal(part.selected.to_numpy(),
                                   part[choice["overall_winner"]].to_numpy())):
            raise ValueError(f"Nested outer fold {fold} selected score differs from winner")
        saved = pd.read_parquet(NESTED_DIR / f"outer_{fold}" / "outer_predictions.parquet")
        common = identity + ["label", *MODEL_NAMES, "selected", "selection_model"]
        if not saved[common].reset_index(drop=True).equals(part[common].reset_index(drop=True)):
            raise ValueError(f"Nested outer fold {fold} scores differ from pooled OOF")
        y = part.label.to_numpy(dtype=np.uint8)
        for row in fold_metrics.itertuples(index=False):
            expected_key = (choice["best_by_model"][choice["overall_winner"]]["candidate_id"]
                            if row.model == "selected" else
                            choice["best_by_model"][row.model]["candidate_id"])
            if (row.candidate_id != expected_key or row.n != len(part) or
                    row.events != int(y.sum())):
                raise ValueError(f"Nested outer fold {fold} metric identity differs")
            calculated = metrics(y, part[row.model].to_numpy())
            if any(not np.isclose(getattr(row, field), value, rtol=1e-8, atol=1e-8)
                   for field, value in calculated.items()):
                raise ValueError(f"Nested outer fold {fold} metric differs: {row.model}")
    pooled = pd.read_csv(NESTED_DIR / "pooled_metrics.csv")
    if len(pooled) != len(MODEL_NAMES) + 1 or set(pooled.model) != set(MODEL_NAMES + ["selected"]):
        raise ValueError("Nested pooled metric rows are incomplete")
    y = scored.label.to_numpy(dtype=np.uint8)
    for row in pooled.itertuples(index=False):
        calculated = metrics(y, scored[row.model].to_numpy())
        if row.n != len(scored) or row.events != int(y.sum()) or any(
            not np.isclose(getattr(row, field), value, rtol=1e-8, atol=1e-8)
            for field, value in calculated.items()
        ):
            raise ValueError(f"Nested pooled metric differs: {row.model}")
    return inner, outer, choices, scored
