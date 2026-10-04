"""Full-cohort tuning for the final CDC model, separate from nested evaluation.

Outer-fold scores are never read here. Every candidate is chosen with five
development folds over 2023+2024, then refitted once on the complete cohort.
The five-fold search scores are selection diagnostics, not unbiased test scores.
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

from src.data.preeclampsia_cdc import OUT
from src.progress import tqdm
from src.training.preeclampsia_cdc import (
    MODEL_DIR, MODEL_NAMES, _fit_model, _predict_model, metrics, run_fingerprint,
)
from src.training.preeclampsia_cdc_nested import _atomic_csv, _atomic_json, _cached_fit, _save_fit
from src.training.preeclampsia_cdc_tuning import (
    GRID, SELECTION_METRIC, _checked_model_checkpoint, _file_sha256, _input_data,
    _prediction_table, _save_model_checkpoint, candidate_id, runtime_versions,
)

FINAL_DIR = OUT / "final_tuning_2023_2024"


def final_fingerprint() -> str:
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in (Path(__file__).name, "preeclampsia_cdc_nested.py",
                                "preeclampsia_cdc_tuning.py")}
    spec = {"source": run_fingerprint(), "grid": GRID,
            "selection_metric": SELECTION_METRIC, "folds": list(range(5)),
            "runtime_versions": runtime_versions(),
            "code_hashes": code_hashes}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _choose(fold_results: pd.DataFrame, fingerprint: str) -> tuple[pd.DataFrame, dict]:
    expected = {(name, candidate_id(name, params), fold)
                for name, candidates in GRID.items() for params in candidates
                for fold in range(5)}
    observed = list(zip(fold_results.model, fold_results.candidate_id, fold_results.fold))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Full-cohort grid has missing or duplicated candidate folds")
    if not fold_results.groupby("fold").n.nunique().eq(1).all() or not (
            fold_results.groupby("fold").events.nunique().eq(1).all()):
        raise ValueError("Full-cohort candidates saw different validation records")
    rows = []
    for (name, key), part in fold_results.groupby(["model", "candidate_id"], sort=False):
        params = next(p for p in GRID[name] if candidate_id(name, p) == key)
        rows.append({"model": name, "candidate_id": key,
            "params_json": json.dumps(params, sort_keys=True),
            "n_oof": int(part.n.sum()), "events_oof": int(part.events.sum()),
            "cv_log_loss": float(np.average(part.log_loss, weights=part.n)),
            "cv_brier": float(np.average(part.brier, weights=part.n)),
            "auroc_fold_mean": float(part.auroc.mean()),
            "auprc_fold_mean": float(part.auprc.mean()),
            "log_loss_fold_sd": float(part.log_loss.std(ddof=1)),
            "fit_seconds_total": float(part.fit_seconds.sum())})
    summary = pd.DataFrame(rows).sort_values(
        ["model", "cv_log_loss", "candidate_id"]).reset_index(drop=True)
    if summary.n_oof.nunique() != 1 or summary.events_oof.nunique() != 1:
        raise ValueError("Full-cohort candidates do not cover the same labels")
    selected = {}
    for name in MODEL_NAMES:
        row = summary.loc[summary.model.eq(name)].iloc[0]
        selected[name] = {"candidate_id": row.candidate_id,
            "params": json.loads(row.params_json), "cv_log_loss": float(row.cv_log_loss)}
    winner = min(MODEL_NAMES, key=lambda name: (selected[name]["cv_log_loss"], name))
    choice = {"fingerprint": fingerprint, "development_years": [2023, 2024],
        "development_n": int(summary.n_oof.iloc[0]),
        "development_events": int(summary.events_oof.iloc[0]),
        "selection_metric": SELECTION_METRIC, "overall_winner": winner,
        "best_by_model": selected,
        "interpretation": "Five-fold all-data tuning is used only to choose the final fit; its scores are optimistic after selection."}
    return summary, choice


def run_full_cohort_search(resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit the fixed grid on five pooled folds for final-model selection."""
    data = _input_data()
    if set(data.source_year.unique()) != {2023, 2024} or set(data.fold.unique()) != set(range(5)):
        raise ValueError("Final tuning needs both years and all five prepared folds")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = final_fingerprint()
    manifest_path = FINAL_DIR / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Existing final search belongs to another data, code, or grid")
    else:
        _atomic_json({"fingerprint": fingerprint, "grid": GRID,
            "n_fits": 5 * sum(map(len, GRID.values())),
            "selection_metric": SELECTION_METRIC}, manifest_path)
    rows = []
    with tqdm(total=5 * sum(map(len, GRID.values())),
              desc="2023+2024 final model grid", unit="fit") as bar:
        for fold in range(5):
            train = data.loc[data.fold.ne(fold)]
            valid = data.loc[data.fold.eq(fold)]
            if train.empty or valid.empty or set(train.fold.unique()) != set(range(5)) - {fold}:
                raise ValueError(f"Incomplete full-cohort training split {fold}")
            for name, candidates in GRID.items():
                for params in candidates:
                    key = candidate_id(name, params)
                    path = FINAL_DIR / f"fold_{fold}" / f"{name}_{key}.parquet"
                    sidecar = path.with_suffix(".json")
                    identity = {"fingerprint": fingerprint, "fold": fold,
                        "model": name, "candidate_id": key, "params": params,
                        "train_n": len(train), "train_events": int(train.target_hdp_proxy.sum()),
                        "n": len(valid), "events": int(valid.target_hdp_proxy.sum())}
                    cached = _cached_fit(path, sidecar, valid, identity) if resume else None
                    if cached is None:
                        started = perf_counter()
                        if name == "prevalence":
                            prediction = np.full(len(valid), train.target_hdp_proxy.mean(),
                                                 dtype=np.float32)
                        else:
                            model = _fit_model(name, train,
                                f"Final grid fold {fold}: {name} {key[:6]}", params)
                            prediction = _predict_model(name, model, valid)
                            del model
                        scored = _prediction_table(valid, prediction)
                        row = {**identity, "params_json": json.dumps(params, sort_keys=True),
                            "fit_seconds": perf_counter() - started,
                            "rss_gb": psutil.Process().memory_info().rss / 1e9,
                            **metrics(scored.label.to_numpy(), scored.prediction.to_numpy())}
                        saved = _save_fit(scored, path, sidecar, row)
                        del scored, prediction
                    else:
                        _, saved = cached
                    rows.append({field: value for field, value in saved.items()
                                 if field != "score_sha256"})
                    bar.set_postfix_str(f"fold={fold} {name} {key[:6]}", refresh=False)
                    bar.update()
            del train, valid
            gc.collect()
    results = pd.DataFrame(rows)
    summary, best = _choose(results, fingerprint)
    if best["development_n"] != len(data) or best["development_events"] != int(data.target_hdp_proxy.sum()):
        raise ValueError("Final tuning does not cover every cohort record")
    results_path = FINAL_DIR / "candidate_fold_metrics.csv"
    _atomic_csv(results, results_path)
    summary_path = FINAL_DIR / "candidate_summary.csv"
    _atomic_csv(summary, summary_path)
    best["candidate_summary_sha256"] = _file_sha256(summary_path)
    best["candidate_fold_metrics_sha256"] = _file_sha256(results_path)
    _atomic_json(best, FINAL_DIR / "best_configurations.json")
    return results, summary, best


def load_full_cohort_best() -> dict:
    best = json.loads((FINAL_DIR / "best_configurations.json").read_text())
    if best.get("fingerprint") != final_fingerprint():
        raise ValueError("Final selection belongs to another data, code, or grid")
    summary_path = FINAL_DIR / "candidate_summary.csv"
    if not summary_path.exists() or best.get("candidate_summary_sha256") != _file_sha256(summary_path):
        raise ValueError("Final selection disagrees with candidate summary")
    summary = pd.read_csv(summary_path, dtype={"candidate_id": str})
    expected = {(name, candidate_id(name, p)) for name, candidates in GRID.items()
                for p in candidates}
    if len(summary) != len(expected) or set(zip(summary.model, summary.candidate_id)) != expected:
        raise ValueError("Final candidate summary is incomplete")
    results_path = FINAL_DIR / "candidate_fold_metrics.csv"
    if (not results_path.exists() or best.get("candidate_fold_metrics_sha256") !=
            _file_sha256(results_path)):
        raise ValueError("Final selection disagrees with saved fold metrics")
    reconstructed_summary, reconstructed = _choose(pd.read_csv(
        results_path, dtype={"candidate_id": str}, float_precision="round_trip"),
        best["fingerprint"])
    ordered = summary.sort_values(["model", "candidate_id"]).reset_index(drop=True)
    checked = reconstructed_summary.sort_values(["model", "candidate_id"]).reset_index(drop=True)
    if not ordered[["model", "candidate_id", "params_json"]].equals(
            checked[["model", "candidate_id", "params_json"]]) or any(
            not np.allclose(ordered[column], checked[column], rtol=1e-10, atol=1e-10)
            for column in ("n_oof", "events_oof", "cv_log_loss", "cv_brier")):
        raise ValueError("Final candidate summary disagrees with fold metrics")
    if (best["overall_winner"] != reconstructed["overall_winner"] or
            any(best["best_by_model"][name]["candidate_id"] !=
                reconstructed["best_by_model"][name]["candidate_id"]
                or best["best_by_model"][name]["params"] !=
                reconstructed["best_by_model"][name]["params"]
                or not np.isclose(best["best_by_model"][name]["cv_log_loss"],
                                  reconstructed["best_by_model"][name]["cv_log_loss"],
                                  rtol=1e-10, atol=1e-10)
                for name in MODEL_NAMES) or
            best["development_n"] != reconstructed["development_n"] or
            best["development_events"] != reconstructed["development_events"]):
        raise ValueError("Final selection does not minimize the saved fold scores")
    return best


def refit_full_cohort_best_models(resume: bool = True) -> dict:
    """Refit chosen configurations on all rows, without scoring training rows."""
    best = load_full_cohort_best()
    data = _input_data()
    total, events = len(data), int(data.target_hdp_proxy.sum())
    if (total, events) != (best["development_n"], best["development_events"]):
        raise ValueError("Final refit cohort differs from the selection cohort")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MODEL_DIR / "global_full_cv_2023_2024.json"
    if resume and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if (saved.get("fingerprint") != best["fingerprint"] or
                saved.get("best_by_model") != best["best_by_model"] or
                saved.get("overall_winner") != best["overall_winner"]):
            raise ValueError("Global models disagree with final full-cohort selection")
        if set(saved.get("artifacts", {})) == set(MODEL_NAMES):
            for name, file_name in saved["artifacts"].items():
                path = Path(file_name)
                if not path.exists() or _file_sha256(path) != saved.get("sha256", {}).get(name):
                    raise ValueError(f"Global final model missing or changed: {name}")
            return saved
    artifacts = {}
    prevalence_path = MODEL_DIR / "global_full_cv_2023_2024_prevalence.json"
    _atomic_json({"n": total, "events": events, "prevalence": events / total},
                 prevalence_path)
    artifacts["prevalence"] = str(prevalence_path)
    for name in tqdm(MODEL_NAMES[1:], desc="Final full-cohort refit", unit="model"):
        chosen = best["best_by_model"][name]
        path = MODEL_DIR / f"global_full_cv_2023_2024_{name}_{chosen['candidate_id']}.joblib"
        identity = {"fingerprint": best["fingerprint"], "stage": "global_full_cv_2023_2024",
            "model": name, "candidate_id": chosen["candidate_id"],
            "params": chosen["params"], "training_n": total, "training_events": events}
        if not (resume and _checked_model_checkpoint(path, identity)):
            model = _fit_model(name, data, f"Global full-cohort: {name}", chosen["params"])
            _save_model_checkpoint(model, path, identity)
            del model
        artifacts[name] = str(path)
        gc.collect()
    manifest = {"fingerprint": best["fingerprint"], "source_years": [2023, 2024],
        "n": total, "events": events, "overall_winner": best["overall_winner"],
        "best_by_model": best["best_by_model"], "artifacts": artifacts,
        "sha256": {name: _file_sha256(Path(path)) for name, path in artifacts.items()},
        "warning": "All available labels were used for selection and fitting; no internal performance claim is made for these saved models."}
    _atomic_json(manifest, manifest_path)
    return manifest
