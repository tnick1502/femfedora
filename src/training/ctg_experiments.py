"""Grouped nested model selection, independent calibration, and locked CTG test."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.features.ctg_preprocessing import CTG_FEATURES, CTGInputGuard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/experiments/ctg_uci"
MODELS = ROOT / "models/exported/ctg_uci"
CACHE = ROOT / "tmp/ctg_experiments"
VERSION = "ctg_nested_v1_20261017"
SEED = 20261017
FAMILIES = ["A_multiclass", "B_sqrt", "B_balanced", "C_ovr", "D_hierarchy", "logistic"]
CAT_GRID = list(ParameterGrid({"depth": [3, 5, 7], "learning_rate": [.03, .10],
                               "l2_leaf_reg": [3, 10], "iterations": [300, 600]}))
LOG_GRID = list(ParameterGrid({"C": [.1, 1., 10.], "class_weight": [None, "balanced"]}))
MODES = ["raw", "calibrated", "threshold"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    primary = ROOT / "data/processed/ctg_uci/ctg_primary.parquet"
    splits = ROOT / "data/processed/ctg_uci/ctg_splits.parquet"
    data = pd.read_parquet(primary).merge(
        pd.read_parquet(splits)[["source_excel_row", "split"]], on="source_excel_row", validate="one_to_one")
    CTGInputGuard().fit_transform(data[list(CTG_FEATURES)])
    assert data.groupby("source_file")["split"].nunique().max() == 1
    assert data.groupby("feature_duplicate_group")["split"].nunique().max() == 1
    assert len(data) == 2126
    fingerprint = hashlib.sha256((digest(primary) + digest(splits) + digest(__file__)
        + digest(ROOT / 'data/processed/ctg_uci/preprocessor.joblib') + VERSION + json.dumps(CAT_GRID)).encode()).hexdigest()
    for directory in [OUT, MODELS, CACHE / fingerprint[:16]]:
        directory.mkdir(parents=True, exist_ok=True)
    return data, fingerprint


def grouped_folds(frame, n_splits, seed):
    y = frame.NSP.to_numpy(dtype=int) - 1
    result = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        frame, y, frame.source_file))
    for train, valid in result:
        assert set(frame.iloc[train].source_file).isdisjoint(frame.iloc[valid].source_file)
        assert set(frame.iloc[train].feature_duplicate_group).isdisjoint(frame.iloc[valid].feature_duplicate_group)
        assert len(np.unique(y[train])) == len(np.unique(y[valid])) == 3
    return result


def normalize(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return p / p.sum(axis=1, keepdims=True)


def heads_to_prob(heads, family):
    if family == "D_hierarchy":
        a, b = np.asarray(heads).T
        return normalize(np.column_stack([1 - a, a * (1 - b), a * b]))
    return normalize(heads)


class CTGFamily:
    def __init__(self, family, params, seed=SEED):
        self.family, self.params, self.seed = family, params, seed

    @staticmethod
    def features(frame):
        x = CTGInputGuard().transform(frame).drop(columns=["DS", "Width"])
        x["Tendency"] = x.Tendency.astype(int)
        return x

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if self.family == "logistic":
            template = joblib.load(ROOT / "data/processed/ctg_uci/preprocessor.joblib")
            self.models_ = [Pipeline([("features", clone(template)), ("model", LogisticRegression(
                C=self.params["C"], class_weight=self.params["class_weight"], max_iter=3000))])]
            self.models_[0].fit(X, y)
            return self
        x = self.features(X)
        base = dict(self.params, random_seed=self.seed, thread_count=2, verbose=False,
                    allow_writing_files=False, bootstrap_type="Bayesian", bagging_temperature=1.,
                    random_strength=1., use_best_model=False)
        weights = {"B_sqrt": "SqrtBalanced", "B_balanced": "Balanced"}.get(self.family)
        if weights:
            base["auto_class_weights"] = weights
        self.models_ = []
        if self.family in ["A_multiclass", "B_sqrt", "B_balanced"]:
            model = CatBoostClassifier(**base, loss_function="MultiClass", classes_count=3)
            model.fit(x, y, cat_features=["Tendency"])
            self.models_.append(model)
        elif self.family == "C_ovr":
            for label in range(3):
                model = CatBoostClassifier(**base, loss_function="Logloss")
                model.fit(x, (y == label).astype(int), cat_features=["Tendency"])
                self.models_.append(model)
        elif self.family == "D_hierarchy":
            for mask, binary in [(np.ones(len(y), dtype=bool), y != 0), (y != 0, y == 2)]:
                model = CatBoostClassifier(**base, loss_function="Logloss")
                model.fit(x.loc[mask], binary[mask].astype(int), cat_features=["Tendency"])
                self.models_.append(model)
        else:
            raise ValueError(self.family)
        return self

    def predict_heads(self, X):
        if self.family == "logistic":
            return self.models_[0].predict_proba(X)
        x = self.features(X)
        if self.family in ["A_multiclass", "B_sqrt", "B_balanced"]:
            return self.models_[0].predict_proba(x)
        return np.column_stack([model.predict_proba(x)[:, 1] for model in self.models_])


class HeadCalibrator:
    """Separate regularized sigmoid fits; hierarchical second fit is conditional."""
    def __init__(self, family):
        self.family = family

    def fit(self, heads, y):
        heads, y = np.asarray(heads), np.asarray(y)
        self.models_ = []
        for j in range(heads.shape[1]):
            mask = (y != 0) if self.family == "D_hierarchy" and j == 1 else np.ones(len(y), dtype=bool)
            binary = (y == 2) if self.family == "D_hierarchy" and j == 1 else (
                (y != 0) if self.family == "D_hierarchy" else (y == j))
            model = LogisticRegression(C=1., max_iter=1000)
            model.fit(logit(np.clip(heads[mask, j], 1e-6, 1 - 1e-6)).reshape(-1, 1), binary[mask])
            self.models_.append(model)
        return self

    def predict_proba(self, heads):
        h = np.asarray(heads)
        calibrated = np.column_stack([m.predict_proba(logit(np.clip(h[:, j], 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
                                      for j, m in enumerate(self.models_)])
        return heads_to_prob(calibrated, self.family)


def decisions(p, threshold=None):
    p = np.asarray(p)
    if threshold is None:
        return p.argmax(axis=1)
    pred = p[:, :2].argmax(axis=1)
    pred[p[:, 2] >= threshold] = 2
    return pred


def tune_threshold(y, p):
    choices = [None, *np.round(np.arange(.10, .901, .05), 2).tolist()]
    scores = [f1_score(y, decisions(p, t), average="macro", labels=[0, 1, 2], zero_division=0) for t in choices]
    return choices[int(np.argmax(scores))]


def sensitivity_threshold(y, p, target=.90):
    feasible = []
    for threshold in np.linspace(.01, .99, 99):
        pred = decisions(p, threshold)
        recall = ((pred == 2) & (y == 2)).sum() / max(1, (y == 2).sum())
        specificity = ((pred != 2) & (y != 2)).sum() / max(1, (y != 2).sum())
        if recall >= target:
            feasible.append((specificity, threshold))
    return float(max(feasible)[1]) if feasible else .01


def metrics(y, p, pred=None):
    y, p = np.asarray(y), normalize(p)
    pred = p.argmax(axis=1) if pred is None else np.asarray(pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), recall_path=float(recall[2]),
                precision_path=float(precision[2]), f1_path=float(f1[2]),
                recall_normal=float(recall[0]), recall_suspect=float(recall[1]),
                ap_path=float(average_precision_score(y == 2, p[:, 2])) if (y == 2).any() else np.nan,
                roc_auc_path=float(roc_auc_score(y == 2, p[:, 2])) if len(np.unique(y == 2)) == 2 else np.nan,
                log_loss=float(log_loss(y, p, labels=[0, 1, 2])),
                brier=float(np.mean(np.sum((np.eye(3)[y] - p) ** 2, axis=1))),
                path_as_normal=float(((y == 2) & (pred == 0)).sum() / max(1, (y == 2).sum())),
                specificity_path=float(((y != 2) & (pred != 2)).sum() / max(1, (y != 2).sum())))


def search_grid(frame, family, folds, seed):
    grid = LOG_GRID if family == "logistic" else CAT_GRID
    y, X = frame.NSP.to_numpy(dtype=int) - 1, frame[list(CTG_FEATURES)]

    def evaluate(item):
        idx, params = item
        heads = np.zeros((len(frame), 2 if family == "D_hierarchy" else 3))
        f1s, losses = [], []
        for k, (train, valid) in enumerate(folds):
            model = CTGFamily(family, params, seed + k).fit(X.iloc[train], y[train])
            heads[valid] = model.predict_heads(X.iloc[valid])
            values = metrics(y[valid], heads_to_prob(heads[valid], family))
            f1s.append(values["macro_f1"])
            losses.append(values["log_loss"])
        return dict(candidate=idx, params=params, mean_macro_f1=float(np.mean(f1s)),
                    sd_macro_f1=float(np.std(f1s, ddof=1)), mean_log_loss=float(np.mean(losses)),
                    fold_f1=f1s), heads

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, enumerate(grid)))
    best = min(range(len(results)), key=lambda i: (-results[i][0]["mean_macro_f1"],
                                                  results[i][0]["mean_log_loss"], i))
    return results[best][0]["params"], results[best][1], [item[0] for item in results]


def _prediction_frame(frame, p, pred, **extra):
    result = frame[["source_excel_row", "source_file", "NSP"]].copy()
    for j in range(3):
        result[f"p{j}"] = p[:, j]
    result["pred"] = pred + 1
    for key, value in extra.items():
        result[key] = value
    return result


def run_nested():
    data, fingerprint = load_data()
    train = data[data.split == "train"].reset_index(drop=True)
    outer = grouped_folds(train, 5, SEED)
    layout, inner_layout = [], []
    for k, (fit_idx, hold_idx) in enumerate(outer):
        layout.extend([dict(source_excel_row=int(train.iloc[j].source_excel_row), outer_fold=k) for j in hold_idx])
        fit = train.iloc[fit_idx].reset_index(drop=True)
        for j, (_, inner_hold) in enumerate(grouped_folds(fit, 3, SEED + 100 + k)):
            inner_layout.extend([dict(outer_fold=k, inner_fold=j, source_excel_row=int(fit.iloc[z].source_excel_row))
                                 for z in inner_hold])
    pd.DataFrame(layout).to_csv(OUT / "outer_folds.csv", index=False)
    pd.DataFrame(inner_layout).to_csv(OUT / "inner_folds.csv", index=False)
    rows, predictions, grids, parameters = [], [], [], []
    for k, (fit_idx, hold_idx) in enumerate(outer):
        fit, hold = train.iloc[fit_idx].reset_index(drop=True), train.iloc[hold_idx]
        inner = grouped_folds(fit, 3, SEED + 100 + k)
        y_fit, y_hold = fit.NSP.to_numpy(dtype=int) - 1, hold.NSP.to_numpy(dtype=int) - 1
        for family in FAMILIES:
            cache = CACHE / fingerprint[:16] / f"outer_{k}_{family}.joblib"
            start = time.monotonic()
            if cache.exists():
                payload = joblib.load(cache)
            else:
                params, oof_heads, grid = search_grid(fit, family, inner, SEED + 1000 * k)
                calibrator = HeadCalibrator(family).fit(oof_heads, y_fit)
                calibrated_fit = calibrator.predict_proba(oof_heads)
                threshold = tune_threshold(y_fit, calibrated_fit)
                model = CTGFamily(family, params, SEED + k).fit(fit[list(CTG_FEATURES)], y_fit)
                heads = model.predict_heads(hold[list(CTG_FEATURES)])
                raw, calibrated = heads_to_prob(heads, family), calibrator.predict_proba(heads)
                payload = dict(params=params, grid=grid, threshold=threshold, raw=raw, calibrated=calibrated)
                joblib.dump(payload, cache)
            grids.extend([dict(family=family, outer_fold=k, **r) for r in payload["grid"]])
            parameters.append(dict(family=family, outer_fold=k, params=payload["params"], threshold=payload["threshold"]))
            for mode in MODES:
                p = payload["raw"] if mode == "raw" else payload["calibrated"]
                pred = decisions(p, payload["threshold"] if mode == "threshold" else None)
                rows.append(dict(family=family, mode=mode, outer_fold=k, **metrics(y_hold, p, pred)))
                predictions.append(_prediction_frame(hold, p, pred, family=family, mode=mode, outer_fold=k))
            print(f"outer {k + 1}/5 · {family}: {time.monotonic() - start:.1f}s · grid complete", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    pooled = []
    for (family, mode), frame in prediction_frame.groupby(["family", "mode"]):
        pooled.append(dict(family=family, mode=mode, **metrics(frame.NSP.to_numpy() - 1,
                        frame[["p0", "p1", "p2"]].to_numpy(), frame.pred.to_numpy() - 1)))
    summary = pd.DataFrame(pooled)
    eligible = summary[~((summary.family == "C_ovr") & (summary["mode"] == "raw"))]
    winner = eligible.sort_values(["macro_f1", "log_loss"], ascending=[False, True]).iloc[0]
    pd.DataFrame(rows).to_csv(OUT / "nested_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "nested_pooled_metrics.csv", index=False)
    prediction_frame.to_parquet(OUT / "nested_predictions.parquet", index=False)
    pd.DataFrame(grids).to_json(OUT / "nested_grid.json", orient="records", indent=2)
    write_json(OUT / "nested_parameters.json", parameters)
    write_json(OUT / "nested_selection.json", dict(fingerprint=fingerprint, family=winner.family,
               diagnostic_mode=winner["mode"], pooled_macro_f1=float(winner.macro_f1),
               search_version=VERSION, cat_grid=CAT_GRID, selection="outer_train_CV_only"))
    return summary, pd.DataFrame(rows)


def validation_policies(frame, heads, family):
    y = frame.NSP.to_numpy(dtype=int) - 1
    raw = heads_to_prob(heads, family)
    calibrated = np.zeros((len(frame), 3))
    preds = {"raw": decisions(raw), "calibrated": np.zeros(len(frame), dtype=int),
             "threshold": np.zeros(len(frame), dtype=int)}
    for fit, hold in grouped_folds(frame, 3, SEED + 500):
        calibrator = HeadCalibrator(family).fit(heads[fit], y[fit])
        p_fit = calibrator.predict_proba(heads[fit])
        threshold = tune_threshold(y[fit], p_fit)
        calibrated[hold] = calibrator.predict_proba(heads[hold])
        preds["calibrated"][hold] = decisions(calibrated[hold])
        preds["threshold"][hold] = decisions(calibrated[hold], threshold)
    rows = [dict(mode=mode, **metrics(y, raw if mode == "raw" else calibrated, pred))
            for mode, pred in preds.items()]
    table = pd.DataFrame(rows)
    eligible = table[table["mode"] != "raw"] if family == "C_ovr" else table
    chosen = eligible.sort_values(["macro_f1", "log_loss"], ascending=[False, True]).iloc[0]["mode"]
    return chosen, table, raw, calibrated, preds


class CTGBundle:
    def __init__(self, models, family, mode, calibrator, threshold):
        self.models, self.family, self.mode = models, family, mode
        self.calibrator, self.threshold = calibrator, threshold
        self.classes_ = np.array([1, 2, 3])

    def predict_proba(self, X):
        CTGInputGuard().transform(X)
        heads = np.mean([model.predict_heads(X) for model in self.models], axis=0)
        return heads_to_prob(heads, self.family) if self.mode == "raw" else self.calibrator.predict_proba(heads)

    def predict(self, X):
        return decisions(self.predict_proba(X), self.threshold if self.mode == "threshold" else None) + 1


def grouped_bootstrap(frame, repetitions=1000, seed=SEED):
    rng, groups = np.random.default_rng(seed), frame.source_file.unique()
    indices = {g: np.flatnonzero(frame.source_file.to_numpy() == g) for g in groups}
    values = []
    for _ in range(repetitions):
        ix = np.concatenate([indices[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        f = frame.iloc[ix]
        values.append(metrics(f.NSP.to_numpy() - 1, f[["p0", "p1", "p2"]].to_numpy(), f.pred.to_numpy() - 1))
    boot = pd.DataFrame(values)
    return pd.DataFrame({"lower": boot.quantile(.025), "upper": boot.quantile(.975)})


def run_final():
    data, fingerprint = load_data()
    selection = json.loads((OUT / "nested_selection.json").read_text())
    assert selection["fingerprint"] == fingerprint
    train = data[data.split == "train"].reset_index(drop=True)
    valid = data[data.split == "validation"].reset_index(drop=True)
    test = data[data.split == "test"].reset_index(drop=True)
    champion_family = selection["family"]
    y_train, y_valid = train.NSP.to_numpy(dtype=int) - 1, valid.NSP.to_numpy(dtype=int) - 1
    validation_layout = []
    for k, (_, hold) in enumerate(grouped_folds(valid, 3, SEED + 500)):
        validation_layout.extend([dict(source_excel_row=int(valid.iloc[j].source_excel_row), calibration_fold=k) for j in hold])
    pd.DataFrame(validation_layout).to_csv(OUT / 'validation_calibration_folds.csv', index=False)
    fits, policy_rows, validation_predictions, grid_rows = {}, [], [], []
    for family in FAMILIES:
        cache = CACHE / fingerprint[:16] / f"final_{family}.joblib"
        start = time.monotonic()
        if cache.exists():
            payload = joblib.load(cache)
        else:
            params, _, grid = search_grid(train, family, grouped_folds(train, 3, SEED + 800), SEED + 8000)
            model = CTGFamily(family, params, SEED + 9000).fit(train[list(CTG_FEATURES)], y_train)
            payload = dict(params=params, grid=grid, models=[model])
            joblib.dump(payload, cache)
        grid_rows.extend([dict(family=family, **r) for r in payload["grid"]])
        model_sets = {"single": payload["models"]}
        if family == champion_family:
            ensemble_cache = CACHE / fingerprint[:16] / f"ensemble_{family}.joblib"
            if ensemble_cache.exists():
                ensemble = joblib.load(ensemble_cache)
            else:
                ensemble = [CTGFamily(family, payload["params"], SEED + 10000 + k).fit(
                    train.iloc[fit][list(CTG_FEATURES)], y_train[fit])
                    for k, (fit, _) in enumerate(grouped_folds(train, 5, SEED))]
                joblib.dump(ensemble, ensemble_cache)
            model_sets["fold_ensemble"] = ensemble
        for architecture, models in model_sets.items():
            heads = np.mean([model.predict_heads(valid[list(CTG_FEATURES)]) for model in models], axis=0)
            mode, table, raw, calibrated, preds = validation_policies(valid, heads, family)
            for r in table.to_dict("records"):
                policy_rows.append(dict(family=family, architecture=architecture, **r))
            for md in MODES:
                validation_predictions.append(_prediction_frame(valid, raw if md == "raw" else calibrated,
                    preds[md], family=family, architecture=architecture, mode=md))
            calibrator = HeadCalibrator(family).fit(heads, y_valid)
            p_cal = calibrator.predict_proba(heads)
            threshold = tune_threshold(y_valid, p_cal) if mode == "threshold" else None
            fits[(family, architecture)] = dict(models=models, mode=mode, calibrator=calibrator,
                threshold=threshold, params=payload["params"], validation_score=float(table.loc[table["mode"] == mode, "macro_f1"].iloc[0]),
                sensitivity_threshold=sensitivity_threshold(y_valid, p_cal))
        print(f"final grid · {family}: {time.monotonic() - start:.1f}s", flush=True)
    single = fits[(champion_family, "single")]
    ensemble = fits[(champion_family, "fold_ensemble")]
    architecture = "fold_ensemble" if ensemble["validation_score"] >= single["validation_score"] + .01 else "single"
    locked = dict(fingerprint=fingerprint, family=champion_family, architecture=architecture,
                  ensemble_minimum_gain=.01, selection_source="train_nested_CV_then_validation_crossfit",
                  policies={f"{fam}/{arch}": {k: v for k, v in value.items() if k not in ["models", "calibrator"]}
                            for (fam, arch), value in fits.items()}, test_used_for_selection=False)
    # Test probability generation starts only AFTER this file is persisted.
    write_json(OUT / "locked_decision.json", locked)
    pd.DataFrame(policy_rows).to_csv(OUT / "validation_crossfit_metrics.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_parquet(OUT / "validation_predictions.parquet", index=False)
    pd.DataFrame(grid_rows).to_json(OUT / "final_grid.json", orient="records", indent=2)
    result_rows, test_predictions = [], []
    champion_bundle = None
    for (family, arch), fitted in fits.items():
        bundle = CTGBundle(fitted["models"], family, fitted["mode"], fitted["calibrator"], fitted["threshold"])
        prob = bundle.predict_proba(test[list(CTG_FEATURES)])
        pred = bundle.predict(test[list(CTG_FEATURES)]) - 1
        result_rows.append(dict(family=family, architecture=arch, mode=fitted["mode"],
                               champion=(family == champion_family and arch == architecture),
                               **metrics(test.NSP.to_numpy() - 1, prob, pred)))
        test_predictions.append(_prediction_frame(test, prob, pred, family=family, architecture=arch))
        joblib.dump(bundle, MODELS / f"{family}_{arch}.joblib")
        if family == champion_family and arch == architecture:
            champion_bundle = bundle
            champion_frame = test_predictions[-1]
    prior = train.NSP.value_counts(normalize=True).reindex([1, 2, 3]).to_numpy()
    baseline = np.tile(prior, (len(test), 1))
    result_rows.append(dict(family="always_normal", architecture="constant", mode="raw", champion=False,
                           **metrics(test.NSP.to_numpy() - 1, baseline)))
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "test_metrics.csv", index=False)
    pd.concat(test_predictions, ignore_index=True).to_parquet(OUT / "test_predictions.parquet", index=False)
    ci = grouped_bootstrap(champion_frame)
    ci.to_csv(OUT / "champion_test_intervals.csv")
    joblib.dump(champion_bundle, MODELS / "champion.joblib")
    loaded = joblib.load(MODELS / "champion.joblib")
    assert np.allclose(loaded.predict_proba(test[list(CTG_FEATURES)]), champion_bundle.predict_proba(test[list(CTG_FEATURES)]))
    assert np.array_equal(loaded.predict(test[list(CTG_FEATURES)]), champion_bundle.predict(test[list(CTG_FEATURES)]))
    for k, model in enumerate(champion_bundle.models):
        if model.family != "logistic":
            for h, head in enumerate(model.models_):
                head.save_model(str(MODELS / f"champion_fold{k}_head{h}.cbm"))
    import catboost, sklearn
    write_json(MODELS / "model_card.json", dict(locked, features=list(CTG_FEATURES),
        n_train=len(train), n_calibration=len(valid), n_test=len(test), target_codes={1:"normal",2:"suspect",3:"pathologic"},
        model_sha256=digest(MODELS / "champion.joblib"), versions={"catboost":catboost.__version__,"sklearn":sklearn.__version__},
        test_metrics=results[results.champion].to_dict("records"),
        limitations=["Expert CTG labels, not pregnancy outcomes", "No patient identifier", "35 pathological test segments; grouped uncertainty",
                     "Test distributions examined during earlier data preparation; not a prospectively blind cohort"] ))
    return results, pd.DataFrame(policy_rows), ci
