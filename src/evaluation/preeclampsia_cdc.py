"""Audited pooled and temporal evaluation of the recorded CDC HDP proxy."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.data.preeclampsia_cdc import OUT, OUTCOMES
from src.progress import tqdm
from src.training.preeclampsia_cdc import (MODEL_NAMES, TEMPORAL_SENSITIVITIES,
    SECONDARY_TARGETS, _verify_and_load, metrics, run_fingerprint,
    validate_prediction_artifact)


def _validate_saved_outcomes(frame: pd.DataFrame, source: pd.DataFrame,
                             year: int, *, gestation: bool = False) -> None:
    """Match every saved evaluation label to its audited source record."""
    identity = ["source_year", "source_row", "fold"]
    expected = source.loc[source.source_year.eq(year)].sort_values(identity).reset_index(drop=True)
    scored = frame.sort_values(identity).reset_index(drop=True)
    if len(scored) != len(expected) or scored.duplicated(identity[:2]).any():
        raise ValueError(f"Saved {year} scores do not cover the audited source once")
    fields = identity + OUTCOMES + (["oegest_comb"] if gestation else [])
    for field in fields:
        actual = scored["label"] if field == "target_hdp_proxy" and "label" in scored else scored[field]
        reference = expected[field]
        if not np.array_equal(actual.to_numpy(dtype=float), reference.to_numpy(dtype=float),
                              equal_nan=True):
            raise ValueError(f"Saved {year} {field} differs from the audited source")


def _validate_probability_columns(frame: pd.DataFrame, names: list[str], context: str) -> None:
    for name in names:
        probabilities = frame[name].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"{context}: invalid probabilities for {name}")


def calibration_parameters(y: np.ndarray, p: np.ndarray) -> dict:
    """Offset calibration intercept (slope 1) and separate joint calibration slope."""
    y = np.asarray(y, dtype=np.uint8)
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6))
    intercept = brentq(lambda a: float(expit(x + a).mean() - y.mean()), -30, 30)
    if np.ptp(x) < 1e-12:
        return {"calibration_in_the_large": float(intercept),
                "calibration_slope": np.nan, "joint_calibration_intercept": np.nan}
    model = LogisticRegression(C=1e12, max_iter=300, solver="lbfgs").fit(x.reshape(-1, 1), y)
    return {"calibration_in_the_large": float(intercept),
            "calibration_slope": float(model.coef_[0, 0]),
            "joint_calibration_intercept": float(model.intercept_[0])}


def calibration_offset(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration-in-the-large for subgroup audits without fitting a slope."""
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6))
    observed = float(np.mean(y))
    if not 0 < observed < 1:
        raise ValueError("Both outcome classes are required")
    return float(brentq(lambda a: float(expit(x + a).mean() - observed), -30, 30))


def apply_intercept_update(p: np.ndarray, offset: float) -> np.ndarray:
    """Apply an offset fitted on a separate, later labeled calibration set."""
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6))
    return expit(x + offset)


def fixed_threshold_from_development(p: np.ndarray, fraction: float = .10) -> float:
    """Score cutoff from development predictions, without test scores or labels."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    return float(np.quantile(p, 1-fraction))


def operating_point(y: np.ndarray, flagged: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    return {"flagged_fraction": float(np.mean(flagged)), "true_positive": int(tp),
            "false_positive": int(fp), "true_negative": int(tn), "false_negative": int(fn),
            "sensitivity": float(tp/(tp+fn)) if tp+fn else np.nan,
            "specificity": float(tn/(tn+fp)) if tn+fp else np.nan,
            "ppv": float(tp/(tp+fp)) if tp+fp else np.nan}


def top_tenth_operating_point(y: np.ndarray, p: np.ndarray) -> dict:
    """Descriptive held-out rank budget, distinct from a deployable fixed cutoff."""
    k = int(np.ceil(.10 * len(p)))
    flagged = np.zeros(len(p), dtype=bool)
    flagged[np.argpartition(p, -k)[-k:]] = True
    result = operating_point(y, flagged)
    return {**result, "sensitivity_at_10pct": result["sensitivity"],
            "specificity_at_10pct": result["specificity"], "ppv_at_10pct": result["ppv"]}


def observed_guideline_rule(frame: pd.DataFrame, include_ivf: bool = False) -> np.ndarray:
    """Observable subset of USPSTF factors: diabetes, nulliparity, BMI, age, IVF.

    Missing/unmeasured full-guideline factors are never assumed absent. This
    partial rule can therefore be indeterminate; it is not a clinical checklist.
    """
    diabetes = frame.rf_pdiab.to_numpy(dtype=float)
    age = np.where(frame.mager.notna(), frame.mager.ge(35).astype(float), np.nan)
    bmi = np.where(frame.bmi.notna(), frame.bmi.ge(30).astype(float), np.nan)
    parity_known = frame.priorlive.notna() & frame.priordead.notna()
    nulliparity = np.where(parity_known,
        (frame.priorlive.eq(0) & frame.priordead.eq(0)).astype(float), np.nan)
    factors = [age, bmi, nulliparity]
    if include_ivf:
        if "rf_artec" not in frame:
            raise ValueError("IVF field is required")
        factors.append(frame.rf_artec.to_numpy(dtype=float))
    moderate = np.column_stack(factors)
    known_yes = np.nansum(moderate == 1, axis=1)
    possible_yes = known_yes + np.isnan(moderate).sum(axis=1)
    positive = (diabetes == 1) | (known_yes >= 2)
    negative = (diabetes == 0) & (possible_yes < 2)
    return np.where(positive, 1.0, np.where(negative, 0.0, np.nan))


def compare_guideline_at_budget(frame: pd.DataFrame, model: str = "hist_gb",
                                include_ivf: bool = False) -> pd.DataFrame:
    status = observed_guideline_rule(frame, include_ivf=include_ivf)
    known = np.isfinite(status)
    y = frame.loc[known, "label"].to_numpy(dtype=np.uint8)
    rule_flags = status[known].astype(bool)
    budget = float(rule_flags.mean())
    scores = frame.loc[known, model].to_numpy()
    k = int(rule_flags.sum())
    model_flags = np.zeros(len(scores), dtype=bool)
    if k:
        model_flags[np.argpartition(scores, -k)[-k:]] = True
    rows = []
    for name, flags in [("observed_guideline_subset", rule_flags), (model, model_flags)]:
        rows.append({"model": name, "rule_variant": "early_with_ivf" if include_ivf else "core_preconception",
                     "n": len(y), "coverage": float(known.mean()),
                     "alert_budget": budget, "cases_per_1000": 1000 * float(y.mean()),
                     "captured_per_1000": 1000 * float(np.sum(y[flags])) / len(y),
                     **operating_point(y, flags)})
    return pd.DataFrame(rows)


def decision_curve(y: np.ndarray, probabilities: np.ndarray,
                   thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Net benefit for the *recorded proxy*; not an aspirin treatment policy."""
    thresholds = np.asarray(thresholds if thresholds is not None else np.arange(.02, .251, .01))
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Thresholds must be in (0,1)")
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(probabilities)
    rows = []
    for threshold in thresholds:
        flag = p >= threshold
        weight = threshold/(1-threshold)
        rows.append({"threshold": threshold,
                     "model_net_benefit": (np.sum(flag & (y == 1)) - weight*np.sum(flag & (y == 0)))/len(y),
                     "treat_all_net_benefit": float(y.mean() - weight*(1-y.mean())),
                     "treat_none_net_benefit": 0.0,
                     "flagged_fraction": float(flag.mean())})
    return pd.DataFrame(rows)


def validate_and_summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((OUT / "fold_run.json").read_text())
    if manifest["fingerprint"] != run_fingerprint():
        raise ValueError("OOF artifacts have a different model configuration")
    audit = json.loads((OUT / "audit.json").read_text())
    expected = sum(x["eligible_singleton_rows"] for x in audit["year_checks"].values())
    source = _verify_and_load(["source_year", "source_row", "fold", *OUTCOMES])
    diagnostics, parts, metric_parts = [], [], []
    for fold in tqdm(range(5), desc="Evaluate pooled held-out folds", unit="fold"):
        data = pd.read_parquet(OUT / f"oof_fold_{fold}.parquet")
        if not data.fold.eq(fold).all() or data.label.isna().any():
            raise ValueError(f"Invalid OOF file for fold {fold}")
        validate_prediction_artifact(data, source.loc[source.fold.eq(fold)],
                                     MODEL_NAMES, name=f"evaluation fold {fold}")
        saved = pd.read_csv(OUT / f"fold_metrics_{fold}.csv")
        if (len(saved) != len(MODEL_NAMES) or set(saved.model) != set(MODEL_NAMES) or
                not saved.fold.eq(fold).all() or not saved.n.eq(len(data)).all()):
            raise ValueError(f"Fold {fold} metrics do not reconcile")
        y = data.label.to_numpy(dtype=np.uint8)
        if not saved.events.eq(int(y.sum())).all():
            raise ValueError(f"Fold {fold} event counts do not reconcile")
        for row in saved.itertuples():
            calculated = metrics(y, data[row.model].to_numpy())
            if any(not np.isclose(getattr(row, key), value, rtol=1e-8, atol=1e-8)
                   for key, value in calculated.items()):
                raise ValueError(f"Fold {fold} saved metrics disagree with OOF scores")
        for name in MODEL_NAMES[1:]:
            p = data[name].to_numpy()
            diagnostics.append({"fold": fold, "model": name,
                                **calibration_parameters(y, p),
                                **top_tenth_operating_point(y, p)})
        parts.append(data)
        metric_parts.append(saved)
    del source
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != expected or oof.duplicated(["source_year", "source_row"]).any():
        raise ValueError("OOF predictions do not cover all cohort rows exactly once")
    observed = oof.groupby(["source_year", "fold"], observed=True).label.agg(["size", "sum"])
    reference = pd.read_csv(OUT / "folds_all.csv").set_index(["year", "fold"])
    for key, row in observed.iterrows():
        if row["size"] != reference.loc[key, "rows"] or row["sum"] != reference.loc[key, "events"]:
            raise ValueError(f"Fold/year audit mismatch: {key}")
    diag = pd.DataFrame(diagnostics)
    result = pd.concat(metric_parts, ignore_index=True).merge(
        diag, on=["fold", "model"], how="left", validate="one_to_one")
    result.to_csv(OUT / "fold_metrics_with_diagnostics.csv", index=False)
    return result, oof


def year_breakdown(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in tqdm((2023, 2024), desc="Metrics by year", unit="year"):
        part = oof.loc[oof.source_year.eq(year)]
        y = part.label.to_numpy(dtype=np.uint8)
        for name in MODEL_NAMES:
            rows.append({"year": year, "model": name, "n": len(part),
                         "events": int(y.sum()), **metrics(y, part[name].to_numpy())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "pooled_cv_metrics_by_year.csv", index=False)
    return result


def temporal_summary(bootstrap_rounds: int = 0, seed: int = 20261001) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((OUT / "temporal_run.json").read_text())
    if manifest["fingerprint"] != run_fingerprint():
        raise ValueError("Temporal artifacts have a different model configuration")
    dev = pd.read_parquet(OUT / "temporal_dev_oof.parquet")
    test = pd.read_parquet(OUT / "temporal_test_2024.parquet")
    if not dev.source_year.eq(2023).all() or not test.source_year.eq(2024).all():
        raise ValueError("Temporal train/test years mixed")
    if dev.duplicated(["source_year", "source_row"]).any() or test.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Repeated source records")
    source = _verify_and_load(["source_year", "source_row", "fold", *OUTCOMES])
    _validate_saved_outcomes(dev, source, 2023)
    _validate_saved_outcomes(test, source, 2024)
    del source
    audit = json.loads((OUT / "audit.json").read_text())
    for year, frame in ((2023, dev), (2024, test)):
        expected = audit["year_checks"][str(year)]
        if len(frame) != expected["eligible_singleton_rows"] or int(frame.label.sum()) != expected["eligible_singleton_events"]:
            raise ValueError(f"Temporal {year} predictions do not reconcile with audited cohort")
    names = MODEL_NAMES + TEMPORAL_SENSITIVITIES
    _validate_probability_columns(dev, names, "2023 temporal development")
    _validate_probability_columns(test, names, "2024 temporal test")
    y = test.label.to_numpy(dtype=np.uint8)
    rows = []
    for name in tqdm(names, desc="2024 temporal metrics", unit="model"):
        p = test[name].to_numpy()
        threshold = (fixed_threshold_from_development(dev[name].to_numpy())
                     if name != "prevalence" else np.nan)
        fixed_point = (operating_point(y, p >= threshold) if name != "prevalence"
                       else {key: np.nan for key in operating_point(y, np.zeros(len(y), dtype=bool))})
        rows.append({"model": name, "n": len(test), "events": int(y.sum()),
                     "development_threshold_10pct": threshold,
                     **metrics(y, p), **calibration_parameters(y, p),
                     **{f"fixed_{k}": v for k, v in fixed_point.items()}})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "temporal_diagnostics_2024.csv", index=False)
    return summary, temporal_bootstrap(bootstrap_rounds, seed, test=test)


def temporal_bootstrap(bootstrap_rounds: int = 200, seed: int = 20261001,
                       *, test: pd.DataFrame | None = None) -> pd.DataFrame:
    """Paired record bootstrap, run after faster diagnostics in notebook 06."""
    if bootstrap_rounds < 0:
        raise ValueError("bootstrap_rounds cannot be negative")
    if bootstrap_rounds == 0:
        return pd.DataFrame()
    if test is None:
        manifest = json.loads((OUT / "temporal_run.json").read_text())
        if manifest["fingerprint"] != run_fingerprint():
            raise ValueError("Temporal artifacts have a different model configuration")
        test = pd.read_parquet(OUT / "temporal_test_2024.parquet")
        if not test.source_year.eq(2024).all():
            raise ValueError("Bootstrap requires 2024 temporal test")
        source = _verify_and_load(["source_year", "source_row", "fold", *OUTCOMES])
        _validate_saved_outcomes(test, source, 2024)
        del source
    names = MODEL_NAMES + TEMPORAL_SENSITIVITIES
    _validate_probability_columns(test, names, "2024 temporal bootstrap")
    y = test.label.to_numpy(dtype=np.uint8)
    predictions = {name: test[name].to_numpy() for name in names[1:]}
    rng = np.random.default_rng(seed)
    bootstrap = []
    # A paired, record-level bootstrap preserves the same 2024 resample for all models.
    for repeat in tqdm(range(bootstrap_rounds), desc="2024 paired bootstrap", unit="replicate"):
        index = rng.integers(0, len(test), size=len(test), dtype=np.int32)
        y_rep = y[index]
        for name in names[1:]:
            p_rep = predictions[name][index]
            scores = metrics(y_rep, p_rep)
            bootstrap.append({"repeat": repeat, "model": name, "auroc": scores["auroc"],
                              "auprc": scores["auprc"], "brier": scores["brier"]})
    boot = pd.DataFrame(bootstrap)
    if not boot.empty:
        boot.to_csv(OUT / "temporal_bootstrap_2024.csv", index=False)
    return boot


def secondary_outcome_metrics(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Discrimination only: HDP scores are not probabilities of composites."""
    rows = []
    for outcome in ("target_hdp_preterm37", "target_hdp_preterm34"):
        known = frame[outcome].notna()
        y = frame.loc[known, outcome].to_numpy(dtype=np.uint8)
        for name in models:
            p = frame.loc[known, name].to_numpy()
            rows.append({"outcome": outcome, "model": name, "n": len(y),
                         "events": int(y.sum()),
                         "auroc": float(roc_auc_score(y, p)),
                         "auprc": float(average_precision_score(y, p))})
    return pd.DataFrame(rows)


def secondary_trained_summary(main_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare score ranking on composites and on PTB without recorded HDP."""
    manifest = json.loads((OUT / "secondary_temporal_run.json").read_text())
    if manifest["fingerprint"] != run_fingerprint() or manifest["targets"] != SECONDARY_TARGETS:
        raise ValueError("Secondary model manifest does not match current specification")
    dev = pd.read_parquet(OUT / "secondary_dev_oof.parquet")
    test = pd.read_parquet(OUT / "secondary_test_2024.parquet")
    source = _verify_and_load(["source_year", "source_row", "fold", *OUTCOMES,
                               "oegest_comb"])
    _validate_saved_outcomes(dev, source, 2023, gestation=True)
    _validate_saved_outcomes(test, source, 2024, gestation=True)
    del source
    _validate_probability_columns(dev, list(SECONDARY_TARGETS), "2023 composite development")
    _validate_probability_columns(test, list(SECONDARY_TARGETS), "2024 composite test")
    audit = json.loads((OUT / "audit.json").read_text())
    for year, frame in ((2023, dev), (2024, test)):
        if not frame.source_year.eq(year).all() or len(frame) != audit["year_checks"][str(year)]["eligible_singleton_rows"]:
            raise ValueError(f"Incomplete secondary predictions for {year}")
        if frame.duplicated(["source_year", "source_row"]).any():
            raise ValueError("Duplicate secondary predictions")
    if len(main_test) != len(test) or main_test.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Main and secondary temporal tests have different grains")
    if (np.array_equal(test.source_year.to_numpy(), main_test.source_year.to_numpy()) and
            np.array_equal(test.source_row.to_numpy(), main_test.source_row.to_numpy())):
        joined = test
        joined["label"] = main_test.label.to_numpy()
        joined["hist_gb"] = main_test.hist_gb.to_numpy()
    else:
        joined = test.merge(main_test[["source_year", "source_row", "label", "hist_gb"]],
            on=["source_year", "source_row"], how="inner", validate="one_to_one")
    if len(joined) != len(test) or not joined.label.eq(joined.target_hdp_proxy).all():
        raise ValueError("Main and secondary temporal test records or labels differ")
    scores = ["hist_gb", *SECONDARY_TARGETS]
    rows = []
    for endpoint in SECONDARY_TARGETS.values():
        known = joined[endpoint].notna()
        y = joined.loc[known, endpoint].to_numpy(dtype=np.uint8)
        for name in scores:
            p = joined.loc[known, name].to_numpy()
            result = metrics(y, p)
            matched = SECONDARY_TARGETS.get(name) == endpoint
            rows.append({"evaluation_endpoint": endpoint, "score": name, "subset": "all_known_gestation",
                         "n": len(y), "events": int(y.sum()), "auroc": result["auroc"],
                         "auprc": result["auprc"],
                         "brier_if_matching_target": result["brier"] if matched else np.nan,
                         "calibration_offset_if_matching_target": calibration_offset(y, p) if matched else np.nan})
    for threshold in (37, 34):
        subset = joined.loc[joined.label.eq(0) & joined.oegest_comb.notna()]
        y = subset.oegest_comb.lt(threshold).to_numpy(dtype=np.uint8)
        for name in scores:
            p = subset[name].to_numpy()
            rows.append({"evaluation_endpoint": f"preterm{threshold}_without_recorded_hdp",
                         "score": name, "subset": "recorded_hdp_negative_only",
                         "n": len(y), "events": int(y.sum()),
                         "auroc": float(roc_auc_score(y, p)),
                         "auprc": float(average_precision_score(y, p)),
                         "brier_if_matching_target": np.nan,
                         "calibration_offset_if_matching_target": np.nan})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "secondary_trained_comparison_2024.csv", index=False)
    fold_rows = []
    for fold in range(5):
        part = dev.loc[dev.fold.eq(fold)]
        for name, endpoint in SECONDARY_TARGETS.items():
            known = part[endpoint].notna()
            y = part.loc[known, endpoint].to_numpy(dtype=np.uint8)
            fold_rows.append({"fold": fold, "model": name, "target": endpoint,
                              "n": len(y), "events": int(y.sum()),
                              **metrics(y, part.loc[known, name].to_numpy())})
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUT / "secondary_development_fold_metrics.csv", index=False)
    return comparison, fold_metrics


def temporal_subgroup_audit(dev: pd.DataFrame, test: pd.DataFrame,
                            models: list[str] | None = None) -> pd.DataFrame:
    """Calibration and alert rates at each model's 2023 OOF cutoff."""
    names = models or ["logistic_spline", "hist_gb", "catboost", "lightgbm", "hist_gb_no_masks",
                       "hist_gb_early", "hist_gb_demographic"]
    if not dev.source_year.eq(2023).all() or not test.source_year.eq(2024).all():
        raise ValueError("Subgroup audit requires 2023 development and 2024 test")
    thresholds = {name: fixed_threshold_from_development(dev[name].to_numpy()) for name in names}
    groups = {"no_recorded_pregestational_diabetes": test.rf_pdiab.eq(0),
              "pregestational_diabetes": test.rf_pdiab.eq(1),
              "art_reported_yes": test.art_source_reported.eq(1) & test.rf_artec.eq(1),
              "infertility_treatment_without_art": test.art_source_reported.eq(1) &
                  test.rf_inftr.eq(1) & test.rf_artec.eq(0),
              "no_infertility_treatment": test.art_source_reported.eq(1) &
                  test.rf_inftr.eq(0) & test.rf_artec.eq(0),
              "art_not_reported": test.art_source_reported.eq(0),
              "art_unknown_reported": test.art_status_unknown_reported.eq(1)}
    groups.update({f"{name}_yes": test[name].eq(1) for name in
                   ["bmi_missing", "prior_live_birth_count_missing", "priorterm_missing"]})
    groups.update({f"race_code_{int(code)}": test.mrace6.eq(code)
                   for code in sorted(test.mrace6.dropna().unique())})
    groups.update({f"ethnicity_code_{int(code)}": test.mhisp_r.eq(code)
                   for code in sorted(test.mhisp_r.dropna().unique())})
    rows = []
    for group, mask in tqdm(groups.items(), desc="2024 subgroup audit", unit="group"):
        part = test.loc[mask]
        if len(part) < 100 or part.label.nunique() < 2:
            continue
        y = part.label.to_numpy(dtype=np.uint8)
        for name in names:
            p = part[name].to_numpy()
            fixed = operating_point(y, p >= thresholds[name])
            rows.append({"group": group, "model": name, "n": len(part),
                         "events": int(y.sum()), **metrics(y, p),
                         "mean_calibration_gap_pp": 100 * (y.mean()-p.mean()),
                         "calibration_in_the_large": calibration_offset(y, p),
                         "development_threshold_10pct": thresholds[name],
                         **{f"fixed_{key}": value for key, value in fixed.items()}})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "temporal_subgroups.csv", index=False)
    return result


def art_reclassification(dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Core-versus-ART score movement at fixed 2023-derived 10% alert budgets."""
    if not dev.source_year.eq(2023).all() or not test.source_year.eq(2024).all():
        raise ValueError("ART reclassification requires 2023 development and 2024 test")
    base = "hist_gb"
    early = "hist_gb_early"
    base_cutoff = fixed_threshold_from_development(dev[base].to_numpy())
    early_cutoff = fixed_threshold_from_development(dev[early].to_numpy())
    groups = {"ART_yes_reported": test.art_source_reported.eq(1) & test.rf_artec.eq(1),
              "infertility_treatment_without_ART": test.art_source_reported.eq(1) &
                  test.rf_inftr.eq(1) & test.rf_artec.eq(0),
              "no_infertility_treatment": test.art_source_reported.eq(1) &
                  test.rf_inftr.eq(0) & test.rf_artec.eq(0),
              "ART_unknown_reported": test.art_status_unknown_reported.eq(1),
              "ART_not_reported_core_fallback": test.art_source_reported.eq(0)}
    rows = []
    for group, mask in groups.items():
        part = test.loc[mask]
        if part.empty:
            continue
        base_flag = part[base].to_numpy() >= base_cutoff
        early_flag = part[early].to_numpy() >= early_cutoff
        gained = ~base_flag & early_flag
        removed = base_flag & ~early_flag
        y = part.label.to_numpy(dtype=np.uint8)
        rows.append({"group": group, "n": len(part), "events": int(y.sum()),
                     "observed_rate": float(y.mean()),
                     "core_mean_prediction": float(part[base].mean()),
                     "early_mean_prediction": float(part[early].mean()),
                     "changed_individual_scores": int(np.sum(~np.isclose(
                         part[base].to_numpy(), part[early].to_numpy(), rtol=0, atol=1e-7))),
                     "core_alerts": int(base_flag.sum()), "early_alerts": int(early_flag.sum()),
                     "newly_alerted": int(gained.sum()), "newly_alerted_events": int(y[gained].sum()),
                     "no_longer_alerted": int(removed.sum()),
                     "no_longer_alerted_events": int(y[removed].sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "temporal_art_reclassification.csv", index=False)
    return result
