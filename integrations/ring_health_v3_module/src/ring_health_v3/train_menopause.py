"""Train/evaluate menopausal-transition models on the longitudinal generator.

Pipeline: simulate women -> landmarks (ring window, logs, symptoms, labs) -> MCAR input masking ->
night-level VMS model -> landmark features -> stage classifier + FMP-by-horizon classifiers.
Person-disjoint train / validation / test. Selection on validation only.
"""
from __future__ import annotations

import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score, brier_score_loss

from .cycle_engine import CycleModel
from .menopause_features import FEATURES, NIGHT_FEATURES, extract, night_table
from .sim_menopause import STAGES, FMPAgeSampler, sample_woman, simulate_landmarks

HORIZONS = [6, 12, 24, 36, 48, 60]
BLEED = ["days_since_last_period", "periods_last_6m", "periods_last_12m", "median_interval", "interval_sd", "max_interval", "intervals_ge60",
         "persistent_diff7", "log_history_days", "last_interval"]
RING = ["ring_nights_90", "ring_cycles_completed", "ring_ovulatory_fraction", "ring_temp_shift_mean", "temp_night_sd", "temp_spike_rate", "vms_index_30",
        "vms_nights_30", "eff_mean_30", "awake_mean_30", "lnhrv_age_resid", "hr_mean_90", "sleep_mean_30", "ring_days_since_luteal", "temp_cycle_power"]
LABS = ["ln_fsh", "ln_amh"]
COV = ["smoker", "bmi", "parous", "college"]

_G = {}


def _init(cycle_params, nhanes, night_model):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # noqa: BLE001
        pass
    _G["cm"] = CycleModel(cycle_params); _G["fmp"] = FMPAgeSampler(nhanes); _G["night"] = night_model


def availability(rng, mode="random"):
    if mode == "random":
        return {"logs": rng.random() < 0.85, "lpd": rng.random() < 0.75, "ring": rng.random() < 0.92, "fsh": rng.random() < 0.25, "amh": rng.random() < 0.15,
                "cov": {k: rng.random() < 0.6 for k in COV}}
    full = {"logs": True, "lpd": True, "ring": True, "fsh": False, "amh": False, "cov": {k: False for k in COV}}
    if mode == "full":
        return full
    if mode == "ring_only":
        return {**full, "logs": False, "lpd": False}
    if mode == "lpd_ring":
        return {**full, "logs": False}
    if mode == "logs_only":
        return {**full, "ring": False}
    if mode == "age_only":
        return {**full, "logs": False, "lpd": False, "ring": False}
    if mode == "full_plus_labs":
        return {**full, "fsh": True, "amh": True, "cov": {k: True for k in COV}}
    raise ValueError(mode)


def landmark_rows(args):
    wid, seed, k, mode, want_nights = args
    rng = np.random.default_rng(seed)
    w = sample_woman(rng, _G["fmp"])
    lms = simulate_landmarks(rng, w, k)
    rows, nights = [], []
    for j, lm in enumerate(lms):
        win = lm["window"]; av = availability(rng, mode)
        age = lm["landmark_age"]
        daily = win["daily"] if av["ring"] else None
        starts = win["logged_starts"] if av["logs"] else []
        labs = {"fsh": lm["fsh"] if av["fsh"] else None, "amh": lm["amh"] if av["amh"] else None}
        cov = {kk: (lm["woman"][kk] if av["cov"][kk] else None) for kk in COV}
        # onboarding answer "date of your last period" (recall error grows with elapsed time)
        lpd = None
        if av["lpd"] and np.isfinite(lm["months_since_last_true_period"]):  # onboarding answer, independent of logging
            days_ago = lm["months_since_last_true_period"] * 30.44
            days_ago = max(0.0, days_ago * (1 + rng.normal(0, 0.10)) + rng.normal(0, 3))
            lpd = win["as_of"] - pd.Timedelta(days=int(round(days_ago)))
        f = extract(daily, starts, win["as_of"], age, cycle_model=_G["cm"], night_model=_G["night"], symptom_dates=win["symptom_dates"],
                    symptom_logging=win["symptom_logger"], log_history_days=win["log_history_days"] if av["logs"] else None, labs=labs, covariates=cov,
                    last_period_date=lpd)
        f.update(woman=wid, landmark=j, stage=lm["stage"], years_to_fmp=lm["years_to_fmp"], fmp_age=lm["fmp_age"], landmark_age=age,
                 months_since_last_true_period=lm["months_since_last_true_period"], mode=mode, has_logs=av["logs"], has_ring=av["ring"],
                 vms_class=lm["woman"]["vms_class"], true_flash_nights_30=float((win["flashes"][-30:] > 0).mean()), fsh_true=lm["fsh"], amh_true=lm["amh"],
                 has_lpd=lpd is not None)
        rows.append(f)
        if want_nights and av["ring"]:
            nt = night_table(win["daily"]); nt["flash"] = (win["flashes"] > 0).astype(int); nt["woman"] = wid
            nights.append(nt.dropna(subset=["d_temp"]).iloc[::2])
    return rows, nights


def simulate_table(n, seed0, k, mode, cycle_params, nhanes, night_model=None, want_nights=False, processes=2):
    args = [(f"w{seed0}_{i}", seed0 + 104729 * i, k, mode, want_nights) for i in range(n)]
    # spawn: forking after OpenMP (HistGradientBoosting) initialisation can deadlock workers
    with get_context("spawn").Pool(processes, initializer=_init, initargs=(cycle_params, nhanes, night_model)) as pool:
        res = pool.map(landmark_rows, args, chunksize=8)
    rows = pd.DataFrame([r for a, _ in res for r in a])
    nights = pd.concat([x for _, b in res for x in b], ignore_index=True) if want_nights else None
    return rows, nights


def fit_night_model(nights):
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    m.fit(nights[NIGHT_FEATURES], nights["flash"])
    return m


def hgb(seed=0, **kw):
    base = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=25, random_state=seed)
    base.update(kw)
    return HistGradientBoostingClassifier(**base)


def stage_target(df):
    return df["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()


def fmp_target(df, h):
    return (df["years_to_fmp"] * 12 <= h).astype(int).to_numpy()


def feature_sets():
    return {"age_only": ["age"], "age_bleeding": ["age"] + BLEED, "age_ring": ["age"] + RING + ["hot_flash_days_30"],
            "full": FEATURES}


def train_all(train, val):
    models = {"stage": {}, "fmp": {}}
    report = {"stage": {}, "fmp": {}}
    for name, cols in feature_sets().items():
        y = stage_target(train)
        if name == "age_only":
            m = LogisticRegression(max_iter=2000, C=1.0).fit(np.c_[train.age, (train.age - 50) ** 2 / 10], y)
            pv = m.predict_proba(np.c_[val.age, (val.age - 50) ** 2 / 10])
        else:
            m = hgb().fit(train[cols], y); pv = m.predict_proba(val[cols])
        models["stage"][name] = {"model": m, "features": cols}
        yv = stage_target(val)
        report["stage"][name] = {"val_log_loss": float(log_loss(yv, pv, labels=range(4))), "val_macro_f1": float(f1_score(yv, pv.argmax(1), average="macro")),
                                 "val_balanced_accuracy": float(balanced_accuracy_score(yv, pv.argmax(1)))}
        models["fmp"][name] = {}; report["fmp"][name] = {}
        for h in HORIZONS:
            y = fmp_target(train, h)
            if name == "age_only":
                m = LogisticRegression(max_iter=2000).fit(np.c_[train.age, (train.age - 50) ** 2 / 10], y)
                pv = m.predict_proba(np.c_[val.age, (val.age - 50) ** 2 / 10])[:, 1]
            else:
                m = hgb().fit(train[cols], y); pv = m.predict_proba(val[cols])[:, 1]
            models["fmp"][name][h] = {"model": m, "features": cols}
            yv = fmp_target(val, h)
            pre = val.years_to_fmp.to_numpy() > 0
            report["fmp"][name][h] = {"val_auc_all": float(roc_auc_score(yv, pv)), "val_auc_pre_fmp": float(roc_auc_score(yv[pre], pv[pre])) if 0 < yv[pre].mean() < 1 else None,
                                      "val_brier": float(brier_score_loss(yv, pv))}
    return models, report


def availability_pattern(df):
    b = df.days_since_last_period.notna().to_numpy(); r = (df.ring_nights_90 >= 20).fillna(False).to_numpy()
    lab = (df.ln_fsh.notna() | df.ln_amh.notna()).to_numpy()
    return np.where(lab, "labs", np.where(b & r, "bleeding_ring", np.where(b, "bleeding_only", np.where(r, "ring_only", "age_only"))))


def choose_routing(models, val):
    """Per input pattern pick the model with the best VALIDATION stage log loss (+ mean FMP AUC tie-break)."""
    pat = availability_pattern(val); routing = {}; detail = {}
    cand = {"labs": ["full"], "bleeding_ring": ["full", "age_bleeding", "age_ring"], "bleeding_only": ["full", "age_bleeding"],
            "ring_only": ["full", "age_ring"], "age_only": ["full", "age_only"]}
    yv = stage_target(val)
    for pname, names in cand.items():
        m = pat == pname
        if m.sum() < 50:
            routing[pname] = names[0]; continue
        scores = {}
        for nm in names:
            P = predict_stage(models, nm, val[m]); F = predict_fmp(models, nm, val[m])
            aucs = [roc_auc_score(fmp_target(val[m], h), F[:, j]) for j, h in enumerate(HORIZONS) if 0 < fmp_target(val[m], h).mean() < 1]
            scores[nm] = {"stage_log_loss": float(log_loss(yv[m], P, labels=range(4))), "fmp_mean_auc": float(np.mean(aucs)) if aucs else None}
        routing[pname] = min(scores, key=lambda k: (scores[k]["stage_log_loss"], -(scores[k]["fmp_mean_auc"] or 0)))
        detail[pname] = {"n": int(m.sum()), "scores": scores, "chosen": routing[pname]}
    return routing, detail


def straw_rule_mask(df):
    """Same condition as the API: >=12 months since last period from the onboarding date or long logs."""
    reliable = df.get("has_lpd", pd.Series(False, index=df.index)).fillna(False).to_numpy() | (df.log_history_days >= 400).fillna(False).to_numpy()
    return (df.days_since_last_period >= 365).fillna(False).to_numpy() & reliable


def predict_routed(models, routing, df, straw_rule=True):
    pat = availability_pattern(df)
    P = np.zeros((len(df), 4)); F = np.zeros((len(df), len(HORIZONS)))
    for pname in np.unique(pat):
        m = pat == pname; nm = routing.get(pname, "full")
        P[m] = predict_stage(models, nm, df[m]); F[m] = predict_fmp(models, nm, df[m])
    if straw_rule:
        rule = straw_rule_mask(df)
        P[rule] = [0, 0, 0, 1]; F[rule] = 1.0
    return P, F


def evaluate_routed(models, routing, test):
    P, F = predict_routed(models, routing, test); y = stage_target(test)
    g3 = lambda a: np.where(a == 3, 2, np.where(a == 2, 1, a))
    out = {"n": int(len(test)), "stage_macro_f1": float(f1_score(y, P.argmax(1), average="macro")), "stage_grouped3_macro_f1": float(f1_score(g3(y), g3(P.argmax(1)), average="macro")),
           "stage_recall": {s: float((P.argmax(1)[y == i] == i).mean()) for i, s in enumerate(STAGES)}}
    pre = (test.years_to_fmp > 0).to_numpy() & test.landmark_age.between(42, 56).to_numpy()
    for j, h in enumerate(HORIZONS):
        yh = fmp_target(test, h)
        out[f"fmp_auc_{h}m_pre_fmp_42_56"] = float(roc_auc_score(yh[pre], F[pre, j])) if 0 < yh[pre].mean() < 1 else None
    med = median_months(F)
    out["c_index_time_to_fmp_pre"] = concordance(test.years_to_fmp.to_numpy()[pre], med[pre])
    out["median_abs_error_months_pre"] = float(np.median(np.abs(np.clip(med[pre], 0, 72) - np.clip(test.years_to_fmp.to_numpy()[pre] * 12, 0, 72))))
    # false 'confirmed postmenopause' among women still before FMP
    rule = straw_rule_mask(test)
    out["straw_rule_false_confirmations_pre_fmp"] = float(rule[(test.years_to_fmp > 0).to_numpy()].mean())
    return out


def predict_stage(models, name, df):
    m = models["stage"][name]
    if name == "age_only":
        return m["model"].predict_proba(np.c_[df.age, (df.age - 50) ** 2 / 10])
    return m["model"].predict_proba(df[m["features"]])


def predict_fmp(models, name, df):
    out = []
    for h in HORIZONS:
        m = models["fmp"][name][h]
        p = m["model"].predict_proba(np.c_[df.age, (df.age - 50) ** 2 / 10])[:, 1] if name == "age_only" else m["model"].predict_proba(df[m["features"]])[:, 1]
        out.append(p)
    P = np.maximum.accumulate(np.vstack(out).T, axis=1)  # monotone CDF over horizons
    return P


def median_months(P):
    hz = np.array(HORIZONS, float)
    med = np.full(len(P), np.nan)
    for i, row in enumerate(P):
        if row[0] >= 0.5:
            med[i] = 6 * 0.5 / max(row[0], 1e-6) if row[0] > 0 else 6
        elif row[-1] < 0.5:
            med[i] = 72.0
        else:
            j = int(np.argmax(row >= 0.5)); x0, x1, y0, y1 = hz[j - 1], hz[j], row[j - 1], row[j]
            med[i] = x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-9)
    return med


def concordance(time_true, pred_rank):
    """Harrell's C without censoring (synthetic truth fully observed)."""
    t = np.asarray(time_true); r = np.asarray(pred_rank)
    n = len(t); idx = np.random.default_rng(0).choice(n, min(n, 4000), replace=False)
    t, r = t[idx], r[idx]
    dt = t[:, None] - t[None, :]; dr = r[:, None] - r[None, :]
    m = dt != 0
    conc = ((dt * dr) > 0) & m; ties = (dr == 0) & m
    return float((conc.sum() + 0.5 * ties.sum()) / m.sum())


def evaluate(models, test, name):
    P = predict_stage(models, name, test); y = stage_target(test)
    out = {"n": int(len(test)), "women": int(test.woman.nunique()), "stage_macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
           "stage_balanced_accuracy": float(balanced_accuracy_score(y, P.argmax(1))), "stage_log_loss": float(log_loss(y, P, labels=range(4))),
           "stage_recall": {s: float((P.argmax(1)[y == i] == i).mean()) for i, s in enumerate(STAGES)},
           "stage_confusion": pd.crosstab(pd.Series(y, name="true"), pd.Series(P.argmax(1), name="pred")).reindex(index=range(4), columns=range(4), fill_value=0).values.tolist()}
    F = predict_fmp(models, name, test)
    pre = (test.years_to_fmp > 0).to_numpy() & test.landmark_age.between(42, 56).to_numpy()
    for j, h in enumerate(HORIZONS):
        yh = fmp_target(test, h)
        out[f"fmp_auc_{h}m_pre_fmp_42_56"] = float(roc_auc_score(yh[pre], F[pre, j])) if 0 < yh[pre].mean() < 1 else None
        out[f"fmp_brier_{h}m_pre_fmp_42_56"] = float(brier_score_loss(yh[pre], F[pre, j]))
        out[f"fmp_calibration_{h}m"] = {"mean_pred": float(F[pre, j].mean()), "observed": float(yh[pre].mean())}
    med = median_months(F)
    out["c_index_time_to_fmp_pre"] = concordance(test.years_to_fmp.to_numpy()[pre], med[pre])
    out["median_abs_error_months_pre"] = float(np.median(np.abs(np.clip(med[pre], 0, 72) - np.clip(test.years_to_fmp.to_numpy()[pre] * 12, 0, 72))))
    return out
