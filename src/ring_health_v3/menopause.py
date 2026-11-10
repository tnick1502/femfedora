"""Menopausal-transition analysis: STRAW+10 stage, FMP timing, nocturnal VMS burden.

Final stage = deterministic STRAW+10 bleeding criteria where they are definitive (>=12 months amenorrhea),
otherwise the calibrated ML distribution (bleeding + ring + symptoms + optional labs). FMP timing: probability
that the final menstrual period has occurred or will occur within 6..60 months. All models are trained on the
synthetic longitudinal generator (sim_menopause) whose age-at-menopause distribution is fitted on real NHANES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cycle_engine import CycleModel
from .menopause_features import FEATURES, NIGHT_FEATURES, bleeding_features, extract, night_table
from .sim_menopause import STAGES

HORIZONS = [6, 12, 24, 36, 48, 60]
STAGE_LABELS_RU = {"reproductive": "репродуктивный период", "early_transition": "ранний менопаузальный переход",
                   "late_transition": "поздний менопаузальный переход", "postmenopause": "постменопауза"}


def straw_rules(bf, logs_reliable, has_last_period_date=False, logs_ok_for_patterns=None):
    flags = {"amenorrhea_12m": False, "interval_ge60": False, "persistent_diff7": False}
    ds = bf.get("days_since_last_period")
    # >=12 months only from a reliable source: the onboarding date or complete/long logs (stale partial logs are not proof)
    if ds is not None and np.isfinite(ds) and ds >= 365 and (logs_reliable or has_last_period_date):
        flags["amenorrhea_12m"] = True
    if logs_reliable if logs_ok_for_patterns is None else logs_ok_for_patterns:
        if (np.isfinite(bf.get("max_interval", np.nan)) and bf["max_interval"] >= 60) or (np.isfinite(ds or np.nan) and ds >= 60):
            flags["interval_ge60"] = True
        if np.isfinite(bf.get("persistent_diff7", np.nan)) and bf["persistent_diff7"] >= 2:
            flags["persistent_diff7"] = True
    return flags


def _grouped(probs, label, basis):
    g = {"reproductive": probs["reproductive"], "transition": probs["early_transition"] + probs["late_transition"], "postmenopause": probs["postmenopause"]}
    lab = "postmenopause" if basis != "model" else max(g, key=g.get)
    return {"label": lab, "probabilities": {k: round(v, 3) for k, v in g.items()}}


def analyze_menopause(bundle, cycle_model: CycleModel, daily, starts, as_of, profile, *, last_period_date=None, logs_complete=False,
                      logging_since=None, symptoms=None, labs=None):
    age = float(profile["age"])
    as_of = pd.Timestamp(as_of)
    excl = [k for k in ["pregnant", "breastfeeding", "hormonal_contraception", "menopausal_hormone_therapy", "hysterectomy", "bilateral_oophorectomy"] if profile.get(k)]
    if excl:
        return {"status": "out_of_scope", "reason": "Bleeding-based staging not applicable: " + ", ".join(excl)}
    if age < 35:
        return {"status": "not_applicable_age", "reason": "Menopause-transition models cover age 35-65; early ovarian insufficiency is out of scope"}
    log_hist = (as_of - pd.Timestamp(logging_since)).days if logging_since else None
    sym = symptoms or {}
    labs = labs or {}
    lab_vals = {"fsh": labs.get("fsh"), "amh": labs.get("amh")}
    cov = {"smoker": profile.get("smoker"), "bmi": profile.get("bmi"), "parous": profile.get("parous"), "college": profile.get("college_education")}
    sym_days = sorted({pd.Timestamp(x) for x in list(sym.get("hot_flash_dates", [])) + list(sym.get("night_sweat_dates", []))})  # one day counts once
    f = extract(daily, starts, as_of, min(age, 65), cycle_model=cycle_model, night_model=bundle["vms"]["night_model"],
                symptom_dates=sym_days, symptom_logging=sym.get("logging_enabled"),
                log_history_days=log_hist, labs=lab_vals, covariates=cov, last_period_date=last_period_date)
    X = pd.DataFrame([f])[FEATURES]
    has_bleeding = bool(np.isfinite(f["days_since_last_period"]))
    has_ring = bool(np.isfinite(f["ring_nights_90"]) and f["ring_nights_90"] >= 20)
    m = bundle["models"]
    has_labs = any(v is not None for v in lab_vals.values())
    # input pattern -> model, chosen on VALIDATION data (tools/train_ring_health_v3.py, choose_routing)
    pattern = "labs" if has_labs else ("bleeding_ring" if has_bleeding and has_ring else ("bleeding_only" if has_bleeding else ("ring_only" if has_ring else "age_only")))
    which = m.get("routing", {}).get(pattern, "full")
    age_X = np.c_[[age], [(age - 50) ** 2 / 10]]
    inp = (lambda feats: age_X) if which == "age_only" else (lambda feats: X[feats])
    st = m["stage"][which]["model"].predict_proba(inp(m["stage"][which]["features"]))[0]
    age_st = m["stage"]["age_only"]["model"].predict_proba(age_X)[0]
    fm = np.array([m["fmp"][which][h]["model"].predict_proba(inp(m["fmp"][which][h]["features"]))[0, 1] for h in HORIZONS])
    fm = np.maximum.accumulate(fm)
    bf = {k: f[k] for k in f}
    rules = straw_rules(bf, logs_reliable=bool(logs_complete) or (log_hist is not None and log_hist >= 400), has_last_period_date=last_period_date is not None,
                        logs_ok_for_patterns=bool(logs_complete) or (log_hist is not None and log_hist >= 180))
    probs = dict(zip(STAGES, map(float, st)))
    basis = "model"
    label = max(probs, key=probs.get)
    if rules["amenorrhea_12m"]:
        label, basis = "postmenopause", "STRAW+10: >=12 months without a period"
        fm = np.ones_like(fm)
    sufficient = has_bleeding or has_ring
    if not sufficient:
        status = "insufficient_data"
    else:
        status = "ok"
    # median months to FMP (from monotone CDF over horizons)
    med = None
    if fm[0] >= 0.5:
        med = "<6"
    elif fm[-1] < 0.5:
        med = ">60"
    else:
        j = int(np.argmax(fm >= 0.5)); x0, x1, y0, y1 = HORIZONS[j - 1], HORIZONS[j], fm[j - 1], fm[j]
        med = round(float(x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-9)), 1)
    vms = {"index_last30": None, "share_of_scored_nights_with_likely_hot_flashes_last30": None}
    vms["self_reported_symptom_days_last30"] = (None if not np.isfinite(f["hot_flash_days_30"]) else int(f["hot_flash_days_30"]))
    if np.isfinite(f["vms_index_30"]):
        vms.update({"index_last30": round(f["vms_index_30"], 3), "share_of_scored_nights_with_likely_hot_flashes_last30": round(f["vms_nights_30"], 3)})
    conf = "high" if (has_bleeding and has_ring) else ("medium" if sufficient else "low")
    res = {
        "status": status,
        "stage": {"label": label, "label_ru": STAGE_LABELS_RU[label], "basis": basis, "probabilities": {k: round(v, 3) for k, v in probs.items()},
                  "grouped": _grouped(probs, label, basis), "model_used": which,
                  "age_only_reference": dict(zip(STAGES, [round(float(x), 3) for x in age_st])), "straw10_criteria": rules},
        "final_menstrual_period": {"probability_by_months": {str(h): round(float(p), 3) for h, p in zip(HORIZONS, fm)}, "median_months": med,
                                   "already_confirmed": bool(rules["amenorrhea_12m"]),
                                   "definition": "P(final menstrual period has occurred or occurs within N months); FMP is confirmed only after 12 months without bleeding"},
        "vasomotor": vms,
        "signals": {k: (None if not np.isfinite(f[k]) else round(float(f[k]), 3)) for k in ["ring_ovulatory_fraction", "ring_cycles_completed", "temp_cycle_power", "temp_night_sd", "lnhrv_age_resid",
                                                                                          "eff_mean_30", "awake_mean_30", "days_since_last_period", "max_interval", "persistent_diff7", "hot_flash_days_30"]},
        "data_quality": {"input_pattern": pattern, "symptoms_used": bool(sym.get("logging_enabled")), "bleeding_information": bool(has_bleeding), "ring_nights_last90": None if not np.isfinite(f["ring_nights_90"]) else int(f["ring_nights_90"]),
                         "labs_used": [k for k, v in lab_vals.items() if v is not None], "confidence": conf,
                         "missing_inputs_policy": "not imputed; trees route missing inputs learned under random masking"},
    }
    if status == "insufficient_data":
        age_fm = np.maximum.accumulate(np.array([m["fmp"]["age_only"][h]["model"].predict_proba(np.c_[[age], [(age - 50) ** 2 / 10]])[0, 1] for h in HORIZONS]))
        res["stage"] = {"label": None, "label_ru": None, "basis": "insufficient_data", "probabilities": None, "grouped": None,
                        "age_only_reference": res["stage"]["age_only_reference"], "straw10_criteria": rules}
        res["final_menstrual_period"] = {"probability_by_months": None, "median_months": None, "already_confirmed": False,
                                         "age_only_reference_by_months": {str(h): round(float(p), 3) for h, p in zip(HORIZONS, age_fm)}}
        res["needed"] = ["date of last period or period logs", "or >= 20 nights of ring data in the last 90 days"]
    return res
