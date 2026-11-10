"""Real-data survival model of age at natural menopause (NHANES 2015-2016 + 2017-2018).

Event: no period in past 12 months (RHQ031=2) with reason menopause (RHD043=7); event age RHQ060.
Censoring: still menstruating (RHQ031=1) at interview age; hysterectomy (RHD043=3 / RHD280=1) or bilateral
oophorectomy before natural menopause at the surgery age (RHQ291 / RHQ332). Pregnancy, breastfeeding and
'other' reasons are excluded. Interview age 35-65 limits long-recall bias. Survey weights are used for KM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load(nhanes_dir: Path) -> pd.DataFrame:
    parts = []
    for sfx, wave in [("I", "2015-2016"), ("J", "2017-2018")]:
        r = pd.read_sas(nhanes_dir / f"RHQ_{sfx}.xpt"); d = pd.read_sas(nhanes_dir / f"DEMO_{sfx}.xpt"); b = pd.read_sas(nhanes_dir / f"BMX_{sfx}.xpt")
        x = r.merge(d[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "WTINT2YR"]], on="SEQN").merge(b[["SEQN", "BMXBMI"]], on="SEQN", how="left")
        x["wave"] = wave
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def build(df: pd.DataFrame, age_lo=35, age_hi=65) -> pd.DataFrame:
    x = df[(df.RIAGENDR == 2) & df.RIDAGEYR.between(age_lo, age_hi)].copy()
    age = x.RIDAGEYR.astype(float)
    lmp = x.RHQ060.where(x.RHQ060.between(19, 60))
    hyst_age = x.RHQ291.where(x.RHQ291.between(19, 60)); ooph_age = x.RHQ332.where(x.RHQ332.between(19, 60))
    surg_age = pd.concat([hyst_age.where(x.RHD280 == 1), ooph_age.where(x.RHQ305 == 1)], axis=1).min(axis=1)
    out = pd.DataFrame({"seqn": x.SEQN, "wave": x.wave, "weight": x.WTINT2YR / 2, "interview_age": age, "race": x.RIDRETH3,
                        "education": x.DMDEDUC2, "bmi": x.BMXBMI, "menarche": x.RHQ010.where(x.RHQ010.between(8, 20)),
                        "parity": x.RHQ171.where(x.RHQ171.between(0, 15)).fillna(pd.Series(np.where(x.RHQ131 == 2, 0.0, np.nan), index=x.index))})
    out["time"] = np.nan; out["event"] = 0; out["group"] = "excluded"
    menstruating = x.RHQ031 == 1
    out.loc[menstruating, ["time", "group"]] = [np.nan, "menstruating"]; out.loc[menstruating, "time"] = age[menstruating]
    natural = (x.RHQ031 == 2) & (x.RHD043 == 7) & lmp.notna()
    # oophorectomy at (or within 1 y of) the reported last period = surgical menopause -> censor at surgery
    surg_first = natural & surg_age.notna() & (surg_age <= lmp + 1)
    nat = natural & ~surg_first
    out.loc[nat, "time"] = lmp[nat]; out.loc[nat, "event"] = 1; out.loc[nat, "group"] = "natural_menopause"
    surg = ((x.RHQ031 == 2) & (x.RHD043 == 3) & surg_age.notna()) | surg_first
    out.loc[surg, "time"] = surg_age[surg]; out.loc[surg, "event"] = 0; out.loc[surg, "group"] = "surgery_censored"
    out = out[out.group != "excluded"].dropna(subset=["time"])
    return out


def weighted_km(time, event, weight, grid):
    order = np.argsort(time)
    t = np.asarray(time)[order]; e = np.asarray(event)[order]; w = np.asarray(weight)[order]
    surv = []; s = 1.0; at_risk = w.sum(); j = 0; ut = np.unique(t)
    curve = {}
    for u in ut:
        m = t == u
        d = w[m & (e == 1)].sum()
        if at_risk > 0:
            s *= 1 - d / at_risk
        curve[u] = s
        at_risk -= w[m].sum()
    keys = np.array(sorted(curve)); vals = np.array([curve[k] for k in keys])
    return np.array([vals[keys <= g][-1] if (keys <= g).any() else 1.0 for g in grid])


def fit(nhanes_dir: Path) -> dict:
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter
    raw = load(nhanes_dir)
    d = build(raw)
    grid = np.arange(35, 61)
    km = weighted_km(d.time.values, d.event.values, d.weight.values, grid)
    median = float(grid[np.argmax(km <= 0.5)]) if (km <= 0.5).any() else None
    res = {"source": "NHANES 2015-2018 RHQ/DEMO/BMX (real, public use)", "n": int(len(d)), "events": int(d.event.sum()),
           "groups": d.group.value_counts().to_dict(), "km_survival_by_age": dict(zip(grid.tolist(), np.round(km, 4).tolist())), "km_median_age": median}
    # parametric fits (unweighted, robust)
    dd = d[["time", "event"]].copy(); dd["time"] = dd.time.clip(lower=20)
    fits = {}
    for name, F in [("weibull", WeibullAFTFitter), ("loglogistic", LogLogisticAFTFitter)]:
        f = F().fit(dd, "time", "event")
        fits[name] = {"aic": float(f.AIC_), "params": {str(k): float(v) for k, v in f.params_.items()}, "median": float(f.predict_median(dd.iloc[:1]).iloc[0])}
    res["parametric"] = fits
    # covariates (complete case): menarche, parity, education, race
    cov = d.dropna(subset=["menarche", "parity", "education"]).copy()
    cov["parity_ge1"] = (cov.parity >= 1).astype(float); cov["parity_ge3"] = (cov.parity >= 3).astype(float)
    cov["menarche_c"] = cov.menarche - 13; cov["college"] = (cov.education >= 4).astype(float)
    cov["black"] = (cov.race == 4).astype(float); cov["hispanic"] = cov.race.isin([1, 2]).astype(float); cov["asian"] = (cov.race == 6).astype(float)
    f = LogLogisticAFTFitter().fit(cov[["time", "event", "parity_ge1", "parity_ge3", "menarche_c", "college", "black", "hispanic", "asian"]].assign(time=lambda z: z.time.clip(lower=20)), "time", "event")
    res["covariate_model_loglogistic_aft"] = {"n": int(len(cov)), "coef_log_time": {str(k): float(v) for k, v in f.params_.items()},
                                              "concordance": float(f.concordance_index_), "years_effect_at_median": {k[1]: float(np.expm1(v) * 51) for k, v in f.params_.items() if k[0] == "alpha_" and k[1] != "Intercept"}}
    return res, d
