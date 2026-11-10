"""Longitudinal generator of the menopausal transition for smart-ring users.

Per woman (parameters: params.MENOPAUSE + real NHANES survival fit):
  age at FMP      ~ inverse of real NHANES 2015-2018 weighted Kaplan-Meier (natural menopause),
                    shifted by covariates (college +1.79 y, parous +0.92 y from the same NHANES fit;
                    current smoking -1.2 y, Paramsothy 2017)
  STRAW+10 timing ~ total transition median 5.0 y, late transition (first >=60-day cycle -> FMP)
                    median 2.5 y (Paramsothy 2017 ranges 4.4-8.6 / 2.2-3.4); smokers transit faster (HR 1.87)
  bleeding        ~ stage-dependent cycle intervals; ovulatory share follows SWAN Daily Hormone Study
                    (100% at 10 y, 87.9% at 5 y, 22.8% in the final year before FMP; Santoro 2017)
  hormones        ~ FSH rise from 6.1 y, acceleration 2.05 y before FMP, plateau 2 y after (Randolph 2011);
                    AMH log-linear decline (calibrated in evaluation against Finkelstein 2020 Table 4)
  VMS             ~ four SWAN trajectory classes with published shares (Tepper 2016); nocturnal hot flash
                    raises HR ~4 bpm and reduces HF-HRV ~38% (de Zambotti 2013); effects on nightly aggregates
                    (temperature, awake time, efficiency) are ASSUMPTIONS (magnitudes not verified)
  ring nights     ~ cycle generator (sim_cycles) with ovulatory/anovulatory cycles + VMS + stage effects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sim_cycles import SIGNALS, apply_missingness, latent_vectors, render_signals, sample_person

STAGES = ["reproductive", "early_transition", "late_transition", "postmenopause"]
# AMH trajectory: log-linear in years-to-FMP; slope/spreads calibrated to Finkelstein 2020 Table 4 (generator_checks)
AMH = {"slope": 0.35, "person_sd": 0.7, "visit_sd": 0.6}  # grid-fitted: Table-4 MAE 0.040, r 0.94


class FMPAgeSampler:
    def __init__(self, nhanes_json: Path | dict):
        r = json.loads(Path(nhanes_json).read_text()) if not isinstance(nhanes_json, dict) else nhanes_json
        km = r["km_survival_by_age"]
        ages = np.array(sorted(float(a) for a in km)); s = np.array([km[str(int(a))] if str(int(a)) in km else km[a] for a in ages])
        # extend tail beyond 60 linearly to 0 at 63, and below 35 to 1
        self.ages = np.r_[30.0, ages, 63.0]; self.s = np.r_[1.0, s, 0.0]
        self.cov = {"college": 1.79, "parous": 0.92, "smoker": -1.22}

    def cdf_target(self):
        """NHANES (real) shape for ages >= 45; early tail rescaled to InterLACE (<40: 1.2%, 40-44: 4.7%),
        because NHANES self-report gives 13% before 45 vs 5.9% in 301k pooled women."""
        F = 1 - self.s
        f45 = float(np.interp(45, self.ages, F))
        tgt = np.where(self.ages >= 45, 0.059 + (F - f45) * (1 - 0.059) / (1 - f45),
                       np.interp(self.ages, [30, 40, 45], [0.0, 0.012, 0.059]))
        return np.maximum.accumulate(tgt)

    def sample(self, rng, college, parous, smoker):
        u = rng.random()
        a = float(np.interp(u, self.cdf_target(), self.ages)) + 0.3  # +0.3 y offsets truncation to last bleed
        a += rng.uniform(-0.5, 0.5)  # NHANES reports integer ages
        a += self.cov["college"] * (college - 0.35) + self.cov["parous"] * (parous - 0.80) + self.cov["smoker"] * (smoker - 0.13)
        return float(np.clip(a, 38.0, 62.0))


def vms_intensity(cls, tau):
    """Relative nightly hot-flash intensity by trajectory class; tau = years until FMP (negative after)."""
    sig = lambda x: 1 / (1 + np.exp(-x))
    if cls == "low":
        return 0.06 * np.ones_like(tau)
    if cls == "early_onset":
        return sig((11 - tau) / 1.5) * np.where(tau > 0, 1.0, np.exp(tau / 4.0))
    if cls == "late_onset":
        return sig((0.3 - tau) / 0.4) * np.exp(np.minimum(0, tau + 1) / 6.0)
    return sig((10 - tau) / 1.0) * np.where(tau > -10, 0.95, 0.4)


def ela(tau):
    return float(np.interp(tau, [0.0, 1.0, 5.0, 10.0], [0.20, 0.26, 0.879, 1.0]))


def sample_woman(rng, fmp_sampler, overrides=None):
    o = overrides or {}
    college = int(rng.random() < 0.35); parous = int(rng.random() < 0.80); smoker = int(rng.random() < 0.13)
    bmi = float(np.clip(np.exp(rng.normal(np.log(27.5), 0.2)), 17, 55))
    a_fmp = fmp_sampler.sample(rng, college, parous, smoker)
    d_total = float(np.exp(rng.normal(np.log(5.0), 0.45))) / (1.4 if smoker else 1.0)
    d_late = min(float(np.exp(rng.normal(np.log(2.5), 0.5))), 0.85 * d_total)
    w = dict(college=college, parous=parous, smoker=smoker, bmi=bmi, a_fmp_target=a_fmp, d_total=d_total, d_late=d_late,
             vms_class=str(rng.choice(["low", "early_onset", "late_onset", "high"], p=[0.270, 0.184, 0.290, 0.256])),
             vms_scale=float(rng.gamma(2.0, 0.6)), symptom_logger=bool(rng.random() < 0.4),
             amh_offset=float(rng.normal(0, 0.7)), fsh_noise=float(rng.uniform(0.35, 0.6)))
    # irregular share 6% (not 12%): calibrated so that the STRAW 'persistent >=7-day difference' criterion fires in
    # reproductive-age cycles about as often as in real FedCycleData women aged 30-40 (12-21%)
    w["cycle_person"] = sample_person(rng, age=40.0, irregular=bool(rng.random() < 0.06))
    w.update(o)
    return w


def bleeding_sequence(rng, w, age_start=34.0):
    # women whose FMP precedes age_start get no cycles (FMP age kept as sampled)
    """Cycle list from age_start to FMP. Each item: start_age, length (days), ovulation day or None, stage."""
    cp = w["cycle_person"]
    a_fmp = w["a_fmp_target"]; dt, dl = w["d_total"], w["d_late"]
    age = age_start - rng.uniform(0, 0.08)
    cycles = []
    first_late_done = False
    while True:
        tau = a_fmp - age
        stage = "reproductive" if tau > dt else ("early_transition" if tau > dl else "late_transition")
        ov = rng.random() < ela(tau)
        mu_f = cp["mu_f"] - 0.15 * max(0.0, age - 40)  # late-reproductive follicular shortening (assumption)
        if stage == "reproductive":
            if ov:
                f = int(np.clip(np.round(rng.normal(mu_f, cp["sd_f"])), 7, 40)); L = f + int(np.clip(np.round(rng.normal(cp["mu_l"], cp["sd_l"])), 7, 17))
            else:
                L = int(np.clip(np.round(rng.normal(mu_f + cp["mu_l"] + 3, 5)), 21, 45)); f = None
        elif stage == "early_transition":
            # STRAW+10 defines this stage by OBSERVABLE variability (persistent >=7-day differences within 10 cycles),
            # so follicular variability is raised to >=4.5 d from onset (Paramsothy 2017 onset definition)
            if ov:
                f = int(np.clip(np.round(rng.normal(mu_f - 1.0, max(cp["sd_f"] + 1.5, 4.5))), 7, 40)); L = f + int(np.clip(np.round(rng.normal(cp["mu_l"] - 0.5, cp["sd_l"] + 0.5)), 6, 17))
            else:
                f = None
                L = int(np.clip(np.round(rng.normal(44, 8)), 32, 59)) if rng.random() < 0.65 else int(np.clip(np.round(rng.normal(21, 3)), 15, 25))
        else:
            if not first_late_done:
                L = int(np.clip(np.round(np.exp(rng.normal(np.log(80), 0.35))), 60, 250)); f = None; first_late_done = True
            elif ov:
                f = int(np.clip(np.round(rng.normal(mu_f, cp["sd_f"] + 2)), 7, 45)); L = f + int(np.clip(np.round(rng.normal(cp["mu_l"] - 1, cp["sd_l"] + 0.7)), 5, 16))
            else:
                f = None
                L = int(np.clip(np.round(np.exp(rng.normal(np.log(55), 0.5))), 18, 300))
        if age + L / 365.25 > a_fmp:
            break
        cycles.append({"start_age": age, "length": L, "ovulation": f, "stage": stage, "menses": int(np.clip(np.round(rng.normal(cp["menses_len"], 1)), 2, 9))})
        age += L / 365.25
    fmp_age = cycles[-1]["start_age"] if cycles else a_fmp
    return cycles, float(fmp_age)


def stage_at(w, fmp_age, age):
    tau = fmp_age - age
    if tau <= 0:
        return "postmenopause"
    if tau <= w["d_late"]:
        return "late_transition"
    if tau <= w["d_total"]:
        return "early_transition"
    return "reproductive"


def hormones(rng, w, tau):
    """FSH (IU/L) and AMH (pg/mL) at tau years before FMP (piecewise log-linear trajectories)."""
    ln_fsh = np.interp(-tau, [-12, -6.1, -2.05, 0, 2.0, 10], np.log([6.5, 7.0, 14, 45, 75, 75]))
    fsh = float(np.exp(ln_fsh + rng.normal(0, w["fsh_noise"] * (1.4 if 0 < tau < 4 else 0.6))))
    ln_amh = np.log(142) + AMH["slope"] * (tau - 4.75) + w["amh_offset"] * AMH["person_sd"] / 0.7 + rng.normal(0, AMH["visit_sd"])
    amh = float(np.clip(np.exp(ln_amh), 1.0, 8000.0))
    if tau <= 0:
        amh = float(np.clip(np.exp(np.log(2.0) + rng.normal(0, 0.4)), 1.0, 10))
    return fsh, amh


def window(rng, w, cycles, fmp_age, landmark_age, n_days):
    """Simulate n_days of ring nights ending at landmark_age (inclusive) + logged periods + symptoms."""
    cp = dict(w["cycle_person"])
    cp["age"] = landmark_age
    cp["lnhrv_base"] = cp["lnhrv_base"] - 0.025 * (landmark_age - 40)
    end_day = landmark_age * 365.25
    start_day = end_day - n_days + 1
    starts = [c["start_age"] * 365.25 for c in cycles]
    # cycles overlapping window (+ one before to anchor)
    sel = [i for i, c in enumerate(cycles) if starts[i] + c["length"] > start_day - 40 and starts[i] <= end_day]
    wcycles = []
    if sel:
        for i in sel:
            wcycles.append({"length": cycles[i]["length"], "ovulation": cycles[i]["ovulation"], "menses": cycles[i]["menses"]})
        t0 = starts[sel[0]]
    else:
        t0 = start_day
    # post-FMP (or no cycle overlapping): pad with an anovulatory 'cycle' until window end
    covered = t0 + sum(c["length"] for c in wcycles)
    if covered < end_day + 1:
        wcycles.append({"length": int(np.ceil(end_day + 1 - covered)) + 1, "ovulation": None, "menses": 0})
    lat = latent_vectors(cp, wcycles)
    n_all = len(lat["prog"])
    day_abs = t0 + np.arange(n_all)
    keep = (day_abs >= start_day) & (day_abs <= end_day)
    ages = day_abs / 365.25
    tau = fmp_age - ages
    vms = vms_intensity(w["vms_class"], tau) * w["vms_scale"]
    flashes = rng.poisson(np.clip(vms, 0, 6))
    late_or_post = (tau <= w["d_late"]).astype(float)
    post = (tau <= 0).astype(float)
    extra = {"temp": 0.05 * flashes + rng.normal(0, 0.02, n_all) * flashes, "hr": 0.3 * flashes + 0.8 * post, "lnhrv": -0.035 * flashes - 0.12 * late_or_post,
             "eff": -1.6 * flashes, "awake": flashes * rng.uniform(3, 12, n_all), "sleep": -17 * post - 4 * flashes}
    clean, conf = render_signals(rng, cp, lat, extra)
    obs, worn = apply_missingness(rng, cp, clean)
    obs = obs[keep].reset_index(drop=True); flashes_w = flashes[keep]
    dates = pd.date_range("2030-01-01", periods=len(obs), freq="D")
    obs.insert(0, "date", dates)
    # logged period starts (within log history), with imperfect logging
    log_hist_days = int(rng.choice([90, 180, 365, 730], p=[0.15, 0.25, 0.35, 0.25]))
    rel = float(rng.choice([1.0, 0.9, 0.75], p=[0.6, 0.25, 0.15]))
    logged = []
    for s in starts:
        if end_day - log_hist_days <= s <= end_day and rng.random() < rel:
            logged.append(dates[0] + pd.Timedelta(days=int(round(s - start_day))))
    symptoms = []
    if w["symptom_logger"]:
        for i in np.flatnonzero(flashes_w > 0):
            if rng.random() < 0.7 and i >= len(dates) - 30:
                symptoms.append(dates[i])
        day_flash = rng.poisson(np.clip(vms[keep] * 1.5, 0, 10))
        for i in np.flatnonzero(day_flash > 0):
            if rng.random() < 0.5 and i >= len(dates) - 30 and dates[i] not in symptoms:
                symptoms.append(dates[i])
    return {"daily": obs, "logged_starts": sorted(logged), "log_history_days": log_hist_days, "log_reliability": rel,
            "symptom_dates": sorted(symptoms), "symptom_logger": w["symptom_logger"], "flashes": flashes_w, "as_of": dates[-1],
            "true_starts_in_window": [dates[0] + pd.Timedelta(days=int(round(s - start_day))) for s in starts if start_day <= s <= end_day]}


def simulate_landmarks(rng, w, k, lo=38.0, hi=62.0):
    """Landmark ages drawn from one fixed distribution for every woman (population cross-section);
    drawing them relative to her own FMP would bias P(stage | age)."""
    cycles, fmp_age = bleeding_sequence(rng, w)
    out = []
    for _ in range(k):
        L = float(rng.uniform(lo, hi))
        n_days = int(rng.choice([60, 120, 180, 270], p=[0.2, 0.3, 0.3, 0.2]))
        win = window(rng, w, cycles, fmp_age, L, n_days)
        tau = fmp_age - L
        last_true_start = max([c["start_age"] for c in cycles if c["start_age"] <= L] or [np.nan])
        fsh, amh = hormones(rng, w, tau)
        out.append({"landmark_age": L, "fmp_age": fmp_age, "years_to_fmp": tau, "stage": stage_at(w, fmp_age, L),
                    "months_since_last_true_period": (L - last_true_start) * 12 if np.isfinite(last_true_start) else np.nan,
                    "fsh": fsh, "amh": amh, "window": win, "woman": {k_: w[k_] for k_ in ["college", "parous", "smoker", "bmi", "vms_class", "d_total", "d_late"]}})
    return out


FINKELSTEIN_T4 = {  # P(FMP <=12 mo) by AMH band (pg/mL) and age group, Finkelstein 2020 JCEM Table 4
    "<48": [((0, 10), 0.51), ((10, 25), 0.22), ((25, 50), 0.12), ((50, 100), 0.09), ((100, 200), 0.05), ((200, 400), 0.02), ((400, 1e9), 0.02)],
    "48-51": [((0, 5), 0.67), ((5, 10), 0.43), ((10, 25), 0.40), ((25, 50), 0.23), ((50, 100), 0.19), ((100, 200), 0.10), ((200, 1e9), 0.01)],
    ">=51": [((0, 2), 0.82), ((2, 5), 0.67), ((5, 10), 0.72), ((10, 25), 0.53), ((25, 50), 0.35), ((50, 100), 0.23), ((100, 1e9), 0.10)],
}


def generator_checks(fmp_sampler, n=6000, seed=4242):
    rng = np.random.default_rng(seed)
    fm, ela_rows, amh_rows = [], [], []
    for _ in range(n):
        w = sample_woman(rng, fmp_sampler)
        cyc, fmp = bleeding_sequence(rng, w)
        fm.append(fmp)
        for c in cyc:
            tau = fmp - c["start_age"]
            if tau <= 12:
                ela_rows.append((tau, c["ovulation"] is not None))
        # SWAN-like visit: age 42-53 while still menstruating, AMH measured
        L = rng.uniform(42, 53)
        if L < fmp and any(c["start_age"] <= L for c in cyc):
            _, amh = hormones(rng, w, fmp - L)
            amh_rows.append((L, amh, (fmp - L) <= 1.0))
    fm = np.array(fm)
    ages = np.arange(38, 63)
    cdf = [float((fm <= a).mean()) for a in ages]
    e = pd.DataFrame(ela_rows, columns=["tau", "ov"])
    bins = [0, 1, 2, 3, 5, 7, 10, 12]
    e["b"] = pd.cut(e.tau, bins, include_lowest=True)
    eg = e.groupby("b", observed=True).ov.mean()
    a = pd.DataFrame(amh_rows, columns=["age", "amh", "fmp12"])
    comp = []
    for grp, rows in FINKELSTEIN_T4.items():
        sub = a[(a.age < 48)] if grp == "<48" else (a[(a.age >= 48) & (a.age < 51)] if grp == "48-51" else a[a.age >= 51])
        for (lo, hi), pub in rows:
            s = sub[(sub.amh >= lo) & (sub.amh < hi)]
            comp.append({"age_group": grp, "amh_band": f"{lo}-{hi if hi < 1e8 else 'inf'}", "published": pub, "generator": float(s.fmp12.mean()) if len(s) >= 20 else None, "n": int(len(s))})
    comp = pd.DataFrame(comp)
    ok = comp.dropna()
    return {"fmp_age": {"median": float(np.median(fm)), "mean": float(fm.mean()), "sd": float(fm.std()), "lt40": float((fm < 40).mean()), "lt45": float((fm < 45).mean()), "ge55": float((fm >= 55).mean())},
            "fmp_cdf": {"ages": ages.tolist(), "cdf": cdf},
            "ela": {"years": [float((iv.left + iv.right) / 2) for iv in eg.index], "generator": eg.round(3).tolist(), "published_points": {"10": 1.0, "5": 0.879, "<1": 0.228}},
            "finkelstein_table4": comp.to_dict(orient="records"),
            "finkelstein_mae": float((ok.published - ok.generator).abs().mean()), "finkelstein_corr": float(np.corrcoef(ok.published, ok.generator)[0, 1])}
