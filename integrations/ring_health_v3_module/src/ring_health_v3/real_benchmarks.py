"""Real-data benchmarks (no synthetic data involved).

1) TempID (Zenodo 20160611, CC-BY-4.0): skin-wearable hourly temperature, 59 cycle series with reference
   ovulation. Night value = mean of 21:00-03:59 hours assigned to the morning date (authors' window, >=6 hours).
   Temperature-only path of the cycle engine with a FLAT prior over ovulation day inside the series
   (no period dates are available). Retrospective estimate at series end vs reference.
   Caveat: authors centred/aligned and bidirectionally imputed the series upstream.
2) FedCycleData (Marquette, real): prospective calendar prediction. For each woman, cycles in order;
   predict cycle n's ovulation day and length from cycles < n using the same empirical-Bayes personalisation
   as the engine (population priors are re-estimated leaving the woman out). Reports MAE and +-2 d accuracy,
   and fertile-window day classification of the calendar-only prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .cycle_engine import CycleModel, discrete_normal, max3


def tempid_nights(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv, parse_dates=["measurement_timestamp", "ovulation_timestamp"])
    d["hour"] = d.measurement_timestamp.dt.hour
    night = d[(d.hour >= 21) | (d.hour <= 3)].copy()
    night["morning"] = (night.measurement_timestamp + pd.to_timedelta(np.where(night.hour >= 21, 1, 0), unit="D")).dt.normalize()
    g = night.groupby(["subject_id", "morning"]).agg(temp=("temperature_celsius", "mean"), n=("temperature_celsius", "size"), ov=("ovulation_timestamp", "first")).reset_index()
    g = g[g.n >= 6]
    return g


def tempid_benchmark(csv: Path, params: dict, amplitude_scale=1.0):
    nights = tempid_nights(csv)
    model = CycleModel(params)
    rows = []
    for sid, g in nights.groupby("subject_id"):
        g = g.sort_values("morning").reset_index(drop=True)
        dates = pd.date_range(g.morning.min(), g.morning.max(), freq="D")
        fr = pd.DataFrame({"date": dates}).merge(g[["morning", "temp"]].rename(columns={"morning": "date"}), on="date", how="left")
        fr = fr.rename(columns={"temp": "temperature_deviation"})
        for c in ["resting_hr", "lnhrv", "respiratory_rate"]:
            fr[c] = np.nan
        fr["cycle_day"] = np.arange(1, len(fr) + 1); fr["_ill"] = False
        y = fr.temperature_deviation
        noise = model.person_noise(fr.assign(hrv_ms=np.nan))
        grid = np.arange(4, len(fr) - 3)
        prior = np.ones(len(grid)) / len(grid)
        per = model.population_personal(30)
        m, vv = per["amp"]["temperature_deviation"]
        per["amp"]["temperature_deviation"] = (m * amplitude_scale, vv * amplitude_scale ** 2)
        post = model.cycle_posterior(fr, grid, prior, 0.10, per, noise, cycle_len=None, today_day=None, baselines={"temperature_deviation": float(y.median()), "resting_hr": None, "lnhrv": None, "respiratory_rate": None})
        po = post["p_o"] / post["p_o"].sum()
        est_day = int(grid[np.argmax(po)])
        true_day = int((g.ov.iloc[0].normalize() - dates[0]).days) + 1
        rows.append({"series": sid, "person": sid.split("_")[0], "nights": int(y.notna().sum()), "est_day": est_day, "true_day": true_day,
                     "err": est_day - true_day, "p_ovulatory": 1 - post["p_anov"], "max3": max3(po)})
    r = pd.DataFrame(rows)
    det = (r.p_ovulatory >= 0.8) & (r.max3 >= 0.6)
    return r, {"series": int(len(r)), "people_by_prefix": int(r.person.nunique()), "mae_all": float(r.err.abs().mean()), "median_ae_all": float(r.err.abs().median()),
               "within1_all": float((r.err.abs() <= 1).mean()), "within2_all": float((r.err.abs() <= 2).mean()),
               "detection_rate": float(det.mean()), "mae_detected": float(r.err[det].abs().mean()) if det.any() else None,
               "within2_detected": float((r.err[det].abs() <= 2).mean()) if det.any() else None}


def fedcycle_calendar(csv: Path, params: dict):
    d = pd.read_csv(csv)
    for c in ["LengthofCycle", "EstimatedDayofOvulation", "CycleNumber", "CycleWithPeakorNot", "Age"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Age"] = d.groupby("ClientID").Age.transform("first")
    d = d.sort_values(["ClientID", "CycleNumber"])
    from .train_cycle import real_priors
    raw = pd.read_csv(csv)
    rows = []
    for cid, g in d.groupby("ClientID"):
        # leave-one-woman-out population priors (no information from this woman)
        pri = real_priors(raw, exclude_client=cid)
        pp = json.loads(json.dumps(params))
        for k_src, k_dst in [("follicular_mean", "follicular_mean"), ("follicular_between_sd", "follicular_between_sd"), ("follicular_within_sd", "follicular_within_sd"),
                             ("follicular_age_slope", "follicular_age_slope"), ("luteal_between_sd", "luteal_between_sd")]:
            pp["priors"][k_dst] = pri[k_src]
        model = CycleModel(pp)
        g = g.reset_index(drop=True)
        hist = []
        for i, r in g.iterrows():
            age = r.Age if np.isfinite(r.Age) else 30
            per = model.update_personal(age, hist)
            if i >= 1 and np.isfinite(r.EstimatedDayofOvulation) and r.CycleWithPeakorNot == 1:
                O = int(r.EstimatedDayofOvulation); L = int(r.LengthofCycle)
                grid = np.arange(6, 60)
                pr = discrete_normal(grid, per["foll"]["mean"], per["foll"]["sd"]); pr /= pr.sum()
                pred_o = int(grid[np.argmax(pr)])
                pred_len = per["foll"]["mean"] + per["luteal"]["mean"]
                cs = np.r_[0, np.cumsum(pr)]
                days = np.arange(1, L + 1)
                pf = np.array([cs[np.searchsorted(grid, dd + 5, side="right")] - cs[np.searchsorted(grid, dd)] for dd in days])
                truth = (days >= O - 5) & (days <= O)
                rows.append({"client": cid, "n_hist": len(hist), "err_o": pred_o - O, "err_len": pred_len - L, "pf": pf, "truth": truth})
            # update history with the observed (true) ovulation day as a near-certain observation
            if np.isfinite(r.EstimatedDayofOvulation) and r.CycleWithPeakorNot == 1 and 18 <= r.LengthofCycle <= 60:
                hist.append({"E_o": float(r.EstimatedDayofOvulation), "V_o": 1.0, "p_ovul": 1.0, "length": int(r.LengthofCycle), "amp_est": {}})
            elif 18 <= r.LengthofCycle <= 60:
                hist.append({"E_o": float(r.LengthofCycle - 13), "V_o": 9.0, "p_ovul": 0.2, "length": int(r.LengthofCycle), "amp_est": {}})
    r = pd.DataFrame(rows)
    from sklearn.metrics import roc_auc_score
    pf = np.concatenate(r.pf.values); tr = np.concatenate(r.truth.values)
    out = {"cycles_scored": int(len(r)), "women": int(r.client.nunique()),
           "ovulation_day_mae": float(r.err_o.abs().mean()), "ovulation_within2": float((r.err_o.abs() <= 2).mean()),
           "cycle_length_mae": float(r.err_len.abs().mean()), "cycle_length_within2": float((r.err_len.abs() <= 2).mean()),
           "calendar_fertile_day_auroc": float(roc_auc_score(tr, pf)),
           "calendar_fertile_sensitivity_at_0.30": float((pf[tr] >= 0.30).mean()), "calendar_fertile_specificity_at_0.30": float((pf[~tr] < 0.30).mean())}
    by = r.assign(h=np.minimum(r.n_hist, 6)).groupby("h").agg(ov_mae=("err_o", lambda x: x.abs().mean()), len_mae=("err_len", lambda x: x.abs().mean()), n=("err_o", "size"))
    out["by_history_cycles"] = by.round(3).reset_index().to_dict(orient="records")
    return r, out


def _series_frames(nights):
    out = {}
    for sid, g in nights.groupby("subject_id"):
        g = g.sort_values("morning").reset_index(drop=True)
        dates = pd.date_range(g.morning.min(), g.morning.max(), freq="D")
        fr = pd.DataFrame({"date": dates}).merge(g[["morning", "temp"]].rename(columns={"morning": "date"}), on="date", how="left").rename(columns={"temp": "temperature_deviation"})
        for c in ["resting_hr", "lnhrv", "respiratory_rate"]:
            fr[c] = np.nan
        fr["cycle_day"] = np.arange(1, len(fr) + 1); fr["_ill"] = False
        out[sid] = (fr, int((g.ov.iloc[0].normalize() - dates[0]).days) + 1)
    return out


def _fit_device(model, frames, ids, lag):
    """Amplitude mean/var and noise SD for a new device from training series with reference ovulation."""
    g = model.templates["temperature_deviation"]
    amps, sds = [], []
    from .cycle_engine import K_MIN
    for sid in ids:
        fr, O = frames[sid]
        y = fr.temperature_deviation.to_numpy(); m = np.isfinite(y)
        k = fr.cycle_day.to_numpy()[m] - (O + lag)
        x = g[np.clip(k - K_MIN, 0, len(g) - 1)]; yc = y[m] - y[m].mean(); xc = x - x.mean()
        a = float(xc @ yc / (xc @ xc)); amps.append(a); sds.append(np.std(yc - a * xc))
    amps = np.array(amps)
    return float(amps.mean()), float(max(amps.var(), 1e-4)), float(np.median(sds))


def tempid_device_cv(csv: Path, params: dict, folds=3, seed=0):
    """Grouped (by person prefix) CV: fit device profile (amplitude, noise, rise lag) on train folds."""
    nights = tempid_nights(csv); frames = _series_frames(nights)
    sids = sorted(frames); persons = sorted({s.split("_")[0] for s in sids})
    rng = np.random.default_rng(seed); rng.shuffle(persons)
    fold_of = {p: i % folds for i, p in enumerate(persons)}
    model = CycleModel(params)
    rows = []
    for f in range(folds):
        tr = [s for s in sids if fold_of[s.split("_")[0]] != f]; te = [s for s in sids if fold_of[s.split("_")[0]] == f]
        best = None
        for lag in [-2, -1, 0, 1, 2, 3]:
            am, av, sd = _fit_device(model, frames, tr, lag)
            # choose lag by training-fold likelihood of the reference day
            ll = 0.0
            for sid in tr:
                fr, O = frames[sid]
                post = _post_flat(model, fr, am, av, sd, lag)
                grid, po = post
                ll += np.log(max(po[grid == O].sum(), 1e-6))
            if best is None or ll > best[0]:
                best = (ll, lag, am, av, sd)
        _, lag, am, av, sd = best
        for sid in te:
            fr, O = frames[sid]
            grid, po = _post_flat(model, fr, am, av, sd, lag)
            est = int(grid[np.argmax(po)])
            rows.append({"series": sid, "fold": f, "lag": lag, "amp_mean": am, "noise_sd": sd, "est_day": est, "true_day": O, "err": est - O,
                         "max3": max3(po)})
    r = pd.DataFrame(rows)
    return r, {"series": int(len(r)), "people_by_prefix": len(persons), "folds": folds, "mae": float(r.err.abs().mean()), "median_ae": float(r.err.abs().median()),
               "within1": float((r.err.abs() <= 1).mean()), "within2": float((r.err.abs() <= 2).mean()), "within3": float((r.err.abs() <= 3).mean()),
               "fitted_amplitude_mean_c": float(r.amp_mean.mean()), "fitted_noise_sd_c": float(r.noise_sd.mean()), "fitted_lag_days": r.lag.value_counts().to_dict()}


def _post_flat(model, fr, am, av, sd, lag):
    y = fr.temperature_deviation
    grid = np.arange(4, len(fr) - 2)
    per = model.population_personal(30)
    per["amp"]["temperature_deviation"] = (am, av)
    noise = {"temperature_deviation": {"sd": sd, "rho": model.p["noise"]["temperature_deviation"]["rho"]},
             "resting_hr": model.p["noise"]["resting_hr"], "lnhrv": model.p["noise"]["lnhrv"], "respiratory_rate": model.p["noise"]["respiratory_rate"]}
    g2 = grid + lag  # template origin = reference day + lag (same sign as _fit_device)
    post = model.cycle_posterior(fr, g2, np.ones(len(grid)) / len(grid), 1e-6, per, noise,
                                 baselines={"temperature_deviation": float(y.median()), "resting_hr": None, "lnhrv": None, "respiratory_rate": None})
    po = post["p_o"] / post["p_o"].sum()
    return grid, po
