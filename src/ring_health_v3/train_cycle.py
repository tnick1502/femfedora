"""Fit the cycle engine: templates, amplitude/noise hyper-parameters (synthetic train people),
phase-length priors (real FedCycleData), tempering and decision thresholds (synthetic validation people)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cycle_engine import K_MAX, K_MIN, MODEL_SIGNALS, to_model_frame
from .sim_cycles import simulate_person

N_END = 5


def make_people(n, seed, n_cycles=13, **kw):
    rng = np.random.default_rng(seed)
    return [simulate_person(rng, n_cycles=n_cycles, **kw) for _ in range(n)]


def real_priors(fedcycle_csv, exclude_client=None) -> dict:
    d = pd.read_csv(fedcycle_csv) if not isinstance(fedcycle_csv, pd.DataFrame) else fedcycle_csv.copy()
    if exclude_client is not None:
        d = d[d.ClientID != exclude_client]
    for c in ["LengthofCycle", "EstimatedDayofOvulation", "LengthofLutealPhase", "Age", "CycleWithPeakorNot", "LengthofMenses"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Age"] = d.groupby("ClientID").Age.transform("first")
    v = d[(d.CycleWithPeakorNot == 1) & d.EstimatedDayofOvulation.notna() & d.LengthofLutealPhase.between(5, 25)].copy()

    def comp(x):
        g = v.groupby("ClientID")[x]; n = g.size(); keep = n >= 3
        w = g.var(ddof=1)[keep]; m = g.mean()[keep]
        within_rms = float(np.sqrt(np.nanmean(w)))
        between = float(np.sqrt(max(m.var() - np.nanmean(w / n[keep]), 0)))
        return float(v[x].mean()), between, within_rms, float(np.sqrt(np.nanmedian(w)))
    fm, fb, fw, fwm = comp("EstimatedDayofOvulation")
    lm, lb, lw, lwm = comp("LengthofLutealPhase")
    va = v.dropna(subset=["Age"])
    slope = float(np.polyfit(va.Age - 30, va.EstimatedDayofOvulation, 1)[0])
    return {"source": "FedCycleData (Marquette), real; %d cycles / %d women" % (len(v), v.ClientID.nunique()),
            "follicular_mean": round(fm, 2), "follicular_between_sd": round(fb, 2), "follicular_within_sd": round(fw, 2), "follicular_within_sd_median": round(fwm, 2),
            "follicular_age_slope": round(slope, 3), "luteal_mean_fedcycle": round(lm, 2), "luteal_between_sd": round(lb, 2), "luteal_within_sd_rms": round(lw, 2),
            "luteal_within_sd_median": round(lwm, 2), "no_peak_fraction": round(float((d.CycleWithPeakorNot == 0).mean()), 3)}


def fit_templates(people):
    """Least squares with cycle fixed effects: y = mu_c + beta_k + gamma_j + e."""
    nK = K_MAX - K_MIN + 1
    P = nK + N_END
    out = {}
    for s in MODEL_SIGNALS:
        XtX = np.zeros((P, P)); Xty = np.zeros(P)
        for sim in people:
            f = to_model_frame(sim["daily"]); t = sim["truth"]
            y_all = f[s].to_numpy(float)
            ok = np.isfinite(y_all) & ~t.fever.to_numpy()
            for ci, g in t.groupby("cycle_index"):
                if not g.ovulatory.iloc[0]:
                    continue
                idx = g.index.to_numpy(); m = ok[idx]
                if m.sum() < 8:
                    continue
                cd = g.cycle_day.to_numpy()[m]; O = g.ovulation_day.iloc[0]; L = g.cycle_length.iloc[0]
                k = np.clip(cd - O, K_MIN, K_MAX).astype(int) - K_MIN
                j = (L + 1 - cd)
                X = np.zeros((m.sum(), P)); X[np.arange(m.sum()), k] = 1
                je = (j >= 1) & (j <= N_END); X[np.flatnonzero(je), nK + j[je] - 1] = 1
                y = y_all[idx][m]
                X = X - X.mean(0); y = y - y.mean()
                XtX += X.T @ X; Xty += X.T @ y
        beta = np.linalg.lstsq(XtX + 1e-6 * np.eye(P), Xty, rcond=None)[0]
        bk = beta[:nK]; gj = beta[nK:]
        kk = np.arange(K_MIN, K_MAX + 1)
        base = bk[(kk >= -15) & (kk <= -9)].mean(); plateau = bk[(kk >= 5) & (kk <= 9)].mean()
        amp = plateau - base
        g = (bk - base) / amp
        # light smoothing, keep endpoints
        g = np.convolve(np.r_[g[0], g, g[-1]], [0.25, 0.5, 0.25], mode="valid")
        e = np.clip(1 + gj / amp, -0.5, 1.5)
        out[s] = {"g": g.round(4).tolist(), "end": e.round(4).tolist(), "amplitude_mean": float(amp)}
    return out


def fit_amplitude_noise(people, templates):
    from .cycle_engine import CycleModel  # noqa
    res = {}
    kk_off = -K_MIN
    for s in MODEL_SIGNALS:
        g = np.asarray(templates[s]["g"]); e = np.asarray(templates[s]["end"])
        rows = []; sds = []; rhos = []
        for pi, sim in enumerate(people):
            f = to_model_frame(sim["daily"]); t = sim["truth"]
            y_all = f[s].to_numpy(float); ok = np.isfinite(y_all) & ~t.fever.to_numpy()
            resid_all = np.full(len(y_all), np.nan)
            for ci, gr in t.groupby("cycle_index"):
                idx = gr.index.to_numpy(); m = ok[idx]
                if m.sum() < 8:
                    continue
                y = y_all[idx][m]; cd = gr.cycle_day.to_numpy()[m]
                if gr.ovulatory.iloc[0]:
                    O = gr.ovulation_day.iloc[0]; L = gr.cycle_length.iloc[0]
                    x = g[np.clip(cd - O + kk_off, 0, len(g) - 1).astype(int)]
                    j = L + 1 - cd; ee = np.ones(len(cd)); mm = (j >= 1) & (j <= len(e)); ee[mm] = e[j[mm] - 1]; x = x * ee
                    xc = x - x.mean(); yc = y - y.mean()
                    a = float(xc @ yc / (xc @ xc)); r = yc - a * xc
                    se2 = float(r.var() / (xc @ xc))
                    rows.append((pi, a, se2)); resid_all[idx[m]] = r
                else:
                    resid_all[idx[m]] = y - y.mean()
            rr = resid_all[np.isfinite(resid_all)]
            if len(rr) > 30:
                sds.append(np.std(rr))
                x0 = resid_all[:-1]; x1 = resid_all[1:]; mm = np.isfinite(x0) & np.isfinite(x1)
                if mm.sum() > 30:
                    rhos.append(np.corrcoef(x0[mm], x1[mm])[0, 1])
        A = pd.DataFrame(rows, columns=["person", "a", "se2"])
        total = max(A.a.var() - A.se2.mean(), 1e-6)
        pm = A.groupby("person").agg(a=("a", "mean"), n=("a", "size"))
        within = max(float(A.groupby("person").a.var().mean() - A.se2.mean()), 0.1 * total)
        between = max(float(pm.a.var() - within / pm.n.mean()), 0.1 * total)
        res[s] = {"amplitude": {"mean": float(A.a.mean()), "between_var": between, "within_var": within},
                  "noise": {"sd": float(np.median(sds)), "rho": float(np.clip(np.median(rhos), 0.05, 0.8)), "sd_floor": float(np.quantile(sds, 0.03))}}
    return res


def build_params(train_people, priors_real, anov_by_age):
    templates = fit_templates(train_people)
    an = fit_amplitude_noise(train_people, templates)
    params = {
        "version": "cycle-engine-v3",
        "templates": {s: {"g": templates[s]["g"], "end": templates[s]["end"]} for s in MODEL_SIGNALS},
        "amplitude": {s: an[s]["amplitude"] for s in MODEL_SIGNALS},
        "noise": {s: an[s]["noise"] for s in MODEL_SIGNALS},
        "offset_sd": {"temperature_deviation": 0.2, "resting_hr": 3.0, "lnhrv": 0.25, "respiratory_rate": 0.8},
        "priors": {"follicular_mean": priors_real["follicular_mean"], "follicular_between_sd": priors_real["follicular_between_sd"],
                   "follicular_within_sd": priors_real["follicular_within_sd"], "follicular_age_slope": priors_real["follicular_age_slope"],
                   "luteal_mean": 12.8, "luteal_between_sd": priors_real["luteal_between_sd"], "luteal_within_sd": 1.3,
                   "anov_by_age": anov_by_age, "anov_prior_strength": 20.0, "within_sd_pseudo_cycles": 3.0, "real_prior_source": priors_real},
        "lh": {"positive_prob_by_offset": {"-3": 0.10, "-2": 0.40, "-1": 0.90, "0": 0.50, "1": 0.10}, "false_positive": 0.02, "anovulatory_positive": 0.04},
        "tempering": 1.0,
        "thresholds": {},
    }
    return params


def save(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1))
