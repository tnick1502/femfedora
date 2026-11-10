"""Day-by-day evaluation of the deployed fertility analysis on held-out synthetic women.

Each woman: 14 simulated cycles. Prospective evaluation on cycles 12 and 13 (index 11, 12): for every
morning the analysis sees only data and period logs up to that date. Retrospective ovulation
detection: all completed cycles in the history at the end of cycle 13. Truth never enters inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cycle_engine import CycleModel
from .fertility import analyze_fertility
from .sim_cycles import CORE, SIGNALS, sample_person, simulate_person

SCENARIOS = ["realistic", "missing30", "missing50", "block7_ovulation", "temperature_only", "no_temperature",
             "cold_start", "irregular", "age40_45", "lh_users", "missed_logs", "calendar_only", "mismatch"]

# Deliberately mismatched physiology (engine keeps its trained parameters): weaker/later temperature rise, noisier,
# weaker HR/HRV/RR effects, shorter luteal phase, doubled anovulation -> robustness to generator misspecification
MISMATCH = {"temp_amp": 0.7, "temp_sd": 1.3, "hr_amp": 0.6, "lnhrv_amp": 0.5, "rr_amp": 0.5, "anov_rate": 2.0}


def scenario_person(rng, scenario):
    if scenario == "mismatch":
        p = sample_person(rng)
        for k, f in MISMATCH.items():
            p[k] = p[k] * f
        p["mu_l"] = max(8.0, p["mu_l"] - 1.0); p["rise_mid"] = p["rise_mid"] + 0.7
        return p
    kw = {}
    if scenario == "irregular":
        kw["irregular"] = True
    if scenario == "age40_45":
        kw["age"] = float(rng.uniform(40, 45))
    over = {}
    if scenario == "lh_users":
        over["lh_user"] = True
    if scenario == "missed_logs":
        over["log_reliability"] = 0.75
    return sample_person(rng, overrides=over, **kw)


def apply_scenario(rng, sim, scenario, eval_cycles):
    d = sim["daily"].copy(); t = sim["truth"]
    n = len(d)
    if scenario in ("missing30", "missing50"):
        rate = 0.30 if scenario == "missing30" else 0.50
        d.loc[rng.random(n) < rate, SIGNALS] = np.nan
    if scenario == "block7_ovulation":
        for ci in eval_cycles:
            g = t[t.cycle_index == ci]
            if g.ovulatory.iloc[0]:
                c = g.index[0] + int(g.ovulation_day.iloc[0]) - 1
                d.loc[max(0, c - 3):c + 3, SIGNALS] = np.nan
    if scenario == "temperature_only":
        d[["resting_hr", "hrv_ms", "respiratory_rate"]] = np.nan
    if scenario == "no_temperature":
        d["temperature_deviation"] = np.nan
    if scenario == "calendar_only":
        d[SIGNALS] = np.nan
    return d


def run_person(args):
    pid, seed, scenario, params, thresholds = args[:5]
    anov = args[5] if len(args) > 5 else None
    rng = np.random.default_rng(seed)
    person = scenario_person(rng, scenario)
    if anov is not None:
        person["anov_rate"] = anov
    sim = simulate_person(rng, n_cycles=14, person=person)
    model = CycleModel(params)
    t = sim["truth"]
    eval_cycles = [11, 12]
    daily = apply_scenario(rng, sim, scenario, eval_cycles)
    starts_true = sim["true_starts"]
    logged = sim["logged_starts"]
    visible = sim["logged_visible"]          # (logged date, date from which the log can exist)
    lh = sim["lh"]
    age = person["age"]
    rel = 0.97 if person["log_reliability"] >= 1.0 else 0.9   # as the API: period_logs_complete=True -> 0.97
    rows = []
    for ci in eval_cycles:
        g = t[t.cycle_index == ci]
        c_start = g.date.iloc[0]
        nxt = starts_true[ci + 1]
        O = g.ovulation_day.iloc[0]; true_ov_date = c_start + pd.Timedelta(days=int(O) - 1) if np.isfinite(O) else None
        if scenario == "cold_start":
            # new user: ring worn from 0-10 days before this cycle; one anchor = logged start or onboarding answer with recall error
            cold_lo = max(0, int(np.searchsorted(daily.date.values, (c_start - pd.Timedelta(days=int(rng.integers(0, 11)))).to_datetime64())))
            cold_anchor = c_start + pd.Timedelta(days=int(rng.choice([0, 0, 0, -1, 1, -2, 2])))
        for idx, tr in g.iterrows():
            as_of = tr.date
            if scenario == "cold_start":
                hist = daily.iloc[cold_lo:idx + 1]
                st = [s for s, vis in visible if c_start - pd.Timedelta(days=3) <= s and vis <= as_of][-1:] or [min(as_of, cold_anchor)]
            else:
                hist = daily.iloc[:idx + 1]
                st = [s for s, vis in visible if vis <= as_of]
            lh_now = [(dd, r) for dd, r in lh if dd <= as_of]
            res = analyze_fertility(model, hist, st, as_of, age=age, lh=lh_now, thresholds=thresholds, log_reliability=rel)
            if res.get("status") != "ok":
                rows.append({"pid": pid, "scenario": scenario, "cycle": ci, "date": as_of, "status": res.get("status")})
                continue
            cur_start = pd.Timestamp(res["cycle_start"])
            ov = res["ovulation"]; npd = res["next_period"]; intl = res["_internal"]
            rows.append({
                "pid": pid, "scenario": scenario, "cycle": ci, "date": as_of, "status": "ok", "true_cycle_day": int(tr.cycle_day),
                "anchor_correct": cur_start == c_start, "ovulatory": bool(tr.ovulatory), "true_ov_day": float(O) if np.isfinite(O) else np.nan,
                "fertile": bool(tr.fertile), "p_fertile": res["fertile_window"]["today_probability"], "in_window": res["fertile_window"]["in_window_today"],
                "ov_status": ov["status"], "ov_date_est": pd.Timestamp(ov["date"]) if ov["date"] else pd.NaT, "true_ov_date": true_ov_date,
                "next_est": pd.Timestamp(npd["date"]), "next_true": nxt, "p_post2": intl["p_post2"], "p_anov": intl["p_anov"],
                "conc": res["ovulation"]["concentration_3d"], "nights_after": intl["nights_after"],
                "ov_mode_date": pd.Timestamp(intl["ov_mode_date"]), "ov_median_date": pd.Timestamp(intl["ov_median_date"]),
                "amp_temp": intl["amp"].get("temperature_deviation"), "amp_hr": intl["amp"].get("resting_hr"), "amp_hrv": intl["amp"].get("lnhrv"), "amp_rr": intl["amp"].get("respiratory_rate"),
                "p_today_raw": res["fertile_window"]["today_probability"], "cycle_day_est": res["cycle_day"], "n_hist": res["insights"]["completed_cycles"],
                "cov14": res["data_quality"]["coverage_last14"], "days_since_temp": res["data_quality"]["days_since_last_temperature"], "confidence": res["data_quality"]["confidence"],
                "phase": res["phase"], "age": age, "irregular": person["irregular"], "lh_user": person["lh_user"], "illness_true": bool(tr.fever),
            })
    # retrospective history at end of last evaluated cycle
    last_idx = t[t.cycle_index == eval_cycles[-1]].index[-1]
    as_of = t.date.iloc[last_idx]
    if scenario != "cold_start":
        run = model.run(daily.iloc[:last_idx + 1], [s for s, vis in visible if vis <= as_of], as_of, age=age, lh=[(dd, r) for dd, r in lh if dd <= as_of],
                        log_reliability=rel)
        hrows = []
        for h in run["history"]:
            if "p_ovulatory" not in h:
                continue
            st = pd.Timestamp(run["dates"][h["start_index"]])
            # nearest true start (a log entered 1-2 days early must not map to the previous cycle)
            ts = np.array(starts_true, dtype="datetime64[ns]")
            j = int(np.argmin(np.abs((ts - st.to_datetime64()).astype("timedelta64[D]").astype(int))))
            tc = t[t.cycle_index == j]
            O = tc.ovulation_day.iloc[0]
            true_ov_date = starts_true[j] + pd.Timedelta(days=int(O) - 1) if np.isfinite(O) else None
            est_date = st + pd.Timedelta(days=h["ovulation_day_mode"] - 1)
            hrows.append({"pid": pid, "scenario": scenario, "true_cycle": j, "ovulatory": bool(np.isfinite(O)), "p_ovulatory": h["p_ovulatory"], "max3": h["max3"],
                          "err_days": (est_date - true_ov_date).days if true_ov_date is not None else np.nan, "logged_offset": (st - starts_true[j]).days,
                          "length_logged": h["length"], "temp_nobs": h["nobs"]["temperature_deviation"], "phys_nobs": max(h["nobs"].values()),
                          "nights_before": h.get("nights_before", 0), "nights_after": h.get("nights_after", 0), "quality": h["quality"]})
    else:
        hrows = []
    return rows, hrows


def run_many(n, seed0, scenario, params, thresholds=None, processes=2, anov=None):
    from multiprocessing import get_context
    args = [(f"{scenario}_{i}", seed0 + i * 7919, scenario, params, thresholds, anov) for i in range(n)]
    with get_context("fork").Pool(processes) as pool:
        res = pool.map(run_person, args, chunksize=4)
    days = pd.DataFrame([r for a, _ in res for r in a]); hist = pd.DataFrame([r for _, b in res for r in b])
    return days, hist
