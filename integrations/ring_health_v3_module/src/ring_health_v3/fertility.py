"""Oura-like cycle analysis built on the Bayesian cycle engine.

Outputs: cycle day & phase, fertile-window probability today + 7-day forecast + window dates,
ovulation status (predicted / detected) with date & uncertainty, next-period forecast,
past-cycle insights (ovulation dates, luteal length, possible anovulatory cycles), data quality.
Every number is model output on synthetic-trained parameters; not a contraceptive or diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .cycle_engine import CycleModel, max3

DEFAULT_THRESHOLDS = {
    "window_probability": 0.30,     # day shown as fertile window
    "detect_post_mass": 0.90,       # P(ovulation already happened >=2 days ago)
    "detect_concentration": 0.60,   # posterior mass in best 3-day span
    "detect_min_temp_nights_after": 3,
    "history_detect_p_ovulatory": 0.80,
    "history_detect_concentration": 0.60,
    "anovulatory_flag": 0.70,
}


def _quantile(grid, p, q):
    c = np.cumsum(p) / max(p.sum(), 1e-300)
    return int(grid[min(np.searchsorted(c, q), len(grid) - 1)])


def _iso(ts):
    return str(pd.Timestamp(ts).date())


def next_period_distribution(post, personal, today_day, log_reliability):
    """P(next period starts on cycle day n), n > today. For each hypothesis (ovulation day O, or anovulatory) the
    cycle-length pmf is conditioned on N > today separately; the posterior over O already contains the survival
    evidence, so it is not applied twice."""
    grid = post["grid"]; po = post["p_o"]; pa = post["p_anov"]
    lut = personal["luteal"]; anl = personal["anov_len"]
    horizon = np.arange(today_day + 1, today_day + 120)
    pm = np.zeros(len(horizon))
    keep = po > 1e-6
    if keep.any():
        Os = grid[keep][:, None]; ws = po[keep][:, None]
        pmf = ndtr((horizon[None, :] - Os - 1 + 0.5 - lut["mean"]) / lut["sd"]) - ndtr((horizon[None, :] - Os - 1 - 0.5 - lut["mean"]) / lut["sd"])
        pmf = pmf / np.maximum(pmf.sum(1, keepdims=True), 1e-300)
        pm += (ws * pmf).sum(0)
    pa_pmf = ndtr((horizon - 1 + 0.5 - anl["mean"]) / anl["sd"]) - ndtr((horizon - 1 - 0.5 - anl["mean"]) / anl["sd"])
    if pa_pmf.sum() > 1e-12:
        pm += pa * pa_pmf / pa_pmf.sum()
    s = pm.sum()
    if s < 1e-12:
        return horizon, np.ones(len(horizon)) / len(horizon), True
    return horizon, pm / s, False


def phase_of_day(cycle_day, menses_len, p_fertile, window_threshold, ovulation_status, ov_day, days_to_period):
    if cycle_day <= menses_len:
        return "menstruation"
    if ovulation_status == "detected" and ov_day is not None:
        if cycle_day == ov_day:
            return "ovulation"
        if cycle_day > ov_day:
            return "late_luteal" if days_to_period is not None and days_to_period <= 4 else "luteal"
    if p_fertile >= window_threshold:
        return "fertile_window"
    if ovulation_status != "detected" and ov_day is not None and cycle_day > ov_day + 1:
        return "luteal_expected"
    return "follicular"


def analyze_fertility(model: CycleModel, daily: pd.DataFrame, starts, as_of, *, age=None, lh=None, log_reliability=0.9,
                      menses_length=None, thresholds=None, ml=None):
    th = {**DEFAULT_THRESHOLDS, **(model.p.get("thresholds") or {}), **(thresholds or {})}
    as_of = pd.Timestamp(as_of)
    if not starts:
        return {"status": "anchor_required", "message": "Need at least one period start date (or current cycle day)."}
    run = model.run(daily, starts, as_of, age=age, lh=lh, log_reliability=log_reliability)
    if run.get("anchor_stale") or run["current"] is None:
        return {"status": "anchor_stale", "message": "The latest logged period start is more than 100 days ago; ask for the most recent period start date.",
                "needed": ["cycle.period_starts (most recent start)"]}
    post = run["current"]; per = run["personal"]; frame = run["frame"]
    start_date = pd.Timestamp(run["dates"][post["start_index"]])
    t = post["today_day"]
    grid, po, pa = post["grid"], post["p_o"], post["p_anov"]
    # fertile probability for cycle day d: P(O in [d, d+5])
    cs = np.r_[0, np.cumsum(po)]

    def p_fert(d):
        lo = np.searchsorted(grid, d); hi = np.searchsorted(grid, d + 5, side="right")
        return float(cs[hi] - cs[lo])
    p_today = p_fert(t)
    p_post = float(po[grid < t].sum())               # ovulation before today
    p_post2 = float(po[grid <= t - 2].sum())
    conc = max3(po / max(po.sum(), 1e-300))
    pov = 1 - pa
    temp_obs = frame["temperature_deviation"].iloc[post["start_index"]:].notna().to_numpy()
    cond = po / max(po.sum(), 1e-300)
    ov_mode = int(grid[np.argmax(cond)])
    q10, q50, q90 = (_quantile(grid, cond, q) for q in (0.1, 0.5, 0.9))
    nights_after = int(temp_obs[ov_mode:].sum()) if ov_mode < len(temp_obs) else 0
    days_since_temp = int(len(temp_obs) - 1 - np.flatnonzero(temp_obs)[-1]) if temp_obs.any() else None
    coverage14 = float(frame["temperature_deviation"].tail(14).notna().mean())
    detected = (p_post2 >= th["detect_post_mass"] and conc >= th["detect_concentration"] and nights_after >= th["detect_min_temp_nights_after"]
                and (days_since_temp is not None and days_since_temp <= 5))
    if detected:
        ov_status = "detected"
    elif pa >= th["anovulatory_flag"] and t >= 25:
        ov_status = "not_detected_possible_anovulatory_cycle"
    elif float(po[grid < t].sum() / max(po.sum(), 1e-300)) >= 0.5:  # conditional on an ovulatory cycle
        ov_status = "likely_passed_not_confirmed"
    else:
        ov_status = "predicted"
    # next period
    horizon, pn, degenerate = next_period_distribution(post, per, t, log_reliability)
    n10, n50, n90 = (_quantile(horizon, pn, q) for q in (0.1, 0.5, 0.9))
    # overdue: compare with prior expectation not conditioned on "no period yet"
    expected_len = per["foll"]["mean"] + per["luteal"]["mean"]
    overdue = t > expected_len + 2.5 * np.sqrt(per["foll"]["sd"] ** 2 + per["luteal"]["sd"] ** 2) + 2
    # window dates over the current cycle (calendar)
    days = np.arange(1, max(t + 7, int(q90) + 3) + 1)
    pf = np.array([p_fert(d) for d in days])
    win = days[pf >= th["window_probability"]]
    window = None
    if len(win):
        # contiguous block nearest to today/forward
        blocks = np.split(win, np.flatnonzero(np.diff(win) > 1) + 1)
        blk = min(blocks, key=lambda b: (0 if b[-1] >= t else 1, abs(b[0] - t)))
        window = {"start": _iso(start_date + pd.Timedelta(days=int(blk[0]) - 1)), "end": _iso(start_date + pd.Timedelta(days=int(blk[-1]) - 1)),
                  "peak_day": _iso(start_date + pd.Timedelta(days=int(days[np.argmax(pf)]) - 1)), "status": "past" if blk[-1] < t else ("now" if blk[0] <= t else "upcoming")}
    menses_len = int(round(menses_length if menses_length else 5))
    days_to_period = int(n50 - t)
    ov_day_for_phase = ov_mode if (detected or ov_status in ("predicted", "likely_passed_not_confirmed")) else None
    phase = phase_of_day(t, menses_len, p_today, th["window_probability"], ov_status, ov_day_for_phase, days_to_period)
    forecast = [{"date": _iso(as_of + pd.Timedelta(days=h)), "cycle_day": t + h, "fertile_probability": round(p_fert(t + h), 3)} for h in range(7)]
    # confidence
    hist_n = per["n_ovulatory_equiv"]
    if coverage14 >= 0.7 and hist_n >= 2:
        conf = "high"
    elif coverage14 >= 0.4 or hist_n >= 1:
        conf = "medium"
    else:
        conf = "low"
    evidence = "physiology_and_history" if coverage14 > 0 else ("history_only" if hist_n > 0 else "population_prior_only")
    # history insights
    hist = []
    for h in run["history"]:
        st = pd.Timestamp(run["dates"][h["start_index"]])
        item = {"start": _iso(st), "length": h.get("length"), "quality": h["quality"]}
        if "p_ovulatory" in h:
            # detection needs physiological evidence: a calendar-only posterior (cycle length + luteal prior) is an estimate, not a detection
            phys = max(h["nobs"].values()) if h.get("nobs") else 0
            evid = phys >= th.get("history_min_signal_nights", 8) and h.get("nights_before", 0) >= 3 and h.get("nights_after", 0) >= 3
            det = h["p_ovulatory"] >= th["history_detect_p_ovulatory"] and h["max3"] >= th["history_detect_concentration"] and evid
            item.update(ovulation_detected=bool(det), ovulation_date=_iso(st + pd.Timedelta(days=h["ovulation_day_mode"] - 1)) if det else None,
                        ovulation_date_estimate=_iso(st + pd.Timedelta(days=h["ovulation_day_mode"] - 1)), estimate_basis="physiology" if evid else "calendar_or_sparse_data",
                        luteal_length=int(h["length"] - h["ovulation_day_mode"]) if det else None,
                        possible_anovulatory=bool(1 - h["p_ovulatory"] >= th["anovulatory_flag"]),
                        temperature_shift_c=round(h["temp_shift"][0], 2) if det and h.get("temp_shift") else None)
        hist.append(item)
    lengths = [h["length"] for h in hist if h.get("quality") == "ok"]
    luts = [h["luteal_length"] for h in hist if h.get("luteal_length")]
    insights = {
        "completed_cycles": len(lengths),
        "cycle_length_median": float(np.median(lengths)) if lengths else None,
        "cycle_length_range": [int(min(lengths)), int(max(lengths))] if lengths else None,
        "cycle_variability_sd": round(float(np.std(lengths, ddof=1)), 1) if len(lengths) >= 3 else None,
        "regularity": (None if len(lengths) < 3 else ("regular" if max(lengths) - min(lengths) <= 7 else ("somewhat_irregular" if max(lengths) - min(lengths) <= 14 else "irregular"))),
        "luteal_length_median": float(np.median(luts)) if luts else None,
        "short_luteal_cycles": int(sum(1 for x in luts if x < 10)),
        "possible_anovulatory_cycles": int(sum(1 for h in hist if h.get("possible_anovulatory"))),
        "personal_follicular_mean": round(per["foll"]["mean"], 1), "personal_luteal_mean": round(per["luteal"]["mean"], 1),
    }
    inferred_now = [x for x in run["inferred_starts"] if x["where"] == "current"]
    inferred_hist = [x for x in run["inferred_starts"] if x["where"] == "history"]
    res = {
        "status": "ok", "as_of": _iso(as_of), "cycle_start": _iso(start_date), "cycle_day": int(t), "phase": phase,
        "cycle_start_source": "inferred_from_physiology" if inferred_now else "logged",
        "unlogged_period_suggestions": [{"date": _iso(run["dates"][x["index"]]), "probability": round(x["probability"], 3),
                                          "action": "ask_user_to_confirm_period_start"} for x in inferred_now + inferred_hist],
        "fertile_window": {"today_probability": round(p_today, 3), "in_window_today": bool(p_today >= th["window_probability"]), "window": window, "forecast_7d": forecast},
        "ovulation": {"status": ov_status, "date": _iso(start_date + pd.Timedelta(days=(ov_mode if detected else q50) - 1)) if ov_status != "not_detected_possible_anovulatory_cycle" else None,
                      "interval_80": [_iso(start_date + pd.Timedelta(days=q10 - 1)), _iso(start_date + pd.Timedelta(days=q90 - 1))],
                      "probability_already_passed": round(p_post, 3), "probability_ovulatory_cycle": round(pov, 3), "concentration_3d": round(conc, 3)},
        "next_period": {"date": _iso(start_date + pd.Timedelta(days=n50 - 1)), "interval_80": [_iso(start_date + pd.Timedelta(days=n10 - 1)), _iso(start_date + pd.Timedelta(days=n90 - 1))],
                        "days_until": days_to_period, "late_or_unlogged": bool(overdue)},
        "insights": insights, "history": hist,
        "data_quality": {"temperature_nights_last14": int(round(coverage14 * 14)), "coverage_last14": round(coverage14, 2), "days_since_last_temperature": days_since_temp,
                          "possible_illness_nights_last14": run["illness_nights_recent"], "evidence": evidence, "confidence": conf,
                          "observed_nights_current_cycle": post["nobs"], "imputation": "none (missing nights marginalised in the likelihood)"},
        "_internal": {"amp": {k: (float(np.sum(cond * a[0])) if a is not None else None) for k, a in post["amp_post"].items()}, "p_post2": p_post2, "ov_mode_date": _iso(start_date + pd.Timedelta(days=ov_mode - 1)), "ov_median_date": _iso(start_date + pd.Timedelta(days=q50 - 1)), "p_anov": pa, "ov_mode_day": ov_mode, "q": [q10, q50, q90], "next_q": [n10, n50, n90], "nights_after": nights_after},
    }
    if ml is not None:
        res["fertile_window"]["today_probability_ml"] = ml(res, run)
    return res
