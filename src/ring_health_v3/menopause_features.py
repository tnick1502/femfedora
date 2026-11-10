"""Feature extraction for the menopausal-transition models (identical at training and inference).

No missing-value indicators are used as features: HistGradientBoosting routes NaN natively, and training
masks optional inputs at random (independent of outcome) so that 'not provided' carries no outcome signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cycle_engine import CycleModel

NIGHT_FEATURES = ["d_temp", "d_eff", "d_awake", "d_lnhrv", "d_hr", "d_rr", "d_sleep", "temp_minus_7d_min"]

FEATURES = [
    "age",
    # bleeding pattern (STRAW+10 supportive criteria)
    "days_since_last_period", "periods_last_6m", "periods_last_12m", "median_interval", "interval_sd", "max_interval",
    "intervals_ge60", "persistent_diff7", "log_history_days", "last_interval",
    # ring physiology
    "ring_nights_90", "ring_cycles_completed", "ring_ovulatory_fraction", "ring_temp_shift_mean", "temp_night_sd", "temp_spike_rate",
    "vms_index_30", "vms_nights_30", "eff_mean_30", "awake_mean_30", "lnhrv_age_resid", "hr_mean_90", "sleep_mean_30",
    "ring_days_since_luteal", "temp_cycle_power",
    # self-report
    "hot_flash_days_30",
    # optional labs / covariates
    "ln_fsh", "ln_amh", "smoker", "bmi", "parous", "college",
]


def cycle_power(temp: np.ndarray, min_obs=40):
    """Max R^2 of a sinusoid with period 21-40 d fitted on observed nights only (least-squares periodogram)."""
    t = np.arange(len(temp), dtype=float); m = np.isfinite(temp)
    if m.sum() < min_obs:
        return np.nan
    t, y = t[m], temp[m] - np.nanmean(temp)
    ss = float(y @ y)
    if ss <= 0:
        return np.nan
    best = 0.0
    for P in range(21, 41):
        X = np.c_[np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P), np.ones_like(t)]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        best = max(best, 1 - float(r @ r) / ss)
    return best


def night_table(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["lnhrv"] = np.log(d["hrv_ms"].where(d["hrv_ms"] > 0))
    out = pd.DataFrame(index=d.index)
    for src, dst in [("temperature_deviation", "d_temp"), ("sleep_efficiency", "d_eff"), ("awake_minutes", "d_awake"), ("lnhrv", "d_lnhrv"),
                     ("resting_hr", "d_hr"), ("respiratory_rate", "d_rr"), ("sleep_minutes", "d_sleep")]:
        x = pd.to_numeric(d[src], errors="coerce") if src in d else pd.Series(np.nan, index=d.index)
        base = x.rolling(30, min_periods=7).median().shift(1)
        out[dst] = x - base
    t = pd.to_numeric(d["temperature_deviation"], errors="coerce")
    out["temp_minus_7d_min"] = t - t.rolling(7, min_periods=3).min().shift(1)
    return out


def bleeding_features(starts, as_of, log_history_days=None, last_period_date=None):
    """Bleeding-pattern features. log_history_days not None means the user logs periods in the app, so an empty
    log is informative (counts are 0, not missing). days_since uses the most recent of logs and the onboarding date."""
    as_of = pd.Timestamp(as_of)
    s = sorted(pd.Timestamp(x) for x in starts if pd.Timestamp(x) <= as_of)
    f = {k: np.nan for k in ["days_since_last_period", "periods_last_6m", "periods_last_12m", "median_interval", "interval_sd", "max_interval",
                              "intervals_ge60", "persistent_diff7", "log_history_days", "last_interval"]}
    logging = log_history_days is not None
    f["log_history_days"] = float(max(log_history_days, 0)) if logging else np.nan
    cands = [x for x in s[-1:]]
    if last_period_date is not None and pd.Timestamp(last_period_date) <= as_of:
        cands.append(pd.Timestamp(last_period_date))
    if cands:
        f["days_since_last_period"] = float(min(max((as_of - max(cands)).days, 0), 1095))
    if logging:
        f["periods_last_6m"] = float(sum(1 for x in s if (as_of - x).days <= 182))
        f["periods_last_12m"] = float(sum(1 for x in s if (as_of - x).days <= 365))
        f["intervals_ge60"] = 0.0
    iv = np.diff([x.toordinal() for x in s if (as_of - x).days <= 730]).astype(float)
    if len(iv):
        f["median_interval"] = float(np.median(iv)); f["max_interval"] = float(iv.max()); f["intervals_ge60"] = float((iv >= 60).sum())
        f["last_interval"] = float(iv[-1])
        f["interval_sd"] = float(np.std(iv, ddof=1)) if len(iv) >= 3 else np.nan
        last10 = iv[-10:]
        f["persistent_diff7"] = float((np.abs(np.diff(last10)) >= 7).sum()) if len(last10) >= 3 else np.nan
    return f


def extract(daily, starts, as_of, age, *, cycle_model: CycleModel, night_model=None, symptom_dates=None, symptom_logging=None,
            log_history_days=None, labs=None, covariates=None, last_period_date=None):
    as_of = pd.Timestamp(as_of)
    f = {k: np.nan for k in FEATURES}
    f["age"] = float(int(age))
    f.update(bleeding_features(starts, as_of, log_history_days, last_period_date))
    d = daily[daily["date"] <= as_of].tail(365).reset_index(drop=True) if daily is not None and len(daily) else None
    if d is not None and len(d):
        t90 = d.tail(90)
        f["ring_nights_90"] = float(t90["temperature_deviation"].notna().sum())
        if f["ring_nights_90"] >= 20:
            temp = d["temperature_deviation"].astype(float)
            diffs = temp.diff().dropna()
            f["temp_night_sd"] = float(np.median(np.abs(diffs - diffs.median())) * 1.4826 / np.sqrt(2)) if len(diffs) > 10 else np.nan
            base = temp.rolling(30, min_periods=7).median().shift(1)
            sp = ((temp - base) > 0.35).where(temp.notna() & base.notna()).tail(60)   # share among OBSERVED nights
            f["temp_spike_rate"] = float(sp.mean()) if sp.notna().sum() >= 10 else np.nan
            f["temp_cycle_power"] = cycle_power(temp.tail(120).to_numpy(float))
            lnhrv = np.log(t90["hrv_ms"].where(t90["hrv_ms"] > 0))
            f["lnhrv_age_resid"] = float(lnhrv.mean() - (np.log(50) - 0.025 * (age - 30))) if lnhrv.notna().sum() >= 10 else np.nan
            f["hr_mean_90"] = float(t90["resting_hr"].mean()) if t90["resting_hr"].notna().sum() >= 10 else np.nan
            t30 = d.tail(30)
            f["eff_mean_30"] = float(t30["sleep_efficiency"].mean()); f["awake_mean_30"] = float(t30["awake_minutes"].mean()) if "awake_minutes" in t30 else np.nan
            f["sleep_mean_30"] = float(t30["sleep_minutes"].mean())
            if night_model is not None:
                nt = night_table(d).tail(30)
                ok = nt["d_temp"].notna() | nt["d_eff"].notna()
                if ok.sum() >= 10:
                    pr = night_model.predict_proba(nt.loc[ok, NIGHT_FEATURES])[:, 1]
                    f["vms_index_30"] = float(pr.mean()); f["vms_nights_30"] = float((pr >= 0.5).mean())  # fraction of scored nights
            # luteal activity from the cycle engine on the ring window (logged starts within window)
            st = [x for x in starts if d["date"].iloc[0] <= pd.Timestamp(x) <= as_of]
            if len(st) >= 2:
                run = cycle_model.run(d, st, as_of, age=min(age, 50), infer_unlogged=False)
                hist = [h for h in run["history"] if "p_ovulatory" in h]
                if hist:
                    f["ring_cycles_completed"] = float(len(hist))
                    f["ring_ovulatory_fraction"] = float(np.mean([h["p_ovulatory"] for h in hist]))
                    sh = [h["temp_shift"][0] for h in hist if h.get("temp_shift")]
                    f["ring_temp_shift_mean"] = float(np.mean(sh)) if sh else np.nan
                    last_ov = [pd.Timestamp(run["dates"][h["start_index"]]) + pd.Timedelta(days=h["ovulation_day_mode"]) for h in hist if h["p_ovulatory"] >= 0.8]
                    f["ring_days_since_luteal"] = float((as_of - last_ov[-1]).days) if last_ov else np.nan
            # with no period logs: temperature-only luteal evidence over the window (cycle engine needs anchors)
    if symptom_logging:
        sd = [pd.Timestamp(x) for x in (symptom_dates or [])]
        f["hot_flash_days_30"] = float(sum(1 for x in sd if 0 <= (as_of - x).days < 30))
    labs = labs or {}
    if labs.get("fsh") is not None:
        f["ln_fsh"] = float(np.log(max(labs["fsh"], 0.5)))
    if labs.get("amh") is not None:
        f["ln_amh"] = float(np.log(max(labs["amh"], 1.0)))
    cov = covariates or {}
    for k in ["smoker", "bmi", "parous", "college"]:
        if cov.get(k) is not None:
            f[k] = float(cov[k])
    return f
