"""Physiology-driven generator of ring nights for ovulatory/anovulatory menstrual cycles.

Latent structure per woman (hierarchical, parameters from params.py):
  phase lengths (follicular/luteal) fitted on real FedCycleData variance components;
  progesterone-like luteal effect drives temperature, RHR, ln(RMSSD), respiratory rate;
  estradiol-like pre-ovulatory bump adds an RHR rise inside the fertile window;
  confounders (fever, alcohol, travel, outliers, slow drift) and realistic wear/log missingness.
Truth columns are kept separate from observed columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .params import v

SIGNALS = ["temperature_deviation", "resting_hr", "hrv_ms", "respiratory_rate", "sleep_minutes", "sleep_efficiency", "awake_minutes"]
CORE = ["temperature_deviation", "resting_hr", "hrv_ms", "respiratory_rate"]


def ar1(rng, n, sd, rho):
    """Stationary AR(1) with marginal SD `sd`."""
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    e[0] = rng.normal(0, sd)
    return lfilter([1.0], [1.0, -rho], e)


def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def sample_person(rng, age=None, irregular=None, overrides=None):
    o = overrides or {}
    age = float(rng.uniform(20, 45) if age is None else age)
    # Correlated person means (real FedCycleData correlation -0.42)
    r = v("cycle", "corr_person_means_foll_luteal")
    z1, z2 = rng.normal(size=2)
    z2 = r * z1 + np.sqrt(1 - r * r) * z2
    mu_f = v("cycle", "follicular_mean") + v("cycle", "follicular_age_slope_per_year") * (age - 30) + v("cycle", "follicular_between_sd") * z1
    mu_l = v("cycle", "luteal_mean") + v("cycle", "luteal_between_sd") * z2
    irregular = bool(rng.random() < v("cycle", "irregular_fraction")) if irregular is None else irregular
    sd_f = float(np.exp(np.log(v("cycle", "follicular_within_sd_median")) + v("cycle", "follicular_within_sdlog") * rng.normal()))
    if irregular:
        sd_f = float(rng.uniform(4.0, 9.0))
    sd_l = float(np.exp(np.log(v("cycle", "luteal_within_sd_median")) + v("cycle", "luteal_within_sdlog") * rng.normal()))
    base_anov = v("cycle", "anovulatory_rate_lt35") if age < 35 else (v("cycle", "anovulatory_rate_35_40") if age < 40 else v("cycle", "anovulatory_rate_40_45"))
    # person heterogeneity: Beta with mean base_anov, concentration 20
    anov = float(rng.beta(base_anov * 20, (1 - base_anov) * 20)) * (1.6 if irregular else 1.0)
    adherence_group = rng.choice(["good", "medium", "poor"], p=[0.60, 0.30, 0.10])
    lo, hi = {"good": (0.93, 0.99), "medium": (0.80, 0.93), "poor": (0.55, 0.80)}[adherence_group]
    log_rel = float(rng.choice([1.0, 0.9, 0.75], p=[0.70, 0.20, 0.10]))
    person = dict(
        age=age, irregular=irregular,
        mu_f=clip(mu_f, 9, 28), mu_l=clip(mu_l, 9, 16.5), sd_f=clip(sd_f, 0.6, 9.0), sd_l=clip(sd_l, 0.4, 3.0),
        anov_rate=clip(anov, 0.0, 0.6), menses_len=clip(rng.normal(v("cycle", "menses_length_mean"), v("cycle", "menses_length_between_sd")), 3, 8),
        # temperature
        temp_amp=clip(rng.normal(v("signal", "temp_luteal_amplitude_mean"), v("signal", "temp_luteal_amplitude_sd")), 0.08, 0.80),
        temp_sd=clip(np.exp(np.log(v("signal", "temp_noise_sd_median")) + v("signal", "temp_noise_sdlog") * rng.normal()), 0.07, 0.32),
        temp_rho=float(rng.uniform(0.2, 0.5)), rise_mid=float(rng.uniform(0.8, 2.0)), rise_w=float(rng.uniform(0.35, 0.9)),
        # HR
        hr_base=clip(rng.normal(60, 7), 42, 88), hr_amp=clip(rng.normal(v("signal", "hr_luteal_plateau_mean"), v("signal", "hr_luteal_plateau_sd")), 0.3, 7.5),
        hr_pre_w=clip(rng.normal(v("signal", "hr_preovulatory_weight"), 0.12), 0.0, 0.8), hr_sd=float(rng.uniform(1.5, 3.0)), hr_rho=float(rng.uniform(0.35, 0.6)),
        # HRV (ln RMSSD)
        lnhrv_base=float(rng.normal(np.log(50) - v("signal", "rmssd_decline_per_decade_ln") * (age - 30) / 10, 0.35)),
        lnhrv_amp=clip(rng.normal(v("signal", "lnhrv_luteal_drop_mean"), v("signal", "lnhrv_luteal_drop_sd")), 0.0, 0.3),
        lnhrv_sd=float(rng.uniform(0.12, 0.22)), lnhrv_rho=float(rng.uniform(0.2, 0.45)),
        # RR
        rr_base=clip(rng.normal(14.5, 1.3), 10, 20), rr_amp=clip(rng.normal(v("signal", "rr_luteal_rise_mean"), v("signal", "rr_luteal_rise_sd")), 0.0, 1.2),
        rr_sd=float(rng.uniform(0.2, 0.45)), rr_rho=float(rng.uniform(0.3, 0.55)),
        sleep_base=clip(rng.normal(430, 40), 300, 540), eff_base=clip(rng.normal(88, 4), 70, 97), awake_base=clip(rng.normal(35, 12), 10, 80),
        alcohol_rate=float(rng.uniform(0, v("signal", "alcohol_night_fraction_max"))), fever_rate=float(rng.gamma(2.0, v("signal", "fever_episodes_per_year") / 2.0)),
        adherence=float(rng.uniform(lo, hi)), adherence_group=str(adherence_group), gap_mean=float(rng.choice([1.3, 2.0, 4.0], p=[0.5, 0.35, 0.15])),
        long_gap=bool(rng.random() < v("missing", "long_gap_person_fraction")), log_reliability=log_rel,
        lh_user=bool(rng.random() < v("lh", "user_fraction")),
    )
    person.update(o)
    return person


def sample_cycles(rng, person, n_cycles, anov_override=None):
    cycles = []
    rw = v("cycle", "corr_within_foll_luteal")
    for _ in range(n_cycles):
        anov_p = person["anov_rate"] if anov_override is None else anov_override
        if rng.random() < anov_p:
            length = int(np.clip(np.round(rng.normal(person["mu_f"] + person["mu_l"] + 4, 7)), 21, 75))
            cycles.append({"length": length, "ovulation": None, "menses": int(np.clip(np.round(rng.normal(person["menses_len"], 0.8)), 2, 9))})
            continue
        z1, z2 = rng.normal(size=2)
        z2 = rw * z1 + np.sqrt(1 - rw * rw) * z2
        f = int(np.clip(np.round(person["mu_f"] + person["sd_f"] * z1), 8, 60))
        l_ = int(np.clip(np.round(person["mu_l"] + person["sd_l"] * z2), 7, 18))
        cycles.append({"length": f + l_, "ovulation": f, "menses": int(np.clip(np.round(rng.normal(person["menses_len"], 0.7)), 2, 9))})
    return cycles


def latent_vectors(person, cycles, start_offset=0):
    """Daily latent effects. Luteal effect decays across the menses boundary with signal-specific lags."""
    n = sum(c["length"] for c in cycles)
    prog = np.zeros(n)        # temperature/RR luteal shape (0..1)
    prog_hr = np.zeros(n)     # HR/HRV shape, slower decay
    e2 = np.zeros(n)          # pre-ovulatory bump
    cday = np.zeros(n, int); cid = np.zeros(n, int); ovd = np.full(n, np.nan); clen = np.zeros(n, int)
    pos = 0
    for i, c in enumerate(cycles):
        L = c["length"]; d = np.arange(1, L + 1)
        cday[pos:pos + L] = d; cid[pos:pos + L] = i; clen[pos:pos + L] = L
        if c["ovulation"] is not None:
            O = c["ovulation"]; k = d - O
            rise = 1 / (1 + np.exp(-(k - person["rise_mid"]) / person["rise_w"]))
            to_next = L + 1 - d
            decline = 1 / (1 + np.exp(-(to_next - 1.6) / 0.55))
            shape = rise * decline
            prog[pos:pos + L] = np.maximum(prog[pos:pos + L], shape)
            hr_decline = 1 / (1 + np.exp(-(to_next - 0.2) / 0.8))
            prog_hr[pos:pos + L] = np.maximum(prog_hr[pos:pos + L], rise * hr_decline)
            e2[pos:pos + L] = np.exp(-0.5 * ((k + 1.0) / 2.2) ** 2) * (k <= 1)
            ovd[pos:pos + L] = O
            # carry-over into the next cycle's first days (residual progesterone effect)
            nxt = min(n, pos + L + 4)
            tail = np.arange(1, nxt - (pos + L) + 1)
            prog_hr[pos + L:nxt] = np.maximum(prog_hr[pos + L:nxt], rise[-1] * hr_decline[-1] * np.exp(-tail / 1.5))
        pos += L
    return dict(prog=prog, prog_hr=prog_hr, e2=e2, cycle_day=cday, cycle_index=cid, ovulation_day=ovd, cycle_length=clen)


def render_signals(rng, person, lat, extra=None):
    """Nightly signals in Oura-like units. extra: optional dict of additive daily effects (menopause module)."""
    n = len(lat["prog"])
    extra = extra or {}
    temp_amp_cycle = np.ones(n)
    # cycle-to-cycle amplitude variation
    for i in np.unique(lat["cycle_index"]):
        temp_amp_cycle[lat["cycle_index"] == i] = max(0.2, rng.normal(1.0, 0.12))
    # confounders
    fever = np.zeros(n); alcohol = np.zeros(n); travel = np.zeros(n)
    n_fever = rng.poisson(person["fever_rate"] * n / 365)
    for _ in range(n_fever):
        s = int(rng.integers(0, n)); dur = int(rng.integers(2, 7)); peak = rng.uniform(0.5, 1.5)
        prof = np.sin(np.linspace(0.3, np.pi - 0.3, dur)) * peak
        fever[s:s + dur] = np.maximum(fever[s:s + dur], prof[: len(fever[s:s + dur])])
    alcohol[rng.random(n) < person["alcohol_rate"]] = 1.0
    for _ in range(rng.poisson(1.0 * n / 365)):
        s = int(rng.integers(0, n)); dur = int(rng.integers(2, 5)); travel[s:s + dur] = rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
    drift = ar1(rng, n, 0.07, 0.985)
    outlier = (rng.random(n) < 0.02) * rng.standard_t(3, n) * 0.25
    alc_t = alcohol * rng.uniform(0.05, 0.25, n)
    temp = (person["temp_amp"] * temp_amp_cycle * lat["prog"] - 0.45 * person["temp_amp"]
            + ar1(rng, n, person["temp_sd"], person["temp_rho"]) + drift + fever + alc_t + travel + outlier + extra.get("temp", 0))
    hr = (person["hr_base"] + person["hr_amp"] * (lat["prog_hr"] + person["hr_pre_w"] * lat["e2"])
          + ar1(rng, n, person["hr_sd"], person["hr_rho"]) + fever * 8 + alcohol * rng.uniform(2, 6, n) + extra.get("hr", 0))
    lnhrv = (person["lnhrv_base"] - person["lnhrv_amp"] * lat["prog_hr"] + ar1(rng, n, person["lnhrv_sd"], person["lnhrv_rho"])
             - fever * 0.25 - alcohol * rng.uniform(0.05, 0.25, n) + extra.get("lnhrv", 0))
    rr = (person["rr_base"] + person["rr_amp"] * lat["prog"] + ar1(rng, n, person["rr_sd"], person["rr_rho"]) + fever * 1.0 + extra.get("rr", 0))
    sleep = person["sleep_base"] + rng.normal(0, 45, n) - 15 * lat["prog"] * (lat["cycle_day"] > lat["cycle_length"] - 4) - fever * 25 + extra.get("sleep", 0)
    eff = person["eff_base"] + rng.normal(0, 4, n) - fever * 4 - alcohol * 2 + extra.get("eff", 0)
    awake = person.get("awake_base", 35.0) + rng.gamma(2.0, 6.0, n) - 12 + fever * 15 + extra.get("awake", 0)
    df = pd.DataFrame({
        "temperature_deviation": np.round(temp, 2), "resting_hr": np.round(np.clip(hr, 35, 130), 0),
        "hrv_ms": np.round(np.clip(np.exp(lnhrv), 5, 250), 0), "respiratory_rate": np.round(np.clip(rr, 8, 28), 1),
        "sleep_minutes": np.round(np.clip(sleep, 60, 720), 0), "sleep_efficiency": np.round(np.clip(eff, 40, 100), 0),
        "awake_minutes": np.round(np.clip(awake, 0, 300), 0),
    })
    truth = pd.DataFrame({"fever": fever > 0.2, "alcohol": alcohol > 0, "travel": travel != 0})
    return df, truth


def apply_missingness(rng, person, df):
    n = len(df)
    target_missing = 1 - person["adherence"]
    q = target_missing / max(person["gap_mean"] * (1 - target_missing), 1e-6)
    worn = np.ones(n, bool)
    t = 0
    while t < n:
        if rng.random() < q:
            g = int(rng.geometric(1 / person["gap_mean"]))
            worn[t:t + g] = False; t += g
        t += 1
    if person["long_gap"] and n > 40:
        s = int(rng.integers(0, n - 7)); worn[s:s + int(rng.integers(7, 22))] = False
    out = df.copy()
    out.loc[~worn, SIGNALS] = np.nan
    # channel-specific dropout (short sleep / poor signal)
    for c, pr in [("temperature_deviation", 0.02), ("hrv_ms", 0.04), ("respiratory_rate", 0.04), ("resting_hr", 0.01)]:
        out.loc[rng.random(n) < pr, c] = np.nan
    return out, worn


def logged_period_starts(rng, person, true_start_idx):
    """Returns sorted unique (logged_index, true_index) pairs. A log can only be entered once the period has begun."""
    logged = {}
    for i, s in enumerate(true_start_idx):
        if i > 0 and rng.random() > person["log_reliability"]:
            continue
        jitter = int(rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03])) * int(rng.choice([-1, 1]))
        if i == 0:
            jitter = 0
        logged.setdefault(int(s + jitter), int(s))
    return sorted(logged.items())


def lh_tests(rng, person, cycles, start_idx):
    if not person["lh_user"]:
        return []
    probs = v("lh", "positive_prob_by_offset"); fp = v("lh", "false_positive")
    out = []
    for c, s in zip(cycles, start_idx):
        if rng.random() > 0.7:
            continue
        expected = int(round(person["mu_f"]))
        first = max(6, expected - 6)
        for d in range(first, min(c["length"], expected + 6) + 1):
            if rng.random() > 0.8:
                continue
            if c["ovulation"] is None:
                pos = rng.random() < 0.04
            else:
                pos = rng.random() < probs.get(d - c["ovulation"], fp)
            out.append((int(s + d - 1), bool(pos)))
    return out


def simulate_person(rng, n_cycles=13, age=None, person=None, start="2024-01-01", anov_override=None, signal_scale=None):
    person = person or sample_person(rng, age=age)
    if signal_scale:
        for k, s in signal_scale.items():
            person[k] *= s
    cycles = sample_cycles(rng, person, n_cycles, anov_override=anov_override)
    lat = latent_vectors(person, cycles)
    clean, conf = render_signals(rng, person, lat)
    obs, worn = apply_missingness(rng, person, clean)
    n = len(clean)
    dates = pd.date_range(start, periods=n, freq="D")
    starts = np.r_[0, np.cumsum([c["length"] for c in cycles])[:-1]]
    truth = pd.DataFrame({"date": dates, "cycle_index": lat["cycle_index"], "cycle_day": lat["cycle_day"], "cycle_length": lat["cycle_length"],
                          "ovulation_day": lat["ovulation_day"], "ovulatory": np.isfinite(lat["ovulation_day"]), "worn": worn,
                          "fever": conf.fever.values, "alcohol": conf.alcohol.values})
    truth["fertile"] = truth.ovulatory & (truth.cycle_day >= truth.ovulation_day - 5) & (truth.cycle_day <= truth.ovulation_day)
    obs.insert(0, "date", dates)
    pairs = logged_period_starts(rng, person, starts)
    lh = lh_tests(rng, person, cycles, starts)
    return {"person": person, "cycles": cycles, "true_starts": [dates[i] for i in starts], "logged_starts": [dates[i] for i, _ in pairs if 0 <= i < n],
            "logged_visible": [(dates[i], dates[min(n - 1, max(i, j))]) for i, j in pairs if 0 <= i < n],
            "lh": [(dates[i], r) for i, r in lh if 0 <= i < n], "daily": obs, "clean": clean, "truth": truth}
