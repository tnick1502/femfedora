"""Multi-signal Bayesian cycle engine (hidden ovulation day per cycle).

For each cycle the latent ovulation day O (or an anovulatory hypothesis) explains nightly signals:
    y_s(t) = mu_c + a_s * g_s(t - O) * e_s(days_to_next_start) + AR(1) noise
mu_c (cycle offset) and a_s (amplitude, personal prior) are Gaussian and integrated out exactly;
the AR(1) correlation uses the real time between observed nights, so missing nights are
marginalised (never imputed). Prior over O comes from the woman's own past cycles (empirical
Bayes, shrunk to real-data population priors). Optional evidence: urine LH tests, cycle end
(luteal-length likelihood) and "no new period yet" survival with imperfect logging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

MODEL_SIGNALS = ["temperature_deviation", "resting_hr", "lnhrv", "respiratory_rate"]
K_MIN, K_MAX = -20, 25
O_MIN = 6
MAX_SPLIT_SEGMENT = 100  # days; longer single segments are not split (anchor considered stale)
FIRST_DAY = {"temperature_deviation": 2, "resting_hr": 4, "lnhrv": 4, "respiratory_rate": 2}


def to_model_frame(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["lnhrv"] = np.log(d["hrv_ms"].where(d["hrv_ms"] > 0)) if "hrv_ms" in d else np.nan
    for c in MODEL_SIGNALS:
        if c not in d:
            d[c] = np.nan
    return d


def discrete_normal(grid, mean, sd):
    g = np.asarray(grid, float)
    p = ndtr((g + 0.5 - mean) / sd) - ndtr((g - 0.5 - mean) / sd)
    return np.maximum(p, 1e-12)


def _sf(x, mean, sd):
    return ndtr(-(np.asarray(x, float) - mean) / sd)


class CycleModel:
    def __init__(self, params: dict):
        self.p = params
        self.templates = {s: np.asarray(params["templates"][s]["g"], float) for s in MODEL_SIGNALS}
        self.end = {s: np.asarray(params["templates"][s]["end"], float) for s in MODEL_SIGNALS}

    # ---------- signal model helpers
    def g(self, s, k):
        idx = np.clip(np.asarray(k) - K_MIN, 0, K_MAX - K_MIN).astype(int)
        return self.templates[s][idx]

    def end_factor(self, s, to_next):
        e = np.ones_like(to_next, dtype=float)
        tbl = self.end[s]
        m = (to_next >= 1) & (to_next <= len(tbl))
        e[m] = tbl[to_next[m].astype(int) - 1]
        return e

    def person_noise(self, frame):
        """Personal marginal SD per signal from robust first differences, shrunk to population."""
        out = {}
        for s in MODEL_SIGNALS:
            pop = self.p["noise"][s]
            x = frame[s].to_numpy(float)
            idx = np.flatnonzero(np.isfinite(x))
            sd = pop["sd"]
            if len(idx) > 10:
                consecutive = np.diff(idx) == 1
                dif = np.diff(x[idx])[consecutive]
                if len(dif) >= 8:
                    mad = np.median(np.abs(dif - np.median(dif))) * 1.4826
                    est = mad / np.sqrt(2 * (1 - pop["rho"]))
                    n0 = 12.0
                    sd = float(np.sqrt((n0 * pop["sd"] ** 2 + len(dif) * est ** 2) / (n0 + len(dif))))
            out[s] = {"sd": max(sd, pop["sd_floor"]), "rho": pop["rho"]}
        return out

    @staticmethod
    def illness_mask(frame):
        """Nights likely affected by fever/acute illness: large temperature excursion with HR rise."""
        t = frame["temperature_deviation"].to_numpy(float)
        hr = frame["resting_hr"].to_numpy(float)
        tm = pd.Series(t).rolling(21, min_periods=5, center=False).median().shift(1).to_numpy()
        hm = pd.Series(hr).rolling(21, min_periods=5, center=False).median().shift(1).to_numpy()
        hot = np.isfinite(t) & np.isfinite(tm) & (t - tm > 0.75)
        hrup = np.isfinite(hr) & np.isfinite(hm) & (hr - hm > 5)
        very_hot = np.isfinite(t) & np.isfinite(tm) & (t - tm > 1.1)
        return (hot & (hrup | ~np.isfinite(hr))) | very_hot

    # ---------- per-cycle likelihood
    def signal_loglik(self, s, days, y, grid, noise, amp_prior, cycle_len=None, baseline=None):
        """Returns (loglik(O) - loglik(anov), absolute loglik(anov), amplitude posterior per O)."""
        n = len(y)
        if n < 3:
            return np.zeros(len(grid)), 0.0, None
        sd, rho = noise["sd"], noise["rho"]
        tau = self.p["offset_sd"][s]
        dist = np.abs(days[:, None] - days[None, :])
        S0 = sd ** 2 * rho ** dist + tau ** 2
        S0[np.diag_indices(n)] += 1e-9
        cf = cho_factor(S0, lower=True, check_finite=False)
        logdet = 2 * np.sum(np.log(np.diag(cf[0])))
        yc = y - (np.median(y) if baseline is None else baseline)
        Sy = cho_solve(cf, yc, check_finite=False)
        a0 = float(yc @ Sy)
        K = days[:, None] - grid[None, :]
        G = self.g(s, K)
        if cycle_len is not None:
            G = G * self.end_factor(s, (cycle_len + 1 - days))[:, None]
        SG = cho_solve(cf, G, check_finite=False)
        b = G.T @ Sy
        c = np.einsum("ij,ij->j", G, SG)
        m, vv = amp_prior
        quad = a0 - 2 * m * b + m * m * c - vv * (b - m * c) ** 2 / (1 + vv * c)
        ll = -0.5 * (quad + np.log1p(vv * c))
        ll0 = -0.5 * a0
        ll0_abs = -0.5 * (a0 + logdet + n * np.log(2 * np.pi))
        post_amp = ((m / vv + b) / (1 / vv + c), 1 / (1 / vv + c))
        return ll - ll0, ll0_abs, post_amp

    def lh_loglik(self, grid, lh):
        if not lh:
            return np.zeros(len(grid)), 0.0
        probs = {int(k): v_ for k, v_ in self.p["lh"]["positive_prob_by_offset"].items()}
        fp = self.p["lh"]["false_positive"]
        ll = np.zeros(len(grid)); ll0 = 0.0
        for day, positive in lh:
            pr = np.array([probs.get(int(day - O), fp) for O in grid])
            pr = np.clip(pr, 1e-3, 1 - 1e-3)
            ll += np.log(pr if positive else 1 - pr)
            p0 = self.p["lh"]["anovulatory_positive"]
            ll0 += np.log(p0 if positive else 1 - p0)
        return ll, ll0

    def cycle_posterior(self, frame_cycle, grid, prior_o, prior_anov, personal, noise, *, cycle_len=None, today_day=None, lh=None, log_reliability=0.9, baselines=None):
        """frame_cycle: rows of this cycle (cycle_day, signals). Returns posterior dict incl. log evidence."""
        lam = self.p["tempering"]
        logp = np.log(prior_o) + np.log(max(1 - prior_anov, 1e-9))
        logp0 = np.log(max(prior_anov, 1e-9))
        const = 0.0
        amp_post = {}
        nobs = {}
        if isinstance(frame_cycle, pd.DataFrame):
            seg = {k: frame_cycle[k].to_numpy(float) for k in MODEL_SIGNALS}
            seg["cycle_day"] = frame_cycle["cycle_day"].to_numpy(float); seg["_ill"] = frame_cycle["_ill"].to_numpy(bool)
        else:
            seg = frame_cycle
        cdays = seg["cycle_day"]; ill = seg["_ill"]
        for s in MODEL_SIGNALS:
            ys = seg[s]
            use = seg.get("_use_" + s)
            if use is not None:
                # global masks (computed once per request): identical nights under every segmentation hypothesis
                m = use
            else:
                m = np.isfinite(ys) & (cdays >= FIRST_DAY[s]) & ~ill
            days = cdays[m]
            y = ys[m]
            nobs[s] = int(m.sum())
            if use is None and len(y) >= 3:
                r = np.abs(y - np.median(y)) > 4.5 * noise[s]["sd"]
                days, y = days[~r], y[~r]
            ll, ll0_abs, ap = self.signal_loglik(s, days, y, grid, noise[s], personal["amp"][s], cycle_len=cycle_len,
                                                 baseline=None if baselines is None else baselines[s])
            logp = logp + lam * ll
            const += lam * ll0_abs
            amp_post[s] = ap
        if lh:
            ll, ll0 = self.lh_loglik(grid, lh)
            logp, logp0 = logp + ll, logp0 + ll0
        lut = personal["luteal"]
        anl = personal["anov_len"]
        if cycle_len is not None:
            # completed cycle: luteal length likelihood, anovulatory length likelihood
            logp = logp + np.log(discrete_normal(cycle_len - grid, lut["mean"], lut["sd"]))
            logp0 = logp0 + np.log(discrete_normal(np.array([cycle_len]), anl["mean"], anl["sd"]))[0]
            logp = np.where(cycle_len - grid >= 5, logp, -np.inf)
        elif today_day is not None:
            # no new period logged through today_day: L >= today - O  (period could also be unlogged)
            q = 1 - log_reliability
            surv = _sf(today_day - grid - 0.5, lut["mean"], lut["sd"])
            logp = logp + np.log(surv + (1 - surv) * q + 1e-12)
            surv0 = float(_sf(today_day - 0.5, anl["mean"], anl["sd"]))
            logp0 = logp0 + np.log(surv0 + (1 - surv0) * q + 1e-12)
        allp = np.r_[logp, logp0]
        lz = logsumexp(allp)
        post = np.exp(allp - lz)
        return {"grid": grid, "p_o": post[:-1], "p_anov": float(post[-1]), "amp_post": amp_post, "nobs": nobs, "log_evidence": float(lz + const)}

    # ---------- personalisation (empirical Bayes)
    def population_personal(self, age):
        pr = self.p["priors"]
        mu_f = pr["follicular_mean"] + pr["follicular_age_slope"] * ((age if age is not None else 30) - 30)
        base_anov = np.interp(age if age is not None else 30, [20, 35, 40, 45, 55], pr["anov_by_age"])
        return {
            "foll": {"mean": mu_f, "sd": float(np.sqrt(pr["follicular_between_sd"] ** 2 + pr["follicular_within_sd"] ** 2))},
            "foll_mu": (mu_f, pr["follicular_between_sd"] ** 2), "foll_within": pr["follicular_within_sd"],
            "luteal": {"mean": pr["luteal_mean"], "sd": float(np.sqrt(pr["luteal_between_sd"] ** 2 + pr["luteal_within_sd"] ** 2))},
            "anov": float(base_anov), "anov_len": {"mean": pr["follicular_mean"] + pr["luteal_mean"] + 4, "sd": 7.0},
            "amp": {s: (self.p["amplitude"][s]["mean"], self.p["amplitude"][s]["between_var"] + self.p["amplitude"][s]["within_var"]) for s in MODEL_SIGNALS},
            "n_cycles": 0, "n_ovulatory_equiv": 0.0,
        }

    def update_personal(self, age, completed):
        """completed: list of dicts with E_o, V_o, p_ovul, length, amp_est{s:(mean,var)}"""
        pr = self.p["priors"]
        per = self.population_personal(age)
        if not completed:
            return per
        w = np.array([c["p_ovul"] for c in completed]); E = np.array([c["E_o"] for c in completed]); V = np.array([c["V_o"] for c in completed])
        Lc = np.array([c["length"] for c in completed], float)
        sw_pop = pr["follicular_within_sd"]; k0 = pr["within_sd_pseudo_cycles"]
        # personal within-SD (follicular), robust to an occasional outlier cycle
        if w.sum() >= 1.5:
            mu_hat = np.sum(w * E) / w.sum()
            dev2 = np.minimum((E - mu_hat) ** 2, (3.5 * max(sw_pop, 2.0)) ** 2)
            s2 = (k0 * sw_pop ** 2 + np.sum(w * np.maximum(dev2 - V * 0.5, 0))) / (k0 + w.sum())
        else:
            s2 = sw_pop ** 2
        sw = float(np.sqrt(s2))
        m0, v0 = per["foll_mu"]
        prec = 1 / v0 + np.sum(w / (s2 + V)); mean = (m0 / v0 + np.sum(w * E / (s2 + V))) / prec
        per["foll"] = {"mean": float(mean), "sd": float(np.sqrt(s2 + 1 / prec))}
        # luteal
        L = Lc - E
        lw_pop = pr["luteal_within_sd"]
        if w.sum() >= 1.5:
            lmu = np.sum(w * L) / w.sum()
            ldev2 = np.minimum((L - lmu) ** 2, 16.0)
            ls2 = (k0 * lw_pop ** 2 + np.sum(w * np.maximum(ldev2 - V * 0.5, 0))) / (k0 + w.sum())
        else:
            ls2 = lw_pop ** 2
        lv0 = pr["luteal_between_sd"] ** 2
        lprec = 1 / lv0 + np.sum(w / (ls2 + V)); lmean = (pr["luteal_mean"] / lv0 + np.sum(w * L / (ls2 + V))) / lprec
        per["luteal"] = {"mean": float(np.clip(lmean, 8, 17)), "sd": float(np.clip(np.sqrt(ls2 + 1 / lprec), 0.8, 4.0))}
        # anovulation rate: Beta(prior mean*20) + soft counts
        a = per["anov"] * pr["anov_prior_strength"] + np.sum(1 - w); b = (1 - per["anov"]) * pr["anov_prior_strength"] + np.sum(w)
        per["anov"] = float(np.clip(a / (a + b), 0.02, 0.6))
        # anovulatory cycle length tracks personal cycle length
        per["anov_len"] = {"mean": float(per["foll"]["mean"] + per["luteal"]["mean"] + 4), "sd": 7.0}
        # amplitudes: hierarchical normal (between person + within-person cycle variation)
        for s in MODEL_SIGNALS:
            A = self.p["amplitude"][s]
            ests = [(c["amp_est"][s][0], c["amp_est"][s][1], c["p_ovul"]) for c in completed if c["amp_est"].get(s) is not None]
            if not ests:
                continue
            mb, vb, vw = A["mean"], A["between_var"], A["within_var"]
            prec = 1 / vb + sum(wi / (vw + vi) for _, vi, wi in ests)
            mean = (mb / vb + sum(wi * ai / (vw + vi) for ai, vi, wi in ests)) / prec
            per["amp"][s] = (float(mean), float(vw + 1 / prec))
        per["n_cycles"] = len(completed); per["n_ovulatory_equiv"] = float(w.sum())
        return per

    # ---------- whole-history analysis
    def segment_cycles(self, dates, starts):
        """Logged starts -> list of (start_idx, end_idx_exclusive or None, length or None, quality)."""
        idx = [int(np.searchsorted(dates, s)) for s in starts]
        segs = []
        for i, s in enumerate(idx):
            if i + 1 < len(idx):
                L = idx[i + 1] - s
                q = "ok" if 18 <= L <= 60 else ("too_short" if L < 18 else "too_long_possible_missed_log")
                segs.append((s, idx[i + 1], L, q))
            else:
                segs.append((s, None, None, "current"))
        return segs

    def grid_for(self, length=None, today_day=None):
        if length is not None:
            return np.arange(O_MIN, max(O_MIN + 1, length - 4))
        hi = max(45, (today_day or 1) + 20)
        return np.arange(O_MIN, min(hi, 80) + 1)

    # ---------- segment posteriors and unlogged-period inference
    def _post(self, arr, s, e, per, noise, lh_idx, baselines, log_rel):
        """Posterior for segment [s, e) as a completed cycle (e given) or as the current cycle (e None)."""
        end = arr["n"] if e is None else e
        L = end - s
        fc = {k: arr[k][s:end] for k in MODEL_SIGNALS}
        for k in MODEL_SIGNALS:
            fc["_use_" + k] = arr["_use_" + k][s:end]
        fc["_ill"] = arr["_ill"][s:end]; fc["cycle_day"] = np.arange(1, L + 1, dtype=float)
        grid = self.grid_for(length=L) if e is not None else self.grid_for(today_day=L)
        # prior NOT renormalised on the (truncated) grid: mass of impossible O values is excluded, so log evidence stays
        # comparable between segmentation hypotheses
        prior = discrete_normal(grid, per["foll"]["mean"], per["foll"]["sd"])
        cl = [(i - s + 1, r) for i, r in lh_idx if s <= i < end]
        return self.cycle_posterior(fc, grid, prior, per["anov"], per, noise, cycle_len=L if e is not None else None,
                                    today_day=L if e is None else None, lh=cl, log_reliability=log_rel, baselines=baselines)

    def _split(self, arr, s, e, per, noise, lh_idx, baselines, log_rel, min_len=18):
        """Probability that an unlogged period start exists inside [s, e); returns (best_index, probability)."""
        end = arr["n"] if e is None else e
        q = max(1 - log_rel, 0.02)
        base = self._post(arr, s, e, per, noise, lh_idx, baselines, log_rel)["log_evidence"]
        hi = end - (3 if e is None else min_len)
        cands = list(range(s + min_len, hi + 1))
        if not cands:
            return None, 0.0
        evs = []
        for n in cands:
            a = self._post(arr, s, n, per, noise, lh_idx, baselines, log_rel)["log_evidence"]
            b = self._post(arr, n, e, per, noise, lh_idx, baselines, log_rel)["log_evidence"]
            evs.append(a + b + np.log(q))
        evs = np.asarray(evs)
        allv = np.r_[base, evs]
        w = np.exp(allv - logsumexp(allv))
        k = int(np.argmax(w[1:]))
        return cands[k], float(w[1:].sum())

    def expected_cycle(self, per):
        return per["foll"]["mean"] + per["luteal"]["mean"], float(np.sqrt(per["foll"]["sd"] ** 2 + per["luteal"]["sd"] ** 2))

    def run(self, daily, starts, as_of, *, age=None, lh=None, log_reliability=0.9, max_history_cycles=12, infer_unlogged=True):
        """daily: DataFrame with date + signals (continuous calendar up to as_of). starts: sorted Timestamps."""
        frame = to_model_frame(daily).reset_index(drop=True)
        dates = frame["date"].to_numpy("datetime64[D]")
        frame["_ill"] = self.illness_mask(frame)
        arr = {k: frame[k].to_numpy(float) for k in MODEL_SIGNALS}
        arr["_ill"] = frame["_ill"].to_numpy(bool); arr["n"] = len(frame)
        clean = frame.loc[~frame["_ill"]]
        noise = self.person_noise(clean)
        baselines = {sname: (float(clean[sname].median()) if clean[sname].notna().sum() >= 5 else None) for sname in MODEL_SIGNALS}
        lh = lh or []
        lh_idx = [(int(np.searchsorted(dates, np.datetime64(d, "D"))), bool(r)) for d, r in lh if np.datetime64(d, "D") >= dates[0]]
        segs = self.segment_cycles(dates, [np.datetime64(s, "D") for s in starts])
        logged_idx = [sg[0] for sg in segs if 0 <= sg[0] < len(frame)]
        any_use = np.zeros(len(frame), bool)
        for sname in MODEL_SIGNALS:
            y = arr[sname]
            use = np.isfinite(y) & ~arr["_ill"]
            for i0 in logged_idx:  # early-cycle nights after a LOGGED start (residual luteal carry-over)
                use[i0:i0 + FIRST_DAY[sname] - 1] = False
            if baselines[sname] is not None:
                lim = 5 * noise[sname]["sd"] + abs(self.p["amplitude"][sname]["mean"])
                use &= ~(np.abs(np.nan_to_num(y, nan=baselines[sname]) - baselines[sname]) > lim)
            arr["_use_" + sname] = use
            any_use |= use
        arr["_any"] = any_use
        completed, history, inferred = [], [], []
        current = None

        def add_completed(s, e, inferred_flag=False):
            L = e - s
            per = self.update_personal(age, completed[-max_history_cycles:])
            post = self._post(arr, s, e, per, noise, lh_idx, baselines, log_reliability)
            grid = post["grid"]; pov = 1 - post["p_anov"]
            po = post["p_o"] / max(post["p_o"].sum(), 1e-300)
            E = float(np.sum(po * grid)); Var = float(np.sum(po * (grid - E) ** 2))
            amp_est = {}
            for sname, ap in post["amp_post"].items():
                if ap is None:
                    amp_est[sname] = None
                else:
                    am = float(np.sum(po * ap[0])); av = float(np.sum(po * (ap[1] + (ap[0] - am) ** 2)))
                    amp_est[sname] = (am, av)
            completed.append({"E_o": E, "V_o": max(Var, 0.25), "p_ovul": pov, "length": L, "amp_est": amp_est})
            mode = int(grid[np.argmax(po)])
            seg_any = arr["_any"][s:e]
            history.append({"start_index": s, "length": L, "quality": "inferred_start" if inferred_flag else "ok", "p_ovulatory": pov,
                            "ovulation_day_mean": E, "ovulation_day_sd": float(np.sqrt(Var)), "ovulation_day_mode": mode,
                            "max3": float(max3(po)), "nobs": post["nobs"], "temp_shift": amp_est.get("temperature_deviation"),
                            "nights_before": int(seg_any[:max(mode - 1, 0)].sum()), "nights_after": int(seg_any[mode:].sum())})

        for (s, e, L, qlt) in segs:
            if qlt == "current":
                current = s
                continue
            if s < 0 or (e is not None and e > len(frame)) or s >= len(frame):
                continue
            if qlt == "too_short":
                history.append({"start_index": s, "length": L, "quality": qlt})
                continue
            per = self.update_personal(age, completed[-max_history_cycles:])
            mu, sd = self.expected_cycle(per)
            if infer_unlogged and (qlt != "ok" or L > mu + max(10, 3 * sd)) and 36 <= L <= MAX_SPLIT_SEGMENT:
                n, pr = self._split(arr, s, e, per, noise, lh_idx, baselines, log_reliability)
                if n is not None and pr >= 0.8:
                    inferred.append({"index": n, "probability": pr, "where": "history"})
                    add_completed(s, n, False); add_completed(n, e, True)
                    continue
            if qlt != "ok":
                history.append({"start_index": s, "length": L, "quality": qlt})
                continue
            add_completed(s, e)
        per = self.update_personal(age, completed[-max_history_cycles:])
        out = {"history": history, "personal": per, "noise": noise, "frame": frame, "dates": dates, "current": None,
               "illness_nights_recent": int(frame["_ill"].tail(14).sum()), "inferred_starts": inferred, "baselines": baselines}
        if current is None:
            return out
        if len(frame) - current > MAX_SPLIT_SEGMENT:
            # anchor too old for a cycle: ask for the latest period start instead of guessing many unlogged cycles
            out["anchor_stale"] = True
            return out
        s = current; s_inferred = False
        for _ in range(2):
            mu, sd = self.expected_cycle(per)
            today_day = len(frame) - s
            if not infer_unlogged or today_day < max(mu + max(6, 2 * sd), 30):
                break
            n, pr = self._split(arr, s, None, per, noise, lh_idx, baselines, log_reliability)
            if n is None or pr < 0.8:
                break
            inferred.append({"index": n, "probability": pr, "where": "current"})
            add_completed(s, n, s_inferred)
            per = self.update_personal(age, completed[-max_history_cycles:])
            s = n; s_inferred = True
        out["personal"] = per
        post = self._post(arr, s, None, per, noise, lh_idx, baselines, log_reliability)
        post.update(start_index=s, today_day=len(frame) - s)
        out["current"] = post
        return out


def max3(p):
    if len(p) < 3:
        return float(np.sum(p))
    return float(np.max(np.convolve(p, np.ones(3), mode="valid")))
