"""Metrics for prospective (daily) and retrospective (per cycle) fertility evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def boot_ci(values, groups, fn, n=300, seed=7):
    """Person-level bootstrap CI for fn(sub-dataframe)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        try:
            out.append(fn(values.iloc[rows]))
        except Exception:
            pass
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))] if out else None


def day_metrics(d, threshold_col="in_window"):
    d = d[d.status == "ok"]
    y = d.fertile.astype(int).to_numpy(); p = d.p_fertile.to_numpy(); f = d[threshold_col].astype(bool).to_numpy()
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum()); fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return {"days": int(len(d)), "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None, "brier": float(brier_score_loss(y, p)),
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": 2 * prec * sens / max(prec + sens, 1e-9),
            "accuracy": (tp + tn) / max(len(d), 1)}


def cycle_metrics(d):
    d = d[d.status == "ok"]
    rows = []
    for (pid, cyc), g in d.groupby(["pid", "cycle"]):
        g = g.sort_values("date")
        ovul = bool(g.ovulatory.iloc[0])
        det = g[g.ov_status == "detected"]
        first = det.iloc[0] if len(det) else None
        final = det.iloc[-1] if len(det) else None
        true_O = g.true_ov_day.iloc[0]
        r = {"pid": pid, "cycle": cyc, "ovulatory": ovul, "detected": first is not None}
        if ovul:
            r["delay_days"] = float(first.true_cycle_day - true_O) if first is not None else np.nan
            r["first_err"] = float((first.ov_date_est - first.true_ov_date).days) if first is not None else np.nan
            r["final_err"] = float((final.ov_date_est - final.true_ov_date).days) if final is not None else np.nan
            fw = g[g.fertile]; flagged = int(fw.in_window.sum())
            r["window_days_flagged"] = flagged; r["window_overlap_ge3"] = flagged >= 3; r["window_any"] = flagged >= 1
            # prospective ovulation-date prediction error at cycle day 5 and at O-3
            for tag, day in [("pred_err_day5", 5), ("pred_err_Om3", int(true_O) - 3)]:
                row = g[g.true_cycle_day == day]
                r[tag] = float((row.ov_date_est.iloc[0] - row.true_ov_date.iloc[0]).days) if len(row) and pd.notna(row.ov_date_est.iloc[0]) else np.nan
            post = g[g.true_cycle_day == int(true_O) + 3]
            r["next_err_Op3"] = float((post.next_est.iloc[0] - post.next_true.iloc[0]).days) if len(post) else np.nan
        row = g[g.true_cycle_day == 3]
        r["next_err_day3"] = float((row.next_est.iloc[0] - row.next_true.iloc[0]).days) if len(row) else np.nan
        last = g.iloc[-1]
        r["false_detection"] = (not ovul) and first is not None
        rows.append(r)
    c = pd.DataFrame(rows)
    ov = c[c.ovulatory]; an = c[~c.ovulatory]
    m = {"cycles": int(len(c)), "ovulatory_cycles": int(len(ov)), "anovulatory_cycles": int(len(an)),
         "realtime_detection_rate": float(ov.detected.mean()) if len(ov) else None,
         "realtime_final_mae": float(ov.final_err.abs().mean()) if ov.detected.any() else None,
         "realtime_final_within2_of_all": float((ov.final_err.abs() <= 2).mean()) if len(ov) else None,
         "realtime_median_delay_days": float(ov.delay_days.median()) if ov.detected.any() else None,
         "anovulatory_false_detection": float(an.detected.mean()) if len(an) else None,
         "window_cycles_overlap_ge3": float(ov.window_overlap_ge3.mean()) if len(ov) else None,
         "window_cycles_any": float(ov.window_any.mean()) if len(ov) else None,
         "ovulation_prediction_mae_day5": float(ov.pred_err_day5.abs().mean()) if len(ov) else None,
         "ovulation_prediction_mae_Om3": float(ov.pred_err_Om3.abs().mean()) if len(ov) else None,
         "next_period_mae_day3": float(c.next_err_day3.abs().mean()),
         "next_period_within2_day3": float((c.next_err_day3.abs() <= 2).mean()),
         "next_period_mae_after_ovulation": float(ov.next_err_Op3.abs().mean()) if len(ov) else None,
         "next_period_within2_after_ovulation": float((ov.next_err_Op3.abs() <= 2).mean()) if len(ov) else None}
    return m, c


def retro_metrics(h, p_thr, c_thr, min_nights=8):
    if h is None or not len(h):
        return {}
    h = h[h.logged_offset.abs() <= 2].copy()
    ev = h["phys_nobs"] if "phys_nobs" in h else h["temp_nobs"]
    h["evid"] = ev >= min_nights
    if "nights_before" in h:
        h["evid"] &= (h["nights_before"] >= 3) & (h["nights_after"] >= 3)
    ov = h[h.ovulatory]; an = h[~h.ovulatory]
    det = (ov.p_ovulatory >= p_thr) & (ov.max3 >= c_thr) & ov.evid
    err = ov.err_days.abs()
    return {"cycles": int(len(h)), "ovulatory": int(len(ov)), "anovulatory": int(len(an)),
            "detection_rate": float(det.mean()), "mae_detected": float(err[det].mean()) if det.any() else None,
            "within1_detected": float((err[det] <= 1).mean()) if det.any() else None,
            "within2_detected": float((err[det] <= 2).mean()) if det.any() else None,
            "within2_all_ovulatory": float(((err <= 2) & det).mean()),
            "anovulatory_false_detection": float(((an.p_ovulatory >= p_thr) & (an.max3 >= c_thr) & an.evid).mean()) if len(an) else None}
