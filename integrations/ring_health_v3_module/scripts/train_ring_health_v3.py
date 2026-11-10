"""Ring Health v3 reproducible pipeline.

python -m tools.train_ring_health_v3 cycle      # fit cycle engine, validate thresholds, test all scenarios
python -m tools.train_ring_health_v3 menopause  # NHANES fit, generator checks, VMS/stage/FMP models, test
python -m tools.train_ring_health_v3 real       # real-data benchmarks (FedCycleData, TempID)
Outputs: models/exported/ring_health/v3, research/experiments/ring_health/v3
Selection rules are fixed in code (validation only); test people never influence choices.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ring_health_v3.params import registry  # noqa: E402

OUT_MODELS = ROOT / "models/exported/ring_health/v3"
OUT_EXP = ROOT / "research/experiments/ring_health/v3"
FEDCYCLE = ROOT / "data/external/ring_health/fedcycle/rawFedCycleData.csv"
TEMPID = ROOT / "data/external/ring_health/tempid/temperature_ovulation_dataset.csv"
NHANES_DIR = ROOT / "data/external/ring_health/nhanes"
SEEDS = {"cycle_train": 101, "cycle_val": 20000, "cycle_val_anov": 30000, "cycle_test": 900000, "cycle_test_anov": 950000,
         "meno_night": 7000, "meno_train": 71000, "meno_val": 72000, "meno_test": 73000}
TEMPERING = 0.7   # compared 0.7 vs 1.0 on validation (window AUROC equal; detection 94.5% vs 90.1% at <=10% false alarms)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def select_realtime(d):
    """max within2(all ovulatory) s.t. false detection <= 10% on anovulatory-enriched validation."""
    best = []
    for a, b, c in itertools.product([0.9, 0.95, 0.98, 0.99], [0.5, 0.6, 0.7, 0.8], [3, 4, 5]):
        det = (d.p_post2 >= a) & (d.conc >= b) & (d.nights_after >= c) & (d.days_since_temp.fillna(99) <= 5)
        rows = []
        for (pid, cy), g in d.assign(det=det).groupby(["pid", "cycle"]):
            dd = g[g.det].sort_values("date"); ovu = g.ovulatory.iloc[0]
            # the deployed rule reports the posterior MODE when detected
            err = (dd.ov_mode_date.iloc[-1] - dd.true_ov_date.iloc[-1]).days if (len(dd) and ovu) else np.nan
            rows.append((pid.endswith("_anov"), ovu, len(dd) > 0, err))
        c_ = pd.DataFrame(rows, columns=["enr", "ovu", "det", "err"])
        ov = c_[c_.ovu & ~c_.enr]; an = c_[~c_.ovu]
        best.append(dict(detect_post_mass=a, detect_concentration=b, detect_min_temp_nights_after=c, det=ov.det.mean(), fa=an.det.mean(),
                         w2=((ov.err.abs() <= 2) & ov.det).mean()))
    B = pd.DataFrame(best)
    ok = B[B.fa <= 0.10].sort_values(["w2", "det"], ascending=False)
    return ok.iloc[0].to_dict(), B


def select_window(d):
    real = d[~d.pid.str.endswith("_anov")]
    res = []
    for tw in np.round(np.arange(0.15, 0.56, 0.05), 2):
        f = real.p_fertile >= tw; y = real.fertile
        tp = (f & y).sum(); sens = tp / y.sum(); prec = tp / max(f.sum(), 1)
        res.append((tw, 2 * prec * sens / (prec + sens)))
    return float(max(res, key=lambda x: x[1])[0]), res


def select_retro(h):
    h = h[h.logged_offset.abs() <= 2].copy()
    h["evid"] = (h.phys_nobs >= 8) & (h.nights_before >= 3) & (h.nights_after >= 3)
    ov = h[h.ovulatory]; an = h[~h.ovulatory]
    rows = []
    for p, c in itertools.product([0.8, 0.85, 0.9, 0.95, 0.98], [0.4, 0.5, 0.6, 0.7]):
        det = (ov.p_ovulatory >= p) & (ov.max3 >= c) & ov.evid; fa = ((an.p_ovulatory >= p) & (an.max3 >= c) & an.evid).mean()
        rows.append(dict(history_detect_p_ovulatory=p, history_detect_concentration=c, det=det.mean(), fa=fa, w2all=((ov.err_days.abs() <= 2) & det).mean()))
    R = pd.DataFrame(rows)
    return R[R.fa <= 0.10].sort_values("w2all", ascending=False).iloc[0].to_dict(), R


def cycle_stage():
    from src.ring_health_v3.eval_cycle import SCENARIOS, run_many
    from src.ring_health_v3.metrics_cycle import boot_ci, cycle_metrics, day_metrics, retro_metrics
    from src.ring_health_v3.train_cycle import build_params, make_people, real_priors
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    log("real priors from FedCycleData")
    pri = real_priors(FEDCYCLE); jdump(pri, OUT_EXP / "real_priors_fedcycle.json")
    log("simulate 300 training women x 13 cycles and fit templates/noise")
    train = make_people(300, SEEDS["cycle_train"])
    P = build_params(train, pri, [0.07, 0.07, 0.10, 0.15, 0.25]); P["tempering"] = TEMPERING
    log("validation runs (100 realistic + 60 anovulatory-enriched women)")
    d1, h1 = run_many(100, SEEDS["cycle_val"], "realistic", P)
    d2, h2 = run_many(60, SEEDS["cycle_val_anov"], "realistic", P, anov=0.35)
    d = pd.concat([d1, d2.assign(pid=d2.pid + "_anov")]); h = pd.concat([h1, h2.assign(pid=h2.pid + "_anov")])
    d = d[d.status == "ok"]
    rt, grid_rt = select_realtime(d); tw, grid_w = select_window(d); rr, grid_r = select_retro(h)
    P["thresholds"] = {"window_probability": tw, "detect_post_mass": rt["detect_post_mass"], "detect_concentration": rt["detect_concentration"],
                       "detect_min_temp_nights_after": int(rt["detect_min_temp_nights_after"]), "history_detect_p_ovulatory": rr["history_detect_p_ovulatory"],
                       "history_detect_concentration": rr["history_detect_concentration"], "anovulatory_flag": 0.70, "history_min_signal_nights": 8}
    jdump({"realtime_choice": rt, "window_threshold": tw, "window_grid": grid_w, "retro_choice": rr}, OUT_EXP / "validation_selection.json")
    grid_rt.to_csv(OUT_EXP / "validation_realtime_grid.csv", index=False); grid_r.to_csv(OUT_EXP / "validation_retro_grid.csv", index=False)
    OUT_MODELS.mkdir(parents=True, exist_ok=True); jdump(P, OUT_MODELS / "cycle_engine_params.json")
    log("thresholds", P["thresholds"])
    results = {}
    th = P["thresholds"]
    for sc in ["realistic", "irregular", "age40_45"]:
        n = 250 if sc == "realistic" else 80
        log("test scenario", sc, n)
        dd, hh = run_many(n, SEEDS["cycle_test"] + 1000 * SCENARIOS.index(sc), sc, P)
        dm = day_metrics(dd); cm, cyc = cycle_metrics(dd)
        rm = retro_metrics(hh, th["history_detect_p_ovulatory"], th["history_detect_concentration"]) if len(hh) else {}
        ok = dd[dd.status == "ok"]
        if sc == "realistic":
            dm["auroc_ci95_person_bootstrap"] = boot_ci(ok.reset_index(drop=True), ok.pid.to_numpy(), lambda x: __import__("sklearn.metrics").metrics.roc_auc_score(x.fertile, x.p_fertile), n=200)
            dd.to_pickle(OUT_EXP / "test_realistic_days.pkl"); hh.to_pickle(OUT_EXP / "test_realistic_history.pkl"); cyc.to_csv(OUT_EXP / "test_realistic_cycles.csv", index=False)
        results[sc] = {"women": n, "days": dm, "cycles": cm, "retrospective": rm,
                       "anchor_correct_fraction": float(ok.anchor_correct.mean()) if len(ok) else None,
                       "status_counts": dd.status.value_counts().to_dict()}
        jdump(results, OUT_EXP / "test_cycle_metrics.json")
    log("anovulatory-enriched test set (false detections)")
    dd, hh = run_many(100, SEEDS["cycle_test_anov"], "realistic", P, anov=0.35)
    cm, _ = cycle_metrics(dd); rm = retro_metrics(hh, th["history_detect_p_ovulatory"], th["history_detect_concentration"])
    results["anovulatory_enriched"] = {"women": 100, "cycles": cm, "retrospective": rm}
    jdump(results, OUT_EXP / "test_cycle_metrics.json")
    log("cycle stage done")


def paired_stage(n=100):
    """Masking scenarios on the SAME simulated women; paired deltas vs 'realistic' with person-level bootstrap CIs.
    'mismatch' uses deliberately misspecified physiology (engine unchanged) as an upper-bound check."""
    from sklearn.metrics import roc_auc_score
    from src.ring_health_v3.eval_cycle import run_many
    from src.ring_health_v3.metrics_cycle import cycle_metrics, day_metrics, retro_metrics
    P = json.loads((OUT_MODELS / "cycle_engine_params.json").read_text()); th = P["thresholds"]
    out, store = {}, {}
    import os
    n = int(os.getenv("PAIRED_N", n))
    scen = os.getenv("PAIRED_SC", "realistic,missing30,missing50,block7_ovulation,temperature_only,no_temperature,cold_start,calendar_only,missed_logs,lh_users,mismatch").split(",")
    for sc in scen:
        log("paired scenario", sc, n)
        dd, hh = run_many(n, SEEDS["cycle_test"] + 777, sc, P)
        dd = dd[dd.status == "ok"].copy(); dd["w"] = dd.pid.str.rsplit("_", n=1).str[-1].astype(int)
        cm, cyc = cycle_metrics(dd)
        rm = retro_metrics(hh, th["history_detect_p_ovulatory"], th["history_detect_concentration"]) if len(hh) else {}
        if len(hh):
            hh = hh.copy(); hh["w"] = hh.pid.str.rsplit("_", n=1).str[-1].astype(int)
        cyc["w"] = cyc.pid.str.rsplit("_", n=1).str[-1].astype(int)
        store[sc] = (dd, hh, cyc)
        out[sc] = {"women": n, "days": day_metrics(dd), "cycles": cm, "retrospective": rm}
        jdump(out, OUT_EXP / "test_cycle_metrics_paired.json")
    # paired bootstrap deltas (same women, resampled by woman index)
    rng = np.random.default_rng(5)
    base_d, base_h, base_c = store["realistic"]
    women = np.unique(base_d.w)

    def stats(d, h, c, ws):
        dsel = pd.concat([d[d.w == x] for x in ws]); auc = roc_auc_score(dsel.fertile, dsel.p_fertile)
        csel = pd.concat([c[c.w == x] for x in ws]); ov = csel[csel.ovulatory]; rt = float(ov.detected.mean()) if len(ov) else np.nan
        r2 = np.nan
        if h is not None and len(h):
            hs = pd.concat([h[h.w == x] for x in ws]); hs = hs[hs.logged_offset.abs() <= 2]
            ev = (hs.phys_nobs >= 8) & (hs.nights_before >= 3) & (hs.nights_after >= 3)
            det = (hs.p_ovulatory >= th["history_detect_p_ovulatory"]) & (hs.max3 >= th["history_detect_concentration"]) & ev
            o = hs.ovulatory; r2 = float(((hs.err_days.abs() <= 2) & det)[o].mean())
        return np.array([auc, rt, r2])
    deltas = {}
    for sc, (d, h, c) in store.items():
        if sc == "realistic":
            continue
        point = stats(d, h, c, women) - stats(base_d, base_h, base_c, women)
        boots = []
        for _ in range(200):
            ws = rng.choice(women, len(women), replace=True)
            boots.append(stats(d, h, c, ws) - stats(base_d, base_h, base_c, ws))
            if len(boots) >= 100:
                break
        b = np.array(boots)
        deltas[sc] = {k: {"delta": float(point[i]), "ci95": [float(np.nanquantile(b[:, i], 0.025)), float(np.nanquantile(b[:, i], 0.975))]}
                      for i, k in enumerate(["window_auroc", "realtime_detection", "history_within2_all"])}
    out["_paired_deltas_vs_realistic"] = deltas
    jdump(out, OUT_EXP / "test_cycle_metrics_paired.json")
    log("paired done")


def real_stage():
    from src.ring_health_v3.real_benchmarks import fedcycle_calendar, tempid_benchmark, tempid_device_cv
    P = json.loads((OUT_MODELS / "cycle_engine_params.json").read_text())
    _, fc = fedcycle_calendar(FEDCYCLE, P)
    _, t0 = tempid_benchmark(TEMPID, P)
    _, t1 = tempid_device_cv(TEMPID, P)
    jdump({"fedcycle_calendar_prospective": fc, "tempid_oura_profile": t0, "tempid_device_profile_cv": t1}, OUT_EXP / "real_benchmarks.json")
    log("real stage done", json.dumps({"fedcycle": {k: fc[k] for k in ["ovulation_day_mae", "cycle_length_mae", "calendar_fertile_day_auroc"]}, "tempid_cv_mae": t1["mae"]}))


def menopause_stage():
    from sklearn.metrics import roc_auc_score
    from src.ring_health_v3 import train_menopause as tm
    from src.ring_health_v3.nhanes_anm import fit as nhanes_fit
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    nh, rows = nhanes_fit(NHANES_DIR); jdump(nh, OUT_EXP / "nhanes_age_at_menopause.json"); jdump(nh, OUT_MODELS / "nhanes_age_at_menopause.json")
    from src.ring_health_v3.sim_menopause import AMH, FMPAgeSampler, generator_checks
    chk = generator_checks(FMPAgeSampler(nh), n=8000); chk["amh_parameters"] = AMH
    jdump(chk, OUT_EXP / "generator_checks.json")
    log("generator checks", chk["fmp_age"], "Finkelstein MAE", round(chk["finkelstein_mae"], 3))
    P = json.loads((OUT_MODELS / "cycle_engine_params.json").read_text())
    log("night-level VMS model: simulate 500 women")
    _, nights = tm.simulate_table(500, SEEDS["meno_night"], 3, "random", P, nh, None, want_nights=True)
    night_model = tm.fit_night_model(nights)
    log("simulate train/val/test landmark tables")
    train, _ = tm.simulate_table(3000, SEEDS["meno_train"], 4, "random", P, nh, night_model)
    val, _ = tm.simulate_table(700, SEEDS["meno_val"], 4, "random", P, nh, night_model)
    models, rep = tm.train_all(train, val)
    routing, routing_detail = tm.choose_routing(models, val)
    log("routing (validation)", routing)
    joblib.dump({"night_model": night_model, "night_features": tm.NIGHT_FEATURES}, OUT_MODELS / "menopause_vms_night.joblib")
    joblib.dump({"stage": {k: v for k, v in models["stage"].items()}, "fmp": models["fmp"], "features": tm.FEATURES, "horizons": tm.HORIZONS, "routing": routing},
                OUT_MODELS / "menopause_models.joblib")
    res = {"validation": rep, "routing": routing, "routing_validation_detail": routing_detail, "test": {}, "test_deployed_routed": {}}
    for mode in ["random", "full", "ring_only", "lpd_ring", "logs_only", "age_only", "full_plus_labs"]:
        log("test mode", mode)
        test, _ = tm.simulate_table(1000, SEEDS["meno_test"], 3, mode, P, nh, night_model)
        res["test"][mode] = {name: tm.evaluate(models, test, name) for name in tm.feature_sets()}
        res["test_deployed_routed"][mode] = tm.evaluate_routed(models, routing, test)
        if mode == "random":
            test.to_pickle(OUT_EXP / "menopause_test_random.pkl")
            # night VMS model: held-out night-level evaluation
        jdump(res, OUT_EXP / "test_menopause_metrics.json")
    _, nt = tm.simulate_table(150, SEEDS["meno_test"] + 5, 3, "full", P, nh, None, want_nights=True)
    pr = night_model.predict_proba(nt[tm.NIGHT_FEATURES])[:, 1]
    res["vms_night_model_test"] = {"nights": int(len(nt)), "prevalence": float(nt.flash.mean()), "auroc": float(roc_auc_score(nt.flash, pr))}
    jdump(res, OUT_EXP / "test_menopause_metrics.json")
    log("menopause stage done")


def manifest():
    import sklearn
    arts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT_MODELS.glob("*")) if p.is_file()}
    m = {"version": "ring-health-v3.0.0", "sklearn": sklearn.__version__, "numpy": np.__version__, "artifacts": arts, "seeds": SEEDS, "tempering": TEMPERING,
         "training_data": "synthetic (literature + real-data-fitted parameters); real data used for priors/calibration/benchmarks only",
         "clinical_validation": "not_established"}
    jdump(m, ROOT / "models/registry/ring_health_v3.json")
    jdump(registry(), OUT_EXP / "parameter_registry.json")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("cycle", "all"):
        cycle_stage()
    if stage == "paired":
        paired_stage()
    if stage in ("real", "all"):
        real_stage()
    if stage in ("menopause", "all"):
        menopause_stage()
    manifest()
