"""Build demo request/response JSON pairs for app integration (synthetic people; never real users)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ring_health_v3.engine import NIGHT_COLS, RingHealthV3  # noqa: E402
from src.ring_health_v3.sim_cycles import sample_person, simulate_person  # noqa: E402
from src.ring_health_v3.sim_menopause import FMPAgeSampler, bleeding_sequence, sample_woman, window  # noqa: E402

OUT = ROOT / "data/samples/ring_health/v3"


def nights(df):
    rows = []
    for r in df.to_dict(orient="records"):
        x = {"date": str(pd.Timestamp(r["date"]).date())}
        for c in NIGHT_COLS:
            v = r.get(c)
            x[c] = None if v is None or pd.isna(v) else float(v)
        rows.append(x)
    return rows


def profile(age, **kw):
    return {"age": age, **kw}


def fertility_examples():
    ex = {}
    for seed in range(424242, 424242 + 500):  # pick a typical demo profile: ~28-d cycles, recent cycles ovulatory
        rng = np.random.default_rng(seed)
        per = sample_person(rng, age=31.0, irregular=False, overrides={"adherence": 0.95, "long_gap": False, "log_reliability": 1.0})
        if not (14.0 <= per["mu_f"] <= 16.0 and per["sd_f"] <= 2.0 and 12.5 <= per["mu_l"] <= 14 and per["temp_amp"] >= 0.25):
            continue
        sim = simulate_person(rng, n_cycles=7, person=per)
        if all(c["ovulation"] is not None for c in sim["cycles"][-3:]):
            break
    t = sim["truth"]
    last = t[t.cycle_index == 6]
    as_of = last.date.iloc[int(last.ovulation_day.iloc[0]) + 2] if last.ovulatory.iloc[0] else last.date.iloc[16]
    hist = sim["daily"][sim["daily"].date <= as_of]
    starts = [str(s.date()) for s in sim["logged_starts"] if s <= as_of]
    base = {"as_of": str(as_of.date()), "modules": ["fertility"], "profile": profile(31), "cycle": {"period_starts": starts, "period_logs_complete": True}}
    ex["fertility_regular"] = {**base, "nights": nights(hist)}
    fw = last.date.iloc[max(0, int(last.ovulation_day.iloc[0]) - 3)] if last.ovulatory.iloc[0] else last.date.iloc[12]
    h2 = sim["daily"][sim["daily"].date <= fw]
    ex["fertility_in_window"] = {**base, "as_of": str(fw.date()), "cycle": {"period_starts": [s for s in starts if pd.Timestamp(s) <= fw], "period_logs_complete": True}, "nights": nights(h2)}
    tmp = hist.copy(); tmp[["resting_hr", "hrv_ms", "respiratory_rate"]] = np.nan
    ex["fertility_temperature_only"] = {**base, "nights": nights(tmp)}
    gap = hist.copy()
    if last.ovulatory.iloc[0]:
        c = last.index[0] + int(last.ovulation_day.iloc[0]) - 1
        gap.loc[(gap.index >= c - 3) & (gap.index <= c + 2), NIGHT_COLS] = np.nan
    ex["fertility_gap_around_ovulation"] = {**base, "nights": nights(gap)}
    cs = pd.Timestamp(starts[-1])
    ex["fertility_new_user"] = {**base, "cycle": {"period_starts": [starts[-1]]}, "nights": nights(hist[hist.date >= cs - pd.Timedelta(days=5)])}
    ex["fertility_calendar_only"] = {**base, "nights": []}
    if last.ovulatory.iloc[0]:
        a6 = last.date.iloc[min(len(last) - 1, int(last.ovulation_day.iloc[0]) + 5)]
        ex["fertility_ovulation_confirmed"] = {**base, "as_of": str(a6.date()), "nights": nights(sim["daily"][sim["daily"].date <= a6])}
    lh = [{"date": str(d.date()), "result": "positive" if r else "negative"} for d, r in sim["lh"] if d <= as_of]
    if not lh and last.ovulatory.iloc[0]:
        O = int(last.ovulation_day.iloc[0])
        lh = [{"date": str((last.date.iloc[0] + pd.Timedelta(days=O - 1 + k)).date()), "result": "positive" if k == -1 else "negative"} for k in [-3, -2, -1, 0]]
    ex["fertility_with_lh_tests"] = {**base, "lh_tests": lh, "nights": nights(hist)}
    # Oura API v2 format
    sleep, ready = [], []
    ok = lambda v: v is not None and not pd.isna(v)
    for r in hist.to_dict(orient="records"):
        day = str(pd.Timestamp(r["date"]).date())
        if ok(r["temperature_deviation"]):
            ready.append({"day": day, "temperature_deviation": float(r["temperature_deviation"])})
        if ok(r["resting_hr"]):
            sleep.append({"day": day, "type": "long_sleep", "lowest_heart_rate": float(r["resting_hr"]), "average_hrv": float(r["hrv_ms"]) if ok(r["hrv_ms"]) else None,
                          "average_breath": float(r["respiratory_rate"]) if ok(r["respiratory_rate"]) else None,
                          "total_sleep_duration": float(r["sleep_minutes"] * 60) if ok(r["sleep_minutes"]) else None,
                          "efficiency": float(r["sleep_efficiency"]) if ok(r["sleep_efficiency"]) else None,
                          "awake_time": float(r["awake_minutes"] * 60) if ok(r["awake_minutes"]) else None})
    ex["fertility_oura_format"] = {**base, "oura": {"sleep": sleep, "daily_readiness": ready}}
    return ex


def menopause_examples():
    ex = {}
    S = FMPAgeSampler(json.loads((ROOT / "models/exported/ring_health/v3/nhanes_age_at_menopause.json").read_text()))
    for name, target_stage, seed in [("menopause_late_transition", "late_transition", 5), ("menopause_postmenopause", "postmenopause", 9), ("menopause_reproductive_45", "reproductive", 3)]:
        rng = np.random.default_rng(seed)
        for _ in range(500):
            w = sample_woman(rng, S); cyc, fmp = bleeding_sequence(rng, w)
            L = {"late_transition": fmp - 0.6 * w["d_late"], "postmenopause": fmp + 1.8, "reproductive": 45.0}[target_stage]
            if target_stage == "reproductive" and fmp - 45 < w["d_total"] + 1:
                continue
            if target_stage != "reproductive" and not (44 < L < 58):
                continue
            win = window(rng, w, cyc, fmp, L, 180)
            last_true = max(c["start_age"] for c in cyc if c["start_age"] <= L)
            lpd = win["as_of"] - pd.Timedelta(days=int(round((L - last_true) * 365.25)))
            req = {"as_of": str(win["as_of"].date()), "modules": ["menopause"], "profile": profile(int(L), smoker=bool(w["smoker"]), bmi=round(w["bmi"], 1)),
                   "cycle": {"period_starts": [str(s.date()) for s in win["logged_starts"]], "last_period_date": str(lpd.date()), "logging_since": str((win["as_of"] - pd.Timedelta(days=win["log_history_days"])).date())},
                   "symptoms": {"logging_enabled": True, "hot_flash_dates": [str(s.date()) for s in win["symptom_dates"]]}, "nights": nights(win["daily"])}
            ex[name] = req
            break
    lt = json.loads(json.dumps(ex["menopause_late_transition"]))
    lt["labs"] = {"fsh_iu_l": {"value": 38.0, "date": lt["as_of"]}}
    ex["menopause_late_transition_with_fsh"] = lt
    nolog = json.loads(json.dumps(ex["menopause_late_transition"])); nolog["cycle"] = {}
    ex["menopause_ring_only"] = nolog
    ins = {"as_of": "2031-01-01", "modules": ["menopause"], "profile": profile(48), "nights": []}
    ex["menopause_insufficient"] = ins
    return ex


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = RingHealthV3()
    ex = {**fertility_examples(), **menopause_examples()}
    summary = {}
    for name, req in ex.items():
        res = eng.analyze(req)
        (OUT / f"{name}_input.json").write_text(json.dumps(req, indent=1))
        (OUT / f"{name}_output.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        mod = req["modules"][0]
        summary[name] = {k: res[mod].get(k) for k in (["status", "phase", "cycle_day"] if mod == "fertility" else ["status"])}
        if mod == "menopause" and res[mod].get("stage"):
            summary[name]["stage"] = res[mod]["stage"]["label"]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
