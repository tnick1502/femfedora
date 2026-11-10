"""10 realistic synthetic demo users (different ages, goals, logging habits, health profiles).
Physiology comes from the v3 generators; behaviour (wear, logging, symptoms, labs, Oura export) is scripted per persona.
All dates are re-based so that 'as_of' = AS_OF. Never real people."""
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

AS_OF = pd.Timestamp("2026-11-09")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/ring_health_v3_module/demo_users"


def rebase(dates, last):
    return [pd.Timestamp(d) + (AS_OF - pd.Timestamp(last)) for d in dates]


def nights(df):
    rows = []
    for r in df.to_dict(orient="records"):
        x = {"date": str(pd.Timestamp(r["date"]).date())}
        for c in NIGHT_COLS:
            v = r.get(c)
            x[c] = None if v is None or pd.isna(v) else (int(v) if c in ("resting_hr", "hrv_ms", "sleep_minutes", "sleep_efficiency", "awake_minutes") else float(v))
        rows.append(x)
    return rows


def iso(d):
    return str(pd.Timestamp(d).date())


def to_oura(rows):
    sleep, ready = [], []
    for r in rows:
        if r["temperature_deviation"] is not None:
            ready.append({"id": f"rd-{r['date']}", "day": r["date"], "score": None, "temperature_deviation": r["temperature_deviation"], "temperature_trend_deviation": None})
        if r["resting_hr"] is not None:
            sleep.append({"id": f"sl-{r['date']}", "day": r["date"], "type": "long_sleep", "lowest_heart_rate": r["resting_hr"], "average_hrv": r["hrv_ms"],
                          "average_breath": r["respiratory_rate"], "total_sleep_duration": None if r["sleep_minutes"] is None else r["sleep_minutes"] * 60,
                          "efficiency": r["sleep_efficiency"], "awake_time": None if r["awake_minutes"] is None else r["awake_minutes"] * 60})
            if np.random.default_rng(len(r["date"]) + int(r["date"][-2:])).random() < 0.08:  # occasional nap documents (ignored by adapter)
                sleep.append({"id": f"nap-{r['date']}", "day": r["date"], "type": "rest", "lowest_heart_rate": r["resting_hr"] + 6, "total_sleep_duration": 1500})
    return {"sleep": sleep, "daily_readiness": ready}


def cycle_user(seed, age, n_cycles, person_over, day_in_last_cycle, history_days=None, irregular=False):
    """Simulate; cut at a cycle day of the LAST cycle; rebase dates to AS_OF."""
    for s in range(seed, seed + 400):
        rng = np.random.default_rng(s)
        per = sample_person(rng, age=age, irregular=irregular, overrides=person_over)
        sim = simulate_person(rng, n_cycles=n_cycles, person=per)
        t = sim["truth"]; last = t[t.cycle_index == n_cycles - 1]
        if not last.ovulatory.iloc[0]:
            continue
        cd = day_in_last_cycle(last)
        if cd is None or cd > len(last):
            continue
        idx = last.index[0] + cd - 1
        d = sim["daily"].iloc[:idx + 1].copy()
        if history_days:
            d = d.iloc[-history_days:]
        shift = AS_OF - d.date.iloc[-1]
        d["date"] = d.date + shift
        starts = [s_ + shift for s_ in sim["logged_starts"] if s_ + shift <= AS_OF]
        lh = [(dd + shift, r) for dd, r in sim["lh"] if dd + shift <= AS_OF]
        truth = {"true_ovulation_current_cycle": iso(last.date.iloc[0] + shift + pd.Timedelta(days=int(last.ovulation_day.iloc[0]) - 1)) if last.ovulatory.iloc[0] else None,
                 "true_current_cycle_start": iso(last.date.iloc[0] + shift), "cycle_lengths": [c["length"] for c in sim["cycles"]],
                 "ovulation_days": [c["ovulation"] for c in sim["cycles"]]}
        return d, starts, lh, per, truth
    raise RuntimeError("no persona sample")


def meno_user(seed, target, age_hint, over=None, n_days=180):
    S = FMPAgeSampler(json.loads((ROOT / "models/exported/ring_health/v3/nhanes_age_at_menopause.json").read_text()))
    for s in range(seed, seed + 3000):
        rng = np.random.default_rng(s)
        w = sample_woman(rng, S, overrides=over or {})
        cyc, fmp = bleeding_sequence(rng, w)
        L = target(w, fmp)
        if L is None or not (age_hint - 0.9 <= L <= age_hint + 0.9):
            continue
        win = window(rng, w, cyc, fmp, L, n_days)
        last_true = max([c["start_age"] for c in cyc if c["start_age"] <= L] or [np.nan])
        shift = AS_OF - win["as_of"]
        d = win["daily"].copy(); d["date"] = d.date + shift
        truth = {"age_at_fmp": round(fmp, 2), "years_to_fmp": round(fmp - L, 2), "stage": ("postmenopause" if L >= fmp else ("late_transition" if fmp - L <= w["d_late"] else ("early_transition" if fmp - L <= w["d_total"] else "reproductive"))),
                 "vms_class": w["vms_class"], "nights_with_hot_flash_last30": int((win["flashes"][-30:] > 0).sum())}
        all_starts = [AS_OF - pd.Timedelta(days=int(round((L - c["start_age"]) * 365.25))) for c in cyc if c["start_age"] <= L]
        return d, win, all_starts, last_true, L, w, shift, truth
    raise RuntimeError("no menopause persona")


def build_users(off):
    users = []

    # 1. Anna 24: TTC, diligent, LH tests, today inside fertile window
    d, st, lh, per, tr = cycle_user(1001 + off, 24.0, 7, {"adherence": 0.97, "long_gap": False, "log_reliability": 1.0, "lh_user": True, "irregular": False},
                                    lambda c: None if c is None else int(c.ovulation_day.iloc[0]) - 2)
    users.append(("01_anna_24_ttc_lh", {"name": "Анна", "age": 24, "goal": "планирует беременность, отслеживает фертильное окно",
        "behaviour": "носит кольцо почти каждую ночь (~97%), отмечает каждую менструацию в день начала, делает LH-тесты (тест-полоски) в середине цикла, профиль заполнен полностью",
        "health": "здорова, ИМТ 21, не курит, не рожала, цикл регулярный"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 24, "smoker": False, "bmi": 21.0, "parous": False, "college_education": True},
         "cycle": {"period_starts": [iso(x) for x in st], "period_logs_complete": True, "logging_since": iso(d.date.iloc[0]), "menses_length": 5},
         "lh_tests": [{"date": iso(x), "result": "positive" if r else "negative"} for x, r in lh], "nights": nights(d)}, tr))

    # 2. Maria 29: irregular 31-38 d cycles, charging gaps, ovulation confirmed a week ago
    d, st, lh, per, tr = cycle_user(2002 + off, 29.0, 6, {"adherence": 0.82, "long_gap": False, "log_reliability": 1.0, "gap_mean": 2.0, "mu_f": 20.0, "sd_f": 3.0},
                                    lambda c: None if c is None else int(c.ovulation_day.iloc[0]) + 7)
    users.append(("02_maria_29_long_cycles", {"name": "Мария", "age": 29, "goal": "хочет понимать свой цикл, планирует беременность через год",
        "behaviour": "снимает кольцо на зарядку раз в 4-5 дней (пропуски по 1-3 ночи, ~82% ночей), месячные отмечает, LH-тесты не делает, ИМТ не указала",
        "health": "длинные циклы с поздней овуляцией, без диагнозов"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 29, "smoker": False},
         "cycle": {"period_starts": [iso(x) for x in st], "period_logs_complete": True}, "nights": nights(d)}, tr))

    # 3. Ekaterina 33: regular, forgot to log the last period -> engine suggests it; data arrive as Oura API export
    d, st, lh, per, tr = cycle_user(3003 + off, 33.0, 6, {"adherence": 0.93, "long_gap": False, "log_reliability": 1.0, "irregular": False},
                                    lambda c: None if c is None else 9)
    st_logged = st[:-1]  # forgot to log the current period
    rows = nights(d)
    users.append(("03_ekaterina_33_forgot_period_oura", {"name": "Екатерина", "age": 33, "goal": "следит за циклом и сном, данные приходят прямо из Oura",
        "behaviour": "данные передаются в формате Oura API (sleep + daily_readiness), последнюю менструацию (9 дней назад) забыла отметить; в sleep встречаются дневные 'rest'-сессии",
        "health": "один ребёнок, ИМТ 24, регулярный цикл"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 33, "parous": True, "bmi": 24.0},
         "cycle": {"period_starts": [iso(x) for x in st_logged], "period_logs_complete": False}, "oura": to_oura(rows)}, tr))

    # 4. Darya 27: new user, 12 nights of ring data, one period date from onboarding
    d, st, lh, per, tr = cycle_user(4004 + off, 27.0, 5, {"adherence": 0.95, "long_gap": False, "log_reliability": 1.0},
                                    lambda c: None if c is None else 12, history_days=12)
    users.append(("04_darya_27_new_user", {"name": "Дарья", "age": 27, "goal": "только что купила кольцо, хочет видеть прогноз месячных и окна",
        "behaviour": "кольцо 12 ночей, при онбординге указала только дату начала последней менструации, профиль почти пустой",
        "health": "без особенностей"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 27}, "cycle": {"period_starts": [iso(st[-1])]}, "nights": nights(d)}, tr))

    # 5. Olga 36: cold with fever for 3 nights in this cycle; LH not used
    d, st, lh, per, tr = cycle_user(5005 + off, 36.0, 6, {"adherence": 0.9, "long_gap": False, "log_reliability": 1.0, "irregular": False},
                                    lambda c: None if c is None else int(c.ovulation_day.iloc[0]) - 6)
    d = d.reset_index(drop=True); k = len(d) - 5
    d.loc[k:k + 2, "temperature_deviation"] = (d.loc[k:k + 2, "temperature_deviation"] + [0.9, 1.3, 0.8]).round(2)
    d.loc[k:k + 2, "resting_hr"] = d.loc[k:k + 2, "resting_hr"] + [8, 11, 7]
    d.loc[k:k + 2, "hrv_ms"] = (d.loc[k:k + 2, "hrv_ms"] * 0.7).round(0)
    d.loc[k:k + 2, "respiratory_rate"] = (d.loc[k:k + 2, "respiratory_rate"] + 1.1).round(1)
    users.append(("05_olga_36_fever", {"name": "Ольга", "age": 36, "goal": "отслеживает цикл, двое детей, не планирует беременность (без гормональной контрацепции)",
        "behaviour": "носит кольцо ~90% ночей; 3 ночи назад заболела ОРВИ с температурой - в данных скачок температуры кожи и пульса",
        "health": "двое детей, ИМТ 26, не курит"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 36, "parous": True, "bmi": 26.0, "smoker": False},
         "cycle": {"period_starts": [iso(x) for x in st], "period_logs_complete": True, "menses_length": 4}, "nights": nights(d)}, tr))

    # 6. Irina 41: both modules; cycles shortening, one anovulatory-looking cycle, symptoms logging on
    d, st, lh, per, tr = cycle_user(6006 + off, 41.0, 8, {"adherence": 0.88, "long_gap": False, "log_reliability": 1.0, "mu_f": 13.0, "anov_rate": 0.18},
                                    lambda c: None if c is None else 16)
    users.append(("06_irina_41_cycle_and_perimenopause", {"name": "Ирина", "age": 41, "goal": "замечает, что циклы стали короче; хочет понимать цикл и начало перименопаузы",
        "behaviour": "кольцо ~88% ночей, месячные отмечает, ведёт дневник симптомов (приливов пока нет), сдавала ФСГ полгода назад - не указала",
        "health": "двое детей, ИМТ 23, высшее образование, циклы стали короче"},
        {"as_of": iso(AS_OF), "modules": ["fertility", "menopause"], "profile": {"age": 41, "parous": True, "bmi": 23.0, "smoker": False, "college_education": True},
         "cycle": {"period_starts": [iso(x) for x in st], "period_logs_complete": True, "logging_since": iso(d.date.iloc[0]), "last_period_date": iso(st[-1])},
         "symptoms": {"logging_enabled": True, "hot_flash_dates": [], "night_sweat_dates": []}, "nights": nights(d)}, tr))

    # 7. Elena 46: menopause tracking, late transition, irregular periods logged, hot flashes logged
    d, win, all_st, last_true, L, w, shift, tr = meno_user(7007 + off, lambda w, f: f - 0.55 * w["d_late"] if w["vms_class"] in ("early_onset", "high") else None, 46.5,
                                                         over={"symptom_logger": True})
    logged = [s for s in all_st if (AS_OF - s).days <= 365]
    hf = [x + shift for x in win["symptom_dates"]]
    users.append(("07_elena_46_late_transition", {"name": "Елена", "age": 46, "goal": "отслеживает менопаузальный переход: нерегулярные месячные и приливы",
        "behaviour": "кольцо 6 месяцев, месячные отмечает последний год, отмечает приливы и ночную потливость в дневнике",
        "health": f"ИМТ {w['bmi']:.0f}, {'курит' if w['smoker'] else 'не курит'}, рожала"},
        {"as_of": iso(AS_OF), "modules": ["menopause"], "profile": {"age": 46, "parous": True, "smoker": bool(w["smoker"]), "bmi": round(w["bmi"], 1)},
         "cycle": {"period_starts": [iso(x) for x in logged], "logging_since": iso(AS_OF - pd.Timedelta(days=365)), "last_period_date": iso(logged[-1]) if logged else None},
         "symptoms": {"logging_enabled": True, "hot_flash_dates": [iso(x) for x in hf], "night_sweat_dates": [iso(x) for x in hf[::3]]}, "nights": nights(d)}, tr))

    # 8. Natalya 49: no period logging, onboarding date only, FSH lab, smoker, ring 4 months
    d, win, all_st, last_true, L, w, shift, tr = meno_user(8008 + off, lambda w, f: f - 0.4 if w["smoker"] == 1 or True else None, 49.0, over={"smoker": 1, "symptom_logger": False}, n_days=120)
    lpd = AS_OF - pd.Timedelta(days=int(round((L - last_true) * 365.25)) + 3)
    users.append(("08_natalya_49_no_logs_fsh", {"name": "Наталья", "age": 49, "goal": "хочет понять, близка ли менопауза",
        "behaviour": "месячные в приложении НЕ ведёт; при онбординге указала дату последней менструации (по памяти, ±несколько дней); кольцо 4 месяца; вписала ФСГ из анализа 2 недели назад; дневник симптомов не ведёт",
        "health": "курит, ИМТ " + f"{w['bmi']:.0f}"},
        {"as_of": iso(AS_OF), "modules": ["menopause"], "profile": {"age": 49, "smoker": True, "bmi": round(w["bmi"], 1)},
         "cycle": {"last_period_date": iso(lpd)}, "labs": {"fsh_iu_l": {"value": round(float(np.clip(np.exp(np.log(38) + 0.3 * np.random.default_rng(8).normal()), 18, 90)), 1), "date": iso(AS_OF - pd.Timedelta(days=14))}},
         "nights": nights(d)}, tr))

    # 9. Svetlana 53: postmenopause (>12 months), persistent hot flashes
    d, win, all_st, last_true, L, w, shift, tr = meno_user(9009 + off, lambda w, f: f + 1.6 if w["vms_class"] in ("late_onset", "high") else None, 53.0, over={"symptom_logger": True}, n_days=90)
    lpd = AS_OF - pd.Timedelta(days=int(round((L - last_true) * 365.25)))
    hf = [x + shift for x in win["symptom_dates"]]
    users.append(("09_svetlana_53_postmenopause", {"name": "Светлана", "age": 53, "goal": "постменопауза, беспокоят приливы и сон",
        "behaviour": "кольцо 3 месяца, дату последней менструации указала при онбординге, отмечает приливы почти каждый день",
        "health": "двое детей, ИМТ " + f"{w['bmi']:.0f}" + ", гипотензия (не влияет на модель)"},
        {"as_of": iso(AS_OF), "modules": ["menopause"], "profile": {"age": 53, "parous": True, "bmi": round(w["bmi"], 1), "smoker": False, "college_education": False},
         "cycle": {"last_period_date": iso(lpd)}, "symptoms": {"logging_enabled": True, "hot_flash_dates": [iso(x) for x in hf], "night_sweat_dates": []}, "nights": nights(d)}, tr))

    # 10. Tatiana 31: stopped the pill 4 months ago, wears the ring rarely, sporadic logging, minimal profile
    d, st, lh, per, tr = cycle_user(1010 + off, 31.0, 5, {"adherence": 0.62, "long_gap": True, "log_reliability": 0.85, "gap_mean": 4.0},
                                    lambda c: None if c is None else 20)
    users.append(("10_tatiana_31_sparse_data", {"name": "Татьяна", "age": 31, "goal": "4 месяца назад отменила оральные контрацептивы, ждёт восстановления цикла",
        "behaviour": "носит кольцо нерегулярно (~60% ночей, был перерыв больше недели), месячные отмечает через раз, профиль не заполнен",
        "health": "без особенностей; гормональная контрацепция сейчас НЕ принимается (иначе модуль вернул бы out_of_scope)"},
        {"as_of": iso(AS_OF), "modules": ["fertility"], "profile": {"age": 31, "hormonal_contraception": False},
         "cycle": {"period_starts": [iso(x) for x in st]}, "nights": nights(d)}, tr))

    return users


def rng_range(lst):
    return f"{min(lst)}-{max(lst)}"


ACCEPT = {
    "01": lambda f, m, t: f["phase"] == "fertile_window" and abs((pd.Timestamp(f["ovulation"]["date"]) - pd.Timestamp(t["true_ovulation_current_cycle"])).days) <= 1 and max(t["cycle_lengths"]) - min(t["cycle_lengths"]) <= 5,
    "02": lambda f, m, t: f["ovulation"]["status"] == "detected" and abs((pd.Timestamp(f["ovulation"]["date"]) - pd.Timestamp(t["true_ovulation_current_cycle"])).days) <= 2 and f["data_quality"]["possible_illness_nights_last14"] == 0 and np.mean(t["cycle_lengths"]) >= 31,
    "03": lambda f, m, t: f["cycle_start_source"] == "inferred_from_physiology" and len(f["unlogged_period_suggestions"]) == 1 and abs((pd.Timestamp(f["unlogged_period_suggestions"][0]["date"]) - pd.Timestamp(t["true_current_cycle_start"])).days) <= 1 and max(t["cycle_lengths"]) - min(t["cycle_lengths"]) <= 6,
    "04": lambda f, m, t: f["status"] == "ok",
    "05": lambda f, m, t: f["data_quality"]["possible_illness_nights_last14"] >= 2 and f["phase"] == "fertile_window",
    "06": lambda f, m, t: f["status"] == "ok" and m["status"] == "ok" and m["stage"]["label"] in ("reproductive", "early_transition") and f["ovulation"]["status"] in ("detected", "likely_passed_not_confirmed"),
    "07": lambda f, m, t: m["stage"]["label"] == "late_transition" and 0.3 <= m["final_menstrual_period"]["probability_by_months"]["12"] <= 0.85,
    "08": lambda f, m, t: m["stage"]["grouped"]["label"] == "transition" and m["final_menstrual_period"]["probability_by_months"]["12"] >= 0.3,
    "09": lambda f, m, t: m["final_menstrual_period"]["already_confirmed"],
    "10": lambda f, m, t: f["status"] == "ok" and f["data_quality"]["confidence"] in ("low", "medium") and f["cycle_start_source"] == "logged" and f["cycle_day"] <= 35,
}


def main():
    eng = RingHealthV3()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for off in range(0, 60):
        for folder, persona, req, truth in build_users(off * 97):
            key = folder[:2]
            if key in chosen:
                continue
            req["cycle"] = {k: v for k, v in req["cycle"].items() if v is not None}
            res = eng.analyze(req)
            try:
                ok = ACCEPT[key](res.get("fertility"), res.get("menopause"), truth)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                if truth.get("cycle_lengths"):
                    persona["health"] += f"; фактические длины последних циклов {rng_range(truth['cycle_lengths'])} дней"
                chosen[key] = (folder, persona, req, truth)
        if len(chosen) == len(ACCEPT):
            break
    users = [chosen[k] for k in sorted(chosen)]
    summary = []
    for folder, persona, req, truth in users:
        req["cycle"] = {k: v for k, v in req["cycle"].items() if v is not None}
        res = eng.analyze(req)
        p = OUT / folder; p.mkdir(parents=True, exist_ok=True)
        (p / "input.json").write_text(json.dumps(req, ensure_ascii=False, indent=1))
        (p / "output.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        (p / "_ground_truth_for_qa.json").write_text(json.dumps({"note": "Synthetic hidden truth for QA only. NEVER send to the API or show as data.", **truth}, ensure_ascii=False, indent=1, default=str))
        persona["folder"] = folder
        (p / "persona.json").write_text(json.dumps(persona, ensure_ascii=False, indent=1))
        line = {"folder": folder, **{k: persona[k] for k in ["name", "age", "goal"]}}
        if "fertility" in res:
            f = res["fertility"]
            line["fertility"] = {k: f.get(k) for k in ["status", "cycle_day", "phase", "cycle_start_source"]}
            if f.get("status") == "ok":
                line["fertility"].update(today=f["fertile_window"]["today_probability"], ovulation=f["ovulation"]["status"], ovulation_date=f["ovulation"]["date"],
                                         next_period=f["next_period"]["date"], confidence=f["data_quality"]["confidence"], suggestions=f["unlogged_period_suggestions"],
                                         illness_nights=f["data_quality"]["possible_illness_nights_last14"])
        if "menopause" in res:
            m = res["menopause"]
            line["menopause"] = {"status": m.get("status")}
            if m.get("status") == "ok":
                line["menopause"].update(stage=m["stage"]["label"], grouped=m["stage"]["grouped"]["label"], basis=m["stage"]["basis"],
                                         fmp12=m["final_menstrual_period"]["probability_by_months"]["12"], median=m["final_menstrual_period"]["median_months"],
                                         vms=m["vasomotor"], model=m["stage"]["model_used"])
        line["truth"] = truth
        summary.append(line)
    (OUT / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
