"""Generate markdown result tables for the v3 report from experiment JSON (no manual transcription)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research/experiments/ring_health/v3"


def f(x, p=3):
    return "—" if x is None else f"{x:.{p}f}".replace(".", ",")


def pc(x):
    return "—" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def ci(c):
    return "" if not c else f" [{f(c[0])}–{f(c[1])}]"


def cycle_main():
    r = json.loads((EXP / "test_cycle_metrics.json").read_text())
    v = r["realistic"]; d = v["days"]; c = v["cycles"]; t = v["retrospective"]; a = r.get("anovulatory_enriched", {})
    rows = ["| Метрика | Значение |", "|---|---|",
            f"| Фертильный день сегодня: AUROC / AP / Brier | {f(d['auroc'])}{ci(d.get('auroc_ci95_person_bootstrap'))} / {f(d['average_precision'])} / {f(d['brier'])} |",
            f"| Чувствительность / специфичность / точность | {f(d['sensitivity'], 2)} / {f(d['specificity'], 2)} / {f(d['precision'], 2)} |",
            f"| Окно: ≥3 из 6 дней отмечены / хотя бы 1 | {pc(c['window_cycles_overlap_ge3'])} / {pc(c['window_cycles_any'])} циклов |",
            f"| Подтверждение овуляции в реальном времени: доля циклов / MAE итоговой даты / медианная задержка | {pc(c['realtime_detection_rate'])} / {f(c['realtime_final_mae'], 2)} дня / {f(c['realtime_median_delay_days'], 0)} дн. |",
            f"| Ложное подтверждение в ановуляторных циклах: реальное время / история | {pc(c['anovulatory_false_detection'])} ({c['anovulatory_cycles']} циклов) / {pc(t['anovulatory_false_detection'])} ({t['anovulatory']} циклов) |",
            f"| История ({t['ovulatory']} овуляторных циклов): найдено / MAE / ±1 / ±2 | {pc(t['detection_rate'])} / {f(t['mae_detected'], 2)} / {pc(t['within1_detected'])} / {pc(t['within2_detected'])} (всех: {pc(t['within2_all_ovulatory'])}) |",
            f"| Прогноз даты овуляции на 5-й день / за 3 дня до неё, MAE | {f(c['ovulation_prediction_mae_day5'], 1)} / {f(c['ovulation_prediction_mae_Om3'], 1)} дня |",
            f"| Следующие месячные, MAE: день 3 / через 3 дня после овуляции; в пределах ±2 | {f(c['next_period_mae_day3'], 2)} / {f(c['next_period_mae_after_ovulation'], 2)} дня; {pc(c['next_period_within2_day3'])} / {pc(c['next_period_within2_after_ovulation'])} |"]
    if a:
        rows.append(f"| Тест с обогащением ановуляцией ({a['retrospective']['anovulatory']} ановуляторных циклов в истории): ложное подтверждение история / реальное время; обнаружение овуляторных | {pc(a['retrospective']['anovulatory_false_detection'])} / {pc(a['cycles']['anovulatory_false_detection'])}; {pc(a['retrospective']['detection_rate'])} |")
    sub = ["", "| Подгруппа | AUROC окна | Чувств. / специф. | Окно ≥3 дн. | История: найдено / MAE / ±2 всех | Месячные после овуляции, MAE |", "|---|---|---|---|---|---|"]
    for k, name in [("age40_45", "Возраст 40–45"), ("irregular", "Нерегулярные циклы")]:
        v = r[k]; t = v["retrospective"]
        sub.append(f"| {name} | {f(v['days']['auroc'])} | {f(v['days']['sensitivity'], 2)} / {f(v['days']['specificity'], 2)} | {pc(v['cycles']['window_cycles_overlap_ge3'])} | {pc(t['detection_rate'])} / {f(t['mae_detected'], 2)} / {pc(t['within2_all_ovulatory'])} | {f(v['cycles']['next_period_mae_after_ovulation'], 2)} |")
    return "\n".join(rows + sub)


def paired():
    r = json.loads((EXP / "test_cycle_metrics_paired.json").read_text())
    names = {"realistic": "реалистичные пропуски (база)", "missing30": "+30% ночей пропущено", "missing50": "+50% ночей пропущено", "block7_ovulation": "7 ночей подряд вокруг овуляции",
             "temperature_only": "только температура", "no_temperature": "без температуры (ЧСС, HRV, дыхание)", "cold_start": "новая пользовательница: 1 дата, без истории",
             "calendar_only": "нет данных кольца (календарь)", "missed_logs": "25% месячных не залогировано", "lh_users": "+ LH-тесты",
             "mismatch": "**физиология не совпадает с обучением** (температура ×0,7, шум ×1,3, ЧСС ×0,6, HRV/дыхание ×0,5, лютеиновая −1 день, ановуляция ×2)"}
    dl = r.get("_paired_deltas_vs_realistic", {})
    rows = ["| Сценарий (одни и те же 100 женщин) | AUROC окна (Δ к базе, 95% ДИ) | Чувств. / специф. | Подтверждение в реальном времени | История: найдено / MAE / ±2 всех (Δ ±2) |", "|---|---|---|---|---|"]
    for k, nm in names.items():
        if k not in r:
            continue
        v = r[k]; d = v["days"]; c = v["cycles"]; t = v.get("retrospective") or {}
        da = dl.get(k, {}).get("window_auroc"); dh = dl.get(k, {}).get("history_within2_all")
        dtxt = "" if not da else f" ({'+' if da['delta'] >= 0 else '−'}{f(abs(da['delta']))}, {f(da['ci95'][0])}…{f(da['ci95'][1])})"
        htxt = "" if not dh or dh["delta"] != dh["delta"] else f" ({'+' if dh['delta'] >= 0 else '−'}{f(abs(dh['delta']) * 100, 1)} п.п.)"
        hist = "— (нет физиологии: только календарная оценка)" if k == "calendar_only" else ("—" if not t.get("detection_rate") else f"{pc(t['detection_rate'])} / {f(t.get('mae_detected'), 2)} / {pc(t.get('within2_all_ovulatory'))}{htxt}")
        rows.append(f"| {nm} | {f(d['auroc'])}{dtxt} | {f(d['sensitivity'], 2)} / {f(d['specificity'], 2)} | {pc(c['realtime_detection_rate'])} | {hist} |")
    return "\n".join(rows)


def meno():
    r = json.loads((EXP / "test_menopause_metrics.json").read_text())
    dep = r["test_deployed_routed"]; t = r["test"]
    names = [("age_only", "Только возраст"), ("ring_only", "Только кольцо (без логов и даты)"), ("lpd_ring", "Кольцо + дата последней менструации (без логов)"),
             ("logs_only", "Логи + дата, без кольца"), ("full", "Кольцо + логи + дата + симптомы"), ("full_plus_labs", "+ ФСГ/АМГ и ковариаты"), ("random", "Смесь, как в приложении (случайная доступность)")]
    rows = ["| Доступные входы (развёрнутая система: маршрутизация + правило STRAW) | FMP 12 / 24 / 36 мес, AUROC | C-индекс | Медиана ошибки, мес | Стадия macro-F1 (4 кл. / 3 группы) | Recall: репр. / ранний / поздний / пост |", "|---|---|---|---|---|---|"]
    for k, nm in names:
        if k not in dep:
            continue
        m = dep[k]; rc = m["stage_recall"]
        rows.append(f"| {nm} | {f(m['fmp_auc_12m_pre_fmp_42_56'], 2)} / {f(m['fmp_auc_24m_pre_fmp_42_56'], 2)} / {f(m['fmp_auc_36m_pre_fmp_42_56'], 2)} | {f(m['c_index_time_to_fmp_pre'], 2)} | {f(m['median_abs_error_months_pre'], 1)} | {f(m['stage_macro_f1'], 2)} / {f(m['stage_grouped3_macro_f1'], 2)} | {f(rc['reproductive'], 2)} / {f(rc['early_transition'], 2)} / {f(rc['late_transition'], 2)} / {f(rc['postmenopause'], 2)} |")
    rows += ["", f"Маршрутизация, выбранная на валидации: {json.dumps(r['routing'], ensure_ascii=False)}.",
             f"Ложное «постменопауза подтверждена» у женщин до FMP (смесь входов): {pc(dep['random']['straw_rule_false_confirmations_pre_fmp'])}.",
             f"Модель ночного прилива: AUROC {f(r['vms_night_model_test']['auroc'])} на {r['vms_night_model_test']['nights']} ночах (доля ночей с приливами {pc(r['vms_night_model_test']['prevalence'])})."]
    # contribution of the ring when logs exist: full vs age_bleeding models on the 'full' availability mode
    fu = t["full"]
    rows.append(f"Вклад кольца при наличии логов (режим «всё», без лабораторий): FMP AUROC 12/24/36 полная модель {f(fu['full']['fmp_auc_12m_pre_fmp_42_56'], 2)}/{f(fu['full']['fmp_auc_24m_pre_fmp_42_56'], 2)}/{f(fu['full']['fmp_auc_36m_pre_fmp_42_56'], 2)} против «возраст + кровотечения» {f(fu['age_bleeding']['fmp_auc_12m_pre_fmp_42_56'], 2)}/{f(fu['age_bleeding']['fmp_auc_24m_pre_fmp_42_56'], 2)}/{f(fu['age_bleeding']['fmp_auc_36m_pre_fmp_42_56'], 2)}; стадия F1 {f(fu['full']['stage_macro_f1'], 2)} против {f(fu['age_bleeding']['stage_macro_f1'], 2)}.")
    return "\n".join(rows)


def real():
    r = json.loads((EXP / "real_benchmarks.json").read_text())
    fc = r["fedcycle_calendar_prospective"]; t1 = r["tempid_device_profile_cv"]; t0 = r["tempid_oura_profile"]
    return (f"| Проверка | Результат |\n|---|---|\n"
            f"| FedCycleData: календарный прогноз следующего цикла (априоры leave-one-woman-out), {fc['cycles_scored']} циклов, {fc['women']} женщин | Овуляция MAE {f(fc['ovulation_day_mae'], 2)} дня (±2: {pc(fc['ovulation_within2'])}); длина цикла MAE {f(fc['cycle_length_mae'], 2)} (±2: {pc(fc['cycle_length_within2'])}); календарное окно AUROC {f(fc['calendar_fertile_day_auroc'])}, чувств./специф. {f(fc['calendar_fertile_sensitivity_at_0.30'], 2)}/{f(fc['calendar_fertile_specificity_at_0.30'], 2)} |\n"
            f"| TempID ({t1['series']} циклов, {t1['people_by_prefix']} человек): профиль Oura / профиль устройства из CV по людям | MAE {f(t0['mae_all'], 1)} / {f(t1['mae'], 1)} дня; ±2: {pc(t0['within2_all'])} / {pc(t1['within2'])}; подогнанная амплитуда {f(t1['fitted_amplitude_mean_c'], 2)} °C при шуме {f(t1['fitted_noise_sd_c'], 2)} °C; лаг {t1['fitted_lag_days']} |")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["cycle_main", "paired", "meno", "real"]:
        print(f"<!-- {name} -->"); print(globals()[name]()); print()
