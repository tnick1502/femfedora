"""Generate the compact PTB notebook sequence from versioned cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/notebooks/preterm_cdc"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


SETUP = '''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
plt.rcParams.update({"figure.dpi": 115, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .16})
BLUE, ORANGE, GOLD, SLATE = "#28628A", "#BA6234", "#B78918", "#3B4855"'''


def save(name: str, cells: list):
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbf.validate(notebook)
    path = OUT / name
    if path.exists():
        existing = nbf.read(path, as_version=4)
        if [(cell.cell_type, cell.source) for cell in existing.cells] == [
            (cell.cell_type, cell.source) for cell in notebook.cells
        ]:
            return  # Keep validated outputs on an unchanged notebook.
    nbf.write(notebook, path)


save("01_prepare_data.ipynb", [
    md('''# CDC PTB · подготовка когорты

## tl;dr

Этот ноутбук собирает одноплодные живорождения резиденток США с известным акушерским сроком при родах. В проверенном прогоне: **3 480 382 записи за 2023** и **3 512 780 за 2024** год. Строка CDC — ребёнок при рождении, а не наблюдаемая по неделям беременность.

## Контекст и методы

Источник: `data/processed/cdc_natality/natality_ml_2023_2024.sqlite`, подготовленный из NCHS/NBER. Контракт: [MODEL_INPUT_CONTRACT.md](MODEL_INPUT_CONTRACT.md). Классы исхода: `<32`, `32–33`, `34–36`, `≥37` завершённых недель. Только одно разбиение 2023 года: train 60%, validation 20%, calibration 20%; 2024 сохраняется целиком для временной проверки. ID матери нет, поэтому разные беременности одной женщины могут попасть в разные части.'''),
    code(SETUP + '\nfrom src.data.preterm_cdc import OUT, BASE_FEATURES, CLASS_NAMES, prepare_all, load_year'),
    md('## Данные · построение и сверка численности'),
    code('manifest = prepare_all()\ncounts = pd.DataFrame({year: info["counts"] for year, info in manifest["years"].items()}).T.fillna(0).astype(int)\ndisplay(counts)\nassert counts.loc["2023", "rows"] > 3_000_000 and counts.loc["2024", "rows"] > 3_000_000'),
    md('## Результаты · исходы и пропуски'),
    code('year_counts = counts[[f"band_{i}" for i in range(4)]].rename(columns={f"band_{i}": CLASS_NAMES[i] for i in range(4)})\nshare = year_counts.div(year_counts.sum(axis=1), axis=0) * 100\ndisplay(share.round(2).rename_axis("Год"))\nfig, ax = plt.subplots(figsize=(8, 4.5))\nshare.plot.bar(stacked=True, ax=ax, color=[ORANGE, GOLD, BLUE, SLATE], width=.65)\nax.set(xlabel="Год родов", ylabel="Доля живорождений, %", title="Срок при одноплодном живорождении · CDC")\nax.legend(title="Завершённые недели", bbox_to_anchor=(1.02, 1), loc="upper left")\nplt.tight_layout(); plt.show()'),
    code('missing = pd.DataFrame({year: info["missing_features"] for year, info in manifest["years"].items()}).T\nmissing_pct = missing.div(counts["rows"], axis=0) * 100\ndisplay(missing_pct.round(2).T.rename(columns={"2023": "2023, %", "2024": "2024, %"}))\nfig, ax = plt.subplots(figsize=(9, 4.5))\nmissing_pct.T.plot.barh(ax=ax, color=[BLUE, ORANGE])\nax.set(xlabel="Пропущено, %", ylabel="Признак", title="Полнота доступных до родов полей")\nax.legend(title="Год"); plt.tight_layout(); plt.show()'),
    md('''## Выводы

Результаты относятся только к одноплодным **живорождениям**. Возраст матери записан при родах и служит приближением более раннего возраста. Многоплодие, мертворождения, самопроизвольность родов и события беременности по неделям этой таблицей не определяются. Следующий ноутбук подбирает параметры только по 2023 году.'''),
])

save("02_train_baselines.ipynb", [
    md('''# CDC PTB · три baseline и подбор параметров

## tl;dr

Сравниваются частота исхода, мультиномиальная логистика, LightGBM и CatBoost. Все обучаемые модели видят одинаковые признаки и строки. В проверенном прогоне на validation 2023 CatBoost получил log loss **0,34729**, LightGBM **0,34745**, частотный контроль **0,36053**; разница двух бустингов мала. Выбор параметров и типа модели делает **один** валидационный набор; фолдов нет.

## Контекст и методы

Обучение кандидатов ограничено воспроизводимой случайной подвыборкой 500 000 строк train, валидация — 200 000 строк validation. В финальном ноутбуке выбранная конфигурация переобучается на всех строках train+validation. Ограниченный поиск содержит три значения `C` логистики и по три заранее заданные конфигурации LightGBM/CatBoost; он не доказывает глобальный оптимум. Early stopping использует только validation 2023. Калибровочная часть 2023 и метки 2024 здесь не используются.'''),
    code(SETUP + '\nfrom src.training.preterm_cdc import run_search'),
    md('## Результаты · единый критерий выбора'),
    code('candidates, selected = run_search()\ncols = ["id", "family", "iterations", "n", "multiclass_log_loss", "multiclass_brier", "auprc_lt32", "auprc_lt34", "auprc_lt37"]\ndisplay(candidates[cols].round(5))\nprint("Выбранный тип:", selected["champion_family"], "; набор:", selected["best_by_family"][selected["champion_family"]]["candidate"]["id"])'),
    code('baseline_loss = candidates.loc[candidates.family.eq("prevalence"), "multiclass_log_loss"].iloc[0]\nplot = candidates.loc[candidates.family.ne("prevalence")].copy()\nplot["gain"] = baseline_loss - plot.multiclass_log_loss\nplot = plot.sort_values("gain")\ncolors = plot.family.map({"logreg": BLUE, "lightgbm": ORANGE, "catboost": GOLD})\nfig, ax = plt.subplots(figsize=(9, 5))\nax.barh(plot.id, plot.gain, color=colors)\nax.set(xlabel="Уменьшение log loss относительно частотного контроля · validation 2023", ylabel="Конфигурация", title="Подбор параметров на одном отложенном наборе")\nplt.tight_layout(); plt.show()'),
    md('''## Выводы

Метрики validation использованы для выбора, поэтому не являются независимой оценкой победителя. Перестановочная важность и абляции ниже также исследовательские на validation; итоговую проверку переноса даёт 2024. Все исходы объединены в одну модель, так что оценки `<32 ≤ <34 ≤ <37` согласованы.'''),
])

save("03_ablations.ipynb", [
    md('''# CDC PTB · абляции групп признаков

## tl;dr

Для выбранного в 02 типа модели по одному удаляются анамнез преждевременных родов, вся акушерская история, BMI, хронические состояния и курение до беременности. В проверенном прогоне удаление всей акушерской истории увеличило log loss на **0,00710**. Каждый вариант заново обучается на тех же строках 2023 года и оценивается на том же validation.

## Контекст и методы

Абляция показывает изменение качества **прогноза** при удалении информации, а не причинное влияние признака. Маска пропуска удаляется вместе со своим полем. Параметры типа модели зафиксированы по 02; нового выбора по абляциям не делаем.'''),
    code(SETUP + '\nfrom src.training.preterm_cdc import run_ablations'),
    md('## Результаты'),
    code('ablations = run_ablations()\ncols = ["ablation", "removed", "n", "multiclass_log_loss", "delta_log_loss_vs_full", "auprc_lt32", "auprc_lt37"]\ndisplay(ablations[cols].round(5))'),
    code('plot = ablations.loc[ablations.ablation.ne("full")].sort_values("delta_log_loss_vs_full")\nfig, ax = plt.subplots(figsize=(9, 4.7))\nax.barh(plot.ablation, plot.delta_log_loss_vs_full, color=BLUE)\nax.axvline(0, color=SLATE, linewidth=1)\nax.set(xlabel="Изменение log loss после удаления; >0 означает ухудшение", ylabel="Удалённая группа", title="Абляции · validation 2023")\nplt.tight_layout(); plt.show()'),
    md('''## Выводы

Validation уже использовалась при подборе модели; эти различия служат диагностикой, без доверительного интервала и без независимого утверждения о пользе поля. Взаимосвязанные признаки способны частично заменять друг друга.'''),
])

save("04_feature_importance.ipynb", [
    md('''# CDC PTB · перестановочная важность

## tl;dr

На фиксированной подвыборке validation 2023 проверяется, насколько ухудшается log loss после перестановки каждого исходного признака. Наибольший средний прирост в проверенном прогоне дал анамнез преждевременных родов: **+0,01016**. Маска пропуска переставляется вместе с полем. Для оценки разброса используются три повторения; это не кросс-валидационные фолды.

## Контекст и методы

Результат относится к выбранной модели и этой выборке из 50 000 записей. Важность не является коэффициентом причинного эффекта. Коррелирующие факторы способны делить вклад между собой.'''),
    code(SETUP + '\nfrom src.training.preterm_cdc import run_importance'),
    md('## Результаты'),
    code('importance = run_importance()\nsummary = importance.groupby("feature").agg(mean_delta=("delta_log_loss", "mean"), sd_delta=("delta_log_loss", "std"), mean_delta_brier=("delta_brier", "mean")).sort_values("mean_delta")\ndisplay(summary.round(5))'),
    code('fig, ax = plt.subplots(figsize=(9, 5.3))\nax.barh(summary.index, summary.mean_delta, xerr=summary.sd_delta, color=BLUE, error_kw={"capsize": 3})\nax.axvline(0, color=SLATE, linewidth=1)\nax.set(xlabel="Рост многоклассового log loss после перестановки", ylabel="Признак", title="Важность на validation 2023 · 3 перестановки")\nplt.tight_layout(); plt.show()'),
    md('''## Выводы

Полоски ошибки показывают разброс **трёх перестановок одного validation-набора**, а не доверительные интервалы по пациенткам. Из-за отсутствия ID матери нельзя гарантировать независимость повторных беременностей.'''),
])

save("05_temporal_validation.ipynb", [
    md('''# CDC PTB · итоговая проверка 2023 → 2024

## tl;dr

Выбранная в 02 конфигурация обучается заново на train+validation 2023 года. Оставшаяся часть 2023 года используется только для температурной калибровки четырёх вероятностей. На полном 2024 году в проверенном прогоне CatBoost получил log loss **0,34673** против **0,36037** у частотного контроля; подобранная температура ≈ **1,00**.

## Контекст и методы

Температура сохраняет сумму четырёх вероятностей равной 1 и порядок накопленных рисков `<32`, `<34`, `<37`. Эта проверка временная, но **не слепая**: исходы 2024 года уже изучались в проекте. Нет ID матери, мертворождений, дат начала родов и достоверного разделения spontaneous/indicated PTB. Результат не является персональным недельным риском.'''),
    code(SETUP + '\nfrom src.data.preterm_cdc import OUT\nfrom src.training.preterm_cdc import run_temporal'),
    md('## Результаты · качество и калибровка'),
    code('metrics, metadata = run_temporal()\ncols = ["dataset", "model", "n", "multiclass_log_loss", "multiclass_brier", "auprc_lt32", "auprc_lt34", "auprc_lt37", "brier_lt32", "brier_lt37"]\ndisplay(metrics[cols].round(5))\nprint("Температура:", round(metadata["temperature"], 3), "; модель:", metadata["family"], "; 2024 записей:", f"{metadata[\'temporal_test_rows\']:,}")'),
    code('test = pd.read_parquet(OUT / "temporal_predictions_2024.parquet")\nprobs = test[[f"class_{i}" for i in range(4)]].to_numpy(dtype=float)\nassert np.allclose(probs.sum(axis=1), 1, atol=2e-6)\nrisk32 = probs[:, 0]; risk34 = probs[:, :2].sum(axis=1); risk37 = probs[:, :3].sum(axis=1)\nassert np.all(risk32 <= risk34 + 1e-8) and np.all(risk34 <= risk37 + 1e-8)\nprint("Проверка согласованности <32 ≤ <34 ≤ <37: пройдена")'),
    code('from sklearn.calibration import calibration_curve\nfig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\nfor ax, threshold, risk in zip(axes, [32, 34, 37], [risk32, risk34, risk37]):\n    observed = (test.risk_band.to_numpy() < {32: 1, 34: 2, 37: 3}[threshold]).astype(int)\n    obs, pred = calibration_curve(observed, risk, n_bins=12, strategy="quantile")\n    ax.plot(pred * 100, obs * 100, "o-", color=BLUE, label="Модель")\n    end = max(pred.max(), obs.max()) * 110\n    ax.plot([0, end], [0, end], "--", color=SLATE, label="Идеальная калибровка")\n    ax.set(xlim=(0, end), ylim=(0, end), xlabel="Средний прогноз, %", ylabel="Фактическая доля, %", title=f"Роды <{threshold} недель · 2024")\naxes[0].legend(); plt.tight_layout(); plt.show()'),
    code('part = metrics.loc[metrics.dataset.eq("2024 temporal")].set_index("model")\nmodel_key = metadata["family"] + "_temperature"\nfig, axes = plt.subplots(1, 2, figsize=(11, 4))\nx = np.arange(3); labels = ["<32", "<34", "<37"]\nfor ax, metric, title in [(axes[0], "auprc", "AUPRC · выше лучше"), (axes[1], "brier", "Brier · ниже лучше")]:\n    control = [part.loc["prevalence", f"{metric}_lt{w}"] for w in [32, 34, 37]]\n    model = [part.loc[model_key, f"{metric}_lt{w}"] for w in [32, 34, 37]]\n    ax.bar(x - .18, control, width=.36, color=SLATE, label="Частота")\n    ax.bar(x + .18, model, width=.36, color=BLUE, label=metadata["family"] )\n    ax.set_xticks(x, labels)\n    ax.set(xlabel="Срок родов", ylabel=metric.upper() if metric == "auprc" else "Brier", title=f"{title} · 2024")\naxes[0].legend(); plt.tight_layout(); plt.show()'),
    md('''## Выводы и пределы применения

Сравнивайте модель с частотным контролем и проверяйте прежде всего log loss, Brier и калибровку. AUPRC зависит от частоты исхода. Порог для клинического решения здесь не выбирался. Модель обучена на записях о живорождении в США; перенос на продолжающуюся беременность и персональные недели требует продольной когорты с точным сроком и типом родов.'''),
])

save("06_woman_week_forecast.ipynb", [
    md('''# CDC PTB · прогноз для женщины на текущем сроке

## tl;dr

Модель принимает **полностью заполненные** данные одной женщины с продолжающейся одноплодной беременностью и выдаёт вероятности родов до 32, 34 и 37 недель, в ближайшие 28 дней и по завершённым неделям. На допустимых для индивидуального прогноза записях 2024 года при сроке 28+0 средний прогноз `<37` составил **8,104%**, наблюдаемая доля — **8,163%**, AUROC — **0,636**. При той же доле отмеченных записей преимущество модели перед правилом из трёх вопросов анамнеза мало.

## Метод

CatBoost из 05 даёт четыре вероятности исхода. На **отдельной calibration-части 2023** оценивается распределение конкретных недель внутри каждого класса для децилей модельного риска `<37`. Затем вероятности нормируются на условие «беременность продолжается на указанном сроке». Ни один параметр не выбирается по 2024 году; фолдов нет.

### Ключевые допущения

Внутри текущей недели день родов условно считается равномерным, потому что CDC хранит только завершённую неделю. Распределение внутри четырёх исходных классов заимствуется из групп с похожим модельным риском. Целый возраст на предполагаемую дату родов приближает целый возраст при фактических родах в исходной базе. Исходы без живорождения отсутствуют.'''),
    code(SETUP + '\nfrom src.data.preterm_cdc import OUT\nfrom src.evaluation.preterm_woman_inputs import audit_input_contract, evaluate_serving_scope\nfrom src.training.preterm_woman import build_week_profile, evaluate_week_profile, export_week_bundle\nfrom src.inference.preterm_woman import predict_woman, get_feature_importance'),
    md('## Входы модели и маски пропусков'),
    code('profile = build_week_profile()\nbundle = export_week_bundle()\ninput_audit = audit_input_contract()\nassert all(v == 0 for year in input_audit["missing_mask_mismatches"].values() for v in year.values())\nassert input_audit["prediction_input_parity_mismatches"] == 0\nprint("Совпадение входных признаков:", input_audit["prediction_input_parity_rows"], "проверенных записей, 0 ошибок")\nprint("Маски источника:", sum(input_audit["missing_mask_mismatches"]["2023"].values()) + sum(input_audit["missing_mask_mismatches"]["2024"].values()), "расхождений на полном 2023+2024")\nprint("17 признаков:", ", ".join(bundle["feature_order"]))\nprint("Для нового прогноза незаполненное поле → INSUFFICIENT_DATA; возраст на предполагаемую дату родов — целое число")'),
    md('## Точность на допустимом входе · 2024'),
    code('serving = evaluate_serving_scope()\nrows = []\nfor horizon, item in serving["horizons"].items():\n    rows.append({"исход": horizon, "записей": serving["eligible_2024_landmark_rows"],\n                 "наблюдалось": item["observed_rate"], "прогноз": item["mean_predicted"],\n                 "AUROC": item["auroc"], "AUPRC": item["auprc"],\n                 "Brier": item["brier"], "Brier частоты": item["frequency_brier"],\n                 "улучшение Brier": item["relative_brier_improvement"],\n                 "AUROC без предыдущих живорождений": item["nulliparous_auroc"]})\ndisplay(pd.DataFrame(rows).round(5))\nprint("Подходящих записей 2024:", serving["eligible_2024_rows"], "из", serving["total_2024_rows"])\nprint("Оценка 2024 использует возраст при фактических родах, а вход — возраст на предполагаемую дату")'),
    code('comparison = pd.DataFrame([{"исход": name, "метод": label, "доля отмеченных": serving["checklist_selected_fraction"],\n                                 "PPV": result["ppv"], "чувствительность": result["sensitivity"]}\n    for name, item in serving["horizons"].items()\n    for label, result in (("3 вопроса анамнеза", item["checklist"]),\n                          ("модель при той же доле", item["model_at_same_selection"]))])\ndisplay(comparison.round(4))'),
    code('weekly_scope = serving["weekly_metrics"]\nprint("Недельный log loss:", round(weekly_scope["weekly_log_loss"], 5),\n      "; частотный контроль:", round(weekly_scope["frequency_weekly_log_loss"], 5))\npart = pd.DataFrame(serving["weekly_calibration"])\npart = part.loc[part.birth_week.ne("37+")]\nfig, ax = plt.subplots(figsize=(10, 4.5))\nx = np.arange(len(part))\nax.bar(x - .18, part.observed_rate * 100, width=.36, color=SLATE, label="Наблюдалось")\nax.bar(x + .18, part.mean_predicted * 100, width=.36, color=BLUE, label="Прогноз")\nax.set_xticks(x, part.birth_week.astype(str))\nax.set(xlabel="Завершённая неделя при родах", ylabel="Доля среди продолжающихся на 28+0, %",\n       title="Допустимые записи: распределение родов по неделям · 2024")\nax.legend(); plt.tight_layout(); plt.show()'),
    md('## Дополнительная диагностика · все записи 2024, включая пропуски'),
    code('metrics, calibration = evaluate_week_profile()\nmain = metrics.loc[metrics.horizon.eq("before_37")].copy()\ndisplay(main[["landmark_week", "n_ongoing_live_births", "observed_rate", "mean_predicted", "auroc", "auprc", "brier", "weekly_log_loss", "frequency_weekly_log_loss"]].round(5))\nshort = metrics.loc[metrics.horizon.eq("next_4_weeks_preterm")].copy()\ndisplay(short[["landmark_week", "events", "observed_rate", "mean_predicted", "auroc", "auprc", "brier"]].round(5))\nprint("Частотный контроль для недельного log loss рассчитан на calibration 2023")'),
    code('fig, axes = plt.subplots(2, 2, figsize=(11, 9))\nfor ax, week in zip(axes.ravel(), [20, 28, 32, 36]):\n    part = calibration.loc[(calibration.landmark_week.eq(week)) & (calibration.horizon.eq("before_37"))]\n    n = int(main.loc[main.landmark_week.eq(week), "n_ongoing_live_births"].iloc[0])\n    ax.plot(part.score_decile, 100 * part.observed_rate, "o-", color=BLUE, label="Наблюдалось")\n    ax.plot(part.score_decile, 100 * part.mean_predicted, "s--", color=ORANGE, label="Прогноз")\n    ax.set(xlabel="Дециль исходного риска", ylabel="Роды <37 недель, %", ylim=(0, 23), title=f"Все записи на {week}+0 · 2024 · n={n:,}")\naxes[0, 0].legend(); plt.tight_layout(); plt.show()\ncal_lt37 = calibration.loc[calibration.horizon.eq("before_37")]\nprint("Максимальное отклонение среднего прогноза от доли по показанным децилям:", round(100 * (cal_lt37.mean_predicted - cal_lt37.observed_rate).abs().max(), 3), "п.п.")'),
    md('## Важность признаков'),
    code('importance = pd.DataFrame(get_feature_importance())\ndisplay(importance.round(5))\nplot = importance.sort_values("permutation_delta_log_loss")\nfig, ax = plt.subplots(figsize=(9, 5))\nax.barh(plot.feature, plot.permutation_delta_log_loss, color=BLUE)\nax.set(xlabel="Рост log loss после перестановки · полные записи calibration 2023", ylabel="Признак", title="Важность итоговой модели на допустимом входе")\nplt.tight_layout(); plt.show()'),
    md('## Прогноз для одной женщины · вымышленный профиль'),
    code('woman = {"gestational_week": 28, "gestational_day": 3, "age_at_due_date_years": 30,\n         "singleton_confirmed": True, "pregnancy_ongoing": True, "dating_confirmed": True,\n         "prepregnancy_bmi": 24, "previous_live_births": 1, "previous_terminations": 0,\n         "previous_preterm_birth": False, "chronic_hypertension": False,\n         "preexisting_diabetes": False, "previous_cesarean": False,\n         "prepregnancy_cigarettes_per_day": 0}\nforecast = predict_woman(woman)\nassert forecast["status"] == "OK"\ndisplay(pd.Series({k: v for k, v in forecast["probabilities"].items() if isinstance(v, float)}).rename("вероятность"))\nassert predict_woman({**woman, "prepregnancy_bmi": None})["status"] == "INSUFFICIENT_DATA"\nassert predict_woman({**woman, "previous_live_births": 0, "previous_preterm_birth": True})["status"] == "OOD"\nprint("Все маски прогноза:", forecast["missing_masks"])'),
    code('weekly = pd.DataFrame(forecast["probabilities"]["week_of_birth"])\nassert np.isclose(weekly.probability.sum() + forecast["probabilities"]["at_or_after_37_weeks"], 1)\nfig, ax = plt.subplots(figsize=(10, 4))\nax.bar(weekly.completed_week.astype(str), weekly.probability * 100, color=BLUE)\nax.set(xlabel="Завершённая неделя при родах", ylabel="Оценка, %", title="Иллюстрация: условное распределение преждевременных родов")\nplt.tight_layout(); plt.show()'),
    md('''## Выводы

Проверка 2024 относится к **записям об одноплодных живорождениях США, переживших соответствующую неделю**. Для прямого ввода применяются правила полноты и области модели; таблица всех записей дана только как дополнительная диагностика. Модель различает исходы умеренно и мало улучшает три вопроса анамнеза. Средняя калибровка не гарантирует точность для отдельной женщины. В данных нет точной даты родов, визитов, цервикометрии, симптомов, вмешательств, мертворождений и типа начала родов. Недельный прогноз — приближение по завершённым неделям. Вымышленный профиль иллюстрирует формат результата. Поля и ограничения: [WOMAN_FORECAST_CONTRACT.md](WOMAN_FORECAST_CONTRACT.md).'''),
])

print(f"Built 6 notebooks in {OUT}")
