"""Generate reproducible CDC HDP-proxy notebooks without running model fits."""
from pathlib import Path
import ast
import re
from copy import deepcopy
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "research/notebooks/preeclampsia_cdc"
DEST.mkdir(parents=True, exist_ok=True)


def md(value): return nbf.v4.new_markdown_cell(value)
def code(value):
    ast.parse(value)
    return nbf.v4.new_code_cell(value)
def save(name, cells):
    books[name] = cells


def write_book(name, cells):
    book = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    path = DEST / name
    if path.exists():
        previous = nbf.read(path, as_version=4)
        for current, old in zip(book.cells, previous.cells):
            if current.cell_type == old.cell_type == "code" and current.source == old.source:
                current.outputs = old.outputs
                current.execution_count = old.execution_count
    nbf.validate(book)
    nbf.write(book, path)

books = {}

setup = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/data/preeclampsia_cdc.py").exists())
sys.path.insert(0, str(ROOT))
from src.progress import tqdm
plt.rcParams.update({"figure.figsize": (8, 4.8), "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .2})
BLUE, ORANGE, GREY = "#255B8B", "#D17A31", "#666D78"
MODEL_PALETTE = [("logistic_spline", BLUE), ("hist_gb", ORANGE),
                 ("catboost", "#287D69"), ("lightgbm", "#8062A5")]
RACE6 = {1:"White",2:"Black",3:"AIAN",4:"Asian",5:"NHOPI",6:"Multiple"}  # NCHS MRACE6; social reporting categories'''

save("01_prepare_data.ipynb", [
    md('''# CDC 2023+2024 · подготовка когорты HDP-proxy

Общая когорта включает **все** живорождения резиденток США после одноплодной беременности за 2023–2024 годы с известными `RF_GHYPE` и `RF_EHYPE`, **при записанном отсутствии хронической АГ** (`RF_PHYPE=0`). Каждой записи назначается один из пяти стабильных фолдов; оба года представлены в каждом фолде. Женщины с хронической АГ исключены из основной оценки, поскольку совместная кодировка с `RF_GHYPE` систематически нарушает интерпретацию отрицательных меток. Их количество показано в аудите.

Основная метка: `RF_GHYPE=Y OR RF_EHYPE=Y`. `RF_GHYPE` объединяет гестационную гипертензию и преэклампсию; **отдельный диагноз ПЭ недоступен**. Вторичные метки: HDP-proxy **и** роды до 37 / 34 недель. `OEGEST_COMB` используется только для меток и аудита, никогда как предиктор. Неизвестный срок делает вторичную метку пропущенной. Это сочетание событий не подтверждает преждевременную ПЭ.

Возраст матери записан при родах, поэтому используется лишь как приближение возраста до беременности. Исходы доступны только для живорождений; идентификатора матери и времени начала болезни нет. [NCHS User Guide](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/DVS/natality/UserGuide2024.pdf).'''),
    code(setup + '\nfrom src.data.preeclampsia_cdc import OUT, FEATURES, OUTCOMES, MISSING_MASKS, QUALITY_MASKS, prepare_all, audit_race_imputation'),
    md('## 1. Сверка источников и сборка общего файла'),
    code('audit = prepare_all()\nchecks = pd.DataFrame(audit["year_checks"]).T\ndisplay(checks.drop(columns="outcomes"))\nprint("Parquet:", OUT / "eligible_all_2023_2024.parquet")'),
    md('## 2. Исключения, частоты и вторичные метки'),
    code('cols = ["singleton_before_chronic_exclusion", "excluded_chronic_rows", "excluded_chronic_recorded_events", "eligible_singleton_rows", "eligible_singleton_events", "eligible_singleton_prevalence"]\ndisplay(checks[cols])\noutcome_table = pd.DataFrame([{"year":year,"outcome":name,**value} for year, row in audit["year_checks"].items() for name,value in row["outcomes"].items()])\ndisplay(outcome_table)\nassert checks.excluded_chronic_rows.sum() > 0\nassert (checks.eligible_singleton_rows + checks.excluded_chronic_rows == checks.singleton_before_chronic_exclusion).all()'),
    md('## 3. Фолды всех записей и пропуски'),
    code('folds = pd.read_csv(OUT / "folds_all.csv")\ndisplay(folds)\nassert folds.rows.sum() == checks.eligible_singleton_rows.sum()\nassert folds.groupby("year").fold.nunique().eq(5).all()\nfig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\nfor year, color in [(2023, BLUE), (2024, ORANGE)]:\n    part = folds.loc[folds.year.eq(year)]\n    axes[0].plot(part.fold, part.rows, marker="o", label=str(year), color=color)\n    axes[1].plot(part.fold, part.prevalence, marker="o", label=str(year), color=color)\naxes[0].set(xlabel="Фолд", ylabel="Записи", title="Объём по фолдам")\naxes[1].set(xlabel="Фолд", ylabel="Доля HDP-proxy", title="Частота метки")\nfor ax in axes: ax.legend(); ax.set_xticks(range(5))\nplt.tight_layout(); plt.show()'),
    code('from matplotlib.ticker import PercentFormatter\nmissing = pd.read_csv(OUT / "missingness_by_year.csv")\ntable = missing.pivot(index="field", columns="year", values="missing_fraction")\ndisplay(table.round(4))\nprint("ART: доля пропусков в процентах, а не во всём наборе")\ndisplay((table.loc[["rf_inftr", "rf_artec"]] * 100).round(3).rename_axis("field"))\nplot_table=table.loc[table.max(axis=1).gt(0)].sort_values(2024).tail(15)\nax=plot_table.plot.barh(figsize=(9, 6), color=[BLUE, ORANGE])\nax.set(xlim=(0,.10), xlabel="Доля записей с пропуском, 0–10%", title="15 полей с наибольшей долей пропусков · CDC 2023–24")\nax.xaxis.set_major_formatter(PercentFormatter(1))\nplt.tight_layout(); plt.show()\nassert not set(["oegest_comb", "target_hdp_preterm37", "target_hdp_preterm34"]) & set(FEATURES)'),
    md("""Здесь 2024 `RF_INFTR` и `RF_ARTEC` находятся около **7%**, а не 100%. Оба названия существуют в исходном CSV. Пропуски включают целиком неотчётные области и небольшую долю неизвестных индивидуальных ответов; следующая таблица разделяет эти причины по флагам CDC."""),
    md('## 4. Маски входа: доля, исход и смена отчётности между годами'),
    code('masks=pd.read_csv(OUT/"input_masks_by_year.csv")\ndisplay(masks.round(4))\nassert set(MISSING_MASKS + QUALITY_MASKS)==set(masks["mask"])\nselected=masks.loc[((masks["mask"].isin(MISSING_MASKS)) & masks.value.eq(1)) | (masks["mask"].eq("art_source_reported") & masks.value.eq(0)) | (masks["mask"].eq("art_status_unknown_reported") & masks.value.eq(1))].copy()\nselected["state"]=selected["mask"].replace({"art_source_reported":"art_not_reported"})\nselected["fraction"]=selected.apply(lambda row: row.rows/int(checks.loc[str(row.year),"eligible_singleton_rows"]),axis=1)\nlabels=list(selected.state.unique())\nfig,axes=plt.subplots(1,2,figsize=(13,4.5))\nfor year,color,shift in [(2023,BLUE,-.08),(2024,ORANGE,.08)]:\n    part=selected.loc[selected.year.eq(year)]\n    position=np.array([labels.index(name) for name in part.state])+shift\n    positive=part.rows.gt(0).to_numpy()\n    axes[0].scatter(part.fraction.to_numpy()[positive],position[positive],label=str(year),color=color,s=65)\n    axes[1].scatter(part.event_rate.to_numpy()[positive],position[positive],label=str(year),color=color,s=65)\naxes[0].set(xscale="log",xlim=(1e-4,.2),xlabel="Доля записей, логарифмическая шкала",title="Пропуск или неотчётность")\naxes[1].set(xlabel="Доля HDP-proxy в этом состоянии",title="Частота записанной метки")\nfor ax in axes: ax.set_yticks(range(len(labels)),labels); ax.legend()\nplt.tight_layout();plt.show()'),
    md('## 5. Импутация расы в источнике NCHS'),
    code('''race_imp=audit_race_imputation()
race_counts=race_imp.pivot_table(index=["year","mrace6"],columns="imputed",values="rows",fill_value=0)
race_counts.columns=["not_imputed" if int(code)==0 else "imputed" for code in race_counts.columns]
race_counts["fraction_imputed"]=race_counts.imputed/(race_counts.imputed+race_counts.not_imputed)
display(race_counts.round(4))
by_year=race_imp.groupby(["year","imputed"]).rows.sum().unstack(fill_value=0)
print("Доля импутированной расы в выбранной когорте:")
display((by_year[1]/by_year.sum(axis=1)).rename("fraction_imputed").round(4))
fig,ax=plt.subplots(figsize=(8,4.5))
x=np.arange(6)
for year,color,shift in [(2023,BLUE,-.18),(2024,ORANGE,.18)]:
    part=race_counts.loc[year].reindex(range(1,7))
    ax.bar(x+shift,100*part.fraction_imputed,width=.36,label=str(year),color=color)
ax.set_xticks(x,[RACE6[code] for code in range(1,7)])
ax.set(xlabel="Опубликованный код расы",ylabel="Импутировано, %",title="Доля импутации расы внутри выбранной когорты")
ax.legend();plt.tight_layout();plt.show()'''),
    md('''## Интерпретация

`audit.json` содержит SHA-256 архивов, количества до/после исключения и сверку общего `RF_GHYPE` с официальной статистикой NCHS. `input_masks_by_year.csv` различает пропуск индивидуального значения, неотчётность источника ART и неизвестный ответ при отчётности. `race_imputation_by_year.csv` вычисляется отдельно по исходным архивам на **той же когорте**: опубликованный `MRACE6` может быть импутирован даже при отсутствии пропусков в подготовленном файле. Маски BMI/анамнеза входят в основную модель; маска отчётности ART служит только для выбора маршрута ранней модели. Фолды дают распределение метрик на смешанных годах; отдельный тест переноса 2023→2024 находится в ноутбуке 06. См. `MODEL_INPUT_CONTRACT.md`.'''),
])

save("02_train_baselines.ipynb", [
    md('''# CDC HDP-proxy · фиксированные baseline, вложенный подбор и итоговое обучение

**В этой рабочей папке начинайте с 02:** подготовленный Parquet и аудит уже есть. 01 нужен только при намеренной пересборке когорты; повторная запись Parquet может изменить контрольную сумму и сделать существующие кэши несовместимыми. На каждом фолде модели видят остальные четыре фолда из объединённых 2023+2024 и предсказывают только отложенные записи. Обучение не запускается при создании ноутбуков.

Предзаданные ориентиры: частота метки, линейная логистика, логистика со сплайнами возраста/BMI и категориями числа прошлых живорождений, неглубокий HistGradientBoosting, CatBoost и LightGBM с фиксированными параметрами. Затем **в этом же ноутбуке** выполняются вложенный подбор на общих фолдах и отдельный пятифолдовый выбор параметров для итогового обучения на всей когорте. Все основные бустинги получают одинаковые признаки, включая `NaN` и явные маски; у CatBoost здесь нет категориальных входов, поэтому его преимущество обработки категорий не предполагается заранее. Для отсутствующего BMI, числа прошлых живорождений и числа других исходов предыдущих беременностей есть явные маски. Логистическая импутация обучается только на train; градиентный бустинг получает NaN и те же маски. Обязательные возраст и три двоичных фактора анамнеза проверяются без импутации. Нет хронической АГ и курения в признаках. Классы не перебалансируются. Коэффициенты штрафной логистики не следует интерпретировать как скорректированные OR; для объяснения используются частоты, графики и отложенная перестановочная важность. Коды `MRACE6` расшифровываются по руководству NCHS как социальные категории регистрации, а не биологический механизм. ВРТ и демография исследуются отдельно на временном тесте в 06. Пять фолдов дают **распределение метрик, не доверительный интервал и не проверку будущего года**. Уже сохранённые фиксированные результаты сверяются и используются повторно. Подробности — в [MODEL_BENCHMARKS.md](MODEL_BENCHMARKS.md).'''),
    code(setup + '\nfrom src.data.preeclampsia_cdc import OUT, FEATURES\nfrom src.training.preeclampsia_cdc import fit_folds, MODEL_NAMES, HGB_PARAMS, CATBOOST_PARAMS, LIGHTGBM_PARAMS'),
    code('''import shutil
import psutil
from src.training.preeclampsia_cdc import run_fingerprint
from src.training.preeclampsia_cdc_tuning import runtime_versions
from src.training.preeclampsia_cdc_nested import NESTED_DIR, nested_fingerprint
from src.training.preeclampsia_cdc_final import FINAL_DIR, final_fingerprint
source_path = OUT / "eligible_all_2023_2024.parquet"
assert source_path.exists(), "Сначала запустите 01_prepare_data.ipynb"
audit = json.loads((OUT / "audit.json").read_text())
fold_table = pd.read_csv(OUT / "folds_all.csv")
assert len(fold_table) == 10 and fold_table.groupby("year").fold.nunique().eq(5).all()
assert int(fold_table.rows.sum()) == sum(x["eligible_singleton_rows"] for x in audit["year_checks"].values())
assert int(fold_table.events.sum()) == sum(x["eligible_singleton_events"] for x in audit["year_checks"].values())
assert not set(FEATURES) & {"source_year", "fold", "target_hdp_proxy", "rf_ghype", "rf_ehype", "oegest_comb"}
for manifest_name in ("fold_run.json", "temporal_run.json"):
    manifest_path = OUT / manifest_name
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["fingerprint"] == run_fingerprint(), f"Устаревший кэш: {manifest_name}"
for search_dir, expected_fingerprint in ((NESTED_DIR, nested_fingerprint()), (FINAL_DIR, final_fingerprint())):
    manifest_path = search_dir / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["fingerprint"] == expected_fingerprint, f"Устаревший поиск: {search_dir}"
free_gb = shutil.disk_usage(OUT).free / 1e9
memory_gb = psutil.virtual_memory().total / 1e9
available_memory_gb = psutil.virtual_memory().available / 1e9
print(f"Данные: {fold_table.rows.sum():,} строк, {fold_table.events.sum():,} событий; свободный диск: {free_gb:.1f} ГБ; ОЗУ: {memory_gb:.1f} ГБ, сейчас доступно {available_memory_gb:.1f} ГБ")
print("Версии:", runtime_versions())
print("Признаки:", FEATURES)
print("Модели:", MODEL_NAMES)
print("HGB:", HGB_PARAMS)
print("CatBoost:", CATBOOST_PARAMS)
print("LightGBM:", LIGHTGBM_PARAMS)
if free_gb < 10:
    print("ВНИМАНИЕ: для промежуточных прогнозов подбора может не хватить места на диске")
if memory_gb < 16:
    print("ВНИМАНИЕ: полный поиск на миллионах строк может исчерпать ОЗУ")
if available_memory_gb < 8:
    print("ВНИМАНИЕ: перед полным поиском освободите ОЗУ; доступная память сейчас низкая")'''),
    md('## Обучение и OOF-прогнозы всех записей'),
    code('fold_metrics = fit_folds(resume=True)\ndisplay(fold_metrics.round(4))'),
    md('## Покрытие фолдов и разброс качества фиксированных моделей'),
    code('coverage=fold_metrics.loc[fold_metrics.model.eq("prevalence"),["fold","n","events","prevalence"]].sort_values("fold")\nexpected_rows=sum(item["eligible_singleton_rows"] for item in json.loads((OUT/"audit.json").read_text())["year_checks"].values())\nassert coverage.fold.tolist()==list(range(5)) and coverage.n.sum()==expected_rows\ndisplay(coverage)\nspread=fold_metrics.groupby("model")[["auroc","auprc","brier","log_loss"]].agg(["mean","std","min","max"])\ndisplay(spread.round(5))\nfig,axes=plt.subplots(1,2,figsize=(12,4))\naxes[0].bar(coverage.fold,coverage.n,color=BLUE)\naxes[0].set(xlabel="Фолд",ylabel="Записей",title="Покрытие 2023+2024")\naxes[1].plot(coverage.fold,100*coverage.prevalence,marker="o",color=ORANGE)\naxes[1].set(xlabel="Фолд",ylabel="HDP-proxy, %",title="Доля записанной метки")\nfor ax in axes:ax.set_xticks(range(5))\nplt.tight_layout();plt.show()'),
    code('fig, axes = plt.subplots(2, 2, figsize=(14, 9))\nfor ax, metric, ylabel in zip(axes.flat,["auroc","auprc","brier","log_loss"],["AUROC","AUPRC","Brier","Log loss"]):\n    for name,color in [("logistic","#6A91B1"),*MODEL_PALETTE]:\n        part=fold_metrics.loc[fold_metrics.model.eq(name)].sort_values("fold")\n        ax.plot(part.fold,part[metric],marker="o",label=name,color=color)\n    ax.set(xlabel="Отложенный фолд",ylabel=ylabel,title=f"{ylabel} · фиксированные модели")\n    ax.set_xticks(range(5))\naxes[0,0].legend(ncol=2,fontsize=9)\nplt.tight_layout();plt.show()'),
    md('''Выше — оценки **фиксированных** конфигураций. Ниже параметры выбираются заново внутри каждого внешнего фолда; внешние прогнозы и их графики смотрите в 03. Для подбора только по 2023 и проверки переноса на 2024 используйте 06. Результаты каждого fit сохраняются, поэтому повторный запуск продолжает подтверждённые вычисления.'''),
])

save("03_evaluate_folds.ipynb", [
    md('''# CDC HDP-proxy · оценка фиксированных и настроенных моделей на пяти общих фолдах

Здесь сначала оцениваются фиксированные модели из 02, затем — внешние прогнозы настроенных моделей из того же ноутбука. Каждая строка оценивается ровно один раз вне своего обучения. Показаны распределения AUROC/AUPRC/Brier, калибровка, чувствительность при **описательном** бюджете верхних 10% прогнозов и сравнение с наблюдаемой частью факторов USPSTF. Для применения к будущему году фиксированный порог получают из 2023 OOF и проверяют в 06.

`calibration_in_the_large` рассчитывается с наклоном логита, фиксированным равным 1; калибровочный наклон оценивается отдельно. Правило USPSTF частичное: прежняя ПЭ, почечная/аутоиммунная болезнь, семейный анамнез и др. отсутствуют. Неизвестные измеренные факторы не превращаются в отсутствие риска.'''),
    code(setup + '\nfrom src.data.preeclampsia_cdc import OUT\nfrom src.training.preeclampsia_cdc import MODEL_NAMES, metrics\nfrom src.evaluation.preeclampsia_cdc import validate_and_summarize, year_breakdown, compare_guideline_at_budget, secondary_outcome_metrics, calibration_offset\nfrom sklearn.calibration import calibration_curve\nfrom sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay'),
    code('fold_metrics, oof = validate_and_summarize()\ndisplay(fold_metrics.round(4))\nspread = fold_metrics.groupby("model")[["auroc", "auprc", "brier"]].agg(["mean", "std", "min", "max"])\ndisplay(spread.round(4))'),
    code('fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\nfor ax, metric in zip(axes, ["auroc", "auprc", "brier"]):\n    fold_metrics.boxplot(column=metric, by="model", ax=ax, grid=False)\n    ax.set(title=metric.upper(), xlabel="", ylabel=metric); ax.tick_params(axis="x", rotation=20)\nplt.suptitle(""); plt.tight_layout(); plt.show()'),
    md('## ROC/PR на случайной выборке 300 000 записей; численные метрики и калибровка — на всех OOF-записях'),
    code('plot_sample = oof.sample(n=min(300_000, len(oof)), random_state=20261001)\nfig, axes = plt.subplots(1, 3, figsize=(17, 4.8))\nfor name, color in MODEL_PALETTE:\n    RocCurveDisplay.from_predictions(plot_sample.label, plot_sample[name], ax=axes[0], name=name, curve_kwargs={"color": color})\n    PrecisionRecallDisplay.from_predictions(plot_sample.label, plot_sample[name], ax=axes[1], name=name, curve_kwargs={"color": color})\n    obs, pred = calibration_curve(oof.label, oof[name], n_bins=12, strategy="quantile")\n    axes[2].plot(pred, obs, marker="o", label=name, color=color)\naxes[2].plot([0,.5],[0,.5],"k--")\naxes[2].set(xlim=(0,.5),ylim=(0,.5),xlabel="Средний прогноз",ylabel="Наблюдаемая доля",title="Калибровка · полные OOF")\naxes[2].legend(); plt.tight_layout(); plt.show()'),
    md('## Годы, вторичные сочетания исходов и правило'),
    code('by_year = year_breakdown(oof)\ndisplay(by_year.round(4))\nsecondary = secondary_outcome_metrics(oof, MODEL_NAMES[1:])\nsecondary.to_csv(OUT / "pooled_cv_secondary_outcomes.csv", index=False)\ndisplay(secondary.round(4))\nrule = compare_guideline_at_budget(oof)\nrule.to_csv(OUT / "pooled_cv_guideline_comparison.csv", index=False)\ndisplay(rule.round(4))'),
    md('## Калибровка по группам, включая отсутствие измеренного фактора высокого риска'),
    code('groups = {"no_recorded_pregestational_diabetes": oof.rf_pdiab.eq(0), "pregestational_diabetes": oof.rf_pdiab.eq(1), "bmi_under_30": oof.bmi.lt(30), "bmi_30_plus": oof.bmi.ge(30)}\ngroups.update({f"{name}_yes":oof[name].eq(1) for name in ["bmi_missing","prior_live_birth_count_missing","priorterm_missing"]})\ngroups.update({f"race_code_{int(code)}": oof.mrace6.eq(code) for code in sorted(oof.mrace6.dropna().unique())})\ngroups.update({f"education_code_{int(code)}": oof.meduc.eq(code) for code in sorted(oof.meduc.dropna().unique())})\ngroups.update({f"ethnicity_code_{int(code)}": oof.mhisp_r.eq(code) for code in sorted(oof.mhisp_r.dropna().unique())})\nsub = []\nfor group, mask in tqdm(groups.items(), desc="Subgroup calibration", unit="group"):\n    part = oof.loc[mask]\n    if len(part) < 100 or part.label.nunique() < 2: continue\n    for name in [model for model,_ in MODEL_PALETTE]:\n        sub.append({"group":group,"model":name,"n":len(part),"events":int(part.label.sum()),**metrics(part.label.to_numpy(), part[name].to_numpy()),"calibration_in_the_large":calibration_offset(part.label.to_numpy(),part[name].to_numpy()),"mean_calibration_gap_pp":100*(part.label.mean()-part[name].mean())})\nsub = pd.DataFrame(sub)\nsub.to_csv(OUT / "pooled_cv_subgroups.csv", index=False)\ndisplay(sub.round(4))\nselected=sub.loc[sub.group.str.startswith("race_code_")|sub.group.str.contains("diabetes")]\ngroup_order=sorted(selected.group.unique())\nfig,ax=plt.subplots(figsize=(10,max(5,.55*len(group_order))))\nfor offset,(name,color) in zip(np.linspace(-.27,.27,len(MODEL_PALETTE)),MODEL_PALETTE):\n    part=selected.loc[selected.model.eq(name)].set_index("group").reindex(group_order)\n    ax.scatter(part.mean_calibration_gap_pp,np.arange(len(group_order))+offset,label=name,color=color,s=45)\nax.axvline(0,color="black",ls="--"); ax.set_yticks(np.arange(len(group_order)),group_order)\nax.set(xlabel="Наблюдаемая − прогноз, п.п.",title="Средняя калибровка по группам")\nax.legend(); plt.tight_layout(); plt.show()'),
    md('''## Аудит импутации расы на тех же отложенных прогнозах

Флаг `MRACEIMP` восстановлен из исходных CSV в отдельной таблице по `source_year, source_row`. Он нужен только для аудита и не входит во вход модели. Таблица сверяется с опубликованным кодом `MRACE6` и агрегированным аудитом 01. Разрывы калибровки показаны по опубликованной расе отдельно до и после исключения записей с импутацией.'''),
    code('''from src.evaluation.preeclampsia_cdc_race import build_race_imputation_rows, race_imputation_sensitivity, RACE_ROWS
if not RACE_ROWS.exists():
    build_race_imputation_rows()
race_sensitivity = race_imputation_sensitivity(oof, ["logistic_spline", "hist_gb", "catboost", "lightgbm"])
race_sensitivity.to_csv(OUT / "pooled_cv_race_imputation_sensitivity.csv", index=False)
display(race_sensitivity.loc[race_sensitivity.model.eq("hist_gb")].round(4))
non_imputed = race_sensitivity.loc[race_sensitivity.subset.eq("non_imputed") & race_sensitivity.model.eq("hist_gb")]
fig, ax = plt.subplots(figsize=(10, 4.8))
for year, color in [(2023, BLUE), (2024, ORANGE)]:
    part = non_imputed.loc[non_imputed.year.eq(year)].sort_values("race_code")
    ax.plot(part.race_code, part.mean_calibration_gap_pp, marker="o", label=str(year), color=color)
ax.axhline(0, color="black", ls="--")
ax.set_xticks(range(1, 7), [RACE6[n] for n in range(1, 7)])
ax.set(xlabel="Опубликованный код расы", ylabel="Наблюдаемая − прогноз, п.п.",
       title="HGB · только записи без импутации расы")
ax.legend(); plt.tight_layout(); plt.show()'''),
    md('''## Ограничения

Это качество прогнозирования записанного HDP-proxy, не клинический риск ПЭ. Факторы анамнеза и метка заполняются в одном свидетельстве, поэтому ошибки их регистрации могут быть связаны. Пропущенный BMI может отражать процесс заполнения; его связь с меткой не доказывает причину. Раса в записи не биологический механизм и может отражать социальное неравенство/качество кодировки; сравнение групп обязательно интерпретировать с этой оговоркой. Отдельный флаг импутации позволяет проверить устойчивость групповых результатов, но исключение импутированных строк не делает расу биологическим предиктором. Если обнаружится недооценка или переоценка по группе, сначала нужно проверить качество регистрации исхода и перенос; групповую коррекцию по тем же меткам автоматически не обучаем. Одна женщина может встретиться в обоих годах; идентификатора нет. В исследовании нет мертворождений. Для проверок переноса и фиксированного порога используйте 06.'''),
])

save("04_feature_correlations.ipynb", [
    md('''# CDC HDP-proxy · признаки и одномерные связи

Воспроизводимая случайная 5% подвыборка общей когорты выбирается **без учёта метки и фолда**. Корреляции между признаками проверяют дублирование; для связи с бинарной меткой показаны прежде всего частоты по категориям и не скорректированные отношения шансов с 95% интервалом. Эти показатели описательные и не доказывают причинного влияния.'''),
    code(setup + '\nfrom src.data.preeclampsia_cdc import OUT, ENRICHED_FEATURES\nfrom src.evaluation.preeclampsia_cdc_features import analysis_sample'),
    code('sample = analysis_sample(fraction=.05)\nprint(f"n={len(sample):,}, events={int(sample.target_hdp_proxy.sum()):,}, frequency={sample.target_hdp_proxy.mean():.4f}")\ndisplay(sample.groupby(["source_year","fold"]).target_hdp_proxy.agg(["size","sum","mean"]).round(4))'),
    md('## Корреляции самих признаков'),
    code('corr = sample[ENRICHED_FEATURES].corr(method="spearman")\ndisplay(corr.round(3))\nfig, ax = plt.subplots(figsize=(11,9))\nim = ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm")\nax.set_xticks(range(len(corr)),corr.columns,rotation=55,ha="right")\nax.set_yticks(range(len(corr)),corr.index)\nax.set(title=f"Spearman · CDC 2023–24 без хронической АГ, n={len(sample):,}")\nfig.colorbar(im,ax=ax); plt.tight_layout(); plt.show()'),
    md('## Частоты по клинически понятным категориям'),
    code('art_state=np.select([sample.art_source_reported.eq(0),sample.art_status_unknown_reported.eq(1),sample.rf_artec.eq(1),sample.rf_artec.eq(0)],["not_reported","unknown_reported","ART_yes","ART_no"],default="invalid")\nbins = {"age":pd.cut(sample.mager,[9,19,24,29,34,39,44,55]), "bmi":pd.cut(sample.bmi,[12.9,18.5,25,30,35,40,70]), "prior_live_births":sample.prior_live_birth_count_cat.fillna(-1), "pregestational_diabetes":sample.rf_pdiab, "ART_reporting_state":pd.Series(art_state,index=sample.index)}\nfig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes=axes.ravel()\nfor ax,(name,categories) in zip(axes,bins.items()):\n    counts=sample.groupby(categories,observed=True).target_hdp_proxy.agg(["size","sum","mean"])\n    display(counts.rename(columns={"size":"n","sum":"events","mean":"event_rate"}).assign(variable=name).round(4))\n    counts["mean"].plot.bar(ax=ax,color=BLUE); ax.axhline(sample.target_hdp_proxy.mean(),ls="--",color="black")\n    ax.set(xlabel=name,ylabel="Доля HDP-proxy",title=name)\naxes[-1].axis("off"); fig.suptitle("CDC 2023–24 · без хронической АГ · 5% случайная выборка",y=1.01); plt.tight_layout(); plt.show()'),
    md('## Одномерные OR и интервалы'),
    code('binary = ["rf_pdiab","rf_ppterm","rf_cesar","no_prior_live_birth","age_35_plus","bmi_30_plus","rf_artec"]\nrows=[]\nfor feature in binary:\n    part=sample.loc[sample[feature].isin([0,1]),[feature,"target_hdp_proxy"]]\n    a=int(((part[feature]==1)&(part.target_hdp_proxy==1)).sum()); b=int(((part[feature]==1)&(part.target_hdp_proxy==0)).sum())\n    c=int(((part[feature]==0)&(part.target_hdp_proxy==1)).sum()); d=int(((part[feature]==0)&(part.target_hdp_proxy==0)).sum())\n    aa,bb,cc,dd=[v+.5 for v in [a,b,c,d]]\n    log_or=np.log(aa*dd/(bb*cc)); se=np.sqrt(1/aa+1/bb+1/cc+1/dd)\n    rows.append({"feature":feature,"n":len(part),"or_unadjusted":np.exp(log_or),"ci_low":np.exp(log_or-1.96*se),"ci_high":np.exp(log_or+1.96*se)})\nassociation=pd.DataFrame(rows); display(association.round(3))\nfig,ax=plt.subplots(figsize=(8,5))\nax.errorbar(association.or_unadjusted,association.feature,xerr=[association.or_unadjusted-association.ci_low,association.ci_high-association.or_unadjusted],fmt="o",color=BLUE)\nax.axvline(1,ls="--",color="black");ax.set(xscale="log",xlabel="Не скорректированное OR и 95% интервал",title="CDC 2023–24 · одномерная связь с HDP-proxy")\nplt.tight_layout();plt.show()'),
    md('''OR получены из 2×2 таблиц в выборке и не являются независимыми скорректированными эффектами или рекомендацией по лечению. Коэффициенты штрафной сплайн-логистики отдельно как OR не публикуются: маски, one-hot категории и регуляризация не дают такой интерпретации. `PRIORTERM` смешивает разные исходы прошлых беременностей, включая потери и прерывания; он не является чистым числом выкидышей. `RF_ARTEC` доступен только после зачатия и при отчётности штата. Категории с очень малым числом событий трактуйте осторожно. Качество моделей и перестановочная важность оцениваются отдельно.'''),
])

save("05_feature_importance.ipynb", [
    md('''# CDC HDP-proxy · отложенная важность признаков

На воспроизводимой 5% подвыборке каждого отложенного фолда считаем изменение AUPRC и Brier при трёх независимых перестановках. BMI и показатели анамнеза переставляются **вместе со своей маской пропуска**; отдельно проверяются семейства возраста/BMI и предыдущих беременностей. Маска сама по себе не переставляется, потому что это создало бы невозможные пары «значение известно, маска=1». Добавочную ценность масок проверяет `hist_gb_no_masks` в 06. Это важность для **данной обученной модели и записанной метки**, не причинный эффект. Сначала запустите 02. Обучения здесь нет.'''),
    code(setup + '\nimport joblib, gc\nfrom src.data.preeclampsia_cdc import OUT, ROOT, FEATURES\nfrom src.training.preeclampsia_cdc import MODEL_DIR, SPLINE_FEATURES, run_fingerprint\nfrom src.evaluation.preeclampsia_cdc_features import analysis_sample, heldout_permutation_importance'),
    code('manifest=json.loads((OUT/"fold_run.json").read_text())\nassert manifest["fingerprint"]==run_fingerprint()\nsample=analysis_sample(fraction=.05)\ndisplay(sample.groupby("fold").target_hdp_proxy.agg(["size","sum","mean"]).round(4))'),
    code('parts=[]\nfor fold in tqdm(range(5),desc="Held-out importance",unit="fold"):\n    heldout=sample.loc[sample.fold.eq(fold)].copy()\n    for name, features in [(name, SPLINE_FEATURES if name=="logistic_spline" else FEATURES) for name,_ in MODEL_PALETTE]:\n        model=joblib.load(MODEL_DIR/f"fold_{fold}_{name}.joblib")\n        groups={"previous_pregnancy_history":[x for x in ["prior_live_birth_count_cat","prior_live_birth_count_missing","priorterm","priorterm_missing","rf_ppterm","rf_cesar"] if x in features],"age_bmi_family":[x for x in ["mager","bmi","bmi_missing"] if x in features]}\n        parts.append(heldout_permutation_importance(model,heldout,features,name,fold,repeats=3,groups=groups))\n        del model\n    gc.collect()\nimportance=pd.concat(parts,ignore_index=True)\nimportance.to_csv(OUT/"heldout_feature_importance.csv",index=False)\ndisplay(importance.head().round(5))'),
    code('fold_means=importance.groupby(["model","feature_kind","feature","fold"],as_index=False)[["delta_auprc","delta_brier"]].mean()\nsummary=fold_means.groupby(["model","feature_kind","feature"])[["delta_auprc","delta_brier"]].agg(["mean","std"])\ndisplay(summary.round(5))\nfig,axes=plt.subplots(2,2,figsize=(16,13))\nfor ax,(name,color) in zip(axes.flat,MODEL_PALETTE):\n    part=fold_means.loc[fold_means.model.eq(name)&fold_means.feature_kind.eq("individual")]\n    order=part.groupby("feature").delta_auprc.mean().sort_values().index\n    for index,feature in enumerate(order):\n        vals=part.loc[part.feature.eq(feature)].delta_auprc.to_numpy()\n        ax.scatter(vals,np.full(len(vals),index),color=color,alpha=.55)\n        ax.scatter(vals.mean(),index,marker="D",color="black")\n    ax.set_yticks(range(len(order)),order);ax.axvline(0,color="black")\n    ax.set(xlabel="Падение AUPRC",title=name)\nplt.tight_layout();plt.show()'),
    code('grouped=fold_means.loc[fold_means.feature_kind.eq("group")]\nif not grouped.empty:\n    fig,ax=plt.subplots(figsize=(9,5))\n    group_order=sorted(grouped.feature.unique())\n    for offset,(name,color) in zip(np.linspace(-.25,.25,len(MODEL_PALETTE)),MODEL_PALETTE):\n        part=grouped.loc[grouped.model.eq(name)]\n        for index,feature in enumerate(group_order):\n            vals=part.loc[part.feature.eq(feature)].delta_auprc.to_numpy()\n            ax.scatter(vals,np.full(len(vals),index+offset),color=color,alpha=.55,s=22)\n            ax.scatter(vals.mean(),index+offset,marker="D",color=color,s=55)\n        ax.scatter([],[],color=color,label=name)\n    ax.set_yticks(range(len(group_order)),group_order);ax.axvline(0,color="black")\n    ax.set(xlabel="Падение AUPRC",title="Совместная перестановка связанных признаков")\n    ax.legend();plt.tight_layout();plt.show()'),
    md('## Усреднённые кривые зависимости возраста и BMI'),
    code('heldout=sample.loc[sample.fold.eq(0)].sample(n=min(5000,int(sample.fold.eq(0).sum())),random_state=20261001).copy()\nfig,axes=plt.subplots(1,2,figsize=(13,4.5))\nfor name,color in MODEL_PALETTE:\n    model=joblib.load(MODEL_DIR/f"fold_0_{name}.joblib")\n    features=SPLINE_FEATURES if name=="logistic_spline" else FEATURES\n    for ax,feature,grid in [(axes[0],"mager",np.arange(18,46,2)),(axes[1],"bmi",np.arange(18,46,2))]:\n        values=[]\n        for value in tqdm(grid,desc=f"{name} {feature}",leave=False):\n            changed=heldout[features].copy();changed[feature]=value\n            if feature=="bmi": changed["bmi_missing"]=0\n            p=model.predict_proba(changed)[:,1]\n            values.append(p.mean())\n        ax.plot(grid,values,marker="o",label=name,color=color)\n    del model\naxes[0].set(xlabel="Возраст при родах",ylabel="Средняя модельная вероятность",title="Зависимость от возраста")\naxes[1].set(xlabel="BMI до беременности",ylabel="Средняя модельная вероятность",title="Зависимость от BMI")\nfor ax in axes:ax.legend()\nplt.tight_layout();plt.show()'),
    md('''Усреднённые кривые меняют один вход модели на фиксированной подвыборке и могут создавать нереалистичные сочетания признаков; это диагностический график модели. Разброс перестановочной важности по пяти фолдам и повторам не следует называть доверительным интервалом. Для клинической интерпретации учитывайте неполную метку, неучтённый аспирин и возможную общую ошибку регистрации факторов и исхода. Перестановочная важность не превращает коэффициенты штрафной логистики в клинические OR.'''),
])

save("06_temporal_validation.ipynb", [
    md('''# CDC HDP-proxy · перенос 2023 → 2024

**Подготовленный Parquet уже есть; 01 повторно запускать не требуется.** Временную проверку желательно провести до просмотра новых результатов смешанных фолдов 02–05. Пять фолдов **внутри 2023** служат разработкой; затем модели обучаются на всём 2023 и оцениваются на подходящих записях 2024. Это ретроспективная временная проверка, а не внешняя клиническая валидация. Отдельно проверяется, дают ли явные маски пропусков добавку к бустингу, который уже умеет обрабатывать NaN. Предикторы ВРТ и демографии проверяются отдельными чувствительными моделями. Две дополнительные HGB обучаются на составные исходы HDP + роды <37/<34 недель и сверяются с преждевременными родами без записанного HDP; это не диагноз преждевременной ПЭ. Калибровка и переклассификация при фиксированном бюджете 10% оцениваются по ART и социальным группам.

Порог для приблизительно 10% отметок задаётся по OOF-прогнозам 2023 без доступа к распределению прогнозов 2024. Парные интервалы для метрик **основной HDP-proxy** строятся бутстрепом по всем записям 2024; для отдельно обученных составных исходов показываются 2023-фолды и точечные метрики 2024. Без ID матери бутстреп-интервалы могут быть слишком узкими и не учитывают систематическую ошибку метки. Практически заметные различия в метриках **заранее**: ΔAUROC ≥0,01, ΔAUPRC ≥0,01 или абсолютное улучшение Brier ≥0,002; меньшие различия описываются как практически малые даже при узком ДИ. Эти пороги — исследовательская договорённость, не установленный клинический MCID.

2024 год **уже просмотрен экспертом**, включая сравнение составных исходов и вариантов масок. CatBoost и LightGBM добавлены после этого просмотра как исследовательские дополнительные кандидаты; их результаты 2024 не являются независимым подтверждением выбора архитектуры. Поэтому этот тест остаётся ретроспективной проверкой переноса, а не независимой оценкой выбранного победителя. Автоматический выбор типа и параметров во всех трёх поисках использует один критерий — минимум взвешенного OOF `log loss` на соответствующих обучающих фолдах. Сплайн-логистика показывается отдельно как заранее заданный интерпретируемый ориентир; практическая малость разности метрик обсуждается после расчёта, но не подменяет алгоритм выбора. Все варианты 2024 показываются без настройки. Прогноз — частота записанного HDP-proxy, не индивидуальный риск подтверждённой преэклампсии.'''),
    code(setup + '\nfrom src.data.preeclampsia_cdc import OUT\nfrom src.training.preeclampsia_cdc import fit_temporal, fit_secondary_temporal, MODEL_NAMES, TEMPORAL_SENSITIVITIES, SECONDARY_TARGETS, metrics\nfrom src.evaluation.preeclampsia_cdc import temporal_summary, temporal_bootstrap, secondary_outcome_metrics, secondary_trained_summary, temporal_subgroup_audit, art_reclassification, compare_guideline_at_budget, decision_curve\nfrom sklearn.calibration import calibration_curve'),
    md('## Разработка на 2023, финальные модели и полный тест 2024'),
    code('preliminary=fit_temporal(resume=True)\ndev=pd.read_parquet(OUT/"temporal_dev_oof.parquet")\nresearch_candidates=pd.DataFrame([{"model":name,**metrics(dev.label.to_numpy(),dev[name].to_numpy())} for name in MODEL_NAMES])\nprint("Сплайн-логистика: интерпретируемый ориентир; тип модели выбирается поиском по log loss")\ndisplay(research_candidates.round(4))\ndisplay(preliminary.round(4))'),
    md('## Основная оценка и фиксированный по 2023 порог'),
    code('summary,_=temporal_summary(bootstrap_rounds=0)\ndisplay(summary.round(4))'),
    code('test=pd.read_parquet(OUT/"temporal_test_2024.parquet")\ndev=pd.read_parquet(OUT/"temporal_dev_oof.parquet")\nassert test.source_year.eq(2024).all() and dev.source_year.eq(2023).all()\nfig,axes=plt.subplots(1,2,figsize=(12,4.5))\nfor name,color in MODEL_PALETTE:\n    obs,pred=calibration_curve(test.label,test[name],n_bins=12,strategy="quantile")\n    axes[0].plot(pred,obs,marker="o",label=name,color=color)\n    row=summary.loc[summary.model.eq(name)].iloc[0]\n    axes[1].scatter(row.fixed_flagged_fraction,row.fixed_sensitivity,label=name,color=color,s=80)\naxes[0].plot([0,.5],[0,.5],"k--");axes[0].set(xlim=(0,.5),ylim=(0,.5),xlabel="Средний прогноз",ylabel="Наблюдаемая доля",title="Калибровка 2024")\naxes[1].set(xlabel="Фактическая доля отметок 2024",ylabel="Чувствительность",title="Порог фиксирован по 2023")\nfor ax in axes:ax.legend()\nplt.tight_layout();plt.show()'),
    md('## Составные вторичные исходы, частичное правило и группы'),
    code('secondary=secondary_outcome_metrics(test,MODEL_NAMES[1:]+TEMPORAL_SENSITIVITIES)\nsecondary.to_csv(OUT/"temporal_secondary_outcomes.csv",index=False)\nprint("Основные модели: только AUROC/AUPRC на составных метках; Brier здесь не применим")\ndisplay(secondary.round(4))\nrule=pd.concat([compare_guideline_at_budget(test),compare_guideline_at_budget(test,model="hist_gb_early",include_ivf=True)],ignore_index=True)\nrule.to_csv(OUT/"temporal_guideline_comparison.csv",index=False)\ndisplay(rule.round(4))'),
    md('## Отдельное обучение на составных исходах и проверка специфичности'),
    code('secondary_fits=fit_secondary_temporal(resume=True)\ndisplay(secondary_fits.round(4))\ncomposite_comparison,composite_folds=secondary_trained_summary(test)\ndisplay(composite_folds.groupby("model")[["auroc","auprc","brier"]].agg(["mean","std"]).round(4))\ndisplay(composite_comparison.round(4))'),
    md('''## Абсолютный масштаб при одинаковом числе отметок

На 1000 записей с определимым **частичным** правилом сравниваем число наблюдаемых случаев и найденных случаев при том же бюджете отметок. Для составных исходов используются отдельно обученные HGB 2023 и записи 2024 с известным сроком беременности. Это описательная ретроспективная проверка: правило не включает предыдущую ПЭ, почечные и аутоиммунные болезни и не равно полному USPSTF; аспирин в файле не известен.'''),
    code('''secondary_scores = pd.read_parquet(OUT / "secondary_test_2024.parquet",
    columns=["source_year", "source_row", *SECONDARY_TARGETS])
absolute_frame = test.merge(secondary_scores, on=["source_year", "source_row"],
    how="inner", validate="one_to_one")
assert len(absolute_frame) == len(test)
absolute_rows = []
for endpoint, model in [("target_hdp_proxy", "hist_gb"),
                        ("target_hdp_preterm37", "hist_gb_hdp_preterm37"),
                        ("target_hdp_preterm34", "hist_gb_hdp_preterm34")]:
    label_field = "label" if endpoint == "target_hdp_proxy" else endpoint
    known = absolute_frame.loc[absolute_frame[label_field].notna()].copy()
    known["label"] = known[label_field].astype(np.uint8)
    comparison = compare_guideline_at_budget(known, model=model)
    comparison.insert(0, "endpoint", endpoint)
    absolute_rows.append(comparison)
absolute_table = pd.concat(absolute_rows, ignore_index=True)
absolute_table.to_csv(OUT / "temporal_absolute_per_1000.csv", index=False)
display(absolute_table[["endpoint", "model", "n", "coverage", "alert_budget",
    "cases_per_1000", "captured_per_1000", "sensitivity", "ppv"]].round(3))
absolute_summary = []
for endpoint, model in [("target_hdp_proxy", "hist_gb"),
                        ("target_hdp_preterm37", "hist_gb_hdp_preterm37"),
                        ("target_hdp_preterm34", "hist_gb_hdp_preterm34")]:
    part = absolute_table.loc[absolute_table.endpoint.eq(endpoint)].set_index("model")
    reference = part.loc["observed_guideline_subset"]
    learned = part.loc[model]
    absolute_summary.append({"endpoint": endpoint, "n_comparable": int(reference.n),
        "cases_per_1000": reference.cases_per_1000,
        "rule_captured_per_1000": reference.captured_per_1000,
        "model_captured_per_1000": learned.captured_per_1000,
        "gain_per_1000": learned.captured_per_1000 - reference.captured_per_1000,
        "alert_budget": reference.alert_budget})
absolute_summary = pd.DataFrame(absolute_summary)
absolute_summary.to_csv(OUT / "temporal_absolute_summary_per_1000.csv", index=False)
display(absolute_summary.round(3))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, row in zip(axes, absolute_summary.itertuples()):
    ax.bar(["Правило", "Модель"], [row.rule_captured_per_1000, row.model_captured_per_1000],
           color=[GREY, BLUE])
    ax.set(title=row.endpoint.replace("target_", ""), ylabel="Найдено на 1000",
           ylim=(0, max(1, row.rule_captured_per_1000, row.model_captured_per_1000) * 1.2))
fig.suptitle("Проверка 2024 · одинаковое число отметок в каждой паре")
plt.tight_layout(); plt.show()
del secondary_scores, absolute_frame, absolute_rows
gc.collect()'''),
    code('matrix=composite_comparison.pivot(index="evaluation_endpoint",columns="score",values="auroc")\nfig,ax=plt.subplots(figsize=(10,4.8))\nim=ax.imshow(matrix.to_numpy(),vmin=float(matrix.min().min())-.02,vmax=float(matrix.max().max())+.02,cmap="Blues",aspect="auto")\nfor i in range(len(matrix)):\n    for j in range(len(matrix.columns)): ax.text(j,i,f"{matrix.iloc[i,j]:.3f}",ha="center",va="center",color="black")\nax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=25,ha="right")\nax.set_yticks(range(len(matrix)),matrix.index)\nax.set(title="AUROC · целевой составной исход и роды без записанного HDP")\nfig.colorbar(im,ax=ax,label="AUROC");plt.tight_layout();plt.show()'),
    md('''Строки `preterm*_without_recorded_hdp` рассчитаны **только среди записанных HDP-отрицательных** живорождений. Это проверка общей склонности к преждевременным родам, а не формальный отрицательный контроль и не доказательство специфичности к преэклампсии. Brier/калибровка показаны лишь там, где модель обучена на той же составной метке. Интервал `ILLB_R` остаётся аудитным полем: он измерен до текущих родов, и бинарный порог по нему зависит от длительности текущей беременности у пограничных записей.'''),
    code('''sub=temporal_subgroup_audit(dev,test)
art_groups=["art_reported_yes","infertility_treatment_without_art",
            "no_infertility_treatment","art_unknown_reported","art_not_reported"]
display(sub.loc[sub.group.str.startswith(("race_code_","ethnicity_code_")) |
                (sub.group.isin(art_groups) & sub.model.isin(["hist_gb","hist_gb_early"]))].round(4))
reclass=art_reclassification(dev,test)
display(reclass.round(4))'''),
    code('''race=sub.loc[sub.model.eq("hist_gb") & sub.group.str.startswith("race_code_")].sort_values("group")
art=sub.loc[sub.group.isin(art_groups) & sub.model.isin(["hist_gb","hist_gb_early"])]
art=art.pivot(index="group",columns="model",values=["prevalence","mean_prediction"]).reindex(art_groups)
fig,axes=plt.subplots(1,2,figsize=(15,4.8))
axes[0].bar([RACE6.get(int(group.rsplit("_",1)[-1]),group) for group in race.group],race.mean_calibration_gap_pp,color=BLUE)
axes[0].axhline(0,color=GREY);axes[0].tick_params(axis="x",rotation=25)
axes[0].set(ylabel="Наблюдаемая − прогноз, п.п.",title="Разрыв средней калибровки по коду расы")
x=np.arange(len(art_groups)); width=.26
axes[1].bar(x-width,art[("prevalence","hist_gb")],width,label="Наблюдаемо",color=GREY)
axes[1].bar(x,art[("mean_prediction","hist_gb")],width,label="Основная",color=BLUE)
axes[1].bar(x+width,art[("mean_prediction","hist_gb_early")],width,label="Ранняя модель",color=ORANGE)
axes[1].set_xticks(x,["ART", "Лечение без ART", "Без лечения", "Неизвестно", "Не отчётно"],rotation=30,ha="right")
axes[1].set(ylabel="Доля HDP-proxy",title="Калибровка по состоянию лечения бесплодия")
axes[1].legend()
plt.tight_layout();plt.show()'''),
    md('''`RF_INFTR=1, RF_ARTEC=0` выделено отдельно: отсутствие ART не равнозначно отсутствию лечения бесплодия. Группы на графике взаимоисключающие и покрывают временной тест; число записей и событий показано в таблице выше. Порог здесь означает **верхние 10% прогнозов 2023**, а не клинический порог назначения аспирина. Для неотчётного ART индивидуальные прогнозы ранней и основной моделей равны; их бинарные отметки могут различаться из-за двух разных порогов, рассчитанных на OOF 2023. Коды NCHS: AIAN — American Indian/Alaska Native; NHOPI — Native Hawaiian/Other Pacific Islander. Подгрупповой сдвиг и чувствительность при этом пороге — повод проверить перенос и качество метки, а не автоматически включать расу или обучать отдельную калибровку по каждой группе на том же 2024. Нет сведений о назначении и приёме аспирина; сравнение с частичным правилом USPSTF может быть смещено из-за уже проведённой профилактики. Новый intercept допустимо оценивать только на более поздних независимых размеченных данных и затем проверять на следующем периоде.'''),
    md('''## Калибровка по опубликованной расе после исключения импутированных записей

Флаг NCHS восстановлен по исходным строкам и сверяется с подготовленной когортой. Он используется только для аудита, не для обучения. После исключения изменяется состав подгрупп, поэтому разница разрывов не является эффектом самой импутации.'''),
    code('''from src.evaluation.preeclampsia_cdc_race import build_race_imputation_rows, race_imputation_sensitivity, RACE_ROWS
if not RACE_ROWS.exists():
    build_race_imputation_rows()
race_temporal = race_imputation_sensitivity(test, ["logistic_spline", "hist_gb"])
race_temporal.to_csv(OUT / "temporal_race_imputation_sensitivity.csv", index=False)
display(race_temporal.round(4))'''),
    md('## Decision curves только для записанной метки'),
    code('fig,ax=plt.subplots(figsize=(8,5))\nfor name,color in MODEL_PALETTE:\n    curve=decision_curve(test.label.to_numpy(),test[name].to_numpy())\n    curve.to_csv(OUT/f"temporal_decision_curve_{name}.csv",index=False)\n    ax.plot(curve.threshold,curve.model_net_benefit,label=name,color=color)\nax.plot(curve.threshold,curve.treat_all_net_benefit,"k--",label="Все")\nax.axhline(0,color=GREY,label="Никто")\nax.set(xlabel="Порог вероятности записанной метки",ylabel="Net benefit",title="Decision curve · HDP-proxy, не решение об аспирине")\nax.legend();plt.tight_layout();plt.show()'),
    md('## Парный бутстреп 2024 · запускать после основных графиков'),
    code('boot=temporal_bootstrap(bootstrap_rounds=200)\nintervals=boot.groupby("model")[["auroc","auprc","brier"]].quantile([.025,.975]).unstack().round(4)\ndisplay(intervals)'),
    code('if not boot.empty:\n    comparisons=[]\n    for metric, practical_difference in [("auroc",.01),("auprc",.01),("brier",.002)]:\n        paired=boot.pivot(index="repeat",columns="model",values=metric)\n        for challenger in ["logistic_spline","catboost","lightgbm","hist_gb_no_masks","hist_gb_early","hist_gb_demographic"]:\n            delta=paired[challenger]-paired["hist_gb"]\n            if metric=="brier": delta=-delta  # положительное значение означает улучшение\n            comparisons.append({"metric":metric,"challenger":challenger,"reference":"hist_gb","improvement_mean":delta.mean(),"ci_low":delta.quantile(.025),"ci_high":delta.quantile(.975),"practical_threshold":practical_difference})\n    paired_differences=pd.DataFrame(comparisons)\n    paired_differences["practically_meaningful"] = paired_differences.improvement_mean.ge(paired_differences.practical_threshold)\n    paired_differences["ci_excludes_zero"] = paired_differences.ci_low.gt(0) | paired_differences.ci_high.lt(0)\n    paired_differences.to_csv(OUT/"temporal_paired_differences.csv",index=False)\n    display(paired_differences.round(4))'),
    md('''## Как читать результат

Разница временного теста и смешанных фолдов показывает перенос между годами. Выбор предпочтительного кандидата и порог основаны только на OOF 2023. 2024 уже просмотрен в экспертном разборе и не даёт независимого подтверждения качества. Поскольку метка объединяет гестационную гипертензию с ПЭ и неполно записывается, калибровка и decision curves относятся **только к свидетельству о рождении**. Нельзя выводить эти проценты пациенткам или использовать график для решения об аспирине без клинической валидации. ПЭ у женщин с хронической АГ и мертворождения вне этой когорты.'''),
])

tuned_cells = [
    md('''## 1. Данные, бюджет поиска и исходные конфигурации

Источник — подготовленный и проверенный `eligible_all_2023_2024.parquet`; в поиск входят только строки 2023. В таблице ниже видно число вариантов на модель. `prevalence` не имеет гиперпараметров и служит контрольной строкой. Сетка задаёт небольшие клинически интерпретируемые изменения регуляризации и сложности; она не гарантирует глобальный математический оптимум.'''),
    code('audit=json.loads((OUT/"audit.json").read_text())\nprint("Подготовленный файл:",OUT/"eligible_all_2023_2024.parquet")\nprint("2023:",audit["year_checks"]["2023"]["eligible_singleton_rows"],"строк")\nprint("Всего сочетаний модель × конфигурация × фолд:",5*sum(map(len,GRID.values())))\ngrid_rows=[{"model":name,"candidate_id":candidate_id(name,params),"params":json.dumps(params,ensure_ascii=False,sort_keys=True),"fixed_baseline":params==BASELINE_PARAMS[name]} for name,items in GRID.items() for params in items]\ngrid_table=pd.DataFrame(grid_rows)\ndisplay(grid_table)\nfig,ax=plt.subplots(figsize=(9,4))\ncounts=grid_table.groupby("model",sort=False).size()\nax.barh(counts.index,counts.values,color=BLUE);ax.invert_yaxis()\nax.set(xlabel="Конфигураций",title="Бюджет ограниченного поиска · CDC 2023")\nfor i,count in enumerate(counts): ax.text(count+.08,i,f"{count} × 5",va="center")\nax.set_xlim(0,counts.max()*1.3);plt.tight_layout();plt.show()'),
    md('''## 2. Выполнение поиска

Строки 2024 не загружаются как train или validation в этой функции. Запуск может занять много часов на CPU; один поток поиска позволяет контролировать память. Сохранённые результаты проверяются по идентификаторам, меткам и метрикам.'''),
    code('fold_results, summary, best = run_grid_search(resume=True)\nprint("Общий кандидат:",best["overall_winner"])\ndisplay(summary.round(5))\ndisplay(pd.DataFrame(best["best_by_model"]).T.round(5))'),
    md('## 3. Абляции: изменение относительно фиксированной конфигурации той же модели'),
    code('ablation=ablation_results(fold_results)\nablation_summary=ablation.groupby(["model","candidate_id","params_json","is_fixed_baseline"],as_index=False)[["delta_log_loss","delta_brier","delta_auroc","delta_auprc"]].agg(["mean","std"])\ndisplay(ablation_summary.round(6))\nselected_keys={name:chosen["candidate_id"] for name,chosen in best["best_by_model"].items()}\npaired=ablation.loc[ablation.apply(lambda row: row.candidate_id==selected_keys[row.model],axis=1)].copy()\npaired["delta_millinat"]=1000*paired.delta_log_loss\nfig,ax=plt.subplots(figsize=(10,5))\nfor i,name in enumerate(MODEL_NAMES[1:]):\n    part=paired.loc[paired.model.eq(name)].sort_values("fold")\n    ax.scatter(part.delta_millinat,i+np.linspace(-.13,.13,len(part)),color=MODEL_PALETTE[[x for x,_ in MODEL_PALETTE].index(name)][1] if name in dict(MODEL_PALETTE) else GREY,alpha=.6)\n    ax.scatter(part.delta_millinat.mean(),i,color="black",marker="D")\nax.axvline(0,color="black",ls="--");ax.set_yticks(range(len(MODEL_NAMES)-1),MODEL_NAMES[1:])\nax.set(xlabel="Δ log loss × 1000; ниже — лучше фиксированного baseline",title="Парные изменения по пяти фолдам 2023")\nplt.tight_layout();plt.show()'),
    md('''Точки — именно пять фолдов, ромб — их среднее; это **не** независимые пациенты и не доверительный интервал. Отрицательное изменение `log loss` означает улучшение. При малых различиях важны время, калибровка и перенос, даже если формальный минимум выбран однозначно.'''),
    md('## 4. Карта сетки и цена усложнения'),
    code('fig,axes=plt.subplots(2,2,figsize=(14,10))\nheatmap_specs=[("logistic_spline","n_knots","C"),("hist_gb","max_leaf_nodes","l2_regularization"),("catboost","depth","l2_leaf_reg"),("lightgbm","num_leaves","min_child_samples")]\nfor ax,(name,row_param,col_param) in zip(axes.flat,heatmap_specs):\n    part=summary.loc[summary.model.eq(name)].copy()\n    params=part.params_json.map(json.loads)\n    part[row_param]=params.map(lambda item:item[row_param])\n    part[col_param]=params.map(lambda item:item[col_param])\n    baseline=part.loc[part.candidate_id.eq(candidate_id(name,BASELINE_PARAMS[name])),"cv_log_loss"].iloc[0]\n    matrix=part.pivot(index=row_param,columns=col_param,values="cv_log_loss")\n    sns.heatmap((matrix-baseline)*1000,annot=True,fmt=".2f",cmap="RdBu_r",center=0,ax=ax,cbar_kws={"label":"Δ log loss × 1000"})\n    ax.set(title=name,xlabel=col_param,ylabel=row_param)\nplt.suptitle("2023 OOF · изменение к исходным параметрам той же модели",y=1.01)\nplt.tight_layout();plt.show()\nlinear=summary.loc[summary.model.eq("logistic")].copy()\nlinear["C"]=linear.params_json.map(lambda value:json.loads(value)["C"])\nfig,ax=plt.subplots(figsize=(7,4))\nax.plot(linear.C,linear.cv_log_loss,marker="o",color=BLUE)\nax.set(xscale="log",xlabel="C · слабее регуляризация вправо",ylabel="OOF log loss",title="Линейная логистика · абляция регуляризации")\nplt.tight_layout();plt.show()'),
    md('## 5. Лучшие варианты: качество и вычислительная цена'),
    code('selected_summary=summary.loc[summary.apply(lambda row:row.candidate_id==selected_keys[row.model],axis=1)].set_index("model").reindex(MODEL_NAMES).reset_index()\ndisplay(selected_summary.round(5))\nfig,axes=plt.subplots(2,2,figsize=(13,9))\nfor ax,column,label in zip(axes.flat,["cv_log_loss","cv_brier","auroc_fold_mean","auprc_fold_mean"],["Log loss · ниже лучше","Brier · ниже лучше","AUROC · среднее 5 фолдов","AUPRC · среднее 5 фолдов"]):\n    ax.barh(selected_summary.model,selected_summary[column],color=[GREY,"#6A91B1",BLUE,ORANGE,"#287D69","#8062A5"])\n    ax.invert_yaxis();ax.set(xlabel=label,title=label)\nfig.suptitle("Выбранные конфигурации · 2023 OOF; победитель по log loss",y=1.01)\nplt.tight_layout();plt.show()\nfig,axes=plt.subplots(1,2,figsize=(12,4.5))\naxes[0].barh(selected_summary.model,selected_summary.fit_seconds_total/3600,color=BLUE)\naxes[0].set(xlabel="Сумма времени пяти обучений, часы",title="Цена обучения")\naxes[1].barh(selected_summary.model,selected_summary.rss_gb_max,color=ORANGE)\naxes[1].set(xlabel="Максимальный RSS процесса после fit, ГБ",title="Ориентир памяти, не пик")\nfor ax in axes:ax.invert_yaxis()\nplt.tight_layout();plt.show()'),
    code('selected_fold=fold_results.loc[fold_results.apply(lambda row:row.candidate_id==selected_keys[row.model],axis=1)].copy()\ndisplay(selected_fold[["fold","model","auroc","auprc","brier","log_loss","n","events"]].sort_values(["model","fold"]).round(5))\nfig,axes=plt.subplots(2,2,figsize=(14,9))\nfor ax,metric,title in zip(axes.flat,["log_loss","brier","auroc","auprc"],["Log loss","Brier","AUROC","AUPRC"]):\n    for i,name in enumerate(MODEL_NAMES):\n        part=selected_fold.loc[selected_fold.model.eq(name)].sort_values("fold")\n        ax.scatter(part[metric],i+np.linspace(-.16,.16,5),s=28,alpha=.7,color=BLUE)\n        ax.scatter(part[metric].mean(),i,s=65,marker="D",color=ORANGE)\n    ax.set_yticks(range(len(MODEL_NAMES)),MODEL_NAMES);ax.invert_yaxis()\n    ax.set(xlabel=title,title=f"{title} · пять фолдов 2023")\nfig.suptitle("Разброс после выбора: точки — фолды, ромб — среднее",y=1.01)\nplt.tight_layout();plt.show()'),
    md('''## 6. Калибровка отложенных прогнозов выбранных конфигураций

На графике все записи 2023 имеют прогноз модели, обученной без их фолда. Выбор конфигураций использовал эти же метки, поэтому эти кривые служат **диагностикой после выбора**, а не независимой оценкой настроенного алгоритма.'''),
    code('selected_oof=selected_oof_predictions()\nfig,ax=plt.subplots(figsize=(7,6))\nfor name,color in [("logistic","#6A91B1"),*MODEL_PALETTE]:\n    observed,predicted=calibration_curve(selected_oof.label,selected_oof[name],n_bins=12,strategy="quantile")\n    ax.plot(predicted,observed,marker="o",label=name,color=color)\nax.plot([0,.4],[0,.4],ls="--",color="black")\nax.set(xlim=(0,.4),ylim=(0,.4),xlabel="Средний прогноз в бине",ylabel="Наблюдаемая доля HDP-proxy",title="2023 OOF · калибровка после выбора")\nax.legend();plt.tight_layout();plt.show()'),
    code('temporal_selected=score_selected_on_2024(resume=True)\ndisplay(temporal_selected.round(5))\ntemporal_scores=pd.read_parquet(TUNING_DIR/"selected_temporal_2024.parquet")\nfig,axes=plt.subplots(1,2,figsize=(13,5))\nfor name,color in [("logistic","#6A91B1"),*MODEL_PALETTE]:\n    observed,predicted=calibration_curve(temporal_scores.label,temporal_scores[name],n_bins=12,strategy="quantile")\n    axes[0].plot(predicted,observed,marker="o",label=name,color=color)\naxes[0].plot([0,.4],[0,.4],ls="--",color="black")\naxes[0].set(xlim=(0,.4),ylim=(0,.4),xlabel="Средний прогноз",ylabel="Наблюдаемая доля",title="2024 · калибровка выбранных вариантов")\njoined=temporal_selected.merge(selected_summary[["model","cv_log_loss"]],on="model")\naxes[1].barh(joined.model,1000*(joined.log_loss-joined.cv_log_loss),color=ORANGE)\naxes[1].axvline(0,color="black",ls="--");axes[1].invert_yaxis()\naxes[1].set(xlabel="2024 − 2023 OOF log loss × 1000",title="Изменение между годами")\naxes[0].legend();plt.tight_layout();plt.show()'),
    code('fig,axes=plt.subplots(2,2,figsize=(14,11))\nfor row,(year,frame) in enumerate([(2023,selected_oof),(2024,temporal_scores)]):\n    y=frame.label.to_numpy()\n    axes[row,0].plot([0,1],[0,1],ls="--",color=GREY,label="случайная модель")\n    axes[row,1].axhline(y.mean(),ls="--",color=GREY,label=f"частота {y.mean():.3f}")\n    for name,color in [("logistic","#6A91B1"),*MODEL_PALETTE]:\n        p=frame[name].to_numpy()\n        fpr,tpr,_=roc_curve(y,p)\n        precision,recall,_=precision_recall_curve(y,p)\n        roc_points=np.linspace(0,len(fpr)-1,min(len(fpr),1200)).astype(int)\n        pr_points=np.linspace(0,len(recall)-1,min(len(recall),1200)).astype(int)\n        axes[row,0].plot(fpr[roc_points],tpr[roc_points],label=f"{name} · {auc(fpr,tpr):.3f}",color=color)\n        axes[row,1].plot(recall[pr_points],precision[pr_points],label=f"{name} · {-np.sum(np.diff(recall)*precision[:-1]):.3f}",color=color)\n    axes[row,0].set(xlim=(0,1),ylim=(0,1),xlabel="1 − специфичность",ylabel="Чувствительность",title=f"{year} · ROC / AUROC")\n    axes[row,1].set(xlim=(0,1),ylim=(0,1),xlabel="Полнота",ylabel="Точность",title=f"{year} · PR / AUPRC")\n    axes[row,0].legend(fontsize=8,loc="lower right");axes[row,1].legend(fontsize=8,loc="upper right")\nfig.suptitle("Выбранные варианты: 2023 — после выбора, 2024 — исследовательская проверка",y=1.01)\nplt.tight_layout();plt.show()'),
    md('''Разность `log loss` между 2024 и OOF 2023 объединяет **смену года**, изменение частоты метки и разный объём обучения: OOF-модели обучались на четырёх фолдах, модели для 2024 — на всём 2023. График показывает итоговое изменение качества, но не измеряет отдельно дрейф данных. Выбор настроек по тем же OOF-меткам делает кривые 2023 оптимистичными.'''),
    md('''## 9. Распределение внешних метрик

Каждая точка — один внешний фолд. Строка `selected` отражает **полную процедуру выбора типа модели и параметров внутри фолда**; в разных внешних фолдах тип модели может различаться. Пять точек показывают устойчивость к разделению, но не являются независимыми пациентами или доверительным интервалом.'''),
    code('''plot_models = MODEL_NAMES + ["selected"]
palette = {"prevalence": GREY, "logistic": "#6A91B1", **dict(MODEL_PALETTE), "selected": "#202020"}
display(outer_nested[["outer_fold", "model", "candidate_id", "n", "events", "auroc", "auprc", "brier", "log_loss"]]
        .sort_values(["model", "outer_fold"]).round(5))
spread_nested = outer_nested.groupby("model")[["auroc", "auprc", "brier", "log_loss"]].agg(["mean", "std", "min", "max"])
display(spread_nested.round(5))
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, metric, title in zip(axes.flat, ["auroc", "auprc", "brier", "log_loss"],
                             ["AUROC · выше лучше", "AUPRC · выше лучше", "Brier · ниже лучше", "Log loss · ниже лучше"]):
    for position, name in enumerate(plot_models):
        part = outer_nested.loc[outer_nested.model.eq(name)].sort_values("outer_fold")
        assert part.outer_fold.tolist() == list(range(5))
        ax.scatter(part[metric], position + np.linspace(-.17, .17, 5), color=palette[name], alpha=.7, s=35)
        ax.scatter(part[metric].mean(), position, marker="D", color="black", s=55)
    ax.set_yticks(range(len(plot_models)), plot_models)
    ax.invert_yaxis()
    ax.set(xlabel=title, title=title)
fig.suptitle("Внешние фолды 2023–2024 · настроенные модели и полная процедура выбора", y=1.01)
plt.tight_layout(); plt.show()'''),
    md('''## 10. Стабильность выбранных параметров и разрыв внутренней/внешней оценки

Частоты выбранных конфигураций и типов моделей помогают увидеть нестабильный выбор. Внутренний и внешний `log loss` получены на разных записях и при разном размере обучающей части; их разность описательна и **не является чистой оценкой переобучения**.'''),
    code('''stability = (outer_nested.loc[outer_nested.model.isin(MODEL_NAMES)]
             .groupby(["model", "candidate_id", "params_json"], as_index=False)
             .agg(outer_folds=("outer_fold", "nunique")))
display(stability.sort_values(["model", "outer_folds"], ascending=[True, False]))
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, name in zip(axes.flat, MODEL_NAMES):
    part = stability.loc[stability.model.eq(name)].sort_values("outer_folds")
    ax.barh(part.candidate_id, part.outer_folds, color=palette[name])
    ax.set(xlim=(0, 5.5), xlabel="Внешних фолдов с таким выбором", title=name)
    ax.set_xticks(range(6))
plt.tight_layout(); plt.show()
selection_diagnostic = []
for choice in choices_nested.itertuples(index=False):
    name = choice.overall_winner
    key = getattr(choice, f"{name}_candidate_id")
    inner_part = inner_nested.loc[inner_nested.outer_fold.eq(choice.outer_fold)
                                  & inner_nested.model.eq(name)
                                  & inner_nested.candidate_id.eq(key)]
    external = outer_nested.loc[outer_nested.outer_fold.eq(choice.outer_fold)
                                & outer_nested.model.eq("selected")].iloc[0]
    assert len(inner_part) == 4
    selection_diagnostic.append({"outer_fold": choice.outer_fold, "model": name,
        "inner_log_loss": np.average(inner_part.log_loss, weights=inner_part.n),
        "outer_log_loss": external.log_loss})
selection_diagnostic = pd.DataFrame(selection_diagnostic)
display(selection_diagnostic.round(5))
fig, ax = plt.subplots(figsize=(8, 5))
for row in selection_diagnostic.itertuples(index=False):
    ax.plot([0, 1], [row.inner_log_loss, row.outer_log_loss], marker="o", label=f"фолд {row.outer_fold}: {row.model}")
ax.set_xticks([0, 1], ["Внутренняя CV", "Внешний фолд"])
ax.set(ylabel="Log loss", title="Выбранный тип модели · внутреннее и внешнее качество")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()'''),
    md('''## 11. ROC, PR и калибровка внешних прогнозов

Численные метрики в таблице выше и калибровка ниже считаются по **всем** отложенным записям. Для читаемых ROC/PR-кривых используется случайная выборка до 300 000 записей; их подписи относятся только к этой выборке. Чёрная линия — процедура выбора (`selected`), остальные — каждая семья моделей с собственным внутренним подбором.'''),
    code('''plot_sample = oof_nested.sample(n=min(300_000, len(oof_nested)), random_state=20261001)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for name in plot_models:
    style = {"color": palette[name], "linestyle": "--" if name == "selected" else "-"}
    RocCurveDisplay.from_predictions(plot_sample.label, plot_sample[name], ax=axes[0], name=name, curve_kwargs=style)
    PrecisionRecallDisplay.from_predictions(plot_sample.label, plot_sample[name], ax=axes[1], name=name, curve_kwargs=style)
    observed, predicted = calibration_curve(oof_nested.label, oof_nested[name], n_bins=12, strategy="quantile")
    axes[2].plot(predicted, observed, marker="o", label=name, **style)
axes[2].plot([0, .5], [0, .5], "k:")
axes[0].set(title="ROC · случайная выборка")
axes[1].set(title="Precision–recall · случайная выборка")
axes[2].set(xlim=(0, .5), ylim=(0, .5), xlabel="Средний прогноз", ylabel="Наблюдаемая доля", title="Калибровка · все записи")
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
    md('''## 12. Разбивка внешних прогнозов по году

Здесь фолды смешанные: модели для записей 2023 могут обучаться на части 2024, и наоборот. Поэтому эта таблица описывает неоднородность внутри вложенной оценки, **не** перенос 2023→2024. Для направленной проверки используйте раздел 7 выше.'''),
    code('''year_rows = []
for year, part in oof_nested.groupby("source_year", sort=True):
    for name in plot_models:
        year_rows.append({"year": int(year), "model": name, "n": len(part),
            "events": int(part.label.sum()), **metrics(part.label.to_numpy(), part[name].to_numpy())})
year_nested = pd.DataFrame(year_rows)
year_nested.to_csv(NESTED_DIR / "metrics_by_year.csv", index=False)
display(year_nested.round(5))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for name in plot_models:
    part = year_nested.loc[year_nested.model.eq(name)].sort_values("year")
    axes[0].plot(part.year, part.auprc, marker="o", label=name, color=palette[name])
    axes[1].plot(part.year, part.log_loss, marker="o", label=name, color=palette[name])
axes[0].set(ylabel="AUPRC", title="Смешанный nested CV · AUPRC по году")
axes[1].set(ylabel="Log loss", title="Смешанный nested CV · log loss по году")
for ax in axes:
    ax.set_xticks([2023, 2024])
    ax.set_xlabel("Год записи")
axes[0].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()'''),
    md('''## 13. Вторичные исходы, частичное правило и подгруппы

Внешние прогнозы настроенных моделей сохраняют вторичные метки и поля аудита. Для сочетаний HDP-proxy с родами до 37/34 недель показывается только **дискриминация**: вероятность HDP-proxy не является калиброванной вероятностью составного исхода. Частичное наблюдаемое правило USPSTF сравнивается при одинаковом числе отмеченных записей; пропущенные клинические факторы не считаются отрицательными. Подгруппы нужны для аудита регистрации и калибровки, а не для биологического объяснения различий.'''),
    code('''secondary_nested = secondary_outcome_metrics(oof_nested, MODEL_NAMES[1:] + ["selected"])
secondary_nested.to_csv(NESTED_DIR / "secondary_outcome_discrimination.csv", index=False)
display(secondary_nested.round(5))
rule_nested = pd.concat([
    compare_guideline_at_budget(oof_nested, model="selected"),
], ignore_index=True)
rule_nested.to_csv(NESTED_DIR / "guideline_comparison.csv", index=False)
display(rule_nested.round(5))
groups = {"diabetes_recorded_yes": oof_nested.rf_pdiab.eq(1),
          "diabetes_recorded_no": oof_nested.rf_pdiab.eq(0),
          "bmi_missing": oof_nested.bmi_missing.eq(1),
          "prior_live_birth_count_missing": oof_nested.prior_live_birth_count_missing.eq(1)}
groups.update({f"race_code_{int(code)}": oof_nested.mrace6.eq(code)
               for code in sorted(oof_nested.mrace6.dropna().unique())})
subgroup_rows = []
for group, mask in tqdm(groups.items(), desc="Nested subgroup audit", unit="group"):
    part = oof_nested.loc[mask]
    if len(part) < 100 or part.label.nunique() < 2:
        continue
    subgroup_rows.append({"group": group, "n": len(part), "events": int(part.label.sum()),
        **metrics(part.label.to_numpy(), part.selected.to_numpy()),
        "mean_calibration_gap_pp": 100 * (part.label.mean() - part.selected.mean())})
subgroups_nested = pd.DataFrame(subgroup_rows)
subgroups_nested.to_csv(NESTED_DIR / "selected_subgroups.csv", index=False)
display(subgroups_nested.round(5))
fig, ax = plt.subplots(figsize=(10, max(5, .5 * len(subgroups_nested))))
ax.barh(subgroups_nested.group, subgroups_nested.mean_calibration_gap_pp, color=BLUE)
ax.axvline(0, color="black", ls="--")
ax.set(xlabel="Наблюдаемая − средний прогноз, процентные пункты",
       title="Настроенный алгоритм · разрыв средней калибровки по группам")
plt.tight_layout(); plt.show()'''),
    md('''## Как читать результаты

Минимум `log loss` выбирает конфигурацию по качеству вероятностей в заданной сетке, но не исправляет ошибку регистрации HDP-proxy. Частота и полнота кодирования гестационной гипертензии/преэклампсии, уже существовавшая профилактика, отсутствие ID женщины и факт просмотра 2024 ограничивают клинический вывод. Если преимущество сложной модели мало, опирайтесь на калибровку, перенос и простоту — не на десятые доли процента AUROC. Следующая честная проверка общей модели требует нового периода или независимого клинического набора с подтверждёнными диагнозами.'''),
]


def relocated(cell):
    result = deepcopy(cell)
    if result.cell_type == "markdown":
        result.source = re.sub(r"^## \d+\. ", "## ", result.source)
        result.source = result.source.replace("раздел 7 выше", "ноутбук 06")
    return result

# Keep the original fixed baselines as a reference, then run the nested search
# before any tuned-model evaluation. Existing executed cells retain outputs.
books["02_train_baselines.ipynb"].extend([
    md('''## Настроенные модели: вложенный подбор на общих фолдах

Для каждого внешнего фолда параметры и тип модели выбираются исключительно на остальных четырёх фолдах. Затем модели обучаются на этих четырёх фолдах и прогнозируют отложенный. Таким образом оценка в 03 относится к **процедуре выбора**, а не к параметрам одного готового экземпляра. Прогресс каждого кандидата и каждого фолда показывается при обучении; подтверждённые результаты сохраняются и повторно используются при перезапуске. Полный поиск на миллионах записей запускает пользователь.'''),
    code('''import gc
from src.training.preeclampsia_cdc_tuning import GRID
from src.training.preeclampsia_cdc_nested import NESTED_DIR, run_nested_cv
from src.training.preeclampsia_cdc_final import FINAL_DIR, run_full_cohort_search, refit_full_cohort_best_models
print("Внутренних кандидатных fit:", 5 * 4 * sum(map(len, GRID.values())))
print("Внешних оценок моделей:", 5 * (len(MODEL_NAMES) + 1))
print("Финальных кандидатных fit:", 5 * sum(map(len, GRID.values())))'''),
    code('''inner_nested, outer_nested, choices_nested, oof_nested = run_nested_cv(resume=True)
expected_rows = sum(x["eligible_singleton_rows"] for x in json.loads((OUT / "audit.json").read_text())["year_checks"].values())
assert len(inner_nested) == 5 * 4 * sum(map(len, GRID.values()))
assert len(outer_nested) == 5 * (len(MODEL_NAMES) + 1)
assert len(oof_nested) == expected_rows
assert oof_nested.groupby(["source_year", "fold"]).size().gt(0).all()
nested_n, nested_events = len(oof_nested), int(oof_nested.label.sum())
print("Внешние OOF-прогнозы:", f"{nested_n:,}")
display(choices_nested)
del inner_nested, outer_nested, choices_nested, oof_nested
gc.collect()'''),
    md('''## Выбор финальных параметров и обучение на всех записях

После завершения вложенной оценки проводится **отдельный** пятифолдовый поиск на всей когорте 2023+2024. Он выбирает параметры каждой модели и общий тип по взвешенному `log loss`. Эти поисковые метрики оптимистичны после выбора; внешние метрики вложенного поиска смотрите в 03. Итоговые модели обучаются на всех записях и не оцениваются на этих же строках.'''),
    code('''final_folds, final_summary, final_best = run_full_cohort_search(resume=True)
assert len(final_folds) == 5 * sum(map(len, GRID.values()))
assert final_best["development_n"] == nested_n
assert final_best["development_events"] == nested_events
print("Победитель полного подбора:", final_best["overall_winner"])
display(final_summary.round(5))
display(pd.DataFrame(final_best["best_by_model"]).T)
del final_folds
gc.collect()'''),
    code('''global_manifest = refit_full_cohort_best_models(resume=True)
assert global_manifest["overall_winner"] == final_best["overall_winner"]
print("Итоговый победитель:", global_manifest["overall_winner"])
display(pd.DataFrame([{"model": name, "artifact": path} for name, path in global_manifest["artifacts"].items()]))'''),
])

books["03_evaluate_folds.ipynb"].extend([
    md('''## Внешняя оценка настроенной процедуры · все записи 2023+2024

Раздел читает и проверяет сохранённые прогнозы из 02. **Обучения и подбора здесь нет.** Внешние метки каждого фолда не использовались для выбора параметров этого фолда. Для вывода показаны распределения, стабильность выбора, ROC/PR, калибровка и разрезы. Оба года перемешаны во внешних фолдах: разбивка по году не является проверкой переноса.'''),
    code('''import gc
for old_name in ("oof", "plot_sample", "sub", "groups"):
    globals().pop(old_name, None)
gc.collect()
from src.training.preeclampsia_cdc_nested import NESTED_DIR, load_nested_results
from src.training.preeclampsia_cdc_final import FINAL_DIR, load_full_cohort_best
inner_nested, outer_nested, choices_nested, oof_nested = load_nested_results()
pooled_nested = pd.read_csv(NESTED_DIR / "pooled_metrics.csv")
print("Внутренних проверок:", len(inner_nested), "· внешних оценок:", len(outer_nested))
print("Внешних OOF-прогнозов:", f"{len(oof_nested):,}")
display(choices_nested)
display(pooled_nested.round(5))'''),
    *[relocated(cell) for cell in tuned_cells[17:24]],
    deepcopy(tuned_cells[24]),
    relocated(tuned_cells[25]),
    deepcopy(tuned_cells[26]),
    md('''## Импутация расы и внешний прогноз настроенной процедуры

Та же строковая проверка выполняется для `selected` после вложенного подбора. Здесь тип модели может различаться по внешним фолдам; исключение импутированных строк меняет состав подгруппы и служит проверкой устойчивости описательного разрыва.'''),
    code('''from src.evaluation.preeclampsia_cdc_race import race_imputation_sensitivity
nested_race = race_imputation_sensitivity(oof_nested, ["selected"])
nested_race.to_csv(NESTED_DIR / "selected_race_imputation_sensitivity.csv", index=False)
display(nested_race.round(4))'''),
    code('''for old_name in ("groups", "plot_sample"):
    globals().pop(old_name, None)
gc.collect()'''),
    md('''## Финальный выбор по всей когорте · диагностика, не независимая оценка

Итоговые параметры выбраны в 02 отдельным поиском на пяти общих фолдах. Ни точки ниже, ни их метрики не являются внешней проверкой выбранной итоговой модели. Для оценки процедуры используйте внешние результаты выше.'''),
    code('''final_best = load_full_cohort_best()
final_summary = pd.read_csv(FINAL_DIR / "candidate_summary.csv", dtype={"candidate_id": str})
assert final_best["development_n"] == len(oof_nested)
assert final_best["development_events"] == int(oof_nested.label.sum())
print("Итоговый победитель:", final_best["overall_winner"])
display(final_summary.round(5))
display(pd.DataFrame(final_best["best_by_model"]).T)
final_selected = final_summary.loc[final_summary.apply(
    lambda row: row.candidate_id == final_best["best_by_model"][row.model]["candidate_id"], axis=1)]
fig, ax = plt.subplots(figsize=(10, 4.8))
for name in MODEL_NAMES:
    part = final_summary.loc[final_summary.model.eq(name)]
    ax.scatter(part.cv_log_loss, np.repeat(name, len(part)), color=palette[name], alpha=.55)
    chosen = final_selected.loc[final_selected.model.eq(name)].iloc[0]
    ax.scatter(chosen.cv_log_loss, name, color="black", marker="D", s=70)
ax.set(xlabel="Пятифолдовый log loss; меньше — лучше", title="Финальный подбор · точки кандидатов, ромбы выбранных")
plt.tight_layout(); plt.show()
global_manifest = json.loads((ROOT / "models/exported/preeclampsia_cdc/global_full_cv_2023_2024.json").read_text())
assert global_manifest["overall_winner"] == final_best["overall_winner"]
display(pd.DataFrame([{"model": name, "artifact": path} for name, path in global_manifest["artifacts"].items()]))'''),
    deepcopy(tuned_cells[27]),
])

temporal_cells = books["06_temporal_validation.ipynb"]
books["06_temporal_validation.ipynb"] = temporal_cells[:2] + [
    md('''## Подбор конфигураций только по 2023

До графиков 2024 все конфигурации сравниваются на пяти фолдах 2023. Выбор каждого варианта и общего типа модели основан на взвешенном OOF `log loss` 2023. Результаты выбора на этих же фолдах оптимистичны и служат диагностикой. Сетка была задана заранее; 2024 год уже просматривался в проекте, поэтому последующая временная проверка остаётся исследовательской.'''),
    code('''import gc
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay
from src.training.preeclampsia_cdc_tuning import (GRID, BASELINE_PARAMS, TUNING_DIR,
    candidate_id, run_grid_search, ablation_results, selected_oof_predictions,
    score_selected_on_2024)
sns.set_theme(style="whitegrid", font_scale=1.0)'''),
    *[deepcopy(cell) for cell in tuned_cells[0:14]],
    code('''del selected_oof
gc.collect()'''),
] + temporal_cells[2:] + [
    md('''## Выбранные по 2023 варианты: исследовательская проверка на 2024

Каждый выбранный вариант обучается на всём 2023 и проверяется на 2024. Параметры не переопределяются по меткам 2024; уже имевшийся просмотр результатов 2024 ограничивает силу вывода.'''),
    code('''for old_name in ("dev", "test", "sub"):
    globals().pop(old_name, None)
gc.collect()
selected_oof = selected_oof_predictions()'''),
    *[deepcopy(cell) for cell in tuned_cells[14:17]],
]

for name, cells in books.items():
    write_book(name, cells)
(DEST / "07_hyperparameter_ablations.ipynb").unlink(missing_ok=True)
print(DEST)
