"""Current documentation and research-model registries for the CTG revision."""
from pathlib import Path
import sys
import json,hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REVIEW=ROOT/'research/reviews/ctg_20261017'
UCI=ROOT/'research/experiments/ctg_uci/reviewed_v3'
JNU=ROOT/'research/experiments/jnu_ctg/reviewed_v2'


def write(path,text):Path(path).write_text(text)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package():
    u=pd.read_csv(UCI/'repeat_metrics.csv');up=u[u.family=='selected_procedure']
    j=pd.read_csv(JNU/'profile_metrics.csv');jp=j[j.family=='selected_procedure']
    write(ROOT/'research/notebooks/ctg_uci/README.md',f'''# UCI Cardiotocography · reviewed v3

UCI — benchmark трёх экспертных классов NSP. Префиксы имён не подтверждены как patient IDs; они используются как более строгие группы зависимости. 352 файла объединены в 177 групп. Все данные ранее просмотрены, нового независимого test нет.

## Текущие ноутбуки

1. [01 · подготовка](01_prepare_data.ipynb): данные, распределения, пропуски/типы, префиксы, перекрывающиеся сегменты.
2. [02 · nested Grid Search](02_nested_grid_search.ipynb): 3 повтора 5×3 CV, шесть семейств, group-balanced log loss и One-SE простота.
3. [03 · калибровка и экспорт](03_calibration_ensemble_and_test.ipynb): fit-side OOF temperature, argmax, финальные benchmark-модели.
4. [04 · кластеризация](04_clustering_without_outcomes.ipynb): сохранённый отрицательный результат, ветка закрыта.
5. [05 · ошибки и интервалы](05_results_and_error_analysis.ipynb): все классы, suspect, CLASS, bootstrap префиксов.
6. [06 · UMAP/AE](06_clustering_embeddings_grid_search.ipynb): сохранены 1650 кандидатов; AE+GMM неустойчив.
7. [07 · критический аудит](07_deep_audit_and_recommendations.ipynb): ответы на все 23 пункта, проверенные факты и ограничения.
8. [08 · вход бланка](08_hospital_reports_feature_engineering.ipynb): единый контракт приложения, отказ до модели, различия определений.

## Основной результат

Средние метрики исправленной selected_procedure по трём повторам: group-balanced Macro-F1 **{up.group_macro_f1.mean():.3f}**, recall suspect **{up.group_recall_suspect.mean():.1%}**, recall pathologic **{up.group_recall_path.mean():.1%}**. Это оценка процедуры выбора; не независимый score финального файла модели. Segment-weighted метрики показаны рядом.

CI ресэмплируют группы с кратностью и условны на сохранённых CV fits. One-SE — эвристика простоты. Группы могут объединить разных пациенток и не исключают одну пациентку с разными префиксами. Классы — экспертная морфология, не исход ребёнка.

## Воспроизведение

```bash
.venv/bin/python -m src.training.ctg_review_evidence
.venv/bin/python -m src.training.ctg_prefix_benchmark
.venv/bin/python -m src.training.ctg_review_assessment uci
.venv/bin/python tools/execute_notebooks_inprocess.py research/notebooks/ctg_uci/02_nested_grid_search.ipynb
.venv/bin/python tools/validate_ctg_review.py
```

Общий генератор 12 актуальных ноутбуков: `tools/build_ctg_review_notebooks.py`; `--summaries-only` сохраняет выполненные outputs. Код обучения: `ctg_prefix_benchmark.py`, `ctg_review_models.py`. Фиксированные входы и source hashes в protocol.json; завершённая версия проверяет хеши перед replay.

Новые результаты: `research/experiments/ctg_uci/reviewed_v3/`. Модели: `models/exported/ctg_uci/reviewed_v3/`. Прежние notebooks/documents: `research/reviews/ctg_20261017/before_revision/`. Старые v1/v2 результаты и модели сохранены как история, не являются текущим выбором и не применяются к произвольным бланкам.
''')
    table='| Профиль | Macro-F1 принятых | Recall abnormal | Охват JNU |\n|---|---:|---:|---:|\n'
    for _,r in jp.iterrows():table+=f'| {r.profile} | {r.group_macro_f1:.3f} | {r.group_recall:.1%} | {r.coverage:.2%} |\n'
    write(ROOT/'research/notebooks/jnu_ctg/README.md',f'''# JNU-CTG · reviewed v2 · поля больничного бланка

Источник: [JNU-CTG 1.0.0](https://zenodo.org/records/21800730), CC BY 4.0, 20769 записей /12606 пациенток. Цель моделей — экспертная binary FIGO normal/abnormal. Асфиксия и медицинский риск не являются целями этих классификаторов.

## Текущие ноутбуки

- [01 · подготовка](01_prepare_data.ipynb): сохранён глубокий аудит источника, маски/дубли/метаданные; добавлены UC signatures, published U_* и согласие разных записей пациентки.
- [02 · измерения и контракт](02_report_features.ipynb): прежние вычисления сохранены; проверены определения амплитуды, единицы, редкая низкая STV, единый отказ до модели.
- [03 · переобучение](03_report_classifiers.ipynb): 12 strict report-моделей, четыре профиля, nested CV, отдельные mode/calibration группы, сигнальные ориентиры.
- [04 · проверка](04_classifier_assessment.ipynb): интервалы, отказы, низкая STV, качество, UC signatures, synthetic probes, связь с исходом и сценарный PPV.

## Результат исправленной процедуры

{table}
Все данные уже изучались; это post-review exploratory nested CV, не новая слепая когорта. Метрики patient-balanced по записям, охват — доля полных собственных измерений. Он не означает совместимость с больничным прибором. Модели выбираются внутри fit; финальный full-cohort export не получает независимый test-score.

## Что изменено

Убраны quality/duration из X, дубли STV(ms/BPM) и amplitude(ms/BPM), непроверенные episode-rules. Четыре текущих профиля: core3, report5, report7, report4s5. Fit только на полных полях профиля; отсутствующие поля считаются отказами во всей исходной когорте. В report-моделях нет imputation и missing indicators. Скейлеры/сплайны fit заново внутри fit и сохранены отдельно.

Сетка CatBoost: 16 конфигураций depth/l2/iterations/class weights при learning_rate=0.05; сплайны: 6, линейная: 3. В каждой внешней fit-части independent calibration 20%; оставшаяся proper-fit имеет отдельный mode holdout, не участвующий в grid. Сравнение single/ensemble оценивает одну выбранную конфигурацию на этом holdout и не доказывает превосходства всех возможных ансамблей.

HGB-ориентиры на одинаковых группах и report5-complete когорте: report5, FHR std, signal27, raw-minute210, quarantined F_*, quality-only и report+quality. Raw-minute210 — временные дескрипторы двух исходных каналов, не end-to-end нейросеть и не теоретический потолок качества. Published features не становятся допустимыми входами приложения.

## Единый вход приложения

`src.features.ctg_report_contract`: parse_ctg_report → prepare_report → evaluate_report. При missing/unconfirmed required fields predict_proba не вызывается и вероятность не возвращается. Неподтверждённое определение сохраняется отдельно от пропуска. Сейчас нет внешне валидированных bridges устройств; application mode возвращает abstained /clinical_risk=None. Наличие корректных чисел не превращает expert-label вероятность в риск гипоксии.

В v1 амплитуда — минутный range с акцелерациями. Новый event-excluded descriptor исследован на 128 train-записях, не объявлен эталонным. Полный D/R или Фишер по неполному бланку не рассчитывается. Индивидуальный срок/время до родов неизвестны; 30-min обучение не разрешает 17-min вход или применение с 28 недель.

## Артефакты и воспроизведение

Модели: `models/exported/jnu_ctg/reviewed_v2/`, включая отдельные scalers/bases и 35 fold reference-models. Результаты: `research/experiments/jnu_ctg/reviewed_v2/`. Первичная подготовка и исходные измерения: прежние `v1/` и `report_features_v1/`; reference scalers 02 не используются classifiers v2. Прежние 15 classifiers_v1 сохранены для истории, не являются текущим application-путём.

```bash
.venv/bin/python -m src.training.ctg_review_evidence
.venv/bin/python -m src.training.jnu_review_experiment
.venv/bin/python -m src.training.ctg_review_assessment jnu
.venv/bin/python tools/execute_notebooks_inprocess.py research/notebooks/jnu_ctg/03_report_classifiers.ipynb
.venv/bin/python tools/validate_ctg_review.py
```

Старые notebooks/documents сохранены в `research/reviews/ctg_20261017/before_revision/`. Все 23 решения — в UCI [07](../ctg_uci/07_deep_audit_and_recommendations.ipynb) и `expert_dispositions.csv`.
''')
    write(ROOT/'research/notebooks/jnu_ctg/ANALYSIS_PLAN.md','''# JNU-CTG · план после экспертного ревью

## Выполненная ревизия

1. Сохранены исходные сигналы, маски и операционные измерения v1. Подтверждены проблемы поддержки низкой STV, missingness/quality association, неоднозначных published U_* и разных формул амплитуды.
2. Обучены strict single-unit report-модели: четыре профиля, CatBoost/сплайны/линейная логистика, grouped nested CV 5×3, отдельные mode/calibration группы. Все source rows участвуют в учёте охвата, NaN не получает прогноз.
3. Сравнены семь представлений одним HGB на одинаковых фолдах/когорте. Сырые временные дескрипторы и published F_* — исследовательские ориентиры, не скрытые признаки приложения.
4. Проверены conditional CI, отказы, низкая STV, качество, сигнатуры UC, synthetic probes и независимое сохранение моделей.

## Что требуется для клинической модели

1. Определить с клиницистами цель приложения: воспроизведение заключения, повод для врачебной проверки или клинический исход. Не переносить expert FIGO в индивидуальный риск гипоксии.
2. Собрать независимые обезличенные размеченные бланки целевых приборов: срок на КТГ, длительность, определения/единицы, временная связь с исходом. Сырые сигналы не являются обязательным входом приложения.
3. При наличии пар signal/report или экспертных аннотаций проверить baseline, события и STV/LTV измерения; без эталона новый фильтр не объявлять правильным.
4. Заранее задать клинические потери, политику отказа и допустимую чувствительность, потом проверить их на внешней когорте. Для редких исходов считать независимых пациенток, не число повторных записей.

Исторические планы v1–v3 сохранены в history и before_revision. Результаты и ограничения текущего исследования — в 03/04 и README. Независимого нового test в этой ревизии нет.
''')
    write(ROOT/'research/notebooks/jnu_ctg/CLASSIFIER_PROTOCOL.md','''# Реализованный протокол reviewed_v2

Источник истины: `research/experiments/jnu_ctg/reviewed_v2/protocol.json`; он записан до расчёта новой версии. Код: `jnu_review_experiment.py`, `ctg_review_models.py`.

- Четыре strict-профиля без quality/duration, дублирующих единиц и missing indicators. Входы только полные, охват/отказы по всем 20769 исходным строкам.
- 5 внешних patient-component фолдов. Во внешнем fit выделяется 20% calibration; proper-fit отдельно делится tune/mode-holdout 80/20. Группы не пересекаются.
- На tune: 3-fold Grid Search CatBoost 16, spline 6, linear 3 конфигурации. Критерий group log loss; One-SE parsimony heuristic. Семья процедуры выбирается здесь же, до внешних меток.
- Single/ensemble с одинаковыми выбранными параметрами сравниваются на отдельном mode holdout; критерий +0.005 F1 при ухудшении log loss≤0.005. Затем actual aggregate fit на proper-fit и положительная sigmoid с intercept fit на отдельной calibration. Порог 0.5 фиксирован.
- Все вероятности внешнего фолда получены без его групп в fit/калибровке/выборе. Post-review постановка уже информирована прежними данными; outer CV не устраняет человеческую адаптацию к базе.
- HGB representation comparison: единые outer folds, common report5-complete cohort, одинаковые fit/calibration группы и сетка leaf7/15 × l2=1/10. Early stopping выключен. Median imputer reference-ветки fit только на proper training; reference-модели сохранены.
- 1000 component-bootstrap CI условны на сохранённых fits. Подгруппы/сценарии описательные, модели по ним не менялись. Synthetic patterns не имеют клинических gold labels.
- Final export использует proper-fit/calibration всех уже известных данных. Его отдельный independent test-score отсутствует. Native CatBoost без внешней sigmoid не воспроизводит calibrated bundle один в один.

Исторический classifiers_v1 protocol и notebooks сохранены в before_revision. Исправление переносимости joblib-драйвера описано в driver_pickle_repair.json: изменён только способ запуска модуля, вычислительные checkpoints переиспользованы без изменения алгоритмов/фолдов/параметров.
''')
    for name in ['EXPERIMENT_PLAN.md','EXPERIMENT_PLAN_V2.md']:
        p=ROOT/'research/notebooks/ctg_uci'/name
        original=p.read_text()
        if not original.startswith('> Исторический протокол'):
            write(p,'> Исторический протокол. Текущая исправленная процедура — reviewed_v3 в README и 02. Прежние file-level splits не доказывают независимость пациенток.\n\n'+original)
    write(ROOT/'research/notebooks/ctg_uci/PREGNANCY_APP_INPUT.md','''# Единый вход бланка КТГ

Публичный модуль: `src/features/ctg_report_contract.py`. Он использует один консервативный текстовый parser с сохранением исходных строк. Заключения, ПСП, Фишер и «8 из 8» не входят в X. OCR здесь не реализован.

`evaluate_report(report, bundle, profile)` проверяет обязательные поля, определения, контекст, подтверждение переноса и длительность до вызова модели. При блокере: `status=abstained`, `probability_expert_abnormal=None`, `clinical_risk=None`. Поддержка NaN внутри исторической модели не разрешает её применение к неполному бланку.

Сейчас ни один bridge больничного прибора не прошёл внешнюю валидацию: application mode всегда отказывает. Research mode требует явного операционного контракта и полных полей; результат относится только к экспертному классу JNU.

SisPorto import и старый `ctg_pregnancy_app` — исторические benchmark adapters. Их поля не подставляются вместо STV(ms)/амплитуды другого устройства. Первичный аудит источника JNU находится в JNU 01, не внутри UCI. Пример — в 08 и JNU 02.
''')
    # Current registries; historical binary files and frozen experiment receipts are untouched.
    from src.features.ctg_report_contract import PROFILES,CONTRACT
    export=ROOT/'models/exported/jnu_ctg/reviewed_v2';registry={'status':'research_only','contract':CONTRACT,'clinical_validation':False,'profiles':{}}
    for profile in PROFILES:
        card=json.loads((export/f'{profile}_model_card.json').read_text());family=card['selected_family'];filename=f'{profile}_{family}.joblib'
        registry['profiles'][profile]={'family':family,'bundle':filename,'sha256':sha(export/filename),'columns':PROFILES[profile],'independent_final_model_test':False}
    (export/'research_models.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
    write(export/'README.md','''# JNU reviewed_v2 · research models

12 strict report bundles (4 profiles ×3 families), separate fitted scalers/spline bases, native CatBoost files and 35 HGB reference fold-models. `research_models.json` identifies the family selected by the declared fit-side rule for each profile.

Each bundle contains the actual selected single/ensemble and its sigmoid calibrator. Native `.cbm` files contain trees only. Missing required features raise an error before prediction; there is no imputer or missing indicator in the report models. The `reference_models/` branch has its own fit-only imputer and requires signal/published features; it is not an app input.

Use `src.features.ctg_report_contract.evaluate_report` as the public report entry. Application mode abstains: no external device bridge is validated. Direct predict_proba is research-only and predicts the JNU expert label, not a medical risk. All final models were fit/calibrated on already known data and have no independent final-model test score.

Current executed notebooks: research/notebooks/jnu_ctg/03 and 04. Detailed protocol, coverage, intervals and subgroup errors are in reviewed_v2/ under research/experiments. Do not load untrusted joblib files.
''')
    export_u=ROOT/'models/exported/ctg_uci/reviewed_v3';card=json.loads((export_u/'model_card.json').read_text())
    write(export_u/'README.md',f'''# UCI reviewed_v3 · benchmark only

Selected family on all-data inner CV: **{card['selected_family']}**. Six calibrated bundles, native CatBoost heads and separate logistic preprocessor are saved. Final models fit all 2126 segments; they have no independent final-model test score. The evaluated object is the repeated nested selection procedure.

Input: exact native SisPorto UCI columns; missing values rejected. This is not a model for arbitrary hospital reports. The prefix is a dependence proxy, not verified patient identity. The current application-facing report entry is `src.features.ctg_report_contract` and has no validated medical inference bridge.
''')
    for folder in [ROOT/'models/exported/jnu_ctg/classifiers_v1',ROOT/'models/exported/ctg_uci/audited_v2']:
        (folder/'DEPRECATED_FOR_APPLICATION.json').write_text(json.dumps({'status':'historical_research_only','reason':'Expert-review corrections supersede this application path; use current strict contract.','current_entry':'src.features.ctg_report_contract','clinical_validation':False},indent=2))
    root=ROOT/'README.md';text=root.read_text();start=text.index('Отдельная серия по КТГ:');end=text.index('## CDC HDP-proxy',start)
    replacement=f'''Серии КТГ обновлены после критического разбора 23 замечаний эксперта:

- [UCI — 8 ноутбуков](research/notebooks/ctg_uci/README.md): benchmark с 177 группами-прокси, 3 повтора 5×3 nested CV, proper-score selection/calibration. Средний group Macro-F1 исправленной процедуры {up.group_macro_f1.mean():.3f}. Кластерная ветка сохранена и закрыта для клинических выводов.
- [JNU — 4 ноутбука](research/notebooks/jnu_ctg/README.md): strict report-модели без learned missing fallback, дублирующих единиц и quality/duration в X; одинаковые внешние группы для сигнальных ориентиров. Цель — expert normal/abnormal, не риск асфиксии.
- [Ответ по каждому пункту](research/notebooks/ctg_uci/07_deep_audit_and_recommendations.ipynb), новые модели — `models/exported/ctg_uci/reviewed_v3/` и `models/exported/jnu_ctg/reviewed_v2/`. Прежние notebooks/results сохранены.
- Единый вход бланка: `src.features.ctg_report_contract`. При missing/unconfirmed fields прогноз не вычисляется. Внешне валидированных bridges приборов пока нет, поэтому application mode возвращает отказ и clinical_risk=None.

'''
    text=text[:start]+replacement+text[end:];text=text.replace('аудит новой JNU-CTG — в ноутбуке 08','аудит JNU-CTG — в отдельной серии jnu_ctg, ноутбук 01');write(root,text)
    registry_path=ROOT/'data/registry/open_datasets.json';data=json.loads(registry_path.read_text())
    entries=data['datasets']
    for entry in entries:
        if entry['id']=='jnu_ctg_1_0_0':entry['clinical_limit']='Single centre antepartum; binary expert morphology is not neonatal prognosis. Revised strict report models have exploratory patient-grouped CV only; no validated external device bridge or clinical application.'
        if entry['id']=='uci_cardiotocography_193':entry['clinical_limit']='Expert-class benchmark only; 352 files share 177 normalized prefix proxies, not verified patient IDs. Revised repeated nested prefix-grouped CV; arbitrary hospital report transfer unsupported.'
    write(registry_path,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Current documentation and model registries written')


if __name__=='__main__':package()
