# JNU-CTG · reviewed v2 · поля больничного бланка

Источник: [JNU-CTG 1.0.0](https://zenodo.org/records/21800730), CC BY 4.0, 20769 записей /12606 пациенток. Цель моделей — экспертная binary FIGO normal/abnormal. Асфиксия и медицинский риск не являются целями этих классификаторов.

## Текущие ноутбуки

- [01 · подготовка](01_prepare_data.ipynb): сохранён глубокий аудит источника, маски/дубли/метаданные; добавлены UC signatures, published U_* и согласие разных записей пациентки.
- [02 · измерения и контракт](02_report_features.ipynb): прежние вычисления сохранены; проверены определения амплитуды, единицы, редкая низкая STV, единый отказ до модели.
- [03 · переобучение](03_report_classifiers.ipynb): 12 strict report-моделей, четыре профиля, nested CV, отдельные mode/calibration группы, сигнальные ориентиры.
- [04 · проверка](04_classifier_assessment.ipynb): интервалы, отказы, низкая STV, качество, UC signatures, synthetic probes, связь с исходом и сценарный PPV.

## Результат исправленной процедуры

| Профиль | Macro-F1 принятых | Recall abnormal | Охват JNU |
|---|---:|---:|---:|
| core3 | 0.673 | 63.8% | 99.87% |
| report4s5 | 0.720 | 70.8% | 99.79% |
| report5 | 0.722 | 70.7% | 99.79% |
| report7 | 0.732 | 72.1% | 99.79% |

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
