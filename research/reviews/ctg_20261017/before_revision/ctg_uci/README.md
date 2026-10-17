# КТГ · UCI Cardiotocography · audited v2

Текущая серия после глубокого аудита. Источник: [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography), DOI 10.24432/C51S4N, CC BY 4.0. Цель — экспертный NSP, не исход ребёнка.

## Текущие ноутбуки

1. [01_prepare_data.ipynb](01_prepare_data.ipynb) — XLS, аудит, распределения, первичная таблица и неизменное разбиение; train-only скейлеры сохраняются отдельно и в preprocessor.
2. [02_nested_grid_search.ipynb](02_nested_grid_search.ipynb) — повторная полная сетка всех6семейств,5×3nestedCV; баланс и сегментов, и независимых файлов классов; postprocessing выбирается внутри fit-стороны.
3. [03_calibration_ensemble_and_test.ipynb](03_calibration_ensemble_and_test.ipynb) — финальный grid, proper-score calibration, порог, корректное soft voting, фиксация и исследовательский повторный test; экспорт primary/shortlist.
4. [04_clustering_without_outcomes.ipynb](04_clustering_without_outcomes.ipynb) — сохранённый исходный FCM/KMeans/GMM без меток в fit/выборе.
5. [05_results_and_error_analysis.ipynb](05_results_and_error_analysis.ipynb) — текущие модели, ошибки, PR/ROC, reliability, file-balanced метрики, парные интервалы и независимый пересчёт.
6. [06_clustering_embeddings_grid_search.ipynb](06_clustering_embeddings_grid_search.ipynb) — сохранённые1650сочетаний UMAP/AE/PCA и кластеров, геометрия, seed/file-subsample устойчивость.
7. [07_deep_audit_and_recommendations.ipynb](07_deep_audit_and_recommendations.ipynb) — обнаруженные ошибки, доказательства исправлений и пределы выводов.
8. [08_hospital_reports_feature_engineering.ipynb](08_hospital_reports_feature_engineering.ipynb) — определения реальных больничных показателей, ограниченные входы, feature engineering и новый 5×3 grid; адаптер для приложения по ведению беременности и первичный аудит дородовой JNU-CTG.

Для приложения с неизвестным аппаратом: [формат ввода и ограничения](PREGNANCY_APP_INPUT.md). Модели новой ветки сохранены отдельно в `models/exported/ctg_uci/report_features_v1/`; они не заменяют audited v2. На трёх условно общих измерениях test macro-F1 .546, recall патологии 12/35; на девяти нативных .844 и 31/35. Нативные вариабельности нельзя заменять похожими строками другого прибора; клинический перенос не подтверждён.

Актуальный подробный [протокол v2](EXPERIMENT_PLAN_V2.md). В исходных данных1360/341/425train/validation/test; источник и feature-duplicate groups между ними не пересекаются. Пациентских ID нет; патологических файлов33/7/10. Test уже известен исследованию, новая слепая клиническая когорта отсутствует.

## Исправления

- Hierarchy-ансамбль усредняет готовые три вероятности каждого участника, а не условные головы до перемножения.
- Внутренние/внешние фолды учитывают наличие каждого класса в независимых исходных файлах. Ровные числа сегментов недостаточны.
- Калибровка выбирается по cross-fitted log loss; классификационный cutoff отдельно по macro-F1. Выбор всей политики входит в внешнюю проверку.
- Основной segment macro-F1 сохранён, добавлены показатели с равным суммарным весом исходного файла и paired file bootstrap.
- Новый fingerprint включает источники guard/методов/фолдов, данные, подготовку и версии; экспорт проверяется по SHA-256 и независимому inference roundtrip.

## Запуск и артефакты

После `poetry install`, из корня: `poetry run jupyter lab research/notebooks/ctg_uci`.

01 создаёт данные. Затем02→03→05→07. 04/06 — сохранённая исследовательская ветка; её исходные результаты не перезаписываются аудитом. Ноутбуки можно выполнять через `poetry run python tools/execute_notebooks_inprocess.py <путь>`.

Код текущей процедуры: `src/training/ctg_audit.py`, `ctg_audit_folds.py`; генератор `tools/build_ctg_audited_notebooks.py`. Стандартный `build_ctg_training_notebooks.py` теперь создаёт текущую версию; `--legacy` направляет старый генератор в историю.

Текущие таблицы и проверочные receipts: `research/experiments/ctg_uci/audited_v2/`. Основной bundle: **`models/exported/ctg_uci/audited_v2/champion.joblib`**. Shortlist и параметры — в его `model_card.json` и `best_models.json`. Bundle содержит проверку входа, нужную подготовку, модели, усреднение, калибровку и порог; отдельные `.cbm` содержат только деревья. Вход — DataFrame ровно с21исходным признаком. Новые пропуски вызывают явный отказ; они не маскируются.

Исторические02/03/05и описания — в [history/v1](history/v1/). Старый `ctg_experiments.py`, корневые экспериментальные CSV и старые модели сохраняются для воспроизведения исходного результата; текущий экспорт находится в `audited_v2/`.

## Кластеризация и ограничения

AE+GMM имел20/45патологических сегментов, но только2патологических исходных файла и низкую устойчивость к seed. UMAP+GMM оказался чувствителен к размеру batch при transform. Высокий описательный ARI не делает эти варианты подтверждёнными риск-группами; они не выбирают supervised champion и не добавляются задним числом в X.

Грид — лучший результат среди проверенных конфигураций, не глобальная гарантия. Источник мал и историчен, NSP суррогатный экспертный класс, исходный файл не доказывает независимость пациенток. Для переноса требуется независимая когорта с ясной клинической целью.

## Выполненные результаты audited v2

## Основные варианты по development CV

| Вариант | Mean outer macro-F1 | Mean outer recall P | Test macro-F1 | Test recall P | Test AP P |
|---|---:|---:|---:|---:|---:|
| C_ovr/single | 0.8744 | 0.8623 | 0.8538 | 0.9143 | 0.9103 |
| D_hierarchy/single | 0.8724 | 0.9001 | 0.8454 | 0.9143 | 0.9403 |


Полная таблица всех8кандидатов и ограничения: research/experiments/ctg_uci/audited_v2/README.md. На известном test лучший point macro-F1 у B_balanced (.8838), но primary по development CV не переименовывался.
