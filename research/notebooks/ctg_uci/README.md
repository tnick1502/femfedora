# UCI Cardiotocography · reviewed v3

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

Средние метрики исправленной selected_procedure по трём повторам: group-balanced Macro-F1 **0.855**, recall suspect **70.5%**, recall pathologic **82.6%**. Это оценка процедуры выбора; не независимый score финального файла модели. Segment-weighted метрики показаны рядом.

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
