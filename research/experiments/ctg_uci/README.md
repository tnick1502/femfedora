# Исполненный эксперимент UCI CTG · 17 октября 2026

Протокол и интерпретация: `research/notebooks/ctg_uci/EXPERIMENT_PLAN.md` и README серии. Воспроизводимые расчёты — ноутбуки 02–05; данные и код связаны fingerprint в `nested_selection.json`, `locked_decision.json` и `validation_receipt.json`.

| Выход | Значение |
|---|---|
| `outer_folds.csv`, `inner_folds.csv` | Исходные train-строки и общие фолды для всех моделей. Внешний holdout не входит во внутренние фолды своего внешнего fit. |
| `nested_grid.json`, `nested_parameters.json` | Все 630 комбинаций и найденные настройки каждого семейства/внешнего фолда. `fold_f1` — три внутренние оценки; не независимые клинические испытания. |
| `nested_fold_metrics.csv`, `nested_pooled_metrics.csv`, `nested_predictions.parquet` | Оценка всей процедуры поиска/калибровки/порога на внешних OOF-строках train. |
| `final_grid.json` | 126 комбинаций, повторно оценённых только на всём исходном train с внутренними фолдами. |
| `validation_calibration_folds.csv`, `validation_crossfit_metrics.csv`, `validation_predictions.parquet` | Общая групповая проверка калибраторов, порогов и single/ensemble на validation. |
| `locked_decision.json` | Выбор семейства, архитектуры, postprocessing и порогов, сохранённый перед генерацией test-предсказаний. |
| `test_metrics.csv`, `test_predictions.parquet` | Финальные результаты заранее зафиксированных процедур. Столбец `champion` означает выбор до test, а не максимум в test-таблице. |
| `champion_test_intervals.csv` | 95% percentile bootstrap по исходным файлам для фиксированного выбранного чемпиона, 1000 повторов. Не включает полную изменчивость переобучения и отбора моделей. |
| `test_precision_recall_areas.csv` | Average Precision и трапецеидальная PR-AUC pathologic: разные определения площади. |
| `sensitivity_operating_point.csv` | Отдельный заранее определённый validation-порог для recall ≥0.90; его test-результат не меняет чемпиона. |
| `champion_errors.csv`, `champion_feature_importance.csv` | Ошибки test и внутренняя важность признаков; описательный аудит без нового обучения по этим ошибкам. |
| `cluster_grid.csv`, `cluster_locked_config.json` | Полные outcome-free сетки и выбор по validation silhouette, до анализа test-разметки. |
| `cluster_assignments.parquet`, `cluster_association.csv`, `cluster_test_summary.csv` | Принадлежности, ARI/NMI и состав кластеров по экспертным классам после фиксации. Cluster ID произволен и не означает степень риска. |
| `cluster_projection.parquet`, `fcm_convergence.json` | Двумерная проекция test и сходимость выбранного FCM. Проекция показывает только первые две из десяти PCA-компонент. |
| `validation_receipt.json` | Независимая сверка границ строк/файлов, полноты сеток, test-метрик и исполненности ноутбуков. |

В предсказаниях `NSP` и `pred` имеют коды 1/2/3, а `p0/p1/p2` означают соответственно P(normal), P(suspect), P(pathologic). Все метрики вычислены на естественных долях классов без тестовых весов. FCM-membership — близость к геометрическому центру, не вероятность патологии. `membership_weighted_path_fraction` — описательная взвешенная доля разметки на test; это не дополнительный обученный прогноз.

Интервалы для `suspect` и объединённого `abnormal=NSP>1` добавлены как исследовательское уточнение после просмотра таблицы состава. Обучение, число кластеров и параметры при этом не менялись. Это не отдельные подтверждающие проверки после коррекции множественных сравнений.

Полные модели лежат локально в `models/exported/ctg_uci/` (игнорируются Git). Для применения следует загружать `champion.joblib`, а не использовать отдельную `.cbm` без её калибровки и правил. Экспертные классы КТГ не заменяют реальные исходы или внешнюю клиническую валидацию.
