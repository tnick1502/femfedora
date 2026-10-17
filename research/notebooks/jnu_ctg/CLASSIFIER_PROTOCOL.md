# Реализованный протокол reviewed_v2

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
