# Протокол после глубокого аудита · audited v2

## Вопрос и единица наблюдения

Предсказать экспертный NSP (normal/suspect/pathologic) UCI Cardiotocography. Это воспроизведение интерпретации КТГ; наблюдаемых перинатальных исходов в задаче нет. 2126 сегментов принадлежат 352 исходным файлам. Идентификатора пациентки нет; файл — доступная, но неполная замена пациентской группировки.

Основной estimand сохранён: macro-F1 по сегментам естественного распределения. Дополнительный estimand: равновероятный файл, затем равновероятный сегмент внутри него; вес сегмента 1/число сегментов файла. File-balanced и segment-показатели отвечают разным вопросам и не означают оценку на независимых пациентках.

## Неподвижные данные

- Сохранить train1360 / validation341 / test425 и отсутствие общих файлов/feature duplicates. Патологических исходных файлов соответственно33/7/10. Не выравнивать test искусственно.
- Все21признака независимо сверены с Raw Data XLS. Пропусков нет; нули сохраняются. Новые пропуски отклоняются: импутация/маски на этом источнике не проверены.
- DS/Width оставить в primary, исключить из основной модельной схемы. Tendency категориальная. Метки и метаданные не входят в X.
- CatBoost без scaler. Логистика клонирует и fit всю подготовку заново в каждом fit. Скейлеры01 не применяются к внутренним фолдам уже обученными на всём train.

## 02 · Полная процедура внутри внешней проверки

1. Общие5внешних фолдов на train. Балансировать для каждого NSP и числа сегментов, и присутствие класса в исходных файлах, плюс размеры частей. Детерминированный greedy+moves/swaps не использует X или метрики. Крупные файлы неделимы; одинаковые доли строк не гарантируются. Сохранять поддержку каждым классом.
2. В каждом fit — общие3внутренних фолда тем же способом. Семейства A MultiClass, B SqrtBalanced, B Balanced, C OvR, D hierarchy и multinomial logistic.
3. CatBoost: depth3/5/7 × iterations300/600 × learning_rate.03/.1 × L2 3/10 =24. Logistic: C.1/1/10 × None/Balanced =6. Итого630сочетаний family/outer/params, по3внутренних оценки. Выбор по среднему внутреннему segment macro-F1, tie по log loss/порядку. Это двухэтапный поиск в заданной сетке, не глобальный оптимум.
4. На внутреннем OOF лучшей конфигурации выбрать calibration raw/temperature/sigmoid по grouped cross-fit log loss. Для C калибровка проверяется, но не навязывается при ухудшении. Log loss — proper probability score, не чистая изолированная мера калибровки: он зависит также от разрешающей способности и неопределённости данных.
5. Отдельно сравнить argmax с патологическим порогом. В calibration cross-fit cutoff fit только на fit-стороне, оценка на holdout. Принять threshold policy при выигрыше macro-F1≥.005. Финальный cutoff обучающей стороны fit по её cross-fitted probabilities; calibrator fit на всей этой стороне.
6. Внешний holdout оценивает уже всю процедуру. Он не используется для scaler, early stopping, calibration, cutoff или выбора внутренних параметров. Внутренняя адаптация и её оптимизм оцениваются снаружи.
7. Primary family — лучший средний внешний macro-F1, tie log loss. Shortlist: максимум2семейства в пределах.01 от лучшего. Пять фолдов с общими train не независимы, SD не CI; выбор семейства по внешним данным тоже вносит оптимизм.

## 03 · Финальный fit и soft voting

1. Повторить полный3foldgrid на исходном train для всех6семейств; fit single. Никакого fit деревьев/scaler на validation/test.
2. Для shortlist обучить5моделей на4/5train. Сначала получить три class probabilities каждого участника, затем усреднить. Для hierarchy P(path)=ab, E[ab] отличается от E[a]E[b]. Среднее имеет отдельный calibrator.
3. На validation выбрать calibration по log loss, cutoff по macro-F1 с правилом≥.005. Фолды балансируют положительные исходные файлы. Семь патологических файлов не делают этот выбор точным автоматически.
4. Ensemble принимается при validation cross-fit macro-F1 gain≥.01 относительно single. Иначе single. Правило и shortlist не меняются по test.
5. Fit calibrator на всей validation, деревья не переобучать. Сохранить lock со всеми решениями/fingerprints/версиями до test predict.
6. Исследовательский повторный test; benchmark уже известен до аудита и не становится слепым. Основной bundle и shortlist сохранить в `models/exported/ctg_uci/audited_v2/`.

## 05 · Метрики и независимые проверки

Macro-F1; recall/precision/F1 и AP патологии; balanced accuracy; log loss; multiclass Brier; ROC; confusion matrix; file-balanced показатели и ошибки по исходным патологическим файлам. AP — Average Precision, не трапецеидальная площадь под PR. Добавить always-normal с естественной prevalence.

Shortlist:1500pairedbootstrap по test-файлам, одинаковые sampled files для каждой модели; все сегменты выбранного файла сохраняются вместе. Интервалы условны на фиксированных моделях, без полного разброса обучения/отбора. Ноль в CI разницы не подтверждает превосходство. Известный test ограничивает даже различия с CI вне нуля.

Все test-строки пересчитать отдельно из сохранённых probabilities. Проверить загрузку каждого bundle, predict/predict_proba, классы/схему, суммы вероятностей, grouped boundaries и native CatBoost roundtrip. Сохранить SHA-256 артефактов и итоговый receipt.

## 04/06 · Кластеризация остаётся исследовательской

Существующие результаты FCM/KMeans/GMM, UMAP/AE/PCA сохранены побайтно. NSP/CLASS не участвуют в fit/выборе, последующая связь с NSP не выбирает классификатор. AE+GMM:20/45патологических сегментов, но только2патологических файла; низкая seed/file-subsample устойчивость. Базовый small_cluster не заменяет `path_source_files`/`limited_path_support` из дополнительной таблицы06.

UMAP transform оказался чувствителен к batch: у проверенного UMAP+GMM менялось1/20назначений тех же строк. Его не продвигают в эксплуатационный классификатор. Для будущих cluster features нужен заранее заданный cross-fitted ablation, стабильный transform-контракт и независимая проверка.

## 07 · Аудит

Свести доказанные ошибки/исправления, независимые проверки источника/scaler, новые результаты и пределы: нет patient ID, известный test, экспертный surrogate target, неопределённость лучшей модели. Старые02/03/05 и план сохранены в `history/v1/`. Заявлять готовность только в проверенной исследовательской области.

Источники: [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography), [nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), [soft voting](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier), [UMAP clustering caveats](https://umap-learn.readthedocs.io/en/latest/clustering.html).
