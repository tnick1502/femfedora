# Подготовленные признаки UCI Cardiotocography

`ctg_features.csv` — **2 126 строк, 25 колонок**. Каждая строка — готовый вектор признаков КТГ; сырых сигналов здесь нет. `LB`, `AC`, `FM`, `UC`, `DL`, `DS`, `DP` описывают базовый ритм, акселерации, движения, сокращения и децелерации; `ASTV`, `MSTV`, `ALTV`, `MLTV` — вариабельность; `Width`, `Min`, `Max`, `Nmax`, `Nzeros`, `Mode`, `Mean`, `Median`, `Variance`, `Tendency` — распределение ЧСС. `CLASS` и `NSP` — экспертные классы. `source_row` позволяет отследить исходный лист, `duplicate_group_id` объединяет совпадающие векторы.

Подходит для учебного baseline классификации КТГ, с разделением по `duplicate_group_id` (в исходном профиле 11 полных дублей). `NSP` не является прямым клиническим исходом или маркером преждевременных родов. Источник, лицензия и ограничения — [`data/external/uci_cardiotocography/DATASET.md`](../../external/uci_cardiotocography/DATASET.md).
