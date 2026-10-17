# JNU-CTG · подготовка v1

Источник: https://zenodo.org/records/21800730 (DOI 10.5281/zenodo.21800730).

Подготовка всех 20769 записей / 12606 patient_id. Одна запись — 7200 точек FHR/UC при 4 Hz. Оригинальный архив — `data/external/jnu_ctg/1.0.0/JNU-CTG.zip`; byte-level оригинал сохраняется там, float32 массив — производное представление.

Выполненный ноутбук: `research/notebooks/jnu_ctg/01_prepare_data.ipynb`. Методика и все ограничения — в его README и `ANALYSIS_PLAN.md`. Разбиение: 12464/4165/4140 записей train/validation/test по независимым компонентам patient_id и точных дублей FHR. Цели: бинарный figo_label и отдельная nullable neonatal_asphyxia; отсутствующие исходы не заполнены нулями.

27 собственных сигналовых дескрипторов не равны медицинским STV/LTV. Качество и маски сохранены отдельно. X_raw и X_robust каждой части имеют те же строки, что соответствующий index_targets; использовать record_id для явного соединения. Eligibility-флаги в index_targets необходимы для контроля пригодности; они не являются предикторами и не гарантируют клиническое качество.

Сканы сигналов используют memory map. Packed masks: numpy.unpackbits(..., axis=1, bitorder='little')[:, :7200]. Для нормализации/импутации использовать train-only артефакты из `models/exported/jnu_ctg/preparation_v1/`; для CV — необученный шаблон с повторным fit внутри фолда.

Все большие файлы исключены из обычного Git. Контрольные суммы: `research/experiments/jnu_ctg/preparation_v1/manifest.json`. Модели-классификаторы не обучались; медицинская применимость не подтверждена.
