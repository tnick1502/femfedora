# Индекс CTU-UHB CTG

`record_index.csv` — **552 строки, по одной на исходную запись**. `record_id`, `header_path`, `signal_path` указывают на PhysioNet `.hea` и `.dat` в `data/external/ctu_uhb_ctg/`; `signal_count`, `sample_frequency_hz`, `sample_count` описывают сигнал. `comments` сохраняет полный текст заголовка. Разобранные поля: `cord_ph`, `base_deficit_ecf`, `apgar_1min`, `apgar_5min`, `gestation_weeks`, `birth_weight_g`, `maternal_age`, `gravidity`, `parity`, `diabetes`, `hypertension`, `preeclampsia`, `induced`, `delivery_type`.

Индекс нужен для поиска сигналов и небольших экспериментов по интранатальной КТГ с исходами родов. Часть значений из комментариев источника предварительна; пустые или неизвестные поля нужно проверять перед анализом. Сведения об исследовании, лицензии и ограничениях — [`data/external/ctu_uhb_ctg/DATASET.md`](../../external/ctu_uhb_ctg/DATASET.md).
