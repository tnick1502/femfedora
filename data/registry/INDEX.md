# Реестр открытых данных

## Файлы

- `open_datasets.json` — официальные страницы, URL загрузки, версии, лицензии и допустимые роли.
- `cdc_natality_baseline_contract.json` — features, labels, split и запрещённые claims для CDC baseline.
- `natality2023.dct`, `natality2024.dct` — схемы fixed-width NCHS-файлов, подготовленные NBER.
- `checksums.sha256` — SHA-256 фактически загруженных архивов; создаётся после проверки.
- `longitudinal_datasets.json` — новые продольные источники CPP, Marić, DREAM и обе когорты NLSY79, включая состояние загрузки.
- `longitudinal_checksums.sha256` — SHA-256 загруженных архивов новых источников.

## Политика

Тяжёлые данные хранятся локально вне Git. Реестр и код подготовки версионируются. Новый источник нельзя использовать в обучении, пока не определены grain, версия, лицензия, target, допустимые predictors, missingness и ограничения клинического утверждения.

CDC/NCHS, CTU-UHB, UCI CTG и TPEHG остаются независимыми наборами. Совпадение названий полей не является основанием для построчного объединения.

Для каждого источника рядом с данными в `data/external/<набор>/DATASET.md` лежит карточка состава, полей и применимости. DREAM/SDY2187 сейчас представлен только публичным манифестом: сами архивы требуют авторизованной загрузки через ImmPort. Для CPP `visitfle.csv` остаётся позиционно размеченным файлом; клинические траектории из него нельзя строить до расшифровки полей по исходным формам.

## Локально подготовленные представления

| Путь | Grain | Назначение |
|---|---|---|
| `data/processed/cdc_natality/natality_ml_2023_2024.sqlite` | одна запись о рождении у резидентки США (`RESTATUS` 1/2/3) | лёгкие PTB/GDM/HDP-proxy baseline; 2023 development, 2024 temporal test |
| `data/processed/preeclampsia_cdc/eligible_all_2023_2024.parquet` | одна одноплодная запись о живорождении с известной HDP-proxy меткой | общая когорта 2023+2024 с 5 фолдами и масками входа; отдельный анализ обучает на 2023 и проверяет 2024 |
| `data/processed/ctu_uhb_ctg/record_index.csv` | одна КТГ-запись | индекс WFDB-сигналов, исходов и материнских факторов |
| `data/processed/uci_cardiotocography/ctg_features.csv` | один размеченный feature-вектор | учебный CTG baseline с `duplicate_group_id` |
| `data/processed/tpehg/record_index.csv` | одна EHG-запись | индекс WFDB-сигналов и PTB label |
| `data/processed/tpehg/tpehgdb_features__*.csv` | канал одной EHG-записи | опубликованные авторами signal-features для лёгкого baseline |

Prepared-файлы намеренно исключены из Git, но уже находятся в локальном проекте. Для воспроизведения нужно запустить downloader, preparation и profiling из `src/data/`.
