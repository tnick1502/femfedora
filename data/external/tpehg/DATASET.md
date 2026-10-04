# Term-Preterm EHG Database (TPEHG), 1.0.1

## Что изучалось и что лежит в папке

300 записей электрогистерографии матки (EHG) из PhysioNet; 38 беременностей завершились преждевременными родами по принятой в источнике метке. В `1.0.1/files/tpehgdb/` — пары сигналов `.dat` и заголовков `.hea`, а рядом — опубликованные авторами файлы извлечённых признаков `.fvl` и руководство `tpehgdb.pdf`. Производные `data/processed/tpehg/record_index.csv` и `tpehgdb_features__*.csv` создают индекс и табличные signal-features.

## Поля

| Где | Поля / значение |
|---|---|
| `.hea` / `.dat` | Три исходных канала EHG и их фильтрованные версии; заголовок содержит частоту дискретизации и число отсчётов. |
| Комментарии `.hea` | `Gestation` — срок родов; `Rectime` — срок при записи; `Age`, `Parity`, `Abortions`, `Weight`, `Hypertension`, `Diabetes`, `Placental_position`, кровотечения по триместрам, `Funneling`, `Smoker`. |
| `record_index.csv` | `record_id`, пути к сигналам, `gestation_weeks`, `recording_gestational_age_weeks`, материнские поля, `target_preterm`, `recorded_before_26_weeks`. |
| `tpehgdb_features__*.csv` | `record`, `chann`, `gestation`, `rec_time`, `group`, `rms`, `fmed`, `fpeak`, `samp_en`, `premature`, `early`; отдельный файл на полосу фильтра. |

Важно различать срок **записи** (`Rectime`/`rec_time`) и срок **родов** (`Gestation`/`gestation`). При раннем прогнозе нельзя передавать срок родов или построенные из него `premature`, `early`, `group` в признаки модели.

## Куда подходит

Для исследовательского signal-processing baseline преждевременных родов и проверки эффекта недели записи. Набор мал, положительных исходов 38; оценки нестабильны и требуют доверительных интервалов, разделения по записи и внешней проверки. Не переносится непосредственно на обычную клиническую антенатальную популяцию.

Источник: [PhysioNet TPEHG](https://physionet.org/content/tpehgdb/1.0.1/), DOI [10.13026/C2FW2V](https://doi.org/10.13026/C2FW2V). Условия: Open Data Commons Attribution License 1.0. Версия и хэш каталогов указаны в реестре проекта.
