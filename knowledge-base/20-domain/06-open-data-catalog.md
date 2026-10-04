---
status: active
effective_date: 2026-09-23
tags:
  - datasets
  - open-data
  - catalog
---

# Каталог открытых данных

## Загруженные наборы

| Источник | Роль | Доступ/лицензия | Локальное расположение |
|---|---|---|---|
| CDC/NCHS Natality 2023–2024 | PTB, GDM и HDP-proxy baseline; U.S. resident births; 2024 как temporal test | public use; условия CDC/NCHS | `data/external/cdc_natality/` |
| CTU-UHB CTG 1.0.0 | outcome-linked КТГ benchmark | ODC Attribution 1.0 | `data/external/ctu_uhb_ctg/1.0.0/` |
| UCI Cardiotocography | учебная экспертная классификация | CC BY 4.0 | `data/external/uci_cardiotocography/1.0.0/` |
| TPEHG 1.0.1 | EHG/PTB R&D | ODC Attribution 1.0 | `data/external/tpehg/1.0.1/` |

Машиночитаемый реестр, checksums и результаты intake находятся в `data/registry/` и `research/notebooks/`.

## Контролируемые источники — не скачаны

| Источник | Ценность | Причина отсутствия |
|---|---|---|
| nuMoM2b | продольные PE/PTB/GDM модели | требуется заявка и Data Use Agreement |
| NICHD Fetal Growth Studies | серийные УЗИ и траектории роста | контролируемый доступ |
| Consortium on Safe Labor | интранатальные и родовые процессы | контролируемый доступ; вторая очередь |
| локальная когорта | true PE/FGR/ИЦН и переносимость | нужен протокол, правовое основание и деидентификация |

## Стандарты, не train set

INTERGROWTH-21st используется для детерминированных стандартов роста. Его графики и формулы не являются patient-level обучающей выборкой.

## Правила использования

1. Наборы хранятся отдельно и не объединяются на уровне строк.
2. CDC — запись о родах, а не продольная история.
3. CDC prepared baseline ограничен `RESTATUS ∈ {1,2,3}`; нерезиденты США (`4`) остаются в raw-архиве, но исключаются из аналитической популяции.
4. UCI CTG — признаки и экспертные классы, а не raw signal/outcome.
5. CTU-UHB — интранатальный контекст, не антенатальный скрининг.
6. TPEHG — малый исследовательский набор с 38 preterm events.
7. Любое prepared-представление сохраняет проверяемую связь с записью и версией источника; конкретные поля определены его data contract и registry.
8. Публичная доступность файла не доказывает законность обработки. Наборы с именами, неясным происхождением или неопределённым target не попадают в обучение до правового основания, минимизации, деидентификации и проверки на синтетику/утечку.

## Официальные страницы

- [CDC/NCHS Vital Statistics Online](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm)
- [CTU-UHB CTG 1.0.0](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
- [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography)
- [TPEHG 1.0.1](https://physionet.org/content/tpehgdb/1.0.1/)

## Связанные заметки

- [[00-risk-model-catalog]]
- [[05-data-and-prediction-contracts]]
- [[../10-project/01-current-ml-scope]]
