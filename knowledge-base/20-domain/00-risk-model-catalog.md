---
status: active
effective_date: 2026-09-23
tags:
  - ml
  - risk-models
  - catalog
---

# Каталог ML-задач

| Задача | Статус | Первая честная цель | Данные сейчас | Следующий уровень |
|---|---|---|---|---|
| Преэклампсия/HDP | P0 | HDP-proxy в записи родов; не ранняя PE | CDC/NCHS | landmark PE на nuMoM2b/локальной когорте |
| Преждевременные роды | P0 | PTB `<37`, PTB `<32` | CDC/NCHS | spontaneous PTB survival с competing risks |
| КТГ | P0 signal | outcome-linked intrapartum benchmark | CTU-UHB | CNN/TCN, затем SSM challenger |
| EHG | R&D | term/preterm classification | TPEHG | внешняя когорта и устойчивость |
| ГСД | P1 baseline | GDM flag в записи родов | CDC/NCHS | ранний прогноз до ОГТТ в продольной когорте |
| ЗРП/FGR | data collection | открытой допустимой цели нет; low birth weight не является FGR | CDC даёт только исследовательский low-BW label | trajectory FGR по серийным УЗИ/допплеру |
| ИЦН | data collection | нет допустимой открытой ML-цели | нет | локальная серия цервикометрии и вмешательств |

## Правило именования моделей

Имя модели должно отражать фактический label и момент данных, например:

- `cdc2023_2024_ptb37_logreg_v0`;
- `cdc2023_2024_hdp_proxy_logreg_v0`;
- `ctu_uhb_ph_outcome_tcn_v0`;
- `tpehg_term_preterm_features_logreg_v0`.

Запрещены имена, которые обещают больше, чем label: `preeclampsia_detector`, если target — объединённый флаг HDP; `fetal_distress`, если target — экспертный CTG-класс; `fgr_model`, если target — SGA по массе при рождении.

## Карточки направлений

- [[01-preeclampsia-hdp]]
- [[02-preterm-birth]]
- [[03-ctg-ehg]]
- [[04-deferred-candidates]]
- [[05-data-and-prediction-contracts]]
- [[06-open-data-catalog]]
