---
status: canonical
effective_date: 2026-09-23
tags:
  - data-contract
  - leakage
  - validation
---

# Контракты данных и прогнозов

## Минимальный ClinicalEvent

```json
{
  "person_id": "...",
  "pregnancy_id": "...",
  "fetus_id": "...",
  "encounter_id": "...",
  "concept_code": "...",
  "value": "...",
  "unit": "...",
  "observed_at": "...",
  "available_at": "...",
  "ga_days": 142,
  "source": "manual|lis|ehr|document|device",
  "quality_flag": "verified",
  "provenance": {"document_id": "...", "span": "..."}
}
```

## Task specification до обучения

Каждый эксперимент фиксирует:

| Поле | Смысл |
|---|---|
| Population | кто включён |
| Index time | когда строится прогноз |
| Prediction horizon | до какого момента прогнозируется событие |
| Outcome | точное определение, источник и подтверждение |
| Predictors | только факты, доступные к index time |
| Exclusions | заранее заданные исключения |
| Action | какое решение теоретически может поддержать сигнал |
| Abstention | когда модель обязана не отвечать |

## Правило point-in-time

Признак допустим, только если `available_at <= index_time` и он не вычислен из исхода или факта, появившегося после исхода. `observed_at` недостаточно: результат мог быть получен раньше, но подписан позже.

## Split

- все беременности одной женщины находятся в одном split;
- все окна одного сигнала находятся в одном split;
- временной test откладывается после train/validation;
- центр, устройство и версия parser контролируются во внешней проверке;
- threshold выбирается на validation, никогда на test.

## Пропуски

Маска пропуска не заменяет причину. Где возможно, хранятся `not_due`, `clinician_decision`, `patient_declined`, `unavailable`, `lost_to_followup`, `data_error` и фактическое окно плана.

## Минимальная data card

- источник, версия, дата получения и лицензия;
- population, grain и количество записей/пациентов;
- схема и единицы;
- целевые переменные и их ограничения;
- missingness, duplicates и известные ошибки;
- допустимые и запрещённые claims;
- checksum исходного файла;
- точный код подготовки;
- связь prepared dataset с исходной версией.
