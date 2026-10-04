---
status: canonical
effective_date: 2026-09-23
tags:
  - architecture
  - security
  - provenance
---

# Целевая архитектура

Эта заметка сохраняет долгосрочную архитектуру продукта. Она не расширяет текущий [[01-current-ml-scope|ML-scope]].

## Поток данных

| Слой | Компоненты | Основной контроль |
|---|---|---|
| Источники | МИС/EHR, LIS, PACS, документы, ручной ввод, устройства | идентичность, правовое основание, время |
| Ingestion | FHIR/CSV/API, parser, unit normalization, terminology mapping | quarantine, provenance, проверка критичных полей |
| Clinical Event Store | Person, Pregnancy, Fetus, Encounter, Event | immutable history, исправления как новые версии |
| Care Plan | окна гестации, условия и ожидаемые события | просрочки и причины пропусков |
| Rule Engine | версионируемые калькуляторы | golden cases и deterministic replay |
| Feature/Model | point-in-time feature views, registry, calibration, OOD | leakage tests, rollback, abstention |
| API/UI | timeline, cards, review queue | RBAC, audit, explanation |
| Monitoring | data/model/process drift | alert, freeze и incident workflow |

## План наблюдения — отдельный слой

Care Plan описывает, какое событие ожидается в каком гестационном окне и при каком условии. Задача имеет состояния `planned`, `due`, `completed`, `overdue`, `not_applicable`, `declined/deferred`. Пропуск обследования не превращается в отрицательный клинический факт. Полный контракт: [[../20-domain/08-event-store-and-care-plan]].

## Извлечение из документов

| Класс | Примеры | Режим |
|---|---|---|
| C0 | оформление и заголовки | автоматически |
| C1 | некритичные наблюдения | автоматически при высокой уверенности |
| C2 | АД, GA, лаборатория, длина шейки | источник обязателен; ручная проверка при низкой уверенности |
| C3 | диагноз, аллергия, тромбофилия, противопоказание, исход | обязательное подтверждение человеком |

Для C2/C3 измеряются precision, recall, unit accuracy, date/GA accuracy, hallucination rate и доля ручных исправлений. Средний F1 по всем полям не считается достаточным.

Ручная очередь создаётся при конфликте единиц, биологически невозможном значении, несовместимых дате/GA, неоднозначной беременности, изменении категории карточки или извлечении C3-факта только из текста. Исправление не удаляет исходный parser output и автора подтверждения.

## Безопасность

- RBAC/ABAC по организации, роли, пациентке и цели доступа;
- шифрование при передаче и хранении;
- неизменяемый журнал чтения, изменения, расчёта и экспорта;
- сегрегация исследовательского и клинического контуров;
- псевдонимизация обучающих выборок и минимизация данных;
- backup/restore, incident response, model card, data sheet и SBOM;
- независимый rollback rule, parser и model версий.

## API результата карточки

```json
{
  "card": "preeclampsia",
  "evaluated_at": "...",
  "ga_days": 154,
  "status": "ELEVATED",
  "mode": "RULE_PLUS_SHADOW",
  "rule": {"version": "...", "score": 3, "reasons": [], "missing_critical": []},
  "model": {"visible": false, "version": "...", "risk": 0.083, "interval": [0.052, 0.124], "ood": false},
  "next_step": {"protocol_ref": "...", "due_at": "..."},
  "provenance": ["event/123", "event/456"]
}
```

## Обязательные поставки зрелого продукта

Репозиторий и воспроизводимая сборка; data dictionary и event schema; реестр правил с golden cases; data sheets и model cards; point-in-time pipeline и leakage tests; TRIPOD+AI/PROBAST+AI отчёт; UI timeline/cards/review; audit/RBAC/backup/monitoring; обезличенный экспорт для независимой проверки; руководство и протокол обучения персонала.

Проверяемые сценарии, матрица требований и реестр рисков описаны в [[05-verification-and-risk-controls]].
