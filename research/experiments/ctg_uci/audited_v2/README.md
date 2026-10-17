# Результаты после глубокого аудита · audited v2

Полная повторная nested-сетка всех6семейств и финальная сетка выполнены. Подготовка/источник сверены независимо, все текущие test-строки пересчитаны ручными формулами и проверкой сохранённых bundle. Внешних данных и идентификаторов пациенток не появилось.

## Основные варианты по development CV

| Вариант | Mean outer macro-F1 | Mean outer recall P | Test macro-F1 | Test recall P | Test AP P |
|---|---:|---:|---:|---:|---:|
| C_ovr/single | 0.8744 | 0.8623 | 0.8538 | 0.9143 | 0.9103 |
| D_hierarchy/single | 0.8724 | 0.9001 | 0.8454 | 0.9143 | 0.9403 |

Primary C_ovr/single. Разница C/D по CV мала; уникальное превосходство не установлено. Обе процедуры оставлены single: ансамбль не выполнил условие validation gain≥.01. Для обоих selected policy — temperature calibration и argmax, отдельный патологический cutoff не принят.

На известном test максимум macro-F1 среди всех зафиксированных кандидатов у B_balanced/single:0.8838. Это выше основного C и альтернативного D на данном разделе; результат не скрыт и не использован для смены primary после test. Вложенная CV ранжировала его третьим, ниже C/D. Все кандидатные модели сохранены для прозрачного сравнения.

Test recall32/35=91.43% опирается лишь на10патологических файлов. При равном весе исходного файла recall85.76%. Парный95%file-bootstrap CI разницы D−C macro-F1 −.0351…+.0133 включает0. Bootstrap относится к фиксированным моделям, не всему процессу обучения.

## Артефакты

- Текущие02/03/05и аудиторский07 находятся в research/notebooks/ctg_uci/.
- nested_grid.json:630комбинаций, по3внутренних оценки. final_grid.json:126комбинаций.
- selection.json и locked_decision.json: решения до нового test; known-test статус указан.
- test_metrics.csv:все8кандидатов, основные и file-balanced метрики.
- test_paired_intervals.csv:1500парных повторов по исходным файлам.
- validation_receipt.json:ручной пересчёт, grouped boundaries, native/bundle roundtrip, notebook hashes, SHA-256.
- Модели:models/exported/ctg_uci/audited_v2/champion.joblib и best_models.json.
- Предыдущие supervised-ноутбуки:research/notebooks/ctg_uci/history/v1/. Старые04/06и их результаты сохранены побайтно.

## Ограничения

Известный test не является слепым. NSP — экспертный surrogate, не исход ребёнка. Source-file grouping не устанавливает независимость пациенток. Внутренние cross-fit scores — оценки подбора; внешний слой оценивает всю процедуру, но выбор среди семейств сохраняет оптимизм. UMAP/AE-кластеры не подтверждают устойчивые группы риска и не используются в supervised X.
