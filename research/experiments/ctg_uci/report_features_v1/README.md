# Реальные бланки КТГ: выполненный эксперимент

Цель приложения — дородовое наблюдение, аппарат заранее неизвестен. UCI содержит экспертный NSP, а не клинический исход. Новая ветка изолирована; audited v2 и кластеризация сохранены.

## Метод

Три контракта измерений, исходные/производные признаки, A/Bsqrt/Bbalanced/C OvR/D hierarchy/логистика. 5×3 вложенные фолды по исходным записям, одинаковые с audited v2. 1380 кандидатов на внешних fit-частях = 4140 внутренних оценок; ещё 276 кандидатов и 828 оценок в финальном grid. Mean inner macro-F1 выбирает модель и engineering совместно. Аргмакс фиксирован. Validation отдельно выбирает пятифолдовый soft vote при приросте ≥.01 и обучает scalar temperature. Test ранее известен, новая слепая внешняя проверка отсутствует.

## Результаты

| Входы | Outer macro-F1 | Test macro-F1 | Recall suspect | Recall pathology | Precision pathology | AP pathology |
|---|---:|---:|---:|---:|---:|---:|
| core2 | 0.4745 | 0.4744 | 67.2% | 2.9% | 2.3% | 0.2119 |
| common3 | 0.5224 | 0.5463 | 79.3% | 34.3% | 17.6% | 0.2525 |
| sisporto9 | 0.8542 | 0.8436 | 70.7% | 88.6% | 77.5% | 0.9222 |

core2: baseline+accelerations. common3 добавляет сумму DL+DS+DP как условный proxy общей частоты децелераций. sisporto9 требует девять нативных показателей, включая вариабельности SisPorto; на приложенной фотографии их нет.

С 3 показателями найдены 12/35 патологических и 46/58 подозрительных случаев; с 9 — 31/35 и 41/58. Это разные условия доступности измерений, а не взаимозаменяемые модели. Для common3 precision pathology всего 17.6%; медицинское автоматическое применение не обосновано. Интервал recall pathology по исходным файлам common3 0–.533, native9 .700–.966; test содержит лишь 10 файлов с патологией.

Новые функции тех же входов не восстановили утраченную информацию: mean outer F1 raw/engineered common3 .526/.522, native9 .868/.848. В 302 train-сегментах одинаковые common3-входы принадлежат группам с разными NSP. Производные не устраняют такую неоднозначность.

## Экспорт

- core2: {'family': 'B_sqrt', 'engineered': True, 'params': {'depth': 3, 'iterations': 300, 'l2_leaf_reg': 10, 'learning_rate': 0.06}}; members=1, temperature=1.0509.
- common3: {'family': 'B_balanced', 'engineered': True, 'params': {'depth': 3, 'iterations': 600, 'l2_leaf_reg': 3, 'learning_rate': 0.06}}; members=5, temperature=0.9343.
- sisporto9: {'family': 'C_ovr', 'engineered': True, 'params': {'depth': 3, 'iterations': 600, 'l2_leaf_reg': 3, 'learning_rate': 0.06}}; members=1, temperature=1.4765.

`models/exported/ctg_uci/report_features_v1/`: bundle, schema, нативные деревья, границы train и отдельный файл скейлеров каждого профиля. Победили деревья; скейлер им не нужен, список скейлеров явно пуст. Для логистических кандидатов scaler fit выполнялся внутри каждого fit-фолда; unit test независимо проверяет медианы/IQR и неизменность при predict.

## Приложение

Шаблон JSON/CSV, JSON Schema и ручная транскрипция: `data/processed/ctg_uci/report_features_v1/`. Адаптер `src/features/ctg_pregnancy_app.py` принимает неизвестное устройство для хранения; clinical_risk=null. STV ms не подменяет UCI MSTV, высокие/низкие эпизоды не подменяют ASTV/ALTV, «20 потерянных ударов» не означает 20 BPM. 8/8 и галочки не являются признаками или независимой меткой.

Полный разбор и исходники графиков — в выполненном `08_hospital_reports_feature_engineering.ipynb`. Общие рекомендации по вводу — `PREGNANCY_APP_INPUT.md`.

## Дополнительный дородовый источник

JNU-CTG 1.0.0 скачана с https://zenodo.org/records/21800730, сверена по опубликованному MD5; metadata/README/LICENSE сохранены. Проверено 20769 waveform-файлов, 12606 patient_id; FIGO binary 9598/11171. Исход neonatal_asphyxia: 154 положительных записи от 91 пациентки, 58 пропусков метки. Это другой endpoint; новая модель на нём в этой ветке не обучалась. Единицы/экстрактор F_STV/F_LTV требуют проверки; небезопасные поздние исходы и непроверенные latent features не добавлялись в X. Источник даёт основу для отдельного дородового обучения, но не доказывает совместимость аппаратов.

## Проверки

`validation_receipt.json`: ручной пересчёт F1/recall/log loss/Brier/AP и file weights, границы фолдов, исходные hashes, roundtrip bundle/CBM, повторный расчёт temperature только по validation. `jnu_candidate/source_audit.json`: источник и ограничения нового архива. Unit tests проверяют единицы, пропуски, запрет выходных оценок в X, независимость batch, scaler и типы ввода приложения.
