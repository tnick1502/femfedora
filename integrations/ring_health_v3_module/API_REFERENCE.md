# API_REFERENCE — Ring Health v3: входы, выходы, статусы

Два модуля: **цикл/фертильность** (окно фертильности, овуляция, следующие месячные — по образцу Oura Cycle Insights / Fertile Window) и **менопаузальный переход** (стадия STRAW+10, вероятность последней менструации в горизонтах 6–60 мес, ночные приливы). Модели обучены на синтетике (генератор откалиброван по литературе и реальным данным FedCycleData и NHANES). Клиническая точность на пользователях кольца не установлена. Ответ всегда содержит `mode=demonstration`, `clinical_validation=not_established` и пустой массив `recommendations`.

## Как подключить

| Способ | Вызов |
|---|---|
| Python | `from src.ring_health_v3.engine import RingHealthV3; RingHealthV3().analyze(payload)` — объект создаётся один раз, потокобезопасен |
| HTTP | `uvicorn src.ring_health_v3.api:app --port 8790`; `POST /v3/analyze`, `GET /health/ready`; ключ — `Authorization: Bearer $RING_HEALTH_API_KEY` |
| Ядро femfedora | `RingFertilityComponent().compute(values, ctx)` и `RingMenopauseComponent().compute(values, ctx)` из `src/ring_health_v3/adapter_core.py` — карточка в режиме SHADOW со статусами `ok / insufficient_data / not_applicable / MANUAL_REVIEW / COMPONENT_ERROR` |

Время ответа: обычно 20–60 мс; до ~0,3 с, если в длинном цикле ищется незалогированная менструация. Если последняя дата начала месячных старше 100 дней, сразу возвращается `anchor_stale` с просьбой указать последнюю дату — модель не угадывает множество незалогированных циклов.

## Вход

```json
{
  "as_of": "2025-04-10",
  "modules": ["fertility", "menopause"],
  "profile": {"age": 31, "pregnant": false, "breastfeeding": false, "hormonal_contraception": false,
              "menopausal_hormone_therapy": false, "hysterectomy": false, "bilateral_oophorectomy": false,
              "smoker": null, "bmi": null, "parous": null, "college_education": null},
  "cycle": {"period_starts": ["2025-01-03", "2025-01-31", "2025-03-01", "2025-03-29"],
            "last_period_date": null, "period_logs_complete": true, "logging_since": "2025-01-01", "menses_length": 5},
  "lh_tests": [{"date": "2025-03-12", "result": "positive"}],
  "symptoms": {"logging_enabled": true, "hot_flash_dates": [], "night_sweat_dates": []},
  "labs": {"fsh_iu_l": {"value": 38, "date": "2025-04-01"}, "amh_pg_ml": null},
  "nights": [{"date": "2025-04-10", "temperature_deviation": 0.21, "resting_hr": 57, "hrv_ms": 41,
              "respiratory_rate": 14.6, "sleep_minutes": 432, "sleep_efficiency": 88, "awake_minutes": 31}]
}
```

Вместо `nights` можно передать `oura` — документы Oura API v2: `{"sleep": [...], "daily_readiness": [...]}`. Используется самый длинный `long_sleep` за день, температура берётся из `daily_readiness.temperature_deviation`, секунды переводятся в минуты.

| Поле ночи | Источник Oura | Единицы |
|---|---|---|
| temperature_deviation | daily_readiness.temperature_deviation | °C, отклонение от личной базы (не абсолютная температура и не °F) |
| resting_hr | sleep.lowest_heart_rate | уд/мин |
| hrv_ms | sleep.average_hrv | мс, RMSSD |
| respiratory_rate | sleep.average_breath | вдохов/мин |
| sleep_minutes, sleep_efficiency, awake_minutes | sleep.total_sleep_duration/60, efficiency, awake_time/60 | мин, %, мин |

Правила: даты в формате YYYY-MM-DD (локальная дата кольца); дубликаты и даты позже `as_of` отклоняются (HTTP 422); история — не больше 800 дней. `null` означает пропуск, а `0` — это измерение. Обязательны только `as_of` и `profile.age`. Для фертильности дополнительно нужна хотя бы одна дата начала менструации.

## Пропуски: как устроено

- **Ничего не заполняется.** Пропущенная ночь просто отсутствует в правдоподобии байесовской модели цикла: корреляция AR(1) считается по фактическим интервалам между ночами. Это проверяется тестом.
- Любой сигнал можно не передавать. Только температура — рабочий режим. Без температуры модель опирается на ЧСС, HRV и дыхание. Совсем без ночей остаётся календарь по личной истории.
- Незалогированная менструация: если текущий цикл неправдоподобно длинный, модель сравнивает гипотезы «новый цикл начался в день N» по маргинальному правдоподобию. Когда вероятность ≥ 0.8, она возвращает `cycle_start_source=inferred_from_physiology` и `unlogged_period_suggestions` с действием `ask_user_to_confirm_period_start`. Покажите пользовательнице вопрос «Начались ли месячные DD.MM?».
- Ночи с вероятной лихорадкой (большой скачок температуры вместе с ЧСС) исключаются из оценки и считаются в `possible_illness_nights_last14`.
- В модели менопаузы признаки-индикаторы пропусков не используются. Необязательные входы при обучении скрывались случайно, независимо от исхода, поэтому «не передано» не несёт сигнала об исходе. Это ошибка, найденная в ревью preterm и КТГ: здесь она исключена дизайном.

## Выход: фертильность (`fertility`)

| Поле | Смысл |
|---|---|
| status | `ok`, `anchor_required` (нет ни одной даты начала месячных), `anchor_stale` (последняя дата старше 100 дней), `out_of_scope` (беременность, ГВ, гормональная контрацепция, операции), `not_applicable_age` (младше 18 или старше 50; синтетическая проверка охватывала 20–45 лет) |
| cycle_day, cycle_start, cycle_start_source | день цикла и откуда взято начало цикла (`logged` или `inferred_from_physiology`) |
| phase | `menstruation`, `follicular`, `fertile_window`, `ovulation`, `luteal`, `late_luteal`, `luteal_expected` |
| fertile_window.today_probability | P(сегодня входит в 6-дневное окно: овуляция − 5 … овуляция) |
| fertile_window.in_window_today, window{start,end,peak_day,status}, forecast_7d | окно для календаря и вероятности на 7 дней вперёд |
| ovulation.status | `predicted` (прогноз), `detected` (подтверждена устойчивым физиологическим сдвигом), `likely_passed_not_confirmed`, `not_detected_possible_anovulatory_cycle` |
| ovulation.date, interval_80, probability_ovulatory_cycle | дата: при `detected` — мода апостериорного распределения, иначе медиана; 80% интервал |
| next_period.date, interval_80, days_until, late_or_unlogged | прогноз следующей менструации |
| insights | длина цикла (медиана, диапазон, SD), регулярность, длина лютеиновой фазы, короткие лютеиновые фазы, возможные ановуляторные циклы |
| history[] | прошлые циклы: `ovulation_detected` (только при ≥8 ночах сигналов и ≥3 ночах до и после оценки), дата, `ovulation_date_estimate` с `estimate_basis` (physiology / calendar_or_sparse_data), лютеиновая фаза, температурный сдвиг °C |
| data_quality | покрытие ночей, давность последней температуры, ночи с подозрением на болезнь, `evidence` (physiology_and_history / history_only / population_prior_only), `confidence` (high/medium/low) |

Это **не** вероятность зачатия и **не** контрацепция. Подтверждение овуляции гормонами (LH-тесты) учитывается, если передано, но ответ остаётся модельной гипотезой.

## Выход: менопауза (`menopause`)

| Поле | Смысл |
|---|---|
| status | `ok`, `insufficient_data` (нет ни даты последней менструации/логов, ни ≥20 ночей кольца за 90 дней), `out_of_scope` (гормональная терапия, контрацепция, операции, беременность), `not_applicable_age` (младше 35; обучение покрывало 38–62 года) |
| stage.label, label_ru, probabilities | `reproductive`, `early_transition`, `late_transition`, `postmenopause`; вероятности всех четырёх |
| stage.basis, straw10_criteria | `STRAW+10: >=12 months without a period` — детерминированно, только если источник надёжен: дата последней менструации из онбординга или полные/длинные (≥ 400 дней) логи; иначе `model`. Флаги критериев: интервал ≥ 60 дней, стойкая разница длин циклов ≥ 7 дней |
| stage.age_only_reference | что дал бы один возраст (для сравнения в UI) |
| final_menstrual_period.probability_by_months | P(последняя менструация уже была или будет в течение 6/12/24/36/48/60 мес), монотонно |
| final_menstrual_period.median_months | ожидаемое время до последней менструации: `<6`, число или `>60` |
| vasomotor | `self_reported_symptom_days_last30` — дни с приливами или потливостью по дневнику (null, если дневник не ведётся); `index_last30` — средняя вероятность прилива по ночам кольца; `share_of_scored_nights_with_likely_hot_flashes_last30` — доля ночей с вероятностью ≥ 0,5 (модель ночи: температура, пробуждения, эффективность сна, HRV, ЧСС относительно личной базы). Самоотчёт учитывается, только если `symptoms.logging_enabled=true`; одна дата с приливом и ночной потливостью считается один раз |
| data_quality.input_pattern, stage.model_used | какие входы были (labs / bleeding_ring / bleeding_only / ring_only / age_only) и какая модель выбрана — маршрутизация подобрана на валидации |
| signals | доля овуляторных циклов по кольцу, вариабельность температуры, HRV относительно возрастной нормы, сон, параметры кровотечений |

Последняя менструация подтверждается только ретроспективно — после 12 месяцев без кровотечений. До этого момента модель даёт вероятность, а не диагноз.

## Примеры

В `examples/` лежат пары `*_input.json` / `*_output.json`. Фертильность: регулярный цикл, день в окне, только температура, разрыв вокруг овуляции, новый пользователь, только календарь, LH-тесты, формат Oura. Менопауза: поздний переход, тот же случай с ФСГ, только кольцо без логов, постменопауза, 45 лет в репродуктивной стадии, недостаточно данных.


## Машинные описания

- `schemas/request.schema.json` — JSON Schema запроса (из pydantic, авторитетная).
- `schemas/response.schema.json` — JSON Schema ответа, выведенная из 26 реальных ответов.
- `schemas/openapi.json` — OpenAPI HTTP-сервиса.
- `types/ring_health_v3.ts` — TypeScript-типы с комментариями.

## Полный список статусов

| Где | Значение | Когда |
|---|---|---|
| fertility.status | ok | расчёт выполнен (возможно, с низким доверием) |
| fertility.status | anchor_required | нет ни одной даты начала месячных |
| fertility.status | anchor_stale | последняя дата старше 100 дней |
| fertility.status | out_of_scope | беременность, ГВ, гормональная контрацепция, ЗГТ, гистерэктомия или удаление яичников |
| fertility.status | not_applicable_age | младше 18 или старше 50 |
| fertility.ovulation.status | predicted / detected / likely_passed_not_confirmed / not_detected_possible_anovulatory_cycle | см. AGENT_INTEGRATION §4.2 |
| fertility.phase | menstruation / follicular / fertile_window / ovulation / luteal / late_luteal / luteal_expected | фаза на `as_of` |
| fertility.data_quality.confidence | high / medium / low | high: ≥ 70% ночей с температурой за 14 дней и ≥ 2 прошлых овуляции; low: почти нет данных |
| menopause.status | ok / insufficient_data / out_of_scope / not_applicable_age | insufficient_data — нет ни даты/логов месячных, ни ≥ 20 ночей кольца за 90 дней |
| menopause.stage.label | reproductive / early_transition / late_transition / postmenopause | STRAW+10 |
| menopause.stage.basis | model / «STRAW+10: >=12 months without a period» / insufficient_data | |
| menopause.data_quality.input_pattern | labs / bleeding_ring / bleeding_only / ring_only / age_only | какие входы были, по ним выбрана модель |
