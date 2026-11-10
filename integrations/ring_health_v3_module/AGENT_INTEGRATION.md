# AGENT_INTEGRATION — как встроить модуль Ring Health v3 в приложение

Этот файл написан для агента-разработчика. Задача «добавь модуль ring health» решается по шагам ниже. Справочник полей — `API_REFERENCE.md`; машинные схемы — `schemas/`; типы для фронтенда — `types/ring_health_v3.ts`; примеры вход/выход — `examples/`; тестовые пользователи — `demo_users/`.

## 0. Что это и чего это НЕ делает

Один вызов `RingHealthV3().analyze(request) -> response` возвращает два независимых блока:
- `fertility` — цикл и фертильность. День и фаза цикла; вероятность фертильного дня сегодня и прогноз на 7 дней; даты окна; овуляция (прогноз или подтверждение) с датой; следующие месячные; история прошлых циклов; качество данных; подсказка подтвердить незалогированные месячные.
- `menopause` — менопаузальный переход. Стадия STRAW+10; вероятность последней менструации (FMP) в течение 6–60 мес и медианное время; ночные приливы.

НЕ делает: не даёт рекомендаций и диагнозов, это не контрацепция и не вероятность зачатия. Каждый ответ содержит `clinical_validation: "not_established"`, `mode: "demonstration"`, `recommendations: []` — показывай пользователю дисклеймер (раздел 6).

## 1. Установка

```bash
# из корня модуля (папка, где лежит этот файл)
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # scikit-learn строго 1.9.1 — иначе движок откажется стартовать
python -m pytest tests -q                # 24 теста должны пройти
python scripts/run_demo.py               # прогон 10 тестовых женщин, сверка с сохранёнными ответами
```

Если модуль положен внутрь другого репозитория, импортируй пакет как `src.ring_health_v3`, запуская из корня модуля. Либо добавь корень модуля в `PYTHONPATH`. Если артефакты лежат отдельно, задай `RING_HEALTH_V3_ROOT=/путь/к/корню_модуля` — там должны быть `models/exported/ring_health/v3/` и `models/registry/ring_health_v3.json`.

## 2. Три способа вызова

### 2.1 Python (рекомендуется для бэкенда)

```python
from src.ring_health_v3.engine import RingHealthV3

ENGINE = RingHealthV3()          # создать ОДИН раз при старте: проверяет SHA256 артефактов и грузит модели (~1 с)

def analyze_user(request: dict) -> dict:
    """request — dict по схеме schemas/request.schema.json. Потокобезопасно, без состояния между вызовами."""
    try:
        return ENGINE.analyze(request)
    except ValueError as e:        # pydantic ValidationError тоже наследует ValueError
        return {"error": "invalid_input", "detail": str(e)}
```

### 2.2 HTTP-сервис

```bash
export RING_HEALTH_API_KEY=$(openssl rand -hex 32)
uvicorn src.ring_health_v3.api:app --host 127.0.0.1 --port 8790 --no-access-log
# или Docker: docker build -f deployment/Dockerfile -t ring-health-v3 . && docker compose -f deployment/compose.yaml up -d
```

- `POST /v3/analyze`, заголовок `Authorization: Bearer <ключ>`, тело — request JSON → 200 с response JSON.
- Ошибки: 401 — неверный ключ; 413 — тело больше 3 МБ; 422 — вход не прошёл проверку (`{"error":"invalid_input","details":[...]}`).
- `GET /health/ready` возвращает версию и SHA реестра. OpenAPI — `schemas/openapi.json`.
- Тело запроса не логируется. Ключ хранится только на бэкенде, в мобильный клиент его не класть.

### 2.3 Ядро femfedora (карточка, режим SHADOW)

```python
from src.ring_health_v3.adapter_core import RingFertilityComponent, RingMenopauseComponent
values = {"nights": [...], "period_starts": ["2026-10-22", ...], "last_period_date": None, "lh_tests": [], "symptoms": {}, "labs": {}}
# вместо "nights" можно передать "oura": {"sleep": [...], "daily_readiness": [...]}
ctx = {"as_of": "2026-11-09", "profile": {"age": 24, ...}}   # лишние ключи профиля игнорируются
card = RingFertilityComponent().compute(values, ctx)
# card = {"component": "ring.fertility.v3", "mode": "SHADOW", "status": "ok|insufficient_data|not_applicable|MANUAL_REVIEW|COMPONENT_ERROR", "result": {...}}
```

## 3. Как собрать request из данных приложения

| Данные приложения | Поле request | Правило |
|---|---|---|
| сегодняшняя локальная дата пользовательницы | `as_of` | YYYY-MM-DD в её часовом поясе. Ничего после `as_of` передавать нельзя (422) |
| какие экраны нужны | `modules` | `["fertility"]`, `["menopause"]` или оба. Цикл — 18–50 лет, менопауза — 35+ |
| профиль | `profile` | Обязателен только `age`. Флаги `pregnant`, `breastfeeding`, `hormonal_contraception`, `menopausal_hormone_therapy`, `hysterectomy`, `bilateral_oophorectomy` по умолчанию false. Если любой true — модуль вернёт `out_of_scope`. `smoker`, `bmi`, `parous`, `college_education` необязательны (null = неизвестно) |
| отметки начала месячных в календаре | `cycle.period_starts` | Все известные даты, без дублей. Можно старше истории кольца |
| вопрос онбординга «когда начались последние месячные?» | `cycle.last_period_date` | Особенно важно для 40+ и для тех, кто не ведёт календарь |
| пользовательница подтвердила, что отмечает все месячные | `cycle.period_logs_complete` | true только при явном подтверждении |
| с какой даты ведёт календарь в приложении | `cycle.logging_since` | Помогает модели менопаузы отличить «не было месячных» от «не отмечала» |
| средняя длительность кровотечения | `cycle.menses_length` | Для фазы «менструация»; по умолчанию 5 |
| LH-тесты | `lh_tests` | `[{"date", "result": "positive"|"negative"}]` |
| дневник симптомов | `symptoms` | `logging_enabled: true`, если она ведёт дневник (тогда «нет отметки» = «не было»); `hot_flash_dates`, `night_sweat_dates` |
| анализы | `labs.fsh_iu_l`, `labs.amh_pg_ml` | `{"value", "date"}`, ФСГ в МЕ/л, АМГ в пг/мл |
| данные кольца | `nights` ИЛИ `oura` | `nights` — одна строка на локальную ночь, null = нет данных. `oura` — сырые документы Oura API v2 `sleep` и `daily_readiness`, всё за последние до 800 дней |

Пример сборки (Python):

```python
def build_request(user, today, ring_nights, period_starts, lh, symptoms, labs, modules):
    return {
        "as_of": today.isoformat(),
        "modules": modules,
        "profile": {"age": user.age, "pregnant": user.pregnant, "breastfeeding": user.breastfeeding,
                    "hormonal_contraception": user.on_pill, "menopausal_hormone_therapy": user.on_hrt,
                    "hysterectomy": user.hysterectomy, "bilateral_oophorectomy": user.ovaries_removed,
                    "smoker": user.smoker, "bmi": user.bmi, "parous": user.has_children, "college_education": user.higher_education},
        "cycle": {"period_starts": [d.isoformat() for d in period_starts], "last_period_date": iso_or_none(user.onboarding_last_period),
                  "period_logs_complete": user.confirms_all_periods_logged, "logging_since": iso_or_none(user.calendar_started)},
        "lh_tests": [{"date": t.date.isoformat(), "result": "positive" if t.positive else "negative"} for t in lh],
        "symptoms": {"logging_enabled": user.symptom_diary_on, "hot_flash_dates": [...], "night_sweat_dates": [...]},
        "labs": {"fsh_iu_l": {"value": labs.fsh, "date": labs.fsh_date.isoformat()} if labs.fsh else None},
        "nights": [{"date": n.date.isoformat(), "temperature_deviation": n.temp_dev, "resting_hr": n.lowest_hr, "hrv_ms": n.avg_hrv,
                    "respiratory_rate": n.avg_breath, "sleep_minutes": n.total_sleep_min, "sleep_efficiency": n.efficiency,
                    "awake_minutes": n.awake_min} for n in ring_nights],
    }
```

Когда вызывать: после каждой синхронизации кольца, после изменения календаря, LH или симптомов, и раз в сутки утром. Ответ привязан к `as_of` — не переиспользуй его на следующий день. Типичное время 20–60 мс, до ~0,3 с.

## 4. Как показывать ответ

### 4.1 Статусы

| Блок | `status` | Что показать |
|---|---|---|
| fertility | `ok` | Экран цикла (4.2) |
| fertility | `anchor_required` | «Укажите дату начала последних месячных» + поле даты |
| fertility | `anchor_stale` | «Последняя отмеченная менструация была больше 100 дней назад — укажите дату последних месячных» |
| fertility | `out_of_scope` | «Прогноз цикла недоступен при {reason}» (контрацепция, беременность, ГВ, операция, ЗГТ) |
| fertility | `not_applicable_age` | Скрыть экран цикла |
| menopause | `ok` | Экран перехода (4.3) |
| menopause | `insufficient_data` | «Укажите дату последних месячных или носите кольцо хотя бы 3 недели». Поле `needed` содержит список; можно показать `stage.age_only_reference` с пометкой «по возрасту» |
| menopause | `out_of_scope`, `not_applicable_age` | Скрыть или пояснить `reason` |
| любой | HTTP 422 / `ValueError` | Ошибка данных на стороне приложения: дубль даты, дата в будущем, NaN, строка вместо числа — исправить сборку request |

### 4.2 Фертильность → UI (как в Oura Cycle Insights)

| Виджет | Поля |
|---|---|
| Главная карточка: «День цикла N · фаза» | `cycle_day`, `phase` (подписи в 4.4) |
| Плашка «Фертильное окно» | `fertile_window.in_window_today`, `today_probability`. `window.start`…`window.end`, `window.status`: past / now / upcoming |
| Мини-график на 7 дней | `fertile_window.forecast_7d[].fertile_probability` (0–1) |
| Бейдж овуляции | `ovulation.status`: predicted → «Ожидается ~{date}»; detected → «Овуляция подтверждена по физиологии {date}»; likely_passed_not_confirmed → «Вероятно, прошла ~{date}, ждём подтверждения»; not_detected_possible_anovulatory_cycle → «В этом цикле подъём не обнаружен». Интервал — `interval_80` |
| Следующие месячные | `next_period.date`, `days_until`, `interval_80`. Если `late_or_unlogged` — «Месячные задерживаются или не отмечены» |
| Подтверждение незалогированных месячных | Если `unlogged_period_suggestions` не пуст: «Похоже, месячные начались {date}. Подтвердить?» → да: добавить дату в календарь и пересчитать. `cycle_start_source = inferred_from_physiology` — день цикла посчитан от предполагаемой даты |
| Инсайты | `insights`: медиана и диапазон длины цикла, `regularity`, `luteal_length_median`, `possible_anovulatory_cycles`, `short_luteal_cycles` |
| История | `history[]`: старт, длина, `ovulation_detected` + `ovulation_date` (если false — можно показать `ovulation_date_estimate` с пометкой «оценка»), `luteal_length`, `temperature_shift_c` |
| Качество данных | `data_quality.confidence` (high/medium/low) → полупрозрачность прогноза. `coverage_last14`, `days_since_last_temperature`; `possible_illness_nights_last14 > 0` → «Похоже, вы болели — эти ночи исключены» |

Цвета календаря: менструация (дни 1…`menses_length` от `cycle_start` и окно `next_period.interval_80`), фертильное окно (`window`), день овуляции (`ovulation.date`), лютеиновая фаза — после овуляции.

### 4.3 Менопауза → UI

| Виджет | Поля |
|---|---|
| Стадия | `stage.grouped.label` (reproductive / transition / postmenopause) крупно; подстадия `stage.label_ru`. Вероятности `stage.probabilities` можно дать полоской. При `stage.basis` STRAW — «Подтверждено: 12 месяцев без менструаций» |
| Признаки | `stage.straw10_criteria`: `interval_ge60` → «был перерыв ≥ 60 дней»; `persistent_diff7` → «длина циклов меняется на ≥ 7 дней» |
| Прогноз FMP | `final_menstrual_period.probability_by_months` — шкала 6/12/24/36/48/60 мес; `median_months` — «ожидаемо через ~N мес» (строки `<6` и `>60` бывают); `already_confirmed` |
| Приливы | `vasomotor.self_reported_symptom_days_last30` — по дневнику; `vasomotor.index_last30` — оценка по ночам кольца (слабый сигнал, подавать как «ночи с признаками приливов») |
| Сигналы | `signals.lnhrv_age_resid` (HRV относительно возрастной нормы, ln), `eff_mean_30`, `awake_mean_30`, `ring_ovulatory_fraction` |
| Доверие | `data_quality.confidence`, `input_pattern`, `labs_used` |

### 4.4 Подписи фаз (RU)

menstruation — «Менструация»; follicular — «Фолликулярная фаза»; fertile_window — «Фертильное окно»; ovulation — «Овуляция»; luteal — «Лютеиновая фаза»; late_luteal — «Скоро месячные»; luteal_expected — «Вероятно, лютеиновая фаза».

## 5. Типы и схемы

- `types/ring_health_v3.ts` — TypeScript-интерфейсы запроса и ответа с комментариями.
- `schemas/request.schema.json` — JSON Schema запроса (из pydantic, авторитетная).
- `schemas/response.schema.json` — JSON Schema ответа, выведенная из всех примеров.
- `schemas/openapi.json` — OpenAPI HTTP-сервиса.

Пример клиента (TypeScript):

```ts
import type { AnalyzeRequest, AnalyzeResponse } from "./ring_health_v3";
export async function analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const r = await fetch(`${process.env.RING_HEALTH_URL}/v3/analyze`, {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RING_HEALTH_API_KEY}` },
    body: JSON.stringify(req),
  });
  if (r.status === 422) throw new Error("ring-health: invalid input " + JSON.stringify(await r.json()));
  if (!r.ok) throw new Error(`ring-health: HTTP ${r.status}`);
  return r.json();
}
```

## 6. Обязательный дисклеймер в UI

«Оценка по данным кольца и вашему календарю. Модель обучена на синтетических данных и не проверена клинически. Не является диагнозом, методом контрацепции или вероятностью зачатия.»

## 7. Проверка интеграции (чек-лист агента)

1. `python -m pytest tests -q` → 24 passed.
2. `python scripts/run_demo.py` → 10/10 OK.
3. Отправить в свой эндпоинт `demo_users/*/input.json` и сравнить с `output.json`. Поле `artifact_registry_sha256` должно совпасть.
4. Проверить UI на всех статусах. Готовые входы для них: `examples/fertility_new_user_*` (мало данных), `examples/fertility_calendar_only_*` (нет кольца), `examples/fertility_out_of_scope_*`, `examples/menopause_insufficient_*`, `demo_users/03_*` (подсказка о незалогированных месячных), `demo_users/05_*` (ночи болезни).
5. Не отправлять в API файлы `_ground_truth_for_qa.json` — это скрытая «истина» генератора только для QA.
