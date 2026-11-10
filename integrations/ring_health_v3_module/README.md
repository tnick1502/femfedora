# Ring Health v3 — модуль анализа цикла и менопаузы по данным умного кольца

Самодостаточный модуль: код, обученные модели, контракт входов и выходов, документация для агента-интегратора, примеры и 10 тестовых пользовательниц.

> Агенту: начни с **AGENT_INTEGRATION.md**. Задача «добавь модуль» = §1 установка → §2 вызов → §3 сборка request из данных приложения → §4 отрисовка ответа → §7 чек-лист.

## Что внутри

| Путь | Что это |
|---|---|
| `AGENT_INTEGRATION.md` | **Пошаговая интеграция**: установка, вызов (Python / HTTP / ядро), сборка запроса из данных приложения, статусы → UI, виджеты, код |
| `API_REFERENCE.md` | Все поля входа и выхода, единицы, статусы |
| `TECHNICAL_DESCRIPTION.md` | Как устроены модели, на чём обучены, качество, ограничения, переобучение |
| `types/ring_health_v3.ts` | TypeScript-типы запроса и ответа |
| `schemas/` | JSON Schema запроса и ответа, OpenAPI |
| `src/ring_health_v3/` | Python-пакет (точка входа `engine.RingHealthV3`) |
| `models/` | Артефакты (параметры модели цикла, модели менопаузы) + реестр с SHA256 |
| `examples/` | 16 пар `*_input.json` / `*_output.json`: все режимы и статусы |
| `demo_users/` | 10 реалистичных тестовых женщин (24–53 года, разные цели, привычки и профили) + `README_AGENT.md` |
| `tests/` | 24 теста: контракт, причинность, пропуски, отказоустойчивость, HTTP |
| `scripts/run_demo.py` | Прогон всех примеров и демо-пользовательниц со сверкой ответов |
| `deployment/` | Dockerfile + compose |
| `reports/` | Полный исследовательский отчёт, метрики, графики, доказательная база |

## Быстрый старт

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests -q            # 24 passed
python scripts/run_demo.py           # 26 identical, 0 different
python -c "
import json; from src.ring_health_v3.engine import RingHealthV3
r = RingHealthV3().analyze(json.load(open('demo_users/01_anna_24_ttc_lh/input.json')))
print(r['fertility']['phase'], r['fertility']['fertile_window']['today_probability'], r['fertility']['ovulation'])"
uvicorn src.ring_health_v3.api:app --port 8790     # HTTP: POST /v3/analyze
```

Требования: Python 3.12+, scikit-learn **1.9.1** (артефакты привязаны к версии); остальное — в `requirements.txt`.

## Важно

Модели обучены на синтетике, откалиброванной по опубликованным исследованиям (Oura, SWAN, InterLACE, Ava, WHOOP) и реальным открытым данным (FedCycleData, NHANES). Клиническая точность на пользователях кольца не установлена. Это не диагноз, не контрацепция и не вероятность зачатия. Каждый ответ несёт `clinical_validation: "not_established"`. Дисклеймер для UI — в AGENT_INTEGRATION §6.
