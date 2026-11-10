"""Build evidence report, examples, charts and a portable v2 research release."""
from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ring_health.api_v2 import app
from src.ring_health.engine_v2 import Engine

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/"research/experiments/ring_health/v2"


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main():
    engine=Engine()
    metrics=json.loads((EXP/"deployed_missingness_metrics.json").read_text())
    training=json.loads((EXP/"training_metrics.json").read_text())
    real=json.loads((EXP/"real_thermal_metrics.json").read_text())
    samples=ROOT/"data/samples/ring_health/v2"
    base=json.loads((ROOT/"data/samples/ring_health/fertility_demo_input.json").read_text())
    examples={"fertility":base}
    temp=json.loads(json.dumps(base));temp["daily"]=[{"date":r["date"],"temperature_deviation":r.get("temperature_deviation")} for r in temp["daily"]]
    examples["temperature_only"]=temp
    gap=json.loads(json.dumps(temp))
    for row in gap["daily"][-7:]:row["temperature_deviation"]=None
    examples["seven_day_gap"]=gap
    prior=json.loads(json.dumps(base));prior["daily"]=[]
    examples["calendar_only"]=prior
    examples["menopause"]=json.loads((ROOT/"data/samples/ring_health/menopause_demo_input.json").read_text())
    for name,payload in examples.items():
        dump(samples/(name+"_input.json"),payload)
        dump(samples/(name+"_output.json"),engine.analyze(payload))
    dump(ROOT/"docs/ring_health_v2.openapi.json",app.openapi())
    modes=["complete","random10","random30","random50","random70","block3","block7","block14","temperature_only","no_temperature","metric_dropout","illness_nonwear","missed_period_log"]
    labels=["Complete","10% missing","30% + cold start","50% + cold start","70% + cold start","3-day block","7-day block","14-day block","Temperature only","No temperature","Channel dropout","Illness nonwear","Unlogged period"]
    fig,ax=plt.subplots(1,2,figsize=(13,6),layout="constrained")
    rates=[metrics["regimes"][m]["detection_rate_all_ovulatory"]*100 for m in modes]
    ax[0].barh(labels[::-1],rates[::-1],color="#387e90")
    ax[0].set(xlim=(0,100),xlabel="Cycles with a thermal-shift detection (%)",title="Synthetic heldout people: actual API")
    values=[metrics["regimes"][m]["mae_detected_only"] for m in modes]
    ax[1].barh(labels[::-1],[v if v is not None else 0 for v in values][::-1],color="#b77b47")
    for i,m in enumerate(modes[::-1]):
        v=metrics["regimes"][m]
        ax[0].text(v["detection_rate_all_ovulatory"]*100+1,i,f'{v["detected_cycles"]}/{v["ovulatory_cycles"]}',va="center",fontsize=8)
        if v["mae_detected_only"] is None:ax[1].text(.2,i,"N/A: no detected events",va="center",fontsize=8)
    ax[1].set(xlabel="MAE, days (detected cycles only)",title="Read together with coverage on the left")
    for a in ax:a.spines[["top","right"]].set_visible(False)
    fig.suptitle("Research scenarios; these results do not establish performance on Oura users")
    fig.savefig(EXP/"missingness_coverage.png",dpi=160);plt.close(fig)
    rows=[]
    for m in modes:
        v=metrics["regimes"][m];mae="—" if v["mae_detected_only"] is None else f'{v["mae_detected_only"]:.2f}'
        rows.append(f'| {m} | {v["forecast_coverage"]:.1%} | {v["positive_f1_all_days_unavailable_as_negative"]:.3f} | {v["detected_cycles"]}/{v["ovulatory_cycles"]} ({v["detection_rate_all_ovulatory"]:.1%}) | {mae} | {v["within2_all_ovulatory"]:.1%} |')
    report='''# Перепроверка моделей кольца, версия 2

Технический результат: обученные артефакты, строгая схема входа, API, контейнер, примеры и воспроизводимая оценка. Научный результат: удалось обосновать отдельные физиологические предпосылки и обработку пропусков. Точность на реальной популяции пользователей Oura и сопоставимость с Oura пока не установлены. Синтетическая генерация не устраняет отсутствие совместных реальных измерений кольца и эталонных исходов.

## Что исправлено после независимой проверки

Два независимых рецензента проверили методологию и развёртывание. Убраны требования ко всем семи каналам и нескольким предыдущим циклам; достаточно температуры и одного достоверного начала цикла либо текущего дня цикла. HR, HRV, дыхание и сон необязательны. Отсутствие температуры допускает только календарный режим. LH для работы API не нужен, но требуется как независимый эталон для будущей проверки.

Признаки для утреннего прогноза используют только доступную историю. Шаги текущего дня исключены. Даты и конечные числа проверяются строго; обязательны as_of и контекст. Oura-адаптер объединяет даты readiness, sleep и activity: температура больше не теряется из-за отсутствующего sleep. Для длинного или неопределённого цикла модель возвращает явный статус без ложного точного события. Устаревшее температурное событие при неполных записях менструаций требует уточнения ориентира; консервативный горизонт 14 дней является эксплуатационным допущением, а не клиническим правилом. Эта поправка проверялась на том же стресс-наборе: это регрессионная проверка исправления, не новая независимая внешняя валидация.

Модель не определяет дату новой менструации по одному падению температуры. Если начало нового цикла не отмечено, автоматическое восстановление цикла остаётся нерешённой задачей. В приложении нужен короткий запрос подтверждения начала цикла при статусе uncertain_cycle_anchor.

## Научные источники и генерация

- [Bull et al., 2019](https://www.nature.com/articles/s41746-019-0152-7): 612 613 отобранных овуляторных циклов; средние длительности цикла 29.3 дня, фолликулярной фазы 16.9, лютеиновой 12.4; SD 5.2, 5.3 и 2.4. Это маргинальные параметры выбранной популяции, не совместное распределение всех сигналов.
- [Oura, JMIR 2025](https://www.jmir.org/2025/1/e60667): ретроспективное определение даты относительно LH; 1 155 овуляторных циклов, detection 96.4%, MAE 1.26 дня, 87.9% в пределах ±2 дней. Исключены случаи с более 40% пропусков за 60 дней. Это не F1 прогнозирования будущих фертильных дней и не оценка ложных сигналов в ановуляторных циклах.
- [Исследование запястной температуры, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6265623/): средний сдвиг 0.33°C и неодинаковая выраженность двухфазности. Перенос запястье→палец требует отдельной проверки.
- [Пульс, 2017](https://www.nature.com/articles/s41598-017-01433-9): средние различия +2.1 bpm в фертильном окне и +3.8 bpm в середине лютеиновой фазы относительно менструации. Временные формы и корреляции генератора остаются допущениями.
- [Модель со скрытым состоянием, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5575519/): методологическая опора для пропусков в наблюдениях; наша реализация не является воспроизведением опубликованной модели.
- [TempID](https://zenodo.org/records/20160611): отдельный профиль амплитуды и шума обучен на реальных обработанных данных. Это другой сенсор, данные уже выровнены и двунаправленно заполнены авторами. Исходную утечку устранить невозможно. Исправлено ночное окно на авторское 21:00–03:59 с отнесением к утру и минимумом 6 часовых измерений.
- [STRAW+10](https://pmc.ncbi.nlm.nih.gov/articles/PMC3580996/): стадия репродуктивного старения определяется анамнезом кровотечений; прогноз будущей финальной менструации нельзя валидировать на одном поперечном срезе.

Все численные предпосылки и их происхождение сохранены в models/configs/training/ring_health_evidence_v2.json. Доля ановуляции 10%, болезни, корреляция шума, распределение пропусков, эффекты HRV/дыхания и совместные зависимости явно помечены как сценарные допущения. Они не являются оценками распространённости в целевой популяции. Нельзя просто дополнить 200 анкет искусственными показателями кольца и считать полученные связи обнаруженными в исследовании.

Сгенерированы 360 отдельных участников по пять циклов; разделение по участникам 60/20/20, зафиксированные seed и списки людей. Для обучения используются случайные и блоковые пропуски, выпадение целых каналов, нерегулярные записи менструаций. Выбор модели по validation log loss, порога — на validation; желаемые метрики Oura в подборе не используются. Генератор, таблицы, split и артефакты включены в архив.

## Как обрабатываются редкие измерения

Измерения не заполняются придуманными температурами или LH. Температурная модель интегрирует неизвестное смещение и медленный дрейф, учитывает корреляцию соседних ночей и суммирует правдоподобие только по наблюдённым точкам. Используется смесь возможных амплитуд и компонента отсутствия овуляции; это не валидированная вероятность ановуляции у конкретного человека. Условные квантили даты события не являются клинически калиброванным доверительным интервалом.

При менее восьми температурных ночей или коротком временном охвате возвращается календарная оценка. После длинного перерыва температура перестаёт считаться актуальным свидетельством. Для сообщения о температурном сдвиге нужны наблюдения до и после предполагаемого события, достаточный сдвиг и ограниченная неопределённость. Даже такой статус не означает подтверждения овуляции гормонами.

ML-модели получают показатели наличия, давности и количества наблюдений; бустинг работает с NaN, альтернативный лес имеет imputer только для признаков. Это не восстановление медицинских измерений. Экспериментальные ML-вероятности выдаются отдельно от температурной оценки, чтобы не скрывать их синтетическое происхождение.

## Результаты повторной оценки

Оценивался именно Engine.analyze, используемый API: 72 отложенных синтетических участника, каждый допустимый день последнего заранее выбранного цикла начиная с дня 7, 13 сценариев, 22 230 запросов. Результат без прогноза остаётся в знаменателе. Один и тот же участник встречается в разных сценариях; это парные стресс-проверки, не независимые выборки. random30/50/70 одновременно проверяют холодный старт с одним известным началом цикла; различия с complete нельзя приписывать только доле пропусков. Блоки намеренно расположены около эталонного события для стресс-проверки; эталон не передаётся модели.

| Сценарий | Дни с прогнозом | F1 фертильного дня, порог 0.5 | Сдвиг обнаружен / овуляторные циклы | MAE обнаруженных, дни | Верно ±2 дня / все овуляторные |
|---|---:|---:|---:|---:|---:|
'''+"\n".join(rows)+'''

Покрытие прогнозом включает календарный режим и само по себе не означает наличие физиологического сигнала. Ошибка только среди обнаруженных циклов может выглядеть хорошо при низком обнаружении. В этом тесте всего пять ановуляторных циклов: даже ноль ложных сигналов не позволяет заявлять высокую специфичность. Интервалы bootstrap по людям и все агрегаты находятся в deployed_missingness_metrics.json. Время вычисления локально: p50 '''+f'{metrics["latency_ms_p50_p95"][0]:.1f}'+''' мс, p95 '''+f'{metrics["latency_ms_p50_p95"][1]:.1f}'+''' мс; это последовательный тест на данном компьютере, не нагрузочная гарантия API.

![Покрытие и ошибка](../experiments/ring_health/v2/missingness_coverage.png)

Отдельный тест обученных классификаторов на отложенных синтетических участниках с аугментацией пропусков:

| Модель | Метрика | Значение | Область доказательства |
|---|---|---:|---|
| Фертильный день, HGB | positive F1 | '''+f'{training["fertile_target"]["test"]["positive_f1"]:.3f}'+''' | Синтетические данные, validation-порог 0.27 |
| Постовуляторное состояние, HGB | positive F1 | '''+f'{training["postovulatory_target"]["test"]["positive_f1"]:.3f}'+''' | Синтетические данные, validation-порог 0.36 |
| Репродуктивная стадия, HGB | macro F1 | '''+f'{training["stage"]["test"]["macro_f1"]:.3f}'+''' | Только синтетика; не диагноз менопаузы |

Реальные TempID, 3-fold CV по 39 предполагаемым участникам и 59 сериям: MAE всех оценок '''+f'{real["complete"]["mae_all_estimates"]:.2f}'+''' дня; фильтр пригодности пропустил только '''+str(real["complete"]["reported_cycles"])+''' из 59 циклов ('''+f'{real["complete"]["coverage"]:.1%}'+'''). MAE этих трёх циклов 2.00 дня не свидетельствует о точности уровня Oura. Исходная обработка авторами, другой сенсор, отбор циклов и отсутствие проспективного протокола ограничивают даже эту слабую оценку. Это отдельный ретроспективный бенчмарк температурного профиля, а не проверка Oura-пути API.

Прежний возрастной baseline NHANES сохранён в v1: предсказывает текущий статус по анкете, не будущую менопаузу по кольцу. Анкета Jordan после удаления дубля содержит 199 строк, не содержит временных рядов кольца и имеет противоречия кодировки. Она не даёт независимой проверки wearable-модели. Реестры открытых источников и скачиваний из v1 сохранены.

## Готовность и проверка

Проверены API-валидация, причинность признаков, temperature-only, отсутствие температуры, перерывы, один/нулевой ориентир цикла, устаревшее событие, Oura union, контрольные суммы и отказ при повреждении артефактов. Linux-контейнер собран и проверен HTTP-запросом с теми же артефактами, токеном и обработкой ошибок. Версии зависимостей serving закреплены; артефакты загружаются один раз при старте. Контейнер работает без root, с read-only FS; запросы с медицинскими данными не пишутся access-log. Проверка контейнера не заменяет нагрузочную проверку и инфраструктурные настройки конкретного приложения.

Оценочная карточка глубокого аудита: проверено 6 материальных классов дефектов; исправления и оставшиеся ограничения записаны в review_findings.json. Методологические ограничения не считаются исправленными кодом. Покрытие таблиц отчёта: 13/13 сценариев; происхождение: 7/7 основных количественных источников размечены; категории dashboard/navigation — неприменимы. Оценки не являются медицинской сертификацией.

| Критерий готовности | Итог | Основание |
|---|---|---|
| Прослеживаемость источников и допущений | Выполнен в проверенном объёме | 7 источников, отдельные device profiles |
| Разделение участников и выбор по validation | Выполнен | Зафиксированные split, без подбора под Oura |
| Пропуски и покрытие фактического inference | Выполнен с ограничениями | 13 сценариев; потерянное начало цикла не восстанавливается автоматически |
| Строгий контракт и артефакты | Выполнен | 47 тестов, startup SHA256, ограниченный вход |
| Переносимость backend | Выполнен для локального smoke test | Docker HTTP 200/401/422; не нагрузочный тест |
| Читаемость графика и знаменателей | Выполнен | Ошибка показана рядом с покрытием |
| Независимая клиническая проверка кольца | Не выполнен | Нет подходящей парной когорты |
| Точность немного ниже Oura / будущая менопауза | Не подтверждены | Сопоставимых независимых исходов нет |

Итого 6/8 критериев исследовательской поставки выполнены в указанном объёме. Это оценка готовности поставки, не точность модели. Для Brier/AP в all-days оценке отсутствующий прогноз условно кодируется нулём как штраф за отказ; такие значения не доказывают калибровку вероятностей. 1 200 строк stage test — четыре маски 300 участников, а не 1 200 независимых женщин.

Развёртывание: docs/ring_health_v2_deployment_ru.md. Технически готов исследовательский backend для демонстрации. Научно обосновано показывать происхождение, наблюдённые данные, неопределённость и статусы. Нельзя заявлять «немного хуже Oura», подтверждённую овуляцию, вероятность беременности или прогноз возраста наступления менопаузы. Для такого заявления нужна новая независимая выборка сырых ночных данных целевого кольца, LH/УЗИ-эталон, ановуляторные циклы, зафиксированные endpoints, пропуски, coverage и person-level разделение до подбора модели.
'''
    (ROOT/"research/reports/ring_health_v2_review_ru.md").write_text(report)
    deployment='''# Ring Health v2: запуск в приложении

Это исследовательский backend с обученными моделями. Клиническая точность на пользователях кольца не установлена. Ответ содержит clinical_validation=not_established; клиент должен сохранять эту информацию и различать календарную оценку и температурное свидетельство.

Из корня распакованного архива:

```bash
export RING_HEALTH_API_KEY="$(openssl rand -hex 32)"
docker compose -f deployment/ring_health/compose.yaml up --build -d
curl --fail http://127.0.0.1:8788/health/ready
curl --fail http://127.0.0.1:8788/v2/analyze \\
  -H "Authorization: Bearer $RING_HEALTH_API_KEY" \\
  -H 'Content-Type: application/json' \\
  --data-binary @data/samples/ring_health/v2/fertility_input.json
```

Секрет хранится в backend/secret manager. Мобильное приложение обращается к вашему backend; не встраивайте общий ключ в клиент. Compose слушает только loopback. Внешний HTTPS, ограничение частоты, управление пользователями, хранение согласий и мониторинг настраиваются в инфраструктуре приложения. В этой работе внешний сервер не публиковался. Остановка: docker compose -f deployment/ring_health/compose.yaml down.

POST /v2/analyze: обязательны as_of (локальная YYYY-MM-DD), context.age, пять явных булевых флагов контекста и ровно один источник daily либо oura. Полная схема: docs/ring_health_v2.openapi.json. Даты без времени, уникальные, не позже as_of; история максимум 730 календарных дней, тело максимум 2 MB. Сигналы — конечные числа в диапазонах схемы или null; пропущенный канал также допустим. temperature_deviation — отклонение ночной температуры в °C, не абсолютные 36.5°C. Не передавайте Celsius deviation как Fahrenheit. HR — bpm, HRV — ms, дыхание — в минуту, длительности сна — минуты.

Фертильность: возраст 18–45 в области сценарной проверки; нужен хотя бы один известный первый день менструации в period_start_dates либо текущий cycle_day. Не требуются ручные ежедневные измерения, LH или полный набор каналов. Дополнительная история улучшает персональный prior. period_logs_complete=true задаётся только если все начала менструаций действительно внесены; отсутствие записи при false не трактуется как отсутствие менструации.

Статусы: thermal_shift_detected — гипотеза температурного сдвига; estimated_with_gaps — неполное температурное свидетельство; calendar_only / calendar_only_stale_measurements — календарный prior; anchor_required / uncertain_cycle_anchor — требуется подтверждение ориентира цикла; cycle_out_of_support — превышена поддерживаемая длина; outside_model_population — задан исключающий контекст. При отсутствии forecast не рисуйте нулевые вероятности: это отсутствие результата. Поле reference_event_date отсутствует/равно null без достаточного свидетельства. Вероятности модели не являются вероятностью зачатия. Следующий семидневный forecast является модельным сценарием с неопределённостью.

Менопауза: для экспериментального классификатора стадий требуется возраст 35–65, 90 календарных дней и 20 ночей температуры; остальные каналы необязательны. Вероятности обучены только на синтетике. future_onset_prediction и diagnosis не выдаются. Беременность, грудное вскармливание, гормональные препараты, гистерэктомия/двусторонняя овариэктомия исключены из области этих моделей.

Oura-адаптер принимает daily_readiness, sleep, daily_activity. Используется объединение дат. Проверяйте семантику экспортируемых полей вашей версии Oura и локальную дату завершения сна. Артефакты joblib загружаются только из доверенного release; контрольная сумма подтверждает целостность, не безопасность неизвестного pickle.

Воспроизведение в Python 3.12:

```bash
python3.12 -m venv .venv-ring
.venv-ring/bin/pip install -r requirements-ring-health.txt -r requirements-ring-health-serving.txt httpx==0.28.1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-ring/bin/python -m tools.train_ring_health_v2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-ring/bin/python -m tools.evaluate_ring_health_v2
.venv-ring/bin/python -m pytest tests/unit/test_ring_health_v2.py -q
.venv-ring/bin/python -m tools.build_ring_health_v2_release
```

Запуск не требует повторного обучения: готовые .joblib и профили входят в архив. Для сохранения эталонной оценки храните исходный release; повторное обучение перезаписывает артефакты. Текущие split и seed сохранены в research/experiments/ring_health/v2. Синтетические таблицы содержат истинные скрытые исходы для обучения/оценки: передавать их как вход приложения нельзя. Реальные данные TempID включены с атрибуцией Zenodo CC-BY 4.0; исходный авторский код и лицензия сохранены отдельно. Частная пользовательская анкета в переносимый архив v2 не включена.
'''
    (ROOT/"docs/ring_health_v2_deployment_ru.md").write_text(deployment)
    # Portable release includes data for replay, never private user questionnaire.
    files=[]
    directories=["src/ring_health","models/exported/ring_health/v2","data/samples/ring_health/v2","data/processed/ring_health/v2","data/external/ring_health/tempid","data/external/ring_health/provenance","research/experiments/ring_health/v2","deployment/ring_health"]
    for directory in directories:files.extend(p for p in (ROOT/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="release_manifest.json")
    names=["src/__init__.py","models/registry/ring_health_v2.json","models/configs/training/ring_health_evidence_v2.json","models/configs/training/ring_health_simulator_v1.json","requirements-ring-health.txt","requirements-ring-health-serving.txt",".dockerignore","docs/ring_health_v2_deployment_ru.md","docs/ring_health_v2.openapi.json","research/reports/ring_health_v2_review_ru.md","tools/train_ring_health_v2.py","tools/evaluate_ring_health_v2.py","tools/build_ring_health_v2_release.py","tests/unit/test_ring_health_v2.py","data/registry/ring_health_sources.json","data/registry/ring_health_downloads.json","data/samples/ring_health/fertility_demo_input.json","data/samples/ring_health/menopause_demo_input.json"]
    files.extend(ROOT/n for n in names)
    files.append(ROOT/"data/processed/ring_health/synthetic_stage_features.csv")
    files.append(ROOT/"data/processed/ring_health/DATASET.md")
    if (EXP/"deployment_verification.json").exists():files.append(EXP/"deployment_verification.json")
    release={"release":"ring-health-v2","clinical_validation":"not_established","python":platform.python_version(),"files":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}}
    manifest=EXP/"release_manifest.json";dump(manifest,release);files.append(manifest)
    output=ROOT/"output/ring_health_mvp_v2.zip";output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,"ring_health_mvp_v2/"+str(p.relative_to(ROOT)))
    print(json.dumps({"archive":str(output),"bytes":output.stat().st_size,"files":len(set(files)),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__=="__main__":main()
