# Ring Health v2: запуск в приложении

Это исследовательский backend с обученными моделями. Клиническая точность на пользователях кольца не установлена. Ответ содержит clinical_validation=not_established; клиент должен сохранять эту информацию и различать календарную оценку и температурное свидетельство.

Из корня распакованного архива:

```bash
export RING_HEALTH_API_KEY="$(openssl rand -hex 32)"
docker compose -f deployment/ring_health/compose.yaml up --build -d
curl --fail http://127.0.0.1:8788/health/ready
curl --fail http://127.0.0.1:8788/v2/analyze \
  -H "Authorization: Bearer $RING_HEALTH_API_KEY" \
  -H 'Content-Type: application/json' \
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
