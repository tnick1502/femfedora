#!/bin/bash
# Сборка и запуск в Docker, затем прогон сценариев на реальных моделях.
# Результаты: ../tmp/live/ (каталог tmp/ исключён из Git). Секреты в файлы не пишутся.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=../tmp/live; mkdir -p "$OUT"
PORT=$(grep -E '^EMO_HTTP_PORT=' .env | cut -d= -f2); PORT=${PORT:-8086}
echo "1/4 Сборка и запуск..."
docker compose up --build -d > "$OUT/compose_up.log" 2>&1 || { echo "Compose не запустился: см. $OUT/compose_up.log"; exit 1; }
echo "2/4 Ожидание готовности API на порту ${PORT}..."
for i in $(seq 1 60); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
docker compose ps > "$OUT/ps.txt" 2>&1
curl -fsS "http://127.0.0.1:$PORT/health" > "$OUT/health.json" 2>&1 || echo "API не отвечает" | tee -a "$OUT/health.json"
echo "3/4 Калибровочный прогон классификатора (около 2 минут)..."
docker compose exec -T api python tools/live_check.py calibrate > "$OUT/calibration.json" 2> "$OUT/calibration.err"
echo "4/4 Сценарии чата (около 5 минут)..."
docker compose exec -T api python tools/live_check.py chat > "$OUT/chat.json" 2> "$OUT/chat.err"
docker compose logs --tail=300 api worker migrate > "$OUT/logs.txt" 2>&1
echo "Готово. Результаты: $OUT. Демо: http://127.0.0.1:$PORT/demo/"
