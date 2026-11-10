#!/bin/bash
# Сборка и запуск в режиме разработки: правки кода подхватываются без пересборки.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=../tmp/live; mkdir -p "$OUT"
PORT=$(grep -E '^EMO_HTTP_PORT=' .env | cut -d= -f2); PORT=${PORT:-8086}
echo "Сборка и запуск..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d > "$OUT/compose_up.log" 2>&1 \
  || { echo "Compose не запустился: см. $OUT/compose_up.log"; tail -20 "$OUT/compose_up.log"; exit 1; }
for i in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 2; done
docker compose ps > "$OUT/ps.txt" 2>&1
docker compose logs --tail=60 api migrate > "$OUT/logs.txt" 2>&1
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Готово. Демо: http://127.0.0.1:${PORT}/demo/"
else
  echo "API не отвечает: см. $OUT/logs.txt"; tail -30 "$OUT/logs.txt"; exit 1
fi
