#!/bin/sh
set -eu
cd /opt/emo-core/source
# Перед обновлением сохраняется текущая БД; изменения в Git не затираются.
/opt/emo-core/ops/backup.sh
git pull --ff-only origin master
cd emo_core
docker compose -f docker-compose.yml -f /opt/emo-core/ops/compose.server.yml build api
docker compose -f docker-compose.yml -f /opt/emo-core/ops/compose.server.yml \
    up -d --no-build --wait --wait-timeout 180
