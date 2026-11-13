#!/bin/sh
set -eu
umask 077
backup_dir=/opt/emo-core/backups
mkdir -p "$backup_dir"
backup_file="$backup_dir/emo-$(date -u +%Y%m%dT%H%M%SZ).dump"
cd /opt/emo-core/source/emo_core
# Секреты остаются в окружении контейнера; pg_dump использует локальное соединение.
docker compose -f docker-compose.yml -f /opt/emo-core/ops/compose.server.yml \
    exec -T db sh -c 'exec pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom' \
    > "$backup_file.tmp"
mv "$backup_file.tmp" "$backup_file"
# Удаляются только собственные резервные копии этого задания старше семи суток.
find "$backup_dir" -maxdepth 1 -type f -name 'emo-*.dump' -mtime +7 -delete
