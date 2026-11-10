import os
import secrets
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from cryptography.fernet import Fernet
from dotenv import dotenv_values


def init_env(directory: Path) -> None:
    """Создаёт .env в directory и заполняет отсутствующие локальные секреты.

    За основу берёт .env.example. Существующие непустые значения сохраняет;
    пароль из готового DATABASE_URL использует при заполнении POSTGRES_PASSWORD.
    При отсутствии DSN формирует адрес Compose-хоста db. Записывает файл с
    правами 0600; значения секретов не выводит.
    """
    target = directory / ".env"
    if not target.exists():
        target.write_text((directory / ".env.example").read_text())
    current = dotenv_values(target)
    generated = {
        "DEMO_JWT_SECRET": secrets.token_urlsafe(48),
        "ADMIN_API_TOKEN": secrets.token_urlsafe(48),
        "MODEL_SECRETS_KEY": Fernet.generate_key().decode(),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
    }
    if current.get("DATABASE_URL"):
        existing_dsn = urlparse(current["DATABASE_URL"])
        if existing_dsn.password:
            generated["POSTGRES_PASSWORD"] = unquote(existing_dsn.password)
    additions = {}
    for key, value in generated.items():
        if not current.get(key):
            additions[key] = value
    merged = {**current, **additions}
    if not current.get("DATABASE_URL"):
        additions["DATABASE_URL"] = (
            f"postgresql+asyncpg://{quote(merged.get('POSTGRES_USER') or 'emo', safe='')}:"
            f"{quote(merged['POSTGRES_PASSWORD'], safe='')}@db:5432/{merged.get('POSTGRES_DB') or 'emo'}"
        )
    lines = target.read_text().splitlines()
    for key, value in additions.items():
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith(key + "="):
                lines[index] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
    target.write_text("\n".join(lines) + "\n")
    os.chmod(target, 0o600)
