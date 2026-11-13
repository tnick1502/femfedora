import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from emo_core.core.config import get_settings
from emo_core.db.models import Questionnaire
from emo_core.services.retention import prune
from emo_core.services.screening_engine import validate_material


def main():
    """Разбирает команду CLI: подготовка .env, проверка конфигурации, импорт редакции или очистка данных."""
    parser = argparse.ArgumentParser(
        description="EMO Core: администрирование локальной конфигурации"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser(
        "import-questionnaire", help="Импортировать проверенную русскую редакцию"
    )
    importer.add_argument("path", type=Path)
    sub.add_parser("init-env", help="Создать .env и заполнить недостающие локальные секреты")
    sub.add_parser("check-config")
    sub.add_parser("prune", help="Удалить данные старше RETENTION_DAYS")
    args = parser.parse_args()
    if args.command == "init-env":
        from emo_core.core.env_setup import init_env

        init_env(Path.cwd())
        print(".env готов; существующие значения сохранены, секреты не выводятся")
        return
    if args.command == "check-config":
        settings = get_settings()
        print(
            f"Configuration valid; env={settings.app_env}; AI={settings.ai_mode}; secrets omitted"
        )
        return
    data = None
    if args.command == "import-questionnaire":
        data = validate_material(json.loads(args.path.read_text()))
    asyncio.run(execute(args.command, data))


async def execute(command, data):
    """Выполняет очистку по сроку хранения либо импорт новой неизменяемой редакции шкалы.

    При импорте отключает предыдущие активные версии того же кода и языка в одной
    транзакции. Совпадение версии вызывает ValueError. Пул БД освобождает при выходе.
    """
    from emo_core.db.session import Session, engine

    try:
        if command == "prune":
            settings = get_settings()
            print(
                f"Removed records: {await prune(Session, settings.retention_days, prompt_days=settings.prompt_trace_retention_days)}"
            )
        else:
            async with Session() as session, session.begin():
                existing = await session.scalar(
                    select(Questionnaire).where(
                        Questionnaire.code == data["code"],
                        Questionnaire.version == data["version"],
                        Questionnaire.language == data["language"],
                    )
                )
                if existing:
                    raise ValueError("Версия неизменяема. Для исправлений создайте новую версию.")
                previous = await session.scalars(
                    select(Questionnaire).where(
                        Questionnaire.code == data["code"],
                        Questionnaire.language == data["language"],
                        Questionnaire.enabled.is_(True),
                    )
                )
                for questionnaire in previous:
                    questionnaire.enabled = False
                session.add(Questionnaire(**{**data, "enabled": True}))
            print("Approved questionnaire imported")
    finally:
        await engine.dispose()
