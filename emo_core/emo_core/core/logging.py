import logging


def configure_logging(level: str) -> None:
    """Задаёт уровень и формат журналов приложения.

    Подавляет сообщения HTTP-клиентов, SQLAlchemy и SDK ниже CRITICAL,
    чтобы их диагностический вывод не раскрывал URL и параметры запросов.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # External HTTP logs may contain sensitive URLs; application never logs payloads.
    for name in ("httpx", "httpcore", "httpx2", "httpcore2", "sqlalchemy.engine", "typesafe_sdk"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
