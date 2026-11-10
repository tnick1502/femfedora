import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from emo_core.core.config import Settings
from emo_core.db.base import Base
from emo_core.db.models import Questionnaire
from emo_core.main import create_app

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "emo_core/questionnaires"
REGISTRY = [json.loads(p.read_text()) for p in REGISTRY_DIR.glob("*.json")]


@pytest.fixture
def config():
    """Создаёт тестовые настройки с явным mock и демо-авторизацией без чтения .env."""
    return Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite+aiosqlite://",
        ai_mode="mock",
        demo_auth_enabled=True,
        demo_jwt_secret="unit-test-only-demo-signing-secret-32-chars",
    )


@pytest_asyncio.fixture
async def sessions(tmp_path):
    """Создаёт тестовую схему и загружает встроенные шкалы, затем отдаёт фабрику сессий.

    TEST_DATABASE_URL должен указывать на отдельную расходуемую БД: перед тестом
    все ORM-таблицы удаляются. Без переменной используется временный файл SQLite.
    """
    url = os.getenv("TEST_DATABASE_URL") or f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine = create_async_engine(url)
    # TEST_DATABASE_URL must point to an expendable, isolated test database.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session, session.begin():
        for data in REGISTRY:
            session.add(Questionnaire(**data))
    yield factory
    await engine.dispose()


@pytest.fixture
def app(config, sessions):
    """Собирает приложение на тестовых настройках и фабрике сессий."""
    return create_app(config, sessions)


@pytest_asyncio.fixture
async def client(app):
    """Отдаёт HTTP-клиент к ASGI-приложению без открытия сетевого порта."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def auth(client):
    """Выпускает JWT двух независимых демо-профилей anna и maria."""
    result = {}
    for name in ("anna", "maria"):
        response = await client.post("/dev/token", json={"user": name})
        result[name] = {"Authorization": "Bearer " + response.json()["access_token"]}
    return result


async def conversation(client, auth, user="anna"):
    """Создаёт беседу выбранного тестового профиля и возвращает её UUID."""
    response = await client.post(
        "/api/v1/emo/conversations", headers=auth[user], json={"title": "Разговор"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def send(client, auth, cid, text, user="anna", client_message_id=None):
    """Отправляет реплику через API с заданным либо новым UUID идемпотентности."""
    return await client.post(
        f"/api/v1/emo/conversations/{cid}/messages",
        headers=auth[user],
        json={"content": text, "client_message_id": client_message_id or str(uuid4())},
    )


def synthetic_material(code="GAD2", version="test-v1"):
    """Создаёт искусственные пункты шкалы для проверки механизма подсчёта и импорта.

    Тексты, ключи и отметка approved относятся только к тестовой фикстуре;
    они не являются русской редакцией шкалы или клиническим одобрением.
    """
    count, maximum = {"GAD2": (2, 6), "GAD7": (7, 21), "EPDS": (10, 30), "PHQ9": (9, 27)}[code]
    return dict(
        code=code,
        version=version,
        language="ru",
        enabled=True,
        license_reference="test-fixture-only-not-clinical",
        definition={
            "clinical_review_status": "approved",
            "reviewed_by": "test-fixture",
            "approval_reference": "synthetic-unit-test-only",
            "items": [
                {
                    "id": str(i),
                    "text": f"Синтетический пункт {i}",
                    "options": ["A", "B", "C", "D"],
                    "score_map": [3, 2, 1, 0] if code == "EPDS" and i == 1 else [0, 1, 2, 3],
                }
                for i in range(count)
            ],
        },
        scoring_rules={
            "maximum": maximum,
            "safety_items": [{"item_id": str(count - 1), "minimum_score": 1}]
            if code in ("EPDS", "PHQ9")
            else [],
        },
    )


@pytest_asyncio.fixture
async def enabled(sessions):
    """Добавляет искусственную редакцию GAD-2 в тестовую БД и возвращает её UUID."""
    async with sessions() as session, session.begin():
        q = Questionnaire(**synthetic_material())
        session.add(q)
        await session.flush()
        return str(q.id)
