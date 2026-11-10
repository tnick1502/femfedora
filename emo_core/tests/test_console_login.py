import jwt
import pytest
from pydantic import ValidationError

from emo_core.core.config import Settings
from emo_core.main import create_app

PASSWORD = "Unit-test-only*&^%password"


@pytest.fixture
def config():
    """Включает вход в консоль на тестовом логине и пароле."""
    return Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite+aiosqlite://",
        ai_mode="mock",
        demo_auth_enabled=True,
        demo_jwt_secret="unit-test-only-demo-signing-secret-32-chars",
        app_login="admin",
        app_password=PASSWORD,
    )


async def gate(client, password=PASSWORD, login="admin"):
    """Отправляет запрос входа с тестовым либо переданным паролем."""
    return await client.post("/auth/login", json={"login": login, "password": password})


async def test_nothing_is_issued_without_login(client):
    """Проверяет защиту выпуска токенов и администрирования обязательным входом."""
    assert (await client.get("/demo/config")).json()["login_required"] is True
    assert (await client.post("/dev/token", json={"user": "anna"})).status_code == 401
    assert (await gate(client, "wrong-password")).status_code == 401
    assert (await gate(client, PASSWORD, "someone")).status_code == 401
    assert (await client.get("/api/v1/emo/admin/models")).status_code == 403


async def test_login_opens_profiles_admin_and_api_key(client, config):
    """Проверяет доступ сессии консоли и различие сроков пользовательского JWT и API-ключа."""
    session = (await gate(client)).json()
    headers = {"Authorization": "Bearer " + session["access_token"]}
    user = (await client.post("/dev/token", headers=headers, json={"user": "anna"})).json()
    api = (
        await client.post("/dev/token", headers=headers, json={"user": "anna", "purpose": "api"})
    ).json()
    assert api["purpose"] == "api" and api["expires_at"] > user["expires_at"]
    for token in (user["access_token"], api["access_token"]):
        response = await client.get(
            "/api/v1/emo/settings", headers={"Authorization": "Bearer " + token}
        )
        assert response.status_code == 200
    claims = jwt.decode(api["access_token"], options={"verify_signature": False})
    assert (claims["exp"] - claims["iat"]) == config.api_key_ttl_days * 86400
    # Сессия консоли открывает администрирование, но не является пользовательским токеном.
    admin = await client.get(
        "/api/v1/emo/admin/models", headers={"X-Admin-Token": session["access_token"]}
    )
    assert admin.status_code == 200
    assert (await client.get("/api/v1/emo/settings", headers=headers)).status_code == 401
    # Пользовательский токен не открывает ни консоль, ни администрирование.
    as_user = {"Authorization": "Bearer " + user["access_token"]}
    assert (
        await client.post("/dev/token", headers=as_user, json={"user": "maria"})
    ).status_code == 401
    assert (
        await client.get(
            "/api/v1/emo/admin/models", headers={"X-Admin-Token": user["access_token"]}
        )
    ).status_code == 403


async def test_repeated_failures_are_throttled(client):
    """Проверяет временную блокировку входа после серии неверных паролей."""
    for _ in range(5):
        assert (await gate(client, "wrong-password")).status_code == 401
    assert (await gate(client)).status_code == 429


def test_password_never_in_openapi_or_repr(config):
    """Проверяет отсутствие пароля в OpenAPI и строковом представлении настроек."""
    assert PASSWORD not in repr(config)
    assert PASSWORD not in str(create_app(config).openapi())


@pytest.mark.parametrize(
    "kwargs", [{"app_login": "admin"}, {"app_login": "admin", "app_password": "short"}]
)
def test_incomplete_login_configuration_fails(kwargs):
    """Проверяет отказ при неполной или слишком короткой паре логин/пароль."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, demo_jwt_secret="x" * 40, **kwargs)
