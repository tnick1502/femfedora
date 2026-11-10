import json
from datetime import timedelta

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from emo_core.core.auth import DEMO_USERS
from emo_core.db.base import utcnow


async def test_jwks_auth_and_claims(client, auth, app, monkeypatch):
    """Проверяет загрузку JWKS, подпись JWT и обязательные issuer/audience/expiry."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update(kid="test-key", use="sig")
    original_client = httpx.AsyncClient

    def handler(request):
        """Возвращает тестовый публичный ключ только по ожидаемому URL JWKS."""
        assert str(request.url) == "https://id.test/jwks"
        return httpx.Response(200, json={"keys": [jwk]})

    def patched_client(*args, **kwargs):
        """Подменяет HTTP-транспорт загрузчика JWKS локальным обработчиком."""
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    app.state.settings.jwt_jwks_url = "https://id.test/jwks"
    claims = {
        "sub": str(DEMO_USERS["anna"]),
        "iat": utcnow(),
        "exp": utcnow() + timedelta(hours=1),
        "iss": app.state.settings.jwt_issuer,
        "aud": app.state.settings.jwt_audience,
    }
    token = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})
    assert (
        await client.get("/api/v1/emo/settings", headers={"Authorization": "Bearer " + token})
    ).status_code == 200
    for changes in (
        {"aud": "wrong"},
        {"iss": "wrong"},
        {"exp": utcnow() - timedelta(hours=1)},
        {"sub": "not-a-uuid"},
    ):
        bad = jwt.encode(
            {**claims, **changes}, private, algorithm="RS256", headers={"kid": "test-key"}
        )
        assert (
            await client.get("/api/v1/emo/settings", headers={"Authorization": "Bearer " + bad})
        ).status_code == 401


async def test_demo_disabled_cannot_mint(client, app):
    """Проверяет отказ в выпуске демо-токена при отключённой демо-авторизации."""
    app.state.settings.demo_auth_enabled = False
    assert (await client.post("/dev/token", json={"user": "anna"})).status_code == 404
