import hmac
import time
from datetime import timedelta
from uuid import UUID

import httpx
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from emo_core.db.base import utcnow

bearer = HTTPBearer(auto_error=False)
DEMO_USERS = {
    "anna": UUID("11111111-1111-4111-8111-111111111111"),
    "maria": UUID("22222222-2222-4222-8222-222222222222"),
}


CONSOLE_ISSUER = "emo-console"
LOGIN_WINDOW_SECONDS, LOGIN_MAX_FAILURES = 300, 5


def console_login(request: Request, login: str, password: str) -> dict:
    """Проверяет логин и пароль консоли и выпускает JWT её сессии.

    После пяти ошибок с одного адреса за 300 секунд возвращает HTTP 429.
    Срок токена задаёт console_session_hours; возвращаются токен, тип и время
    истечения в ISO-формате. Неверная пара даёт 401, отключённый вход — 404.
    Счётчик попыток хранится в памяти текущего процесса API.
    """
    settings = request.app.state.settings
    if not settings.login_required:
        raise HTTPException(404, "Вход по паролю не настроен")
    failures = request.app.state.login_failures
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = [t for t in failures.get(client, []) if t > now - LOGIN_WINDOW_SECONDS]
    if len(recent) >= LOGIN_MAX_FAILURES:
        raise HTTPException(429, "Слишком много попыток входа. Повторите позже.")
    # Сравнение за постоянное время; оба поля проверяются всегда.
    valid = hmac.compare_digest(login.encode(), settings.app_login.encode()) & hmac.compare_digest(
        password.encode(), settings.app_password.encode()
    )
    if not valid:
        failures[client] = [*recent, now]
        raise HTTPException(401, "Неверный логин или пароль")
    failures.pop(client, None)
    issued = utcnow()
    expires = issued + timedelta(hours=settings.console_session_hours)
    token = jwt.encode(
        {
            "sub": "console",
            "iss": CONSOLE_ISSUER,
            "aud": settings.jwt_audience,
            "iat": issued,
            "exp": expires,
        },
        settings.demo_jwt_secret,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat()}


def console_token_valid(settings, token: str | None) -> bool:
    """Проверяет подпись, издателя, аудиторию и срок JWT консоли.

    Возвращает False для отсутствующего или недействительного токена,
    а также если вход по паролю не настроен. Исключения JWT наружу не передаёт.
    """
    if not token or not settings.login_required:
        return False
    try:
        jwt.decode(
            token,
            settings.demo_jwt_secret,
            algorithms=["HS256"],
            issuer=CONSOLE_ISSUER,
            audience=settings.jwt_audience,
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
        return True
    except jwt.PyJWTError:
        return False


async def console_session(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
) -> None:
    """Требует действительную сессию консоли, если включён вход по паролю.

    FastAPI использует эту зависимость для демо-операций. При отсутствии
    настроенной пары логин/пароль пропускает запрос; иначе отказ возвращает 401.
    """
    settings = request.app.state.settings
    if settings.login_required and not console_token_valid(
        settings, credentials.credentials if credentials else None
    ):
        raise HTTPException(401, "Требуется вход", headers={"WWW-Authenticate": "Bearer"})


def demo_token(name, settings, ttl: timedelta = timedelta(hours=2)):
    """Выпускает HS256-токен одного из двух демонстрационных профилей.

    name — anna или maria; ttl задаёт срок действия, по умолчанию два часа.
    При выключенной демо-авторизации, неизвестном имени или в production
    возвращает HTTP 404. Токен консоли и пользовательский токен имеют разных издателей.
    """
    if not settings.demo_auth_enabled or settings.app_env == "production" or name not in DEMO_USERS:
        raise HTTPException(404, "Не найдено")
    now = utcnow()
    return jwt.encode(
        {
            "sub": str(DEMO_USERS[name]),
            "iss": "emo-demo",
            "aud": settings.jwt_audience,
            "iat": now,
            "exp": now + ttl,
        },
        settings.demo_jwt_secret,
        algorithm="HS256",
    )


async def current_user(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
) -> UUID:
    """Проверяет Bearer JWT и возвращает UUID пользователя из sub.

    В development/test допускает HS256 только для фиксированных демо-профилей.
    Основные токены проверяет по RS256 и JWKS из серверных настроек; кеш ключей
    действует десять минут и обновляется при неизвестном kid. Ошибки подписи,
    полей токена и получения ключей превращает в HTTP 401 без раскрытия токена.
    """
    if not credentials:
        raise HTTPException(401, "Требуется Bearer token", headers={"WWW-Authenticate": "Bearer"})
    settings = request.app.state.settings
    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
        if (
            header.get("alg") == "HS256"
            and settings.demo_auth_enabled
            and settings.app_env != "production"
        ):
            payload = jwt.decode(
                token,
                settings.demo_jwt_secret,
                algorithms=["HS256"],
                issuer="emo-demo",
                audience=settings.jwt_audience,
                options={"require": ["sub", "exp", "iat", "iss", "aud"]},
            )
            user_id = UUID(payload["sub"])
            if user_id not in DEMO_USERS.values():
                raise ValueError("Unknown demo subject")
            return user_id
        if header.get("alg") != "RS256" or not settings.jwt_jwks_url:
            raise ValueError("Unsupported authentication")
        # Bounded async JWKS fetch; one refresh for unknown kid. Never use token-supplied URLs.
        cache = request.app.state.jwks
        if cache.get("expires", utcnow()) <= utcnow() or header.get("kid") not in cache.get(
            "keys", {}
        ):
            async with request.app.state.jwks_lock:
                async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
                    result = await client.get(settings.jwt_jwks_url)
                    result.raise_for_status()
                    keys = result.json()["keys"]
                cache.update(
                    keys={
                        k["kid"]: k
                        for k in keys
                        if k.get("kty") == "RSA" and k.get("use", "sig") == "sig"
                    },
                    expires=utcnow() + timedelta(minutes=10),
                )
        key = jwt.PyJWK.from_dict(cache["keys"][header["kid"]]).key
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
        return UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError, TypeError, httpx.HTTPError):
        raise HTTPException(
            401, "Недействительный токен", headers={"WWW-Authenticate": "Bearer"}
        ) from None
