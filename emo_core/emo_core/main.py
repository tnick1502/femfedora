import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from emo_core.api import (
    checkins,
    complaints,
    conversations,
    models,
    proactive,
    questionnaires,
    settings,
    trends,
    voice,
)
from emo_core.audio.ingestion import VoiceService
from emo_core.core.auth import console_login, console_session, current_user, demo_token
from emo_core.core.config import get_settings
from emo_core.core.logging import configure_logging
from emo_core.db.base import utcnow
from emo_core.db.session import Session, engine
from emo_core.providers.factory import providers
from emo_core.schemas.user import StrictModel
from emo_core.services.policies import policy
from emo_core.services.proactive_engine import ProactiveMessagingEngine


class DemoLogin(StrictModel):
    user: str
    # session — короткий токен интерфейса; api — ключ для интеграции и скриптов.
    purpose: Literal["session", "api"] = "session"


class ConsoleLogin(StrictModel):
    login: str
    password: str


class WorkerRun(StrictModel):
    now: str | None = None


class JevProbe(StrictModel):
    text: str
    route: Literal["normal", "recommend_professional", "urgent"] = "normal"


def create_app(config=None, sessions=None, llm=None, jev=None):
    """Создаёт FastAPI-приложение с API, авторизацией, обработчиками ошибок и демо-клиентом.

    config, sessions, llm и jev позволяют передать зависимости в тестах. В production
    требует документированного одобрения safety policy; dev-маршруты подключает
    только в development/test. Соединение с БД проверяет при запуске lifespan.
    """
    config = config or get_settings()
    configure_logging(config.log_level)
    if config.app_env == "production":
        safety_policy = policy(config, "safety_rules.yaml")
        if not all(
            (
                safety_policy.get("clinical_review_status") == "approved",
                safety_policy.get("reviewed_by"),
                safety_policy.get("approval_reference"),
            )
        ):
            raise ValueError(
                "Production requires a documented approval of safety templates and routing policy"
            )

    @asynccontextmanager
    async def lifespan(app):
        """Проверяет соединение с БД при старте и освобождает собственный пул при завершении."""
        async with app.state.sessions() as session:
            await session.execute(text("SELECT 1"))
        yield
        if sessions is None:
            await engine.dispose()

    app = FastAPI(
        title="EMO Core",
        version="0.1.0",
        lifespan=lifespan,
        description="Многопользовательская эмоциональная поддержка. Не медицинская диагностика.",
    )
    app.state.settings = config
    app.state.sessions = sessions or Session
    default_llm, default_jev = providers(config)
    app.state.llm, app.state.jev = llm or default_llm, jev or default_jev
    app.state.jwks, app.state.jwks_lock = {}, asyncio.Lock()
    app.state.login_failures = {}
    app.state.voice = VoiceService(config)
    app.state.chat_slots = asyncio.Semaphore(config.chat_concurrency)
    router = APIRouter(prefix="/api/v1/emo")
    for module in (
        conversations,
        models,
        questionnaires,
        checkins,
        complaints,
        trends,
        settings,
        proactive,
        voice,
    ):
        router.include_router(module.router)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # FastAPI's default errors echo invalid input (potential medical text).
        """Возвращает HTTP 422 с местом и типом ошибки, исключая исходные значения запроса."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in error.errors()
                ]
            },
        )

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        # Avoid propagating SQL parameters / provider bodies into logs or responses.
        """Возвращает общий HTTP 500 без текста исключения и параметров внешних систем."""
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса"})

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Добавляет запрет кэширования и защитные заголовки; для Swagger разрешает его ресурсы."""
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500, content={"detail": "Внутренняя ошибка сервиса"}
            )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(self), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'"
        )
        if request.url.path in ("/docs", "/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
            )
        return response

    @app.get("/health")
    async def health():
        """Проверяет доступность БД запросом SELECT 1 и возвращает status=ok."""
        async with app.state.sessions() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/demo/config")
    async def demo_config():
        """Возвращает режим AI и признаки доступности входа и демо-авторизации без секретов."""
        return {
            "demo_auth": config.demo_auth_enabled and config.app_env != "production",
            "ai_mode": config.ai_mode,
            "clinical_materials_require_approval": True,
            "login_required": config.login_required,
        }

    @app.post("/auth/login")
    async def auth_login(body: ConsoleLogin, request: Request):
        """Проверяет логин и пароль консоли, возвращает токен её сессии."""
        return console_login(request, body.login, body.password)

    if config.app_env in ("development", "test"):

        @app.post("/dev/token", dependencies=[Depends(console_session)])
        async def login(body: DemoLogin):
            """Выпускает JWT демо-профиля: на два часа для сессии либо api_key_ttl_days для API."""
            ttl = (
                timedelta(days=config.api_key_ttl_days)
                if body.purpose == "api"
                else timedelta(hours=2)
            )
            return {
                "access_token": demo_token(body.user, config, ttl),
                "token_type": "bearer",
                "expires_at": (utcnow() + ttl).isoformat(),
                "purpose": body.purpose,
            }

        async def live_jev():
            """Выбирает текущий глобальный профиль Jev либо провайдера из зависимостей приложения."""
            from emo_core.services.model_registry import ModelRegistry

            async with app.state.sessions() as session, session.begin():
                snapshot = await ModelRegistry(config).resolve(session)
            return providers(snapshot.settings)[1] if snapshot.overridden else app.state.jev

        @app.post("/dev/jev/analyze", dependencies=[Depends(console_session)])
        async def probe_input(body: JevProbe):
            """Возвращает маркеры и намерение Jev для синтетического текста в dev-калибровке."""
            analysis = await (await live_jev()).analyze(body.text)
            return {"markers": analysis.markers, "intent": analysis.intent}

        @app.post("/dev/jev/check-output", dependencies=[Depends(console_session)])
        async def probe_output(body: JevProbe):
            """Возвращает оценки Jev и результат программных проверок синтетического ответа."""
            from emo_core.services.safety_guard import SafetyGuard

            jev = await live_jev()
            guard = SafetyGuard(config, jev)
            return {
                "scores": await jev.check_output(body.text, body.route),
                "deterministic_passed": guard.deterministic_output(body.text),
                "unverified_contact": guard.unverified_contacts(body.text),
            }

        @app.post("/dev/worker/run")
        async def worker(body: WorkerRun, user_id: UUID = Depends(current_user)):
            """Запускает один цикл инициативных сообщений для JWT-профиля на указанное aware-время."""
            from datetime import datetime

            try:
                now = datetime.fromisoformat(body.now) if body.now else utcnow()
                if now.tzinfo is None:
                    raise ValueError("timezone")
            except ValueError:
                raise HTTPException(422, "Ожидается ISO timestamp с часовым поясом") from None
            return await ProactiveMessagingEngine(config, app.state.sessions).run_once(
                now, only_user=user_id
            )

    demo = Path(__file__).resolve().parents[1] / "demo"
    app.mount("/demo", StaticFiles(directory=demo, html=True), name="demo")

    @app.get("/", include_in_schema=False)
    async def home():
        """Перенаправляет корневой URL на демо-клиент /demo/."""
        return RedirectResponse("/demo/")

    return app


app = create_app()
