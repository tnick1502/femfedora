import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from emo_core.core.auth import console_token_valid

admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def current_admin(request: Request, token: str | None = Depends(admin_header)) -> None:
    """Проверяет X-Admin-Token: секрет администратора или JWT сессии консоли.

    Обычный пользовательский JWT административных прав не даёт. Если вход
    администратора не настроен, возвращает 404; неверный секрет — 403.
    """
    settings = request.app.state.settings
    expected = settings.admin_api_token
    # Сессия консоли (вход по логину и паролю) даёт те же права, что и ADMIN_API_TOKEN.
    if console_token_valid(settings, token):
        return
    if not expected and not settings.login_required:
        raise HTTPException(404, "Администрирование не настроено")
    if not token or not expected or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(403, "Требуется вход администратора")


async def admin_session(request: Request, _: None = Depends(current_admin)):
    """Открывает транзакцию БД после проверки административного доступа.

    Передаёт сессию обработчику через yield. Успешное завершение фиксирует
    изменения, исключение откатывает транзакцию.
    """
    async with request.app.state.sessions() as session, session.begin():
        yield session
