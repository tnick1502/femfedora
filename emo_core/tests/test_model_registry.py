import json
from uuid import UUID

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from emo_core.core.env_setup import init_env
from emo_core.db.models import ModelProfile
from emo_core.main import create_app
from emo_core.services.model_registry import ModelRegistry
from tests.conftest import conversation, send

P = "/api/v1/emo"
ADMIN_TOKEN = "unit-test-admin-token-with-more-than-32-characters"
API_KEY = "provider-test-secret-key-never-exposed"


@pytest.fixture
def admin(app):
    """Настраивает тестовый админ-доступ, ключ шифрования и разрешённые хосты провайдеров."""
    app.state.settings.admin_api_token = ADMIN_TOKEN
    app.state.settings.model_secrets_key = Fernet.generate_key().decode()
    app.state.settings.allowed_llm_hosts = "llm.test"
    app.state.settings.allowed_jev_hosts = "api.typesafe.ai"
    return {"X-Admin-Token": ADMIN_TOKEN}


def profile(**changes):
    """Создаёт тело тестового LLM-профиля с возможностью переопределить поля."""
    return {
        "kind": "llm",
        "name": "Test provider",
        "model": "example-v1",
        "base_url": "https://llm.test/v1",
        "api_key": API_KEY,
        "timeout_seconds": 15,
        **changes,
    }


async def test_admin_required_not_just_user_jwt(client, auth, admin):
    """Проверяет, что пользовательский JWT не даёт права менять глобальные модели."""
    for headers in ({}, auth["anna"], {"X-Admin-Token": "wrong"}):
        assert (await client.get(P + "/admin/models", headers=headers)).status_code == 403
        assert (
            await client.post(P + "/admin/models", headers=headers, json=profile())
        ).status_code == 403
        assert (
            await client.patch(
                P + "/admin/model-runtime",
                headers=headers,
                json={"ai_mode": "disabled", "expected_revision": 0},
            )
        ).status_code == 403
    assert (await client.get(P + "/admin/models", headers=admin)).status_code == 200


async def test_create_edit_encryption_and_no_secret_return(client, admin, sessions, app, caplog):
    """Проверяет создание и правку профиля, шифрование ключа и отсутствие секрета в API."""
    response = await client.post(P + "/admin/models", headers=admin, json=profile())
    assert response.status_code == 201, response.text
    pid = response.json()["id"]
    assert API_KEY not in response.text
    async with sessions() as s:
        row = await s.get(ModelProfile, UUID(pid))
        assert row.encrypted_api_key and API_KEY not in row.encrypted_api_key
        assert ModelRegistry(app.state.settings).key(row) == API_KEY
    response = await client.patch(
        P + "/admin/models/" + pid, headers=admin, json={"model": "example-v2"}
    )
    assert response.status_code == 200 and response.json()["model"] == "example-v2"
    async with sessions() as s:
        row = await s.get(ModelProfile, UUID(pid))
        assert ModelRegistry(app.state.settings).key(row) == API_KEY
    for path in ("/admin/models", "/admin/model-events"):
        response = await client.get(P + path, headers=admin)
        assert API_KEY not in response.text and "encrypted_api_key" not in response.text
    assert API_KEY not in caplog.text
    assert API_KEY not in repr(app.state.settings)


async def test_env_credential_reference(client, admin, app, sessions):
    """Проверяет хранение ссылки на ключ окружения вместо зашифрованного значения в БД."""
    app.state.settings.llm_api_key = API_KEY
    data = profile(credential_ref="llm_default")
    data.pop("api_key")
    response = await client.post(P + "/admin/models", headers=admin, json=data)
    assert response.status_code == 201, response.text
    async with sessions() as s:
        row = await s.scalar(select(ModelProfile))
        assert row.encrypted_api_key is None and row.credential_ref == "llm_default"
        assert ModelRegistry(app.state.settings).key(row) == API_KEY


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "http://llm.test/v1"},
        {"base_url": "https://evil.test/v1"},
        {"base_url": "https://llm.test/v1?api_key=secret"},
        {"base_url": "https://token@llm.test/v1"},
        {"kind": "jev", "model": "jev-latest", "base_url": "https://api.typesafe.ai"},
    ],
)
async def test_reject_unapproved_endpoints_and_floating_jev(client, admin, changes):
    """Проверяет отказ для неразрешённых хостов и неприкреплённой версии Jev."""
    assert (
        await client.post(P + "/admin/models", headers=admin, json=profile(**changes))
    ).status_code == 422


async def test_global_switch_persists_across_app_instances(
    client, auth, admin, app, sessions, config
):
    """Проверяет сохранение глобального выбора моделей между экземплярами приложения."""
    pid = (await client.post(P + "/admin/models", headers=admin, json=profile())).json()["id"]
    current = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]
    response = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={"llm_profile_id": pid, "expected_revision": current["revision"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["llm"]["profile_id"] == pid
    for user in ("anna", "maria"):
        data = (await client.get(P + "/models", headers=auth[user])).json()
        assert data["llm"]["configured_model"] == "example-v1" and data["llm"]["profile_id"] == pid
    # A fresh API instance reads the same shared DB, rather than process-local state.
    other = create_app(config, sessions)
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=other), base_url="http://test"
    ) as second:
        data = (await second.get(P + "/models", headers=auth["maria"])).json()
        assert data["llm"]["profile_id"] == pid
    cid = await conversation(client, auth)
    answer = (await send(client, auth, cid, "Привет")).json()["assistant_message"]
    assert (
        answer["metadata"]["decision_report"]["models"]["llm"]["configured_model"] == "example-v1"
    )
    assert (
        answer["metadata"]["decision_report"]["models"]["llm"]["model"]
        == "deterministic-support-v1"
    )


async def test_conflict_and_invalid_live_update_rollback(client, admin):
    """Проверяет конфликт ревизии и откат некорректного перехода в live."""
    current = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]["revision"]
    assert (
        await client.patch(
            P + "/admin/model-runtime",
            headers=admin,
            json={"ai_mode": "disabled", "expected_revision": current + 1},
        )
    ).status_code == 409
    assert (
        await client.patch(
            P + "/admin/model-runtime",
            headers=admin,
            json={"ai_mode": "live", "expected_revision": current},
        )
    ).status_code == 409
    state = (await client.get(P + "/admin/models", headers=admin)).json()
    assert state["runtime"]["ai_mode"] == "mock" and state["runtime"]["revision"] == current
    assert (await client.get(P + "/admin/model-events", headers=admin)).json()["items"] == []


async def test_turn_off_globally_without_restart(client, auth, admin):
    """Проверяет отключение моделей для новых сообщений без перезапуска приложения."""
    current = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]["revision"]
    result = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={"ai_mode": "disabled", "expected_revision": current},
    )
    assert result.status_code == 200, result.text
    for user in ("anna", "maria"):
        cid = await conversation(client, auth, user)
        response = await send(client, auth, cid, "Привет", user)
        report = response.json()["assistant_message"]["metadata"]["decision_report"]
        assert report["models"]["ai_mode"] == "disabled" and report["result"]["llm_calls"] == 0


async def test_missing_encryption_key_rejects_inline_key(client, admin, app):
    """Проверяет отказ сохранить явный API-ключ без MODEL_SECRETS_KEY."""
    app.state.settings.model_secrets_key = ""
    result = await client.post(P + "/admin/models", headers=admin, json=profile())
    assert result.status_code == 422 and "MODEL_SECRETS_KEY" in result.text


def test_init_env_no_overwrite_and_no_secret_output(tmp_path, capsys):
    """Проверяет генерацию секретов .env и сохранение уже заполненных значений."""
    (tmp_path / ".env.example").write_text(
        "DEMO_JWT_SECRET=\nADMIN_API_TOKEN=\nMODEL_SECRETS_KEY=\nPOSTGRES_PASSWORD=\nDATABASE_URL=\n"
    )
    init_env(tmp_path)
    original = (tmp_path / ".env").read_text()
    from dotenv import dotenv_values

    values = dotenv_values(tmp_path / ".env")
    assert len(values["ADMIN_API_TOKEN"]) >= 32
    Fernet(values["MODEL_SECRETS_KEY"].encode())
    assert values["POSTGRES_PASSWORD"] in values["DATABASE_URL"]
    init_env(tmp_path)
    assert (tmp_path / ".env").read_text() == original
    assert values["ADMIN_API_TOKEN"] not in capsys.readouterr().out


async def test_live_switch_changes_calls_for_both_users_and_keeps_past_reports(
    client, auth, admin, app, monkeypatch
):
    """Проверяет общий выбор модели для двух профилей и неизменность старых отчётов."""
    from types import SimpleNamespace

    import httpx

    from emo_core.providers.jev_provider import TypeSafeJevProvider

    app.state.settings.external_ai_allowed = True
    app.state.settings.ai_processing_region = "test-region"
    llm = (await client.post(P + "/admin/models", headers=admin, json=profile())).json()["id"]
    jev = (
        await client.post(
            P + "/admin/models",
            headers=admin,
            json=profile(kind="jev", model="jev-1.14.0", base_url="https://api.typesafe.ai"),
        )
    ).json()["id"]
    runtime = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]
    response = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={
            "llm_profile_id": llm,
            "jev_profile_id": jev,
            "ai_mode": "live",
            "expected_revision": runtime["revision"],
        },
    )
    assert response.status_code == 200, response.text
    original = httpx.AsyncClient
    calls = []
    rotated = "rotated-provider-key-test-only"

    def handler(request):
        """Записывает HTTP-вызовы тестовой LLM и возвращает ответ совместимого API."""
        data = json.loads(request.content)
        calls.append((data["model"], request.headers["Authorization"]))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Спасибо, что поделилась. Что сейчас хотелось бы обсудить?"
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(**kw, transport=httpx.MockTransport(handler))
    )

    async def ask(self, state, questions):
        """Возвращает оценки подменённого TypeSafe SDK для сценария live-переключения."""
        return SimpleNamespace(
            nouls={key: SimpleNamespace(noul=0.1) for key in questions if key != "intent"},
            choices={"intent": SimpleNamespace(choice="support")},
        )

    monkeypatch.setattr(TypeSafeJevProvider, "ask", ask)
    past = []
    for user in ("anna", "maria"):
        cid = await conversation(client, auth, user)
        await client.patch(P + "/settings", headers=auth[user], json={"external_ai_consent": True})
        response = await send(client, auth, cid, "Привет", user)
        assert response.status_code == 200, response.text
        answer = response.json()["assistant_message"]
        past.append((cid, answer["id"], user, answer["metadata"]["decision_report"]))
    assert calls == [("example-v1", "Bearer " + API_KEY)] * 2
    response = await client.patch(
        P + "/admin/models/" + llm, headers=admin, json={"model": "example-v2", "api_key": rotated}
    )
    assert response.status_code == 200, response.text
    for user in ("anna", "maria"):
        cid = await conversation(client, auth, user)
        response = await send(client, auth, cid, "Продолжим", user)
        assert response.status_code == 200, response.text
        assert (
            response.json()["assistant_message"]["metadata"]["decision_report"]["models"]["llm"][
                "model"
            ]
            == "example-v2"
        )
    assert calls[-2:] == [("example-v2", "Bearer " + rotated)] * 2
    for cid, mid, user, report in past:
        stored = (
            await client.get(
                f"{P}/conversations/{cid}/messages/{mid}/decision-report", headers=auth[user]
            )
        ).json()
        assert stored == report and stored["models"]["llm"]["model"] == "example-v1"
        assert API_KEY not in str(stored) and rotated not in str(stored)


async def test_no_external_calls_without_individual_consent(client, auth, admin, app):
    """Проверяет отсутствие внешних вызовов без личного согласия при глобальном live."""
    app.state.settings.external_ai_allowed = True
    app.state.settings.ai_processing_region = "test-region"
    app.state.settings.llm_api_key = API_KEY
    app.state.settings.typesafe_api_key = API_KEY
    app.state.settings.llm_model = "example-v1"
    app.state.settings.llm_base_url = "https://llm.test/v1"
    revision = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]["revision"]
    result = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={"ai_mode": "live", "expected_revision": revision},
    )
    assert result.status_code == 200, result.text
    cid = await conversation(client, auth)
    report = (await send(client, auth, cid, "Привет")).json()["assistant_message"]["metadata"][
        "decision_report"
    ]
    stage = next(s for s in report["stages"] if s["stage"] == "jev_analysis")
    assert stage["reasons"] == ["external_ai_consent_missing"]
    assert report["result"]["llm_calls"] == 0 and report["result"]["jev_input_called"] is False


@pytest.mark.postgres
async def test_concurrent_global_runtime_updates_one_winner(client, admin):
    """Проверяет один успешный выбор модели при конкурентной правке одной ревизии."""
    import asyncio
    import os

    if not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"):
        pytest.skip("Global row locking requires PostgreSQL")
    revision = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]["revision"]
    results = await asyncio.gather(
        *(
            client.patch(
                P + "/admin/model-runtime",
                headers=admin,
                json={"ai_mode": mode, "expected_revision": revision},
            )
            for mode in ("disabled", "mock")
        )
    )
    assert sorted(r.status_code for r in results) == [200, 409]
    events = (await client.get(P + "/admin/model-events", headers=admin)).json()["items"]
    assert len(events) == 1 and events[0]["revision"] == revision + 1


async def test_disabled_mode_works_when_stored_key_cannot_be_decrypted(client, admin, app):
    """Проверяет возможность отключить AI при недоступном ключе шифрования старого профиля."""
    pid = (await client.post(P + "/admin/models", headers=admin, json=profile())).json()["id"]
    revision = (await client.get(P + "/admin/models", headers=admin)).json()["runtime"]["revision"]
    result = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={"llm_profile_id": pid, "expected_revision": revision},
    )
    assert result.status_code == 200, result.text
    # The administrator must retain a way to disable AI after loss/rotation of the master key.
    app.state.settings.model_secrets_key = Fernet.generate_key().decode()
    revision = result.json()["revision"]
    result = await client.patch(
        P + "/admin/model-runtime",
        headers=admin,
        json={"ai_mode": "disabled", "expected_revision": revision},
    )
    assert result.status_code == 200, result.text
    assert result.json()["ai_mode"] == "disabled"


def test_init_env_preserves_password_from_existing_database_url(tmp_path):
    """Проверяет извлечение существующего пароля из DSN без его замены."""
    from dotenv import dotenv_values

    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql+asyncpg://emo:existing%40password@db:5432/emo\n"
    )
    init_env(tmp_path)
    values = dotenv_values(tmp_path / ".env")
    assert values["POSTGRES_PASSWORD"] == "existing@password"
    assert values["DATABASE_URL"] == "postgresql+asyncpg://emo:existing%40password@db:5432/emo"
