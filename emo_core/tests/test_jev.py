from types import SimpleNamespace

import pytest

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.providers.jev_provider import TypeSafeJevProvider
from emo_core.providers.mock_providers import MockJevProvider
from emo_core.schemas.emotion import CATEGORIES, EXTRA_SIGNALS
from tests.conftest import conversation, send


async def test_multiple_markers():
    """Проверяет одновременное распознавание нескольких категорий mock-классификатором."""
    result = await MockJevProvider().analyze("Мне тревожно и я устала. Плохо сплю")
    assert all(result.markers[k] == 1 for k in ("anxiety", "fatigue", "sleep_problem"))


async def test_real_adapter_contract(config):
    """Проверяет схему вопросов TypeSafe SDK, преобразование оценок и фиксацию версии Jev."""
    config.jev_gate_enabled = False
    provider = TypeSafeJevProvider(config)

    async def ask(state, questions):
        """Проверяет отправленные вопросы SDK и возвращает заранее заданные оценки."""
        assert set((*CATEGORIES, *EXTRA_SIGNALS)) <= questions.keys()
        assert all(questions[k].model_dump()["type"] == "noul" for k in CATEGORIES)
        return SimpleNamespace(
            nouls={
                k: SimpleNamespace(noul=0.9 if k == "anxiety" else 0.1)
                for k in (*CATEGORIES, *EXTRA_SIGNALS)
            },
            choices={"intent": SimpleNamespace(choice="support")},
        )

    provider.ask = ask
    result = await provider.analyze("текст")
    assert result.markers["anxiety"] == 0.9
    assert result.model_version == "jev-1.14.0"


@pytest.mark.parametrize("value", [float("nan"), 2, -1])
def test_reject_invalid_jev(value):
    """Проверяет отказ от нечисловых, неполных или выходящих за диапазон оценок Jev."""
    with pytest.raises(ProviderUnavailable):
        TypeSafeJevProvider.read_nouls(
            SimpleNamespace(nouls={"a": SimpleNamespace(noul=value)}), ["a"]
        )


async def test_classifier_failure_has_no_fabricated_complaints(client, auth, app):
    """Проверяет отсутствие выдуманных жалоб при недоступном классификаторе."""

    class Broken:
        async def analyze(self, text):
            """Имитирует недоступность Jev без резервной mock-классификации."""
            raise ProviderUnavailable("unavailable")

    app.state.jev = Broken()
    cid = await conversation(client, auth)
    response = await send(client, auth, cid, "Мне тревожно")
    assert response.json()["assistant_message"]["metadata"]["analysis_available"] is False
    assert (await client.get("/api/v1/emo/complaints", headers=auth["anna"])).json()["items"] == []
