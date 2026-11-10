import yaml

from emo_core.schemas.emotion import EmotionAnalysis
from emo_core.services.state_evaluator import StateEvaluator
from emo_core.services.support_protocol_selector import SupportProtocolSelector


def test_protocol_rules_and_safety_priority(config, tmp_path):
    """Проверяет выбор протокола поддержки и приоритет срочного маршрута над ним."""
    state = StateEvaluator().evaluate(
        EmotionAnalysis(markers={"anxiety": 0.9}), [], "normal", postpartum=True
    )
    assert state.support_protocol == "anxiety_support" and state.postpartum
    assert SupportProtocolSelector(config).select(state) == (None, None)
    (tmp_path / "protocols").mkdir()
    protocol = {
        "id": "anxiety_support",
        "version": "test",
        "clinical_review_status": "approved",
        "reviewed_by": "fixture",
        "approval_reference": "fixture",
    }
    (tmp_path / "protocols/anxiety_support.yaml").write_text(yaml.safe_dump(protocol))
    (tmp_path / "base_system_prompt.md").write_text("Не назначай лечение")
    config.policy_dir = str(tmp_path)
    selector = SupportProtocolSelector(config)
    chosen, _ = selector.select(state)
    assert chosen["id"] == "anxiety_support"
    assert "Не назначай лечение" in selector.system_prompt(state, chosen, None)
    state.safety_route = "urgent"
    assert selector.select(state) == (None, None)


def test_low_mood_is_support_need():
    """Проверяет представление сниженного настроения как потребности в поддержке без диагноза."""
    state = StateEvaluator().evaluate(EmotionAnalysis(markers={"low_mood": 0.9}), [], "normal")
    assert state.primary_support_need == "low_mood"
    assert "diagnosis" not in state.model_dump()


def test_equal_markers_use_fixed_priority_independent_of_provider_key_order():
    """Проверяет постоянный приоритет равных маркеров независимо от порядка ключей провайдера."""
    states = [
        StateEvaluator().evaluate(EmotionAnalysis(markers=markers), [], "normal")
        for markers in ({"fatigue": 0.9, "anxiety": 0.9}, {"anxiety": 0.9, "fatigue": 0.9})
    ]
    assert all(
        s.primary_support_need == "anxiety" and s.support_protocol == "anxiety_support"
        for s in states
    )
