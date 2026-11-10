import re

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.schemas.emotion import CATEGORIES, EXTRA_SIGNALS, EmotionAnalysis
from emo_core.services import safety_lexicon

PATTERNS = {
    "anxiety": r"тревож|волнуюсь|беспокоюсь",
    "fear_for_baby": r"боюсь за (?:малыша|реб[её]нка)",
    "low_mood": r"груст|плачу|подавлен",
    "loss_of_interest": r"ничего не радует|нет интереса",
    "sleep_problem": r"не могу (?:у|за)снуть|плохо сплю|бессонниц",
    "fatigue": r"устала|нет сил",
    "irritability": r"раздражает|раздражительн",
    "loneliness": r"одиноко|одинока",
    "relationship_stress": r"ссор|муж не поддерживает",
    "overwhelm": r"не справляюсь|слишком много",
    "physical_discomfort": r"болит|тошнит",
    "self_reported_improvement": r"стало легче|стало лучше",
    "self_reported_worsening": r"стало хуже|стало тяжелее",
    "functional_impact": r"не могу работать|не могу встать",
    "request_for_support": r"поддерж|поговор",
    "request_for_medical_advice": r"лекарств|дозировк|какие таблетки",
    "need_for_clarification": r"не знаю что чувствую",
    "maternal_estrangement": r"плохая мать|ужасная мать|ничего не чувствую к (?:реб[её]нку|малышу)|реб[её]нку (?:будет|было бы) лучше без меня",
}
# Признаки безопасности демонстрационный классификатор берёт из того же явного словаря,
# что и детерминированные правила: он не изображает независимую клиническую оценку.
LEXICON_MARKERS = {
    "possible_self_harm": {"self_harm_ideation", "self_harm_intent", "self_harm_act"},
    "self_harm_intent": {"self_harm_intent", "self_harm_act"},
    "possible_urgent_medical_issue": {"acute_medical"},
    "possible_psychosis": {"psychosis_command"},
    "possible_harm_to_baby": {"harm_to_baby_intent", "harm_to_baby_act"},
    "possible_violence_exposure": {"violence_exposure", "violence_immediate"},
}


def self_report_text(text):
    """Удаляет цитаты и некоторые отрицания, чужие и давние события для mock-классификатора."""
    text = re.sub(r'«[^»]*»|“[^”]*”|"[^"]*"', "", text.lower())
    # Conservative deterministic development fixture, not a clinical classifier.
    clauses = re.split(r"[.!?;\n]|\bно\b|\bа сейчас\b", text)
    return " ".join(
        c
        for c in clauses
        if not re.search(
            r"подруга|сестра|она (?:сказала|говорит)|раньше|в прошлом|год назад|\bне тревож|\bнет тревог|\bне груст|\bне чувствую тревог|\bне хочу навредить|\bне собираюсь",
            c,
        )
    )


class MockJevProvider:
    async def analyze(self, text):
        """Возвращает детерминированные оценки 0/1 по регулярным выражениям.

        Защитные маркеры берёт из того же словаря, что и явные правила сервиса.
        Источник помечает mock; независимую оценку клинической модели не выполняет.
        """
        current = self_report_text(text)
        risks = set(safety_lexicon.assess(text).risks)
        markers = {}
        for code in (*CATEGORIES, *EXTRA_SIGNALS):
            if code in LEXICON_MARKERS:
                markers[code] = float(bool(risks & LEXICON_MARKERS[code]))
            else:
                markers[code] = float(bool(re.search(PATTERNS[code], current)))
        return EmotionAnalysis(
            markers=markers, model_version="deterministic-demo-v1", source="mock"
        )

    async def check_output(self, text, route):
        """Возвращает нулевые оценки всех проверок ответа для детерминированных тестов."""
        from emo_core.providers.jev_provider import OUTPUT_RULES

        return dict.fromkeys(OUTPUT_RULES, 0.0)


class MockLLMProvider:
    async def generate(self, messages, repair=False):
        # Context is deliberately visible in deterministic tests without replaying private text.
        """Возвращает один из шаблонных ответов без вызова внешней модели.

        Учитывает число пользовательских реплик и системное требование рекомендации.
        Аргумент repair оставлен для совместимости с интерфейсом LLMProvider.
        """
        system = messages[0]["content"] if messages else ""
        referral = (
            " Об этом стоит поговорить с врачом или специалистом по психическому здоровью."
            if "Обязательно предложи" in system
            else ""
        )
        prior = sum(m["role"] == "user" for m in messages)
        if prior > 2:
            return (
                "Я помню, что мы уже говорили о твоём самочувствии. Можно вернуться к тому, что сейчас для тебя важно."
                + referral
                + " Что хотелось бы обсудить?"
            )
        return (
            "Спасибо, что поделилась тем, как тебе сейчас. Можно рассказать об этом в своём темпе."
            + referral
            + " Что сейчас помогло бы почувствовать больше поддержки?"
        )


class DisabledJevProvider:
    async def analyze(self, text):
        """Сообщает external_ai_disabled через ProviderUnavailable, не обрабатывая текст."""
        raise ProviderUnavailable("external_ai_disabled")

    async def check_output(self, text, route):
        """Сообщает external_ai_disabled через ProviderUnavailable вместо проверки ответа."""
        raise ProviderUnavailable("external_ai_disabled")


class DisabledLLMProvider:
    async def generate(self, messages, repair=False):
        """Сообщает external_ai_disabled через ProviderUnavailable вместо генерации."""
        raise ProviderUnavailable("external_ai_disabled")
