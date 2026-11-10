import math
import re
from dataclasses import dataclass, field

from emo_core.core.exceptions import ProviderUnavailable
from emo_core.db.models import SafetyEvent
from emo_core.schemas.decision import OutputCheck
from emo_core.services import safety_lexicon
from emo_core.services.policies import policy

# Маркеры Jev, которые при уверенном срабатывании означают возможную непосредственную опасность.
URGENT_MARKERS = (
    "self_harm_intent",
    "possible_urgent_medical_issue",
    "possible_psychosis",
    "possible_harm_to_baby",
)
# Маркеры, которые уже на пороге проверки требуют рекомендации специалиста.
REVIEW_MARKERS = (*URGENT_MARKERS, "possible_self_harm")
# Маркеры без срочности: рекомендация специалиста при уверенном срабатывании.
REFERRAL_MARKERS = ("possible_violence_exposure", "request_for_medical_advice")
SELF_HARM_MARKERS = ("possible_self_harm", "self_harm_intent")
# Маркеры с пониженным порогом проверки.
SENSITIVE_MARKERS = (*SELF_HARM_MARKERS, "possible_urgent_medical_issue")
MENTAL_STATE_RISKS = {"possible_psychosis", "mental_state_red_flag"}
OTHER_URGENT_RISKS = {
    "self_harm_intent",
    "self_harm_act",
    "harm_to_baby_intent",
    "harm_to_baby_act",
    "psychosis_command",
    "violence_immediate",
    "possible_harm_to_baby",
}
SELF_HARM_RISKS = {"self_harm_intent", "self_harm_act", "self_harm_ideation", "possible_self_harm"}
MEDICAL_RISKS = {"acute_medical", "possible_urgent_medical_issue"}
REFERRAL_WORDS = re.compile(
    r"врач|специалист|психолог|психотерапевт|психиатр|акушер|гинеколог|скор\w+ помощ|\b112\b|\b103\b"
)
# Телефон: последовательность цифр с разделителями; проверяются номера от десяти цифр,
# чтобы годы, сроки и давление не принимались за контакты.
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s()\-]{8,}\d")
DOMAIN = re.compile(r"\b((?:[a-z0-9а-я-]+\.)+(?:ru|рф|com|org|net|su|info|online|io|me))\b")
NEGATION_BEFORE = re.compile(
    r"(?:\bне (?:могу|можем|может|буду|стану|вправе|берусь|имею права|стоит|нужно|надо|следует)\b|"
    r"\bнельзя\b|\bникто не (?:может|знает)\b|\bневозможно\b|\bне\s)[^.!?]*$"
)
FORBIDDEN_OUTPUT = tuple(
    re.compile(pattern)
    for pattern in (
        # Диагнозы
        r"(?:у (?:вас|тебя)|это точно|это похоже на|у тебя явно|у вас явно) (?:\w+ )?(?:депресси|психоз|расстройств|невроз|птср)",
        r"(?:послеродов\w+|клиническ\w+) депресси\w+ у (?:тебя|вас)",
        # Ложные заверения
        r"(?:точно|гарантирую|обещаю|уверяю|можешь не сомневаться).{0,30}(?:здоров|безопас|ничего страшного|все будет хорошо|все в порядке|все обойдется)",
        r"(?:абсолютно|совершенно|полностью).{0,15}(?:здоров|безопасн)",
        r"(?:ребенок|малыш).{0,12}(?:точно здоров|в полной безопасности|в порядке,? не переживай)",
        r"это (?:просто|всего лишь) (?:гормоны|усталость|нервы|капризы)",
        r"(?:не нужно|не надо|не стоит|незачем) (?:идти|обращаться|ходить|звонить).{0,20}(?:врач|специалист|психолог|скор)",
        r"не (?:ходи|обращайся|звони).{0,20}(?:врач|психолог|специалист|скор)",
        # Лечение и дозы
        r"(?:принимай|принимайте|выпей|выпейте|назначаю|отмени|отмените|пейте|попробуй принимать|начни принимать|(?:прекрати|перестань|бросай|брось) (?:принимать|пить)).{0,50}(?:мг|таблет|препарат|антидепрессант|лекарств|успокоительн|снотворн)",
        r"(?:\d+\s*мг|\d+\s*таблет).{0,30}(?:в день|ежедневно|на ночь|утром|вечером)",
        # Зависимость и ложная идентичность
        r"(?:только я|никто кроме меня).{0,30}(?:понимаю|помогу|нужн)",
        r"больше никому не (?:рассказывай|говори)",
        r"(?<!не )я (?:твой|ваш) (?:врач|психолог|психотерапевт|акушер)|я (?:тоже|живой) человек|я скучала|ты нужна мне|не оставляй меня",
    )
)


@dataclass
class InputDecision:
    route: str = "normal"
    risks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    lexicon_route: str = "normal"

    @property
    def self_harm(self) -> bool:
        """Проверяет наличие маркеров мыслей, намерения или действий по причинению себе вреда."""
        return bool(SELF_HARM_RISKS & set(self.risks))

    @property
    def mental_state_only(self) -> bool:
        """Выделяет признаки изменения психического состояния без других срочных рисков."""
        risks = set(self.risks)
        return bool(risks & MENTAL_STATE_RISKS) and not risks & (OTHER_URGENT_RISKS | MEDICAL_RISKS)

    @property
    def medical_only(self) -> bool:
        """Выделяет медицинские риски без срочных намерений, действий и психотических признаков."""
        risks = set(self.risks)
        urgent_other = risks & {
            "self_harm_intent",
            "self_harm_act",
            "harm_to_baby_intent",
            "harm_to_baby_act",
            "psychosis_command",
            "violence_immediate",
            "possible_psychosis",
            "possible_harm_to_baby",
        }
        return bool(risks & MEDICAL_RISKS) and not urgent_other


class SafetyGuard:
    def __init__(self, settings, jev):
        """Сохраняет провайдера проверки и загружает текущую safety policy."""
        self.settings = settings
        self.jev = jev
        self.rules = policy(settings, "safety_rules.yaml")

    # ------------------------------------------------------------------ input
    def deterministic(self, text) -> safety_lexicon.LexiconResult:
        """Возвращает маршрут и коды риска из явного словаря по исходному тексту."""
        return safety_lexicon.assess(text)

    def deterministic_route(self, text) -> str:
        """Возвращает только маршрут словарной проверки без обращения к Jev."""
        return self.deterministic(text).route

    def assess_input(self, text, analysis) -> InputDecision:
        """Объединяет явные правила и оценки Jev в InputDecision.

        Модель может усилить маршрут, но не отменить сработавшее правило. Порог
        чувствительных маркеров ниже общего; слабые сигналы записывает как uncertain.
        Недоступный анализ на обычном маршруте требует рекомендации специалиста.
        """
        lexicon = self.deterministic(text)
        decision = InputDecision(
            route=lexicon.route, risks=list(lexicon.risks), lexicon_route=lexicon.route
        )
        if lexicon.route == "urgent":
            decision.reasons.append("deterministic_urgent_rule")
        elif lexicon.route == "recommend_professional":
            decision.reasons.append("deterministic_referral_rule")
        if not analysis.available:
            decision.reasons.append(
                "analysis_skipped_for_urgent"
                if lexicon.route == "urgent"
                else "classifier_unavailable"
            )
            if decision.route == "normal":
                # Без классификатора сообщение не считается автоматически безопасным.
                decision.route = "recommend_professional"
            return decision
        action = self.settings.jev_threshold
        for key in (*REVIEW_MARKERS, *REFERRAL_MARKERS):
            value = analysis.markers.get(key, 0)
            review = (
                self.settings.jev_sensitive_review_threshold
                if key in SENSITIVE_MARKERS
                else self.settings.jev_review_threshold
            )
            if value < review:
                continue
            confident = value >= action
            if key in REFERRAL_MARKERS and not confident:
                continue
            # Слабый сигнал самоповреждения (между низким и обычным порогом проверки) даёт
            # рекомендацию специалиста, но не текст о мыслях о причинении себе вреда.
            weak = not confident and value < self.settings.jev_review_threshold
            decision.reasons.append(
                f"{key}:above_"
                + (
                    "action_threshold"
                    if confident
                    else "low_threshold"
                    if weak
                    else "review_threshold"
                )
            )
            decision.risks.append(
                key
                if not weak
                else "self_harm_uncertain"
                if key in SELF_HARM_MARKERS
                else "medical_uncertain"
            )
            if confident and key in URGENT_MARKERS:
                decision.route = "urgent"
            elif decision.route == "normal":
                decision.route = "recommend_professional"
        decision.risks = sorted(set(decision.risks))
        if not decision.reasons:
            decision.reasons.append("no_risk_rule_triggered")
        return decision

    def input_route(self, text, analysis) -> str:
        """Возвращает маршрут объединённой проверки правил и уже полученного анализа."""
        return self.assess_input(text, analysis).route

    # ----------------------------------------------------------------- output
    @staticmethod
    def deterministic_output(text):
        """Проверяет текст ответа по программным запретам и возвращает признак допуска.

        Учитывает отрицание перед совпадением, чтобы не блокировать честный отказ
        обещать благополучие или назначать лечение. Jev не вызывает.
        """
        lowered = text.lower().replace("ё", "е")
        for pattern in FORBIDDEN_OUTPUT:
            for match in pattern.finditer(lowered):
                # «Не могу обещать, что всё точно будет хорошо» и «не принимай лекарства без
                # врача» — честный отказ, а не нарушение: отрицание перед совпадением его снимает.
                if not NEGATION_BEFORE.search(lowered[max(0, match.start() - 60) : match.start()]):
                    return False
        return True

    @staticmethod
    def has_referral(text) -> bool:
        """Ищет явную рекомендацию обратиться к врачу или специалисту в тексте ответа."""
        return bool(REFERRAL_WORDS.search(text.lower().replace("ё", "е")))

    async def evaluate_output(self, text, route, referral_required=None) -> OutputCheck:
        """Возвращает OutputCheck после правил, проверки контактов и Jev.

        Если обязательность рекомендации не задана, определяет её по маршруту.
        Программный запрет, неизвестный контакт и пропущенная рекомендация блокируют
        ответ до вызова Jev. Пустые или некорректные оценки и сбой провайдера также
        дают safe=False; результаты проверки сохраняются в структурированном отчёте.
        """
        if referral_required is None:
            referral_required = route == "recommend_professional"
        deterministic = self.deterministic_output(text)
        if not deterministic:
            return OutputCheck(
                safe=False,
                deterministic_passed=False,
                reasons=["deterministic_output_rule_blocked"],
            )
        if self.unverified_contacts(text):
            # Модель не должна придумывать телефоны и сайты помощи.
            return OutputCheck(
                safe=False,
                deterministic_passed=False,
                reasons=["unverified_contact_blocked"],
            )
        referral = self.has_referral(text) if referral_required else None
        if referral is False:
            return OutputCheck(
                safe=False,
                deterministic_passed=True,
                referral_present=False,
                reasons=["required_referral_missing"],
            )
        try:
            values = await self.jev.check_output(text, route)
            valid = bool(values) and all(
                isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                for v in values.values()
            )
            if not valid:
                raise ProviderUnavailable("jev_invalid_response")
            reasons = [
                key for key, value in values.items() if value >= self.settings.jev_output_threshold
            ]
            return OutputCheck(
                safe=not reasons,
                deterministic_passed=True,
                referral_present=referral,
                jev_called=True,
                jev_available=True,
                jev_scores=values,
                reasons=reasons or ["all_output_checks_passed"],
            )
        except ProviderUnavailable:
            return OutputCheck(
                safe=False,
                deterministic_passed=True,
                referral_present=referral,
                jev_called=True,
                jev_available=False,
                reasons=["output_classifier_unavailable"],
            )

    async def output_safe(self, text, route):
        """Возвращает только safe из полной проверки ответа evaluate_output."""
        return (await self.evaluate_output(text, route)).safe

    # -------------------------------------------------------------- templates
    def emergency_text(self, country=None) -> str:
        """Берёт текст экстренных контактов страны из политики с предусмотренным fallback."""
        contacts = self.rules.get("emergency_contacts") or {}
        entry = contacts.get(country or self.rules.get("default_country")) or contacts.get(
            "default"
        )
        return (entry or {}).get("text", "")

    def support_entry(self, country=None) -> dict:
        """Возвращает запись линии поддержки для заданной страны либо пустой словарь."""
        contacts = self.rules.get("support_contacts") or {}
        return contacts.get(country or self.rules.get("default_country")) or {}

    def support_text(self, country=None) -> str:
        """Возвращает текст линии поддержки из записи политики либо пустую строку."""
        return self.support_entry(country).get("text", "")

    def contacts(self, country=None) -> dict:
        """Собирает страну, экстренные номера, линию поддержки и статус проверки политики для API."""
        code = country or self.rules.get("default_country")
        emergency = self.rules.get("emergency_contacts") or {}
        entry = emergency.get(code) or emergency.get("default") or {}
        support = self.support_entry(country)
        fields = ("label", "phone", "phone_display", "hours", "site", "operator", "text")
        return {
            "country": code if code in emergency else None,
            "emergency": {"text": entry.get("text", ""), "phones": entry.get("phones") or []},
            "support": {k: support[k] for k in fields if k in support} or None,
            "clinical_review_status": self.rules.get("clinical_review_status"),
        }

    def allowed_contacts(self):
        """Собирает разрешённые номера и домены из политики, добавляя встроенные 112 и 103."""
        phones, sites = {"112", "103"}, set()
        for group in ("emergency_contacts", "support_contacts"):
            for entry in (self.rules.get(group) or {}).values():
                phones |= {re.sub(r"\D", "", p) for p in PHONE.findall(entry.get("text", ""))}
                if entry.get("phone"):
                    phones.add(re.sub(r"\D", "", str(entry["phone"])))
                if entry.get("site"):
                    sites.add(entry["site"].lower())
        return phones, sites

    def unverified_contacts(self, text) -> bool:
        """Проверяет наличие телефона или домена вне разрешённого списка.

        Для телефонов сравнивает последовательности цифр длиной от десяти и учитывает
        эквивалентность российских префиксов 8/+7. Для сайтов допускает домен
        политики и его поддомены. True означает, что контакт нужно заблокировать.
        """
        phones, sites = self.allowed_contacts()
        lowered = text.lower()
        for found in PHONE.findall(lowered):
            digits = re.sub(r"\D", "", found)
            if len(digits) < 10:
                continue
            same_number = len(digits) == 11 and (
                # 8 800… и +7 800… — один и тот же номер.
                "8" + digits[1:] in phones or "7" + digits[1:] in phones
            )
            if digits not in phones and not same_number:
                return True
        return any(
            not any(domain == site or domain.endswith("." + site) for site in sites)
            for domain in DOMAIN.findall(lowered)
        )

    def template(self, kind, country=None):
        """Подставляет контакты страны в выбранный шаблон политики и нормализует пробелы."""
        text = self.rules[f"{kind}_template"] if kind != "self_harm_addendum" else self.rules[kind]
        text = text.replace("{emergency}", self.emergency_text(country))
        text = text.replace("{support}", self.support_text(country))
        return re.sub(r"\s+", " ", text).strip()

    def urgent_kind(self, decision: InputDecision) -> str:
        """Выбирает медицинский, психический или общий срочный шаблон по составу рисков."""
        if decision.medical_only and "medical_urgent_template" in self.rules:
            return "medical_urgent"
        if decision.mental_state_only and "mental_state_urgent_template" in self.rules:
            return "mental_state_urgent"
        return "urgent"

    def record(self, session, user_id, message_id, risk_type, route, action, details=None):
        """Добавляет SafetyEvent с кодами риска, маршрутом, действием и версией политики.

        Событие фиксирует транзакция вызывающего кода. В details следует передавать
        коды и технические признаки, исключая текст переписки и секреты.
        """
        session.add(
            SafetyEvent(
                user_id=user_id,
                message_id=message_id,
                risk_type=risk_type,
                risk_route=route,
                action_taken=action,
                policy_version=self.rules["version"],
                # Только коды правил, без текста сообщения.
                details=details or {},
            )
        )
