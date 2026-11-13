from pydantic import BaseModel, Field

CATEGORIES = (
    "anxiety",
    "fear_for_baby",
    "low_mood",
    "loss_of_interest",
    "sleep_problem",
    "fatigue",
    "irritability",
    "loneliness",
    "relationship_stress",
    "overwhelm",
    "physical_discomfort",
)
EXTRA_SIGNALS = (
    "self_reported_improvement",
    "self_reported_worsening",
    "functional_impact",
    "request_for_support",
    "request_for_medical_advice",
    "need_for_clarification",
    "possible_self_harm",
    "possible_urgent_medical_issue",
    "possible_psychosis",
    "possible_harm_to_baby",
    # Намерение, план или уже совершённое действие — отдельно от мыслей без намерения.
    "self_harm_intent",
    # Насилие или угрозы со стороны партнёра/близких.
    "possible_violence_exposure",
    # «Красный флаг» MBRRACE-UK: устойчивое ощущение несостоятельности как матери
    # или отчуждение от ребёнка. Учитывается только при повторении во времени.
    "maternal_estrangement",
)


class EmotionAnalysis(BaseModel):
    intent: str = "support"
    markers: dict[str, float] = Field(default_factory=dict)
    model_version: str = "unavailable"
    source: str = "jev"
    available: bool = True
    error_code: str | None = None
    # Пропущенная классификация не равна 24 отрицательным результатам.
    scope: str = "full"  # full | irrelevant | unavailable
    gate_score: float | None = None
    input_calls: int = 1


class SupportState(BaseModel):
    primary_support_need: str = "general"
    secondary_support_need: str | None = None
    distress_pattern: str = "unknown"
    safety_route: str = "normal"
    recommend_specialist: bool = False
    screening_offer: str | None = None
    support_protocol: str = "general_support"
    support_basis: str = "no_support_marker_above_threshold"
    postpartum: bool = False
    perinatal_stage: str = "unknown"
    gestational_week: int | None = None
    postpartum_weeks: int | None = None
    # Коды оснований рекомендации специалиста (не диагнозы).
    specialist_reasons: list[str] = Field(default_factory=list)
    # Нужно ли обязательно упомянуть специалиста именно в этом ответе.
    referral_required: bool = False
    current_signals: list[str] = Field(default_factory=list)
    recent_signals: list[str] = Field(default_factory=list)
    analysis_scope: str = "full"
    change: str = "unknown"
    functional_impact: bool = False
    screening_reason: str | None = None
    screening_attempt_id: str | None = None
    birth_trauma_followup: bool = False
    context_conflicts: list[str] = Field(default_factory=list)
