from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, StrictInt

from emo_core.schemas.user import StrictModel


class AttemptCreate(StrictModel):
    questionnaire_id: UUID | None = None
    attempt_id: UUID | None = None
    answers: dict[str, StrictInt] = Field(default_factory=dict, max_length=30)
    auxiliary_answers: dict[str, StrictInt] = Field(default_factory=dict, max_length=10)
    client_request_id: UUID | None = None
    expected_revision: StrictInt | None = Field(None, ge=0)
    complete: bool = False


class AttemptUpdate(StrictModel):
    expected_revision: StrictInt = Field(ge=0)
    answers: dict[str, StrictInt] = Field(default_factory=dict, max_length=30)
    auxiliary_answers: dict[str, StrictInt] = Field(default_factory=dict, max_length=10)


class QuestionnaireView(StrictModel):
    id: UUID
    code: str
    version: str
    language: str
    enabled: bool
    definition: dict[str, Any]
    scoring_rules: dict[str, Any]
    license_reference: str | None
    definition_hash: str


class QuestionnaireCatalog(StrictModel):
    items: list[QuestionnaireView]
    unavailable: list[dict[str, Any]]


class AttemptProgress(StrictModel):
    answered: int
    required: int
    missing_item_ids: list[str]


class ScoreBand(StrictModel):
    minimum: int
    maximum: int
    code: str | None = None
    label: str


class ScreeningResult(StrictModel):
    total_score: int
    maximum_score: int
    item_scores: dict[str, int]
    auxiliary_answers: dict[str, int]
    auxiliary_in_total: Literal[False]
    interpretation: ScoreBand | None
    recommended_action: Literal[
        "immediate_safety_assessment",
        "gad7_or_professional_assessment",
        "professional_assessment",
        "discuss_if_concerned",
    ]
    next_step: str
    safety_flags: list[str]
    scoring_method: Literal["sum_of_item_score_maps"]
    calculated_by: Literal["python"]
    models_used: list[str]
    scoring_version: str
    source_references: list[str]
    interpretation_note: str
    not_a_diagnosis: Literal[True]
    questionnaire_hash: str
    completed_at: datetime


class AttemptView(StrictModel):
    id: UUID
    user_id: UUID
    pregnancy_id: UUID | None
    questionnaire_id: UUID
    questionnaire_code: str
    questionnaire_version: str
    questionnaire_hash: str
    questionnaire: QuestionnaireView
    answers: dict[str, int]
    auxiliary_answers: dict[str, int]
    total_score: int | None
    status: Literal["in_progress", "completed"]
    revision: int
    client_request_id: UUID | None
    safety_flags: list[str]
    safety_guidance: str | None
    next_step: str | None
    context: dict[str, Any]
    started_at: datetime
    completed_at: datetime | None
    progress: AttemptProgress
    result: ScreeningResult | None
    not_a_diagnosis: bool = True


class AttemptPage(StrictModel):
    items: list[AttemptView]
    limit: int
    offset: int


class ScheduleAction(StrictModel):
    action: Literal["offered", "declined", "snoozed"]


class CheckinCreate(StrictModel):
    anxiety_score: int = Field(ge=0, le=10)
    mood_score: int = Field(ge=0, le=10)
    sleep_quality: int | None = Field(None, ge=0, le=10)
