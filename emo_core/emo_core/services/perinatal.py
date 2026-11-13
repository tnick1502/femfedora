"""Перинатальный контекст пользователя.

EMO не ведёт собственный акушерский календарь: срок, предполагаемая дата родов, дата родов
и исход приходят из основного приложения. Здесь они только приводятся к текущему моменту,
чтобы однажды переданная «28-я неделя» не оставалась двадцать восьмой навсегда.
"""

from dataclasses import asdict, dataclass
from datetime import datetime

TERM_DAYS = 280  # 40 недель от первого дня последней менструации
MAX_PREGNANCY_WEEKS = 43  # дальше данные считаются устаревшими, а не «44-й неделей»
PERINATAL_POSTPARTUM_DAYS = 365  # перинатальный период: беременность и год после родов


@dataclass(frozen=True)
class PerinatalContext:
    stage: str = "unknown"  # preconception | pregnant | postpartum | after_loss | unknown
    gestational_week: int | None = None
    trimester: int | None = None
    postpartum_weeks: int | None = None
    source: str = "none"

    @property
    def postpartum(self) -> bool:
        """Проверяет, что текущий этап равен postpartum."""
        return self.stage == "postpartum"

    def as_dict(self) -> dict:
        """Возвращает этап, срок, триместр и источник контекста как словарь."""
        return asdict(self)


def _trimester(week: int) -> int:
    """Определяет триместр: до 13 недель включительно, 14–27 или от 28 недель."""
    return 1 if week <= 13 else 2 if week <= 27 else 3


def same_assessment_episode(attempt, user):
    """Use explicit episode ID and known birth date; don't infer a new pregnancy from prose."""
    if attempt.pregnancy_id != user.pregnancy_id:
        return False
    birth = attempt.context.get("birth_date")
    return not (birth and user.birth_date and birth != user.birth_date.isoformat())


def perinatal_context(user, now: datetime) -> PerinatalContext:
    """Приводит переданные сведения о беременности или родах к моменту now.

    Сначала учитывает исход loss, затем дату родов в пределах 365 дней.
    Срок беременности вычисляет по предполагаемой дате родов либо прибавляет
    полные недели к сроку на дату его передачи. Срок вне 0–43 недель даёт unknown
    с указанием устаревшего источника. Профиль не меняет; даты ожидаются с поясом.
    """
    explicit = getattr(user, "reproductive_stage", None)
    if explicit in ("preconception", "unknown", "after_loss"):
        return PerinatalContext(stage=explicit, source="reproductive_stage")
    if getattr(user, "pregnancy_outcome", None) == "loss":
        return PerinatalContext(stage="after_loss", source="pregnancy_outcome")
    birth = getattr(user, "birth_date", None)
    if birth is not None and birth <= now:
        days = (now - birth).days
        if days <= PERINATAL_POSTPARTUM_DAYS:
            return PerinatalContext(
                stage="postpartum", postpartum_weeks=days // 7, source="birth_date"
            )
        return PerinatalContext(source="birth_date_beyond_perinatal_period")
    week, source = None, "none"
    due = getattr(user, "estimated_due_date", None)
    if due is not None:
        week, source = (TERM_DAYS - (due - now.date()).days) // 7, "estimated_due_date"
    elif getattr(user, "gestational_week", None) is not None:
        recorded = getattr(user, "gestational_week_recorded_at", None)
        elapsed = (now - recorded).days // 7 if recorded else 0
        week, source = user.gestational_week + max(elapsed, 0), "gestational_week"
    if week is None:
        return PerinatalContext(
            stage=explicit or "unknown", source="reproductive_stage" if explicit else "none"
        )
    if not 0 <= week <= MAX_PREGNANCY_WEEKS:
        # Срок не обновлялся или дата родов не передана: не выдумываем текущий статус.
        return PerinatalContext(source=f"{source}_stale")
    return PerinatalContext(
        stage="pregnant", gestational_week=week, trimester=_trimester(week), source=source
    )
