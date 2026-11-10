from emo_core.schemas.emotion import CATEGORIES, SupportState

PROTOCOLS = {
    "anxiety": "anxiety_support",
    "fear_for_baby": "anxiety_support",
    "low_mood": "low_mood_support",
    "loss_of_interest": "low_mood_support",
    "overwhelm": "overload_support",
    "fatigue": "overload_support",
    "sleep_problem": "overload_support",
    "loneliness": "social_support",
    "relationship_stress": "social_support",
}
DEFAULT_PERSISTENCE = {"window_days": 30, "min_distinct_days": 3, "min_span_days": 14}


class StateEvaluator:
    """Явные правила текущей потребности в поддержке. Не диагностика.

    Частота упоминаний не приравнивается к выраженности: рекомендация специалиста возникает
    из длительности (жалоба повторяется в разные дни на протяжении двух и более недель),
    влияния на повседневную жизнь, результата шкалы или сигнала безопасности.
    """

    def __init__(self, rules=None):
        """Загружает правила длительности жалоб, дополняя отсутствующие значения стандартными."""
        self.persistence = {**DEFAULT_PERSISTENCE, **((rules or {}).get("persistence") or {})}

    def pattern(self, code, recurrence, complaints):
        """Определяет характер повторения маркера: unknown, recurring или persistent.

        С историей учитывает число разных дней и интервал между первым и последним.
        Без истории поддерживает прежний счётчик карточки, который может дать
        только recurring, но не основание persistent.
        """
        if recurrence is None:
            # Совместимость: без истории наблюдений известен только счётчик карточки.
            repeated = any(
                c.category == code and c.details.get("observations", 0) >= 3 for c in complaints
            )
            return "recurring" if repeated else "unknown"
        seen = recurrence.get(code)
        if not seen or seen["days"] < self.persistence["min_distinct_days"]:
            return "unknown"
        return (
            "persistent" if seen["span_days"] >= self.persistence["min_span_days"] else "recurring"
        )

    def evaluate(
        self,
        analysis,
        complaints,
        route,
        postpartum=False,
        threshold=0.7,
        *,
        perinatal=None,
        recurrence=None,
        screening=None,
    ):
        """Выбирает потребность в поддержке и основания для обращения к специалисту.

        Маркеры выше threshold сортирует по убыванию оценки, при равенстве — по
        порядку CATEGORIES. Учитывает маршрут безопасности, длительность наблюдений,
        влияние на повседневную жизнь, свежие шкалы и перинатальный контекст.
        Возвращает SupportState; медицинский диагноз не вычисляет и БД не меняет.
        """
        markers = sorted(
            ((k, v) for k, v in analysis.markers.items() if k in CATEGORIES and v >= threshold),
            key=lambda kv: (-kv[1], CATEGORIES.index(kv[0])),
        )
        main = markers[0][0] if markers else "general"
        pattern = self.pattern(main, recurrence, complaints)
        reasons = []
        if route != "normal":
            reasons.append("safety_route")
        if pattern == "persistent":
            reasons.append("persistent_complaint")
        if analysis.markers.get("functional_impact", 0) >= threshold:
            reasons.append("functional_impact")
        if (
            recurrence is not None
            and self.pattern("maternal_estrangement", recurrence, complaints) != "unknown"
        ):
            reasons.append("persistent_maternal_estrangement")
        for result in screening or []:
            if result.get("safety_flags"):
                reasons.append(f"screening_safety_item:{result['code']}")
            elif result.get("followup"):
                reasons.append(f"screening_followup:{result['code']}")
        if perinatal is not None:
            postpartum = perinatal.postpartum
        return SupportState(
            primary_support_need=main,
            secondary_support_need=markers[1][0] if len(markers) > 1 else None,
            distress_pattern=pattern,
            safety_route=route,
            recommend_specialist=bool(reasons),
            specialist_reasons=reasons,
            support_protocol=PROTOCOLS.get(main, "general_support"),
            postpartum=postpartum,
            perinatal_stage=perinatal.stage if perinatal else "unknown",
            gestational_week=perinatal.gestational_week if perinatal else None,
            postpartum_weeks=perinatal.postpartum_weeks if perinatal else None,
        )
