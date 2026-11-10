from pathlib import Path

import yaml

from emo_core.services.policies import DEFAULT_DIR

STAGE_LINES = {
    "pregnant": "Контекст: беременность{week}. Не оценивай течение беременности и не давай акушерских заключений.",
    "postpartum": "Контекст: послеродовой период{week}. Усталость и перепады настроения в первые дни часты, но не объясняй ими любые жалобы и не отговаривай от обращения за помощью.",
    "after_loss": "Контекст: беременность завершилась потерей. Не упоминай ребёнка, роды или шкалы беременности как предстоящие, не предлагай «думать о будущем»; признай утрату, не торопи и не ищи причин.",
}


class SupportProtocolSelector:
    def __init__(self, settings):
        """Выбирает каталог политик из policy_dir либо встроенного набора."""
        self.directory = Path(settings.policy_dir) if settings.policy_dir else DEFAULT_DIR

    def load_approved(self, key):
        """Читает YAML-протокол и возвращает его только с документированным статусом approved.

        Требует reviewed_by и approval_reference. При отсутствии файла или реквизитов
        возвращает None; само содержание на клиническую корректность не оценивает.
        """
        path = self.directory / "protocols" / f"{key}.yaml"
        if not path.exists():
            return None
        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            protocol.get("clinical_review_status") != "approved"
            or not protocol.get("reviewed_by")
            or not protocol.get("approval_reference")
        ):
            return None
        return protocol

    def select(self, state):
        """Выбирает основной и послеродовой утверждённые протоколы; для urgent возвращает два None."""
        if state.safety_route == "urgent":
            return None, None
        return self.load_approved(state.support_protocol), self.load_approved(
            "postpartum_support"
        ) if state.postpartum else None

    def allowed_actions(self, state):
        """Собирает разрешённые действия и требования к текущему ответу.

        Учитывает обязательность рекомендации, её недавнее напоминание и допустимость
        добровольного предложения шкалы. Эти инструкции задаёт код по SupportState.
        """
        actions = [
            "выслушивание",
            "один уточняющий вопрос",
            "предложение реальной поддержки близких",
        ]
        if state.referral_required:
            lines = [
                "Обязательно предложи обратиться к врачу или квалифицированному специалисту: спокойно, без запугивания и без диагноза."
            ]
        elif state.recommend_specialist:
            lines = [
                "Обращение к специалисту уже рекомендовано недавно: не повторяй рекомендацию в каждом ответе, но поддержи, если она сама заговорит об этом."
            ]
        else:
            lines = []
        if state.screening_offer and state.safety_route == "normal":
            lines.append(
                "Допустимо один раз, если это уместно, упомянуть, что в приложении есть короткий необязательный опросник о самочувствии. Не настаивай и не называй результат диагнозом."
            )
        else:
            lines.append("Не предлагай опросники и тесты в этом ответе.")
        return lines + ["Разрешены: " + ", ".join(actions) + "."]

    def system_prompt(
        self, state, protocol, postpartum, support_contact="", address="ты", privacy_note=False
    ):
        """Собирает системные правила, перинатальный контекст и разрешённые действия.

        Добавляет только переданные утверждённые протоколы, выбранную форму обращения,
        правило псевдонимов и ограничения контактов. Личную историю сюда не включает:
        она передаётся отдельно как недоверенные данные.
        """
        base = (self.directory / "base_system_prompt.md").read_text(encoding="utf-8")
        stage = STAGE_LINES.get(state.perinatal_stage)
        if stage:
            week = (
                f", {state.gestational_week}-я неделя"
                if state.gestational_week is not None
                else f", {state.postpartum_weeks} нед. после родов"
                if state.postpartum_weeks is not None
                else ""
            )
            base += "\n" + stage.format(week=week)
        if protocol:
            base += "\nУтверждённый протокол:\n" + yaml.safe_dump(protocol, allow_unicode=True)
        if postpartum:
            base += "\nДополнительный контекст:\n" + yaml.safe_dump(postpartum, allow_unicode=True)
        contacts = (
            "Линию помощи можно назвать один раз, без нажима: "
            + support_contact
            + " Экстренные службы: 112, скорая 103. Других телефонов, сайтов и организаций не называй."
            if support_contact
            else "В этом ответе не называй телефонов, сайтов и организаций. Экстренные службы"
            " (112, 103) упоминай, только если она описывает опасность."
        )
        form = (
            "Обращайся на «вы»."
            if address == "вы"
            else "Обращайся на «ты», как в остальных сообщениях приложения; не переходи на «вы»."
        )
        privacy = (
            "\nЛичные данные в тексте заменены обозначениями в квадратных скобках ([ИМЯ_1],"
            " [ТЕЛЕФОН_1] и подобными). Используй их как обычные имена и не обсуждай сами обозначения."
            " Обозначения не склоняются: пиши их точно так, как они встретились, а если нужен"
            " другой падеж — назови человека по роли («с мужем», «врачу») вместо обозначения."
            if privacy_note
            else ""
        )
        return (
            base
            + "\n"
            + form
            + privacy
            + "\n"
            + contacts
            + "\n"
            + "\n".join(self.allowed_actions(state))
        )
