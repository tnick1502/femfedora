"""Граница приватности для всех исходящих обращений к внешним моделям.

Один объект :class:`PrivacyBoundary` создаётся на запрос и применяется ко всему, что уходит
наружу: к реплике для классификатора, к системному контексту, истории и резюме беседы для LLM,
к ответу модели при его проверке. Псевдонимы согласованы внутри запроса (одно имя — один
псевдоним во всех частях промпта) и подставляются обратно в ответ перед сохранением.

Скрываются прямые идентификаторы: имена, телефоны, почта, ссылки и ники, адреса, учреждения
с номером или названием, номера документов и длинные номера, дата рождения, а также слова,
которые пользователь сам перечислил в личном кабинете. Содержание переживаний и срок
беременности не скрываются: без них поддержка невозможна. Это снижение риска, а не гарантия:
имя, не входящее в словарь и написанное без подсказывающих слов, может остаться в тексте.

Короткие числа заменяются только после явного маркера («паспорт», «полис», «№»), чтобы
сроки, баллы и давление не превращались в псевдонимы.
"""

import re
from dataclasses import dataclass, field

POLICY = "privacy_boundary_v1"
LEVELS = ("standard", "strict")

# Частые русские имена и уменьшительные формы. Основа + падежное окончание.
_NAMES_A = (
    "Анна Аня Анечка Мария Маша Машенька Марина Екатерина Катя Катюша Елена Лена Леночка Ольга "
    "Оля Наталья Наталия Наташа Татьяна Таня Ирина Ира Светлана Света Юлия Юля Анастасия Настя "
    "Дарья Даша Дашенька Александра Саша Сашенька Саня Шура Евгения Женя Виктория Вика Ксения "
    "Ксюша Полина Поля Алина Алёна Алена Кристина Валерия Лера София Софья Соня Сонечка "
    "Елизавета Лиза Маргарита Рита Вероника Ника Диана Яна Карина Арина Ульяна Варвара Варя "
    "Василиса Вася Милана Алиса Ева Злата Таисия Тася Кира Галина Галя Людмила Люда Мила "
    "Нина Валентина Валя Лариса Зинаида Зина Тамара Раиса Лидия Лида Антонина Тоня Зоя Инна "
    "Оксана Жанна Алла Эльвира Регина Камила Амина Мадина Лилия Лиля Нелли Эмилия Мирослава "
    "Ярослава Серафима Агата Стефания Стеша Олеся Леся Снежана Ангелина Геля Эвелина Майя "
    "Никита Илья Илюша Миша Мишенька Дима Димочка Серёжа Сережа Лёша Леша Алёша Алеша Коля "
    "Петя Ваня Ванечка Вова Володя Гриша Паша Костя Стёпа Степа Лёва Лева Сева Гоша Жора "
    "Данила Савва Кузьма Фома Лука Тёма Тема Рома Гена Федя Витя Толя Боря Слава Кирюша "
    "Андрюша Тимоша Матвейка Егорка"
).split()
_NAMES_CONSONANT = (
    "Александр Алексей Андрей Антон Артём Артем Артур Борис Вадим Валентин Валерий Василий "
    "Виктор Виталий Владимир Владислав Вячеслав Геннадий Георгий Глеб Григорий Даниил Данил "
    "Денис Дмитрий Евгений Егор Иван Игорь Кирилл Константин Леонид Максим Марк Матвей Михаил "
    "Николай Олег Павел Пётр Петр Роман Руслан Сергей Станислав Степан Тимофей Тимур Фёдор "
    "Федор Филипп Эдуард Юрий Ярослав Арсений Богдан Давид Захар Макар Мирон Назар Платон "
    "Родион Семён Семен Святослав Тихон Эмиль Рустам Ринат Марат Ильдар Айдар Дамир Амир "
    "Мирослав Всеволод Герман Демид Елисей Лев Любовь Нинель Эстер"
).split()
# Имена, совпадающие с обычными словами: считаются именем только не в начале предложения.
_AMBIGUOUS = {
    "Вера",
    "Надежда",
    "Любовь",
    "Роза",
    "Лев",
    "Слава",
    "Мила",
    "Ника",
    "Майя",
    "Поля",
    "Лука",
}
_NAMES_A += ["Вера", "Надежда", "Роза"]
_END_A = r"(?:а|ы|е|у|ой|ою|и|я|ю|ей|ею|ь|ью)?"
_END_C = r"(?:а|у|ом|ем|ём|е|я|ю|ы|и|ой)?"


def _stem(name: str) -> str:
    """Удаляет конечную а, я или ь из словарного имени для поиска его падежных форм."""
    return name[:-1] if name[-1] in "аяь" else name


def _term_stem(term: str) -> str:
    """Отбрасывает короткое падежное окончание русского слова для поиска защищённых терминов."""
    return term[:-1] if len(term) > 3 and term[-1].lower() in "аяьыиеоуюй" else term


_NAME = re.compile(
    r"(?<![\w\[])(?:"
    + "|".join(sorted({re.escape(_stem(n)) + _END_A for n in _NAMES_A}, key=len, reverse=True))
    + "|"
    + "|".join(
        sorted(
            {
                re.escape(_stem(n) if n[-1] in "йь" else n)
                + (r"(?:й|я|ю|ем|е|ь|и)?" if n[-1] in "йь" else _END_C)
                for n in _NAMES_CONSONANT
            },
            key=len,
            reverse=True,
        )
    )
    + r")(?![\w\]])"
)
_PATRONYMIC = (
    r"[А-ЯЁ][а-яё]+(?:(?:ович|евич|ьич|ич)(?:а|у|ем|е)?|(?:овн|евн|ичн|иничн)(?:а|ы|е|у|ой))"
)
# «меня зовут Гульнара», «мужа зовут Тимур», «мой врач Соколова», «доктор Иван Петрович».
_TRIGGERED = re.compile(
    r"(?P<lead>\b(?i:зовут|звать|имя|муж(?:а|у|ем|е)?|супруг(?:а|у|ом|е)?|парень|парн(?:я|ю|ем)|"
    r"сын(?:а|у|ом|е|очек|очка)?|доч(?:ь|ери|ерью|ка|ку|ки|кой|ке|енька|еньку)|малыш(?:а|у|ом|ка|ку)?|"
    r"мам(?:а|у|ы|ой|е)|пап(?:а|у|ы|ой|е)|свекров(?:ь|и|ью)|т[её]щ(?:а|у|и|ей|е)|"
    r"сестр(?:а|у|ы|ой|е)|брат(?:а|у|ом|е)?|подруг(?:а|у|и|ой|е)|бабушк(?:а|у|и|ой|е)|"
    r"врач(?:а|у|ом|е)?|доктор(?:а|у|ом|е)?|акушер\w*|гинеколог\w*|психолог\w*|психиатр\w*|"
    r"заведующ\w+|медсестр\w+|фамилия)\s+(?:—\s*|-\s*|по имени\s+)?)"
    r"(?P<name>[А-ЯЁ][а-яё]{1,20}(?:\s+[А-ЯЁ][а-яё]{1,20}){0,2})"
)
_FULL_NAME = re.compile(
    r"(?<![\w\[])(?:[А-ЯЁ][а-яё]{1,20}\s+)?[А-ЯЁ][а-яё]{1,20}\s+" + _PATRONYMIC + r"(?![\w\]])"
)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINK = re.compile(
    r"(?:https?://|www\.)\S+|(?<![\w@])@[A-Za-z][\w.]{2,}|\b(?:t\.me|vk\.com|instagram\.com|wa\.me)/\S+"
)
_PHONE = re.compile(
    r"(?<![\w.\[])(?:\+7|8|7)[\s(\-]*\d{3}[\s)\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
)
_SNILS = re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[ -]\d{2}(?!\d)")
_MARKED_NUMBER = re.compile(
    r"(?P<lead>\b(?:паспорт\w*|снилс\w*|полис\w*|омс|инн|медкарт\w*|карт[аеыу]|договор\w*|"
    r"истори\w+ болезни|обменн\w+ карт\w+|сери[яию]|номер\w*|№)\s*(?:№\s*)?:?\s*)"
    r"(?P<number>\d[\d\s\-]{2,24}\d)",
    flags=re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"(?<![\w.\[])\d(?:[\s\-]?\d){8,}(?![\w\]])")
_ADDRESS = re.compile(
    r"\b(?:ул\.|улиц[аеуыей]+|просп(?:ект\w*|\.)|пр-т|пер(?:еулок\w*|\.)|шоссе|бульвар\w*|"
    r"наб(?:ережн\w+|\.)|площад\w+|микрорайон\w*|мкр\.?)\s+[А-ЯЁ0-9][\w\-]*(?:\s+[А-ЯЁ][\w\-]*)?"
    r"(?:,?\s*(?:д\.|дом)\s*\d+[\w/]*)?(?:,?\s*(?:корп\.|корпус|стр\.)\s*\d+)?"
    r"(?:,?\s*(?:кв\.|квартира)\s*\d+)?",
)
_FACILITY = re.compile(
    r"\b(?:роддом\w*|род\.\s?дом\w*|женск\w+ консультаци\w+|жк|поликлиник\w+|больниц\w+|"
    r"клиник\w+|перинатальн\w+ центр\w*|медцентр\w*|гкб|црб)\s*"
    r"(?:№\s*\d+|N\s*\d+|«[^»]{2,60}»|\"[^\"]{2,60}\"|им(?:ени|\.)\s+[А-ЯЁ][а-яё]+)",
    flags=re.IGNORECASE,
)
_BIRTH_DATE = re.compile(
    r"(?P<lead>\b(?:дата рождения|родилась|год рождения|г\.\s?р\.|д\.\s?р\.)\s*:?\s*)"
    r"(?P<date>\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}\s+[а-яё]+\s+\d{4}|\d{4})",
    flags=re.IGNORECASE,
)
_ANY_DATE = re.compile(r"(?<![\w.])\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}(?!\d)")
# Слово с заглавной буквы не в начале предложения (уровень strict).
_CAPITALIZED = re.compile(r"(?<=[а-яёa-z,;:—–\-»\"] )(?<![.!?] )[А-ЯЁ][а-яё]{2,20}(?![\w\]])")
_SENTENCE_START = re.compile(r"(?:^|[.!?…]\s+|\n\s*)$")
_ALIAS = re.compile(
    r"\[(?:ИМЯ|ТЕЛЕФОН|ПОЧТА|ССЫЛКА|АДРЕС|УЧРЕЖДЕНИЕ|НОМЕР|ДАТА|СКРЫТО)_\d+(?:\.\d+)?\]"
)
# Слова с заглавной буквы, которые не являются личными данными.
_NOT_PERSONAL = {
    "России",
    "Россия",
    "Москва",
    "Москве",
    "Москвы",
    "Бог",
    "Боже",
    "Господи",
    "Интернет",
    "Госуслуги",
    "Госуслугах",
    "Новый",
    "Рождество",
    "Пасха",
}


@dataclass
class PrivacyReport:
    enabled: bool
    level: str = "standard"
    policy: str = POLICY
    entities: dict[str, int] = field(default_factory=dict)
    restored_in_reply: int = 0

    def as_dict(self) -> dict:
        """Возвращает режим и счётчики замен по типам без найденных персональных значений."""
        return {
            "enabled": self.enabled,
            "level": self.level,
            "policy": self.policy,
            # Только количество по типам: значения в отчёт и журналы не попадают.
            "entities": dict(sorted(self.entities.items())),
            "restored_in_reply": self.restored_in_reply,
        }


class PrivacyBoundary:
    """Обратимая псевдонимизация в пределах одного запроса."""

    def __init__(
        self, enabled=True, level="standard", protected_terms=(), public_numbers=(), known_names=()
    ):
        """Создаёт обратимую замену персональных данных для одного запроса.

        protected_terms задают дополнительные скрываемые слова, public_numbers —
        номера, которые нужно оставить, known_names — ранее встреченные имена.
        Таблицы псевдонимов и исходных форм хранятся в памяти экземпляра.
        """
        self.enabled = bool(enabled)
        self.level = level if level in LEVELS else "standard"
        self._by_entity: dict[tuple[str, str], int] = {}
        self._forms: dict[tuple[str, int], list[str]] = {}
        self._alias_to_text: dict[str, str] = {}
        self._text_to_alias: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        # Имена, уже встреченные в этом запросе: их остальные упоминания (в ответах
        # помощника, в памяти) скрываются и без подсказывающих слов.
        self._learned: dict[str, re.Pattern] = {}
        self._public = {re.sub(r"\D", "", n)[-10:] for n in public_numbers if n}
        terms = sorted(
            {t.strip() for t in protected_terms if t and len(t.strip()) >= 2}, key=len, reverse=True
        )
        # Слова из кабинета ищутся без учёта регистра, с падежным окончанием для одного слова.
        self._terms = [
            (
                term,
                re.compile(
                    r"(?<![\w\[])"
                    + (
                        re.escape(_term_stem(term)) + r"[а-яё]{0,3}"
                        if re.fullmatch(r"[А-Яа-яЁё]{3,}", term)
                        else re.escape(term)
                    )
                    + r"(?![\w\]])",
                    flags=re.IGNORECASE,
                ),
            )
            for term in terms
        ]
        self.report = PrivacyReport(enabled=self.enabled, level=self.level)
        for name in known_names:
            self._learn(str(name))

    def _learn(self, surface):
        """Запоминает основу допустимого имени и шаблон поиска его падежных форм."""
        for word in surface.split():
            stem = _NAME_ENDINGS.sub("", word)
            if len(stem) >= 4 and word not in _AMBIGUOUS and stem not in self._learned:
                self._learned[stem] = re.compile(
                    r"(?<![\w\[])" + re.escape(stem) + r"[а-яё]{0,3}(?![\w\]])"
                )

    def learned_names(self):
        """Возвращает первые исходные формы имён, которым в этом запросе назначены псевдонимы."""
        return [forms[0] for (kind, _), forms in self._forms.items() if kind == "ИМЯ"]

    # ------------------------------------------------------------------ aliases
    def _alias(self, kind: str, surface: str, entity_key: str | None = None) -> str:
        """Назначает обозначение сущности и запоминает исходный текст для восстановления.

        Одинаковая сущность получает один номер; другая падежная форма — суффикс
        формы. Счётчик отчёта повышает только при появлении новой сущности.
        """
        if surface in self._text_to_alias:
            return self._text_to_alias[surface]
        key = (kind, (entity_key or surface).lower())
        if key not in self._by_entity:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            self._by_entity[key] = self._counters[kind]
            self.report.entities[_KIND_CODES[kind]] = (
                self.report.entities.get(_KIND_CODES[kind], 0) + 1
            )
        index = self._by_entity[key]
        forms = self._forms.setdefault((kind, index), [])
        forms.append(surface)
        # Другая падежная форма того же имени получает тот же номер с суффиксом формы.
        alias = f"[{kind}_{index}]" if len(forms) == 1 else f"[{kind}_{index}.{len(forms)}]"
        if kind == "ИМЯ":
            self._learn(surface)
        self._alias_to_text[alias] = surface
        self._text_to_alias[surface] = alias
        return alias

    def _sub(
        self, pattern, kind, text, group=None, lead=None, entity=None, accept=None, split=False
    ):
        """Заменяет совпадения шаблона псевдонимами заданного типа.

        group выбирает скрываемую группу, lead сохраняет вводный текст, entity
        задаёт ключ объединения форм, accept может отклонить совпадение.
        При split выдаёт отдельный псевдоним каждому слову.
        """

        def replace(match):
            """Преобразует одно принятое совпадение в псевдоним, сохраняя разрешённый префикс."""
            value = match.group(group) if group else match.group(0)
            if accept is not None and not accept(match, value, text):
                return match.group(0)
            prefix = match.group(lead) if lead else ""
            if split:
                # «Мария Соколова» — два псевдонима: модель обратится по имени, а не по ФИО.
                return prefix + " ".join(
                    self._alias(kind, word, entity(word) if entity else None)
                    for word in value.split()
                )
            return prefix + self._alias(kind, value, entity(value) if entity else None)

        return pattern.sub(replace, text)

    # ------------------------------------------------------------------ public
    def mask(self, text):
        """Заменяет распознанные персональные значения и пользовательские термины псевдонимами.

        Обновляет таблицу восстановления и счётчики. В strict дополнительно скрывает
        даты и некоторые слова с заглавной буквы; разрешённые публичные номера
        сохраняет. При выключенном режиме или пустом тексте возвращает исходную строку.
        """
        if not self.enabled or not text:
            return text
        for term, pattern in self._terms:
            text = self._sub(pattern, "СКРЫТО", text, entity=lambda _value, term=term: term)
        text = self._sub(_EMAIL, "ПОЧТА", text)
        text = self._sub(_LINK, "ССЫЛКА", text)
        text = self._sub(
            _PHONE,
            "ТЕЛЕФОН",
            text,
            entity=lambda v: re.sub(r"\D", "", v)[-10:],
            accept=lambda _m, v, _t: re.sub(r"\D", "", v)[-10:] not in self._public,
        )
        text = self._sub(_SNILS, "НОМЕР", text)
        text = self._sub(_MARKED_NUMBER, "НОМЕР", text, group="number", lead="lead")
        text = self._sub(
            _LONG_NUMBER,
            "НОМЕР",
            text,
            accept=lambda _m, v, _t: re.sub(r"\D", "", v)[-10:] not in self._public,
        )
        text = self._sub(_ADDRESS, "АДРЕС", text)
        text = self._sub(_FACILITY, "УЧРЕЖДЕНИЕ", text)
        text = self._sub(_BIRTH_DATE, "ДАТА", text, group="date", lead="lead")
        text = self._sub(_FULL_NAME, "ИМЯ", text)
        text = self._sub(
            _TRIGGERED, "ИМЯ", text, group="name", lead="lead", entity=_name_key, split=True
        )
        text = self._sub(_NAME, "ИМЯ", text, entity=_name_key, accept=_plausible_name)
        for pattern in list(self._learned.values()):
            text = self._sub(pattern, "ИМЯ", text, entity=_name_key)
        if self.level == "strict":
            text = self._sub(_ANY_DATE, "ДАТА", text)
            text = self._sub(
                _CAPITALIZED,
                "СКРЫТО",
                text,
                entity=_name_key,
                accept=lambda _m, v, _t: v not in _NOT_PERSONAL,
            )
        return text

    def mask_messages(self, messages):
        """Возвращает копии сообщений с обезличенным content.

        Первым проходом собирает имена из всей последовательности, чтобы скрыть
        упоминания, встречавшиеся до знакомства с именем. Остальные поля сохраняет.
        """
        if self.enabled:
            # Первый проход только собирает имена: имя может быть названо позже, чем упомянуто.
            for message in messages:
                self.mask(message["content"])
        return [{**m, "content": self.mask(m["content"])} for m in messages]

    def unmask(self, text):
        """Восстанавливает известные обозначения в ответе и считает восстановленные вхождения."""
        if not self.enabled or not text or not self._alias_to_text:
            return text

        def restore(match):
            """Восстанавливает точный псевдоним или его известную базовую форму; чужой оставляет."""
            alias = match.group(0)
            if alias in self._alias_to_text:
                self.report.restored_in_reply += 1
                return self._alias_to_text[alias]
            # Модель могла употребить базовый псевдоним без суффикса формы.
            base = re.sub(r"\.\d+\]$", "]", alias)
            if base in self._alias_to_text:
                self.report.restored_in_reply += 1
                return self._alias_to_text[base]
            return alias

        return _ALIAS.sub(restore, text)

    def preview(self, text):
        """Возвращает обезличенный текст и отчёт замен, обновляя только состояние экземпляра."""
        masked = self.mask(text)
        return {"text": masked, **self.report.as_dict()}


_KIND_CODES = {
    "ИМЯ": "name",
    "ТЕЛЕФОН": "phone",
    "ПОЧТА": "email",
    "ССЫЛКА": "link",
    "АДРЕС": "address",
    "УЧРЕЖДЕНИЕ": "facility",
    "НОМЕР": "number",
    "ДАТА": "date",
    "СКРЫТО": "protected_term",
}
_NAME_ENDINGS = re.compile(r"(?:ою|ею|ой|ей|ом|ем|ём|ью|[аыеуияюьй])$")


def _name_key(value: str) -> str:
    """Строит грубый ключ имени без падежного окончания для объединения его форм."""
    return " ".join(_NAME_ENDINGS.sub("", part.lower()) or part.lower() for part in value.split())


def _plausible_name(match, value, text) -> bool:
    """Отсеивает неоднозначные слова вроде «Вера» в начале предложения по контексту совпадения."""
    if _stem(value) in {_stem(a) for a in _AMBIGUOUS} or value in _AMBIGUOUS:
        # «Вера в себя» в начале предложения — слово, не имя.
        return not _SENTENCE_START.search(text[: match.start()])
    return True


def boundary_for(server_settings, user_preferences: dict | None, public_numbers=()):
    """Собирает правила замены по серверной политике и предпочтениям профиля.

    always и never перекрывают пользовательский переключатель; user учитывает
    его значение. Передаёт защищённые слова, запомненные имена и публичные номера.
    """
    privacy = (user_preferences or {}).get("privacy") or {}
    mode = getattr(server_settings, "anonymize_outbound", "user")
    enabled = (
        True if mode == "always" else False if mode == "never" else privacy.get("anonymize", True)
    )
    return PrivacyBoundary(
        enabled=enabled,
        level=privacy.get("level", "standard"),
        protected_terms=privacy.get("protected_terms") or (),
        public_numbers=public_numbers,
        known_names=(user_preferences or {}).get("learned_names") or (),
    )
