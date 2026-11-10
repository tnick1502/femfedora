"""Versioned source material. Source verification is not clinical product approval."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from emo_core.core.exceptions import InvalidMaterial

EXPECTED = {"GAD2": (2, 6), "GAD7": (7, 21), "EPDS": (10, 30), "PHQ9": (9, 27)}
MATERIAL_FIELDS = (
    "code",
    "version",
    "language",
    "definition",
    "scoring_rules",
    "license_reference",
)


def material_hash(data):
    """Вычисляет SHA-256 редакции по текстам, правилам, языку и основанию использования."""
    material = {key: data.get(key) for key in MATERIAL_FIELDS}
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache
def bundled_hashes():
    """Читает встроенные JSON-редакции и кэширует множество их контрольных сумм."""
    directory = Path(__file__).resolve().parents[1] / "questionnaires"
    return frozenset(material_hash(json.loads(p.read_text())) for p in directory.glob("*.json"))


def validate_material(data):
    """Проверяет редакцию шкалы и возвращает исходный словарь без изменений.

    Требует документированного одобрения либо совпадения с проверенной встроенной
    редакцией. Проверяет число и идентификаторы пунктов, варианты, ключи, диапазон
    порогов и независимые safety-пункты. При нарушении правил поднимает InvalidMaterial;
    неполная структура словаря может вызвать KeyError.
    """
    definition, rules = data["definition"], data["scoring_rules"]
    if data["code"] not in EXPECTED:
        raise InvalidMaterial("Неизвестная шкала")
    approved = all(
        (
            definition.get("clinical_review_status") == "approved",
            definition.get("reviewed_by"),
            definition.get("approval_reference"),
        )
    )
    source_verified = (
        definition.get("source_review_status") == "verified"
        and definition.get("source_review_reference")
        and material_hash(data) in bundled_hashes()
    )
    if not data.get("license_reference") or not (approved or source_verified):
        raise InvalidMaterial("Требуются основание использования и проверенная редакция")
    count, maximum = EXPECTED[data["code"]]
    items = definition.get("items", [])
    if len(items) != count or len({i["id"] for i in items}) != count:
        raise InvalidMaterial("Неверное число или повторение пунктов")
    for item in items:
        options, scores = item.get("options", []), item.get("score_map", [])
        if (
            not isinstance(item.get("id"), str)
            or not item.get("text")
            or len(options) != 4
            or not all(isinstance(o, str) and o for o in options)
            or any(type(s) is not int for s in scores)
            or sorted(scores) != [0, 1, 2, 3]
        ):
            raise InvalidMaterial("Нужны четыре варианта и проверенный ключ каждого пункта")
    auxiliary = definition.get("auxiliary_items", [])
    ids = {item["id"] for item in items}
    aux_ids = {item["id"] for item in auxiliary}
    if ids & aux_ids or len(aux_ids) != len(auxiliary):
        raise InvalidMaterial("Повторение дополнительных пунктов")
    for item in auxiliary:
        if (
            not isinstance(item.get("id"), str)
            or not item.get("text")
            or len(item.get("options", [])) != 4
            or not all(isinstance(o, str) and o for o in item.get("options", []))
            or item.get("scored") is not False
        ):
            raise InvalidMaterial("Дополнительные пункты не должны входить в сумму")
    if rules.get("maximum") != maximum:
        raise InvalidMaterial("Неверная максимальная сумма")
    followup = rules.get("followup_score")
    if followup is not None and (type(followup) is not int or not 0 <= followup <= maximum):
        raise InvalidMaterial("Некорректный порог следующего шага")
    thresholds = rules.get("thresholds", [])
    if thresholds:
        expected_minimum = 0
        for band in thresholds:
            low, high = band.get("minimum"), band.get("maximum")
            if (
                type(low) is not int
                or type(high) is not int
                or low != expected_minimum
                or high < low
                or high > maximum
                or not band.get("label")
            ):
                raise InvalidMaterial("Пороги должны без пропусков покрывать диапазон шкалы")
            expected_minimum = high + 1
        if expected_minimum != maximum + 1:
            raise InvalidMaterial("Пороги не покрывают диапазон шкалы")
    for rule in rules.get("safety_items", []):
        if rule.get("item_id") not in ids or rule.get("minimum_score") not in (1, 2, 3):
            raise InvalidMaterial("Некорректный пункт безопасности")
    if data["code"] in ("EPDS", "PHQ9") and not rules.get("safety_items"):
        raise InvalidMaterial("Нужна независимая обработка пункта безопасности")
    return data


def questionnaire_view(questionnaire):
    """Собирает снимок редакции для API: материал, UUID, доступность и контрольную сумму."""
    data = {key: getattr(questionnaire, key) for key in MATERIAL_FIELDS}
    return {
        **data,
        "id": str(questionnaire.id),
        "enabled": questionnaire.enabled,
        "definition_hash": material_hash(data),
    }
