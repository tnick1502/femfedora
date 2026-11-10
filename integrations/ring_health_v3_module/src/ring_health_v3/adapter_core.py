"""Adapter for the femfedora core risk module: component.compute(values, ctx) -> card.

Core contract (core_architecture v0.5): ML components run in SHADOW mode; insufficient_data -> task in todo;
invalid clinical input -> MANUAL_REVIEW; unexpected failure -> COMPONENT_ERROR only for this card.
values: {"nights": [...], "period_starts": [...], "last_period_date": ..., "lh_tests": [...], "symptoms": {...}, "labs": {...}}
ctx:    {"as_of": "YYYY-MM-DD", "profile": {"age": .., flags..}}
"""
from __future__ import annotations

from .engine import RingHealthV3

_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RingHealthV3()
    return _ENGINE


def _profile(ctx_profile):
    from .schema import Profile
    return {k: v for k, v in (ctx_profile or {}).items() if k in Profile.model_fields}  # core profiles may carry extra keys


def _payload(values, ctx, module):
    cyc = {"period_starts": values.get("period_starts", []), "last_period_date": values.get("last_period_date"),
           "period_logs_complete": bool(values.get("period_logs_complete", False)), "logging_since": values.get("logging_since"),
           "menses_length": values.get("menses_length")}
    p = {"as_of": ctx["as_of"], "modules": [module], "profile": _profile(ctx.get("profile")), "cycle": {k: v for k, v in cyc.items() if v is not None},
         "lh_tests": values.get("lh_tests", []), "symptoms": values.get("symptoms", {}), "labs": values.get("labs", {})}
    if values.get("oura") is not None:
        p["oura"] = values["oura"]
    else:
        p["nights"] = values.get("nights", [])
    return p


def _card(component, status, result=None, missing=None, error=None):
    card = {"component": component, "mode": "SHADOW", "status": status, "provenance": "synthetic_trained_demo_model",
            "clinical_validation": "not_established", "result": result}
    if missing:
        card["insufficient_data"] = missing
    if error:
        card["error"] = error
    return card


class RingFertilityComponent:
    id = "ring.fertility.v3"

    def compute(self, values, ctx):
        try:
            r = engine().analyze(_payload(values, ctx, "fertility"))["fertility"]
        except ValueError as e:
            return _card(self.id, "MANUAL_REVIEW", error=str(e))
        except Exception as e:  # noqa: BLE001
            return _card(self.id, "COMPONENT_ERROR", error=type(e).__name__)
        if r["status"] == "ok":
            return _card(self.id, "ok", r)
        if r["status"] in ("anchor_required", "anchor_stale"):
            return _card(self.id, "insufficient_data", r, missing=["cycle.period_start (most recent)"])
        return _card(self.id, "not_applicable", r)


class RingMenopauseComponent:
    id = "ring.menopause.v3"

    def compute(self, values, ctx):
        try:
            r = engine().analyze(_payload(values, ctx, "menopause"))["menopause"]
        except ValueError as e:
            return _card(self.id, "MANUAL_REVIEW", error=str(e))
        except Exception as e:  # noqa: BLE001
            return _card(self.id, "COMPONENT_ERROR", error=type(e).__name__)
        if r["status"] == "ok":
            return _card(self.id, "ok", r)
        if r["status"] == "insufficient_data":
            return _card(self.id, "insufficient_data", r, missing=["cycle.last_period_date", "ring.nights>=20"])
        return _card(self.id, "not_applicable", r)
