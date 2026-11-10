"""Strict request contract v3 (pydantic). Dates are local calendar dates YYYY-MM-DD.

Missing values: use null or omit. Zero is a measurement, never "missing". Nothing is imputed.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def calendar_date(v):
    if isinstance(v, date):
        return v
    if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        raise ValueError("Expected local date YYYY-MM-DD")
    return date.fromisoformat(v)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _num(v):
    if v is None:
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError("Expected finite number or null")
    return v


class Night(Strict):
    date: date
    temperature_deviation: float | None = Field(default=None, ge=-3, le=3, description="Nightly skin temperature deviation from personal baseline, deg C (Oura daily_readiness.temperature_deviation)")
    resting_hr: float | None = Field(default=None, ge=25, le=180, description="Lowest/resting nightly HR, bpm (Oura sleep.lowest_heart_rate)")
    hrv_ms: float | None = Field(default=None, ge=1, le=400, description="Nightly average RMSSD, ms (Oura sleep.average_hrv)")
    respiratory_rate: float | None = Field(default=None, ge=5, le=40, description="breaths/min (Oura sleep.average_breath)")
    sleep_minutes: float | None = Field(default=None, ge=0, le=1440)
    sleep_efficiency: float | None = Field(default=None, ge=0, le=100)
    awake_minutes: float | None = Field(default=None, ge=0, le=1000, description="Awake time inside the sleep period, minutes (Oura sleep.awake_time/60)")

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v):
        return calendar_date(v)

    @field_validator("temperature_deviation", "resting_hr", "hrv_ms", "respiratory_rate", "sleep_minutes", "sleep_efficiency", "awake_minutes", mode="before")
    @classmethod
    def _n(cls, v):
        return _num(v)


class Profile(Strict):
    age: float = Field(ge=16, le=70)
    pregnant: bool = False
    breastfeeding: bool = False
    hormonal_contraception: bool = False
    menopausal_hormone_therapy: bool = False
    hysterectomy: bool = False
    bilateral_oophorectomy: bool = False
    smoker: bool | None = None
    bmi: float | None = Field(default=None, ge=12, le=70)
    parous: bool | None = None
    college_education: bool | None = None

    @field_validator("age", "bmi", mode="before")
    @classmethod
    def _n(cls, v):
        return _num(v)


class Cycle(Strict):
    period_starts: list[date] = Field(default_factory=list, max_length=60)
    last_period_date: date | None = Field(default=None, description="Onboarding answer: first day of the most recent period (may be years ago)")
    period_logs_complete: bool = Field(default=False, description="True only if every period start in the history is logged")
    logging_since: date | None = Field(default=None, description="Date from which periods are logged in the app")
    menses_length: int | None = Field(default=None, ge=1, le=12)

    @field_validator("period_starts", mode="before")
    @classmethod
    def _l(cls, v):
        if not isinstance(v, list):
            raise ValueError("Expected array of dates")
        return [calendar_date(x) for x in v]

    @field_validator("last_period_date", "logging_since", mode="before")
    @classmethod
    def _d(cls, v):
        return None if v is None else calendar_date(v)


class LHTest(Strict):
    date: date
    result: Literal["positive", "negative"]

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v):
        return calendar_date(v)


class Symptoms(Strict):
    logging_enabled: bool = Field(default=False, description="True if the user logs symptoms; then absence of a log means 'no symptom'")
    hot_flash_dates: list[date] = Field(default_factory=list, max_length=400)
    night_sweat_dates: list[date] = Field(default_factory=list, max_length=400)

    @field_validator("hot_flash_dates", "night_sweat_dates", mode="before")
    @classmethod
    def _l(cls, v):
        return [calendar_date(x) for x in v]


class Lab(Strict):
    value: float = Field(gt=0)
    date: date

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v):
        return calendar_date(v)


class Labs(Strict):
    fsh_iu_l: Lab | None = None
    amh_pg_ml: Lab | None = None


class OuraExport(Strict):
    sleep: list[dict] = Field(default_factory=list, max_length=4000)
    daily_readiness: list[dict] = Field(default_factory=list, max_length=800)


class AnalyzeRequest(Strict):
    as_of: date
    modules: list[Literal["fertility", "menopause"]] = Field(default_factory=lambda: ["fertility", "menopause"])
    profile: Profile
    cycle: Cycle = Field(default_factory=Cycle)
    lh_tests: list[LHTest] = Field(default_factory=list, max_length=400)
    symptoms: Symptoms = Field(default_factory=Symptoms)
    labs: Labs = Field(default_factory=Labs)
    nights: list[Night] | None = Field(default=None, max_length=800)
    oura: OuraExport | None = None

    @field_validator("as_of", mode="before")
    @classmethod
    def _d(cls, v):
        return calendar_date(v)

    @model_validator(mode="after")
    def _check(self):
        if self.nights is not None and self.oura is not None:
            raise ValueError("Supply nights or oura, not both")
        ds = [n.date for n in self.nights or []]
        if len(ds) != len(set(ds)):
            raise ValueError("Duplicate night dates")
        if ds and max(ds) > self.as_of:
            raise ValueError("Night after as_of")
        if ds and (self.as_of - min(ds)).days > 800:
            raise ValueError("History longer than 800 days")
        ps = self.cycle.period_starts
        if len(ps) != len(set(ps)) or any(x > self.as_of for x in ps):
            raise ValueError("Duplicate or future period start")
        if self.cycle.last_period_date and self.cycle.last_period_date > self.as_of:
            raise ValueError("last_period_date after as_of")
        if any(t.date > self.as_of for t in self.lh_tests):
            raise ValueError("LH test after as_of")
        if self.cycle.logging_since and self.cycle.logging_since > self.as_of:
            raise ValueError("logging_since after as_of")
        for lab in [self.labs.fsh_iu_l, self.labs.amh_pg_ml]:
            if lab is not None and lab.date > self.as_of:
                raise ValueError("Lab date after as_of")
        if any(x > self.as_of for x in self.symptoms.hot_flash_dates + self.symptoms.night_sweat_dates):
            raise ValueError("Symptom date after as_of")
        return self
