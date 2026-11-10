"""Strict mobile/backend contract. Dates are local calendar dates, not epochs."""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def calendar_date(value):
    if not isinstance(value,str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}",value):
        raise ValueError("Expected exact local date YYYY-MM-DD")
    return date.fromisoformat(value)


class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)


class Daily(StrictModel):
    date: date
    temperature_deviation: float|None=Field(default=None,ge=-3,le=3)
    resting_hr: float|None=Field(default=None,ge=25,le=180)
    hrv_ms: float|None=Field(default=None,ge=0,le=400)
    respiratory_rate: float|None=Field(default=None,ge=5,le=40)
    sleep_minutes: float|None=Field(default=None,ge=0,le=1440)
    sleep_efficiency: float|None=Field(default=None,ge=0,le=100)
    steps: float|None=Field(default=None,ge=0,le=100000)

    @field_validator("date",mode="before")
    @classmethod
    def validate_date(cls,v):return calendar_date(v)

    @field_validator("temperature_deviation","resting_hr","hrv_ms","respiratory_rate","sleep_minutes","sleep_efficiency","steps",mode="before")
    @classmethod
    def validate_number(cls,v):
        if v is None:return v
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError("Expected finite JSON number or null")
        return v


class Context(StrictModel):
    age: float=Field(ge=18,le=65)
    pregnant: bool=Field(strict=True)
    breastfeeding: bool=Field(strict=True)
    hormonal_medication: bool=Field(strict=True)
    hysterectomy: bool=Field(strict=True)
    bilateral_oophorectomy: bool=Field(strict=True)
    period_start_dates: list[date]=Field(default_factory=list,max_length=24)
    cycle_day: int|None=Field(default=None,ge=1,le=90,strict=True)
    period_logs_complete: bool=Field(default=False,strict=True)

    @field_validator("period_start_dates",mode="before")
    @classmethod
    def validate_starts(cls,v):
        if not isinstance(v,list):raise ValueError("Expected array of dates")
        return [calendar_date(x) for x in v]

    @field_validator("age",mode="before")
    @classmethod
    def validate_age(cls,v):
        if isinstance(v,bool) or not isinstance(v,(int,float)):raise ValueError("Expected numeric age")
        return v


class OuraExport(StrictModel):
    sleep: list[dict]=Field(default_factory=list,max_length=3000)
    daily_readiness: list[dict]=Field(default_factory=list,max_length=730)
    daily_activity: list[dict]=Field(default_factory=list,max_length=730)


class AnalyzeRequest(StrictModel):
    as_of: date
    context: Context
    daily: list[Daily]|None=Field(default=None,max_length=730)
    oura: OuraExport|None=None

    @field_validator("as_of",mode="before")
    @classmethod
    def validate_as_of(cls,v):return calendar_date(v)

    @model_validator(mode="after")
    def validate_source(self):
        if (self.daily is None)==(self.oura is None):raise ValueError("Supply exactly one of daily or oura")
        dates=[x.date for x in self.daily or []]
        if len(dates)!=len(set(dates)):raise ValueError("Duplicate daily dates")
        if dates and ((self.as_of-min(dates)).days>730 or max(dates)>self.as_of):raise ValueError("Dates after as_of or history longer than730days")
        starts=self.context.period_start_dates
        if len(starts)!=len(set(starts)) or any(x>self.as_of for x in starts):raise ValueError("Duplicate/future period starts")
        return self
