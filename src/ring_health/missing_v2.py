"""Observed-only, causal features with explicit masks and observation age."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SIGNALS


def causal_features(d):
    out=pd.DataFrame(index=d.index)
    # Steps are full-day activity and unavailable at morning analysis. They are
    # intentionally omitted, rather than leaking today's future activity.
    for c in [x for x in SIGNALS if x!="steps"]:
        x=pd.to_numeric(d[c],errors="coerce") if c in d else pd.Series(np.nan,index=d.index)
        baseline=x.shift(1).rolling(28,min_periods=3).median()
        out[c+"_delta"]=x-baseline
        out[c+"_change3"]=x.rolling(3,min_periods=1).mean()-x.shift(3).rolling(7,min_periods=2).median()
        out[c+"_std14"]=x.rolling(14,min_periods=3).std()
        out[c+"_count14"]=x.notna().rolling(14,min_periods=1).sum()
        index=pd.Series(np.arange(len(x)),index=x.index,dtype=float)
        last=index.where(x.notna()).ffill()
        out[c+"_age"]=index-last
    return out


def mask_measurements(d,regime,rng):
    d=d.copy()
    n=len(d)
    signals=[c for c in SIGNALS if c in d]
    rate={"complete":0.,"random10":.1,"random30":.3,"random50":.5,"random70":.7}.get(regime,0.)
    if rate:d.loc[rng.random(n)<rate,signals]=np.nan
    if regime.startswith("block"):
        size=int(regime.replace("block",""))
        start=int(rng.integers(0,max(1,n-size+1)))
        d.loc[d.index[start:start+size],signals]=np.nan
    if regime=="temperature_only":d.loc[:,[c for c in signals if c!="temperature_deviation"]]=np.nan
    if regime=="no_temperature":d.loc[:,"temperature_deviation"]=np.nan
    if regime=="metric_dropout":
        for c in signals:
            d.loc[rng.random(n)<rng.uniform(.1,.6),c]=np.nan
    if regime=="illness_nonwear":
        abnormal=d.temperature_deviation.abs()>.4
        mask=(rng.random(n)<.15)|(abnormal&(rng.random(n)<.65))
        d.loc[mask,signals]=np.nan
    return d


def normalize_oura(payload):
    from .schema_v2 import Daily,calendar_date
    records={}
    for endpoint in ["daily_readiness","daily_activity"]:
        seen=set()
        for r in payload.get(endpoint,[]):
            day=r.get("day")
            calendar_date(day)
            if day in seen:raise ValueError("Duplicate day in "+endpoint)
            seen.add(day)
            row=records.setdefault(day,{"date":day})
            if endpoint=="daily_readiness":row["temperature_deviation"]=r.get("temperature_deviation")
            else:row["steps"]=r.get("steps")
    best={}
    for r in payload.get("sleep",[]):
        day=r.get("day")
        calendar_date(day)
        if r.get("type")!="long_sleep":continue
        duration=r.get("total_sleep_duration")
        if duration is not None and (isinstance(duration,bool) or not isinstance(duration,(int,float)) or not np.isfinite(duration)):raise ValueError("Invalid sleep duration")
        if day not in best or (duration or 0)>(best[day].get("total_sleep_duration") or 0):best[day]=r
    for day,r in best.items():
        row=records.setdefault(day,{"date":day})
        row.update(resting_hr=r.get("lowest_heart_rate"),hrv_ms=r.get("average_hrv"),respiratory_rate=r.get("average_breath"),sleep_minutes=r.get("total_sleep_duration")/60 if r.get("total_sleep_duration") is not None else None,sleep_efficiency=r.get("efficiency"))
    # Union of dates keeps readiness-only temperatures and activity-only days.
    return [Daily.model_validate(records[k]).model_dump(mode="json") for k in sorted(records)]
