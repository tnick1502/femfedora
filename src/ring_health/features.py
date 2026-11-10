"""Causal features. All windows end at the latest observed day.

No centred filters, full-cycle centring, future cycle lengths or outcome fields
are accepted as model inputs. An Oura nightly deviation is not core temperature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = ["temperature_deviation", "resting_hr", "hrv_ms", "respiratory_rate", "sleep_minutes", "sleep_efficiency", "steps"]


def daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Single-person/cycle sequence, ordered and reindexed to calendar days.

    Today's full sleep episode is available at the morning prediction time.
    Baselines use previous nights only; imputation is left to training pipelines.
    """
    d = frame.copy()
    if "date" in d:
        d["date"] = pd.to_datetime(d["date"])
        if d.date.duplicated().any():
            raise ValueError("Duplicate dates; aggregate main sleep sessions first")
        d = d.sort_values("date").set_index("date").asfreq("D").reset_index()
    f = pd.DataFrame(index=d.index)
    for c in SIGNALS:
        if c not in d:
            continue
        x = pd.to_numeric(d[c], errors="coerce")
        past = x.shift(1).rolling(28, min_periods=5)
        base = past.median()
        f[c + "_delta"] = x - base
        f[c + "_mean3_delta"] = x.rolling(3, min_periods=2).mean() - base
        f[c + "_mean7_delta"] = x.rolling(7, min_periods=4).mean() - base
        f[c + "_change3"] = x.rolling(3, min_periods=2).mean() - x.shift(3).rolling(3, min_periods=2).mean()
        f[c + "_std7"] = x.rolling(7, min_periods=4).std()
        f[c + "_observed7"] = x.notna().rolling(7, min_periods=1).mean()
    return f


def window_features(frame: pd.DataFrame, age: float | None = None) -> dict:
    """Summarize up to 90 observed calendar days for a stage demo."""
    d = frame.copy()
    if "date" in d:
        d["date"] = pd.to_datetime(d.date)
        if d.date.duplicated().any():
            raise ValueError("Duplicate dates")
        d = d.sort_values("date").set_index("date").asfreq("D").reset_index()
    d = d.tail(90)
    features = {}
    for c in SIGNALS:
        x = pd.to_numeric(d.get(c, pd.Series(np.nan, index=d.index)), errors="coerce")
        features[c + "_mean"] = float(x.mean())
        features[c + "_std"] = float(x.std())
        features[c + "_iqr"] = float(x.quantile(.75) - x.quantile(.25))
        features[c + "_missing"] = float(x.isna().mean())
    t = pd.to_numeric(d.get("temperature_deviation", pd.Series(np.nan, index=d.index)), errors="coerce")
    # Autocorrelation uses only already-observed history; no gap filling.
    ac = [t.corr(t.shift(k)) for k in range(20, 36) if len(t) > k + 10]
    features["temperature_periodicity"] = float(np.nanmax(ac)) if np.isfinite(ac).any() else np.nan
    features["temperature_successive_diff"] = float(t.diff().abs().median())
    if age is not None:
        features["age"] = float(age)
    return features


def oura_to_daily(payload: dict) -> pd.DataFrame:
    """Normalize already-authorized Oura v2 JSON exports, without network access.

    Choose the longest `long_sleep` episode per Oura day. Oura assigns `day`;
    do not derive it from UTC start times. Units are explicit.
    """
    readiness = {x["day"]: x for x in payload.get("daily_readiness", [])}
    activity = {x["day"]: x for x in payload.get("daily_activity", [])}
    sleeps = {}
    for s in payload.get("sleep", []):
        if s.get("type") != "long_sleep":
            continue
        day = s["day"]
        if day not in sleeps or (s.get("total_sleep_duration") or 0) > (sleeps[day].get("total_sleep_duration") or 0):
            sleeps[day] = s
    rows = []
    for day, s in sorted(sleeps.items()):
        r, a = readiness.get(day, {}), activity.get(day, {})
        rows.append({"date": day, "temperature_deviation": r.get("temperature_deviation"),
                     "resting_hr": s.get("lowest_heart_rate"), "hrv_ms": s.get("average_hrv"),
                     "respiratory_rate": s.get("average_breath"),
                     "sleep_minutes": s.get("total_sleep_duration", np.nan) / 60 if s.get("total_sleep_duration") is not None else np.nan,
                     "sleep_efficiency": s.get("efficiency"), "steps": a.get("steps")})
    return pd.DataFrame(rows, columns=["date"] + SIGNALS)
