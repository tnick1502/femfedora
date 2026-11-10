"""Inspectable preparation and audit of supplied and public datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document

from .features import daily_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/external/ring_health"
PROCESSED = ROOT / "data/processed/ring_health"
RESULTS = ROOT / "research/experiments/ring_health/v1"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))


def prepare_jordan():
    d = pd.read_excel(RAW / "data set menopause.xlsx")
    doc = Document(RAW / "menapouse study questionnaire.docx")
    text = "\n".join(p.text for p in doc.paragraphs)
    text += "\n" + "\n".join(" | ".join(c.text for c in r.cells) for t in doc.tables for r in t.rows)
    (PROCESSED / "questionnaire_extracted.txt").write_text(text)
    cols = d.columns.tolist()
    duplicates = int(d.duplicated().sum())
    clean = d.drop_duplicates().copy()
    # Counts 68/52/80 agree with Table 1 of Assaf et al. Questionnaire order
    # supports the mapping; the original numeric codebook is unavailable.
    # Explicitly named "stage_proxy" to preserve that uncertainty.
    clean["stage_proxy"] = clean["How do you evaluate your health status"] - 1
    clean["participant_id"] = [f"jordan_{i:03d}" for i in clean.index]
    rename = {"Unnamed: 3": "work_type_code", "Increase Bp Thyroid gland": "hypertension_code", "Unnamed: 5": "thyroid_code"}
    clean = clean.rename(columns=rename)
    # Unknown coding and downstream/aggregate fields are not used as inputs.
    features = ["Age", "hypertension_code", "thyroid_code", "Back pain", "DM", "Heart disease", "Varcoise", "Constipation", "Respiratory", "Tumors"]
    clean[["participant_id", "stage_proxy"] + features].to_csv(PROCESSED / "jordan_stage_proxy.csv", index=False)
    aggregate_error = int((d.totalqol != d[["emotional", "sexual", "health", "occupational"]].sum(axis=1)).sum())
    conflicts = int(((d["How do you evaluate your health status"] != 3) & d["Loss menstuartion"].notna()).sum())
    return {"rows_raw": len(d), "columns_raw": len(cols), "exact_duplicate_rows": duplicates,
            "rows_prepared": len(clean), "age_range": [int(d.Age.min()), int(d.Age.max())],
            "stage_raw_counts": {str(k): int(v) for k,v in d["How do you evaluate your health status"].value_counts().items()},
            "missing_per_column": {c:int(v) for c,v in d.isna().sum().items() if v},
            "totalqol_not_equal_sum_subscales": aggregate_error,
            "cause_recorded_outside_stage3": conflicts,
            "qol_response_values": sorted(d[cols[30:53]].stack().unique().tolist()),
            "features_used": features,
            "excluded": ["Loss menstuartion", "all QOL items and aggregate scores", "derived agecat", "categorical Children/Pregnancies with no coding key", "social/demographic codes with ambiguous mappings"],
            "target_status": "Proxy mapping supported by questionnaire order and published counts; original codebook absent. Current stage only, no future menopause event.",
            "source_doi": "10.12688/f1000research.125887.2", "dataset_doi": "10.6084/m9.figshare.20822032.v2"}


def prepare_nhanes():
    profiles, datasets = [], []
    for suffix, year in [("I", 2015), ("J", 2017)]:
        tables = {t: pd.read_sas(RAW / f"nhanes/{t}_{suffix}.xpt") for t in ["DEMO", "RHQ", "BMX", "BPX", "SLQ"]}
        d = tables["RHQ"]
        for name in ["DEMO", "BMX", "BPX", "SLQ"]:
            assert tables[name].SEQN.is_unique
            d = d.merge(tables[name], on="SEQN", how="left", validate="one_to_one")
        candidate = d[(d.RIAGENDR == 2) & d.RIDAGEYR.between(35, 60)].copy()
        positives = (candidate.RHQ031 == 2) & (candidate.RHD043 == 7)
        negatives = candidate.RHQ031 == 1
        excluded_surgery = (candidate.RHD280 == 1) | (candidate.RHQ305 == 1)
        keep = (positives | negatives) & ~excluded_surgery
        p = candidate.loc[keep].copy()
        o = pd.DataFrame({"participant_id": year * 1000000 + p.SEQN.astype(int), "cohort": year,
                          "target": positives.loc[keep].astype(int), "age": p.RIDAGEYR,
                          "bmi": p.BMXBMI, "waist_cm": p.BMXWAIST,
                          "pulse_bpm": p.BPXPLS, "sleep_hours": p.SLD012.where(p.SLD012.between(3, 14)),
                          "menarche_age": p.RHQ010.where(p.RHQ010.between(6, 20)),
                          "exam_weight": p.WTMEC2YR, "stratum": p.SDMVSTRA, "psu": p.SDMVPSU})
        datasets.append(o)
        profiles.append({"cohort": year, "candidate_35_60": len(candidate), "included": len(o),
                         "positive": int(o.target.sum()), "negative": int((o.target == 0).sum()),
                         "excluded": int((~keep).sum()), "known_surgery_excluded": int(excluded_surgery.sum()),
                         "missing_fraction": {c:float(v) for c,v in o.isna().mean().items()},
                         "scope": "Current self-reported natural menopause versus at least one period in 12 months. Includes peri in comparator; not future onset.",
                         "limits": "Some surgery and pregnancy-related fields suppressed above age44; complete exclusion cannot be guaranteed. Ever hormone use is not current use."})
    pd.concat(datasets).to_csv(PROCESSED / "nhanes_current_status.csv", index=False)
    return profiles


def prepare_tempid():
    d = pd.read_csv(RAW / "tempid/temperature_ovulation_dataset.csv")
    d["timestamp"] = pd.to_datetime(d.measurement_timestamp)
    d["event_date"] = pd.to_datetime(d.ovulation_timestamp)
    d["date"] = d.timestamp.dt.floor("D")
    d["participant_id"] = d.subject_id.str.replace(r"_[0-9]+$", "", regex=True)
    assert not d.duplicated(["subject_id", "timestamp"]).any()
    # Last 1-hour endpoint is not a complete day: require >= 4 readings in
    # the preregistered 00:00..05:59 nighttime window.
    night = d[d.timestamp.dt.hour < 6]
    a = night.groupby(["participant_id", "subject_id", "date", "event_date"], as_index=False).agg(temperature_deviation=("temperature_celsius", "mean"), observations=("temperature_celsius", "size"))
    a = a[a.observations >= 4].copy()
    a["relative_event_day"] = (a.date - a.event_date).dt.days
    a["fertile_target"] = a.relative_event_day.between(-5, 0).astype(int)
    a["postovulatory_target"] = (a.relative_event_day >= 2).astype(int)
    parts = []
    for _, g in a.groupby("subject_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        # A fixed7-day baseline prevents expanding-window sample size and
        # startup missingness from encoding position in the event-aligned
        # published extraction. The first7days are warmup, then discarded.
        x=g.temperature_deviation
        base=x.shift(1).rolling(7,min_periods=7).median()
        f=pd.DataFrame({"signal_delta":x-base,
                        "signal_mean3_delta":x.rolling(3,min_periods=3).mean()-base,
                        "signal_mean7_delta":x.rolling(7,min_periods=7).mean()-base,
                        "signal_change3":x.rolling(3,min_periods=3).mean()-x.shift(3).rolling(3,min_periods=3).mean(),
                        "signal_std7":x.rolling(7,min_periods=7).std()})
        parts.append(pd.concat([g, f], axis=1))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["signal_delta","signal_std7"]).reset_index(drop=True)
    out.to_csv(PROCESSED / "tempid_daily_research.csv", index=False)
    return {"hourly_rows": len(d), "cycle_series": int(d.subject_id.nunique()), "participant_groups_inferred": int(d.participant_id.nunique()),
            "daily_rows": len(out), "aligned_reference_dates": d.ovulation_timestamp.nunique(),
            "per_series_mean_range": [float(v) for v in [d.groupby("subject_id").temperature_celsius.mean().min(), d.groupby("subject_id").temperature_celsius.mean().max()]],
            "upstream_cautions": ["All series are aligned to one ovulation date; timestamps and relative_event_day are target-only and banned from X", "Every series mean is exactly36.5C: full-series normalization upstream; offset-invariant fixed-window features used", "Author data already filtered and bidirectionally imputed; cannot establish causal/raw performance", "Seven startup days discarded and fixed-size windows used to reduce alignment-boundary leakage", "Participant grouping inferred by removing final cycle suffix; no independent identity mapping", "Non-ring skin sensor; transfer to Oura unvalidated"],
            "evaluation_status": "Processed-data research benchmark only, not prospective real-world validation"}


def prepare_all():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    audits = {"jordan": prepare_jordan(), "nhanes": prepare_nhanes(), "tempid": prepare_tempid()}
    save_json(RESULTS / "data_audit.json", audits)
    return audits
