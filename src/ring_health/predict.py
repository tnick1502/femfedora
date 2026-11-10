"""MVP JSON inference: python -m src.ring_health.predict --input example.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import ROOT
from .features import SIGNALS, daily_features, oura_to_daily, window_features
from .ovulation_filter import ovulation_posterior

EXPORT=ROOT/"models/exported/ring_health/v1"
FLAGS=["pregnant","breastfeeding","hormonal_medication","hysterectomy","bilateral_oophorectomy"]
RANGES={"temperature_deviation":(-3,3),"resting_hr":(30,130),"hrv_ms":(1,250),"respiratory_rate":(5,35),"sleep_minutes":(120,900),"sleep_efficiency":(20,100),"steps":(0,80000)}


def score(name,feature_frame):
    artifact=joblib.load(EXPORT/f"{name}.joblib")
    missing=set(artifact["features"])-set(feature_frame)
    if missing:raise ValueError(f"Missing computed features:{sorted(missing)}")
    p=artifact["pipeline"].predict_proba(feature_frame[artifact["features"]])[-1]
    return artifact,p


def predict(payload):
    if not isinstance(payload,dict):raise ValueError("Input must be a JSON object")
    result={"mode":"demonstration","model_version":"ring-health-v1","medical_recommendation":False,
            "validation_status":"Ring ML models trained only on synthetic scenarios. Prospective Oura performance is unknown."}
    context=payload.get("context",{})
    if not isinstance(context,dict):raise ValueError("context must be a JSON object")
    if any(context.get(k) is not False for k in FLAGS):
        return {**result,"status":"ineligible_or_unknown_context","required_false_flags":FLAGS}
    age=context.get("age")
    if not isinstance(age,(int,float)) or not 18<=age<=65:
        return {**result,"status":"age_missing_or_out_of_support"}
    d=oura_to_daily(payload["oura"]) if "oura" in payload else pd.DataFrame(payload.get("daily",[]))
    if d.empty or "date" not in d:
        return {**result,"status":"no_daily_data"}
    d["date"]=pd.to_datetime(d.date,errors="raise").dt.normalize()
    if d.date.duplicated().any():raise ValueError("Duplicate daily dates")
    as_of=pd.Timestamp(payload.get("as_of",d.date.max())).normalize()
    if (d.date>as_of).any():raise ValueError("Observed rows after as_of date are not permitted")
    d=d.sort_values("date").set_index("date").asfreq("D").reset_index()
    # Include absent days up to as_of so a stale last measurement cannot pass.
    d=d.set_index("date").reindex(pd.date_range(d.date.min(),as_of,freq="D")).rename_axis("date").reset_index()
    invalid={}
    for c,(lo,hi) in RANGES.items():
        if c not in d:d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors="coerce")
        bad=d[c].notna()&~d[c].between(lo,hi)
        if bad.any():invalid[c]=int(bad.sum())
    if invalid:return {**result,"status":"out_of_range_values","invalid_counts":invalid}
    result.update(as_of=str(as_of.date()),status="ok",observed_calendar_days=len(d))
    quality={c:float(d.tail(28)[c].notna().mean()) for c in SIGNALS}
    result["coverage_last28"]=quality
    quality90={c:float(d.tail(90)[c].notna().mean()) for c in SIGNALS}
    result["coverage_last90"]=quality90
    if age>=35 and len(d)>=90 and all(v>=.70 for v in quality.values()) and all(v>=.70 for v in quality90.values()):
        f=pd.DataFrame([window_features(d,age)])
        _,p=score("synthetic_stage_hybrid",f)
        result["menopause_stage_demo"]={"status":"scored" if p.max()>=.60 else "uncertain",
                    "probabilities":dict(zip(["reproductive","transition","postmenopause"],map(float,p))),
                    "model":"synthetic_stage_hybrid","provenance":"synthetic_only",
                    "future_menopause_risk":None,"time_to_menopause":None}
    else:
        result["menopause_stage_demo"]={"status":"insufficient_or_out_of_support","requirements":"age35..65,90calendar days,at least70% valid nights per signal over both28and90days"}
    if age>43:
        result["fertility_demo"]={"status":"out_of_training_age_support","age_support":[22,43]}
        return result
    if age<22:
        result["fertility_demo"]={"status":"out_of_training_age_support","age_support":[22,43]}
        return result
    if len(d)<35 or min(quality.values())<.70 or d.tail(3).temperature_deviation.isna().any():
        result["fertility_demo"]={"status":"insufficient_recent_data"}
        return result
    starts=sorted(set(context.get("period_start_dates",[])))
    if len(starts)<4:
        result["fertility_demo"]={"status":"insufficient_cycle_history","required":"4observed period starts; future dates prohibited"}
        return result
    starts=pd.DatetimeIndex(pd.to_datetime(starts)).sort_values().unique()
    if (starts>as_of).any():raise ValueError("Future period starts are not permitted")
    lengths=np.diff(starts.values).astype("timedelta64[D]").astype(int)[-6:]
    day=int((as_of-starts[-1]).days)+1
    if (lengths<19).any() or (lengths>65).any() or not 1<=day<=60:
        result["fertility_demo"]={"status":"cycle_history_out_of_support"}
        return result
    f=daily_features(d).tail(1).copy()
    f["cycle_day"]=day
    f["previous_cycle_median"]=float(np.median(lengths))
    f["previous_cycle_std"]=float(np.std(lengths))
    _,p=score("synthetic_fertility_hybrid",f)
    _,pc=score("synthetic_fertility_confirmation",f)
    result["fertility_demo"]={"status":"scored","cycle_day":day,"fertile_day_probability":float(p[1]),
            "postovulatory_signal_probability":float(pc[1]),"model":"synthetic_fertility_hybrid","provenance":"synthetic_only",
            "interpretation":"Current-day research signal; not pregnancy probability or confirmed ovulation"}
    try:result["fertility_demo"]["timing_distribution"]=ovulation_posterior(d,starts.astype(str).tolist(),as_of)
    except ValueError as e:result["fertility_demo"]["timing_distribution"]={"status":"insufficient_data","reason":str(e)}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=predict(json.loads(args.input.read_text()))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__=="__main__":main()
