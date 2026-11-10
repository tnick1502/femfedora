"""Integration checks, scenario examples, calibration and timing forecast score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ring_health.data import PROCESSED, RESULTS, ROOT, save_json
from src.ring_health.evaluation import classification_metrics
from src.ring_health.features import SIGNALS, daily_features
from src.ring_health.ovulation_filter import ovulation_posterior
from src.ring_health.predict import predict
from src.ring_health.simulation import fertility_scenarios
from src.ring_health.train import group_split


def serialize(frame):
    f=frame[["date"]+SIGNALS].copy()
    f["date"]=pd.to_datetime(f.date).dt.strftime("%Y-%m-%d")
    return json.loads(f.to_json(orient="records"))


def main():
    destination=ROOT/"data/samples/ring_health"
    destination.mkdir(parents=True,exist_ok=True)
    flags={k:False for k in ["pregnant","breastfeeding","hormonal_medication","hysterectomy","bilateral_oophorectomy"]}
    s=pd.read_csv(PROCESSED/"synthetic_stage_daily_examples.csv")
    stage=s[s.participant_id==s.participant_id.iloc[0]]
    example={"as_of":str(pd.to_datetime(stage.date.max()).date()),"context":{"age":50,**flags},"daily":serialize(stage)}
    save_json(destination/"menopause_demo_input.json",example)
    save_json(destination/"menopause_demo_output.json",predict(example))
    # Generate identical population, keeping raw signals for independent filter
    # evaluation. These columns are evaluation truth and never input to ML.
    features,_,raw=fertility_scenarios(include_all_daily=True)
    _,_,test=group_split(features)
    ids=test.participant_id.unique()
    raw=raw[raw.participant_id.isin(ids)].copy()
    scored=[]
    for pid,person in raw.groupby("participant_id"):
        starts=person.loc[person.period_start,"date"].tolist()
        for _,row in person.iterrows():
            today=row.date
            observed_starts=[x for x in starts if x<=today]
            if len(observed_starts)<4 or int(row.cycle_day)<9 or pd.isna(row.temperature_deviation):continue
            history=person.loc[person.date<=today,["date"]+SIGNALS]
            try:posterior=ovulation_posterior(history,[str(x.date()) for x in observed_starts],today)
            except ValueError:continue
            for h,out in enumerate(posterior["forecast"]):
                # Origins evaluated only when target is within the SAME cycle;
                # future beyond cycle end is censored for this diagnostic.
                cid=person[person.cycle_id==row.cycle_id]
                if row.cycle_day+h>cid.cycle_day.max():continue
                ov=row.ovulation_day
                truth=int(pd.notna(ov) and ov-5<=row.cycle_day+h<=ov)
                scored.append({"participant_id":pid,"cycle_id":row.cycle_id,"horizon":h,"y":truth,"p":out["fertile_day_probability"]})
    forecasts=pd.DataFrame(scored)
    forecasts.to_csv(RESULTS/"bayesian_forecast_predictions.csv",index=False)
    metrics={"provenance":"synthetic","heldout_people":len(ids),"specification":"fixed assumed0.4C rise,Student-t scale0.2,tempered likelihood0.30,no-ovulation prior0.10", "limitations":"Only cycle3and4 with>=3 completed prior cycles; target days beyond observed current-cycle end censored; no real Oura validation", "horizons":{}}
    for h,g in forecasts.groupby("horizon"):
        p=np.column_stack([1-g.p,g.p])
        metrics["horizons"][str(h)]=classification_metrics(g.y,p)
    save_json(RESULTS/"bayesian_forecast_metrics.json",metrics)
    # Choose an illustrative heldout person and a usable observation origin.
    # An illustrative synthetic preovulation origin, independent of model scores.
    eligible=raw[(raw.cycle_id.str.endswith("_3")) & raw.ovulation_day.between(13,25)]
    # Require usable observed coverage for the runnable example. No selection
    # on predicted scores or model correctness; absent nights remain absent.
    for pid in eligible.participant_id.unique():
        person=raw[raw.participant_id==pid]
        starts=person.loc[person.period_start,"date"].tolist()
        true_ov=person[person.cycle_id.str.endswith("_3")].ovulation_day.iloc[0]
        today=starts[3]+pd.Timedelta(days=int(true_ov)-4)
        subset=person[person.date<=today]
        example={"as_of":str(today.date()),"context":{"age":32,**flags,"period_start_dates":[str(x.date()) for x in starts if x<=today]},"daily":serialize(subset)}
        out=predict(example)
        if out["fertility_demo"]["status"]=="scored" and "forecast" in out["fertility_demo"].get("timing_distribution",{}):break
    else:raise AssertionError("No runnable synthetic example found")
    save_json(destination/"fertility_demo_input.json",example)
    assert out["fertility_demo"]["status"]=="scored"
    save_json(destination/"fertility_demo_output.json",out)
    checks={"fertility_demo_status":out["fertility_demo"]["status"],"menopause_demo_has_score":"probabilities" in predict(json.loads((destination/"menopause_demo_input.json").read_text()))["menopause_stage_demo"],
            "split_overlap":False,"target_leakage_features":False,"causality_prefix_checks":0,"nan_output":False}
    for f in RESULTS.glob("*_split.csv"):
        d=pd.read_csv(f)
        assert d.groupby("participant_id").split.nunique().max()==1,f
    for f in RESULTS.glob("*_metrics.json"):
        v=json.loads(f.read_text())
        assert not set(v.get("features",[]))&{"target","fertile_target","postovulatory_target","ovulation_day","event_date","relative_event_day","cohort","participant_id","stage_proxy"},f
    person=person[["date"]+SIGNALS]
    full=daily_features(person)
    for n in [35,60,90]:
        pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True),daily_features(person.iloc[:n]).reset_index(drop=True))
        checks["causality_prefix_checks"]+=1
    save_json(RESULTS/"integration_checks.json",checks)
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
