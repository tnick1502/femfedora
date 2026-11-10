"""Evaluate exactly the deployed v2 analyzer, including gaps and fallback modes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,f1_score

from src.ring_health.data import ROOT,save_json
from src.ring_health.engine_v2 import Engine
from src.ring_health.features import SIGNALS
from src.ring_health.missing_v2 import mask_measurements

OUT=ROOT/"research/experiments/ring_health/v2"


def ci(values,seed=29426):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(values,(500,len(values)),replace=True),axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def main():
    engine=Engine()
    raw=pd.read_csv(ROOT/"data/processed/ring_health/v2/synthetic_daily.csv",parse_dates=["date"])
    split=json.loads((OUT/"person_split.json").read_text())
    raw=raw[raw.participant_id.isin(split["test"])].copy()
    modes=["complete","random10","random30","random50","random70","block3","block7","block14","temperature_only","no_temperature","metric_dropout","illness_nonwear","missed_period_log"]
    rows=[];events=[];latencies=[]
    for mode_index,mode in enumerate(modes):
        print("EVALUATE exact API",mode,flush=True)
        for pi,(pid,person) in enumerate(raw.groupby("participant_id")):
            person=person.reset_index(drop=True)
            # Each regime uses ALL heldout people; final cycle fixed before
            # inspecting outcomes. Every day from day7 to observed cycle end.
            current=person[person.cycle_id==person.cycle_id.iloc[-1]]
            rng=np.random.default_rng(29427+pi*43+mode_index)
            d=mask_measurements(person,mode,rng)
            if mode.startswith("block"):
                # A second declared stress design: block centered on reference
                # ovulation (or day16 in anovulatory cycles), label only used
                # to place the stress mask, never passed to analyzer.
                d=person.copy();size=int(mode[5:])
                center=int(current.index[0]+(current.ovulation_day.iloc[0] if current.ovulatory.iloc[0] else 16)-1)
                lo=max(0,center-size//2);d.loc[lo:lo+size-1,SIGNALS]=np.nan
            starts=person.loc[person.period_start,"date"].tolist()
            if mode=="missed_period_log":starts=starts[:-1]
            elif mode in ["random30","random50","random70"]:
                # Cold start: one last known anchor, not6perfect prior cycles.
                starts=starts[-1:]
            first_detection=None;estimated=None;forecasted_days=0;thermal_days=0
            for idx,truth in current.iterrows():
                if truth.cycle_day<7:continue
                today=truth.date
                history=d.loc[:idx,["date"]+SIGNALS].copy()
                history["date"]=history.date.dt.strftime("%Y-%m-%d")
                payload={"as_of":str(today.date()),"context":{"age":32,"pregnant":False,"breastfeeding":False,"hormonal_medication":False,"hysterectomy":False,"bilateral_oophorectomy":False,"period_start_dates":[str(x.date()) for x in starts if x<=today],"period_logs_complete":False},"daily":json.loads(history.to_json(orient="records"))}
                t=time.perf_counter();result=engine.analyze(payload);latencies.append((time.perf_counter()-t)*1000)
                f=result["fertility"];available="forecast" in f
                probability=f["forecast"][0]["fertile_day_probability"] if available else 0.
                forecasted_days+=int(available)
                thermal_days+=int(f.get("evidence_source")=="observed_temperature_and_cycle_prior")
                if f["status"]=="thermal_shift_detected" and first_detection is None:
                    first_detection=int(truth.cycle_day)
                    estimated=(pd.Timestamp(f["reference_event_date"])-current.date.min()).days+1
                learned=f.get("experimental_ml_signals",{}).get("fertile_target_probability")
                rows.append({"regime":mode,"participant_id":pid,"cycle_id":truth.cycle_id,"cycle_day":int(truth.cycle_day),"y":int(truth.fertile_target),"probability":probability,"forecast_available":available,"status":f["status"],"ml_probability":learned})
            ov=float(current.ovulation_day.iloc[0]) if current.ovulatory.iloc[0] else None
            events.append({"regime":mode,"participant_id":pid,"cycle_id":current.cycle_id.iloc[0],"ovulatory":bool(current.ovulatory.iloc[0]),"reference_day":ov,"alarm_day":first_detection,"estimate":estimated,"error":abs(estimated-ov) if estimated is not None and ov is not None else None,"delay":first_detection-ov if first_detection is not None and ov is not None else None,"forecasted_days":forecasted_days,"thermal_days":thermal_days})
    days=pd.DataFrame(rows);cycles=pd.DataFrame(events)
    days.to_csv(OUT/"deployed_daily_predictions.csv",index=False);cycles.to_csv(OUT/"deployed_cycle_predictions.csv",index=False)
    metrics={}
    for mode,g in days.groupby("regime"):
        c=cycles[cycles.regime==mode];ov=c[c.ovulatory];detected=ov[ov.alarm_day.notna()];anov=c[~c.ovulatory]
        covered=((ov.error<=2)&ov.alarm_day.notna()).astype(float)
        metrics[mode]={"days":len(g),"cycles":len(c),"forecast_coverage":float(g.forecast_available.mean()),"positive_prevalence":float(g.y.mean()),"positive_f1_all_days_unavailable_as_negative":float(f1_score(g.y,g.probability>=.5,zero_division=0)),"average_precision_all_days":float(average_precision_score(g.y,g.probability)),"brier_all_days":float(brier_score_loss(g.y,g.probability)),"ovulatory_cycles":len(ov),"detected_cycles":len(detected),"detection_rate_all_ovulatory":float(len(detected)/len(ov)),"detection_rate_ci95":ci(ov.alarm_day.notna().astype(float)),"mae_detected_only":float(detected.error.mean()) if len(detected) else None,"within2_all_ovulatory":float(covered.mean()),"within2_ci95":ci(covered),"median_alarm_delay_detected":float(detected.delay.median()) if len(detected) else None,"anovulatory_cycles":len(anov),"anovulatory_false_alarm_fraction":float(anov.alarm_day.notna().mean()) if len(anov) else None,"thermal_evidence_day_fraction":float(c.thermal_days.sum()/len(g))}
    save_json(OUT/"deployed_missingness_metrics.json",{"provenance":"synthetic_literature_constrained","test_people":len(split["test"]),"regimes":metrics,"latency_ms_p50_p95":np.quantile(latencies,[.5,.95]).tolist(),"evaluation":"Actual Engine.analyze on every eligible day of fixed final cycle; no sampling of favorable days; misses remain in all-cycle denominator","clinical_comparison":"Not equivalent to Oura validation"})
    print("END TO END EVALUATION COMPLETE",len(days),"requests",flush=True)


if __name__=="__main__":main()
