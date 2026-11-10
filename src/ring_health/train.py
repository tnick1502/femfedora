"""Train independent real-data benchmarks and explicitly synthetic MVP models.

Run: python -m src.ring_health.train
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PROCESSED, RESULTS, ROOT, prepare_all, save_json
from .evaluation import classification_metrics, cluster_ci, threshold_on_validation
from .simulation import fertility_scenarios, stage_scenarios

EXPORT = ROOT / "models/exported/ring_health/v1"
REGISTRY = ROOT / "models/registry/ring_health_v1.json"


def candidates():
    return {
        "dummy":make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),DummyClassifier(strategy="prior")),
        "logistic":make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.3,max_iter=1500)),
        "hist_gradient":make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=4,early_stopping=False,random_state=701)),
        "random_forest":make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),RandomForestClassifier(n_estimators=220,max_depth=9,min_samples_leaf=8,max_features=.8,random_state=702,n_jobs=2)),
    }


def group_split(d,group="participant_id",seed=701):
    ids=d[group].unique()
    train_ids, hold_ids=train_test_split(ids,test_size=.4,random_state=seed)
    val_ids,test_ids=train_test_split(hold_ids,test_size=.5,random_state=seed+1)
    assert not set(train_ids)&set(val_ids) and not set(train_ids)&set(test_ids) and not set(val_ids)&set(test_ids)
    return [d[d[group].isin(ids)].copy() for ids in [train_ids,val_ids,test_ids]]


def run_split(name,train,val,test,features,target,provenance,stress=None):
    print(f"TRAIN {name}: {len(train)}/{len(val)}/{len(test)} rows, {len(features)} features",flush=True)
    models, validations={},{}
    for model_name,model in candidates().items():
        model.fit(train[features],train[target])
        models[model_name]=model
        validations[model_name]=classification_metrics(val[target],model.predict_proba(val[features]))
    selected=min(validations,key=lambda m:validations[m]["log_loss"])
    k=len(models[selected].classes_)
    threshold=threshold_on_validation(val[target],models[selected].predict_proba(val[features])) if k==2 else .5
    result={"provenance":provenance,"target":target,"features":features,"selected_by":"minimum validation log loss, candidates fixed before test access", "selected":selected,
            "split":{"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"train_people":train.participant_id.nunique(),"validation_people":val.participant_id.nunique(),"test_people":test.participant_id.nunique()},
            "validation":validations,"test":{}}
    rows=[]
    for model_name,model in models.items():
        p=model.predict_proba(test[features])
        result["test"][model_name]=classification_metrics(test[target],p,threshold if model_name==selected else .5)
        if model_name==selected:
            result["test_selected_ci"]=cluster_ci(test[target],p,test.participant_id,threshold)
            prediction=test[["participant_id",target]].copy()
            for c in ["cycle_id","cycle_day","ovulation_day","ovulatory","age"]:
                if c in test:
                    prediction[c]=test[c]
            for j in range(k):
                prediction[f"p_{j}"]=p[:,j]
            prediction.to_csv(RESULTS/f"{name}_test_predictions.csv",index=False)
            if "exam_weight" in test:
                result["test_survey_weighted_secondary"]=classification_metrics(test[target],p,threshold,test.exam_weight)
            if "age" in test:
                subgroups={}
                for lo,hi in [(35,44),(45,54),(55,65)]:
                    mask=test.age.between(lo,hi).to_numpy()
                    if mask.sum()>10 and len(np.unique(test.loc[mask,target]))==k:
                        subgroups[f"age_{lo}_{hi}"]=classification_metrics(test.loc[mask,target],p[mask],threshold)
                result["subgroups"]=subgroups
    model=models[selected]
    if stress is not None:
        result["shifted_synthetic_test"]=classification_metrics(stress[target],model.predict_proba(stress[features]),threshold)
    # Negative control, fixed specification and shuffled labels only in train.
    control=candidates()["logistic"]
    control.fit(train[features],np.random.default_rng(810).permutation(train[target]))
    result["shuffled_train_labels_control"]=classification_metrics(test[target],control.predict_proba(test[features]))
    result["threshold"]=threshold
    result["classes"]=[int(x) for x in model.classes_]
    joblib.dump({"pipeline":model,"features":features,"threshold":threshold,"provenance":provenance,"target":target,"classes":result["classes"],"name":name},EXPORT/f"{name}.joblib")
    for split_name,frame in [("train",train),("validation",val),("test",test)]:
        rows += [{"participant_id":x,"split":split_name} for x in frame.participant_id.unique()]
    pd.DataFrame(rows).to_csv(RESULTS/f"{name}_split.csv",index=False)
    save_json(RESULTS/f"{name}_metrics.json",result)
    return result


def grouped_benchmark(name,d,features,target,n_splits=5):
    """All candidate models are reported; no test-based winner or deployment."""
    print(f"GROUP-CV {name}",flush=True)
    results={"status":"fixed-specification research benchmark; no model selected using OOF results","features":features,"target":target,"models":{}}
    fold_manifest=[]
    for model_name in ["dummy","logistic","hist_gradient","random_forest"]:
        k=d[target].nunique()
        p=np.zeros((len(d),k))
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=n_splits).split(d,d[target],d.participant_id)):
            assert not set(d.iloc[tr].participant_id)&set(d.iloc[te].participant_id)
            model=candidates()[model_name]
            model.fit(d.iloc[tr][features],d.iloc[tr][target])
            p[te]=model.predict_proba(d.iloc[te][features])
            if model_name=="dummy":
                fold_manifest += [{"participant_id":x,"test_fold":fold} for x in d.iloc[te].participant_id.unique()]
        results["models"][model_name]=classification_metrics(d[target],p)
        if model_name=="logistic":
            results["logistic_ci"]=cluster_ci(d[target],p,d.participant_id)
        out=d[["participant_id",target]].copy()
        if "subject_id" in d:
            out["subject_id"]=d.subject_id
        for j in range(k):out[f"p_{j}"]=p[:,j]
        out.to_csv(RESULTS/f"{name}_{model_name}_oof.csv",index=False)
    pd.DataFrame(fold_manifest).drop_duplicates().to_csv(RESULTS/f"{name}_folds.csv",index=False)
    save_json(RESULTS/f"{name}_metrics.json",results)
    return results


def event_metrics(d,threshold):
    """Detection time is first alarm; estimated event=alarm minus2 days.

    Evaluate all ovulatory cycles, missed cycles, detection delay, and false
    alarms in anovulatory cycles. No retrospective search for the best peak.
    """
    rows=[]
    for cid,g in d.groupby("cycle_id"):
        alarm=g[g.p_1>=threshold]
        ov=g.ovulation_day.iloc[0]
        day=float(alarm.cycle_day.iloc[0]) if len(alarm) else None
        rows.append({"cycle_id":cid,"ovulatory":bool(g.ovulatory.iloc[0]),"true_day":None if pd.isna(ov) else float(ov),"alarm_day":day,
                     "estimated_day":None if day is None else day-2})
    events=pd.DataFrame(rows)
    ov=events[events.ovulatory]
    detected=ov[ov.alarm_day.notna()]
    error=(detected.estimated_day-detected.true_day).abs()
    anov=events[~events.ovulatory]
    return {"ovulatory_cycles":len(ov),"detected_cycles":len(detected),"detection_rate":float(len(detected)/len(ov)),
            "mae_days_detected_only":float(error.mean()),"within_2_days_all_ovulatory":float((error<=2).sum()/len(ov)),
            "median_alarm_delay_days":float((detected.alarm_day-detected.true_day).median()),
            "anovulatory_cycles":len(anov),"anovulatory_false_alarm_fraction":float(anov.alarm_day.notna().mean()),
            "warning":"synthetic only; first-alarm specification fixed before scoring"}


def main():
    EXPORT.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    protocol={"date":"2026-11-09","version":"v1","seeds":{"split":701,"stage_train":704,"stage_shift":1704,"fertility_train":905,"fertility_shift":1905},
              "candidate_families":["dummy","logistic_C0.3","hist_gradient_leaf15_l2_4","random_forest_depth9_leaf8"],
              "selection":"validation log loss; binary threshold validation F1; no test-driven changes", "split_policy":"person disjoint60/20/20; NHANES2015-16 train/validation,2017-18 temporal test; processed TempID and Jordan5fold grouped CV",
              "synthetic_policy":"independent model artifacts, never mixed with real records; physiology-informed assumptions, no inferred clinical validity",
              "claim_policy":"current stage and current fertile-day signal; no validated future FMP/time-to-menopause output"}
    save_json(RESULTS/"protocol.json",protocol)
    audits=prepare_all()
    results={}
    nh=pd.read_csv(PROCESSED/"nhanes_current_status.csv")
    dev=nh[nh.cohort==2015]
    tr_ix,va_ix=train_test_split(np.arange(len(dev)),test_size=.25,stratify=dev.target,random_state=701)
    for label,features in [("age_only",["age"]),("context",["age","bmi","waist_cm","pulse_bpm","sleep_hours","menarche_age"])]:
        name="nhanes_"+label
        results[name]=run_split(name,dev.iloc[tr_ix],dev.iloc[va_ix],nh[nh.cohort==2017],features,"target","real_nhanes_current_status_no_ring")
    jo=pd.read_csv(PROCESSED/"jordan_stage_proxy.csv")
    results["jordan_proxy"]=grouped_benchmark("jordan_proxy",jo,[c for c in jo if c not in ["participant_id","stage_proxy"]],"stage_proxy")
    tmp=pd.read_csv(PROCESSED/"tempid_daily_research.csv")
    features=[c for c in tmp if c.startswith("signal_")]
    for target in ["fertile_target","postovulatory_target"]:
        name="tempid_"+target
        results[name]=grouped_benchmark(name,tmp,features,target)
    stage,sample=stage_scenarios()
    stage.to_csv(PROCESSED/"synthetic_stage_features.csv",index=False)
    sample.to_csv(PROCESSED/"synthetic_stage_daily_examples.csv",index=False)
    shift,_=stage_scenarios(400,seed=1704,stress_test=True)
    split=group_split(stage)
    ring=[c for c in stage if c not in ["participant_id","target","provenance","age"]]
    for suffix,feat in [("ring_only",ring),("hybrid",ring+["age"])]:
        name="synthetic_stage_"+suffix
        results[name]=run_split(name,*split,feat,"target","synthetic_mechanistic_scenario_not_clinically_validated",stress=shift)
    fertility,events,samples=fertility_scenarios()
    fertility.to_csv(PROCESSED/"synthetic_fertility_features.csv",index=False)
    events.to_csv(PROCESSED/"synthetic_fertility_events.csv",index=False)
    samples.to_csv(PROCESSED/"synthetic_fertility_daily_examples.csv",index=False)
    fs,_,_=fertility_scenarios(100,seed=1905,stress_test=True)
    fsplit=group_split(fertility)
    physiological=[c for c in fertility if any(c.startswith(s+"_") for s in ["temperature_deviation","resting_hr","hrv_ms","respiratory_rate","sleep_minutes","sleep_efficiency","steps"])]
    calendar=["cycle_day","previous_cycle_median","previous_cycle_std"]
    for suffix,features,target in [("calendar",calendar,"fertile_target"),("ring_only",physiological,"fertile_target"),("hybrid",physiological+calendar,"fertile_target"),("confirmation",physiological+calendar,"postovulatory_target")]:
        name="synthetic_fertility_"+suffix
        results[name]=run_split(name,*fsplit,features,target,"synthetic_mechanistic_scenario_not_clinically_validated",stress=fs)
    pred=pd.read_csv(RESULTS/"synthetic_fertility_confirmation_test_predictions.csv")
    results["synthetic_ovulation_events"]=event_metrics(pred,results["synthetic_fertility_confirmation"]["threshold"])
    save_json(RESULTS/"ovulation_event_metrics.json",results["synthetic_ovulation_events"])
    files=[p for p in PROCESSED.glob("*") if p.is_file()]
    environment={"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"pandas":pd.__version__,
                 "code_base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                 "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"src/ring_health").glob("*.py"))},
                 "data_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    save_json(RESULTS/"environment.json",environment)
    save_json(RESULTS/"all_results.json",results)
    save_json(REGISTRY,{"id":"ring-health-v1","status":"research_and_synthetic_demo","date":"2026-11-09","artifacts":{p.stem:{"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"provenance":results[p.stem]["provenance"]} for p in EXPORT.glob("*.joblib")},"results":str((RESULTS/"all_results.json").relative_to(ROOT)),"environment":environment})
    print("COMPLETED",len(results),"experiments",flush=True)


if __name__=="__main__":main()
