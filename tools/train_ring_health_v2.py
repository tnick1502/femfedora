"""Fit v2 missing-aware models and real-data thermal profiles, without target metrics."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold,train_test_split
from sklearn.pipeline import make_pipeline

from src.ring_health.data import ROOT,save_json
from src.ring_health.evaluation import classification_metrics,threshold_on_validation
from src.ring_health.scenarios_v2 import generate,training_table
from src.ring_health.thermal_v2 import posterior

OUT=ROOT/"research/experiments/ring_health/v2"
EXPORT=ROOT/"models/exported/ring_health/v2"


def thermal_cycles():
    d=pd.read_csv(ROOT/"data/external/ring_health/tempid/temperature_ovulation_dataset.csv")
    d["date"]=pd.to_datetime(d.measurement_timestamp)
    # Author-defined nocturnal21:00..03:59 window, assigned to morning date.
    d=d[(d.date.dt.hour>=21)|(d.date.dt.hour<4)].copy()
    d["day"]=d.date.dt.floor("D")+pd.to_timedelta((d.date.dt.hour>=21).astype(int),unit="D")
    d=d.groupby(["subject_id","day","ovulation_timestamp"],as_index=False).agg(temperature_deviation=("temperature_celsius","mean"),n=("temperature_celsius","size"))
    d=d[d.n>=6].copy();d["participant_id"]=d.subject_id.str.replace(r"_[0-9]+$","",regex=True)
    return d


def fit_profile(d):
    amplitudes=[];noise=[]
    for _,g in d.groupby("subject_id"):
        rel=(g.day-pd.to_datetime(g.ovulation_timestamp)).dt.days
        pre=g.loc[rel.between(-9,-3),"temperature_deviation"]
        post=g.loc[rel.between(3,9),"temperature_deviation"]
        if len(pre)>=3 and len(post)>=3:amplitudes.append(float(post.median()-pre.median()))
        if len(pre)>=4:noise.extend((pre.diff().dropna()/np.sqrt(2)).tolist())
    a=np.maximum(.005,np.quantile(amplitudes,[.1,.25,.5,.75,.9]))
    sigma=max(.03,float(np.std(noise)))
    return {"amplitudes":a.tolist(),"amplitude_weights":[.2]*5,"noise_sigma":sigma,"rho":.5,"rise_days":2.,"likelihood_weight":.5,"detection_min_shift":.025,"provenance":"fit_to_processed_TempID_training_people_only"}


def real_thermal_cv(d):
    rows=[]
    regimes=["complete","random10","random30","random50","random70","block3","block7","block14"]
    for fold,(tr,te) in enumerate(GroupKFold(3).split(d,groups=d.participant_id)):
        profile=fit_profile(d.iloc[tr])
        for sid,g in d.iloc[te].groupby("subject_id"):
            g=g.sort_values("day").reset_index(drop=True)
            days=(g.day-g.day.min()).dt.days.to_numpy()+1
            truth=int((pd.Timestamp(g.ovulation_timestamp.iloc[0])-g.day.min()).days)+1
            for ri,regime in enumerate(regimes):
                rng=np.random.default_rng(41000+ri+sum(map(ord,sid)))
                y=g.temperature_deviation.to_numpy().copy()
                if regime.startswith("random"):y[rng.random(len(y))<int(regime[6:])/100]=np.nan
                if regime.startswith("block"):
                    gap=int(regime[5:]);center=truth-1;lo=max(0,center-gap//2);y[lo:lo+gap]=np.nan
                # Uniform event prior within observed support. No cycle-day or
                # event-aligned timestamp is an input; reference is target only.
                grid=np.arange(5,max(6,int(days.max())-2))
                prior=np.r_[np.full(len(grid),.9/len(grid)),.1]
                p=posterior(days,y,grid,prior,profile)
                est=int(grid[np.argmax(p[:-1])])
                usable=int(np.isfinite(y).sum())>=8
                q=np.cumsum(p[:-1]/p[:-1].sum())
                width=int(grid[min(np.searchsorted(q,.95),len(grid)-1)]-grid[min(np.searchsorted(q,.05),len(grid)-1)])
                rows.append({"fold":fold,"participant_id":g.participant_id.iloc[0],"subject_id":sid,"regime":regime,"truth":truth,"estimate":est,"absolute_error":abs(est-truth),"observed_nights":int(np.isfinite(y).sum()),"reported":bool(usable and width<=12),"interval_width":width})
    pd.DataFrame(rows).to_csv(OUT/"real_thermal_predictions.csv",index=False)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/"protocol.json",{"version":"v2","selection":"validation log loss; no test-targeted optimization or required score","test_policy":"new synthetic seeds and heldout people; real3fold personCV, no Oura clinical comparator equivalence","synthetic_seed":29418,"random_split":29420,"missing_regimes":["0/10/30/50/70%","3/7/14day blocks","channel dropout","illness-related missingness","missed period logs"],"claim":"Deployable research API, clinical performance unvalidated"})
    real=thermal_cycles();cv=real_thermal_cv(real)
    summary={}
    for mode,g in cv.groupby("regime"):
        accepted=g[g.reported]
        summary[mode]={"cycles":len(g),"reported_cycles":len(accepted),"coverage":float(g.reported.mean()),"mae_all_estimates":float(g.absolute_error.mean()),"mae_reported":float(accepted.absolute_error.mean()) if len(accepted) else None,"within2_all_cycles":float(((g.absolute_error<=2)&g.reported).mean()),"limitations":"processed aligned non-ring data, no anovulatory specificity or prospective validation"}
    save_json(OUT/"real_thermal_metrics.json",summary)
    profiles={"tempid_processed":fit_profile(real),"oura_literature":{"amplitudes":[.08,.15,.25,.33,.45,.55,.7],"amplitude_weights":[.05,.1,.15,.25,.2,.15,.1],"noise_sigma":.10,"rho":.5,"rise_days":2.,"likelihood_weight":.5,"detection_min_shift":.12,"provenance":"literature_constrained_uncalibrated_joint_model"}}
    save_json(EXPORT/"thermal_profiles.json",profiles)
    print("REAL thermal evaluation complete",flush=True)
    raw=generate();data=training_table(raw)
    processed=ROOT/"data/processed/ring_health/v2";processed.mkdir(exist_ok=True)
    raw.to_csv(processed/"synthetic_daily.csv",index=False);data.to_csv(processed/"synthetic_training_features.csv",index=False)
    ids=raw.participant_id.unique();tr,hold=train_test_split(ids,test_size=.4,random_state=29420);va,te=train_test_split(hold,test_size=.5,random_state=29421)
    save_json(OUT/"person_split.json",{"train":tr.tolist(),"validation":va.tolist(),"test":te.tolist()})
    features=[c for c in data if c not in ["participant_id","cycle_id","fertile_target","postovulatory_target"]]
    results={}
    for target in ["fertile_target","postovulatory_target"]:
        train=data[data.participant_id.isin(tr)];val=data[data.participant_id.isin(va)];test=data[data.participant_id.isin(te)]
        options={"hist_gradient":HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=5,min_samples_leaf=35,early_stopping=False,random_state=29422),"forest":make_pipeline(SimpleImputer(strategy="median",add_indicator=True),RandomForestClassifier(n_estimators=140,max_depth=12,min_samples_leaf=10,n_jobs=2,random_state=29423))}
        validations={}
        for name,model in options.items():
            model.fit(train[features],train[target]);validations[name]=classification_metrics(val[target],model.predict_proba(val[features]))
        selected=min(validations,key=lambda k:validations[k]["log_loss"]);model=options[selected]
        threshold=threshold_on_validation(val[target],model.predict_proba(val[features]))
        p=model.predict_proba(test[features]);metrics=classification_metrics(test[target],p,threshold)
        joblib.dump({"pipeline":model,"features":features,"threshold":threshold,"target":target,"provenance":"synthetic_literature_constrained_missingness_augmented"},EXPORT/f"{target}.joblib")
        results[target]={"selected":selected,"validation":validations,"test":metrics,"threshold":threshold}
        print("FITTED",target,selected,metrics["positive_f1"],flush=True)
    # Stage remains an explicitly experimental model. Augmentation drops whole
    # channels, with indicator features; it does not manufacture clinical labels.
    stage=pd.read_csv(ROOT/"data/processed/ring_health/synthetic_stage_features.csv")
    ids=stage.participant_id.to_numpy();tr,hold=train_test_split(ids,test_size=.4,random_state=29420);va,te=train_test_split(hold,test_size=.5,random_state=29421)
    sf=[c for c in stage if c not in ["participant_id","target","provenance"]]
    rng=np.random.default_rng(29424);aug=[]
    for rep in range(4):
        s=stage.copy()
        for c in ["resting_hr","hrv_ms","respiratory_rate","sleep_minutes","sleep_efficiency","steps"]:
            m=rng.random(len(s))<.30
            s.loc[m,[x for x in sf if x.startswith(c+"_") and not x.endswith("_missing")]]=np.nan
            s.loc[m,c+"_missing"]=1.
        aug.append(s)
    aug=pd.concat(aug);m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=12,l2_regularization=5,early_stopping=False,random_state=29425)
    m.fit(aug[aug.participant_id.isin(tr)][sf],aug[aug.participant_id.isin(tr)].target)
    test=aug[aug.participant_id.isin(te)];results["stage"]={"test":classification_metrics(test.target,m.predict_proba(test[sf])),"provenance":"synthetic_only_not_menopause_diagnosis"}
    joblib.dump({"pipeline":m,"features":sf,"provenance":"synthetic_only_not_menopause_diagnosis"},EXPORT/"stage.joblib")
    save_json(OUT/"training_metrics.json",results)
    manifest={"version":"ring-health-v2","python":platform.python_version(),"sklearn":sklearn.__version__,"clinical_validation":"not_established","artifacts":{p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in EXPORT.iterdir() if p.is_file()},"literature_and_assumptions":"models/configs/training/ring_health_evidence_v2.json"}
    save_json(ROOT/"models/registry/ring_health_v2.json",manifest)
    print("V2 training complete",flush=True)


if __name__=="__main__":main()
