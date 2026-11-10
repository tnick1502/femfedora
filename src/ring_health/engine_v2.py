"""Versioned app inference; artifact loading occurs once at service startup."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .features import SIGNALS,window_features
from .missing_v2 import causal_features,normalize_oura
from .schema_v2 import AnalyzeRequest
from .thermal_v2 import analyze_cycle

ROOT=Path(__file__).resolve().parents[2]


class Engine:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        self.directory=self.root/"models/exported/ring_health/v2"
        manifest_path=self.root/"models/registry/ring_health_v2.json"
        self.manifest=json.loads(manifest_path.read_text())
        if self.manifest["sklearn"]!=sklearn.__version__:raise RuntimeError("Artifact scikit-learn version mismatch")
        for name,item in self.manifest["artifacts"].items():
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=item["sha256"]:raise RuntimeError("Artifact checksum mismatch: "+name)
        self.models={name:joblib.load(self.directory/(name+".joblib")) for name in ["fertile_target","postovulatory_target","stage"]}
        self.profiles=json.loads((self.directory/"thermal_profiles.json").read_text())
        self.model_digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def analyze(self,payload):
        request=payload if isinstance(payload,AnalyzeRequest) else AnalyzeRequest.model_validate(payload)
        result={"model_version":"ring-health-v2","artifact_manifest_sha256":self.model_digest,"as_of":request.as_of.isoformat(),
                "clinical_validation":"not_established","recommendations":[],"measurement_imputation":"none; missing measurements remain unobserved"}
        context=request.context
        if any(getattr(context,k) for k in ["pregnant","breastfeeding","hormonal_medication","hysterectomy","bilateral_oophorectomy"]):
            return {**result,"status":"outside_model_population","fertility":{"status":"not_assessed"},"menopause":{"status":"not_assessed"}}
        rows=[x.model_dump(mode="json") for x in request.daily] if request.daily is not None else normalize_oura(request.oura.model_dump())
        for x in rows:
            if pd.Timestamp(x["date"]).date()>request.as_of:raise ValueError("Observed date after as_of")
        if rows and (request.as_of-pd.Timestamp(min(x["date"] for x in rows)).date()).days>730:raise ValueError("History exceeds730days")
        d=pd.DataFrame(rows,columns=["date"]+SIGNALS)
        if len(d):
            d["date"]=pd.to_datetime(d.date)
            d=d.set_index("date").sort_index()
            d=d.reindex(pd.date_range(d.index.min(),request.as_of,freq="D")).rename_axis("date").reset_index()
        else:d=pd.DataFrame({"date":[pd.Timestamp(request.as_of)],**{c:[np.nan] for c in SIGNALS}})
        for c in SIGNALS:d[c]=pd.to_numeric(d[c],errors="raise").astype(float)
        observed={c:int(d.tail(28)[c].notna().sum()) for c in SIGNALS}
        result.update(status="ok",observed_nights_last28=observed)
        starts=sorted(context.period_start_dates)
        if context.cycle_day is not None:
            anchor=request.as_of-timedelta(days=context.cycle_day-1)
            if starts and starts[-1]!=anchor:raise ValueError("cycle_day conflicts with latest period start")
            if not starts:starts=[anchor]
        if context.age>45:
            fertility={"status":"outside_evaluated_age_support","age_support":[18,45]}
        else:
            fertility=analyze_cycle(d,[x.isoformat() for x in starts],request.as_of,self.profiles["oura_literature"],logs_complete=context.period_logs_complete)
            fertility["model_profile"]="oura_literature"
            fertility["profile_evidence"]="literature_constrained; joint model not validated on Oura cohort"
            # Optional physiology can contribute a separate learned signal;
            # never call it measured fertility or hormone-confirmed ovulation.
            if starts and 1<=(request.as_of-starts[-1]).days+1<=50 and max(observed[c] for c in SIGNALS if c!="steps")>=3:
                f=causal_features(d).tail(1).copy()
                lengths=np.diff(pd.to_datetime(starts).values).astype("timedelta64[D]").astype(int)[-6:]
                lengths=lengths[(lengths>=19)&(lengths<=50)]
                f["cycle_day"]=(request.as_of-starts[-1]).days+1
                f["previous_cycle_median"]=float(np.median(lengths)) if len(lengths) else np.nan
                f["previous_cycle_std"]=float(np.std(lengths)) if len(lengths) else np.nan
                learned={}
                for name in ["fertile_target","postovulatory_target"]:
                    artifact=self.models[name]
                    learned[name+"_probability"]=float(artifact["pipeline"].predict_proba(f[artifact["features"]])[0,1])
                fertility["experimental_ml_signals"]={**learned,"training_data":"synthetic_literature_constrained_missingness_augmented","calibration_on_real_ring_cohort":"not_established"}
        result["fertility"]=fertility
        if context.age>=35 and len(d)>=90 and d.tail(90).temperature_deviation.notna().sum()>=20:
            f=pd.DataFrame([window_features(d,context.age)])
            m=self.models["stage"];p=m["pipeline"].predict_proba(f[m["features"]])[0]
            result["menopause"]={"status":"experimental_model_output","stage_probabilities":dict(zip(["reproductive","transition","postmenopause"],map(float,p))),"training_data":"synthetic_only","ring_stage_validation":"not_established","future_onset_prediction":None,"diagnosis":None}
        else:result["menopause"]={"status":"insufficient_history_or_age_support","required":"Age35..65;90calendar days with at least20observed temperature nights; other signals optional","future_onset_prediction":None}
        return result
