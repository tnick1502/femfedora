"""Literature-constrained scenarios. Joint distributions remain assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SIGNALS
from .missing_v2 import causal_features,mask_measurements


def generate(n_people=360,seed=29418):
    rng=np.random.default_rng(seed)
    raw=[]
    for person in range(n_people):
        typical=float(np.clip(rng.normal(29.3,4.4),20,45))
        within=float(rng.uniform(1.5,4.))
        segments=[]
        pos=0
        for cycle in range(5):
            length=int(np.clip(np.round(rng.normal(typical,within)),19,60))
            luteal=int(np.clip(np.round(rng.normal(12.4,2.4)),7,20))
            ov=max(6,length-luteal)
            ovulatory=bool(rng.random()>.10) # stress assumption, not prevalence estimate
            segments.append((cycle,pos,length,ov if ovulatory else np.nan))
            pos+=length
        d=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=pos)})
        temp=np.zeros(pos);hr_effect=np.zeros(pos);luteal_vector=np.zeros(pos)
        for cycle,start,length,ov in segments:
            day=np.arange(1,length+1)
            if np.isfinite(ov):
                a=float(np.clip(rng.normal(.33,.12),.05,.70))
                if rng.random()<.15:a*=.35 # weak signal transport stress
                rise=np.clip((day-ov)/rng.uniform(1,3),0,1)
                temp[start:start+length]=a*rise
                luteal_vector[start:start+length]=rise
                hr_effect[start:start+length]=3.8*rise+2.1*np.exp(-.5*((day-(ov-2))/2.3)**2)*(1-rise)
        common=rng.normal(0,1,pos)
        temp_noise=rng.normal(0,rng.uniform(.06,.12),pos)
        for k in range(1,pos):temp_noise[k]+=.5*temp_noise[k-1]
        illness=np.zeros(pos)
        if rng.random()<.35:
            k=int(rng.integers(0,max(1,pos-7)));illness[k:k+int(rng.integers(2,6))]=rng.uniform(.4,.9)
        d["temperature_deviation"]=temp+temp_noise+illness+rng.normal(0,.07)
        d["resting_hr"]=rng.normal(60,7)+hr_effect+common*2+illness*8
        d["hrv_ms"]=np.exp(rng.normal(3.7,.3)-.05*luteal_vector+rng.normal(0,.12,pos))
        d["respiratory_rate"]=rng.normal(15,1.2)+.2*luteal_vector+rng.normal(0,.3,pos)
        d["sleep_minutes"]=np.clip(rng.normal(430,40,pos)-illness*20,180,800)
        d["sleep_efficiency"]=np.clip(rng.normal(88,5,pos),40,100)
        d["steps"]=np.maximum(0,rng.normal(6500,2000,pos))
        d["participant_id"]=f"v2_{seed}_{person}"
        d["cycle_day"]=np.concatenate([np.arange(1,length+1) for _,_,length,_ in segments])
        d["ovulation_day"]=np.concatenate([np.full(length,ov) for _,_,length,ov in segments])
        d["cycle_id"]=np.concatenate([np.full(length,f"v2_{seed}_{person}_{cycle}") for cycle,_,length,_ in segments])
        d["period_start"]=d.cycle_day==1
        d["ovulatory"]=d.ovulation_day.notna()
        d["fertile_target"]=(d.ovulatory & (d.cycle_day>=d.ovulation_day-5)&(d.cycle_day<=d.ovulation_day)).astype(int)
        d["postovulatory_target"]=(d.ovulatory & (d.cycle_day>=d.ovulation_day+2)).astype(int)
        raw.append(d)
    return pd.concat(raw,ignore_index=True)


def training_table(raw,seed=29419):
    rng=np.random.default_rng(seed)
    parts=[]
    regimes=["complete","random10","random30","random50","block7","temperature_only","no_temperature","metric_dropout","illness_nonwear"]
    for _,person in raw.groupby("participant_id",sort=False):
        person=person.reset_index(drop=True)
        for replicate in range(2):
            mode=regimes[int(rng.integers(len(regimes)))]
            d=mask_measurements(person,mode,rng)
            f=causal_features(d)
            starts=person.index[person.period_start].to_numpy()
            logged=[int(x) for i,x in enumerate(starts) if i==0 or rng.random()>.20]
            # Calendar inputs must reflect logged history, not hidden true starts.
            f["cycle_day"]=np.nan;f["previous_cycle_median"]=np.nan;f["previous_cycle_std"]=np.nan
            for i,origin in enumerate(logged):
                end=logged[i+1] if i+1<len(logged) else len(d)
                lengths=np.diff(logged[:i+1]);lengths=lengths[(lengths>=19)&(lengths<=50)]
                f.loc[origin:end-1,"cycle_day"]=np.arange(1,end-origin+1)
                if len(lengths):
                    f.loc[origin:end-1,"previous_cycle_median"]=np.median(lengths[-6:])
                    f.loc[origin:end-1,"previous_cycle_std"]=np.std(lengths[-6:])
            for c in ["participant_id","cycle_id","fertile_target","postovulatory_target"]:f[c]=person[c]
            # First cycle warmup only. Subsequent sparse current nights remain.
            parts.append(f[person.cycle_id!=person.cycle_id.iloc[0]])
    return pd.concat(parts,ignore_index=True)
