"""New, isolated CTG summary-feature experiment; preserves all audited v2 outputs.

Family and feature engineering are jointly selected inside nested grouped CV.
The fixed classification policy is argmax. Independent validation fits only a
scalar temperature and decides whether a five-fold soft vote earns its complexity.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

from src.features.ctg_report_features import PROFILES, ReportFeatures, uci_profile, validate_profile
from src.training.ctg_experiments import ROOT, SEED, load_data, normalize, write_json, digest
from src.training.ctg_audit import score, ProbabilityCalibrator
from src.training.ctg_audit_folds import balanced_group_folds, fold_inventory

OUT = ROOT/'research/experiments/ctg_uci/report_features_v1'
EXPORT = ROOT/'models/exported/ctg_uci/report_features_v1'
VERSION = 'ctg_report_features_v1_20261017'
CAT_GRID = list(ParameterGrid({'depth':[3,5], 'iterations':[300,600],
                              'l2_leaf_reg':[3,10], 'learning_rate':[.06]}))
LOG_GRID = list(ParameterGrid({'C':[.1,1.,10.], 'class_weight':[None,'balanced']}))
FAMILIES = ['A_multiclass','B_sqrt','B_balanced','C_ovr','D_hierarchy','logistic']


def context():
    data, original = load_data()
    sources = [Path(__file__), ROOT/'src/features/ctg_report_features.py',
        ROOT/'src/training/ctg_audit.py', ROOT/'src/training/ctg_audit_folds.py',
        ROOT/'src/training/ctg_experiments.py', ROOT/'src/features/ctg_preprocessing.py']
    import importlib.metadata
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','joblib']}
    fp = hashlib.sha256((original+''.join(digest(p) for p in sources)+
                         json.dumps(versions,sort_keys=True)+VERSION).encode()).hexdigest()
    cache = ROOT/'tmp/ctg_report_features_v1'/fp[:16]
    for directory in [OUT,EXPORT,cache]: directory.mkdir(parents=True,exist_ok=True)
    return data,fp,cache,versions


def candidates():
    result=[]
    for engineered in [False,True]:
        for family in FAMILIES:
            for params in (LOG_GRID if family=='logistic' else CAT_GRID):
                result.append(dict(family=family,engineered=engineered,params=params))
    return result


class SummaryModel:
    def __init__(self,profile,candidate,seed=SEED):
        self.profile,self.candidate,self.seed=profile,candidate,seed

    def fit(self,X,y):
        self.features_=ReportFeatures(self.profile,self.candidate['engineered']).fit(X)
        x=self.features_.transform(X); y=np.asarray(y,dtype=int)
        self.training_bounds_=pd.DataFrame({'min':X.min(),'max':X.max()})
        family=self.candidate['family']; params=self.candidate['params']
        if family=='logistic':
            self.models_=[Pipeline([('scale',RobustScaler()),('model',LogisticRegression(
                **params,max_iter=4000))]).fit(x,y)]
            return self
        base=dict(params,random_seed=self.seed,thread_count=1,verbose=False,
            allow_writing_files=False,bootstrap_type='Bayesian',bagging_temperature=1.,
            random_strength=1.,use_best_model=False)
        weights={'B_sqrt':'SqrtBalanced','B_balanced':'Balanced'}.get(family)
        if weights:base['auto_class_weights']=weights
        self.models_=[]
        if family in ['A_multiclass','B_sqrt','B_balanced']:
            self.models_.append(CatBoostClassifier(**base,loss_function='MultiClass',classes_count=3).fit(x,y))
        else:
            heads=([(np.ones(len(y),bool),y==j) for j in range(3)] if family=='C_ovr'
                   else [(np.ones(len(y),bool),y!=0),(y!=0,y==2)])
            for mask,binary in heads:
                if len(np.unique(binary[mask]))!=2:raise ValueError('Binary head lost a class')
                self.models_.append(CatBoostClassifier(**base,loss_function='Logloss').fit(x.loc[mask],binary[mask]))
        return self

    def predict_proba(self,X):
        x=self.features_.transform(X);family=self.candidate['family']
        if family in ['A_multiclass','B_sqrt','B_balanced','logistic']:
            return normalize(self.models_[0].predict_proba(x))
        heads=np.column_stack([m.predict_proba(x)[:,1] for m in self.models_])
        if family=='D_hierarchy':
            a,b=heads.T
            return normalize(np.column_stack([1-a,a*(1-b),a*b]))
        return normalize(heads)


class SummaryBundle:
    def __init__(self,models,calibrator,profile):
        self.models,self.calibrator,self.profile=models,calibrator,profile
        self.classes_=np.array([1,2,3])
        self.clinically_validated_=False

    def predict_proba(self,X):
        X=validate_profile(X,self.profile)
        return self.calibrator.predict_proba(normalize(np.mean([m.predict_proba(X) for m in self.models],axis=0)))

    def predict(self,X):return self.predict_proba(X).argmax(axis=1)+1

    def support_report(self,X):
        X=validate_profile(X,self.profile)
        bounds=self.models[0].training_bounds_
        return pd.DataFrame({'out_of_training_range':((X<bounds['min'])|(X>bounds['max'])).any(axis=1)},index=X.index)


def grid_search(frame,profile,folds,seed):
    """Explicit exhaustive finite grid; every fitted transform stays on fit side."""
    x=uci_profile(frame,profile);y=frame.NSP.to_numpy()-1
    def evaluate(item):
        i,candidate=item; p=np.zeros((len(frame),3));rows=[]
        for j,(fit,hold) in enumerate(folds):
            model=SummaryModel(profile,candidate,seed+j).fit(x.iloc[fit],y[fit])
            p[hold]=model.predict_proba(x.iloc[hold])
            rows.append(score(frame.iloc[hold],p[hold]))
        row=dict(candidate=i,**candidate,mean_macro_f1=float(np.mean([r['macro_f1'] for r in rows])),
                 mean_log_loss=float(np.mean([r['log_loss'] for r in rows])))
        return row,p
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(evaluate,enumerate(candidates())))
    best=lambda ids:min(ids,key=lambda i:(-results[i][0]['mean_macro_f1'],results[i][0]['mean_log_loss'],i))
    chosen={'selected':best(range(len(results)))}
    for mode in [False,True]:chosen['engineered' if mode else 'raw']=best(
        [i for i,(row,_) in enumerate(results) if row['engineered']==mode])
    return {label:dict(candidate=candidates()[i],inner_scores=results[i][0],p_oof=results[i][1])
            for label,i in chosen.items()},[r[0] for r in results]


def prediction_table(frame,p,**extra):
    t=frame[['source_excel_row','source_file','NSP']].copy()
    for j in range(3):t[f'p{j}']=p[:,j]
    t['pred']=p.argmax(axis=1)+1
    for k,v in extra.items():t[k]=v
    return t


def run_nested():
    data,fp,cache,versions=context();train=data[data.split=='train'].reset_index(drop=True)
    outer=balanced_group_folds(train,5,SEED+21000)
    fold_inventory(train,outer).to_csv(OUT/'outer_fold_support.csv',index=False)
    layout=[];inner_layout=[];rows=[];predictions=[];grids=[];chosen=[]
    for k,(fit,hold) in enumerate(outer):
        f,h=train.iloc[fit].reset_index(drop=True),train.iloc[hold]
        layout.extend(dict(source_excel_row=int(train.iloc[i].source_excel_row),outer_fold=k) for i in hold)
        inner=balanced_group_folds(f,3,SEED+22000+k)
        for j,(_,v) in enumerate(inner):
            inner_layout.extend(dict(outer_fold=k,inner_fold=j,source_excel_row=int(f.iloc[i].source_excel_row)) for i in v)
        for profile in PROFILES:
            path=cache/f'nested_{profile}_{k}.joblib'
            if path.exists():payload=joblib.load(path)
            else:
                start=time.monotonic();selections,grid=grid_search(f,profile,inner,SEED+41000+k*100)
                output={}
                for mode,s in selections.items():
                    model=SummaryModel(profile,s['candidate'],SEED+42000+k).fit(uci_profile(f,profile),f.NSP.to_numpy()-1)
                    output[mode]=dict(candidate=s['candidate'],p=model.predict_proba(uci_profile(h,profile)))
                payload=dict(selections=output,grid=grid,seconds=time.monotonic()-start)
                joblib.dump(payload,path)
            for mode,s in payload['selections'].items():
                rows.append(dict(profile=profile,mode=mode,outer_fold=k,**score(h,s['p'])))
                predictions.append(prediction_table(h,s['p'],profile=profile,mode=mode,outer_fold=k))
                chosen.append(dict(profile=profile,mode=mode,outer_fold=k,**s['candidate']))
            grids.extend(dict(profile=profile,outer_fold=k,**r) for r in payload['grid'])
            pd.DataFrame(rows).to_csv(OUT/'nested_fold_metrics.csv',index=False)
            write_json(OUT/'progress.json',dict(stage='nested',completed=len(rows)//3,total=15,profile=profile,outer_fold=k,fingerprint=fp))
            print(f'{profile} outer {k+1}/5: {payload["seconds"]:.1f}s',flush=True)
    results=pd.DataFrame(rows)
    summary=results.groupby(['profile','mode']).agg(mean_macro_f1=('macro_f1','mean'),
        sd_macro_f1=('macro_f1','std'),mean_recall_path=('recall_path','mean'),
        mean_ap_path=('ap_path','mean'),mean_file_macro_f1=('file_macro_f1','mean')).reset_index()
    summary.to_csv(OUT/'nested_summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'nested_predictions.parquet',index=False)
    pd.DataFrame(grids).to_csv(OUT/'nested_grid_results.csv',index=False)
    pd.DataFrame(layout).to_csv(OUT/'outer_folds.csv',index=False)
    pd.DataFrame(inner_layout).to_csv(OUT/'inner_folds.csv',index=False)
    write_json(OUT/'nested_choices.json',chosen)
    write_json(OUT/'protocol.json',dict(version=VERSION,fingerprint=fp,environment=versions,
        profiles=PROFILES,candidates=candidates(),outer_folds=5,inner_folds=3,
        classification='fixed_argmax; final temperature preserves argmax',
        selector='mean inner segment macro-F1; tie log_loss; family and engineering selected jointly',
        test_status='previously observed; descriptive research evaluation, not a new blind cohort',
        missingness='structural absences select a profile; required missing inputs reject inference'))
    return summary,results


def run_final():
    data,fp,cache,versions=context()
    train,valid,test=[data[data.split==s].reset_index(drop=True) for s in ['train','validation','test']]
    inner=balanced_group_folds(train,3,SEED+31000)
    outer=balanced_group_folds(train,5,SEED+21000)
    rows=[];predictions=[];cards=[];all_grid=[]
    for profile in PROFILES:
        path=cache/f'final_{profile}.joblib'
        if path.exists():payload=joblib.load(path)
        else:
            start=time.monotonic();selected,grid=grid_search(train,profile,inner,SEED+43000)
            candidate=selected['selected']['candidate']
            model=SummaryModel(profile,candidate,SEED+44000).fit(uci_profile(train,profile),train.NSP.to_numpy()-1)
            ensemble=[SummaryModel(profile,candidate,SEED+45000+k).fit(uci_profile(train.iloc[fit],profile),
                train.iloc[fit].NSP.to_numpy()-1) for k,(fit,_) in enumerate(outer)]
            xv=uci_profile(valid,profile)
            pv_single=model.predict_proba(xv)
            pv_ensemble=normalize(np.mean([m.predict_proba(xv) for m in ensemble],axis=0))
            sv,se=score(valid,pv_single),score(valid,pv_ensemble)
            use_ensemble=se['macro_f1']>=sv['macro_f1']+.01
            models=ensemble if use_ensemble else [model]
            pv=pv_ensemble if use_ensemble else pv_single
            cal=ProbabilityCalibrator('temperature').fit(pv,valid.NSP.to_numpy()-1)
            bundle=SummaryBundle(models,cal,profile)
            payload=dict(candidate=candidate,grid=grid,bundle=bundle,validation_single=sv,
                validation_ensemble=se,ensemble_accepted=use_ensemble,
                validation_raw=pv,seconds=time.monotonic()-start)
            joblib.dump(payload,path)
        bundle=payload['bundle'];candidate=payload['candidate']
        # All model/ensemble/calibration choices above precede any test prediction.
        for split,frame in [('validation',valid),('test',test)]:
            p=bundle.predict_proba(uci_profile(frame,profile))
            rows.append(dict(profile=profile,split=split,**score(frame,p)))
            predictions.append(prediction_table(frame,p,profile=profile,split=split))
        model_path=EXPORT/f'{profile}_best.joblib';joblib.dump(bundle,model_path)
        # Export learned scalers explicitly even when the chosen tree needs none.
        scalers=[m.models_[0].named_steps['scale'] for m in bundle.models if candidate['family']=='logistic']
        joblib.dump(scalers,EXPORT/f'{profile}_scalers.joblib')
        native=[]
        for i,m in enumerate(bundle.models):
            if candidate['family']!='logistic':
                for j,tree in enumerate(m.models_):
                    pth=EXPORT/f'{profile}_member{i}_head{j}.cbm';tree.save_model(str(pth));native.append(pth.name)
        bounds=pd.DataFrame({'min':uci_profile(train,profile).min(),'max':uci_profile(train,profile).max()})
        bounds.to_csv(EXPORT/f'{profile}_training_bounds.csv')
        schema=dict(profile=profile,required_inputs=list(PROFILES[profile]),allow_missing=False,
            units={n:('bpm' if n=='baseline_bpm' else 'percent' if n.endswith('_pct') else
                      'events/second' if n.endswith('_per_s') else 'native SisPorto 2 definition; not Dawes-Redman') for n in PROFILES[profile]},
            extractor='UCI SisPorto 2; other extractors unverified',outcome='expert NSP 1/2/3',
            clinically_validated=False,temperature=bundle.calibrator.temperature_,
            scalers='RobustScaler inside each logistic member' if scalers else 'none required for CatBoost; explicit empty scaler list saved',
            candidate=candidate,ensemble_accepted=payload['ensemble_accepted'],members=len(bundle.models))
        write_json(EXPORT/f'{profile}_schema.json',schema)
        cards.append(dict(**schema,model=model_path.name,sha256=digest(model_path),native_models=native,
            validation_single=payload['validation_single'],validation_ensemble=payload['validation_ensemble']))
        all_grid.extend(dict(profile=profile,**r) for r in payload['grid'])
        print(f'Final {profile}: {candidate}, ensemble={payload["ensemble_accepted"]}; {payload["seconds"]:.1f}s',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'final_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'final_predictions.parquet',index=False)
    pd.DataFrame(all_grid).to_csv(OUT/'final_grid_results.csv',index=False)
    # "Champion" is defined within the screenshot-relevant common3 profile only.
    common=joblib.load(EXPORT/'common3_best.joblib');joblib.dump(common,EXPORT/'champion_common3.joblib')
    write_json(EXPORT/'model_card.json',dict(version=VERSION,fingerprint=fp,environment=versions,
        primary='common3_best.joblib',models=cards,clinical_use=False,
        transfer='restricted-input UCI experiment; hospital report measurement bridge and labeled external validation absent',
        final_fit='trees/scalers fit train only; temperature fit validation only; no train+validation refit',
        ensemble_rule='accept five-fold probability soft vote if validation macro-F1 increases >=.01',
        limitations=['NSP is an expert trace label, not a diagnosis or neonatal outcome',
            'source_file is not a proven patient ID','known reused test','one unlabelled report cannot validate transfer',
            'fewer features lose information; unmatched STV/LTV are never imputed into UCI columns']))
    write_json(OUT/'progress.json',dict(stage='complete',fingerprint=fp))
    return metrics,pd.DataFrame(cards)


def class_details():
    predictions=pd.read_parquet(OUT/'final_predictions.parquet');rows=[]
    for profile in PROFILES:
        t=predictions[(predictions.profile==profile)&(predictions.split=='test')]
        cm=confusion_matrix(t.NSP,t.pred,labels=[1,2,3]);np.save(OUT/f'{profile}_test_confusion.npy',cm)
        pr,re,f1,support=precision_recall_fscore_support(t.NSP,t.pred,labels=[1,2,3],zero_division=0)
        for i in range(3):rows.append(dict(profile=profile,NSP=i+1,precision=pr[i],recall=re[i],f1=f1[i],
            support=int(support[i]),correct=int(cm[i,i])))
    result=pd.DataFrame(rows);result.to_csv(OUT/'test_class_details.csv',index=False);return result


def bootstrap_results(repetitions=2000):
    """Block resampling fixed predictions; excludes model-selection uncertainty."""
    t=pd.read_parquet(OUT/'final_predictions.parquet');t=t[t.split=='test']
    reference=t[t.profile=='common3'].reset_index(drop=True)
    files=reference.source_file.unique();blocks={f:np.flatnonzero(reference.source_file.to_numpy()==f) for f in files}
    rng=np.random.default_rng(SEED+46000);rows=[]
    for profile in PROFILES:
        p=t[t.profile==profile].set_index('source_excel_row').loc[reference.source_excel_row].reset_index()
        y=p.NSP.to_numpy();pred=p.pred.to_numpy();values=[]
        for _ in range(repetitions):
            idx=np.concatenate([blocks[f] for f in rng.choice(files,len(files),replace=True)])
            if len(np.unique(y[idx]))!=3:continue
            recalls=[np.mean(pred[idx][y[idx]==label]==label) for label in [2,3]]
            values.append([f1_score(y[idx],pred[idx],average='macro',labels=[1,2,3],zero_division=0),*recalls])
        vals=np.array(values)
        for j,metric in enumerate(['macro_f1','recall_suspect','recall_path']):
            low,high=np.quantile(vals[:,j],[.025,.975])
            rows.append(dict(profile=profile,metric=metric,lower=low,upper=high,valid_bootstraps=len(vals)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'test_file_bootstrap.csv',index=False);return result
