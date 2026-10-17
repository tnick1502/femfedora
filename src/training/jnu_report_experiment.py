"""Sealed-test, patient-grouped exhaustive search for JNU report classifiers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,ParameterGrid

from src.features.jnu_ctg_report_extraction import ROOT,DEST,PROFILE_FIELDS,sha256
from src.training.jnu_report_models import (ReportClassifier,ReportBundle,SigmoidCalibration,
    patient_weights,score,choose_research_threshold)

OUT=ROOT/'research/experiments/jnu_ctg/classifiers_v1'
EXPORT=ROOT/'models/exported/jnu_ctg/classifiers_v1'
SEED=20261017
CAT_GRID=list(ParameterGrid(dict(depth=[3,4,6],learning_rate=[.03,.1],
    l2_leaf_reg=[3,10],iterations=[300,700],auto_class_weights=[None,'SqrtBalanced','Balanced'])))
SPLINE_GRID=list(ParameterGrid(dict(n_knots=[3,5,7],C=[.03,.3,3.,30.],
    interactions=[False,True],weight_policy=[None,'Balanced'])))
LINEAR_GRID=list(ParameterGrid(dict(C=[.01,.1,1.,10.,100.],weight_policy=[None,'Balanced'])))
FAMILIES=['catboost','spline_logit','linear']
GRID={'catboost':CAT_GRID,'spline_logit':SPLINE_GRID,'linear':LINEAR_GRID}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def initialize():
    for d in [OUT,EXPORT]:d.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/'data/processed/jnu_ctg/v1/splits.parquet']
    inputs += [DEST/f'X_{p}_{s}.parquet' for p in PROFILE_FIELDS for s in ['train','validation','test']]
    inputs += [ROOT/'src/training/jnu_report_models.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in inputs}
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','joblib']}
    fingerprint=hashlib.sha256(json.dumps(dict(hashes=hashes,versions=versions),sort_keys=True).encode()).hexdigest()
    cache=ROOT/'tmp/jnu_report_classifiers_v1'/fingerprint[:16]
    cache.mkdir(parents=True,exist_ok=True)
    if (OUT/'test_predictions.parquet').exists():
        frozen=json.loads((OUT/'selection_frozen.json').read_text())
        if frozen.get('cache_fingerprint')!=cache.name:
            raise RuntimeError('This version already evaluated test. Changed source requires an explicit new research version, not another blind-test claim.')
    # Protocol exists before ANY new test predictions or model selection.
    write_json(OUT/'protocol.json',dict(version='jnu_report_classifiers_v1',fingerprint=fingerprint,
        input_sha256=hashes,environment=versions,seed=SEED,target='figo_label:0 normal / 1 abnormal',
        profiles=PROFILE_FIELDS,grids=GRID,outer_folds=5,inner_folds=3,ensemble_members=5,
        patient_weight='1/current-cohort-record-count per patient; training mean normalized to 1',
        selector='maximum mean inner patient-weighted macro-F1; exact ties log_loss then complexity',
        mode_selector='ensemble only if mean inner F1 >= single+0.005 AND log_loss <= single+0.005',
        primary_threshold=.5,alternative_threshold='max patient-weighted F1 on separate operating set; research only',
        calibration='positive-slope sigmoid of actual single/ensemble on validation calibration half',
        eligibility='all allocated records; NO selection using raw quality/STV-minutes flag',
        test='sealed until every profile/family/aggregation/calibrator/threshold/champion is frozen',
        exact_tree_prefix_reuse=True,early_stopping=False,clinical_validation=False))
    return fingerprint,cache


def metadata(split):
    frame=pd.read_parquet(ROOT/'data/processed/jnu_ctg/v1/splits.parquet')
    return frame.loc[frame.split==split].reset_index(drop=True)


def inputs(frame,profile):
    # The source X has exactly one row per original ID. All rows including NaN
    # are retained, and joins never carry identifiers/targets into predictors.
    part=frame.split.unique()
    if len(part)!=1:raise ValueError('One fixed partition required')
    table=pd.read_parquet(DEST/f'X_{profile}_{part[0]}.parquet').set_index('record_id')
    return table.loc[frame.record_id,PROFILE_FIELDS[profile]].reset_index(drop=True)


def grouped_folds(frame,n,seed):
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    folds=list(splitter.split(np.zeros(len(frame)),frame.figo_label,frame.patient_component))
    for fit,hold in folds:
        assert not set(frame.iloc[fit].patient_component)&set(frame.iloc[hold].patient_component)
        assert set(frame.iloc[fit].figo_label)=={0,1} and set(frame.iloc[hold].figo_label)=={0,1}
    assert sorted(np.concatenate([b for _,b in folds]).tolist())==list(range(len(frame)))
    return folds


def fit_model(frame,x,profile,family,params,seed):
    return ReportClassifier(profile,family,params,seed).fit(x,frame.figo_label.to_numpy(int),
        frame.patient_id.to_numpy(),frame.record_id.to_numpy(),frame.patient_component.to_numpy())


def grid_search(frame,x,profile,family,folds,seed,cache,label):
    """Exhaustive finite Cartesian grid; 300-tree points reuse exact prefixes.

    The model seed depends on fold, not candidate. Prefix equality is checked
    independently in unit tests and against a standalone real-data fit.
    """
    candidates=GRID[family]
    if family=='catboost':
        groups=list(ParameterGrid(dict(depth=[3,4,6],learning_rate=[.03,.1],
            l2_leaf_reg=[3,10],auto_class_weights=[None,'SqrtBalanced','Balanced'])))
    else:groups=candidates
    lookup={json.dumps(p,sort_keys=True):i for i,p in enumerate(candidates)}
    folder=cache/f'grid_{label}_{profile}_{family}';folder.mkdir(parents=True,exist_ok=True)
    def evaluate(item):
        k,params=item;path=folder/f'{k:03d}.joblib'
        if path.exists():return joblib.load(path)
        values=[300,700] if family=='catboost' else [None]
        preds={v:np.full(len(frame),np.nan) for v in values}
        foldscores={v:[] for v in values}
        start=time.monotonic()
        for j,(fit,hold) in enumerate(folds):
            cfg={**params,'iterations':700} if family=='catboost' else params
            model=fit_model(frame.iloc[fit],x.iloc[fit],profile,family,cfg,seed+j)
            for v in values:
                p=model.predict_positive(x.iloc[hold],ntree_end=v or 0)
                preds[v][hold]=p
                metrics=score(frame.iloc[hold].figo_label,p,frame.iloc[hold].patient_id)
                foldscores[v].append(dict(fold=j,macro_f1=metrics['patient_macro_f1'],
                    log_loss=metrics['patient_log_loss'],ap=metrics['patient_ap']))
        results=[]
        for v in values:
            cfg={**params,'iterations':v} if family=='catboost' else params
            f=pd.DataFrame(foldscores[v]);candidate=lookup[json.dumps(cfg,sort_keys=True)]
            row=dict(candidate=candidate,params=cfg,mean_f1=f.macro_f1.mean(),
                sd_f1=f.macro_f1.std(ddof=1),mean_log_loss=f.log_loss.mean(),mean_ap=f.ap.mean(),
                fit_seconds=time.monotonic()-start)
            results.append(dict(row=row,p=preds[v],fold_scores=foldscores[v]))
        joblib.dump(results,path,compress=3)
        return results
    output=[];completed=0;started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate,item) for item in enumerate(groups)]
        for future in as_completed(futures):
            output.extend(future.result());completed+=1
            if completed%6==0 or completed==len(groups):
                write_json(OUT/'progress.json',dict(stage='grid',label=label,profile=profile,family=family,
                    complete=completed,total=len(groups),seconds=time.monotonic()-started))
                print(f'{label} {profile} {family}: {completed}/{len(groups)} fits-groups',flush=True)
    output.sort(key=lambda r:r['row']['candidate'])
    assert len(output)==len(candidates) and all(np.isfinite(r['p']).all() for r in output)
    best=min(range(len(output)),key=lambda i:(-output[i]['row']['mean_f1'],output[i]['row']['mean_log_loss'],
        complexity(family,output[i]['row']['params'])))
    return output[best],output


def complexity(family,params):
    if family=='catboost':return (params['depth'],params['iterations'],-params['l2_leaf_reg'])
    return (params.get('interactions',False),params.get('n_knots',0),params['C'])


def choose_aggregation(frame,x,profile,family,params,folds,single_result,seed,cache,label):
    """Mode selection occurs INSIDE outer fit; outer assessment remains unseen."""
    path=cache/f'aggregation_{label}_{profile}_{family}.joblib'
    if path.exists():return joblib.load(path)
    if family=='linear':
        result=dict(mode='single',single_f1=single_result['row']['mean_f1'],ensemble_f1=None,
            single_log_loss=single_result['row']['mean_log_loss'],ensemble_log_loss=None,
            ensemble_p=None,params=params)
        joblib.dump(result,path);return result
    ensemble_p=np.full(len(frame),np.nan);scores=[]
    for j,(fit,hold) in enumerate(folds):
        f=frame.iloc[fit].reset_index(drop=True);xf=x.iloc[fit].reset_index(drop=True)
        subfolds=grouped_folds(f,5,seed+100+j)
        predicted=[]
        for k,(subfit,_) in enumerate(subfolds):
            model=fit_model(f.iloc[subfit],xf.iloc[subfit],profile,family,params,seed+j*10+k)
            assert not set(model.fit_components_)&set(frame.iloc[hold].patient_component)
            predicted.append(model.predict_positive(x.iloc[hold]))
        p=np.mean(predicted,axis=0);ensemble_p[hold]=p
        scores.append(score(frame.iloc[hold].figo_label,p,frame.iloc[hold].patient_id))
    f1=float(np.mean([s['patient_macro_f1'] for s in scores]));loss=float(np.mean([s['patient_log_loss'] for s in scores]))
    single_f1=single_result['row']['mean_f1'];single_loss=single_result['row']['mean_log_loss']
    mode='ensemble5' if f1>=single_f1+.005 and loss<=single_loss+.005 else 'single'
    result=dict(mode=mode,single_f1=single_f1,ensemble_f1=f1,single_log_loss=single_loss,
        ensemble_log_loss=loss,ensemble_p=ensemble_p,params=params)
    joblib.dump(result,path,compress=3);return result


def fit_aggregate(frame,x,profile,family,params,mode,seed):
    if mode=='single':return [fit_model(frame,x,profile,family,params,seed)]
    return [fit_model(frame.iloc[fit],x.iloc[fit],profile,family,params,seed+k)
        for k,(fit,_) in enumerate(grouped_folds(frame,5,seed+1000))]


def predict_aggregate(models,x):return np.mean([m.predict_positive(x) for m in models],axis=0)


def layouts():
    train=metadata('train');valid=metadata('validation')
    outer=grouped_folds(train,5,SEED+1000)
    records=[];support=[]
    for k,(fit,hold) in enumerate(outer):
        for i in hold:records.append(dict(stage='outer',outer_fold=k,fold=k,record_id=train.iloc[i].record_id))
        f=train.iloc[fit].reset_index(drop=True)
        for j,(_,h) in enumerate(grouped_folds(f,3,SEED+2000+k)):
            for i in h:records.append(dict(stage='inner',outer_fold=k,fold=j,record_id=f.iloc[i].record_id))
        for part,ids in [('fit',fit),('assessment',hold)]:
            d=train.iloc[ids]
            support.append(dict(outer_fold=k,part=part,records=len(d),patients=d.patient_id.nunique(),
                components=d.patient_component.nunique(),normal=int((d.figo_label==0).sum()),abnormal=int(d.figo_label.sum())))
    cal,op=grouped_folds(valid,2,SEED+3000)[0]
    validation=valid[['record_id','patient_id','patient_component']].copy()
    validation['role']='operating_point';validation.loc[cal,'role']='calibration'
    assert not set(valid.iloc[cal].patient_component)&set(valid.iloc[op].patient_component)
    pd.DataFrame(records).to_parquet(OUT/'fold_assignments.parquet',index=False)
    pd.DataFrame(support).to_csv(OUT/'outer_support.csv',index=False)
    validation.to_parquet(OUT/'validation_roles.parquet',index=False)
    return outer,cal,op


def nested_search(cache):
    train=metadata('train');outer,_,_=layouts()
    metrics=[];predictions=[];choices=[];grids=[]
    for k,(fit,hold) in enumerate(outer):
        f=train.iloc[fit].reset_index(drop=True);h=train.iloc[hold].reset_index(drop=True)
        inner=grouped_folds(f,3,SEED+2000+k)
        for profile in PROFILE_FIELDS:
            xf,xh=inputs(f,profile),inputs(h,profile)
            for family in FAMILIES:
                path=cache/f'outer_{k}_{profile}_{family}.joblib'
                if path.exists():payload=joblib.load(path)
                else:
                    selected,grid=grid_search(f,xf,profile,family,inner,SEED+4000+k*100,cache,f'outer{k}')
                    params=selected['row']['params']
                    agg=choose_aggregation(f,xf,profile,family,params,inner,selected,
                        SEED+5000+k*100,cache,f'outer{k}')
                    models=fit_aggregate(f,xf,profile,family,params,agg['mode'],SEED+6000+k*100)
                    assert all(not set(m.fit_components_)&set(h.patient_component) for m in models)
                    p=predict_aggregate(models,xh)
                    payload=dict(params=params,aggregation=agg,grid=[r['row'] for r in grid],p=p,
                        fit_records=f.record_id.to_list(),assessment_records=h.record_id.to_list())
                    joblib.dump(payload,path,compress=3)
                p=payload['p'];m=score(h.figo_label,p,h.patient_id)
                metrics.append(dict(profile=profile,family=family,outer_fold=k,**m))
                choices.append(dict(profile=profile,family=family,outer_fold=k,params=payload['params'],
                    **{a:b for a,b in payload['aggregation'].items() if a not in ['ensemble_p','params']}))
                tab=h[['record_id','patient_id','patient_component','figo_label']].copy()
                tab['p']=p;tab['profile']=profile;tab['family']=family;tab['outer_fold']=k
                predictions.append(tab)
                grids.extend(dict(profile=profile,family=family,outer_fold=k,**r) for r in payload['grid'])
                pd.DataFrame(metrics).to_csv(OUT/'nested_fold_metrics.csv',index=False)
                write_json(OUT/'progress.json',dict(stage='nested_complete',completed=len(metrics),total=75,
                    outer_fold=k,profile=profile,family=family))
                print(f'COMPLETE outer {k+1}/5 {profile} {family}: F1={m["patient_macro_f1"]:.4f}',flush=True)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'nested_predictions.parquet',index=False)
    pd.DataFrame(grids).to_json(OUT/'nested_grid_results.json',orient='records',indent=1)
    write_json(OUT/'nested_choices.json',choices)
    result=pd.DataFrame(metrics)
    summary=result.groupby(['profile','family']).agg(mean_f1=('patient_macro_f1','mean'),
        sd_f1=('patient_macro_f1','std'),mean_ap=('patient_ap','mean'),
        mean_recall=('patient_recall','mean'),mean_log_loss=('patient_log_loss','mean')).reset_index()
    summary.to_csv(OUT/'nested_summary.csv',index=False)
    return summary


def final_training(cache,summary):
    """Freeze EVERY choice without using test features/predictions/labels."""
    if (OUT/'selection_frozen.json').exists():
        frozen=json.loads((OUT/'selection_frozen.json').read_text())
        if frozen.get('cache_fingerprint')==cache.name:
            for relative,expected in frozen['export_sha256'].items():assert sha256(ROOT/relative)==expected,relative
            return frozen['choices'],frozen['champions']
    train,valid=metadata('train'),metadata('validation')
    folds=grouped_folds(train,3,SEED+7000)
    roles=pd.read_parquet(OUT/'validation_roles.parquet').set_index('record_id').loc[valid.record_id,'role'].to_numpy()
    cal=np.flatnonzero(roles=='calibration');op=np.flatnonzero(roles=='operating_point')
    choices=[];grids=[];predictions=[];exports={};champions={}
    for profile in PROFILE_FIELDS:
        candidate_summary=summary[summary.profile==profile].sort_values(['mean_f1','mean_log_loss'],ascending=[False,True])
        champions[profile]=candidate_summary.iloc[0].family
        xt,xv=inputs(train,profile),inputs(valid,profile)
        for family in FAMILIES:
            path=cache/f'final_{profile}_{family}.joblib'
            if path.exists():payload=joblib.load(path)
            else:
                selected,grid=grid_search(train,xt,profile,family,folds,SEED+8000,cache,'final')
                params=selected['row']['params']
                agg=choose_aggregation(train,xt,profile,family,params,folds,selected,SEED+9000,cache,'final')
                models=fit_aggregate(train,xt,profile,family,params,agg['mode'],SEED+10000)
                pv=predict_aggregate(models,xv)
                calibrator=SigmoidCalibration().fit(pv[cal],valid.iloc[cal].figo_label,valid.iloc[cal].patient_id)
                pc=calibrator.transform(pv)
                threshold=choose_research_threshold(valid.iloc[op].figo_label.to_numpy(int),pc[op],valid.iloc[op].patient_id)
                bundle=ReportBundle(profile,family,models,calibrator,threshold=.5)
                payload=dict(params=params,aggregation=agg,bundle=bundle,raw=pv,calibrated=pc,
                    research_threshold=threshold,grid=[r['row'] for r in grid])
                joblib.dump(payload,path,compress=3)
            export=EXPORT/f'{profile}_{family}.joblib';joblib.dump(payload['bundle'],export,compress=3)
            exports[str(export.relative_to(ROOT))]=sha256(export)
            for j,m in enumerate(payload['bundle'].models):
                if family=='catboost':
                    native=EXPORT/f'{profile}_catboost_{j}.cbm';m.model_.save_model(native)
                    exports[str(native.relative_to(ROOT))]=sha256(native)
                else:
                    scaler=EXPORT/f'{profile}_{family}_{j}_preprocessor.joblib';joblib.dump(m.basis_,scaler,compress=3)
                    exports[str(scaler.relative_to(ROOT))]=sha256(scaler)
            choices.append(dict(profile=profile,family=family,params=payload['params'],
                mode=payload['aggregation']['mode'],research_threshold=payload['research_threshold'],
                calibrator_slope=payload['bundle'].calibrator.slope_,calibrator_intercept=payload['bundle'].calibrator.intercept_,
                champion=family==champions[profile]))
            tab=valid[['record_id','patient_id','patient_component','figo_label']].copy()
            tab['role']=roles;tab['profile']=profile;tab['family']=family
            tab['p_raw']=payload['raw'];tab['p']=payload['calibrated'];predictions.append(tab)
            grids.extend(dict(profile=profile,family=family,**r) for r in payload['grid'])
            print(f'FROZEN {profile} {family}: {payload["aggregation"]["mode"]}',flush=True)
    write_json(OUT/'final_choices.json',choices)
    pd.DataFrame(grids).to_json(OUT/'final_grid_results.json',orient='records',indent=1)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'validation_predictions.parquet',index=False)
    write_json(OUT/'selection_frozen.json',dict(cache_fingerprint=cache.name,champions=champions,choices=choices,export_sha256=exports,
        test_predictions_created=False,selected_with='train nested OOF only',
        calibrators_fit='validation calibration only',threshold_fit='validation operating-point only'))
    return choices,champions


def test_once():
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    for relative,expected in frozen['export_sha256'].items():assert sha256(ROOT/relative)==expected,relative
    path=OUT/'test_predictions.parquet'
    if path.exists():
        receipt=json.loads((OUT/'test_evaluation_receipt.json').read_text())
        assert receipt['frozen_sha256']==sha256(OUT/'selection_frozen.json')
        assert receipt['predictions_sha256']==sha256(path)
        return pd.read_csv(OUT/'test_metrics.csv')
    test=metadata('test');train=metadata('train');rows=[];predictions=[]
    for item in frozen['choices']:
        profile,family=item['profile'],item['family']
        bundle=joblib.load(EXPORT/f'{profile}_{family}.joblib')
        x=inputs(test,profile);p=bundle.predict_proba(x)[:,1]
        for mode,threshold in [('primary_0.5',.5),('research_macro_threshold',item['research_threshold'])]:
            rows.append(dict(profile=profile,family=family,policy=mode,threshold=threshold,
                champion=item['champion'],**score(test.figo_label,p,test.patient_id,threshold)))
        tab=test[['record_id','patient_id','patient_component','figo_label']].copy()
        tab['p']=p;tab['p_raw']=bundle.raw_probability(x);tab['profile']=profile;tab['family']=family
        tab['core_available']=bundle.support(x).core_available.to_numpy();predictions.append(tab)
    prevalence=float(np.average(train.figo_label,weights=patient_weights(train.patient_id)))
    for name,value in [('dummy_probability',prevalence),('dummy_majority',float(prevalence>=.5))]:
        rows.append(dict(profile='all',family=name,policy='primary_0.5',threshold=.5,champion=False,
            **score(test.figo_label,np.full(len(test),value),test.patient_id)))
    pd.concat(predictions,ignore_index=True).to_parquet(path,index=False)
    pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
    write_json(OUT/'test_evaluation_receipt.json',dict(frozen_sha256=sha256(OUT/'selection_frozen.json'),
        predictions_sha256=sha256(path),metrics_sha256=sha256(OUT/'test_metrics.csv'),
        test_used_for_selection=False,one_frozen_evaluation=True))
    return pd.DataFrame(rows)


def run_all():
    _,cache=initialize();summary=nested_search(cache)
    final_training(cache,summary)
    return test_once()


if __name__=='__main__':print(run_all().to_string(index=False))
