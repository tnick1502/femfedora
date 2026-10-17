"""Audited CTG v2: balanced recording folds, nested policy selection, soft voting.

The v1 modules/results remain historical evidence. No test metric is a selector.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from src.features.ctg_preprocessing import CTGInputGuard
from src.training.ctg_experiments import (ROOT, OUT, MODELS, SEED, CTG_FEATURES, FAMILIES,
    CAT_GRID, LOG_GRID, CTGFamily, decisions, heads_to_prob, normalize, tune_threshold,
    metrics, load_data, digest, write_json)
from src.training.ctg_audit_folds import balanced_group_folds, fold_inventory

AUDIT = OUT / 'audited_v2'
EXPORT = MODELS / 'audited_v2'
VERSION = 'ctg_audit_v2_20261017'


def environment():
    return {p: importlib.metadata.version(p) for p in
            ['numpy','pandas','scipy','scikit-learn','catboost','joblib','umap-learn']}


def context():
    data, legacy_fingerprint = load_data()
    paths = [Path(__file__), ROOT/'src/training/ctg_audit_folds.py',
             ROOT/'src/training/ctg_experiments.py', ROOT/'src/features/ctg_preprocessing.py']
    fingerprint = hashlib.sha256((legacy_fingerprint + ''.join(digest(p) for p in paths) +
        json.dumps(environment(),sort_keys=True) + VERSION).encode()).hexdigest()
    cache = ROOT/'tmp/ctg_audited_v2'/fingerprint[:16]
    for p in [AUDIT,EXPORT,cache]: p.mkdir(parents=True,exist_ok=True)
    return data, fingerprint, cache


def group_weights(frame):
    """Population: sample a recording uniformly, then one of its segments."""
    w = 1/frame.source_file.map(frame.source_file.value_counts()).to_numpy(dtype=float)
    return w * len(w)/w.sum()


def score(frame, p, pred=None):
    y = frame.NSP.to_numpy(dtype=int)-1
    p = normalize(p)
    pred = decisions(p) if pred is None else np.asarray(pred)
    values = metrics(y,p,pred)
    w=group_weights(frame)
    cm=confusion_matrix(y,pred,labels=[0,1,2],sample_weight=w)
    recall=np.diag(cm)/np.maximum(cm.sum(axis=1),1e-12)
    values.update(file_macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1,2],sample_weight=w,zero_division=0)),
        file_recall_path=float(recall[2]), file_log_loss=float(log_loss(y,p,labels=[0,1,2],sample_weight=w)),
        file_ap_path=float(average_precision_score(y==2,p[:,2],sample_weight=w)))
    return values


def soft_vote(models, X, family):
    """Average class probabilities, never products of averaged conditional heads."""
    return normalize(np.mean([heads_to_prob(m.predict_heads(X),family) for m in models],axis=0))


class ProbabilityCalibrator:
    def __init__(self, method='raw'): self.method=method

    def fit(self,p,y):
        p=normalize(p); y=np.asarray(y,dtype=int)
        if self.method=='temperature':
            logits=np.log(np.clip(p,1e-8,1))
            objective=lambda t: log_loss(y,softmax(logits/np.exp(t),axis=1),labels=[0,1,2])
            result=minimize_scalar(objective,bounds=(-2.,2.),method='bounded')
            if not result.success: raise RuntimeError('Temperature calibration failed')
            self.temperature_=float(np.exp(result.x))
        elif self.method=='sigmoid':
            self.models_=[]
            for j in range(3):
                x=np.log(p[:,j]/np.maximum(1-p[:,j],1e-8)).reshape(-1,1)
                self.models_.append(LogisticRegression(C=1.,max_iter=1000).fit(x,(y==j).astype(int)))
        elif self.method!='raw': raise ValueError(self.method)
        return self

    def predict_proba(self,p):
        p=normalize(p)
        if self.method=='raw': return p
        if self.method=='temperature': return softmax(np.log(p)/self.temperature_,axis=1)
        return normalize(np.column_stack([m.predict_proba(
            np.log(p[:,j]/np.maximum(1-p[:,j],1e-8)).reshape(-1,1))[:,1] for j,m in enumerate(self.models_)]))


def select_policy(frame, p, seed):
    """All inputs belong to the current fit side, never the outer holdout/test.

    Calibration optimizes a proper probability score (log loss). Threshold policy
    is evaluated with recording cross-fitting and needs >=.005 F1 improvement.
    Its internal selection bias is assessed by the untouched outer folds.
    """
    y=frame.NSP.to_numpy(dtype=int)-1
    folds=balanced_group_folds(frame,3,seed)
    rows=[]; crossfit={}; threshold_predictions={}
    for method in ['raw','temperature','sigmoid']:
        cf=np.zeros_like(p); threshold_pred=np.zeros(len(y),dtype=int)
        for fit,hold in folds:
            cal=ProbabilityCalibrator(method).fit(p[fit],y[fit])
            cf[hold]=cal.predict_proba(p[hold])
            cutoff=tune_threshold(y[fit],cal.predict_proba(p[fit]))
            threshold_pred[hold]=decisions(cf[hold],cutoff)
        crossfit[method]=cf;threshold_predictions[method]=threshold_pred
        rows.append(dict(calibration=method,decision='argmax',**score(frame,cf)))
        rows.append(dict(calibration=method,decision='threshold',**score(frame,cf,threshold_pred)))
    table=pd.DataFrame(rows)
    # Proper scoring for probabilities; deterministic preference for simpler ties.
    cal_order={m:i for i,m in enumerate(['raw','temperature','sigmoid'])}
    chosen=table[table.decision=='argmax'].assign(complexity=lambda d:d.calibration.map(cal_order)).sort_values(
        ['log_loss','complexity']).iloc[0].calibration
    arg=table[(table.calibration==chosen)&(table.decision=='argmax')].iloc[0]
    thr=table[(table.calibration==chosen)&(table.decision=='threshold')].iloc[0]
    decision='threshold' if thr.macro_f1>=arg.macro_f1+.005 else 'argmax'
    cutoff=tune_threshold(y,crossfit[chosen]) if decision=='threshold' else None
    cal=ProbabilityCalibrator(chosen).fit(p,y)
    return dict(calibration=chosen,decision=decision,threshold=cutoff,calibrator=cal,
        validation_macro_f1=float(thr.macro_f1 if decision=='threshold' else arg.macro_f1),
        validation_log_loss=float(arg.log_loss)),table,crossfit[chosen]


class AuditedCTGBundle:
    def __init__(self,models,family,policy):
        self.models,self.family,self.policy=models,family,policy
        self.classes_=np.array([1,2,3])

    def predict_proba(self,X):
        X=CTGInputGuard().transform(X)
        raw=soft_vote(self.models,X,self.family)
        return self.policy['calibrator'].predict_proba(raw)

    def predict(self,X):
        return decisions(self.predict_proba(X),self.policy['threshold'])+1


def search(frame,family,folds,seed):
    params_grid=LOG_GRID if family=='logistic' else CAT_GRID
    y=frame.NSP.to_numpy(dtype=int)-1;X=frame[list(CTG_FEATURES)]
    def evaluate(item):
        candidate,params=item
        p=np.zeros((len(frame),3));fold_scores=[]
        for j,(fit,hold) in enumerate(folds):
            model=CTGFamily(family,params,seed+j).fit(X.iloc[fit],y[fit])
            p[hold]=heads_to_prob(model.predict_heads(X.iloc[hold]),family)
            fold_scores.append(score(frame.iloc[hold],p[hold]))
        return dict(candidate=candidate,params=params,
            mean_macro_f1=float(np.mean([r['macro_f1'] for r in fold_scores])),
            mean_log_loss=float(np.mean([r['log_loss'] for r in fold_scores])),
            mean_file_macro_f1=float(np.mean([r['file_macro_f1'] for r in fold_scores])),
            fold_macro_f1=[r['macro_f1'] for r in fold_scores]),p
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(evaluate,enumerate(params_grid)))
    best=min(range(len(results)),key=lambda i:(-results[i][0]['mean_macro_f1'],results[i][0]['mean_log_loss'],i))
    return results[best][0]['params'],results[best][1],[r[0] for r in results]


def prediction_frame(frame,p,pred,**extra):
    result=frame[['source_excel_row','source_file','NSP']].copy()
    for j in range(3):result[f'p{j}']=p[:,j]
    result['pred']=np.asarray(pred)+1
    for key,value in extra.items():result[key]=value
    return result


def run_nested():
    data,fingerprint,cache=context()
    train=data[data.split=='train'].reset_index(drop=True)
    folds=balanced_group_folds(train,5,SEED+21000)
    inventory=fold_inventory(train,folds)
    inventory.to_csv(AUDIT/'outer_fold_support.csv',index=False)
    layout=[];inner_layout=[];rows=[];grids=[];predictions=[];policies=[]
    for k,(fit,hold) in enumerate(folds):
        for i in hold:layout.append(dict(source_excel_row=int(train.iloc[i].source_excel_row),outer_fold=k))
        f,h=train.iloc[fit].reset_index(drop=True),train.iloc[hold]
        inner=balanced_group_folds(f,3,SEED+22000+k)
        for j,(_,iv) in enumerate(inner):
            inner_layout.extend(dict(outer_fold=k,inner_fold=j,source_excel_row=int(f.iloc[i].source_excel_row)) for i in iv)
        for family in FAMILIES:
            path=cache/f'nested_{k}_{family}.joblib'
            if path.exists():payload=joblib.load(path)
            else:
                start=time.monotonic()
                params,p_oof,grid=search(f,family,inner,SEED+23000+k*100)
                policy,policy_grid,_=select_policy(f,p_oof,SEED+24000+k)
                model=CTGFamily(family,params,SEED+25000+k).fit(f[list(CTG_FEATURES)],f.NSP.to_numpy()-1)
                raw=heads_to_prob(model.predict_heads(h[list(CTG_FEATURES)]),family)
                p=policy['calibrator'].predict_proba(raw)
                pred=decisions(p,policy['threshold'])
                payload=dict(params=params,policy=policy,policy_grid=policy_grid,p=p,pred=pred,grid=grid,
                    seconds=time.monotonic()-start)
                joblib.dump(payload,path)
            rows.append(dict(family=family,outer_fold=k,**score(h,payload['p'],payload['pred'])))
            predictions.append(prediction_frame(h,payload['p'],payload['pred'],family=family,outer_fold=k))
            grids.extend(dict(family=family,outer_fold=k,**r) for r in payload['grid'])
            policies.append(dict(family=family,outer_fold=k,params=payload['params'],
                **{a:b for a,b in payload['policy'].items() if a!='calibrator'}))
            pd.DataFrame(rows).to_csv(AUDIT/'nested_fold_metrics.csv',index=False)
            write_json(AUDIT/'progress.json',dict(stage='nested',completed=len(rows),total=30,
                family=family,outer_fold=k,fingerprint=fingerprint))
            print(f'Audited nested {k+1}/5 {family}: {payload["seconds"]:.1f}s',flush=True)
    results=pd.DataFrame(rows)
    summary=results.groupby('family').agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),
        mean_recall_path=('recall_path','mean'),mean_ap_path=('ap_path','mean'),
        mean_log_loss=('log_loss','mean'),mean_file_macro_f1=('file_macro_f1','mean')).reset_index()
    summary=summary.sort_values(['mean_macro_f1','mean_log_loss'],ascending=[False,True]).reset_index(drop=True)
    best=summary.iloc[0]
    # Retain a second strong contender only if within .01 on the pre-test primary.
    shortlist=summary[summary.mean_macro_f1>=best.mean_macro_f1-.01].head(2).family.tolist()
    summary['shortlisted']=summary.family.isin(shortlist)
    summary.to_csv(AUDIT/'nested_summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(AUDIT/'nested_predictions.parquet',index=False)
    pd.DataFrame(grids).to_json(AUDIT/'nested_grid.json',orient='records',indent=2)
    pd.DataFrame(layout).to_csv(AUDIT/'outer_folds.csv',index=False)
    pd.DataFrame(inner_layout).to_csv(AUDIT/'inner_folds.csv',index=False)
    write_json(AUDIT/'nested_policies.json',policies)
    write_json(AUDIT/'selection.json',dict(fingerprint=fingerprint,primary=best.family,shortlist=shortlist,
        selection='mean outer-fold segment macro-F1; entire policy chosen within fit folds',
        shortlist_rule='top two within .01 of best mean macro-F1; no test',versions=environment()))
    return summary,results


def paired_intervals(predictions, reference, repetitions=2000):
    """Paired recording bootstrap of fixed predictors, not retraining uncertainty."""
    candidates=predictions.candidate.unique()
    aligned={c:p.sort_values('source_excel_row').reset_index(drop=True) for c,p in predictions.groupby('candidate')}
    first=aligned[reference];groups=first.source_file.unique()
    indices={g:np.flatnonzero(first.source_file.to_numpy()==g) for g in groups}
    rng=np.random.default_rng(SEED+33000)
    points={};samples={c:[] for c in candidates}
    for c,f in aligned.items():
        assert f.source_excel_row.tolist()==first.source_excel_row.tolist()
        points[c]=score(f,f[['p0','p1','p2']].to_numpy(),f.pred.to_numpy()-1)
    for _ in range(repetitions):
        ix=np.concatenate([indices[g] for g in rng.choice(groups,len(groups),replace=True)])
        for c,f in aligned.items():
            sub=f.iloc[ix]
            # Metrics only; avoid recomputing all proper scores in bootstrap.
            y=sub.NSP.to_numpy()-1;pred=sub.pred.to_numpy()-1
            recall=float(np.mean(pred[y==2]==2)) if (y==2).any() else np.nan
            samples[c].append((f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0),recall))
    rows=[]
    for c in candidates:
        values=np.asarray(samples[c]);delta=values-np.asarray(samples[reference])
        rows.append(dict(candidate=c,macro_f1=points[c]['macro_f1'],recall_path=points[c]['recall_path'],
            macro_f1_low=float(np.nanquantile(values[:,0],.025)),macro_f1_high=float(np.nanquantile(values[:,0],.975)),
            recall_low=float(np.nanquantile(values[:,1],.025)),recall_high=float(np.nanquantile(values[:,1],.975)),
            delta_macro_f1_vs_primary=points[c]['macro_f1']-points[reference]['macro_f1'],
            delta_low=float(np.nanquantile(delta[:,0],.025)),delta_high=float(np.nanquantile(delta[:,0],.975))))
    return pd.DataFrame(rows)


def run_final():
    data,fingerprint,cache=context()
    selected=json.loads((AUDIT/'selection.json').read_text())
    if selected['fingerprint']!=fingerprint:raise ValueError('Stale nested selection')
    train=data[data.split=='train'].reset_index(drop=True)
    valid=data[data.split=='validation'].reset_index(drop=True)
    test=data[data.split=='test'].reset_index(drop=True)
    inner=balanced_group_folds(train,3,SEED+31000)
    calibration_folds=balanced_group_folds(valid,3,SEED+32000)
    fold_inventory(valid,calibration_folds).to_csv(AUDIT/'validation_fold_support.csv',index=False)
    fits={};validation=[];grids=[];parameters=[]
    shortlist=selected['shortlist']
    for family in FAMILIES:
        path=cache/f'final_{family}.joblib'
        if path.exists():payload=joblib.load(path)
        else:
            params,_,grid=search(train,family,inner,SEED+35000)
            model=CTGFamily(family,params,SEED+36000).fit(train[list(CTG_FEATURES)],train.NSP.to_numpy()-1)
            payload=dict(params=params,models=[model],grid=grid)
            joblib.dump(payload,path)
        model_sets={'single':payload['models']}
        grids.extend(dict(family=family,**r) for r in payload['grid'])
        parameters.append(dict(family=family,params=payload['params']))
        if family in shortlist:
            path=cache/f'ensemble_{family}.joblib'
            if path.exists():ensemble=joblib.load(path)
            else:
                ensemble=[CTGFamily(family,payload['params'],SEED+37000+k).fit(
                    train.iloc[fit][list(CTG_FEATURES)],train.iloc[fit].NSP.to_numpy()-1)
                    for k,(fit,_) in enumerate(balanced_group_folds(train,5,SEED+21000))]
                joblib.dump(ensemble,path)
            model_sets['fold_ensemble']=ensemble
        for architecture,models in model_sets.items():
            p=soft_vote(models,valid[list(CTG_FEATURES)],family)
            policy,table,cf=select_policy(valid,p,SEED+32000)
            candidate=f'{family}/{architecture}'
            fits[candidate]=AuditedCTGBundle(models,family,policy)
            for r in table.to_dict('records'):validation.append(dict(candidate=candidate,**r))
        write_json(AUDIT/'progress.json',dict(stage='final',family=family,fingerprint=fingerprint))
        print(f'Audited final {family}: complete',flush=True)
    choices={}
    for family in shortlist:
        a=f'{family}/single';b=f'{family}/fold_ensemble'
        choices[family]=b if fits[b].policy['validation_macro_f1']>=fits[a].policy['validation_macro_f1']+.01 else a
    primary=choices[selected['primary']]
    lock=dict(fingerprint=fingerprint,primary=primary,shortlist=[choices[f] for f in shortlist],
        selection_source='nested train CV then grouped validation policy and architecture',
        ensemble_rule='soft-vote class probabilities; require >=.01 validation macro-F1 improvement',
        probability_selection='group cross-fitted log loss, raw vs temperature vs sigmoid',
        threshold_rule='group-cross-fitted gain >=.005 macro-F1; refit threshold on cross-fitted probabilities',
        parameters=parameters,policies={c:{k:v for k,v in b.policy.items() if k!='calibrator'} for c,b in fits.items()},
        test_used_for_selection=False,test_status='previously inspected benchmark; exploratory retest',versions=environment())
    # The lock is persisted BEFORE any test prediction or metric in this procedure.
    write_json(AUDIT/'locked_decision.json',lock)
    pd.DataFrame(validation).to_csv(AUDIT/'validation_policy_metrics.csv',index=False)
    pd.DataFrame(grids).to_json(AUDIT/'final_grid.json',orient='records',indent=2)
    metrics_rows=[];predictions=[]
    for candidate,bundle in fits.items():
        p=bundle.predict_proba(test[list(CTG_FEATURES)]);pred=bundle.predict(test[list(CTG_FEATURES)])-1
        metrics_rows.append(dict(candidate=candidate,primary=candidate==primary,shortlisted=candidate in lock['shortlist'],**score(test,p,pred)))
        predictions.append(prediction_frame(test,p,pred,candidate=candidate))
        joblib.dump(bundle,EXPORT/f'{candidate.replace("/","_")}.joblib')
    pd.DataFrame(metrics_rows).to_csv(AUDIT/'test_metrics.csv',index=False)
    prediction_table=pd.concat(predictions,ignore_index=True)
    prediction_table.to_parquet(AUDIT/'test_predictions.parquet',index=False)
    chosen_predictions=prediction_table[prediction_table.candidate.isin(lock['shortlist'])]
    paired_intervals(chosen_predictions,primary,repetitions=1500).to_csv(AUDIT/'test_paired_intervals.csv',index=False)
    joblib.dump(fits[primary],EXPORT/'champion.joblib')
    for candidate in lock['shortlist']:
        b=fits[candidate]
        for i,m in enumerate(b.models):
            if m.family!='logistic':
                for j,head in enumerate(m.models_):head.save_model(str(EXPORT/f'{candidate.replace("/","_")}_fold{i}_head{j}.cbm'))
    write_json(EXPORT/'model_card.json',dict(lock,features=list(CTG_FEATURES),classes={1:'normal',2:'suspect',3:'pathologic'},
        train_segments=len(train),calibration_segments=len(valid),test_segments=len(test),
        model_sha256=digest(EXPORT/'champion.joblib'),test_metrics=[r for r in metrics_rows if r['primary']],
        limitations=['Expert CTG labels, not neonatal outcomes','No patient IDs; source file is a proxy',
            'Only 10 pathologic test recordings; intervals conditional on fitted model',
            'Known test reused during method audit; external validation required',
            'CV algorithm/selection is evaluated; no guarantee of globally best parameters']))
    write_json(AUDIT/'progress.json',dict(stage='complete',fingerprint=fingerprint,primary=primary))
    return pd.DataFrame(metrics_rows),lock
