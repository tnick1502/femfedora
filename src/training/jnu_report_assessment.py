"""Post-freeze uncertainty, failure analysis and robustness of report models."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.training.jnu_report_experiment import (OUT,EXPORT,ROOT,DEST,SEED,metadata,inputs,
    grouped_folds,fit_model,write_json)
from src.training.jnu_report_models import patient_weights,score


def check_tree_prefixes():
    """Verify every non-iteration grid combination on real TRAIN measurements."""
    from sklearn.model_selection import ParameterGrid
    train=metadata('train').iloc[:1500].reset_index(drop=True)
    x=inputs(train,'sonomed_fields');rows=[]
    grid=ParameterGrid(dict(depth=[3,4,6],learning_rate=[.03,.1],l2_leaf_reg=[3,10],
        auto_class_weights=[None,'SqrtBalanced','Balanced']))
    for params in grid:
        short=fit_model(train,x,'sonomed_fields','catboost',{**params,'iterations':300},SEED+14000)
        long=fit_model(train,x,'sonomed_fields','catboost',{**params,'iterations':700},SEED+14000)
        difference=float(np.max(np.abs(short.predict_positive(x)-long.predict_positive(x,ntree_end=300))))
        if difference>1e-13:raise AssertionError(f'Non-equivalent tree prefix: {params}, {difference}')
        rows.append(dict(**params,max_probability_difference=difference,train_rows=len(train)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'tree_prefix_reuse_checks.csv',index=False)
    return result


def bootstrap_scores(y,p,patients,components,draws,threshold=.5):
    """Component resampling; patient weights fixed before applying multiplicity.

    AP/AUC handle tied predictions at their complete threshold groups. Uses
    frozen probabilities; intervals do not include retraining/selection noise.
    """
    y=np.asarray(y,int);p=np.asarray(p,float)
    _,inverse=np.unique(components,return_inverse=True)
    base=patient_weights(patients)
    order=np.argsort(-p,kind='stable');sortedp=p[order];sorted_y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sortedp)!=0),len(y)-1]
    pred=p>=threshold
    masks=np.column_stack([(y==0)&~pred,(y==0)&pred,(y==1)&~pred,(y==1)&pred]).astype(float)
    results=[]
    eps=np.finfo(float).eps
    losses=-(y*np.log(np.clip(p,eps,1-eps))+(1-y)*np.log1p(-np.clip(p,eps,1-eps)))
    for start in range(0,len(draws),100):
        w=draws[start:start+100,inverse]*base[None,:]
        tn,fp,fn,tp=(w@masks).T
        divide=lambda a,b:np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
        weighted=w[:,order]
        cumulative_tp=np.cumsum(weighted*sorted_y,axis=1)[:,ends]
        cumulative_fp=np.cumsum(weighted*(1-sorted_y),axis=1)[:,ends]
        positive=cumulative_tp[:,-1];negative=cumulative_fp[:,-1]
        precision=divide(cumulative_tp,cumulative_tp+cumulative_fp)
        recall_increment=np.diff(np.column_stack([np.zeros(len(w)),cumulative_tp]),axis=1)/positive[:,None]
        ap=np.nansum(recall_increment*precision,axis=1)
        tpr=np.column_stack([np.zeros(len(w)),cumulative_tp/positive[:,None]])
        fpr=np.column_stack([np.zeros(len(w)),cumulative_fp/negative[:,None]])
        auc=np.trapezoid(tpr,fpr,axis=1)
        results.append(pd.DataFrame(dict(
            macro_f1=.5*(divide(2*tp,2*tp+fp+fn)+divide(2*tn,2*tn+fp+fn)),
            recall=divide(tp,tp+fn),specificity=divide(tn,tn+fp),
            precision=divide(tp,tp+fp),npv=divide(tn,tn+fn),
            ap=ap,roc_auc=auc,brier=(w@((p-y)**2))/w.sum(axis=1),
            log_loss=(w@losses)/w.sum(axis=1))))
    return pd.concat(results,ignore_index=True)


def uncertainty(repetitions=1000):
    predictions=pd.read_parquet(OUT/'test_predictions.parquet')
    test=metadata('test');n=test.patient_component.nunique()
    rng=np.random.default_rng(SEED+11000)
    draws=rng.multinomial(n,np.full(n,1/n),size=repetitions)
    np.savez_compressed(OUT/'bootstrap_component_multiplicities.npz',draws=draws)
    summary=[];paired=[];all_samples=[]
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    samples={}
    for (profile,family),frame in predictions.groupby(['profile','family'],sort=False):
        frame=frame.set_index('record_id').loc[test.record_id].reset_index()
        b=bootstrap_scores(frame.figo_label,frame.p,frame.patient_id,frame.patient_component,draws)
        point=score(frame.figo_label,frame.p,frame.patient_id)
        samples[(profile,family)]=b
        for metric in b:
            lo,hi=np.nanquantile(b[metric],[.025,.975])
            summary.append(dict(profile=profile,family=family,metric=metric,
                estimate=point[f'patient_{metric}'],ci_low=lo,ci_high=hi,
                repetitions=repetitions,champion=frozen['champions'][profile]==family))
        b['profile']=profile;b['family']=family;b['draw']=np.arange(repetitions);all_samples.append(b)
    for profile in frozen['champions']:
        for metric in ['macro_f1','recall','ap','brier']:
            delta=samples[(profile,'catboost')][metric]-samples[(profile,'spline_logit')][metric]
            lo,hi=np.nanquantile(delta,[.025,.975])
            paired.append(dict(profile=profile,metric=metric,comparison='CatBoost minus spline model',
                mean_bootstrap_difference=delta.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(summary).to_csv(OUT/'test_bootstrap_intervals.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_family_differences.csv',index=False)
    pd.concat(all_samples,ignore_index=True).to_parquet(OUT/'bootstrap_metric_samples.parquet',index=False)
    return pd.DataFrame(summary),pd.DataFrame(paired)


def calibration_tables():
    frame=pd.read_parquet(OUT/'test_predictions.parquet')
    rows=[];changes=[]
    for (profile,family),f in frame.groupby(['profile','family']):
        w=patient_weights(f.patient_id)
        bins=np.minimum((f.p.to_numpy()*10).astype(int),9)
        for k in range(10):
            mask=bins==k
            if not mask.any():continue
            rows.append(dict(profile=profile,family=family,bin=k,records=int(mask.sum()),
                patient_mass=w[mask].sum(),mean_probability=np.average(f.p[mask],weights=w[mask]),
                observed_fraction=np.average(f.figo_label[mask],weights=w[mask])))
        for name,col in [('uncalibrated','p_raw'),('sigmoid','p')]:
            m=score(f.figo_label,f[col],f.patient_id)
            changes.append(dict(profile=profile,family=family,probabilities=name,
                brier=m['patient_brier'],log_loss=m['patient_log_loss'],ap=m['patient_ap']))
    pd.DataFrame(rows).to_csv(OUT/'test_reliability_bins.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'calibration_changes.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(changes)


def robustness():
    """Frozen models only. No model/threshold is changed after these results."""
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    test=metadata('test');rows=[];changes=[]
    quality=pd.read_parquet(DEST/'measurements.parquet').set_index('record_id').loc[test.record_id]
    for profile,family in frozen['champions'].items():
        bundle=joblib.load(EXPORT/f'{profile}_{family}.joblib');x=inputs(test,profile)
        base=bundle.predict_proba(x)[:,1]
        scenarios={'original':x.copy(),'rounding':x.copy()}
        roundx=scenarios['rounding']
        roundx['baseline_bpm']=roundx.baseline_bpm.round(0)
        for c in [k for k in x if k.startswith(('stv','amplitude'))]:roundx[c]=roundx[c].round(1)
        stv=[c for c in x if c in ['stv_ms','stv_4s_ms','stv_bpm']]
        if stv:
            scenarios['missing_STV']=x.copy();scenarios['missing_STV'][stv]=np.nan
        scenarios['missing_baseline']=x.copy();scenarios['missing_baseline']['baseline_bpm']=np.nan
        for scenario,xx in scenarios.items():
            p=bundle.predict_proba(xx)[:,1];s=bundle.support(xx)
            m=score(test.figo_label,p,test.patient_id)
            rows.append(dict(profile=profile,family=family,scenario=scenario,
                core_coverage=s.core_available.mean(),complete_profile_coverage=s.all_profile_fields_available.mean(),
                median_probability_change=np.median(np.abs(p-base)),
                p95_probability_change=np.quantile(np.abs(p-base),.95),
                classification_flip_fraction=np.mean((p>=.5)!=(base>=.5)),
                forced_research_prediction=True,**m))
        masks={'full_cohort':np.ones(len(test),bool),
            'reference_quality_pass':quality.report_quality_eligible.to_numpy(bool),
            'reference_quality_fail':~quality.report_quality_eligible.to_numpy(bool),
            'core_missing':~bundle.support(x).core_available.to_numpy(bool)}
        for name,mask in masks.items():
            d=test.loc[mask];both=d.figo_label.nunique()==2 and len(d)>=20
            metrics=score(d.figo_label,base[mask],d.patient_id) if both else {}
            changes.append(dict(profile=profile,family=family,cohort=name,records=len(d),
                patients=d.patient_id.nunique(),abnormal_records=int(d.figo_label.sum()),
                metrics_supported=both,**metrics))
    pd.DataFrame(rows).to_csv(OUT/'stress_tests.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'quality_subgroups.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(changes)


def block_permute(frame,rng):
    """Label vectors permuted between equally sized patient groups.

    Preserves within-patient pattern and group size, but small singleton strata
    may remain unchanged. This is a sanity check, NOT a permutation p-value.
    """
    y=frame.figo_label.to_numpy(int).copy()
    groups=[np.asarray(v) for v in frame.groupby('patient_component',sort=True).indices.values()]
    strata={}
    for ids in groups:strata.setdefault(len(ids),[]).append(ids)
    changed=0
    for items in strata.values():
        order=rng.permutation(len(items))
        original=[y[ids].copy() for ids in items]
        for i,ids in enumerate(items):
            y[ids]=original[order[i]]
            changed+=int(order[i]!=i)*len(ids)
    return y,changed/len(y)


def negative_controls():
    """Predeclared fixed configurations, train-only held-out patient groups."""
    train=metadata('train');fit,hold=grouped_folds(train,5,SEED+1000)[0]
    f=train.iloc[fit].reset_index(drop=True);h=train.iloc[hold].reset_index(drop=True)
    profile='sonomed_fields';xf,xh=inputs(f,profile),inputs(h,profile)
    rows=[]
    for seed in range(5):
        yy,changed=block_permute(f,np.random.default_rng(SEED+12000+seed))
        permuted=f.copy();permuted['figo_label']=yy
        for family,params in [
            ('catboost',dict(depth=3,iterations=300,learning_rate=.1,l2_leaf_reg=10,auto_class_weights=None)),
            ('spline_logit',dict(C=.3,n_knots=5,interactions=True,weight_policy=None))]:
            model=fit_model(permuted,xf,profile,family,params,SEED+13000+seed)
            p=model.predict_positive(xh)
            rows.append(dict(control='same_size_blocks',family=family,repetition=seed,
                fit_records=len(f),assessment_records=len(h),shuffled_fit_group_row_fraction=changed,
                actual_changed_label_fraction=float(np.mean(yy!=f.figo_label.to_numpy())),
                **score(h.figo_label,p,h.patient_id)))
    # Equal-size block permutation retains group-size/target association. Add a
    # second sanity check with exactly one randomly chosen record per component.
    # The choice uses only grouping/seed; no time ordering or target selection.
    onefit=f.groupby('patient_component',group_keys=False).sample(n=1,random_state=SEED+15000).reset_index(drop=True)
    onehold=h.groupby('patient_component',group_keys=False).sample(n=1,random_state=SEED+15001).reset_index(drop=True)
    xfit,xhold=inputs(onefit,profile),inputs(onehold,profile)
    for seed in range(5):
        shuffled=onefit.copy()
        shuffled['figo_label']=np.random.default_rng(SEED+16000+seed).permutation(onefit.figo_label.to_numpy())
        for family,params in [
            ('catboost',dict(depth=3,iterations=300,learning_rate=.1,l2_leaf_reg=10,auto_class_weights=None)),
            ('spline_logit',dict(C=.3,n_knots=5,interactions=True,weight_policy=None))]:
            model=fit_model(shuffled,xfit,profile,family,params,SEED+17000+seed)
            p=model.predict_positive(xhold)
            rows.append(dict(control='one_record_per_component',family=family,repetition=seed,
                fit_records=len(onefit),assessment_records=len(onehold),shuffled_fit_group_row_fraction=np.nan,
                actual_changed_label_fraction=float(np.mean(shuffled.figo_label.to_numpy()!=onefit.figo_label.to_numpy())),
                **score(onehold.figo_label,p,onehold.patient_id)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'train_only_negative_controls.csv',index=False)
    return result


def duration_prediction_drift():
    """No FIGO labels assigned to shorter windows; this is NOT accuracy."""
    d=pd.read_parquet(ROOT/'research/experiments/jnu_ctg/report_features_v1/duration_prefix_train.parquet')
    frozen=json.loads((OUT/'selection_frozen.json').read_text());profile='variability'
    bundle=joblib.load(EXPORT/f'{profile}_{frozen["champions"][profile]}.joblib')
    reference=d.query('prefix_min==30').set_index('record_id').sort_index()
    rows=[]
    columns=['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm']
    p30=bundle.predict_proba(reference[columns])[:,1]
    for duration in [10,17,20,30]:
        part=d.query('prefix_min==@duration').set_index('record_id').loc[reference.index]
        p=bundle.predict_proba(part[columns])[:,1]
        rows.extend(dict(record_id=record,prefix_min=duration,p=p[i],p30=p30[i],
            absolute_change=abs(p[i]-p30[i]),no_short_window_labels=True)
            for i,record in enumerate(reference.index))
    result=pd.DataFrame(rows);result.to_parquet(OUT/'duration_probability_drift.parquet',index=False)
    return result


def error_cases():
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    test=metadata('test');predictions=pd.read_parquet(OUT/'test_predictions.parquet')
    rows=[]
    for profile,family in frozen['champions'].items():
        p=predictions[(predictions.profile==profile)&(predictions.family==family)].set_index('record_id').loc[test.record_id]
        x=inputs(test,profile).set_axis(test.record_id)
        frame=p[['patient_id','figo_label','p','core_available']].join(x)
        frame['predicted_class']=(frame.p>=.5).astype(int)
        frame=frame[frame.predicted_class!=frame.figo_label].copy()
        frame['error_type']=np.where(frame.figo_label==1,'false_negative','false_positive')
        frame['wrong_class_confidence']=np.where(frame.figo_label==1,1-frame.p,frame.p)
        frame['profile']=profile;frame['family']=family
        rows.append(frame.reset_index())
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(OUT/'test_error_cases.csv',index=False)
    return result


def run_assessment():
    from src.features.jnu_ctg_report_extraction import sha256
    frozen_hash=sha256(OUT/'selection_frozen.json');code_hash=sha256(Path(__file__))
    receipt_path=OUT/'assessment_receipt.json'
    if receipt_path.exists():
        cached=json.loads(receipt_path.read_text())
        if cached.get('frozen_sha256')==frozen_hash and cached.get('code_sha256')==code_hash:
            if all(sha256(ROOT/p)==h for p,h in cached['artifacts_sha256'].items()):
                return pd.read_csv(OUT/'test_bootstrap_intervals.csv'),pd.read_csv(OUT/'paired_family_differences.csv')
    intervals,paired=uncertainty(1000)
    calibration_tables();robustness();negative_controls();duration_prediction_drift();error_cases()
    names=['bootstrap_component_multiplicities.npz','test_bootstrap_intervals.csv','paired_family_differences.csv',
        'bootstrap_metric_samples.parquet','test_reliability_bins.csv','calibration_changes.csv','stress_tests.csv',
        'quality_subgroups.csv','train_only_negative_controls.csv','duration_probability_drift.parquet','test_error_cases.csv']
    write_json(receipt_path,dict(frozen_sha256=frozen_hash,code_sha256=code_hash,
        component_bootstrap_repetitions=1000,conditional_on_frozen_models=True,
        test_used_for_selection=False,negative_control_train_only=True,
        measurement_perturbations_not_clinically_validated=True,clinical_validation=False,
        artifacts_sha256={str((OUT/n).relative_to(ROOT)):sha256(OUT/n) for n in names}))
    return intervals,paired
