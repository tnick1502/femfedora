"""Conditional component-bootstrap and error audits after revised CV completes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.training.ctg_review_evidence import ROOT,OUT as REVIEW,write_json
from src.training.ctg_review_models import weights,multiclass_scores,binary_scores
from src.training.jnu_report_assessment import bootstrap_scores

UCI=ROOT/'research/experiments/ctg_uci/reviewed_v3'
JNU=ROOT/'research/experiments/jnu_ctg/reviewed_v2'


def multiclass_bootstrap(frame,draws):
    y=frame.NSP.to_numpy(int)-1;p=frame[['p0','p1','p2']].to_numpy();pred=p.argmax(axis=1)
    groups=frame.review_group.to_numpy();_,inverse=np.unique(groups,return_inverse=True)
    base=weights(groups);mask=np.column_stack([(y==i)&(pred==j) for i in range(3) for j in range(3)]).astype(float)
    b=bootstrap_scores((y==2).astype(int),p[:,2],groups,groups,draws)
    result=[]
    for start in range(0,len(draws),100):
        w=draws[start:start+100,inverse]*base;cm=(w@mask).reshape(-1,3,3)
        diag=cm[:,np.arange(3),np.arange(3)];den=cm.sum(axis=1)+cm.sum(axis=2)
        f1=np.divide(2*diag,den,out=np.zeros_like(diag),where=den>0)
        recall=np.divide(diag,cm.sum(axis=2),out=np.full_like(diag,np.nan),where=cm.sum(axis=2)>0)
        loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
        result.append(pd.DataFrame(dict(macro_f1=f1.mean(axis=1),recall_normal=recall[:,0],recall_suspect=recall[:,1],recall_path=recall[:,2],log_loss=(w@loss)/w.sum(axis=1))))
    r=pd.concat(result,ignore_index=True);r['ap_path']=b.ap;r['auc_path']=b.roc_auc;return r


def uci_assessment(repetitions=1000):
    predictions=pd.read_parquet(UCI/'nested_predictions.parquet');rows=[];samples=[];confusions=[];patterns=[];per_class=[]
    first=predictions[(predictions.repetition==0)&(predictions.family=='selected_procedure')].sort_values('source_excel_row')
    n=first.review_group.nunique();rng=np.random.default_rng(20261019);draws=rng.multinomial(n,np.full(n,1/n),size=repetitions)
    np.savez_compressed(UCI/'bootstrap_proxy_multiplicities.npz',draws=draws)
    point=pd.read_csv(UCI/'repeat_metrics.csv').groupby('family').mean(numeric_only=True)
    for family,d in predictions.groupby('family'):
        bs=[]
        for rep,g in d.groupby('repetition'):
            g=g.sort_values('source_excel_row');assert g.source_excel_row.tolist()==first.source_excel_row.tolist()
            bs.append(multiclass_bootstrap(g,draws))
            y=g.NSP.to_numpy(int)-1;pred=g[['p0','p1','p2']].to_numpy().argmax(axis=1)
            cm=confusion_matrix(y,pred,labels=[0,1,2])
            from sklearn.metrics import average_precision_score,roc_auc_score
            w=weights(g.review_group);prob=g[['p0','p1','p2']].to_numpy()
            weighted=confusion_matrix(y,pred,labels=[0,1,2],sample_weight=w)
            for label in range(3):
                tp=weighted[label,label];actual=weighted[label].sum();predicted=weighted[:,label].sum()
                per_class.append(dict(family=family,repetition=rep,label=label,
                    recall=tp/actual,precision=tp/predicted if predicted else 0,
                    ap=average_precision_score(y==label,prob[:,label],sample_weight=w),
                    roc_auc=roc_auc_score(y==label,prob[:,label],sample_weight=w)))
            for i in range(3):
                for j in range(3):confusions.append(dict(family=family,repetition=rep,actual=i,predicted=j,segments=int(cm[i,j])))
            if family=='selected_procedure':
                for label,z in g.assign(predicted=pred).groupby('CLASS'):
                    patterns.append(dict(repetition=rep,CLASS=label,segments=len(z),proxy_groups=z.review_group.nunique(),
                        normal_predictions=int((z.predicted==0).sum()),suspect_predictions=int((z.predicted==1).sum()),path_predictions=int((z.predicted==2).sum())))
        average=sum(bs)/len(bs)
        for metric in average:
            lo,hi=average[metric].quantile([.025,.975])
            rows.append(dict(family=family,metric=metric,estimate=float(point.loc[family,'group_'+metric]),ci_low=lo,ci_high=hi,bootstrap=repetitions,
                estimand='mean metric over three repetitions; fixed CV fits, group resampling'))
        average['family']=family;average['draw']=np.arange(repetitions);samples.append(average)
    pd.DataFrame(rows).to_csv(UCI/'conditional_intervals.csv',index=False)
    pd.concat(samples,ignore_index=True).to_parquet(UCI/'bootstrap_metric_samples.parquet',index=False)
    pd.DataFrame(confusions).to_csv(UCI/'repeat_confusions.csv',index=False)
    pd.DataFrame(patterns).to_csv(UCI/'CLASS_pattern_errors.csv',index=False)
    pd.DataFrame(per_class).to_csv(UCI/'per_class_metrics.csv',index=False)
    assigned=pd.read_parquet(UCI/'outer_assignments.parquet').merge(pd.read_parquet(ROOT/'data/processed/ctg_uci/ctg_primary.parquet')[['source_excel_row','NSP']],on='source_excel_row',validate='many_to_one')
    assigned.groupby(['repetition','outer_fold','NSP']).agg(segments=('source_excel_row','size'),proxy_groups=('review_group','nunique')).reset_index().to_csv(UCI/'outer_class_support.csv',index=False)
    return pd.DataFrame(rows)


def jnu_calibration_support():
    pred=pd.read_parquet(JNU/'nested_predictions.parquet');rows=[]
    for family,g in pred[(pred.profile=='report5')&pred.accepted].groupby('family'):
        g=g.copy();g['_weight']=weights(g.patient_id);g['bin']=np.minimum((g.p*10).astype(int),9)
        for k,z in g.groupby('bin'):
            w=z['_weight'].to_numpy()
            rows.append(dict(family=family,bin=int(k),records=len(z),patient_components=z.patient_component.nunique(),
                mean_probability=np.average(z.p,weights=w),observed_fraction=np.average(z.figo_label,weights=w)))
    result=pd.DataFrame(rows);result.to_csv(JNU/'calibration_bin_support.csv',index=False)
    from src.training.jnu_review_experiment import load_data
    source=load_data()[['record_id','baseline_bpm','stv_ms','amplitude_bpm','acc15_per_10min','dec15_per_10min']]
    errors=pred[(pred.family=='selected_procedure')&pred.accepted&((pred.p>=.5).astype(int)!=pred.figo_label)].copy()
    errors['error_type']=np.where(errors.figo_label==1,'false_negative','false_positive')
    errors['wrong_class_confidence']=np.where(errors.figo_label==1,1-errors.p,errors.p)
    errors=errors.merge(source,on='record_id',validate='many_to_one');errors.to_csv(JNU/'selected_procedure_error_cases.csv',index=False)
    return result


def jnu_assessment(repetitions=1000):
    pred=pd.read_parquet(JNU/'nested_predictions.parquet');rows=[];coverage=[];confusions=[];quality=[];all_samples=[]
    frozen=pd.read_csv(JNU/'profile_metrics.csv').set_index(['profile','family'])
    for profile,frame in pred.groupby('profile'):
        valid=frame[(frame.family=='selected_procedure')&frame.accepted].sort_values('record_id')
        n=valid.patient_component.nunique();rng=np.random.default_rng(20261019);draws=rng.multinomial(n,np.full(n,1/n),size=repetitions)
        for family,g in frame.groupby('family'):
            total=g.copy();g=g[g.accepted].sort_values('record_id');assert g.record_id.tolist()==valid.record_id.tolist()
            b=bootstrap_scores(g.figo_label,g.p,g.patient_id,g.patient_component,draws)
            for metric in ['macro_f1','recall','specificity','precision','ap','roc_auc','log_loss','brier']:
                lo,hi=b[metric].quantile([.025,.975]);column='group_auc' if metric=='roc_auc' else 'group_'+metric
                rows.append(dict(profile=profile,family=family,metric=metric,estimate=float(frozen.loc[(profile,family),column]),ci_low=lo,ci_high=hi,bootstrap=repetitions,
                    conditional_on='accepted inputs and fixed exploratory CV fits'))
            b['profile']=profile;b['family']=family;b['draw']=np.arange(repetitions);all_samples.append(b)
            predicted=np.where(total.accepted,(total.p>=.5).astype(int),2)
            for actual in [0,1]:
                for decision in [0,1,2]:confusions.append(dict(profile=profile,family=family,actual=actual,predicted=decision,records=int(((total.figo_label==actual)&(predicted==decision)).sum())))
            if family=='selected_procedure':
                for flag,z in total.groupby('report_quality_eligible'):
                    zz=z[z.accepted]
                    quality.append(dict(profile=profile,quality_flag=bool(flag),source_records=len(z),accepted_records=len(zz),
                        **binary_scores(zz.figo_label,zz.p,zz.patient_id)))
        for label,z in frame[frame.family=='selected_procedure'].groupby('figo_label'):
            coverage.append(dict(profile=profile,actual=int(label),records=len(z),accepted=int(z.accepted.sum()),abstained=int((~z.accepted).sum()),coverage=z.accepted.mean()))
    pd.DataFrame(rows).to_csv(JNU/'conditional_intervals.csv',index=False)
    pd.DataFrame(coverage).to_csv(JNU/'coverage_summary.csv',index=False)
    pd.DataFrame(confusions).to_csv(JNU/'confusions_with_abstention.csv',index=False)
    pd.DataFrame(quality).to_csv(JNU/'quality_subgroups.csv',index=False)
    pd.concat(all_samples,ignore_index=True).to_parquet(JNU/'bootstrap_metric_samples.parquet',index=False)
    from src.training.jnu_review_experiment import load_data
    measurements=load_data()[['record_id','stv_ms','amplitude_bpm']]
    signatures=pd.read_parquet(REVIEW/'jnu_uc_signature_records.parquet')[['record_id','raw_uc_max']]
    diagnostic=pred[(pred.profile=='report5')&(pred.family=='selected_procedure')].merge(measurements,on='record_id',validate='one_to_one').merge(signatures,on='record_id',validate='one_to_one')
    diagnostic['stv_stratum']=pd.cut(diagnostic.stv_ms,[-np.inf,4,15,np.inf],right=False).astype(str)
    diagnostic['uc_signature']=np.where(diagnostic.raw_uc_max==100,'max100',np.where(diagnostic.raw_uc_max==127,'max127','other'))
    subgroups=[]
    for kind in ['stv_stratum','uc_signature']:
        for name,g in diagnostic.groupby(kind):
            z=g[g.accepted];support=len(z)>=50 and z.figo_label.value_counts().min()>=10 and z.figo_label.nunique()==2
            subgroups.append(dict(grouping=kind,stratum=name,source_records=len(g),accepted_records=len(z),
                abnormal_records=int(g.figo_label.sum()),metrics_supported=bool(support),
                **(binary_scores(z.figo_label,z.p,z.patient_id) if support else {})))
    pd.DataFrame(subgroups).to_csv(JNU/'input_subgroup_metrics.csv',index=False)
    from src.features.ctg_report_contract import research_patterns,prepare_report
    source=load_data();source=source[(source.split=='train')&np.isfinite(source[['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm']]).all(axis=1)]
    support=[]
    for name,report in research_patterns().items():
        target,_=prepare_report(report,'report5');r=target.iloc[0]
        mask=(source.baseline_bpm-r.baseline_bpm).abs()<=10
        mask&=(source.stv_ms-r.stv_ms).abs()<=1
        mask&=(source.amplitude_bpm-r.amplitude_bpm).abs()<=3
        mask&=np.isclose(source.acc15_per_10min,r.acc15_per_10min,rtol=0,atol=1e-9)
        mask&=np.isclose(source.dec15_per_10min,r.dec15_per_10min,rtol=0,atol=1e-9)
        z=source[mask]
        support.append(dict(pattern=name,original_train_records_in_box=len(z),patients=z.patient_id.nunique(),
            FIGO_abnormal_fraction_if_n_ge20=z.figo_label.mean() if len(z)>=20 else np.nan,
            box='baseline ±10 BPM, STV ±1 ms, range ±3 BPM; AC/DC rates exact; descriptive not clinical validation'))
    pd.DataFrame(support).to_csv(JNU/'synthetic_joint_support.csv',index=False)
    references=pd.read_parquet(JNU/'representation_predictions.parquet');first=references[references.representation=='report5'].sort_values('record_id')
    n=first.patient_component.nunique();rng=np.random.default_rng(20261020);draws=rng.multinomial(n,np.full(n,1/n),size=repetitions)
    comparisons=[];reference_samples={};reference_rows=[]
    for name,g in references.groupby('representation'):
        g=g.sort_values('record_id');assert g.record_id.tolist()==first.record_id.tolist()
        b=bootstrap_scores(g.figo_label,g.p,g.patient_id,g.patient_component,draws);reference_samples[name]=b
        point=binary_scores(g.figo_label,g.p,g.patient_id)
        for metric in ['macro_f1','ap','roc_auc']:
            lo,hi=b[metric].quantile([.025,.975]);key='group_auc' if metric=='roc_auc' else 'group_'+metric
            reference_rows.append(dict(representation=name,metric=metric,estimate=point[key],ci_low=lo,ci_high=hi))
    for name,b in reference_samples.items():
        if name=='report5':continue
        for metric in ['macro_f1','ap','roc_auc']:
            delta=b[metric]-reference_samples['report5'][metric];lo,hi=delta.quantile([.025,.975])
            comparisons.append(dict(comparison=f'{name} minus report5',metric=metric,mean=delta.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(reference_rows).to_csv(JNU/'representation_intervals.csv',index=False)
    pd.DataFrame(comparisons).to_csv(JNU/'representation_paired_differences.csv',index=False)
    # Hypothetical label-shift illustration, not a known clinical prevalence.
    prevalence=[]
    for (profile,family),r in frozen.iterrows():
        for pi in [.05,.15,.30,.54]:
            ppv=r.group_recall*pi/(r.group_recall*pi+(1-r.group_specificity)*(1-pi))
            prevalence.append(dict(profile=profile,family=family,assumed_FIGO_prevalence=pi,ppv_under_constant_sensitivity_specificity=ppv))
    pd.DataFrame(prevalence).to_csv(JNU/'hypothetical_label_shift_ppv.csv',index=False)
    jnu_calibration_support()
    return pd.DataFrame(rows)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='uci':uci_assessment()
    elif sys.argv[1]=='jnu':jnu_assessment()
