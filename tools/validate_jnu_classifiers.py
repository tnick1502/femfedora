"""Independent numerical/boundary/export checks of the completed JNU run."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features.jnu_ctg_report_extraction import sha256,PROFILE_FIELDS
from src.training.jnu_report_experiment import OUT,EXPORT,metadata,inputs
from src.features.ctg_russian_reports import EXAMPLE_TEXT,parse_report_text


def assert_hashes(receipt,key):
    for relative,expected in receipt[key].items():assert sha256(ROOT/relative)==expected,relative


def validate():
    protocol=json.loads((OUT/'protocol.json').read_text())
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    evaluation=json.loads((OUT/'test_evaluation_receipt.json').read_text())
    assert_hashes(protocol,'input_sha256');assert_hashes(frozen,'export_sha256')
    assert evaluation['frozen_sha256']==sha256(OUT/'selection_frozen.json')
    assert evaluation['predictions_sha256']==sha256(OUT/'test_predictions.parquet')
    assert evaluation['metrics_sha256']==sha256(OUT/'test_metrics.csv')
    train,valid,test=[metadata(p) for p in ['train','validation','test']]
    groups={p:set(d.patient_component) for p,d in zip(['train','validation','test'],[train,valid,test])}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    folds=pd.read_parquet(OUT/'fold_assignments.parquet')
    outer=folds.query("stage=='outer'").merge(train[['record_id','patient_component']],on='record_id',validate='one_to_one')
    assert len(outer)==len(train) and outer.groupby('patient_component').fold.nunique().max()==1
    for k in range(5):
        hold=set(outer.loc[outer.fold==k,'record_id'])
        inner=folds[(folds.stage=='inner')&(folds.outer_fold==k)].merge(train[['record_id','patient_component']],on='record_id',validate='one_to_one')
        assert not hold&set(inner.record_id) and len(inner)+len(hold)==len(train)
        assert inner.groupby('patient_component').fold.nunique().max()==1
    roles=pd.read_parquet(OUT/'validation_roles.parquet')
    cal=roles[roles.role=='calibration'];op=roles[roles.role=='operating_point']
    assert not set(cal.patient_component)&set(op.patient_component)
    assert set(cal.record_id)|set(op.record_id)==set(valid.record_id)
    nested=pd.read_parquet(OUT/'nested_predictions.parquet')
    predictions=pd.read_parquet(OUT/'test_predictions.parquet')
    vpred=pd.read_parquet(OUT/'validation_predictions.parquet')
    test_metrics=pd.read_csv(OUT/'test_metrics.csv')
    checks=[]
    for item in frozen['choices']:
        profile,family=item['profile'],item['family']
        bundle=joblib.load(EXPORT/f'{profile}_{family}.joblib')
        xtrain=inputs(train,profile).set_axis(train.record_id)
        counts=pd.Series(0,index=train.record_id)
        for i,model in enumerate(bundle.models):
            ids=model.fit_record_ids_.tolist();fit=train.set_index('record_id').loc[ids]
            assert set(ids)<=set(train.record_id)
            assert not set(model.fit_components_)&(groups['validation']|groups['test'])
            assert set(model.fit_components_)==set(fit.patient_component)
            assert len(ids)==len(set(ids))
            counts.loc[ids]+=1
            xx=xtrain.loc[ids]
            np.testing.assert_allclose(model.training_medians_,xx.median().fillna(0))
            if family!='catboost':
                b=model.basis_;array=xx.to_numpy()
                median=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in array.T])
                np.testing.assert_allclose(b.medians_,median)
                complete=np.where(np.isnan(array),median,array)
                np.testing.assert_allclose(b.centers_,np.median(complete,axis=0))
                q=np.subtract(*np.percentile(complete,[75,25],axis=0));q[q<10*np.finfo(float).eps]=1
                np.testing.assert_allclose(b.scales_,q)
                saved=joblib.load(EXPORT/f'{profile}_{family}_{i}_preprocessor.joblib')
                np.testing.assert_array_equal(b.transform(xx.iloc[:5]),saved.transform(xx.iloc[:5]))
            else:
                native=CatBoostClassifier();native.load_model(EXPORT/f'{profile}_catboost_{i}.cbm')
                sample=inputs(test,profile).iloc[:31]
                np.testing.assert_array_equal(native.predict_proba(model._tree_features(sample))[:,1],model.predict_positive(sample))
        assert (counts==(4 if item['mode']=='ensemble5' else 1)).all()
        assert len(bundle.models)==(5 if item['mode']=='ensemble5' else 1)
        assert set(bundle.calibrator.fit_patients_)==set(cal.patient_id)
        xt=inputs(test,profile)
        stored=predictions[(predictions.profile==profile)&(predictions.family==family)].set_index('record_id').loc[test.record_id]
        p=bundle.predict_proba(xt)[:,1]
        np.testing.assert_array_equal(p,stored.p.to_numpy())
        np.testing.assert_allclose(bundle.predict_proba(xt).sum(axis=1),1)
        # BLAS may round the same dot product differently for 7 vs 4140 rows.
        np.testing.assert_allclose(p[:7],bundle.predict_proba(xt.iloc[:7,::-1])[:,1],rtol=0,atol=1e-12)
        try:bundle.predict_proba(xt.assign(figo_label=0))
        except ValueError:pass
        else:raise AssertionError('Target accepted as a feature')
        # Independent pandas patient weights and manual confusion arithmetic.
        sizes=test.groupby('patient_id').record_id.transform('size').to_numpy()
        w=1./sizes;y=test.figo_label.to_numpy(int)
        metric=test_metrics[(test_metrics.profile==profile)&(test_metrics.family==family)&(test_metrics.policy=='primary_0.5')].iloc[0]
        tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1],sample_weight=w).ravel()
        f1=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
        np.testing.assert_allclose([metric.patient_macro_f1,metric.patient_recall,metric.patient_specificity],[f1,tp/(tp+fn),tn/(tn+fp)])
        assert metric.record_tn+metric.record_fp+metric.record_fn+metric.record_tp==len(test)
        np.testing.assert_allclose(metric.patient_ap,average_precision_score(y,p,sample_weight=w),rtol=1e-12)
        np.testing.assert_allclose(metric.patient_roc_auc,roc_auc_score(y,p,sample_weight=w),rtol=1e-12)
        vv=vpred[(vpred.profile==profile)&(vpred.family==family)].set_index('record_id').loc[valid.record_id]
        np.testing.assert_array_equal(vv.p_raw.to_numpy(),bundle.raw_probability(inputs(valid,profile)))
        vo=vv[vv.role=='operating_point'].copy()
        ow=1./vo.groupby('patient_id').figo_label.transform('size').to_numpy()
        thresholds=pd.DataFrame({'p':vo.p,'positive':ow*vo.figo_label,'negative':ow*(1-vo.figo_label)}).groupby('p').sum().sort_index(ascending=False).cumsum()
        ttp,tfp=thresholds.positive,thresholds.negative
        tfn,ttn=thresholds.positive.iloc[-1]-ttp,thresholds.negative.iloc[-1]-tfp
        values=.5*(2*ttp/(2*ttp+tfp+tfn)+2*ttn/(2*ttn+tfp+tfn))
        assert values.loc[item['research_threshold']]>=values.max()-1e-12
        oo=nested[(nested.profile==profile)&(nested.family==family)]
        assert len(oo)==len(train) and oo.record_id.is_unique and set(oo.record_id)==set(train.record_id)
        audit=bundle.audit_report(parse_report_text(EXAMPLE_TEXT)['measurements'])
        assert audit['clinical_risk'] is None and not audit['supported_for_clinical_inference']
        checks.append(dict(profile=profile,family=family,mode=item['mode'],all_train_rows_retained=True,
            fitted_preprocessing_independently_checked=True,group_boundaries=True,export_roundtrip=True,
            calibration_only_calibration_patients=True,threshold_only_operating_patients=True,
            no_raw_signal_in_X=True,clinical_validation=False))
        card=dict(profile=profile,family=family,champion=item['champion'],parameters=item['params'],aggregation=item['mode'],
            inputs=PROFILE_FIELDS[profile],feature_contract='JNU_report_operational_1.0',duration_min_supported_in_research=30,
            target={'0':'normal expert FIGO','1':'abnormal expert FIGO'},
            primary_threshold=.5,research_macro_threshold=item['research_threshold'],
            training_patient_weights='1/current cohort record count',test_used_for_selection=False,
            calibration={'slope':bundle.calibrator.slope_,'intercept':bundle.calibrator.intercept_},
            clinically_validated=False,external_russian_report_validation=False,
            limitations=['single Chinese centre','in-house signal extractor not a verified vendor bridge',
                'no separate suspect/pathologic classes','no shorter-window validation','not neonatal asphyxia probability'])
        (EXPORT/f'{profile}_{family}_model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
    for stage in ['preparation_v1','report_features_v1']:
        old=json.loads((ROOT/f'research/experiments/jnu_ctg/{stage}/validation_receipt.json').read_text())
        if 'artifacts_sha256' in old:assert_hashes(old,'artifacts_sha256')
    original_manifest=json.loads((ROOT/'research/experiments/jnu_ctg/preparation_v1/manifest.json').read_text())
    assert_hashes(original_manifest,'artifacts_sha256')
    assert original_manifest['code_sha256']==sha256(ROOT/'src/features/jnu_ctg_preparation.py')
    for stage,filename in [('preparation_v1','01_prepare_data.ipynb'),('report_features_v1','02_report_features.ipynb')]:
        old=json.loads((ROOT/f'research/experiments/jnu_ctg/{stage}/presentation_receipt.json').read_text())
        relative=f'research/notebooks/jnu_ctg/{filename}'
        assert sha256(ROOT/relative)==old['sha256'][relative]
    uci=json.loads((ROOT/'research/experiments/ctg_uci/audited_v2/validation_receipt.json').read_text())
    assert_hashes(uci,'artifact_sha256')
    registry={'feature_contract':'JNU_report_operational_1.0','target':'binary expert FIGO normal/abnormal',
        'selection':'mean outer nested-CV train Macro-F1; frozen before test',
        'clinical_validation':False,'primary_threshold':.5,'profiles':{}}
    for profile,family in frozen['champions'].items():
        metric=test_metrics[(test_metrics.profile==profile)&(test_metrics.family==family)&(test_metrics.policy=='primary_0.5')].iloc[0]
        filename=f'{profile}_{family}.joblib'
        registry['profiles'][profile]={'family':family,'bundle':filename,'sha256':sha256(EXPORT/filename),
            'model_card':f'{profile}_{family}_model_card.json','columns':PROFILE_FIELDS[profile],
            'test_patient_weighted_record_metrics':{k:float(metric[f'patient_{k}']) for k in ['macro_f1','recall','specificity','precision','ap','roc_auc']},
            'routing_requires_confirmed_field_definitions':True,'missing_profile_fields_not_validated_for_application':True}
    (EXPORT/'research_champions.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
    result=dict(passed=True,model_bundles=len(checks),records_train=len(train),records_test=len(test),
        checks=checks,all_nested_oof_rows_once=True,finite_probabilities=True,
        manual_test_arithmetic=True,exact_native_model_roundtrip=True,
        previous_preparation_and_UCI_artifacts_preserved=True,clinical_validation=False)
    write=result.copy();write['artifacts_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in EXPORT.glob('*') if p.is_file()}
    (OUT/'validation_receipt.json').write_text(json.dumps(write,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':validate()
