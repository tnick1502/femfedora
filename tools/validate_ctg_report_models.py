"""Independent arithmetic, boundaries, exports and scaler-fit audit for notebook 08."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.training.ctg_report_models import context,OUT,EXPORT,candidates
from src.features.ctg_report_features import PROFILES,uci_profile
from src.training.ctg_audit import context as audited_context,ProbabilityCalibrator
from src.training.ctg_audit_folds import balanced_group_folds
from src.training.ctg_experiments import digest,SEED
from tools.validate_ctg_audited import manual_scores


def validate():
    data,fp,_,_=context();card=json.loads((EXPORT/'model_card.json').read_text())
    assert card['fingerprint']==fp and not card['clinical_use']
    partitions={s:set(f.source_excel_row) for s,f in data.groupby('split')}
    layout=pd.read_csv(OUT/'outer_folds.csv')
    assert set(layout.source_excel_row)==partitions['train'] and layout.source_excel_row.is_unique
    old_layout=pd.read_csv(ROOT/'research/experiments/ctg_uci/audited_v2/outer_folds.csv')
    pd.testing.assert_frame_equal(layout.sort_values('source_excel_row').reset_index(drop=True),
        old_layout.sort_values('source_excel_row').reset_index(drop=True),check_dtype=False)
    joined=layout.merge(data[['source_excel_row','source_file','feature_duplicate_group']],on='source_excel_row',validate='one_to_one')
    for group in ['source_file','feature_duplicate_group']:assert joined.groupby(group).outer_fold.nunique().max()==1
    inner=pd.read_csv(OUT/'inner_folds.csv')
    for k,f in inner.groupby('outer_fold'):
        assert set(f.source_excel_row)==partitions['train']-set(layout.loc[layout.outer_fold==k,'source_excel_row'])
        j=f.merge(data[['source_excel_row','source_file','feature_duplicate_group']],on='source_excel_row',validate='one_to_one')
        for group in ['source_file','feature_duplicate_group']:assert j.groupby(group).inner_fold.nunique().max()==1
    grid=pd.read_csv(OUT/'nested_grid_results.csv');final_grid=pd.read_csv(OUT/'final_grid_results.csv')
    assert len(grid)==5*len(PROFILES)*len(candidates())==1380
    assert len(final_grid)==len(PROFILES)*len(candidates())==276
    nested=pd.read_parquet(OUT/'nested_predictions.parquet')
    for _,f in nested.groupby(['profile','mode']):
        assert set(f.source_excel_row)==partitions['train'] and f.source_excel_row.is_unique
    pred=pd.read_parquet(OUT/'final_predictions.parquet');metrics=pd.read_csv(OUT/'final_metrics.csv')
    verified=[]
    for (profile,split),f in pred.groupby(['profile','split']):
        assert set(f.source_excel_row)==partitions[split] and f.source_excel_row.is_unique
        p=f[['p0','p1','p2']].to_numpy();y=f.NSP.to_numpy()-1;prediction=f.pred.to_numpy()-1
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        np.testing.assert_array_equal(prediction,p.argmax(axis=1))
        row=metrics[(metrics.profile==profile)&(metrics.split==split)].iloc[0]
        for key,value in manual_scores(y,p,prediction).items():assert abs(row[key]-value)<1e-12,(profile,split,key)
        weights=1/f.source_file.map(f.source_file.value_counts()).to_numpy()
        weighted=manual_scores(y,p,prediction,weights)
        for key in ['macro_f1','recall_path','log_loss','ap_path']:assert abs(weighted[key]-row['file_'+key])<1e-12
        frame=data.set_index('source_excel_row').loc[f.source_excel_row]
        bundle=joblib.load(EXPORT/f'{profile}_best.joblib')
        np.testing.assert_allclose(bundle.predict_proba(uci_profile(frame,profile)),p,atol=1e-12)
        np.testing.assert_allclose(bundle.predict_proba(uci_profile(frame,profile).iloc[:1]),p[:1],atol=1e-12)
        verified.append(profile+'/'+split)
    train=data[data.split=='train'].reset_index(drop=True);valid=data[data.split=='validation'].reset_index(drop=True)
    outer=balanced_group_folds(train,5,SEED+21000)
    from catboost import CatBoostClassifier
    for spec in card['models']:
        profile=spec['profile'];path=EXPORT/spec['model'];assert digest(path)==spec['sha256']
        bundle=joblib.load(path);assert len(bundle.models)==spec['members']
        x=uci_profile(valid,profile);raw=np.mean([m.predict_proba(x) for m in bundle.models],axis=0)
        recal=ProbabilityCalibrator('temperature').fit(raw,valid.NSP.to_numpy()-1)
        assert abs(recal.temperature_-bundle.calibrator.temperature_)<1e-7
        sv,se=spec['validation_single']['macro_f1'],spec['validation_ensemble']['macro_f1']
        assert spec['ensemble_accepted']==(se>=sv+.01)
        for i,model in enumerate(bundle.models):
            fit=outer[i][0] if len(bundle.models)>1 else np.arange(len(train))
            expected=model.features_.transform(uci_profile(train.iloc[fit],profile))
            if model.candidate['family']=='logistic':
                scale=model.models_[0].named_steps['scale']
                np.testing.assert_allclose(scale.center_,np.median(expected,axis=0))
                expected_iqr=np.subtract(*np.percentile(expected,[75,25],axis=0));expected_iqr[expected_iqr<10*np.finfo(float).eps]=1
                np.testing.assert_allclose(scale.scale_,expected_iqr)
            else:
                for j,head in enumerate(model.models_):
                    native=CatBoostClassifier().load_model(str(EXPORT/f'{profile}_member{i}_head{j}.cbm'))
                    np.testing.assert_allclose(native.predict_proba(expected.iloc[:3]),head.predict_proba(expected.iloc[:3]),atol=1e-12)
    old=json.loads((ROOT/'research/experiments/ctg_uci/audited_v2/validation_receipt.json').read_text())
    assert audited_context()[1]==old['fingerprint']
    for relative,sha in old['artifact_sha256'].items():assert digest(ROOT/relative)==sha,relative
    for spec in old['notebooks']:assert digest(ROOT/'research/notebooks/ctg_uci'/spec['name'])==spec['sha256']
    from src.training.ctg_embeddings import legacy_snapshot
    preserved=json.loads((ROOT/'research/experiments/ctg_uci/embeddings/preserved_legacy.json').read_text())['sha256']
    assert legacy_snapshot()==preserved
    receipt=dict(fingerprint=fp,passed=True,nested_candidates=1380,inner_evaluations=4140,
        final_candidates=276,final_inner_evaluations=828,verified_predictions=verified,
        grouped_boundaries=True,same_outer_folds_as_v2=True,manual_arithmetic=True,
        inference_roundtrip=True,batch_invariant=True,native_roundtrip=True,scaler_fit_verified=True,
        calibration_validation_only_recomputed=True,previous_results_unchanged=True,clinical_transfer_validated=False)
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=False,indent=2))
