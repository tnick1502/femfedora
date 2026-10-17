"""Independent boundaries, mask arithmetic and fitted-parameter checks for JNU 01."""
from pathlib import Path
import json
import sys
import zipfile,io
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.features.jnu_ctg_preparation import PROCESSED,OUT,MODELS,digest


def validate():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['code_sha256']==digest(ROOT/'src/features/jnu_ctg_preparation.py')
    for relative,sha in manifest['artifacts_sha256'].items():assert digest(ROOT/relative)==sha,relative
    original=pd.read_parquet(PROCESSED/'metadata_original.parquet')
    clean=pd.read_parquet(PROCESSED/'metadata_clean.parquet')
    primary=pd.read_parquet(PROCESSED/'jnu_primary.parquet');index=pd.read_parquet(PROCESSED/'signal_index.parquet')
    splits=pd.read_parquet(PROCESSED/'splits.parquet');quality=pd.read_parquet(PROCESSED/'signal_quality.parquet')
    assert len(original)==len(clean)==len(primary)==len(index)==len(splits)==20769
    assert primary.record_id.is_unique and primary.patient_id.nunique()==12606
    assert set(primary.record_id)==set(original.record_id)
    assert set(splits.split)=={'train','validation','test'}
    for key in ['patient_id','patient_component']:assert splits.groupby(key).split.nunique().max()==1
    for key in ['fhr_sha256','waveform_sha256']:
        j=splits.merge(index[['record_id',key]],on='record_id',validate='one_to_one')
        assert j.groupby(key).split.nunique().max()==1
    np.testing.assert_allclose(splits.groupby('patient_id').inverse_patient_record_count.sum(),1)
    assert original.neonatal_asphyxia.isna().sum()==clean.neonatal_asphyxia.isna().sum()==58
    np.testing.assert_array_equal(original.neonatal_asphyxia.fillna(-1),clean.neonatal_asphyxia.fillna(-1))
    assert set(clean.figo_label.dropna().unique())=={0,1}
    for column in [c for c in clean if c.startswith('apgar_')]:
        maximum=10 if column in ['apgar_1min','apgar_5min','apgar_10min'] else 2
        x=clean[column].dropna();assert x.between(0,maximum).all() and np.isclose(x,x.round()).all()
    contract=json.loads((PROCESSED/'feature_contract.json').read_text());columns=contract['primary_features']
    assert len(columns)==27 and not set(columns)&set(original.columns)
    assert not any(c.startswith(('apgar_','NetFea','birth_','delivery_')) for c in columns)
    parameters=json.loads((MODELS/'tabular_parameters.json').read_text())
    train=primary[(primary.split=='train')&primary.primary_training_eligible]
    assert len(train)==parameters['fit_rows']
    x=train[columns].to_numpy(dtype=float);medians=np.nanmedian(x,axis=0)
    medians=np.where(np.isfinite(medians),medians,0)
    np.testing.assert_allclose(parameters['median_imputation'],medians)
    filled=np.where(np.isnan(x),medians,x)
    np.testing.assert_allclose(parameters['robust_center'],np.median(filled,axis=0))
    iqr=np.subtract(*np.percentile(filled,[75,25],axis=0));iqr[iqr<10*np.finfo(float).eps]=1
    np.testing.assert_allclose(parameters['robust_scale'],iqr)
    bundle=joblib.load(MODELS/'tabular_preprocessor_train.joblib')
    template=joblib.load(MODELS/'preprocessor_template_unfitted.joblib')
    assert not hasattr(template.named_steps['columns'],'transformers_')
    cloned=clone(template);assert not hasattr(cloned.named_steps['columns'],'transformers_')
    for split in ['train','validation','test']:
        raw=pd.read_parquet(PROCESSED/f'X_raw_{split}.parquet')
        target=pd.read_parquet(PROCESSED/f'index_targets_{split}.parquet')
        assert raw.record_id.equals(target.record_id)
        stored=np.load(PROCESSED/f'X_robust_{split}.npy')
        actual=bundle.transform(raw[columns]).astype(np.float32)
        np.testing.assert_array_equal(actual,stored)
        assert np.isfinite(stored).all() and stored.shape[1]==54
        np.testing.assert_array_equal(stored[:,27:],raw[columns].isna().to_numpy(dtype=float))
    signal=np.load(PROCESSED/'signals_original_float32.npy',mmap_mode='r')
    assert signal.shape==(20769,7200,2) and signal.dtype==np.float32
    normalizer=joblib.load(MODELS/'signal_normalizer_train.joblib')
    # Compute the population moments with different chunk boundaries and explicit
    # formulas rather than calling the fitting implementation again.
    total=np.zeros(2);sum_sq=np.zeros(2);n=np.zeros(2,dtype=np.int64)
    positions=train.signal_row.to_numpy()
    for start in range(0,len(positions),113):
        a=np.asarray(signal[positions[start:start+113]],dtype=np.float64)
        for channel in range(2):
            v=a[:,:,channel];ok=np.isfinite(v)
            if channel==0:ok&=(v>=30)&(v<=240)
            values=v[ok];n[channel]+=len(values);total[channel]+=values.sum();sum_sq[channel]+=np.dot(values,values)
    means=total/n;scales=np.sqrt(np.maximum(sum_sq/n-means**2,0));scales[scales<1e-12]=1
    np.testing.assert_array_equal(n,normalizer.count_)
    np.testing.assert_allclose(means,normalizer.mean_,atol=1e-10)
    np.testing.assert_allclose(scales,normalizer.scale_,atol=1e-9)
    rng=np.random.default_rng(20261017)
    rows=np.unique(np.r_[rng.choice(len(primary),30,replace=False),primary.loc[~primary.has_valid_fhr,'signal_row']])
    rows=rows.astype(int)
    with np.load(PROCESSED/'signal_masks_packed.npz') as masks:
        for name in masks.files:
            assert masks[name].shape==(20769,900)
        observed=np.unpackbits(masks['fhr_observed'][rows],axis=1,bitorder='little')[:,:7200].astype(bool)
        raw=signal[rows,:,0]
        np.testing.assert_array_equal(observed,np.isfinite(raw)&(raw>=30)&(raw<=240))
        src=np.unpackbits(masks['fhr_source_present'][rows],axis=1,bitorder='little')[:,:7200].astype(bool)
        np.testing.assert_array_equal(src,np.isfinite(raw)&(raw>0))
    for row in rows:
        source=signal[row];transformed=normalizer.transform_one(source)
        assert np.isfinite(transformed['values']).all()
        assert not transformed['interpolated_mask'][0,0] and not transformed['interpolated_mask'][-1,0]
        np.testing.assert_array_equal(transformed['values'][~transformed['used_mask']],0)
        assert transformed['observed_mask'][:,1].all() # verified finite UC throughout this source
    q=primary[['record_id','has_valid_fhr']].merge(quality,on='record_id',validate='one_to_one')
    np.testing.assert_array_equal(q.has_valid_fhr,q.fhr_observed_fraction>0)
    # Independently check ID -> source CSV -> stored row with another parser.
    check_index=index.iloc[rows]
    with zipfile.ZipFile(ROOT/'data/external/jnu_ctg/1.0.0/JNU-CTG.zip') as archive:
        for item in check_index.itertuples(index=False):
            csv_values=pd.read_csv(io.BytesIO(archive.read(item.source_member))).to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(csv_values,signal[item.signal_row])
    balance=pd.read_csv(OUT/'split_balance.csv')
    assert balance.asphyxia_positive_patients.sum()==91
    assert (balance.asphyxia_positive_patients>=15).all()
    assert balance.abnormal_record_fraction.max()-balance.abnormal_record_fraction.min()<.01
    receipt=dict(passed=True,rows=20769,patients=12606,features=27,transformed_columns=54,
        metadata_source_preserved=True,nullable_outcome_preserved=True,apgar_domains_checked=True,
        patient_and_exact_duplicate_boundaries=True,source_hashes_checked=True,
        scaler_and_imputer_train_only_independently_recomputed=True,
        all_column_missing_masks_verified=True,signal_moments_train_only_recomputed=True,
        packed_signal_masks_checked=True,source_csv_to_array_link_sample_checked=True,roundtrip=True,models_trained=False,
        clinical_validation=False,test_used_for_model_selection=False)
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=False,indent=2))
