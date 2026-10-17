"""Reproducible train-only checks and report-profile preprocessing exports."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform
import re

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.features.jnu_ctg_preparation import preprocessor_template
from src.features.jnu_ctg_report_extraction import (
    ROOT,SOURCE,DEST,OUT,VERSION,ExtractionConfig,PROFILE_FIELDS,
    extract_report_features,profile_frame,sha256)
from src.features.ctg_russian_reports import EXAMPLE_TEXT,parse_report_text,engineer_report

MODELS=ROOT/'models/exported/jnu_ctg/report_features_v1'
SEED=20261017


def export_preprocessing(measurements):
    MODELS.mkdir(parents=True,exist_ok=True)
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    train_ids=set(splits.loc[splits.split=='train','record_id'])
    fit_mask=measurements.record_id.isin(train_ids)&measurements.report_quality_eligible
    rows=[]
    for name,features in PROFILE_FIELDS.items():
        x=profile_frame(measurements,name)
        template=preprocessor_template(features)
        joblib.dump(template,MODELS/f'{name}_template_unfitted.joblib')
        fitted=template.fit(x.loc[fit_mask])
        joblib.dump(fitted,MODELS/f'{name}_robust_train.joblib')
        for part in ['train','validation','test']:
            ids=splits.loc[splits.split==part,['record_id']]
            joined=ids.merge(pd.concat([measurements[['record_id']],x],axis=1),on='record_id',validate='one_to_one')
            matrix=fitted.transform(joined[features])
            assert matrix.shape==(len(joined),2*len(features)) and np.isfinite(matrix).all()
            np.save(DEST/f'X_{name}_robust_{part}.npy',matrix.astype(np.float32))
        rows.append(dict(profile=name,report_inputs=len(features),with_missing_masks=2*len(features),
            fit_train_records=int(fit_mask.sum()),constant_train_fields=','.join(x.loc[fit_mask].columns[x.loc[fit_mask].nunique()<=1])))
    pd.DataFrame(rows).to_csv(OUT/'profiles.csv',index=False)
    pd.DataFrame({'record_id':measurements.loc[fit_mask,'record_id']}).to_parquet(DEST/'preprocessor_fit_ids.parquet',index=False)
    return pd.DataFrame(rows)


def train_sensitivity(measurements, sample_size=128):
    """Independent-patient sample, selected without targets or feature values."""
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    index=pd.read_parquet(SOURCE/'signal_index.parquet')
    signals=np.load(SOURCE/'signals_original_float32.npy',mmap_mode='r')
    chosen=splits.loc[splits.split=='train'].sort_values('record_id').drop_duplicates('patient_component')
    chosen=chosen.sample(n=min(sample_size,len(chosen)),random_state=SEED).merge(index,on='record_id',validate='one_to_one')
    chosen[['record_id','patient_id','patient_component','signal_row']].to_parquet(OUT/'sensitivity_train_sample.parquet',index=False)
    configs={f'{method}_{window}s':replace(ExtractionConfig(),baseline_method=method,baseline_window_s=window)
        for method in ['median','median_trim'] for window in [180,300,600]}
    configs.update({'no_fill':replace(ExtractionConfig(),max_internal_gap_s=0),
        'fill_1s':replace(ExtractionConfig(),max_internal_gap_s=1),
        'return_1bpm':replace(ExtractionConfig(),return_band_bpm=1),
        'return_3bpm':replace(ExtractionConfig(),return_band_bpm=3),
        'smooth_1s':replace(ExtractionConfig(),event_smoothing_s=1)})
    rows=[];durations=[]
    for row in chosen.itertuples():
        wave=signals[row.signal_row]
        for name,cfg in configs.items():
            f=extract_report_features(wave,cfg)
            rows.append(dict(record_id=row.record_id,recipe=name,**{k:f[k] for k in
                ['baseline_bpm','acc15_count','dec15_count','stv_ms','amplitude_bpm','eligible_stv_min']}))
        # The record-level FIGO label is intentionally not propagated to prefixes.
        for duration in [10,17,20,30]:
            f=extract_report_features(wave[:duration*240])
            durations.append(dict(record_id=row.record_id,prefix_min=duration,**{k:f[k] for k in
                ['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm','report_quality_eligible']}))
    sens=pd.DataFrame(rows);duration=pd.DataFrame(durations)
    sens.to_parquet(OUT/'sensitivity_train.parquet',index=False)
    duration.to_parquet(OUT/'duration_prefix_train.parquet',index=False)
    reference=sens.query("recipe=='median_trim_300s'").set_index('record_id')
    summaries=[]
    for name,part in sens.groupby('recipe'):
        delta=part.set_index('record_id')[reference.select_dtypes('number').columns]-reference.select_dtypes('number')
        for col in delta:
            finite=delta[col].dropna()
            summaries.append(dict(recipe=name,feature=col,paired_n=len(finite),
                median_absolute_change=float(finite.abs().median()),p95_absolute_change=float(finite.abs().quantile(.95)),
                changed_fraction=float((finite.abs()>1e-8).mean())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'sensitivity_summary.csv',index=False)
    (OUT/'sensitivity_protocol.json').write_text(json.dumps(dict(seed=SEED,train_only=True,
        no_label_based_selection=True,independent_components=len(chosen),
        configs={k:asdict(v) for k,v in configs.items()},prefix_labels_assigned=False),indent=2))
    return sens,duration,summary


def compare_published_train(measurements):
    published=pd.read_parquet(SOURCE/'published_features_quarantined.parquet')
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    train=measurements.merge(splits.loc[splits.split=='train',['record_id']],on='record_id',validate='one_to_one')
    both=train.merge(published,on='record_id',validate='one_to_one')
    mapping={'baseline_bpm':'F_BL','acc15_count':'F_A_N','dec15_count':'F_D_N','stv_ms':'F_STV','amplitude_bpm':'F_LTV'}
    rows=[]
    for own,pub in mapping.items():
        if pub not in both:continue
        z=both[[own,pub]].apply(pd.to_numeric,errors='coerce').dropna()
        # The release does not give a complete unit/recipe contract. Numerical
        # subtraction would misleadingly resemble a measurement error in ms/bpm.
        rows.append(dict(ours=own,published=pub,n=len(z),ours_median=z[own].median(),
            published_median=z[pub].median(),
            spearman=z.corr(method='spearman').iloc[0,1],unit_equivalence_confirmed=False,
            measurement_error_estimated=False))
    result=pd.DataFrame(rows);result.to_csv(OUT/'published_comparison_train.csv',index=False)
    return result


def example_report_outputs():
    parsed=parse_report_text(EXAMPLE_TEXT)
    x,audit=engineer_report({**parsed['measurements'],'analysis_context':'antenatal',
        'device':'unknown','confirmed':False,'source':'manual transcription of user example'})
    # JSON unknowns are null, not nonstandard NaN; score fields never enter X.
    def serial(value):
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if isinstance(value,list):return [serial(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    (OUT/'example_report_parse.json').write_text(json.dumps(serial(parsed),ensure_ascii=False,indent=2,allow_nan=False))
    (OUT/'example_report_audit.json').write_text(json.dumps(serial(audit),ensure_ascii=False,indent=2,allow_nan=False))
    x.to_parquet(DEST/'example_report_engineered.parquet',index=False)
    template={key:None for key in audit['values']}
    template.update(device=None,software_version=None,analysis_context='antenatal',gestational_weeks=None,
        confirmed=False,definitions={},analysis_window='whole_recording',event_exposure=None)
    (DEST/'report_input_template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2))
    return parsed,x,audit


def write_contracts():
    from src.features.ctg_russian_reports import FIELDS,META
    requirements={profile:{'measurement_contract':VERSION} for profile in PROFILE_FIELDS}
    for profile in ['variability','sonomed_fields','mfmon_fields']:
        requirements[profile]['stv_epoch_s']=3.75
    requirements['mak_fields']['stv_epoch_s']=4
    requirements['sonomed_fields'].update(variation_episode_rule='3consecutive_11bpm',acc10_category='inclusive_ge10')
    requirements['mfmon_fields']['variation_episode_rule']='5consecutive_32ms'
    contract=dict(version=VERSION,mode='report_numbers_only',clinical_validation=False,
        native_measurements=FIELDS,metadata_fields=sorted(META),profiles=PROFILE_FIELDS,
        definition_requirements=requirements,unknown_is_zero=False,
        report_missing_masks='all columns',vendor_names_prove_equivalence=False,
        classifier_trained=False,clinical_risk_output=None)
    (DEST/'report_profile_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
    document=ROOT/'research/notebooks/jnu_ctg/REPORT_FEATURE_METHODS.md'
    sources=[]
    for line in document.read_text().splitlines():
        if re.match(r'\| (S\d|P\d|G\d|D\d)',line):
            parts=line.split('|')
            sources.append(dict(id=parts[1].strip(),description=parts[2].strip(),
                urls=re.findall(r'\]\((https?://[^)]+)\)',parts[2]),critical_decision=parts[3].strip()))
    assert len(sources)==18
    (OUT/'evidence_sources.json').write_text(json.dumps(dict(review_date='2026-10-17',
        exhaustive_systematic_review=False,methods_document_sha256=sha256(document),sources=sources),ensure_ascii=False,indent=2))
    return contract


def validate_exports(measurements):
    """Independent reconstruction of preprocessing moments and identities."""
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    fit_ids=set(pd.read_parquet(DEST/'preprocessor_fit_ids.parquet').record_id)
    assert fit_ids <= set(splits.loc[splits.split=='train','record_id'])
    assert not fit_ids & set(splits.loc[splits.split!='train','record_id'])
    checks=[]
    for profile,cols in PROFILE_FIELDS.items():
        x=profile_frame(measurements,profile)
        fit=x.loc[measurements.record_id.isin(fit_ids)]
        saved=joblib.load(MODELS/f'{profile}_robust_train.joblib')
        # Use sklearn internals only to inspect fitted values, never reuse them
        # in the independent expected median/IQR computation.
        ct=saved.named_steps['columns']
        numeric=ct.named_transformers_['scaled']
        median=fit.median().to_numpy()
        np.testing.assert_allclose(numeric.named_steps['imputer'].statistics_,median)
        complete=fit.fillna(fit.median()).to_numpy()
        expected_center=np.median(complete,axis=0)
        q25,q75=np.percentile(complete,[25,75],axis=0)
        np.testing.assert_allclose(numeric.named_steps['scaler'].center_,expected_center)
        expected_scale=q75-q25;expected_scale[expected_scale<10*np.finfo(float).eps]=1.
        np.testing.assert_allclose(numeric.named_steps['scaler'].scale_,expected_scale)
        template=joblib.load(MODELS/f'{profile}_template_unfitted.joblib')
        assert not hasattr(template.named_steps['columns'],'transformers_')
        probe=x.iloc[:3].copy();probe.iloc[0,:]=np.nan
        transformed=saved.transform(probe)
        assert (transformed[0,len(cols):]==1).all() and np.isfinite(transformed).all()
        for split in ['train','validation','test']:
            exported=pd.read_parquet(DEST/f'X_{profile}_{split}.parquet')
            expected_ids=splits.loc[splits.split==split,'record_id'].tolist()
            assert exported.record_id.tolist()==expected_ids
            expected=measurements[['record_id']].join(x).set_index('record_id').loc[expected_ids]
            pd.testing.assert_frame_equal(exported[cols].reset_index(drop=True),expected.reset_index(drop=True))
            np.testing.assert_allclose(np.load(DEST/f'X_{profile}_robust_{split}.npy'),
                saved.transform(exported[cols]),rtol=1e-6,atol=1e-6)
        checks.append(dict(profile=profile,train_only=True,median_verified=True,iqr_verified=True,
            all_column_missing_masks=True,template_unfitted=True))
    assert measurements.record_id.is_unique and len(measurements)==len(splits)
    groups={part:set(splits.loc[splits.split==part,'patient_component']) for part in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    assert measurements.loc[measurements.signal_loss_pct==100,'baseline_bpm'].isna().all()
    for col in ['movements_per_hour','vstv_ms','sinusoidal_min','psp','fisher_score','krebs_score','dr_criteria_met']:
        assert measurements[col].isna().all()
    receipt=dict(version=VERSION,records=len(measurements),profiles=checks,fit_records=len(fit_ids),
        clinical_validation=False,models_trained=False,test_used_for_selection=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        artifacts_sha256={str(p.relative_to(ROOT)):sha256(p) for directory in [DEST,MODELS]
            for p in sorted(directory.glob('*')) if p.is_file()})
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt
