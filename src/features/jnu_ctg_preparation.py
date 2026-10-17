"""Reproducible JNU-CTG preparation, isolated from the historical UCI series.

Signal descriptors here are project-defined measurements, never substitutes for
device STV/LTV. Raw waveforms and clinical source values remain recoverable.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/external/jnu_ctg/1.0.0'
PROCESSED=ROOT/'data/processed/jnu_ctg/v1'
OUT=ROOT/'research/experiments/jnu_ctg/preparation_v1'
MODELS=ROOT/'models/exported/jnu_ctg/preparation_v1'
ARCHIVE=SOURCE/'JNU-CTG.zip'
EXPECTED_MD5='ac1cfcba2f1b3336596544211d776771'
FS=4
N_SAMPLES=7200
SEED=20261017
SHORT_GAP_S=2.
FHR_ENGINEERING_RANGE=(30.,240.)
SPLITS={'train':.6,'validation':.2,'test':.2}
VERSION='jnu_preparation_v1_20261017'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.generic):x=x.item()
        if isinstance(x,np.ndarray):return convert(x.tolist())
        if isinstance(x,dict):return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [convert(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,Path):return str(x)
        return x
    path.write_text(json.dumps(convert(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False))


def directories():
    for p in [PROCESSED,OUT,MODELS]:p.mkdir(parents=True,exist_ok=True)


def verify_source():
    directories()
    if not ARCHIVE.exists():raise FileNotFoundError(f'Download official archive to {ARCHIVE}')
    md5=hashlib.md5(ARCHIVE.read_bytes()).hexdigest()
    if md5!=EXPECTED_MD5:raise ValueError('Archive MD5 differs from published Zenodo checksum')
    with zipfile.ZipFile(ARCHIVE) as z:
        text=z.read('JNU-CTG/metadata.csv').decode('utf-8-sig')
        original_columns=next(csv.reader(io.StringIO(text)))
        frame=pd.read_csv(io.StringIO(text),low_memory=False)
        names={Path(n).stem:n for n in z.namelist() if '/waveforms/csv_raw_7200/' in n and n.endswith('.csv')}
    assert len(names)==len(frame)==20769
    assert frame.record_id.is_unique and frame.patient_id.notna().all()
    assert set(names)==set(frame.record_id) and frame.patient_id.nunique()==12606
    frame.to_parquet(PROCESSED/'metadata_original.parquet',index=False)
    registry=pd.DataFrame({'position':range(len(original_columns)),'source_name':original_columns,'canonical_name':frame.columns})
    registry['duplicate_source_header']=registry.source_name.duplicated(keep=False)
    registry.to_csv(OUT/'source_column_registry.csv',index=False)
    receipt=dict(version=VERSION,source_url='https://zenodo.org/records/21800730',
        doi='10.5281/zenodo.21800730',archive_sha256=digest(ARCHIVE),archive_md5=md5,
        rows=len(frame),columns=len(frame.columns),patients=frame.patient_id.nunique(),
        duplicate_headers={k:v for k,v in Counter(original_columns).items() if v>1},
        source_definitions='README and LICENSE inside original archive; no downloaded code executed')
    write_json(OUT/'source_receipt.json',receipt)
    return frame,names,receipt


def runs(mask):
    mask=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def bounded_interpolation(x,observed,max_gap_samples=8):
    """Only internal short gaps; no edge extrapolation or long-gap bridging."""
    out=np.where(observed,np.asarray(x,dtype=float),np.nan)
    filled=np.zeros(len(x),dtype=bool)
    for start,end in runs(~observed):
        if start>0 and end<len(x) and end-start<=max_gap_samples:
            out[start:end]=np.linspace(out[start-1],out[end],end-start+2)[1:-1]
            filled[start:end]=True
    return out,filled


def signal_masks(wave):
    fhr,uc=np.asarray(wave,dtype=float).T
    present=np.isfinite(fhr)&(fhr>0)
    range_flag=present&((fhr<FHR_ENGINEERING_RANGE[0])|(fhr>FHR_ENGINEERING_RANGE[1]))
    observed=present&~range_flag
    cleaned,interpolated=bounded_interpolation(fhr,observed,round(FS*SHORT_GAP_S))
    # Broad, provisional engineering envelope, not clinical normality limits.
    # Original samples and separate reasons are retained. Source-present and
    # analysis-valid masks are distinct; this rule needs device validation.
    jumps=np.zeros(len(fhr),bool)
    jumps[1:]=observed[1:]&observed[:-1]&(np.abs(np.diff(fhr))>25)
    return dict(fhr_source_present=present,fhr_observed=observed,fhr_interpolated=interpolated,
        fhr_range_flag=range_flag,fhr_jump_flag=jumps,
        uc_observed=np.isfinite(uc),uc_zero=np.isfinite(uc)&(uc==0)),cleaned


def basic_stats(values,prefix):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    names=['mean','median','std','min','max','q05','q25','q75','q95','iqr','mad']
    if not len(x):return {prefix+n:np.nan for n in names}
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    values=[x.mean(),q[2],x.std(ddof=0),x.min(),x.max(),q[0],q[1],q[3],q[4],q[3]-q[1],np.median(np.abs(x-q[2]))]
    return {prefix+n:float(v) for n,v in zip(names,values)}


def signal_features(wave):
    """Fixed observed-sample descriptors; no labels or fitted population values."""
    fhr,uc=np.asarray(wave,dtype=float).T;masks,short=signal_masks(wave)
    observed=masks['fhr_observed'];valid=fhr[observed]
    features=basic_stats(valid,'fhr_bpm_')
    features.update(basic_stats(uc[masks['uc_observed']],'uc_device_'))
    adjacent=observed[1:]&observed[:-1];diff=np.diff(fhr)[adjacent]
    features['fhr_adjacent_mean_abs_bpm']=float(np.mean(np.abs(diff))) if len(diff) else np.nan
    features['fhr_adjacent_rmssd_bpm']=float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan
    first=fhr[:1200][observed[:1200]];last=fhr[-1200:][observed[-1200:]]
    features['fhr_first_last_5min_median_delta_bpm']=(float(np.median(last)-np.median(first)) if len(first) and len(last) else np.nan)
    minute_ranges=[]
    # At least 80% observed samples in a minute: an engineering coverage rule,
    # not an implementation of a commercial minute-selection algorithm.
    for start in range(0,len(fhr),60*FS):
        segment=fhr[start:start+60*FS];mask=observed[start:start+60*FS]
        if mask.mean()>=.8:minute_ranges.append(float(np.ptp(segment[mask])))
    features['fhr_observed_minute_range_median_bpm']=float(np.median(minute_ranges)) if minute_ranges else np.nan
    # RR derived from Doppler FHR is not a true beat-to-beat RR series. All
    # differences stay inside minutes; acceleration/deceleration exclusion is
    # not attempted, so this descriptor must never be called device STV.
    rr=np.divide(60000.,fhr,out=np.zeros_like(fhr),where=observed)
    complete=len(fhr)//240*240
    counts=observed[:complete].reshape(-1,16,15).sum(axis=2)
    totals=rr[:complete].reshape(-1,16,15).sum(axis=2)
    epoch_rr=np.divide(totals,counts,out=np.full_like(totals,np.nan),where=counts>=12)
    differences=np.abs(np.diff(epoch_rr,axis=1))
    epoch_diffs=differences[np.isfinite(differences)]
    features['project_rr_epoch_adjacent_abs_ms']=float(np.mean(epoch_diffs)) if len(epoch_diffs) else np.nan
    gaps=runs(~observed)
    same=np.zeros(len(fhr),dtype=bool)
    same[1:]=observed[1:]&observed[:-1]&(np.diff(fhr)==0)
    constant_runs=[(a-1,b) for a,b in runs(same)]
    uc_same=np.zeros(len(uc),dtype=bool)
    uc_same[1:]=masks['uc_observed'][1:]&masks['uc_observed'][:-1]&(np.diff(uc)==0)
    uc_constant_runs=[(a-1,b) for a,b in runs(uc_same)]
    quality={name+'_fraction':float(mask.mean()) for name,mask in masks.items()}
    quality.update(fhr_longest_gap_s=max([b-a for a,b in gaps],default=0)/FS,
        fhr_gap_count=len(gaps),fhr_long_gap_over_2s_count=sum(b-a>8 for a,b in gaps),
        fhr_observed_unique_values=len(np.unique(valid)),
        uc_negative_fraction=float(np.mean(np.isfinite(uc)&(uc<0))),
        uc_at_100_fraction=float(np.mean(uc==100)),
        fhr_longest_constant_valid_run_s=max([b-a for a,b in constant_runs],default=0)/FS,
        fhr_fraction_in_constant_runs_30s=sum(b-a for a,b in constant_runs if b-a>=30*FS)/len(fhr),
        uc_longest_constant_run_s=max([b-a for a,b in uc_constant_runs],default=0)/FS,
        fhr_nonfinite_fraction=float(np.mean(~np.isfinite(fhr))))
    audit=dict(observed_mean=features['fhr_bpm_mean'],observed_min=features['fhr_bpm_min'],
        observed_max=features['fhr_bpm_max'],observed_std=features['fhr_bpm_std'],
        short_gap_mean=float(np.nanmean(short)) if observed.any() else np.nan)
    source_values=fhr[masks['fhr_source_present']]
    audit['positive_source_mean']=float(source_values.mean()) if len(source_values) else np.nan
    audit['positive_source_min']=float(source_values.min()) if len(source_values) else np.nan
    audit['positive_source_max']=float(source_values.max()) if len(source_values) else np.nan
    if observed.any():
        all_gap_filled=np.interp(np.arange(len(fhr)),np.flatnonzero(observed),fhr[observed])
        audit['all_gap_linear_mean_audit_only']=float(all_gap_filled.mean())
    else:audit['all_gap_linear_mean_audit_only']=np.nan
    return features,quality,masks,audit


def scan_signals(frame,names,force=False):
    directories()
    algorithm=''.join(inspect.getsource(f) for f in [runs,bounded_interpolation,signal_masks,basic_stats,signal_features])
    fp=hashlib.sha256((digest(ARCHIVE)+algorithm+str(SHORT_GAP_S)+str(FHR_ENGINEERING_RANGE)).encode()).hexdigest()
    marker=OUT/'signal_scan_receipt.json'
    if marker.exists() and not force:
        saved=json.loads(marker.read_text())
        if saved['fingerprint']==fp and all((PROCESSED/n).exists() for n in
            ['signals_original_float32.npy','signal_features.parquet','signal_quality.parquet','signal_index.parquet','signal_masks_packed.npz']):
            return tuple(pd.read_parquet(PROCESSED/n) for n in ['signal_features.parquet','signal_quality.parquet','signal_index.parquet'])
    shape=(len(frame),N_SAMPLES,2)
    signals=np.lib.format.open_memmap(PROCESSED/'signals_original_float32.npy',mode='w+',dtype=np.float32,shape=shape)
    packed={k:np.zeros((len(frame),N_SAMPLES//8),dtype=np.uint8) for k in
            ['fhr_source_present','fhr_observed','fhr_interpolated','fhr_range_flag','fhr_jump_flag','uc_observed','uc_zero']}
    feature_rows=[];quality_rows=[];index_rows=[];audits=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for i,row in enumerate(frame.itertuples(index=False)):
            payload=z.read(names[row.record_id]) # zipfile checks this member's CRC
            wave=np.loadtxt(io.BytesIO(payload),delimiter=',',skiprows=1,dtype=np.float32)
            if wave.shape!=(N_SAMPLES,2):raise ValueError(f'Wrong waveform shape: {row.record_id}: {wave.shape}')
            if payload.splitlines()[0].decode('utf-8-sig').strip()!='FHR,UC':raise ValueError('Unexpected signal columns')
            signals[i]=wave
            features,quality,masks,audit=signal_features(wave)
            for key,mask in masks.items():packed[key][i]=np.packbits(mask,bitorder='little')
            feature_rows.append(dict(record_id=row.record_id,**features))
            quality_rows.append(dict(record_id=row.record_id,**quality))
            index_rows.append(dict(record_id=row.record_id,signal_row=i,source_member=names[row.record_id],
                csv_sha256=hashlib.sha256(payload).hexdigest(),
                fhr_sha256=hashlib.sha256(np.ascontiguousarray(wave[:,0]).tobytes()).hexdigest(),
                waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest()))
            audits.append(dict(record_id=row.record_id,**audit))
            if (i+1)%2000==0:print(f'JNU signals {i+1}/{len(frame)}',flush=True)
    signals.flush();del signals
    for rows,name in [(feature_rows,'signal_features.parquet'),(quality_rows,'signal_quality.parquet'),(index_rows,'signal_index.parquet')]:
        pd.DataFrame(rows).to_parquet(PROCESSED/name,index=False)
    np.savez_compressed(PROCESSED/'signal_masks_packed.npz',**packed)
    comparisons=frame[['record_id','F_Mean','F_Min','F_Max','F_SD']].merge(pd.DataFrame(audits),on='record_id',validate='one_to_one')
    comparisons.to_parquet(OUT/'published_raw_comparison.parquet',index=False)
    write_json(marker,dict(fingerprint=fp,records=len(frame),samples_per_channel=N_SAMPLES,
        sampling_hz=FS,shape=list(shape),dtype='float32',source_member_crc_checked=True,
        mask_bitorder='little',short_internal_interpolation_max_s=SHORT_GAP_S,
        primary_descriptors='FHR analysis envelope 30..240; finite UC including zeros',
        fhr_engineering_range=list(FHR_ENGINEERING_RANGE),
        artifact_flags='original samples retained; separate range mask; jumps not automatically removed',
        signal_file_sha256=digest(PROCESSED/'signals_original_float32.npy')))
    return pd.DataFrame(feature_rows),pd.DataFrame(quality_rows),pd.DataFrame(index_rows)


def clean_metadata(frame):
    """Conservative parsing with an explicit source-value repair log.

    No diagnosis is inferred from Apgar and no malformed value is guessed. All
    clinical fields are excluded from the primary signal-only predictor matrix.
    """
    clean=frame.copy();issues=[];checks=[]
    published=list(frame.loc[:,'F_BL':].columns.drop('figo_label'))
    text={'record_id','gestational_age','fetal_presentation'}
    binary=[c for c in frame if (pd.api.types.is_bool_dtype(frame[c]) or c=='amniotic_fluid_bloody')]
    binary+=['figo_label','neonatal_asphyxia','pregnancy_anemia','gestational_diabetes','liver_disease',
        'gestational_hypertension','thyroid_disease','fetal_growth_restriction','oligohydramnios','prom','meconium_stained_amniotic_fluid']
    integer=['gravidity','parity','figo_label','neonatal_asphyxia']+[c for c in frame if c.startswith('apgar_')]
    positive=['maternal_age','height_cm','weight_kg','bmi','birth_weight_kg','birth_length_cm','birth_head_circumference_cm']
    def record(column,mask,reason):
        for i in np.flatnonzero(np.asarray(mask)):
            issues.append(dict(record_id=frame.iloc[i].record_id,column=column,
                raw_value=str(frame.iloc[i][column]),reason=reason))
    for column in frame:
        if column in published or column=='patient_id':continue
        original=frame[column]
        if column in text:
            clean[column]=original.astype('string').str.strip().replace({'':pd.NA,'/':pd.NA,'--':pd.NA})
            continue
        if column in binary and column not in ['figo_label','neonatal_asphyxia']:
            s=original.astype('string').str.strip().str.lower()
            mapped=s.map({'true':1.,'false':0.,'0':0.,'1':1.,'0.0':0.,'1.0':1.})
            bad=original.notna()&mapped.isna();record(column,bad,'unrecognized_binary_or_sentinel')
            clean[column]=mapped.astype('Float64');continue
        s=original.astype('string').str.strip()
        number=pd.to_numeric(s,errors='coerce').astype(float)
        malformed=original.notna()&number.isna();record(column,malformed,'non_numeric_or_sentinel')
        nonfinite=number.notna()&~np.isfinite(number);record(column,nonfinite,'nonfinite')
        number.loc[nonfinite]=np.nan
        invalid=pd.Series(False,index=frame.index)
        if column in integer:invalid|=number.notna()&~np.isclose(number,number.round())
        if column.startswith('apgar_'):
            maximum=10 if column in ['apgar_1min','apgar_5min','apgar_10min'] else 2
            invalid|=number.notna()&~number.between(0,maximum)
        if column in ['figo_label','neonatal_asphyxia']:invalid|=number.notna()&~number.isin([0,1])
        if column in ['gravidity','parity']:invalid|=number.notna()&(number<(1 if column=='gravidity' else 0))
        if column in positive:invalid|=number.notna()&(number<=0)
        record(column,invalid,'invalid_domain_or_noninteger');number.loc[invalid]=np.nan
        clean[column]=number
    def parse_gestation(value):
        if pd.isna(value):return np.nan
        match=re.fullmatch(r'(\d{1,2})(?:\+(\d))?',str(value).strip())
        if not match:return np.nan
        weeks=int(match.group(1));days=int(match.group(2) or 0)
        return weeks+days/7 if 0<weeks<=50 and days<=6 else np.nan
    clean['gestational_age_reported_weeks']=clean.gestational_age.map(parse_gestation)
    record('gestational_age',frame.gestational_age.notna()&clean.gestational_age_reported_weeks.isna(),'unparsed_gestation')
    def crosscheck(name,mask):
        for rid in clean.loc[mask,'record_id']:checks.append(dict(record_id=rid,check=name))
    for minute in [1,5,10]:
        total=f'apgar_{minute}min';components=[c for c in clean if c.startswith(total+'_')]
        complete=clean[components].notna().all(axis=1)&clean[total].notna()
        crosscheck(total+'_component_sum_disagrees',complete&(clean[components].sum(axis=1)!=clean[total]))
    crosscheck('parity_exceeds_gravidity',clean.parity.notna()&clean.gravidity.notna()&(clean.parity>clean.gravidity))
    crosscheck('gravidity_above_20_review_only',clean.gravidity>20)
    bmi_from_values=clean.weight_kg/(clean.height_cm/100)**2
    crosscheck('bmi_discrepancy_over_1_review_only',clean.bmi.notna()&bmi_from_values.notna()&((clean.bmi-bmi_from_values).abs()>1))
    crosscheck('asphyxia_positive_both_apgar_above_7_review_only',
        (clean.neonatal_asphyxia==1)&(clean.apgar_1min>7)&(clean.apgar_5min>7))
    repeated=[]
    for c in [n for n in frame.columns if n not in published+['record_id','figo_label']]:
        n=clean.groupby('patient_id')[c].nunique(dropna=True)
        repeated.append(dict(column=c,patients_with_conflicting_values=int((n>1).sum())))
    pd.DataFrame(repeated).to_csv(OUT/'patient_metadata_consistency.csv',index=False)
    repairs=pd.DataFrame(issues,columns=['record_id','column','raw_value','reason'])
    contradictions=pd.DataFrame(checks,columns=['record_id','check'])
    repairs.to_parquet(OUT/'metadata_normalization_issues.parquet',index=False)
    repairs.groupby(['column','reason']).size().rename('rows').reset_index().to_csv(OUT/'metadata_normalization_summary.csv',index=False)
    contradictions.to_csv(OUT/'cross_field_flags.csv',index=False)
    clean.to_parquet(PROCESSED/'metadata_clean.parquet',index=False)
    frame[['record_id',*published]].to_parquet(PROCESSED/'published_features_quarantined.parquet',index=False)
    roles=[]
    for c in frame:
        if c in ['record_id','patient_id']:role='identifier'
        elif c in ['figo_label','neonatal_asphyxia']:role='target_only'
        elif c in published:role='published_feature_definition_or_fit_scope_unverified'
        elif c in ['maternal_age','gravidity','parity','height_cm','weight_kg','bmi']:
            role='potential_covariate_timing_unverified'
        elif c in ['pregnancy_anemia','gestational_diabetes','liver_disease','gestational_hypertension',
                   'thyroid_disease','fetal_growth_restriction','oligohydramnios','prom',
                   'meconium_stained_amniotic_fluid','gestational_age','bpd_mm','ac_mm','fl_mm','hc_mm']:
            role='timing_unverified_exclude'
        else:role='delivery_or_postnatal_exclude'
        if role=='identifier':kind='identifier'
        elif c in binary:kind='binary'
        elif c in ['gravidity','parity']:kind='count'
        elif c.startswith('apgar_'):kind='ordinal_score'
        elif c=='fetal_presentation':kind='nominal_category'
        elif c=='gestational_age':kind='encoded_time_interval'
        elif c in published:kind='published_numeric_definition_unverified'
        else:kind='numeric_measure_or_count'
        roles.append(dict(column=c,role=role,primary_predictor_allowed=False,
            statistical_type=kind,
            source_dtype=str(frame[c].dtype),source_missing=int(frame[c].isna().sum()),
            clean_missing=int(clean[c].isna().sum()),unique_nonmissing=int(clean[c].nunique())))
    dictionary=pd.DataFrame(roles);dictionary.to_csv(OUT/'metadata_column_roles.csv',index=False)
    filename=frame.record_id.str.extract(r'^(\d+)_(\d+)_(\d+)$')
    write_json(OUT/'filename_audit.json',dict(parse_failures=int(filename.isna().any(axis=1).sum()),
        unique_prefixes=int(filename[0].nunique()),unique_suffixes=int(filename[2].nunique()),
        unique_patient_ids=int(frame.patient_id.nunique()),chronology_recoverable=False,
        decision='Use explicit patient_id; prefix is record-unique, common suffix is not a reliable date'))
    return clean,dictionary,repairs,contradictions


def patient_components(frame,index):
    """Keep patient IDs and exact repeated input signals on the same split side."""
    patients=frame.patient_id.to_numpy();parent={int(p):int(p) for p in np.unique(patients)}
    def find(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    aligned=frame[['record_id','patient_id']].merge(index,on='record_id',validate='one_to_one')
    duplicate_rows=[]
    for key in ['fhr_sha256','waveform_sha256']:
        for value,group in aligned[aligned[key].duplicated(keep=False)].groupby(key):
            pp=group.patient_id.unique()
            for patient in pp[1:]:union(int(pp[0]),int(patient))
            duplicate_rows.append(dict(hash_type=key,sha256=value,recordings=len(group),patients=len(pp)))
    result=pd.DataFrame({'record_id':frame.record_id,'patient_component':[find(int(p)) for p in patients]})
    pd.DataFrame(duplicate_rows,columns=['hash_type','sha256','recordings','patients']).to_csv(OUT/'exact_signal_duplicate_groups.csv',index=False)
    return result


def split_patients(frame,index):
    components=patient_components(frame,index)
    base=frame[['record_id','patient_id','figo_label','neonatal_asphyxia']].merge(components,on='record_id',validate='one_to_one')
    patients=base.groupby('patient_id').agg(patient_component=('patient_component','first'),
        outcome=('neonatal_asphyxia','first'))
    counts=base.assign(normal=(base.figo_label==0).astype(int),abnormal=(base.figo_label==1).astype(int)).groupby('patient_component').agg(
        records=('record_id','size'),normal=('normal','sum'),abnormal=('abnormal','sum'))
    pc=patients.groupby('patient_component').agg(patients=('outcome','size'),
        positive_patients=('outcome',lambda s:int((s==1).sum())),
        known_patients=('outcome',lambda s:int(s.notna().sum())))
    groups=counts.join(pc)
    groups['stratum']=(np.where(groups.normal>0,'N','')+np.where(groups.abnormal>0,'A','')+
        np.where(groups.positive_patients>0,'_positive',np.where(groups.known_patients==0,'_unknown','_negative')))
    keys=list(SPLITS);target=np.array(list(SPLITS.values()))
    metrics=['records','normal','abnormal','patients','positive_patients','known_patients']
    totals=groups[metrics].sum().to_numpy(dtype=float)
    candidates=[]
    for attempt in range(64):
        rng=np.random.default_rng(SEED+attempt);assignment={}
        for _,sub in groups.groupby('stratum',sort=True):
            ids=rng.permutation(sub.index.to_numpy());quotas=len(ids)*target
            allocated=np.floor(quotas).astype(int)
            for j in np.argsort(-(quotas-allocated),kind='stable')[:len(ids)-allocated.sum()]:allocated[j]+=1
            offset=0
            for j,n in enumerate(allocated):
                for component in ids[offset:offset+n]:assignment[int(component)]=j
                offset+=n
        sums=np.array([groups.loc[[g for g,a in assignment.items() if a==j],metrics].sum().to_numpy(dtype=float) for j in range(3)])
        fraction=sums/np.maximum(totals,1)
        objective=float(np.square(fraction-target[:,None]).sum())
        if (sums[:,4]==0).any() or (sums[:,1:3]==0).any():objective+=100
        candidates.append((objective,attempt,assignment,sums))
    objective,attempt,assignment,sums=min(candidates,key=lambda v:(v[0],v[1]))
    base['split']=base.patient_component.map(lambda g:keys[assignment[int(g)]])
    base['inverse_patient_record_count']=1/base.patient_id.map(base.patient_id.value_counts())
    assert base.groupby('patient_id').split.nunique().max()==1
    assert base.groupby('patient_component').split.nunique().max()==1
    for name in ['fhr_sha256','waveform_sha256']:
        aligned=base.merge(index[['record_id',name]],on='record_id',validate='one_to_one')
        assert aligned.groupby(name).split.nunique().max()==1
    base.to_parquet(PROCESSED/'splits.parquet',index=False)
    balance=[]
    for split,part in base.groupby('split',sort=False):
        pp=part.groupby('patient_id').neonatal_asphyxia.first()
        balance.append(dict(split=split,records=len(part),patients=part.patient_id.nunique(),components=part.patient_component.nunique(),
            normal_records=int((part.figo_label==0).sum()),abnormal_records=int((part.figo_label==1).sum()),
            abnormal_record_fraction=float((part.figo_label==1).mean()),
            asphyxia_positive_records=int((part.neonatal_asphyxia==1).sum()),
            asphyxia_positive_patients=int((pp==1).sum()),asphyxia_known_patients=int(pp.notna().sum()),
            asphyxia_unknown_records=int(part.neonatal_asphyxia.isna().sum()),
            asphyxia_patient_prevalence=float((pp==1).sum()/pp.notna().sum())))
    balance=pd.DataFrame(balance);balance.to_csv(OUT/'split_balance.csv',index=False)
    write_json(OUT/'split_design.json',dict(seed=SEED,attempt_selected=attempt,candidates=64,objective=objective,
        fractions=SPLITS,primary_unit='patient_id + connected exact signal duplicates',
        selection_inputs=metrics+['target-presence strata'],feature_or_model_scores_used=False,
        rare_positive_patients=int((patients.outcome==1).sum()),components=len(groups),
        patient_groups=len(patients),test_status='allocated and sealed before model development'))
    return base,balance


class JNUTabularGuard(BaseEstimator,TransformerMixin):
    def __init__(self,features):self.features=features
    def fit(self,X,y=None):
        self.transform(X);self.n_features_in_=len(self.features);self.feature_names_in_=np.array(self.features,dtype=object)
        return self
    def transform(self,X):
        if not isinstance(X,pd.DataFrame):raise TypeError('Use a named numeric DataFrame')
        if X.columns.duplicated().any() or set(X.columns)!=set(self.features):raise ValueError('Unexpected columns; labels and metadata must never enter X')
        x=X[list(self.features)].astype(float)
        if np.isinf(x.to_numpy()).any():raise ValueError('Infinity is not a missing value')
        return x
    def get_feature_names_out(self,input_features=None):return np.array(self.features,dtype=object)


def preprocessor_template(features):
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',RobustScaler())])
    columns=ColumnTransformer([('scaled',numeric,list(features)),
        ('missing',MissingIndicator(features='all',error_on_new=False),list(features))],sparse_threshold=0,verbose_feature_names_out=True)
    return Pipeline([('guard',JNUTabularGuard(tuple(features))),('columns',columns)])


class JNUSignalNormalizer:
    """Train-observed moments, bounded gap filling and mandatory validity masks."""
    def fit_rows(self,signals,indices):
        counts=np.zeros(2,dtype=np.int64);totals=np.zeros(2);squares=np.zeros(2)
        for start in range(0,len(indices),64):
            batch=np.asarray(signals[np.asarray(indices[start:start+64])],dtype=np.float64)
            for channel in range(2):
                values=batch[:,:,channel];valid=np.isfinite(values)
                if channel==0:valid&=(values>=FHR_ENGINEERING_RANGE[0])&(values<=FHR_ENGINEERING_RANGE[1])
                v=values[valid];counts[channel]+=len(v);totals[channel]+=v.sum();squares[channel]+=np.square(v).sum()
        if (counts==0).any():raise ValueError('A channel has no observed training samples')
        self.mean_=totals/counts
        variance=np.maximum(squares/counts-self.mean_**2,0)
        self.scale_=np.sqrt(variance);self.scale_[self.scale_<1e-12]=1.
        self.count_=counts;return self
    def transform_one(self,wave,max_internal_gap_s=SHORT_GAP_S):
        wave=np.asarray(wave,dtype=float)
        if wave.shape!=(N_SAMPLES,2):raise ValueError('Expected 7200 x 2 FHR/UC')
        observed=np.isfinite(wave);observed[:,0]&=(wave[:,0]>=FHR_ENGINEERING_RANGE[0])&(wave[:,0]<=FHR_ENGINEERING_RANGE[1])
        fhr,filled=bounded_interpolation(wave[:,0],observed[:,0],round(FS*max_internal_gap_s))
        used=observed.copy();used[:,0]|=filled
        cleaned=wave.copy();cleaned[:,0]=fhr
        normalized=(cleaned-self.mean_)/self.scale_
        normalized[~used]=0. # numeric placeholder only; used_mask is mandatory
        interpolation=np.zeros_like(used);interpolation[:,0]=filled
        return dict(values=normalized.astype(np.float32),observed_mask=observed,
            used_mask=used,interpolated_mask=interpolation)


def prepare_tables_and_scalers(clean,features,quality,index,splits):
    columns=[c for c in features if c!='record_id']
    primary=splits.merge(index[['record_id','signal_row']],on='record_id',validate='one_to_one').merge(features,on='record_id',validate='one_to_one')
    coverage=quality.set_index('record_id').fhr_observed_fraction
    primary['has_valid_fhr']=primary.record_id.map(coverage)>0
    primary['fhr_coverage_at_least_80pct']=primary.record_id.map(coverage)>=.8
    primary['primary_training_eligible']=primary.has_valid_fhr&primary.figo_label.isin([0,1])
    primary['clinical_outcome_training_eligible']=primary.has_valid_fhr&primary.neonatal_asphyxia.isin([0,1])
    primary.to_parquet(PROCESSED/'jnu_primary.parquet',index=False)
    schema=[]
    for c in columns:
        unit='ms (project epoch descriptor; not device STV)' if c=='project_rr_epoch_adjacent_abs_ms' else 'bpm' if c.startswith('fhr_') else 'device UC units'
        schema.append(dict(feature=c,unit=unit,definition_version=VERSION,allows_missing=True,
            source='observed raw signal',not_vendor_metric=True))
    pd.DataFrame(schema).to_csv(PROCESSED/'primary_feature_schema.csv',index=False)
    template=preprocessor_template(columns);joblib.dump(template,MODELS/'preprocessor_template_unfitted.joblib')
    train=primary[(primary.split=='train')&primary.primary_training_eligible]
    fitted=preprocessor_template(columns).fit(train[columns])
    joblib.dump(fitted,MODELS/'tabular_preprocessor_train.joblib')
    learned=fitted.named_steps['columns'];numeric=learned.named_transformers_['scaled']
    imputer=numeric.named_steps['imputer'];scaler=numeric.named_steps['scaler']
    joblib.dump(imputer,MODELS/'numeric_imputer_train.joblib');joblib.dump(scaler,MODELS/'robust_scaler_train.joblib')
    joblib.dump(learned.named_transformers_['missing'],MODELS/'missing_indicator_all_columns.joblib')
    feature_names=fitted.get_feature_names_out().tolist()
    for split in SPLITS:
        part=primary[primary.split==split].copy()
        part[['record_id','patient_id','patient_component','signal_row','figo_label','neonatal_asphyxia','inverse_patient_record_count',
            'has_valid_fhr','fhr_coverage_at_least_80pct','primary_training_eligible','clinical_outcome_training_eligible']].to_parquet(PROCESSED/f'index_targets_{split}.parquet',index=False)
        part[['record_id',*columns]].to_parquet(PROCESSED/f'X_raw_{split}.parquet',index=False)
        matrix=fitted.transform(part[columns]);assert np.isfinite(matrix).all()
        np.save(PROCESSED/f'X_robust_{split}.npy',matrix.astype(np.float32))
    parameters=dict(fit_split='train',fit_rows=len(train),features=columns,transformed_features=feature_names,
        median_imputation=imputer.statistics_,robust_center=scaler.center_,robust_scale=scaler.scale_,
        missing_indicators='one for every input feature; unseen missingness does not fail',
        cv_rule='clone the unfitted template and fit inside each training fold; prepared matrices are not valid nested-CV inputs',
        clipping='none',outlier_deletion='none',models_trained=False)
    write_json(MODELS/'tabular_parameters.json',parameters)
    signals=np.load(PROCESSED/'signals_original_float32.npy',mmap_mode='r')
    normalizer=JNUSignalNormalizer().fit_rows(signals,train.signal_row.to_numpy())
    joblib.dump(normalizer,MODELS/'signal_normalizer_train.joblib')
    write_json(MODELS/'signal_normalization.json',dict(fit_split='train',channel_order=['FHR','UC'],
        mean=normalizer.mean_,scale=normalizer.scale_,observed_sample_count=normalizer.count_,
        fhr_validity='finite within provisional engineering envelope 30..240; original extremes retained separately',uc_validity='finite; zero retained',
        gap_rule='bounded internal interpolation <=2 seconds; mask returned; no extrapolation',
        missing_placeholder=0,mask_required=True))
    write_json(PROCESSED/'feature_contract.json',dict(primary_features=columns,quality_features=[c for c in quality if c!='record_id'],
        targets=['figo_label','neonatal_asphyxia'],ids=['record_id','patient_id','patient_component','signal_row'],
        source_published_features_allowed=False,clinical_covariates_allowed=False,
        hospital_STV_equivalence=False,test_selection_allowed=False))
    return primary,parameters


def metadata_and_feature_profiles(original,clean,primary,quality):
    profile=[]
    for c in original:
        raw=original[c];s=clean[c]
        row=dict(column=c,original_missing=int(raw.isna().sum()),clean_missing=int(s.isna().sum()),
            additional_missing=int(s.isna().sum()-raw.isna().sum()),source_dtype=str(raw.dtype),
            unique_nonmissing=int(s.nunique()))
        profile.append(row)
    pd.DataFrame(profile).to_csv(OUT/'metadata_missingness.csv',index=False)
    contract=json.loads((PROCESSED/'feature_contract.json').read_text())
    cols=contract['primary_features'];train=primary[primary.split=='train']
    rows=[]
    for c in cols:
        s=train[c]
        rows.append(dict(feature=c,n=len(s),missing=int(s.isna().sum()),zero=int((s==0).sum()),
            min=s.min(),q01=s.quantile(.01),q25=s.quantile(.25),median=s.median(),q75=s.quantile(.75),q99=s.quantile(.99),max=s.max()))
    pd.DataFrame(rows).to_csv(OUT/'train_feature_profile.csv',index=False)
    qtrain=quality.merge(primary[['record_id','split']],on='record_id',validate='one_to_one')
    qtrain[qtrain.split=='train'].drop(columns=['record_id','split']).describe().T.to_csv(OUT/'train_signal_quality_profile.csv')
    comparisons=pd.read_parquet(OUT/'published_raw_comparison.parquet').merge(primary[['record_id','split']],on='record_id',validate='one_to_one')
    comparisons=comparisons[comparisons.split=='train']
    measures=[('F_Mean','observed_mean'),('F_Mean','positive_source_mean'),('F_Mean','short_gap_mean'),('F_Mean','all_gap_linear_mean_audit_only'),
              ('F_Min','observed_min'),('F_Max','observed_max'),('F_SD','observed_std')]
    rows=[]
    for published,computed in measures:
        a,b=comparisons[published],comparisons[computed];ok=a.notna()&b.notna();error=(a[ok]-b[ok]).abs()
        rows.append(dict(published=published,recomputed=computed,rows=int(ok.sum()),
            within_0_01_fraction=float((error<=.01).mean()),median_abs_error=float(error.median()),max_abs_error=float(error.max())))
    pd.DataFrame(rows).to_csv(OUT/'published_feature_agreement_train.csv',index=False)


def gap_sensitivity(primary):
    """Train-only numerical sensitivity; no clinical labels or model scores."""
    signals=np.load(PROCESSED/'signals_original_float32.npy',mmap_mode='r')
    rows=[]
    train=primary[primary.split=='train']
    for record in train.itertuples(index=False):
        fhr=np.asarray(signals[record.signal_row,:,0],dtype=float)
        observed=np.isfinite(fhr)&(fhr>=FHR_ENGINEERING_RANGE[0])&(fhr<=FHR_ENGINEERING_RANGE[1])
        gaps=runs(~observed);base=fhr[observed].mean() if observed.any() else np.nan
        for seconds in [0.,1.,2.]:
            n=0;added=0.
            for a,b in gaps:
                if a>0 and b<len(fhr) and b-a<=seconds*FS:
                    n+=b-a;added+=(fhr[a-1]+fhr[b])/2*(b-a)
            mean=(fhr[observed].sum()+added)/(observed.sum()+n) if observed.sum()+n else np.nan
            rows.append(dict(record_id=record.record_id,max_gap_s=seconds,filled_fraction=n/len(fhr),
                abs_mean_change_bpm=abs(mean-base),remaining_missing_fraction=(~observed).mean()-n/len(fhr)))
    table=pd.DataFrame(rows)
    summary=table.groupby('max_gap_s').agg(records=('record_id','size'),
        mean_filled_fraction=('filled_fraction','mean'),p95_filled_fraction=('filled_fraction',lambda s:s.quantile(.95)),
        median_abs_mean_change_bpm=('abs_mean_change_bpm','median'),
        p95_abs_mean_change_bpm=('abs_mean_change_bpm',lambda s:s.quantile(.95)),
        mean_remaining_missing_fraction=('remaining_missing_fraction','mean')).reset_index()
    summary.to_csv(OUT/'gap_sensitivity_train.csv',index=False)
    return summary


def prepare_all():
    frame,names,source=verify_source()
    features,quality,index=scan_signals(frame,names)
    clean,dictionary,repairs,contradictions=clean_metadata(frame)
    splits,balance=split_patients(clean,index)
    primary,parameters=prepare_tables_and_scalers(clean,features,quality,index,splits)
    metadata_and_feature_profiles(frame,clean,primary,quality)
    gap_sensitivity(primary)
    import importlib.metadata
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','nbformat']}
    artifacts={str(p.relative_to(ROOT)):digest(p) for directory in [PROCESSED,MODELS] for p in sorted(directory.glob('*')) if p.is_file()}
    manifest=dict(version=VERSION,source=source,environment=versions,code_sha256=digest(Path(__file__)),
        split_design=json.loads((OUT/'split_design.json').read_text()),primary_features=parameters['features'],
        all_records_preserved=True,clinical_use_validated=False,models_trained=False,artifacts_sha256=artifacts)
    write_json(OUT/'manifest.json',manifest)
    return primary,balance,dictionary,repairs,contradictions
