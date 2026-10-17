"""Recompute expert-review claims; separate observations from interpretations."""
from pathlib import Path
import json,re,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,cohen_kappa_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/reviews/ctg_20261017'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


def prefix_proxy(filename):
    """Conservative initials-like proxy; S+digits acquisition codes stay separate.

    The source does not certify patient identity. This grouping may merge
    different patients and cannot rule out different prefixes for one patient.
    """
    stem=Path(str(filename)).stem.casefold()
    if re.fullmatch(r's\d+',stem):return stem
    match=re.match(r'[a-z]+',stem)
    return match.group(0) if match else stem.split('_')[0]


def uci_groups():
    data=pd.read_parquet(ROOT/'data/processed/ctg_uci/ctg_primary.parquet')
    data['prefix_proxy']=data.source_file.map(prefix_proxy)
    parent={g:g for g in data.prefix_proxy.unique()}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    # A feature-identical group may bridge proxy groups: preserve both rules.
    for _,frame in data.groupby('feature_duplicate_group',dropna=True):
        groups=sorted(frame.prefix_proxy.unique())
        for group in groups[1:]:parent[find(group)]=find(groups[0])
    data['review_group']=data.prefix_proxy.map(find)
    data[['source_excel_row','source_file','prefix_proxy','review_group','feature_duplicate_group']].to_parquet(OUT/'uci_group_assignments.parquet',index=False)
    return data


def run_evidence():
    OUT.mkdir(parents=True,exist_ok=True)
    uci=uci_groups();old=pd.read_parquet(ROOT/'data/processed/ctg_uci/ctg_splits.parquet')
    uci=uci.merge(old[['source_excel_row','split']],on='source_excel_row',validate='one_to_one')
    support=uci.groupby('review_group').agg(segments=('NSP','size'),files=('source_file','nunique'),
        split_count=('split','nunique'),path_segments=('NSP',lambda x:int((x==3).sum())))
    support.to_csv(OUT/'uci_proxy_group_support.csv')
    path=uci[(uci.split=='test')&(uci.NSP==3)]
    train=uci[uci.split=='train'];test=uci[uci.split=='test']
    same=train.prefix_proxy.unique();same_path=train.loc[train.NSP==3,'prefix_proxy'].unique()
    original=dict(records=len(uci),source_files=uci.source_file.nunique(),prefix_proxies=uci.prefix_proxy.nunique(),
        merged_review_groups=uci.review_group.nunique(),proxy_groups_crossing_old_splits=int((support.split_count>1).sum()),
        old_test_records=len(test),old_test_path_segments=len(path),old_test_path_files=path.source_file.nunique(),
        old_test_path_proxy_groups=path.prefix_proxy.nunique(),
        old_test_path_files_shared_prefix_train=path[path.prefix_proxy.isin(same)].source_file.nunique(),
        old_test_path_files_shared_path_prefix_train=path[path.prefix_proxy.isin(same_path)].source_file.nunique(),
        largest_test_path_file=path.source_file.value_counts().head(1).to_dict(),
        patient_identity_verified=False)
    # Original worksheet retains b/e positions. Never infer overlap from row order.
    archive=ROOT/'data/external/uci_cardiotocography/1.0.0/cardiotocography.zip'
    with zipfile.ZipFile(archive) as z:
        entry=next(n for n in z.namelist() if n.lower().endswith('ctg.xls'))
        import io
        raw=pd.read_excel(io.BytesIO(z.read(entry)),sheet_name='Raw Data',header=0)
    raw=raw.loc[raw['FileName'].notna()].copy()
    pairs=[]
    for file,g in raw.groupby('FileName'):
        g=g.sort_values('b')
        for i in range(1,len(g)):
            pairs.append(dict(file=file,previous_end=float(g.iloc[i-1]['e']),start=float(g.iloc[i]['b']),
                overlaps=bool(g.iloc[i]['b']<g.iloc[i-1]['e'])))
    pd.DataFrame(pairs).to_csv(OUT/'uci_adjacent_interval_overlap.csv',index=False)
    original.update(adjacent_pairs=len(pairs),overlapping_adjacent_pairs=sum(x['overlaps'] for x in pairs))
    write_json(OUT/'uci_claims.json',original)

    base=ROOT/'data/processed/jnu_ctg/v1';dest=ROOT/'data/processed/jnu_ctg/report_features_v1'
    meta=pd.read_parquet(base/'metadata_clean.parquet')
    splits=pd.read_parquet(base/'splits.parquet')
    measurements=pd.read_parquet(dest/'measurements.parquet')
    features=pd.read_parquet(base/'signal_features.parquet')
    d=measurements.merge(splits,on='record_id',validate='one_to_one')
    y=d.figo_label.to_numpy(int);t=d[d.split=='train']
    rows=[]
    for col,limits in [('stv_ms',[-np.inf,4,15,np.inf]),('amplitude_bpm',[-np.inf,10,30,np.inf])]:
        t=t.copy();t['bin']=pd.cut(t[col],limits,right=False)
        for label,g in t.groupby('bin',observed=True):
            rows.append(dict(feature=col,bin=str(label),records=len(g),patients=g.patient_id.nunique(),abnormal_fraction=g.figo_label.mean()))
    pd.DataFrame(rows).to_csv(OUT/'jnu_low_high_variation_support.csv',index=False)
    missing=[]
    for split,g in d.groupby('split'):
        for col in ['baseline_bpm','stv_ms','stv_4s_ms','amplitude_bpm']:
            z=g[g[col].isna()]
            missing.append(dict(split=split,feature=col,records=len(z),patients=z.patient_id.nunique(),abnormal=int(z.figo_label.sum())))
    pd.DataFrame(missing).to_csv(OUT/'jnu_missingness_by_class.csv',index=False)
    univariate=[];bins=[]
    for col in ['acc15_per_10min','acc10_per_10min','signal_loss_pct','stv_ms','amplitude_bpm']:
        valid=t[col].notna()
        univariate.append(dict(feature=col,records=int(valid.sum()),auc=float(roc_auc_score(t.loc[valid,'figo_label'],t.loc[valid,col]))))
        labels=pd.qcut(t[col],10,duplicates='drop')
        for label,g in t.groupby(labels,observed=True):bins.append(dict(feature=col,bin=str(label),mean=g[col].mean(),records=len(g),abnormal_fraction=g.figo_label.mean()))
    pd.DataFrame(univariate).to_csv(OUT/'jnu_train_univariate_auc.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'jnu_train_feature_deciles.csv',index=False)
    # All unordered within-patient pairs, symmetrized; no chronology claimed.
    left=[];right=[]
    for _,g in d.groupby('patient_id'):
        labels=g.figo_label.to_numpy(int)
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):left.extend([labels[i],labels[j]]);right.extend([labels[j],labels[i]])
    agreement=dict(unordered_pairs=len(left)//2,symmetrized_pair_kappa=float(cohen_kappa_score(left,right)),
        interpretation='Different recordings, not repeated annotations of the same trace; not inter-rater reliability.')
    # Outcome copied per recording: record association is descriptive, not independent n.
    outcome=d[['record_id','patient_id','figo_label']].merge(meta[['record_id','neonatal_asphyxia']],on='record_id',validate='one_to_one')
    outcome.groupby('figo_label').neonatal_asphyxia.agg(['count','sum','mean']).to_csv(OUT/'jnu_outcome_record_association.csv')
    patient=outcome.groupby('patient_id').agg(any_abnormal=('figo_label','max'),outcome=('neonatal_asphyxia','first'),records=('record_id','size'))
    patient.groupby('any_abnormal').outcome.agg(['count','sum','mean']).to_csv(OUT/'jnu_outcome_patient_association.csv')
    write_json(OUT/'jnu_label_agreement.json',agreement)
    # Suspected published channel attribution: evidence, not proof of swapped raw channels.
    pub=meta[['record_id']+[c for c in meta if c.startswith('U_') and ('STTF' in c or c in ['U_LPF_ST10','U_LPF_ST11'])]]
    compare=pub.merge(features[['record_id','fhr_bpm_std','uc_device_std','fhr_bpm_iqr','uc_device_iqr']],on='record_id',validate='one_to_one')
    compare=compare[compare.record_id.isin(t.record_id)]
    corr=[]
    for col in pub.columns[1:]:
        for reference in ['fhr_bpm_std','uc_device_std','fhr_bpm_iqr','uc_device_iqr']:
            for method in ['pearson','spearman']:
                a=compare[[col,reference]].dropna()
                corr.append(dict(published_feature=col,reference=reference,method=method,records=len(a),correlation=a[col].corr(a[reference],method=method)))
    pd.DataFrame(corr).to_csv(OUT/'jnu_published_U_channel_correlations.csv',index=False)
    # Scan actual released raw UC maxima; exact values are signatures, not device IDs.
    signals=np.load(base/'signals_original_float32.npy',mmap_mode='r');maxima=[]
    for start in range(0,len(signals),256):
        a=np.asarray(signals[start:start+256,:,1],float)
        maxima.extend(np.max(np.where(np.isfinite(a),a,-np.inf),axis=1).tolist())
    index=pd.read_parquet(base/'signal_index.parquet')[['record_id','signal_row']]
    index['raw_uc_max']=np.asarray(maxima)[index.signal_row.to_numpy(int)]
    uc=d[['record_id','patient_id','figo_label','split']].merge(index[['record_id','raw_uc_max']],on='record_id',validate='one_to_one')
    uc.groupby('raw_uc_max').agg(records=('record_id','size'),patients=('patient_id','nunique'),abnormal_fraction=('figo_label','mean')).sort_values('records',ascending=False).to_csv(OUT/'jnu_raw_uc_max_signatures.csv')
    uc.to_parquet(OUT/'jnu_uc_signature_records.parquet',index=False)
    print('Review evidence computed:',original)
    return original


if __name__=='__main__':run_evidence()
