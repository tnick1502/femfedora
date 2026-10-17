"""Label-free representations and detector sensitivity for the JNU revision."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from src.training.ctg_review_evidence import ROOT,OUT as REVIEW
from src.features.jnu_ctg_report_extraction import baseline_curve,detect_events,ExtractionConfig


def raw_time_features():
    destination=ROOT/'data/processed/jnu_ctg/review_v2/raw_minute_features.parquet'
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if destination.exists():
        cached=pd.read_parquet(destination)
        if cached.attrs.get('implementation_sha256')!=implementation:raise RuntimeError('Representation cache implementation changed; use a new version')
        return cached
    base=ROOT/'data/processed/jnu_ctg/v1';index=pd.read_parquet(base/'signal_index.parquet')
    signals=np.load(base/'signals_original_float32.npy',mmap_mode='r');blocks=[]
    for start in range(0,len(index),128):
        part=index.iloc[start:start+128];raw=np.asarray(signals[part.signal_row.to_numpy(int)],float)
        fhr=raw[:,:,0].reshape(len(part),30,240);uc=raw[:,:,1].reshape(len(part),30,240)
        fhr=np.where((fhr>=30)&(fhr<=240)&np.isfinite(fhr),fhr,np.nan)
        uc=np.where(np.isfinite(uc)&(uc>=0),uc,np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            stats={'fhr_mean':np.nanmean(fhr,axis=2),'fhr_std':np.nanstd(fhr,axis=2),
                'fhr_p10':np.nanquantile(fhr,.1,axis=2),'fhr_p90':np.nanquantile(fhr,.9,axis=2),
                'uc_mean':np.nanmean(uc,axis=2),'uc_std':np.nanstd(uc,axis=2)}
            diff=np.abs(np.diff(fhr,axis=2));stats['fhr_adjacent_abs']=np.nanmean(diff,axis=2)
        block=pd.DataFrame({f'{name}_minute{minute:02d}':values[:,minute] for name,values in stats.items() for minute in range(30)})
        block.insert(0,'record_id',part.record_id.to_numpy());blocks.append(block)
    result=pd.concat(blocks,ignore_index=True);result.attrs['implementation_sha256']=implementation
    destination.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(destination,index=False)
    return result


def baseline_only_amplitude(wave):
    """Alternative research descriptor, not a replacement vendor amplitude.

    Remove detected >=10-bpm excursions and 5s margins before minute ranges;
    require 30s original baseline observations. Baseline/event estimates still
    need independent annotations. No label or classifier score is consulted.
    """
    fhr=np.asarray(wave)[:,0];base,smooth,observed,filled,clean=baseline_curve(fhr)
    events=detect_events(smooth,base,observed,filled)
    keep=observed&np.isfinite(base)
    for e in events.itertuples():keep[max(0,int(e.start_s*4)-20):min(len(keep),int(e.end_s*4)+20)]=False
    ranges=[]
    for a in range(0,len(fhr),240):
        mask=keep[a:a+240]
        if mask.sum()>=120:ranges.append(float(np.ptp(fhr[a:a+240][mask])))
    return float(np.mean(ranges)) if ranges else np.nan


def extractor_probe():
    path=REVIEW/'jnu_amplitude_exclusion_sensitivity.csv'
    if path.exists():return pd.read_csv(path)
    base=ROOT/'data/processed/jnu_ctg/v1';index=pd.read_parquet(base/'signal_index.parquet').set_index('record_id')
    d=pd.read_parquet(base/'splits.parquet');m=pd.read_parquet(ROOT/'data/processed/jnu_ctg/report_features_v1/measurements.parquet').set_index('record_id')
    train=d[d.split=='train'].sort_values('record_id').sample(128,random_state=20261018)
    signals=np.load(base/'signals_original_float32.npy',mmap_mode='r');rows=[]
    for r in train.itertuples():
        original=m.loc[r.record_id];alternative=baseline_only_amplitude(signals[int(index.loc[r.record_id,'signal_row'])])
        rows.append(dict(record_id=r.record_id,minute_range_including_accelerations=original.amplitude_bpm,
            baseline_only_candidate_bpm=alternative,acc15_count=original.acc15_count,figo_label=r.figo_label))
    result=pd.DataFrame(rows);result.to_csv(path,index=False);return result
