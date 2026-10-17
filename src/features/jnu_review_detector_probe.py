"""Controlled acceleration-exclusion probe; not a model-input selection tool."""
import numpy as np
import pandas as pd
from src.training.ctg_review_evidence import ROOT,OUT
from src.features.jnu_ctg_report_extraction import baseline_curve,detect_events,minute_measurements


def controlled_amplitude_probe():
    destination=OUT/'jnu_amplitude_controlled_ablation.csv'
    if destination.exists():return pd.read_csv(destination)
    base=ROOT/'data/processed/jnu_ctg/v1';index=pd.read_parquet(base/'signal_index.parquet').set_index('record_id')
    source=pd.read_parquet(base/'splits.parquet');sample=source[source.split=='train'].sort_values('record_id').sample(128,random_state=20261018)
    wave=np.load(base/'signals_original_float32.npy',mmap_mode='r');rows=[]
    for record in sample.itertuples():
        fhr=np.asarray(wave[int(index.loc[record.record_id,'signal_row']),:,0],float)
        baseline,smooth,observed,filled,clean=baseline_curve(fhr)
        events=detect_events(smooth,baseline,observed,filled)
        minute=minute_measurements(clean,observed,filled,baseline,events)
        keep=observed.copy()
        for event in events.itertuples():
            if event.kind=='acceleration':
                a=max(0,int(event.start_s*4)-20);b=min(len(fhr),int(event.end_s*4)+20);keep[a:b]=False
        before=[];after=[];minutes_before=0
        for k,m in enumerate(minute.itertuples()):
            if not m.ltv_eligible:continue
            minutes_before+=1;a=k*240;b=a+240;retained=keep[a:b]
            if retained.sum()<120:continue
            before.append(float(np.ptp(fhr[a:b][observed[a:b]])));after.append(float(np.ptp(fhr[a:b][retained])))
        original=float(np.mean(before)) if before else np.nan;ablated=float(np.mean(after)) if after else np.nan
        if before:assert ablated<=original+1e-9
        rows.append(dict(record_id=record.record_id,figo_label=record.figo_label,
            eligible_minutes_before=minutes_before,common_eligible_minutes=len(before),
            range_before_same_minutes_bpm=original,range_after_acceleration_exclusion_bpm=ablated,
            method='same eligible minutes; remove acceleration spans and 5s margins; require >=30s retained original samples'))
    result=pd.DataFrame(rows);result.to_csv(destination,index=False);return result
