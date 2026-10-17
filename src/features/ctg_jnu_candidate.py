"""Source audit of an antenatal candidate cohort; does not train a new endpoint."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd

from src.training.ctg_experiments import ROOT,write_json

SOURCE=ROOT/'data/external/jnu_ctg/1.0.0'
OUT=ROOT/'research/experiments/ctg_uci/report_features_v1/jnu_candidate'
EXPECTED_MD5='ac1cfcba2f1b3336596544211d776771'


def audit_jnu():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=SOURCE/'JNU-CTG.zip'
    checksum=hashlib.md5(archive.read_bytes()).hexdigest()
    if checksum!=EXPECTED_MD5:raise ValueError('JNU archive is incomplete or checksum does not match Zenodo')
    with zipfile.ZipFile(archive) as z:
        # Read selected members; never execute downloaded Python or extract paths.
        for name in ['README.md','LICENSE.txt','metadata.csv']:
            (SOURCE/name).write_bytes(z.read('JNU-CTG/'+name))
        names=[n for n in z.namelist() if '/waveforms/csv_raw_7200/' in n and n.endswith('.csv')]
        frame=pd.read_csv(SOURCE/'metadata.csv',low_memory=False)
        assert len(names)==len(frame)==20769 and frame.record_id.is_unique
        assert {Path(n).stem for n in names}==set(frame.record_id)
        name=names[0];wave=pd.read_csv(io.BytesIO(z.read(name)))
    assert wave.shape==(7200,2) and list(wave)==['FHR','UC']
    assert frame.patient_id.notna().all() and frame.patient_id.nunique()==12606
    support=[]
    for target in ['figo_label','neonatal_asphyxia']:
        assert set(frame[target].dropna().unique())=={0,1}
        for label in [0,1]:
            selected=frame[frame[target]==label]
            support.append(dict(target=target,label=label,recordings=len(selected),
                patients_with_label=selected.patient_id.nunique(),missing_target=int(frame[target].isna().sum())))
    support=pd.DataFrame(support);support.to_csv(OUT/'target_support.csv',index=False)
    selected=['F_BL','F_A_N','F_D_N','F_LTV','F_STV']
    missing=frame.isna().sum().rename('missing').to_frame();missing['fraction']=missing['missing']/len(frame)
    missing.to_csv(OUT/'missingness.csv')
    stats=frame[selected].describe().T;stats.to_csv(OUT/'summary_feature_ranges.csv')
    receipt=dict(source='https://zenodo.org/records/21800730',doi='10.5281/zenodo.21800730',
        archive_md5=checksum,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_members_checked=True,waveform_count=len(names),rows=len(frame),columns=len(frame.columns),
        unique_patients=frame.patient_id.nunique(),license_file='CC BY 4.0 text included inside archive',
        outcome_missing=int(frame.neonatal_asphyxia.isna().sum()),
        positive_outcome_patients=frame.loc[frame.neonatal_asphyxia==1,'patient_id'].nunique(),
        inconsistent_patient_outcome=int((frame.groupby('patient_id').neonatal_asphyxia.nunique()>1).sum()),
        hospital_mapping_verified=False,models_trained=False,
        blockers=['F_STV/F_LTV names do not document equality to the hospital extractor',
            'Binary FIGO normal/abnormal cannot validate separate suspect/pathologic classes',
            'Rare neonatal outcome is distinct from morphology and UCI NSP',
            'Delivery and neonatal columns are unavailable at antenatal prediction time',
            'Gestational age is described at delivery; recording-time gestation must be established',
            'Single center; independent device and clinical validation still needed'])
    write_json(OUT/'source_audit.json',receipt)
    write_json(SOURCE/'provenance.json',receipt)
    sample=dict(record_id=Path(name).stem,FHR_zero_pct=float(100*(wave.FHR==0).mean()),
        UC_zero_pct=float(100*(wave.UC==0).mean()),sampling_hz=4,
        note='FHR zero masked; UC zero not automatically treated as missing')
    write_json(OUT/'waveform_sample_audit.json',sample)
    return support,stats,receipt,wave
