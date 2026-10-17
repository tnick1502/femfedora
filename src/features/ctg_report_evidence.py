"""Inspectable source and measurement evidence for notebook 08."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.ctg_report_features import REPORT_FIELDS, PROFILES, uci_profile, audit_hospital_report
from src.training.ctg_experiments import ROOT, CTG_FEATURES, load_data, write_json

OUT=ROOT/'research/experiments/ctg_uci/report_features_v1'
INPUTS=ROOT/'data/processed/ctg_uci/report_features_v1'
SOURCES={
    'uci': 'https://archive.ics.uci.edu/dataset/193/cardiotocography',
    'sisporto2_validation': 'https://www.sciencedirect.com/science/article/abs/pii/S030121150400212X',
    'sisporto_features_primary_study': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7516409/',
    'sonomed_manual': 'https://cordismed.ru/image/pdf/spektromed-sonomed-200-instrukcija.pdf',
    'mfmon_manual': 'https://usontec.com/uploads/s/9/2/c/92ckun9gej9h/file/raxxmfuv.pdf',
    'dawes_redman_manufacturer': 'https://www.huntleigh-diagnostics.com/wp-content/uploads/2025/07/749339EN-1.1-English-DAWES-REDMAN-CTG-ANALYSIS-Clinical-Application-Guide.pdf',
}


def example_report():
    """Manual transcription of the attached crop; no inferred device or label."""
    return dict(device=None,software_version=None,analysis_context=None,gestational_weeks=None,
        extractor_contract=None,event_exposure=None,
        provenance='Manual transcription of user attachment; device, gestation and context not verified; no clinical label',
        duration_min=17.,signal_loss_pct=8.5,baseline_bpm=131.4,
        accelerations_10_15_count=5,accelerations_15_15_count=1,
        decelerations_15_15_count=0,decelerations_over_20_lost_beats_count=0,
        high_variation_min=13.,low_variation_min=0.,sinusoidal_min=0.,
        oscillation_amplitude_ms=47.6,oscillation_amplitude_bpm=14.1,
        oscillation_frequency_per_min=2.2,stv_ms=7.6,stv_bpm=2.2,
        vstv_ms=3.7,vstv_bpm=1.1,movements_per_hour=7.1)


def mapping_table():
    rows=[
        ('Базальная ЧСС, BPM','LB','conditional','Единицы совпадают; алгоритм базальной линии и версия прибора требуют проверки'),
        ('Акцелерации >15 BPM, ≥15 s','AC','conditional','count/(60×duration_min); совпадения порога недостаточно для равенства детекторов'),
        ('Акцелерации >10 BPM, ≥15 s','нет','incompatible','Другой порог: не подставлять в AC и не складывать с >15'),
        ('Все децелерации >15 BPM, ≥15 s','DL+DS+DP','conditional_proxy','Сумма опубликованных категорий UCI; сверить взаимоисключение и правила детектора'),
        ('Децелерации >20 потерянных ударов','нет','incompatible','Площадь события, не амплитуда >20 BPM и не число DP'),
        ('STV, ms / BPM','MSTV','incompatible','Эпохальная STV и исходная SisPorto MSTV не взаимозаменяемы'),
        ('ВКИ, ms / BPM','нет','incompatible','В руководстве Сономед это vSTV от удара к удару; не доказано равенство UCI MSTV'),
        ('Минуты высокой / низкой вариабельности','ASTV / ALTV','incompatible','Эпизоды с иными правилами; длительность / время записи не восстанавливает ASTV/ALTV'),
        ('Амплитуда осцилляций','Width / MLTV','incompatible','Минутная амплитуда не ширина общей гистограммы; MLTV требует нативного определения'),
        ('Частота осцилляций, 1/min','Nmax / Nzeros','incompatible','Частота во времени не число пиков/нулей гистограммы'),
        ('Частота шевелений, 1/hour','FM','conditional_not_used','Деление на 3600 меняет единицы; ручные/автоматические отметки могут различаться'),
        ('Время анализа, min','e-b','audit_only','Деноминатор и точность частот; время сегмента восстановлено из XLS'),
        ('Потери сигнала, %','нет','report_only','Качество и маска доступности; для обучения такого столбца в UCI нет'),
        ('Синусоидальный ритм, min','нет','report_only','Нельзя восстанавливать через CLASS или NSP'),
        ('Галочки, баллы, «8 из 8», FIGO','CLASS / NSP','exclude_outcome','Итоговая оценка прибора не исходный предиктор и не независимая клиническая метка'),
        ('Срок беременности, контекст, устройство','нет','required_metadata','В UCI нет построчной проверки этих факторов; не выдумывать обучающие значения'),
    ]
    return pd.DataFrame(rows,columns=['hospital_field','UCI_candidate','status','reason'])


def prepare_evidence():
    for path in [OUT,INPUTS]:path.mkdir(parents=True,exist_ok=True)
    data,_=load_data()
    path=ROOT/'data/external/uci_cardiotocography/1.0.0/files/CTG.xls'
    raw=pd.read_excel(path,sheet_name='Raw Data',header=0)
    raw['source_excel_row']=raw.index+2
    raw=raw[raw.SegFile.notna()].copy()
    joined=data.merge(raw[['source_excel_row','FileName','SegFile','b','e','AC','FM','UC','DL','DS','DP']],
        on='source_excel_row',suffixes=('','_count'),validate='one_to_one')
    assert len(joined)==2126
    assert (joined.source_file==joined.FileName).all() and (joined.segment_file==joined.SegFile).all()
    duration=joined.e-joined.b
    assert (duration>0).all()
    reconstruction=[]
    for name in ['AC','FM','UC','DL','DS','DP']:
        error=np.abs(joined[name]-joined[name+'_count']/duration)
        assert error.max()<1e-12
        assert np.allclose(joined[name+'_count'],joined[name+'_count'].round())
        reconstruction.append(dict(feature=name,unit='events/second',denominator='e-b',
            rows=len(error),max_abs_error=float(error.max())))
    joined['duration_s']=duration
    exposure=joined[['source_excel_row','source_file','NSP','split','duration_s']+
                    [n+'_count' for n in ['AC','FM','UC','DL','DS','DP']]]
    exposure.to_parquet(INPUTS/'uci_exposure_audit.parquet',index=False)
    pd.DataFrame(reconstruction).to_csv(OUT/'event_rate_reconstruction.csv',index=False)
    descriptions={
        'LB':'Базальная линия SisPorto', 'AC':'Акцелерации / s','FM':'Шевеления / s',
        'UC':'Сокращения матки / s','DL':'Лёгкие децелерации / s','DS':'Тяжёлые децелерации / s',
        'DP':'Пролонгированные децелерации / s','ASTV':'% abnormal STV SisPorto',
        'MSTV':'Mean STV SisPorto (native definition)','ALTV':'% abnormal LTV SisPorto',
        'MLTV':'Mean LTV SisPorto (native definition)',
        'Width':'Ширина гистограммы = Max−Min','Min':'Минимум гистограммы',
        'Max':'Максимум гистограммы','Nmax':'Количество пиков гистограммы',
        'Nzeros':'Количество нулей гистограммы','Mode':'Мода гистограммы',
        'Mean':'Среднее гистограммы','Median':'Медиана гистограммы',
        'Variance':'Дисперсия гистограммы','Tendency':'Категория асимметрии −1/0/1'}
    train=data[data.split=='train'];dictionary=[]
    for name in CTG_FEATURES:
        dictionary.append(dict(feature=name,definition=descriptions[name],missing=int(data[name].isna().sum()),
            train_min=float(train[name].min()),train_median=float(train[name].median()),
            train_max=float(train[name].max()),train_zero_pct=float(100*(train[name]==0).mean()),
            train_nonzero_files=int(train.loc[train[name]!=0,'source_file'].nunique()),
            in_core2=name in ['LB','AC'],in_sisporto9=name in ['LB','AC','UC','DL','DP','ASTV','MSTV','ALTV','MLTV']))
    dictionary=pd.DataFrame(dictionary);dictionary.to_csv(OUT/'uci_feature_dictionary.csv',index=False)
    mapping=mapping_table();mapping.to_csv(OUT/'hospital_feature_mapping.csv',index=False)
    for profile in PROFILES:
        x=uci_profile(data,profile)
        x.insert(0,'source_excel_row',data.source_excel_row)
        x.to_parquet(INPUTS/f'{profile}_features.parquet',index=False)
    report=example_report();audit=audit_hospital_report(report)
    write_json(INPUTS/'example_report_transcribed.json',report)
    template={key:None for key in report};template['provenance']='Fill measurements with units in field names; no scores or conclusions'
    write_json(INPUTS/'hospital_report_template.json',template)
    pd.DataFrame([template]).to_csv(INPUTS/'hospital_report_template.csv',index=False)
    from src.features.ctg_pregnancy_app import hospital_report_schema
    write_json(INPUTS/'hospital_report_schema.json',hospital_report_schema())
    write_json(OUT/'example_compatibility.json',audit)
    write_json(OUT/'sources.json',dict(accessed='2026-10-17',sources=SOURCES,
        notes=['Manufacturer manuals are primary documents, not proof of the attached device identity',
            'Russian report layout resembles Sonomed/MFMon family; identity not inferred',
            'Later SisPorto studies explain related definitions but do not prove equality of software versions',
            'No hospital labels, no paired device-versus-SisPorto measurements available'],
        xls_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return data,exposure,dictionary,mapping,audit


def temporal_counterexample():
    slow=np.repeat([120.,140.],16);fast=np.tile([120.,140.],16)
    rows=[]
    for label,x in [('two_blocks',slow),('alternating',fast)]:
        rows.append(dict(sequence=label,mean=x.mean(),minimum=x.min(),maximum=x.max(),
            variance=x.var(),mean_abs_adjacent_bpm=np.abs(np.diff(x)).mean()))
    return pd.DataFrame(rows),slow,fast
