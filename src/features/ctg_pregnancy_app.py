"""Device-neutral storage adapter for an antenatal pregnancy-tracking app.

Accepting a report for storage does not authorize model inference. The original
measurements, units, unknown metadata and missing masks remain inspectable.
"""
from __future__ import annotations
import numpy as np
from src.features.ctg_report_features import REPORT_FIELDS,META_FIELDS,audit_hospital_report


def hospital_report_schema():
    properties={}
    for key in sorted(REPORT_FIELDS):
        properties[key]={'type':['integer' if key.endswith('_count') else 'number','null'],'minimum':0}
        if key=='signal_loss_pct':properties[key]['maximum']=100
        if key in ['baseline_bpm','duration_min']:properties[key]['exclusiveMinimum']=0
    for key in sorted(META_FIELDS):properties[key]={'type':['string','null']}
    properties['gestational_weeks']={'type':['number','null'],'exclusiveMinimum':0,'maximum':50}
    properties['analysis_context']={'enum':['antenatal','intrapartum','unknown',None]}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema',
        'title':'CTG antenatal report measurements — storage, not clinical inference',
        'type':'object','additionalProperties':False,'properties':properties}


def prepare_pregnancy_app_record(record):
    extras=set(record)-REPORT_FIELDS-META_FIELDS
    if extras:raise ValueError(f'Unsupported measurement fields: {sorted(extras)}')
    canonical={key:record.get(key) for key in sorted(REPORT_FIELDS|META_FIELDS)}
    for key in META_FIELDS-{'gestational_weeks'}:
        if canonical[key] is not None and not isinstance(canonical[key],str):
            raise ValueError(f'{key}: string or null required')
    for key in REPORT_FIELDS:
        value=canonical[key]
        if value is not None:
            if isinstance(value,bool):raise ValueError(f'{key}: boolean is not a measurement')
            try:value=float(value)
            except (ValueError,TypeError) as exc:raise ValueError(f'{key}: numeric value required') from exc
            if not np.isfinite(value) or value<0:raise ValueError(f'{key}: nonnegative finite value required')
            if key.endswith('_count') and not value.is_integer():raise ValueError(f'{key}: integer count required')
            if key=='signal_loss_pct' and value>100:raise ValueError('Signal loss cannot exceed 100 percent')
            if key in ['baseline_bpm','duration_min'] and value<=0:raise ValueError(f'{key}: positive value required')
            canonical[key]=value
    context=canonical['analysis_context']
    if context not in [None,'unknown','antenatal','intrapartum']:
        raise ValueError('analysis_context must be antenatal, intrapartum or unknown')
    gestation=canonical['gestational_weeks']
    if gestation is not None:
        gestation=float(gestation)
        if not np.isfinite(gestation) or gestation<=0 or gestation>50:raise ValueError('Invalid gestational_weeks')
        canonical['gestational_weeks']=gestation
    missing={key:canonical[key] is None for key in REPORT_FIELDS}
    audit=None
    if canonical['duration_min'] is not None:audit=audit_hospital_report(canonical)
    reasons=[]
    if context!='antenatal':reasons.append('Для дородового сценария нужно подтвердить контекст записи')
    if gestation is None:reasons.append('Срок беременности не указан')
    if canonical['device'] in [None,'','unknown']:reasons.append('Аппарат не указан; показатели сохранены с исходными единицами')
    if audit is None:reasons.append('Не указана длительность анализа; частоты из счётчиков не рассчитаны')
    elif audit['issues']:reasons.append('Совместимость измерений с обученной моделью не подтверждена')
    return dict(schema_version='ctg_antenatal_report_storage_v1',intended_context='antenatal',
        record=canonical,missing_mask=missing,derived={} if audit is None else audit['derived'],
        storage_status='accepted_with_unknowns' if reasons else 'accepted',notices=reasons,
        model_prediction_eligible=False,clinical_risk=None,inference_status='external_antenatal_validation_required')
