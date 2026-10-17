"""Single application-facing CTG report contract with abstention before models.

Historical SisPorto feature importers are benchmark adapters, not app parsers.
No model in this repository has a validated bridge to a hospital device.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.features.ctg_russian_reports import parse_report_text,normalize_report,engineer_report

CONTRACT='JNU_report_single_unit_2.0'
PROFILES={
    'core3':['baseline_bpm','acc15_per_10min','dec15_per_10min'],
    'report5':['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm'],
    'report7':['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm','acc10_per_10min','dec20lost_per_10min'],
    'report4s5':['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_4s_ms','amplitude_bpm'],
}
# Empty deliberately: a caller-provided string is not external validation.
VALIDATED_APPLICATION_BRIDGES={}


def parse_ctg_report(text):
    """Conservative text parser; no OCR and no inference from a printed conclusion."""
    return parse_report_text(text)


def prepare_report(report,profile):
    if profile not in PROFILES:raise ValueError('Unknown report profile')
    x,audit=engineer_report(report);definitions=report.get('definitions',{});blockers=[]
    if 'stv_4s_ms' in PROFILES[profile]:x['stv_4s_ms']=x.stv_ms
    selected=x.reindex(columns=PROFILES[profile])
    missing=selected.columns[~np.isfinite(selected.iloc[0].to_numpy(float))].tolist()
    if missing:blockers.append('missing_required_measurements:'+','.join(missing))
    if report.get('confirmed') is not True:blockers.append('transcription_unconfirmed')
    if report.get('analysis_context')!='antenatal':blockers.append('antenatal_context_unconfirmed')
    if not np.isfinite(x.duration_min.iloc[0]) or abs(x.duration_min.iloc[0]-30)>1e-8:blockers.append('duration_not_validated_30min')
    if definitions.get('measurement_contract')!=CONTRACT:blockers.append('measurement_bridge_unconfirmed')
    if 'stv_ms' in selected and definitions.get('stv_epoch_s')!=3.75:blockers.append('stv_3_75s_recipe_unconfirmed')
    if 'stv_4s_ms' in selected and definitions.get('stv_epoch_s')!=4:blockers.append('stv_4s_recipe_unconfirmed')
    if 'amplitude_bpm' in selected and definitions.get('amplitude_recipe')!='observed_minute_range_including_accelerations':blockers.append('amplitude_recipe_unconfirmed')
    if 'acc10_per_10min' in selected and definitions.get('acc10_category')!='inclusive_ge10':blockers.append('acc10_category_unconfirmed')
    if definitions.get('event_exposure')!='full_recording_duration':blockers.append('event_denominator_unconfirmed')
    # Unknown definitions stay visible in the audit, never become NaN sent to a model.
    audit.update(profile=profile,required_fields=PROFILES[profile],missing_required=missing,
        blockers=blockers,contract=CONTRACT,clinical_risk=None,inference_authorized=False)
    return selected,audit


def evaluate_report(report,bundle,profile,*,mode='application'):
    """Return no probability for blocked input; model is not called in that case."""
    if mode not in ['application','research']:raise ValueError('Unknown mode')
    x,audit=prepare_report(report,profile)
    blockers=list(audit['blockers'])
    if mode=='application':blockers.append('no_externally_validated_application_bridge')
    result=dict(status='abstained',profile=profile,reasons=blockers,
        probability_expert_abnormal=None,clinical_risk=None,supported_for_clinical_inference=False)
    if blockers:return result
    probability=float(bundle.predict_proba(x)[0,1])
    result.update(status='research_prediction',probability_expert_abnormal=probability)
    return result


def partial_dr_checklist(report):
    """Incomplete Philips D/R-2002 requirements, not a Dawes-Redman implementation.

    Source: Philips application note 4522 991 20871, Aug 2016, page 3.
    A summary lacks GA centiles, terminal epochs/artifacts and event severity.
    Therefore overall criteria_met and clinical_risk are never inferred here.
    """
    parsed=normalize_report(report);v=parsed['values'];d=report.get('definitions',{})
    verified=d.get('clinical_stv_definition')=='Philips_Dawes_Redman_2002'
    stv=v['stv_ms'];rows=[]
    rows.append(dict(requirement='STV >3 ms in the specified implementation',
        status=('not_evaluable' if not verified or not np.isfinite(stv) else 'met' if stv>3 else 'not_met')))
    for requirement in ['Gestation-dependent high-variation LTV centiles',
        'Verified high-variation episode and sinusoidal assessment',
        'Movement/acceleration combination with verified event definitions',
        'Deceleration area categories including >100 lost beats',
        'Terminal epoch deceleration and artifact checks']:
        rows.append(dict(requirement=requirement,status='not_evaluable'))
    return dict(status='incomplete_requirements_only',criteria_met=None,clinical_risk=None,
        requirements=rows,source='https://www.philips.com/c-dam/b2bhc/master/healthcare/room-layout/mri/CTG_analysis_based_on_Dawes_Redman_criteria.pdf')


def research_patterns():
    """Plausibility probes, not synthetic clinically validated gold labels."""
    common=dict(duration_min=30,signal_loss_pct=5,baseline_bpm=135,acc10_count=0,dec20lost_count=0,
        confirmed=True,analysis_context='antenatal',definitions={'measurement_contract':CONTRACT,
        'stv_epoch_s':3.75,'amplitude_recipe':'observed_minute_range_including_accelerations',
        'acc10_category':'inclusive_ge10','event_exposure':'full_recording_duration'})
    return {name:{**common,**fields} for name,fields in {
        'quiet':dict(stv_ms=3.5,amplitude_bpm=6,acc15_count=0,dec15_count=0),
        'reactive':dict(stv_ms=9,amplitude_bpm=25,acc15_count=9,acc10_count=12,dec15_count=0),
        'high_variation':dict(stv_ms=17,amplitude_bpm=36,acc15_count=0,dec15_count=0),
        'decelerative':dict(stv_ms=5,amplitude_bpm=12,acc15_count=0,dec15_count=9,dec20lost_count=3),
    }.items()}
