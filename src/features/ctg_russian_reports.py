"""Numbers/text on Russian CTG summaries -> explicit, missing-aware inputs.

No OCR engine, no diagnosis and no reconstruction of a waveform from aggregates.
Text input may be transcribed/OCR'd, but every parsed value retains its source
line and must be confirmed before medical use. Definitions are caller metadata.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


FIELDS = {
    'duration_min':'min', 'signal_loss_pct':'%', 'baseline_bpm':'bpm',
    'acc10_count':'count', 'acc15_count':'count', 'dec15_count':'count',
    'dec20lost_count':'count', 'high_variation_min':'min', 'low_variation_min':'min',
    'sinusoidal_min':'min', 'amplitude_bpm':'bpm', 'amplitude_ms':'ms',
    'oscillation_frequency_per_min':'1/min', 'stv_ms':'ms', 'stv_bpm':'bpm',
    'vstv_ms':'ms', 'vstv_bpm':'bpm', 'movements_per_hour':'1/h',
    'contractions_count':'count', 'fhr_min_bpm':'bpm', 'fhr_max_bpm':'bpm',
}
CONCLUSIONS = {'psp','fisher_score','krebs_score','figo','nst','dr_criteria_met',
               'assessment','doctor_conclusion','parameters_normal'}
META = {'device','software_version','analysis_context','gestational_weeks',
        'definitions','source','confirmed','analysis_window','event_exposure'}
MISSING = {'','/','—','–','-','нет данных','не определено','неизвестно','nan','none'}


def number(value):
    if value is None:
        return np.nan
    if isinstance(value,str):
        if value.strip().lower() in MISSING:
            return np.nan
        value=value.strip().replace(',','.')
    if isinstance(value,(bool,np.bool_)):
        raise ValueError('Boolean is not a measurement')
    answer=float(value)
    if np.isnan(answer):
        return np.nan
    if not np.isfinite(answer):
        raise ValueError('Infinite measurement')
    return answer


def normalize_report(report):
    """Canonical units only; structured values may explicitly state other units.

    Counts remain counts. Rates can be derived only with a known duration.
    Unknown units/definitions and conclusions never become predictor numbers.
    """
    extras=set(report)-set(FIELDS)-META-CONCLUSIONS
    if extras:
        raise ValueError(f'Unknown fields: {sorted(extras)}')
    values={}; issues=[]
    for field,unit in FIELDS.items():
        raw=report.get(field)
        if isinstance(raw,dict):
            got_unit=raw.get('unit',unit)
            val=number(raw.get('value'))
            if got_unit==unit:
                pass
            elif unit=='min' and got_unit=='s':
                val/=60
            elif unit=='1/h' and got_unit=='1/min':
                val*=60
            else:
                raise ValueError(f'Incompatible units for {field}: {got_unit}, expected {unit}')
        else:
            val=number(raw)
        if np.isfinite(val) and val<0:
            raise ValueError(f'Negative {field}')
        if unit=='count' and np.isfinite(val) and not val.is_integer():
            raise ValueError(f'Event count must be integer: {field}')
        values[field]=val
    duration=values['duration_min']; loss=values['signal_loss_pct']
    if np.isfinite(duration) and duration<=0:
        raise ValueError('Duration must be positive')
    if np.isfinite(loss) and loss>100:
        raise ValueError('Loss percentage >100')
    if np.isfinite(values['baseline_bpm']) and values['baseline_bpm']<=0:
        raise ValueError('Baseline must be positive')
    for field in ['high_variation_min','low_variation_min','sinusoidal_min']:
        if np.isfinite(duration) and values[field]>duration+1e-8:
            raise ValueError(f'{field} exceeds duration')
    if np.isfinite(duration) and values['high_variation_min']+values['low_variation_min']>duration+1e-8:
        issues.append('High+low variation exceeds duration: verify rounding and denominators')
    definitions=report.get('definitions',{})
    if definitions.get('acc10_category')=='inclusive_ge10' and values['acc15_count']>values['acc10_count']:
        raise ValueError('Nested acceleration counts are inconsistent')
    if values['dec20lost_count']>values['dec15_count']:
        issues.append('Significant decelerations exceed dec15: verify event definitions')
    lo,hi=values['fhr_min_bpm'],values['fhr_max_bpm']
    if np.isfinite(lo) and np.isfinite(hi) and lo>hi:
        raise ValueError('Minimum exceeds maximum FHR')
    for key in ['device','software_version','analysis_context']:
        if report.get(key) in [None,'','unknown']:
            issues.append(f'Unknown {key}')
    if report.get('analysis_context') not in [None,'','unknown','antenatal']:
        issues.append('Outside intended antenatal application context')
    if report.get('confirmed') is not True:
        issues.append('Report transcription not confirmed')
    if not definitions:
        issues.append('Measurement algorithms unverified')
    if not np.isfinite(duration):
        issues.append('Missing duration: event rates cannot be computed')
    if not np.isfinite(loss):
        issues.append('Signal quality unknown')
    # Unit agreement is only a numerical plausibility check, not a conversion.
    lb=values['baseline_bpm']
    for prefix in ['stv','vstv']:
        ms,bpm=values[f'{prefix}_ms'],values[f'{prefix}_bpm']
        if np.isfinite(ms) and np.isfinite(bpm) and np.isfinite(lb):
            approx=ms*lb**2/60000
            if abs(approx-bpm)>.3+abs(bpm)*.2:
                issues.append(f'{prefix}: ms/bpm pair requires unit/algorithm review')
    return dict(values=values, missing={k:bool(np.isnan(v)) for k,v in values.items()},
        issues=issues, excluded_conclusions={k:report[k] for k in CONCLUSIONS if k in report},
        metadata={k:report.get(k) for k in META}, clinical_risk=None,
        vendor_bridge_validated=False, status='research_measurements_only')


def engineer_report(report):
    """Derive information algebraically available on the report, without fitting.

    Missing stays NaN. These transforms do not add independent information and
    must be ablated against the original values in later nested CV.
    """
    parsed=normalize_report(report)
    x=parsed['values'].copy(); duration=x['duration_min']; lb=x['baseline_bpm']
    for name in ['acc10','acc15','dec15','dec20lost','contractions']:
        x[f'{name}_per_10min']=x[f'{name}_count']/duration*10
    for kind in ['high','low','sinusoidal']:
        x[f'{kind}_variation_fraction' if kind!='sinusoidal' else 'sinusoidal_fraction']=x[f'{kind}_variation_min' if kind!='sinusoidal' else 'sinusoidal_min']/duration
    x['baseline_rr_ms']=60000/lb if np.isfinite(lb) else np.nan
    x['nominal_nonlost_min']=duration*(1-x['signal_loss_pct']/100)
    # Do not divide event counts by nominal_nonlost_min: event exposure rules
    # differ by device, and this is not known valid event observation time.
    x['amplitude_relative_to_baseline']=x['amplitude_bpm']/lb
    x['stv_relative_to_rr']=x['stv_ms']/x['baseline_rr_ms']
    x['stv_to_ltv_ratio']=x['stv_ms']/x['amplitude_ms'] if x['amplitude_ms']>0 else np.nan
    # Only genuine nested categories permit subtraction.
    definitions=report.get('definitions',{})
    if definitions.get('acc10_category')=='inclusive_ge10':
        x['small_acc10_15_count']=x['acc10_count']-x['acc15_count']
    elif definitions.get('acc10_category')=='exclusive_10_to15':
        x['small_acc10_15_count']=x['acc10_count']
    else:
        x['small_acc10_15_count']=np.nan
    for name in ['acc10_per_10min','acc15_per_10min','dec15_per_10min','dec20lost_per_10min']:
        x[f'log1p_{name}']=float(np.log1p(x[name]))
    # Mean/max/dynamics/entropy cannot be calculated from these aggregates.
    return pd.DataFrame([x]), parsed


def report_profile_input(report, profile):
    """Produce the same named columns as research X; retain semantic blockers.

    Unknown-definition STV/episode values remain available in the audit, but
    cannot silently become a version-matched predictor. This function prepares
    research input; it neither loads a classifier nor returns medical risk.
    """
    from src.features.jnu_ctg_report_extraction import PROFILE_FIELDS, VERSION
    if profile not in PROFILE_FIELDS:
        raise ValueError('Unknown profile')
    x,audit=engineer_report(report)
    definitions=report.get('definitions',{})
    epoch=definitions.get('stv_epoch_s')
    if profile=='mak_fields':
        x['stv_4s_ms']=x['stv_ms'] if epoch==4 else np.nan
        if epoch!=4:audit['issues'].append('MAK research profile requires confirmed 4-second STV recipe')
    elif 'stv_ms' in PROFILE_FIELDS[profile] and epoch!=3.75:
        x['stv_ms']=np.nan;x['stv_bpm']=np.nan
        audit['issues'].append('Research profile requires confirmed 3.75-second STV recipe')
    for prefix,rule in [('sonomed_rule','3consecutive_11bpm'),('mfmon_rule','5consecutive_32ms')]:
        for kind in ['high','low']:
            if f'{prefix}_{kind}_fraction' in PROFILE_FIELDS[profile]:
                x[f'{prefix}_{kind}_fraction']=(x[f'{kind}_variation_fraction']
                    if definitions.get('variation_episode_rule')==rule else np.nan)
                if definitions.get('variation_episode_rule')!=rule:
                    audit['issues'].append(f'Unverified {prefix} {kind} episode definition')
    if profile=='sonomed_fields' and definitions.get('acc10_category')!='inclusive_ge10':
        x['acc10_per_10min']=np.nan
        audit['issues'].append('Research AC10 field is inclusive >=10; report category unverified')
    if definitions.get('measurement_contract')!=VERSION:
        audit['issues'].append('No verified measurement bridge to the research extractor')
    selected=x.reindex(columns=PROFILE_FIELDS[profile])
    audit['profile']=profile
    audit['profile_missing_fields']=selected.columns[selected.iloc[0].isna()].tolist()
    audit['inference_authorized']=False
    return selected,audit


def _numeric_after_label(tail):
    tail=re.sub(r'\([^)]*\)','',tail).replace('✓','').replace('√','').strip(' :;=\t')
    match=re.match(r'([+-]?\d+(?:[.,]\d+)?)',tail)
    return match.group(1) if match else None


def parse_report_text(text):
    """Conservative Russian-label parser, preserving rejected/ambiguous lines.

    The parser is intentionally line-based; it must not capture reference
    intervals, a score or the next line as a measurement. No image OCR claimed.
    """
    result={}; evidence=[]; unparsed=[]; conflicts=[]
    rules=[
        (r'(?:время анализа|длительность (?:анализа|записи))','duration_min'),
        (r'потер[ия] сигнала','signal_loss_pct'),
        (r'(?:базальн\w* (?:чсс|ритм)|бчсс)','baseline_bpm'),
        (r'высокая вариабельность','high_variation_min'),
        (r'низкая вариабельность','low_variation_min'),
        (r'синусоидальн\w* ритм','sinusoidal_min'),
        (r'частота осцилляций','oscillation_frequency_per_min'),
        (r'частота шевелений','movements_per_hour'),
        (r'количество схваток','contractions_count'),
    ]
    for line in text.splitlines():
        original=line.strip()
        if not original:
            continue
        low=original.lower().replace('ё','е'); found={}
        event=re.search(r'(акцелераци\w*|децелераци\w*)\s*[>≥]\s*(10|15|20)\s*(.*)',low)
        if event:
            kind,threshold,rest=event.groups()
            if kind.startswith('дец') and threshold=='20' and re.match(r'ударов',rest):
                v=_numeric_after_label(re.sub(r'^ударов','',rest))
                if v is not None:found['dec20lost_count']=v
            elif threshold in ['10','15']:
                tail=re.sub(r'^(?:bpm|врм|уд/?мин)\s*(?:и|,|/)\s*15\s*(?:s|с|сек)\.?','',rest).strip()
                # Require the duration/units to have matched; unknown event
                # wording is left unparsed instead of guessing a count.
                if tail!=rest.strip():
                    v=_numeric_after_label(tail)
                    if v is not None:
                        key=f'acc{threshold}_count' if kind.startswith('акц') else 'dec15_count' if threshold=='15' else None
                        if key:found[key]=v
        elif re.match(r'\s*(stv|квв|вки|vstv|амплитуда осцилляций)\b',low):
            prefix='amplitude' if low.startswith('амплитуда') else 'vstv' if re.match(r'(вки|vstv)',low) else 'stv'
            tail=re.sub(r'^(stv|квв|вки|vstv|амплитуда осцилляций)\s*','',low)
            tail=re.sub(r'\([^)]*\)','',tail).replace('✓','')
            for unit,pat in [('ms',r'мс|ms'),('bpm',r'bpm|врм|уд/мин')]:
                match=re.search(r'(?<![\d.,])([+-]?\d+(?:[.,]\d+)?)\s*(?:'+pat+r')\b',tail)
                if match:found[f'{prefix}_{unit}']=match.group(1)
        else:
            for pattern,key in rules:
                match=re.match(pattern,low)
                if match:
                    tail=low[match.end():]
                    v=_numeric_after_label(tail)
                    if v is not None:
                        # Parse only units the line actually supports; a rate
                        # per minute must not silently become a rate per hour.
                        unit_ok=True; factor=1.
                        if key=='movements_per_hour':
                            if re.search(r'(?:1/мин|/min)',tail):factor=60.
                            elif not re.search(r'(?:1/час|/ч\b|/h\b)',tail):unit_ok=False
                        elif key=='oscillation_frequency_per_min':
                            unit_ok=bool(re.search(r'(?:1/мин|/мин|/min)',tail))
                        elif key=='signal_loss_pct':unit_ok='%' in tail
                        elif key=='duration_min' or key.endswith('_variation_min') or key=='sinusoidal_min':
                            if re.search(r'\b(?:сек|с|s)\b',tail):factor=1/60
                            elif not re.search(r'\b(?:мин|min)\b',tail):unit_ok=False
                        elif key=='baseline_bpm':unit_ok=bool(re.search(r'bpm|врм|уд/мин',tail))
                        if unit_ok:found[key]=number(v)*factor
                    break
        if found:
            for key,val in found.items():
                val=number(val)
                if key in result and result[key]!=val:
                    conflicts.append(dict(field=key,previous=result[key],new=val,line=original))
                    result[key]=None  # Never silently choose conflicting OCR.
                elif not any(c['field']==key for c in conflicts):
                    result[key]=val
                evidence.append(dict(field=key,value=val,line=original))
        else:
            unparsed.append(original)
    return dict(measurements=result,evidence=evidence,unparsed_lines=unparsed,
        conflicts=conflicts,requires_confirmation=True,ocr_accuracy_evaluated=False)


EXAMPLE_TEXT = '''Время анализа 17,0 (10–60) мин
Потери сигнала 8,5 %
Базальная ЧСС 131,4 (115–160) BPM
Акцелерации >10 BPM и 15 с 5
Акцелерации >15 BPM и 15 с ✓ 1 (≥1)
Децелерации >15 BPM и 15 с 0
Децелерации >20 ударов ✓ 0 (≤0)
Высокая вариабельность ✓ 13 (≥3) мин
Низкая вариабельность 0 мин
Синусоидальный ритм ✓ 0 (≤0) мин
Амплитуда осцилляций 47,6 мс; 14,1 BPM
Частота осцилляций 2,2 1/мин
STV ✓ 7,6 (≥4) мс; 2,2 BPM
ВКИ 3,7 мс; 1,1 BPM
Частота шевелений ✓ 7,1 (≥6) 1/час
Оценка КТГ плода 1 (Пользовательская)
8 из 8 параметров в норме'''
