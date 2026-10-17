"""Explicit feature contracts for CTG summaries; vendor metrics are not aliases.

These contracts describe research inputs. They do not establish clinical or
cross-device validity. Missing fields are never replaced with a normal value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


PROFILES = {
    'core2': ('baseline_bpm', 'accelerations_per_s'),
    'common3': ('baseline_bpm', 'accelerations_per_s', 'decelerations_per_s'),
    'sisporto9': ('baseline_bpm', 'accelerations_per_s', 'contractions_per_s',
        'light_decelerations_per_s', 'prolonged_decelerations_per_s',
        'abnormal_stv_pct', 'mean_stv_sisporto', 'abnormal_ltv_pct', 'mean_ltv_sisporto'),
}
ALIASES = {
    'baseline_bpm': 'LB', 'accelerations_per_s': 'AC', 'contractions_per_s': 'UC',
    'light_decelerations_per_s': 'DL', 'prolonged_decelerations_per_s': 'DP',
    'abnormal_stv_pct': 'ASTV', 'mean_stv_sisporto': 'MSTV',
    'abnormal_ltv_pct': 'ALTV', 'mean_ltv_sisporto': 'MLTV',
}


def uci_profile(frame, profile):
    """Training-only bridge from official UCI columns, not a hospital adapter.

    Sum DL+DS+DP is the explicit total-event proxy under the mutually exclusive
    SisPorto 2 duration-category interpretation. It is not a count of lost beats.
    """
    result = pd.DataFrame(index=frame.index)
    for name in PROFILES[profile]:
        result[name] = (frame[['DL', 'DS', 'DP']].sum(axis=1) if name == 'decelerations_per_s'
                        else frame[ALIASES[name]])
    return validate_profile(result, profile)


def validate_profile(frame, profile):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Pass a named DataFrame with explicit units')
    expected = list(PROFILES[profile])
    if set(frame.columns) != set(expected) or frame.columns.duplicated().any():
        raise ValueError(f'Expected exactly {expected}; missing/extra fields are not imputed')
    x = frame[expected].copy().astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Missing/non-finite input: a missing measurement is not zero')
    if (x < 0).any().any() or (x.baseline_bpm <= 0).any():
        raise ValueError('Negative measurements or non-positive baseline')
    for name in x:
        if name.endswith('_pct') and (x[name] > 100).any():
            raise ValueError(f'{name} must be a percentage in [0,100]')
    return x


class ReportFeatures(BaseEstimator, TransformerMixin):
    """Stateless engineering. All learned scaling lives inside the fitted model.

    1200 seconds is a reference exposure, not a claim of a 20-minute recording.
    The 0.5 rate pseudocount stabilizes a descriptor; it is not an observed event.
    No medical decision threshold or report conclusion is encoded as a predictor.
    """
    def __init__(self, profile='common3', engineered=False):
        self.profile, self.engineered = profile, engineered

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(PROFILES[self.profile], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.transform(X)
        return self

    def transform(self, X):
        x = validate_profile(X, self.profile)
        if not self.engineered:
            return x
        x['baseline_rr_ms'] = 60000 / x.baseline_bpm
        for name in list(x):
            if name.endswith('_per_s'):
                x[name.replace('_per_s', '_log_events_per_20min')] = np.log1p(1200*x[name])
                x[name.replace('_per_s', '_present')] = (x[name] > 0).astype(float)
        if 'decelerations_per_s' in x:
            a, d = x.accelerations_per_s*1200, x.decelerations_per_s*1200
            x['decel_accel_log_ratio'] = np.log((d+.5)/(a+.5))
            x['event_balance'] = (a-d)/(a+d+1.)
        if self.profile == 'sisporto9':
            x['abnormal_stv_fraction'] = x.abnormal_stv_pct/100
            x['abnormal_ltv_fraction'] = x.abnormal_ltv_pct/100
            x['abnormal_variability_product'] = x.abnormal_stv_fraction*x.abnormal_ltv_fraction
            x['relative_mean_stv'] = x.mean_stv_sisporto/x.baseline_bpm
            x['relative_mean_ltv'] = x.mean_ltv_sisporto/x.baseline_bpm
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError('Non-finite engineered feature')
        return x

    def get_feature_names_out(self, input_features=None):
        sample = pd.DataFrame([{name: (130. if name == 'baseline_bpm' else 0.)
                                for name in PROFILES[self.profile]}])
        return np.array(self.transform(sample).columns, dtype=object)


REPORT_FIELDS = {
    'duration_min', 'signal_loss_pct', 'baseline_bpm', 'accelerations_10_15_count',
    'accelerations_15_15_count', 'decelerations_15_15_count',
    'decelerations_over_20_lost_beats_count', 'high_variation_min', 'low_variation_min',
    'sinusoidal_min', 'oscillation_amplitude_ms', 'oscillation_amplitude_bpm',
    'oscillation_frequency_per_min', 'stv_ms', 'stv_bpm', 'vstv_ms', 'vstv_bpm',
    'movements_per_hour',
}
META_FIELDS = {'device', 'software_version', 'analysis_context', 'gestational_weeks',
               'extractor_contract', 'event_exposure', 'provenance'}


def audit_hospital_report(report):
    """Parse measurements and return explicit missingness/compatibility status.

    Readable values and unit conversions are available even when inference is
    blocked. Unknown device/definition is not repaired by setting a boolean flag.
    """
    extras = set(report) - REPORT_FIELDS - META_FIELDS
    if extras:
        raise ValueError(f'Unsupported fields, possibly conclusions: {sorted(extras)}')
    values = {}
    for key in REPORT_FIELDS:
        value = report.get(key)
        values[key] = None if value is None else float(value)
        if values[key] is not None and (not np.isfinite(values[key]) or values[key] < 0):
            raise ValueError(f'Invalid value for {key}')
    duration = values['duration_min']
    if duration is None or duration <= 0:
        raise ValueError('Positive analysis duration in minutes is required')
    if values['baseline_bpm'] is not None and values['baseline_bpm'] <= 0:
        raise ValueError('Baseline must be positive')
    loss = values['signal_loss_pct']
    if loss is not None and loss > 100:
        raise ValueError('Signal loss must be a percentage')
    for name in REPORT_FIELDS:
        val = values[name]
        if name.endswith('_count') and val is not None and not val.is_integer():
            raise ValueError(f'{name} is an event count, not an event rate')
        if name in ['high_variation_min', 'low_variation_min', 'sinusoidal_min'] and val is not None and val > duration:
            raise ValueError(f'{name} exceeds analysis duration')
    hi, lo = values['high_variation_min'], values['low_variation_min']
    if hi is not None and lo is not None and hi+lo > duration+1e-8:
        raise ValueError('High/low variation durations overlap or have incompatible denominators')
    derived = {'duration_s': 60*duration}
    for raw, canonical in [('accelerations_15_15_count','accelerations_per_s'),
                            ('decelerations_15_15_count','decelerations_per_s')]:
        derived[canonical] = (None if values[raw] is None else values[raw]/(60*duration))
    for key in ['high_variation_min','low_variation_min','sinusoidal_min']:
        derived[key.replace('_min','_fraction')] = None if values[key] is None else values[key]/duration
    derived['nominal_nonlost_min'] = None if loss is None else duration*(1-loss/100)
    lb = values['baseline_bpm']
    derived['baseline_rr_ms'] = None if lb is None else 60000/lb
    derived['stv_bpm_linear_unit_check'] = (None if lb is None or values['stv_ms'] is None
                                           else values['stv_ms']*lb**2/60000)
    issues = []
    for key in ['device','software_version','analysis_context','gestational_weeks']:
        if report.get(key) in [None,'','unknown']:
            issues.append(f'Unverified metadata: {key}')
    # Only a native, version-matched extractor contract can pass. Other devices
    # require a paired measurement bridge + external labeled validation first.
    if report.get('extractor_contract') != 'UCI_SisPorto2_native_events_v1':
        issues.append('No verified measurement bridge to UCI SisPorto 2')
    if report.get('event_exposure') != 'wall_time':
        issues.append('Event denominator (wall-time vs valid signal) is unverified')
    for key in ['baseline_bpm','accelerations_15_15_count','decelerations_15_15_count']:
        if values[key] is None:
            issues.append(f'Missing core measurement: {key}')
    return dict(status='blocked_incompatible_measurements' if issues else 'native_research_only',
                measurements=values, derived=derived,
                missing_mask={key: value is None for key,value in values.items()},
                issues=issues, clinical_validation=False)


def report_to_common3(report):
    audit = audit_hospital_report(report)
    if audit['issues']:
        raise ValueError('; '.join(audit['issues']))
    return pd.DataFrame([{'baseline_bpm': audit['measurements']['baseline_bpm'],
        'accelerations_per_s': audit['derived']['accelerations_per_s'],
        'decelerations_per_s': audit['derived']['decelerations_per_s']}])
