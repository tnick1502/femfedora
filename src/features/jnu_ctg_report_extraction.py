"""Auditable research measurements for report-only CTG modelling.

An independent operational implementation, NOT a reproduction of a vendor's
medical software. Every computed measurement depends on the versioned baseline,
event, missingness and minute rules below. No fitted parameters or labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from src.features.jnu_ctg_preparation import bounded_interpolation, runs

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/processed/jnu_ctg/v1'
DEST = ROOT / 'data/processed/jnu_ctg/report_features_v1'
OUT = ROOT / 'research/experiments/jnu_ctg/report_features_v1'
VERSION = 'JNU_report_operational_1.0'


@dataclass(frozen=True)
class ExtractionConfig:
    fs: int = 4
    baseline_window_s: int = 300
    baseline_method: str = 'median_trim'
    max_internal_gap_s: float = 2.
    min_baseline_segment_s: int = 60
    event_smoothing_s: int = 3
    return_band_bpm: float = 2.
    minute_min_observed_fraction: float = .75  # STRICT > .75
    epoch_min_observed_fraction: float = .8
    min_report_observed_fraction: float = .8  # engineering cohort, not clinical


def mean_or_nan(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def baseline_curve(fhr, config=ExtractionConfig()):
    """1 Hz medians -> centred rolling median -> optional twice trimmed mean.

    Each contiguous usable segment is independent. No smoothing across a long
    gap, no edge extrapolation. Baselines for segments <60 s remain unknown.
    This deliberately simple baseline must be benchmarked against annotations.
    """
    x = np.asarray(fhr, float)
    if config.fs != 4 or len(x) % config.fs:
        raise ValueError('Version 1 requires 4 Hz and a whole number of seconds')
    if config.baseline_method not in ('median', 'median_trim'):
        raise ValueError('Unknown baseline method')
    if config.baseline_window_s < config.min_baseline_segment_s:
        raise ValueError('Baseline window shorter than required support')
    observed = np.isfinite(x) & (x >= 30) & (x <= 240)
    clean, filled = bounded_interpolation(x, observed, int(config.fs * config.max_internal_gap_s))
    used = observed | filled
    baseline = np.full(len(x), np.nan)
    smoothed = np.full(len(x), np.nan)
    # Process at original resolution before downsampling: a missing quarter
    # second must not disappear into a valid one-second aggregate.
    for a, b in runs(used):
        y = clean[a:b]
        smoothed[a:b] = pd.Series(y).rolling(config.fs * config.event_smoothing_s + 1,
            center=True, min_periods=1).median().to_numpy()
        seconds = len(y) // config.fs
        if seconds < config.min_baseline_segment_s:
            continue
        y1 = np.median(y[:seconds * config.fs].reshape(-1, config.fs), axis=1)
        series = pd.Series(y1)
        window = config.baseline_window_s + 1
        base = series.rolling(window, center=True, min_periods=config.min_baseline_segment_s).median()
        if config.baseline_method == 'median_trim':
            for _ in range(2):
                stable = series.where((series - base).abs() <= 10.)
                proposed = stable.rolling(window, center=True,
                    min_periods=config.min_baseline_segment_s).mean()
                base = proposed.fillna(base)
        times = np.arange(len(y)) / config.fs
        # Constant extension is within an observed segment, never into a gap.
        baseline[a:b] = np.interp(times, np.arange(seconds), base.to_numpy())
    return baseline, smoothed, observed, filled, clean


EVENT_COLUMNS = ['kind', 'start_s', 'end_s', 'duration_s', 'peak_bpm',
                 'area_beats', 'censored', 'observed_fraction', 'filled_fraction']


def detect_events(fhr, baseline, observed, filled, config=ExtractionConfig()):
    """Excursions outside a +/-2 bpm return band; peak >=10, duration >=15 s.

    Duration measures the excursion, NOT time spent above the peak threshold.
    A gap or recording boundary censors a candidate; it is not counted as a
    complete event. No stitching across gaps. Both nested >=10 and >=15 counts
    are available; an exclusive 10..15 category is also saved explicitly.
    """
    fhr, baseline = np.asarray(fhr), np.asarray(baseline)
    valid = np.isfinite(fhr) & np.isfinite(baseline) & (observed | filled)
    rows = []
    for kind, sign in [('acceleration', 1), ('deceleration', -1)]:
        residual = sign * (fhr - baseline)
        for a, b in runs(valid & (residual > config.return_band_bpm)):
            peak = float(np.max(residual[a:b]))
            duration = (b-a) / config.fs
            if peak < 10 or duration < 15:
                continue
            censored = a == 0 or b == len(fhr) or not valid[a-1] or not valid[b]
            rows.append(dict(kind=kind, start_s=a/config.fs, end_s=b/config.fs,
                duration_s=duration, peak_bpm=peak,
                area_beats=float(np.maximum(residual[a:b], 0).sum() / (60 * config.fs)),
                censored=bool(censored), observed_fraction=float(observed[a:b].mean()),
                filled_fraction=float(filled[a:b].mean())))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def episode_mask(condition, minimum_run):
    answer = np.zeros(len(condition), dtype=bool)
    for a, b in runs(np.asarray(condition, dtype=bool)):
        if b-a >= minimum_run:
            answer[a:b] = True
    return answer


def five_of_six_mask(condition, eligible):
    """Union of complete eligible six-minute windows with >=5 high minutes.

    Only an operational primitive; no gestation-dependent confirmation.
    """
    answer = np.zeros(len(condition), dtype=bool)
    for a in range(len(condition)-5):
        if eligible[a:a+6].all() and np.sum(condition[a:a+6]) >= 5:
            answer[a:a+6] = True
    return answer


def minute_measurements(clean, observed, filled, baseline, events, config=ExtractionConfig()):
    """No inter-minute epoch pair; never fill missing epoch by zero.

    STV requires 16 epochs of >=80% ORIGINAL observations, all working samples
    available after bounded fill, >75% original minute coverage, no known or
    censored >=15 bpm deceleration. LTV uses original observed points only.
    """
    samples = config.fs * 60
    exclusion = np.zeros(len(clean), bool)
    for e in events.itertuples():
        if e.kind == 'deceleration' and e.peak_bpm >= 15:
            exclusion[int(e.start_s*config.fs):int(e.end_s*config.fs)] = True
    rows = []
    for a in range(0, len(clean)-samples+1, samples):
        b = a+samples
        obs = observed[a:b]
        eligible = (obs.mean() > config.minute_min_observed_fraction
            and not exclusion[a:b].any() and np.isfinite(baseline[a:b]).all())
        row = dict(minute=a/samples, observed_fraction=float(obs.mean()),
            deceleration_excluded=bool(exclusion[a:b].any()), ltv_eligible=bool(eligible))
        y = clean[a:b][obs]
        rr = 60000/y
        row['ltv_point_range_bpm'] = float(np.ptp(y)) if eligible else np.nan
        row['ltv_point_range_ms'] = float(np.ptp(rr)) if eligible else np.nan
        y_epoch = clean[a:b].reshape(16, 15)
        obs_epoch = obs.reshape(16, 15)
        complete = eligible and np.isfinite(y_epoch).all() and (obs_epoch.mean(axis=1) >= config.epoch_min_observed_fraction).all()
        row['stv_eligible'] = bool(complete)
        for k in ['stv_epoch_bpm', 'stv_epoch_mean_rr_ms', 'stv_rr_of_mean_bpm_ms', 'ltv_epoch_range_ms', 'stv_4s_mean_rr_ms']:
            row[k] = np.nan
        if complete:
            epoch_bpm = y_epoch.mean(axis=1)
            epoch_rr = (60000/y_epoch).mean(axis=1)
            row['stv_epoch_bpm'] = float(np.abs(np.diff(epoch_bpm)).mean())
            row['stv_epoch_mean_rr_ms'] = float(np.abs(np.diff(epoch_rr)).mean())
            row['stv_rr_of_mean_bpm_ms'] = float(np.abs(np.diff(60000/epoch_bpm)).mean())
            row['ltv_epoch_range_ms'] = float(np.ptp(epoch_rr))
        # A separate 4-second recipe is required for the MAK manual description.
        # Do not overwrite the 3.75 s measurement or use its epoch support mask.
        epoch4 = clean[a:b].reshape(15, 16)
        obs4 = obs.reshape(15, 16)
        complete4 = eligible and np.isfinite(epoch4).all() and (obs4.mean(axis=1)>=config.epoch_min_observed_fraction).all()
        if complete4:
            row['stv_4s_mean_rr_ms'] = float(np.abs(np.diff((60000/epoch4).mean(axis=1))).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def oscillation_diagnostics(clean, observed, baseline, minute, config=ExtractionConfig()):
    """Research proxies, not vendor oscillation frequency or sinusoidal diagnosis."""
    crossing, cycles, peakfreq, concentration = [], [], [], []
    for row in minute.itertuples():
        a = int(row.minute * 60 * config.fs); b = a + 60*config.fs
        if not row.stv_eligible:
            continue
        residual = clean[a:b] - baseline[a:b]
        smooth = pd.Series(residual).rolling(13, center=True, min_periods=1).mean().to_numpy()
        # Schmitt trigger eliminates zero chatter; count BOTH crossing directions.
        state = 0; transitions = 0
        for val in smooth[::config.fs]:
            new = 1 if val > 1 else -1 if val < -1 else state
            if state and new != state:
                transitions += 1
            state = new
        crossing.append(transitions); cycles.append(transitions/2.)
        freq, power = periodogram(smooth, fs=config.fs, detrend='linear')
        band = (freq >= .5/60) & (freq <= 10/60)
        if power[band].sum() > 0:
            peakfreq.append(float(freq[band][np.argmax(power[band])]*60))
            concentration.append(float(power[(freq>=3/60)&(freq<=5/60)].sum()/power[band].sum()))
    return dict(osc_crossings_hysteresis_per_min=mean_or_nan(crossing),
        osc_cycles_hysteresis_per_min=mean_or_nan(cycles),
        osc_spectral_peak_per_min=mean_or_nan(peakfreq),
        narrowband_3_5_power_fraction=mean_or_nan(concentration))


def extract_report_features(wave, config=ExtractionConfig(), diagnostics=False):
    wave = np.asarray(wave)
    fhr = wave[:, 0] if wave.ndim == 2 else wave
    if len(fhr) < 240 or len(fhr) % 4:
        raise ValueError('Need at least one minute at 4 Hz')
    base, smooth, observed, filled, clean = baseline_curve(fhr, config)
    events = detect_events(smooth, base, observed, filled, config)
    minute = minute_measurements(clean, observed, filled, base, events, config)
    valid_base = np.isfinite(base) & observed
    exposure_min = len(fhr)/(4*60)
    f = dict(duration_min=exposure_min, signal_loss_pct=100*(1-observed.mean()),
        source_loss_pct=100*(1-(np.isfinite(fhr)&(fhr>0)).mean()),
        observed_min=observed.sum()/240, filled_fraction=filled.mean(),
        baseline_coverage_fraction=valid_base.mean(), baseline_bpm=mean_or_nan(base[valid_base]),
        baseline_p05_bpm=float(np.quantile(base[valid_base], .05)) if valid_base.any() else np.nan,
        baseline_p95_bpm=float(np.quantile(base[valid_base], .95)) if valid_base.any() else np.nan,
        adjacent_jump_gt25_fraction=float(((np.abs(np.diff(fhr))>25)&observed[1:]&observed[:-1]).sum()/max(1,(observed[1:]&observed[:-1]).sum())),
        event_censored_candidates=int(events.censored.sum()) if len(events) else 0,
        eligible_ltv_min=int(minute.ltv_eligible.sum()), eligible_stv_min=int(minute.stv_eligible.sum()))
    complete = events[~events.censored.astype(bool)]
    has_exposure = valid_base.sum() >= config.min_baseline_segment_s * config.fs
    subsets = {
        'acc10': complete[(complete.kind=='acceleration') & (complete.peak_bpm>=10)],
        'acc15': complete[(complete.kind=='acceleration') & (complete.peak_bpm>=15)],
        'acc10_15exclusive': complete[(complete.kind=='acceleration') & (complete.peak_bpm>=10) & (complete.peak_bpm<15)],
        'dec15': complete[(complete.kind=='deceleration') & (complete.peak_bpm>=15)],
    }
    for name, table in subsets.items():
        f[f'{name}_count'] = float(len(table)) if has_exposure else np.nan
        f[f'{name}_per_10min'] = len(table)/exposure_min*10 if has_exposure else np.nan
        f[f'{name}_duration_s'] = float(table.duration_s.sum()) if has_exposure else np.nan
        f[f'{name}_area_beats'] = float(table.area_beats.sum()) if has_exposure else np.nan
        f[f'{name}_max_peak_bpm'] = float(table.peak_bpm.max()) if len(table) else np.nan
    dec = subsets['dec15']
    f['dec20lost_count'] = float((dec.area_beats>=20).sum()) if has_exposure else np.nan
    f['dec20lost_per_10min'] = f['dec20lost_count']/exposure_min*10
    for seconds in [120, 180, 300]:
        f[f'dec_duration_ge{seconds}s_count'] = float((dec.duration_s>=seconds).sum()) if has_exposure else np.nan
    for key, col in [('stv_ms','stv_epoch_mean_rr_ms'), ('stv_bpm','stv_epoch_bpm'), ('stv_4s_ms','stv_4s_mean_rr_ms'),
                     ('stv_rr_of_mean_bpm_ms','stv_rr_of_mean_bpm_ms'),
                     ('amplitude_bpm','ltv_point_range_bpm'), ('amplitude_ms','ltv_point_range_ms'),
                     ('ltv_epoch_range_ms','ltv_epoch_range_ms')]:
        f[key] = mean_or_nan(minute[col])
    eligible = minute.ltv_eligible.to_numpy(bool)
    ltv_bpm = minute.ltv_point_range_bpm.to_numpy()
    ltv_ms = minute.ltv_point_range_ms.to_numpy()
    for name, high, low, run in [
        ('sonomed_rule', eligible & (ltv_bpm>=11), eligible & (ltv_bpm<11), 3),
        ('mfmon_rule', eligible & (ltv_ms>=32), eligible & (ltv_ms<32), 5)]:
        for kind, mask in [('high',high), ('low',low)]:
            f[f'{name}_{kind}_min'] = float(episode_mask(mask,run).sum()) if eligible.any() else np.nan
    f['dr_5of6_32ms_candidate_min'] = float(five_of_six_mask(ltv_ms>32,eligible).sum()) if eligible.any() else np.nan
    # Fixed documented thresholds, with a fallback described in the MAK manual.
    # LTV handling of long accelerations is not a complete vendor reproduction.
    high_threshold,low_threshold=32.,30.
    if len(minute)>=20 and not five_of_six_mask(ltv_ms[:20]>32,eligible[:20]).any():
        high_threshold,low_threshold=24.,22.
    f['mak_high_threshold_ms']=high_threshold
    f['mak_rule_candidate_high_min']=float(five_of_six_mask(ltv_ms>high_threshold,eligible).sum()) if eligible.any() else np.nan
    f['mak_rule_candidate_low_min']=float(five_of_six_mask(ltv_ms<low_threshold,eligible).sum()) if eligible.any() else np.nan
    f.update(oscillation_diagnostics(clean,observed,base,minute,config))
    # These cannot be recovered from 4 Hz FHR and external UC.
    for key in ['movements_per_hour','vstv_ms','sinusoidal_min','psp','fisher_score','krebs_score','dr_criteria_met']:
        f[key] = np.nan
    f['report_quality_eligible'] = bool(observed.mean() >= config.min_report_observed_fraction
        and f['baseline_coverage_fraction'] >= config.min_report_observed_fraction and f['eligible_stv_min']>=5)
    if diagnostics:
        return f, events, minute, dict(baseline=base,smoothed=smooth,observed=observed,filled=filled,clean=clean)
    return f


# These sets describe numbers potentially present on a paper report. They are
# research input profiles; a vendor name is never a claim of vendor equivalence.
PROFILE_FIELDS = {
    'minimal': ['baseline_bpm','acc15_per_10min','dec15_per_10min'],
    'variability': ['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm'],
    'sonomed_fields': ['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm',
        'duration_min','signal_loss_pct','acc10_per_10min','dec20lost_per_10min','amplitude_ms','stv_bpm',
        'sonomed_rule_high_fraction','sonomed_rule_low_fraction'],
    'mfmon_fields': ['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm',
        'duration_min','signal_loss_pct','dec20lost_per_10min','mfmon_rule_high_fraction','mfmon_rule_low_fraction'],
    'mak_fields': ['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_4s_ms','amplitude_ms',
        'duration_min','signal_loss_pct','dec20lost_per_10min'],
}


def profile_frame(frame, profile):
    if profile not in PROFILE_FIELDS:
        raise ValueError('Unknown profile')
    x = frame.copy()
    for prefix in ['sonomed_rule','mfmon_rule']:
        for kind in ['high','low']:
            key = f'{prefix}_{kind}_fraction'
            if key in PROFILE_FIELDS[profile] and key not in x:
                x[key] = x[f'{prefix}_{kind}_min']/x.duration_min
    return x[PROFILE_FIELDS[profile]].astype(float)


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def extract_dataset(force=False):
    """Pure transforms of fixed train/validation/test; no test score inspected."""
    DEST.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    source_receipt = json.loads((ROOT/'research/experiments/jnu_ctg/preparation_v1/manifest.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(dict(version=VERSION,config=asdict(ExtractionConfig()),
        code=sha256(Path(__file__)),source=source_receipt),sort_keys=True).encode()).hexdigest()
    stamp = OUT/'extraction_manifest.json'
    if not force and stamp.exists() and json.loads(stamp.read_text())['fingerprint']==fingerprint:
        saved=json.loads(stamp.read_text())
        if all((ROOT/p).exists() and sha256(ROOT/p)==h for p,h in saved['artifacts_sha256'].items()):
            return pd.read_parquet(DEST/'measurements.parquet')
    signals=np.load(SOURCE/'signals_original_float32.npy',mmap_mode='r')
    index=pd.read_parquet(SOURCE/'signal_index.parquet')
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    rows=[]
    for i,row in enumerate(index.itertuples(index=False)):
        rows.append(dict(record_id=row.record_id,**extract_report_features(signals[row.signal_row])))
        if (i+1)%2000==0:
            print(f'Извлечено {i+1}/{len(index)}',flush=True)
    measurements=pd.DataFrame(rows)
    measurements.to_parquet(DEST/'measurements.parquet',index=False)
    # Target-free files for future fold-local pipelines; targets remain separate.
    for profile in PROFILE_FIELDS:
        x=profile_frame(measurements,profile)
        x.insert(0,'record_id',measurements.record_id)
        for name in ['train','validation','test']:
            part=splits.loc[splits.split==name,['record_id']].merge(x,on='record_id',validate='one_to_one')
            part.to_parquet(DEST/f'X_{profile}_{name}.parquet',index=False)
    schema=[]
    for col in measurements.columns[1:]:
        profiles=[p for p,cols in PROFILE_FIELDS.items() if col in cols]
        unit=('count' if col.endswith('_count') else 'bpm' if col.endswith('_bpm') or col in ['baseline_bpm','stv_bpm','amplitude_bpm']
            else 'ms' if col.endswith('_ms') else 's' if col.endswith('_s') else 'min' if col.endswith('_min')
            else 'events/10min' if col.endswith('_per_10min') else 'fraction' if col.endswith('_fraction')
            else '%' if col.endswith('_pct') else 'beats' if col.endswith('_beats') else 'see methods')
        schema.append(dict(feature=col,unit=unit,report_profiles=';'.join(profiles),
            role='report_candidate' if profiles else 'unavailable_or_diagnostic',
            vendor_equivalence_verified=False))
    pd.DataFrame(schema).to_csv(DEST/'measurement_dictionary.csv',index=False)
    owned=[DEST/'measurements.parquet',DEST/'measurement_dictionary.csv']
    owned += [DEST/f'X_{p}_{s}.parquet' for p in PROFILE_FIELDS for s in ['train','validation','test']]
    receipt=dict(version=VERSION,fingerprint=fingerprint,config=asdict(ExtractionConfig()),
        records=len(measurements),labels_used=False,test_used_for_selection=False,
        clinical_validation=False,vendor_equivalence_verified=False,
        artifacts_sha256={str(p.relative_to(ROOT)):sha256(p) for p in owned})
    stamp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return measurements
