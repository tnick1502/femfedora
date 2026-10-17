"""Formula/event/missingness tests, not a claim of detector clinical accuracy."""
import numpy as np
import pytest

from src.features.jnu_ctg_report_extraction import (
    ExtractionConfig, baseline_curve, detect_events, extract_report_features,
    minute_measurements, episode_mask, five_of_six_mask)
from src.features.ctg_russian_reports import (
    EXAMPLE_TEXT, parse_report_text, normalize_report, engineer_report, report_profile_input)


def test_exact_epoch_formula_and_no_cross_minute_pair():
    epochs=np.tile([120.,150.],8)
    y=np.r_[np.repeat(epochs,15),np.repeat(epochs+10,15)]
    obs=np.ones(len(y),bool)
    import pandas as pd
    empty=pd.DataFrame(columns=['kind','peak_bpm','start_s','end_s'])
    minutes=minute_measurements(y,obs,~obs,np.full(len(y),130.),empty)
    assert minutes.iloc[0].stv_epoch_mean_rr_ms==pytest.approx(100.)
    assert minutes.iloc[0].stv_epoch_bpm==30
    assert minutes.iloc[0].ltv_point_range_ms==pytest.approx(100.)
    assert minutes.stv_eligible.all()


def test_missing_epoch_not_zero_and_deceleration_exclusion():
    import pandas as pd
    y=np.full(480,140.);obs=np.ones(480,bool);obs[30:45]=False;y[30:45]=np.nan
    event=pd.DataFrame([dict(kind='deceleration',peak_bpm=20,start_s=70,end_s=90)])
    m=minute_measurements(y,obs,np.zeros(480,bool),np.full(480,140.),event)
    assert not m.stv_eligible.any()
    assert m.iloc[0].ltv_eligible
    assert not m.iloc[1].ltv_eligible
    assert m.stv_epoch_mean_rr_ms.isna().all()


def test_four_second_epochs_have_separate_exact_formula():
    import pandas as pd
    y=np.repeat(np.resize([120.,150.],15),16)
    obs=np.ones(240,bool)
    m=minute_measurements(y,obs,~obs,np.full(240,130.),pd.DataFrame(columns=['kind','peak_bpm']))
    assert m.iloc[0].stv_4s_mean_rr_ms==pytest.approx(100.)
    assert m.iloc[0].stv_epoch_mean_rr_ms!=pytest.approx(100.)


def test_lost_beats_area_duration_and_peak_are_separate():
    y=np.full(1200,140.);y[240:960]=120.
    obs=np.ones(len(y),bool)
    events=detect_events(y,np.full(len(y),140.),obs,~obs)
    assert len(events)==1
    assert events.iloc[0].duration_s==180
    assert events.iloc[0].area_beats==60
    assert events.iloc[0].peak_bpm==20
    assert not events.iloc[0].censored


def test_event_duration_is_excursion_not_threshold_dwell():
    y=np.full(480,140.);y[120:200]=145.;y[160]=156.
    obs=np.ones(len(y),bool)
    e=detect_events(y,np.full(len(y),140.),obs,~obs)
    assert len(e)==1 and e.iloc[0].duration_s==20 and e.iloc[0].peak_bpm==16


def test_gaps_and_edges_censor_incomplete_events():
    y=np.full(720,140.);y[:100]=160.;y[300:500]=120.;y[390:430]=np.nan
    obs=np.isfinite(y)
    e=detect_events(y,np.full(len(y),140.),obs,np.zeros(len(y),bool))
    assert len(e)==3 and e.censored.all()


def test_no_bridge_long_gaps_no_boundary_fill():
    y=np.full(2400,140.);y[600:680]=0.;y[1000:1008]=0.;y[:4]=0
    b,s,obs,filled,clean=baseline_curve(y)
    assert not filled[600:680].any() and np.isnan(b[600:680]).all()
    assert filled[1000:1008].all() and np.isfinite(clean[1000:1008]).all()
    assert not filled[:4].any()


def test_flat_signal_zero_events_is_not_all_missing():
    flat=extract_report_features(np.full(7200,140.))
    missing=extract_report_features(np.zeros(7200))
    assert flat['acc15_count']==0 and flat['stv_ms']==0
    assert flat['baseline_bpm']==140
    assert np.isnan(missing['acc15_count']) and np.isnan(missing['stv_ms'])
    assert missing['signal_loss_pct']==100 and not missing['report_quality_eligible']
    for key in ['movements_per_hour','vstv_ms','sinusoidal_min','psp']:
        assert np.isnan(flat[key])


def test_minute_loss_boundary_is_explicit():
    import pandas as pd
    y=np.full(240,140.);obs=np.ones(240,bool);obs[:60]=False
    m=minute_measurements(y,obs,~obs,np.full(240,140.),pd.DataFrame(columns=['kind','peak_bpm']))
    assert not m.iloc[0].ltv_eligible  # exactly 25% loss


def test_episode_definitions_differ_and_gaps_break_runs():
    assert episode_mask([1,1,0,1,1,1,0],3).sum()==3
    assert episode_mask([1,1,1,1,0,1],5).sum()==0
    assert five_of_six_mask(np.array([1,1,0,1,1,1]),np.ones(6,bool)).sum()==6
    assert five_of_six_mask(np.ones(6,bool),np.array([1,1,0,1,1,1])).sum()==0


def test_parse_example_without_reference_limits_or_score():
    p=parse_report_text(EXAMPLE_TEXT)
    assert len(p['measurements'])==18
    assert p['measurements']['stv_ms']==7.6
    assert p['measurements']['acc15_count']==1
    assert p['measurements']['dec20lost_count']==0
    assert p['measurements']['movements_per_hour']==7.1
    assert p['requires_confirmation'] and len(p['unparsed_lines'])==2
    x,a=engineer_report(p['measurements'])
    assert x.iloc[0].acc15_per_10min==pytest.approx(10/17)
    assert x.iloc[0].nominal_nonlost_min==pytest.approx(15.555)
    assert np.isnan(x.iloc[0].small_acc10_15_count)  # nesting unknown
    assert a['clinical_risk'] is None and a['vendor_bridge_validated'] is False


def test_missing_units_conclusions_and_conflicts():
    p=normalize_report({'duration_min':{'value':1020,'unit':'s'},'stv_ms':'/','fisher_score':8})
    assert p['values']['duration_min']==17 and p['missing']['stv_ms']
    assert p['excluded_conclusions']=={'fisher_score':8}
    assert parse_report_text('Частота шевелений 2 1/мин')['measurements']['movements_per_hour']==120
    assert parse_report_text('Время анализа 1020 с')['measurements']['duration_min']==17
    assert not parse_report_text('Потери сигнала 10')['measurements']
    assert parse_report_text('STV 7 мс\nSTV 9 мс')['measurements']['stv_ms'] is None
    with pytest.raises(ValueError):normalize_report({'stv_ms':{'value':7,'unit':'bpm'}})
    with pytest.raises(ValueError):normalize_report({'duration_min':0})
    with pytest.raises(ValueError):normalize_report({'acc15_count':1.2})
    with pytest.raises(ValueError):normalize_report({'signal_loss_pct':101})


def test_inclusive_acceleration_categories_only_when_explicit():
    x,_=engineer_report({'duration_min':20,'acc10_count':5,'acc15_count':2,
        'definitions':{'acc10_category':'inclusive_ge10'}})
    assert x.iloc[0].small_acc10_15_count==3
    with pytest.raises(ValueError):normalize_report({'acc10_count':1,'acc15_count':2,
        'definitions':{'acc10_category':'inclusive_ge10'}})


def test_definition_gated_model_input_does_not_alias_unknown_stv():
    p=parse_report_text(EXAMPLE_TEXT)['measurements']
    x,audit=report_profile_input(p,'sonomed_fields')
    assert np.isnan(x.iloc[0].stv_ms) and np.isnan(x.iloc[0].sonomed_rule_high_fraction)
    assert audit['values']['stv_ms']==7.6 and not audit['inference_authorized']
    assert np.isnan(x.iloc[0].acc10_per_10min)
    p['definitions']={'stv_epoch_s':4}
    x,audit=report_profile_input(p,'mak_fields')
    assert x.iloc[0].stv_4s_ms==7.6


def test_native_report_profile_roundtrip():
    from src.features.jnu_ctg_report_extraction import PROFILE_FIELDS,VERSION,profile_frame
    import pandas as pd
    t=np.arange(7200)/4
    f=extract_report_features(140+3*np.sin(2*np.pi*t/13))
    for profile in PROFILE_FIELDS:
        report={k:f[k] for k in ['duration_min','signal_loss_pct','baseline_bpm','acc10_count',
            'acc15_count','dec15_count','dec20lost_count','amplitude_bpm','amplitude_ms','stv_bpm']}
        report['stv_ms']=f['stv_4s_ms'] if profile=='mak_fields' else f['stv_ms']
        rule='mfmon_rule' if profile=='mfmon_fields' else 'sonomed_rule'
        report.update(high_variation_min=f[f'{rule}_high_min'],low_variation_min=f[f'{rule}_low_min'])
        report['definitions']={'measurement_contract':VERSION,'stv_epoch_s':4 if profile=='mak_fields' else 3.75,
            'acc10_category':'inclusive_ge10','variation_episode_rule':'5consecutive_32ms' if profile=='mfmon_fields' else '3consecutive_11bpm'}
        actual,_=report_profile_input(report,profile)
        expected=profile_frame(pd.DataFrame([f]),profile)
        np.testing.assert_allclose(actual,expected,equal_nan=True)
