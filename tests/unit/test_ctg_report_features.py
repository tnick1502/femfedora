import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.ctg_report_features import (ReportFeatures, uci_profile,
    audit_hospital_report, report_to_common3, validate_profile)
from src.features.ctg_report_evidence import example_report, temporal_counterexample
from src.training.ctg_report_models import SummaryModel,SummaryBundle
from src.training.ctg_audit import ProbabilityCalibrator


def test_report_units_and_missing_are_not_zero():
    report=example_report();audit=audit_hospital_report(report)
    assert audit['derived']['accelerations_per_s']==pytest.approx(1/1020)
    assert audit['derived']['high_variation_fraction']==pytest.approx(13/17)
    assert audit['measurements']['decelerations_15_15_count']==0
    assert not audit['missing_mask']['decelerations_15_15_count']
    report['decelerations_15_15_count']=None
    missing=audit_hospital_report(report)
    assert missing['derived']['decelerations_per_s'] is None
    assert missing['missing_mask']['decelerations_15_15_count']
    with pytest.raises(ValueError,match='Missing core'):report_to_common3(report)


def test_no_stv_alias_or_outcome_input():
    report=example_report()
    assert audit_hospital_report(report)['status']=='blocked_incompatible_measurements'
    with pytest.raises(ValueError,match='measurement bridge'):report_to_common3(report)
    report['NSP']=1
    with pytest.raises(ValueError,match='conclusions'):audit_hospital_report(report)
    x=pd.DataFrame([dict(baseline_bpm=130,accelerations_per_s=.001,decelerations_per_s=0,stv_ms=7)])
    with pytest.raises(ValueError,match='exactly'):validate_profile(x,'common3')


def test_count_and_duration_errors_rejected():
    for key,value in [('duration_min',0),('signal_loss_pct',101),
                      ('accelerations_15_15_count',.2),('high_variation_min',18)]:
        report=example_report();report[key]=value
        with pytest.raises(ValueError):audit_hospital_report(report)


def test_engineering_batch_independent_and_roundtrip(tmp_path):
    x=pd.DataFrame(dict(baseline_bpm=[120,135,150]*6,
                       accelerations_per_s=[0,.001,.01]*6,
                       decelerations_per_s=[.004,0,.001]*6))
    transformer=ReportFeatures('common3',True).fit(x)
    assert np.allclose(transformer.transform(x).iloc[:1],transformer.transform(x.iloc[:1]))
    y=np.tile([0,1,2],6)
    candidate=dict(family='logistic',engineered=True,params=dict(C=1.,class_weight=None))
    model=SummaryModel('common3',candidate).fit(x,y)
    scaler=model.models_[0].named_steps['scale']
    expected=np.median(transformer.transform(x),axis=0)
    np.testing.assert_allclose(scaler.center_,expected)
    hold=x.copy();hold['baseline_bpm']+=500
    center=scaler.center_.copy();model.predict_proba(hold)
    np.testing.assert_array_equal(scaler.center_,center)
    cal=ProbabilityCalibrator('temperature').fit(model.predict_proba(x),y)
    bundle=SummaryBundle([model],cal,'common3')
    path=tmp_path/'bundle.joblib';joblib.dump(bundle,path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(x),bundle.predict_proba(x),rtol=0,atol=1e-12)
    assert bundle.support_report(hold).out_of_training_range.all()
    broken=x.copy();broken.iloc[0,1]=np.nan
    with pytest.raises(ValueError,match='missing measurement'):bundle.predict_proba(broken)


def test_temporal_information_cannot_be_reconstructed_from_histogram():
    table,_,_=temporal_counterexample()
    np.testing.assert_array_equal(table.iloc[0,1:5],table.iloc[1,1:5])
    assert table.iloc[0].mean_abs_adjacent_bpm != table.iloc[1].mean_abs_adjacent_bpm


def test_pregnancy_app_can_store_unknown_device_without_inventing_a_risk():
    from src.features.ctg_pregnancy_app import prepare_pregnancy_app_record
    record=prepare_pregnancy_app_record(example_report())
    assert record['storage_status']=='accepted_with_unknowns'
    assert record['record']['device'] is None
    assert record['clinical_risk'] is None and not record['model_prediction_eligible']
    partial=prepare_pregnancy_app_record({'baseline_bpm':130,'stv_ms':7.6})
    assert partial['derived']=={} and partial['missing_mask']['duration_min']
    with pytest.raises(ValueError):prepare_pregnancy_app_record({'baseline_bpm':0})
    with pytest.raises(ValueError):prepare_pregnancy_app_record({'accelerations_15_15_count':True})
    with pytest.raises(ValueError):prepare_pregnancy_app_record({'device':False})
    from src.features.ctg_pregnancy_app import hospital_report_schema
    import jsonschema
    jsonschema.validate(example_report(),hospital_report_schema())
    jsonschema.validate({'baseline_bpm':130,'stv_ms':7.6},hospital_report_schema())
