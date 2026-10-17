"""Leakage boundaries, grouped weighting, probability and export contracts."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.training.jnu_report_models import (patient_weights,training_weights,checked_frame,
    ReportBasis,ReportClassifier,SigmoidCalibration,ReportBundle,choose_research_threshold)


def fixture_frame(n=160):
    rng=np.random.default_rng(417)
    x=pd.DataFrame({'baseline_bpm':rng.uniform(110,170,n),'acc15_per_10min':rng.integers(0,7,n)/3,
        'dec15_per_10min':rng.integers(0,4,n)/3})
    y=((x.baseline_bpm>145)|(x.dec15_per_10min>2/3)).to_numpy(int)
    return x,y,np.repeat(np.arange(n//2),2)


def test_each_patient_has_equal_total_weight_and_class_weight_is_explicit():
    patients=[1,2,2,2,3,3];y=[0,0,1,1,1,1]
    w=patient_weights(patients)
    np.testing.assert_allclose(pd.Series(w).groupby(patients).sum(),1)
    w2=training_weights(y,patients,'Balanced')
    assert w2.mean()==pytest.approx(1)
    assert w2[np.array(y)==0].sum()==pytest.approx(w2[np.array(y)==1].sum())


def test_schema_units_masks_and_column_order():
    x,_,_=fixture_frame()
    np.testing.assert_array_equal(checked_frame(x[x.columns[::-1]],'minimal'),x)
    with pytest.raises(ValueError):checked_frame(x.assign(figo_label=0),'minimal')
    bad=x.copy();bad.iloc[0,0]=np.inf
    with pytest.raises(ValueError):checked_frame(bad,'minimal')
    bad=x.copy();bad.iloc[0,1]=-1
    with pytest.raises(ValueError):checked_frame(bad,'minimal')


def test_basis_fit_is_unchanged_by_holdout_and_new_missing_columns_are_masked():
    x,_,_=fixture_frame();basis=ReportBasis('minimal',5,True).fit(x.iloc[:100])
    expected=x.iloc[:100].median().to_numpy()
    np.testing.assert_allclose(basis.medians_,expected)
    probe=x.iloc[100:].copy();probe.iloc[0,:]=np.nan
    transformed=basis.transform(probe)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[0,-3:],1)
    np.testing.assert_allclose(basis.medians_,expected)


def test_zero_heavy_quantiles_constant_and_empty_features_do_not_break_splines():
    x,_,_=fixture_frame();x['acc15_per_10min']=0.;x.iloc[-1,1]=5.
    x['dec15_per_10min']=np.nan
    basis=ReportBasis('minimal',7,True).fit(x)
    assert np.isfinite(basis.transform(x)).all()
    assert basis.medians_[2]==0


@pytest.mark.parametrize('family,params',[
    ('catboost',dict(iterations=20,depth=3,learning_rate=.1,l2_leaf_reg=3,auto_class_weights='Balanced')),
    ('spline_logit',dict(C=.3,n_knots=5,interactions=True,weight_policy='Balanced')),
    ('linear',dict(C=.3,weight_policy=None))])
def test_model_roundtrip_finite_probabilities_and_no_metadata_in_X(tmp_path,family,params):
    x,y,patients=fixture_frame()
    model=ReportClassifier('minimal',family,params).fit(x,y,patients,np.arange(len(y)),patients)
    probe=x.iloc[:7].copy();probe.iloc[0,:]=np.nan
    p=model.predict_positive(probe)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    target=tmp_path/'model.joblib';joblib.dump(model,target)
    np.testing.assert_array_equal(p,joblib.load(target).predict_positive(probe))


def test_exact_catboost_tree_prefix_reuse():
    x,y,patients=fixture_frame()
    params=dict(depth=3,learning_rate=.03,l2_leaf_reg=10,auto_class_weights=None)
    a=ReportClassifier('minimal','catboost',{**params,'iterations':20},seed=81).fit(x,y,patients)
    b=ReportClassifier('minimal','catboost',{**params,'iterations':40},seed=81).fit(x,y,patients)
    np.testing.assert_allclose(a.predict_positive(x),b.predict_positive(x,ntree_end=20),rtol=0,atol=1e-14)


def test_calibration_is_monotone_and_preserves_probability_contract():
    x,y,patients=fixture_frame()
    p=np.linspace(.02,.98,len(y))
    cal=SigmoidCalibration().fit(p,y,patients)
    transformed=cal.transform(p)
    assert cal.slope_>0 and (np.diff(transformed)>0).all()
    assert np.isfinite(transformed).all()


def test_research_threshold_matches_bruteforce_and_handles_probability_ties():
    y=np.array([0,1,0,1,1,0]);p=np.array([.2,.2,.3,.6,.6,.8]);patients=np.arange(6)
    t=choose_research_threshold(y,p,patients)
    brute=max(f1_score(y,p>=v,average='macro') for v in np.unique(p))
    assert f1_score(y,p>=t,average='macro')==pytest.approx(brute)


def test_unvalidated_hospital_report_never_returns_clinical_risk():
    bundle=ReportBundle('minimal','catboost',[],None)
    audit=bundle.audit_report({'duration_min':17,'baseline_bpm':131.4,'acc15_count':1,'dec15_count':0})
    assert audit['clinical_risk'] is None and not audit['supported_for_clinical_inference']
    assert any('30 min' in m for m in audit['issues'])
