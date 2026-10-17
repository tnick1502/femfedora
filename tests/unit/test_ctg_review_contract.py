import copy
import numpy as np
import pandas as pd
import pytest
import joblib
from src.features.ctg_report_contract import (PROFILES,prepare_report,evaluate_report,research_patterns,partial_dr_checklist)
from src.training.ctg_review_models import StrictBinaryModel,weights,select_one_se,FixedTemperature
from src.training.ctg_review_evidence import prefix_proxy


class Spy:
    def __init__(self):self.calls=0
    def predict_proba(self,X):self.calls+=1;return np.array([[.6,.4]])


def test_application_never_calls_research_model_even_with_claimed_contract():
    spy=Spy();result=evaluate_report(research_patterns()['reactive'],spy,'report5')
    assert spy.calls==0 and result['probability_expert_abnormal'] is None
    assert result['clinical_risk'] is None and result['status']=='abstained'


@pytest.mark.parametrize('change',[{'stv_ms':None},{'duration_min':17},{'confirmed':False},{'definitions':{}}])
def test_required_missing_or_unverified_measurement_abstains_before_prediction(change):
    report={**research_patterns()['reactive'],**change};spy=Spy()
    result=evaluate_report(report,spy,'report5',mode='research')
    assert spy.calls==0 and result['status']=='abstained' and result['probability_expert_abnormal'] is None


def test_zero_accelerations_are_observed_zero_not_missing_or_automatic_diagnosis():
    spy=Spy();report=research_patterns()['quiet'];result=evaluate_report(report,spy,'report5',mode='research')
    assert spy.calls==1 and result['probability_expert_abnormal']==pytest.approx(.4)
    assert result['clinical_risk'] is None


def test_low_stv_summary_does_not_fabricate_full_dawes_redman_classification():
    r=research_patterns()['quiet'];r=copy.deepcopy(r);r['definitions']['clinical_stv_definition']='Philips_Dawes_Redman_2002'
    result=partial_dr_checklist(r)
    assert result['criteria_met'] is None and result['clinical_risk'] is None
    assert result['requirements'][0]['status']=='met'  # 3.5 > 3, other required evidence is absent.


def test_profiles_exclude_device_quality_duration_duplicate_units_and_masks():
    for cols in PROFILES.values():
        assert 'signal_loss_pct' not in cols and 'duration_min' not in cols
        assert not {'stv_ms','stv_bpm'}<=set(cols)
        assert not {'amplitude_ms','amplitude_bpm'}<=set(cols)
        assert not any('missing' in col for col in cols)


def test_prefix_proxy_merges_initials_and_suffixes_but_preserves_numeric_acquisition_codes():
    assert prefix_proxy('Mabp_12.txt')==prefix_proxy('Mabp_25.txt')
    assert prefix_proxy('Hasc3.txt')==prefix_proxy('Hasc_1.txt')
    assert prefix_proxy('S8001038.dsp')!=prefix_proxy('S8001039.dsp')


def test_patient_weights_and_one_se_are_explicit_parsimony_not_max_f1():
    ids=np.array([1,1,1,2,3,3]);w=weights(ids)
    np.testing.assert_allclose(pd.Series(w).groupby(ids).sum(),1)
    rows=[dict(name='complex',mean_loss=.50,sd_loss=.04,folds=4),dict(name='simple',mean_loss=.515,sd_loss=.04,folds=4)]
    chosen,cutoff=select_one_se(rows,lambda r:0 if r['name']=='simple' else 1)
    assert chosen['name']=='simple' and cutoff==pytest.approx(.52)


@pytest.mark.parametrize('family,params',[('linear',{'C':1.}),('spline',{'C':1.,'n_knots':5}),('catboost',dict(depth=3,iterations=15,learning_rate=.05,l2_leaf_reg=3,auto_class_weights=None))])
def test_models_have_no_missing_fallback_and_roundtrip(tmp_path,family,params):
    rng=np.random.default_rng(18);x=pd.DataFrame(dict(baseline_bpm=rng.uniform(110,170,160),acc15_per_10min=rng.integers(0,8,160)/3,dec15_per_10min=rng.integers(0,3,160)/3))
    y=(x.baseline_bpm>140).astype(int);groups=np.repeat(np.arange(80),2)
    m=StrictBinaryModel(family,params).fit(x,y,groups,np.arange(len(x)))
    bad=x.iloc[:1].copy();bad.iloc[0,0]=np.nan
    with pytest.raises(ValueError):m.predict_proba(bad)
    path=tmp_path/'bundle.joblib';joblib.dump(m,path)
    np.testing.assert_array_equal(m.predict_proba(x),joblib.load(path).predict_proba(x))
    assert not any('missing' in c for c in m.columns_)


def test_multiclass_component_bootstrap_matches_independent_weighted_metrics():
    from sklearn.metrics import f1_score,log_loss,recall_score,average_precision_score
    from src.training.ctg_review_assessment import multiclass_bootstrap
    y=np.array([0,1,2,0,1,2,0,1,2,0]);groups=np.repeat(list('abcde'),2)
    p=np.array([[.8,.1,.1],[.2,.6,.2],[.2,.3,.5],[.3,.4,.3],[.2,.6,.2],[.1,.2,.7],[.5,.3,.2],[.1,.7,.2],[.2,.3,.5],[.6,.1,.3]])
    frame=pd.DataFrame(dict(NSP=y+1,review_group=groups,p0=p[:,0],p1=p[:,1],p2=p[:,2]))
    draws=np.array([[1,2,0,1,1],[0,1,1,2,1]])
    actual=multiclass_bootstrap(frame,draws)
    for k,draw in enumerate(draws):
        w=np.repeat(draw,2)/2;pred=p.argmax(axis=1)
        assert actual.iloc[k].macro_f1==pytest.approx(f1_score(y,pred,labels=[0,1,2],average='macro',sample_weight=w))
        assert actual.iloc[k].log_loss==pytest.approx(log_loss(y,p,labels=[0,1,2],sample_weight=w))
        assert actual.iloc[k].recall_path==pytest.approx(recall_score(y,pred,labels=[0,1,2],average=None,sample_weight=w)[2])
        assert actual.iloc[k].ap_path==pytest.approx(average_precision_score(y==2,p[:,2],sample_weight=w))
