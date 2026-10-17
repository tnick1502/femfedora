import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.training.ctg_audit import ProbabilityCalibrator, soft_vote, group_weights, AuditedCTGBundle
from src.features.ctg_preprocessing import CTG_FEATURES


class FixedHeads:
    def __init__(self,a,b): self.heads=np.array([a,b])
    def predict_heads(self,X): return np.tile(self.heads,(len(X),1))


def valid_frame():
    frame=pd.DataFrame({c:[0.,0.] for c in CTG_FEATURES})
    frame['LB']=130.;frame['Min']=100.;frame['Max']=150.;frame['Width']=50.
    return frame


def test_hierarchical_soft_vote_has_independent_probability_oracle():
    models=[FixedHeads(.9,.9),FixedHeads(.1,.1)]
    # Per-model P(path): .81 and .01. Their mixture is .41, not .5*.5=.25.
    actual=soft_vote(models,valid_frame(),'D_hierarchy')
    np.testing.assert_allclose(actual,np.tile([.5,.09,.41],(2,1)),atol=1e-12)


def test_temperature_preserves_argmax_and_bundle_serialization(tmp_path):
    p=np.tile([[.55,.3,.15],[.2,.55,.25],[.15,.25,.6]],(30,1))
    y=np.tile([0,1,2],30)
    cal=ProbabilityCalibrator('temperature').fit(p,y)
    q=cal.predict_proba(p)
    np.testing.assert_array_equal(p.argmax(axis=1),q.argmax(axis=1))
    assert log_loss(y,q)<log_loss(y,p)
    policy={'calibrator':ProbabilityCalibrator('raw').fit(p,y),'threshold':.4}
    bundle=AuditedCTGBundle([FixedHeads(.9,.9),FixedHeads(.1,.1)],'D_hierarchy',policy)
    path=tmp_path/'model.joblib';joblib.dump(bundle,path)
    restored=joblib.load(path)
    np.testing.assert_allclose(restored.predict_proba(valid_frame()),[[.5,.09,.41]]*2)
    np.testing.assert_array_equal(restored.predict(valid_frame()),[3,3])


def test_file_weights_equal_total_contribution_per_recording():
    frame=pd.DataFrame({'source_file':['a','a','a','b','c','c']})
    weights=group_weights(frame)
    mass=pd.Series(weights).groupby(frame.source_file).sum()
    np.testing.assert_allclose(mass,[2.,2.,2.])
