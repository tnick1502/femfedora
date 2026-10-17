import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,brier_score_loss
from src.training.jnu_report_assessment import bootstrap_scores,block_permute


def test_bootstrap_metric_arithmetic_matches_sklearn_with_ties():
    y=np.array([0,1,0,1,1,0,1,0]);p=np.array([.1,.2,.2,.6,.6,.9,.9,.9])
    patients=np.array([1,1,2,3,3,4,5,5]);components=patients.copy()
    draws=np.array([[1,1,1,1,1],[2,0,1,2,0],[0,2,1,0,2]])
    b=bootstrap_scores(y,p,patients,components,draws)
    _,inverse,counts=np.unique(patients,return_inverse=True,return_counts=True)
    for i in range(len(draws)):
        w=draws[i,inverse]/counts[inverse]
        assert b.iloc[i].ap==pytest.approx(average_precision_score(y,p,sample_weight=w))
        assert b.iloc[i].roc_auc==pytest.approx(roc_auc_score(y,p,sample_weight=w))
        assert b.iloc[i].macro_f1==pytest.approx(f1_score(y,p>=.5,sample_weight=w,average='macro'))
        assert b.iloc[i].brier==pytest.approx(brier_score_loss(y,p,sample_weight=w))


def test_group_permutation_keeps_group_sizes_and_exact_label_prevalence():
    f=pd.DataFrame({'patient_component':[1,1,2,2,3,3,4,5,6],
        'figo_label':[0,1,1,1,0,0,1,0,1]})
    yy,changed=block_permute(f,np.random.default_rng(431))
    assert yy.sum()==f.figo_label.sum() and 0<=changed<=1
    assert sorted(tuple(yy[ii]) for ii in [[0,1],[2,3],[4,5]])==[(0,0),(0,1),(1,1)]
