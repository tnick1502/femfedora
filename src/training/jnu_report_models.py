"""Patient-weighted classifiers using only versioned CTG report fields.

No raw signal, outcome, patient ID or precomputed global scaler enters X.
All preprocessing is fitted again inside the current fitting cohort.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import (confusion_matrix, f1_score, average_precision_score,
    roc_auc_score, log_loss, brier_score_loss)

from src.features.jnu_ctg_report_extraction import PROFILE_FIELDS,VERSION


def patient_weights(patient_ids, normalize=False):
    ids=np.asarray(patient_ids)
    _,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
    w=1./counts[inverse]
    return w/w.mean() if normalize else w


def training_weights(y,patients,policy=None):
    y=np.asarray(y,int); w=patient_weights(patients,normalize=True)
    if policy not in [None,'Balanced','SqrtBalanced']:
        raise ValueError('Unknown class weighting')
    if policy:
        totals=np.bincount(y,weights=w,minlength=2)
        if (totals<=0).any():raise ValueError('Both classes required')
        multiplier=totals.max()/totals
        if policy=='SqrtBalanced':multiplier=np.sqrt(multiplier)
        w=w*multiplier[y];w=w/w.mean()
    return w


def checked_frame(X,profile):
    if not isinstance(X,pd.DataFrame):raise TypeError('Named DataFrame required')
    required=PROFILE_FIELDS[profile]
    if X.columns.duplicated().any() or set(X.columns)!=set(required):
        raise ValueError('Unexpected/missing columns; identifiers and labels are forbidden')
    x=X[required].astype(float)
    if np.isinf(x.to_numpy()).any():raise ValueError('Infinity is not a missing value')
    if (x<0).any().any():raise ValueError('Negative report measurements')
    if (x.baseline_bpm.dropna()<=0).any():raise ValueError('Baseline must be positive')
    if 'duration_min' in x and (x.duration_min.dropna()<=0).any():raise ValueError('Duration must be positive')
    if 'signal_loss_pct' in x and (x.signal_loss_pct>100).any():raise ValueError('Loss >100%')
    for col in [c for c in x if c.endswith('_fraction')]:
        if (x[col]>1).any():raise ValueError('Fraction >1')
    return x


class ReportBasis:
    """Fold-local median/IQR + linear or quantile B-splines and optional pairs.

    Constant/binary columns remain linear. Duplicate quantile knots are removed
    independently for each feature; a zero-heavy count never produces invalid
    repeated knots. Unknown values have their own indicator for EVERY input.
    """
    def __init__(self,profile,n_knots=5,interactions=False,linear=False):
        self.profile=profile;self.n_knots=n_knots
        self.interactions=interactions;self.linear=linear

    def fit(self,X):
        x=checked_frame(X,self.profile).to_numpy()
        self.columns_=list(PROFILE_FIELDS[self.profile])
        self.medians_=np.array([np.nanmedian(v) if np.isfinite(v).any() else 0. for v in x.T])
        z=np.where(np.isnan(x),self.medians_,x)
        self.centers_=np.median(z,axis=0)
        self.scales_=np.subtract(*np.percentile(z,[75,25],axis=0))
        self.scales_[self.scales_<10*np.finfo(float).eps]=1.
        z=(z-self.centers_)/self.scales_
        self.splines_=[];self.names_=[];self.slices_={}
        offset=0
        for j,col in enumerate(self.columns_):
            knots=np.unique(np.quantile(z[:,j],np.linspace(0,1,self.n_knots)))
            if self.linear or len(np.unique(z[:,j]))<=2 or len(knots)<2:
                spline=None; width=1
            else:
                spline=SplineTransformer(degree=2,knots=knots[:,None],
                    include_bias=False,extrapolation='constant').fit(z[:,[j]])
                width=spline.n_features_out_
            self.splines_.append(spline)
            self.slices_[col]=slice(offset,offset+width);offset+=width
            self.names_.extend([f'{col}:basis{k}' for k in range(width)])
        stv='stv_4s_ms' if 'stv_4s_ms' in self.columns_ else 'stv_ms'
        amp='amplitude_bpm' if 'amplitude_bpm' in self.columns_ else 'amplitude_ms'
        planned=[('acc15_per_10min','dec15_per_10min'),('baseline_bpm',stv),
                 ('acc15_per_10min',stv),('dec15_per_10min',amp)]
        self.pairs_=[p for p in planned if all(k in self.columns_ for k in p)] if self.interactions else []
        for a,b in self.pairs_:
            wa=self.slices_[a].stop-self.slices_[a].start
            wb=self.slices_[b].stop-self.slices_[b].start
            self.names_.extend([f'{a}*{b}:{i}:{j}' for i in range(wa) for j in range(wb)])
        self.names_.extend([f'{c}__missing' for c in self.columns_])
        return self

    def transform(self,X):
        x=checked_frame(X,self.profile).to_numpy()
        missing=np.isnan(x)
        z=(np.where(missing,self.medians_,x)-self.centers_)/self.scales_
        blocks=[s.transform(z[:,[j]]) if s is not None else z[:,[j]] for j,s in enumerate(self.splines_)]
        basic=np.column_stack(blocks);features=[basic]
        for a,b in self.pairs_:
            aa,bb=basic[:,self.slices_[a]],basic[:,self.slices_[b]]
            features.append((aa[:,:,None]*bb[:,None,:]).reshape(len(x),-1))
        features.append(missing.astype(float))
        result=np.column_stack(features)
        if not np.isfinite(result).all():raise ValueError('Non-finite basis')
        return result


class ReportClassifier:
    def __init__(self,profile,family,params,seed=20261017):
        self.profile=profile;self.family=family;self.params=dict(params);self.seed=seed

    def fit(self,X,y,patients,record_ids=None,groups=None):
        x=checked_frame(X,self.profile);y=np.asarray(y,int)
        if set(np.unique(y))!={0,1}:raise ValueError('Require binary FIGO 0/1')
        self.fit_record_ids_=np.asarray(record_ids) if record_ids is not None else np.arange(len(y))
        self.fit_patient_ids_=np.unique(patients)
        self.fit_components_=np.unique(groups) if groups is not None else np.unique(patients)
        self.bounds_=pd.DataFrame({'min':x.min(),'max':x.max()})
        self.training_medians_=x.median().fillna(0)
        self.fit_rows_=len(y);self.classes_=np.array([0,1])
        if self.family=='catboost':
            self.model_=CatBoostClassifier(**self.params,loss_function='Logloss',
                random_seed=self.seed,thread_count=1,verbose=False,allow_writing_files=False,
                bootstrap_type='Bayesian',bagging_temperature=1.,random_strength=1.,
                use_best_model=False,boosting_type='Plain',border_count=128)
            features=self._tree_features(x)
            self.model_.fit(features,y,sample_weight=patient_weights(patients,normalize=True))
        elif self.family in ['spline_logit','linear']:
            self.basis_=ReportBasis(self.profile,n_knots=self.params.get('n_knots',5),
                interactions=self.params.get('interactions',False),linear=self.family=='linear').fit(x)
            self.model_=LogisticRegression(C=self.params['C'],max_iter=2500,tol=1e-5,
                solver='lbfgs',random_state=self.seed)
            weights=training_weights(y,patients,self.params.get('weight_policy'))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                self.model_.fit(self.basis_.transform(x),y,sample_weight=weights)
        else:raise ValueError('Unknown family')
        return self

    @staticmethod
    def _tree_features(x):
        return pd.concat([x,x.isna().astype(float).add_suffix('__missing')],axis=1)

    def predict_positive(self,X,ntree_end=0):
        x=checked_frame(X,self.profile)
        if self.family=='catboost':
            return self.model_.predict_proba(self._tree_features(x),ntree_end=ntree_end)[:,1]
        return self.model_.predict_proba(self.basis_.transform(x))[:,1]


class SigmoidCalibration:
    """Positive-slope logit calibration of the ACTUAL exported aggregate."""
    def fit(self,p,y,patients):
        p=np.asarray(p,float);y=np.asarray(y,float)
        z=logit(np.clip(p,1e-6,1-1e-6));w=patient_weights(patients);w=w/w.sum()
        def objective(theta):
            a,b=theta;v=a*z+b
            loss=np.dot(w,np.logaddexp(0,v)-y*v)+1e-5*((a-1)**2+b*b)
            delta=w*(expit(v)-y)
            return loss,np.array([np.dot(delta,z)+2e-5*(a-1),delta.sum()+2e-5*b])
        fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.05,20.),(-10.,10.)])
        if not fit.success:raise RuntimeError(f'Calibration did not converge: {fit.message}')
        self.slope_,self.intercept_=map(float,fit.x)
        self.fit_rows_=len(y);self.fit_patients_=np.unique(patients)
        return self

    def transform(self,p):
        return expit(self.slope_*logit(np.clip(p,1e-6,1-1e-6))+self.intercept_)


class ReportBundle:
    def __init__(self,profile,family,models,calibrator,threshold=.5):
        self.profile=profile;self.family=family;self.models=models
        self.calibrator=calibrator;self.threshold=float(threshold)
        self.classes_=np.array([0,1]);self.clinically_validated_=False
        self.measurement_contract_=VERSION

    def raw_probability(self,X):
        return np.mean([m.predict_positive(X) for m in self.models],axis=0)

    def predict_proba(self,X):
        p=self.calibrator.transform(self.raw_probability(X))
        return np.column_stack([1-p,p])

    def predict(self,X):return (self.predict_proba(X)[:,1]>=self.threshold).astype(int)

    def support(self,X):
        x=checked_frame(X,self.profile)
        core=['baseline_bpm','acc15_per_10min','dec15_per_10min']
        low=pd.concat([m.bounds_['min'] for m in self.models],axis=1).min(axis=1)
        high=pd.concat([m.bounds_['max'] for m in self.models],axis=1).max(axis=1)
        return pd.DataFrame({'core_available':x[core].notna().all(axis=1),
            'all_profile_fields_available':x.notna().all(axis=1),
            'outside_fit_range':((x<low)|(x>high)).any(axis=1),
            'duration_unsupported':(x.duration_min!=30) if 'duration_min' in x else False},index=x.index)

    def audit_report(self,report):
        from src.features.ctg_russian_reports import report_profile_input
        x,audit=report_profile_input(report,self.profile)
        if audit['values']['duration_min']!=30:
            audit['issues'].append('Training records were 30 min; other durations unvalidated')
        audit.update(clinical_risk=None,clinical_validation=False,
            supported_for_clinical_inference=False,research_profile=self.profile)
        return audit


def score(y,p,patients,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid predictions')
    results={}
    for prefix,w in [('record',None),('patient',patient_weights(patients))]:
        pred=p>=threshold
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=w).ravel()
        div=lambda a,b:float(a/b) if b else np.nan
        results.update({f'{prefix}_macro_f1':float(f1_score(y,pred,average='macro',sample_weight=w,zero_division=0)),
            f'{prefix}_recall':div(tp,tp+fn),f'{prefix}_specificity':div(tn,tn+fp),
            f'{prefix}_precision':div(tp,tp+fp),f'{prefix}_npv':div(tn,tn+fn),
            f'{prefix}_balanced_accuracy':(div(tp,tp+fn)+div(tn,tn+fp))/2,
            f'{prefix}_ap':float(average_precision_score(y,p,sample_weight=w)),
            f'{prefix}_roc_auc':float(roc_auc_score(y,p,sample_weight=w)),
            f'{prefix}_log_loss':float(log_loss(y,p,sample_weight=w,labels=[0,1])),
            f'{prefix}_brier':float(brier_score_loss(y,p,sample_weight=w)),
            f'{prefix}_tn':float(tn),f'{prefix}_fp':float(fp),f'{prefix}_fn':float(fn),f'{prefix}_tp':float(tp)})
    return results


def choose_research_threshold(y,p,patients):
    """Max patient-balanced F1 on a separate operating-point set, not clinical."""
    w=patient_weights(patients)
    order=np.argsort(-p,kind='stable');ps=np.asarray(p)[order]
    yy=np.asarray(y)[order];ww=w[order]
    end=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
    tp=np.cumsum(ww*yy)[end];fp=np.cumsum(ww*(1-yy))[end]
    fn=np.dot(w,y)-tp;tn=np.dot(w,1-np.asarray(y))-fp
    f1=.5*(2*tp/np.maximum(2*tp+fp+fn,1e-15)+2*tn/np.maximum(2*tn+fp+fn,1e-15))
    best=np.flatnonzero(f1>=f1.max()-1e-12)
    selected=best[np.argmin(np.abs(ps[end][best]-.5))]
    return float(ps[end][selected])
