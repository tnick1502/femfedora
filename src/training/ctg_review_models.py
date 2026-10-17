"""Strict complete-input research classifiers for the expert-review revision."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,OneHotEncoder,SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,log_loss,confusion_matrix,average_precision_score,roc_auc_score
from src.features.ctg_preprocessing import CTG_FEATURES,CTGInputGuard


def weights(groups,normalize=False):
    a=np.asarray(groups);_,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
    w=1./counts[inverse]
    return w/w.mean() if normalize else w


def complete_frame(X,columns):
    if not isinstance(X,pd.DataFrame) or X.columns.duplicated().any():raise ValueError('Named unique feature columns required')
    if set(X)!=set(columns):raise ValueError('Missing/extra feature columns')
    x=X.loc[:,columns].astype(float)
    if not np.isfinite(x.to_numpy()).all():raise ValueError('Incomplete input: abstain before prediction')
    if (x<0).any().any():raise ValueError('Negative measurements')
    return x


class StrictBinaryModel:
    """No imputation and no missing indicators; one measured unit per quantity."""
    def __init__(self,family,params,seed=20261018):self.family=family;self.params=dict(params);self.seed=seed
    def fit(self,X,y,groups,ids):
        self.columns_=list(X);x=complete_frame(X,self.columns_);y=np.asarray(y,int)
        if set(y)!={0,1}:raise ValueError('Both binary classes required')
        self.fit_ids_=np.asarray(ids);self.fit_groups_=np.unique(groups);self.classes_=np.array([0,1])
        self.bounds_=pd.DataFrame({'min':x.min(),'max':x.max()})
        w=weights(groups,True)
        if self.family=='catboost':
            self.model_=CatBoostClassifier(**self.params,loss_function='Logloss',random_seed=self.seed,
                thread_count=1,verbose=False,allow_writing_files=False,bootstrap_type='Bayesian',
                bagging_temperature=1.,random_strength=1.,boosting_type='Plain',border_count=128)
            self.model_.fit(x,y,sample_weight=w)
        else:
            self.scaler_=RobustScaler().fit(x);z=self.scaler_.transform(x);self.splines_=[]
            if self.family=='spline':
                for j in range(z.shape[1]):
                    knots=np.unique(np.quantile(z[:,j],np.linspace(0,1,self.params['n_knots'])))
                    s=SplineTransformer(degree=2,knots=knots[:,None],include_bias=False,extrapolation='linear').fit(z[:,[j]]) if len(knots)>2 else None
                    self.splines_.append(s)
                z=self._basis(z)
            self.model_=LogisticRegression(C=self.params['C'],max_iter=3000,tol=1e-5,random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);self.model_.fit(z,y,sample_weight=w)
        return self
    def _basis(self,z):return np.column_stack([s.transform(z[:,[j]]) if s is not None else z[:,[j]] for j,s in enumerate(self.splines_)])
    def predict_proba(self,X,ntree_end=0):
        x=complete_frame(X,self.columns_)
        if self.family=='catboost':return self.model_.predict_proba(x,ntree_end=ntree_end)
        z=self.scaler_.transform(x)
        return self.model_.predict_proba(self._basis(z) if self.family=='spline' else z)


class PrefixCTGModel:
    def __init__(self,family,params,seed=20261018):self.family=family;self.params=dict(params);self.seed=seed
    def fit(self,X,y,groups,ids):
        x=CTGInputGuard().fit_transform(X).drop(columns=['DS','Width']);x['Tendency']=x.Tendency.astype(int)
        y=np.asarray(y,int);self.fit_ids_=np.asarray(ids);self.fit_groups_=np.unique(groups);self.classes_=np.arange(3)
        w=weights(groups,True);self.models_=[]
        if self.family=='logistic':
            self.transformer_=ColumnTransformer([('numeric',RobustScaler(),[c for c in x if c!='Tendency']),
                ('category',OneHotEncoder(categories=[[-1,0,1]],handle_unknown='error',sparse_output=False),['Tendency'])])
            z=self.transformer_.fit_transform(x)
            m=LogisticRegression(C=self.params['C'],max_iter=4000,tol=1e-5)
            with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);m.fit(z,y,sample_weight=w)
            self.models_.append(m);return self
        base=dict(self.params,random_seed=self.seed,thread_count=1,verbose=False,allow_writing_files=False,
            bootstrap_type='Bayesian',bagging_temperature=1.,random_strength=1.,boosting_type='Plain',border_count=128)
        class_weights={'B_sqrt':'SqrtBalanced','B_balanced':'Balanced'}.get(self.family)
        if class_weights:base['auto_class_weights']=class_weights
        if self.family in ['A_multiclass','B_sqrt','B_balanced']:
            m=CatBoostClassifier(**base,loss_function='MultiClass',classes_count=3)
            m.fit(x,y,sample_weight=w,cat_features=['Tendency']);self.models_.append(m)
        else:
            masks=[np.ones(len(y),bool)]*(3 if self.family=='C_ovr' else 2)
            labels=[y==j for j in range(3)] if self.family=='C_ovr' else [y!=0,y==2]
            if self.family=='D_hierarchy':masks[1]=y!=0
            for mask,binary in zip(masks,labels):
                m=CatBoostClassifier(**base,loss_function='Logloss')
                m.fit(x.loc[mask],binary[mask].astype(int),sample_weight=w[mask],cat_features=['Tendency']);self.models_.append(m)
        return self
    def predict_proba(self,X,ntree_end=0):
        x=CTGInputGuard().transform(X).drop(columns=['DS','Width']);x['Tendency']=x.Tendency.astype(int)
        if self.family=='logistic':return self.models_[0].predict_proba(self.transformer_.transform(x))
        if self.family in ['A_multiclass','B_sqrt','B_balanced']:return self.models_[0].predict_proba(x,ntree_end=ntree_end)
        p=np.column_stack([m.predict_proba(x,ntree_end=ntree_end)[:,1] for m in self.models_])
        if self.family=='D_hierarchy':p=np.column_stack([1-p[:,0],p[:,0]*(1-p[:,1]),p[:,0]*p[:,1]])
        p=np.clip(p,1e-12,1);return p/p.sum(axis=1,keepdims=True)


class FixedTemperature:
    """Fixed calibration procedure; current fit-side OOF only, no holdout labels."""
    def fit(self,p,y,groups,ids):
        p=np.clip(p,1e-10,1);p=p/p.sum(axis=1,keepdims=True);w=weights(groups)
        loss=lambda t:log_loss(y,softmax(np.log(p)/np.exp(t),axis=1),sample_weight=w,labels=np.arange(p.shape[1]))+.001*t*t
        r=minimize_scalar(loss,bounds=(-2.,2.),method='bounded')
        if not r.success:raise RuntimeError('Calibration failed')
        self.temperature_=float(np.exp(r.x));self.fit_ids_=np.asarray(ids);return self
    def transform(self,p):return softmax(np.log(np.clip(p,1e-10,1))/self.temperature_,axis=1)


def multiclass_scores(y,p,groups):
    y=np.asarray(y,int);pred=p.argmax(axis=1);w=weights(groups);cm=confusion_matrix(y,pred,labels=[0,1,2],sample_weight=w)
    recall=np.divide(cm.diagonal(),cm.sum(axis=1),out=np.zeros(3),where=cm.sum(axis=1)>0)
    return dict(group_macro_f1=f1_score(y,pred,labels=[0,1,2],average='macro',sample_weight=w,zero_division=0),
        segment_macro_f1=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
        group_recall_normal=recall[0],group_recall_suspect=recall[1],group_recall_path=recall[2],
        group_log_loss=log_loss(y,p,sample_weight=w,labels=[0,1,2]),group_ap_path=average_precision_score(y==2,p[:,2],sample_weight=w),
        group_auc_path=roc_auc_score(y==2,p[:,2],sample_weight=w))


def binary_scores(y,p,groups):
    y=np.asarray(y,int);p=np.asarray(p,float);w=weights(groups);pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=w).ravel()
    return dict(group_macro_f1=f1_score(y,pred,average='macro',sample_weight=w),
        group_recall=tp/(tp+fn),group_specificity=tn/(tn+fp),group_precision=tp/(tp+fp) if tp+fp else 0,
        group_log_loss=log_loss(y,np.column_stack([1-p,p]),sample_weight=w,labels=[0,1]),
        group_ap=average_precision_score(y,p,sample_weight=w),group_auc=roc_auc_score(y,p,sample_weight=w),
        group_brier=np.average((p-y)**2,weights=w),records=len(y),groups=len(np.unique(groups)))


def select_one_se(results,complexity):
    """Parsimony heuristic, not an inferential CI for dependent CV folds."""
    best=min(results,key=lambda r:r['mean_loss']);cutoff=best['mean_loss']+best['sd_loss']/np.sqrt(best['folds'])
    eligible=[r for r in results if r['mean_loss']<=cutoff+1e-12]
    return min(eligible,key=lambda r:(complexity(r),r['mean_loss'])),float(cutoff)
