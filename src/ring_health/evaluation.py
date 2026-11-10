"""Metrics, cluster bootstrap and validation-only threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y, p, threshold=.5, weights=None):
    y, p = np.asarray(y).astype(int), np.asarray(p)
    k = p.shape[1]
    prediction = (p[:,1] >= threshold).astype(int) if k == 2 else np.argmax(p,axis=1)
    values = {"n":len(y), "accuracy":float(accuracy_score(y,prediction,sample_weight=weights)),
              "balanced_accuracy":float(balanced_accuracy_score(y,prediction,sample_weight=weights)),
              "macro_f1":float(f1_score(y,prediction,average="macro",zero_division=0,sample_weight=weights)),
              "log_loss":float(log_loss(y,p,labels=np.arange(k),sample_weight=weights)),
              "confusion_matrix":confusion_matrix(y,prediction,labels=np.arange(k),sample_weight=weights).tolist()}
    if k == 2:
        values.update(roc_auc=float(roc_auc_score(y,p[:,1],sample_weight=weights)),
                      average_precision=float(average_precision_score(y,p[:,1],sample_weight=weights)),
                      brier=float(brier_score_loss(y,p[:,1],sample_weight=weights)),
                      positive_f1=float(f1_score(y,prediction,zero_division=0,sample_weight=weights)),
                      precision=float(precision_score(y,prediction,zero_division=0,sample_weight=weights)),
                      sensitivity=float(recall_score(y,prediction,zero_division=0,sample_weight=weights)),
                      positive_prevalence=float(np.average(y,weights=weights)),threshold=float(threshold))
    else:
        values["brier"] = float(np.average(np.sum((p-np.eye(k)[y])**2,axis=1),weights=weights))
    return values


def threshold_on_validation(y,p):
    thresholds = np.linspace(.1,.8,71)
    scores = [f1_score(y,p[:,1]>=t,zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def cluster_ci(y,p,groups,threshold=.5,n_bootstrap=200,seed=330):
    y,p,groups = np.asarray(y),np.asarray(p),np.asarray(groups)
    unique = np.unique(groups)
    positions = {g:np.flatnonzero(groups==g) for g in unique}
    rng = np.random.default_rng(seed)
    values=[]
    for _ in range(n_bootstrap):
        ix=np.concatenate([positions[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix])) < p.shape[1]:
            continue
        values.append(classification_metrics(y[ix],p[ix],threshold))
    keys=["macro_f1","accuracy","brier"]+(["roc_auc","average_precision","positive_f1"] if p.shape[1]==2 else [])
    return {"method":"percentile participant bootstrap; conditional on fitted model and split; not survey-design or external-validation uncertainty", "replicates":len(values),
            "intervals_95":{k:[float(v) for v in np.quantile([x[k] for x in values],[.025,.975])] for k in keys}}
