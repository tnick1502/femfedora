"""Independent arithmetic, boundaries, artifact and notebook checks for CTG v2."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.ctg_audit import AUDIT,EXPORT,context,CTG_FEATURES
from src.training.ctg_experiments import digest
from src.training.ctg_embeddings import legacy_snapshot


def manual_ap(y,p,w=None):
    w=np.ones(len(y)) if w is None else np.asarray(w)
    order=np.argsort(-p,kind='stable');y=np.asarray(y)[order];p=np.asarray(p)[order];w=w[order]
    tp=np.cumsum(w*y);total=np.cumsum(w)
    ends=np.r_[np.flatnonzero(np.diff(p)!=0),len(p)-1]
    recall=tp[ends]/np.sum(w*y)
    return float(np.sum(np.diff(np.r_[0.,recall])*tp[ends]/total[ends]))


def manual_scores(y,p,pred,w=None):
    w=np.ones(len(y)) if w is None else np.asarray(w)
    cm=np.zeros((3,3));np.add.at(cm,(y,pred),w)
    f1=np.divide(2*np.diag(cm),cm.sum(axis=0)+cm.sum(axis=1),out=np.zeros(3),where=(cm.sum(axis=0)+cm.sum(axis=1))>0)
    return dict(macro_f1=float(f1.mean()),recall_path=float(cm[2,2]/cm[2].sum()),
        log_loss=float(np.average(-np.log(p[np.arange(len(y)),y]),weights=w)),
        brier=float(np.average(np.square(p-np.eye(3)[y]).sum(axis=1),weights=w)),
        ap_path=manual_ap(y==2,p[:,2],w))


def validate():
    data,fingerprint,_=context()
    lock=json.loads((AUDIT/'locked_decision.json').read_text())
    assert lock['fingerprint']==fingerprint and not lock['test_used_for_selection']
    train=data[data.split=='train'];partitions={s:set(f.source_excel_row) for s,f in data.groupby('split')}
    layout=pd.read_csv(AUDIT/'outer_folds.csv')
    assert set(layout.source_excel_row)==partitions['train'] and layout.source_excel_row.is_unique
    joined=layout.merge(data[['source_excel_row','source_file','feature_duplicate_group']],on='source_excel_row',validate='one_to_one')
    assert joined.groupby('source_file').outer_fold.nunique().max()==1
    assert joined.groupby('feature_duplicate_group').outer_fold.nunique().max()==1
    inner=pd.read_csv(AUDIT/'inner_folds.csv')
    for k,frame in inner.groupby('outer_fold'):
        assert set(frame.source_excel_row)==partitions['train']-set(layout.loc[layout.outer_fold==k,'source_excel_row'])
        j=frame.merge(data[['source_excel_row','source_file']],on='source_excel_row',validate='one_to_one')
        assert j.groupby('source_file').inner_fold.nunique().max()==1
    grid=pd.read_json(AUDIT/'nested_grid.json');final=pd.read_json(AUDIT/'final_grid.json')
    assert len(grid)==630 and len(final)==126
    nested=pd.read_parquet(AUDIT/'nested_predictions.parquet')
    for _,f in nested.groupby('family'):
        assert set(f.source_excel_row)==partitions['train'] and f.source_excel_row.is_unique
    results=pd.read_csv(AUDIT/'test_metrics.csv')
    predictions=pd.read_parquet(AUDIT/'test_predictions.parquet')
    verified=[]
    for candidate,f in predictions.groupby('candidate'):
        assert set(f.source_excel_row)==partitions['test'] and f.source_excel_row.is_unique
        y=f.NSP.to_numpy(dtype=int)-1;p=f[['p0','p1','p2']].to_numpy();pred=f.pred.to_numpy(dtype=int)-1
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        row=results[results.candidate==candidate].iloc[0]
        values=manual_scores(y,p,pred)
        w=1/f.source_file.map(f.source_file.value_counts()).to_numpy()
        weighted=manual_scores(y,p,pred,w)
        for key,value in values.items():assert abs(value-row[key])<1e-12,(candidate,key,value,row[key])
        for key in ['macro_f1','recall_path','log_loss','ap_path']:
            assert abs(weighted[key]-row['file_'+key])<1e-12
        bundle=joblib.load(EXPORT/f'{candidate.replace("/","_")}.joblib')
        x=data.set_index('source_excel_row').loc[f.source_excel_row,list(CTG_FEATURES)]
        np.testing.assert_allclose(bundle.predict_proba(x),p,atol=1e-12)
        np.testing.assert_array_equal(bundle.predict(x),f.pred)
        verified.append(candidate)
    assert results.primary.sum()==1
    for candidate in lock['shortlist']:
        bundle=joblib.load(EXPORT/f'{candidate.replace("/","_")}.joblib')
        if bundle.family!='logistic':
            from catboost import CatBoostClassifier
            x=bundle.models[0].features(train[list(CTG_FEATURES)].head(3))
            for i,model in enumerate(bundle.models):
                for j,head in enumerate(model.models_):
                    native=CatBoostClassifier().load_model(str(EXPORT/f'{candidate.replace("/","_")}_fold{i}_head{j}.cbm'))
                    np.testing.assert_allclose(native.predict_proba(x),head.predict_proba(x),atol=1e-12)
    preserved=json.loads((ROOT/'research/experiments/ctg_uci/embeddings/preserved_legacy.json').read_text())['sha256']
    assert legacy_snapshot()==preserved
    notebooks=[]
    for name in ['02_nested_grid_search.ipynb','03_calibration_ensemble_and_test.ipynb',
                 '05_results_and_error_analysis.ipynb','07_deep_audit_and_recommendations.ipynb']:
        path=ROOT/'research/notebooks/ctg_uci'/name;book=nbformat.read(path,as_version=4);nbformat.validate(book)
        code=[c for c in book.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        notebooks.append(dict(name=name,code_cells=len(code),figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs),sha256=digest(path)))
    artifact_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(EXPORT.glob('*')) if p.is_file()}
    artifact_hashes.update({str(p.relative_to(ROOT)):digest(p) for p in sorted(AUDIT.glob('*')) if p.suffix in ['.csv','.parquet','.json'] and p.name!='validation_receipt.json'})
    receipt=dict(fingerprint=fingerprint,nested_candidates=630,final_candidates=126,
        independent_test_rows=verified,manual_arithmetic=['confusion-count F1/recall','negative log likelihood','Brier sum','tie-aware AP integral','file weights'],
        grouped_boundaries=True,legacy_clustering_unchanged=True,model_roundtrip=True,native_roundtrip=True,
        notebooks=notebooks,artifact_sha256=artifact_hashes,
        limitations=['Previously inspected test','No patient IDs','Fixed-model bootstrap does not include full retraining uncertainty'])
    (AUDIT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifact_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':validate()
