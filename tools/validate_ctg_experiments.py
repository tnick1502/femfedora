"""Independent row-boundary and score reconciliation for the CTG experiment."""
from pathlib import Path
import json
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, recall_score, adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.ctg_experiments import OUT, MODELS, CAT_GRID, CTG_FEATURES, load_data


def validate():
    data, fingerprint = load_data()
    partition = {part: set(frame.source_excel_row) for part, frame in data.groupby('split')}
    outer = pd.read_csv(OUT/'outer_folds.csv')
    assert set(outer.source_excel_row) == partition['train']
    audit = outer.merge(data[['source_excel_row','source_file']],on='source_excel_row',validate='one_to_one')
    assert audit.groupby('source_file').outer_fold.nunique().max() == 1
    inner = pd.read_csv(OUT/'inner_folds.csv')
    for fold, part in inner.groupby('outer_fold'):
        expected = partition['train'] - set(outer.loc[outer.outer_fold==fold,'source_excel_row'])
        assert set(part.source_excel_row)==expected
        groups=part.merge(data[['source_excel_row','source_file']],on='source_excel_row',validate='one_to_one')
        assert groups.groupby('source_file').inner_fold.nunique().max()==1
    grid=pd.read_json(OUT/'nested_grid.json')
    assert len(grid)==630
    for (family,fold), part in grid.groupby(['family','outer_fold']):
        assert len(part)==(6 if family=='logistic' else len(CAT_GRID))
        assert part.candidate.is_unique
    nested=pd.read_parquet(OUT/'nested_predictions.parquet')
    for _, part in nested.groupby(['family','mode']):
        assert set(part.source_excel_row)==partition['train'] and part.source_excel_row.is_unique
    calibration=pd.read_csv(OUT/'validation_calibration_folds.csv')
    assert set(calibration.source_excel_row)==partition['validation']
    groups=calibration.merge(data[['source_excel_row','source_file']],on='source_excel_row',validate='one_to_one')
    assert groups.groupby('source_file').calibration_fold.nunique().max()==1
    final_grid=pd.read_json(OUT/'final_grid.json')
    assert len(final_grid)==126
    locked=json.loads((OUT/'locked_decision.json').read_text())
    assert locked['fingerprint']==fingerprint and not locked['test_used_for_selection']
    results=pd.read_csv(OUT/'test_metrics.csv')
    predictions=pd.read_parquet(OUT/'test_predictions.parquet')
    for (family,architecture), part in predictions.groupby(['family','architecture']):
        assert set(part.source_excel_row)==partition['test'] and part.source_excel_row.is_unique
        p=part[['p0','p1','p2']].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        assert np.isfinite(p).all() and (p>=0).all()
        y=part.NSP.to_numpy()-1; pred=part.pred.to_numpy()-1
        row=results[(results.family==family)&(results.architecture==architecture)].iloc[0]
        np.testing.assert_allclose(row.macro_f1,f1_score(y,pred,average='macro',labels=[0,1,2]),atol=1e-12)
        np.testing.assert_allclose(row.recall_path,recall_score(y==2,pred==2),atol=1e-12)
        np.testing.assert_allclose(row.ap_path,average_precision_score(y==2,p[:,2]),atol=1e-12)
        np.testing.assert_allclose(row.log_loss,log_loss(y,p,labels=[0,1,2]),atol=1e-12)
        np.testing.assert_allclose(row.brier,np.square(np.eye(3)[y]-p).sum(axis=1).mean(),atol=1e-12)
    assert results.champion.sum()==1
    bundle=joblib.load(MODELS/'champion.joblib')
    test=data[data.split=='test']
    champion=predictions[(predictions.family==locked['family'])&(predictions.architecture==locked['architecture'])]
    assert test.source_excel_row.tolist()==champion.source_excel_row.tolist()
    np.testing.assert_allclose(bundle.predict_proba(test[list(CTG_FEATURES)]),champion[['p0','p1','p2']].to_numpy())
    cluster_grid=pd.read_csv(OUT/'cluster_grid.csv')
    cluster_lock=json.loads((OUT/'cluster_locked_config.json').read_text())
    assert 'without outcomes' in cluster_lock['selection']
    assert cluster_grid.chosen.sum()==3
    assignments=pd.read_parquet(OUT/'cluster_assignments.parquet')
    association=pd.read_csv(OUT/'cluster_association.csv')
    cluster_summary=pd.read_csv(OUT/'cluster_test_summary.csv')
    for (_,part), frame in assignments.groupby(['algorithm','split']):
        assert set(frame.source_excel_row)==partition[part]
    for (algorithm,part), frame in assignments.groupby(['algorithm','split']):
        row=association[(association.algorithm==algorithm)&(association.split==part)].iloc[0]
        np.testing.assert_allclose(row.ari,adjusted_rand_score(frame.NSP,frame.cluster))
        np.testing.assert_allclose(row.nmi,normalized_mutual_info_score(frame.NSP,frame.cluster))
        membership_columns=[c for c in frame if c.startswith('membership_') and c!='membership_entropy']
        np.testing.assert_allclose(frame[membership_columns].sum(axis=1),1)
        if part=='test':
            for cluster, inside in frame.groupby('cluster'):
                row=cluster_summary[(cluster_summary.algorithm==algorithm)&(cluster_summary.cluster==cluster)].iloc[0]
                np.testing.assert_allclose(row.path_fraction,(inside.NSP==3).mean())
                np.testing.assert_allclose(row.suspect_fraction,(inside.NSP==2).mean())
                np.testing.assert_allclose(row.abnormal_fraction,(inside.NSP!=1).mean())
                np.testing.assert_allclose(row.enrichment,(inside.NSP==3).mean()/(frame.NSP==3).mean())
    from catboost import CatBoostClassifier
    if bundle.family!='logistic':
        x=bundle.models[0].features(data[data.split=='train'][list(CTG_FEATURES)].head(3))
        for h,head in enumerate(bundle.models[0].models_):
            native=CatBoostClassifier().load_model(str(MODELS/f'champion_fold0_head{h}.cbm'))
            np.testing.assert_allclose(native.predict_proba(x),head.predict_proba(x))
    executed=[]
    for name in ['02_nested_grid_search.ipynb','03_calibration_ensemble_and_test.ipynb',
                 '04_clustering_without_outcomes.ipynb','05_results_and_error_analysis.ipynb']:
        book=nbformat.read(ROOT/'research/notebooks/ctg_uci'/name,as_version=4)
        nbformat.validate(book)
        code=[cell for cell in book.cells if cell.cell_type=='code']
        assert all(cell.execution_count is not None for cell in code)
        assert not [o for c in code for o in c.outputs if o.output_type=='error']
        executed.append(dict(notebook=name,code_cells=len(code),figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)))
    receipt=dict(fingerprint=fingerprint,outer_folds=5,inner_folds=3,nested_candidates=len(grid),
        final_candidates=len(final_grid),group_boundaries='verified',test_scores='independently_recomputed',
        cluster_statistics='independently_recomputed',native_model_roundtrip='verified',
        notebooks=executed,limits=['No patient IDs','35 pathologic test segments from 10 recordings',
        'Fixed model bootstrap does not cover full retraining variability'])
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    validate()
