"""Independent checks for revised CTG grouping, strict inference and fit artifacts."""
from pathlib import Path
import sys
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,log_loss,average_precision_score,roc_auc_score
from src.training.ctg_review_evidence import ROOT,OUT as REVIEW
from src.features.ctg_preprocessing import CTG_FEATURES
from src.features.ctg_report_contract import PROFILES,research_patterns,evaluate_report,parse_ctg_report
from src.features.ctg_russian_reports import EXAMPLE_TEXT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes(receipt,key):
    for relative,expected in receipt[key].items():assert sha(ROOT/relative)==expected,relative


def uci_validate():
    out=ROOT/'research/experiments/ctg_uci/reviewed_v3';exports=ROOT/'models/exported/ctg_uci/reviewed_v3'
    protocol=json.loads((out/'protocol.json').read_text());hashes(protocol,'input_sha256')
    done=json.loads((out/'completed.json').read_text());hashes(done,'artifacts_sha256')
    d=pd.read_parquet(REVIEW/'uci_group_assignments.parquet');assignment=pd.read_parquet(out/'outer_assignments.parquet')
    assert d.source_file.nunique()==352 and d.review_group.nunique()==177 and len(d)==2126
    assert d.groupby('feature_duplicate_group').review_group.nunique().max()==1
    assert d.groupby('prefix_proxy').review_group.nunique().max()==1
    for repetition,g in assignment.groupby('repetition'):
        assert len(g)==2126 and g.source_excel_row.is_unique and g.review_group.nunique()==177
        assert g.groupby('review_group').outer_fold.nunique().max()==1
    assert assignment.repetition.nunique()==3
    pred=pd.read_parquet(out/'nested_predictions.parquet');metrics=pd.read_csv(out/'repeat_metrics.csv')
    for (rep,family),g in pred.groupby(['repetition','family']):
        assert len(g)==2126 and g.source_excel_row.is_unique
        p=g[['p0','p1','p2']].to_numpy();np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12);assert np.isfinite(p).all()
        count=g.groupby('review_group').source_excel_row.transform('size').to_numpy();w=1/count;y=g.NSP.to_numpy(int)-1
        cm=confusion_matrix(y,p.argmax(axis=1),labels=[0,1,2],sample_weight=w)
        macro=np.mean(2*cm.diagonal()/(cm.sum(axis=0)+cm.sum(axis=1)))
        r=metrics[(metrics.repetition==rep)&(metrics.family==family)].iloc[0]
        np.testing.assert_allclose([r.group_macro_f1,r.group_log_loss],[macro,log_loss(y,p,sample_weight=w,labels=[0,1,2])],atol=1e-12)
    native=0
    primary=pd.read_parquet(ROOT/'data/processed/ctg_uci/ctg_primary.parquet');X=primary[list(CTG_FEATURES)]
    for path in exports.glob('*.joblib'):
        if path.name.endswith('preprocessor.joblib'):continue
        b=joblib.load(path);m=b.model
        assert set(m.fit_ids_)==set(primary.source_excel_row)
        assert set(m.fit_groups_)==set(d.review_group)
        assert set(b.calibrator.fit_ids_)==set(primary.source_excel_row)
        np.testing.assert_allclose(b.predict_proba(X.iloc[:23]).sum(axis=1),1)
        if m.family=='logistic':
            cols=m.transformer_.transformers_[0][2];s=m.transformer_.named_transformers_['numeric']
            np.testing.assert_allclose(s.center_,X[cols].median())
            q=X[cols].quantile(.75)-X[cols].quantile(.25);q[q==0]=1
            np.testing.assert_allclose(s.scale_,q)
            saved=joblib.load(exports/'logistic_preprocessor.joblib')
            transformed=X.drop(columns=['DS','Width']).copy();transformed.Tendency=transformed.Tendency.astype(int)
            np.testing.assert_array_equal(saved.transform(transformed),m.transformer_.transform(transformed))
        else:
            transformed=X.drop(columns=['DS','Width']).iloc[:31].copy();transformed.Tendency=transformed.Tendency.astype(int)
            for j,model in enumerate(m.models_):
                copy=CatBoostClassifier();copy.load_model(exports/f'{m.family}_{j}.cbm')
                np.testing.assert_array_equal(copy.predict_proba(transformed),model.predict_proba(transformed));native+=1
    return dict(records=2126,prefix_proxy_groups=177,repeats=3,native_models= native,
        patient_identity_verified=False,independent_new_test=False,manual_metrics=True)


def jnu_validate():
    from src.training.jnu_review_experiment import load_data,eligible,xframe,representations
    out=ROOT/'research/experiments/jnu_ctg/reviewed_v2';exports=ROOT/'models/exported/jnu_ctg/reviewed_v2'
    protocol=json.loads((out/'protocol.json').read_text());hashes(protocol,'input_sha256')
    done=json.loads((out/'completed.json').read_text());hashes(done,'artifacts_sha256')
    d=load_data();indexed=d.set_index('record_id');assignment=pd.read_parquet(out/'outer_assignments.parquet')
    assert len(assignment)==20769 and assignment.record_id.is_unique
    assert assignment.groupby('patient_component').outer_fold.nunique().max()==1
    pred=pd.read_parquet(out/'nested_predictions.parquet');metrics=pd.read_csv(out/'profile_metrics.csv')
    for (profile,family),g in pred.groupby(['profile','family']):
        assert len(g)==len(d) and g.record_id.is_unique
        ordered=g.set_index('record_id').loc[d.record_id]
        np.testing.assert_array_equal(ordered.accepted.to_numpy(),eligible(d,profile))
        assert ordered.loc[~ordered.accepted,'p'].isna().all()
        z=g[g.accepted];assert np.isfinite(z.p).all()
        counts=z.groupby('patient_id').record_id.transform('size').to_numpy();w=1/counts;y=z.figo_label.to_numpy(int);p=z.p.to_numpy()
        cm=confusion_matrix(y,p>=.5,labels=[0,1],sample_weight=w);f1=np.mean(2*cm.diagonal()/(cm.sum(axis=0)+cm.sum(axis=1)))
        r=metrics[(metrics.profile==profile)&(metrics.family==family)].iloc[0]
        np.testing.assert_allclose([r.group_macro_f1,r.group_ap,r.group_auc],[f1,average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w)],atol=1e-12)
    cache=ROOT/'tmp/jnu_reviewed_v2'/protocol['fingerprint'][:16]
    checked_members=0;checked_outer_bundles=0
    def validate_bundle(bundle,outer_hold_ids=None):
        nonlocal checked_members
        cal_ids=set(bundle.calibrator.fit_ids_);cal_components=set(indexed.loc[list(cal_ids)].patient_component)
        all_ids=set()
        for j,m in enumerate(bundle.models):
            ids=set(m.fit_ids_);fit=indexed.loc[m.fit_ids_]
            assert not ids&cal_ids and not set(m.fit_groups_)&cal_components
            if outer_hold_ids is not None:
                assert not ids&outer_hold_ids and not cal_ids&outer_hold_ids
                assert not set(m.fit_groups_)&set(indexed.loc[list(outer_hold_ids)].patient_component)
                assert not cal_components&set(indexed.loc[list(outer_hold_ids)].patient_component)
            assert m.columns_==PROFILES[bundle.profile] and not any('missing' in c for c in m.columns_)
            assert np.isfinite(fit[m.columns_]).all().all()
            if m.family!='catboost':
                np.testing.assert_allclose(m.scaler_.center_,fit[m.columns_].median())
                q=fit[m.columns_].quantile(.75)-fit[m.columns_].quantile(.25);q[q<10*np.finfo(float).eps]=1
                np.testing.assert_allclose(m.scaler_.scale_,q)
            else:assert m.model_.feature_names_==PROFILES[bundle.profile]
            all_ids|=ids;checked_members+=1
        assert len(bundle.models)==(5 if bundle.mode=='ensemble5' else 1)
        if bundle.mode=='ensemble5':
            counts=pd.Series([rid for m in bundle.models for rid in m.fit_ids_]).value_counts();assert (counts==4).all()
        probe=xframe(indexed.loc[sorted(all_ids)[:17]],bundle.profile)
        p=bundle.predict_proba(probe);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(axis=1),1)
        bad=probe.iloc[:1].copy();bad.iloc[0,0]=np.nan
        try:bundle.predict_proba(bad)
        except ValueError:pass
        else:raise AssertionError('Strict model accepted missing input')
    for k in range(5):
        hold=set(assignment.loc[assignment.outer_fold==k,'record_id'])
        for profile in PROFILES:
            bundles,choices,grid,champion=joblib.load(cache/f'outer{k}_{profile}.joblib')
            h=indexed.loc[assignment.loc[assignment.outer_fold==k,'record_id']]
            for family,b in bundles.items():
                validate_bundle(b,hold);checked_outer_bundles+=1
                accepted=eligible(h,profile);p=b.predict_proba(xframe(h.loc[accepted],profile))[:,1]
                stored=pred[(pred.profile==profile)&(pred.family==family)].set_index('record_id').loc[h.loc[accepted].index,'p']
                np.testing.assert_allclose(p,stored,rtol=0,atol=1e-12)
    final_models=0
    for path in exports.glob('*.joblib'):
        if path.name.endswith('_preprocessor.joblib'):continue
        b=joblib.load(path);validate_bundle(b);final_models+=1
        for j,m in enumerate(b.models):
            if m.family=='catboost':
                native=CatBoostClassifier();native.load_model(exports/f'{b.profile}_{b.family}_{j}.cbm')
                xx=xframe(indexed.loc[m.fit_ids_[:17]],b.profile)
                np.testing.assert_array_equal(native.predict_proba(xx),m.predict_proba(xx))
            else:
                saved=joblib.load(exports/f'{b.profile}_{b.family}_{j}_preprocessor.joblib')
                np.testing.assert_array_equal(saved['scaler'].center_,m.scaler_.center_)
        report=parse_ctg_report(EXAMPLE_TEXT)['measurements'];audit=b.assess_report(report)
        assert audit['probability_expert_abnormal'] is None and audit['clinical_risk'] is None
    assert final_models==12
    registry=json.loads((exports/'research_models.json').read_text())
    assert set(registry['profiles'])==set(PROFILES) and registry['clinical_validation'] is False
    for profile,item in registry['profiles'].items():
        card=json.loads((exports/f'{profile}_model_card.json').read_text())
        assert item['family']==card['selected_family'] and item['columns']==PROFILES[profile]
        assert sha(exports/item['bundle'])==item['sha256']
    # Same representation model/cohort, imputer fit and actual probability roundtrips.
    reps=representations(d);reference=pd.read_parquet(out/'representation_predictions.parquet');reference_checks=0
    for k in range(5):
        cohort=None
        for name,X in reps.items():
            card=joblib.load(exports/'reference_models'/f'outer{k}_{name}.joblib')
            ids=[set(card[key]) for key in ['fit_ids','calibration_ids','hold_ids']]
            assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
            component_sets=[set(indexed.loc[list(s)].patient_component) for s in ids]
            assert not component_sets[0]&component_sets[1] and not component_sets[0]&component_sets[2] and not component_sets[1]&component_sets[2]
            if cohort is None:cohort=ids
            else:assert ids==cohort
            position=pd.Series(np.arange(len(d)),index=d.record_id)
            fit=X.iloc[position.loc[card['fit_ids']].to_numpy()]
            medians=fit.median().fillna(0).to_numpy();np.testing.assert_allclose(card['model'][0].statistics_,medians)
            xx=X.iloc[position.loc[card['hold_ids'][:17]].to_numpy()]
            p=card['calibrator'].transform(card['model'].predict_proba(xx))[:,1]
            stored=reference[(reference.label==f'outer{k}')&(reference.representation==name)].set_index('record_id').loc[card['hold_ids'][:17],'p']
            np.testing.assert_allclose(p,stored,rtol=0,atol=1e-12);reference_checks+=1
    return dict(source_records=len(d),final_models=final_models,outer_bundles=checked_outer_bundles,
        checked_report_model_members=checked_members,reference_models=reference_checks,
        no_missing_inference=True,no_duplicate_variability_units=True,no_quality_duration_in_report_X=True,
        actual_aggregate_calibration_disjoint=True,reference_cohorts_identical=True,clinical_validation=False)


def validate():
    archive=json.loads((REVIEW/'before_revision_manifest.json').read_text())
    for old,entry in archive.items():assert sha(ROOT/entry['archive'])==entry['sha256'],old
    # Historical binary models stay intact; historical notebook hashes resolve to the archive.
    frozen=json.loads((ROOT/'research/experiments/jnu_ctg/classifiers_v1/selection_frozen.json').read_text());hashes(frozen,'export_sha256')
    old=json.loads((ROOT/'research/experiments/ctg_uci/audited_v2/validation_receipt.json').read_text())
    for relative,expected in old['artifact_sha256'].items():
        if relative.endswith(('.joblib','.cbm')):assert sha(ROOT/relative)==expected,relative
    uci=uci_validate();jnu=jnu_validate()
    result=dict(passed=True,review_points=23,archived_notebooks=12,archived_documents=len(archive),
        uci=uci,jnu=jnu,historical_model_binaries_preserved=True,
        checks_scope='specified numerical/group/input/export invariants; not exhaustive proof of scientific correctness',
        clinical_validation=False)
    (REVIEW/'validation_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':validate()
