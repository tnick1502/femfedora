"""JNU v2: strict single-unit report profiles, honest mode holdout, signal references.

Post-review exploratory nested CV; the old test is not presented as unseen.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid,StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features.ctg_report_contract import PROFILES,CONTRACT,evaluate_report,research_patterns
from src.features.jnu_review_representations import raw_time_features
from src.training.ctg_review_evidence import ROOT,write_json
from src.training.ctg_review_models import StrictBinaryModel,FixedTemperature,weights,binary_scores,select_one_se
from src.training.jnu_report_models import SigmoidCalibration

OUT=ROOT/'research/experiments/jnu_ctg/reviewed_v2'
EXPORT=ROOT/'models/exported/jnu_ctg/reviewed_v2'
SEED=20261018
FAMILIES=['linear','spline','catboost']
CAT=list(ParameterGrid(dict(depth=[3,5],l2_leaf_reg=[3,10],iterations=[300,700],auto_class_weights=[None,'Balanced'])))
GRID={'catboost':[{**p,'learning_rate':.05} for p in CAT],
    'spline':list(ParameterGrid(dict(n_knots=[3,5],C=[.3,3.,30.]))),'linear':[{'C':v} for v in [.1,1.,10.]]}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_data():
    d=pd.read_parquet(ROOT/'data/processed/jnu_ctg/v1/splits.parquet')
    m=pd.read_parquet(ROOT/'data/processed/jnu_ctg/report_features_v1/measurements.parquet')
    return d.merge(m,on='record_id',validate='one_to_one').reset_index(drop=True)


def xframe(d,profile):return d[PROFILES[profile]].astype(float)
def eligible(d,profile):return np.isfinite(xframe(d,profile)).all(axis=1).to_numpy()
def folds(d,n,seed):
    cv=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(d,d.figo_label,d.patient_component))
    for a,b in cv:
        assert not set(d.iloc[a].patient_component)&set(d.iloc[b].patient_component)
        assert set(d.iloc[a].figo_label)=={0,1} and set(d.iloc[b].figo_label)=={0,1}
    return cv


def complexity(r):
    p=r['params'];rank={'linear':0,'spline':1,'catboost':2}[r['family']]
    if r['family']=='catboost':return (rank,p['depth'],p['iterations'],-p['l2_leaf_reg'],p['auto_class_weights'] is not None)
    return (rank,p.get('n_knots',0),p['C'])


def fit_model(d,profile,family,params,seed):
    return StrictBinaryModel(family,params,seed).fit(xframe(d,profile),d.figo_label,d.patient_component,d.record_id)


def search(d,profile,family,cv,seed,cache,label):
    folder=cache/label/profile/family;folder.mkdir(parents=True,exist_ok=True)
    candidates=GRID[family]
    def evaluate(item):
        j,params=item;path=folder/f'{j:03d}.joblib'
        if path.exists():return joblib.load(path)
        p=np.full(len(d),np.nan);losses=[]
        for k,(fit,hold) in enumerate(cv):
            f=d.iloc[fit];h=d.iloc[hold];m=fit_model(f,profile,family,params,seed+k)
            assert not set(m.fit_groups_)&set(h.patient_component)
            p[hold]=m.predict_proba(xframe(h,profile))[:,1]
            losses.append(binary_scores(h.figo_label,p[hold],h.patient_id)['group_log_loss'])
        r=dict(family=family,params=params,mean_loss=float(np.mean(losses)),sd_loss=float(np.std(losses,ddof=1)),folds=len(cv),fold_losses=losses)
        joblib.dump(r,path,compress=3);return r
    result=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluate,item) for item in enumerate(candidates)]):result.append(future.result())
    result.sort(key=lambda r:json.dumps(r['params'],sort_keys=True))
    best,cutoff=select_one_se(result,complexity);return best,result,cutoff


def aggregate_fit(d,profile,family,params,mode,seed):
    if mode=='single':return [fit_model(d,profile,family,params,seed)]
    return [fit_model(d.iloc[fit],profile,family,params,seed+k) for k,(fit,_) in enumerate(folds(d,5,seed+20))]


def aggregate_predict(models,X):return np.mean([m.predict_proba(X) for m in models],axis=0)


class BinarySigmoid:
    def fit(self,p,y,groups,ids):
        self.mapping_=SigmoidCalibration().fit(p[:,1],y,groups);self.fit_ids_=np.asarray(ids);return self
    def transform(self,p):
        positive=self.mapping_.transform(p[:,1]);return np.column_stack([1-positive,positive])


class StrictReportBundle:
    def __init__(self,profile,family,models,calibrator,mode):
        self.profile=profile;self.family=family;self.models=models;self.calibrator=calibrator;self.mode=mode;self.contract=CONTRACT
    def predict_proba(self,X):return self.calibrator.transform(aggregate_predict(self.models,X))
    def assess_report(self,report,mode='application'):return evaluate_report(report,self,self.profile,mode=mode)


def train_procedure(d,profile,cache,label):
    """Calibration is disjoint; mode holdout never participates in grid search."""
    accepted=d.loc[eligible(d,profile)].reset_index(drop=True)
    proper_index,cal_index=folds(accepted,5,SEED+700)[0]
    proper=accepted.iloc[proper_index].reset_index(drop=True);cal=accepted.iloc[cal_index].reset_index(drop=True)
    tune_index,mode_index=folds(proper,5,SEED+800)[0]
    tune=proper.iloc[tune_index].reset_index(drop=True);mode_hold=proper.iloc[mode_index].reset_index(drop=True)
    cv=folds(tune,3,SEED+900);selected=[];bundles={};choices=[];grids=[]
    for family in FAMILIES:
        best,table,cutoff=search(tune,profile,family,cv,SEED+1000,cache,label);selected.append(best)
        grids.extend([dict(profile=profile,label=label,**r) for r in table]);mode='single';mode_scores=[]
        if family!='linear':
            for candidate in ['single','ensemble5']:
                members=aggregate_fit(tune,profile,family,best['params'],candidate,SEED+1100)
                p=aggregate_predict(members,xframe(mode_hold,profile))[:,1]
                mode_scores.append(dict(mode=candidate,**binary_scores(mode_hold.figo_label,p,mode_hold.patient_id)))
            a,b=mode_scores
            if b['group_macro_f1']>=a['group_macro_f1']+.005 and b['group_log_loss']<=a['group_log_loss']+.005:mode='ensemble5'
        models=aggregate_fit(proper,profile,family,best['params'],mode,SEED+1200)
        assert all(not set(m.fit_groups_)&set(cal.patient_component) for m in models)
        p=aggregate_predict(models,xframe(cal,profile))
        calibrator=BinarySigmoid().fit(p,cal.figo_label,cal.patient_component,cal.record_id)
        bundles[family]=StrictReportBundle(profile,family,models,calibrator,mode)
        choices.append(dict(label=label,profile=profile,family=family,params=best['params'],mode=mode,
            inner_loss=best['mean_loss'],one_se_cutoff=cutoff,mode_holdout_scores=mode_scores,
            proper_fit_records=len(proper),calibration_records=len(cal),mode_selection_records=len(mode_hold),
            calibration_slope=calibrator.mapping_.slope_,calibration_intercept=calibrator.mapping_.intercept_))
        print(label,profile,family,mode,'complete',flush=True)
    champion=select_one_se(selected,complexity)[0]['family']
    for r in choices:r['selected_family']=champion
    return bundles,choices,grids,champion


def representations(d):
    base=ROOT/'data/processed/jnu_ctg/v1';simple=pd.read_parquet(base/'signal_features.parquet').set_index('record_id').loc[d.record_id]
    pub=pd.read_parquet(base/'metadata_clean.parquet').set_index('record_id').loc[d.record_id]
    raw=raw_time_features().set_index('record_id').loc[d.record_id]
    result={'report5':xframe(d,'report5').reset_index(drop=True),
        'fhr_std_only':simple[['fhr_bpm_std']].reset_index(drop=True),
        'signal27':simple.reset_index(drop=True),'raw_minute210':raw.reset_index(drop=True),
        'published_F_quarantined':pub[[c for c in pub if c.startswith('F_')]].apply(pd.to_numeric,errors='raise').reset_index(drop=True),
        'quality_only':d[['signal_loss_pct']].reset_index(drop=True),
        'report5_plus_quality':d[PROFILES['report5']+['signal_loss_pct']].reset_index(drop=True)}
    return result


def representation_fold(d,fit,hold,reps,cache,label):
    rows=[];predictions=[];grids=[];common=eligible(d,'report5')
    # Same common assessment and fit cohort for all representations in this comparison.
    f=d.iloc[fit].loc[common[fit]].copy().reset_index().rename(columns={'index':'original_position'})
    h=d.iloc[hold].loc[common[hold]].copy().reset_index().rename(columns={'index':'original_position'})
    a,b=folds(f,5,SEED+700)[0];proper=f.iloc[a].reset_index(drop=True);cal=f.iloc[b].reset_index(drop=True)
    inner=folds(proper,3,SEED+1400)
    for name,X in reps.items():
        path=cache/f'{label}_representation_{name}.joblib'
        if path.exists():r=joblib.load(path)
        else:
            configurations=list(ParameterGrid(dict(max_leaf_nodes=[7,15],l2_regularization=[1.,10.])))
            def make(params):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingClassifier(**params,learning_rate=.05,max_iter=250,early_stopping=False,random_state=SEED))
            def evaluate(params):
                losses=[]
                for fit_i,hold_i in inner:
                    ff=proper.iloc[fit_i];hh=proper.iloc[hold_i];m=make(params)
                    m.fit(X.iloc[ff.original_position],ff.figo_label,histgradientboostingclassifier__sample_weight=weights(ff.patient_component,True))
                    p=m.predict_proba(X.iloc[hh.original_position])[:,1]
                    losses.append(binary_scores(hh.figo_label,p,hh.patient_id)['group_log_loss'])
                return dict(params=params,mean_loss=float(np.mean(losses)),sd_loss=float(np.std(losses,ddof=1)),folds=3)
            with ThreadPoolExecutor(max_workers=4) as pool:table=list(pool.map(evaluate,configurations))
            best,_=select_one_se(table,lambda r:(r['params']['max_leaf_nodes'],-r['params']['l2_regularization']))
            m=make(best['params']);m.fit(X.iloc[proper.original_position],proper.figo_label,histgradientboostingclassifier__sample_weight=weights(proper.patient_component,True))
            pc=m.predict_proba(X.iloc[cal.original_position]);calibrator=BinarySigmoid().fit(pc,cal.figo_label,cal.patient_component,cal.record_id)
            p=calibrator.transform(m.predict_proba(X.iloc[h.original_position]))[:,1]
            reference_dir=EXPORT/'reference_models';reference_dir.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(model=m,calibrator=calibrator,columns=list(X),
                fit_ids=proper.record_id.to_numpy(),calibration_ids=cal.record_id.to_numpy(),hold_ids=h.record_id.to_numpy()),
                reference_dir/f'{label}_{name}.joblib',compress=3)
            metrics=dict(representation=name,**binary_scores(h.figo_label,p,h.patient_id))
            pred=h[['record_id','patient_id','patient_component','figo_label']].copy();pred['p']=p;pred['representation']=name
            r=(metrics,pred,table);joblib.dump(r,path,compress=3)
        metric,pred,table=r;rows.append(dict(label=label,**metric));predictions.append(pred.assign(label=label));grids.extend([dict(label=label,representation=name,**v) for v in table])
        print(label,'representation',name,'complete',flush=True)
    return rows,predictions,grids


def run():
    for p in [OUT,EXPORT]:p.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'src/training/ctg_review_models.py',ROOT/'src/features/ctg_report_contract.py',
        ROOT/'src/features/jnu_review_representations.py',ROOT/'src/training/jnu_report_models.py',
        ROOT/'data/processed/jnu_ctg/report_features_v1/measurements.parquet',ROOT/'data/processed/jnu_ctg/v1/splits.parquet']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    environment={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','catboost']}
    fingerprint=hashlib.sha256(json.dumps(dict(hashes=hashes,environment=environment),sort_keys=True).encode()).hexdigest()
    if (OUT/'completed.json').exists():
        done=json.loads((OUT/'completed.json').read_text());assert done['fingerprint']==fingerprint,'Completed version changed'
        for relative,expected in done['artifacts_sha256'].items():assert digest(ROOT/relative)==expected,relative
        return pd.read_csv(OUT/'profile_metrics.csv')
    cache=ROOT/'tmp/jnu_reviewed_v2'/fingerprint[:16];cache.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',dict(fingerprint=fingerprint,input_sha256=hashes,environment=environment,profiles=PROFILES,grids=GRID,
        outer_folds=5,inner_folds=3,eligibility='complete required profile measurements; coverage over every source row',
        missingness='no imputation or indicators in report models; missing/unconfirmed required fields abstain before inference',
        removed_inputs=['signal_loss_pct','duration_min','duplicate variability units','unverified episode rules'],
        selector='one-SE group-log-loss parsimony heuristic',
        mode_selection='separate grouped 20% mode holdout excluded from grid; compare chosen params at identical holdout',
        calibration='separate 20% outer-fit patients, actual selected aggregate, positive sigmoid with intercept',
        decision='fixed 0.5; no post-review threshold tuning',
        evaluation='all JNU data already seen; exploratory post-review grouped nested CV; no new blind test',
        signal_reference='same outer patient folds/common report5 cohort/HGB grid; not a mathematical or clinical ceiling',
        clinical_validation=False))
    d=load_data();outer=folds(d,5,SEED+100);all_predictions=[];choices=[];grids=[];assignments=[];coverage=[];reference_metrics=[];reference_predictions=[];reference_grids=[]
    reps=representations(d)
    for k,(fit,hold) in enumerate(outer):
        f=d.iloc[fit].reset_index(drop=True);h=d.iloc[hold].reset_index(drop=True);label=f'outer{k}'
        assignments.extend([dict(record_id=r.record_id,patient_component=r.patient_component,outer_fold=k) for r in h.itertuples()])
        for profile in PROFILES:
            checkpoint=cache/f'{label}_{profile}.joblib'
            if checkpoint.exists():bundles,c,g,champion=joblib.load(checkpoint)
            else:
                bundles,c,g,champion=train_procedure(f,profile,cache,label);joblib.dump((bundles,c,g,champion),checkpoint,compress=3)
            choices.extend(c);grids.extend(g)
            valid=eligible(h,profile)
            for family,bundle in bundles.items():
                p=np.full(len(h),np.nan);p[valid]=bundle.predict_proba(xframe(h.loc[valid],profile))[:,1]
                frame=h[['record_id','patient_id','patient_component','figo_label','split','report_quality_eligible']].copy()
                frame['accepted']=valid;frame['p']=p;frame['family']=family;frame['profile']=profile;frame['outer_fold']=k;all_predictions.append(frame)
                if family==champion:all_predictions.append(frame.assign(family='selected_procedure'))
            coverage.append(dict(outer_fold=k,profile=profile,records=len(h),accepted=int(valid.sum()),normal_records=int((h.figo_label==0).sum()),
                abnormal_records=int(h.figo_label.sum()),abstain_normal=int(((~valid)&(h.figo_label==0)).sum()),abstain_abnormal=int(((~valid)&(h.figo_label==1)).sum())))
        r,p,g=representation_fold(d,fit,hold,reps,cache,label);reference_metrics.extend(r);reference_predictions.extend(p);reference_grids.extend(g)
        write_json(OUT/'progress.json',dict(stage='nested',outer_fold=k))
    pred=pd.concat(all_predictions,ignore_index=True);pred.to_parquet(OUT/'nested_predictions.parquet',index=False)
    pd.DataFrame(assignments).to_parquet(OUT/'outer_assignments.parquet',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage_by_class.csv',index=False);write_json(OUT/'choices.json',choices);write_json(OUT/'grid_results.json',grids)
    rows=[]
    for (profile,family),g in pred.groupby(['profile','family']):
        assert len(g)==len(d) and g.record_id.is_unique
        z=g[g.accepted]
        rows.append(dict(profile=profile,family=family,coverage=len(z)/len(g),abstained=len(g)-len(z),**binary_scores(z.figo_label,z.p,z.patient_id)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'profile_metrics.csv',index=False)
    ref=pd.concat(reference_predictions,ignore_index=True);ref.to_parquet(OUT/'representation_predictions.parquet',index=False)
    pd.DataFrame(reference_metrics).to_csv(OUT/'representation_fold_metrics.csv',index=False);write_json(OUT/'representation_grid_results.json',reference_grids)
    summary=[]
    for name,g in ref.groupby('representation'):summary.append(dict(representation=name,**binary_scores(g.figo_label,g.p,g.patient_id)))
    pd.DataFrame(summary).to_csv(OUT/'representation_metrics.csv',index=False)
    # Final bundles use held-out calibration patients and have no independent test score.
    final=[];patterns=[]
    for profile in PROFILES:
        checkpoint=cache/f'final_{profile}.joblib'
        if checkpoint.exists():bundles,c,g,champion=joblib.load(checkpoint)
        else:
            bundles,c,g,champion=train_procedure(d,profile,cache,'final_all_known_data');joblib.dump((bundles,c,g,champion),checkpoint,compress=3)
        final.extend(c)
        for family,bundle in bundles.items():
            target=EXPORT/f'{profile}_{family}.joblib';joblib.dump(bundle,target,compress=3)
            for j,model in enumerate(bundle.models):
                if family=='catboost':model.model_.save_model(EXPORT/f'{profile}_{family}_{j}.cbm')
                else:joblib.dump(dict(scaler=model.scaler_,splines=model.splines_,columns=model.columns_),EXPORT/f'{profile}_{family}_{j}_preprocessor.joblib')
            for name,report in research_patterns().items():
                if profile=='report4s5':report={**report,'definitions':{**report['definitions'],'stv_epoch_s':4}}
                audit=bundle.assess_report(report,mode='research');app=bundle.assess_report(report)
                assert app['probability_expert_abnormal'] is None
                patterns.append({**audit,'profile':profile,'family':family,'pattern':name})
        write_json(EXPORT/f'{profile}_model_card.json',dict(profile=profile,selected_family=champion,contract=CONTRACT,columns=PROFILES[profile],
            clinical_validation=False,external_bridge=False,missing_required='abstain; no prediction',
            duration_min=30,target='expert FIGO normal/abnormal, not prognosis',final_fit_test_independent=False,
            family_choices=c))
    write_json(OUT/'final_choices.json',final);write_json(OUT/'synthetic_pattern_probes.json',patterns)
    artifacts=list(OUT.glob('*.csv'))+list(OUT.glob('*.parquet'))+list(OUT.glob('*choices.json'))+list(OUT.glob('*probes.json'))+list(EXPORT.rglob('*'))
    write_json(OUT/'completed.json',dict(fingerprint=fingerprint,clinical_validation=False,models=12,
        artifacts_sha256={str(p.relative_to(ROOT)):digest(p) for p in artifacts if p.is_file()}))
    print('JNU strict report models and same-fold signal references completed',flush=True);return result


if __name__=='__main__':
    # Avoid pickling driver-local classes as __main__.StrictReportBundle.
    from src.training.jnu_review_experiment import run as imported_run
    imported_run()
