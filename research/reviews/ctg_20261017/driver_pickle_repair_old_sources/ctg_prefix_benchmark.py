"""UCI v3: repeated nested prefix-proxy CV, all observations already seen.

This is an exploratory benchmark revision, not a new independent test.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,importlib.metadata,time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.features.ctg_preprocessing import CTG_FEATURES
from src.training.ctg_review_evidence import ROOT,uci_groups,write_json
from src.training.ctg_audit_folds import balanced_group_folds
from src.training.ctg_review_models import PrefixCTGModel,FixedTemperature,multiclass_scores,select_one_se

OUT=ROOT/'research/experiments/ctg_uci/reviewed_v3'
EXPORT=ROOT/'models/exported/ctg_uci/reviewed_v3'
SEED=20261018
FAMILIES=['logistic','A_multiclass','B_sqrt','B_balanced','C_ovr','D_hierarchy']
CAT_GRID=list(ParameterGrid(dict(depth=[3,5,7],learning_rate=[.03,.1],l2_leaf_reg=[3,10],iterations=[300,600])))
LOG_GRID=[{'C':v} for v in [.1,1.,10.]]
REPEATS=3


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def folds(d,n,seed):
    return balanced_group_folds(d.assign(source_file=d.review_group),n,seed)


def complexity(row):
    p=row['params'];family=row['family']
    if family=='logistic':return (0,p['C'])
    rank={'A_multiclass':1,'B_sqrt':2,'B_balanced':3,'D_hierarchy':4,'C_ovr':5}[family]
    return (rank,p['depth'],p['iterations'],-p['l2_leaf_reg'],p['learning_rate'])


def search(d,family,cv,seed,cache,label):
    directory=cache/label/family;directory.mkdir(parents=True,exist_ok=True)
    grid=LOG_GRID if family=='logistic' else list(ParameterGrid(dict(depth=[3,5,7],learning_rate=[.03,.1],l2_leaf_reg=[3,10])))
    def evaluate(item):
        number,params=item;path=directory/f'{number:03d}.joblib'
        if path.exists():return joblib.load(path)
        limits=[0] if family=='logistic' else [300,600]
        proba={limit:np.full((len(d),3),np.nan) for limit in limits};losses={limit:[] for limit in limits}
        for k,(fit,hold) in enumerate(cv):
            f=d.iloc[fit];h=d.iloc[hold];cfg=params if family=='logistic' else {**params,'iterations':600}
            model=PrefixCTGModel(family,cfg,seed+k).fit(f[list(CTG_FEATURES)],f.NSP.to_numpy(int)-1,f.review_group,f.source_excel_row)
            assert not set(model.fit_groups_)&set(h.review_group)
            for limit in limits:
                p=model.predict_proba(h[list(CTG_FEATURES)],ntree_end=limit)
                proba[limit][hold]=p
                losses[limit].append(multiclass_scores(h.NSP.to_numpy(int)-1,p,h.review_group)['group_log_loss'])
        result=[]
        for limit in limits:
            cfg=params if family=='logistic' else {**params,'iterations':limit};values=np.array(losses[limit])
            result.append(dict(family=family,params=cfg,mean_loss=float(values.mean()),sd_loss=float(values.std(ddof=1)),folds=len(cv),fold_losses=values.tolist(),p=proba[limit]))
        joblib.dump(result,path,compress=3);return result
    result=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluate,item) for item in enumerate(grid)]):result.extend(future.result())
    result.sort(key=lambda r:json.dumps(r['params'],sort_keys=True))
    chosen,cutoff=select_one_se(result,complexity)
    return chosen,[{k:v for k,v in r.items() if k!='p'} for r in result],cutoff


class PrefixBundle:
    def __init__(self,model,calibrator):self.model=model;self.calibrator=calibrator
    def predict_proba(self,X):return self.calibrator.transform(self.model.predict_proba(X))


def choose_family(candidates):return select_one_se(candidates,complexity)[0]['family']


def run():
    for p in [OUT,EXPORT]:p.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'src/training/ctg_review_models.py',ROOT/'src/training/ctg_review_evidence.py',ROOT/'data/processed/ctg_uci/ctg_primary.parquet']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','catboost']}
    fingerprint=hashlib.sha256(json.dumps({'hashes':hashes,'versions':versions},sort_keys=True).encode()).hexdigest()
    if (OUT/'completed.json').exists():
        done=json.loads((OUT/'completed.json').read_text())
        if done['fingerprint']!=fingerprint:raise RuntimeError('Completed revision changed: create a new version')
        for relative,expected in done['artifacts_sha256'].items():assert digest(ROOT/relative)==expected,relative
        return pd.read_csv(OUT/'repeat_metrics.csv')
    cache=ROOT/'tmp/ctg_prefix_reviewed_v3'/fingerprint[:16];cache.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',dict(fingerprint=fingerprint,input_sha256=hashes,environment=versions,
        grouping='initials-like prefix proxy; S+digits kept; union exact feature duplicates; identity unverified',
        repeats=REPEATS,outer=5,inner=3,cat_grid=CAT_GRID,logistic_grid=LOG_GRID,
        selector='minimum group-balanced log loss, one-SE parsimony heuristic within each fit',
        calibration='fixed temperature on current fit-side selected-configuration OOF; outer labels never enter calibration',
        decision='argmax; no posthoc threshold tuning',sample_weight='1/current-fit-prefix-segment-count, normalized mean 1',
        evaluation='all 2126 observations already seen; exploratory repeated nested CV, not a new blind test',
        ensemble='single predeclared; previous mode comparison not evidence against all ensembles',clinical_application=False))
    d=uci_groups().reset_index(drop=True);assignments=[];predictions=[];metrics=[];all_choices=[];grids=[]
    for repetition in range(REPEATS):
        outer=folds(d,5,SEED+repetition*100)
        for k,(fit,hold) in enumerate(outer):
            label=f'rep{repetition}_outer{k}';checkpoint=cache/f'{label}.joblib'
            assignments.extend([dict(repetition=repetition,outer_fold=k,source_excel_row=int(d.iloc[i].source_excel_row),review_group=d.iloc[i].review_group) for i in hold])
            if checkpoint.exists():saved=joblib.load(checkpoint)
            else:
                f=d.iloc[fit].reset_index(drop=True);h=d.iloc[hold].reset_index(drop=True)
                inner=folds(f,3,SEED+1000+repetition*100+k);chosen=[];tables=[];blocks=[];rows=[];choices=[]
                for family in FAMILIES:
                    best,table,cutoff=search(f,family,inner,SEED+2000+k,cache,label);chosen.append(best)
                    tables.extend([dict(repetition=repetition,outer_fold=k,**r) for r in table])
                    cal=FixedTemperature().fit(best['p'],f.NSP.to_numpy(int)-1,f.review_group,f.source_excel_row)
                    m=PrefixCTGModel(family,best['params'],SEED+3000+k).fit(f[list(CTG_FEATURES)],f.NSP.to_numpy(int)-1,f.review_group,f.source_excel_row)
                    assert not set(m.fit_groups_)&set(h.review_group)
                    p=cal.transform(m.predict_proba(h[list(CTG_FEATURES)]))
                    b=h[['source_excel_row','source_file','review_group','NSP','CLASS']].copy()
                    for j in range(3):b[f'p{j}']=p[:,j]
                    b['family']=family;b['repetition']=repetition;b['outer_fold']=k;blocks.append(b)
                    rows.append(dict(repetition=repetition,outer_fold=k,family=family,**multiclass_scores(h.NSP.to_numpy(int)-1,p,h.review_group)))
                    choices.append(dict(repetition=repetition,outer_fold=k,family=family,params=best['params'],mean_loss=best['mean_loss'],one_se_cutoff=cutoff,temperature=cal.temperature_,fit_groups=f.review_group.nunique(),hold_groups=h.review_group.nunique()))
                    print(label,family,'complete',flush=True)
                selected=choose_family(chosen)
                procedure=next(b for b in blocks if b.family.iloc[0]==selected).copy();procedure['family']='selected_procedure';blocks.append(procedure)
                rows.append(dict(repetition=repetition,outer_fold=k,family='selected_procedure',**multiclass_scores(h.NSP.to_numpy(int)-1,procedure[['p0','p1','p2']].to_numpy(),h.review_group)))
                for c in choices:c['selected_family']=selected
                saved=(blocks,rows,choices,tables);joblib.dump(saved,checkpoint,compress=3)
            blocks,rows,choices,tables=saved;predictions.extend(blocks);metrics.extend(rows);all_choices.extend(choices);grids.extend(tables)
            write_json(OUT/'progress.json',dict(repetition=repetition,outer_fold=k,stage='nested'))
    pred=pd.concat(predictions,ignore_index=True);pred.to_parquet(OUT/'nested_predictions.parquet',index=False)
    pd.DataFrame(assignments).to_parquet(OUT/'outer_assignments.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False);write_json(OUT/'nested_choices.json',all_choices);write_json(OUT/'grid_results.json',grids)
    repeat=[]
    for (rep,family),g in pred.groupby(['repetition','family']):
        assert len(g)==2126 and g.source_excel_row.is_unique
        repeat.append(dict(repetition=rep,family=family,**multiclass_scores(g.NSP.to_numpy(int)-1,g[['p0','p1','p2']].to_numpy(),g.review_group)))
    result=pd.DataFrame(repeat);result.to_csv(OUT/'repeat_metrics.csv',index=False)
    # Export after evaluation; its fit uses all data, so it has no independent test score.
    final_cv=folds(d,3,SEED+9000);final=[];final_models={}
    for family in FAMILIES:
        best,table,cutoff=search(d,family,final_cv,SEED+10000,cache,'final_all_data');final.append(best)
        write_json(OUT/f'final_grid_{family}.json',table)
        cal=FixedTemperature().fit(best['p'],d.NSP.to_numpy(int)-1,d.review_group,d.source_excel_row)
        m=PrefixCTGModel(family,best['params'],SEED+11000).fit(d[list(CTG_FEATURES)],d.NSP.to_numpy(int)-1,d.review_group,d.source_excel_row)
        bundle=PrefixBundle(m,cal);joblib.dump(bundle,EXPORT/f'{family}.joblib',compress=3);final_models[family]=bundle
        for j,model in enumerate(m.models_):
            if family!='logistic':model.save_model(EXPORT/f'{family}_{j}.cbm')
        if family=='logistic':joblib.dump(m.transformer_,EXPORT/'logistic_preprocessor.joblib')
    champion=choose_family(final)
    write_json(EXPORT/'model_card.json',dict(selected_family=champion,clinical_application=False,benchmark_only=True,
        target=['normal','suspect','pathologic'],inputs=list(CTG_FEATURES),grouping='prefix proxy, not verified patient ID',
        selection='one-SE group log loss on all-data inner CV',final_model_independent_test=False,
        corrected_procedure_evaluation='3x5 outer / 3 inner CV, post-review exploratory',
        family_choices=[{k:v for k,v in r.items() if k!='p'} for r in final]))
    files=list(OUT.glob('*.csv'))+list(OUT.glob('*.parquet'))+list(OUT.glob('*choices.json'))+list(EXPORT.glob('*'))
    write_json(OUT/'completed.json',dict(fingerprint=fingerprint,selected_family=champion,clinical_validation=False,
        artifacts_sha256={str(p.relative_to(ROOT)):digest(p) for p in files if p.is_file()}))
    print('UCI repeated nested prefix benchmark completed; selected',champion,flush=True)
    return result


if __name__=='__main__':run()
