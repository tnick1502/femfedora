"""Build scientific figures, summary tables and a checked result appendix."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/private/tmp/ring_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.ring_health.data import RESULTS,ROOT,save_json


def main():
    r=json.loads((RESULTS/"all_results.json").read_text())
    plots=RESULTS/"figures"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white"})
    rows=[]
    for name,value in r.items():
        models=value.get("test",value.get("models",{}))
        for model,metrics in models.items():
            rows.append({"experiment":name,"provenance":value.get("provenance","real_processed_research_only"),"evaluation":"test" if "test" in value else "grouped_oof", "model":model,"selected_on_validation":model==value.get("selected"),**{k:v for k,v in metrics.items() if not isinstance(v,list)}})
    pd.DataFrame(rows).to_csv(RESULTS/"experiment_summary.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,5.6),layout="constrained")
    for ax,name,title in zip(axes,["jordan_proxy","tempid_fertile_target","tempid_postovulatory_target"],["Jordan: current stage proxy\n199 people; 5-fold grouped CV","TempID: fertile day\n1239 days; 39 inferred people","TempID: postovulatory day\nProcessed signals; not prospective"]):
        models=r[name]["models"]
        values=[models[m]["macro_f1"] for m in models]
        ax.bar(range(4),values,color=["#b7bfc8","#245a81","#3d8c86","#a46f48"])
        ax.set_xticks(range(4),["Prior","Logistic","Boosting","Forest"],rotation=20)
        ax.set_ylim(0,1)
        ax.set_title(title,fontsize=12,pad=14)
        ax.set_ylabel("Macro F1 at fixed 0.5 / argmax")
        for i,v in enumerate(values):ax.text(i,v+.025,f"{v:.3f}",ha="center")
    fig.suptitle("REAL DATA BENCHMARKS — no ring-specific clinical validation",fontsize=15)
    fig.savefig(plots/"real_benchmarks.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout="constrained")
    names=["synthetic_stage_ring_only","synthetic_stage_hybrid"]
    x=np.arange(2)
    for dx,kind,color in [(-.18,"test","#245a81"),(.18,"shifted_synthetic_test","#ca8e53")]:
        vals=[r[n][kind][r[n]["selected"]]["macro_f1"] if kind=="test" else r[n][kind]["macro_f1"] for n in names]
        axes[0].bar(x+dx,vals,.36,label="Same simulator" if kind=="test" else "Shifted simulator",color=color)
        for i,v in enumerate(vals):axes[0].text(i+dx,v+.02,f"{v:.3f}",ha="center")
    axes[0].set_xticks(x,["Ring only","Ring+age"])
    axes[0].set_ylim(0,1)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Current reproductive stage — SYNTHETIC")
    axes[0].legend(loc="lower right")
    fnames=["synthetic_fertility_calendar","synthetic_fertility_ring_only","synthetic_fertility_hybrid","synthetic_fertility_confirmation"]
    vals=[r[n]["test"][r[n]["selected"]]["positive_f1"] for n in fnames]
    axes[1].bar(range(4),vals,color=["#b7bfc8","#245a81","#3d8c86","#a46f48"])
    axes[1].set_xticks(range(4),["Calendar","Ring only","Hybrid","Postovulation"],rotation=18)
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel("Positive-class F1; validation threshold")
    axes[1].set_title("Fertile-day prediction vs later detection — SYNTHETIC")
    for i,v in enumerate(vals):axes[1].text(i,v-.06 if v>.93 else v+.025,f"{v:.3f}",ha="center",color="white" if v>.93 else "black")
    fig.suptitle("Simulator performance measures scenario recovery, not performance in women",fontsize=14)
    fig.savefig(plots/"synthetic_benchmarks.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
    calrows=[]
    for ax,name,target,title in zip(axes,["nhanes_context","synthetic_fertility_hybrid"],["target","fertile_target"],["NHANES 2017-18 current status — REAL","Current fertile day — SYNTHETIC"]):
        d=pd.read_csv(RESULTS/f"{name}_test_predictions.csv")
        obs,prob=calibration_curve(d[target],d.p_1,n_bins=10,strategy="quantile")
        ax.plot([0,1],[0,1],"--",color="#999999",label="Ideal")
        ax.plot(prob,obs,"o-",color="#245a81",label="Heldout predictions")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted probability",ylabel="Observed fraction",title=title)
        ax.legend()
        for a,b in zip(prob,obs):calrows.append({"experiment":name,"mean_prediction":a,"observed_fraction":b})
    fig.suptitle("Calibration diagnostic: 10 equal-count bins; no confidence bands",fontsize=14)
    fig.savefig(plots/"calibration.png",dpi=160)
    pd.DataFrame(calrows).to_csv(RESULTS/"calibration_bins.csv",index=False)
    plt.close(fig)
    fore=json.loads((RESULTS/"bayesian_forecast_metrics.json").read_text())
    fig,ax=plt.subplots(figsize=(9,4.6),layout="constrained")
    hs=list(range(7))
    for key,label,color in [("average_precision","Average precision","#245a81"),("positive_f1","Positive F1 at 0.5","#a46f48")]:
        ax.plot(hs,[fore["horizons"][str(h)][key] for h in hs],"o-",label=label,color=color)
    ax.set(xlabel="Days ahead from prediction origin",ylabel="Score",ylim=(0,1),xticks=hs,title="7-day timing forecast — SYNTHETIC; observed same-cycle horizons")
    ax.legend()
    fig.savefig(plots/"forecast_horizons.png",dpi=160)
    plt.close(fig)
    # Record final implementation and artifact hashes after validation/reporting
    # additions. Training results are unchanged; reproduction is not preregistered.
    env=json.loads((RESULTS/"environment.json").read_text())
    paths=list((ROOT/"src/ring_health").glob("*.py"))+list((ROOT/"tools").glob("*ring_health*.py"))+[ROOT/"tests/unit/test_ring_health.py",ROOT/"models/configs/training/ring_health_simulator_v1.json"]
    env["code_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    env["note"]="Final code fingerprints include inference and validation. Early exploratory results were reviewed; this is a reproducible research benchmark, not a preregistered blinded validation."
    save_json(RESULTS/"environment.json",env)
    registry_path=ROOT/"models/registry/ring_health_v1.json"
    registry=json.loads(registry_path.read_text())
    registry["environment"]=env
    registry["input_contract"]="docs/ring_health_input_contract_ru.md"
    registry["report"]="research/reports/ring_health_mvp_ru.md"
    save_json(registry_path,registry)
    summary=[]
    for n in ["nhanes_age_only","nhanes_context","synthetic_stage_ring_only","synthetic_stage_hybrid","synthetic_fertility_calendar","synthetic_fertility_ring_only","synthetic_fertility_hybrid","synthetic_fertility_confirmation"]:
        v=r[n]
        m=v["test"][v["selected"]]
        summary.append(f"| {n} | {v['selected']} | {m['macro_f1']:.3f} | {m.get('positive_f1',float('nan')):.3f} | {m.get('roc_auc',float('nan')):.3f} | {m['brier']:.3f} |")
    appendix="\n".join(["# Итоговые метрики ring health v1","","Числа сформированы из сохранённых прогнозов и метрик. Для multiclass Brier — сумма по трём классам; это другой масштаб, чем бинарный Brier. NaN означает, что показатель не применим.","","| Эксперимент | Выбранная модель | Macro F1 | F1 положительного класса | ROC AUC | Brier |","|---|---|---:|---:|---:|---:|",*summary,"","Выбор семейства — по validation log loss. Порог бинарной классификации — по validation F1; классы этапов — argmax. Подробные результаты каждого кандидата находятся в experiment_summary.csv."])
    (ROOT/"research/reports/ring_health_metrics_ru.md").write_text(appendix)
    print("Created4figures, summaryCSV, calibrationCSV and result appendix.")


if __name__=="__main__":main()
