"""Static report figures (PNG, light surface). Palette: dataviz reference slots 1-3 + neutral gray."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "research/experiments/ring_health/v3"
FIG = EXP / "figures"
C1, C2, C3, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#d8d6d0", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "legend.frameon": False})


def fig_cycle_example():
    from src.ring_health_v3.cycle_engine import CycleModel
    from src.ring_health_v3.fertility import analyze_fertility
    from src.ring_health_v3.sim_cycles import sample_person, simulate_person
    P = json.loads((ROOT / "models/exported/ring_health/v3/cycle_engine_params.json").read_text())
    m = CycleModel(P)
    for seed in range(31, 600):  # typical wear (~92%), ovulatory last cycle, no fever in it
        rng = np.random.default_rng(seed)
        per = sample_person(rng, age=30.0, irregular=False, overrides={"long_gap": False, "log_reliability": 1.0, "adherence": 0.92})
        if not (14.5 <= per["mu_f"] <= 16.5 and per["temp_amp"] >= 0.25):
            continue
        sim = simulate_person(rng, n_cycles=6, person=per)
        t = sim["truth"]; d = sim["daily"]; cyc = t[t.cycle_index == 5]
        if cyc.ovulatory.iloc[0] and abs(cyc.ovulation_day.iloc[0] - per["mu_f"]) <= 2 and not cyc.fever.any() and d.loc[cyc.index, "temperature_deviation"].notna().mean() >= 0.85:
            break
    probs, dates, det_day = [], [], None
    for i in cyc.index:
        r = analyze_fertility(m, d.iloc[:i + 1], [s for s in sim["logged_starts"] if s <= d.date.iloc[i]], d.date.iloc[i], age=30)
        probs.append(r["fertile_window"]["today_probability"]); dates.append(d.date.iloc[i])
        if det_day is None and r["ovulation"]["status"] == "detected":
            det_day = (d.date.iloc[i], pd.Timestamp(r["ovulation"]["date"]))
    x = (cyc.cycle_day).to_numpy()
    O = int(cyc.ovulation_day.iloc[0])
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1]})
    ax[0].axvspan(O - 5 - 0.5, O + 0.5, color=C3, alpha=0.12, lw=0)
    ax[0].plot(x, probs, color=C1, lw=2)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("P(fertile today)")
    ax[0].set_title("A synthetic cycle as the app sees it each morning")
    ax[0].text(O - 5, 0.92, "true fertile window", color=INK2, fontsize=9)
    ax[1].plot(x, d.loc[cyc.index, "temperature_deviation"], color=C2, lw=2, marker="o", ms=3)
    ax[1].set_ylabel("temperature dev., °C")
    ax[2].plot(x, d.loc[cyc.index, "resting_hr"], color=C1, lw=2, marker="o", ms=3)
    ax[2].set_ylabel("resting HR, bpm"); ax[2].set_xlabel("cycle day")
    for a in ax:
        a.axvline(O, color=INK2, lw=1, ls="--")
    ax[1].text(O + 0.3, ax[1].get_ylim()[1] * 0.85, "true ovulation", color=INK2, fontsize=9)
    if det_day:
        dd = int((det_day[0] - cyc.date.iloc[0]).days) + 1; est = int((det_day[1] - cyc.date.iloc[0]).days) + 1
        ax[0].annotate(f"detected on day {dd}\n(estimate: day {est})", xy=(dd, probs[dd - 1]), xytext=(dd + 1.5, 0.55), color=INK, fontsize=9,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.tight_layout(); fig.savefig(FIG / "cycle_example.png", dpi=160); plt.close(fig)


def fig_scenarios():
    r = json.loads((EXP / "test_cycle_metrics.json").read_text())
    pr = EXP / "test_cycle_metrics_paired.json"
    if pr.exists():  # masking scenarios on the same women; subgroups from the main test
        paired = json.loads(pr.read_text())
        r = {**{k: r[k] for k in ["irregular", "age40_45"]}, **{k: v for k, v in paired.items() if not k.startswith("_")}}
    order = ["realistic", "missing30", "missing50", "block7_ovulation", "temperature_only", "no_temperature", "cold_start", "irregular", "age40_45", "lh_users", "missed_logs", "calendar_only", "mismatch"]
    names = {"realistic": "realistic", "missing30": "+30% nights missing", "missing50": "+50% nights missing", "block7_ovulation": "7-night gap at ovulation",
             "temperature_only": "temperature only", "no_temperature": "no temperature", "cold_start": "new user (no history)", "irregular": "irregular cycles",
             "age40_45": "age 40-45", "lh_users": "with LH tests", "missed_logs": "25% periods not logged", "calendar_only": "no ring data (calendar)", "mismatch": "mismatched physiology"}
    rows = [(names[s], r[s]["days"]["auroc"], (r[s].get("retrospective") or {}).get("within2_all_ovulatory")) for s in order if s in r]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    y = np.arange(len(rows))[::-1]
    ax[0].barh(y, [a for _, a, _ in rows], color=C1, height=0.55)
    for yi, (_, a, _) in zip(y, rows):
        ax[0].text(a + 0.005, yi, f"{a:.2f}", va="center", color=INK, fontsize=9)
    ax[0].set_xlim(0.5, 1.0); ax[0].set_yticks(y, [n for n, _, _ in rows]); ax[0].set_title("Fertile window today: AUROC")
    vals = [b if b is not None else np.nan for _, _, b in rows]
    ax[1].barh(y, np.nan_to_num(vals), color=C3, height=0.55)
    for yi, b in zip(y, vals):
        ax[1].text((b if np.isfinite(b) else 0) + 0.01, yi, "n/a" if not np.isfinite(b) else f"{b:.0%}", va="center", color=INK, fontsize=9)
    ax[1].set_xlim(0, 1.05); ax[1].set_title("Past cycles: ovulation found within ±2 d")
    ax[1].axvline(0.879, color=GRAY, lw=1, ls="--"); ax[1].text(0.885, y[-1] - 0.9, "Oura 2025: 87.9%", color=INK2, fontsize=8)
    fig.suptitle("Robustness on held-out synthetic women (scenarios paired on the same 40 women; subgroups separate)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "cycle_scenarios.png", dpi=160); plt.close(fig)


def fig_menopause():
    r = json.loads((EXP / "test_menopause_metrics.json").read_text())["test"]["random"]
    hz = [6, 12, 24, 36, 48, 60]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name, col, lab in [("age_only", GRAY, "age only"), ("age_bleeding", C2, "age + bleeding pattern"), ("full", C1, "age + bleeding + ring + symptoms")]:
        v = [r[name][f"fmp_auc_{h}m_pre_fmp_42_56"] for h in hz]
        ax.plot(hz, v, color=col, lw=2, marker="o", ms=6)
        ax.text(hz[-1] + 1.2, v[-1], lab, color=INK, fontsize=9, va="center")
    ax.scatter([12, 24, 36], [0.881, 0.891, 0.896], color=INK, marker="D", s=28, zorder=5)
    ax.text(12, 0.905, "Finkelstein 2020 (SWAN, AMH+age+BMI)", color=INK2, fontsize=8)
    ax.set_xlim(4, 85); ax.set_ylim(0.6, 1.0); ax.set_xlabel("horizon, months"); ax.set_ylabel("AUROC (pre-FMP women 42-56)")
    ax.set_title("Final menstrual period within N months")
    fig.tight_layout(); fig.savefig(FIG / "menopause_fmp_auc.png", dpi=160); plt.close(fig)


def fig_generator_checks():
    nh = json.loads((EXP / "nhanes_age_at_menopause.json").read_text())
    chk = json.loads((EXP / "generator_checks.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ages = np.array(sorted(int(a) for a in nh["km_survival_by_age"])); km = np.array([nh["km_survival_by_age"][str(a)] for a in ages])
    ax[0].step(ages, 1 - km, where="post", color=GRAY, lw=2)
    ga = np.array(chk["fmp_cdf"]["ages"]); gc = np.array(chk["fmp_cdf"]["cdf"])
    ax[0].plot(ga, gc, color=C1, lw=2)
    ax[0].text(53.5, 0.55, "generator", color=C1, fontsize=9); ax[0].text(41, 0.30, "NHANES 2015-18\n(real, weighted KM)", color=INK2, fontsize=9)
    ax[0].set_xlabel("age"); ax[0].set_ylabel("share with natural menopause"); ax[0].set_title("Age at final period")
    ax[0].set_xlim(38, 62)
    e = chk["ela"]
    ax[1].plot(e["years"], e["generator"], color=C1, lw=2, marker="o", ms=5)
    ax[1].scatter([10, 5, 0.5], [1.0, 0.879, 0.228], color=INK, marker="D", s=30, zorder=5)
    ax[1].text(5.3, 0.95, "Santoro 2017 (SWAN DHS)", color=INK2, fontsize=8)
    ax[1].invert_xaxis(); ax[1].set_xlabel("years before final period"); ax[1].set_ylabel("ovulatory cycles"); ax[1].set_ylim(0, 1.08)
    ax[1].set_title("Share of ovulatory cycles")
    fig.tight_layout(); fig.savefig(FIG / "menopause_generator_checks.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in sys.argv[1:] or ["cycle_example", "scenarios", "menopause", "generator_checks"]:
        globals()["fig_" + f]()
        print("ok", f)
