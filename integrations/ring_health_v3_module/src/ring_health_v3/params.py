"""Evidence registry for every generator / prior parameter.

kind:
  literature   - value read in the cited primary source
  fitted_real  - estimated by us on a real open dataset (script: tools/fit_real_priors_v3.py)
  derived      - arithmetic on literature values
  assumption   - engineering choice, NOT a population estimate; varied in stress tests
"""
from __future__ import annotations

SRC = {
    "fedcycle": "https://epublications.marquette.edu/data_nfp/7/ (mirror raw CSV: https://raw.githubusercontent.com/chiomajaco6/MenstrualCyclePrediction/main/rawFedCycleData.csv)",
    "fehring2006": "Fehring et al. 2006, JOGNN, 141 women / 1060 cycles (Marquette)",
    "bull2019": "https://www.nature.com/articles/s41746-019-0152-7",
    "maijala2019": "https://doi.org/10.1186/s12905-019-0844-9",
    "oura2025": "https://www.jmir.org/2025/1/e60667",
    "alzueta2024": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11416332/",
    "goodale2019": "https://www.jmir.org/2019/4/e13404/",
    "whoop2026": "https://github.com/alexgonzl/menstrual_cycle_physiology_sleep_whoop (npj Digit Med 2026)",
    "shilaih2017": "https://www.nature.com/articles/s41598-017-01433-9",
    "lynch2014": "BioCycle, Lynch et al. 2014 (anovulation 5.5-12.8% serum definitions)",
    "who1980_behre2000": "WHO 1980 LH->ovulation 32 h median; Behre 2000: 76.3% ovulate the day after urine LH surge",
    "wilcox1995": "https://www.nejm.org/doi/full/10.1056/NEJM199512073332301",
    "gold2013": "https://academic.oup.com/aje/article/178/1/70/108605",
    "interlace2019": "https://eprints.whiterose.ac.uk/151718/",
    "paramsothy2017": "https://escholarship.org/content/qt4pv4j4fr/qt4pv4j4fr.pdf",
    "santoro2017": "https://academic.oup.com/jcem/article/102/7/2218/3063809",
    "randolph2011": "https://academic.oup.com/jcem/article/96/3/746/2596797",
    "finkelstein2020": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7067546",
    "avis2015": "https://core.ac.uk/download/56530091.pdf",
    "tepper2016": "https://escholarship.org/content/qt64w2r98p/qt64w2r98p.pdf",
    "dezambotti2013": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3713094",
    "oura_peri_2024": "https://femtechinsider.com/oura-ring-perimenopause-report/ (company report, not peer reviewed)",
    "nhanes_rhq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/RHQ_J.htm",
    "straw10": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3580996/",
}


def p(value, kind, source, note=""):
    return {"value": value, "kind": kind, "source": source, "note": note}


CYCLE = {
    # Hierarchical phase lengths: fitted on real FedCycleData (1499 peak cycles, 157 women)
    "follicular_mean": p(16.0, "fitted_real", SRC["fedcycle"], "estimated ovulation day; Bull 2019 reports 16.9 (BBT-based, app users)"),
    "follicular_between_sd": p(2.8, "fitted_real", SRC["fedcycle"]),
    "follicular_within_sd_median": p(1.9, "fitted_real", SRC["fedcycle"], "q10 0.95, q90 3.5"),
    "follicular_within_sdlog": p(0.48, "derived", SRC["fedcycle"], "log-normal fitted to q50/q90"),
    "follicular_age_slope_per_year": p(-0.14, "fitted_real", SRC["fedcycle"], "WHOOP 2026: cycle 29.1 d at 24 -> 26.9 d at 44"),
    "luteal_mean": p(12.8, "derived", SRC["fedcycle"] + "; " + SRC["bull2019"], "FedCycle 13.2 (mucus peak), Bull 12.4 (BBT); midpoint"),
    "luteal_between_sd": p(1.5, "fitted_real", SRC["fedcycle"]),
    "luteal_within_sd_median": p(1.1, "fitted_real", SRC["fedcycle"], "Fehring 2006 within-woman range 3.6 d"),
    "luteal_within_sdlog": p(0.5, "assumption", SRC["fedcycle"], "real q90 3.1 includes peak-day measurement error"),
    "corr_person_means_foll_luteal": p(-0.42, "fitted_real", SRC["fedcycle"]),
    "corr_within_foll_luteal": p(-0.20, "fitted_real", SRC["fedcycle"]),
    "menses_length_mean": p(5.25, "fitted_real", SRC["fedcycle"]),
    "menses_length_between_sd": p(0.9, "fitted_real", SRC["fedcycle"]),
    "irregular_fraction": p(0.12, "assumption", "npj Women's Health 2026 review: ~20% marked variability; Apple WHS within-person SD 4-6 d", "subgroup with follicular within-SD 4-9 d"),
    "anovulatory_rate_lt35": p(0.07, "literature", SRC["lynch2014"] + "; FedCycle no-peak 8.8%"),
    "anovulatory_rate_35_40": p(0.10, "assumption", "interpolated; age breakdown not verified"),
    "anovulatory_rate_40_45": p(0.15, "assumption", "Santoro 2017: ELA ~100% at 10 y pre-FMP, 87.9% at 5 y"),
}

SIGNAL = {
    "temp_luteal_amplitude_mean": p(0.30, "derived", SRC["maijala2019"], "set so simulated luteal-follicular mean difference matches Oura 0.30+-0.12 C"),
    "temp_luteal_amplitude_sd": p(0.12, "literature", SRC["maijala2019"]),
    "temp_rise_range": p([0.3, 0.7], "literature", SRC["oura2025"], "algorithm searches 0.3-0.7 C"),
    "temp_noise_sd_median": p(0.16, "derived", SRC["maijala2019"], "night-to-night SD 0.20+-0.07 follicular incl. confounders"),
    "temp_noise_sdlog": p(0.30, "assumption", SRC["maijala2019"]),
    "temp_ar_rho": p([0.2, 0.5], "assumption", ""),
    "hr_luteal_plateau_mean": p(3.4, "literature", SRC["goodale2019"] + "; review 2.7-3.9 bpm; " + SRC["shilaih2017"], "follicular low -> late luteal ~+3.9 bpm"),
    "hr_luteal_plateau_sd": p(1.2, "assumption", ""),
    "hr_preovulatory_weight": p(0.35, "derived", SRC["shilaih2017"], "+2.1 bpm fertile vs +3.8 mid-luteal"),
    "hr_noise_sd": p([1.5, 3.0], "assumption", ""),
    "lnhrv_luteal_drop_mean": p(0.10, "literature", SRC["whoop2026"], "HRV varies 9.4-14.3% of mean across cycle"),
    "lnhrv_luteal_drop_sd": p(0.05, "assumption", ""),
    "lnhrv_noise_sd": p([0.12, 0.22], "assumption", ""),
    "rr_luteal_rise_mean": p(0.45, "literature", SRC["whoop2026"] + "; " + SRC["goodale2019"], "RR varies 2.7-3.3% of mean"),
    "rr_luteal_rise_sd": p(0.15, "assumption", ""),
    "rr_noise_sd": p([0.2, 0.45], "assumption", ""),
    "rmssd_decline_per_decade_ln": p(0.25, "assumption", "age-related HRV decline; RMSSD per decade not verified"),
    "fever_episodes_per_year": p(2.0, "assumption", ""),
    "alcohol_night_fraction_max": p(0.25, "assumption", ""),
}

LH = {
    "user_fraction": p(0.30, "assumption", ""),
    "positive_prob_by_offset": p({-3: 0.10, -2: 0.40, -1: 0.90, 0: 0.50, 1: 0.10}, "derived", SRC["who1980_behre2000"], "offset = test day - ovulation day"),
    "false_positive": p(0.02, "assumption", ""),
}

MISSING = {
    "adherence_mixture": p({"good": [0.60, 0.93, 0.99], "medium": [0.30, 0.80, 0.93], "poor": [0.10, 0.55, 0.80]}, "assumption",
                           "Maijala 2019: data on 97.4% of days, 4/22 women >50% missing in some cycles; Oura excludes >40% missing"),
    "long_gap_person_fraction": p(0.15, "assumption", ""),
    "period_log_reliability": p({"1.0": 0.70, "0.9": 0.20, "0.75": 0.10}, "assumption", ""),
    "period_log_jitter": p({"0": 0.85, "1": 0.12, "2": 0.03}, "assumption", ""),
}

MENOPAUSE = {
    "fmp_age_median": p(52.5, "literature", SRC["gold2013"], "SWAN 52.54; InterLACE 50.0 (recall); NHANES fit in fitted_real"),
    "fmp_age_sd": p(3.8, "derived", SRC["interlace2019"] + " (SD 4.4)", "SWAN prospective distribution narrower"),
    "premature_fraction": p(0.012, "literature", SRC["interlace2019"], "<40 y: 1.2%"),
    "early_mt_duration_median_by_onset_quartile": p([5.14, 3.27, 2.18, 1.68], "literature", SRC["paramsothy2017"]),
    "late_mt_duration_median_by_onset_quartile": p([3.42, 2.55, 2.47, 2.15], "literature", SRC["paramsothy2017"]),
    "total_mt_duration_median_by_onset_quartile": p([8.57, 6.84, 5.08, 4.37], "literature", SRC["paramsothy2017"]),
    "smoking_total_mt_hr": p(1.87, "literature", SRC["paramsothy2017"]),
    "smoking_onset_years_earlier": p(1.22, "literature", SRC["paramsothy2017"], "46.05 vs 47.27"),
    "ela_by_years_to_fmp": p({10: 1.00, 5: 0.879, 1: 0.228}, "literature", SRC["santoro2017"], "fraction cycles with luteal activity"),
    "cycle_length_stable_until_years_before_fmp": p(4.0, "literature", SRC["santoro2017"]),
    "fsh_rise_onset_years_before": p(6.10, "literature", SRC["randolph2011"]),
    "fsh_acceleration_years_before": p(2.05, "literature", SRC["randolph2011"]),
    "fsh_plateau_years_after": p(2.00, "literature", SRC["randolph2011"]),
    "e2_decline_start_years_before": p(2.03, "literature", SRC["randolph2011"]),
    "vms_trajectory_shares": p({"low": 0.270, "early_onset": 0.184, "late_onset": 0.290, "high": 0.256}, "literature", SRC["tepper2016"]),
    "vms_median_total_duration_years": p(7.4, "literature", SRC["avis2015"]),
    "hot_flash_hr_rise_bpm": p(4.0, "literature", SRC["dezambotti2013"]),
    "hot_flash_hf_hrv_drop": p(0.38, "literature", SRC["dezambotti2013"]),
    "hot_flash_duration_min": p([1, 5], "literature", SRC["dezambotti2013"]),
    "peri_hrv_lower_fraction": p([0.20, 0.30], "literature", SRC["oura_peri_2024"], "company report"),
    "post_sleep_less_hours_per_week": p(2.0, "literature", SRC["oura_peri_2024"], "company report"),
    "amh_age_bmi_auc_12_24_36": p([0.881, 0.891, 0.896], "literature", SRC["finkelstein2020"], "external benchmark for FMP prediction"),
    "fsh_age_bmi_auc_12_24_36": p([0.885, 0.877, 0.871], "literature", SRC["finkelstein2020"]),
    "nightly_hot_flash_skin_temp_rise": p(0.05, "assumption", "UNVERIFIED in sources: +0.05 C on the nightly deviation per nocturnal flash; also +3..12 min awake, -1.6% efficiency, -0.035 ln-RMSSD, +0.3 bpm lowest HR"),
    "fmp_age_distribution": p("NHANES 2015-2018 weighted KM (natural menopause), early tail rescaled to InterLACE (<40: 1.2%, 40-44: 4.7%)", "fitted_real", SRC["nhanes_rhq"] + "; " + SRC["interlace2019"],
                              "real NHANES median 52.0; generator median 51.4, SD 4.4"),
    "fmp_covariates_years": p({"college": 1.79, "parous": 0.92, "smoker": -1.22}, "fitted_real", SRC["nhanes_rhq"] + " (college, parity; log-logistic AFT); smoking from " + SRC["paramsothy2017"]),
    "generator_total_transition_median_years": p(5.0, "derived", SRC["paramsothy2017"], "range of quartile medians 4.37-8.57; smokers /1.4"),
    "generator_late_transition_median_years": p(2.5, "derived", SRC["paramsothy2017"], "range 2.15-3.42"),
    "ela_interpolation": p({"0": 0.20, "1": 0.26, "5": 0.879, "10": 1.0}, "derived", SRC["santoro2017"], "generator last-year mean 0.25 vs published 0.228"),
    "amh_trajectory": p({"ln_amh_at_4.75y": "ln(142)", "slope_per_year": 0.35, "person_sd": 0.7, "visit_sd": 0.6}, "fitted_real",
                        SRC["finkelstein2020"], "grid-fitted to published Table 4 P(FMP<=12 mo | AMH band, age): MAE 0.040, r 0.94 (published-table fit, not individual data)"),
    "stage_hrv_drop_ln": p(0.12, "derived", SRC["oura_peri_2024"], "Oura report: HRV 20-30% lower in perimenopause (part is age); applied in late transition/post"),
}


def registry():
    return {"cycle": CYCLE, "signal": SIGNAL, "lh": LH, "missing": MISSING, "menopause": MENOPAUSE, "sources": SRC}


def v(group, key):
    return {"cycle": CYCLE, "signal": SIGNAL, "lh": LH, "missing": MISSING, "menopause": MENOPAUSE}[group][key]["value"]
