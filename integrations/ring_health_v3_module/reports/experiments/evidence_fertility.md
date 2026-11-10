# Fertility / ovulation evidence for smart-ring synthetic generator (verified 2026-11-09)

Rules: every number below was read on the cited URL during this session. "DERIVED" means arithmetic on verified numbers. "UNVERIFIED" means it could not be checked this session (usually because pmc.ncbi.nlm.nih.gov / pubmed returned a captcha, and Europe PMC, Zenodo API, Hugging Face API and GitHub API are blocked from the sandbox).

---

## 1. Finger (Oura) temperature specifics

| Parameter | Value | Population / device | URL |
|---|---|---|---|
| Luteal minus follicular amplitude, finger skin | **0.30 °C (SD 0.12)**, p<0.001 | Maijala 2019, 22 women, 41 cycles, Oura gen1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6883568/ |
| Same, oral wake-up temperature | 0.23 °C (SD 0.09); skin minus oral 0.07 °C (SD 0.10) | same | same |
| Women meeting the 0.15 °C phase-difference criterion | skin 21/22, oral 18/22 | same | same |
| Day-to-day SD of nightly finger temperature | **follicular 0.20 °C (SD 0.07); luteal 0.24 °C (SD 0.08)** (oral: 0.17 / 0.19) | same | same |
| Ring vs oral daily correlation | r = 0.563 (95% CI 0.519–0.604); phase-difference r = 0.589 | same | same |
| Ring resolution / calibration | 0.07 °C resolution, not calibrated (relative values only) | same | same |
| Rise duration (days) | **Not reported** in Maijala or in Oura JMIR 2025 | — | — |
| Rise targeted by Oura's algorithm | "sustained rise around 0.3–0.7 °C" | Oura JMIR 2025 (Thigpen et al.) | https://jmir.org/2025/1/e60667 |
| Oura algorithm structure | centre at 0, reject values >2 SD from population mean, linear-fill imputation, Butterworth band-pass (grid-searched), hysteresis thresholding into follicular/luteal; reject luteal phases outside 7–17 d and follicular outside 10–90 d | same | same |
| Fraction with clear oscillation (proxy for biphasic) | **96/116 = 83%** had cosinor r²>0.25; **83/116 = 72%** met r²>0.4 | Alzueta 2024 J Biol Rhythms (= PMC11416332), Oura, 56 young (25.6±5.5 y) + 40 midlife (46.6±2.8 y) | https://escholarship.org/content/qt34r0k43v/qt34r0k43v.pdf ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11416332/ |
| LH kit vs temperature-cosinor agreement | 82% | same | same |
| Age effect on finger temp | Mesor higher in midlife (p=0.03); amplitude no group difference (p=0.48); period p=0.16 | same | same |
| Timing (cosinor) | menses onset ~7.05 d (SD 3.27) after temperature acrophase; ovulation ~19.5 d (SD 5.04) after acrophase | same | same |
| Absolute °C by phase in PMC11416332 | **Not reported** (data scaled/centred) | same | same |
| Alzueta 2022 (IJWH) Oura | 26 women (24.4±1.1 y), 1 cycle each; temp midluteal vs menses t=4.14, late luteal t=5.12 (no °C given); only sleep efficiency changed (midluteal −1.96 %, p=0.026) | Oura | https://www.dovepress.com/tracking-sleep-temperature-heart-rate-and-daily-symptoms-across-the-me-peer-reviewed-fulltext-article-IJWH |
| NC° + Oura FDA 510(k) K202897 | cleared **2021-06-24**; 40 women (31.3±4.9 y), 223 cycles (155 complete), cycle 28.6±4.4 d; ovulation confirmed against **87 positive LH tests**; Oura gave **+1.6 extra green days in luteal phase** vs oral; no Pearl Index for Oura input | Natural Cycles + Oura | https://www.accessdata.fda.gov/cdrh_docs/pdf20/K202897.pdf |
| NC with Oura effectiveness (company page) | typical use **93%**, perfect use **98%**; "ovulation confirmed in 100% of cases with both oral and Oura data" (4-month study, size not stated) | Natural Cycles marketing/research page | https://naturalcycles.com/research-library/how-effective-is-natural-cycles-when-used-with-the-oura-ring |
| Oura JMIR 2025 already-known numbers (re-verified) | 1155 cycles, 96.4% detected (1113/1155), MAE 1.26 d (SEM 0.02), ±1 d 68%, ±2 d 87.9%, ±3 d 95%; calendar MAE 3.44 d | 964 participants, 32.8±5.5 y, cycle 29.1±3.4 d | https://jmir.org/2025/1/e60667 |

## 2. HR, HRV, respiratory rate across the cycle (wearables)

| Parameter | Value | Population / device | URL |
|---|---|---|---|
| Wrist temp vs menses (b, °C) | follicular −0.24, fertile −0.25, early luteal +0.01 (ns), late luteal **+0.20**; means 34.11 / 33.87 / 33.88 / – / 34.32 °C | Goodale 2019 JMIR, Ava bracelet, 193 women, 708 analysable cycles, 33.0±3.7 y | https://jmir.org/2019/4/e13404/ |
| Resting pulse vs menses (bpm) | follicular **−1.54**, fertile −0.03 (ns), early luteal **+2.01**, late luteal **+2.46**; means 57.92 / 56.56 / – / 59.98 / 60.47 | same | same |
| DERIVED follicular → late-luteal pulse swing | ≈ +3.9 bpm (60.47 − 56.56) | same | same |
| Respiratory rate vs menses (br/min) | follicular −0.39, fertile **−0.48**, early luteal −0.20, late luteal **+0.22**; means 16.86 / 16.57 / 16.40 / 16.68 / 17.04 | same | same |
| DERIVED RR swing fertile → late luteal | ≈ +0.64 br/min | same | same |
| HRV (Ava "HRV ratio", not RMSSD) | follicular +0.11, fertile +0.08, late luteal −0.20 vs menses (only trends after Bonferroni); means 1.78 / 1.86 / 1.86 / 1.62 | same | same |
| Phase definitions used | fertile = OV−5..OV; early luteal OV+1..OV+7; late luteal OV+8..menses; LH peak ≈ 1 d before ovulation | same | same |
| WHOOP cohort size | 2,596 women, 42,759 cycles, 1.2 M days | npj Digit Med 2026, DOI 10.1038/s41746-026-02799-9 | https://alexgonzl.github.io/menstrual_cycle_physiology_sleep_whoop/ |
| WHOOP within-cycle range (% of mean) | HRV **9.4% (24-d cycle) to 14.3% (34-d cycle)**; RR **2.7–3.3%**; HRV range falls 13.3% → 10.0% from age 24 to 44 | WHOOP (bioRxiv 10.1101/2025.09.11.675620) | https://www.biorxiv.org/content/10.1101/2025.09.11.675620.full |
| WHOOP timing | RHR minimum / HRV maximum at cycle day 6 (34-d cycle) vs day 4 (24-d cycle); RR minimum day 13 (34-d) vs day 8 (24-d); HRV peaks during menstruation | same | same |
| WHOOP individual within-cycle variation (% personal mean) | HRV 38%, RHR 14%, RR 6%, skin temp 3%, SpO2 2% | same | same |
| Sleep confound | −10% sleep duration → RHR **+1.2%** (95% CI 0.9–1.4) ≈ +0.7 bpm at 60 bpm | same | same |
| Scoping review 2026 (40 studies) | RHR rises **~2.7–3.9 bpm** follicular → luteal; temperature nadir often >5 d before ovulation; RR dips then peaks premenstrually; HRV highest follicular, lowest premenstrual | Johnson et al. npj Women's Health 2026, DOI 10.1038/s44294-026-00146-7 | https://www.news-medical.net/news/20260527/Wearable-devices-may-help-detect-menstrual-health-changes-earlier.aspx |
| Oura HR/RMSSD (Alzueta 2024) | HR higher at ovulatory, mid-luteal and late-luteal vs menses (p<0.01); RMSSD lower in midlife overall; late-luteal RMSSD < menses in young only (p=0.019); **no bpm / ms values in text** | Oura | https://escholarship.org/content/qt34r0k43v/qt34r0k43v.pdf |
| Shilaih 2017 Sci Rep pulse (+2.1 bpm fertile, +3.8 mid-luteal) | **UNVERIFIED this session** (given as known) | Ava | — |
| RMSSD in ms across phases from any ring/wrist study | **UNVERIFIED** (not found in text of any source opened) | — | — |

## 3. Cycle-length statistics, phase variability, anovulation

| Parameter | Value | Population | URL |
|---|---|---|---|
| Cycle / follicular / luteal (mean±SD) | **28.9±3.4 / 16.5±3.4 / 12.4±2.0 d**; 95% ranges 22–36 / 10–22 / 9–16 d | Fehring 2006, 141 women, 1,060 cycles, ovulation = 2nd Clearblue peak day | https://scispace.com/pdf/variability-in-the-phases-of-the-menstrual-cycle-4yyngl5ir5.pdf |
| Within-woman mean range (max−min) | luteal **3.6±2.5 d**, follicular **5.8±2.6 d**, cycle **6.7±3.5 d** | same (≈7.5 cycles/woman) | same |
| DERIVED within-woman SD (range/≈2.7 for n≈7) | luteal ≈1.3 d, follicular ≈2.1 d, cycle ≈2.5 d (approximation, not in source) | same | same |
| % women with >7 d difference | cycle 42.5%, follicular 33.6%, luteal 9% | same | same |
| Cycle length by age (Fehring) | <25: 29.2±3.5; 25–35: 29.2±3.4; >35: 27.9±2.6 d | same | same |
| Apple Women's Health Study | 12,608 women, 165,668 cycles; mean 28.7 (SD 6.1), median 28; vs 35–39: <20 +1.6, 20–24 +1.4, 25–29 +1.1, 30–34 +0.6, >50 +2.0 d; within-person SD 4–6 d; variability vs 35–39: <20 +45%, 45–49 +44%, >50 +200%; Asian +1.6 d, Hispanic +0.7 d; class-3 obesity +1.5 d | Li et al. npj Digit Med 2023 (PMC10226714) | https://www.nature.com/articles/s41746-023-00848-1 |
| WHOOP age trend | 28.4±2.4 d; **29.1 d at 24 y → 26.9 d at 44 y**; 27% of cycles deviate ≥3 d from personal median (min 25% at age 33) | WHOOP 2026 | https://www.biorxiv.org/content/10.1101/2025.09.11.675620.full |
| Kindara/Sympto (Symul 2019) | follicular median 16 d, 90% of ovulations between day 10 and 24, only ~24% on days 14–15; luteal median 12 (Kindara) / 13 (Sympto); ~20% of luteal phases ≤10 d; BBT shift mid-follicular→mid-luteal ≈ **0.36 °C** | 109,161 reliably-estimated cycles | https://infosciences.epfl.ch/record/268599/files/s41746-019-0139-4.pdf |
| Sporadic anovulation, regular cycles | serum algorithms **5.5–12.8%**; urine algorithms **3.4–18.6%** | Lynch 2014 Fertil Steril, BioCycle, 259 women (18–44) | https://researchconnect.buffalo.edu/en/publications/assessment-of-anovulation-in-eumenorrheic-women-comparison-of-ovu/ |
| Anovulation (LH-based) | **23/193 cycles = 11.9%** anovulatory (DERIVED %) | Zhu 2021, 57 healthy women | https://doaj.org/article/ac0197400a6743929dbf73fbb13827b5 |
| Ava cohort LH exclusions | 25/237 women had no confirmed LH surge (excluded) | Goodale 2019 | https://jmir.org/2019/4/e13404/ |
| Anovulation **by age** | **UNVERIFIED** (no age-stratified figure found; Lynch abstract has no age breakdown) | — | — |
| Bull 2019 numbers / age trends | **UNVERIFIED this session** (nature.com fetch failed) | — | — |

## 4. Ovulation detection accuracy (temperature and multiparameter)

| Method | Result | Population / device | URL |
|---|---|---|---|
| Oura physiology algorithm | detection 96.4%, MAE 1.26 d, ±2 d 87.9%. By regularity: typical 97.5%/MAE 1.23; moderately irregular 96.0%/1.23; abnormally irregular 94.0%/1.48 (±2 d 81.8%). By cycle length: 22–26 d 93.2%/1.34; 27–31 d 98.0%/1.18; 32–35 d 97.3%/1.26; ≥36 d 90%/1.70 (±2 d 77%). By age: 18–26 97.0%/1.35; 27–34 96.6%/1.29; 35–43 96.0%/1.17; ≥44 95%/1.19 | Oura JMIR 2025, reference = day after last positive LH | https://jmir.org/2025/1/e60667 |
| Maijala HALF_LOCS (midpoint of adjacent min/max) | sensitivity **83.3%** (window −3..+2 d), 74.4% (−4..+1), ~95% within ±4 d; mean offset 0.6 d (SD 1.5). RISE_0.15 offset 0.6 (SD 1.6); HALF_PEAKS 1.4 (SD 1.5) | Oura, 22 women, 80 verified ovulations | https://pmc.ncbi.nlm.nih.gov/articles/PMC6883568/ |
| Maijala menses prediction | ±1 d 50.0%, ±2 d 71.9%, ±3 d 81.4%, ±4 d 86.5% | same | same |
| Wrist skin temp vs oral BBT | sensitivity **0.62 vs 0.23**; specificity **0.26 vs 0.70**; PPV 86.2% vs 84.8%; post-ovulatory rise up to **0.50 °C wrist vs 0.20 °C BBT** | Zhu 2021 JMIR (Ava), 57 women, 193 cycles | https://doaj.org/article/ac0197400a6743929dbf73fbb13827b5 |
| Ava multiparameter RF (fertile window) | fertile window detected in **90%** of cycles; sensitivity 0.81, specificity 0.93, F1 0.78; with 50% nights dropped: accuracy >86%, F1 0.72 | Goodale 2019, 85 validation cycles | https://jmir.org/2019/4/e13404/ |
| Symul HMM (Kindara/Sympto) | 10-state HMM on BBT+mucus; "reliable" if uncertainty ≤±1.5 d, shift ≥0.15 °C, confidence ≥0.75 → only 14% of standard cycles; **no ground-truth accuracy reported** | 2.7 M cycles | https://infosciences.epfl.ch/record/268599/files/s41746-019-0139-4.pdf |
| Shilaih 2018 (PMC6265623) | **UNVERIFIED** (captcha) | Ava | https://pmc.ncbi.nlm.nih.gov/articles/PMC6265623 |
| Three-over-six rule, CUSUM, Tempdrop accuracy | **UNVERIFIED** (no primary source opened) | — | — |

## 5. LH surge timing, fertile window, conception probabilities

| Parameter | Value | URL |
|---|---|---|
| LH rise → ovulation | WHO 1980 (n=177): median **32 h** (95% CI 23.6–38.2); 90% ovulate 16–48 h after initial rise | https://se.clearblue.com/sites/default/files/HCP_Publications/Articles-Pregnancy/Evidence_for_using_LH%2B1_as_marker_for_conception.pdf |
| LH peak → ovulation | WHO 1980 median **16.5 h** (9.5–23); Collins 1991 17.5 h; Fritz 1992 peak→rupture 37.6±4.2 h | same |
| First positive urine LH → ovulation | Miller & Soules 1996: **20±3 h** (95% CI 14–26) | same |
| Ovulation day relative to urine LH surge day | Behre 2000: 14.8% surge day, **76.3% next day**; Pearlstone 1994: 35% LH+1, 61% LH+2; Luciano 1990: 6% / 70.8% / 21% on LH+1/+2/+3 | same |
| Conventions in use | Oura JMIR 2025: ovulation = **day after last positive LH test**; Goodale: LH peak ≈1 d before ovulation; Fehring: 2nd Clearblue peak day | jmir.org/2025/1/e60667 ; jmir.org/2019/4/e13404/ ; scispace PDF above |
| Fertile window | 6 days ending on ovulation day (all conceptions had intercourse in this window) | Wilcox 1995 NEJM: https://static1.squarespace.com/static/54694fa6e4b0eaec4530f99d/t/564b6821e4b0202af56ed68c/1447782433893/TIMING+OF+SEXUAL+INTERCOURSE+IN+RELATION+TO+OVULATION+NEJM+1995.pdf |
| Day-specific conception probability (±SE) | −5: 0.10 (0.08); −4: 0.16 (0.06); −3: 0.14 (0.08); −2: 0.27 (0.07); −1: 0.31 (0.06); 0: 0.33 (0.09); +1: 0 (95% CI 0–0.12) | same; 221 women, 625 ovulatory cycles, 192 pregnancies; ovulation from urinary E1G/PdG ratio |

## 6. Wear adherence and missingness

| Parameter | Value | Population / device | URL |
|---|---|---|---|
| Ring data availability | **97.4% of days per cycle** (range 83.3–100); 96.6% per participant before interpolation (80.7–100); oral thermometer 92.9% (60.6–100) | Maijala 2019, Oura | https://pmc.ncbi.nlm.nih.gov/articles/PMC6883568/ |
| Long gaps | 4/22 participants had >50% missing skin data in some cycles (only longest continuous segment used) | same | same |
| Bracelet sync exclusions | 14/237 women excluded for syncing <80% of days; validation cycles <80% sync removed | Goodale 2019, Ava | https://jmir.org/2019/4/e13404/ |
| Robustness to missingness | 50% random nightly drop → fertile-window accuracy >86% | same | same |
| Oura production rule | cycle excluded if >40% physiology missing in last 60 days; 47 cycles excluded for insufficient data; linear imputation; calendar fallback | Oura JMIR 2025 | https://jmir.org/2025/1/e60667 |
| Alzueta 2024 | ~36 d of data per person (~1.2 cycles); ring-night missingness not reported; diary missing 9.35% AM / 13.84% PM | Oura | https://escholarship.org/content/qt34r0k43v/qt34r0k43v.pdf |
| PMC11294004 | **UNVERIFIED** (could not identify; PMC captcha) | — | — |

## 7. Datasets (access status checked 2026-11-09)

| Dataset | Status / access | License | Size | Labels / columns | URL |
|---|---|---|---|---|---|
| **mcPHASES** (PhysioNet) | Only v1.0.0 (2025-09-09, DOI 10.13026/5b8h-jx61). Restricted access, and Files section says **"The files for this project are no longer available."** No newer version. | PhysioNet Restricted Health Data License 1.5.0 + DUA 1.5.0 | 23 tables; 42 participants consented to release (50 enrolled 2022; 20 did 2024 interval) | Fitbit Sense, Dexcom G6 (interval 1), Mira LH/E3G/PdG; 4 phases from Mira (menstruation, late-follicular, ovulation, luteal) | https://www.physionet.org/content/mcphases/ |
| `haammar/fem-state` (Hugging Face) | Public, no gating visible. Source not stated; columns (Fitbit-style wrist_temp_delta, glucose_mean, stress_score_mean, 4-phase label) **look derived from mcPHASES (inference, not confirmed)** — check provenance and license before use | openrail (as listed) | 5.67k rows, CSV | 11 ordinal symptoms, sleep_minutes/awake/efficiency, resting_hr, steps_sum, stress_score_mean, wrist_temp_delta, glucose_mean, hrv_rmssd, is_weekend; label `phase` (Menstrual/Follicular/Fertility/Luteal) | https://huggingface.co/datasets/haammar/fem-state |
| **FedCycleData** (Marquette, Fehring "Menstrual Cycle Data", 2012) | Open download from Marquette ePublications (.sav main file 329 KB; CSV "FedCycleData071012 (2).csv", 290 kB). The direct CSV link returned a client error to WebFetch this session | No CC license; "anonymized for dissemination… Data reuse agreed to by subjects in consent form" | 290 kB CSV, 80 columns | per-cycle: ClientID, CycleNumber, Group, CycleWithPeakorNot, ReproductiveCategory, LengthofCycle, MeanCycleLength, **EstimatedDayofOvulation**, **LengthofLutealPhase**, FirstDayofHigh, TotalNumberofHighDays, TotalHighPostPeak, TotalNumberofPeakDays, TotalDaysofFertility, LengthofMenses, MensesScoreDayOne…Day15, MeanBleedingIntensity, NumberofDaysofIntercourse, IntercourseInFertileWindow, UnusualBleeding, Age, Height, Weight, BMI, Method, demographics/reproductive history | https://epublications.marquette.edu/data_nfp/7/ ; CSV: https://epublications.marquette.edu/cgi/viewcontent.cgi?filename=0&article=1007&context=data_nfp&type=additional |
| FedCycleData **raw GitHub mirror** (fetched successfully) | Public repo chiomajaco6/MenstrualCyclePrediction, branch `main`, file `rawFedCycleData.csv`, 80-column header identical to above; first rows `nfp8122,1,0,1,0,29,27.33,17,12,...`. WebFetch saw ~257 rows / 27 ClientIDs, **but that may be a truncation by the fetch tool; full row count UNVERIFIED** | inherits Marquette terms | — | as above | **https://raw.githubusercontent.com/chiomajaco6/MenstrualCyclePrediction/main/rawFedCycleData.csv** |
| Kaggle "Menstrual Cycle Data" (nikitabisht) | Page exists; file/license details not rendered to fetcher | UNVERIFIED | UNVERIFIED | same source per repo README | https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data |
| **Zenodo 20160611** ("Monitoring female fertile ground: pilot study of continuous body temperature using a skin wearable device and ML", Läänelaid, Salumets, Altmäe et al.) | Open, v1, 2026-05-13 | **CC BY 4.0** | 2.5 MB: `temperature_ovulation_dataset.csv` (2.3 MB, cleaned/annotated/imputed temperature + ovulation), `MaxRange.csv` (146.8 kB, hourly means) | participants/device/LH not stated on page; code: github.com/nach00gar/Monitoring-female-fertile-ground-temperature-measurement-using-a-skin-wearable-and-ML-integration | https://zenodo.org/records/20160611 |
| Zenodo 17822639 WAVES (SRI, Taitz/Baker) | Open, code only (notebooks, cosinor/wavelet BBT feature extraction), no data | CC BY 4.0 | 1.4 MB | — | https://zenodo.org/records/17822639 |
| Symul/Kindara/Sympto | Raw data restricted (on request with Kindara/Sympto permission); aggregated values + code public | — | — | — | https://lasy.github.io/FAM-Public-Repo/ |
| WHOOP 2026 | Code public on GitHub; data access not described | — | 42,759 cycles | — | https://alexgonzl.github.io/menstrual_cycle_physiology_sleep_whoop/ |
| MCAnalysis (Juli app) | Juli data not public; package ships synthetic example data | — | — | — | https://arxiv.org/html/2604.12536 ; https://github.com/kyradelray/mcanalysis |
| Natural Cycles Dryad data | **UNVERIFIED / not found** (searches returned only papers and NC's user data-export help page) | — | — | — | — |
| Other (Oura-specific open menstrual dataset) | none found | — | — | — | — |

### Notes for the generator
- Suggested finger-temp parameters (Oura-like): luteal step ≈ 0.30 °C (SD 0.12 between women), nightly noise SD ≈ 0.20 °C follicular / 0.24 °C luteal; allow 0.3–0.7 °C at the high end (Oura algorithm target); about 17–28% of women/cycles without a clean oscillation (Alzueta 2024 r² thresholds); anovulatory cycles about 3–19% depending on definition (Lynch 2014), about 12% in Zhu 2021.
- HR: follicular nadir to late luteal about +4 bpm (Goodale), 2.7–3.9 bpm range (2026 review); RR about +0.6 br/min from fertile window to late luteal; HRV: use relative amplitude 9–14% of mean (WHOOP), peaking at menses/early follicular.
- Missingness: about 2.6% nights missing at baseline, with a heavy tail (some users >50% missing in some cycles); the production-like cutoff is >40% missing in the last 60 days.
