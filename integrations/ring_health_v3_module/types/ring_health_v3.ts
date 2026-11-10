// Ring Health v3 — request/response types (mirror of src/ring_health_v3/schema.py and engine outputs).
// Dates are local calendar dates "YYYY-MM-DD". null = unknown/missing (never use 0 for missing).

export type ISODate = string;

export interface Night {
  date: ISODate;
  temperature_deviation?: number | null; // °C deviation from personal baseline (Oura daily_readiness.temperature_deviation), -3..3
  resting_hr?: number | null;            // bpm, lowest nightly HR (Oura sleep.lowest_heart_rate), 25..180
  hrv_ms?: number | null;                // ms, nightly average RMSSD (Oura sleep.average_hrv), 1..400
  respiratory_rate?: number | null;      // breaths/min (Oura sleep.average_breath), 5..40
  sleep_minutes?: number | null;         // total sleep, min
  sleep_efficiency?: number | null;      // %, 0..100
  awake_minutes?: number | null;         // awake time inside sleep period, min
}

export interface Profile {
  age: number;                           // required, 16..70
  pregnant?: boolean; breastfeeding?: boolean; hormonal_contraception?: boolean;
  menopausal_hormone_therapy?: boolean; hysterectomy?: boolean; bilateral_oophorectomy?: boolean; // any true -> out_of_scope
  smoker?: boolean | null; bmi?: number | null; parous?: boolean | null; college_education?: boolean | null;
}

export interface AnalyzeRequest {
  as_of: ISODate;                                       // user's local "today"; nothing may be dated after it
  modules?: Array<"fertility" | "menopause">;           // default both
  profile: Profile;
  cycle?: {
    period_starts?: ISODate[];                          // logged period start dates (unique)
    last_period_date?: ISODate | null;                  // onboarding answer
    period_logs_complete?: boolean;                     // user confirms every period is logged
    logging_since?: ISODate | null;                     // calendar logging start
    menses_length?: number | null;                      // 1..12
  };
  lh_tests?: Array<{ date: ISODate; result: "positive" | "negative" }>;
  symptoms?: { logging_enabled?: boolean; hot_flash_dates?: ISODate[]; night_sweat_dates?: ISODate[] };
  labs?: { fsh_iu_l?: { value: number; date: ISODate } | null; amh_pg_ml?: { value: number; date: ISODate } | null };
  nights?: Night[] | null;                              // EITHER nights OR oura (max 800 days)
  oura?: { sleep?: object[]; daily_readiness?: object[] } | null; // raw Oura API v2 documents
}

export type FertilityStatus = "ok" | "anchor_required" | "anchor_stale" | "out_of_scope" | "not_applicable_age";
export type Phase = "menstruation" | "follicular" | "fertile_window" | "ovulation" | "luteal" | "late_luteal" | "luteal_expected";
export type OvulationStatus = "predicted" | "detected" | "likely_passed_not_confirmed" | "not_detected_possible_anovulatory_cycle";

export interface FertilityOk {
  status: "ok";
  as_of: ISODate; cycle_start: ISODate; cycle_day: number; phase: Phase;
  cycle_start_source: "logged" | "inferred_from_physiology";
  unlogged_period_suggestions: Array<{ date: ISODate; probability: number; action: "ask_user_to_confirm_period_start" }>;
  fertile_window: {
    today_probability: number;            // P(today is within the 6-day window ending on ovulation day)
    in_window_today: boolean;
    window: { start: ISODate; end: ISODate; peak_day: ISODate; status: "past" | "now" | "upcoming" } | null;
    forecast_7d: Array<{ date: ISODate; cycle_day: number; fertile_probability: number }>;
  };
  ovulation: {
    status: OvulationStatus; date: ISODate | null; interval_80: [ISODate, ISODate];
    probability_already_passed: number; probability_ovulatory_cycle: number; concentration_3d: number;
  };
  next_period: { date: ISODate; interval_80: [ISODate, ISODate]; days_until: number; late_or_unlogged: boolean };
  insights: {
    completed_cycles: number; cycle_length_median: number | null; cycle_length_range: [number, number] | null;
    cycle_variability_sd: number | null; regularity: "regular" | "somewhat_irregular" | "irregular" | null;
    luteal_length_median: number | null; short_luteal_cycles: number; possible_anovulatory_cycles: number;
    personal_follicular_mean: number; personal_luteal_mean: number;
  };
  history: Array<{
    start: ISODate; length: number | null; quality: "ok" | "inferred_start" | "too_short" | "too_long_possible_missed_log";
    ovulation_detected?: boolean; ovulation_date?: ISODate | null; ovulation_date_estimate?: ISODate;
    estimate_basis?: "physiology" | "calendar_or_sparse_data"; luteal_length?: number | null;
    possible_anovulatory?: boolean; temperature_shift_c?: number | null;
  }>;
  data_quality: {
    temperature_nights_last14: number; coverage_last14: number; days_since_last_temperature: number | null;
    possible_illness_nights_last14: number; evidence: "physiology_and_history" | "history_only" | "population_prior_only";
    confidence: "high" | "medium" | "low"; observed_nights_current_cycle: Record<string, number>; imputation: string;
  };
}
export type Fertility = FertilityOk | { status: Exclude<FertilityStatus, "ok">; reason?: string; message?: string; needed?: string[] };

export type Stage = "reproductive" | "early_transition" | "late_transition" | "postmenopause";
export interface MenopauseOk {
  status: "ok" | "insufficient_data";
  stage: {
    label: Stage | null; label_ru: string | null; basis: string;               // "model" | "STRAW+10: >=12 months without a period" | "insufficient_data"
    probabilities: Record<Stage, number> | null;
    grouped: { label: "reproductive" | "transition" | "postmenopause"; probabilities: Record<string, number> } | null;
    model_used?: "full" | "age_bleeding" | "age_ring" | "age_only";
    age_only_reference: Record<Stage, number>;
    straw10_criteria: { amenorrhea_12m: boolean; interval_ge60: boolean; persistent_diff7: boolean };
  };
  final_menstrual_period: {
    probability_by_months: Record<"6" | "12" | "24" | "36" | "48" | "60", number> | null; // P(FMP happened or happens within N months), monotone
    median_months: number | "<6" | ">60" | null; already_confirmed: boolean; definition?: string;
    age_only_reference_by_months?: Record<string, number>;                           // only when insufficient_data
  };
  vasomotor: { index_last30: number | null; share_of_scored_nights_with_likely_hot_flashes_last30: number | null; self_reported_symptom_days_last30: number | null };
  signals: Record<string, number | null>;
  data_quality: { input_pattern: "labs" | "bleeding_ring" | "bleeding_only" | "ring_only" | "age_only"; symptoms_used: boolean; bleeding_information: boolean;
                  ring_nights_last90: number | null; labs_used: string[]; confidence: "high" | "medium" | "low"; missing_inputs_policy: string };
  needed?: string[];
}
export type Menopause = MenopauseOk | { status: "out_of_scope" | "not_applicable_age"; reason: string };

export interface AnalyzeResponse {
  model_version: string; artifact_registry_sha256: string; as_of: ISODate;
  mode: "demonstration"; clinical_validation: "not_established"; training_data: string; recommendations: [];
  data_quality: { nights_supplied: number; coverage: Record<string, unknown> };
  fertility?: Fertility;
  menopause?: Menopause;
}
