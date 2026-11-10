# Prepared data v2

These are synthetic research scenarios, not observed ring users. `synthetic_daily.csv` contains 360 people with five cycles each, observed channels before masking, and hidden simulation targets. Target, cycle and participant fields are never passed as sensor inputs to inference. `synthetic_training_features.csv` contains causal features and observation-count/age indicators after missingness augmentation. Missing measurements remain missing; no real sensor values or outcomes were fabricated to complete respondents.

Train/validation/test split is by participant (60/20/20), seed 29420/29421. Generation seed 29418. Exact IDs, protocol, aggregate metrics and literature assumptions are saved in `research/experiments/ring_health/v2` and `models/configs/training/ring_health_evidence_v2.json`. The stage classifier uses the separate v1 `synthetic_stage_features.csv` (1500 simulated people), with optional-channel dropout added in v2; its 1200 test rows are four masks of 300 people, not 1200 independent participants.

TempID v2 is read directly from its downloaded CSV using the author-defined 21:00–03:59 nocturnal window and at least six hourly samples. This differs from the historical v1 preparation in the parent directory. Author-side bidirectional imputation and alignment remain limitations. No clinical equivalence to Oura is established.
