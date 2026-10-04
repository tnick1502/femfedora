"""Bounded, reproducible feature diagnostics for the pooled CDC HDP study."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data.preeclampsia_cdc import ENRICHED_FEATURES, MISSING_MASKS, QUALITY_MASKS, OUT, SEED
from src.progress import tqdm
from src.training.preeclampsia_cdc import predict_logistic


SAMPLE_COLUMNS = list(dict.fromkeys(
    ["source_year", "source_row", "fold", "target_hdp_proxy", "rf_phype", "rf_ghype", "rf_ehype", "rf_artec"]
    + ENRICHED_FEATURES + QUALITY_MASKS
))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analysis_sample(fraction: float = 0.05) -> pd.DataFrame:
    """Independent Bernoulli sample of rows; no label-dependent selection."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    data_path = OUT / "eligible_all_2023_2024.parquet"
    audit = json.loads((OUT / "audit.json").read_text())
    digest = _sha256(data_path)
    if digest != audit["prepared_file"]["sha256"]:
        raise ValueError("Prepared data differ from audited SHA-256")
    sample_path = OUT / "feature_analysis_sample.parquet"
    meta_path = OUT / "feature_analysis_sample.json"
    spec = {"prepared_sha256": digest, "seed": SEED, "fraction": fraction,
            "columns": SAMPLE_COLUMNS}
    if sample_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("spec") == spec and saved.get("sha256") == _sha256(sample_path):
            return pd.read_parquet(sample_path)
    source = pq.ParquetFile(data_path)
    rng = np.random.default_rng(SEED)
    parts = []
    for batch in tqdm(source.iter_batches(batch_size=100_000, columns=SAMPLE_COLUMNS),
                      desc="Independent feature sample", unit="batch"):
        frame = batch.to_pandas()
        selected = rng.random(len(frame)) < fraction
        if selected.any():
            parts.append(frame.loc[selected])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["source_year", "source_row"]).any():
        raise ValueError("Sample contains repeated birth rows")
    if result.groupby("fold").size().min() < 1000 or result.fold.nunique() != 5:
        raise ValueError("Sample is too small or missing a fold")
    result.to_parquet(sample_path, index=False)
    meta_path.write_text(json.dumps({"spec": spec, "sha256": _sha256(sample_path),
                                     "rows": len(result)}, indent=2))
    return result


def heldout_permutation_importance(model, frame: pd.DataFrame, features: list[str],
                                   model_name: str, fold: int, repeats: int = 3,
                                   seed: int = SEED,
                                   groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """AUPRC loss and Brier deterioration when a held-out feature is shuffled."""
    if not frame.fold.eq(fold).all() or frame.target_hdp_proxy.nunique() != 2:
        raise ValueError("Importance requires one held-out fold with both classes")
    if frame[features].isna().all().any():
        raise ValueError("An entire feature is missing in the held-out sample")
    X = frame[features].copy().reset_index(drop=True)
    y = frame.target_hdp_proxy.to_numpy(dtype=np.uint8)

    def predict(data):
        if model_name.startswith("logistic"):
            return predict_logistic(model, data)
        return model.predict_proba(data)[:, 1]

    baseline = predict(X)
    baseline_ap = average_precision_score(y, baseline)
    baseline_brier = brier_score_loss(y, baseline)
    rng = np.random.default_rng(seed + fold)
    rows = []
    linked_masks = {"bmi": "bmi_missing", "prior_live_birth_count_cat": "prior_live_birth_count_missing",
                    "priorterm": "priorterm_missing"}
    feature_sets = {name: [name] + ([linked_masks[name]] if name in linked_masks and linked_masks[name] in features else [])
                    for name in features if name not in MISSING_MASKS}
    feature_sets.update(groups or {})
    for feature, members in tqdm(feature_sets.items(), desc=f"Fold {fold}: {model_name} importance", unit="feature", leave=False):
        if not set(members).issubset(features):
            raise ValueError(f"Invalid feature group: {feature}")
        for repeat in range(repeats):
            shuffled = X.copy()
            permutation = rng.permutation(len(X))
            for member in members:
                shuffled[member] = X[member].to_numpy()[permutation]
            prediction = predict(shuffled)
            rows.append({"fold": fold, "model": model_name, "feature": feature,
                         "feature_kind": "group" if feature in (groups or {}) else "individual",
                         "members": ",".join(members),
                         "repeat": repeat, "sample_rows": len(frame),
                         "sample_events": int(y.sum()), "baseline_auprc": baseline_ap,
                         "baseline_brier": baseline_brier,
                         "delta_auprc": baseline_ap-average_precision_score(y, prediction),
                         "delta_brier": brier_score_loss(y, prediction)-baseline_brier})
    return pd.DataFrame(rows)
