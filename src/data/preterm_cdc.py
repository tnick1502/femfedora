"""Point-in-time-constrained research cohort for preterm live birth in CDC Natality.

One row is one *singleton live-birth record*, not a prospectively observed
pregnancy. Any downstream week profile is a research approximation from the
recorded week at birth, not an observed pregnancy trajectory.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/cdc_natality/natality_ml_2023_2024.sqlite"
OUT = ROOT / "data/processed/preterm_cdc"
SEED = 20261004

BASE_FEATURES = (
    "maternal_age_at_birth",
    "prepregnancy_bmi",
    "previous_live_births",
    "previous_terminations",
    "previous_preterm_birth",
    "chronic_hypertension",
    "preexisting_diabetes",
    "previous_cesarean",
    "prepregnancy_cigarettes_per_day",
)
FEATURES = BASE_FEATURES + tuple(f"{name}_missing" for name in BASE_FEATURES[1:])
CLASS_NAMES = ("<32", "32–33", "34–36", "≥37")

SOURCE_COLUMNS = (
    "source_row", "mager", "bmi", "priorlive", "priordead", "priorterm",
    "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar", "cig_0",
    "oegest_comb", "mrace6", "meduc",
)


def _uniform_from_source_row(rows: np.ndarray) -> np.ndarray:
    """Stable pseudo-random 0..1 assignment independent of chunk size."""
    x = rows.astype(np.uint64) + np.uint64(SEED) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def prepare_chunk(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    if year not in (2023, 2024):
        raise ValueError("Expected 2023 or 2024")
    gestation = pd.to_numeric(raw["oegest_comb"], errors="coerce")
    if gestation.isna().any():
        raise ValueError("Query must exclude unknown gestational age")
    if (~gestation.between(17, 47)).any():
        raise ValueError("Unexpected gestational age in prepared source")
    prior_live = pd.to_numeric(raw["priorlive"], errors="coerce")
    prior_dead = pd.to_numeric(raw["priordead"], errors="coerce")
    prior_births = (prior_live + prior_dead).where(prior_live.notna() & prior_dead.notna())
    values = {
        "maternal_age_at_birth": raw["mager"],
        "prepregnancy_bmi": raw["bmi"],
        "previous_live_births": prior_births,
        "previous_terminations": raw["priorterm"],
        "previous_preterm_birth": raw["rf_ppterm"],
        "chronic_hypertension": raw["rf_phype"],
        "preexisting_diabetes": raw["rf_pdiab"],
        "previous_cesarean": raw["rf_cesar"],
        "prepregnancy_cigarettes_per_day": raw["cig_0"],
    }
    frame = pd.DataFrame({name: pd.to_numeric(s, errors="coerce").astype("float32")
                          for name, s in values.items()})
    for name in BASE_FEATURES[1:]:
        frame[f"{name}_missing"] = frame[name].isna().astype("int8")
    frame.insert(0, "source_row", raw["source_row"].to_numpy(dtype="int64"))
    frame.insert(0, "source_year", year)
    frame["gestational_weeks_at_birth"] = gestation.to_numpy(dtype="int8")
    frame["risk_band"] = np.select(
        [gestation.lt(32), gestation.lt(34), gestation.lt(37)], [0, 1, 2], default=3
    ).astype("int8")
    frame["reported_race_code_audit_only"] = pd.to_numeric(raw["mrace6"], errors="coerce").astype("float32")
    frame["education_code_audit_only"] = pd.to_numeric(raw["meduc"], errors="coerce").astype("float32")
    if year == 2023:
        u = _uniform_from_source_row(frame["source_row"].to_numpy())
        frame["partition"] = np.select([u < .60, u < .80], ["train", "validation"], default="calibration")
    else:
        frame["partition"] = "temporal_test"
    return frame


def prepare_all(chunk_size: int = 200_000, force: bool = False) -> dict:
    """Write one compact Parquet file per year; never use delivery fields as features."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if all((OUT / f"cohort_{year}.parquet").exists() for year in (2023, 2024)):
            return manifest
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    audit: dict[str, dict] = {}
    connection = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    try:
        for year in (2023, 2024):
            counts = Counter()
            missing = Counter()
            output_path = OUT / f"cohort_{year}.parquet"
            temporary = output_path.with_suffix(".building.parquet")
            temporary.unlink(missing_ok=True)
            writer = None
            query = (
                f"SELECT {', '.join(SOURCE_COLUMNS)} FROM births_baseline "
                "WHERE source_year = ? AND dplural = 1 AND oegest_comb IS NOT NULL "
                "ORDER BY source_row"
            )
            try:
                for raw in pd.read_sql_query(query, connection, params=(year,), chunksize=chunk_size):
                    frame = prepare_chunk(raw, year)
                    if frame.empty:
                        continue
                    counts["rows"] += len(frame)
                    counts.update({f"band_{k}": int(v) for k, v in frame["risk_band"].value_counts().items()})
                    counts.update({f"partition_{k}": int(v) for k, v in frame["partition"].value_counts().items()})
                    missing.update({name: int(frame[name].isna().sum()) for name in BASE_FEATURES})
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                raise RuntimeError(f"Empty source cohort for {year}")
            temporary.replace(output_path)
            audit[str(year)] = {"counts": dict(counts), "missing_features": dict(missing),
                                "output": str(output_path.relative_to(ROOT))}
    finally:
        connection.close()
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "population": "US-resident singleton live-birth records with known obstetric gestation",
        "unit": "live-birth record; maternal ID unavailable",
        "seed": SEED,
        "partition": "2023: 60% train / 20% validation / 20% calibration by stable source-row hash; 2024: temporal test",
        "class_names": CLASS_NAMES,
        "class_rule": "<32, 32–33, 34–36, >=37 completed gestational weeks at birth",
        "features": FEATURES,
        "prohibited_predictors": ["gestational_weeks_at_birth", "risk_band", "source_row", "source_year", "reported_race_code_audit_only", "education_code_audit_only"],
        "age_caveat": "Maternal age is recorded at birth and is only a proxy for age before pregnancy.",
        "claim_limit": "Static research baseline among singleton live births; not a gestational-week or spontaneous-preterm-birth predictor.",
        "years": audit,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_year(year: int, columns: list[str] | None = None) -> pd.DataFrame:
    path = OUT / f"cohort_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)
