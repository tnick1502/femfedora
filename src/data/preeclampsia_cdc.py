"""Reproducible CDC natality cohort for an offline HDP proxy benchmark.

One row is a live birth, not a longitudinal pregnancy. The main analysis is
restricted to singleton births to avoid duplicate pregnancy labels for twins.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data/processed/cdc_natality/natality_ml_2023_2024.sqlite"
OUT = ROOT / "data/processed/preeclampsia_cdc"
RAW_INPUTS = ["mager", "bmi", "priorlive", "priordead", "priorterm",
              "rf_pdiab", "rf_ppterm", "rf_cesar"]
NUMERIC = ["mager", "bmi", "prior_live_birth_count_cat", "priorterm"]
BINARY = ["rf_pdiab", "rf_ppterm", "rf_cesar"]
MISSING_MASKS = ["bmi_missing", "prior_live_birth_count_missing", "priorterm_missing"]
FEATURES = NUMERIC + BINARY + MISSING_MASKS
ENGINEERED = ["no_prior_live_birth", "prior_live_birth_count_cat", *MISSING_MASKS,
              "age_35_plus", "bmi_30_plus", "bmi_35_plus"]
ENRICHED_FEATURES = list(dict.fromkeys(FEATURES + ENGINEERED))
EARLY_FEATURES = FEATURES + ["rf_inftr", "rf_artec"]
DEMOGRAPHIC_FEATURES = FEATURES + ["mrace6", "mhisp_r", "meduc"]
QUALITY_MASKS = ["core_scoring_ready", "art_source_reported", "art_status_unknown_reported"]
OUTCOMES = ["target_hdp_proxy", "target_hdp_preterm37", "target_hdp_preterm34"]
AUDIT = ["mrace6", "mhisp_r", "meduc", "bmi_r", "rf_phype", "rf_ghype", "rf_ehype",
         "cig_0", "rf_inftr", "rf_artec", "f_rf_inft", "f_rf_inf_art",
         "illb_r", "lbo_rec", "oegest_comb"]
COLUMNS = list(dict.fromkeys(["source_year", "source_row", "target_hdp_proxy"] + RAW_INPUTS + AUDIT))
NCHS_GHYPE = {2023: 361_780, 2024: 377_267}
SEED = 20261001


def source_hashes() -> dict[str, str]:
    registry = ROOT / "data/registry/checksums.sha256"
    result = {}
    for line in registry.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        if "cdc_natality" in relative:
            result[relative] = digest
    return result


def fingerprint_config() -> str:
    content = json.dumps({"raw_inputs": RAW_INPUTS, "features": FEATURES,
                          "engineered": ENGINEERED, "quality_masks": QUALITY_MASKS,
                          "early": EARLY_FEATURES,
                          "demographic": DEMOGRAPHIC_FEATURES, "seed": SEED,
                          "cohort": "resident singleton, known HDP proxy, recorded no chronic hypertension",
                          "design": "pooled 2023+2024 five folds plus temporal analysis",
                          "outcomes": OUTCOMES, "sources": source_hashes()}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def split_bucket(rows: np.ndarray) -> np.ndarray:
    """Stable 64-bit splitmix hash; 0..999 buckets independent of row order."""
    z = rows.astype(np.uint64) + np.uint64(SEED) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(1000)).astype(np.uint16)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in COLUMNS[2:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # The 2023/24 primary cohort has complete age and historical risk flags.
    # If this changes, refuse to silently learn an unsupported missing code.
    for column in ["mager", *BINARY]:
        if df[column].isna().any():
            raise ValueError(f"Required early-history input is missing: {column}")
    for column in BINARY:
        bad = df[column].notna() & ~df[column].isin([0, 1])
        if bad.any():
            raise ValueError(f"Unexpected code in {column}: {df.loc[bad, column].unique()}")
    if not df["target_hdp_proxy"].isin([0, 1]).all():
        raise ValueError("Cohort includes unknown or invalid target")
    if not df["mager"].dropna().between(10, 55).all():
        raise ValueError("Invalid maternal age")
    if not df["bmi"].dropna().between(13, 69.9).all():
        raise ValueError("Invalid prepregnancy BMI")
    for column in ["priorlive", "priordead", "priorterm"]:
        if not df[column].dropna().between(0, 30).all():
            raise ValueError(f"Invalid {column}")
    # Unknown input stays unknown: it must never be interpreted as absence of risk.
    known_prior_live_birth = df["priorlive"].notna() & df["priordead"].notna()
    df["no_prior_live_birth"] = np.where(known_prior_live_birth, (df["priorlive"].eq(0) & df["priordead"].eq(0)).astype(float), np.nan)
    df["prior_live_birth_count_cat"] = np.where(known_prior_live_birth,
        np.minimum(df["priorlive"].fillna(0) + df["priordead"].fillna(0), 3), np.nan)
    df["bmi_missing"] = df["bmi"].isna().astype(np.int8)
    df["prior_live_birth_count_missing"] = (~known_prior_live_birth).astype(np.int8)
    df["priorterm_missing"] = df["priorterm"].isna().astype(np.int8)
    df["core_scoring_ready"] = 1
    df["art_source_reported"] = (df["f_rf_inft"].eq(1) & df["f_rf_inf_art"].eq(1)).astype(np.int8)
    df["art_status_unknown_reported"] = (df["art_source_reported"].eq(1) &
        (df["rf_inftr"].isna() | df["rf_artec"].isna())).astype(np.int8)
    if (df.loc[df.art_source_reported.eq(0), ["rf_inftr", "rf_artec"]].notna().any().any()):
        raise ValueError("Unreported ART source contains an interpreted treatment code")
    if (not df["rf_inftr"].dropna().isin([0, 1]).all() or
            not df["rf_artec"].dropna().isin([0, 1]).all()):
        raise ValueError("ART fields contain an unsupported code")
    df["age_35_plus"] = np.where(df["mager"].notna(), df["mager"].ge(35).astype(float), np.nan)
    for threshold in (30, 35):
        df[f"bmi_{threshold}_plus"] = np.where(df["bmi"].notna(), df["bmi"].ge(threshold).astype(float), np.nan)
    if df["rf_phype"].ne(0).any():
        raise ValueError("The primary cohort must have recorded absence of chronic hypertension")
    for threshold in (37, 34):
        df[f"target_hdp_preterm{threshold}"] = np.where(
            df["oegest_comb"].notna(),
            (df["target_hdp_proxy"].eq(1) & df["oegest_comb"].lt(threshold)).astype(float),
            np.nan,
        )
    # LBO_REC is current live-birth order and is audit-only: 1 + previous live births.
    known_order = known_prior_live_birth & df["lbo_rec"].notna() & df["lbo_rec"].lt(8)
    if (df.loc[known_order, "lbo_rec"] !=
            df.loc[known_order, "priorlive"] + df.loc[known_order, "priordead"] + 1).any():
        raise ValueError("Previous live-birth counts contradict recorded live-birth order")
    return df


def cohort_query(year: int) -> str:
    if year not in (2023, 2024):
        raise ValueError(year)
    return f"SELECT {', '.join(COLUMNS)} FROM births_baseline WHERE source_year={year} AND restatus IN ('1','2','3') AND dplural=1 AND rf_phype=0 AND target_hdp_proxy IN (0,1) ORDER BY source_row"


def iter_cohort(year: int, chunksize: int = 100_000):
    with sqlite3.connect(DATABASE) as connection:
        for chunk in pd.read_sql_query(cohort_query(year), connection, chunksize=chunksize):
            yield clean_frame(chunk)


def audit_database() -> dict:
    if not DATABASE.exists():
        raise FileNotFoundError(DATABASE)
    for relative, expected in source_hashes().items():
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    with sqlite3.connect(DATABASE) as connection:
        metadata = pd.read_sql_query("SELECT * FROM source_metadata ORDER BY source_year", connection)
        if metadata.source_year.tolist() != [2023, 2024]:
            raise ValueError("Unexpected source years")
        checks = {}
        for year in (2023, 2024):
            m = metadata.loc[metadata.source_year.eq(year)].iloc[0]
            resident = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=?", (year,)).fetchone()[0]
            distinct_rows = connection.execute("SELECT count(DISTINCT source_row) FROM births_baseline WHERE source_year=?", (year,)).fetchone()[0]
            ghype = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND rf_ghype=1", (year,)).fetchone()[0]
            known = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND target_hdp_proxy IN (0,1)", (year,)).fetchone()[0]
            wrong_labels = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND target_hdp_proxy IS NOT (CASE WHEN rf_ghype IS NULL OR rf_ehype IS NULL THEN NULL WHEN rf_ghype=1 OR rf_ehype=1 THEN 1 ELSE 0 END)", (year,)).fetchone()[0]
            singleton = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND dplural=1 AND target_hdp_proxy IN (0,1)", (year,)).fetchone()[0]
            events = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND dplural=1 AND target_hdp_proxy=1", (year,)).fetchone()[0]
            eligible = connection.execute("SELECT count(*),sum(target_hdp_proxy),sum(CASE WHEN rf_phype=1 THEN 1 ELSE 0 END) FROM births_baseline WHERE source_year=? AND dplural=1 AND rf_phype=0 AND target_hdp_proxy IN (0,1)", (year,)).fetchone()
            chronic = connection.execute("SELECT count(*),sum(target_hdp_proxy) FROM births_baseline WHERE source_year=? AND dplural=1 AND rf_phype=1 AND target_hdp_proxy IN (0,1)", (year,)).fetchone()
            cooccurrence = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND dplural=1 AND target_hdp_proxy IN (0,1) AND rf_phype=1 AND rf_ghype=1", (year,)).fetchone()[0]
            eclampsia_only = connection.execute("SELECT count(*) FROM births_baseline WHERE source_year=? AND dplural=1 AND target_hdp_proxy IN (0,1) AND rf_ehype=1 AND rf_ghype=0", (year,)).fetchone()[0]
            assert resident == distinct_rows == int(m.row_count)
            assert int(m.raw_row_count) - resident >= 0
            assert ghype == NCHS_GHYPE[year], (year, ghype)
            assert wrong_labels == 0, (year, wrong_labels)
            assert eligible[2] == 0
            checks[str(year)] = {"raw_rows": int(m.raw_row_count), "resident_rows": resident, "foreign_excluded": int(m.raw_row_count)-resident, "ghype_yes_official_match": ghype, "unknown_target": resident-known, "singleton_before_chronic_exclusion": singleton, "events_before_chronic_exclusion": events, "excluded_chronic_rows": chronic[0], "excluded_chronic_recorded_events": chronic[1], "eligible_singleton_rows": eligible[0], "eligible_singleton_events": eligible[1], "eligible_singleton_prevalence": eligible[1]/eligible[0], "chronic_and_gestational_hypertension_cooccurrence": cooccurrence, "eclampsia_without_ghype": eclampsia_only}
    return {"source_hashes": source_hashes(), "config_fingerprint": fingerprint_config(), "year_checks": checks, "grain": "one singleton live birth; maternal identity unavailable"}


def audit_race_imputation() -> pd.DataFrame:
    """Audit NCHS race imputation on the exact cohort, without using it to fit models."""
    audit = json.loads((OUT / "audit.json").read_text())
    needed = {"restatus", "dplural", "rf_phype", "rf_ghype", "rf_ehype",
              "mrace6", "mraceimp"}
    rows = []
    for year in (2023, 2024):
        source = ROOT / f"data/external/cdc_natality/{year}/natality{year}us.csv.zip"
        observed = 0
        with ZipFile(source) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one natality CSV for {year}")
            with archive.open(members[0]) as stream:
                chunks = pd.read_csv(stream, usecols=lambda name: name.lower() in needed,
                                     dtype="string", chunksize=200_000)
                for chunk in chunks:
                    chunk.columns = chunk.columns.str.lower()
                    if set(chunk.columns) != needed:
                        raise ValueError(f"Missing race-audit source column for {year}")
                    eligible = (chunk.restatus.isin(["1", "2", "3"]) & chunk.dplural.eq("1") &
                                chunk.rf_phype.eq("N") & chunk.rf_ghype.isin(["Y", "N"]) &
                                chunk.rf_ehype.isin(["Y", "N"]))
                    selected = chunk.loc[eligible, ["mrace6", "mraceimp"]].copy()
                    if not selected.mrace6.isin(["1", "2", "3", "4", "5", "6"]).all():
                        raise ValueError(f"Unexpected maternal race code for {year}")
                    if not selected.mraceimp.fillna("").isin(["", "1", "2"]).all():
                        raise ValueError(f"Unexpected race-imputation code for {year}")
                    selected["imputed"] = selected.mraceimp.isin(["1", "2"])
                    observed += len(selected)
                    counts = selected.groupby(["mrace6", "imputed"], observed=True).size()
                    rows.extend({"year": year, "mrace6": int(race), "imputed": int(imputed),
                                 "rows": int(count)} for (race, imputed), count in counts.items())
        if observed != audit["year_checks"][str(year)]["eligible_singleton_rows"]:
            raise ValueError(f"Race-imputation audit does not match {year} cohort size")
    result = pd.DataFrame(rows).groupby(["year", "mrace6", "imputed"], as_index=False).rows.sum()
    result.to_csv(OUT / "race_imputation_by_year.csv", index=False)
    return result


def prepare_all() -> dict:
    """Write one pooled 2023+2024 table, with five folds on every row."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.progress import tqdm

    OUT.mkdir(parents=True, exist_ok=True)
    audit = audit_database()
    output = OUT / "eligible_all_2023_2024.parquet"
    building = OUT / "eligible_all_2023_2024.building.parquet"
    building.unlink(missing_ok=True)
    writer = None
    fold_counts = np.zeros((2, 5, 2), dtype=np.int64)
    missing_rows = []
    mask_rows = []
    try:
        for year in (2023, 2024):
            seen = 0
            missing = {x: 0 for x in list(dict.fromkeys(ENRICHED_FEATURES + QUALITY_MASKS + AUDIT + OUTCOMES))}
            mask_counts = {name: [0, 0] for name in MISSING_MASKS + QUALITY_MASKS}
            outcome_counts = {name: [0, 0] for name in OUTCOMES}
            progress = tqdm(total=audit["year_checks"][str(year)]["eligible_singleton_rows"], desc=f"CDC {year}: prepare", unit="rows", unit_scale=True)
            for chunk in iter_cohort(year):
                composite_id = (np.uint64(year) << np.uint64(32)) | chunk.source_row.to_numpy(dtype=np.uint64)
                chunk["fold"] = (split_bucket(composite_id) % 5).astype(np.int8)
                for value, mask in (("bmi", "bmi_missing"), ("prior_live_birth_count_cat", "prior_live_birth_count_missing"), ("priorterm", "priorterm_missing")):
                    if not np.array_equal(chunk[value].isna().to_numpy(), chunk[mask].eq(1).to_numpy()):
                        raise ValueError(f"Missingness mask mismatch: {value}")
                if not chunk.core_scoring_ready.eq(1).all():
                    raise ValueError("A row lacks the required core inputs")
                for name in mask_counts:
                    present = chunk[name].eq(1)
                    mask_counts[name][0] += int(present.sum())
                    mask_counts[name][1] += int(chunk.loc[present, "target_hdp_proxy"].sum())
                for column in list(dict.fromkeys(ENRICHED_FEATURES + QUALITY_MASKS + AUDIT + OUTCOMES)):
                    missing[column] = missing.get(column, 0) + int(chunk[column].isna().sum())
                    if column not in OUTCOMES:
                        chunk[column] = chunk[column].astype(np.float32)
                for name in OUTCOMES:
                    outcome_counts[name][0] += int(chunk[name].notna().sum())
                    outcome_counts[name][1] += int(chunk[name].sum())
                chunk["source_year"] = chunk["source_year"].astype(np.int16)
                chunk["source_row"] = chunk["source_row"].astype(np.int32)
                chunk["target_hdp_proxy"] = chunk["target_hdp_proxy"].astype(np.uint8)
                chunk["fold"] = chunk["fold"].astype(np.int8)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(building, table.schema, compression="zstd")
                writer.write_table(table)
                seen += len(chunk)
                progress.update(len(chunk))
                year_index = year - 2023
                for fold in range(5):
                    rows = chunk.loc[chunk.fold.eq(fold), "target_hdp_proxy"]
                    fold_counts[year_index, fold, 0] += len(rows)
                    fold_counts[year_index, fold, 1] += int(rows.sum())
            progress.close()
            if seen != audit["year_checks"][str(year)]["eligible_singleton_rows"]:
                raise ValueError(f"Year {year}: count mismatch")
            audit["year_checks"][str(year)]["outcomes"] = {name: {"known": counts[0], "events": counts[1]} for name, counts in outcome_counts.items()}
            for column in missing:
                missing_rows.append({"year": year, "field": column, "missing_fraction": missing[column]/seen})
            for name, (count, events) in mask_counts.items():
                mask_rows.append({"year": year, "mask": name, "value": 1, "rows": count,
                                  "events": events, "event_rate": events/count if count else np.nan})
                total_events = audit["year_checks"][str(year)]["eligible_singleton_events"]
                mask_rows.append({"year": year, "mask": name, "value": 0, "rows": seen-count,
                                  "events": total_events-events,
                                  "event_rate": (total_events-events)/(seen-count) if seen > count else np.nan})
    finally:
        if writer is not None:
            writer.close()
    os.replace(building, output)
    fold_rows = []
    for year_index, year in enumerate((2023, 2024)):
        for fold in range(5):
            n, events = fold_counts[year_index, fold]
            fold_rows.append({"year": year, "fold": fold, "rows": int(n), "events": int(events), "prevalence": float(events/n)})
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(OUT / "folds_all.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(OUT / "missingness_by_year.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(OUT / "input_masks_by_year.csv", index=False)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    audit["prepared_file"] = {"path": str(output.relative_to(ROOT)), "sha256": digest.hexdigest()}
    audit["cross_validation"] = {"folds": fold_rows, "assignment": "stable splitmix64((year << 32) | source_row) modulo 5; 2023 and 2024 are pooled; every eligible row belongs to exactly one fold", "temporal_test": "separate 2023 development to 2024 evaluation notebook"}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
