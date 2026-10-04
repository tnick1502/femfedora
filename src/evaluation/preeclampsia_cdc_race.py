"""Record-level NCHS race-imputation audit, never used as a model input."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.preeclampsia_cdc import OUT, ROOT
from src.progress import tqdm

RACE_ROWS = OUT / "race_imputation_rows.parquet"
SOURCE_FIELDS = {"restatus", "dplural", "rf_phype", "rf_ghype", "rf_ehype",
                 "mrace6", "mraceimp"}


def _selected_race_rows(chunk: pd.DataFrame, year: int, offset: int) -> pd.DataFrame:
    """Use raw CSV row positions, the same IDs assigned by prepare_cdc_chunk."""
    chunk.columns = chunk.columns.str.lower()
    if set(chunk.columns) != SOURCE_FIELDS:
        raise ValueError(f"Missing NCHS race audit fields in {year}")
    eligible = (chunk.restatus.isin(["1", "2", "3"]) & chunk.dplural.eq("1") &
                chunk.rf_phype.eq("N") & chunk.rf_ghype.isin(["Y", "N"]) &
                chunk.rf_ehype.isin(["Y", "N"]))
    part = chunk.loc[eligible, ["mrace6", "mraceimp"]]
    if not part.mrace6.isin(["1", "2", "3", "4", "5", "6"]).all():
        raise ValueError(f"Unexpected MRACE6 in {year}")
    if not part.mraceimp.fillna("").isin(["", "1", "2"]).all():
        raise ValueError(f"Unexpected MRACEIMP in {year}")
    return pd.DataFrame({
        "source_year": np.full(len(part), year, dtype=np.int16),
        "source_row": part.index.to_numpy(dtype=np.int32) - chunk.index[0] + offset,
        "mrace6": part.mrace6.to_numpy(dtype=np.int8),
        "race_imputed": part.mraceimp.isin(["1", "2"]).to_numpy(dtype=np.uint8),
    })


def build_race_imputation_rows() -> Path:
    """Stream source archives into an independent sidecar; retain model caches."""
    audit = json.loads((OUT / "audit.json").read_text())
    temporary = RACE_ROWS.with_suffix(".building.parquet")
    writer = None
    totals = {}
    try:
        for year in (2023, 2024):
            source = ROOT / f"data/external/cdc_natality/{year}/natality{year}us.csv.zip"
            with ZipFile(source) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"Expected one CSV in {source}")
                offset, selected = 0, 0
                with archive.open(members[0]) as stream:
                    chunks = pd.read_csv(stream, usecols=lambda name: name.lower() in SOURCE_FIELDS,
                                         dtype="string", chunksize=200_000)
                    for chunk in tqdm(chunks, desc=f"Race audit {year}", unit="chunk"):
                        part = _selected_race_rows(chunk, year, offset)
                        offset += len(chunk)
                        selected += len(part)
                        table = pa.Table.from_pandas(part, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                        writer.write_table(table)
            expected = audit["year_checks"][str(year)]
            if offset != expected["raw_rows"] or selected != expected["eligible_singleton_rows"]:
                raise ValueError(f"Race sidecar fails source/cohort count check for {year}")
            totals[year] = selected
        if writer is not None:
            writer.close()
            writer = None
        sidecar = pd.read_parquet(temporary)
        source_rows = pd.read_parquet(OUT / "eligible_all_2023_2024.parquet",
                                      columns=["source_year", "source_row", "mrace6"])
        keys = ["source_year", "source_row"]
        if (len(sidecar) != len(source_rows) or sidecar.duplicated(keys).any() or
                source_rows.duplicated(keys).any()):
            raise ValueError("Race sidecar has missing or duplicate cohort rows")
        sidecar = sidecar.sort_values(keys).reset_index(drop=True)
        source_rows = source_rows.sort_values(keys).reset_index(drop=True)
        for field in [*keys, "mrace6"]:
            if not np.array_equal(sidecar[field].to_numpy(), source_rows[field].to_numpy()):
                raise ValueError(f"Race sidecar disagrees with prepared cohort: {field}")
        aggregate = pd.read_csv(OUT / "race_imputation_by_year.csv")
        grouped = sidecar.groupby(["source_year", "mrace6", "race_imputed"], observed=True).size()
        expected_grouped = aggregate.set_index(["year", "mrace6", "imputed"]).rows
        if not grouped.equals(expected_grouped.reindex(grouped.index)):
            raise ValueError("Record-level race flag does not reconcile with source audit")
        temporary.replace(RACE_ROWS)
        return RACE_ROWS
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def race_imputation_sensitivity(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compare mean calibration by published race with/without imputed rows."""
    if not RACE_ROWS.exists():
        raise FileNotFoundError(f"Run build_race_imputation_rows() first: {RACE_ROWS}")
    required = ["source_year", "source_row", "mrace6", "label", *models]
    if len(set(required)) != len(required) or not set(required).issubset(frame):
        raise ValueError("Race sensitivity needs unique keys, labels, race, and scores")
    keys = ["source_year", "source_row"]
    joined = frame[required].merge(pd.read_parquet(RACE_ROWS), on=keys,
                                   how="left", validate="one_to_one", suffixes=("", "_source"))
    if (len(joined) != len(frame) or joined.race_imputed.isna().any() or
            not joined.mrace6.eq(joined.mrace6_source).all()):
        raise ValueError("Race sidecar does not match scored cohort")
    records = []
    for (year, race), group in joined.groupby(["source_year", "mrace6"], observed=True):
        for subset, part in (("all", group), ("non_imputed", group.loc[group.race_imputed.eq(0)]),
                             ("imputed", group.loc[group.race_imputed.eq(1)])):
            if part.empty:
                continue
            observed = part.label.to_numpy(dtype=float)
            for model in models:
                predicted = part[model].to_numpy(dtype=float)
                records.append({"year": int(year), "race_code": int(race), "subset": subset,
                                "model": model, "n": len(part), "events": int(observed.sum()),
                                "event_rate": float(observed.mean()),
                                "mean_prediction": float(predicted.mean()),
                                "mean_calibration_gap_pp": float(100 * (observed.mean() - predicted.mean()))})
    return pd.DataFrame(records)
