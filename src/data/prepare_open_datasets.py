"""Prepare model-ready tables and signal indexes from registered open data.

The script keeps every source separate. It never joins records from unrelated
cohorts into a synthetic longitudinal pregnancy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CDC_USECOLS = [
    "mager",
    "mbstate_rec",
    "restatus",
    "mrace6",
    "mhisp_r",
    "dmar",
    "meduc",
    "priorlive",
    "priordead",
    "priorterm",
    "lbo_rec",
    "tbo_rec",
    "illb_r",
    "ilop_r",
    "ilp_r",
    "cig_0",
    "m_ht_in",
    "bmi",
    "bmi_r",
    "pwgt_r",
    "rf_pdiab",
    "rf_gdiab",
    "rf_phype",
    "rf_ghype",
    "rf_ehype",
    "rf_ppterm",
    "rf_inftr",
    "rf_fedrg",
    "rf_artec",
    "f_rf_inft",
    "f_rf_inf_art",
    "rf_cesar",
    "rf_cesarn",
    "dplural",
    "sex",
    "oegest_comb",
    "dbwt",
]

CDC_NUMERIC_RANGES = {
    "mager": (10, 55),
    "mbstate_rec": (1, 3),
    "mrace6": (1, 6),
    "mhisp_r": (0, 9),
    "meduc": (1, 8),
    "priorlive": (0, 30),
    "priordead": (0, 30),
    "priorterm": (0, 30),
    "lbo_rec": (1, 8),
    "tbo_rec": (1, 8),
    "cig_0": (0, 97),
    "m_ht_in": (30, 78),
    "bmi": (13, 69.9),
    "bmi_r": (1, 6),
    "pwgt_r": (75, 375),
    "rf_cesarn": (0, 30),
    "dplural": (1, 5),
    "oegest_comb": (17, 47),
    "dbwt": (227, 8165),
}

YES_NO_COLUMNS = [
    "rf_pdiab",
    "rf_gdiab",
    "rf_phype",
    "rf_ghype",
    "rf_ehype",
    "rf_ppterm",
    "rf_cesar",
]


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            output = (destination / member.filename).resolve()
            if destination_root not in output.parents and output != destination_root:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def normalized_yes_no(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    return values.map({"Y": 1, "N": 0}).astype("Int8")


def prepare_cdc_chunk(chunk: pd.DataFrame, year: int, row_offset: int) -> pd.DataFrame:
    chunk.columns = [column.lower() for column in chunk.columns]
    chunk.insert(0, "source_row", range(row_offset, row_offset + len(chunk)))
    chunk.insert(0, "source_year", year)

    chunk["restatus"] = chunk["restatus"].astype("string").str.strip()
    # The national resident-birth denominator excludes category 4
    # (foreign residents). Retain the source field for population audits.
    chunk = chunk[chunk["restatus"].isin(["1", "2", "3"])].copy()
    chunk["dmar"] = chunk["dmar"].astype("string").str.strip().replace({"": pd.NA})
    chunk["sex"] = chunk["sex"].astype("string").str.strip().replace({"": pd.NA})
    for column in YES_NO_COLUMNS:
        chunk[column] = normalized_yes_no(chunk[column])
    for column in ("f_rf_inft", "f_rf_inf_art"):
        flag = pd.to_numeric(chunk[column], errors="coerce")
        chunk[column] = flag.where(flag.isin([0, 1])).astype("Int8")
    chunk["rf_inftr"] = normalized_yes_no(chunk["rf_inftr"]).where(chunk["f_rf_inft"].eq(1)).astype("Int8")
    for column in ("rf_fedrg", "rf_artec"):
        raw = chunk[column].astype("string").str.strip().str.upper()
        decoded = normalized_yes_no(raw)
        not_applicable_zero = raw.eq("X").fillna(False) & chunk["rf_inftr"].eq(0).fillna(False)
        decoded = decoded.mask(not_applicable_zero, 0)
        chunk[column] = decoded.where(chunk["f_rf_inf_art"].eq(1)).astype("Int8")
    # 888 means no preceding event; 999 is unknown. Preserve the distinction.
    for column in ("illb_r", "ilop_r", "ilp_r"):
        numeric = pd.to_numeric(chunk[column], errors="coerce")
        chunk[column] = numeric.where(numeric.between(0, 300) | numeric.eq(888))
    for column, (minimum, maximum) in CDC_NUMERIC_RANGES.items():
        numeric = pd.to_numeric(chunk[column], errors="coerce")
        chunk[column] = numeric.where(numeric.between(minimum, maximum))

    gestation = chunk["oegest_comb"]
    chunk["target_ptb37"] = gestation.lt(37).where(gestation.notna()).astype("Int8")
    chunk["target_ptb32"] = gestation.lt(32).where(gestation.notna()).astype("Int8")
    chunk["target_gdm"] = chunk["rf_gdiab"].astype("Int8")
    hdp_known = chunk[["rf_ghype", "rf_ehype"]].notna().all(axis=1)
    hdp_positive = chunk[["rf_ghype", "rf_ehype"]].eq(1).any(axis=1)
    chunk["target_hdp_proxy"] = hdp_positive.where(hdp_known).astype("Int8")
    chunk["target_eclampsia"] = chunk["rf_ehype"].astype("Int8")
    chunk["target_low_birthweight_2500g"] = (
        chunk["dbwt"].lt(2500).where(chunk["dbwt"].notna()).astype("Int8")
    )
    return chunk


def prepare_cdc_sqlite() -> Path:
    output = PROJECT_ROOT / "data" / "processed" / "cdc_natality" / "natality_ml_2023_2024.sqlite"
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_suffix(".building.sqlite")
    building.unlink(missing_ok=True)

    sources = {
        2023: PROJECT_ROOT / "data" / "external" / "cdc_natality" / "2023" / "natality2023us.csv.zip",
        2024: PROJECT_ROOT / "data" / "external" / "cdc_natality" / "2024" / "natality2024us.csv.zip",
    }
    connection = sqlite3.connect(building)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")

    total_rows = 0
    metadata_rows: list[tuple[int, str, int, int, str]] = []
    try:
        for year, archive_path in sources.items():
            if not archive_path.exists():
                raise FileNotFoundError(archive_path)
            with ZipFile(archive_path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise RuntimeError(f"Expected one CSV in {archive_path}, found {members}")
                year_rows = 0
                raw_rows = 0
                with archive.open(members[0]) as stream:
                    reader = pd.read_csv(
                        stream,
                        usecols=lambda name: name.lower() in CDC_USECOLS,
                        chunksize=100_000,
                        low_memory=False,
                    )
                    for chunk in reader:
                        source_chunk_rows = len(chunk)
                        prepared = prepare_cdc_chunk(chunk, year, raw_rows)
                        prepared.to_sql("births_baseline", connection, if_exists="append", index=False)
                        raw_rows += source_chunk_rows
                        year_rows += len(prepared)
                        total_rows += len(prepared)
                        print(f"CDC {year}: {year_rows:,} resident rows")
                metadata_rows.append(
                    (
                        year,
                        str(archive_path.relative_to(PROJECT_ROOT)),
                        raw_rows,
                        year_rows,
                        "restatus in {1,2,3}; excludes foreign residents (4)",
                    )
                )

        connection.execute(
            "CREATE TABLE source_metadata (source_year INTEGER, source_path TEXT, raw_row_count INTEGER, "
            "row_count INTEGER, population_filter TEXT)"
        )
        connection.executemany("INSERT INTO source_metadata VALUES (?, ?, ?, ?, ?)", metadata_rows)
        connection.execute("CREATE UNIQUE INDEX births_source_row ON births_baseline(source_year, source_row)")
        for target in ["target_ptb37", "target_ptb32", "target_gdm", "target_hdp_proxy"]:
            connection.execute(f"CREATE INDEX idx_{target} ON births_baseline(source_year, {target})")
        connection.commit()
    finally:
        connection.close()
    os.replace(building, output)
    print(f"prepared CDC SQLite: {output.relative_to(PROJECT_ROOT)} ({total_rows:,} rows)")
    return output


def header_metadata(lines: list[str], dataset_id: str) -> dict[str, str | int | float]:
    """Extract a small, stable set of modelling fields from WFDB comments."""
    comments = [line[1:].strip() for line in lines if line.startswith("#")]
    if dataset_id == "ctu_uhb_ctgdb_1_0_0":
        fields = {
            "pH": "cord_ph",
            "BDecf": "base_deficit_ecf",
            "Apgar1": "apgar_1min",
            "Apgar5": "apgar_5min",
            "Gest. weeks": "gestation_weeks",
            "Weight(g)": "birth_weight_g",
            "Age": "maternal_age",
            "Gravidity": "gravidity",
            "Parity": "parity",
            "Diabetes": "diabetes",
            "Hypertension": "hypertension",
            "Preeclampsia": "preeclampsia",
            "Induced": "induced",
            "Deliv. type": "delivery_type",
        }
    else:
        fields = {
            "Gestation": "gestation_weeks",
            "Rectime": "recording_gestational_age_weeks",
            "Age": "maternal_age",
            "Parity": "parity",
            "Abortions": "abortions",
            "Weight": "maternal_weight_kg",
            "Hypertension": "hypertension",
            "Diabetes": "diabetes",
            "Funneling": "funneling",
            "Smoker": "smoker",
        }

    result: dict[str, str | int | float] = {}
    for label, column in fields.items():
        for comment in comments:
            if comment == label or comment.startswith(label + " "):
                value = comment[len(label):].strip()
                try:
                    result[column] = float(value) if "." in value else int(value)
                except ValueError:
                    result[column] = value
                break
    if dataset_id == "tpehgdb_1_0_1":
        gestation = result.get("gestation_weeks")
        recording_age = result.get("recording_gestational_age_weeks")
        result["target_preterm"] = int(float(gestation) < 37) if gestation != "" and gestation is not None else ""
        result["recorded_before_26_weeks"] = (
            int(float(recording_age) < 26) if recording_age != "" and recording_age is not None else ""
        )
    return result


def build_header_index(root: Path, output: Path, dataset_id: str) -> None:
    records: list[dict[str, str | int]] = []
    for header_path in sorted(root.rglob("*.hea")):
        lines = header_path.read_text(encoding="latin-1", errors="replace").splitlines()
        first = lines[0].split() if lines else []
        comments = [line[1:].strip() for line in lines if line.startswith("#")]
        record = {
            "dataset_id": dataset_id,
            "record_id": header_path.stem,
            "header_path": str(header_path.relative_to(PROJECT_ROOT)),
            "signal_path": str(header_path.with_suffix(".dat").relative_to(PROJECT_ROOT)),
            "signal_count": int(first[1]) if len(first) > 1 and first[1].isdigit() else "",
            "sample_frequency_hz": first[2] if len(first) > 2 else "",
            "sample_count": first[3] if len(first) > 3 else "",
            "comments": " | ".join(comments),
        }
        record.update(header_metadata(lines, dataset_id))
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["dataset_id"])
        writer.writeheader()
        writer.writerows(records)
    print(f"signal index: {output.relative_to(PROJECT_ROOT)} ({len(records)} records)")


def prepare_signal_archives() -> None:
    ctu_files = PROJECT_ROOT / "data" / "external" / "ctu_uhb_ctg" / "1.0.0" / "files"
    if not ctu_files.exists():
        raise FileNotFoundError(ctu_files)
    build_header_index(
        ctu_files,
        PROJECT_ROOT / "data" / "processed" / "ctu_uhb_ctg" / "record_index.csv",
        "ctu_uhb_ctgdb_1_0_0",
    )

    tpehg_files = PROJECT_ROOT / "data" / "external" / "tpehg" / "1.0.1" / "files"
    if not tpehg_files.exists():
        raise FileNotFoundError(tpehg_files)
    build_header_index(
        tpehg_files,
        PROJECT_ROOT / "data" / "processed" / "tpehg" / "record_index.csv",
        "tpehgdb_1_0_1",
    )
    for feature_file in tpehg_files.rglob("*.fvl"):
        destination = PROJECT_ROOT / "data" / "processed" / "tpehg" / f"{feature_file.stem}.csv"
        lines = [line.strip() for line in feature_file.read_text(encoding="latin-1").splitlines() if line.strip()]
        rows = [
            [cell.strip() for cell in line.split("|")]
            for line in lines
            if "|" in line and not set(line.replace("+", "").replace("|", "").strip()) <= {"-"}
        ]
        if len(rows) > 1:
            rows[0] = [
                column.lower().replace(".", "").replace(" ", "_").replace("_/_", "_")
                for column in rows[0]
            ]
            with destination.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)


def prepare_uci() -> None:
    archive_path = PROJECT_ROOT / "data" / "external" / "uci_cardiotocography" / "1.0.0" / "cardiotocography.zip"
    extracted = archive_path.parent / "files"
    extract_zip_safely(archive_path, extracted)
    xls_files = list(extracted.rglob("*.xls"))
    if len(xls_files) != 1:
        raise RuntimeError(f"Expected one XLS file, found: {xls_files}")
    converted = extracted / "CTG.xlsx"
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "xlsx", "--outdir", str(extracted), str(xls_files[0])],
        check=True,
        capture_output=True,
        text=True,
    )
    workbook = pd.ExcelFile(converted)
    preferred = next((name for name in workbook.sheet_names if name.strip().lower() == "data"), None)
    if preferred is None:
        preferred = next((name for name in workbook.sheet_names if name.strip().lower() == "raw data"), None)
    if preferred is None:
        raise RuntimeError(f"No data sheet found in {converted}: {workbook.sheet_names}")
    if preferred.strip().lower() == "data":
        raw = pd.read_excel(converted, sheet_name=preferred, header=1)
        feature_names = [
            "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV",
            "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance",
            "Tendency",
        ]
        frame = pd.concat([raw.iloc[:, 10:31], raw.iloc[:, [43, 45]]], axis=1)
        frame.columns = feature_names + ["CLASS", "NSP"]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame[frame["NSP"].isin([1, 2, 3])].reset_index(drop=True)
    else:
        frame = pd.read_excel(converted, sheet_name=preferred)
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
    duplicate_key = pd.util.hash_pandas_object(frame.drop(columns=["CLASS"], errors="ignore"), index=False)
    frame.insert(0, "duplicate_group_id", pd.factorize(duplicate_key, sort=False)[0])
    frame.insert(0, "source_row", range(len(frame)))
    output = PROJECT_ROOT / "data" / "processed" / "uci_cardiotocography" / "ctg_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"UCI CTG: {output.relative_to(PROJECT_ROOT)} ({len(frame)} rows, {len(frame.columns)} columns)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["cdc", "signals", "uci"])
    args = parser.parse_args()
    if args.only in (None, "cdc"):
        prepare_cdc_sqlite()
    if args.only in (None, "signals"):
        prepare_signal_archives()
    if args.only in (None, "uci"):
        prepare_uci()


if __name__ == "__main__":
    main()
