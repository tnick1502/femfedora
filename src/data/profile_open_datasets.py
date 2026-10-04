"""Create a compact, inspectable intake report for the open datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "open_datasets.json"
REPORT_JSON = PROJECT_ROOT / "research" / "reports" / "open_data_intake.json"
REPORT_MD = PROJECT_ROOT / "research" / "reports" / "open_data_intake.md"
CHECKSUM_PATH = PROJECT_ROOT / "data" / "registry" / "checksums.sha256"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def profile_archives() -> tuple[list[dict], list[str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    profiles: list[dict] = []
    checksum_lines: list[str] = []
    for entry in registry["datasets"]:
        path = PROJECT_ROOT / entry["local_path"]
        profile = {
            "dataset_id": entry["id"],
            "local_path": entry["local_path"],
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "source_valid": False,
            "file_count": None,
            "sha256": None,
        }
        if path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
            profile["file_count"] = len(files)
            profile["source_valid"] = bool(files)
            profile["size_bytes"] = sum(item.stat().st_size for item in files)
            if profile["source_valid"]:
                profile["sha256"] = directory_sha256(path)
                checksum_lines.append(f"{profile['sha256']}  {entry['local_path']}/")
        elif path.exists():
            try:
                with ZipFile(path) as archive:
                    profile["file_count"] = len(archive.infolist())
                    profile["source_valid"] = archive.testzip() is None
            except (BadZipFile, OSError):
                profile["source_valid"] = False
            if profile["source_valid"]:
                profile["sha256"] = sha256(path)
                checksum_lines.append(f"{profile['sha256']}  {entry['local_path']}")
        profiles.append(profile)
    return profiles, checksum_lines


def profile_cdc() -> dict:
    database = PROJECT_ROOT / "data" / "processed" / "cdc_natality" / "natality_ml_2023_2024.sqlite"
    if not database.exists():
        return {"available": False, "path": str(database.relative_to(PROJECT_ROOT))}

    connection = sqlite3.connect(database)
    targets = ["target_ptb37", "target_ptb32", "target_gdm", "target_hdp_proxy", "target_eclampsia"]
    features = ["mager", "meduc", "priorlive", "cig_0", "bmi", "pwgt_r", "rf_ppterm", "dplural"]
    try:
        rows_by_year = {
            str(year): rows
            for year, rows in connection.execute(
                "SELECT source_year, COUNT(*) FROM births_baseline GROUP BY source_year ORDER BY source_year"
            )
        }
        prevalence = []
        for target in targets:
            query = (
                f"SELECT source_year, COUNT({target}) AS known, SUM({target}) AS positive, "
                f"AVG({target}) AS prevalence FROM births_baseline "
                f"GROUP BY source_year ORDER BY source_year"
            )
            for year, known, positive, rate in connection.execute(query):
                prevalence.append(
                    {
                        "source_year": year,
                        "target": target,
                        "known": known,
                        "positive": int(positive or 0),
                        "prevalence": rate,
                    }
                )
        missingness = []
        for feature in features:
            query = (
                f"SELECT source_year, COUNT(*) AS n, SUM(CASE WHEN {feature} IS NULL THEN 1 ELSE 0 END) "
                f"FROM births_baseline GROUP BY source_year ORDER BY source_year"
            )
            for year, n, missing in connection.execute(query):
                missingness.append(
                    {
                        "source_year": year,
                        "feature": feature,
                        "missing": missing,
                        "missing_rate": missing / n,
                    }
                )
        duplicate_keys = connection.execute(
            "SELECT COUNT(*) FROM (SELECT source_year, source_row, COUNT(*) n FROM births_baseline "
            "GROUP BY source_year, source_row HAVING n > 1)"
        ).fetchone()[0]
        source_metadata = [
            {
                "source_year": year,
                "raw_rows": raw_rows,
                "resident_rows": resident_rows,
                "excluded_rows": raw_rows - resident_rows,
                "population_filter": population_filter,
            }
            for year, raw_rows, resident_rows, population_filter in connection.execute(
                "SELECT source_year, raw_row_count, row_count, population_filter "
                "FROM source_metadata ORDER BY source_year"
            )
        ]
    finally:
        connection.close()
    return {
        "available": True,
        "path": str(database.relative_to(PROJECT_ROOT)),
        "size_bytes": database.stat().st_size,
        "rows_by_year": rows_by_year,
        "source_metadata": source_metadata,
        "target_prevalence": prevalence,
        "feature_missingness": missingness,
        "duplicate_source_keys": duplicate_keys,
    }


def profile_uci() -> dict:
    path = PROJECT_ROOT / "data" / "processed" / "uci_cardiotocography" / "ctg_features.csv"
    if not path.exists():
        return {"available": False, "path": str(path.relative_to(PROJECT_ROOT))}
    frame = pd.read_csv(path)
    model_columns = [column for column in frame.columns if column not in {"source_row", "duplicate_group_id"}]
    features_and_target = [column for column in model_columns if column != "CLASS"]
    return {
        "available": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(frame),
        "columns": len(frame.columns),
        "exact_duplicate_rows": int(frame[model_columns].duplicated().sum()),
        "duplicate_features_and_target": int(frame[features_and_target].duplicated().sum()),
        "nsp_counts": {str(int(key)): int(value) for key, value in frame["NSP"].value_counts().sort_index().items()},
        "null_cells": int(frame.isna().sum().sum()),
    }


def profile_signal_index(dataset: str) -> dict:
    path = PROJECT_ROOT / "data" / "processed" / dataset / "record_index.csv"
    if not path.exists():
        return {"available": False, "path": str(path.relative_to(PROJECT_ROOT))}
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing_signal_files = sum(not (PROJECT_ROOT / row["signal_path"]).exists() for row in rows)
    record_ids = [row["record_id"] for row in rows]
    duplicate_ids = sum(count - 1 for count in Counter(record_ids).values() if count > 1)
    result = {
        "available": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "records": len(rows),
        "missing_signal_files": missing_signal_files,
        "duplicate_record_ids": duplicate_ids,
    }
    if dataset == "tpehg":
        known = [row.get("target_preterm", "") for row in rows]
        result["known_preterm_labels"] = sum(value in {"0", "1"} for value in known)
        result["preterm_positive"] = sum(value == "1" for value in known)
    if dataset == "ctu_uhb_ctg":
        result["known_cord_ph"] = sum(row.get("cord_ph", "") != "" for row in rows)
    return result


def render_markdown(report: dict) -> str:
    lines = [
        "# Intake открытых данных",
        "",
        f"Дата проверки: {report['profiled_on']}",
        "",
        "## Архивы",
        "",
        "| Набор | Размер | Проверка | Файлов |",
        "|---|---:|---|---:|",
    ]
    for archive in report["archives"]:
        size_mib = archive["size_bytes"] / 1024 / 1024 if archive["size_bytes"] else 0
        lines.append(
            f"| `{archive['dataset_id']}` | {size_mib:.1f} MiB | "
            f"{'ok' if archive['source_valid'] else 'ошибка/нет'} | {archive['file_count'] or 0} |"
        )

    cdc = report["prepared"]["cdc_natality"]
    lines.extend(["", "## CDC/NCHS baseline", ""])
    if cdc.get("available"):
        lines.extend([
            "| Год | Строк |",
            "|---:|---:|",
            *[f"| {year} | {rows:,} |" for year, rows in cdc["rows_by_year"].items()],
            "",
            "Population: U.S. resident births (`RESTATUS` 1/2/3); foreign residents (`4`) are retained only in the raw archive.",
            "",
            "| Год | Target | Известно | Положительных | Доля |",
            "|---:|---|---:|---:|---:|",
        ])
        for item in cdc["target_prevalence"]:
            lines.append(
                f"| {item['source_year']} | `{item['target']}` | {item['known']:,} | "
                f"{item['positive']:,} | {item['prevalence']:.4%} |"
            )
        lines.append("")
        lines.append(
            "Риск: запись о родах не содержит честной продольной временной линии. HDP и GDM используются только как статические proxy-labels."
        )
    else:
        lines.append("Prepared SQLite ещё не создан.")

    uci = report["prepared"]["uci_cardiotocography"]
    lines.extend(["", "## UCI Cardiotocography", ""])
    if uci.get("available"):
        lines.append(
            f"{uci['rows']:,} строк, {uci['columns']} столбцов, {uci['exact_duplicate_rows']} полных дублей; "
            f"классы NSP: {uci['nsp_counts']}."
        )
        lines.append("Риск: экспертный класс не является прямым клиническим исходом.")
    else:
        lines.append("Prepared CSV ещё не создан.")

    lines.extend(["", "## Сигналы", ""])
    for key in ["ctu_uhb_ctg", "tpehg"]:
        profile = report["prepared"][key]
        if profile.get("available"):
            sentence = (
                f"- `{key}`: {profile['records']} записей; отсутствующих `.dat`: "
                f"{profile['missing_signal_files']}; дублирующихся record id: {profile['duplicate_record_ids']}."
            )
            if key == "tpehg":
                sentence += (
                    f" Preterm: {profile.get('preterm_positive', 0)} из "
                    f"{profile.get('known_preterm_labels', 0)} записей."
                )
            if key == "ctu_uhb_ctg":
                sentence += f" Известный cord pH: {profile.get('known_cord_ph', 0)} записей."
            lines.append(sentence)
        else:
            lines.append(f"- `{key}`: индекс ещё не создан.")

    lines.extend(
        [
            "",
            "## Решение по пригодности",
            "",
            "- CDC/NCHS: пригоден для больших статических baseline и temporal test, не для клинической динамической модели.",
            "- CTU-UHB: пригоден как малый outcome-linked КТГ benchmark.",
            "- UCI CTG: пригоден только как учебный feature benchmark после group-aware дедупликации.",
            "- TPEHG: пригоден для R&D; малое число preterm events требует широких интервалов и внешней проверки.",
            "- Для ИЦН и true FGR подходящего открытого набора в проект не добавлено.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    archives, checksum_lines = profile_archives()
    report = {
        "profiled_on": date.today().isoformat(),
        "archives": archives,
        "prepared": {
            "cdc_natality": profile_cdc(),
            "uci_cardiotocography": profile_uci(),
            "ctu_uhb_ctg": profile_signal_index("ctu_uhb_ctg"),
            "tpehg": profile_signal_index("tpehg"),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    CHECKSUM_PATH.write_text("\n".join(sorted(checksum_lines)) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"wrote {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CHECKSUM_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
