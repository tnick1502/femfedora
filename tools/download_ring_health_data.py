"""Download public research data; never download credentialed datasets implicitly."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/external/ring_health"


def download(item):
    name, url = item
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "2", "--max-time", "90", url, "-o", str(path)], check=True)
    if path.suffix.lower() == ".xpt" and not path.read_bytes().startswith(b"HEADER RECORD"):
        path.unlink()
        raise ValueError(f"Not an XPT: {url}")
    return {"path": str(path.relative_to(ROOT)), "url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    local = Path("/Users/nikita/Downloads/20822032")
    for filename in ["data set menopause.xlsx", "menapouse study questionnaire.docx"]:
        if (local / filename).exists():
            shutil.copy2(local / filename, DEST / filename)
    items = [
        ("tempid/temperature_ovulation_dataset.csv", "https://zenodo.org/records/20160611/files/temperature_ovulation_dataset.csv?download=1"),
        ("tempid/MaxRange.csv", "https://zenodo.org/records/20160611/files/MaxRange.csv?download=1"),
        ("tempid/metadata.json", "https://zenodo.org/api/records/20160611"),
        ("provenance/assaf_article.pdf", "https://www.ammanu.edu.jo/media/w1jdiahg/f1000research-11-162087.pdf"),
        ("provenance/tempid_preprocessing.py", "https://raw.githubusercontent.com/nach00gar/Monitoring-female-fertile-ground-temperature-measurement-using-a-skin-wearable-and-ML-integration/main/src/preprocessing.py"),
        ("provenance/tempid_construction.py", "https://raw.githubusercontent.com/nach00gar/Monitoring-female-fertile-ground-temperature-measurement-using-a-skin-wearable-and-ML-integration/main/src/construction.py"),
        ("provenance/tempid_LICENSE.txt", "https://raw.githubusercontent.com/nach00gar/Monitoring-female-fertile-ground-temperature-measurement-using-a-skin-wearable-and-ML-integration/main/LICENSE"),
    ]
    for year, suffix in [(2015, "I"), (2017, "J")]:
        for table in ["DEMO", "RHQ", "BMX", "BPX", "SLQ"]:
            name = f"{table}_{suffix}"
            base = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
            items += [(f"nhanes/{name}.xpt", base + ".xpt"), (f"nhanes/{name}.htm", base + ".htm")]
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(download, items))
    for filename in ["data set menopause.xlsx", "menapouse study questionnaire.docx"]:
        path = DEST / filename
        if path.exists():
            records.append({"path": str(path.relative_to(ROOT)), "source": str(local / filename),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    target = ROOT / "data/registry/ring_health_downloads.json"
    target.write_text(json.dumps({"retrieved_date": "2026-11-09", "files": records}, indent=2))
    print(json.dumps({"downloaded_files": len(records), "manifest": str(target)}, indent=2))


if __name__ == "__main__":
    main()
