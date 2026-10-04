"""Reproducible downloader for the public datasets registered in data/registry.

The downloaded archives are intentionally ignored by Git. Metadata, URLs and
checksums remain versioned so a model run can be tied to exact source files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "open_datasets.json"
CHECKSUM_PATH = PROJECT_ROOT / "data" / "registry" / "checksums.sha256"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def list_s3_objects(prefix: str) -> list[tuple[str, int]]:
    endpoint = "https://physionet-open.s3.amazonaws.com/"
    continuation: str | None = None
    objects: list[tuple[str, int]] = []
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        request_url = endpoint + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(request_url) as response:
            root = ET.fromstring(response.read())
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:Contents", namespace):
            key = item.findtext("s3:Key", namespaces=namespace)
            size = int(item.findtext("s3:Size", default="0", namespaces=namespace))
            if key and not key.endswith("/"):
                objects.append((key, size))
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace) == "true"
        if not truncated:
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not continuation:
            raise RuntimeError(f"S3 listing for {prefix} is truncated without a continuation token")
    return objects


def download_s3_directory(prefix: str, destination: Path, force: bool = False) -> None:
    objects = list_s3_objects(prefix)
    if not objects:
        raise RuntimeError(f"No objects found for public S3 prefix: {prefix}")
    destination.mkdir(parents=True, exist_ok=True)
    endpoint = "https://physionet-open.s3.amazonaws.com/"

    def fetch(item: tuple[str, int]) -> None:
        key, expected_size = item
        relative = key.removeprefix(prefix)
        output = destination / relative
        if output.exists() and output.stat().st_size == expected_size and not force:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".part")
        url = endpoint + urllib.parse.quote(key, safe="/")
        request = urllib.request.Request(url, headers={"User-Agent": "FemFedora-data/1.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Size mismatch for {key}: {partial.stat().st_size} != {expected_size}")
        partial.replace(output)

    print(f"download S3 directory: s3://physionet-open/{prefix} ({len(objects)} objects)")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(fetch, objects))


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force and valid_zip(destination):
        print(f"skip valid archive: {destination.relative_to(PROJECT_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FemFedora-data/1.0"})
    print(f"download: {url}")
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)

    if not valid_zip(destination):
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", help="Dataset id; repeatable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checksums-only", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    selected = set(args.dataset or [])
    entries = [d for d in registry["datasets"] if not selected or d["id"] in selected]
    missing = selected - {d["id"] for d in entries}
    if missing:
        raise SystemExit(f"Unknown dataset ids: {sorted(missing)}")

    checksum_lines: list[str] = []
    for entry in entries:
        destination = PROJECT_ROOT / entry["local_path"]
        if entry.get("storage", "zip") == "s3_directory":
            if not args.checksums_only:
                download_s3_directory(entry["mirror_s3_prefix"], destination, force=args.force)
            if not destination.is_dir() or not any(destination.rglob("*")):
                print(f"missing or invalid: {destination.relative_to(PROJECT_ROOT)}", file=sys.stderr)
                continue
            checksum_lines.append(f"{directory_sha256(destination)}  {entry['local_path']}/")
        else:
            if not args.checksums_only:
                download(entry["url"], destination, force=args.force)
            if not destination.exists() or not valid_zip(destination):
                print(f"missing or invalid: {destination.relative_to(PROJECT_ROOT)}", file=sys.stderr)
                continue
            checksum_lines.append(f"{sha256(destination)}  {entry['local_path']}")

    CHECKSUM_PATH.write_text("\n".join(sorted(checksum_lines)) + "\n", encoding="utf-8")
    print(f"wrote: {CHECKSUM_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
