"""Fetch the raw sensor dataset from its Zenodo archive into data/raw/.

The dataset is NOT committed to this repository (see docs/data_provenance.md for why).
This script downloads it from Zenodo by record ID, verifies each file's checksum against
what Zenodo reports, and extracts it into data/raw/.

Usage:
    python -m src.data.download
    python -m src.data.download --record-id 1234567   # override the default below
    python -m src.data.download --force                # re-download even if data/raw/ exists

TODO once the archive is published: replace ZENODO_RECORD_ID below with the real record id
(the number in the Zenodo URL, e.g. https://zenodo.org/records/1234567 -> 1234567) or pass
--record-id / set the ZENODO_RECORD_ID environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# TODO: fill in once the dataset is uploaded and published on Zenodo.
ZENODO_RECORD_ID = os.environ.get("ZENODO_RECORD_ID", "")
ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected_size or 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def fetch(record_id: str, force: bool = False) -> None:
    if not record_id:
        print(
            "No Zenodo record id set. Pass --record-id, set ZENODO_RECORD_ID, or fill in "
            "ZENODO_RECORD_ID in src/data/download.py once the dataset is published.",
            file=sys.stderr,
        )
        sys.exit(1)

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()) and not force:
        print(f"{RAW_DIR} already has content — use --force to re-download.")
        return

    meta = requests.get(ZENODO_API.format(record_id=record_id), timeout=30).json()
    files = meta.get("files", [])
    if not files:
        print(f"Zenodo record {record_id} has no files (or isn't published yet).", file=sys.stderr)
        sys.exit(1)

    tmp_dir = REPO_ROOT / "data" / "_zenodo_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        filename = f["key"]
        url = f["links"]["self"]
        expected_size = f.get("size")
        expected_checksum = f.get("checksum", "")  # format: "md5:<hex>"
        dest = tmp_dir / filename

        print(f"Downloading {filename} ...")
        _download_file(url, dest, expected_size)

        if expected_checksum.startswith("md5:"):
            actual = _md5(dest)
            expected = expected_checksum.split(":", 1)[1]
            if actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {filename}: expected {expected}, got {actual}. "
                    "Download may be corrupted — try again with --force."
                )
            print(f"  checksum OK ({actual})")

        if filename.endswith(".zip"):
            print(f"  extracting into {RAW_DIR} ...")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(RAW_DIR)

    print(f"Done. Raw data is in {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID, help="Zenodo record id")
    parser.add_argument("--force", action="store_true", help="re-download even if data/raw/ exists")
    args = parser.parse_args()
    fetch(args.record_id, force=args.force)


if __name__ == "__main__":
    main()
