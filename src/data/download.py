"""Fetch the raw sensor dataset into data/raw/.

The dataset is NOT committed to this repository (see docs/data_provenance.md for why).
Two sources are supported:

  - "drive" (default, current): an interim public Google Drive folder, used until the
    dataset is formally published on Zenodo alongside the paper. Anonymized per-group
    zips only — see docs/data_provenance.md for what's excluded and why.
  - "zenodo": the eventual permanent, DOI-citable archive. Not live yet — switch
    DEFAULT_SOURCE below to "zenodo" and fill in ZENODO_RECORD_ID once it's published.

Usage:
    python -m src.data.download
    python -m src.data.download --source zenodo --record-id 1234567
    python -m src.data.download --force                # re-download even if data/raw/ exists
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Switch to "zenodo" once the dataset is published there (see docstring above).
DEFAULT_SOURCE = os.environ.get("DATASET_SOURCE", "drive")

# TODO: fill in once the dataset is uploaded and published on Zenodo.
ZENODO_RECORD_ID = os.environ.get("ZENODO_RECORD_ID", "")
ZENODO_API = "https://zenodo.org/api/records/{record_id}"

# Interim source: public Drive folder (see README.md "Data access" section for the
# human-facing link). Group 4 is absent by design — recorded but excluded from the
# thesis dataset (camera failure made its annotation unusable). File IDs + SHA-256
# checksums captured at upload time (2026-09-10) from data/zenodo_upload/MANIFEST.md.
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1c0LOZ98eEX9RyOY7iio6Rl_NOwFgxTf2"
DRIVE_FILES = {
    "group_1_raw.zip": {"id": "1yXjRNRODFtlCmpIupDoDcPqOKf-uhh8A", "sha256": "ab6535e81f0e7cb685807b9cbfc8ce508ae61cdbf122a426c7fc2c2f26dd2744"},
    "group_2_raw.zip": {"id": "10ymQr8a6O9vAP0lNd2oeRxPlROHMFysW", "sha256": "d691b9c300164f32dbe24640428efec53951d064d962dc265bf6b1268a854720"},
    "group_3_raw.zip": {"id": "17cYz9iIwznJSLGkzn-s35XEBtRJ_Z4qY", "sha256": "9c2eddde29e7fc1022d7fd201fb2c77fba5b8d6993d5224b213d7f9765394ddf"},
    "group_5_raw.zip": {"id": "1Q7q3HNutzAFQzSYLO_OcXIgzZkEqDPIL", "sha256": "1c582f305a5b504876e01248dc87c0e47106eb97ac0c50d1f0ac6482360125b1"},
    "group_6_raw.zip": {"id": "120fwa0dbFJh-KgwELPr8Yo_HRcxjxEKr", "sha256": "a3007fbfac4ac698ace0659e55b75f64c22dec7b4d6e5348aa218d4e6f174dd0"},
    "group_7_raw.zip": {"id": "1p94hpfohmIvEgAe6mkwiMKpQ4sEn17GB", "sha256": "02f592db7c955904cbb2fc7d8c30f8912feb923c716a03f1588c882ee030b21a"},
    "group_8_raw.zip": {"id": "1PQEroCLUVi0c499GaLXn63M0ONlMLV4t", "sha256": "cf3a5b4dd680eb8078031b6712aef0161c368236a9b72b5063218dd5da4b5d39"},
    "group_9_raw.zip": {"id": "1VA0eTCqJXoChLFzrRFZRFPZ7txpwpFAl", "sha256": "5673e7903e4851f96d56c6e2ae574e8218f31b948754e892b13455bd46116705"},
    "group_10_raw.zip": {"id": "1EQgho-mtZZZQAZo73ypxSSfarNib22fy", "sha256": "aa91595ba4686fe39acfa671d5ad96f5c45be70b63563f0b6e47e53495025fe4"},
}


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _drive_download(file_id: str, dest: Path) -> None:
    """Download a (possibly large) public Drive file, handling the "can't scan for
    viruses" interstitial confirmation page Drive shows for files over ~25MB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    resp = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token is None:
        # Newer Drive UI embeds the confirm token in the HTML instead of a cookie.
        m = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
        if m:
            token = m.group(1)

    if token:
        resp = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)

    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def fetch_from_drive(force: bool = False) -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()) and not force:
        print(f"{RAW_DIR} already has content — use --force to re-download.")
        return

    tmp_dir = REPO_ROOT / "data" / "_drive_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for filename, info in DRIVE_FILES.items():
        dest = tmp_dir / filename
        print(f"Downloading {filename} from Drive ...")
        _drive_download(info["id"], dest)

        actual = _sha256(dest)
        if actual != info["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {info['sha256']}, got {actual}. "
                "Download may be corrupted or the Drive folder contents changed — try again "
                f"with --force, or check {DRIVE_FOLDER_URL} directly."
            )
        print(f"  checksum OK ({actual[:16]}...)")

        print(f"  extracting into {RAW_DIR} ...")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(RAW_DIR)

    print(f"Done. Raw data is in {RAW_DIR}")
    print(
        "Note: this pulled from the interim Drive folder, not the permanent Zenodo "
        "archive (not published yet — see src/data/download.py's docstring)."
    )


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


def fetch_from_zenodo(record_id: str, force: bool = False) -> None:
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
    parser.add_argument("--source", choices=["drive", "zenodo"], default=DEFAULT_SOURCE,
                         help="where to fetch the dataset from (default: %(default)s)")
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID, help="Zenodo record id (--source zenodo only)")
    parser.add_argument("--force", action="store_true", help="re-download even if data/raw/ exists")
    args = parser.parse_args()

    if args.source == "drive":
        fetch_from_drive(force=args.force)
    else:
        fetch_from_zenodo(args.record_id, force=args.force)


if __name__ == "__main__":
    main()
