"""Build anonymized, per-group Zenodo upload archives from data/raw/.

Safety model: ALLOW-LIST, not deny-list. For the elan/ subfolder (the only
place real participant names were ever found, per docs/data_provenance.md's
2026-09-03 investigation), only files matching known-safe patterns are
included -- everything else in elan/ is silently skipped, so a future/unknown
unsafe variant can never slip through. OpenEarable/Xsens/OptiTrack sensor
files are included wholesale (spot-checked 2026-09-10: pure numeric/metadata,
no participant-name fields) except OptiTrack filenames, which are renamed to
strip the researcher's own name (present in every group's take filenames,
e.g. "Arda_Group_2_Take_3.csv" -> "group_2_optitrack_take_3.csv") for a
cleaner public archive -- this is a cosmetic rename of the researcher's own
already-public name, not a participant-privacy issue.

Run from repo root: python scripts/build_zenodo_archive.py
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import zipfile

REPO_ROOT = os.getcwd()
RAW_ROOT = os.path.join(REPO_ROOT, "data", "raw")
OUT_ROOT = os.path.join(REPO_ROOT, "data", "zenodo_upload")
GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]

SAFE_ELAN_PATTERNS = [
    re.compile(r"^Group_\d+_individual_build_renamed\.csv$"),
    re.compile(r"^Group_\d+_clean\.csv$"),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_group(g: int, stage_dir: str) -> list[str]:
    """Copy only safe files for group g into stage_dir. Returns skipped elan filenames."""
    src = os.path.join(RAW_ROOT, f"group_{g}")
    skipped = []

    # elan/ -- allow-list only
    elan_src = os.path.join(src, "elan")
    if os.path.isdir(elan_src):
        elan_dst = os.path.join(stage_dir, "elan")
        os.makedirs(elan_dst, exist_ok=True)
        for fn in sorted(os.listdir(elan_src)):
            if any(p.match(fn) for p in SAFE_ELAN_PATTERNS):
                shutil.copy2(os.path.join(elan_src, fn), os.path.join(elan_dst, fn))
            else:
                skipped.append(fn)

    # openearable/, xsens/ -- wholesale copy (sensor data, no name fields)
    for sub in ("openearable", "xsens"):
        sub_src = os.path.join(src, sub)
        if os.path.isdir(sub_src):
            shutil.copytree(sub_src, os.path.join(stage_dir, sub))

    # optitrack/ -- wholesale copy but rename files to strip researcher's name
    opti_src = os.path.join(src, "optitrack")
    if os.path.isdir(opti_src):
        opti_dst = os.path.join(stage_dir, "optitrack")
        os.makedirs(opti_dst, exist_ok=True)
        for fn in sorted(os.listdir(opti_src)):
            m = re.search(r"[Tt]ake[_-]?(\d+)", fn)
            take_num = m.group(1) if m else fn
            new_name = f"group_{g}_optitrack_take_{take_num}.csv"
            shutil.copy2(os.path.join(opti_src, fn), os.path.join(opti_dst, new_name))

    return skipped


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest_lines = ["# Zenodo archive build manifest\n\n"]

    for g in GROUPS:
        stage_dir = os.path.join(OUT_ROOT, f"_stage_group_{g}")
        if os.path.exists(stage_dir):
            shutil.rmtree(stage_dir)
        os.makedirs(stage_dir)

        skipped = stage_group(g, stage_dir)

        zip_path = os.path.join(OUT_ROOT, f"group_{g}_raw.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(stage_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    arcname = os.path.join(f"group_{g}", os.path.relpath(full, stage_dir))
                    zf.write(full, arcname)

        checksum = sha256_of(zip_path)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        shutil.rmtree(stage_dir)  # free disk immediately, one group at a time

        manifest_lines.append(f"## group_{g}_raw.zip\n")
        manifest_lines.append(f"- size: {size_mb:.1f} MB\n")
        manifest_lines.append(f"- sha256: {checksum}\n")
        if skipped:
            manifest_lines.append(f"- excluded elan files (real names, not published): {skipped}\n")
        manifest_lines.append("\n")

        print(f"group_{g}: {size_mb:.1f} MB, sha256={checksum[:16]}..., excluded={skipped}")

    with open(os.path.join(OUT_ROOT, "MANIFEST.md"), "w") as f:
        f.writelines(manifest_lines)

    print("\nDone. Archives + MANIFEST.md written to:", OUT_ROOT)


if __name__ == "__main__":
    main()
