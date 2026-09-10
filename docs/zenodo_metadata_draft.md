# Zenodo upload — metadata draft

Copy/paste these into Zenodo's "New Upload" form. Review and edit before publishing —
this is a draft, not final text.

## Title
Multimodal Wearable Sensing of Group Activity: Raw Sensor Dataset (OpenEarable, Xsens DOT, OptiTrack)

## Authors
Arda Güney — University of Twente
(add ORCID if you have one; affiliation as above)

## Description

Raw multimodal sensor data collected for the MSc thesis "Multimodal Wearable Sensing of
Group Activity" (University of Twente, 2026). Nine groups of three participants each
performed a structured collaborative puzzle-assembly task while wearing:

- **OpenEarable 2.0** (ear-worn: accelerometer, gyroscope, barometer, PPG, skin/environment
  temperature, magnetometer, bone-conduction accelerometer)
- **Xsens DOT** (wrist-worn inertial: orientation, acceleration, angular velocity)
- **OptiTrack** (ceiling-mounted optical motion capture, whole-room, 3 tracked landmarks)

Video and audio were used only to produce ELAN ground-truth annotations of individual,
pairwise, and whole-group activity — they are **not** included in this archive and were
never used as model input.

**Groups**: numbered 1, 2, 3, 5, 6, 7, 8, 9, 10 (Group 4 was recorded but excluded due to
a camera failure that made its annotation unusable — this is why the numbering skips 4).
Groups 1, 7, 8, 9 included the thesis author as a third participant; groups 2, 3, 5, 6, 10
are fully independent triads.

**Anonymization**: participants are identified only as `Participant1`/`Participant2`/
`Participant3` per group throughout this archive. Ground-truth annotation exports are the
already-anonymized variant (`*_individual_build_renamed.csv` / `*_clean.csv`) — the raw
ELAN exports containing participants' real first names were excluded entirely and never
uploaded. OptiTrack take filenames were renamed from the recording software's own
`<researcher-name>_Group_N_Take_M.csv` convention to `group_N_optitrack_take_M.csv`.

**Companion code**: the full reproduction pipeline (raw sync → cleaning → feature
engineering → model training → thesis tables) is at:
https://github.com/ArdaGuney17/multimodal-wearable-sensing-of-group-activity

## Keywords
group activity recognition; wearable sensing; multimodal sensor fusion; human activity
recognition; OpenEarable; Xsens DOT; OptiTrack; collaborative task; social signal processing

## License
**Decide before publishing** — check exactly what your ethics approval permits. Typical
choices for anonymized research data: `CC-BY-4.0` (attribution required, most common) or
`CC0-1.0` (public domain, no restrictions). Note: this can differ from your code's license
(e.g. MIT) — data and code licenses are set separately.

## Related/alternate identifiers
- Is supplement to / Cites: link to the GitHub repo above (relation: "isSupplementTo" or
  "isReferencedBy", whichever Zenodo's dropdown calls it in the version you see)
- Once published, come back and add the resulting DOI to this repo's `README.md` and
  `src/data/download.py` (replace the `<TODO>` placeholder there).

## Upload type
Dataset

## Access right
Open Access (or "Restricted" if your ethics approval requires request-based access —
check this before publishing)

---

## Files to upload
Built by `scripts/build_zenodo_archive.py`, one zip per group, in
`data/zenodo_upload/` (gitignored, not committed — upload these directly to Zenodo):
`group_1_raw.zip` … `group_10_raw.zip` (skipping group_4), plus `MANIFEST.md` listing each
zip's SHA-256 checksum and exactly which files were excluded per group for anonymization —
review that file before uploading, it's your audit trail.
