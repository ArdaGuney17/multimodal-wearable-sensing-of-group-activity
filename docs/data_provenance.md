# Data provenance

## Collection

Data was collected by the thesis author (Arda Güney, University of Twente) from groups of three
participants performing a collaborative puzzle-assembly task, using:

- **OpenEarable 2.0** (ear-worn, per participant)
- **Xsens DOT** (wrist-worn, per participant)
- **OptiTrack** (ceiling-mounted optical motion capture, whole-room)
- Video and audio, used **only** for ELAN annotation ground truth — not as model input.

Full setup details: `thesis_reproduction_targets.md` §1.

## Groups

- Groups 1, 2, 3, 5, 6, 7, 8, 9, 10 are used in all results (9 groups, LOGO evaluation).
- **Group 4 was recorded but excluded** — a camera failure made its annotation unusable, so it
  never entered the modelling dataset. This is why numbering skips 3→5 throughout the raw data,
  derived features, and every results table.
- Groups **1, 7, 8, 9** included the researcher as a third participant (not a fully independent
  triad). Groups **2, 3, 5, 6, 10** are "fully naive" — no researcher participation. Both the full
  9-group results and a naive-5-only sensitivity analysis are reported throughout the thesis
  (Ch.7 §7.6, Ch.8 §8.10, Table 10.1) and reproduced here.

## Ethics / consent

University of Twente ethics approval covered this study; the author has confirmed participant
consent permits publishing the data **provided it is anonymized**. Video/audio itself is not part
of this repo's data or the Zenodo archive — only OptiTrack/Xsens/OpenEarable signal data and
ELAN-derived text annotations are used as model input/output, and participants are referred to
only as `Participant1/2/3` per group, never by name.

**Anonymization verification checklist — must be confirmed before any public upload/push:**
- [ ] No participant real names, emails, or student numbers appear in any raw CSV, filename, or
      embedded metadata (spot-checked programmatically before archiving, not just assumed)
- [ ] No device identifiers (Xsens DOT serials, OpenEarable MAC addresses, etc.) that could be
      traced back to a specific person's personally-owned hardware, if applicable
- [ ] ELAN annotation exports contain only the cleaned tier/label structure (§ above), no free-text
      annotator notes that might reference names
- [ ] Ethics approval documentation itself (not just this author's summary of it) has been checked
      against what's actually about to be published, if the approval terms are more specific than
      "anonymized data may be published"

*(This checklist gets actually run — not just listed — before any raw data leaves this machine.
See the pilot-download review step.)*

## Data hosting: Zenodo

Raw sensor data is archived on **Zenodo** (free, permanent, DOI-citable — the standard venue for
thesis research data) rather than committed to this git repository or GitHub LFS. This repo ships
`src/data/download.py`, which fetches and verifies it by DOI.

**Upload steps (for the author, once the anonymization checklist above is confirmed):**
1. Create a Zenodo account / new upload (claude cannot create accounts on your behalf).
2. Upload the prepared archive (this session will produce a zip — or per-group zips if that's
   more manageable — plus a checksums manifest, once the pilot download is validated).
3. Fill in Zenodo's metadata: title, description (can reuse the thesis abstract), your name as
   author, license for the data (note: data license can differ from the MIT code license — CC-BY
   or CC0 are typical for anonymized research data, but check what your ethics approval permits),
   and link back to this GitHub repo.
4. Publish → note the resulting DOI.
5. Give the DOI back to this session (or fill it in yourself) in `README.md` and
   `src/data/download.py` so the download script actually points at the real record instead of
   the `<TODO>` placeholder.

## What's committed vs. regenerated

Nothing under `data/` is committed to this git repository — not even via LFS. Raw data lives on
Zenodo (above) and is pulled locally by `src/data/download.py`; everything else — synced/
cleaned/windowed intermediate data, engineered feature tables, model predictions — is regenerated
by running the pipeline (`src/`). Both are excluded from version control via `.gitignore` to keep
the repository itself small and fast to clone.

## Known data issues under investigation

See `docs/table_to_source_mapping.md` for the live list — notably a folder in the original Drive
history (`thesis/output/group_3`) whose files were misnamed as `group5_...`, which needs manual
verification before being trusted as a source for group 3.
