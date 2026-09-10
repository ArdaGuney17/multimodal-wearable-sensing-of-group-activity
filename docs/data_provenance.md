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
- [x] **FAILED, then fixed by exclusion** — see finding below. Raw CSVs in `elan/` DO contain real
      participant first names in some file variants. Resolved by only ever publishing the
      already-anonymized variants (see below) — never the raw-name ones.
- [ ] No device identifiers (Xsens DOT serials, OpenEarable MAC addresses, etc.) that could be
      traced back to a specific person's personally-owned hardware, if applicable — not yet checked.
- [x] ELAN annotation exports: confirmed the anonymized variants contain only the cleaned
      `Participant1/2/3` / `Whole_Group` tier structure, no free-text notes.
- [ ] Ethics approval documentation itself (not just this author's summary of it) has been checked
      against what's actually about to be published, if the approval terms are more specific than
      "anonymized data may be published"

### Finding: real names in raw ELAN files (2026-09-03)

Programmatically scanned the first column (tier/speaker) of every `elan/*.csv` file across all 9
groups. Result: **8 of 9 groups' raw ELAN exports contain real participant first names** instead
of the anonymized `Participant1/2/3` labels — including the thesis author's own name ("Arda") in
the 3 groups where he participated (7, 8, 9). Full list of names found: Ali, Rick, Lennart (g10);
Arda, Khalil, Shizin (g2); Chinyu, Egemen (g3); Adarsh, Mintan, Ali (g5); Gozluk, Kas, Mark (g6,
possibly nicknames); Adam, Kadın (g7, "Kadın" is Turkish for "woman" — likely a generic label, not
a name); Arda, Ewoud, Jennifer (g8); Arda, Roy, Concetta (g9).

**Update (2026-09-04), from reading `FINAL_ARDA_THESIS.ipynb`'s own code** (the notebook that
builds every group's `_individual_build_renamed.csv` — see `docs/table_to_source_mapping.md`'s
"Raw sensor sync & cleaning" row). Its hardcoded per-group `NAME` dicts confirm the list above and
add one previously-missed name — **Group 3's third participant is `"long"`** (lowercase, code
literal at `FINAL_ARDA_THESIS.ipynb` line 5902; the file scan above had only found "Chinyu, Egemen"
for group 3, one short). Separately, this same notebook's code reads a raw-name `Group_1.csv`
directly (`ELAN_PATH` at line 926, names `{"Arda","Bas","Rachel"}`) — contradicting the "Group_1
only ever had this safe variant downloaded" note below, which was only ever true of what had been
*downloaded to this machine* at scan time, not of what exists in Drive. **A raw-name `Group_1.csv`
almost certainly exists in Drive and has not yet been located/added to the not-safe list below** —
treat Group 1 as unconfirmed-safe (not yet cleared) until that file is found there and excluded
the same way as the other 8 groups' raw files.

**The fix already exists in the data itself**: each group has an `_individual_build_renamed.csv`
(and/or `_clean.csv`) variant that uses only `Participant1/2/3`/`Whole_Group` labels — these are
the properly anonymized versions ("renamed" = names replaced with pseudonyms). Confirmed clean via
the same scan. Group_1 only ever had this safe variant downloaded **to this machine** — see the
update above regarding Drive itself.

**Not-safe files (real names, must NEVER be published)**: the plain `Group_N.csv` (all 9 groups,
including Group 1 — see update above), every `_with_individual_build.csv`, the
`_BACKUP_before_sync_patch.csv` pair in group_9, and group_3's
`_concatenated.csv`/`_Part1.csv`/`_Part2.csv`.

**Decision (2026-09-03): exclude.** The raw-name files stay on this local machine only — never
committed (already true, `data/raw/` is gitignored) and, critically, **never included in the
Zenodo archive** built later. When the Zenodo upload bundle is prepared, it must explicitly skip:
`Group_N.csv`, `Group_N_with_individual_build.csv`, `Group_9_*_BACKUP_before_sync_patch.csv`,
`Group_3_concatenated.csv`, `Group_3_Part1.csv`, `Group_3_Part2.csv` — only `*_individual_build_
renamed.csv` and `*_clean.csv` (or group_1's single safe file) ship publicly. This exclusion list
must be encoded directly in whatever script builds that archive, not just remembered.

## Data hosting: Google Drive now, Zenodo eventually

Raw sensor data is **not** committed to this git repository or GitHub LFS. As of 2026-09-10 it's
hosted on a public Google Drive folder as an interim measure — anonymized per-group zips, built by
`scripts/build_zenodo_archive.py` using the exclusion list above:

**[multimodal-group-activity-recognition-raw-data](https://drive.google.com/drive/folders/1c0LOZ98eEX9RyOY7iio6Rl_NOwFgxTf2)**
(anyone-with-the-link, viewer access; SHA-256 checksums for every file in that folder's
`MANIFEST.md`). `src/data/download.py --source drive` (the current default) fetches and verifies
it from there.

The **permanent** home is still meant to be **Zenodo** (free, permanent, DOI-citable — the
standard venue for thesis research data) — published alongside the thesis paper, not before, per
the author's own call on timing. `docs/zenodo_metadata_draft.md` has the metadata drafted and
ready. Once that upload happens:
1. Publish on Zenodo, using the same anonymized archives (already built, in `data/zenodo_upload/`
   locally) and the drafted metadata.
2. Note the resulting DOI.
3. Fill it into `README.md`'s Data section and `src/data/download.py`'s `ZENODO_RECORD_ID` (or
   pass `--record-id`), and switch `DEFAULT_SOURCE` in that file from `"drive"` to `"zenodo"`.
4. The Drive folder can stay up as a mirror, or come down — author's call.

## What's committed vs. regenerated

Nothing under `data/` is committed to this git repository — not even via LFS. Raw data lives on
Drive for now (Zenodo eventually, see above) and is pulled locally by `src/data/download.py`;
everything else — synced/cleaned/windowed intermediate data, engineered feature tables, model
predictions — is regenerated by running the pipeline (`src/`). Both are excluded from version
control via `.gitignore` to keep the repository itself small and fast to clone.

## Known data issues under investigation

See `docs/table_to_source_mapping.md` for the live list — notably a folder in the original Drive
history (`thesis/output/group_3`) whose files were misnamed as `group5_...`, which needs manual
verification before being trusted as a source for group 3.
