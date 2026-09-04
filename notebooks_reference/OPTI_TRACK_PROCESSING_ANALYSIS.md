# Analysis: OPTI_TRACK_PROCESSING.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\OPTI_TRACK_PROCESSING.ipynb`

Code-only extract used for line references below: `OPTI_TRACK_PROCESSING_CODE_ONLY.py` (19,568 lines). Verified directly against the `.ipynb` JSON: 86 total cells (70 code, 16 markdown). The extract's own `# --- CELL N (code cell #M) ---` numbering has gaps at every dropped markdown cell (e.g. `CELL 9` → `CELL 11`, `CELL 16` → `CELL 18`, `CELL 20` → `CELL 22`, ...) — `code cell #M` (1-70) is the reliable sequential index; `CELL N` is the notebook's raw 0-indexed cell-array position minus dropped markdown. Both are given below as `CELL n (code cell #m)`.

## Cell counts

Total cells: 86
Code cells: 70
Markdown cells: 16

## Import statements (deduplicated)

```python
import os
import csv
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
```

Note: no `google.colab.drive.mount(...)` call anywhere in the file (confirmed by search — zero matches for `drive.mount`/`google.colab`) — unlike every sibling notebook analyzed in this repo, this one has no mount cell at all in the extracted code, meaning either Drive was already mounted by a session-level cell not captured by the extractor, or the notebook simply assumes a live Colab session with Drive already attached. All paths are hardcoded to `/content/drive/MyDrive/thesis/data/...` regardless. No `sklearn`/`torch`/ML imports anywhere — this notebook never trains or evaluates a model; it is pure data wrangling (raw marker reconstruction, sync, label transfer).

## Markdown headers / outline (in order)

The 16 markdown cells split the file into two large, mirror-image passes over the same 9 groups: a **raw-cleaning pass** (Group 1 → 2 → 5 → 6 → 7 → 8 → 9 → 10, cells 0–53) building one clean, combined-take OptiTrack CSV per group, then a **labeling pass** (Group 10 → 9 → 8 → 7 → 6 → 5 → 3 → 2 → 1, cells 55–85) that syncs each group's clean OptiTrack CSV against its ELAN annotations and writes a per-row labeled CSV. Group 4 never appears (camera-failure exclusion, consistent with the rest of the repo).

- **(raw pos 0–9, no MD header before it) Group 1 — raw marker cleaning**, CELL 0 (code cell #1) through CELL 9 (code cell #10):
  - CELL 0 (#1): `INSPECT MARKER TRACKS` — reads Motive multi-header CSV, per-marker frame-presence summary + trajectory plots. No cleaning yet.
  - CELL 1 (#2): `FAST FIRST PASS 3-LANDMARK RECONSTRUCTION` — Hungarian-algorithm (`scipy.optimize.linear_sum_assignment`) frame-by-frame nearest-neighbor assignment of raw unlabeled Motive markers onto 3 persistent landmark identities (`EXPECTED_LANDMARKS=3`, `MAX_ASSIGN_DIST=0.35`, `MAX_REASONABLE_JUMP=0.45`).
  - CELL 2 (#3): `CONSERVATIVE 3-LANDMARK RECONSTRUCTION` — same idea, tighter thresholds (`MAX_ASSIGN_DIST=0.18`, `MAX_REASONABLE_JUMP=0.25`, `MAX_MISSING_FRAMES_MEMORY=240`), leaves NaN instead of forcing an assignment when uncertain.
  - CELL 3 (#4): `BALANCED 3-LANDMARK TRACKER` — nearest-neighbor + velocity prediction, a middle ground between the two above.
  - CELL 4 (#5): `TRACKLET SUCCESSOR INSPECTION` — builds a marker-tracklet summary and proposes successor chains (candidates for manual stitching) via `find_successors`.
  - CELL 5 (#6): `MANUAL TRACKLET STITCHING` — a **hand-curated** `CHAINS` dict (e.g. `{"take_1": {"landmark1": ["Unlabeled 1669", "Unlabeled 1682", "Unlabeled 1726"], ...}}`) mapping specific raw Motive marker-track IDs, per take, onto landmark1/2/3 identities; `build_stitched_clean` assembles the clean per-landmark XYZ series from these hand-picked chains. **This hand-curated marker-identity mapping is unique per group/take and cannot be regenerated automatically** — it is the main human-in-the-loop step of the raw-cleaning pass.
  - CELL 6 (#7): `FIND CANDIDATE FRAGMENTS FOR MISSING GAPS` — `get_missing_gaps`/`find_candidates_for_gap` propose additional marker fragments to fill remaining NaN gaps in the stitched result.
  - CELL 7 (#8): `UPDATED MANUAL TRACKLET STITCHING` — an updated/expanded `CHAINS` dict incorporating some of CELL 6's gap-fill candidates.
  - CELL 9 (#10): `COMBINE TAKE 1 + TAKE 2` — concatenates the per-take stitched/smoothed CSVs, applies a real capture-clock offset between takes (e.g. `TAKE2_OFFSET_S = 1385.836`, derived from the takes' real capture-start timestamps), recomputes `landmark{i}_available`/`active_clean_landmarks`, and writes `group_1_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 10, MD "Group 2") Group 2 raw cleaning**, CELL 11 (#11) `START INSPECTION` → CELL 12 (#12) `MANUAL TRACKLET STITCHING` → CELL 13 (#13) `COMBINE TAKE 1–5` → `group_2_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 17, MD "GROUP 5") Group 3 raw cleaning** (yes — cell content is titled "GROUP 3" despite the group-5 markdown header, likely an authoring-order artifact): CELL 14 (#14) `START INSPECTION` → CELL 15 (#15) `MANUAL TRACKLET STITCHING` → CELL 16 (#16) `COMBINE TAKE 1–5` → `group_3_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 21, MD "GROUP 6") Group 5 raw cleaning**: CELL 18 (#17) `START INSPECTION` → CELL 19 (#18) `MANUAL TRACKLET STITCHING` → CELL 20 (#19) `COMBINE TAKE 1–5` → `group_5_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 27, MD "GROUP 7") Group 6 raw cleaning**: CELL 22 (#20) `FILE INTEGRITY / TIMESTAMP AUDIT` (row-count/frame-monotonicity/duplicate/gap diagnostic, no output CSV) → CELL 23 (#21) `CHECK IF TAKE 3 IS DUPLICATE PREFIX OF TAKE 2` (diagnostic) → CELL 24 (#22) `START INSPECTION` → CELL 25 (#23) `MANUAL TRACKLET STITCHING` → CELL 26 (#24) `COMBINE TAKE 1–2` → `group_6_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 32, MD "GROUP 8") Group 7 raw cleaning**: CELL 28 (#25) `START INSPECTION` → CELL 29 (#26) `MANUAL TRACKLET STITCHING` → CELL 30 (#27) `GAP CANDIDATE INSPECTION FOR CURRENT STITCHING` → CELL 31 (#28) `COMBINE TAKE 1–4` → `group_7_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 37, MD "GROUP 9") Group 8 raw cleaning**: CELL 33 (#29) `START INSPECTION` → CELL 34 (#30) `MANUAL TRACKLET STITCHING` → CELL 35 (#31) `GAP CANDIDATE INSPECTION FOR CURRENT STITCHING` → CELL 36 (#32) `COMBINE TAKE 1–5` → `group_8_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 44, MD "GROUP 10") Group 9 raw cleaning**: CELL 38 (#33) `START INSPECTION` → CELL 39 (#34) `MANUAL TRACKLET STITCHING` → CELL 40 (#35) `FOCUSED MARKER SUMMARY FOR TAKE 5 AND TAKE 6` → CELL 41 (#36) `FOCUSED GAP CANDIDATE INSPECTION` → CELL 42 (#37) `MANUAL TRACKLET STITCHING UPDATED` → CELL 43 (#38) `COMBINE TAKE 1–6` → `group_9_optitrack_cleaned_combined_240hz.csv`.
- **(no further MD before it) Group 10 raw cleaning**: CELL 45 (#39) `START INSPECTION` → CELL 46 (#40) `MANUAL TRACKLET STITCHING` → CELL 47 (#41) `FOCUSED GAP CANDIDATE INSPECTION` → CELL 48 (#42) `MANUAL TRACKLET STITCHING UPDATED` → CELL 49 (#43) `COMBINE TAKE 1–3` → CELL 50 (#44) `COMBINE TAKE 1–3` (re-run/duplicate of the same combine logic) → `group_10_optitrack_cleaned_combined_240hz.csv`.
- **(raw pos 51, MD "SUMMARY OF THE OPTITRACK DATA")**:
  - CELL 52 (#45): `OPTITRACK FINAL SUMMARY TABLE - ALL GROUPS` — loops `GROUPS = [1,2,3,5,6,7,8,9,10]`, reads each group's `group_{g}_optitrack_cleaned_combined_240hz.csv`, computes per-group and per-take availability/gap/quality statistics → `optitrack_final_summary/optitrack_all_groups_summary.csv` + `optitrack_all_groups_take_level_summary.csv`. This is a **QC/reporting table, not a windowed feature or label grid**.
  - CELL 53 (#46): `OPTITRACK SUPERVISOR VISUALIZATION` — bar plots of per-group landmark availability and active-3-landmark percentage from the CELL 52 summary CSV. No new CSV output.
- **(raw pos 54, MD "LABELING") — the sync + label-transfer pass, all 9 groups, reverse order 10→9→8→7→6→5→3→2→1:**
  - **Group 10**, no separate group-level MD before it: CELL 55 (#47) `OPTITRACK LABELING FROM ELAN - ONE GROUP` — the full-featured template cell (see dedicated section below) → `group_10_optitrack_labeled.csv` + `group_10_optitrack_labeling_summary.csv`. CELL 56 (#48) `VALIDATE FIRST AND LAST SYNC ALIGNMENT` — sanity-checks the single-point shift against both ends of the recording.
  - **Group 9**: CELL 57 (#49) `WHOLE OPTITRACK SYNC MOVE VISUALIZATION` (movement-score peak-candidate detection over the whole recording) → CELL 58 (#50) `CHECK ELAN SYNCHRONIZATION ROWS` (diagnostic) → CELL 59 (#51) `FOCUSED SYNC WINDOW INSPECTION` (zooms on the specific known ELAN sync intervals) → CELL 60 (#52) `FIND CORRECT OPTITRACK AND ELAN FILES` (glob-based file discovery, restricted to `*renamed*.csv`) → CELL 61 (#53) `FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC` → `group_9_optitrack_labeled.csv`.
  - **(raw pos 62, MD "GROUP 8") Group 8**: CELL 63 (#54) `FOCUSED SYNC WINDOW INSPECTION` → CELL 64 (#55) `CORRECTED SYNC INSPECTION AROUND ELAN SYNC TIMES` → CELL 65 (#56) `FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC` → `group_8_optitrack_labeled.csv`.
  - **(raw pos 66, MD "GROUP 7") Group 7**: CELL 67 (#57) `SYNC WINDOW INSPECTION` → CELL 68 (#58) `EXTRA SYNC CHECK` → CELL 69 (#59) `FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC` → `group_7_optitrack_labeled.csv`.
  - **(raw pos 70, MD "Group 6") Group 6**: CELL 71 (#60) `SYNC WINDOW INSPECTION` → CELL 72 (#61) `EXTRA SYNC CANDIDATE COMPARISON` → CELL 73 (#62) `FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC` → `group_6_optitrack_labeled.csv`.
  - **(raw pos 74, MD "GROUP 5") Group 5**: CELL 75 (#63) `SYNC WINDOW INSPECTION` → CELL 76 (#64) `FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC` (final ELAN sync falls after OptiTrack recording ends, so only the first sync point is usable) → `group_5_optitrack_labeled.csv`.
  - **(raw pos 77, MD "NO GROUP 4 SO GROUP 3") Group 3**: CELL 78 (#65) `SYNC WINDOW INSPECTION` → CELL 79 (#66) `FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC` (only one ELAN `synchronization_move` exists at all, so drift validation is impossible) → `group_3_optitrack_labeled.csv`.
  - **(raw pos 80, MD "GROUP 2") Group 2**: CELL 81 (#67) `SYNC INSPECTION` (not all 3 members performed the sync move; inspects all sync-like labels, not just `Whole_Group`) → CELL 82 (#68) `FAST OPTITRACK LABELING FROM ELAN WITH SINGLE PARTICIPANT SYNC` (anchors on `Participant1`'s sync label specifically — `khalil_synchornizaion_move` → `participant1_synchronization_move`) → `group_2_optitrack_labeled.csv`.
  - **(raw pos 83, MD "GROUP 1") Group 1**: CELL 84 (#69) `SYNC WINDOW INSPECTION` → CELL 85 (#70) `FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC` → `group_1_optitrack_labeled.csv`. End of file.

## Sync recovery and Task-1 grid provenance

### 1. Per-group sync mechanism, in full detail

**The algorithm, quoted verbatim (Group 10's full "ONE GROUP" template, CELL 55 / code cell #47, lines 13291–13520):**

```python
def find_elan_sync_interval(elan):
    sync_mask = elan["label"].str.lower().str.contains(
        "sync|synchron", regex=True, na=False
    )

    sync_rows = elan[sync_mask].copy()

    if len(sync_rows) == 0:
        raise ValueError(
            "No synchronization label found in ELAN file. "
            "Check label names manually."
        )

    # Prefer Whole_Group if available, otherwise use earliest sync row
    whole = sync_rows[sync_rows["tier_clean"].str.lower() == "whole_group"]

    if len(whole) > 0:
        chosen = whole.sort_values("start_s").iloc[0]
    else:
        chosen = sync_rows.sort_values("start_s").iloc[0]

    return chosen, sync_rows
```

`load_elan_annotations` (lines 13328–13350) reads the same headerless ELAN export format used throughout this repo (col 0 = tier, col 3 = start_s, col 5 = end_s, col 7 = duration_s, col 8 = label), applies a small per-group `normalize_label` typo dictionary (`synchronizaiton_move`→`synchronization_move`, etc. — the exact dictionary is re-declared, and grows, in every group's cell — see "Function/class definitions" below), and sanitizes tier names.

**How the OptiTrack-side anchor is found — this is the part that answers "how is the ELAN sync interval matched against a movement signature in the OptiTrack stream":**

```python
def compute_optitrack_movement(df):
    out = df.copy()
    for i in [1, 2, 3]:
        dx = out[f"landmark{i}_x"].diff()
        dy = out[f"landmark{i}_y"].diff()
        dz = out[f"landmark{i}_z"].diff()
        out[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)
        out[f"landmark{i}_movement_smooth"] = (
            out[f"landmark{i}_movement"].rolling(120, center=True, min_periods=1).mean()  # ~0.5s @ 240Hz
        )
    movement_cols = ["landmark1_movement_smooth", "landmark2_movement_smooth", "landmark3_movement_smooth"]
    available_movement_count = out[movement_cols].notna().sum(axis=1)
    out["sync_movement_score_tolerant"] = out[movement_cols].mean(axis=1, skipna=True)
    out.loc[available_movement_count < 2, "sync_movement_score_tolerant"] = np.nan
    out["sync_movement_score_strict"] = out[movement_cols].min(axis=1, skipna=False)
    out["sync_movement_score_tolerant_smooth"] = out["sync_movement_score_tolerant"].rolling(240, center=True, min_periods=1).mean()  # ~1s
    out["sync_movement_score_strict_smooth"] = out["sync_movement_score_strict"].rolling(240, center=True, min_periods=1).mean()
    return out

def auto_detect_optitrack_sync(df_movement):
    cand = df_movement[df_movement["sync_movement_score_tolerant_smooth"].notna()].copy()
    if "take" in cand.columns:
        take_starts = cand.groupby("take")["time_s"].min().values
        for t in take_starts:
            cand = cand[~((cand["time_s"] >= t - 2.0) & (cand["time_s"] <= t + 2.0))]  # exclude take-boundary artifacts
    cand["time_bin_2s"] = (cand["time_s"] // 2).astype(int)
    peaks = (
        cand.sort_values("sync_movement_score_tolerant_smooth", ascending=False)
            .groupby("time_bin_2s", as_index=False).head(1)          # best candidate per 2s bin
            .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
            .head(20).copy()                                          # top 20 candidates overall
    )
    return peaks
```

So: **the "similarity metric" is a smoothed 3D per-landmark displacement magnitude, averaged (tolerant) or minimum (strict) across the 3 landmarks, binned into 2-second windows, and ranked by that smoothed movement score to produce the top 20 candidate peak times** (this stands in for a physical "sit-up"/"clap" sync gesture, on the theory that its movement magnitude will visibly dominate ordinary puzzle-assembly motion). There is **no explicit search range/step over shift values and no cross-correlation between the ELAN and OptiTrack signals** — the algorithm does not slide a candidate shift and score alignment quality; instead it independently (a) picks the ELAN sync interval via label-string matching and (b) picks the OptiTrack sync instant via movement-score peak-picking, then simply subtracts the two:

```python
SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID          # exact quote, line 13671
...
optitrack_labeled_base["video_time_s"] = optitrack_labeled_base["time_s"] - SHIFT
```

**Critically, the peak-candidate detector is not run unattended in the saved copy of this notebook.** Group 10's own cell sets `OPTITRACK_SYNC_TIME_MANUAL = 35.95` (with a comment: *"If you visually identified OptiTrack sync time, write it here... For Group 10, from the whole-recording plot, this is very likely 35.95s"*) and only falls back to `auto_detect_optitrack_sync`'s top candidate if this manual override is set to `None` — which it is not, in this saved run. The other 8 groups follow the same two-stage pattern: an earlier, separate cell ("SYNC WINDOW INSPECTION" / "WHOLE OPTITRACK SYNC MOVE VISUALIZATION" / "FOCUSED SYNC WINDOW INSPECTION") runs the same movement-score/candidate-peak logic and plots zoomed windows around the top candidates for **visual human confirmation**, and the human-read value is then hand-transcribed as a literal constant (`OPTITRACK_SYNC_TIME`, or `OPTITRACK_FIRST_SYNC_TIME`/`OPTITRACK_LAST_SYNC_TIME`) into the final "FAST" labeling cell, with a comment like `# From Group 9 sync inspection`. **The sync mechanism is therefore semi-automated: the candidate-generation step (movement-score peak-picking) is a real, reusable algorithm, but the final scalar shift used for every one of the 9 groups in this saved notebook was a human-confirmed constant, not something computed end-to-end without manual intervention.**

For groups with **two** usable ELAN sync intervals (a beginning one and an end one), a linear two-point alignment is used instead of a single constant shift, to correct for OptiTrack/ELAN clock drift over the session:

```python
def compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last):
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b
...
optitrack["video_time_s"] = a * optitrack["time_s"] + b
```

**Confirmed coverage across all 9 study groups** (GROUP is reassigned per cell, not looped — this is a copy-pasted-and-hand-adapted template repeated 9 times across CELLs 55–85, not a single parameterized loop over `GROUP`):

| Group | Sync cell(s) | Method | Hardcoded constant(s) used | Anchor tier |
|---|---|---|---|---|
| 10 | CELL 55 (#47) | single-point, manual-confirmed | `OPTITRACK_SYNC_TIME_MANUAL = 35.95` | `Whole_Group` (preferred) |
| 9 | CELL 57–61 (#49–53) | two-point linear | `OPTITRACK_FIRST_SYNC_TIME=23.141667`, `OPTITRACK_LAST_SYNC_TIME=3606.117833` | earliest/latest sync row |
| 8 | CELL 63–65 (#54–56) | two-point linear | `OPTITRACK_FIRST_SYNC_TIME=111.125`, `OPTITRACK_LAST_SYNC_TIME=2312.608333` | earliest/latest sync row |
| 7 | CELL 67–69 (#57–59) | two-point linear | `OPTITRACK_FIRST_SYNC_TIME=71.119`, `OPTITRACK_LAST_SYNC_TIME=2977.631667` | earliest/latest sync row |
| 6 | CELL 71–73 (#60–62) | two-point linear | `OPTITRACK_FIRST_SYNC_TIME=32.108333`, `OPTITRACK_LAST_SYNC_TIME=1367.339833` | earliest/latest sync row |
| 5 | CELL 75–76 (#63–64) | single-point | `OPTITRACK_FIRST_SYNC_TIME=113.5875` | first sync row only (final ELAN sync occurs after OptiTrack recording ends) |
| 3 | CELL 78–79 (#65–66) | single-point | `OPTITRACK_SYNC_TIME=301.704167` | only one ELAN `synchronization_move` exists at all |
| 2 | CELL 81–82 (#67–68) | single-point, participant-specific | `OPTITRACK_SYNC_TIME=199.883333` | `Participant1` tier specifically (not all 3 members performed the sync move) |
| 1 | CELL 84–85 (#69–70) | single-point | `OPTITRACK_SYNC_TIME=1829.856833` | only one ELAN `synchronization_move` available |

**All 9 study groups (1, 2, 3, 5, 6, 7, 8, 9, 10) are covered with a computed shift; none are missing.** This directly resolves the "raw sensor sync" gap in `docs/table_to_source_mapping.md` — at least for OptiTrack — with real, per-group code, not a single group-1-only artifact like `sensor_sync_fixed.ipynb`.

**Caveat on output filenames vs. the known gap description:** the gap note in `table_to_source_mapping.md` describes the missing sync source as producing files named like `_SHIFTED_EARLY_PEAK.csv` / `_shifted_by_clap_sync_peak.csv`. This notebook's outputs are consistently named `group_{g}_optitrack_labeled.csv` (see Outputs below) — a different naming convention. This notebook is very likely the right algorithmic family (same "movement-peak vs. ELAN sync interval" idea, generalized per group with hand-confirmed constants), but the literal already-synced `*_labeled*.csv` files referenced in the gap note may be a slightly different, later-renamed run of a near-identical script rather than a byte-for-byte match to this file's own output paths — worth a filename/row-count diff against the actual Drive files before declaring the gap fully closed.

### 2. Does the sync mechanism cover sensors beyond OptiTrack?

**No.** Confirmed by exact-string search across the whole 19,568-line extract: zero occurrences of `xsens`, `openearable`, `open_earable`, `Xsens`, or `OpenEarable` anywhere in this file. Every function, path, and column name in this notebook (`landmark{i}_x/y/z`, `optitrack/`, `OPTITRACK_PATH`) is OptiTrack-specific, consistent with the notebook's title. The `find_elan_sync_interval`/movement-score-peak/`SHIFT = Opti - ELAN` pattern is a **generalizable idea** (and structurally very similar to `sensor_sync_fixed.ipynb`'s Xsens/OpenEarable peak-detection approach — see comparison below) but this specific notebook never applies it to Xsens or OpenEarable data. The gap for those two sensor families' sync code remains open.

### 3. Task-1 5s window/label grid — not built here

**No.** This notebook does not build any fixed-width window grid at all. Confirmed by exact-string search: zero occurrences of `WINDOW_S`, `window_start`, `window_end`, `non_interaction`, or `binary_label` anywhere in the file, and no `resample`/`STRIDE` constant of the kind used by every windowing cell in the other analyzed notebooks. The label-transfer step (`assign_elan_labels_to_optitrack` / `fast_assign_labels`) writes one `label_{tier}` **text** column per ELAN tier (`Participant1`, `Participant2`, `Participant3`, the 3 pairwise dyads, `Whole_Group`) at the **native OptiTrack row rate** (240 Hz, unaggregated) — every raw OptiTrack row in `group_{g}_optitrack_labeled.csv` gets the raw ELAN label string(s) active at that instant (concatenated with `" + "` if multiple overlapping labels apply), not a binary interaction/non-interaction flag and not a 5-second aggregate. This notebook's own scope stops exactly at **per-group time alignment + raw-rate label transfer for OptiTrack**; it does not touch cross-group combination, windowing, feature aggregation, or binary-labeling at all. It therefore **cannot be** the lost from-scratch source of `binary_5s_all_sensor_advanced_features.csv` / `TASK1_LABEL_GRID_PATH` — that circularity (described in `master_feature_generator_task1_task2_task3_CORRECTED_V4_ANALYSIS.md`) remains unresolved by this file. A downstream, not-yet-located notebook would need to take this notebook's `group_{g}_optitrack_labeled.csv` (or an equivalent labeled/synced file per sensor) and actually bin it into 5-second windows with a binary vote — that step is absent here.

### 4. Anonymization check

**Clean — no real participant names read or written anywhere in this notebook.** Every `ELAN_PATH` in every one of the 9 groups' sync cells points at the already-anonymized `_individual_build_renamed.csv` variant (e.g. `/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv`), consistent with `docs/data_provenance.md`'s "safe" file list. Several groups additionally include an explicit **glob-based safeguard** against accidentally loading a raw-name file, e.g. Group 9's CELL 60 (#52):

```python
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)
```

— the pattern hard-requires `*renamed*` in the filename, so it structurally cannot select `Group_N.csv` or `Group_N_with_individual_build.csv`. The raw OptiTrack Motive exports (`TAKE_PATHS`, e.g. `Arda_Thesis_Group-1_Take_1.csv`) contain only 3D marker positions, no participant-identifying text. **No file path or content flagged as containing real names anywhere in this notebook.**

### 5. Relationship to already-analyzed sync notebooks

**Genuinely different from `sensor_sync_fixed.ipynb`/`sensor_sync_ROOT.ipynb`, not the same algorithm merely generalized.** Both notebooks share the same *conceptual* anchor idea (a distinctive physical "sync gesture" annotated in ELAN as `synchronization_move`, matched against a movement-magnitude peak in the sensor stream, giving `shift = elan_time - device_time`), but the implementations diverge substantially:

| | `sensor_sync_fixed.ipynb` | `OPTI_TRACK_PROCESSING.ipynb` |
|---|---|---|
| Sensors covered | Xsens DOT + OpenEarable (no OptiTrack) | OptiTrack only |
| Groups covered | Group 1 only, hardcoded `GROUP_DIR`, no group loop | All 9 groups (1,2,3,5,6,7,8,9,10), one hand-copied cell block per group |
| Peak-finding | `scipy.signal.find_peaks(height=14, distance=10)` on the **last 300s** of each device's own accelerometer-magnitude signal | smoothed (rolling-mean) 3-landmark movement-score, peak-picked across candidates from the **whole recording**, binned into 2s windows |
| Correction model | single constant offset per participant per device (`elan_sync_t − detected_peak_time`) | single constant shift (9/10 group-cells) **or** a 2-point linear drift correction (groups 6,7,8,9) |
| Manual-override mechanism | `MANUAL_OVERRIDES` dict, present but empty/unused in the saved run | separate "sync inspection" cells plot candidates for visual confirmation; the confirmed value is hand-transcribed into a literal constant — used for every group in this saved run |

This notebook is best read as an **independent, more complete implementation covering the sensor `sensor_sync_fixed.ipynb` never reached (OptiTrack) and the group coverage it never reached (all 9 instead of 1)** — not a cleanup pass over the same code. Neither notebook's Xsens/OpenEarable-vs-OptiTrack sync code appears to be shared or derived from the other (no shared function names, no shared constants, different peak-detection statistic entirely).

**Relationship to `FINAL_ARDA_THESIS_CODE_ONLY.py`:** confirmed by direct path search — `FINAL_ARDA_THESIS_CODE_ONLY.py` is the notebook that **writes** every `Group_N_individual_build_renamed.csv` used as this notebook's `ELAN_PATH` input, for all 9 groups (e.g. `DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"` at its line 17414, matching this notebook's `ELAN_PATH` at line 14366 exactly). This is a clean two-stage pipeline relationship: **`FINAL_ARDA_THESIS_CODE_ONLY.py` builds/anonymizes the per-group ELAN annotation CSV → `OPTI_TRACK_PROCESSING.ipynb` consumes that exact file as its `ELAN_PATH` to compute the sync shift and label the raw OptiTrack stream.** The filenames line up exactly for all 9 groups, strongly suggesting these are sequential stages of one real (if never consolidated into a single script) pipeline, not two independent, unrelated attempts.

## Hardcoded file paths found in code

### Inputs

- Raw Motive/OptiTrack exports (raw-cleaning pass, all 9 groups): `/content/drive/MyDrive/thesis/data/group_{g}/optitrack/{TakeName}.csv` (e.g. `Arda_Thesis_Group-1_Take_1.csv`) — multi-header-row Motive CSV exports, one file per take (2–6 takes per group).
- Per-take intermediate cleaned/stitched files, e.g. `/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_manual_stitched_UPDATED/group_1_optitrack_take_{n}_manual_stitched_UPDATED_smoothed.csv` — read back by each group's final "COMBINE TAKE" cell.
- Combined per-group cleaned file (produced by the raw-cleaning pass, consumed by the labeling pass and by CELL 52's summary): `/content/drive/MyDrive/thesis/data/group_{g}/optitrack/optitrack_final/group_{g}_optitrack_cleaned_combined_240hz.csv`.
- ELAN annotations (labeling pass, all 9 groups): `/content/drive/MyDrive/thesis/data/group_{g}/elan/Group_{g}_individual_build_renamed.csv` — always the anonymized `_renamed` variant (see anonymization section above); some groups' cells re-discover this path via a `glob.glob(...*renamed*.csv...)` search instead of a literal string.

### Outputs (grouped by pass)

**Raw-cleaning pass, per group:**
- `group_{g}/optitrack_cleaned*/…` — multiple intermediate per-take CSVs across the fast/conservative/balanced/manual-stitched/gap-filled variants (naming varies per group/cell, e.g. `optitrack_cleaned`, `optitrack_cleaned_conservative`, `optitrack_cleaned_manual_stitched`, `optitrack_cleaned_manual_stitched_UPDATED`).
- `group_{g}/optitrack/optitrack_final/group_{g}_optitrack_cleaned_combined_240hz.csv` — the final per-group combined-take output that both CELL 52 (summary) and the entire labeling pass consume as `OPTITRACK_PATH`.
- `/content/drive/MyDrive/thesis/data/optitrack_final_summary/optitrack_all_groups_summary.csv` + `optitrack_all_groups_take_level_summary.csv` (CELL 52 (#45), all 9 groups in one run) — QC/availability report, not a feature or label table.

**Labeling pass, per group:**
- `group_{g}/optitrack/optitrack_labeled/group_{g}_optitrack_labeled.csv` — the full raw-rate (240 Hz) OptiTrack CSV with `video_time_s`, alignment-metadata columns (`optitrack_to_elan_shift_s`/`alignment_a`/`alignment_b`/etc. depending on single- vs two-point method), and one `label_{tier}` text column per ELAN tier. Written unconditionally by every one of the 9 groups' final "FAST ... LABELING" cell.
- `group_{g}/optitrack/optitrack_labeled/group_{g}_optitrack_labeling_summary.csv` — per-tier labeled-row-count/percentage/unique-labels summary, written alongside the labeled CSV for every group.

No cross-group combined output exists anywhere in the labeling pass — each group's labeled CSV and summary stay separate; nothing merges them into one table (consistent with the "no windowing, no cross-group grid" finding in §3 above).

## Function / class definitions

No classes are defined anywhere in the file (pure function/script style, same convention as the rest of this repo's notebooks). Two large families:

**Raw-cleaning pass (heavily duplicated per group, near-identical each time — same copy-paste pattern documented in this repo's other analyses):**
- Marker I/O: `read_motive_unlabeled_csv`, `read_motive_points_by_frame`, `read_motive_long`, `read_motive_long_and_frame_time`, `read_motive_long_and_active` — all parse the same 7-row Motive multi-header CSV format (meta row, marker-name row, marker-id row, axis row) but return different shapes (wide per-frame vs. long per-marker) depending on what the cell needs.
- Landmark reconstruction: `initialize_landmarks`/`initialize_landmarks_conservative`, `choose_best_three_when_too_many`, `reconstruct_three_landmarks_fast`, `reconstruct_conservative`, `balanced_reconstruct` — frame-by-frame nearest-neighbor/Hungarian-algorithm assignment of raw unlabeled markers onto 3 persistent landmark identities, three variants (fast/conservative/balanced) with different distance-threshold tuning.
- Tracklet analysis: `make_tracklet_summary`, `find_successors`, `plot_tracklet_timeline`, `plot_tracklets_xz`/`plot_top_tracklets_xz` — summarize which raw "Unlabeled NNNN" marker tracks exist and propose successor chains for manual stitching.
- Manual stitching: `build_stitched_clean` — assembles clean landmark XYZ series from a hand-curated `CHAINS` dict of marker-track IDs (see outline above); `get_missing_gaps`, `nearest_valid_position`, `find_candidates_for_gap` — propose fragments to fill remaining gaps.
- Cleanup/smoothing: `add_diagnostics`, `smooth_landmarks`, `smooth_short_gaps` (rolling-window gap interpolation, `limit`/`window` params).
- Diagnostics: `inspect_motive_file`, `get_expected_ncols`, `load_raw`, `safe_float`, `column_category`-style helpers in the Group 6 file-integrity audit cell.
- Plotting-only: `plot_active_counts`, `plot_marker_trajectories`, `plot_clean_result`, `plot_landmarks_over_time`, `plot_xyz_over_time`, `plot_xz`, `plot_active`, `plot_stitched`.

**Labeling pass (also re-declared per group with small, group-specific additions to the typo dictionary; the exact same 3 core functions repeat 9 times):**
- `sanitize_name(x)` — strips/regex-replaces non-alphanumerics in a tier name (used to build `label_{tier}` column names).
- `normalize_label(x)` — a per-group typo-correction dict applied to raw ELAN label text (`synchronizaiton_move`→`synchronization_move`, `task_operaitonal_convo`/`task_operaitonal_talk`→`task_operational_convo`, `object_handıver`→`object_handover`, `co_builidng_subpiece`→`co_building_subpiece`, `inspecitng_piece`/`inspectingpiece`→`inspecting_piece`, `pickign_up_pice`→`picking_up_piece`, `picking_up_traget_image`→`picking_up_target_image`, plus group-specific extras — e.g. Group 6 adds `synchronaziton_move`; Group 3 adds `synchronaztion_move`; Group 2 adds `synch_motion`/`sync_motion`/`synch_move`/`sync_move`/`khalil_synchornizaion_move`→`participant1_synchronization_move`/`droping_erarble_syncornaziton_move`→`dropping_earable_synchronization_move`; Group 9 adds `traveling_between_unitsinspecting_other_units`→`traveling_between_units + inspecting_other_units`).
- `load_elan_annotations(elan_path)` — the fixed-column-index ELAN CSV parser (col 0/3/5/7/8) shared by every group.
- `find_elan_sync_interval(elan)` (Group 10 only) / `find_sync_rows(elan)` (two-point groups) / `find_first_sync_row(elan)` / `find_sync_row(elan)` (single-point groups) — locate the sync-labeled ELAN row(s); logic is functionally identical (label contains `sync|synchron`, prefer `Whole_Group` or take earliest/latest) with only the function name and return shape varying per group's cell.
- `compute_optitrack_movement(df)` / the equivalent inline block in the "WHOLE OPTITRACK SYNC MOVE VISUALIZATION" cells — the smoothed movement-score computation (see algorithm section above).
- `auto_detect_optitrack_sync(df_movement)` — top-20 candidate-peak picker (Group 10's cell and the pre-labeling "sync window inspection" cells for the other groups).
- `plot_sync_confirmation(...)` / `plot_sync_window(start_s, end_s, title)` — the human-confirmation plots.
- `assign_elan_labels_to_optitrack(optitrack, elan)` (Group 10's slower, row-iteration version) / `fast_assign_labels(optitrack, elan)` (all other groups' `np.searchsorted`-based faster version) — transfer ELAN interval labels onto every OptiTrack row whose `video_time_s` falls inside the interval, concatenating with `" + "` when multiple labels overlap.
- `compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last)` — the linear `a, b` drift-correction fit (two-point groups only: 6, 7, 8, 9).
- `make_labeling_summary(...)` — per-tier labeled-row-count/percentage/unique-label summary, re-declared per group with minor signature differences (Group 10's version takes extra args to also print the shift/sync metadata).

## Hyperparameter-looking constants (verbatim)

- `EXPECTED_LANDMARKS = 3` — every raw-cleaning cell (3 participants per group, tracked as 3 rigid-body landmarks).
- Landmark-assignment thresholds (raw-cleaning pass, vary per reconstruction variant): `MAX_ASSIGN_DIST = 0.35` / `MAX_REASONABLE_JUMP = 0.45` / `SMOOTH_WINDOW = 5` (fast, CELL 1); `MAX_ASSIGN_DIST = 0.18` / `MAX_REASONABLE_JUMP = 0.25` / `MAX_MISSING_FRAMES_MEMORY = 240` (~1s @ 240Hz) / `SMOOTH_WINDOW = 5` (conservative, CELL 2) — units are meters (OptiTrack world units) and frames.
- OptiTrack native capture rate: `240` Hz, baked into filenames (`..._240hz.csv`) and rolling-window sizes (`rolling(120, ...)` ≈ 0.5s, `rolling(240, ...)` ≈ 1s) rather than a named constant.
- Movement-score smoothing windows: `rolling(120, center=True, min_periods=1)` (~0.5s, per-landmark movement) then `rolling(240, center=True, min_periods=1)` (~1s, combined sync score) — identical in every group's movement-score cell.
- Candidate-peak binning: `time_bin_2s = (time_s // 2).astype(int)`, top-1-per-bin then top-20-overall (`head(20)`) — identical across all groups' peak-detection cells.
- Take-boundary exclusion zone: `±2.0s` around each take's start time, excluded from sync-candidate consideration (avoids false peaks from recording start/stop transients).
- `SYNC_PLOT_WINDOW_S = 15` / `ZOOM_HALF_WINDOW_S = 10` / `TOP_N_ZOOMS = 8` — confirmation-plot window sizes (Group 10's cell and the "whole recording" visualization cells).
- Per-group hardcoded sync-time constants: see the table in "Sync recovery" §1 above — these are the actual load-bearing "hyperparameters" of this notebook (human-confirmed OptiTrack sync instants, one or two floats per group).
- Take-combination offsets, one per group/take-pair, derived from real Motive capture-start-time metadata and hardcoded per group, e.g. Group 1: `TAKE2_OFFSET_S = 1385.836` (from "10:23:10.716 -> 10:46:16.552").
- ELAN sync-label regex: `"sync|synchron"` (case-insensitive substring match), used identically everywhere a sync row is located.

## Potentially broken / disabled cells

- **No automatic end-to-end sync — every group's final shift is a hand-transcribed constant.** As detailed in §1 above, `auto_detect_optitrack_sync`/the movement-score candidate list is a real algorithm but is never the value actually used to shift the data in any of the 9 groups' saved cells; a human read a plot and typed a number (`OPTITRACK_SYNC_TIME_MANUAL = 35.95`, `# From Group 9 sync inspection`, etc.). Re-running this notebook from scratch on new data would require the same manual-confirmation step each time — it is not a push-button pipeline.
- **Manual marker-identity resolution (`CHAINS` dicts) is per-group/per-take hand curation, not reproducible from a rule.** Every group's raw-cleaning pass includes at least one "MANUAL TRACKLET STITCHING" cell with a literal dict of specific Motive `"Unlabeled NNNN"` marker-track IDs assigned to landmark1/2/3 — several groups (7, 8, 9, 10) even have a second "MANUAL TRACKLET STITCHING UPDATED" cell layering in more hand-picked gap-fill fragments (e.g. Group 9's comment: *"Take 2 L3: added Unlabeled 2532... Take 6 L1 kept unchanged because no safe successor was found"*). This is the single least-automatable step in the whole file and would need to be preserved verbatim (or re-derived by a human watching the same trajectory plots) for any from-scratch reproduction.
- **Heavy copy-paste duplication across all 18 per-group cell blocks (9 raw-cleaning + 9 labeling)** — the same core functions (`load_elan_annotations`, `normalize_label`, `find_sync_rows`/variants, `fast_assign_labels`, `make_labeling_summary`, plus the whole Motive-reading/landmark-reconstruction family) are independently re-declared, near-identically, in nearly every cell rather than imported once. A port to `src/preprocessing/` should consolidate these into a small number of shared, group-parameterized functions (e.g. `sync_group_optitrack(group, elan_path, optitrack_path, sync_config)`), taking the per-group constants from the table above as a config dict rather than as literal code duplicated per group.
- **`CELL 49/50 (#43/#44), Group 10 "COMBINE TAKE 1–3"` appears twice in immediate succession** with the same title and (from the outline) apparently overlapping logic — worth diffing directly before porting to confirm whether the second is a genuine fix/rerun of the first or accidental duplication left over from iterative editing (not fully diffed line-by-line in this pass given the size of the file; flagged for a closer look if Group 10's final combined file needs to be trusted).
- **No `TODO`/`FIXME` markers and no `!pip install` cells found anywhere** (checked by direct search). No `if False:` dead branches either — every cell's `to_csv`/`display` calls execute unconditionally when the cell runs.
- **Group 3/1's "single-point sync, drift not validated" is an acknowledged limitation, not a bug**: both cells' own inline comments state plainly that with only one ELAN `synchronization_move` available, there is no way to check for OptiTrack/ELAN clock drift over the session the way the two-point groups can (`drift = last_shift - first_shift` in CELL 61 etc.) — any drift present in groups 1, 2, 3, 5's data would silently propagate into their labeled OptiTrack timestamps.

## Summary

`OPTI_TRACK_PROCESSING.ipynb` is a **two-pass, per-group, semi-manual OptiTrack-only pipeline**: an first pass (cells 0–53) that reconstructs 3 persistent landmark identities per group from raw, noisy Motive marker exports (Hungarian-algorithm nearest-neighbor tracking, plus extensive **hand-curated tracklet stitching** to fix identity swaps and fill gaps) and combines multiple takes per group into one continuous, native-240Hz `group_{g}_optitrack_cleaned_combined_240hz.csv`; and a second pass (cells 55–85) that is exactly the sync-shift computation and ELAN label transfer the user recalled — it locates an ELAN `synchronization_move` interval per group (preferring the `Whole_Group` tier, falling back to a specific participant's tier for Group 2), locates a matching OptiTrack movement-magnitude peak (a real, reusable candidate-detection algorithm, though the constant ultimately used in every one of these 9 saved cells was human-confirmed rather than auto-selected), computes `SHIFT = Opti - ELAN` (or a 2-point linear `a*t+b` drift correction where two ELAN sync intervals are available), and writes one raw-rate (240 Hz), per-tier-labeled OptiTrack CSV per group. **This block genuinely recurs for all 9 study groups** (1, 2, 3, 5, 6, 7, 8, 9, 10 — group 4 excluded as everywhere else in this repo) as 9 separately hand-copied and hand-adapted cell blocks, not one parameterized loop, but the coverage is complete: every group has its own computed shift, hardcoded sync-time constant(s), and labeled output. This is a materially more complete finding than the previously-documented `sensor_sync_fixed.ipynb`/`sensor_sync_ROOT.ipynb` (confirmed group-1-only, and Xsens/OpenEarable rather than OptiTrack) — the two notebooks are independent implementations of the same general anchor-and-shift idea, not one generalized from the other. The notebook's `ELAN_PATH` inputs line up exactly, for all 9 groups, with the `Group_N_individual_build_renamed.csv` files that `FINAL_ARDA_THESIS_CODE_ONLY.py` writes, strongly suggesting these are two sequential stages of one real pipeline (ELAN anonymization/building → OptiTrack sync+labeling). What this notebook does **not** do: (1) it never touches Xsens or OpenEarable data at all (zero references anywhere in the file) — the sync gap for those two sensor families remains open; (2) it never builds any fixed-width (5s or otherwise) window grid or binary interaction/non-interaction label — its label transfer is per-row, at native 240Hz, with free-text ELAN label strings, not binned or binarized in any way. It therefore **does not resolve the ENG3/Task-1 5-second window-grid circularity** documented in `master_feature_generator_task1_task2_task3_CORRECTED_V4_ANALYSIS.md` — that from-scratch windowing/binarization step is still missing and would have to consume this notebook's `group_{g}_optitrack_labeled.csv` outputs (or equivalent labeled/synced files for Xsens and OpenEarable, which this notebook does not produce) as a starting point. No real participant names were found anywhere in this notebook's inputs or outputs — every `ELAN_PATH` is the pre-anonymized `_individual_build_renamed.csv` variant, and several cells even glob-restrict to `*renamed*` as an explicit safeguard.
