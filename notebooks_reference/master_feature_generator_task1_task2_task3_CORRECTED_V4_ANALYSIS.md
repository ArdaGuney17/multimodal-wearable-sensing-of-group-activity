# Analysis: master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`

Code-only extract used for line references below: `master_feature_generator_task1_task2_task3_CORRECTED_V4_CODE_ONLY.py` (7378 lines). That extract's own `--- CELL N (code cell #M) ---` numbering is **not** the raw notebook cell position — the extractor drops markdown cells but leaves numbering gaps where they were. Confirmed mapping (`code-only CELL N` = `raw notebook position N+1`): raw position 1 (title, markdown) has no code-only counterpart; positions 3, 11, 22, 24 are the four section-header markdown cells and show up as the missing code-only numbers 2, 10, 21, 23. Both numbering schemes are given throughout this document as `CELL n (code cell #m)`.

## Cell counts

Total cells: 27
Code cells: 22
Markdown cells: 5

## Import statements (deduplicated)

```python
from collections import Counter, defaultdict
from google.colab import drive
from IPython.display import display
from pathlib import Path
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC
import gc
import glob
import json
import math
import numpy as np
import pandas as pd
import os
import re
import shutil
import warnings
import zipfile
```

Note: the `sklearn.*` imports (clone, SelectKBest/f_classif, LogisticRegression, LeaveOneGroupOut, RobustScaler, LinearSVC/SVC, the metrics block) and the config constants that go with them (`RANDOM_STATE`, `K_LIST`, `FEATURE_SETS_TO_RUN`, `TIME_CONDITIONS`) appear only in CELL 9 (code cell #8) and are **never used anywhere in this notebook** — this notebook only builds and saves feature tables, it never trains or evaluates a model. They look like leftover boilerplate copied from a companion modelling notebook (e.g. `task1_full_comparison_classical_elapsed_dl_with_std.ipynb`). See "Potentially broken / disabled" below.

## Markdown headers / outline (in order)

- (raw pos 1, H1, no code-only cell) **Master Feature Generator — Corrected V4** — top-level notebook description; lists 5 claimed outputs (`binary_5s_all_sensor_advanced_features.csv`, `binary_5s_specialized_oe_merged_all_features.csv`, `interaction_eng3_features.csv`, `activity3_advanced_merged_10s_features.csv`, `activity_tokens_6label_fullstat.csv`) and an "Important provenance note" (the original Task 1 window/label-grid code and the old generic `oe__` generator were **not retained**; this notebook reconstructs them) and a "V4 Task 1 correction" note (OptiTrack/Xsens generators now use source-aware column-mapping/scoring).
  - **CELL 1 (code cell #1)**, lines 1-98 — `0. CONFIGURATION`: mounts Drive (optional), sets `SOURCE_DATA_ROOT`, `RAW_DIR`, `OUTPUT_DATA_ROOT`, `RUN_TASK1/2/3`, `TASK1_LABEL_GRID_PATH`, `RQ3_NORMALIZED_LABEL_PATH`, `BUILD_RECONSTRUCTED_GENERIC_OE`.
  - (raw pos 3, H1, code-only gap CELL 2) **Part A — Task 1** — derives the authoritative 5s interaction label grid, rebuilds OptiTrack/Xsens feature tables, creates a reconstructed generic OpenEarable table, then runs the verified SPECIAL_OE generator for the final Task 1 model input.
  - **CELL 3 (code cell #2)**, lines 99-212 — `A1. AUTHORITATIVE FIVE-SECOND BINARY WINDOW AND LABEL GRID`: bootstraps `TASK1_LABEL_GRID_PATH` from an existing official Task 1 table (first-existing of two candidates), loads it into `base_windows`.
  - **CELL 4 (code cell #3)**, lines 213-273 — `SECTION 0b — advanced statistic expansion (shared by 0c/0d)`: defines `adv_stats` (16-stat expansion), `load_raw_file`, `num` — shared helpers for the 5s OPTI2/XSENS2/generic-OE rebuilders.
  - **CELL 5 (code cell #4)**, lines 274-748 — `A2. TASK 1 OPTI2 FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR`: source-aware OptiTrack 5s feature rebuild → `rebuild_opti2_5s.csv`.
  - **CELL 6 (code cell #5)**, lines 749-1111 — `A3. TASK 1 XSENS2 FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR`: source-aware Xsens 5s feature rebuild (historical simple 16-stat schema) → `rebuild_xsens2_5s.csv`.
  - **CELL 7 (code cell #6)**, lines 1112-1215 — `A2. RECONSTRUCTED GENERIC OPENEAREABLE FEATURES`: reconstructs the lost generic `oe__` feature family (16-stat `adv_stats` over raw acc/gyro/mag) → `rebuild_generic_oe_5s.csv`.
  - **CELL 8 (code cell #7)**, lines 1216-1252 — `A3. MERGE RECONSTRUCTED TASK 1 COMPATIBILITY BASE TABLE`: merges label grid + OPTI2 5s + XSENS2 5s + generic OE → `binary_5s_all_sensor_advanced_features.csv`.
  - **CELL 9 (code cell #8)**, lines 1253-2086 — `SPECIALIZED OE9/OE10-STYLE FEATURES FOR 5s BINARY TASK`: rebuilds specialized head/posture/turn/nod/spectral/jerk/sync OE features for the exact 5s binary windows, drops the old generic `oe__` columns, merges → `binary_5s_specialized_oe_merged_all_features.csv` (**the final Task 1 model input**).
  - (raw pos 11, H1, code-only gap CELL 10) **Part B — Shared ENG3 and Task 2** — regenerates the 24-feature ENG3 table, derives recognition labels, runs the verified ENG7/OE10/OPTI2/XSENS2 generators, creates the final 10s Task 2 merged table.
  - **CELL 11 (code cell #9)**, lines 2087-2552 — `BUILD INTERACTION_ENG3`: 5s ENG window grid + old ENG proximity/motion features + Xsens hand features → `interaction_eng3_features.csv` (+ `interaction_eng3_tensors.npz`).
  - **CELL 12 (code cell #10)**, lines 2553-2870 — `RECOGNITION TASK — LABEL INVENTORY BEFORE GROUPING`: inventories raw pairwise/whole-group annotation labels before deciding a grouping scheme → 5 inventory/summary CSVs.
  - **CELL 13 (code cell #11)**, lines 2871-3291 — `RECOGNITION TASK — CREATE FINAL 5-CLASS INTERACTION DATASET`: maps raw labels to 5 classes (co_building/co_merging/co_inspection/conversation/object_transport) → `eng3_recognition_5class_interaction_only_features.csv` + audit files.
  - **CELL 14 (code cell #12)**, lines 3292-3328 — `B1. CREATE THE THREE-CLASS CORE RECOGNITION TABLE`: filters the 5-class table down to `{co_building, co_merging, conversation}` → `eng3_recognition_3class_core_features.csv`.
  - **CELL 15 (code cell #13)**, lines 3329-3566 — `BUILD INTERACTION_ENG7`: transfer-invariant, self-normalised 10s head/hand/cross-person-coordination features → `interaction_eng7_10s.csv`.
  - **CELL 16 (code cell #14)**, lines 3567-4217 — `BUILD OE9 — RICH OPENEAREABLE-ONLY FEATURES`: rich per-person + cross-person OpenEarable acc/gyro features (10s) → `interaction_oe9_10s.csv`.
  - **CELL 17 (code cell #15)**, lines 4218-4790 — `BUILD OE10 — ADD MAGNETOMETER FEATURES TO EXISTING OE9`: adds magnetometer-derived features to OE9 → `interaction_oe10_10s.csv`.
  - **CELL 18 (code cell #16)**, lines 4791-5569 — `OPTI2 — RICH OPTITRACK FEATURE GENERATION, ROBUST VERSION`: source-aware, rich 10s OptiTrack proximity/spread/formation features → `interaction_opti2_10s.csv`.
  - **CELL 19 (code cell #17)**, lines 5570-6370 — `XSENS2 — RICH XSENS FEATURE GENERATION`: source-aware, rich 10s Xsens acc/gyro/euler features → `interaction_xsens2_10s.csv`.
  - **CELL 20 (code cell #18)**, lines 6371-6681 — `B7. SAFE TASK 2 ADVANCED MERGE`: left-merges OE10 + OPTI2 onto the XSENS2-labelled base (992-row 3-class core) → `activity3_advanced_merged_10s_features.csv` (**the final Task 2 model input**).
  - (raw pos 22, H1, code-only gap CELL 21) **Part C — Task 3 activity tokens** — uses the regenerated `interaction_eng3_features.csv` plus the normalized RQ3 annotation table to build the six-label, 337-dimensional activity-token table for the Task 3 notebook.
  - **CELL 22 (code cell #19)**, lines 6682-6900 — `BUILD FULL-STATISTIC SIX-LABEL ACTIVITY TOKENS`: collapses consecutive-same-label windows into activity tokens with 14 statistics per numeric channel → `activity_tokens_6label_fullstat.csv`.
  - (raw pos 24, H1, code-only gap CELL 23) **Part D — Verification and manifest** — compares regenerated outputs with the existing official thesis files; exact equality expected for traceable generators; the reconstructed historical Task 1 base may differ in its generic `oe__` columns.
  - **CELL 24 (code cell #20)**, lines 6902-7154 — `D1. VERIFY REBUILT OUTPUTS AGAINST OFFICIAL TABLES`: schema/row/column-correlation diff of each generated file against its `REFERENCE_DATA_ROOT` counterpart → verification report + dataset manifest.
  - **CELL 25 (code cell #21)**, lines 7155-7192 — ad hoc text search across `/content/drive/MyDrive/thesis` (`*.ipynb`/`*.py`/`*.txt`) for known column/function names, to help locate the lost original generator code. Diagnostic, not part of the pipeline.
  - **CELL 26 (code cell #22)**, lines 7193-7378 — ad hoc archaeology search across all of `/content/drive/MyDrive` (including inside `.zip` archives) scoring notebook cells that mention `INTERACTION_BINARY_5S_ADVANCED_FEATURES`/`binary_5s_all_sensor_advanced_features.csv` for likely provenance. Diagnostic, not part of the pipeline.

## Hardcoded file paths found in code

### Inputs (upstream data this notebook reads but does not itself produce)

- `SOURCE_DATA_ROOT = "/content/drive/MyDrive/thesis/data"`
- `RAW_DIR = {SOURCE_DATA_ROOT}/ALL_MODEL_READY_FILES_IDENTITY_FIXED` — primary raw sensor source:
  - `group_{gnum}_optitrack_model_ready.csv`
  - `group_{gnum}_xsens_model_ready.csv`
  - `group_{gnum}_openearable_model_ready.csv`
- `{SOURCE_DATA_ROOT}/ALL_MODEL_READY_FILES/group_{group}_optitrack_model_ready.csv` — fallback OptiTrack source (scored against the primary; higher-coordinate-coverage source wins)
- `{SOURCE_DATA_ROOT}/ALL_MODEL_READY_FILES/group_{group}_xsens_model_ready.csv` — fallback Xsens source
- `{SOURCE_DATA_ROOT}/group_{group}/xsens/model_ready/group_{group}_xsens_model_ready.csv` — third fallback Xsens source (Task 1 5s and XSENS2 10s rebuilders only)
- **Bootstrap dependency — `TASK1_LABEL_GRID_SOURCE_CANDIDATES`** (first-existing wins, read once to populate `TASK1_LABEL_GRID_PATH` if it doesn't already exist):
  - `{SOURCE_DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv`
  - `{SOURCE_DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv`
  - (these are the exact *historical/official* versions of two of this notebook's own outputs — see "Potentially broken" below)
- `RQ3_NORMALIZED_LABEL_PATH = {SOURCE_DATA_ROOT}/RQ3_LABEL_NORMALIZATION/rq3_normalized_labels_full.csv` — required by Part C (Task 3 tokens); not produced anywhere in this notebook.
- **Verification-only inputs** (`REFERENCE_DATA_ROOT = SOURCE_DATA_ROOT`, read only for comparison in CELL 24, never merged into a feature table):
  - `{SOURCE_DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv`
  - `{SOURCE_DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv`
  - `{SOURCE_DATA_ROOT}/INTERACTION_ENG3/interaction_eng3_features.csv`
  - `{SOURCE_DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv`
  - `{SOURCE_DATA_ROOT}/PUBLICATION_TASK3_CORRECTED_FINAL/activity_tokens_6label_fullstat.csv`
- Diagnostic-only (CELL 25/26): `/content/drive/MyDrive/thesis/**/*.{ipynb,py,txt}`, `/content/drive/MyDrive/**/*.ipynb`, `/content/drive/MyDrive/**/*.zip`.

### Outputs (grouped by feature family / sensor combo)

All outputs are written under `OUTPUT_DATA_ROOT = {SOURCE_DATA_ROOT}/FEATURE_GENERATOR_REBUILT` (kept separate from `SOURCE_DATA_ROOT` so the official files are never overwritten), **except** `TASK1_LABEL_GRID_PATH` (see flag below) which is written directly under `{SOURCE_DATA_ROOT}/FEATURE_GENERATOR_INPUTS/`.

**Bootstrap side-effect (written into the input tree, not the output tree):**
- `{SOURCE_DATA_ROOT}/FEATURE_GENERATOR_INPUTS/task1_authoritative_5s_label_grid.csv`

**Task 1 — 5s binary interaction detection (OptiTrack + Xsens + OpenEarable):**
- `_parts` intermediates under `{OUTPUT_DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/_parts/`:
  - `task1_window_labels_5s.csv`
  - `rebuild_opti2_5s.csv` + `rebuild_opti2_5s_source_report.csv`
  - `rebuild_xsens2_5s.csv` + `rebuild_xsens2_5s_source_report.csv`
  - `rebuild_generic_oe_5s.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv` — OPTI2(16-stat)+XSENS2(16-stat)+reconstructed-generic-OE compatibility base
- `{OUTPUT_DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/`:
  - `binary_5s_specialized_oe9_oe10_features.csv`
  - `binary_5s_specialized_oe_merged_all_features.csv` — **final Task 1 model input**
  - `SUMMARY_PATH`, `PRED_PATH`, `BEST_PATH`, `EFFECT_PATH` constants (`binary_5s_specialized_oe_summary.csv`, `..._predictions.csv`, `..._best_per_condition.csv`, `..._effect.csv`) are **defined but never written** — dead output paths (see below).

**Shared ENG3 recognition-labelling family (5s):**
- `{OUTPUT_DATA_ROOT}/INTERACTION_ENG3/`:
  - `interaction_eng3_features.csv` + `interaction_eng3_tensors.npz`
  - `recognition_all_pairwise_wholegroup_raw_label_inventory.csv`
  - `recognition_raw_label_summary.csv`
  - `recognition_normalized_label_summary.csv`
  - `recognition_interaction_window_label_inventory.csv`
  - `recognition_interaction_window_label_summary.csv`
  - `eng3_recognition_5class_interaction_only_features.csv`
  - `eng3_recognition_5class_full_audit.csv`
  - `eng3_recognition_5class_label_mapping_audit.csv`
  - `eng3_recognition_5class_raw_label_mapping_audit.csv`
  - `eng3_recognition_3class_core_features.csv` — feeds the recognition labels used by ENG7/OE9/OE10/OPTI2/XSENS2 (10s)

**Task 2 — 10s activity recognition (co_building / co_merging / conversation):**
- `{OUTPUT_DATA_ROOT}/INTERACTION_ENG7/interaction_eng7_10s.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_OE9/interaction_oe9_10s.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_OE10/interaction_oe10_10s.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_OPTI2/interaction_opti2_10s.csv` + `opti2_source_selection_report.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_XSENS2/interaction_xsens2_10s.csv` + `xsens2_source_selection_report.csv`
- `{OUTPUT_DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv` — **final Task 2 model input** (merge of XSENS2 base + OE10 + OPTI2)

**Task 3 — activity tokens:**
- `{OUTPUT_DATA_ROOT}/PUBLICATION_TASK3_CORRECTED_FINAL/activity_tokens_6label_fullstat.csv`

**Verification / manifest (Part D):**
- `{OUTPUT_DATA_ROOT}/MANIFEST/COLUMN_DIFFS/{name}__only_generated.csv` and `{name}__only_reference.csv`, one pair per dataset name (`task1_advanced_base`, `task1_specialized`, `eng3`, `task2_advanced`, `task3_tokens`)
- `{OUTPUT_DATA_ROOT}/MANIFEST/feature_generator_verification.csv`
- `{OUTPUT_DATA_ROOT}/MANIFEST/generated_dataset_manifest.csv`

## Function / class definitions

No classes are defined anywhere in this notebook (pure function/script style). Grouped by feature family; no functions have docstrings unless noted.

**Shared 5s statistic-expansion helpers — CELL 4 (code cell #3), lines 220-271:**
- `adv_stats(out, name, x, t=None)` — writes 16 statistics into `out` under `{name}__{stat}`: mean, std, min, max, range, median, iqr, p10, p90, energy, rms, first_half_mean, second_half_mean, half_delta, mad, slope (linear-fit slope vs. time if `t` given). Used by every 5s Task 1 rebuilder (OPTI2, XSENS2, generic OE).
- `load_raw_file(kind, gnum)` — loads `group_{gnum}_{kind}_model_ready.csv` from `RAW_DIR`, derives a unified `t` column from `video_time_s`/`time_s`.
- `num(r, name, clean=1e6)` — numeric column extractor that NaNs out values `>= clean` in absolute value (corrupt-value guard).

**Optitrack proximity/spatial features:**
- CELL 5 (code cell #4), Task 1 5s OPTI2 (lines 292-746): `task1_standardize_columns`, `task1_find_landmark_mapping` (detects `landmark{i}_x/y/z` vs `p{i}_x/y/z` vs `participant{i}_x/y/z` vs `person{i}_x/y/z` column-naming conventions), `task1_opti_candidate_paths`, `task1_load_opti_candidate`, `task1_opti_source_score` (counts finite coordinate samples inside the actual windows), `task1_choose_opti_source` (picks highest-scoring candidate file). Main body computes per-participant position/speed, pairwise distances, nearest/farthest/mean/spread, centroid, centroid spread, triangle area, and active-speed-threshold counts (0.05/0.10/0.20 m/s), all expanded via `adv_stats` (16 stats each).
- CELL 18 (code cell #16), OPTI2 10s "robust version" (lines 4837-5400): `get_num_col`, `safe_stats` (9-stat expansion: mean/std/min/max/range/median/iqr/p10/p90 — this is the schema that matches the thesis's documented 9 statistics, unlike the 16-stat `adv_stats`), `safe_fraction`, `row_nanmin/max/mean/std`, `centroid_from_stack`, `compute_speed` (drops tracking jumps ≥10 m/s), `triangle_area_2d` (2D shoelace formula), `add_recognition_labels`, `standardize_columns`, `find_landmark_mapping`, `find_available_col`, `load_candidate_file`, `coordinate_score_for_windows`, `choose_best_opti_source`, `compute_window_features` — the large (~270-line) function building all OPTI2 10s features: availability/tracking-quality fractions, per-landmark position stats + 2D/3D speed + moving-fraction-at-threshold, pairwise 2D/3D distance stats + distance-change-rate, nearest/farthest/mean/std/range across pairs, 2D/3D centroid + centroid speed, 2D/3D spread-from-centroid, triangle area/perimeter/compactness (`area/(perimeter**2 + 1e-9)`), and cross-participant speed-mean asymmetry.

**OpenEarable audio/head-motion features:**
- CELL 9 (code cell #8), Task 1 specialized OE (lines 1386-1982): `discover_openearable`, `load_oe`, `sinterp` (NaN outside raw time range), `safe_mean/std/min/max/range/iqr/percentile`, `mad_diff`, `event_count` (rising-edge count), `spectral_entropy`, `band_ratio`, `corr_safe`, `max_lag_corr` (±1s lag search), `aggregate_values` (mean/std/min/max/range across the 3 participants), `circular_diff`, `session_baseline` (per-participant session-level acc/gyro mean/std/percentiles used for self-normalisation), `extract_specialized_oe_features` — the ~300-line function computing: pitch/roll (`atan2(acc_x, sqrt(acc_y²+acc_z²))` / `atan2(acc_y, sqrt(acc_x²+acc_z²))`), head-down fraction/regime-switch-rate, head-activity alternation, per-person acc/gyro energy+entropy+low/nod/high-band ratios, pitch/roll stats, jerk/angular-jerk, turn/burst event rates at p75/p90, active-person counts (acc/gyro/down) with exactly1/atleast2/all3 fractions, dominance-gap and asymmetry per feature, cross-person acc/gyro/pitch correlation and lag-correlation, first-half-vs-second-half deltas, and (if magnetometer present) magnitude/heading stats and pair correlations.
- CELL 16 (code cell #14), OE9 10s (lines 3628-4149): near-identical duplicate of the above helper set and `extract_oe9_features`/`session_baseline`, adapted to fixed 10s windows (no magnetometer).
- CELL 17 (code cell #15), OE10 10s (lines 4269-4718): `discover_openearable`, `load_mag`, `sinterp`, safe-stat family (duplicated again), `aggregate`, `mag_session_baseline`, `extract_mag_features` — adds magnitude z-score stats, heading (`unwrap(atan2(mag_y, mag_x))`), heading velocity, horizontal-strength stats, turn-rate at p75, and cross-person magnitude/heading-velocity correlation + lag-correlation, merged onto the existing OE9 table.

**Xsens hand/wrist IMU features:**
- CELL 6 (code cell #5), Task 1 5s XSENS2 (lines 768-1109): `task1_xsens_candidate_paths`, `task1_load_xsens_candidate`, `task1_sane_xsens_values` (clips |acc|>100, |gyr|>2000, |euler|>360 to NaN), `task1_xsens_source_score`, `task1_choose_xsens_source`. Main body keeps the "historical simple 16-statistic schema": per-axis acc/gyr/euler channels, per-participant acc/gyr/euler norms, per-participant availability, plus window-level available-channel count/mean-abs/energy — all via `adv_stats`.
- CELL 19 (code cell #17), XSENS2 10s (lines 5629-6221): `get_num_col`, `safe_stats`/`safe_fraction` (9-stat schema, same as OPTI2 10s), `add_recognition_labels`, `clean_signal` (same clip thresholds as Task 1's `task1_sane_xsens_values`), `unwrap_degrees`, `vector_norm`, `derivative_norm`/`derivative_1d` (jerk/rate helpers with a `max_value` outlier clip), `interp_nan`, `spectral_features` (entropy + low/mid/high band ratios at 0.1-0.5/0.5-2.0/2.0-6.0 Hz), `corr_feature`, `standardize_columns`, `candidate_paths_for_group`, `load_xsens_file`, `sanity_score_for_source`, `choose_best_xsens_source`, `compute_xsens_window_features` — per-participant acc/gyr/euler axis stats, acc/gyr norms, dynamic acceleration (axis minus window median), jerk/angular-jerk norms, Euler rate, burst fractions at acc-dyn 0.5/1.0/2.0 and gyro 30/60/100, spectral entropy/band ratios, group-level active-participant counts, cross-person acc/gyro correlation + abs-diff stats, and movement-asymmetry summaries.

**Cross-modal / coordination features (ENG3, ENG7):**
- CELL 11 (code cell #9), ENG3 5s (lines 2127-2520): `discover` (requires all 3 sensor files present per group), `load`, `sinterp`, `xoff` (Xsens-to-video time-offset search via label agreement, 0-220s in 0.5s steps), `spec` (dominant frequency + power), `safe_mean`/`safe_max`, `old_eng_features_and_tensor` — 17 legacy ENG features: sorted pairwise distances (close/mid/far), centroid speed, sorted per-participant acc/gyro-energy and speed (min/mid/max), movement coordination (mean pairwise acc-magnitude correlation) — plus a parallel 7-channel tensor (for the accompanying `.npz`), `xsens_hand_features` — hand dominant-frequency, log-power, orientation variance, 3-way hand-motion correlation, `build_eng3` — assembles 5s windows per group, majority-vote interaction label from 4 annotation tiers.
- CELL 15 (code cell #13), ENG7 10s (lines 3368-3564): `discover`, `load`, `xoff`, `sinterp`, `band_ratio`, `spectral_entropy`, `active_mask`, `event_count`, `burstiness` (CV of inter-event gaps), `features` — 4 explicitly-named "transfer-invariant" families: (1) head-posture regime (down-fraction/pitch-range/regime-switch-rate), (2) gaze events (turn-event-rate, nod-band-ratio), (3) hand rhythm (active-fraction, burstiness, spectral entropy, intermittency), (4) cross-person structure (head-activity alternation, hand-coupling-rate, wrist-handover-event-rate, role-split index/talk-fraction/work-fraction) — plus a small OptiTrack proximity baseline for comparison, `session_baseline`, `build`.

**Label engineering / recognition-class construction:**
- CELL 12 (code cell #10) (lines 2578-2750): `discover`, `clean_label_value`, `normalize_label` (lowercases, fixes 4 known typos: "mering"→"merging", "synchornizaion"/"syncornaziton"→"synchronization", "erarble"→"earable"), `is_technical_label`, `load_openearable_labels`.
- CELL 13 (code cell #11) (lines 2919-3074): `normalize_label` (same, plus "handiver"→"handover"), `is_technical_or_sync`, `map_to_5class` — priority-ordered keyword mapping of raw/dominant annotation labels into `{co_building, co_merging, co_inspection, conversation, object_transport}` or `ignore` (documented docstring, quoted in full in the code: "Important: non_interaction is NOT used... mixed labels are mapped by priority: collaborative physical activity, inspection, object transport, conversation").

**Task 2 merge / output-writing:**
- CELL 20 (code cell #18) (lines 6407-6519): `unique_feats` (order-preserving dedup), `add_merge_keys` (rounds `window_start`/`window_end` to `decimals` places for a fuzzy-float join key), `merge_features_left_safe` — tries merge-key rounding at 6→1 decimals and keeps whichever round matched the most base rows (never drops base rows even on total mismatch), `is_xsens_quality_feature`, `is_xsens_raw_euler_posture`, `is_opti2_quality_feature`, `is_opti2_absolute_position_feature` (defined but **never called** — see below), `create_safe_advanced_dataset` — uses XSENS2 as the labelled 992-row base, left-merges a filtered subset of OE10 motion/magnitude columns (renamed `oe__*`) and all `opti2_*`/3 legacy `opti_*` proximity columns.

**Task 3 token construction:**
- CELL 22 (code cell #19) (lines 6700-6892): `first_existing`, `load_six_label_windows` (merges normalized RQ3 labels with ENG3 or OE10 sensor features on rounded group/time keys), `channel_statistics` — 14 stats per channel: mean, std, min, max, range, median, iqr, p10, p25, p75, p90, energy, rms, spectral entropy, `build_fullstat_tokens` — collapses consecutive-same-label windows (`cumsum` of label-change) into one token per activity segment, with duration/start_time plus 14-stat expansion of every sensor channel.

**Verification / manifest:**
- CELL 24 (code cell #20) (lines 6964-7102): `schema_and_value_report(name, generated_path, reference_path)` — row/column counts, common/only-generated/only-reference column sets, matching-key-row count, and median Pearson correlation + count of common numeric columns with correlation ≥0.99 between the regenerated file and its official counterpart.

**Diagnostic archaeology (not part of the pipeline):**
- CELL 25/26 (code cells #21/#22): `extract_notebook_text`, `inspect_notebook` — scores notebook cells across the whole Drive (including inside `.zip` files) by whether they mention the Task 1 output name/folder and contain file-writing patterns, to try to locate the lost original generator notebook.

## Hyperparameter-looking constants (verbatim)

- `WINDOW_SECONDS = 5.0` / `STRIDE_SECONDS = 5.0` (CELL 3, Task 1 label grid)
- `WINDOW_S, STRIDE_S, RESAMPLE_T = 5.0, 5.0, 64` (CELL 11, ENG3)
- `WINDOW_S, STRIDE_S = 10.0, 10.0` (CELL 15, ENG7)
- `WINDOW_S = 10.0` / `STRIDE_S = 10.0` (CELL 16, OE9)
- (OE10, OPTI2, XSENS2 10s cells inherit their windows from the ENG7/XSENS2-derived base table rather than redefining WINDOW_S)
- `RESAMPLE_HZ = 25` (CELLs 9, 15, 16, 17 — OE feature resampling grid)
- `NOD_BAND = (1.0, 3.0)` — Hz, head-nod frequency band (CELLs 9, 15, 16)
- `LOW_MOTION_BAND = (0.2, 1.0)` / `HIGH_MOTION_BAND = (3.0, 8.0)` (CELLs 9, 16 — OE acc/gyro band ratios)
- `MAG_LOW_BAND = (0.2, 1.0)` / `MAG_MID_BAND = (1.0, 3.0)` / `MAG_HIGH_BAND = (3.0, 8.0)` (CELL 9, magnetometer)
- `LOW_BAND = (0.2, 1.0)` / `MID_BAND = (1.0, 3.0)` / `HIGH_BAND = (3.0, 8.0)` (CELL 17, OE10 magnetometer — same values, re-declared)
- Xsens spectral bands (CELL 19): `low = freqs in [0.1, 0.5)`, `mid = [0.5, 2.0)`, `high = [2.0, 6.0)` Hz
- `down_thr = -0.35` — radians (~ -20°), head-down pitch threshold (CELLs 9, 15, 16)
- Speed/activity thresholds:
  - `opti2_active_speed_count_gt_{0_05,0_10,0_20}` — 0.05 / 0.10 / 0.20 m/s (CELL 5, Task 1 OPTI2 5s)
  - `opti2_lm{i}_moving2d_frac_{005,010,020}` — same 0.05/0.10/0.20 m/s thresholds (CELL 18, OPTI2 10s)
  - `compute_speed(...)`: tracking-jump reject `speed < 10.0` m/s (CELL 18)
- Xsens corrupt-value clip thresholds (CELLs 6, 19): `acc: |x| > 100.0 → NaN`, `gyr/gyro: |x| > 2000.0 → NaN`, `euler: |x| > 360.0 → NaN`
- Xsens 10s burst-fraction thresholds (CELL 19): `acc_dyn_gt_{0.5, 1.0, 2.0}`, `gyr_gt_{30.0, 60.0, 100.0}` (deg/s)
- Xsens jerk outlier clips (CELL 19): `derivative_norm(acc_dyn, max_value=500.0)`, `derivative_norm(gyr, max_value=10000.0)`, `derivative_1d(euler, max_value=1000.0)`, `derivative_norm(euler, max_value=2000.0)`
- ENG7 hand-coupling threshold: `active_mask(hand_e[p], 0.5)` (~0.4-0.5s simultaneity), handover anti-phase threshold `da>0.3 & db<-0.3` (CELL 15)
- `max_lag_steps = int(1.0 * fs)` — ±1 second max lag search for cross-person lag-correlation (CELLs 9, 16, 17)
- `spectral_entropy`/`band_ratio` minimum sample count: `len(s) < 8 → NaN` (CELLs 9, 15, 16, 17); XSENS2's `spectral_features` uses `< 16`
- Percentile thresholds for burst/turn events: p75, p80, p90 of each participant's own **session** distribution (self-normalisation design, CELLs 9, 15, 16)
- `adv_stats` 16-stat set (CELL 4): `mean, std, min, max, range, median, iqr, p10, p90, energy, rms, first_half_mean, second_half_mean, half_delta, mad, slope`
- `safe_stats` 9-stat set (CELLs 18, 19): `mean, std, min, max, range, median, iqr, p10, p90` — matches the thesis's documented 9 window statistics exactly
- `channel_statistics` 14-stat set (CELL 22, Task 3 tokens): `mean, std, min, max, range, median, iqr, p10, p25, p75, p90, energy, rms, entropy`
- `CORE = ["co_building", "co_merging", "conversation"]` (CELLs 18, 19, 20 — Task 2 3-class label set)
- `GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]` — explicit 9-group list, group 4 excluded (CELLs 18, 19)
- `MERGE6 = {'social_conversation': 'conversation', 'task_conversation': 'conversation'}` (CELL 22, Task 3 label merge)
- `STAT_NAMES`/token feature-dimension count: notebook's own markdown claims **337-dimensional** activity tokens (raw pos 1); not independently re-derived here (depends on how many sensor columns survive `load_six_label_windows`'s numeric-column filter at run time).
- `dt <= 1e-6` guards throughout (speed/derivative division-by-zero protection)
- Corrupt-value guard used broadly: `abs(v) < 1e6` (or `1e9` for OE10's `load_mag`) → else NaN

## Potentially broken / disabled cells

- **Hardcoded dead branch**: CELL 22 (code cell #19), line 6881: `if False and os.path.exists(TOKEN_PATH):` — the "reload existing token table" branch is permanently disabled by the literal `False`, so Part C **always rebuilds `activity_tokens_6label_fullstat.csv` from scratch** on every run, regardless of whether a prior copy exists. This is different from the `RESUME_EXISTING`-style caching seen in the companion Task 3 notebooks — here caching was clearly attempted (the branch exists) but is switched off.
- **`RUN_TASK2` and `RUN_TASK3` flags are declared but never checked.** `RUN_TASK1` (CELL 1) is checked in exactly 3 places (CELLs 5, 6, 7 — the Task 1 5s OPTI2/XSENS2/generic-OE rebuilders). `RUN_TASK2` and `RUN_TASK3` (also set `True` in CELL 1) are **never referenced again anywhere in the file** — grepping the whole notebook confirms only 3 occurrences of `RUN_TASK1` and zero occurrences of `RUN_TASK2`/`RUN_TASK3` outside their assignment. Setting either to `False` would have **no effect**: all of Part B (ENG3/ENG7/OE9/OE10/OPTI2/XSENS2/Task 2 merge) and Part C (Task 3 tokens) still run unconditionally. Porting to `src/features/` should either wire these flags up properly or drop them — as written they are misleading dead configuration.
- **Circular / self-referential bootstrap for the Task 1 window grid.** `TASK1_LABEL_GRID_PATH` (CELL 1/CELL 3) is populated, on first run, by reading whichever of `binary_5s_specialized_oe_merged_all_features.csv` or `binary_5s_all_sensor_advanced_features.csv` **already exists under `SOURCE_DATA_ROOT`** (i.e. the *official, historical* version of two of this very notebook's own output files) and exporting just the timing/label/provenance columns. This means the notebook is **not actually runnable from raw sensor data alone** — it needs at least one historical official Task 1 output file to exist first, to recover the original 5s window boundaries and interaction labels (the code comment admits this directly: "The exact historical window-grid creation code was not retained"). Once `TASK1_LABEL_GRID_PATH` has been written once, subsequent runs reuse it without touching the official file again — the same "inherit-then-freeze" bootstrap pattern seen in the companion Task 3 `FINAL_V2` notebook's `OLD_TOKEN_PATH`/`NEW_TOKEN_PATH` copy step.
- **Unused sklearn imports/config in CELL 9 (code cell #8).** The specialized-OE cell imports `clone`, `LeaveOneGroupOut`, `RobustScaler`, `SelectKBest`/`f_classif`, `LogisticRegression`, `LinearSVC`/`SVC`, and the full `sklearn.metrics` block, and defines `RANDOM_STATE = 42`, `K_LIST = [40, 80, 120, 200]`, `FEATURE_SETS_TO_RUN = [...]`, `TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]` — none of these are ever used in this cell or anywhere else in the notebook. Likewise the paths `SUMMARY_PATH`, `PRED_PATH`, `BEST_PATH`, `EFFECT_PATH` are defined but **never written to**. This looks like boilerplate copied wholesale from the companion `task1_full_comparison_classical_elapsed_dl_with_std.ipynb` (which does use exactly these names) and left in place after the model-training logic was stripped out, leaving this cell purely a feature-generation cell despite its leftover ML imports.
- **Dead helper function**: `is_opti2_absolute_position_feature` (CELL 20, code cell #18) is defined but never called anywhere in the merge logic — `is_opti2_quality_feature`, `is_xsens_quality_feature`, and `is_xsens_raw_euler_posture` are likewise defined but not obviously invoked in the read code path (`create_safe_advanced_dataset` only filters by column-name prefix, not by these helper predicates). These look like leftover feature-exclusion logic from an earlier version of the merge that is no longer applied — worth checking during port whether absolute-position/quality columns should in fact be filtered out (Table 5.1's "tracking quality" family suggests they're meant to be kept as features, so this is plausibly intentional dead code rather than a bug, but it's inconsistent with the function names existing at all).
- **"Always recreate" comment on Task 2's final merge** (CELL 20, line 6673): `# Always recreate because previous file was broken.` followed by an unconditional `os.remove(ADVANCED_DATA_PATH)` if it exists, then a fresh rebuild. This is evidence the merge logic in this cell was fixed after a prior bug (the surrounding `merge_features_left_safe` docstring literally says "This avoids the previous Shape: (0, 1543) problem") — not currently broken, but a signal that Task 2's merge step had at least one prior failure mode (a merge that produced 0 matched rows) that a porter should be aware of and test against.
- **Two incompatible window-statistic schemas coexist.** The Task 1 5s rebuilders (CELLs 5, 6, 7, via `adv_stats`, CELL 4) expand every base signal into **16** statistics (including `energy`, `rms`, `first_half_mean`/`second_half_mean`/`half_delta`, `mad`, `slope`), while the Task 2 10s generators (CELLs 18, 19, via `safe_stats`) expand into a different, **9**-statistic set (`mean, std, min, max, range, median, iqr, p10, p90`). Only the 9-stat set matches the thesis's Section 5.10.1 "Window Statistic Expansion" table verbatim. This is not a bug (both schemas are internally consistent and intentional per their own cells) but it means "the statistic-expansion schema" is not uniform across this single notebook, and a porter reproducing Table 5.1's exact 9-statistic claim should base that specifically on the CELL 18/19 (`safe_stats`) code path, not CELL 4/5/6/7 (`adv_stats`).
- **Group-specific special-casing**: none found. Source selection (`task1_choose_opti_source`, `task1_choose_xsens_source`, `choose_best_opti_source`, `choose_best_xsens_source`) is a generic per-group scoring/fallback algorithm applied uniformly to all groups — no notebook analogous to the companion notebooks' `NAIVE_GROUPS = {2, 3, 5, 6, 10}` hardcoded subset exists here. `GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]` (group 4 excluded) is applied uniformly, consistent with the 9-group thesis design.
- **No `TODO`/`FIXME` comments** and no `!pip install` cells were found anywhere in the file. `drive.mount` (CELL 1) is wrapped in `try/except`, so Colab is optional, not required.
- **Do all claimed output files actually get written unconditionally?** Yes for the 5 files the notebook's own markdown (raw pos 1) advertises as "Outputs" — `binary_5s_all_sensor_advanced_features.csv` (CELL 8, unconditional), `binary_5s_specialized_oe_merged_all_features.csv` (CELL 9, unconditional), `interaction_eng3_features.csv` (CELL 11, unconditional), `activity3_advanced_merged_10s_features.csv` (CELL 20, unconditional — the file is even force-deleted and rebuilt every run), `activity_tokens_6label_fullstat.csv` (CELL 22, unconditional given the `if False` branch above always takes the rebuild path). The only conditionally-written outputs are the CELL 5/6/7 5s `_parts` intermediates, gated behind `RUN_TASK1` (`True` by default) and, for the generic-OE reconstruction specifically, also `BUILD_RECONSTRUCTED_GENERIC_OE` (`True` by default; if `False`, `binary_5s_all_sensor_advanced_features.csv` would be written **without any `oe__` columns at all**, an empty-frame fallback is used instead).

## Cross-reference against `docs/thesis_reproduction_targets.md` Chapter 5 tables

- **Table 5.1 (OptiTrack)** → CELL 18 `compute_window_features` (lines 5130-5400), and the simpler CELL 5 Task 1 5s OPTI2 body:
  - Pairwise distance → `opti2_pair{ab}_dist2d/dist3d` (CELL 18); `opti2_{name}_dist` per pair (CELL 5). ✓
  - Nearest/middle/farthest pair → `opti2_nearest_pair_dist2d/3d`, `opti2_farthest_pair_dist2d/3d` computed (CELL 18); an explicit **"middle"** pair-distance column is **not** produced by CELL 18 (only nearest/farthest/mean/std/range across the 3 pairs). CELL 11's older `old_eng_features_and_tensor` (ENG3, 5s) *does* have explicit `dist_close/dist_mid/dist_far` (sorted 1st/2nd/3rd pairwise distance) — so the "middle pair" concept exists in the legacy 5s ENG3 code but is not carried into the newer, richer 10s OPTI2 generator. Minor gap to note for the port.
  - Distance change rate → `opti2_pair{ab}_dist2d_change_rate`/`dist3d_change_rate` (CELL 18). ✓
  - Centroid / centroid speed → `opti2_centroid2d_*`/`centroid3d_*`, `opti2_centroid_speed2d/3d` (CELL 18); `opti2_centroid_{axis}`, no explicit centroid-speed column in CELL 5's 5s version (only in the 10s version). ✓ (10s), partial (5s)
  - Group spread → `opti2_spread2d/3d_mean/std/max` (CELL 18); `opti2_group_spread` (CELL 5). ✓
  - Triangle area/perimeter → `opti2_triangle_area2d`, `opti2_triangle_perimeter2d` (CELL 18); `opti2_triangle_area` only, no perimeter, in CELL 5's 5s version. ✓ (10s)
  - Triangle compactness → `opti2_triangle_compactness2d = area/(perimeter**2+1e-9)` (CELL 18) — matches the docs formula exactly. ✓
  - Tracking quality → `opti2_lm{i}_available_frac`, `opti2_active_landmarks_*`, `opti2_all3/atleast2/atleast1_available_frac` (CELL 18). ✓
- **Table 5.2 (OpenEarable)** → CELL 9 `extract_specialized_oe_features`, CELL 16 `extract_oe9_features`, CELL 17 `extract_mag_features`:
  - Acceleration/gyroscope magnitude → `acc_mag`/`gyr_mag` = `np.linalg.norm(...)`. ✓
  - Pitch/roll → `atan2(acc_x, sqrt(acc_y²+acc_z²))` / `atan2(acc_y, sqrt(acc_x²+acc_z²))` — this **resolves the docs' flagged OCR-uncertain formula** exactly as the docs' "general form" guess. ✓
  - Jerk/angular jerk → `np.diff(acc_mag)*fs` / `np.diff(gyr_mag)*fs`. ✓
  - Head-down fraction → `ear_head_down_fraction`, `oe_down_fraction`/`oe_up_fraction` (pitch < -0.35 rad). ✓
  - Turn rate → `turn_rate_p75/p90`, `ear_head_turn_event_rate` (gyro bursts above the person's own session p75/p80/p90). ✓ matches docs' "Gyroscope bursts above percentile thresholds" exactly.
  - Nod-band/entropy → `nod_band_ratio` (1-3 Hz), `spectral_entropy` on acc/gyro. ✓
  - Magnetometer magnitude/heading → `mag_norm`, `heading = unwrap(atan2(mag_y, mag_x))` — matches docs exactly. ✓
  - Active-person count → `oe_acc_active_count`/`gyro_active_count`/`down_count` with exactly1/atleast2/all3/switch-rate. ✓
  - Dominance/asymmetry → `oe_{name}_dominance_gap = max - median`, `oe_{name}_asymmetry = std/(|mean|+eps)` — matches docs' "Max-minus-median and std/mean across people" verbatim. ✓
  - Synchrony/lag → `oe_pair_{acc,gyro,pitch}_corr`, `oe_pair_{acc,gyro}_lagcorr`, `mag_pair_*_corr`/`lagcorr`. ✓
- **Table 5.3 (Xsens)** → CELL 19 `compute_xsens_window_features` (10s, "rich" schema) and CELL 6 (5s, "historical simple 16-statistic schema"):
  - Acceleration/gyroscope norm → `vector_norm` = `sqrt(x²+y²+z²)`. ✓
  - Dynamic acceleration → `acc_x_dyn = acc_x - nanmedian(acc_x)` (per-axis window-median subtraction) — matches docs' "Acceleration axis minus window median" exactly. ✓
  - Jerk/angular jerk → `derivative_norm(acc_dyn)`/`derivative_norm(gyr)`. ✓
  - Euler rate → `derivative_1d`/`derivative_norm` on unwrapped Euler angles. ✓
  - Burst fractions → `acc_dyn_gt_{0.5,1.0,2.0}_frac`, `gyr_gt_{30,60,100}_frac`. ✓
  - Spectral features → `spectral_features`: entropy + low/mid/high band ratios. ✓
  - Pair synchrony → `xsens2_pair{ab}_acc_dyn_corr`/`gyr_corr` + abs-diff stats. ✓
- **Table 5.4 (category summary)** → all 6 rows are represented: Proximity/Spatial-movement/Group-dispersion (OPTI2, CELL 18), Head-movement (OE9/OE10/specialized-OE, CELLs 9/16/17), Hand-movement (XSENS2, CELL 19), Coordination (cross-person synchrony/lag columns spread across OPTI2/OE/XSENS2, **plus** the explicitly-named "cross-person structure" family in ENG7, CELL 15: alternation/coupling-rate/handover-rate/role-split).
- **Section 5.10.1 (9 window statistics)** → `safe_stats` in CELLs 18/19 matches the documented 9-statistic set (mean/std/min/max/range/median/iqr/p10/p90) exactly. The 5s Task 1 rebuilders (`adv_stats`, CELLs 4-7) instead use a **16**-statistic superset (adds energy/rms/first-half-mean/second-half-mean/half-delta/mad/slope) not documented in the thesis text at all — this notebook computes strictly more per-signal statistics for the 5s Task 1 tables than Chapter 5 describes. Flag for the port: decide whether the extra 7 statistics are wanted for Task 1, or whether Task 1 should be switched to the 9-stat schema for consistency with the documented methodology.
- **Section 5.10.2 feature-count claims (531 OptiTrack / 506 OpenEarable / 693 Xsens features, 992 windows: 548 co_building / 311 conversation / 133 co_merging)** — these exact counts could not be independently verified by static reading (they depend on how many raw columns survive per-group at run time); they should be checked against the actual output shape of `activity3_advanced_merged_10s_features.csv` (CELL 20) once the pipeline is run. The class-count breakdown in particular is a good regression check for a port.
- **Nothing in the notebook's code appears to be un-described by the docs** — every feature family the code computes (including the OE10 magnetometer/heading features, the ENG7 "role-split"/"talk-fraction"/"work-fraction" cross-modal indices, and Task 1's 16-stat superset) maps onto at least one of the docs' conceptual categories (Sections 3.1-3.5 / Table 5.4), even where the exact statistic list goes beyond what Chapter 5 enumerates.

## Summary

This notebook is a **feature-regeneration / reconstruction** pipeline, not a from-scratch feature-engineering pipeline: its own markdown and code comments state plainly that the original notebook(s) that created the Task 1 five-second window grid and the old generic OpenEarable feature family were lost, so this "CORRECTED V4" version bootstraps the authoritative window/label grid by copying timing+label columns out of an already-existing official Task 1 output file, then regenerates every sensor feature value from the raw `group_{g}_{sensor}_model_ready.csv` files under a source-aware column-mapping/scoring scheme (so it tolerates `landmark{i}_x` vs `p{i}_x` vs `participant{i}_x` naming and picks whichever of `ALL_MODEL_READY_FILES_IDENTITY_FIXED`/`ALL_MODEL_READY_FILES`/a per-group nested path has the most usable, sane data inside the actual window boundaries). Structurally it processes **all 9 groups uniformly** (`GROUPS = [1,2,3,5,6,7,8,9,10]`, group 4 excluded by design, consistent with the rest of the thesis codebase) with no group-specific exceptions beyond the generic per-group source-selection scoring; it iterates per-group internally but always accumulates into one combined table per feature family (it does not write one file per group — the per-group `_source_report`/`source_selection_report` CSVs are metadata about *which raw file was chosen*, not per-group feature outputs). The notebook produces roughly a dozen distinct numeric feature CSVs across three families and two window sizes: **Task 1 (5s, binary interaction detection)** — `binary_5s_all_sensor_advanced_features.csv` (a compatibility-reconstruction base combining rebuilt 16-stat OPTI2 + XSENS2 + reconstructed generic OE) and, built on top of it by dropping/replacing the generic OE columns with a much richer specialized OE9/OE10-style feature set, `binary_5s_specialized_oe_merged_all_features.csv` (the actual Task 1 model input); **shared ENG3 labelling (5s)** — `interaction_eng3_features.csv` (24 legacy proximity/motion + hand features) which is then relabelled into a 5-class and finally a 3-class (`co_building`/`co_merging`/`conversation`) recognition table, `eng3_recognition_3class_core_features.csv`, that defines the labelled window grid for everything downstream in Task 2; **Task 2 (10s, 3-class activity recognition)** — four parallel rich per-sensor generators (`interaction_eng7_10s.csv`, `interaction_oe9_10s.csv`→`interaction_oe10_10s.csv`, `interaction_opti2_10s.csv`, `interaction_xsens2_10s.csv`), safely left-merged (with a fuzzy rounded-timestamp join and a "never drop base rows" merge strategy that was explicitly patched after a prior "0 rows matched" bug) into the single final `activity3_advanced_merged_10s_features.csv`; and **Task 3 tokens** — `activity_tokens_6label_fullstat.csv`, built by collapsing consecutive same-label 5s windows (from the RQ3-normalized six-label annotation) into activity segments with 14 statistics per numeric sensor channel. A future porter into `src/features/` needs to know: (1) the Task 1 pipeline is **not independently reproducible from raw data** without at least one historical official Task 1 CSV already present to bootstrap the window/label grid — this is an external dependency that must either be supplied as a fixture or have its original window-construction logic recovered/reimplemented; (2) two incompatible window-statistic schemas coexist (16-stat `adv_stats` for the 5s Task 1 tables vs. the thesis-matching 9-stat `safe_stats` for the 10s Task 2 tables) and a decision is needed on which to standardize on; (3) `RUN_TASK2`/`RUN_TASK3` are inert flags and CELL 22's cache-reuse branch is permanently disabled by a literal `if False`, so as written every run is a full, unconditional rebuild of every output — fine for a clean port but worth making the flags actually functional if incremental/partial runs are desired; and (4) CELL 9 carries a large block of unused `sklearn` imports/config and four dead output-path constants that should simply be dropped during the port, since this notebook never trains or evaluates a model — it only manufactures the feature CSVs that the separate Task 1/2/3 modelling notebooks (and, per the task description, `src/models/*.py` in this repo) consume by exact filename.
