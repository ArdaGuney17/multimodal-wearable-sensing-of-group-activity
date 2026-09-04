# Analysis: 07_feature_engineering_ENG7_activity_invariant_EXECUTED.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\07_feature_engineering_ENG7_activity_invariant_EXECUTED.ipynb`

Code-only extract used for line references below: `07_feature_engineering_ENG7_activity_invariant_EXECUTED_CODE_ONLY.py` (19,377 lines). That extract's own `--- CELL N (code cell #M) ---` numbering is the notebook's **0-indexed raw cell position** (i.e. extractor `CELL N` = notebook cell array index `N` = 1-indexed notebook position `N+1`). Confirmed by parsing the `.ipynb` directly: extractor `CELL 1 (code cell #1)` = notebook position 2 (`drive.mount(...)`), extractor `CELL 39 (code cell #28)` = notebook position 40 (the SPECIALIZED-OE cell), etc. Both numbering schemes are given throughout this document as `CELL n (code cell #m)` / `raw position p`.

## Cell counts

Total cells: 49
Code cells: 36
Markdown cells: 13

## Import statements (deduplicated)

```python
from google.colab import drive
from IPython.display import display
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset, DataLoader
import gc
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import numpy as np, pandas as pd
import os
import os, glob, re, gc, warnings
import pandas as pd
import random
import re
import shutil
import torch
import torch.nn as nn
import warnings
```

Note: `drive.mount(...)` (CELL 1, raw pos 2) is a **bare, unguarded call** — unlike several sibling notebooks in this repo, it is not wrapped in `try/except`, so this notebook is Colab-**required**, not Colab-optional (CELL 2's "safe remount" only wraps the `flush_and_unmount()` step in `try/except`; the `drive.mount("/content/drive", force_remount=True)` call itself is unguarded).

## Markdown headers / outline (in order)

- (raw pos 1, H1, no code-only cell) **`INTERACTION_ENG7` — activity-discriminating, transfer-invariant head & hand features** — top-level title, "Designed for co_building vs co_merging vs conversation".
- **CELL 1 (code cell #1)**, raw pos 2 — `drive.mount('/content/drive')`.
- **CELL 2 (code cell #2)**, raw pos 3 — "SAFE GOOGLE DRIVE REMOUNT FIX": flush/unmount, remove local mount folder, remount.
- (raw pos 4, MD, no code-only cell) "After running the cell above, please run the mount cell again." — operator note.
- (raw pos 5, MD, no code-only cell) "> **Why this is a different bet, honestly.** Four prior representations failed on LOGO because they measured *how much* each person moved..." — design-rationale blockquote motivating the self-normalised/invariant-feature approach (angles/fractions/counts/ratios rather than raw magnitudes) used throughout this notebook.
- (raw pos 6, H2, no code-only cell) **"## 1 · Build ENG7 (run once; writes `interaction_eng7_10s.csv`)"**
  - **CELL 6 (code cell #3)**, raw pos 7 — `BUILD INTERACTION_ENG7`: 10s window builder, 4 feature families (head posture, gaze events, hand rhythm, cross-person structure) + a 3-feature OptiTrack proximity baseline → `interaction_eng7_10s.csv`.
- (raw pos 8, H2, no code-only cell) **"## 2 · Test on 3-class recognition + see the physical contrasts"**
  - **CELL 8 (code cell #4)**, raw pos 9 — `ENG7 -> 3-CLASS RECOGNITION`: labels ENG7 10s windows from the ENG3 3-class recognition core table, evaluates RandomForest/LogReg over 9 feature-subset combinations, prints RF feature importances.
- (raw pos 10, MD, no code-only cell) "TESTING ON OPENEARABLE"
  - **CELL 10 (code cell #5)**, raw pos 11 — `BUILD OE9 — RICH OPENEAREABLE-ONLY FEATURES` → `interaction_oe9_10s.csv`.
  - **CELL 11 (code cell #6)**, raw pos 12 — `EVALUATE OE9` — full OE9-only model search (7 feature subsets × 6 k-values × 24 model configs) → `oe9_openearable_only_model_search_summary.csv` / `_predictions.csv`.
  - **CELL 12 (code cell #7)**, raw pos 13 — `FAST FOCUSED OE9 SEARCH` — smaller focused re-run (6 feature subsets × 4 k-values × 7 models) → **`oe9_focused_search_summary.csv` / `oe9_focused_search_predictions.csv`**.
  - **CELL 13 (code cell #8)**, raw pos 14 — `MAGNETOMETER DIAGNOSTIC FOR OPENEAREABLE FILES` — diagnostic scan for mag/heading/compass columns per group; no CSV output.
- (raw pos 15, MD, no code-only cell) "INCLUDING MAGNEOMETER"
  - **CELL 15 (code cell #9)**, raw pos 16 — `BUILD OE10 — ADD MAGNETOMETER FEATURES TO EXISTING OE9` → `interaction_oe10_10s.csv`.
  - **CELL 16 (code cell #10)**, raw pos 17 — `EVALUATE OE10 — DOES MAGNETOMETER HELP?` (6 feature subsets × 5 k-values × 6 models) → **`oe10_mag_focused_search_summary.csv` / `oe10_mag_focused_search_predictions.csv`**.
  - **CELL 17 (code cell #11)**, raw pos 18 — `MISSING OE10 TESTS — MOTION + MAGNETOMETER COMBINATIONS` (13 feature subsets incl. "OE9 motion + MAG magnitude" × 5 k-values × 7 models) → **`oe10_missing_motion_mag_search_summary.csv` / `oe10_missing_motion_mag_search_predictions.csv`** — **this is the search cell that produces Table 7.9 row 1.**
- (raw pos 19, MD, no code-only cell) "OPTITRACK + OPENEEARABLE BEST MODEL"
  - **CELL 19 (code cell #12)**, raw pos 20 — `CLEAN TEST — BEST OE MODEL + MANUAL OPTITRACK FEATURES ONLY` → **`oe_best_plus_manual_optitrack_summary.csv` / `_predictions.csv` / `_used_features.txt`** — **this is the hand-curated-list cell, produces Table 7.9 row 2.**
  - **CELL 20 (code cell #13)**, raw pos 21 — `CLEAN REPRODUCTION — ENG7 PROXIMITY + BEST OE MODEL` → **`eng7_proximity_exact_plus_oebest_summary.csv` / `_predictions.csv` / `_used_features.txt`** — **produces Table 7.9 row 3.**
- (raw pos 22, MD, no code-only cell) "BEST OE + PROXIMITY + ELAPSED TIME"
  - **CELL 22 (code cell #14)**, raw pos 23 — `BEST MODEL + ENG7 PROXIMITY + NON-NORMALIZED ELAPSED TIME` → **`oebest_proximity_elapsed_time_summary.csv` / `_predictions.csv` / `_used_features.txt`** — **produces Table 7.9 row 4 (best OE-contextual config, verified below).**
  - **CELL 23 (code cell #15)**, raw pos 24 — `EXPLAINABILITY FOR BEST MODEL` — LOGO held-out permutation importance for the best config, writes plots/CSVs under `INTERACTION_OE10/explainability_best_model/`.
  - **CELL 24 (code cell #16)**, raw pos 25 — `CHECK EXACT OPTITRACK / PROXIMITY FEATURES USED FROM ENG7` — re-prints/verifies the 3 `opti_*` proximity columns.
  - **CELL 25 (code cell #17)**, raw pos 26 — `PRINT + SAVE EXACT FEATURE LISTS USED IN FINAL MODELS` — re-derives and saves the 3 winning OE-best feature sets (programmatically, not by hand) plus a combined file, under `INTERACTION_OE10/final_feature_lists/`; also prints the reference scores `accuracy=0.685 | macro-F1=0.640 | balanced accuracy=0.664` for "OE best + ENG7 proximity + elapsed time", which match Table 7.9 row 4 (0.6850/0.6400/0.6640) closely enough to confirm this notebook is the source.
  - **CELL 26 (code cell #18)**, raw pos 27 — `OPTITRACK RAW DATA INSPECTION` — diagnostic column/dtype dump of raw OptiTrack files; no CSV output.
- (raw pos 28, MD, no code-only cell) "IMPROVED OPTITRACK FEATURES"
  - **CELL 28 (code cell #19)**, raw pos 29 — `OPTI2 — RICH OPTITRACK FEATURE GENERATION, ROBUST VERSION` (10s, source-aware) → `interaction_opti2_10s.csv` + `opti2_source_selection_report.csv`.
  - **CELL 29 (code cell #20)**, raw pos 30 — `EVALUATE OPTI2 RICH FEATURES WITH CURRENT BEST OE MODEL` — 3-class ablation comparing OPTI2-rich vs. the OE-best config; not one of the 8 target file stems.
- (raw pos 31, MD, no code-only cell) "XSXENS"
  - **CELL 31 (code cell #21)**, raw pos 32 — `XSENS RAW DATA INSPECTION` — diagnostic.
  - **CELL 32 (code cell #22)**, raw pos 33 — `XSENS2 — RICH XSENS FEATURE GENERATION` (10s) → `interaction_xsens2_10s.csv` + `xsens2_source_selection_report.csv`.
  - **CELL 33 (code cell #23)**, raw pos 34 — `EVALUATE XSENS2 RICH FEATURES` — 3-class ablation.
- (raw pos 35, MD, no code-only cell) "TEST FOR THE EFFECT OF TIME AND COMPARİSON WİTH DEEP LEARNING MODELS"
  - **CELL 35 (code cell #24)**, raw pos 36 — `PART 1 — CLEAN CLASSICAL ABLATION` (10s, 3-class, elapsed vs. no elapsed, 9 sensor-combo feature sets) → `INTERACTION_ABLATIONS/classical_elapsed_ablation_summary.csv` / `_predictions.csv`.
  - **CELL 36 (code cell #25)**, raw pos 37 — `FAST EVALUATION ONLY — 5s BINARY ADVANCED FEATURES` — reads the pre-existing (generic-OE) `binary_5s_all_sensor_advanced_features.csv`, 2 models × k=200 → `binary_5s_FAST_2models_k200_*` (4 files). Precursor to the SPECIAL_OE work, still on the *old* generic `oe__` features.
  - **CELL 37 (code cell #26)**, raw pos 38 — `FIXED FOCUSED CELL — IMPROVE OPTI2 AND TEST ADDING OE` — same old 5s binary base file, `K_LIST=[100,200,300]` → `binary_5s_FIXED_FOCUSED_opti_oe_*` (3 files) + `binary_5s_FIXED_FOCUSED_oe_add_effect.csv`. Still pre-SPECIAL_OE.
  - **CELL 38 (code cell #27)**, raw pos 39 — ad hoc keyword scan of the old generic `oe__` columns for head/movement/frequency-sounding names (diagnostic only, motivates CELL 39's rebuild).
  - **CELL 39 (code cell #28)**, raw pos 40 — `SPECIALIZED OE9/OE10-STYLE FEATURES FOR 5s BINARY TASK` — **the Table 7.10 cell** (see dedicated section below). Writes `binary_5s_specialized_oe9_oe10_features.csv`, `binary_5s_specialized_oe_merged_all_features.csv`, `binary_5s_specialized_oe_summary.csv`, `_predictions.csv`, `_best_per_condition.csv`, `_effect.csv`.
  - **CELL 40 (code cell #29)**, raw pos 41 — `EXPLAINABILITY FOR BEST BINARY CLASSICAL MODELS` — LOGO permutation importance for `OPTI2_RELATIVE_ONLY + elapsed, LinearSVC, k=120` and other top configs → `EXPLAINABILITY/explainability_model_metrics.csv` + per-config feature-effect CSVs.
  - **CELL 41 (code cell #30)**, raw pos 42 — `VISUALIZE FEATURE EFFECTS FROM EXPLAINABILITY OUTPUTS` — plots only.
  - **CELL 42 (code cell #31)**, raw pos 43 — `BREAK DOWN "OpenEarable | other" INTO EXACT FEATURE NAMES + BETTER GROUPS` — re-labels the "other" family in the CELL 40 output with finer categories.
  - **CELL 43 (code cell #32)**, raw pos 44 — `SEQUENTIAL DEEP MODELS FOR 5s BINARY INTERACTION DETECTION` (LSTM/Transformer, no elapsed_min) → `DEEP_SEQUENCE_MODELS_NO_ELAPSED/deep_sequence_no_elapsed_*` (3 files). Reads `binary_5s_specialized_oe_merged_all_features.csv`.
  - **CELL 44 (code cell #33)**, raw pos 45 — `FOCUSED FINE-TUNING FOR SEQUENTIAL DL MODELS, NO ELAPSED` → `DEEP_SEQUENCE_FINE_TUNING_NO_ELAPSED/fine_tuned_deep_sequence_*` (3 files).
- (raw pos 46, MD, no code-only cell) "DL OF RECOGNITION TASK"
  - **CELL 46 (code cell #34)**, raw pos 47 — `FIXED: CREATE ADVANCED 3-CLASS MERGED DATASET ROBUSTLY` → `INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv` (same target filename/role as in `master_feature_generator_task1_task2_task3_CORRECTED_V4`, built here from XSENS2 + OE10 + OPTI2 with a different, "robust" merge implementation).
  - **CELL 47 (code cell #35)**, raw pos 48 — `SAFE FULL CELL: 3-CLASS ACTIVITY RECOGNITION WITH SEQUENTIAL DEEP MODELS USING CORRECT ADVANCED FEATURES` → `ACTIVITY_3CLASS_DEEP_SEQUENCE_ADVANCED_NO_ELAPSED/activity3_advanced_deep_sequence_*` (4 files).
  - **CELL 48 (code cell #36)**, raw pos 49 — `FULL STANDALONE MULTIMODAL 3-CLASS DL CELL` (OE only / XSENS only / OPTI2 only / combinations) → `ACTIVITY_3CLASS_DEEP_SEQUENCE_ADVANCED_MULTIMODAL_NO_ELAPSED/activity3_multimodal_deep_sequence_*` (5 files).

None of CELLs 40-48 write any of the 13 target file stems from the task brief; they are downstream explainability/DL work built on top of `binary_5s_specialized_oe_merged_all_features.csv` (CELL 39's output) and the separate Task-2 (10s, 3-class) `activity3_advanced_merged_10s_features.csv` pipeline.

## Table 7.9 / 7.10 source mapping

All 4 rows of Table 7.9 and both feature-set families of Table 7.10 are produced by this single notebook. Two of the 13 requested stems (`oe_best_plus_optitrack_summary.csv`, `oe_best_plus_proximity_speed_summary.csv`) do **not** appear anywhere in this file — see note at the end of this section.

### Shared building block: "OE best" = "OE9 motion + MAG magnitude"

Re-derived identically in CELLs 17, 19, 20, 22, 23 and 25 from `interaction_oe10_10s.csv` (`old_ear` = `ear_*` columns, `oe9_features` = `oe_*` columns, `mag_features` = `mag_*` columns):

```python
motion = [c for c in old_ear + oe9_features
          if ("acc_" in c or "gyro_" in c or "jerk" in c or "turn" in c)]

mag_magnitude = [c for c in mag_features
                  if ("magnitude" in c or "horizontal" in c or "mag_active" in c)]

oe_best = unique_feats(motion + mag_magnitude)   # = "OE9 motion + MAG magnitude", 305 features per Table 7.9
```
`unique_feats` = `list(dict.fromkeys(feats))` (order-preserving de-dup), defined identically in every cell that needs it.

### 1. `oe9_focused_search_summary.csv` / `_predictions.csv`

- **(a) Cell:** CELL 12 (code cell #7), raw pos 13, lines 1515-1890.
- **(b) Feature-set construction:** built from `interaction_oe9_10s.csv` (`old_ear` = `ear_*`, `new_oe` = `oe_*`). Six named subsets tested: `"OE old ear_ only"` = `old_ear`; `"OE9 posture"` = cols containing `"pitch"`/`"roll"`/`"down"`/starting `ear_head_down`/`ear_head_pitch`; `"OE9 motion"` = cols containing `"acc_"`/`"gyro_"`/`"jerk"`/`"turn"`; `"OE9 synchrony/asymmetry"` = cols containing `"pair_"`/`"active_count"`/`"dominance"`/`"asymmetry"`/`"alternation"`; `"OE9 spectral"` = cols containing `"entropy"`/`"band"`; `"OE9 all"` = `old_ear + new_oe`. This is the OE9-only precursor of the "OE9 motion" family — it does **not** yet include the magnetometer (OE10 not built until CELL 15).
- **(c) Model/grid:** `selection_ks = [None, 10, 20, 40]` × 7 models: `logreg_C1` (C=1.0), `linearSVC_C1` (C=1.0), `rbfSVC_C1_gscale` (C=1.0, gamma="scale"), `rbfSVC_C1_g0.1`, `rbfSVC_C3_g0.03`, `rf_leaf4`, `extraTrees_leaf1`. Pipeline: `SimpleImputer(median) → RobustScaler → [SelectKBest(f_classif,k)] → model`, LOGO CV. Includes the exact RBF-SVC C=1 gamma=scale config used for Table 7.9, but on the OE9-only (no-magnetometer) feature universe.
- **(d) Output paths:** `{OUT_DIR}/oe9_focused_search_summary.csv`, `{OUT_DIR}/oe9_focused_search_predictions.csv`, where `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE9"`.
- **(e) Input materialization:** reads `OE9_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE9/interaction_oe9_10s.csv"` (`oe9 = pd.read_csv(OE9_PATH)`) — this is a **downstream feature file** produced earlier in this same notebook (CELL 10). Also reads `REC_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"` (`rec = pd.read_csv(REC_PATH)[...]`) — an **already-materialized** label table from the companion `master_feature_generator` notebook (CELL 14 there).
- **(f) Hardcoded Drive paths:** `OE9_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE9/interaction_oe9_10s.csv"`; `REC_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"`; `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE9"`.

### 2. `oe10_mag_focused_search_summary.csv` / `_predictions.csv`

- **(a) Cell:** CELL 16 (code cell #10), raw pos 17, lines 2623-3002.
- **(b) Feature-set construction:** from `interaction_oe10_10s.csv`. `mag_features` = `mag_*` cols; `mag_heading` = `mag_*` cols containing `"heading"`; `mag_magnitude` = `mag_*` cols containing `"magnitude"`/`"horizontal"`/`"mag_active"`; `mag_pair` = `mag_*` cols containing `"pair"`; `posture` = (`old_ear`+`oe9_features`) cols matching the posture rule above. Six sets: `"MAG only"`, `"MAG heading only"`, `"MAG magnitude only"`, `"MAG pair only"`, `"OE9 posture + MAG"` (=`posture+mag_features`), `"OE10 all acc+gyro+mag"` (=`old_ear+oe9_features+mag_features`). None of these is literally "OE9 motion + MAG magnitude" yet — this cell is exploring whether magnetometer alone helps, not the final fusion.
- **(c) Model/grid:** `selection_ks = [None, 10, 20, 40, 80]` × 6 models: `logreg_C1`, `linearSVC_C1`, `rbfSVC_C1_gscale` (C=1, gamma=scale), `rbfSVC_C3_g0.03`, `rf_leaf4`, `extraTrees_leaf1`. Same pipeline shape as above.
- **(d) Output paths:** `{OUT_DIR}/oe10_mag_focused_search_summary.csv`, `{OUT_DIR}/oe10_mag_focused_search_predictions.csv`, `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **(e) Input materialization:** `OE10_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10/interaction_oe10_10s.csv"` (`oe10 = pd.read_csv(OE10_PATH).copy()`) — downstream feature file produced by CELL 15 of this same notebook. `REC_PATH` as above.
- **(f) Hardcoded paths:** `OE10_PATH`, `REC_PATH` (as above), `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.

### 3. `oe10_missing_motion_mag_search_summary.csv` / `_predictions.csv` — **source of Table 7.9 row 1**

- **(a) Cell:** CELL 17 (code cell #11), raw pos 18, lines 3004-3509.
- **(b) Feature-set construction:** 13 named sets built from `interaction_oe10_10s.csv`, including the exact target row:
  ```python
  motion = [c for c in old_ear + oe9_features
            if ("acc_" in c or "gyro_" in c or "jerk" in c or "turn" in c)]
  mag_magnitude = [c for c in mag_features
                    if ("magnitude" in c or "horizontal" in c or "mag_active" in c)]
  feature_sets = {
      "OE9 motion only": unique_feats(motion),
      "OE9 all acc+gyro": unique_feats(old_ear + oe9_features),
      "MAG only": unique_feats(mag_features),
      "MAG heading only": unique_feats(mag_heading),
      "MAG magnitude only": unique_feats(mag_magnitude),
      "MAG pair only": unique_feats(mag_pair),
      "OE9 motion + MAG": unique_feats(motion + mag_features),
      "OE9 motion + MAG heading": unique_feats(motion + mag_heading),
      "OE9 motion + MAG magnitude": unique_feats(motion + mag_magnitude),   # <-- Table 7.9 row 1
      "OE9 motion + MAG pair": unique_feats(motion + mag_pair),
      "OE9 motion+posture + MAG": unique_feats(motion + posture + mag_features),
      "OE9 motion+sync + MAG": unique_feats(motion + sync + mag_features),
      "OE9 motion+spectral + MAG": unique_feats(motion + spectral + mag_features),
      "OE10 all acc+gyro+mag": unique_feats(old_ear + oe9_features + mag_features),
  }
  ```
- **(c) Model/grid:** `selection_ks = [None, 10, 20, 40, 80]` × 7 models: `logreg_C1`, `linearSVC_C1`, **`rbfSVC_C1_gscale`** (C=1.0, gamma="scale" — the Table 7.9 model), `rbfSVC_C1_g0.1`, `rbfSVC_C3_g0.03`, `rf_leaf4`, `extraTrees_leaf1`. Same `SimpleImputer→RobustScaler→[SelectKBest]→model` LOGO pipeline. `clone(model)` used per fold. The row "OE9 motion + MAG magnitude" × `rbfSVC_C1_gscale` × selection `"all"` (no `SelectKBest`, since Table 7.9 reports "305" = the full feature count for this set with no k restriction) is Table 7.9 row 1.
- **(d) Output paths:** `{OUT_DIR}/oe10_missing_motion_mag_search_summary.csv`, `{OUT_DIR}/oe10_missing_motion_mag_search_predictions.csv`, `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **(e) Input materialization:** `OE10_PATH` (downstream, CELL 15 output), `REC_PATH` (already-materialized, from `master_feature_generator`).
- **(f) Hardcoded paths:** `OE10_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10/interaction_oe10_10s.csv"`; `REC_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"`; `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.

### 4. `oe_best_plus_manual_optitrack_summary.csv` / `_predictions.csv` / `_used_features.txt` — **source of Table 7.9 row 2, and the notebook's one manually-curated feature list**

- **(a) Cell:** CELL 19 (code cell #12), raw pos 20, lines 3511-3930.
- **(b) Feature-set construction — this is the human-curation point.** The cell's own header comment says explicitly: *"Best OE-only feature set: OE9 motion + MAG magnitude ... This cell tests ONLY: 1) OE best only, 2) OE best + manually selected OptiTrack/spatial features. No auto-detection."* The literal, hand-picked column list is:
  ```python
  # STRICT version: distance + speed + centroid movement only
  # I removed: gyrE_min, gyrE_mid, gyrE_max, hand_freq_max
  # because those names are ambiguous and may not be pure OptiTrack.
  OPTITRACK_FEATURES = [
      "dist_close_mean",
      "dist_close_min",
      "dist_mid_mean",
      "dist_far_mean",
      "dist_disp_mean",
      "dist_disp_std",
      "speed_min",
      "speed_mid",
      "speed_max",
      "centroid_speed",
  ]
  ```
  These 10 names are **not** OPTI2 columns — they are pulled from `REC_PATH` (`eng3_recognition_3class_core_features.csv`), i.e. the legacy ENG3 5s "`old_eng_features_and_tensor`" proximity/speed columns (sorted pairwise distances close/mid/far, per-participant speed min/mid/max, centroid speed) documented in `master_feature_generator_task1_task2_task3_CORRECTED_V4_ANALYSIS.md` CELL 11, averaged per 10s ENG7/OE10 window (`sub[c].mean()` over the ENG3 5s sub-windows falling inside it) and prefixed `opti_` on merge (`out[f"opti_{c}"] = ...mean()`). `feature_sets["OE best + manual OptiTrack"] = unique_feats(oe_best_features + opti_features)` where `oe_best_features` is the "OE9 motion + MAG magnitude" 305-feature set (see shared block above) and `opti_features = [f"opti_{c}" for c in available_opti]` (10 cols) → 305 + 10 = **315**, exactly matching Table 7.9 row 2's reported feature count.
- **(c) Model:** single fixed model, no grid — `SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)`, i.e. the exact RBF-SVC C=1 gamma=scale of Table 7.9. Pipeline: `SimpleImputer(median) → RobustScaler → SVC`, LOGO CV, no `SelectKBest` (both rows use the full feature set, consistent with Table 7.9 not reporting a `k`/selection column for these two rows).
- **(d) Output paths:** `{OUT_DIR}/oe_best_plus_manual_optitrack_summary.csv`, `_predictions.csv`, `_used_features.txt`, `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **(e) Input materialization:** `OE10_PATH` (downstream, this notebook's CELL 15), `REC_PATH` (already-materialized, from `master_feature_generator`).
- **(f) Hardcoded paths:** `OE10_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10/interaction_oe10_10s.csv"`; `REC_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"`; `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.

### 5. `eng7_proximity_exact_plus_oebest_summary.csv` / `_predictions.csv` / `_used_features.txt` — **source of Table 7.9 row 3**

- **(a) Cell:** CELL 20 (code cell #13), raw pos 21, lines 3934-4429.
- **(b) Feature-set construction:** "ENG7 proximity" is defined, per the cell's own comment, "EXACTLY like your old code": `eng7_opti = [c for c in eng7.columns if c.startswith("opti_")]` — i.e. the **3** columns computed directly by the ENG7 builder itself (CELL 6, lines 234-239: `opti_nearest_pair_dist_mean`, `opti_all_pairs_dist_mean`, `opti_all_pairs_dist_std`), **not** the same `opti_*` columns as in item 4 above (those came from the ENG3 legacy table and were prefixed `opti_` on merge; these come natively from ENG7's own proximity baseline block). `oe_best` = the same "OE9 motion + MAG magnitude" 305-feature set, renamed `oebest__<col>` on merge to avoid name collisions, then re-merged onto the ENG7 table via a robust rounded-timestamp join (tries `decimals` 6→1, keeps whichever rounding matches the most rows — `add_merge_keys`/best-of-6-roundings pattern reused in CELLs 22/23/25). `feature_sets["OE best + ENG7 proximity"] = unique_feats(oe_best_renamed + eng7_opti)` → 305 + 3 = **308**, matching Table 7.9 row 3.
- **(c) Model:** three models compared — `RF_old_setup` (RandomForest, n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0), `LogReg_old_setup` (max_iter=4000, class_weight="balanced", random_state=0), and **`RBF_best_OE_setup`** = `SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)` — the Table 7.9 row's model. No `SelectKBest`.
- **(d) Output paths:** `{OUT_DIR}/eng7_proximity_exact_plus_oebest_summary.csv`, `_predictions.csv`, `_used_features.txt`, `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **(e) Input materialization:** `ENG7_PATH = ".../INTERACTION_ENG7/interaction_eng7_10s.csv"` (downstream, this notebook's CELL 6), `OE10_PATH` (downstream, CELL 15), `REC_PATH` (already-materialized).
- **(f) Hardcoded paths:** `ENG7_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG7/interaction_eng7_10s.csv"`; `OE10_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10/interaction_oe10_10s.csv"`; `REC_PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"`; `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.

### 6. `oebest_proximity_elapsed_time_summary.csv` / `_predictions.csv` / `_used_features.txt` — **source of Table 7.9 row 4 (best OE-contextual)**

- **(a) Cell:** CELL 22 (code cell #14), raw pos 23, lines 4432-4908.
- **(b) Feature-set construction:** identical `oe_best_renamed` (305) + `eng7_opti` (3, native ENG7 `opti_*` columns) as item 5, plus a non-normalized elapsed-time feature computed in this cell:
  ```python
  eng7["window_mid"] = (eng7["window_start"] + eng7["window_end"]) / 2.0
  group_start = eng7.groupby("group")["window_mid"].transform("min")
  eng7["elapsed_min"] = (eng7["window_mid"] - group_start) / 60.0
  ```
  (minutes since that group's first ENG7 window; explicitly **not** normalized to [0,1] and does **not** use the session end time — the cell's banner comment stresses this). Three feature sets: `"OE best + ENG7 proximity"` (308), `"Elapsed time only"` (`["elapsed_min"]`), `"OE best + ENG7 proximity + elapsed time"` = `unique_feats(oe_best_renamed + eng7_opti + ["elapsed_min"])` → 308 + 1 = **309**, matching Table 7.9 row 4.
- **(c) Model:** single fixed model — `SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)`. No `SelectKBest`.
- **(d) Output paths:** `{OUT_DIR}/oebest_proximity_elapsed_time_summary.csv`, `_predictions.csv`, `_used_features.txt`, `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **(e) Input materialization:** `ENG7_PATH`, `OE10_PATH` (both downstream, this notebook's CELLs 6 and 15), `REC_PATH` (already-materialized).
- **(f) Hardcoded paths:** same three as item 5, plus `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_OE10"`.
- **Corroboration:** CELL 25 (raw pos 26), which re-derives this exact "OE best + ENG7 proximity + elapsed time" feature list for saving to disk, prints as a reminder: *"Expected score from previous run: accuracy = 0.685 | macro-F1 = 0.640 | balanced accuracy = 0.664"* — matching Table 7.9 row 4 (A=0.6850, M=0.6400, B=0.6640) essentially exactly. This is strong confirmation that this notebook (and specifically CELL 22) is the source of Table 7.9's headline row.

### 7-12. `binary_5s_specialized_oe_summary.csv` / `_predictions.csv` / `_best_per_condition.csv` / `_effect.csv`, `binary_5s_specialized_oe9_oe10_features.csv`, `binary_5s_specialized_oe_merged_all_features.csv` — **all six produced by one cell, the source of Table 7.10**

- **(a) Cell:** CELL 39 (code cell #28), raw pos 40, lines 11829-13189 (~1,360 lines — the largest single cell in the notebook).
- **(b) Feature-set construction:**
  - **Input grid/labels are inherited, not recomputed:** the cell loads `base_df = pd.read_csv(OLD_BINARY_PATH)` where `OLD_BINARY_PATH = ".../INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv"` — the 5s window grid, `group`, `binary_label`, pre-existing `opti2_*`/`xsens2_*` columns (16-stat `adv_stats` schema, from the companion `master_feature_generator_task1_task2_task3_CORRECTED_V4` notebook's Task-1 5s rebuilders) and, notably, an **already-present `elapsed_min` column** all come from this file.
  - **SPECIAL_OE features are freshly recomputed** (not read from `interaction_oe9_10s.csv`/`interaction_oe10_10s.csv`) — `extract_specialized_oe_features(oe, ws, we, sess)` is a near-verbatim merge of `extract_oe9_features` (CELL 10) and `extract_mag_features` (CELL 15), re-run directly on the raw `group_{g}_openearable_model_ready.csv` files but evaluated on the **5-second** windows taken from `base_df` instead of 10s windows. Written to `SPECIAL_OE_PATH = ".../binary_5s_specialized_oe9_oe10_features.csv"`.
  - **Merge:** `generic_oe_cols = [c for c in base_df.columns if c.startswith("oe__")]` (the old reconstructed generic-OE family) is **dropped**, then `special_oe_df` is left-merged on `["group","window_start","window_end"]` → `MERGED_SPECIAL_PATH = ".../binary_5s_specialized_oe_merged_all_features.csv"` (**the actual Table 7.10 model input**).
  - **Feature-set selectors (all programmatic prefix/token rules, no hand-picked list here):**
    ```python
    def is_special_oe(c):
        return c.startswith("ear_") or c.startswith("oe_") or c.startswith("mag_")

    def is_opti(c):
        return c.startswith("opti2_") or c.startswith("opti2__")

    def is_xsens(c):
        return c.startswith("xsens2__")

    def is_opti_relative_feature(c):
        if not is_opti(c): return False
        tokens = ["dist","spread","area","speed","active_speed",
                  "pair","nearest","farthest","triangle"]
        return any(tok in c.lower() for tok in tokens)

    special_oe_cols   = [c for c in df.columns if is_special_oe(c)]
    opti_all_cols     = [c for c in df.columns if is_opti(c)]
    opti_relative_cols= [c for c in opti_all_cols if is_opti_relative_feature(c)]
    xsens_cols        = [c for c in df.columns if is_xsens(c)]
    elapsed           = ["elapsed_min"] if "elapsed_min" in df.columns else []

    feature_sets = {
        "SPECIAL_OE": unique_feats(special_oe_cols),
        "OPTI2_RELATIVE_ONLY": unique_feats(opti_relative_cols),
        "SPECIAL_OE + OPTI2_RELATIVE_ONLY": unique_feats(special_oe_cols + opti_relative_cols),
        "SPECIAL_OE + OPTI2_RELATIVE_ONLY + XSENS2": unique_feats(special_oe_cols + opti_relative_cols + xsens_cols),
    }
    ```
    "`OPTI2_RELATIVE_ONLY`" = OPTI2 columns whose name contains a relative-geometry token (distance/spread/area/speed/pair/nearest/farthest/triangle), excluding absolute-position OPTI2 columns that don't match any token. `with_elapsed` variants append `elapsed_min` (`unique_feats(base_feats + elapsed)`).
- **(c) Model/grid:**
  ```python
  models = [
      ("logreg_C1",       LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=42)),
      ("linearSVC_C1",    LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, dual=False, random_state=42)),
      ("rbfSVC_C1_gscale",SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)),
  ]
  K_LIST = [40, 80, 120, 200]
  TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]
  ```
  Nested loop: `FEATURE_SETS_TO_RUN` (4) × `TIME_CONDITIONS` (2) × `K_LIST` (4) × `models` (3) — 96 LOGO runs. Per fold: `median_impute_train_test` (train-median imputation applied to both train/test) → `RobustScaler` (fit on train) → `SelectKBest(f_classif, k=min(k_requested, n_features-1))` → model, binary label = `binary_label`. This exactly reproduces Table 7.10's two model/k combinations: **LogReg (C=1), k=200** for the "SPECIAL_OE" rows and **LinearSVC (C=1), k=80** for the "SPECIAL_OE + OPTI2 relative" rows — both are literal entries in this grid (`logreg_C1` × `k_requested=200`, `linearSVC_C1` × `k_requested=80`), selected as the argmax over `best_per_condition` (grouped by `feature_set` × `time_condition`, sorted by macro-F1 then accuracy).
- **(d) Output paths:** `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_BINARY_5S_SPECIALIZED_OE"`:
  - `SPECIAL_OE_PATH = f"{OUT_DIR}/binary_5s_specialized_oe9_oe10_features.csv"`
  - `MERGED_SPECIAL_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_merged_all_features.csv"`
  - `SUMMARY_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_summary.csv"`
  - `PRED_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_predictions.csv"`
  - `BEST_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_best_per_condition.csv"`
  - `EFFECT_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_effect.csv"`
- **(e) Input materialization:** `OLD_BINARY_PATH = f"{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv"` (`base_df = pd.read_csv(OLD_BINARY_PATH)`, line 11944) is an **already-materialized downstream feature file** produced by the companion `master_feature_generator_task1_task2_task3_CORRECTED_V4` notebook (its CELL 8) — not raw sensor data. Raw OpenEarable sensor files (`group_{g}_openearable_model_ready.csv` under `INPUT_DIR = f"{DATA_ROOT}/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`) **are** read directly for the SPECIAL_OE rebuild itself (this part genuinely computes features from raw signals). The cell hard-fails with `FileNotFoundError` if `OLD_BINARY_PATH` doesn't already exist ("Run the earlier binary feature-generation cell first").
- **(f) Hardcoded paths:** `DATA_ROOT = "/content/drive/MyDrive/thesis/data"`; `INPUT_DIR = f"{DATA_ROOT}/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`; `OLD_BINARY_PATH = f"{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv"`; `OUT_DIR = f"{DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE"` (all 6 outputs above live here).

### 13. `interaction_oe9_10s.csv` and `interaction_oe10_10s.csv` — upstream feature files, both produced by this notebook

- **`interaction_oe9_10s.csv`**: CELL 10 (code cell #5), raw pos 11, lines 389-1038. `extract_oe9_features` computes rich per-person OpenEarable acc/gyro features (energy/entropy/band-ratio/pitch/roll/jerk/active-count/dominance/cross-person-correlation, all self-normalised by per-participant session baselines) on 10s/10s-stride windows built directly from `group_{g}_openearable_model_ready.csv` under `INPUT_DIR = "/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`. This **is** computed from raw sensor data (no upstream feature-file dependency other than raw per-group CSVs). Output: `OUT_PATH = f"{OUT_DIR}/interaction_oe9_{int(WINDOW_S)}s.csv"` = `.../INTERACTION_OE9/interaction_oe9_10s.csv`.
- **`interaction_oe10_10s.csv`**: CELL 15 (code cell #9), raw pos 16, lines 2050-2621. Reads `OE9_PATH` (this notebook's own CELL 10 output — a downstream file), recomputes magnetometer-only features (`extract_mag_features`) directly from the same raw `group_{g}_openearable_model_ready.csv` files, and left-merges the new `mag_*` columns onto the OE9 table on `["group","window_start","window_end"]`. Output: `OUT_PATH = f"{OUT_DIR}/interaction_oe10_10s.csv"` = `.../INTERACTION_OE10/interaction_oe10_10s.csv`.

### Two requested stems NOT found in this notebook

`oe_best_plus_optitrack_summary.csv` and `oe_best_plus_proximity_speed_summary.csv` do **not** occur anywhere in the 19,377-line code-only extract (checked by exact substring search for `proximity_speed`, `plus_optitrack`, and `oe_best_plus`). Only `oe_best_plus_manual_optitrack_*` (item 4 above) exists. These two filenames most likely belong to a different, earlier exploratory notebook/session not covered by this file (possibly an intermediate iteration before the "manual"/"exact ENG7" cells replaced it) — worth re-checking the Drive inventory for a sibling notebook if these two specific CSVs are needed.

## Hardcoded file paths found in code

### Inputs (raw sensor / already-materialized upstream)

- `/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED` — raw per-group `group_{g}_openearable_model_ready.csv` / `_optitrack_model_ready.csv` / `_xsens_model_ready.csv` (used by CELLs 6, 10, 15, 28, 32, 39).
- `/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES` — fallback OptiTrack/Xsens source (CELL 28's `OPTI_DIR_OLD`).
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv` — already-materialized 3-class recognition label/feature table (from `master_feature_generator...CORRECTED_V4`), read by nearly every 3-class-recognition cell (`REC_PATH`, CELLs 8, 11, 12, 16, 17, 19, 20, 22, 23, 24, 25, 29, 33, 43, 44, 46-48).
- `/content/drive/MyDrive/thesis/data/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv` — already-materialized 5s binary base table (`OLD_BINARY_PATH`/`MERGED_FEATURES_PATH`, from `master_feature_generator...CORRECTED_V4`), read by CELLs 36, 37, 39.

### Outputs, by feature family

- `INTERACTION_ENG7/interaction_eng7_10s.csv` (CELL 6).
- `INTERACTION_OE9/interaction_oe9_10s.csv` (CELL 10); `oe9_openearable_only_model_search_summary.csv` / `_predictions.csv` (CELL 11); **`oe9_focused_search_summary.csv` / `_predictions.csv`** (CELL 12).
- `INTERACTION_OE10/interaction_oe10_10s.csv` (CELL 15); **`oe10_mag_focused_search_summary.csv` / `_predictions.csv`** (CELL 16); **`oe10_missing_motion_mag_search_summary.csv` / `_predictions.csv`** (CELL 17); **`oe_best_plus_manual_optitrack_summary.csv` / `_predictions.csv` / `_used_features.txt`** (CELL 19); **`eng7_proximity_exact_plus_oebest_summary.csv` / `_predictions.csv` / `_used_features.txt`** (CELL 20); **`oebest_proximity_elapsed_time_summary.csv` / `_predictions.csv` / `_used_features.txt`** (CELL 22); `INTERACTION_OE10/explainability_best_model/*` (CELL 23); `INTERACTION_OE10/final_feature_lists/*_features.csv`, `*_features.txt`, `combined_final_feature_lists.{csv,txt}` (CELL 25).
- `INTERACTION_OPTI2/interaction_opti2_10s.csv` + `opti2_source_selection_report.csv` (CELL 28).
- `INTERACTION_XSENS2/interaction_xsens2_10s.csv` + `xsens2_source_selection_report.csv` (CELL 32).
- `INTERACTION_ABLATIONS/classical_elapsed_ablation_summary.csv` / `_predictions.csv` (CELL 35); `INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv` (CELL 46).
- `INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_FAST_2models_k200_{summary,predictions,best_per_condition,elapsed_effect}.csv` (CELL 36).
- `INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_FIXED_FOCUSED_opti_oe_{summary,predictions,best_per_condition}.csv`, `binary_5s_FIXED_FOCUSED_oe_add_effect.csv` (CELL 37).
- `INTERACTION_BINARY_5S_SPECIALIZED_OE/` — **`binary_5s_specialized_oe9_oe10_features.csv`, `binary_5s_specialized_oe_merged_all_features.csv`, `binary_5s_specialized_oe_summary.csv`, `_predictions.csv`, `_best_per_condition.csv`, `_effect.csv`** (all CELL 39); `EXPLAINABILITY/explainability_model_metrics.csv` + per-config feature-effect CSVs (CELL 40); `EXPLAINABILITY/BEST_OE_ONLY__SPECIAL_OE_WITH_ELAPSED_feature_importance.csv`, `OE_ONLY_exact_features_previously_called_other.csv`, `OE_ONLY_breakdown_of_previous_other_family.csv`, `OE_ONLY_detailed_feature_family_summary.csv`, `OE_ONLY_top40_exact_features_with_detailed_family.csv` (CELL 42); `DEEP_SEQUENCE_MODELS_NO_ELAPSED/deep_sequence_no_elapsed_{summary,predictions,fold_metrics}.csv` (CELL 43); `DEEP_SEQUENCE_FINE_TUNING_NO_ELAPSED/fine_tuned_deep_sequence_{summary,fold_metrics,predictions}.csv` (CELL 44).
- `ACTIVITY_3CLASS_DEEP_SEQUENCE_ADVANCED_NO_ELAPSED/activity3_advanced_deep_sequence_{summary,fold_metrics,predictions,aggregate_by_setting}.csv` (CELL 47).
- `ACTIVITY_3CLASS_DEEP_SEQUENCE_ADVANCED_MULTIMODAL_NO_ELAPSED/activity3_multimodal_deep_sequence_{summary,fold_metrics,predictions,aggregate_by_setting}.csv`, `activity3_multimodal_best_per_feature_set.csv` (CELL 48).

All paths are rooted at `/content/drive/MyDrive/thesis/data` (Colab Drive mount); no non-Drive absolute paths appear anywhere in the file.

## Function / class definitions

Grouped by cell family (full per-line mapping available on request — 296 `def`/`class` statements total, driven by this notebook's pattern of **re-defining** near-identical helper sets in almost every cell rather than importing shared utilities):

- **CELL 6 (ENG7 builder):** `discover`, `load`, `xoff` (Xsens/OE time-offset search via label agreement), `sinterp`, `band_ratio`, `spectral_entropy`, `active_mask`, `event_count`, `burstiness`, `features` (per-window, 4 feature families), `session_baseline`, `build`.
- **CELL 8:** `ev` (LOGO evaluator for the 9 ENG7 feature-subset combos).
- **CELL 10 (OE9 builder), CELL 39 (SPECIAL_OE rebuild — near-duplicate):** `discover_openearable`, `load_oe`, `sinterp`, `safe_mean/std/min/max/range/iqr/percentile`, `mad_diff`, `event_count`, `spectral_entropy`, `band_ratio`, `corr_safe`, `max_lag_corr`, `aggregate_values`, `session_baseline`, `extract_oe9_features` (CELL 10) / `extract_specialized_oe_features` (CELL 39, same logic + inline magnetometer block + 4 extra "keyword bait" collect() calls: `head_movement_frequency`, `head_turn_frequency`, `head_nod_frequency_band`, `head_posture_switch_frequency`), `build_oe9` (CELL 10 only).
- **CELL 11:** `clean_feature_list`, `run_logo_eval` (full grid evaluator, 7×6 configs).
- **CELL 12, 16, 17, 19, 20, 22, 23, 25** (all the "focused search" / "clean test" / "clean reproduction" cells): each independently re-defines `unique_feats`, `clean_feature_list`, and (from CELL 20 onward) `add_recognition_labels`, `add_merge_keys`, `evaluate_logo`, `print_report` — byte-for-byte near-identical across cells (copy-paste pattern, not shared imports).
- **CELL 13:** `discover_openearable`, `find_possible_mag_cols`, `find_axis_triplets` (magnetometer diagnostic).
- **CELL 15 (OE10 builder):** `discover_openearable`, `load_mag`, `sinterp`, safe-stat family, `corr_safe`, `circular_diff`, `max_lag_corr`, `aggregate`, `mag_session_baseline`, `extract_mag_features`.
- **CELL 23:** `pretty_name`, `feature_group`, `plot_barh` (explainability plotting).
- **CELL 25:** `feature_source`, `feature_group`, `original_name`, `save_feature_list`.
- **CELL 28 (OPTI2 10s builder):** `get_num_col`, `safe_stats` (9-stat), `safe_fraction`, `row_nanmin/max/mean/std`, `centroid_from_stack`, `compute_speed`, `triangle_area_2d`, `add_recognition_labels`, `standardize_columns`, `find_landmark_mapping`, `find_available_col`, `load_candidate_file`, `coordinate_score_for_windows`, `choose_best_opti_source`, `compute_window_features` (main ~500-line feature builder) — near-identical to the OPTI2 builder documented in `master_feature_generator...CORRECTED_V4_ANALYSIS.md` CELL 18.
- **CELL 29:** `is_quality_feature`, `is_absolute_position_feature`, `opti2_group_name`, `evaluate_logo`, `print_report`.
- **CELL 31:** `column_category`, `inspect_file` (Xsens raw-file diagnostic).
- **CELL 32 (XSENS2 10s builder):** `get_num_col`, `safe_stats`, `safe_fraction`, `add_recognition_labels`, `clean_signal`, `unwrap_degrees`, `vector_norm`, `derivative_norm`, `derivative_1d`, `interp_nan`, `spectral_features`, `corr_feature`, `standardize_columns`, `candidate_paths_for_group`, `load_xsens_file`, `sanity_score_for_source`, `choose_best_xsens_source`, `compute_xsens_window_features`.
- **CELL 33, 35 (3-class ablations):** `is_xsens_quality_feature`, `is_xsens_raw_euler_posture`, `xsens_group_name`, `is_opti2_quality_feature`, `is_opti2_absolute_position_feature`, `evaluate_logo`, `print_best_report`, `merge_by_windows`.
- **CELL 36, 37 (5s binary, pre-SPECIAL_OE):** `evaluate_logo_fast` / `evaluate_logo_focused`, `is_oe_feature`, `is_opti_feature`, `is_xsens_feature`, `is_opti_relative_feature`, `selected_feature_source_counts`, `median_impute_train_test`, `get_best`, `print_best_report`.
- **CELL 39 (the Table 7.10 cell):** all of the CELL 10/15 helper set plus `is_special_oe`, `is_opti`, `is_xsens`, `is_opti_relative_feature`, `median_impute_train_test`, `selected_feature_source_counts`, `evaluate_logo`, `get_best`, `print_best_report`.
- **CELL 40:** `is_special_oe`, `is_opti`, `is_xsens`, `is_elapsed`, `is_opti_relative_feature`, `modality_of_feature`, `feature_family`, `median_impute_train_test`, `explain_logo_linear_model` (permutation-importance explainer).
- **CELL 41:** `short_feature_name`, `top_by_signed_effect`, `plot_signed_feature_effects`, `plot_family_importance`.
- **CELL 42:** `detailed_oe_family`, `short_name`.
- **CELL 43, 44 (LSTM/Transformer, 5s binary):** `unique_feats`, `is_special_oe`/`is_opti`/`is_xsens`/`is_opti_relative_feature`, `clean_feature_list`, `median_impute_train_test`, `make_sequences`, `choose_validation_group`, `set_seed`, `make_loader`, `predict_model`, `train_one_fold`, `evaluate_config_logo`/`evaluate_run`; classes `LSTMClassifier`/`RNNClassifier(nn.Module)`, `PositionalEncoding(nn.Module)`, `TransformerClassifier(nn.Module)`, `build_model`.
- **CELL 46 (advanced Task-2 merge):** `unique_feats`, `attach_recognition_label`, `add_merge_keys`, `merge_by_windows`.
- **CELL 47, 48 (Task-2 DL):** `unique_feats`, `add_merge_keys`, `merge_features_left_safe`, `is_xsens_quality_feature`, `is_xsens_raw_euler_posture`, `is_opti2_quality_feature`, `is_opti2_absolute_position_feature`, `create_safe_advanced_dataset` (CELL 47 only), `clean_feature_list`, `set_seed`, `median_impute_train_test`, `make_sequences`, `choose_validation_group`, `make_loader`, `predict_model`, `train_one_fold`, `evaluate_run`; classes `RNNClassifier(nn.Module)`, `PositionalEncoding(nn.Module)`, `TransformerClassifier(nn.Module)`, `build_model`.

## Hyperparameter-looking constants (verbatim)

- `WINDOW_S, STRIDE_S = 10.0, 10.0` — every 10s builder (ENG7, OE9, OE10 inherits it, OPTI2, XSENS2).
- `RESAMPLE_HZ = 25` — OE/OE9/OE10/SPECIAL_OE resampling grid.
- `NOD_BAND = (1.0, 3.0)`, `LOW_MOTION_BAND = (0.2, 1.0)`, `HIGH_MOTION_BAND = (3.0, 8.0)` — OE acc/gyro band ratios (CELLs 6, 10, 39).
- `MAG_LOW_BAND = (0.2, 1.0)`, `MAG_MID_BAND = (1.0, 3.0)`, `MAG_HIGH_BAND = (3.0, 8.0)` — magnetometer bands (CELL 39); CELL 15 re-declares the same trio as `LOW_BAND`/`MID_BAND`/`HIGH_BAND`.
- `down_thr = -0.35` (rad, ≈ -20°) — head-down pitch threshold, hardcoded inline in CELLs 6, 10, 39 (not a named module-level constant).
- `max_lag_steps = int(1.0 * fs)` — ±1s cross-person lag-correlation search (CELLs 10, 15, 39).
- `CORE = ["co_building", "co_merging", "conversation"]` — 3-class label set, redeclared in nearly every cell (18 occurrences).
- `GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]` — explicit 9-group list, group 4 excluded (CELLs 28, 32).
- **Table 7.9 grid (CELLs 11/12/16/17):** `selection_ks`/`K` variants `[None,10,20,40]` → `[None,10,20,40,80]`; models progressively narrowed from a 24-config grid (CELL 11: C∈{0.03,0.1,0.3,1.0,3.0} logreg, C∈{0.03,0.1,0.3,1.0} linearSVC, C∈{0.3,1,3}×gamma∈{scale,0.03,0.1} rbfSVC, leaf∈{1,2,4,8} RF/ExtraTrees) down to the focused 6-7-model set (`logreg_C1`, `linearSVC_C1`, `rbfSVC_C1_gscale`, `rbfSVC_C1_g0.1`/`rbfSVC_C3_g0.03`, `rf_leaf4`, `extraTrees_leaf1`) used from CELL 12 onward.
- **Table 7.9 winning-row model:** `SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)` — fixed, no grid, in CELLs 19, 20, 22.
- **Table 7.10 grid (CELL 39):** `RANDOM_STATE = 42`; `K_LIST = [40, 80, 120, 200]`; `FEATURE_SETS_TO_RUN = ["SPECIAL_OE","OPTI2_RELATIVE_ONLY","SPECIAL_OE + OPTI2_RELATIVE_ONLY","SPECIAL_OE + OPTI2_RELATIVE_ONLY + XSENS2"]`; `TIME_CONDITIONS = ["no_elapsed","with_elapsed"]`; `models = [logreg_C1 (C=1.0, lbfgs), linearSVC_C1 (C=1.0, dual=False, max_iter=10000), rbfSVC_C1_gscale (C=1.0, gamma="scale")]`.
- `elapsed_min` construction (CELLs 22, 23, 25 — freshly computed from ENG7 timestamps): `elapsed_min = (window_mid - group.window_mid.min()) / 60.0`, explicitly NOT normalized and NOT using session end. CELL 39 instead assumes `elapsed_min` **already exists** as a column in the upstream `binary_5s_all_sensor_advanced_features.csv`.
- Manual feature list (CELL 19): `OPTITRACK_FEATURES` (10 names) — see Table 7.9/7.10 section above, quoted in full there.
- Merge-key rounding search: `for decimals in [6, 5, 4, 3, 2, 1]` — used identically in CELLs 20, 22, 23, 25, 46, 47 to fuzzy-join ENG7/OE10/OPTI2/XSENS2 tables on rounded `window_start`/`window_end`.
- DL hyperparameters (CELLs 43/44/47/48): `SEED(S) = 42` (or `[42, 7]`), `MAX_EPOCHS = 60-80`, `PATIENCE = 10-12`, `BATCH_SIZE = 64`.
- `FAST_MODE = True` (CELL 35) — "Set this True first for speed. Later set False for full search." — comment implies an intended-but-possibly-never-executed full (`FAST_MODE=False`) run.
- `K_LIST = [100, 200, 300]` (CELL 37) vs. `K_LIST = [40, 80, 120, 200]` (CELL 39) — two different, non-overlapping-except-at-200 `K_LIST`s for what are functionally similar 5s-binary searches; CELL 37 operates on the old generic-OE base file, CELL 39 (with the different K_LIST) is the one that matches Table 7.10.

## Potentially broken / disabled cells

- **Colab-required, not Colab-optional.** Unlike `task1_full_comparison_classical_elapsed_dl_with_std.ipynb` and `master_feature_generator...CORRECTED_V4.ipynb` (both of which wrap `drive.mount` in `try/except`), CELL 1's `drive.mount('/content/drive')` here is a bare call with no guard, and CELL 2's remount fix only guards the `flush_and_unmount()` step, not the final `drive.mount(..., force_remount=True)`. Running outside Colab will raise immediately. A markdown cell right after (raw pos 4) even says "After running the cell above, please run the mount cell again," implying the mount sequence was flaky enough during authoring to need a documented manual retry — a porter should replace this whole two-cell dance with a plain local filesystem path.
- **Two abandoned/incompletely-adopted `OPTITRACK_FEATURES`-style stems appear in the task brief but not in this file**: `oe_best_plus_optitrack_summary.csv` and `oe_best_plus_proximity_speed_summary.csv` are absent from the 19,377-line extract entirely (see "Table 7.9/7.10 source mapping" above) — likely remnants of an earlier/different notebook version.
- **CELLs 36 and 37 are dead-end precursors, not the Table 7.10 source, despite operating on the same task.** Both read the *old* `binary_5s_all_sensor_advanced_features.csv` (generic `oe__` features, not SPECIAL_OE) and write their own summary/prediction files (`binary_5s_FAST_2models_k200_*`, `binary_5s_FIXED_FOCUSED_opti_oe_*`) that are never read again by any later cell — CELL 38's ad hoc keyword scan and CELL 39's full rebuild supersede them entirely. Not broken, but a porter should not mistake these two cells' outputs for Table 7.10 sources.
- **CELL 39 hard-fails if run standalone.** It raises `FileNotFoundError` if `OLD_BINARY_PATH` (`binary_5s_all_sensor_advanced_features.csv`) does not already exist — this notebook does not build that file itself; it must come from the companion `master_feature_generator_task1_task2_task3_CORRECTED_V4` notebook (or an equivalent upstream run) first. This is the single most important cross-notebook dependency for porting Table 7.10.
- **Heavy copy-paste duplication, not a bug but a maintenance/porting concern.** `unique_feats`, `clean_feature_list`, `add_recognition_labels`, `add_merge_keys`, `evaluate_logo`, `print_report`/`print_best_report`, and the various `safe_*`/`spectral_entropy`/`band_ratio`/`corr_safe`/`max_lag_corr` signal helpers are independently re-defined, near-identically, in 10+ cells rather than imported once — 296 total `def`/`class` statements in the file, the large majority duplicates of a much smaller set of true distinct implementations. A port to `src/features/` should consolidate these into shared modules (e.g. a `signal_stats.py`, a `logo_eval.py`) rather than porting cell-by-cell.
- **No `if False:` dead branches, no `!pip install` cells, and no `TODO`/`FIXME` markers found anywhere in the file** (checked by direct grep). All `raise ValueError`/`raise RuntimeError` calls found are defensive guards (missing columns, zero usable features/rows) rather than evidence of a broken pipeline.
- **`FAST_MODE = True` (CELL 35)** is commented "Set this True first for speed. Later set False for full search." with no evidence elsewhere in the file that it was ever actually re-run with `False` — worth flagging that the full (non-fast) classical ablation for the 10s 3-class task may never have been executed via this specific cell, though this does not affect Tables 7.9/7.10 (both of which come from other cells with no `FAST_MODE`-style shortcut).

## Summary

This notebook is a long, iterative **exploratory feature-engineering-and-model-search** session for the thesis's OpenEarable-focused and SPECIAL_OE experiments (Chapter 7, Section 7.5), not a clean single-pass pipeline: it builds several increasingly rich, self-normalised OpenEarable feature families from raw sensor data — `interaction_eng7_10s.csv` (CELL 6, 4 transfer-invariant feature families + a 3-column OptiTrack proximity baseline), `interaction_oe9_10s.csv` (CELL 10, rich acc/gyro-only), and `interaction_oe10_10s.csv` (CELL 15, OE9 + magnetometer) — then runs a long sequence of LOGO-cross-validated classical-ML "focused search" cells that incrementally narrow in on one winning feature/model combination: "OE9 motion" (acc/gyro/jerk/turn-named columns) fused with "MAG magnitude" (magnetometer magnitude/horizontal-strength/mag-active columns), scored with an RBF-SVC (C=1, gamma="scale"). This exact 305-feature "OE best" set is then extended three more times — with a **manually hand-picked** 10-column OptiTrack/proximity subset (`OPTITRACK_FEATURES`, quoted verbatim above, 315 features total — Table 7.9 row 2, the notebook's only human-curated feature list), with the **programmatically-derived** 3-column native ENG7 proximity baseline (308 features — Table 7.9 row 3), and with both proximity plus a freshly-computed non-normalized `elapsed_min` (309 features — Table 7.9 row 4, corroborated by CELL 25's printed reference scores 0.685/0.640/0.664 matching Table 7.9 almost exactly). Table 7.10's SPECIAL_OE binary-interaction work is entirely contained in one very large cell (CELL 39, ~1,360 lines): it reads an **already-materialized** upstream 5s binary feature table (`binary_5s_all_sensor_advanced_features.csv`, built by the companion `master_feature_generator_task1_task2_task3_CORRECTED_V4` notebook), drops its old generic-OE columns, rebuilds a rich SPECIAL_OE feature set directly from raw OpenEarable sensor files for the exact same 5s windows, and then runs a fully **programmatic** (prefix/substring-token, not hand-picked) 4-feature-set × 2-elapsed-condition × 4-k × 3-model LOGO grid search (`logreg_C1`, `linearSVC_C1`, `rbfSVC_C1_gscale`; `K_LIST=[40,80,120,200]`) that reproduces both Table 7.10 model/selection combinations (LogReg C=1 k=200; LinearSVC C=1 k=80) in a single pass, writing 6 output files including the two the thesis reports from (`binary_5s_specialized_oe_summary.csv`, `binary_5s_specialized_oe_merged_all_features.csv`). Two requested filename stems (`oe_best_plus_optitrack_summary.csv`, `oe_best_plus_proximity_speed_summary.csv`) do not exist in this notebook at all and likely belong elsewhere. For porting: (1) the **single most load-bearing artifact for faithful reproduction is the literal `OPTITRACK_FEATURES` list in CELL 19** — every other feature set in both tables is reconstructible from a small number of documented substring/prefix rules, but this one 10-name list cannot be regenerated from a rule and must be copied verbatim; (2) Table 7.9's winning model is always the same fixed `SVC(C=1.0, gamma="scale", kernel="rbf")` with no `SelectKBest` (full feature set every time), while Table 7.10 always goes through per-fold `SelectKBest(f_classif, k)` after `RobustScaler`; (3) CELL 39 cannot run standalone without the upstream `binary_5s_all_sensor_advanced_features.csv` already existing — this cross-notebook dependency must be preserved or the file supplied as a fixture in the reproduction repo.
