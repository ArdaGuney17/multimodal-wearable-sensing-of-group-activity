# Analysis: GAR_Complete_Selfcontained.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\GAR_Complete_Selfcontained.ipynb`

## Cell counts

Total cells: 28
Code cells: 14
Markdown cells: 14

## Import statements (deduplicated)

```python
from google.colab import drive
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from itertools import combinations
from scipy.signal import welch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gc
import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings
import os
import os, glob, re, gc
import os, glob, re, gc, warnings
import pandas as pd
import re
import subprocess, sys
import torch, torch.nn as nn
import warnings
from collections import Counter
```

Note: `welch` (from `scipy.signal`) is imported in cell 4 but never called — the notebook's own `spec()` helper computes power spectra by hand via `np.fft.rfft`, not `scipy.signal.welch`. Dead import. Also note `train_test_split` (from `sklearn.model_selection`) is **used** in cell 20 but **never imported anywhere in the notebook** — see "Potentially broken / disabled cells" below.

## Markdown headers / outline (in order, with cell numbers)

- (cell 0, H1) Group Activity Recognition — Complete Self-Contained Notebook (notebook-level intro/instructions; lists the 8-part structure and says "Use a GPU runtime for the Transformer")
- (cell 1, H2) `## 0 · Setup`
- (cell 3, H2) `## 1 · Build INTERACTION features (your exact code + gaze)`
- (cell 5, H2) `## 2 · Build RECOGNITION features (your exact code + gaze)`
- (cell 7, H2) `## 3 · Load built features + evaluation harness`
- (cell 10, H2) `## 4 · Raw vs engineered features (LogReg · RF, LOGO)`
- (cell 12, H2) `## 5 · Model comparison + confusion matrices (LogReg · RF · Transformer)`
- (cell 14, H2) `## 5.1 · All 7 Combination of Features`
- (cell 16, H2) `## 5.2 · Depp Dive of the Winner Model (accuracy + honest (leave-one-out, LOGO) feature importance + confusion matrix)` [sic, "Depp"]
- (cell 19, H2) `## 6 · Explainability (permutation importance + sensor rollup)`
- (cell 21, H2) `## 7 · Data imbalance solutions (recognition)`
- (cell 23, H2) `## 8 · Stratification — LOGO vs StratifiedGroupKFold`
- (cell 25, H2) `## 9 · Mutual-gaze test — does shared gaze help?`
- (cell 27, H2) `## Summary`

Mapping the notebook's own 8-part list (from cell 0) to actual cell ranges:

1. **Build interaction features** → cell 3 (markdown) + cell 4 (code, `# CELL 1 (v2)`). Genuinely from-raw feature builder — see Circularity Verdict.
2. **Build recognition features** (both incl. gaze) → cell 5 (markdown) + cell 6 (code, `# CELL 1 — RECOGNITION v2`). Genuinely from-raw feature builder.
3. **Raw vs engineered** → cell 7 (load/harness, code cell 8) + cell 10 (markdown) + cell 11 (code, the actual raw-vs-engineered bar chart).
4. **Model comparison + confusion matrices (incl. Transformer)** → cell 12 (markdown) + cell 13 (code, LogReg/RF/Transformer confusion matrices); notebook's own numbering also spins off two extra sub-parts not listed in cell 0's summary: 5.1 (cell 14/15, feature-group ablation) and 5.2 (cell 16/17/18, winner-model deep dive + forward selection).
5. **Explainability** → cell 19 (markdown) + cell 20 (code, permutation importance).
6. **Imbalance** → cell 21 (markdown) + cell 22 (code, SMOTE/oversample/undersample comparison).
7. **Stratification** → cell 23 (markdown) + cell 24 (code, LOGO vs StratifiedGroupKFold).
8. **Mutual gaze** → cell 25 (markdown) + cell 26 (code, gaze ablation test), plus cell 27 (final markdown Summary, which explicitly discusses the gaze result).

## Hardcoded file paths found in code

### INPUT paths

- `/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED` — `INPUT_DIR` in **both** feature-builder cells (cell 4, interaction; cell 6, recognition). This is the **only** raw-data input either builder reads. Discovery is dynamic: `discover(folder)` globs `group_*_model_ready.csv`, extracts `(group, sensor)` via regex `r"group_(\d+)_([a-z]+)_model_ready"`, and keeps a group only if all three of `openearable`, `xsens`, `optitrack` are present. No group list is hardcoded in the builders themselves.
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG` (note: **no "2"**) and `/content/drive/MyDrive/thesis/data/RECOGNITION_ENG` (no "2") — `INT_DIR`/`REC_DIR` in cell 8 (`# === LOAD + HARNESS ===`), feeding `ENG_CSV = {"interaction": f"{INT_DIR}/interaction_eng_features.csv", "recognition": f"{REC_DIR}/recognition_fixed_features.csv"}`. **This is a pre-existing "engineered" feature CSV that this notebook does NOT build anywhere** — it is a different folder from the `INTERACTION_ENG2`/`RECOGNITION_ENG2` this notebook writes in cells 4/6. See Circularity Verdict for why this matters.
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG/interaction_eng_features.csv` — read again directly in cell 15 (`PATH`, the 5.1 feature-group-ablation cell). Same v1 (no "2") file, not this notebook's own ENG2 output.
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG2/interaction_eng_features.csv` — read directly in cell 17 (5.2 deep-dive) and cell 18 (forward selection). **This one correctly points at the file this notebook itself builds in cell 4.**
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG/interaction_eng_features.csv` and `/content/drive/MyDrive/thesis/data/INTERACTION_ENG2/interaction_eng_features.csv` — both read side-by-side in cell 9, explicitly to print/diff the two files' feature-column lists (`paths = {"INTERACTION_ENG": ..., "INTERACTION_ENG2": ...}`). This cell is direct proof the author was aware ENG and ENG2 are two different, separately-produced files.
- Raw `ALL_MODEL_READY_FILES_IDENTITY_FIXED` is read a **second, independent** time in cell 8's `build_raw_all()` (via `_find(g,s)` = `glob.glob(os.path.join(RAW_DIR, f"group_{g}_{s}_model_ready.csv"))`, `RAW_DIR = "/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`) to build a **separate, from-scratch "TRUE raw baseline"** (per-channel mean/std/min/max over 5s windows) used only for the raw-vs-engineered comparison — this path is also non-circular (reads only the raw model_ready CSVs).
- Nowhere in the notebook is any `..._advanced_features.csv`, `..._merged_all_features.csv`, `task1`/`task2`/`task3` output CSV, or any `PUBLICATION_*`/`binary_5s_*` file read. No circular Task-1-style bootstrap of a window grid was found (contrast with `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`, which reads `binary_5s_specialized_oe_merged_all_features.csv`/`binary_5s_all_sensor_advanced_features.csv` to recover its window grid — see that notebook's own `_ANALYSIS.md`).

### OUTPUT paths

- `{OUT_DIR}/interaction_eng_features.csv` where `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG2"` (cell 4) → `INTERACTION_ENG2/interaction_eng_features.csv`.
- `{OUT_DIR}/interaction_eng_tensors.npz` (cell 4) → `INTERACTION_ENG2/interaction_eng_tensors.npz` (keys `X`, `y`, `groups`).
- `{OUT_DIR}/recognition_{mode}_features.csv` where `OUT_DIR = "/content/drive/MyDrive/thesis/data/RECOGNITION_ENG2"` and `mode in ("fixed","labeled")` (cell 6) → `RECOGNITION_ENG2/recognition_fixed_features.csv` and `RECOGNITION_ENG2/recognition_labeled_features.csv`.
- `{OUT_DIR}/recognition_{mode}_tensors.npz` (cell 6) → `RECOGNITION_ENG2/recognition_fixed_tensors.npz` and `RECOGNITION_ENG2/recognition_labeled_tensors.npz`.
- No other CSV/npz/report is written anywhere else in the notebook — parts 3-9 are all in-memory analysis/plots (matplotlib `plt.show()`), nothing is persisted to disk beyond the four files above.

## Function / class definitions

**Cell 4 — Build INTERACTION features:**
- `discover(folder)` — globs `group_*_model_ready.csv`, groups paths by `(group_id, sensor_name)`, returns only groups that have all three of `openearable`/`xsens`/`optitrack`.
- `load(path, timecol, cols)` — reads a model_ready CSV, coerces `timecol` to numeric `"t"`, drops unparseable-time rows, sorts by time; for each requested column coerces to numeric and nulls out values with `abs() >= 1e6` (sentinel/garbage rejection).
- `xoff(oe, xs)` — brute-force cross-sensor time-offset finder: samples `label_Whole_Group` on both OE and Xsens streams over a shared probe grid (0.2s step, trimmed 5s from each end), then searches offsets `0..220s` in 0.5s steps for the one that maximizes label agreement between the two streams; returns the best offset.
- `spec(sig, fs)` — FFT-based spectral summary of a 1D signal: dominant frequency (argmax of power spectrum excluding DC), spectral centroid, and total power (excluding DC bin); returns `(0,0,0)` if fewer than 8 finite samples.
- `circmean_deg(a)` — circular mean of an array of angles given in degrees (via `atan2(mean(sin), mean(cos))`).
- `feats(oe, xs, ot, ws, we)` — per-window feature extractor (see "Gaze functions" note below for the head-orientation block in detail): computes OptiTrack pairwise-distance stats (`dist_close/mid/far_mean`, `dist_disp_mean/std`), centroid speed, per-participant speed (sorted min/mid/max), OpenEarable head pitch/heading/gaze features, head motion frequency/power, sorted acc/gyro energy std across participants, `move_coord` (mean pairwise correlation of OE accel-magnitude across the 3 participants), Xsens hand motion frequency/power/orientation variance and `hand_coord` (mean pairwise correlation of Xsens accel-magnitude), and a 3-channel `RESAMPLE_T`-length interpolated distance tensor for sequence models. Any non-finite value is replaced with `0.0` before returning.
- `build()` — iterates all discovered groups, loads+time-aligns the three sensor streams (`xs["t"] -= xoff(oe,xs)`), computes a boolean "interaction" flag per raw OE sample from the 4 `GROUP_TIERS` label columns (`label_Participant1_Participant2`, `label_Participant1_Participant3`, `label_Participant2_Participant3`, `label_Whole_Group`), slides a fixed 5.0s/5.0s window grid across the intersection of the three streams' time ranges, majority-votes the window label (`"interaction"` if `inter[m].mean() >= 0.5` else `"non_interaction"`), calls `feats()` per window, and returns `(DataFrame, tensor array, y array, group array)`.

**Cell 6 — Build RECOGNITION features:**
- `mapL(label)` — maps a raw (possibly multi-token, `+`/`|`/`/`-joined) ELAN label string to one of `{"co_building","co_merging","conversation"}` via membership in the `CLASS_SETS` dict, else `""`.
- `discover`, `load`, `xoff`, `circmean` — same logic as cell 4's identically-named functions (re-defined, not reused, so the two build cells are independently self-contained).
- `spec(sig, fs)` — same as cell 4's but returns a 2-tuple `(dominant_freq, power)` instead of 3-tuple (no centroid).
- `mn(v)`, `mx(v)` — safe mean/max that return `0.0` on an empty list.
- `feats(oe, xs, ot, ws, we, home)` — per-window feature extractor for recognition: proximity (`dist_close/mid/far_mean`, `dist_disp_std`), centroid speed, per-participant speed, **new** "home"/location cues (`fromhome_mean/min` = distance from each participant's session-median position `home[i]`; `converge_ratio = min(fromhome)/(dist_far_mean+0.5)`), **new** table/arena-center cues (`center_dist_centroid`, `center_dist_min`, both planar `sqrt(x²+z²)` distances), the identical head-gaze block from cell 4 (see below), hand kinematics (same as cell 4), and a **5-channel** tensor (3 interpolated pairwise-distance channels + 1 mean head-motion-energy channel + 1 mean hand-motion-energy channel) for sequence models.
- `build(mode)` — `mode="fixed"`: same uniform 5.0s/5.0s sliding-window grid as cell 4. `mode="labeled"`: run-length-encodes the per-sample mapped-class codes (`np.diff(codes) != 0`) to get contiguous same-label segments as the windows instead (kept if segment length ≥ 0.5s and inside the valid time range). Both modes restrict to samples whose mapped class is in `KEEP`, majority-vote the label, call `feats()`, and return the same 4-tuple as cell 4's `build()`.

**Cell 8 — Load + harness (raw baseline + engineered-feature evaluation harness):**
- `_norm(s)` / `to_rec_class(label)` — same recognition-label mapping as cell 6's `mapL`, but first passes through a `TYPO` dict (`{"object_handiver":"object_handover", "nspecting_pieces":"inspecting_pieces", "returning_target_image_tr":"returning_target_image", "carriyng_thray_to_central_table":"carrying_tray_to_central_table"}`) to fix specific misspellings before matching `CLASS_SETS`.
- `_find(g, s)` — globs the single raw `group_{g}_{s}_model_ready.csv` path for a group/sensor.
- `_stats(a)` — per-channel `[mean, std, min, max]` concatenated across all channels of an array (the 4-stat "raw baseline" representation, matching thesis §5.2's stated raw representation).
- `xoff_fast(oe, xs)` — same brute-force offset search as cell 4/6's `xoff`, reimplemented a third time with slightly different column-name args, used only to align Xsens for the raw baseline.
- `build_raw_all()` — builds a **separate, independent "TRUE raw baseline"** feature set (not the ENG2 features) directly from the raw model_ready files for the hardcoded `GROUPS=[1,2,3,5,6,7,8,9,10]`: for each 5s window, computes `_stats()` over the raw OE (27 ch), OT (9 ch), and Xsens (27 ch) channels, concatenates into a 252-value `raw{i}` feature vector, and labels each window for both the interaction task (majority of the 4 `GROUP_TIERS` columns non-null) and the recognition task (`to_rec_class` majority vote over `label_Whole_Group`).
- `prep(df, classes)` — filters to given label classes, returns `(X, y, groups, feature_names)` with `X` `fillna(0)`.
- `model(name)` — returns a `StandardScaler` + `LogisticRegression(class_weight="balanced")` or `StandardScaler` + `RandomForestClassifier(n_estimators=400, class_weight="balanced")` pipeline.
- `splitter(cv)` — returns `LeaveOneGroupOut()` if `cv=="logo"` else `StratifiedGroupKFold(5, shuffle=True, random_state=0)`.
- `run(df, classes, name, cv="logo")` — full CV loop: `SimpleImputer(median)` + `StandardScaler` + classifier, fit/predict per fold, returns concatenated true/pred labels plus mean accuracy and macro-F1 across folds. This is the workhorse function reused by cells 11, 24, and (via `model`/`splitter`) 26.

**Cell 13 — Confusion matrices incl. Transformer:**
- `load_seq(task)` — loads the `.npz` tensor file for a task (`TENSORS[task]`, defined earlier in cell 9's `paths`-adjacent code — actually referenced but not itself defined in the shown cells; likely expected to be set from the `ENG_CSV`/tensor-path convention, a possible gap — see "Potentially broken" below), `nan_to_num`s it.
- `TFClassifier(nn.Module)` — small Transformer sequence classifier: `Linear(C, d=64)` input projection, a learned positional-embedding placeholder created in `__init__` (`self.posseq = nn.Parameter(torch.randn(1,4096,d)*0.0)`, immediately overwritten per training run — see "Potentially broken"), `TransformerEncoder` with 2 layers / 4 heads / `dim_feedforward=d*2` / dropout 0.1 / `batch_first=True`, mean-pools over the time axis, then a `Linear(d, n_cls)` head.
- `tf_logo(task, epochs=18)` — trains a fresh `TFClassifier` per Leave-One-Group-Out fold: z-score-normalizes the tensor globally (`flat.mean(0)`/`flat.std(0)`), builds inverse-frequency class weights, trains with Adam (`lr=1e-3`) and `CrossEntropyLoss(weight=cw)` for `epochs` epochs in batches of 64, evaluates on the held-out group, returns concatenated true/predicted label arrays across all folds.

**Cell 15 — Feature-group ablation:**
- `grp(c)` — classifies a feature name into `"Proximity"`/`"Head"`/`"Hand"`/`"Other"` by name-prefix matching.
- `evalset(feats, name)` — LOGO evaluation (`LogReg` or `RF`) over a given feature-name list, returns mean accuracy/F1 across folds. Looped over every non-empty combination (`itertools.combinations`) of the discovered feature groups (i.e., all 2ⁿ−1 subsets) to produce the ablation table.

**Cell 17 — Winner-model deep dive:**
- `logo_pred(feats)` — LOGO true/pred arrays for a given feature list using `LogReg` only (no imputer — uses `fillna(0)` directly).
- (inline loop) leave-one-feature-out importance: for each proximity feature, reruns `logo_pred` with that one feature dropped and records the macro-F1 drop vs. the full-proximity-feature baseline.

**Cell 18 — Forward feature selection:**
- `f1_of(feats)` — LOGO macro-F1 (`SimpleImputer` + `StandardScaler` + `LogReg`) for a given feature subset, using a pre-computed fixed CV split (`cv = list(LeaveOneGroupOut().split(...))`).
- (inline greedy loop) forward-selects up to 10 features, one at a time, always adding whichever remaining feature yields the highest `f1_of(selected + [f])`.

**Cell 20 — Explainability:**
- `sensor(f)` — identical logic to cell 15's `grp(c)`, redefined under a different name.
- (inline) for each task, splits `DATA[(task,"eng")]` 70/30 with `train_test_split` (**not imported anywhere in the notebook — see "Potentially broken" below**), fits an RF, runs `sklearn.inspection.permutation_importance` (`n_repeats=10`, `scoring="f1_macro"`), plots top-10 feature importances and a per-sensor-family importance rollup.

**Cell 22 — Imbalance:**
- No named functions; defines `LR = lambda **k: LogisticRegression(max_iter=2000, **k)` and a `strategies` dict of four `imblearn` pipelines (`class_weight`, `SMOTE`, `oversample`, `undersample`), evaluated with `StratifiedGroupKFold(5)` on the recognition task only.

**Cell 26 — Gaze test:**
- `ev(feats, cv)` — inline LOGO/StratifiedGroupKFold evaluation helper (reuses `splitter`/`model` from cell 8) that returns overall macro-F1 and a per-class F1 dict for a given feature subset.

### Gaze functions — detailed geometric/mathematical logic

The three "mutual-gaze" columns are **not computed by a separate named function** — they are inline expressions inside the head-orientation block of `feats()` in both cell 4 and cell 6 (identical code, duplicated verbatim in both builders). The logic, step by step:

1. **Per-participant heading** (`headg[i]`): for each participant `i`, take the OpenEarable magnetometer channels `p{i}_mag_x/y/z` within the window, compute an instantaneous heading angle `atan2(mag_y, mag_x)` in degrees per sample, then reduce to a single window-level heading via `circmean_deg` (circular mean, so it correctly handles wraparound at ±180°). This gives one heading angle per participant per window — **a magnetometer-derived proxy for which direction the participant's head/ear-worn device is pointing, not true gaze/eye-tracking**.
2. **Pairwise facing agreement** (`_gd`): for the three participant pairs `(1,2), (1,3), (2,3)` (indexed `(0,1),(0,2),(1,2)` into the `headg` list), compute `cos(deg2rad(headg[a] - headg[b]))` for each pair. This is `+1` when a pair's headings are identical (facing exactly the same absolute direction), `-1` when they are exactly opposite (180° apart), `0` when perpendicular (90° apart). This 3-vector is also separately exposed (in cell 4 only) as `head_facing = mean(_gd)` — the notebook's pre-existing (non-gaze) "how aligned is everyone's heading" feature.
3. **`head_antiface = mean(clip(-_gd, 0, 1))`** — negates `_gd` then clips to `[0,1]` before averaging: this keeps only the "facing away from each other" component of each pair (positive exactly when that pair's headings are more than 90° apart) and zeroes out any pair that is instead aligned or perpendicular. High values mean the three participants are, on average, oriented in opposing directions.
4. **`head_colinear = mean(abs(_gd))`** — mean of the *absolute* pairwise cosines: high when pairs are either facing the same way or exactly opposite (both are "on the same axis"), low when pairs are oriented perpendicular to each other. This distinguishes "everyone on the same line of sight axis" (whether co-facing or face-to-face) from "everyone facing different, unrelated directions."
5. **`head_facing_min = min(_gd)`** — the raw (unclipped) cosine of the single *least*-aligned pair in the window — i.e., the worst-case pairwise facing agreement, retaining sign (can be strongly negative for a clearly opposed pair).

If fewer than all 3 participants have any OE samples in the window, `headg` has length < 3 and all three gaze features fall back to `0.0` (via `_gd = np.zeros(3)`).

**Cross-check against the thesis reproduction-targets doc**: `docs/thesis_reproduction_targets.md` §3.3 (Section 5.7, Head-movement features) mentions the general concept "similarity of head-facing direction across participants" as one of the head-movement feature concepts, but the thesis's own feature tables (§3.6/5.10, "Advanced Multimodal Feature Engineering," Table 5.2 "OpenEarable feature families") list only pitch/roll/jerk/head-down-fraction/turn-rate/nod-band-entropy/magnetometer-magnitude/heading/active-person-count/dominance/synchrony-lag as OpenEarable feature families — **there is no mention anywhere in the reproduction-targets doc of `head_antiface`, `head_colinear`, `head_facing_min`, or the term "mutual gaze"/"mutual-gaze" at all** (confirmed via full-document search — zero matches). These three columns appear to be a **notebook-only, ad hoc addition** layered on top of the already-ported feature set, not something derived from or documented in the thesis chapters extracted so far, and not something any current `src/features/*` or `src/models/*` code in this repo currently computes or consumes.

## Hyperparameter-looking constants (verbatim)

- `WINDOW_S,STRIDE_S,RESAMPLE_T=5.0,5.0,64` (cell 4, interaction builder)
- `WINDOW_S,STRIDE_S,RESAMPLE_T=5.0,5.0,64` (cell 6, recognition builder — same values, redefined)
- `GROUP_TIERS=["label_Participant1_Participant2","label_Participant1_Participant3","label_Participant2_Participant3","label_Whole_Group"]` (cell 4)
- `LABEL_TIER="label_Whole_Group"` (cell 6)
- `KEEP={"conversation","co_building","co_merging"}` (cell 6)
- Interaction majority-vote threshold: `"interaction" if inter[m].mean()>=0.5 else "non_interaction"` (cell 4)
- Sentinel/garbage rejection threshold: `df[c].where(df[c].abs()<1e6, np.nan)` (cell 4, cell 6, in `load()`)
- Cross-sensor offset search: `np.arange(0,220,0.5)` (max 220s offset, 0.5s step) in `xoff()` (cell 4, cell 6) and `xoff_fast()` (cell 8)
- Offset-search probe grid: `np.arange(max(oe["t"].min(),5),oe["t"].max()-5,0.2)` (cell 4/6) — probe step 0.2s; cell 8's `xoff_fast` uses `0.5` instead
- `RUN-LENGTH min segment length` for "labeled" recognition windows: `et-st>=0.5` (cell 6, `build(mode="labeled")`)
- `GROUPS=[1,2,3,5,6,7,8,9,10]; WIN=5.0` (cell 8) — the canonical 9-group list (group 4 absent), matching `Global_Cleaning_Before_Model.ipynb`'s `GROUPS`.
- `OE_CH`/`OT_CH`/`XS_CH` channel counts: 27 OE, 9 OT, 27 Xsens raw channels → `4 * (27+9+27) = 252` raw baseline features (cell 8)
- `LogisticRegression(max_iter=2000, class_weight="balanced")` (cells 8, 15, 17, 18, 22, 26)
- `RandomForestClassifier(n_estimators=400, class_weight="balanced"/"balanced_subsample", random_state=0)` (cells 8, 15, 20)
- `StratifiedGroupKFold(5, shuffle=True, random_state=0)` (cell 8's `splitter`, used in cells 22, 24, 26)
- `permutation_importance(..., n_repeats=10, random_state=0, scoring="f1_macro")` (cell 20)
- `train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)` (cell 20)
- `TFClassifier(C,n_cls,d=64,heads=4,layers=2)` (cell 13) — Transformer width 64, 4 heads, 2 layers, dropout 0.1
- `tf_logo(task, epochs=18)` (cell 13) — 18 training epochs per LOGO fold, batch size `64` (hardcoded in the training loop `for i in range(0,len(Xtr),64)`), Adam `lr=1e-3`
- `bn=max(curve,key=lambda r:r[2])` forward-selection cap: `while remaining and len(selected)<10` (cell 18) — max 10 features
- Reference line in the forward-selection plot: `plt.axhline(0.691,ls="--",c="#E07A5F",label="all 20 features (0.69)")` (cell 18) — a hardcoded prior-result F1 value (0.691) used only for plotting, not recomputed.
- Color constants: `TEAL,AMBER,CORAL,INK,SLATE = "#1F7A8C","#F2A65A","#E07A5F","#0E2233","#9FB3BE"` (cell 2)

## Potentially broken / disabled cells

- **`train_test_split` used without being imported anywhere in the notebook.** Cell 20 (`## 6 · Explainability`) calls `train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)`, but no cell in the notebook (checked all 14 code cells) contains `from sklearn.model_selection import train_test_split` or any import that would bind that name. On a genuinely fresh runtime (as the notebook's own title promises — "Self-Contained") this cell would raise `NameError: name 'train_test_split' is not defined`. It would only work if the Colab/Jupyter kernel already had `train_test_split` bound in `globals()` from some earlier, unsaved cell execution.
- **`TENSORS` dict referenced in cell 13 (`load_seq`) but not defined in any shown cell.** `def load_seq(task): d=np.load(TENSORS[task], ...)` — `TENSORS` never appears anywhere else in the code-only dump (checked all 14 code cells). This is either a genuine missing-cell/missing-definition bug, or (more likely, given the notebook's general messiness) a variable the author expected to already exist from an earlier interactive session but that was never actually defined in a saved cell — meaning cell 13 (confusion matrices incl. Transformer) would also `NameError` on a truly fresh run.
- **Parts 3-9's "engineered" baseline is NOT this notebook's own freshly-built ENG2 output for most cells.** Cell 8's `ENG_CSV` dict points at `INTERACTION_ENG`/`RECOGNITION_ENG` (no "2"), a pre-existing feature CSV this notebook never builds. `DATA[(task,"eng")]` — used throughout cells 11 (raw vs. engineered), 13 (confusion matrices), 20 (explainability), 22 (imbalance), 24 (stratification), and 26 (the gaze test itself!) — is loaded from this ENG-v1 path, not from `INTERACTION_ENG2`/`RECOGNITION_ENG2` that cells 4/6 just wrote. Only cells 15 (5.1 ablation — actually also uses the v1 `INTERACTION_ENG` path, not ENG2), 17 (5.2 deep dive), and 18 (forward selection) explicitly hardcode the `INTERACTION_ENG2` path. **This means cell 26's own gaze test — the notebook's headline new contribution — evaluates `head_antiface`/`head_colinear`/`head_facing_min` on a feature file (`INTERACTION_ENG`/`RECOGNITION_ENG`, no "2") that, per this notebook's own code, is never populated with those columns.** Cell 26 has a defensive check for exactly this (`present=[c for c in GAZE if c in F.columns]; if not present: print(f"[{task}] no gaze cols — re-run build cell"); continue`), which strongly suggests the author hit this issue and left a guard rather than fixing the path — i.e., the gaze test only actually runs if the `INTERACTION_ENG`/`RECOGNITION_ENG` folder happens to already contain gaze columns from some other, unincluded run of similar code (perhaps an earlier version of cells 4/6 that wrote to `INTERACTION_ENG` before the author renamed the output folder to `ENG2` for this "self-contained" rewrite). Cell 9 (which diffs the ENG vs ENG2 column lists) is direct evidence the author was tracking this discrepancy but the harness/analysis cells were never updated to consistently use ENG2.
- **Markdown/code contradiction**: cell 7's markdown header says `## 3 · Load built features + evaluation harness`, implying it loads the features "built" in parts 1-2 of *this* notebook — but as detailed above, the code it introduces (cell 8) actually loads a different, externally-produced file.
- **`from scipy.signal import welch`** (cell 4) is imported but never called; dead import.
- **`TFClassifier.__init__`** creates a throwaway placeholder positional-embedding parameter (`self.posseq=nn.Parameter(torch.randn(1, 4096, d)*0.0)  # placeholder, replaced below`) that is immediately discarded and replaced every fold in `tf_logo` (`net.posseq=nn.Parameter(torch.randn(1,X.shape[1],net.d,device=device)*0.02)`). Not broken (final behavior is correct — a per-fold-length positional embedding gets used), just wasteful/confusing code.
- **Typo in a section header**: cell 16's markdown says `## 5.2 · Depp Dive of the Winner Model` (should be "Deep Dive"). Cosmetic only.
- **No `TODO`/`FIXME` comments found** anywhere in the code-only dump.
- **Group handling — verified uniform across the canonical 9 groups.** The two feature-builder cells (4, 6) do **not** hardcode a group list at all; they use `discover(INPUT_DIR)` to dynamically find every group folder under `ALL_MODEL_READY_FILES_IDENTITY_FIXED` that has all three sensor `model_ready` files present, then loop over `groups.items()` uniformly (no per-group branching, no group-specific patches, no special-casing anywhere in cells 4 or 6 — unlike `Global_Cleaning_Before_Model.ipynb`'s `PARTICIPANT_POSITION_MAP` seating chart or its group-1 xsens column drop). Cell 8's `build_raw_all()` (the separate raw baseline) **does** hardcode `GROUPS=[1,2,3,5,6,7,8,9,10]` — the exact same 9-group list (group 4 absent) confirmed canonical in `Global_Cleaning_Before_Model_ANALYSIS.md`. So: yes, this notebook processes all 9 groups uniformly, the same 9 groups as `Global_Cleaning_Before_Model.ipynb`, just via dynamic discovery (parts 1-2) rather than a hardcoded literal (part 3's raw-baseline cell), and with no group-specific special-casing anywhere.

## THE CIRCULARITY VERDICT

**Verdict: NO — parts 1-2 (the actual feature-building code) are genuinely non-circular.** They read *only* from `ALL_MODEL_READY_FILES_IDENTITY_FIXED` (the raw per-sample, per-sensor, per-group cleaned corpus that is the confirmed output of `Global_Cleaning_Before_Model.ipynb`) and derive their own window grid, their own labels, and their own features entirely from scratch — with no read of any already-existing features/publication/task1/task2 output CSV anywhere in cells 4 or 6.

**The exact code that proves this:**

```python
# cell 4 (interaction builder)
INPUT_DIR="/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED"
OUT_DIR  ="/content/drive/MyDrive/thesis/data/INTERACTION_ENG2"
...
def discover(folder):
    g={}
    for p in glob.glob(os.path.join(folder,"group_*_model_ready.csv")):
        m=re.search(r"group_(\d+)_([a-z]+)_model_ready",os.path.basename(p))
        if m: g.setdefault(int(m.group(1)),{})[m.group(2)]=p
    return {k:v for k,v in sorted(g.items()) if all(s in v for s in("openearable","xsens","optitrack"))}
...
def build():
    feats_,tens,Y,G=[],[],[],[]
    for g,paths in groups.items():
        oe=load(paths["openearable"],"video_time_s",OE_COLS)
        xs=load(paths["xsens"],"time_s",XS_COLS)
        ot=load(paths["optitrack"],"video_time_s",POS)
        xs["t"]-=xoff(oe,xs)
        lo=max(oe["t"].min(),xs["t"].min(),ot["t"].min()); hi=min(oe["t"].max(),xs["t"].max(),ot["t"].max())
        rt=oe["t"].values
        inter=np.zeros(len(oe),bool)
        for col in GROUP_TIERS:
            if col in oe.columns:
                inter|=oe[col].notna().values & (oe[col].astype(str).str.strip()!="").values
        for ws in np.arange(lo,hi-WINDOW_S+1e-9,STRIDE_S):
            we=ws+WINDOW_S; m=(rt>=ws)&(rt<we)
            if m.sum()==0: continue
            lab="interaction" if inter[m].mean()>=0.5 else "non_interaction"
            row,ten=feats(oe,xs,ot,ws,we); row["group"]=g
            ...
groups=discover(INPUT_DIR); print("Groups:",list(groups))
F,X,Y,G=build()
F.to_csv(f"{OUT_DIR}/interaction_eng_features.csv",index=False)
```

The window boundaries (`np.arange(lo,hi-WINDOW_S+1e-9,STRIDE_S)`) are computed fresh from `lo`/`hi` (the intersection of the three raw sensors' own timestamp ranges after alignment), not read back from any historical output file. The interaction label (`inter[m].mean()>=0.5`) is computed fresh from the raw `label_*` columns present in the raw `openearable` model_ready CSV, not copied from a prior labeled-window CSV. Cell 6's `build(mode)` (recognition builder) follows the identical pattern — `groups=discover(INPUT_DIR)` from the same raw folder, windows computed fresh (either fixed-grid or run-length-encoded from the raw per-sample labels), nothing read from any features/publication output. This is the opposite pattern from `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`, whose own analysis doc documents it reading `binary_5s_specialized_oe_merged_all_features.csv`/`binary_5s_all_sensor_advanced_features.csv` — its own historical output files — specifically to recover the Task 1 window grid, with the code comment "The exact historical window-grid creation code was not retained." No equivalent pattern, comment, or file read exists anywhere in cells 4 or 6 of this notebook.

**Caveat (not circularity, but a related integrity concern worth flagging alongside the verdict):** while parts 1-2 themselves are clean, most of the notebook's *own downstream evaluation* of the resulting features (parts 3, 4, 6, 7, 8, and 9 — i.e., 6 of the notebook's 8 stated parts) does not actually consume the `INTERACTION_ENG2`/`RECOGNITION_ENG2` files that parts 1-2 just built; it consumes a different, pre-existing `INTERACTION_ENG`/`RECOGNITION_ENG` (no "2") pair of files whose provenance is not shown anywhere in this notebook (see "Potentially broken / disabled cells" above). So while the *feature-building code itself* is provably non-circular and re-derivable from raw `model_ready` files, the notebook as a *whole, end-to-end script* is not fully self-verifying — running it top-to-bottom on a machine that only has `ALL_MODEL_READY_FILES_IDENTITY_FIXED` (and not a pre-existing `INTERACTION_ENG`/`RECOGNITION_ENG` folder) would successfully produce `INTERACTION_ENG2`/`RECOGNITION_ENG2` in parts 1-2, but then most of parts 3-9 would fail to find `INTERACTION_ENG/interaction_eng_features.csv` and error out (or, worse, silently reuse stale data from a previous unrelated run).

## Summary

This notebook is a from-scratch feature-engineering-plus-analysis script whose first two parts (cells 3-6) rebuild two engineered feature sets — "INTERACTION" (binary `interaction`/`non_interaction`, fixed 5s/5s sliding windows) and "RECOGNITION" (3-class `conversation`/`co_building`/`co_merging`, either fixed 5s windows or contiguous same-label segments) — directly and only from the raw, per-group, per-sensor `ALL_MODEL_READY_FILES_IDENTITY_FIXED` corpus (the confirmed output of `Global_Cleaning_Before_Model.ipynb`), auto-discovering all groups that have all three sensors present (which in practice is the canonical 9-group set `[1,2,3,5,6,7,8,9,10]`, matching every other notebook in this project). Both builders compute proximity/spatial features (pairwise distances, centroid speed, per-participant speed), head-orientation features from OpenEarable IMU+magnetometer (pitch, heading-alignment, motion frequency/power, energy), hand-kinematics features from Xsens (motion frequency/power, orientation variance, cross-participant synchrony), and — new to this notebook — three "mutual-gaze" columns (`head_antiface`, `head_colinear`, `head_facing_min`) that are actually a magnetometer-heading-alignment proxy for "who is facing whom," not true eye-gaze tracking; the notebook's own closing summary is candid that this is likely a sensor-limitation finding ("the uncalibrated earable magnetometer can't resolve who-faces-whom"). The output is named `INTERACTION_ENG2`/`RECOGNITION_ENG2` — a **different version number from `ENG3`**, which is what the already-ported `src/models/task3_*.py` modules in this repo currently expect (`INTERACTION_ENG3/interaction_eng3_features.csv`, `eng3_recognition_3class_core_features.csv`, etc.). Based on the feature-name overlap visible in this notebook (proximity/`dist_*`, `head_*`, `hand_*`, `speed_*`, `accE_*`/`gyrE_*`, `move_coord`/`hand_coord`), ENG2 looks like it corresponds to the *earlier, simpler* engineered-feature stage described in thesis §5.4-5.9 (the ~20-30 column "conceptual category" features: proximity, group movement, head-movement, hand-movement, coordination) rather than the much larger "Advanced Multimodal Feature Engineering" stage in thesis §5.10 (531 OptiTrack + 506 OpenEarable + 693 Xsens columns after 9-statistic window expansion) that `ENG3` in `src/models/` presumably reflects — i.e. ENG2 is very likely a **different, smaller, and probably earlier-vintage feature set than ENG3**, not a superset or a drop-in replacement; a human should diff the actual `ENG2` and `ENG3` CSV column lists before assuming either supersedes the other, since this analysis is based on code inspection only (no CSVs were run/read). For porting parts 1-2 into `src/features/`: the two builders duplicate almost all of their helper logic (`discover`, `load`, `xoff`, `spec`, head/hand feature blocks) nearly verbatim between cells 4 and 6 — a real port should factor these into shared, tested functions rather than copy-pasting; the `xoff`/`xoff_fast` cross-sensor synchronization-offset search is reimplemented a third, slightly different way in cell 8 and should be unified; and the notebook is genuinely Colab-only unless the hardcoded `/content/drive/MyDrive/thesis/data/...` paths are parameterized. Parts 3-8 (raw-vs-engineered comparison, model comparison + confusion matrices, the 5.1/5.2 feature-group ablation and forward-selection deep dives, permutation-importance explainability, SMOTE/oversample/undersample imbalance comparison, and LOGO-vs-StratifiedGroupKFold stratification comparison) all read as **exploratory analysis-notebook content**, not reproducible pipeline code: they are plot-and-print cells with no persisted output files, several depend on globals from earlier cells without re-derivation, and (per the caveat above) most of them do not even reliably run against this notebook's own freshly-built ENG2 output. None of parts 3-8 appear to map onto a specific labeled thesis table or figure in `docs/thesis_reproduction_targets.md` (the closest thesis content — Chapter 7's Task 1/Task 2 result tables, and the sensor-combination-ablation design in §6.4 — use different sensor-combination groupings, different CV/model configurations, and are already the target of the separate `task1_*`/`task2_*` notebooks in this folder); this reads as Arda's own exploratory follow-up work layered on top of the ENG2 rebuild, most useful for its gaze-test conclusion (part 9) rather than as a reproduction target in its own right. Part 9 (the mutual-gaze test) is the one genuinely new, notebook-specific contribution — its finding (gaze features likely don't help, per the "uncalibrated magnetometer" summary) is worth preserving even if the surrounding analysis machinery is not ported verbatim, and a future reader should note that `head_antiface`/`head_colinear`/`head_facing_min` are entirely absent from the current `docs/thesis_reproduction_targets.md` task mapping and from any `src/` code — so if these are kept, they need to be explicitly documented as a notebook-only addition, not part of the thesis's originally-described feature set.
