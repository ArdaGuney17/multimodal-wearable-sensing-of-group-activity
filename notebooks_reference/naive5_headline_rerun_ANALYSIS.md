# Analysis: naive5_headline_rerun.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\naive5_headline_rerun.ipynb`

## Cell counts

Total cells: 30
Code cells: 21
Markdown cells: 9

## Import statements (deduplicated)

```python
from IPython.display import display
from collections import Counter, defaultdict
from google.colab import drive
from google.colab import files
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset, DataLoader
import gc
import glob
import json
import math
import numpy as np
import numpy as np, pandas as pd
import os
import os, random
import pandas as pd
import random
import re
import time
import torch
import torch.nn as nn
import warnings
```

(`sklearn.base.clone` is imported inline inside `build_pipeline`'s enclosing cell (cell 8) via the top of that cell's block, not repeated here beyond the `from sklearn.base import clone` already implied by the classical-pipeline cell. `from sklearn.preprocessing import FunctionTransformer` is imported locally inside `build_pipeline()` itself, not at module level.)

## Markdown headers / outline (in order)

- (cell 0, H1) Naive-5 re-run — standalone headline notebook
  - Contains an explicit table mapping each part to a thesis table (quoted verbatim below — this is the single most important piece of evidence in the whole notebook for section 9).
- (cell 1, H2) Part 0 — setup, Drive, and the naive-5 switch
- (cell 6, H2) Part 1 — Interaction + activity recognition (tables 7.2–7.7)
  - (cell 12, H3) 1a — classical models (LOGO)
  - (cell 14, H3) 1b — deep-model definitions
  - (cell 16, H3) 1c — deep sequence models (LOGO)
- (cell 20, H2) Part 2 — Task 3 grammar headline (table 8.7)
- (cell 24, H2) Part 3 — Task 3 persistence (table 8.2)
- (cell 28, H2) Save results

Cell 0's full text, quoted verbatim (this is the notebook's own self-description and is authoritative for section 9):

> Re-runs **only the headline experiments** on the five fully-naive groups **{G2, G3, G5, G6, G10}**
> and compares them with the original nine groups.
> (The author was the third participant in G1, G7, G8, G9. G4 was lost to a camera failure.)
>
> All code below is copied verbatim from your own notebooks; only the *configuration*
> is narrowed so this runs in minutes instead of hours.
>
> | Part | Produces | Thesis table |
> |---|---|---|
> | 1 | Interaction + activity recognition (headline sensors/models only) | 7.2–7.7 |
> | 2 | Task 3 grammar: back-off n-gram over activity tokens | 8.7 |
> | 3 | Task 3 persistence: all-window vs transition-only | 8.2 |
>
> ### How to run
> 1. Set `RUN_ON` in the switch cell (`"all"` or `"naive"`).
> 2. Runtime → Run all.
> 3. Repeat with the other value of `RUN_ON`.
> 4. Send me the two `naive5_results_*.json` files that the last cell writes.

Cell 20's markdown adds an important scoping note: "the neural token models are **not** re-run: they were not the headline and are slow" — i.e. only the deterministic n-gram back-off model is rerun for Task 3 grammar, not the tokenized-Transformer/HMM/hybrid models that appear elsewhere in the full Task 3 notebook.

## Hardcoded file paths found in code

**INPUT paths** (all pre-computed feature/label CSVs from earlier pipeline stages — this notebook does **not** compute features from raw sensor data, and for Part 1 it does **not** consume Task 1/2's already-computed *results* CSVs either; it reloads the same raw per-window *feature* CSVs Task 1/2's full-cohort notebooks read, then re-runs the entire classical+DL pipeline from scratch, restricted to the naive-5 groups via a global `pd.read_csv` monkey-patch — see section 9):

- `{DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv` (`ACTIVITY_DATA_PATH`, Part 1, 10s activity-recognition feature set)
- `{DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv` (`INTERACTION_DATA_PATHS[0]`, Part 1, preferred 5s interaction feature set)
- `{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv` (`INTERACTION_DATA_PATHS[1]`, Part 1, fallback if the specialized-OE file is missing)
- `{DATA_ROOT}/RQ3_LABEL_NORMALIZATION/rq3_normalized_labels_full.csv` (`NORM`, Part 2 token-building, and again as `FULL_LABEL_PATH`, Part 3)
- `{DATA_ROOT}/INTERACTION_ENG3/interaction_eng3_features.csv` (`FEATURE_CANDIDATES[0]`, Part 2, sensor features merged onto activity tokens)
- `{DATA_ROOT}/INTERACTION_OE10/interaction_oe10_10s.csv` (`FEATURE_CANDIDATES[1]`, Part 2, fallback sensor-feature source)
- `{DATA_ROOT}/RQ3_LABEL_NORMALIZATION/rq3_label_normalization_metadata.json` (`META_PATH`, Part 3, optional group/time column hints)
- `{DATA_ROOT}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/activity_tokens_6label_fullstat.csv` (`TOKEN_PATH`, Part 2 — read *if it already exists* thanks to `RESUME_EXISTING=True`, otherwise built from the two inputs above and then written)

**OUTPUT paths** (note the naming pattern flaw discussed in section 8/9: none of these carry the `TAG`/`RUN_ON` value, so a `RUN_ON="naive"` run overwrites the exact same files a `RUN_ON="all"` run would have written):

- `{DATA_ROOT}/ALL_SENSOR_MULTI_TASK_ABLATIONS/` (`OUT_DIR`, Part 1 root)
  - `{OUT_DIR}/{task_name}/{task_name}_feature_counts.csv`
  - `{OUT_DIR}/{task_name}/{sensor_combo}/{task_name}_{sensor_combo}_classical_summary.csv`
  - `{OUT_DIR}/{task_name}/{sensor_combo}/{task_name}_{sensor_combo}_classical_predictions.csv`
  - `{OUT_DIR}/{task_name}/{sensor_combo}/{task_name}_{sensor_combo}_classical_best_per_condition.csv`
  - `{OUT_DIR}/combined_classical_summary.csv`
  - `{OUT_DIR}/combined_classical_best_per_condition.csv`
  - `{OUT_DIR}/FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED/` (`FAST_DL_OUT_DIR`)
    - `fast_dl_plan.csv`
    - `combined_fast_dl_no_elapsed_summary.csv`
    - `combined_fast_dl_no_elapsed_fold_metrics.csv`
    - `combined_fast_dl_no_elapsed_predictions.csv` (only if `SAVE_FAST_DL_PREDICTIONS` is `True` at the point the loop runs — see the cell-11-vs-cell-17 contradiction in section 8)
    - `combined_fast_dl_no_elapsed_best_per_task_sensor.csv`
- `{DATA_ROOT}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/activity_tokens_6label_fullstat.csv` (`TOKEN_PATH`, Part 2 — written only on first build, reused/naive-filtered on reruns)
- `{DATA_ROOT}/RQ3_7LABEL_HISTORY_AWARE_PREDICTION/` (`OUT_DIR`, Part 3 — created via `os.makedirs` but **nothing is ever written into it**; Part 3 only prints and calls `variance_report`/`log_result`, so this directory is dead output)
- `naive5_results_{TAG}.json` (final cell, written to the *current working directory*, not under `DATA_ROOT` — `TAG` is `"full9"` or `"naive5"` depending on `RUN_ON`; this is the **only** output artifact in the whole notebook whose name actually distinguishes the two cohorts). Triggers `google.colab.files.download(fn)` in a try/except (Colab-only, degrades gracefully elsewhere).

## Function / class definitions

- cell 4, function `_gid` — strips non-digit characters from a group value (e.g. `"group_2"` → `2`) via `re.sub(r"\D", "", str(g))`; used everywhere to normalize inconsistent group-column typing/naming before checking membership in `NAIVE_GROUPS`.
- cell 4, function `_read_csv` (monkey-patches `pd.read_csv` at module scope) — calls the original `pd.read_csv`, then if `RUN_ON == "naive"` and the result has a `"group"` column, filters rows to `NAIVE_GROUPS` in place and prints a before/after row-count line; this single patch is what makes every downstream cell's "copied verbatim" full-cohort code produce naive-5-restricted results without being individually rewritten (see section 9).
- cell 4, function `log_result` — appends `{"config": name, **kv}` to `RESULTS[part]` (creating the list if needed) and prints a one-line log entry; the shared accumulator that the final JSON dump is built from.
- cell 5, function `variance_report` — given `y_true/y_pred/groups`, computes per-group accuracy/macro-F1, pools them, computes fold mean±SD, a Student-t 95% CI and a 2000-sample bootstrap 95% CI, prints a formatted block tagged with `[{TAG}]`, calls `log_result(part, name, pooled_macro_f1=..., pooled_acc=..., fold_mean=..., fold_sd=..., ci_lo=..., ci_hi=..., n_folds=...)`, and returns the per-group DataFrame. This is the single function that produces every fold-mean±SD/CI number quoted in Tables 7.11–7.13/8.8.
- cell 8, function `safe_name` — sanitizes a string into a `[A-Za-z0-9_]` token (used for per-class metric-column suffixes).
- cell 8, function `unique_feats` — de-duplicates a list while preserving order (`dict.fromkeys`).
- cell 8, function `find_first_existing` — returns the first path in a list that exists on disk, else `None`.
- cell 8, function `infer_window_seconds` — estimates a dataset's window length from the median `end_col - start_col`, falling back to a caller-supplied default.
- cell 8, function `add_elapsed_min` — derives an `elapsed_min` "time-in-session" feature per group from window midpoint minus that group's minimum midpoint, divided by 60.
- cell 8, function `looks_bad_feature_name` — flags column names containing any of `BAD_TOKENS` (label/target/class/group/window/time/elapsed/pred/...) so they're excluded from feature lists.
- cell 8, function `is_oe_feature` / `is_opti_feature` / `is_xsens_feature` — prefix/token-based modality classifiers for column names (OE = OpenEarable ear-worn audio features, OPTI = OptiTrack motion-capture features, XSENS = XSens IMU features), identical logic to the sibling `task1_full_comparison...` notebook.
- cell 8, function `clean_feature_list` — drops non-feature/meta columns, optionally strips `elapsed_min`, coerces to numeric, and drops near-constant or mostly-missing columns (`<20` finite values or `std < 1e-12`).
- cell 8, function `make_classical_models` — returns a dict of 5 sklearn estimators (`logreg_C1`, `linearSVC_C1`, `rbfSVC_C1_gscale`, `rf_leaf2`, `extraTrees_leaf1`), all seeded with `RANDOM_STATE` and `class_weight="balanced"`.
- cell 8, function `_clip_for_float32` — clips post-`RobustScaler` values to ±1e6 and replaces non-finite values, guarding tree models' internal float32 casts against overflow.
- cell 8, function `build_pipeline` — builds an sklearn `Pipeline` of median-`SimpleImputer` → `RobustScaler` → clip → optional `SelectKBest(f_classif, k)` → cloned model.
- cell 8, function `metric_dict` — computes accuracy/macro-F1/balanced-accuracy plus per-class precision/recall/F1/support for a fixed `label_order`.
- cell 9, function `get_modality_features` — applies the three `is_*_feature` classifiers to a DataFrame's columns and returns a `{"OE":..., "OPTI":..., "XSENS":...}` dict, made mutually exclusive by subtracting already-claimed columns.
- cell 9, function `make_combo_feature_sets` — for each of the 7 entries in `SENSOR_COMBINATIONS`, unions the relevant modality feature lists and produces both a no-elapsed and an elapsed variant.
- cell 9, function `prepare_task` — the task dispatcher: given a task name (one of the 6 in `RUN_TASKS`), filters/derives labels from either `interaction_df_raw` (Task 1) or `activity_df_raw` (all Task-2 variants), builds `combo_features`/`combo_features_elapsed`, and returns a `spec` dict consumed by every downstream runner. Identical branch structure to the sibling `task1_full_comparison...` notebook's `prepare_task`, but here **all 6 branches are reachable** because `RUN_TASKS` includes all 6 (unlike that sibling notebook, where only `interaction_vs_noninteraction` was enabled).
- cell 10, function `sanitize_spec` — replaces ±inf with NaN in every feature column referenced by a spec's `combo_features`/`combo_features_elapsed`, drops now-all-NaN-or-constant columns from those feature lists, and returns the list of columns that contained inf.
- cell 11, function `_canon` — canonicalizes a sensor-combo key (e.g. `"OE+OPTI"` or `"OE_OPTI"`) into a `frozenset` of parts, used to reconcile the naming mismatch between `SENSOR_COMBINATIONS`'s `_`-joined keys and a `WANT` list written with `+`.
- cell 13, function `run_classical_for_task_and_sensor` — for one `(spec, sensor_combo)`: loops `TIME_CONDITIONS × models × K_CLASSICAL`, runs full `LeaveOneGroupOut` per configuration, pools predictions, writes summary/predictions/best-per-condition CSVs, and returns them. Because of the `pd.read_csv` patch, `spec["df"]` was already naive-5-filtered when it was loaded, so `LeaveOneGroupOut` here produces **5 folds, not 9**, purely as a side effect — this function's own code is unmodified from the full-cohort version.
- cell 15, class `RNNClassifier` (`__init__`, `forward`) — LSTM/GRU wrapper (uni- or bi-directional) with a `LayerNorm→Dropout→Linear` head on the last timestep.
- cell 15, class `PositionalEncoding` (`__init__`, `forward`) — standard sinusoidal positional encoding added to a sequence.
- cell 15, class `TransformerClassifier` (`__init__`, `forward`) — linear input projection → positional encoding → `TransformerEncoder` (norm-first, GELU) → `LayerNorm→Dropout→Linear` head on the last timestep.
- cell 15, function `set_seed` — seeds `random`, `numpy`, and `torch` (+ `cuda` if available).
- cell 15, function `build_model` — dispatches a `model_cfg["model_type"]` string to the right class/kwargs.
- cell 15, function `make_loader` — wraps `(X, y)` numpy arrays in a `TensorDataset`/`DataLoader`.
- cell 15, function `make_sequences` — per group, sorts rows by `starts`, and for every position `≥ seq_len-1` builds a sliding window of the previous `seq_len` rows (features) labeled with the current row's target — the sequence-construction routine shared by both the exploratory and fast DL runners.
- cell 15, function `torch_predict` — batched inference returning argmax class ids (defined but the *fast* runner uses its own near-duplicate, `torch_predict_fast`, instead).
- cell 15, function `choose_validation_group` — from the training groups, picks the "best" validation group by (has ≥2 classes, most rows), used to carve an in-fold validation split out of the LOGO training set.
- cell 17, function `should_run_fast_dl_task` / `should_run_fast_dl_sensor` — allow-list gates (both effectively no-ops here since `FAST_DL_TASKS_TO_RUN`/`FAST_DL_SENSOR_COMBOS_TO_RUN` are `None`, meaning "run everything").
- cell 17, function `choose_fast_seq_len` — picks `spec["seq_lens"][-1]` (i.e. the longest configured context) given `FAST_DL_SEQ_CHOICE = "last"`.
- cell 17, function `get_fast_model_configs` — filters `MODEL_CONFIGS` down to `FAST_DL_MODEL_TYPES` (`["bilstm", "transformer"]`).
- cell 17, function `torch_predict_fast` — near-duplicate of `torch_predict` with a different default batch size (`FAST_PRED_BATCH_SIZE=1024`).
- cell 17, function `train_one_fast_dl_run` — the full per-`(spec, sensor_combo, seq_len, k_features, model_cfg, seed)` LOGO training loop: in-fold median-impute → `RobustScaler` → clip → `SelectKBest` → `make_sequences` → trains with an early-stopping loop (`FAST_MAX_EPOCHS=25`, `FAST_PATIENCE=4`) picking the best validation-macro-F1 epoch, evaluates on the held-out test group, and returns pooled `summary`/per-fold `fold_rows`/optional `pred_rows` DataFrames.
- cell 18, function `_log_best` — for a results DataFrame, groups by the given keys, takes the top row by `(macro_f1, accuracy)` per group, and calls `log_result` for each — used to populate `RESULTS["part1_recognition"]` from both `combined_classical` and `combined_fast_dl`.
- cell 19, function `_pick` — returns the first column name from a candidate list that exists in a DataFrame (used to locate group/`y_true`/`y_pred` columns in the DL predictions CSV, which may be named inconsistently).
- cell 22, function `first_existing` — same "first matching column name" helper as `_pick`, independently redefined (not reused) for locating group/time columns in the label-normalization file.
- cell 22, function `load_six_label_windows` — loads `rq3_normalized_labels_full.csv`, merges the 6-label `social_conversation`/`task_conversation` → `conversation` collapse, left-merges numeric sensor columns from whichever `FEATURE_CANDIDATES` path exists, and returns `(df, sensor_cols)`.
- cell 22, function `channel_statistics` — computes a 14-value statistics vector (`STAT_NAMES`: mean/std/min/max/range/median/iqr/p10/p25/p75/p90/energy/rms/spectral-entropy) for one sensor channel's values within a segment.
- cell 22, function `build_fullstat_tokens` — groups by `group`, splits each group's time-ordered rows into contiguous same-label segments (`(label != label.shift()).cumsum()`), and for each segment emits one "token" row: `group`, `label`, `duration`, `start_time`, plus `{channel}__{stat_name}` columns from `channel_statistics` for every sensor column — this is the 6-label activity-token table Part 2's n-gram grammar runs over.
- cell 23, function `seqs_by_group` — sorts each group's tokens by `start_time`/order column and returns `{group: [label, label, ...]}`.
- cell 23, function `fit_ngram` — builds back-off count tables `tabs[h][context_tuple] = Counter(next_label)` for `h = 1..max_h`, plus a unigram fallback `Counter`.
- cell 23, function `predict_ngram` — given a history, tries the longest available context first and backs off to shorter contexts, finally to the unigram mode (hardcoded fallback `"conversation"` if the unigram table is empty).
- cell 26, function `first_existing` — third independent redefinition of the same "first matching column" helper (cells 8/9's `find_first_existing`, cell 22's and cell 26's `first_existing` are three separately-written near-duplicates across the notebook — a copy-paste artifact worth collapsing into one shared utility during porting).

No functions are defined in cells 21, 25, 27, or 29 — those cells are top-level config/script code only.

## Hyperparameter-looking constants (verbatim)

- `RUN_ON = "naive"          # "all" = original 9 groups | "naive" = 5 naive groups`
- `NAIVE_GROUPS    = {2, 3, 5, 6, 10}`
- `RESEARCHER_GRPS = {1, 7, 8, 9}`
- `TAG = "full9" if RUN_ON == "all" else "naive5"`
- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"` (redefined identically 3 times — cells 7, 21, 25)
- `ACTIVITY_DATA_PATH = f"{DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv"`
- `INTERACTION_DATA_PATHS = [...]` (2-element list, see paths section)
- `OUT_DIR = f"{DATA_ROOT}/ALL_SENSOR_MULTI_TASK_ABLATIONS"` (cell 7; later **reassigned** in cell 25 to `os.path.join(DATA_ROOT, "RQ3_7LABEL_HISTORY_AWARE_PREDICTION")` — same variable name reused for two unrelated directories across parts)
- `RUN_TASKS = ["interaction_vs_noninteraction", "conversation_vs_nonconversation", "conversation_vs_building", "conversation_vs_merging", "merging_vs_building", "three_class_activity"]`
- `SENSOR_COMBINATIONS = {"OE": ["OE"], "OPTI": ["OPTI"], "XSENS": ["XSENS"], "OE_OPTI": ["OE","OPTI"], "OE_XSENS": ["OE","XSENS"], "OPTI_XSENS": ["OPTI","XSENS"], "OE_OPTI_XSENS": ["OE","OPTI","XSENS"]}`
- `RUN_CLASSICAL = True`
- `RUN_DL = True`
- `K_CLASSICAL = [40, 80, 120, 200, "all"]` (cell 7) → **narrowed** to `K_CLASSICAL = [80, 200]` (cell 11)
- `TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]`
- `K_DL = [80, 120, 200]` (cell 7 — defined but never read; the actually-used deep-learning grid is `FAST_DL_K`)
- `SEEDS = [42]`
- `MAX_EPOCHS = 80` / `PATIENCE = 12` / `BATCH_SIZE = 64` (cell 7 — belong to the never-invoked "full" DL runner; the notebook only runs the "fast" DL path, which has its own `FAST_*` constants)
- `RUN_DL_MODEL_TYPES = ["lstm", "bilstm", "gru", "transformer"]` (cell 7 — superseded for the actual run by `FAST_DL_MODEL_TYPES`)
- `STOP_AFTER_N_DL_RUNS_PER_TASK_SENSOR = None` (cell 7 — defined, never referenced again)
- `RANDOM_STATE = 42`
- `WANT = ["OE", "OPTI", "XSENS", "OE+OPTI", "OE+XSENS", "OPTI+XSENS", "OE+OPTI+XSENS"]` (cell 11 — all 7 combos kept; "narrow the grid" narrows *models/k*, not sensor combos)
- `FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]` (cell 11, restated identically in cell 17)
- `FAST_DL_K = 120` (cell 11, restated in cell 17)
- `FAST_DL_SEQ_CHOICE = "last"` (cell 11, restated in cell 17)
- `SAVE_FAST_DL_PREDICTIONS = True   # needed for per-group DL variance` (cell 11) — **then overwritten to** `SAVE_FAST_DL_PREDICTIONS = False` in cell 17 (see section 8 — a real cross-cell contradiction)
- `FAST_DL_TASKS_TO_RUN = None` / `FAST_DL_SENSOR_COMBOS_TO_RUN = None` (cell 17)
- `FAST_MAX_EPOCHS = 25` / `FAST_PATIENCE = 4` / `FAST_BATCH_SIZE = 256` / `FAST_PRED_BATCH_SIZE = 1024` (cell 17)
- `MAX_LOGO_FOLDS = None` (cell 17 — "Full LOGO is still honest")
- `ALL_MODEL_CONFIGS = [{"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4}, {"model_type": "bilstm", ...same...}, {"model_type": "gru", ...same...}, {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4}]`
- `RESUME_EXISTING = True` (cell 21 — governs whether `TOKEN_PATH` is reloaded vs. rebuilt)
- `TOKEN_SEEDS = [42, 1, 7]` (cell 21 — "Historical report seeds for token neural models" — dead: the neural token models are explicitly not rerun in this notebook, per cell 20's markdown)
- `SEED_STD_DDOF = 0` (cell 21 — dead, same reason)
- `FOLD_STD_DDOF = 1` (cell 21 — dead in this notebook; `variance_report` hardcodes `ddof=1` itself rather than reading this constant)
- `K_SELECT = 40` (cell 21 — "in-fold sensor feature selection used in the old token notebook" — dead, no selection step runs here)
- `GRAMMAR_ORDERS = [1, 2, 3, 5, 10]` (cell 21) — **not actually used**: cell 23's loop hardcodes `for H in [1, 2, 3, 5]`, silently dropping order 10 from `GRAMMAR_ORDERS`
- `HYBRID_ORDER = 3` / `HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)` (cell 21 — dead, no hybrid model runs here)
- `TASK3_PIPELINE_VERSION = 'final_v2_common_targets'` (cell 21)
- `MERGE6 = {'social_conversation': 'conversation', 'task_conversation': 'conversation'}` (cell 22)
- `STAT_NAMES = ['mean','std','min','max','range','median','iqr','p10','p25','p75','p90','energy','rms','entropy']` (cell 22)
- `BAD_TOKENS = ["label","target","class","group","window","time","elapsed","pred","prediction","correct","fold","split","index"]` (cell 8)
- Model hyperparameters embedded in `make_classical_models`: `C=1.0`, `max_iter=5000` (logreg) / `max_iter=10000` (linearSVC), `class_weight="balanced"`, `n_estimators=500`, `min_samples_leaf=2` (RF) / `min_samples_leaf=1` (ExtraTrees), `gamma="scale"` (RBF SVC)
- `_clip_for_float32(X, limit=1e6)`

## Potentially broken / disabled cells

- **Real cross-cell contradiction (not cosmetic)**: cell 11 sets `SAVE_FAST_DL_PREDICTIONS = True` with the comment `# needed for per-group DL variance` — anticipating cell 19's need to read row-level DL predictions. Cell 17's "FAST DL SETTINGS" block then re-executes `SAVE_FAST_DL_PREDICTIONS = False` with the comment `# False is much faster and lighter`, silently undoing cell 11's intent. On a straight top-to-bottom "Run all," cell 17's value wins (it runs after cell 11 and is the value in effect when the DL training loop actually checks it), so `combined_fast_dl_no_elapsed_predictions.csv` is **never written**, and cell 19's DL branch (`if os.path.exists(dlp) and os.path.getmtime(dlp) >= RUN_START`) falls through to `print("no fresh DL predictions — set SAVE_FAST_DL_PREDICTIONS = True and re-run the DL cell")`. Net effect: **`RESULTS["part1_variance"]` only ever gets populated with classical per-group variances, never deep-model per-group variances**, unless a human notices and re-runs cell 11 after cell 17 (or edits cell 17). This directly affects the deep-model fold-mean±SD/CI numbers a reader might expect this notebook to reproduce for Table 7.11's "Deep no-elapsed" column and any deep-side rows of a Table-7.13-style per-group breakdown.
- **Silent config drop**: `GRAMMAR_ORDERS = [1, 2, 3, 5, 10]` is defined in cell 21 as the "Grammar orders for the updated Task 3 comparison," but cell 23's actual sweep hardcodes `for H in [1, 2, 3, 5]` — order 10 is silently never evaluated in this notebook, even though the constant advertises it. (Table 8.8's naive-cohort grammar block in the thesis only shows h=1,2,3,5 anyway, so the *output* is consistent with the thesis table — but the dead constant is misleading if read as documentation of what the cell does.)
- **Output-path collision across `RUN_ON` values, by design gap rather than by cell content**: none of Part 1's or Part 3's output paths (`OUT_DIR`, `FAST_DL_OUT_DIR`, `combined_classical_summary.csv`, `combined_fast_dl_no_elapsed_*.csv`, etc.) include `TAG` or `RUN_ON`. Only the final `naive5_results_{TAG}.json` does. Per the "How to run" instructions in cell 0 ("Repeat with the other value of `RUN_ON`"), the *intended* workflow is to run the whole notebook twice (once per `RUN_ON` value) — but the second run overwrites every intermediate CSV the first run wrote at those same paths. Cell 19's `if os.path.getmtime(p) < RUN_START: continue` guard against "stale full-9 file" is direct evidence the notebook's own authors were aware of and worked around this within a *single* run's lifetime (to avoid picking up leftover files from a still-earlier run) — but nothing protects the *previous* run's outputs once a second run starts writing. A reader trying to keep both cohorts' intermediate CSVs side-by-side needs to manually move/rename `ALL_SENSOR_MULTI_TASK_ABLATIONS/` between the two `RUN_ON` passes; only the final JSON is inherently safe to keep both copies of.
- **Dead output directory**: Part 3 (cells 25–27) creates `{DATA_ROOT}/RQ3_7LABEL_HISTORY_AWARE_PREDICTION/` via `os.makedirs(OUT_DIR, exist_ok=True)` (cell 25) but never writes any file into it — Part 3 only calls `variance_report(...)` (console output + `RESULTS` dict), so this directory is created and left empty.
- **Duplicated helper functions**: `find_first_existing` (cell 8), `first_existing` (cell 22), and `first_existing` again (cell 26, a separate, non-shared re-definition) all implement the same "return the first candidate that exists/matches" pattern with slightly different signatures (path-existence vs. column-name-matching) — not broken, but a copy-paste smell to consolidate into one utility when porting.
- **`OUT_DIR` variable reused for two unrelated directories** across the notebook (`ALL_SENSOR_MULTI_TASK_ABLATIONS` in cell 7, then reassigned to `RQ3_7LABEL_HISTORY_AWARE_PREDICTION` in cell 25) — harmless only because Part 1's uses of `OUT_DIR` (cells 9, 13, 19) all complete before cell 25 reassigns it, i.e. correctness depends on cells running strictly in written order.
- **No Colab-hard-dependency**: `drive.mount(...)` (cell 3) is *not* wrapped in a try/except here (unlike the sibling `task1_full_comparison...` notebook), so this notebook, unlike that one, is **not** gracefully Colab-optional — running it outside Colab will raise on that import/call unless the cell is skipped or edited.
- **This notebook does NOT contain the "exact optimized Transformer reproduction" cells** (the 211-feature/`seq_len=18`/`k=120`/`seed=42`/4426-sequence-assertion pattern already ported into `src/models/task1.py`'s `run_exact_reproduction`/`run_naive5_reproduction`). Grepping this notebook for `211`, `4426`, `OPTI2_RELATIVE_ONLY`, `is_relative_opti2`, and `task1_optimized_0806` returns **zero matches**. That specific single-config exact-reproduction path was ported from cells 16–17 of the *sibling* `task1_full_comparison_classical_elapsed_dl_with_std.ipynb` notebook (confirmed by that file's own module docstring), not from this one — see section 9/10 for what this means for porting.
- No `!pip install` cells found. No cells reference undefined variables at static-read level. The Turkish-language comments/prints noted in the sibling `task1_full_comparison...` analysis do **not** appear in this notebook — all comments/prints here are English.

## WHICH TASKS/TABLES DOES THIS NOTEBOOK ACTUALLY COVER

This is the single most load-bearing question, and the notebook answers it explicitly in its own cell-0 markdown table (quoted in full above): **Part 1 → Tables 7.2–7.7, Part 2 → Table 8.7, Part 3 → Table 8.2.** Read literally, that table names the *full-cohort* table numbers each part's pipeline is copied from — not the naive-cohort comparison tables those pipelines feed once rerun with `RUN_ON="naive"`. Cross-referencing against `docs/thesis_reproduction_targets.md`, the naive-cohort-specific tables those reruns actually populate are:

- **Part 1 (cells 4–19) → feeds Table 7.11** (full-9 vs. naive-5 comparison across all six recognition tasks) **and is the probable raw source for Table 7.12/7.13's Full9 columns as a consistency check** (the reproduction-targets doc explicitly notes "Nine-group values in Table 7.12 agree with the ablation tables 'to within 0.0004 macro-F1' — i.e. Table 7.12's full9 column is a consistency check against Tables 7.2–7.7, not an independent number").
  - Evidence this is Table 7.11's source: `RUN_TASKS` (cell 7) has exactly the 6 tasks in Table 7.11's 6 rows (`interaction_vs_noninteraction` = Task 1; the five `conversation_vs_*`/`merging_vs_building`/`three_class_activity` variants = Tasks 2a–2e). Table 7.11's caption is "**OptiTrack feature family**" and its "Deep no-elapsed (full9 A/M)" column values match Table 7.12's "OPTI ... " configs exactly for at least one row (Task 2a: `0.880/0.845` appears identically in both tables) — strong evidence Table 7.11 is this notebook's full `SENSOR_COMBINATIONS` grid (cells 9/13/17), filtered post-hoc to `sensor_combo == "OPTI"` (that filtering step itself is **not** performed inside this notebook — it produces the full 7-sensor-combo grid, cell 18's own comment says "expect 7 sensors x 6 tasks x 3 regimes = 126" rows of `RESULTS["part1_recognition"]" — someone/something downstream, likely manual analysis or one of the two sibling notebooks, must then select the OPTI-only rows to render Table 7.11 as printed).
  - Cell 4's monkey-patched `pd.read_csv` is what turns cell 9's `interaction_df_raw`/`activity_df_raw` loads into naive-5-restricted DataFrames, which is what turns cell 13's `LeaveOneGroupOut()` into 5 folds instead of 9 — the source of the "naive5" half of every Table 7.11/7.12 comparison column.
  - Quote proving the mechanism: cell 13's `run_classical_for_task_and_sensor` is verbatim the same LOGO runner as the sibling `task1_full_comparison...` notebook (`logo.split(X, y, groups)` with no `NAIVE_GROUPS` filtering logic anywhere in its body) — the naive restriction happens entirely upstream, invisibly, via the monkey-patched loader.
  - Cell 19's `variance_report(...)` calls (via `_log_best`'s selected best-config-per-`(task, sensor_combo, time_condition)` rows) are exactly the fold-mean±SD/CI computation Table 7.12's "fold mean±SD [CI]" columns and a Table-7.13-style per-group breakdown require — but see the section-8 finding that the deep-model half of this is silently skipped due to the `SAVE_FAST_DL_PREDICTIONS` contradiction.

- **Part 2 (cells 20–23) → feeds Table 8.8's upper block** (Task 3 activity-token back-off n-gram, naive-5 subset). Direct code evidence: cell 23's own in-code comment reads `# BACK-OFF N-GRAM over activity tokens  (thesis Table 8.7 headline)` — i.e. the code itself names Table 8.7 as the *full-cohort* origin of this exact pipeline, and running it under `RUN_ON="naive"` (via the same `pd.read_csv` patch acting on `NORM`/`FEATURE_CANDIDATES` in cell 22's `load_six_label_windows`, and on `TOKEN_PATH` if reloaded) is precisely what `docs/thesis_reproduction_targets.md` §8.10/Table 8.8 describes: "activity-token back-off grammar (116 tokens for the naive subset, vs. 244 for the full cohort)." Cell 23's `for H in [1, 2, 3, 5]: ... variance_report(yt, yp, gg, f"n-gram back-off h={H}", "part2_grammar")` produces exactly the four `n-gram back-off, h=1/2/3/5` rows of Table 8.8's upper block.

- **Part 3 (cells 24–27) → feeds Table 8.8's lower block** (Task 3 persistence: all-window vs. transition-only, naive-5 subset). Direct code evidence: cell 27's in-code comment reads `# PERSISTENCE vs TRANSITION-ONLY  (thesis Table 8.2)`, again naming the full-cohort table this pipeline is copied from; run under `RUN_ON="naive"` it produces the naive-restricted `947`-labelled-window/`111`-transition persistence numbers described in reproduction-targets §8.10. Cell 27's three `variance_report(...)` calls (`"repeat-current (all windows)"`, `"no-self n-gram (transition-only)"`, `"repeat-current (transition-only)"`) map 1:1 onto Table 8.8's `repeat-current, all windows` / `no-self n-gram, transitions` / `repeat-current, transitions` rows.

- **This notebook does NOT produce Table 10.1.** Table 10.1 is a purely descriptive comparison computed directly from raw ELAN annotation timelines (session length, transition counts per tier, individual-build percentage, dominant-share, Mann-Whitney U/p) — grepping this notebook for `ELAN`, `video_time`, `duration`, `transitions_per`, `Individual build`, `Dominant`, `Mann`/`mannwhitney` returns **zero matches**. Nothing in this notebook touches ELAN annotation-tier data or computes session-duration/transition-rate statistics; it only trains/evaluates ML models and n-gram/persistence baselines. Table 10.1 must come from a different, not-yet-analyzed source (outside this notebook and outside the two sibling `after_GL` notebooks per the task framing — possibly an earlier-dated ELAN-processing notebook not in this three-notebook set).

- **This notebook does NOT contain the single "headline configuration per task" reproduction** (Task 1's optimized Transformer with the exact 211-feature/seq=18/k=120 config, and by extension whatever the analogous single-best-config selection is for Tasks 2a–2e) that Table 7.12/7.13's "Config" column names (e.g. `OPTI2_REL transf. s=18` for Task 1, `OPTI transf. s=9` for 2a, `OE+OPTI lstm s=9` for 2e). Those are specific architecture+sequence-length combinations, not simply "the best row in this notebook's full grid" — this notebook computes the full ablation grid (all 7 sensor combos × up to 4 models × several k-values), and Table 7.12's precise configs must be selected/reproduced by whichever of the two sibling notebooks is actually named for that purpose (**`naive5_best_per_task.ipynb`**, per its very name, is the far more likely source of Table 7.12/7.13's single-headline-config-per-task numbers than this notebook).

## Summary

`naive5_headline_rerun.ipynb` is an editorially self-aware "rerun bundle": its own cell-0 markdown explicitly states it is "copied verbatim from your own notebooks," with only *configuration* narrowed, so that three separately-authored pipelines can each be rerun quickly on a restricted cohort. The mechanism that makes this possible without touching any of the copied pipeline code is a single global monkey-patch installed in cell 4: `pd.read_csv` is wrapped so that, whenever `RUN_ON == "naive"`, any DataFrame it loads that happens to have a `"group"` column is silently filtered down to `NAIVE_GROUPS = {2, 3, 5, 6, 10}` before being returned. This is elegant but also opaque — every downstream `pd.read_csv` call in the notebook (Part 1's interaction/activity feature CSVs, Part 2's normalized-label and sensor-feature CSVs and cached token table, Part 3's normalized-label CSV) is silently affected, and nothing in the copied pipeline code itself has any awareness that it's operating on a subset; `LeaveOneGroupOut` simply produces 5 folds instead of 9 because there are only 5 distinct group values left in the data by the time it sees it. Structurally the notebook has three parts, each traceable via in-code comments and file-path/task-name evidence to specific thesis sections: Part 1 (cells 4–19) reruns the full 6-task × 7-sensor-combo classical+deep recognition ablation (the same grid underlying Tables 7.2–7.7) and is the most plausible raw source of Table 7.11's naive-5 columns (once someone filters its output to `sensor_combo == "OPTI"`, per Table 7.11's own caption) and a consistency-check contributor to Table 7.12's full9 column; Part 2 (cells 20–23) reruns the deterministic back-off n-gram grammar over 6-label activity tokens (explicitly commented as "thesis Table 8.7 headline") and produces Table 8.8's upper block; Part 3 (cells 24–27) reruns the persistence-vs-transition-only analysis (explicitly commented as "thesis Table 8.2") and produces Table 8.8's lower block. It does **not** produce Table 10.1 (a separate ELAN-annotation descriptive-statistics table with no code trace anywhere in this notebook) and does **not** contain the single "headline configuration per task" exact-reproduction code (Table 7.12/7.13's specific per-task architecture+seq-length configs) that is a far better match for the sibling `naive5_best_per_task.ipynb`. Reproducibility-wise this notebook is genuinely a *rerun*, not a fresh derivation — it depends on the same upstream feature-engineering CSVs (`binary_5s_specialized_oe_merged_all_features.csv`, `activity3_advanced_merged_10s_features.csv`, `rq3_normalized_labels_full.csv`, etc.) that the full-cohort notebooks already require, so it inherits whatever circularity or provenance gaps those upstream stages have, but does not introduce a new one of its own (it does not, e.g., read a results CSV and rebrand it — it retrains everything from the same raw per-window features, just on fewer groups). It does, however, introduce two concrete reproducibility hazards a porter must fix rather than reproduce faithfully: (1) the `SAVE_FAST_DL_PREDICTIONS` contradiction between cells 11 and 17 that silently drops the deep-model per-group variance report on every straight top-to-bottom run, and (2) the complete absence of `RUN_ON`/`TAG` from every intermediate output path (only the final `naive5_results_{TAG}.json` is safely distinguishable between a `"naive"` and an `"all"` run), meaning a naive5 run silently clobbers a prior full9 run's intermediate CSVs at the same paths. For porting into `src/eval/naive_cohort.py`: the single reusable idea worth extracting is the "restrict-then-rerun-the-existing-pipeline" pattern itself (already validated independently by this repo's own `src/models/task1.py::run_naive5_reproduction` and `src/models/task3_naive5.py`, both of which take the same approach of filtering an already-loaded DataFrame to `NAIVE_GROUPS` before feeding it to unmodified full-cohort code) — but this notebook additionally supplies the previously-unfetched **Task 3 naive-5 source** that `task3_naive5.py`'s own docstring flags as unverified ("no dedicated `naive5_*.ipynb` notebook was fetched for Task 3 ... Treat results from this module as unverified against the original naive5 notebooks specifically"): this notebook's Parts 2–3 are exactly that missing source, and its `REPORT_REFERENCE` table in `task3_naive5.py` (h=1..5 back-off, repeat-current, no-self n-gram values keyed by G2/G3/G5/G6/G10) should now be cross-checked line-for-line against this notebook's cell 22–27 pipeline logic (token-building via `channel_statistics`/`build_fullstat_tokens`, back-off via `fit_ngram`/`predict_ngram`, persistence via the inline `ex`/`tr_tab` loop) to confirm the already-ported module reproduces this notebook's exact algorithm, not just its published numbers.
