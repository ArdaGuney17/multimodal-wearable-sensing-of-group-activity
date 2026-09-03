# Analysis: seven_winners_naive5.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\seven_winners_naive5.ipynb`
Code-only extract: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\seven_winners_naive5_CODE_ONLY.py` (2058 lines)

Cell numbers below are **1-indexed notebook positions** (code + markdown cells together, matching the
`.ipynb`'s own `cells[]` order) — this equals `CODE_ONLY_CELL_N + 1` in the code-only extract's own
`# --- CELL N (code cell #M) ---` labels (verified against the extract's content at multiple points).

## Cell counts

Total cells: 28
Code cells: 23
Markdown cells: 5

## Import statements (deduplicated)

```python
from google.colab import drive
from IPython.display import display
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset, DataLoader
import gc
import glob
import json
import math
import numpy as np
import os
import pandas as pd
import random
import re
import time
import torch
import torch.nn as nn
import warnings
```

Note: many of these are re-imported redundantly cell-to-cell (e.g. `import os, pandas as pd` appears
at the top of cells 3, 4, 6, 7, 9, 25, 26, 27, and `from sklearn.model_selection import
LeaveOneGroupOut` is imported fresh in both cell 16 and cell 21) — a symptom of this notebook being
assembled by pasting together blocks from several source notebooks (exploration cells, the
`task1_full_comparison`-style pipeline, and a separate Task-3 grammar pipeline) rather than being
written as one coherent script.

## Markdown headers / outline (in order)

- (cell 1, H1) **Seven thesis winners, re-run on the five naive groups** — full text quoted below
- (cell 15, H2) Deep-learning helpers and the single-run function
- (cell 18, H2) Experiments 1-6 — recognition winners
- (cell 20, H2) Experiment 7 — next-activity grammar
- (cell 24, H2) Final comparison table

Full text of cell 1 (verbatim, this is the notebook's own framing and is essential context for
Section 9 below):

> # Seven thesis winners, re-run on the five naive groups
>
> Runs **only** the best configuration for each of the seven experiments — six
> deep-learning recognition runs called directly (no sweep) plus the activity-token
> grammar. Six DL runs total, roughly 2-4 minutes.
>
> Set `RUN_ON` in the switch cell: `"naive"` for the five naive groups,
> `"all"` for the full nine.
>
> **Note on experiment 1.** The thesis headline for interaction detection is
> 0.8064 / 0.8062 using `OPTI2_RELATIVE_ONLY`, a tuned feature set that does not
> exist in this notebook. This runs the reproducible ablation winner instead
> (OPTI + Transformer, seq=18), whose full-9 reference is 0.751 / 0.751.

This is the single most important sentence in the notebook: it is a **self-declared, deliberate
substitution** — "experiment 1" in this notebook's "seven winners" is *not* the thesis's actual
Task-1 headline (the exact `OPTI2_RELATIVE_ONLY` Transformer reproduced by
`src/models/task1.py::run_exact_reproduction`, ≈0.8064/0.8062), but the weaker "ablation" config
(plain `OPTI` sensor family, Transformer, seq_len=18) that Table 7.12 lists as a *separate,
second* row ("Task 1: interaction (ablation)").

## Hardcoded file paths found in code

### Diagnostic/exploration reads (cells 3-8) — NOT part of the "seven winners" pipeline

These early cells only print/inspect pre-computed publication CSVs to help the notebook's author
locate data; nothing they load feeds into the actual seven-winners computation in cells 9-28:

- `{R}/PUBLICATION_TASK3_FULL_COMPARISON/task3_token_neural_fold_metrics.csv`
- `{R}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/task3_grammar_long_history_common_targets_fold_metrics.csv`
- `{R}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/appendix_r_legacy_fold_metrics.csv`
- `{R}/RQ3_7LABEL_HISTORY_AWARE_PREDICTION/rq3_7label_history_aware_fold_std.csv`
- `{R}/RQ3_7LABEL_SEGMENT_SENSOR_FORECAST/segment_sensor_forecast_fold_std.csv`
- `{R}/INTERACTION_ENG3/task3_expanding_prefix_segment_prediction_fold_std.csv`
- `{R}/RQ3_7LABEL_SEGMENT_SENSOR_FORECAST/*.csv` (glob)
- `/content/drive/MyDrive/**/*.ipynb`, `**/*.py`, `**/*.csv`, `**/*.json` (glob, searching all of
  Drive for files containing `"OPTI2"`, `"RELATIVE_ONLY"`, `"0.8062"`, `"0.8064"` — a manual audit
  trail of where the exact Task-1 headline number lives elsewhere in Drive)
- `{R}/INTERACTION_BINARY_5S_SPECIALIZED_OE/PUBLICATION_TASK1_OPTIMIZED_0806/task1_optimized_0806_fold_metrics.csv`
- `{R}/INTERACTION_BINARY_5S_SPECIALIZED_OE/PUBLICATION_TASK1_OPTIMIZED_0806/task1_optimized_0806_summary_with_std.csv`
- `{R}/INTERACTION_BINARY_5S_SPECIALIZED_OE/DEEP_SEQUENCE_FINE_TUNING_NO_ELAPSED/fine_tuned_deep_sequence_summary.csv`
- `{R}/INTERACTION_BINARY_5S_SPECIALIZED_OE/DEEP_SEQUENCE_MODELS_NO_ELAPSED/deep_sequence_no_elapsed_summary.csv`
- `{R}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/task3_grammar_primary_common_targets_fold_metrics.csv`

  (`R = "/content/drive/MyDrive/thesis/data"` throughout.) The cell-7 read of
  `task1_optimized_0806_fold_metrics.csv` is a useful cross-check: its printed contents (9 rows,
  one per test group, `test_group`/`validation_group`/`accuracy`/`macro_f1`/...) are exactly the
  shape `src/models/task1.py::run_exact_reproduction` writes — confirming this notebook is aware of
  and consistent with that already-ported exact-reproduction pipeline, but does not re-run it.

### Actual "seven winners" pipeline INPUT (cells 9-23) — recomputes from per-window feature CSVs

This is the answer to "does it consume Task 1/2/3's already-computed full-cohort results, or
recompute from scratch restricted to naive groups?": **it recomputes from scratch**, from the same
kind of raw per-window feature CSVs Task 1/2's own full pipelines consume — it does not read Task
1/2's already-computed result tables as input.

- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"` (cell 11)
- `ACTIVITY_DATA_PATH = f"{DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv"` — 10s activity-recognition feature table (992 rows × 1543 cols per executed output)
- `INTERACTION_DATA_PATHS = [f"{DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv", f"{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv"]` — 5s binary-interaction feature table (4579 rows × 1515 cols); first-existing wins
- `OUT_DIR = f"{DATA_ROOT}/ALL_SENSOR_MULTI_TASK_ABLATIONS"` (cell 11) — output root for the recognition-winners pipeline
- `FAST_DL_OUT_DIR = os.path.join(OUT_DIR, "FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED")` (cell 17) — created via `os.makedirs` but nothing is ever written directly into it in visible code (results are collected in memory and saved by cell 19 instead)
- `NORM = os.path.join(DATA_ROOT, 'RQ3_LABEL_NORMALIZATION', 'rq3_normalized_labels_full.csv')` (cell 21) — the 6-label process-category ground truth for the grammar experiment
- `FEATURE_CANDIDATES = [os.path.join(DATA_ROOT, 'INTERACTION_ENG3', 'interaction_eng3_features.csv'), os.path.join(DATA_ROOT, 'INTERACTION_OE10', 'interaction_oe10_10s.csv')]` (cell 21) — sensor features merged onto the activity tokens (first-existing wins)
- `CORE_OUT = os.path.join(DATA_ROOT, 'PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS')` (cell 20)
- `TOKEN_PATH = os.path.join(CORE_OUT, 'activity_tokens_6label_fullstat.csv')` (cell 21) — **cached** activity-token table; if `RESUME_EXISTING` (True) and this file exists, it is *reloaded* via the patched `pd.read_csv` rather than rebuilt from `NORM`/`FEATURE_CANDIDATES`
- `os.path.join(DATA_ROOT, "group_*", "elan", "*.csv")` (cell 26, glob) — raw ELAN annotation exports, one folder per group
- `os.path.join(DATA_ROOT, "group_*", "elan", "*_individual_build_renamed.csv")` (cell 27, glob, filtered to exclude `"BACKUP"`) — the canonical per-group ELAN file actually used for the Table-10.1-style descriptives
- `os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION", "rq3_normalized_labels_full.csv")` (cell 27, coverage check)

### OUTPUT paths

- `{OUT_DIR}/{task_name}/{task_name}_feature_counts.csv` (cell 13) — one row per sensor combo, `n_no_elapsed`/`n_with_elapsed` feature counts, per task
- `{OUT_DIR}/winners_predictions_{TAG}.csv` (cell 19) — row-level predictions (`y_true`,`y_pred`,`group`,`window_start`,...) for all 6 recognition winners, concatenated
- `{OUT_DIR}/winners_{TAG}.csv` (cell 19) — the 6-row recognition-winners summary table (`experiment`,`sensors`,`model`,`full9_A`,`naive5_A`,`full9_M`,`naive5_M`,`dM`,`fold_mean_M`,`fold_sd_M`,`n_folds`)
- `{CORE_OUT}/activity_tokens_6label_fullstat.csv` (cell 21) — cached token table (written only if it didn't already exist)
- `{OUT_DIR}/seven_winners_{TAG}.csv` (cell 24) — **the headline output**: the 6 recognition rows plus the grammar row appended, 7 rows total
- `elan_group_descriptives.csv`, `elan_annotations_all.csv` (cell 27) — written with **bare relative filenames**, not under `DATA_ROOT`/`OUT_DIR` like everything else in the notebook. In Colab this lands in the ephemeral `/content` working directory, not Google Drive — these two files would **not survive a runtime restart**. This looks like an oversight relative to every other `to_csv(...)` call in the notebook, which is consistently rooted at `DATA_ROOT`/`OUT_DIR`.
- **Cell 28 (the normalized-label recompute that actually reproduces Table 10.1's numbers) never calls `to_csv` at all** — its `d2`/`ph2`/`sh` DataFrames are printed but not persisted anywhere. See Section 8/9.

`TAG = "full9" if RUN_ON == "all" else "naive5"` (cell 9) — so every `{TAG}`-suffixed filename above
is literally named `..._full9.csv` or `..._naive5.csv` depending on the switch. **The persisted
outputs in this saved notebook were all written with `TAG = "full9"`** (see Section 8/9) — i.e. the
files that exist on Drive from this notebook, if anywhere, are `winners_full9.csv`,
`winners_predictions_full9.csv`, `seven_winners_full9.csv`; no `..._naive5.csv` counterpart is
evidenced by this notebook's saved run.

## Function / class definitions

- cell 9, function `_gid(g)` — strips non-digit characters from a group label (e.g. `"group_2"`, `2`, `"2"`) and returns the int group id, or `None` if no digits found.
- cell 9, function `_read_csv(*a, **kw)` — monkeypatches `pd.read_csv` (assigned over `pd.read_csv` itself, guarded by a one-shot `pd._naive_patched` flag): calls the real reader, and if `RUN_ON == "naive"` and the result has a `"group"` column, filters rows to `NAIVE_GROUPS` in place and prints a `[naive5] <file> N -> M` line. This is the mechanism that makes *every* downstream `pd.read_csv` call in the notebook (the activity/interaction feature CSVs in cell 13, the cached token table in cell 21, the ELAN files in cells 26-27 that use plain `pd.read_csv`) automatically cohort-restricted without any of those cells needing their own filtering logic.
- cell 9, function `log_result(part, name, **kv)` — appends `{"config": name, **kv}` to `RESULTS[part]` (a module-level dict-of-lists) and prints a `[logged] ...` line; the single place all headline numbers in this notebook get collected.
- cell 10, function `variance_report(y_true, y_pred, groups, name, part, n_boot=2000, seed=0)` — groups predictions by `groups`, computes per-group accuracy/macro-F1, pooled accuracy/macro-F1, fold mean±SD, a t-distribution 95% CI (`scipy.stats.t.interval`), and a 2000-resample bootstrap 95% CI over the per-group macro-F1 values; prints a table and summary line, calls `log_result(...)`, and returns the per-group DataFrame. This is the shared metric-reporting function used by both the recognition winners (cell 19) and the grammar sweep (cell 23).
- cell 12, function `safe_name(x)` — collapses a string to `[A-Za-z0-9_]` tokens (for safe column-name suffixes).
- cell 12, function `unique_feats(feats)` — de-duplicates a list while preserving order (`dict.fromkeys`).
- cell 12, function `find_first_existing(paths)` — returns the first path in a list that exists on disk, else `None`.
- cell 12, function `infer_window_seconds(df, start_col, end_col, default)` — median `end-start` duration across rows, falling back to `default`.
- cell 12, function `add_elapsed_min(df, group_col, start_col, end_col)` — adds an `elapsed_min` column = minutes since the group's first window (or passes through if the column already exists).
- cell 12, function `looks_bad_feature_name(c)` — True if a column name contains a label/metadata token (`label`,`target`,`class`,`group`,`window`,`time`,`elapsed`,`pred`,`prediction`,`correct`,`fold`,`split`,`index`).
- cell 12, function `is_oe_feature(c)` / `is_opti_feature(c)` / `is_xsens_feature(c)` — prefix/token-based classifiers that assign a column name to the OE ("OpenEarable"), OPTI (OptiTrack), or XSens sensor modality (with fallback token lists for older un-prefixed OptiTrack columns).
- cell 12, function `clean_feature_list(dataframe, feats, label_cols=None, include_elapsed=False)` — drops metadata/label columns, drops `elapsed_min` unless `include_elapsed`, drops columns with fewer than 20 finite values or near-zero variance; returns a de-duplicated survivor list.
- cell 12, function `make_classical_models(n_classes)` — returns a dict of 5 sklearn classifiers (`logreg_C1`, `linearSVC_C1`, `rbfSVC_C1_gscale`, `rf_leaf2`, `extraTrees_leaf1`) with fixed hyperparameters; **defined but never actually invoked anywhere else in this notebook** — classical models are set up (`RUN_CLASSICAL = True` in cell 11) but the "seven winners" pipeline (cell 19) only calls `train_one_fast_dl_run`, which is DL-only. See Section 8.
- cell 12, function `_clip_for_float32(X, limit=1e6)` — clips post-`RobustScaler` values to ±1e6 and replaces non-finite values, to prevent float32 overflow downstream in tree models / torch tensors.
- cell 12, function `build_pipeline(model, k, n_features)` — builds an sklearn `Pipeline` (median imputer → RobustScaler → clip → optional `SelectKBest` → model); same dead-code status as `make_classical_models` (never called from the seven-winners path).
- cell 12, function `metric_dict(y_true, y_pred, label_order, prefix="")` — accuracy/macro-F1/balanced-accuracy plus per-label precision/recall/F1/support, as a flat dict.
- cell 13, function `get_modality_features(df, label_cols=None)` — applies `is_oe_feature`/`is_opti_feature`/`is_xsens_feature` to a DataFrame's columns and makes the three modality sets mutually exclusive.
- cell 13, function `make_combo_feature_sets(df, modality_features, label_cols=None)` — builds the 7 `SENSOR_COMBINATIONS` feature lists (both with and without `elapsed_min`).
- cell 13, function `prepare_task(task_name)` — a 6-branch dispatcher (`interaction_vs_noninteraction`, `conversation_vs_nonconversation`, `conversation_vs_building`, `conversation_vs_merging`, `merging_vs_building`, `three_class_activity`) that filters `activity_df_raw`/`interaction_df_raw` to the right label subset, builds a binary/ternary `task_label` target column, and returns a `spec` dict (df, label order, group/start/end columns, candidate seq_lens, modality/combo feature lists, output dir). Unlike the sibling `task1_full_comparison_classical_elapsed_dl_with_std.ipynb` (where 5 of 6 branches are dead code because `RUN_TASKS` only enables one), **this notebook's `RUN_TASKS` (cell 11) includes all 6 tasks and all 6 are actually driven through `prepare_task` and later through the winners loop** — this is the one place the notebook genuinely runs the full breadth of Table 7.12's experiments.
- cell 14, function `sanitize_spec(spec)` — replaces `+/-inf` with NaN across every feature referenced in `spec["combo_features"]`/`combo_features_elapsed`, drops features that become entirely NaN or constant after that, and mutates `spec` in place (removing dead features from both combo dicts); returns the list of columns that had contained inf.
- cell 16, class `RNNClassifier(nn.Module)` — single (Bi)LSTM/GRU layer → LayerNorm+Dropout+Linear head on the last timestep's hidden state.
- cell 16, class `PositionalEncoding(nn.Module)` — standard sinusoidal positional encoding buffer.
- cell 16, class `TransformerClassifier(nn.Module)` — linear input projection → positional encoding → `nn.TransformerEncoder` (pre-norm, GELU) → LayerNorm+Dropout+Linear head on the last timestep.
- cell 16, function `set_seed(seed)` — seeds `random`, `numpy`, `torch` (+ `torch.cuda` if available).
- cell 16, function `build_model(model_cfg, input_dim, n_classes)` — dispatches on `model_cfg["model_type"]` to construct one of the above 3 architectures (bilstm = `RNNClassifier(..., bidirectional=True)`).
- cell 16, function `make_loader(X, y, batch_size=64, shuffle=False)` — wraps arrays in a `TensorDataset`/`DataLoader`.
- cell 16, function `make_sequences(X, y, groups, starts, seq_len)` — per-group, time-sorted sliding-window sequence builder (label = the *last* window's label in each `seq_len`-length window); groups with fewer than `seq_len` rows are skipped entirely.
- cell 16, function `torch_predict(model, X, batch_size=512)` — batched inference, returns argmax class ids; **defined but unused** by the fast path (superseded by `torch_predict_fast` in cell 17).
- cell 16, function `choose_validation_group(train_groups, y_all, groups_all)` — picks, among the training groups, the one with (in priority order) at least 2 classes present, then the most rows — used as the LOGO inner validation split.
- cell 17, function `should_run_fast_dl_task(task_name)` / `should_run_fast_dl_sensor(sensor_combo)` — allow-list gates against `FAST_DL_TASKS_TO_RUN`/`FAST_DL_SENSOR_COMBOS_TO_RUN` (both `None` here, so both always return True); **unused by the cell-19 direct-call path**, which bypasses these entirely.
- cell 17, function `choose_fast_seq_len(spec)` — picks `spec["seq_lens"][-1]` (i.e. "last"/longest context) given `FAST_DL_SEQ_CHOICE = "last"`; **unused by cell 19**, which hardcodes each experiment's `seq_len` directly in the `WINNERS` dict instead.
- cell 17, function `get_fast_model_configs()` — filters `MODEL_CONFIGS` down to `FAST_DL_MODEL_TYPES`; **unused by cell 19** (same reason).
- cell 17, function `torch_predict_fast(model, X, batch_size=FAST_PRED_BATCH_SIZE)` — same as `torch_predict` but with the larger prediction batch size; this one IS used, inside `train_one_fast_dl_run`.
- cell 17, function `train_one_fast_dl_run(spec, sensor_combo, seq_len, k_features, model_cfg, seed)` — **the actual workhorse**: full LOGO loop over `spec["df"]`'s groups — per fold, picks a validation group via `choose_validation_group`, imputes+scales+clips+`SelectKBest`-selects features (fit on train only), builds sliding-window sequences via `make_sequences`, trains the DL model with early stopping on validation macro-F1 (`FAST_PATIENCE`, up to `FAST_MAX_EPOCHS`), evaluates on the held-out test group, and accumulates per-fold rows and (optionally) row-level predictions. Returns `(summary_dict, fold_df, pred_df)` or `(None, None, None)` if nothing was evaluable. This is the single function the "seven winners" loop calls once per experiment with no sweep — the direct analogue of `src/models/task1.py`'s DL training loop, but generalized across all 6 recognition tasks in this repo's terms rather than Task 1 only.
- cell 22, function `first_existing(columns, candidates)` — like `find_first_existing` but over a column-name list rather than filesystem paths.
- cell 22, function `load_six_label_windows()` — reads `NORM` (per-window `rq3_process_label`), merges `social_conversation`/`task_conversation` into `conversation` (the `MERGE6` dict), left-joins in numeric sensor columns from the first existing `FEATURE_CANDIDATES` file keyed on `(group, rounded time)`; returns `(df, sensor_cols)`.
- cell 22, function `channel_statistics(values)` — 14 summary statistics per channel per segment (mean/std/min/max/range/median/iqr/p10/p25/p75/p90/energy/rms/spectral entropy via FFT power spectrum).
- cell 22, function `build_fullstat_tokens(window_df, sensor_cols)` — groups by `group`, splits into contiguous same-label runs ("segments"/tokens) via `(label != label.shift()).cumsum()`, computes `channel_statistics` per segment per sensor channel, returns the token table + feature column list. This is what produces the "244 tokens" (full cohort) / "116 tokens" (naive subset, via the read_csv patch) referenced in Table 8.7/8.8.
- cell 23, function `fit_ngram(train_seqs, max_h)` — builds back-off n-gram count tables for orders `1..max_h` plus a unigram fallback, from a list of per-group label sequences.
- cell 23, function `predict_ngram(hist, tabs, uni, max_h)` — classic back-off: tries the longest available context first (`min(max_h, len(hist))` down to 1), falls back to the global most-common label (or the literal string `"conversation"` if the unigram table is empty).
- cell 26, function `peek(path, pats)` — reads a file as text and returns which of a pattern list appear in it (used for the earlier Drive-wide OPTI2 search in cell 6, reused here); note this is a dead leftover reference in cell 26's own logic — cell 26 itself doesn't call `peek`, it was defined in cell 6 and is simply still in scope.
- cell 28, function `norm_label(s)` — regex-based typo-robust mapper from the 280 raw ELAN activity-label strings onto 8 process categories (`merging`, `co_building`, `individual_build`, `object_handover`, `conversation`, `inspection`, `moving_transport`, `sync_marker`) plus an `"other"` catch-all; this is the function that makes cell 28's descriptives match the thesis's Table 10.1 numbers exactly (see Section 9), where cell 27's un-normalized version does not.

## Hyperparameter-looking constants (verbatim)

```python
# cell 9 — THE ONLY SWITCH YOU CHANGE
RUN_ON = "all"          # "all" = original 9 groups | "naive" = 5 naive groups
NAIVE_GROUPS    = {2, 3, 5, 6, 10}
RESEARCHER_GRPS = {1, 7, 8, 9}
TAG = "full9" if RUN_ON == "all" else "naive5"

# cell 11 — CONFIG
DATA_ROOT = "/content/drive/MyDrive/thesis/data"
RUN_TASKS = [
    "interaction_vs_noninteraction", "conversation_vs_nonconversation",
    "conversation_vs_building", "conversation_vs_merging",
    "merging_vs_building", "three_class_activity",
]
SENSOR_COMBINATIONS = {
    "OE": ["OE"], "OPTI": ["OPTI"], "XSENS": ["XSENS"],
    "OE_OPTI": ["OE", "OPTI"], "OE_XSENS": ["OE", "XSENS"],
    "OPTI_XSENS": ["OPTI", "XSENS"], "OE_OPTI_XSENS": ["OE", "OPTI", "XSENS"],
}
RUN_CLASSICAL = True
RUN_DL = True
K_CLASSICAL = [40, 80, 120, 200, "all"]
TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]
K_DL = [80, 120, 200]
SEEDS = [42]
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 64
RUN_DL_MODEL_TYPES = ["lstm", "bilstm", "gru", "transformer"]
STOP_AFTER_N_DL_RUNS_PER_TASK_SENSOR = None
RANDOM_STATE = 42

# cell 12 — classical model hyperparameters (defined, never invoked — see Section 8)
BAD_TOKENS = ["label", "target", "class", "group", "window", "time", "elapsed",
              "pred", "prediction", "correct", "fold", "split", "index"]
LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                    solver="liblinear" if n_classes == 2 else "lbfgs", random_state=RANDOM_STATE)
LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000, dual=False)
SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

# cell 16 — DL architectures
ALL_MODEL_CONFIGS = [
    {"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "bilstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4},
]

# cell 17 — fast single-run DL settings
FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]
FAST_DL_K = 120
FAST_DL_SEQ_CHOICE = "last"
FAST_MAX_EPOCHS = 25
FAST_PATIENCE = 4
FAST_BATCH_SIZE = 256
FAST_PRED_BATCH_SIZE = 1024
MAX_LOGO_FOLDS = None
SAVE_FAST_DL_PREDICTIONS = False   # overridden to True in cell 19

# cell 19 — THE SEVEN WINNERS TABLE (the load-bearing constant of the whole notebook)
# task -> (sensor_combo, model_type, seq_len, k, full9_A_reference, full9_M_reference)
WINNERS = {
 "interaction_vs_noninteraction":   ("OPTI",       "transformer", 18, 120, 0.751,  0.751),
 "conversation_vs_nonconversation": ("OPTI",       "transformer",  9, 120, 0.880,  0.845),
 "conversation_vs_building":        ("OPTI_XSENS", "bilstm",       9, 120, 0.881,  0.860),
 "conversation_vs_merging":         ("OE_OPTI",    "transformer",  9, 120, 0.879,  0.866),
 "merging_vs_building":             ("OPTI",       "transformer",  9, 120, 0.811,  0.738),
 "three_class_activity":            ("OE_OPTI",    "lstm",         9, 120, 0.7565, 0.6994),
}
SEED = 42

# cell 20 — Part 2 (grammar) config
TOKEN_SEEDS = [42, 1, 7]
SEED_STD_DDOF = 0
FOLD_STD_DDOF = 1
K_SELECT = 40
GRAMMAR_ORDERS = [1, 2, 3, 5, 10]
HYBRID_ORDER = 3
HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)

# cell 23 — grammar sweep orders actually run
for H in [1, 2, 3, 5]: ...

# cell 24 — hardcoded reference for the 7th ("grammar") winner
"full9_A": 0.604, "full9_M": 0.499   # = Table 8.7's n-gram h=2 published numbers

# cell 26/27 — cohort definitions re-declared (values identical to cell 9)
NAIVE, RESEARCHER = {2, 3, 5, 6, 10}, {1, 7, 8, 9}
```

`NAIVE_GROUPS = {2, 3, 5, 6, 10}` is declared **three times** in this notebook (cell 9 as
`NAIVE_GROUPS`, cell 26/27 as `NAIVE`) — all three declarations use the identical group-ID set, and
this matches `NAIVE_GROUPS` in both `src/models/common.py` (imported by `task1.py`) and
`task3_naive5.py` exactly.

## Potentially broken / disabled cells

- **The notebook's own persisted outputs show it was executed with `RUN_ON = "all"` (TAG =
  `"full9"`), never with `RUN_ON = "naive"`.** Cell 9's saved output literally prints
  `RUN_ON = all | TAG = full9`, and every downstream saved output (cell 19, cell 24/25) is
  self-consistent with a 9-fold, full-cohort run — see Section 9 for full evidence. **This means the
  notebook, as saved, never actually computed or persisted the naive-5 numbers that are the entire
  stated purpose of the file.** Re-running with the switch flipped to `RUN_ON = "naive"` is required
  to get real naive5 numbers out of this code; nothing in the current `.ipynb` file demonstrates that
  was ever done.
- **`make_classical_models`, `build_pipeline`, `should_run_fast_dl_task`, `should_run_fast_dl_sensor`,
  `choose_fast_seq_len`, `get_fast_model_configs`, and `torch_predict` (cells 12, 17) are all defined
  but never called** by the actual "seven winners" path (cell 19 calls `train_one_fast_dl_run`
  directly with a hardcoded `(sensor, model_type, seq_len, k)` per experiment, bypassing the sweep
  infrastructure those functions exist to support). `RUN_CLASSICAL = True` (cell 11) is likewise
  never acted upon — no classical model is trained anywhere in this notebook, despite the
  classical-model machinery being fully present. This mirrors the sibling
  `task1_full_comparison_classical_elapsed_dl_with_std.ipynb`'s "dead-config" pattern but is more
  extreme here: an entire model family (classical ML) is wired up and then unused.
- **Cell 24's hardcoded grammar reference (`full9_A=0.604, full9_M=0.499`, taken from Table 8.7) does
  not match this same notebook's own freshly-computed full9 grammar run one cell earlier** (cell 23's
  printed output: `pooled=0.460 | fold mean+/-SD=0.392+/-0.068`, i.e. pooled accuracy≈0.596,
  macro-F1≈0.460, versus the hardcoded 0.604/0.499). Since back-off n-gram counting is fully
  deterministic (no seed/model randomness involved), this ~0.04 macro-F1 gap between "the number the
  notebook cites as ground truth" and "the number the notebook itself just computed from the same
  244-token table" is a genuine, unexplained reproduction discrepancy — not run-to-run noise. Worth
  investigating (e.g. a difference in how tokens/segments were built, or a different label-merge rule)
  before treating either number as authoritative in a port.
- **Cell 27 (raw, un-normalized ELAN label descriptives) produces numbers that do NOT match thesis
  Table 10.1** (e.g. its G1 `transitions=198` vs. Table 10.1's G1 `134`) — it is effectively a wrong
  first attempt, and it is the one that gets saved to disk (`elan_group_descriptives.csv`,
  `elan_annotations_all.csv`, cell 27). **Cell 28 (which applies `norm_label()` to collapse the 280
  raw label strings onto 8 process categories first) reproduces Table 10.1 exactly** — session
  minutes, transition counts, transitions/min, dominant share, and the cohort-level
  `researcher=4.66 / naive=3.21 changes/min, U=19.0, p=0.032` all match Table 10.1's published values
  verbatim (see Section 9) — **but cell 28 never calls `to_csv`, so its (correct) output is never
  persisted to a file**, only printed to notebook stdout. A port must use cell 28's logic
  (`norm_label` + the `d2`/`rows2` recompute), not cell 27's, and must add the missing save step.
- **Cell 27's two `to_csv` calls use bare relative filenames** (`"elan_group_descriptives.csv"`,
  `"elan_annotations_all.csv"`) instead of being rooted at `DATA_ROOT`/`OUT_DIR` like every other
  output in the notebook — in Colab this writes into the ephemeral `/content` filesystem, not Drive,
  so these files would not survive a session restart. Likely an oversight, and moot anyway since it's
  cell 27's (wrong) numbers being saved, not cell 28's (correct) ones.
- **`TOKEN_PATH` caching (`RESUME_EXISTING = True`, cell 21) silently reloads a pre-existing
  `activity_tokens_6label_fullstat.csv` instead of rebuilding it.** This is not a bug — the reload
  goes through the patched `pd.read_csv`, so a naive-cohort run would still correctly filter the
  cached (full, 244-token) file down to 116 naive tokens at read time — but it does mean the "tokens:
  244" vs. "tokens: 116" figure printed by cell 22 is the only visible confirmation of which cohort
  was actually used for a given run, and it's worth flagging for anyone debugging a future run of
  this notebook.
- No `!pip install` cells; `drive.mount` (cell 3) is a bare call with no try/except, so this notebook
  is **not** Colab-optional the way `task1_full_comparison...ipynb` is — it will raise if
  `google.colab` isn't importable.
- Mixed English/Turkish comments/strings persist in the exploration cells (`"eksikler"`, `"nerede?"`,
  `"ozet:"`, `"YOK"` for "not found") in cells 4-6 — cosmetic, but flag for cleanup if any of that
  exploration logic is ever ported (it likely shouldn't be; see Section 9).

## WHICH TASKS/TABLES DOES THIS NOTEBOOK ACTUALLY COVER

**This notebook targets Table 7.12 (6 of its 7 rows) + Table 8.7's headline row (as an appended 7th
row) — NOT Table 7.11, and its saved/persisted run only ever produced the full-9 (not naive-5)
version of that table. Table 10.1 is separately, and correctly, reproduced by cell 28's code (not
cell 27's), though cell 28's output is never saved to a file. Table 7.13's per-group breakdown could
be produced by this notebook's own `variance_report()` per-group DataFrames if run with
`RUN_ON="naive"`, but that run does not exist in this saved copy.**

Working through the evidence:

**1. "Seven winners" = 6 of Table 7.12's 7 rows, plus Table 8.7's headline as a substitute 7th row.**
Table 7.12 (`docs/thesis_reproduction_targets.md` lines 896-908) has exactly 7 rows: "Task 1
(optimised)", "Task 1 (ablation)", and Task 2a-2e. This notebook's `WINNERS` dict (cell 19, quoted in
full above) has exactly 6 entries — one per recognition task — and its first entry
(`interaction_vs_noninteraction: (OPTI, transformer, 18, 120, 0.751, 0.751)`) matches Table 7.12's
**"Task 1: interaction (ablation)"** row (`Full9 A=0.751, Full9 M=0.751`) *exactly*, not the
"(optimised)" row (`0.806/0.806`). The remaining 5 `WINNERS` entries match Table 7.12's 2a-2e rows'
`Config` and `Full9 A`/`Full9 M` columns exactly:

| `WINNERS` key | `(sensor, model, seq, k)` | notebook full9 (A, M) | Table 7.12 row | Table 7.12 (Full9 A, M) |
|---|---|---|---|---|
| `interaction_vs_noninteraction` | OPTI, transformer, 18, 120 | 0.751, 0.751 | Task 1 (ablation) | 0.751, 0.751 |
| `conversation_vs_nonconversation` | OPTI, transformer, 9, 120 | 0.880, 0.845 | Task 2a | 0.880, 0.845 |
| `conversation_vs_building` | OPTI+XSENS, bilstm, 9, 120 | 0.881, 0.860 | Task 2b | 0.881, 0.860 |
| `conversation_vs_merging` | OE+OPTI, transformer, 9, 120 | 0.879, 0.866 | Task 2c | 0.879, 0.866 |
| `merging_vs_building` | OPTI, transformer, 9, 120 | 0.811, 0.738 | Task 2d | 0.811, 0.738 |
| `three_class_activity` | OE+OPTI, lstm, 9, 120 | 0.7565, 0.6994 | Task 2e | 0.756, 0.699 |

Cell 24 then appends a 7th row, `"next_activity (n-gram h=2)"`, with hardcoded reference
`full9_A=0.604, full9_M=0.499` — this is Table 8.7's headline row verbatim (`docs/
thesis_reproduction_targets.md` line 1143: `n-gram Markov, h=2 (back-off) | 0.604 | 0.499 |
0.426±0.087 [0.359,0.493]`). So: **Table 7.12's own "Task 1 (optimised)" row is the one row of that
table this notebook deliberately does not attempt** (see the markdown cell-1 disclaimer quoted
above), and its place as the 7th "winner" is instead filled by Task 3's grammar headline (Table 8.7),
which is not itself a row of Table 7.12 at all.

**2. Confirmed by executed cell output that this run was full-9, not naive-5.** Cell 9's saved
output:
```
RUN_ON = all | TAG = full9
naive groups: [2, 3, 5, 6, 10]
NOTE: 5 groups -> LOGO gives 5 folds instead of 9.
```
And cell 19's per-experiment output consistently shows `n_folds=9` and 9-row per-group tables (e.g.
for `interaction_vs_noninteraction`: groups `1,2,3,5,6,7,8,9,10`, 9 rows) — i.e. researcher groups
1/7/8/9 were included, meaning `RUN_ON` was `"all"` at execution time, not `"naive"`. Cell 25's saved
final table shows `naive5_A`/`naive5_M` columns numerically almost identical to `full9_A`/`full9_M`
(e.g. interaction: `full9_A=0.7510, naive5_A=0.7508`) — because with `RUN_ON="all"`, the code that
populates the `"naive5_*"` column names (`round(summary["accuracy"], 4)`, `round(summary["macro_f1"],
4)` in cell 19) is fed by a run over the **same 9 groups** as the hardcoded `full9_*` reference
values, so the two columns are actually two independent full-9 computations of the same thing (a
self-consistency check), not full-9-vs-naive-5 at all. Compare this to the real naive-5 numbers this
notebook is supposed to produce (Table 7.12 lines 902-908, e.g. Task 1 ablation naive5
`A=0.627,M=0.626`; Task 2c naive5 `A=0.672,M=0.587`) — nothing in this saved notebook comes close to
those, because it never ran with the naive-5 group restriction active.

**3. Table 7.11 is not attempted by this notebook at all.** Table 7.11 (lines 874-885) reports
classical-no-elapsed / classical-with-elapsed / deep-no-elapsed columns for all 6 tasks — this
notebook never runs classical models (see Section 8) and only runs one specific DL config per task
(the "winner"), not a with/without-elapsed comparison. Table 7.11 would need to come from one of the
other `after_GL` sibling notebooks (`naive5_best_per_task.ipynb` most plausibly, given its name).

**4. Table 7.13 (per-group/per-fold breakdown) is structurally reachable from this notebook's own
`variance_report()` output but was not captured for the naive-5 case.** `variance_report()` (cell 10)
already returns and prints a per-group DataFrame with exactly the `group`/`macro_f1` columns Table
7.13 needs. Cell 19's saved (full9) output shows these per-group tables for all 6 recognition
experiments (e.g. `interaction_vs_noninteraction`: group 1→0.568514, 2→0.705473, ..., 10→0.774651) —
but these are the full-9 per-group values, not the naive-5 ones Table 7.13 actually reports (e.g. its
G2/G3/G5/G6/G10 for "Task 1 (ablation)" are `0.666, 0.829, 0.474, 0.407, 0.519`, pooled `0.626` —
none of which appear anywhere in this saved notebook's output). Re-running with `RUN_ON="naive"`
would populate this table for real.

**5. Table 10.1 is reproduced exactly — by cell 28, not cell 27.** Table 10.1 (`docs/
thesis_reproduction_targets.md` lines 1242-1255) reports Session(min)/Transitions/Transitions-per-min/
Dominant-share per group plus researcher-vs-naive cohort means and Mann-Whitney U/p. Cell 28's
`norm_label()`-based recompute (`d2` DataFrame, printed in full) matches **every value in Table 10.1
verbatim**: e.g. G1 `session_min=30.5, transitions=134, trans_per_min=4.39, dominant_share=0.607`
(Table 10.1 G1: `30.5, 134, 4.39, 0.607` exactly); G3 `45.2, 91, 2.02, 0.797` (Table 10.1 G3
identical); and the cohort comparison line prints `trans_per_min researcher=4.66 naive=3.21 U=19.0
p=0.032`, matching Table 10.1's `researcher mean ... 4.66`, `naive mean ... 3.21`, and `U/p ... 19.0 /
0.03` for the "Transitions per min" row exactly, including the single individually-significant
p-value the thesis calls out. Cell 27's un-normalized recompute (`desc` DataFrame, using the raw ELAN
`label` column without `norm_label()`) gives visibly different, wrong numbers (e.g. G1 transitions
`198` instead of `134`) — it is a superseded first pass, not an alternate valid reproduction. **The
mapping from raw ELAN vocabulary to Table-10.1 categories lives entirely in `norm_label()` (cell
28)** and is the piece to port for Table 10.1, not the `COLS`/`tier`-parsing logic in cell 26/27
(which both cells share and which is fine — it's specifically the label-normalization step that
matters).

**Net mapping:**

| Thesis table | Covered by this notebook? | Where | Caveat |
|---|---|---|---|
| Table 7.11 | No | — | Not attempted (no classical models run here) |
| Table 7.12 | Partially — 6 of 7 rows, with row 1 substituted | cells 9-19, 24-25 | Task-1-optimised row swapped for Task-1-ablation row (see markdown cell 1); **saved run is full9 only, naive5 never executed** |
| Table 7.13 | Structurally yes, not executed | `variance_report()` in cells 10/19 | Same full9-only caveat as above |
| Table 8.7 (Task 3, cited for context) | Row cited as hardcoded reference only | cell 24 | Notebook's own live recompute of the same number (cell 23) disagrees with the hardcoded reference by ~0.04 macro-F1 |
| Table 8.8 (naive Task 3) | No | — | Not attempted (grammar sweep in cell 23 is full9 only per the same `RUN_ON` state) |
| Table 10.1 | Yes, exactly | cell 28 (`norm_label`, `d2`) | Correct version not saved to CSV; cell 27's saved CSV is the wrong (un-normalized) version |

## Summary

`seven_winners_naive5.ipynb` is a compact, purpose-built "final answer" notebook: rather than
re-running the full classical+DL sweep machinery, it hardcodes the single best configuration per
experiment (the `WINNERS` dict) and calls the shared DL trainer (`train_one_fast_dl_run`, itself a
near-duplicate of the fast-path trainer in the sibling `task1_full_comparison...ipynb`) exactly once
per experiment, for 6 of Table 7.12's 7 "recognition winner" rows plus, separately, Task 3's
back-off n-gram grammar headline (Table 8.7) as a 7th row — explicitly substituting the ablation
`OPTI`-only Task-1 config for the true `OPTI2_RELATIVE_ONLY` headline number, which the notebook's
own markdown cell says "does not exist in this notebook." The entire naive-vs-full toggle is a single
switch (`RUN_ON`, cell 9) plus a global monkeypatch of `pd.read_csv` that transparently restricts any
CSV with a `group` column to the 5 naive groups — an elegant, low-duplication design in principle.
In practice, however, **the notebook's persisted execution never flipped that switch**: every saved
output (`TAG = full9`, 9-fold tables, `winners_full9.csv`/`seven_winners_full9.csv` filenames) shows
a full-9-cohort run used to self-check the hardcoded reference numbers against a fresh recomputation
— which it mostly does successfully, down to the 3rd-4th decimal — but this means **none of Tables
7.11-7.13 or 8.8's actual naive-5 numbers are demonstrated anywhere in this file**; reproducing them
requires literally re-running this notebook with `RUN_ON = "naive"` and confirming the output matches
the thesis's naive5 columns (e.g. Task 1 ablation naive5 A/M = 0.627/0.626, per Table 7.12). Two
further, independent findings temper how much to trust this notebook as a ground truth: its own
freshly-computed grammar (n-gram h=2) result disagrees with the very Table 8.7 number it hardcodes as
a reference by roughly 0.04 macro-F1 despite that computation being fully deterministic, suggesting
some upstream difference in how the 244-token table was built versus whatever produced the published
number; and its ELAN-based descriptive-statistics section (aimed at Table 10.1) contains two
competing implementations — a broken/superseded raw-label version (cell 27, the one whose output
actually gets saved to CSV, at a bare relative path that wouldn't even survive a Colab session
restart) and a correct, `norm_label()`-normalized version (cell 28) that reproduces Table 10.1's
numbers exactly but is never written to disk at all. For a port into `src/eval/naive_cohort.py`, the
useful pieces are: (1) the `WINNERS` dict as the canonical list of "headline config per recognition
task," to be run against the already-ported `run_all_dl`/task-spec machinery in
`src/models/common.py` with the group set restricted to `NAIVE_GROUPS`, producing Table 7.12/7.13's
naive5 columns for real; (2) `norm_label()` from cell 28, verbatim, as the raw-ELAN-vocabulary-to-
process-category mapping needed to reproduce Table 10.1, paired with cell 27's tier-parsing/file-
selection logic (headerless 9-column ELAN CSV schema, `*_individual_build_renamed.csv` file
selection, per-tier transition counting) but NOT cell 27's un-normalized `desc` computation; and (3)
a clear-eyed acknowledgment, next to whatever the port produces, that this specific notebook's saved
state does not independently verify any naive-5 number — the actual verification has to come from
running the ported code and checking it against the `docs/thesis_reproduction_targets.md` Table
7.11-7.13/10.1 target values directly, the same way `src/models/task1.py::run_naive5_reproduction`
and `src/models/task3_naive5.py` already document doing.
