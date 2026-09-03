# Analysis: naive5_best_per_task.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\naive5_best_per_task.ipynb`
Code-only extract used for line-level references: `naive5_best_per_task_CODE_ONLY.py` (2040 lines). Note: the extract's own `# --- CELL N (code cell #M) ---` numbering is **off by one** from the real notebook's cell positions (the extract only counts from the first code cell; the notebook's cell 1 is a markdown title cell). Throughout this document, **notebook cell number = extract `CELL N` + 1** — verified directly against the `.ipynb` JSON's `cells[]` array (27 cells) rather than assumed.

## Cell counts

Total cells: 27
Code cells: 21
Markdown cells: 6

## Import statements (deduplicated)

```python
import gc
import glob
import json
import math
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from IPython.display import display
from google.colab import drive
from google.colab import files          # inside try/except in the final save cell

from scipy import stats
from scipy.stats import mannwhitneyu

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC, SVC

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
```

Note: `import numpy as np`, `import pandas as pd`, `from collections import defaultdict, Counter`, and `from sklearn.metrics import f1_score, accuracy_score` are each re-imported redundantly in later cells (harmless — this is a notebook stitched together from at least 4 previously-separate notebooks/scripts, evidenced by the repeated `# ===== SETUP =====` / `# CELL 1 - SETUP` style banner comments reappearing mid-file).

## Markdown headers / outline (in order, with real notebook cell numbers)

- (cell 1, H1) `Naive-5 re-run — best configuration per task`
  - Body: "Runs only the six best-per-task recognition configurations (all six winners are deep models, so the classical sweep is skipped entirely), plus the Task 3 forecasting experiments, plus the researcher-vs-naive descriptive comparison. **Set `RUN_ON` in the switch cell.** Run once with `"naive"` and once with `"all"`. Expected runtime: about 3-5 minutes per run."
- (cell 10, H2) `Part 1 — recognition (best config per task)`
- (cell 15, H2) `Part 2 — next-activity grammar`
- (cell 20, H2) `Part 3 — persistence`
- (cell 24, H2) `Part 4 — researcher vs naive descriptives`
- (cell 26, H2) `Save`

## Hardcoded file paths found in code

### INPUT paths — reads pre-computed feature/results CSVs, does NOT recompute features from scratch

All inputs are **already-computed full-cohort feature/label CSVs from earlier pipeline stages** (produced by other, upstream notebooks). This notebook never re-runs feature engineering; it only re-runs the *modeling* stage restricted to the naive-5 groups, and for Part 2 it optionally reuses an already-built Task 3 token table instead of rebuilding it. Naive-group restriction is applied to these inputs **at load time**, not by recomputing anything (see Section 9 below for the mechanism).

- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"` (notebook cell 6)
- `ACTIVITY_DATA_PATH = f"{DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv"` — 10s activity-recognition feature table (Part 1, tasks 2a–2e)
- `INTERACTION_DATA_PATHS = [f"{DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv", f"{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv"]` — first-existing wins; 5s interaction feature table (Part 1, task 1)
- `os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_predictions.csv")` where `FAST_DL_OUT_DIR = os.path.join(OUT_DIR, "FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED")` — read back by the "BEST PER TASK" cell (notebook cell 14) via `_orig_read_csv` (the **unpatched** `pd.read_csv`, deliberately bypassing the naive-filter monkeypatch — see Section 8/9). This is a **self-produced intermediate file from earlier in the same run** (or a prior run), not an external ground-truth CSV — see the circular-dependency finding below.
- `CORE_OUT = os.path.join(DATA_ROOT, 'PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS')`, `TOKEN_PATH = os.path.join(CORE_OUT, 'activity_tokens_6label_fullstat.csv')` (notebook cell 17) — if `RESUME_EXISTING` (`=True`) and the file exists, **reloads the already-built full-cohort token table** rather than rebuilding it from `NORM`/`FEATURE_CANDIDATES`. This path matches the destination `src/models/task3_tokens.get_or_build_tokens` already writes to (confirmed by `task3_naive5.py`'s `core_out` parameter using the identical folder name), so the naive-5 filter is applied to an already-existing cross-cohort artifact at read time.
- `NORM = os.path.join(DATA_ROOT, 'RQ3_LABEL_NORMALIZATION', 'rq3_normalized_labels_full.csv')` — fallback source for token-building if `TOKEN_PATH` doesn't exist yet (notebook cell 17)
- `FEATURE_CANDIDATES = [os.path.join(DATA_ROOT, 'INTERACTION_ENG3', 'interaction_eng3_features.csv'), os.path.join(DATA_ROOT, 'INTERACTION_OE10', 'interaction_oe10_10s.csv')]` — first-existing wins (notebook cell 17)
- `NORM_DIR = os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION")`, `FULL_LABEL_PATH = os.path.join(NORM_DIR, "rq3_normalized_labels_full.csv")`, `META_PATH = os.path.join(NORM_DIR, "rq3_label_normalization_metadata.json")` (notebook cell 21, Part 3 setup)
- `os.path.join(NORM_DIR, "rq3_normalized_labels_full.csv")` re-read a second time in Part 4 (notebook cell 25), this time explicitly via `_orig_read_csv` (unpatched) — see Section 9.

### OUTPUT paths — exact naming

- `OUT_DIR = f"{DATA_ROOT}/ALL_SENSOR_MULTI_TASK_ABLATIONS"` (notebook cell 6) — **a new output directory distinct from** the sibling `task1_full_comparison_classical_elapsed_dl_with_std.ipynb`'s `PUBLICATION_TASK1_FULL_COMPARISON`/`PUBLICATION_TASK1_OPTIMIZED_0806`, and distinct from `src/models/task1.py`'s `run_naive5_reproduction` output folder.
  - `{OUT_DIR}/{task_name}/{task_name}_feature_counts.csv` (notebook cell 8, one per task)
  - `FAST_DL_OUT_DIR = os.path.join(OUT_DIR, "FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED")` (notebook cell 13)
    - `fast_dl_plan.csv`
    - `combined_fast_dl_no_elapsed_summary.csv` (rewritten after every run in the loop, then again at the end)
    - `combined_fast_dl_no_elapsed_fold_metrics.csv`
    - `combined_fast_dl_no_elapsed_predictions.csv` — only written **if `SAVE_FAST_DL_PREDICTIONS` is `True` at the time this cell executes** (see the contradiction flagged in Section 8 — the value in effect by the end of this cell is actually `False`)
    - `combined_fast_dl_no_elapsed_best_per_task_sensor.csv`
  - `os.path.join(OUT_DIR, f"best_per_task_{TAG}.csv")` (notebook cell 14) — `TAG` is `"naive5"` or `"full9"` depending on `RUN_ON`; this is the Table 7.12/7.13-relevant comparison CSV (`cmp_df`)
- `TOKEN_PATH` (see above) — only written if it didn't already exist (`RESUME_EXISTING` branch)
- `"group_descriptives.csv"` (notebook cell 25) — **relative path, written to the notebook's current working directory, not under `DATA_ROOT`** — this is the Table 10.1 raw-ingredients CSV
- `f"naive5_results_{TAG}.json"` (notebook cell 27) — also a **relative path** in the CWD; bundles every `RESULTS[part]` list (from `log_result()`) plus `tag`, `run_on`, `naive_groups`; downloaded via `google.colab.files.download(fn)` if running in Colab

## Function / class definitions

(notebook cell numbers; all function bodies read and summarized, not just "no docstring")

- cell 4 (`THE ONLY SWITCH`): `_gid(g)` — regex-strips non-digits from a group label and returns `int` (or `None`); also patches `pd.read_csv` in place (`_read_csv` closure) so **every** CSV loaded anywhere in the notebook that has a `group` column is auto-filtered to `NAIVE_GROUPS` when `RUN_ON == "naive"`, printing a `[naive5] <file> <before> -> <after>` line each time; `log_result(part, name, **kv)` — appends a dict to the global `RESULTS[part]` list and prints it.
- cell 5 (`variance helper`): `variance_report(y_true, y_pred, groups, name, part, n_boot=2000, seed=0)` — computes per-group accuracy/macro-F1, pooled macro-F1/accuracy, fold mean±SD, a Student-t 95% CI and a 2000-resample bootstrap 95% CI over the per-group macro-F1 values, prints a formatted block, calls `log_result(...)`, and returns the per-group DataFrame. This is the single function that produces every per-group ("per-fold") row later matched against Tables 7.13 and 8.8.
- cell 7 (`GENERAL HELPERS`): `safe_name`, `unique_feats`, `find_first_existing`, `infer_window_seconds`, `add_elapsed_min` (adds an `elapsed_min` = minutes since group's first window-midpoint column), `looks_bad_feature_name`/`is_oe_feature`/`is_opti_feature`/`is_xsens_feature` (column-name-prefix-based modality classifiers), `clean_feature_list` (drops label/meta columns, near-constant columns, and elapsed unless requested), `make_classical_models(n_classes)` (returns logreg/linearSVC/rbfSVC/RF/ExtraTrees dict — **not used in this notebook's actual run**, see below), `_clip_for_float32` (clips post-`RobustScaler` values to ±1e6 to avoid float32 overflow), `build_pipeline(model, k, n_features)` (impute→scale→clip→optional SelectKBest→model), `metric_dict(y_true, y_pred, label_order, prefix="")` (accuracy/macro-F1/balanced-accuracy + per-class precision/recall/F1/support).
- cell 8 (`LOAD DATASETS AND CREATE TASKS`): `get_modality_features(df, label_cols=None)` — splits columns into mutually-exclusive OE/OPTI/XSENS feature lists by prefix; `make_combo_feature_sets(df, modality_features, label_cols=None)` — builds the 7 `SENSOR_COMBINATIONS` feature lists, with and without `elapsed_min`; `prepare_task(task_name)` — builds one of 6 task specs (`interaction_vs_noninteraction`, `conversation_vs_nonconversation`, `conversation_vs_building`, `conversation_vs_merging`, `merging_vs_building`, `three_class_activity`), each returning a dict with `df`, `label_col`, `target_col`, `label_order`, `group_col`, `seq_lens`, `combo_features`, `combo_features_elapsed`, `out_dir`.
- cell 9 (`FIX D`): `sanitize_spec(spec)` — replaces ±inf with NaN across every feature column referenced by a task spec, drops features that become entirely NaN or constant, prints an inf-count/dead-feature-count summary per task.
- cell 12 (`DL MODEL HELPERS`): `RNNClassifier` (class; LSTM/GRU, optionally bidirectional, `LayerNorm`+`Dropout`+`Linear` head on the last timestep), `PositionalEncoding` (class; standard sinusoidal), `TransformerClassifier` (class; linear input projection → positional encoding → `TransformerEncoder` (`norm_first=True`, GELU) → head on last timestep), `set_seed`, `build_model(model_cfg, input_dim, n_classes)` (dispatches on `model_cfg["model_type"]` to one of the three classes above), `make_loader`, `make_sequences(X, y, groups, starts, seq_len)` (per-group sliding windows sorted by window start), `torch_predict`, `choose_validation_group(train_groups, y_all, groups_all)` (picks, among training groups, the one with ≥2 classes present and the most rows, to hold out as the in-training validation split).
- cell 13 (`FAST DL TRAINING / EVALUATION`): `should_run_fast_dl_task`, `should_run_fast_dl_sensor`, `choose_fast_seq_len(spec)` (returns `seq_lens[-1]`/`[0]`/middle/explicit int per `FAST_DL_SEQ_CHOICE`), `get_fast_model_configs()` (filters the global `MODEL_CONFIGS` down to `FAST_DL_MODEL_TYPES`), `torch_predict_fast`, `train_one_fast_dl_run(spec, sensor_combo, seq_len, k_features, model_cfg, seed)` — the full LOGO training loop: per fold, picks a validation group via `choose_validation_group`, imputes/scales/clips/`SelectKBest`s in-fold, builds sequences, trains with early stopping on validation macro-F1 (`FAST_MAX_EPOCHS`/`FAST_PATIENCE`), evaluates on the held-out test group, and returns a summary dict + per-fold DataFrame + (optionally) per-window prediction rows.
- cell 17 (`BUILD FULL-STATISTIC SIX-LABEL ACTIVITY TOKENS`): `first_existing` (re-defined, task-3 scope), `load_six_label_windows()` — loads `rq3_normalized_labels_full.csv`, merges the 6-label-normalized process label onto a sensor-feature file (first existing of `FEATURE_CANDIDATES`) by `(group, rounded time)`; `channel_statistics(values)` — 14 named statistics (`STAT_NAMES`: mean/std/min/max/range/median/iqr/p10/p25/p75/p90/energy/rms/spectral entropy via FFT) for one numeric channel over one activity segment; `build_fullstat_tokens(window_df, sensor_cols)` — collapses consecutive same-label windows into "tokens" (segments), computing all 14 stats per sensor channel per token, returning a token DataFrame and its feature-column list.
- cell 18 (`BACK-OFF N-GRAM`, thesis Table 8.7/8.8 headline): `seqs_by_group(df)` — per-group ordered label sequences; `fit_ngram(train_seqs, max_h)` — builds order-`h` transition-count tables for `h=1..max_h` plus a unigram fallback; `predict_ngram(hist, tabs, uni, max_h)` — backs off from the longest available matching context down to the unigram (defaulting to `"conversation"` if nothing has ever been seen).
- cell 22 (`PERSISTENCE vs TRANSITION-ONLY`, thesis Table 8.2/8.8 lower block): no new function definitions — inline LOGO loop building `(current label → next label)` examples per group, then three comparisons: repeat-current baseline on all windows, a "no-self" order-1 back-off n-gram restricted to genuine transitions in the training folds, and repeat-current restricted to transition rows (trivially 0 by construction).
- cell 25 (`DESCRIPTIVE COMPARISON: researcher vs naive groups`, thesis Table 10.1): no function definitions — inline per-group descriptive stats (`minutes`, `n_windows`, `n_activities`, `transitions`, `trans_rate`, `mean_run_s`), cohort means, and Mann–Whitney U tests via `scipy.stats.mannwhitneyu`, plus a separate `ph` phase-duration-by-activity breakdown.

## Hyperparameter-looking constants (verbatim)

```python
RUN_ON = "naive"          # "all" = original 9 groups | "naive" = 5 naive groups
NAIVE_GROUPS    = {2, 3, 5, 6, 10}
RESEARCHER_GRPS = {1, 7, 8, 9}
TAG = "full9" if RUN_ON == "all" else "naive5"

DATA_ROOT = "/content/drive/MyDrive/thesis/data"
RUN_TASKS = [
    "interaction_vs_noninteraction",
    "conversation_vs_nonconversation",
    "conversation_vs_building",
    "conversation_vs_merging",
    "merging_vs_building",
    "three_class_activity",
]
SENSOR_COMBINATIONS = {
    "OE": ["OE"], "OPTI": ["OPTI"], "XSENS": ["XSENS"],
    "OE_OPTI": ["OE", "OPTI"], "OE_XSENS": ["OE", "XSENS"],
    "OPTI_XSENS": ["OPTI", "XSENS"], "OE_OPTI_XSENS": ["OE", "OPTI", "XSENS"],
}
RUN_CLASSICAL = True                     # set but never actually invoked in this notebook (see below)
RUN_DL = True
K_CLASSICAL = [40, 80, 120, 200, "all"]  # classical grid — defined but classical sweep never run here
TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]
K_DL = [80, 120, 200]
SEEDS = [42]
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 64
RUN_DL_MODEL_TYPES = ["lstm", "bilstm", "gru", "transformer"]
STOP_AFTER_N_DL_RUNS_PER_TASK_SENSOR = None
RANDOM_STATE = 42

BAD_TOKENS = ["label", "target", "class", "group", "window", "time", "elapsed",
              "pred", "prediction", "correct", "fold", "split", "index"]

ALL_MODEL_CONFIGS = [
    {"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "bilstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4},
]
MODEL_CONFIGS = [m for m in ALL_MODEL_CONFIGS if m["model_type"] in RUN_DL_MODEL_TYPES]

# "PIN THE DL GRID TO THE 6 THESIS WINNERS" cell (notebook cell 11) — later overwritten, see Section 8:
FAST_DL_SENSOR_COMBOS_TO_RUN = ["OPTI", "OPTI_XSENS", "OE_OPTI"]
FAST_DL_MODEL_TYPES          = ["transformer", "bilstm", "lstm"]
FAST_DL_K                    = 120
FAST_DL_SEQ_CHOICE           = "last"
SAVE_FAST_DL_PREDICTIONS     = True

# "FAST DL TRAINING / EVALUATION" cell (notebook cell 13) — values actually in effect at run time:
FAST_DL_TASKS_TO_RUN = None
FAST_DL_SENSOR_COMBOS_TO_RUN = None          # overwrites the pin above — reverts to ALL 7 sensor combos
FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]  # overwrites the pin above — drops "lstm"
FAST_DL_K = 120
FAST_DL_SEQ_CHOICE = "last"
FAST_MAX_EPOCHS = 25
FAST_PATIENCE = 4
FAST_BATCH_SIZE = 256
FAST_PRED_BATCH_SIZE = 1024
MAX_LOGO_FOLDS = None
SAVE_FAST_DL_PREDICTIONS = False             # overwrites the pin above — predictions CSV NOT (re)written

# "BEST PER TASK" cell (notebook cell 14) — the hardcoded reference table, verified below to be
# copied verbatim from thesis Table 7.12's Config/Full9-A/Full9-M columns:
BEST = {   # task: (sensor, model, full9_acc, full9_macroF1)
 "interaction_vs_noninteraction":   ("OPTI",       "transformer", 0.751, 0.751),
 "conversation_vs_nonconversation": ("OPTI",       "transformer", 0.880, 0.845),
 "conversation_vs_building":        ("OPTI_XSENS", "bilstm",      0.881, 0.860),
 "conversation_vs_merging":         ("OE_OPTI",    "transformer", 0.879, 0.866),
 "merging_vs_building":             ("OPTI",       "transformer", 0.811, 0.738),
 "three_class_activity":            ("OE_OPTI",    "lstm",        0.756, 0.699),
}

# Task 3 "GLOBAL CONFIGURATION" cell (notebook cell 16) — mostly dead config in this notebook, see below:
TOKEN_SEEDS = [42, 1, 7]
SEED_STD_DDOF = 0
FOLD_STD_DDOF = 1
K_SELECT = 40
GRAMMAR_ORDERS = [1, 2, 3, 5, 10]     # not actually used — see Section 8
HYBRID_ORDER = 3                      # not used anywhere downstream in this notebook
HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)   # not used anywhere downstream in this notebook
RESUME_EXISTING = True
TASK3_PIPELINE_VERSION = 'final_v2_common_targets'

# Part 4 descriptive-comparison cell (notebook cell 25):
NAIVE, RESEARCHER, WIN_S = {2, 3, 5, 6, 10}, {1, 7, 8, 9}, 5.0
```

`NAIVE_GROUPS = {2, 3, 5, 6, 10}` is confirmed **identical** to the definition used by `src/models/task1.py`'s `run_naive5_reproduction` and `src/models/task3_naive5.py` (both import `NAIVE_GROUPS` from `.common`), and matches `docs/thesis_reproduction_targets.md`'s statement that "this exact group-ID list is used consistently everywhere in the thesis it appears."

## Potentially broken / disabled cells

- **The only evidence of an actual completed run in this saved file stops at notebook cell 13.** Inspecting the `.ipynb` JSON's `outputs` directly: cells 2, 3, 4, 9, 11, 12 have small printed outputs (confirming `RUN_ON = "naive"`, `TAG = naive5`, 5 groups active, `pinned sensors/models` printed, `DL model types` printed), and cell 13 has exactly one output: `Output hidden; open in https://colab.research.google.com to view.` — **every code cell from 14 through 27 (the "BEST PER TASK" comparison, all of Part 2, all of Part 3, all of Part 4, and the Save cell) has zero saved output.** This means there is no execution evidence in this notebook file itself that Table 7.12/7.13's naive-5 numbers, the Table 8.8 grammar/persistence numbers, or the Table 10.1 descriptive comparison were ever actually produced by *this* run — only that the naive-5 FAST-DL sweep (cells 2–13) was launched with the correct group restriction. Whether it finished, and whether cells 14+ ran cleanly afterward, cannot be confirmed from the file.
- **Real contradiction between notebook cells 11 and 13** (flagged, not merely "worth deciding"): cell 11's banner comment reads `"PIN THE DL GRID TO THE 6 THESIS WINNERS\nAll six best-per-task configs are deep models, so the classical\nsweep is not needed at all. 3 sensors x 3 architectures ~ 3 min."` and sets `FAST_DL_SENSOR_COMBOS_TO_RUN = ["OPTI", "OPTI_XSENS", "OE_OPTI"]`, `FAST_DL_MODEL_TYPES = ["transformer", "bilstm", "lstm"]`, `SAVE_FAST_DL_PREDICTIONS = True`. Two cells later, cell 13's banner comment reads `"FAST DL TRAINING / EVALUATION\nRun this instead of the full DL grid."` and immediately **reassigns the same three globals**: `FAST_DL_SENSOR_COMBOS_TO_RUN = None` (reverts to running all 7 sensor combinations, not the pinned 3), `FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]` (silently drops `"lstm"`), and `SAVE_FAST_DL_PREDICTIONS = False` (silently un-does the pin's intent to save predictions). Because Python executes top-to-bottom, **the values actually in force at `get_fast_model_configs()`/the run loop are cell 13's, not cell 11's** — meaning the executed sweep is "all 7 sensors × 2 architectures × 6 tasks × 1 seed," not the advertised "3 sensors × 3 architectures ~3 min," and the per-window predictions file the very next cell (14) depends on is **not written** by this run.
- **Direct consequence — a circular/external-state dependency**, the same class of issue flagged in other notebooks in this project: cell 14 (`BEST PER TASK`) does `pred_path = os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_predictions.csv"); pr = _orig_read_csv(pred_path)` immediately followed by `assert all([gc, yt, yp, mc]), "column detection failed..."`. Given `SAVE_FAST_DL_PREDICTIONS = False` at the point cell 13 actually runs, this file is only present if it was written by an *earlier* execution of this same notebook (with the flag manually flipped back to `True`), or possibly produced by the sibling `seven_winners_naive5.ipynb`. A fresh, from-scratch run of this notebook top-to-bottom as currently written would raise `FileNotFoundError` at cell 14 unless that CSV already exists on Drive.
- **Task 3 config cell defines `GRAMMAR_ORDERS = [1, 2, 3, 5, 10]`, `HYBRID_ORDER = 3`, and `HYBRID_LAMBDAS` (11-point linspace) that are never referenced again anywhere downstream in this notebook** — the actual n-gram sweep (cell 18) hardcodes its own `for H in [1, 2, 3, 5]:` independently (matching Table 8.8's four rows, but omitting order 10 from `GRAMMAR_ORDERS`, and never running any sensor+grammar hybrid experiment despite `HYBRID_ORDER`/`HYBRID_LAMBDAS` being defined). Dead config left over from a fuller Task-3 notebook this cell was presumably copied from.
- **Part 3's naive-5 filtering relies on a column literally named `"group"` being present**, because the monkeypatch condition in cell 4 is `if RUN_ON == "naive" and isinstance(df, pd.DataFrame) and "group" in df.columns`. Part 3's own code resolves the actual group column dynamically: `GROUP_COL_LABEL = first_existing(labels_df.columns, ["group", "group_id", "session", "session_id"])`. If `rq3_normalized_labels_full.csv`'s real column is `group_id` or `session` rather than literally `group`, the auto-filter silently never fires for that load, and Part 3's persistence analysis (feeding Table 8.8's lower block) would silently run on the **full 9-group data** while still being logged/tagged as `"naive5"` — there is no assertion anywhere in this cell that checks the row count actually shrank to 5 groups. This should be verified against the real CSV's header before trusting Part 3's naive-5 numbers, or ported with an explicit mask (as `src/models/task3_naive5.py` already does) rather than relying on the implicit patch.
- **Part 4's "researcher vs naive descriptives" cell (Table 10.1's putative source) computes transitions from a single merged label stream** (`rq3_process_label`, one label per 5-second window) — `tr = (labs[1:] != labs[:-1])`, `trans_rate = tr.mean()` — whereas `docs/thesis_reproduction_targets.md` states Table 10.1's transition counts are "summed over all 7 annotation tiers (3 individuals + 3 pairs + whole-group; 6 tiers for G3)" with "per-tier rate = total/7 (or /6 for G3)." This cell's methodology (single derived-label-stream transitions on 5s windows) is **not obviously the same computation** as the thesis's 7-tier ELAN annotation transition count, even though both are plausibly called "transitions per minute." In addition, this cell's `desc` DataFrame only produces `minutes`, `n_windows`, `n_activities`, `transitions`, `trans_rate`, `mean_run_s` — it does **not** compute Table 10.1's `Individual build (%)` or `Dominant share` columns directly; those would have to be derived manually from the separately-printed `ph` (phase-duration-by-activity) breakdown at the end of the same cell. Treat this cell as producing raw ingredients toward Table 10.1, not a verified 1:1 reproduction of all 6 of its columns.
- No `!pip install` cells. `drive.mount('/content/drive')` (cell 3) is **not** wrapped in a try/except here (unlike the sibling `task1_full_comparison...ipynb`), so this notebook as saved is Colab-required, not Colab-optional — consistent with its recorded output `"Mounted at /content/drive"`.

## Which tasks/tables does this notebook actually cover

**"Best per task" means**: for each of the 6 recognition experiments (Task 1 interaction, and Tasks 2a–2e), the notebook hardcodes the single best-performing (sensor-combo, model-architecture) configuration that had already been identified as the winner in the *full-cohort* deep-learning comparison — the `BEST` dict in notebook cell 14 (quoted above) — and reruns **only that one configuration per task** on the naive-5-restricted data (`RUN_ON = "naive"`), rather than re-running the entire classical+DL grid. This is stated explicitly in the cell-11 banner: `"PIN THE DL GRID TO THE 6 THESIS WINNERS\nAll six best-per-task configs are deep models, so the classical sweep is not needed at all."` — and confirmed by `RUN_CLASSICAL = True` never actually being consulted by any classical-model training call anywhere downstream in this notebook (`make_classical_models`/`build_pipeline` are defined in cell 7 but never invoked — dead code in this particular notebook, presumably live in a sibling notebook).

**Verified exact mapping of the `BEST` dict to thesis Table 7.12**: every one of the 6 hardcoded `(sensor, model, full9_acc, full9_macroF1)` tuples matches Table 7.12's `Config`/`Full9 A`/`Full9 M` columns **exactly**, digit-for-digit, for the 6 non-"optimised" rows:

| `BEST` dict key | notebook tuple | Table 7.12 row | Table 7.12 Full9 A/M |
|---|---|---|---|
| `interaction_vs_noninteraction` | `("OPTI","transformer",0.751,0.751)` | Task 1: interaction (ablation) | 0.751 / 0.751 |
| `conversation_vs_nonconversation` | `("OPTI","transformer",0.880,0.845)` | Task 2a: conv. vs rest | 0.880 / 0.845 |
| `conversation_vs_building` | `("OPTI_XSENS","bilstm",0.881,0.860)` | Task 2b: conv. vs building | 0.881 / 0.860 |
| `conversation_vs_merging` | `("OE_OPTI","transformer",0.879,0.866)` | Task 2c: conv. vs merging | 0.879 / 0.866 |
| `merging_vs_building` | `("OPTI","transformer",0.811,0.738)` | Task 2d: merging vs building | 0.811 / 0.738 |
| `three_class_activity` | `("OE_OPTI","lstm",0.756,0.699)` | Task 2e: three-class | 0.756 / 0.699 |

Note these values are close to, but **not identical to**, Table 7.11's "Deep no-elapsed (full9 A/M)" column for the same tasks (e.g. Task 2b: Table 7.11 gives 0.875/0.852, this dict gives 0.881/0.860) — Table 7.11's deep column is described in its caption as "best of BiLSTM/Transformer at k=120, no elapsed," i.e. re-selected per task across sensor families for that specific table, whereas Table 7.12 pins one specific winning (sensor, model) configuration per task carried over from the earlier ablation tables (7.2–7.7). **This notebook's `BEST` dict is sourced from Table 7.12, not Table 7.11.**

Given that, the concrete table mapping is:

- **Notebook cells 11–14 (Part 1) → thesis Table 7.12's `Naive5 A` / `Naive5 M` / `ΔM` columns**, for 6 of its 7 rows (all except "Task 1: interaction (optimised)", which is `OPTI2_REL transf. s=18` — a *different*, more elaborate feature set/reproduction already covered separately by `src/models/task1.py`'s `run_naive5_reproduction`, not by this notebook). Proof: cell 14's loop `for task, (sens, mdl, f9a, f9m) in BEST.items(): ... variance_report(sub[yt], sub[yp], sub[gc], f"{task} | {sens} | {mdl}", "best_per_task") ... rows.append({"task": task, "sensor": sens, "model": mdl, "full9_A": f9a, "naive5_A": round(a, 4), "dA": round(a - f9a, 4), "full9_M": f9m, "naive5_M": round(m, 4), "dM": round(m - f9m, 4)})` computes exactly the `Naive5 A`/`Naive5 M`/`ΔM` fields Table 7.12 reports, keyed by the same 6 (task, sensor, model) triples.
- **The same cell's `variance_report()` call also produces thesis Table 7.13's per-group (per-fold) macro-F1 breakdown** for those same 6 configurations — `variance_report` builds `per = pd.DataFrame(rows).sort_values("group")` with one row per naive group (G2/G3/G5/G6/G10) and a `macro_f1` column, which is precisely Table 7.13's per-column layout (`G2 | G3 | G5 | G6 | G10 | pooled`).
- **Table 7.11 is only partially and indirectly touched**: the "Deep no-elapsed (naive5 A/M)" column of Table 7.11 is a *different* per-task best-of-BiLSTM/Transformer selection than Table 7.12's pinned configs, and this notebook's FAST-DL sweep (cell 13), if it had run the full un-pinned grid (all 7 sensor combos × bilstm/transformer), *could* supply the raw numbers Table 7.11 needs via `combined_fast_dl.groupby(["task","sensor_combo"]).head(1)` — but the notebook as written never re-derives Table 7.11's specific "OptiTrack feature family" framing or its classical no-elapsed/with-elapsed columns (those require the classical sweep, which — per the cell-11 banner — is deliberately skipped here). **Table 7.11's classical columns are not produced by this notebook at all.**
- **Notebook cells 16–19 (Part 2) → thesis Table 8.8's upper block** (activity-token back-off grammar at `h=1,2,3,5`) and the §8.11 headline number. Proof: cell 19 explicitly prints `"Task 3 winner (n-gram back-off h=2), naive5:"` with `f"  pooled acc      = {r['pooled_acc']:.4f}   (full-9: 0.600)"` and `f"  pooled macro-F1 = {r['pooled_macro_f1']:.4f}   (full-9: 0.500)"`, which are the same full-9 reference numbers (0.604/0.499 in the thesis text, rounded to 0.600/0.500 here) that `docs/thesis_reproduction_targets.md` §8.11 quotes as "the back-off n-gram over activity tokens — 0.604 accuracy / 0.499 macro-F1 at h=2."
- **Notebook cells 21–23 (Part 3) → thesis Table 8.8's lower block** (persistence analysis: repeat-current all-windows / no-self n-gram transitions / repeat-current transitions), matching the row-for-row structure and the `docs/thesis_reproduction_targets.md` description "947 labelled windows, 111 transitions" for the naive subset — this notebook's `ex = pd.DataFrame(...)` / `print(f"\nexamples: {len(ex)} | transitions: ...")` computes exactly these counts (subject to the naive-filter caveat above).
- **Notebook cell 25 (Part 4) → thesis Table 10.1** (Ch.10 §10.5, "researcher-participant vs. naive cohort comparison"). Proof: the cell's own banner comment is `"DESCRIPTIVE COMPARISON: researcher vs naive groups\nExplicitly requested by the supervisor."`, it defines `NAIVE, RESEARCHER, WIN_S = {2, 3, 5, 6, 10}, {1, 7, 8, 9}, 5.0` (the exact same cohort split Table 10.1 uses), computes `minutes`/`transitions`/`trans_rate` per group and cohort means, and runs `mannwhitneyu` per metric — directly mirroring the table's structure and its "for completeness only" Mann–Whitney framing in the docs. As flagged above, the exact transition-counting methodology and the `Individual build (%)`/`Dominant share` columns need independent verification against this cell's output (which was not preserved in the saved `.ipynb`).
- **Table 7.11 in full, and any classical (Logistic Regression / Linear SVC / SVC-RBF / RF / ExtraTrees) naive-5 numbers, are NOT produced anywhere in this notebook** — despite `RUN_CLASSICAL = True` and `make_classical_models`/`build_pipeline` being fully defined, no cell in this file ever calls them. Table 7.11's classical columns must come from one of the sibling notebooks (`seven_winners_naive5.ipynb` and/or `naive5_headline_rerun.ipynb`).

## Summary

This notebook is the "fast path" naive-5 reproduction: rather than re-running the entire classical+deep-learning grid restricted to 5 groups (which the sibling `task1_full_comparison...ipynb`-style notebooks do for the full 9-group cohort), it hardcodes the 6 already-known-best (sensor, model) configurations per recognition task (`BEST` in cell 14, verified above to be copied verbatim from thesis Table 7.12) and reruns only those, plus reruns the already-built Task 3 grammar/persistence pipelines and a researcher-vs-naive descriptive comparison, all restricted to `NAIVE_GROUPS = {2, 3, 5, 6, 10}`. The core mechanism for the "naive-5" restriction is architecturally distinct from what's already been ported into `src/`: instead of an explicit per-function group mask (the pattern `src/models/task1.py`'s `run_naive5_reproduction` and `src/models/task3_naive5.py` both use — `df["group"].map(_gid).isin(NAIVE_GROUPS)`), this notebook monkeypatches `pd.read_csv` itself (cell 4) so that **every** CSV loaded anywhere in the notebook that happens to have a column literally named `"group"` is silently filtered down to the 5 naive groups at load time, with an `_orig_read_csv` escape hatch used deliberately in two places (reading back predictions in cell 14, and reading the full-cohort labels file for the Table 10.1 comparison in cell 25, which needs to see all 9 groups). This is elegant but fragile: it depends on the group column being named exactly `"group"` everywhere, which — as flagged above — is not guaranteed for Part 3's dynamically-resolved `GROUP_COL_LABEL`, so Part 3's naive-5 numbers (feeding Table 8.8's lower block) carry a real risk of silently running on the full 9-group data without any error being raised. More importantly for reproducibility: the notebook's own saved outputs stop after cell 13 (the FAST-DL sweep, whose "hidden" Colab output at least confirms `RUN_ON="naive"` was active with 5 LOGO folds) — cells 14 through 27, covering the entire "BEST PER TASK" comparison (Table 7.12/7.13), both Task 3 blocks (Table 8.8), and the Table 10.1 descriptive comparison, have **zero recorded execution output** in this file, so none of the specific published naive-5 numbers can be cross-checked against this notebook's own run history; they can only be cross-checked against `docs/thesis_reproduction_targets.md`'s transcription of the final PDF. There is also a genuine, unresolved circular-bootstrap risk here similar to what's been seen elsewhere in this project: cell 14 reads back `combined_fast_dl_no_elapsed_predictions.csv`, a file that (per the code as currently written, with `SAVE_FAST_DL_PREDICTIONS` overwritten to `False` in cell 13) is not regenerated by a fresh run of this notebook — meaning a from-scratch execution would fail at cell 14 unless that CSV already exists on Drive from a previous run of this notebook (with the flag manually restored) or from the sibling `seven_winners_naive5.ipynb`. For porting into `src/eval/naive_cohort.py`: (1) reuse the existing explicit-mask pattern from `task1.py`/`task3_naive5.py` rather than a global monkeypatch; (2) port Part 1's `BEST`-dict-driven rerun as the direct source for Table 7.12's `Naive5 A`/`M`/`ΔM` columns and Table 7.13's per-group breakdown (6 of 7 rows; the 7th, "Task 1 optimised," is already covered by `task1.py`); (3) treat Part 2/Part 3 as effectively already superseded by the more carefully-reasoned, already-ported `src/models/task3_naive5.py` (whose own docstring already documents that no naive5 notebook had been fetched for Task 3 at the time it was written — this notebook is that missing source, and its code confirms `task3_naive5.py`'s inferred approach — reuse the existing token table, restrict by group — was correct, since `CORE_OUT`'s folder name and the reuse-existing-table pattern match exactly); (4) port Part 4 as the literal source for Table 10.1, but first verify its transition-counting methodology (single merged-label-stream vs. the thesis's stated 7-ELAN-tier sum) against the actual PDF, and derive the missing `Individual build (%)`/`Dominant share` columns from the `ph` phase-duration breakdown the cell already computes but doesn't name as such; (5) do not port the dead `RUN_CLASSICAL`/`make_classical_models`/`GRAMMAR_ORDERS`/`HYBRID_ORDER`/`HYBRID_LAMBDAS` config — none of it is exercised by this notebook's actual code path.
