# Analysis: Task 1 - task1_full_comparison_classical_elapsed_dl_with_std.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\task1_full_comparison_classical_elapsed_dl_with_std.ipynb`

## Cell counts

Total cells: 17  
Code cells: 11  
Markdown cells: 6

## Import statements (deduplicated)

```python
from IPython.display import display
from google.colab import drive
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset, DataLoader
import gc
import json
import numpy as np
import numpy as np, pandas as pd
import os
import os, random
import pandas as pd
import random
import re
import torch
import torch.nn as nn
import warnings
```

## Markdown headers / outline (in order)

- (cell 1, H1) Task 1 — Full Sensor and Model-Regime Comparison
  - (cell 1, H2) Output convention
  - (cell 6, H2) Data and feature preparation
  - (cell 9, H2) Classical comparison
  - (cell 11, H2) Deep-learning comparison without elapsed time
  - (cell 14, H2) Final publication tables

## Hardcoded file paths found in code

- `/content/drive`
- `/content/drive/MyDrive/thesis/data`
- `/content/drive/MyDrive/thesis/data/`
- `\nkaydedildi -> {OUT_DIR_0806}/task1_optimized_0806_fold_metrics_NAIVE5.csv`
- `]}_feature_counts.csv`
- `activity3_advanced_merged_10s_features.csv`
- `binary_5s_all_sensor_advanced_features.csv`
- `binary_5s_specialized_oe_merged_all_features.csv`
- `combined_classical_best_per_condition_with_std.csv`
- `combined_classical_fold_metrics.csv`
- `combined_classical_summary_with_std.csv`
- `combined_dl_no_elapsed_best_per_task_sensor_with_std.csv`
- `combined_dl_no_elapsed_fold_metrics.csv`
- `combined_dl_no_elapsed_predictions.csv`
- `combined_dl_no_elapsed_summary_partial.csv`
- `combined_dl_no_elapsed_summary_with_std.csv`
- `dl_plan.csv`
- `dl_seed_variability.csv`
- `publication_three_regime_comparison_long.csv`
- `publication_three_regime_comparison_wide.csv`
- `task1_optimized_0806_exact_features.csv`
- `task1_optimized_0806_fold_metrics.csv`
- `task1_optimized_0806_summary_with_std.csv`
- `{OUT_DIR_0806}/task1_optimized_0806_fold_metrics_NAIVE5.csv`
- `{task_name}_{sensor_combo}_classical_best_per_condition_with_std.csv`
- `{task_name}_{sensor_combo}_classical_fold_metrics.csv`
- `{task_name}_{sensor_combo}_classical_predictions.csv`
- `{task_name}_{sensor_combo}_classical_summary_with_std.csv`

## Function / class definitions

- cell 7, function `safe_name` — (no docstring)
- cell 7, function `unique_feats` — (no docstring)
- cell 7, function `find_first_existing` — (no docstring)
- cell 7, function `infer_window_seconds` — (no docstring)
- cell 7, function `add_elapsed_min` — (no docstring)
- cell 7, function `looks_bad_feature_name` — (no docstring)
- cell 7, function `is_oe_feature` — (no docstring)
- cell 7, function `is_opti_feature` — (no docstring)
- cell 7, function `is_xsens_feature` — (no docstring)
- cell 7, function `clean_feature_list` — (no docstring)
- cell 7, function `make_classical_models` — (no docstring)
- cell 7, function `build_pipeline` — (no docstring)
- cell 7, function `metric_dict` — (no docstring)
- cell 8, function `get_modality_features` — (no docstring)
- cell 8, function `make_combo_feature_sets` — (no docstring)
- cell 8, function `prepare_task` — (no docstring)
- cell 10, function `run_classical_for_task_and_sensor` — (no docstring)
- cell 12, class `RNNClassifier` — (no docstring)
- cell 12, class `PositionalEncoding` — (no docstring)
- cell 12, class `TransformerClassifier` — (no docstring)
- cell 12, function `set_seed` — (no docstring)
- cell 12, function `build_model` — (no docstring)
- cell 12, function `make_loader` — (no docstring)
- cell 12, function `make_sequences` — (no docstring)
- cell 12, function `torch_predict` — (no docstring)
- cell 12, function `choose_validation_group` — (no docstring)
- cell 12, function `__init__` — (no docstring)
- cell 12, function `forward` — (no docstring)
- cell 12, function `__init__` — (no docstring)
- cell 12, function `forward` — (no docstring)
- cell 12, function `__init__` — (no docstring)
- cell 12, function `forward` — (no docstring)
- cell 13, function `choose_fast_seq_len` — (no docstring)
- cell 13, function `get_fast_model_configs` — (no docstring)
- cell 13, function `torch_predict_fast` — (no docstring)
- cell 13, function `train_one_fast_dl_run` — (no docstring)
- cell 16, function `is_relative_opti2` — (no docstring)
- cell 16, function `set_exact_seed` — (no docstring)
- cell 16, function `impute_from_training` — (no docstring)
- cell 16, function `predict_loader_exact` — (no docstring)
- cell 16, function `train_exact_fold` — (no docstring)
- cell 17, function `_gid` — (no docstring)

## Hyperparameter-looking constants (verbatim)

- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"`
- `ACTIVITY_DATA_PATH = (`
- `INTERACTION_DATA_PATHS = [`
- `OUT_DIR = f"{DATA_ROOT}/PUBLICATION_TASK1_FULL_COMPARISON"`
- `RUN_TASKS = ["interaction_vs_noninteraction"]`
- `SENSOR_COMBINATIONS = {`
- `RUN_CLASSICAL = True`
- `RUN_DL = True`
- `REPORT_REPRODUCTION_MODE = True`
- `K_CLASSICAL_FULL = [40, 80, 120, 200, "all"]`
- `TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]`
- `RUN_DL_MODEL_TYPES = ["lstm", "bilstm", "gru", "transformer"]`
- `K_DL = [120]`
- `SEEDS = [42]`
- `FAST_MAX_EPOCHS = 25`
- `FAST_PATIENCE = 4`
- `FAST_BATCH_SIZE = 256`
- `FAST_PRED_BATCH_SIZE = 1024`
- `MAX_LOGO_FOLDS = None`
- `SAVE_CLASSICAL_PREDICTIONS = False`
- `SAVE_FAST_DL_PREDICTIONS = False`
- `RANDOM_STATE = 42`
- `BAD_TOKENS = [`
- `CLASSICAL_MODEL_NAMES = ["logreg_C1", "linearSVC_C1"]`
- `CLASSICAL_K_VALUES = [80, 200]`
- `CLASSICAL_MODEL_NAMES = [`
- `CLASSICAL_K_VALUES = K_CLASSICAL_FULL`
- `ALL_MODEL_CONFIGS = [`
- `MODEL_CONFIGS = [m for m in ALL_MODEL_CONFIGS if m["model_type"] in RUN_DL_MODEL_TYPES]`
- `FAST_DL_TASKS_TO_RUN = None`
- `FAST_DL_SENSOR_COMBOS_TO_RUN = None`
- `FAST_DL_MODEL_TYPES = RUN_DL_MODEL_TYPES`
- `FAST_DL_K = 120`
- `FAST_DL_SEQ_CHOICE = "last"`
- `FAST_DL_OUT_DIR = os.path.join(`
- `DATA_PATH = (`
- `OUT_DIR_0806 = (`
- `SEED = 42`
- `SEQ_LEN = 18`
- `MAX_EPOCHS = 80`
- `PATIENCE = 12`
- `BATCH_SIZE = 64`
- `TOKENS = [`
- `NAIVE_GROUPS = {2, 3, 5, 6, 10}`
- `random_state = RANDOM_STATE,`
- `n_estimators = 500,`
- `n_folds = ("test_group", "nunique"),`
- `hidden_size = hidden_dim,`
- `num_layers = num_layers,`
- `dropout = dropout if num_layers > 1 else 0.0,`
- `dropout = dropout,`
- `batch_size = FAST_PRED_BATCH_SIZE,`
- `k = actual_k,`
- `lr = model_cfg["lr"],`
- `batch_size = FAST_BATCH_SIZE,`
- `seed = int(row["seed"]),`
- `K = 120`
- `batch_size = BATCH_SIZE,`
- `num_layers = 2,`
- `dropout = 0.25,`
- `lr = 5e-4,`
- `patience = 4,`

## Potentially broken / disabled cells

Manual review (automated heuristic found nothing, but these are worth flagging):

- **Not broken, but dead-config**: `prepare_task()` (cell 8) implements 6 possible tasks — `interaction_vs_noninteraction`, `conversation_vs_nonconversation`, `conversation_vs_building`, `conversation_vs_merging`, `merging_vs_building`, `three_class_activity` — but `RUN_TASKS = ["interaction_vs_noninteraction"]` (cell 5) means only the interaction task is ever actually executed. The other 5 task branches are fully coded but unreachable given current config. Worth deciding during porting whether those activity-vs-activity comparisons are wanted.
- **Intentional cross-cell dependency, not broken**: Cell 16 ("EXACT OPTIMIZED TRANSFORMER REPRODUCTION") explicitly requires cell 12 ("DL MODEL HELPERS") to have run first (checks `DEVICE`, `TransformerClassifier`, `make_loader`, `make_sequences` are in `globals()` and raises `RuntimeError` if not). Cell 17 ("NAIVE-5 RERUN") similarly requires cell 16 to have run first (checks `df`, `features`, `train_exact_fold`, etc.). These are guarded with explicit runtime checks, so they fail loudly rather than silently — but they are NOT self-contained scripts; porting needs to preserve this dependency order or refactor into an explicit pipeline.
- Cell 16 hardcodes two "magic number" assertions that will raise if source data changes: `if len(features) != 211: raise RuntimeError(...)` and `if len(all_true) != 4426: raise RuntimeError(...)`. These are sanity checks pinned to the exact historical dataset — expected to pass on the canonical data but will hard-fail on any different dataset revision. Important to note for the port (do we keep these guards or relax them?).
- Cell 17 has Turkish-language comments/prints (e.g. `"Once hucre 15'i calistir. Eksik:"`, `"kaydedildi ->"`) — a mix of English and Turkish. Not broken, just inconsistent language; flag for cleanup during port.
- Cell 17 restricts evaluation to `NAIVE_GROUPS = {2, 3, 5, 6, 10}` (5 of the 9 groups) as a deliberate "naive" comparison rerun against the full 9-group result in cell 16 — intentional, not an error.
- No `!pip install` cells and no cells that error out from a missing variable were found. `drive.mount` is wrapped in a try/except that degrades gracefully when not running in Colab (cell 3), so it is Colab-optional, not Colab-required.

## Summary

This notebook trains and compares models for **Task 1 (interaction detection)**: classifying 5-second windows as `interaction` vs `non_interaction` using wearable/optical sensor features (OE = "OpenEarable"/ear-worn audio-derived features, OPTI = OptiTrack motion-capture-derived features, XSENS = XSens IMU-derived features). It reads two pre-computed feature CSVs from Google Drive (`INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv` and `INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv`, first-existing wins) plus an activity feature CSV that is loaded but, given current config, not actually used for a run (only the interaction task is enabled). It runs three parallel "regimes" and compares them: (1) classical ML (Logistic Regression, Linear SVC, with optional RBF-SVC/RandomForest/ExtraTrees in a non-default exploratory mode) over all 7 single/combo sensor-modality subsets (OE, OPTI, XSENS, and their pairwise/triple combinations), with and without an `elapsed_min` "time-in-session" feature, using `SelectKBest` feature selection and Leave-One-Group-Out (LOGO) cross-validation over the 9 study groups; (2) a deep-learning sweep (LSTM, BiLSTM, GRU, Transformer) over the same sensor combinations without elapsed time, using sliding-window sequences; and (3) a final "exact historical reproduction" run (cells 16-17) that reproduces one specific optimized Transformer configuration (`OPTI2_RELATIVE_ONLY` feature set, 211 features, seq_len=18 i.e. 90s context, k=120, seed=42) to match published thesis numbers almost exactly (asserts on feature count and evaluated-sequence count), plus a restricted 5-group "naive" rerun of the same pipeline for comparison. All three regimes write per-fold and pooled mean±SD metrics (accuracy, macro-F1, balanced accuracy) to CSVs under `PUBLICATION_TASK1_FULL_COMPARISON/` and a separate `PUBLICATION_TASK1_OPTIMIZED_0806/` directory, culminating in long- and wide-format "publication" comparison tables. The notebook is Colab-optional (graceful `drive.mount` fallback) and otherwise looks complete and runnable end-to-end, though it depends on upstream feature-engineering notebooks/scripts (not included here) having already produced the three input CSVs.
