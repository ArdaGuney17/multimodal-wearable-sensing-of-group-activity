# Analysis: Task 2 - task2_full_comparison_RESUMABLE_with_std.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\task2_full_comparison_RESUMABLE_with_std.ipynb`

## Cell counts

Total cells: 16  
Code cells: 9  
Markdown cells: 7

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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset, DataLoader
import gc
import json
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import torch.nn as nn
import warnings
```

## Markdown headers / outline (in order)

- (cell 1, H1) Task 2 — Full Comparison with Resume and Checkpoint Support
  - (cell 1, H2) Output convention
  - (cell 6, H2) Data and feature preparation
  - (cell 9, H2) Safe execution and recovery
  - (cell 10, H2) Classical comparison
  - (cell 12, H2) Deep-learning comparison without elapsed time
  - (cell 15, H2) Final publication tables

## Hardcoded file paths found in code

- `/content/drive`
- `/content/drive/MyDrive/thesis/data`
- `]}_feature_counts.csv`
- `activity3_advanced_merged_10s_features.csv`
- `binary_5s_all_sensor_advanced_features.csv`
- `binary_5s_specialized_oe_merged_all_features.csv`
- `combined_classical_best_per_condition_with_std.csv`
- `combined_classical_fold_metrics.csv`
- `combined_classical_summary_with_std.csv`
- `combined_dl_no_elapsed_best_per_task_sensor_with_std.csv`
- `combined_dl_no_elapsed_fold_metrics.csv`
- `combined_dl_no_elapsed_folds_partial.csv`
- `combined_dl_no_elapsed_predictions.csv`
- `combined_dl_no_elapsed_summary_partial.csv`
- `combined_dl_no_elapsed_summary_with_std.csv`
- `dl_plan.csv`
- `dl_seed_variability.csv`
- `publication_three_regime_comparison_long.csv`
- `publication_three_regime_comparison_wide.csv`
- `{run_key}__folds.csv`
- `{run_key}__predictions.csv`
- `{run_key}__summary.csv`
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
- cell 11, function `run_classical_for_task_and_sensor` — (no docstring)
- cell 13, class `RNNClassifier` — (no docstring)
- cell 13, class `PositionalEncoding` — (no docstring)
- cell 13, class `TransformerClassifier` — (no docstring)
- cell 13, function `set_seed` — (no docstring)
- cell 13, function `build_model` — (no docstring)
- cell 13, function `make_loader` — (no docstring)
- cell 13, function `make_sequences` — (no docstring)
- cell 13, function `torch_predict` — (no docstring)
- cell 13, function `choose_validation_group` — (no docstring)
- cell 13, function `__init__` — (no docstring)
- cell 13, function `forward` — (no docstring)
- cell 13, function `__init__` — (no docstring)
- cell 13, function `forward` — (no docstring)
- cell 13, function `__init__` — (no docstring)
- cell 13, function `forward` — (no docstring)
- cell 14, function `choose_fast_seq_len` — (no docstring)
- cell 14, function `get_fast_model_configs` — (no docstring)
- cell 14, function `torch_predict_fast` — (no docstring)
- cell 14, function `train_one_fast_dl_run` — (no docstring)

## Hyperparameter-looking constants (verbatim)

- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"`
- `ACTIVITY_DATA_PATH = (`
- `INTERACTION_DATA_PATHS = [`
- `OUT_DIR = f"{DATA_ROOT}/PUBLICATION_TASK2_FULL_COMPARISON"`
- `RUN_TASKS = [`
- `SENSOR_COMBINATIONS = {`
- `RESUME_EXISTING = True`
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
- `CLASSICAL_SUMMARY_PATH = os.path.join(`
- `CLASSICAL_FOLDS_PATH = os.path.join(`
- `CLASSICAL_BEST_PATH = os.path.join(`
- `CLASSICAL_FILES_COMPLETE = all(`
- `RUN_CLASSICAL = False`
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
- `FAST_DL_RUN_DIR = os.path.join(`
- `DL_SUMMARY_PATH = os.path.join(`
- `DL_FOLDS_PATH = os.path.join(`
- `DL_BEST_PATH = os.path.join(`
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

## Potentially broken / disabled cells

Manual review (automated heuristic found nothing structurally broken):

- **Not broken, but dead-weight load**: cell 8 always loads and validates `ACTIVITY_DATA_PATH` (raises `FileNotFoundError` if missing) AND attempts to load `INTERACTION_DATA_PATHS` (`interaction_df_raw`), even though `RUN_TASKS` (cell 5) only lists the 5 activity-comparison tasks (`conversation_vs_nonconversation`, `conversation_vs_building`, `conversation_vs_merging`, `merging_vs_building`, `three_class_activity`) — `interaction_vs_noninteraction` is never in `RUN_TASKS` here (that's Task 1's job), so `interaction_df_raw` is loaded but unused in this notebook. Not an error, just wasted I/O; the interaction CSV load degrades gracefully (prints a warning) if missing.
- **Identical duplicated global assignment**: `CLASSICAL_SUMMARY_PATH`, `CLASSICAL_FOLDS_PATH`, `CLASSICAL_BEST_PATH`, and `FAST_DL_OUT_DIR` are defined twice — once in cell 11 (classical runner) and again in cell 16 (final tables / "automatic recovery after a Colab runtime disconnect"). Harmless (same values recomputed), but a porting smell — should be defined once in config.
- **This is the "resumable" version of the Task 1/Task 2 pipeline**: cell 11 checks `RESUME_EXISTING` and, if the three classical output CSVs already exist on Drive, reloads them and sets `RUN_CLASSICAL = False` to skip re-running the whole classical grid. Cell 14 (DL loop) checkpoints **per individual run** (`{run_key}__summary.csv` / `__folds.csv` / `__predictions.csv` under `DL_NO_ELAPSED_ALL_ARCHITECTURES/PER_CONFIGURATION_CHECKPOINTS/`) and skips any run whose checkpoint files already exist. This means the notebook's actual behavior is highly state-dependent on what's already in `PUBLICATION_TASK2_FULL_COMPARISON/` on Drive — running it fresh vs. re-running it after a partial prior run will take very different code paths. Important for porting: the "resume" logic is a first-class feature here, not incidental.
- Cell 16 ("FINAL THREE-REGIME COMPARISON TABLES") also has its own "automatic recovery" block that re-loads `combined_classical_best`/`combined_fast_dl_best` from CSV if they aren't already in the notebook's global namespace (e.g., after a fresh restart) before raising `RuntimeError` if still missing. Same defensive pattern as Task 1's cell 15, but more elaborate.
- No `!pip install` cells found. `drive.mount` is optional/graceful (cell 3), same pattern as Task 1.
- No cells found that reference an undefined variable or are entirely commented out.

## Summary

This notebook is the Task 2 (activity/group-behavior recognition) counterpart to the Task 1 notebook, sharing essentially the same feature-engineering helpers, classical-ML pipeline, and DL pipeline almost verbatim, but retargeted at 5 multi-class/binary **activity** classification tasks read from the single 10-second-window feature CSV `INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv`: `conversation_vs_nonconversation`, `conversation_vs_building`, `conversation_vs_merging`, `merging_vs_building`, and `three_class_activity` (co_building / co_merging / conversation). Like Task 1, it evaluates the same three "regimes" — classical ML (Logistic Regression + Linear SVC by default, or a larger grid with RBF-SVC/RandomForest/ExtraTrees) across 7 sensor-modality combinations (OE, OPTI, XSENS, and pairwise/triple combos) with/without elapsed-time features, and a DL sweep (LSTM/BiLSTM/GRU/Transformer) over the same combos — all under Leave-One-Group-Out cross-validation with fold-level mean±SD reporting. The key difference from Task 1 is that this notebook is explicitly built to be **resumable across Colab disconnects**: it checkpoints classical results at the whole-grid level and DL results at the per-configuration level, and every phase first checks Google Drive for already-completed CSV outputs before deciding whether to (re)run. Final outputs are long- and wide-format "publication" comparison tables written to `PUBLICATION_TASK2_FULL_COMPARISON/`, mirroring Task 1's output structure. The notebook is complete and internally consistent; the main porting consideration is that its "skip if checkpoint exists" resume logic is deliberate behavior (useful for a long Colab job) that should probably be reimplemented as an explicit checkpoint/cache mechanism rather than dropped, since it changes what actually executes on a given run.
