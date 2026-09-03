# Analysis: Task 3 - task3_FINAL_V2_with_exact_report_reproduction.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\task3_FINAL_V2_with_exact_report_reproduction.ipynb`

## Cell counts

Total cells: 56  
Code cells: 43  
Markdown cells: 13

## Import statements (deduplicated)

```python
from IPython.display import display
from collections import Counter, defaultdict
from collections import defaultdict, Counter
from google.colab import drive
from hmmlearn import hmm
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy as np, pandas as pd
import numpy as np, pandas as pd, glob, os, re, warnings
import numpy as np, pandas as pd, warnings
import os
import pandas as pd
import pandas as pd, numpy as np, glob, os, re, matplotlib.pyplot as plt
import random
import re
import shutil
import subprocess, sys
import torch
import torch.nn as nn
import warnings
```

## Markdown headers / outline (in order)

- (cell 1, H1) Task 3 — Final V2 Common-Target Publication Analysis
  - (cell 2, H2) Reporting convention
  - (cell 3, H2) How to run this final notebook
- (cell 7, H1) Part I — Build six-label full-statistic activity tokens
- (cell 9, H1) Part II — Corrected common-target activity grammar comparison
- (cell 11, H1) Part III — Leakage-free neural models on the same 217 targets
- (cell 14, H1) Part IV — Final common-target Task 3 comparison
- (cell 16, H1) Optional Appendix A — Window-level history and transition-only evaluation
- (cell 28, H1) Optional Appendix B — Segment-level sensor-feature forecasting
- (cell 42, H1) Optional Appendix C — Fine 13-label versus coarse 4-label expanding-prefix prediction
- (cell 45, H1) Optional Appendix D — HMM and collective-state analyses
  - (cell 49, H2) Appendix R — Exact reproduction of the original report protocol
  - (cell 55, H2) Recommended thesis hierarchy

## Hardcoded file paths found in code

- `*.csv`
- `.csv`
- `.csv.gz`
- `.pkl`
- `/content/drive`
- `/content/drive/MyDrive/thesis/data`
- `/content/drive/MyDrive/thesis/data/`
- `/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED`
- `/content/drive/MyDrive/thesis/data/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv`
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG3`
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv`
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_5class_interaction_only_features.csv`
- `/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/interaction_eng3_features.csv`
- `activity_tokens_6label_fullstat.csv`
- `appendix_r_legacy_fold_metrics.csv`
- `appendix_r_legacy_summary_with_std.csv`
- `appendix_r_report_reproduction_check.csv`
- `best_sensor_forecast_classification_report.csv`
- `best_transition_only_classification_report.csv`
- `detected_numeric_features.csv`
- `detected_sensor_features.csv`
- `feature_file_selection_scores.csv`
- `forecast_examples_h{h}.csv`
- `group_{g}_openearable_model_ready.csv`
- `history_examples_h{h}.csv`
- `history_model_feature_file_scores.csv`
- `interaction_eng3_features.csv`
- `interaction_oe10_10s.csv`
- `label_segment_transition_baselines.csv`
- `label_vs_sensor_history_comparison.csv`
- `model_ready_file_details.json`
- `model_ready_file_summary.csv`
- `rq3_7label_history_aware_fold_std.csv`
- `rq3_7label_history_aware_predictions.csv`
- `rq3_7label_history_aware_summary.csv`
- `rq3_label_normalization_metadata.json`
- `rq3_normalized_labels_full.csv`
- `segment_level_feature_vectors.csv`
- `segment_sensor_forecast_fold_std.csv`
- `segment_sensor_forecast_predictions.csv`
- `segment_sensor_forecast_summary.csv`
- `task3_core_FINAL_V2_PANEL_A_deterministic_common_targets.csv`
- `task3_core_FINAL_V2_PANEL_B_neural_common_targets.csv`
- `task3_core_FINAL_V2_PANEL_C_long_history_sensitivity.csv`
- `task3_core_FINAL_V2_descriptive_combined_table.csv`
- `task3_grammar_long_history_common_targets_fold_metrics.csv`
- `task3_grammar_long_history_common_targets_predictions.csv`
- `task3_grammar_long_history_common_targets_summary_with_std.csv`
- `task3_grammar_primary_common_targets_fold_metrics.csv`
- `task3_grammar_primary_common_targets_predictions.csv`
- `task3_grammar_primary_common_targets_summary_with_std.csv`
- `task3_token_neural_FINAL_V2_COMMON_TARGETS_group_means.csv`
- `task3_token_neural_FINAL_V2_COMMON_TARGETS_seed_group_metrics.csv`
- `task3_token_neural_FINAL_V2_COMMON_TARGETS_seed_results.csv`
- `task3_token_neural_FINAL_V2_COMMON_TARGETS_summary_with_std.csv`
- `{D}/INTERACTION_OE10/interaction_oe10_10s.csv`
- `{D}/INTERACTION_OPTI2/interaction_opti2_10s.csv`
- `{OUT_DIR}/interaction_eng3_features.csv`
- `{OUT_DIR}/recognition_interaction_window_label_inventory.csv`
- `{OUT_DIR}/{RESULT_PREFIX}_fold_std.csv`
- `{OUT_DIR}/{RESULT_PREFIX}_macro_f1_pivot.csv`
- `{OUT_DIR}/{RESULT_PREFIX}_predictions.csv`
- `{OUT_DIR}/{RESULT_PREFIX}_summary.csv`
- `{run_key}__predictions.csv`
- `{run_key}__summary.csv`

## Function / class definitions

- cell 8, function `first_existing` — (no docstring)
- cell 8, function `load_six_label_windows` — (no docstring)
- cell 8, function `channel_statistics` — (no docstring)
- cell 8, function `build_fullstat_tokens` — (no docstring)
- cell 10, function `token_sequences` — (no docstring)
- cell 10, function `fit_backoff_tables` — (no docstring)
- cell 10, function `backoff_probabilities` — (no docstring)
- cell 10, function `fit_second_order_transition` — (no docstring)
- cell 10, function `fit_sensor_pipeline` — (no docstring)
- cell 10, function `aligned_sensor_probabilities` — (no docstring)
- cell 10, function `metric_record` — (no docstring)
- cell 10, function `choose_hybrid_lambda` — (no docstring)
- cell 10, function `save_grammar_fold` — (no docstring)
- cell 10, function `summarize_common_targets` — (no docstring)
- cell 10, function `ngram_predictions_on_common_targets` — (no docstring)
- cell 12, class `TokenSeqNet` — (no docstring)
- cell 12, function `run_token_seed` — (no docstring)
- cell 12, function `__init__` — (no docstring)
- cell 12, function `forward` — (no docstring)
- cell 12, function `sequences_for` — (no docstring)
- cell 12, function `sequence_iterator` — (no docstring)
- cell 18, function `first_existing` — (no docstring)
- cell 19, function `detect_group_time_cols` — (no docstring)
- cell 19, function `is_result_file` — (no docstring)
- cell 19, function `numeric_feature_count_sample` — (no docstring)
- cell 21, function `make_history_examples` — (no docstring)
- cell 22, function `train_ngram_transition_model` — (no docstring)
- cell 22, function `predict_ngram_backoff` — (no docstring)
- cell 23, function `make_ohe` — (no docstring)
- cell 23, function `score_predictions` — (no docstring)
- cell 23, function `build_history_classifier` — (no docstring)
- cell 30, function `first_existing` — (no docstring)
- cell 31, function `detect_group_time_cols` — (no docstring)
- cell 31, function `is_result_file` — (no docstring)
- cell 31, function `numeric_feature_count_sample` — (no docstring)
- cell 35, function `make_forecast_examples` — (no docstring)
- cell 36, function `safe_log_loss` — (no docstring)
- cell 36, function `score_classification` — (no docstring)
- cell 36, function `feature_rmse` — (no docstring)
- cell 36, function `mean_cosine_similarity` — (no docstring)
- cell 36, function `align_proba_to_label_order` — (no docstring)
- cell 36, function `make_decoder_classifier` — (no docstring)
- cell 36, function `get_final_clf` — (no docstring)
- cell 38, function `train_first_order_markov` — (no docstring)
- cell 38, function `predict_first_order_markov` — (no docstring)
- cell 38, function `predict_first_order_markov_no_self` — (no docstring)
- cell 43, function `normalize_text` — (no docstring)
- cell 43, function `make_fine_13_labels` — Same fine-grained setup as previous Task 3:
- cell 43, function `coarse_4_map` — Coarse group-state mapping.
- cell 43, function `make_coarse_4_labels` — (no docstring)
- cell 43, function `add_key_columns` — (no docstring)
- cell 43, function `build_segments_for_label_mode` — Collapse consecutive same labels into activity segments.
- cell 43, function `build_prefix_examples` — For each group:
- cell 43, function `pad_activity_sequences` — (no docstring)
- cell 43, function `pad_numeric_sequences` — (no docstring)
- cell 43, function `preprocess_numeric_fold` — Fit imputer/scaler only on valid training prefix positions.
- cell 43, class `PrefixDataset` — (no docstring)
- cell 43, class `PrefixLSTM` — (no docstring)
- cell 43, class `PrefixCNN1D` — (no docstring)
- cell 43, class `PrefixTransformer` — (no docstring)
- cell 43, function `build_deep_model` — (no docstring)
- cell 43, function `predict_majority` — (no docstring)
- cell 43, function `predict_markov_last` — (no docstring)
- cell 43, function `train_suffix_tables` — Variable-order Markov / suffix-backoff model.
- cell 43, function `predict_suffix_backoff` — (no docstring)
- cell 43, function `train_eval_deep` — (no docstring)
- cell 43, function `score_predictions` — (no docstring)
- cell 43, function `__init__` — (no docstring)
- cell 43, function `__len__` — (no docstring)
- cell 43, function `__getitem__` — (no docstring)
- cell 43, function `__init__` — (no docstring)
- cell 43, function `forward` — (no docstring)
- cell 43, function `__init__` — (no docstring)
- cell 43, function `forward` — (no docstring)
- cell 43, function `__init__` — (no docstring)
- cell 43, function `forward` — (no docstring)
- cell 46, function `fit_hmm` — (no docstring)
- cell 46, function `rep` — (no docstring)
- cell 47, function `norm` — (no docstring)
- cell 47, function `to_coll` — (no docstring)
- cell 47, function `nonempty` — (no docstring)
- cell 47, function `find` — (no docstring)
- cell 47, function `dominant` — (no docstring)
- cell 47, function `learn` — (no docstring)
- cell 47, function `causal` — (no docstring)
- cell 47, function `viterbi` — (no docstring)
- cell 47, function `sc` — (no docstring)
- cell 48, function `norm` — (no docstring)
- cell 48, function `to_coll` — (no docstring)
- cell 48, function `nonempty` — (no docstring)
- cell 48, function `dom` — (no docstring)
- cell 48, function `find` — (no docstring)
- cell 48, function `load_sensor` — (no docstring)
- cell 48, function `agg30` — (no docstring)
- cell 48, function `learn` — (no docstring)
- cell 48, function `causal_next` — forward filter; at each t predict NEXT state (past+present only).
- cell 48, function `rep` — (no docstring)
- cell 50, function `_legacy_fold_table` — (no docstring)
- cell 52, function `pid` — (no docstring)
- cell 52, function `emis_vec` — (no docstring)
- cell 53, function `ngram_tables` — (no docstring)
- cell 53, function `p_ngram` — back-off distribution over next label given history up to t-1 (predicting s[t]).
- cell 53, function `make_xy` — (no docstring)
- cell 53, function `p_sensor` — (no docstring)
- cell 56, function `matches_any` — (no docstring)
- cell 56, function `matching_columns` — (no docstring)
- cell 56, function `detect_sensor_families` — (no docstring)
- cell 56, function `read_sample` — (no docstring)
- cell 56, function `estimate_sampling_information` — (no docstring)
- cell 56, function `compact_dtype_summary` — (no docstring)
- cell 56, function `safe_unique_values` — (no docstring)

## Hyperparameter-looking constants (verbatim)

- `DATA_ROOT = '/content/drive/MyDrive/thesis/data'`
- `CORE_OUT = os.path.join(DATA_ROOT, 'PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS')`
- `RESUME_EXISTING = True`
- `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`
- `TOKEN_SEEDS = [42, 1, 7]`
- `SEED_STD_DDOF = 0`
- `FOLD_STD_DDOF = 1`
- `K_SELECT = 40`
- `GRAMMAR_ORDERS = [1, 2, 3, 5, 10]`
- `HYBRID_ORDER = 3`
- `HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)`
- `TASK3_PIPELINE_VERSION = 'final_v2_common_targets'`
- `OLD_TOKEN_PATH = (`
- `NEW_TOKEN_PATH = (`
- `NORM = os.path.join(`
- `FEATURE_CANDIDATES = [`
- `MERGE6 = {`
- `STAT_NAMES = [`
- `TOKEN_PATH = os.path.join(CORE_OUT, 'activity_tokens_6label_fullstat.csv')`
- `CLASSES = sorted(T['label'].astype(str).unique())`
- `CLASS_TO_ID = {label: index for index, label in enumerate(CLASSES)}`
- `ID_TO_CLASS = {index: label for label, index in CLASS_TO_ID.items()}`
- `N_CLASSES = len(CLASSES)`
- `FIXED_LABEL_IDS = np.arange(N_CLASSES)`
- `SEQUENCES = token_sequences(T)`
- `ALL_GROUPS = sorted(SEQUENCES)`
- `PRIMARY_START = 3`
- `LONG_START = 10`
- `NEURAL_CHECKPOINT_DIR = os.path.join(`
- `DATA_ROOT = "/content/drive/MyDrive/thesis/data"`
- `NORM_DIR = os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION")`
- `FULL_LABEL_PATH = os.path.join(NORM_DIR, "rq3_normalized_labels_full.csv")`
- `META_PATH = os.path.join(NORM_DIR, "rq3_label_normalization_metadata.json")`
- `OUT_DIR = os.path.join(DATA_ROOT, "RQ3_7LABEL_HISTORY_AWARE_PREDICTION")`
- `LABEL_COL = "rq3_process_label"`
- `GROUP_COL_LABEL = meta.get("group_col", None)`
- `TIME_COL_LABEL = meta.get("time_col", None)`
- `GROUP_COL_LABEL = first_existing(labels_df.columns, ["group", "group_id", "session", "session_id"])`
- `TIME_COL_LABEL = first_existing(labels_df.columns, ["window_start", "win_start", "start", "start_time", "window_mid", "time"])`
- `FEATURE_FILE_OVERRIDE = None`
- `KNOWN_FEATURE_FILES = [`
- `GROUP_COL = GROUP_COL_LABEL`
- `TIME_COL = TIME_COL_LABEL`
- `FEATURE_FILE = None`
- `FEATURE_FILE = feature_score_df.iloc[0]["file_path"]`
- `GROUP_COL_FEATURE = feature_score_df.iloc[0]["group_col"]`
- `TIME_COL_FEATURE = feature_score_df.iloc[0]["time_col"]`
- `GROUP_COL = GROUP_COL_FEATURE`
- `TIME_COL = TIME_COL_FEATURE`
- `META_COLS = set([`
- `HISTORY_LENS = [1, 2, 3, 5]`
- `USE_FEATURE_LAGS = True`
- `MAX_SENSOR_FEATURES = 80`
- `OUT_DIR = os.path.join(DATA_ROOT, "RQ3_7LABEL_SEGMENT_SENSOR_FORECAST")`
- `MIN_SEGMENT_WINDOWS = 1`
- `HISTORY_LENS = [1, 2, 3]`
- `MAX_FEATURES = 80`
- `LABEL_ORDER = sorted(segments["label"].unique())`
- `FORECAST_MODELS = ["ridge", "random_forest"]`
- `SEED = 42`
- `OUT_DIR = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3"`
- `INV_PATH = f"{OUT_DIR}/recognition_interaction_window_label_inventory.csv"`
- `ENG3_FEATURES_PATH = f"{OUT_DIR}/interaction_eng3_features.csv"`
- `RESULT_PREFIX = "task3_expanding_prefix_segment_prediction"`
- `MIN_WINDOWS = 15`
- `LABEL_MODES_TO_RUN = [`
- `FEATURE_MODES = [`
- `DEEP_MODELS = [`
- `MAX_EPOCHS = 80`
- `BATCH_SIZE = 32`
- `WEIGHT_DECAY = 1e-4`
- `EMB_DIM = 16`
- `HIDDEN_DIM = 64`
- `MAX_SUFFIX_ORDER = 5`
- `LABEL_COL = "dominant_normalized_label"`
- `OPTITRACK_FEATURES = [`
- `OPENEAREABLE_FEATURES = [`
- `XSENS_FEATURES = [`
- `CANDIDATE_FEATURES = (`
- `SENSOR_FEATURES = [c for c in CANDIDATE_FEATURES if c in features_df.columns]`
- `PATH = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/eng3_recognition_3class_core_features.csv"`
- `CORE = ["co_building","co_merging","conversation"]`
- `EMIT = [c for c in cand if c in df.columns]`
- `INPUT_DIR = "/content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`
- `GROUPS = [1,2,3,5,6,7,8,9,10]; WIN=30.0; K=3`
- `IND = ["label_Participant1","label_Participant2","label_Participant3"]`
- `PAIR = ["label_Participant1_Participant2","label_Participant1_Participant3","label_Participant2_Participant3"]`
- `TYPO = {"object_handiver":"object_handover"}`
- `KEEP = {"individual_phase","conversation","co_building","co_merging"}`
- `INPUT_DIR = f"{D}/ALL_MODEL_READY_FILES_IDENTITY_FIXED"`
- `OPTI2 = f"{D}/INTERACTION_OPTI2/interaction_opti2_10s.csv"`
- `OE10 = f"{D}/INTERACTION_OE10/interaction_oe10_10s.csv"`
- `SENS = pd.merge(opti,oe,on=["group","window_start","window_end"],how="outer")`
- `SCOLS = ocols+ecols`
- `CAT = [c for c in Xdf.columns if c.endswith(tuple(f"lag{l}" for l in range(1,K+1))) or c=="prev_state"]`
- `NUM = [c for c in SCOLS if c in Xdf.columns]`
- `T_LEGACY_TEXT_GROUP = T.assign(group=T["group"].astype(str))`
- `LEGACY_ROWS = []`
- `LEGACY_FOLD_ROWS = []`
- `HIST_LENS = [1, 2, 3, 5]`
- `EPS = 1e-9`
- `LAMBDAS = np.linspace(0.0, 1.0, 11)`
- `REPORT_REFERENCE = {`
- `ROOT_FOLDER = (`
- `SAMPLE_ROWS = 5000`
- `MAX_FILES = 200`
- `OUTPUT_FOLDER = os.path.join(`
- `SUPPORTED_EXTENSIONS = {`
- `TIMESTAMP_WORDS = [`
- `GROUP_WORDS = [`
- `PARTICIPANT_WORDS = [`
- `LABEL_WORDS = [`
- `SENSOR_WORDS = {`
- `k = min(K_SELECT, X_rows.shape[1])`
- `random_state = 42,`
- `n_layers = 3,`
- `dropout = 0.3,`
- `dropout = dropout,`
- `num_layers = n_layers,`
- `dropout = dropout if n_layers > 1 else 0.0,`
- `epochs = 100,`
- `learning_rate = 8e-4,`
- `k = K_SELECT,`
- `lr = learning_rate,`
- `seed = seed,`
- `n_folds = ('held_group', 'nunique'),`
- `n_estimators = 300,`
- `hidden_size = hidden_dim,`
- `num_layers = 1,`
- `hidden_dim = HIDDEN_DIM,`
- `lr = LR,`
- `batch_size = BATCH_SIZE,`

## Potentially broken / disabled cells

Manual review (automated heuristic found nothing entirely commented-out, but this notebook has several important structural quirks):

- **Critical hidden dependency on an notebook not included in this deliverable**: Cell 6 (code cell #3), which runs immediately after the global-config cell and before anything else, hard-requires a file to already exist at `/content/drive/MyDrive/thesis/data/PUBLICATION_TASK3_CORRECTED_FINAL/activity_tokens_6label_fullstat.csv` (`OLD_TOKEN_PATH`) — it `raise FileNotFoundError`s if that file is missing — then copies it to this notebook's own output location (`NEW_TOKEN_PATH` = `PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/activity_tokens_6label_fullstat.csv`). That `PUBLICATION_TASK3_CORRECTED_FINAL` folder is the output of an **earlier "Task 3 CORRECTED_FINAL" notebook that is not one of the 3 notebooks provided for this port**. This notebook is therefore not fully self-contained/reproducible from raw data alone as given — the historical "verified" token table is inherited, not regenerated, in the normal run path.
- **Consequence of the above — Part I's own token-building code is effectively dead code in normal runs**: Cell 8 defines `load_six_label_windows()` / `build_fullstat_tokens()` to build the six-label token table from raw label+sensor CSVs, but immediately after (same cell) checks `if RESUME_EXISTING and os.path.exists(TOKEN_PATH)` — and since cell 6 already copied the file to that exact `TOKEN_PATH`, this condition is normally already `True` before the "build" branch ever runs, so cell 8 just reloads the copied file (`T = pd.read_csv(TOKEN_PATH)`) rather than rebuilding it. The from-scratch build logic only executes if `RESUME_EXISTING=False` or the copied file is deleted. Important for porting: the from-scratch path (`load_six_label_windows`/`build_fullstat_tokens`) is real, working code, but it is not what actually produced the numbers in the published tables under the notebook's default configuration — the copied historical table was.
- **Self-documented no-op cell**: Cell 32 (code cell #23) is explicitly commented `# (merge now applied inside the data-building cell above; this cell is a no-op)` and just prints a label distribution. Intentional, not an error, but confirms the notebook was edited/refactored in place and left a stub cell behind.
- **Extensive `RESUME_EXISTING`/checkpoint reuse throughout**, same pattern as Task 2: Part II's grammar/sensor/hybrid models (cell 10), Part III's 3-seed neural runner (cell 13, "RESUMABLE THREE-SEED TOKEN MODEL RUNNER"), and multiple appendix cells all check for existing output CSVs / per-seed checkpoint files under `NEURAL_CHECKPOINT_DIR` etc. and skip recomputation when found. As in Task 2, the actual code path executed depends heavily on what already exists in `PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/` on Drive from a prior run. The notebook's own markdown (cell 3) states explicitly: "The neural section is resumable... Old neural checkpoints cannot be reloaded into this version" — i.e., the checkpoint format changed from an earlier version of this notebook and is not backward compatible.
- **The notebook is explicitly split into a mandatory "core" and optional, independently-runnable appendices** — this is stated directly in its own markdown (cell 3): run everything from the top through Part IV for the final headline Task 3 results; Appendices A (window-level history/persistence baseline), B (segment-level sensor-feature forecasting), C (fine-13 vs coarse-4 expanding-prefix prediction), and D (HMM/collective-state analyses) are optional supplementary analyses, each largely self-contained with its own data loading ("CELL 1 - SETUP" / "CELL 2 - LOAD..." internal numbering restarts inside each appendix, e.g. appendix A's cells 17-27, appendix B's cells 29-41, appendix C's single large cell 43, appendix D's cells 46-48). Appendix R (cells 50-54) is a **deliberate byte-for-byte reproduction of an older, methodologically different evaluation protocol** — the notebook's own markdown (cell 49) explains the historical report used a looser, per-model-inconsistent evaluation window and un-smoothed argmax, which Parts II-IV explicitly correct; Appendix R exists only to verify the corrected pipeline can still reproduce the original published Table 22 numbers exactly (cell 54 checks reproduced values against `REPORT_REFERENCE` with an 0.0006 tolerance and prints "EXACT"/"DIFFERS"). None of this is "broken" — it is intentional methodological documentation — but a porter should not assume Appendix R's numbers are the ones to trust; the notebook's own "Recommended thesis hierarchy" (cell 55) says the corrected Parts II-IV panel is the headline result and Appendix R is "the reproduction of record" for the older report only.
- Appendix D's cell 47/48 comments explicitly warn: "Viterbi results are smoothing results and must not be described as causal next-state prediction" and separately implements a genuine causal forward-filter version (`causal_next`) alongside the (non-causal) Viterbi decode — a correctness distinction to preserve during porting, not a bug.
- No `!pip install` cells found (the `hmmlearn` import in Appendix D presumably needs a `pip install hmmlearn` that isn't shown, since `hmmlearn` is not a standard Colab pre-install — likely done in an earlier, unincluded setup cell or assumed pre-installed). `drive.mount` is optional/graceful, same pattern as Tasks 1 and 2.
- Cell 56 (very last cell, "LARGE MODEL-READY DATA INSPECTION") is a diagnostic/exploratory utility unrelated to the main pipeline (samples rows from arbitrary large files under `INPUT_DIR`/`ROOT_FOLDER` and reports column-name patterns) — looks like leftover data-exploration tooling, not part of the actual Task 3 modeling pipeline. Safe to leave out of the port.

## Summary

This is by far the largest and most structurally complex of the three notebooks — it is Task 3 (activity-sequence / next-activity forecasting) and is explicitly organized as a mandatory core pipeline (Parts I-IV) plus four optional, mostly independent appendices (A-D) plus a legacy-reproduction appendix (R). **Part I** collapses consecutive same-label sensor windows into "activity tokens" (duration + 14 statistics per numeric sensor channel per token) for a 6-class label scheme, but in practice reuses a token table copied from an earlier, non-included "Task 3 CORRECTED_FINAL" notebook rather than rebuilding it from raw data. **Part II** compares deterministic n-gram/Markov "grammar" models (orders 1/2/3/5/10), a second-order HMM-like transition model, a sensor-only classifier, and a hybrid grammar+sensor model, all under a corrected common-target protocol where every model predicts token position 3 onward (217 shared targets) via Leave-One-Group-Out over the 9 study groups. **Part III** trains leakage-free Transformer/LSTM neural sequence models on the identical 217 targets, averaged over 3 seeds (42, 1, 7) with both seed-level and held-out-group-level standard deviation reported. **Part IV** merges these into final "publication" comparison tables (deterministic and neural kept in separate panels since their metric bases differ), written to `PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/`. The four **optional appendices** cover: (A) an older 7-label window-level history-aware persistence/Markov baseline (flagged by the notebook itself as not the headline comparison); (B) forecasting the next segment's raw sensor-feature vector and decoding it back to a label; (C) an expanding-prefix full-session-history LSTM/CNN1D/Transformer/Markov/suffix-backoff comparison at fine (13-label) vs. coarse (4-label) granularity; and (D) HMM and 5-class "collective state" analyses with both non-causal (Viterbi/smoothing) and causal (forward-filter) decoding, explicitly distinguished. **Appendix R** re-runs the original, methodologically looser evaluation protocol verbatim to numerically verify it still reproduces the historical thesis report's Table 22 (n-gram, HMM2, hybrid model accuracy/macro-F1 targets are hardcoded in `REPORT_REFERENCE` in cell 54 — e.g. `ngram_backoff_h2: (0.604, 0.499)`, `HMM2_categorical: (0.606, 0.443)` — these are excellent candidates to cross-check against `docs/thesis_reproduction_targets.md`). The notebook depends on several upstream Drive artifacts not produced by any of the 3 notebooks given here (most notably the Part-I token table and various `RQ3_LABEL_NORMALIZATION`/`INTERACTION_ENG3`/`ALL_MODEL_READY_FILES_IDENTITY_FIXED` feature files), so a full from-scratch port needs to either locate/regenerate those upstream files or treat them as required external inputs.
