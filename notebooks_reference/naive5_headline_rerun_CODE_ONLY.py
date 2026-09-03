# --- CELL 2 (code cell #1) ---
# ================================================================
# SETUP
# ================================================================

import os
import re
import gc
import json
import random
import warnings
import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_colwidth", None)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# --- CELL 3 (code cell #2) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 4 (code cell #3) ---
# =====================================================================
# THE ONLY SWITCH YOU CHANGE
# =====================================================================
RUN_ON = "naive"          # "all" = original 9 groups | "naive" = 5 naive groups

NAIVE_GROUPS    = {2, 3, 5, 6, 10}
RESEARCHER_GRPS = {1, 7, 8, 9}
TAG = "full9" if RUN_ON == "all" else "naive5"

import pandas as pd, numpy as np, re, os, json
RESULTS = {}          # every headline number lands here

def _gid(g):
    d = re.sub(r"\D", "", str(g))
    return int(d) if d else None

# Wrap read_csv so EVERY table with a 'group' column is filtered automatically.
if not getattr(pd, "_naive_patched", False):
    _orig_read_csv = pd.read_csv
    def _read_csv(*a, **kw):
        df = _orig_read_csv(*a, **kw)
        if RUN_ON == "naive" and isinstance(df, pd.DataFrame) and "group" in df.columns:
            keep = df["group"].map(_gid).isin(NAIVE_GROUPS)
            if keep.any():
                b = len(df); df = df[keep].copy()
                print(f"   [naive5] {os.path.basename(str(a[0]))[:42]:42s} {b:6d} -> {len(df):6d}")
        return df
    pd.read_csv = _read_csv
    pd._naive_patched = True

def log_result(part, name, **kv):
    RESULTS.setdefault(part, []).append({"config": name, **kv})
    print(f"   [logged] {part} | {name} | " + " ".join(f"{k}={v}" for k, v in kv.items()))

print("RUN_ON =", RUN_ON, "| TAG =", TAG)
print("naive groups:", sorted(NAIVE_GROUPS))
print("NOTE: 5 groups -> LOGO gives 5 folds instead of 9.")


# --- CELL 5 (code cell #4) ---
# ---- variance helper: per-group scores, mean +/- SD, 95% CI -------------
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

def variance_report(y_true, y_pred, groups, name, part, n_boot=2000, seed=0):
    d = pd.DataFrame({"g": list(groups), "yt": list(y_true), "yp": list(y_pred)})
    rows = []
    for g, sub in d.groupby("g"):
        rows.append({"group": g, "n": len(sub),
                     "accuracy": accuracy_score(sub.yt, sub.yp),
                     "macro_f1": f1_score(sub.yt, sub.yp, average="macro", zero_division=0)})
    per = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    pooled_f1  = f1_score(d.yt, d.yp, average="macro", zero_division=0)
    pooled_acc = accuracy_score(d.yt, d.yp)
    v = per["macro_f1"].values.astype(float)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    rng = np.random.default_rng(seed)
    if len(v) > 1:
        t_ci = stats.t.interval(0.95, len(v)-1, loc=m, scale=sd/np.sqrt(len(v)))
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
        b_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    else:
        t_ci = (m, m); b_ci = (m, m)
    print(f"\n--- {name} [{TAG}] ---")
    print(per.to_string(index=False))
    print(f"macro_f1: pooled={pooled_f1:.3f} | fold mean+/-SD={m:.3f}+/-{sd:.3f} "
          f"| CI95 t=[{t_ci[0]:.3f},{t_ci[1]:.3f}] boot=[{b_ci[0]:.3f},{b_ci[1]:.3f}]")
    log_result(part, name, pooled_macro_f1=round(pooled_f1,4), pooled_acc=round(pooled_acc,4),
               fold_mean=round(m,4), fold_sd=round(sd,4),
               ci_lo=round(float(t_ci[0]),4), ci_hi=round(float(t_ci[1]),4), n_folds=len(per))
    return per

print("variance_report() ready")


# --- CELL 7 (code cell #5) ---
# ================================================================
# CONFIG
# ================================================================

DATA_ROOT = "/content/drive/MyDrive/thesis/data"

# 10-second activity recognition dataset.
ACTIVITY_DATA_PATH = f"{DATA_ROOT}/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv"

# 5-second binary interaction dataset.
# Preferred path is the specialized OE merged dataset because it contains all sensors plus improved OE features.
INTERACTION_DATA_PATHS = [
    f"{DATA_ROOT}/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv",
    f"{DATA_ROOT}/INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv",
]

OUT_DIR = f"{DATA_ROOT}/ALL_SENSOR_MULTI_TASK_ABLATIONS"
os.makedirs(OUT_DIR, exist_ok=True)

RUN_TASKS = [
    "interaction_vs_noninteraction",
    "conversation_vs_nonconversation",
    "conversation_vs_building",
    "conversation_vs_merging",
    "merging_vs_building",
    "three_class_activity",
]

SENSOR_COMBINATIONS = {
    "OE": ["OE"],
    "OPTI": ["OPTI"],
    "XSENS": ["XSENS"],
    "OE_OPTI": ["OE", "OPTI"],
    "OE_XSENS": ["OE", "XSENS"],
    "OPTI_XSENS": ["OPTI", "XSENS"],
    "OE_OPTI_XSENS": ["OE", "OPTI", "XSENS"],
}

RUN_CLASSICAL = True
RUN_DL = True

# Classical grid.
K_CLASSICAL = [40, 80, 120, 200, "all"]
TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]

# DL grid.
K_DL = [80, 120, 200]
SEEDS = [42]
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 64

# Full model list = ["lstm", "bilstm", "gru", "transformer"]
# For a faster but still meaningful run, use ["gru", "transformer"].
RUN_DL_MODEL_TYPES = ["lstm", "bilstm", "gru", "transformer"]

# Smoke test: set this to 4 first to make sure everything runs.
# Full run: set to None.
STOP_AFTER_N_DL_RUNS_PER_TASK_SENSOR = None

RANDOM_STATE = 42


# --- CELL 8 (code cell #6) ---
# ================================================================
# GENERAL HELPERS
# ================================================================

def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def unique_feats(feats):
    return list(dict.fromkeys(feats))


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def infer_window_seconds(df, start_col, end_col, default):
    if start_col in df.columns and end_col in df.columns:
        dur = pd.to_numeric(df[end_col], errors="coerce") - pd.to_numeric(df[start_col], errors="coerce")
        val = float(np.nanmedian(dur))
        if np.isfinite(val) and val > 0:
            return val
    return float(default)


def add_elapsed_min(df, group_col, start_col, end_col):
    df = df.copy()
    if "elapsed_min" in df.columns:
        df["elapsed_min"] = pd.to_numeric(df["elapsed_min"], errors="coerce")
        return df

    if start_col not in df.columns:
        raise ValueError(f"Cannot create elapsed_min because {start_col} is missing.")

    if end_col in df.columns:
        df["window_mid"] = (
            pd.to_numeric(df[start_col], errors="coerce") +
            pd.to_numeric(df[end_col], errors="coerce")
        ) / 2.0
    else:
        df["window_mid"] = pd.to_numeric(df[start_col], errors="coerce")

    df["elapsed_min"] = (df["window_mid"] - df.groupby(group_col)["window_mid"].transform("min")) / 60.0
    return df


BAD_TOKENS = [
    "label", "target", "class", "group", "window", "time", "elapsed",
    "pred", "prediction", "correct", "fold", "split", "index"
]


def looks_bad_feature_name(c):
    cl = str(c).lower()
    return any(tok in cl for tok in BAD_TOKENS)


def is_oe_feature(c):
    c = str(c)
    cl = c.lower()
    if looks_bad_feature_name(c):
        return False
    return (
        c.startswith("oe__")
        or c.startswith("oe_")
        or c.startswith("oebest__")
        or c.startswith("ear_")
        or c.startswith("mag__")
        or c.startswith("mag_")
    )


def is_opti_feature(c):
    c = str(c)
    cl = c.lower()
    if looks_bad_feature_name(c):
        return False

    # Current advanced merged datasets normally use opti2_ / opti2__ prefixes.
    if c.startswith("opti2_") or c.startswith("opti2__") or c.startswith("opti__") or c.startswith("opti_"):
        return True

    # Fallback for older unprefixed OptiTrack feature names.
    # Kept conservative to avoid catching OE/XSens features.
    fallback_tokens = [
        "dist_close", "dist_mid", "dist_far", "dist_disp",
        "centroid", "spread", "triangle", "perimeter", "compactness",
        "nearest", "farthest", "pairdist", "pair_dist",
        "approach", "separation", "proximity", "relative_pos",
    ]
    if any(tok in cl for tok in fallback_tokens):
        return True

    # Some old OptiTrack columns used simple speed names.
    if cl in {"speed_min", "speed_mid", "speed_max", "centroid_speed"}:
        return True

    return False


def is_xsens_feature(c):
    c = str(c)
    cl = c.lower()
    if looks_bad_feature_name(c):
        return False
    return (
        c.startswith("xsens2__")
        or c.startswith("xsens2_")
        or c.startswith("xsens__")
        or c.startswith("xsens_")
    )


def clean_feature_list(dataframe, feats, label_cols=None, include_elapsed=False):
    label_cols = set(label_cols or [])
    bad_cols = set(label_cols) | {
        "group", "window_start", "window_end", "window_mid", "video_time_s", "time_s",
        "label", "target", "class", "activity", "activity_class", "general_class",
        "binary_label", "recognition_label", "task_label", "pred", "prediction", "correct",
    }

    cleaned = []
    for f in unique_feats(feats):
        if f not in dataframe.columns:
            continue
        if f in bad_cols:
            continue

        fl = str(f).lower()
        if f == "elapsed_min":
            if include_elapsed:
                cleaned.append(f)
            continue

        if not include_elapsed and "elapsed" in fl:
            continue
        if any(tok in fl for tok in ["label", "target", "pred", "prediction", "correct"]):
            continue

        x = pd.to_numeric(dataframe[f], errors="coerce").values
        if np.isfinite(x).sum() < 20:
            continue
        if np.nanstd(x) < 1e-12:
            continue
        cleaned.append(f)

    return unique_feats(cleaned)


def make_classical_models(n_classes):
    return {
        "logreg_C1": LogisticRegression(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear" if n_classes == 2 else "lbfgs",
            multi_class="auto",
            random_state=RANDOM_STATE,
        ),
        "linearSVC_C1": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=10000,
            dual=False,
        ),
        "rbfSVC_C1_gscale": SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "rf_leaf2": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extraTrees_leaf1": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ------------------------------------------------
# Protect against extreme values after RobustScaler
# ------------------------------------------------
def _clip_for_float32(X, limit=1e6):
    """
    RobustScaler can create extremely large but still finite float64 values
    when a feature has an almost-zero training IQR and an extreme test value.

    Tree models such as RandomForest internally use float32, so those values
    can overflow. Clipping at +/-1e6 only affects pathological scaled values
    and leaves the normal scaled feature range unchanged.
    """
    X = np.asarray(X, dtype=np.float64)

    # Safety fallback in case scaling ever produces NaN/inf
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=limit,
        neginf=-limit
    )

    return np.clip(X, -limit, limit)


def build_pipeline(model, k, n_features):
    from sklearn.preprocessing import FunctionTransformer

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),

        # IMPORTANT:
        # clip AFTER scaling, because that is where the overflow occurs
        ("clip", FunctionTransformer(
            _clip_for_float32,
            validate=False
        )),
    ]

    if k != "all":
        actual_k = min(int(k), n_features)
        steps.append(
            ("select", SelectKBest(
                f_classif,
                k=actual_k
            ))
        )

    steps.append(("model", clone(model)))

    return Pipeline(steps)


def metric_dict(y_true, y_pred, label_order, prefix=""):
    out = {
        prefix + "accuracy": accuracy_score(y_true, y_pred),
        prefix + "macro_f1": f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0),
        prefix + "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )

    for lab, p, r, f, s in zip(label_order, pr, rc, f1, sup):
        safe = safe_name(lab)
        out[prefix + f"precision_{safe}"] = p
        out[prefix + f"recall_{safe}"] = r
        out[prefix + f"f1_{safe}"] = f
        out[prefix + f"support_{safe}"] = int(s)

    return out


# --- CELL 9 (code cell #7) ---
# ================================================================
# LOAD DATASETS AND CREATE TASKS
# ================================================================

# ---------- Load activity 10s dataset ----------
if not os.path.exists(ACTIVITY_DATA_PATH):
    raise FileNotFoundError(
        f"Could not find activity dataset:\n{ACTIVITY_DATA_PATH}\n"
        "Run the advanced merged 10s feature-generation cell first."
    )

activity_df_raw = pd.read_csv(ACTIVITY_DATA_PATH)
activity_df_raw["recognition_label"] = activity_df_raw["recognition_label"].astype(str).str.strip()
activity_df_raw = add_elapsed_min(activity_df_raw, "group", "window_start", "window_end")
activity_window_s = infer_window_seconds(activity_df_raw, "window_start", "window_end", default=10.0)

print("Loaded activity dataset:", ACTIVITY_DATA_PATH)
print("Shape:", activity_df_raw.shape)
print("Estimated window seconds:", activity_window_s)
display(activity_df_raw["recognition_label"].value_counts())

# ---------- Load binary interaction 5s dataset ----------
interaction_path = find_first_existing(INTERACTION_DATA_PATHS)
if interaction_path is None:
    print("WARNING: interaction binary dataset not found. Interaction task will be skipped.")
    interaction_df_raw = None
    interaction_window_s = 5.0
else:
    interaction_df_raw = pd.read_csv(interaction_path)
    interaction_df_raw["binary_label"] = interaction_df_raw["binary_label"].astype(str).str.strip()
    interaction_df_raw = add_elapsed_min(interaction_df_raw, "group", "window_start", "window_end")
    interaction_window_s = infer_window_seconds(interaction_df_raw, "window_start", "window_end", default=5.0)

    print("\nLoaded interaction dataset:", interaction_path)
    print("Shape:", interaction_df_raw.shape)
    print("Estimated window seconds:", interaction_window_s)
    display(interaction_df_raw["binary_label"].value_counts())


# ---------- Feature extraction per dataset ----------
def get_modality_features(df, label_cols=None):
    label_cols = label_cols or []

    oe_raw = [c for c in df.columns if is_oe_feature(c)]
    opti_raw = [c for c in df.columns if is_opti_feature(c)]
    xsens_raw = [c for c in df.columns if is_xsens_feature(c)]

    # Make modality sets mutually exclusive if broad fallback captured something unexpectedly.
    oe = clean_feature_list(df, oe_raw, label_cols=label_cols, include_elapsed=False)
    opti = clean_feature_list(df, [c for c in opti_raw if c not in oe], label_cols=label_cols, include_elapsed=False)
    xsens = clean_feature_list(df, [c for c in xsens_raw if c not in oe and c not in opti], label_cols=label_cols, include_elapsed=False)

    return {"OE": oe, "OPTI": opti, "XSENS": xsens}


def make_combo_feature_sets(df, modality_features, label_cols=None):
    combo_sets = {}
    combo_sets_elapsed = {}

    for combo_name, modalities in SENSOR_COMBINATIONS.items():
        feats = []
        for m in modalities:
            feats.extend(modality_features.get(m, []))
        feats = clean_feature_list(df, feats, label_cols=label_cols, include_elapsed=False)
        feats_elapsed = clean_feature_list(
            df,
            feats + (["elapsed_min"] if "elapsed_min" in df.columns else []),
            label_cols=label_cols,
            include_elapsed=True,
        )
        combo_sets[combo_name] = feats
        combo_sets_elapsed[combo_name] = feats_elapsed

    return combo_sets, combo_sets_elapsed


# ---------- Task preparation ----------
def prepare_task(task_name):
    if task_name == "interaction_vs_noninteraction":
        if interaction_df_raw is None:
            return None

        df = interaction_df_raw.copy()
        label_col = "binary_label"
        labels = ["interaction", "non_interaction"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [6, 12, 18]   # 5s windows -> 30/60/90s
        window_s = interaction_window_s

    elif task_name == "conversation_vs_nonconversation":
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        core = ["co_building", "co_merging", "conversation"]
        df = df[df[label_col].isin(core)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = np.where(df[label_col] == "conversation", "conversation", "non_conversation")
        labels = ["conversation", "non_conversation"]
        default_seq_lens = [3, 6, 9]
        window_s = activity_window_s

    elif task_name == "conversation_vs_building":
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        labels = ["conversation", "co_building"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [3, 6, 9]
        window_s = activity_window_s

    elif task_name == "conversation_vs_merging":
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        labels = ["conversation", "co_merging"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [3, 6, 9]
        window_s = activity_window_s

    elif task_name == "merging_vs_building":
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        labels = ["co_merging", "co_building"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [3, 6, 9]
        window_s = activity_window_s

    elif task_name == "three_class_activity":
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        labels = ["co_building", "co_merging", "conversation"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [3, 6, 9]
        window_s = activity_window_s

    else:
        raise ValueError(f"Unknown task: {task_name}")

    if len(df) == 0:
        print("Skipping empty task:", task_name)
        return None

    modality_features = get_modality_features(df, label_cols=[label_col, target_col])
    combo_sets, combo_sets_elapsed = make_combo_feature_sets(df, modality_features, label_cols=[label_col, target_col])

    spec = {
        "task_name": task_name,
        "df": df,
        "label_col": label_col,
        "target_col": target_col,
        "label_order": labels,
        "group_col": "group",
        "start_col": "window_start",
        "end_col": "window_end",
        "window_seconds": window_s,
        "seq_lens": default_seq_lens,
        "modality_features": modality_features,
        "combo_features": combo_sets,
        "combo_features_elapsed": combo_sets_elapsed,
        "out_dir": os.path.join(OUT_DIR, task_name),
    }
    os.makedirs(spec["out_dir"], exist_ok=True)
    return spec


task_specs = []
for task in RUN_TASKS:
    spec = prepare_task(task)
    if spec is not None:
        task_specs.append(spec)

print("\n" + "=" * 100)
print("PREPARED TASKS AND SENSOR COMBINATIONS")
print("=" * 100)

for spec in task_specs:
    print("\nTASK:", spec["task_name"])
    print("Rows:", len(spec["df"]))
    print("Classes:", spec["label_order"])
    print("Window seconds:", spec["window_seconds"])
    print("Seq lens:", spec["seq_lens"])
    print("Modality counts:", {k: len(v) for k, v in spec["modality_features"].items()})
    print("Combination counts, no elapsed:", {k: len(v) for k, v in spec["combo_features"].items()})
    print("Combination counts, with elapsed:", {k: len(v) for k, v in spec["combo_features_elapsed"].items()})
    display(spec["df"][spec["target_col"]].value_counts())
    display(pd.crosstab(spec["df"][spec["group_col"]], spec["df"][spec["target_col"]]))

    feature_count_rows = []
    for combo in SENSOR_COMBINATIONS:
        feature_count_rows.append({
            "task": spec["task_name"],
            "sensor_combo": combo,
            "n_no_elapsed": len(spec["combo_features"][combo]),
            "n_with_elapsed": len(spec["combo_features_elapsed"][combo]),
        })
    pd.DataFrame(feature_count_rows).to_csv(os.path.join(spec["out_dir"], f"{spec['task_name']}_feature_counts.csv"), index=False)


# --- CELL 10 (code cell #8) ---
# =====================================================================
# FIX D — sanitize non-finite feature values
# Xsens ratio/synchrony features can be +/-inf. SimpleImputer fills NaN
# but NOT inf -> ValueError. Convert inf to NaN so the in-fold median
# imputer handles them. No leakage: the imputer is still fit per fold.
# =====================================================================
import numpy as np

def sanitize_spec(spec):
    feats = set()
    for key in ("combo_features", "combo_features_elapsed"):
        for v in spec.get(key, {}).values():
            feats.update(v)
    feats = [f for f in feats if f in spec["df"].columns]
    if not feats:
        return

    df    = spec["df"]
    block = df[feats].apply(pd.to_numeric, errors="coerce")
    arr   = block.to_numpy(dtype="float64")

    inf_mask = np.isinf(arr)
    n_inf    = int(inf_mask.sum())
    bad_cols = list(block.columns[inf_mask.any(axis=0)])

    block = block.replace([np.inf, -np.inf], np.nan)
    df[feats] = block                      # write back as clean numerics

    # drop features that are now entirely NaN or constant (f_classif -> NaN)
    keep  = block.notna().any(axis=0) & (block.nunique(dropna=True) > 1)
    dead  = set(block.columns[~keep])
    if dead:
        for key in ("combo_features", "combo_features_elapsed"):
            for combo, v in spec.get(key, {}).items():
                spec[key][combo] = [f for f in v if f not in dead]

    print(f"{spec['task_name']:35s} inf={n_inf:6d} in {len(bad_cols):3d} cols | dropped {len(dead)} dead feats")
    return bad_cols

all_bad = set()
for spec in task_specs:
    b = sanitize_spec(spec)
    if b: all_bad.update(b)

print(f"\ncolumns that contained inf ({len(all_bad)}):")
for c in sorted(all_bad)[:40]:
    print("  ", c)
if len(all_bad) > 40:
    print(f"   ... and {len(all_bad)-40} more")


# --- CELL 11 (code cell #9) ---
# =====================================================================
# NARROW THE GRID — FIXED sensor-key matching
# The old cell used "OE+OPTI" but this notebook keys them "OE_OPTI",
# so only "OPTI" survived. Match on the SET of modalities instead.
# =====================================================================
import time
RUN_START = time.time()          # used later to ignore stale result files

print("keys available:", list(SENSOR_COMBINATIONS))

def _canon(k):
    return frozenset(p for p in re.split(r"[+_\s]+", str(k).upper().strip()) if p)

# all seven combinations = the full ablation reported in the thesis
WANT = ["OE", "OPTI", "XSENS", "OE+OPTI", "OE+XSENS", "OPTI+XSENS", "OE+OPTI+XSENS"]
_want = {_canon(w) for w in WANT}

SENSOR_COMBINATIONS = {k: v for k, v in SENSOR_COMBINATIONS.items() if _canon(k) in _want}
_missing = _want - {_canon(k) for k in SENSOR_COMBINATIONS}
if _missing:
    print("!! NOT FOUND:", ["+".join(sorted(m)) for m in _missing],
          "-> re-run the config cell that defines SENSOR_COMBINATIONS")

K_CLASSICAL         = [80, 200]
FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]
FAST_DL_K           = 120
FAST_DL_SEQ_CHOICE  = "last"

SAVE_FAST_DL_PREDICTIONS = True   # needed for per-group DL variance

print("sensor combos :", list(SENSOR_COMBINATIONS), f"({len(SENSOR_COMBINATIONS)})")
print("classical k   :", K_CLASSICAL)
print("DL models     :", FAST_DL_MODEL_TYPES, "| k =", FAST_DL_K)
print("tasks         :", list(RUN_TASKS))


# --- CELL 13 (code cell #10) ---
# ================================================================
# CLASSICAL LOGO RUNNER
# ================================================================

def run_classical_for_task_and_sensor(spec, sensor_combo):
    task_name = spec["task_name"]
    df = spec["df"]
    y = df[spec["target_col"]].astype(str).values
    groups = df[spec["group_col"]].values
    label_order = spec["label_order"]
    n_classes = len(label_order)

    models = make_classical_models(n_classes)

    summary_rows = []
    pred_rows = []

    for time_condition in TIME_CONDITIONS:
        feats = spec["combo_features"][sensor_combo] if time_condition == "no_elapsed" else spec["combo_features_elapsed"][sensor_combo]

        if len(feats) == 0:
            print(f"Skipping classical | {task_name} | {sensor_combo} | {time_condition}: no features")
            continue

        for model_name, model in models.items():
            for k in K_CLASSICAL:
                if k != "all" and int(k) > len(feats):
                    continue

                print(
                    f"Running classical | {task_name} | {sensor_combo} | {time_condition} | "
                    f"{model_name} | k={k} | n_features={len(feats)}"
                )

                X = df[feats].apply(pd.to_numeric, errors="coerce").values
                pipe = build_pipeline(model, k, len(feats))
                logo = LeaveOneGroupOut()

                y_true_all = []
                y_pred_all = []
                group_all = []
                row_index_all = []

                for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups), start=1):
                    if len(np.unique(y[tr_idx])) < 2:
                        continue

                    pipe_fold = clone(pipe)
                    pipe_fold.fit(X[tr_idx], y[tr_idx])
                    pred = pipe_fold.predict(X[te_idx])

                    y_true_all.extend(y[te_idx])
                    y_pred_all.extend(pred)
                    group_all.extend(groups[te_idx])
                    row_index_all.extend(te_idx)

                if len(y_true_all) == 0:
                    continue

                metrics = metric_dict(np.array(y_true_all), np.array(y_pred_all), label_order)
                metrics.update({
                    "task": task_name,
                    "sensor_combo": sensor_combo,
                    "time_condition": time_condition,
                    "model": model_name,
                    "k": k,
                    "n_features": len(feats),
                    "n_rows_evaluated": len(y_true_all),
                })
                summary_rows.append(metrics)

                for idx, g, yt, yp in zip(row_index_all, group_all, y_true_all, y_pred_all):
                    pred_rows.append({
                        "task": task_name,
                        "sensor_combo": sensor_combo,
                        "time_condition": time_condition,
                        "model": model_name,
                        "k": k,
                        "row_index": int(idx),
                        "group": g,
                        "y_true": yt,
                        "y_pred": yp,
                        "correct": yt == yp,
                    })

    summary = pd.DataFrame(summary_rows)
    preds = pd.DataFrame(pred_rows)

    if len(summary) == 0:
        return summary, preds, pd.DataFrame()

    summary = summary.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

    best_per_condition = (
        summary
        .sort_values(["time_condition", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("time_condition", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    out_dir = os.path.join(spec["out_dir"], sensor_combo)
    os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_summary.csv"), index=False)
    preds.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_predictions.csv"), index=False)
    best_per_condition.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_best_per_condition.csv"), index=False)

    print("\nTop classical results for", task_name, sensor_combo)
    display(summary.head(10).round(4))
    print("\nBest classical per time condition for", task_name, sensor_combo)
    display(best_per_condition.round(4))

    return summary, preds, best_per_condition


all_classical_summaries = []
all_classical_best = []

if RUN_CLASSICAL:
    for spec in task_specs:
        for sensor_combo in SENSOR_COMBINATIONS:
            print("\n" + "#" * 120)
            print("CLASSICAL TASK:", spec["task_name"], "| SENSOR:", sensor_combo)
            print("#" * 120)
            summary, preds, best = run_classical_for_task_and_sensor(spec, sensor_combo)
            if len(summary) > 0:
                all_classical_summaries.append(summary)
            if len(best) > 0:
                all_classical_best.append(best)

    if len(all_classical_summaries) > 0:
        combined_classical = pd.concat(all_classical_summaries, ignore_index=True)
        combined_classical.to_csv(os.path.join(OUT_DIR, "combined_classical_summary.csv"), index=False)

    if len(all_classical_best) > 0:
        combined_classical_best = pd.concat(all_classical_best, ignore_index=True)
        combined_classical_best.to_csv(os.path.join(OUT_DIR, "combined_classical_best_per_condition.csv"), index=False)

        print("\n" + "=" * 100)
        print("COMBINED CLASSICAL BESTS")
        print("=" * 100)
        display(combined_classical_best.round(4))


# --- CELL 15 (code cell #11) ---
# ================================================================
# DL MODEL HELPERS
# ================================================================

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, rnn_type="lstm", hidden_dim=64, num_layers=1, dropout=0.25, bidirectional=False):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        out = self.encoder(x)
        last = out[:, -1, :]
        return self.head(last)


ALL_MODEL_CONFIGS = [
    {"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "bilstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4},
]

MODEL_CONFIGS = [m for m in ALL_MODEL_CONFIGS if m["model_type"] in RUN_DL_MODEL_TYPES]
print("DL model types:", [m["model_type"] for m in MODEL_CONFIGS])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_cfg, input_dim, n_classes):
    mt = model_cfg["model_type"]
    if mt == "lstm":
        return RNNClassifier(input_dim, n_classes, rnn_type="lstm", hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"], bidirectional=False)
    if mt == "bilstm":
        return RNNClassifier(input_dim, n_classes, rnn_type="lstm", hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"], bidirectional=True)
    if mt == "gru":
        return RNNClassifier(input_dim, n_classes, rnn_type="gru", hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"], bidirectional=False)
    if mt == "transformer":
        return TransformerClassifier(input_dim, n_classes, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_layers=model_cfg["num_layers"], dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"])
    raise ValueError(mt)


def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_sequences(X, y, groups, starts, seq_len):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    starts = np.asarray(starts, dtype=float)

    Xs, ys, gs, sts = [], [], [], []

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(starts[idx])]
        if len(idx) < seq_len:
            continue
        for end_pos in range(seq_len - 1, len(idx)):
            win_idx = idx[end_pos - seq_len + 1:end_pos + 1]
            Xs.append(X[win_idx])
            ys.append(y[idx[end_pos]])
            gs.append(g)
            sts.append(starts[idx[end_pos]])

    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.array([]), np.array([]), np.array([])

    return np.stack(Xs).astype(np.float32), np.array(ys), np.array(gs), np.array(sts)


def torch_predict(model, X, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def choose_validation_group(train_groups, y_all, groups_all):
    candidates = []
    for g in sorted(np.unique(train_groups)):
        mask = groups_all == g
        n_classes = len(np.unique(y_all[mask]))
        n_rows = int(mask.sum())
        candidates.append((n_classes >= 2, n_rows, g))
    candidates = sorted(candidates, reverse=True)
    return candidates[0][2]


# --- CELL 17 (code cell #12) ---
# ================================================================
# FAST DL TRAINING / EVALUATION
# Run this instead of the full DL grid.
# ================================================================

import os
import gc
import time
import numpy as np
import pandas as pd

from IPython.display import display
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

# ------------------------------------------------
# FAST DL SETTINGS
# ------------------------------------------------

RUN_DL = True

# Keep all tasks and all sensor combinations.
# This gives one fast but meaningful DL result per task/sensor.
FAST_DL_TASKS_TO_RUN = None
FAST_DL_SENSOR_COMBOS_TO_RUN = None

# Use only one strong model first.
# GRU is usually faster than Transformer and often stable.
FAST_DL_MODEL_TYPES = ["bilstm", "transformer"]

# Use only one k value.
# The code automatically uses min(k, number of available features).
FAST_DL_K = 120

# Use only one sequence length per task.
# "last" means longest context:
# interaction 5s: seq_len 18 = 90s
# activity 10s: seq_len 9 = 90s
FAST_DL_SEQ_CHOICE = "last"

# Much faster training.
FAST_MAX_EPOCHS = 25
FAST_PATIENCE = 4
FAST_BATCH_SIZE = 256
FAST_PRED_BATCH_SIZE = 1024

# Full LOGO is still honest.
# Set to 3 only for a quick smoke test, not final thesis result.
MAX_LOGO_FOLDS = None

# Save row-level predictions?
# False is much faster and lighter.
SAVE_FAST_DL_PREDICTIONS = False

FAST_DL_OUT_DIR = os.path.join(OUT_DIR, "FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED")
os.makedirs(FAST_DL_OUT_DIR, exist_ok=True)

print("FAST_DL_OUT_DIR:", FAST_DL_OUT_DIR)


def should_run_fast_dl_task(task_name):
    if FAST_DL_TASKS_TO_RUN is None:
        return True
    return task_name in FAST_DL_TASKS_TO_RUN


def should_run_fast_dl_sensor(sensor_combo):
    if FAST_DL_SENSOR_COMBOS_TO_RUN is None:
        return True
    return sensor_combo in FAST_DL_SENSOR_COMBOS_TO_RUN


def choose_fast_seq_len(spec):
    seq_lens = list(spec["seq_lens"])

    if FAST_DL_SEQ_CHOICE == "last":
        return seq_lens[-1]
    if FAST_DL_SEQ_CHOICE == "first":
        return seq_lens[0]
    if FAST_DL_SEQ_CHOICE == "middle":
        return seq_lens[len(seq_lens) // 2]

    return int(FAST_DL_SEQ_CHOICE)


def get_fast_model_configs():
    return [
        cfg for cfg in MODEL_CONFIGS
        if cfg["model_type"] in FAST_DL_MODEL_TYPES
    ]


def torch_predict_fast(model, X, batch_size=FAST_PRED_BATCH_SIZE):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().numpy())

    return np.array(preds)


def train_one_fast_dl_run(spec, sensor_combo, seq_len, k_features, model_cfg, seed):
    set_seed(seed)

    task_name = spec["task_name"]
    df = spec["df"]

    feature_list = spec["combo_features"][sensor_combo].copy()
    label_order = spec["label_order"]
    n_classes = len(label_order)

    if len(feature_list) == 0:
        return None, None, None

    label_to_id = {lab: i for i, lab in enumerate(label_order)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    y_label = df[spec["target_col"]].astype(str).values
    y_all = np.array([label_to_id[v] for v in y_label], dtype=int)

    groups_all = df[spec["group_col"]].values
    starts_all = pd.to_numeric(df[spec["start_col"]], errors="coerce").values

    X_raw = df[feature_list].apply(pd.to_numeric, errors="coerce").values

    logo = LeaveOneGroupOut()

    y_true_all = []
    y_pred_all = []
    fold_rows = []
    pred_rows = []

    fold_counter = 0

    for fold, (trval_idx, te_idx) in enumerate(logo.split(X_raw, y_all, groups_all), start=1):

        fold_counter += 1
        if MAX_LOGO_FOLDS is not None and fold_counter > MAX_LOGO_FOLDS:
            break

        if len(np.unique(y_all[trval_idx])) < 2:
            continue

        test_group = groups_all[te_idx][0]

        train_groups = np.unique(groups_all[trval_idx])
        val_group = choose_validation_group(train_groups, y_all, groups_all)

        val_mask = groups_all[trval_idx] == val_group
        val_idx = trval_idx[val_mask]
        tr_idx = trval_idx[~val_mask]

        if len(np.unique(y_all[tr_idx])) < 2:
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()

        Xtr = imputer.fit_transform(X_raw[tr_idx])
        Xval = imputer.transform(X_raw[val_idx])
        Xte = imputer.transform(X_raw[te_idx])

        Xtr = scaler.fit_transform(Xtr)
        Xval = scaler.transform(Xval)
        Xte = scaler.transform(Xte)

        # Protect against extreme scaled values before conversion to float32
        Xtr = _clip_for_float32(Xtr)
        Xval = _clip_for_float32(Xval)
        Xte = _clip_for_float32(Xte)

        actual_k = min(int(k_features), Xtr.shape[1])

        selector = SelectKBest(f_classif, k=actual_k)
        Xtr = selector.fit_transform(Xtr, y_all[tr_idx])
        Xval = selector.transform(Xval)
        Xte = selector.transform(Xte)

        Xtr_seq, ytr_seq, _, _ = make_sequences(
            Xtr, y_all[tr_idx], groups_all[tr_idx], starts_all[tr_idx], seq_len
        )
        Xval_seq, yval_seq, _, _ = make_sequences(
            Xval, y_all[val_idx], groups_all[val_idx], starts_all[val_idx], seq_len
        )
        Xte_seq, yte_seq, gte_seq, ste_seq = make_sequences(
            Xte, y_all[te_idx], groups_all[te_idx], starts_all[te_idx], seq_len
        )

        if len(Xtr_seq) == 0 or len(Xval_seq) == 0 or len(Xte_seq) == 0:
            continue

        model = build_model(model_cfg, input_dim=actual_k, n_classes=n_classes).to(DEVICE)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=model_cfg["lr"],
            weight_decay=model_cfg["weight_decay"]
        )

        loss_fn = nn.CrossEntropyLoss()

        train_loader = make_loader(
            Xtr_seq,
            ytr_seq,
            batch_size=FAST_BATCH_SIZE,
            shuffle=True
        )

        best_state = None
        best_val_macro = -1
        patience_left = FAST_PATIENCE
        best_epoch = 0

        for epoch in range(1, FAST_MAX_EPOCHS + 1):
            model.train()

            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            val_pred = torch_predict_fast(model, Xval_seq)
            val_macro = f1_score(yval_seq, val_pred, average="macro", zero_division=0)

            if val_macro > best_val_macro:
                best_val_macro = val_macro
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_left = FAST_PATIENCE
                best_epoch = epoch
            else:
                patience_left -= 1

            if patience_left <= 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(DEVICE)

        test_pred = torch_predict_fast(model, Xte_seq)

        y_true_all.extend(yte_seq.tolist())
        y_pred_all.extend(test_pred.tolist())

        fold_metrics_ids = metric_dict(yte_seq, test_pred, list(range(n_classes)))

        fold_rows.append({
            "task": task_name,
            "sensor_combo": sensor_combo,
            "fold": fold,
            "test_group": test_group,
            "model_type": model_cfg["model_type"],
            "seed": seed,
            "seq_len": seq_len,
            "context_seconds": seq_len * spec["window_seconds"],
            "k_features": actual_k,
            "n_features_before_select": len(feature_list),
            "n_sequences": len(Xte_seq),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro,
            "accuracy": fold_metrics_ids["accuracy"],
            "macro_f1": fold_metrics_ids["macro_f1"],
            "balanced_accuracy": fold_metrics_ids["balanced_accuracy"],
        })

        if SAVE_FAST_DL_PREDICTIONS:
            for yt, yp, g, st in zip(yte_seq, test_pred, gte_seq, ste_seq):
                pred_rows.append({
                    "task": task_name,
                    "sensor_combo": sensor_combo,
                    "model_type": model_cfg["model_type"],
                    "seed": seed,
                    "seq_len": seq_len,
                    "context_seconds": seq_len * spec["window_seconds"],
                    "k_features": actual_k,
                    "group": g,
                    "window_start": st,
                    "y_true": id_to_label[int(yt)],
                    "y_pred": id_to_label[int(yp)],
                    "correct": int(yt) == int(yp),
                })

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(y_true_all) == 0:
        return None, None, None

    y_true_lab = np.array([id_to_label[int(v)] for v in y_true_all])
    y_pred_lab = np.array([id_to_label[int(v)] for v in y_pred_all])

    summary = metric_dict(y_true_lab, y_pred_lab, label_order)

    summary.update({
        "task": task_name,
        "sensor_combo": sensor_combo,
        "model_type": model_cfg["model_type"],
        "seed": seed,
        "seq_len": seq_len,
        "context_seconds": seq_len * spec["window_seconds"],
        "k_features": int(k_features),
        "n_features_before_select": len(feature_list),
        "n_sequences_evaluated": len(y_true_lab),
        "time_condition": "no_elapsed",
        "feature_set": sensor_combo,
        "fast_dl": True,
        "max_epochs": FAST_MAX_EPOCHS,
        "patience": FAST_PATIENCE,
        "max_logo_folds": MAX_LOGO_FOLDS,
    })

    return summary, pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


# ================================================================
# RUN FAST DL
# ================================================================

fast_model_configs = get_fast_model_configs()

planned = []

for spec in task_specs:
    if not should_run_fast_dl_task(spec["task_name"]):
        continue

    seq_len = choose_fast_seq_len(spec)

    for sensor_combo in SENSOR_COMBINATIONS:
        if not should_run_fast_dl_sensor(sensor_combo):
            continue

        if sensor_combo not in spec["combo_features"]:
            continue

        if len(spec["combo_features"][sensor_combo]) == 0:
            continue

        for seed in SEEDS:
            for model_cfg in fast_model_configs:
                planned.append({
                    "task": spec["task_name"],
                    "sensor_combo": sensor_combo,
                    "seq_len": seq_len,
                    "k": min(FAST_DL_K, len(spec["combo_features"][sensor_combo])),
                    "model": model_cfg["model_type"],
                    "seed": seed,
                    "n_features_before_select": len(spec["combo_features"][sensor_combo]),
                })

planned_df = pd.DataFrame(planned)

print("=" * 100)
print("FAST DL PLAN")
print("=" * 100)
print("Number of displayed DL configs:", len(planned_df))

if MAX_LOGO_FOLDS is None:
    print("LOGO folds: full leave-one-group-out")
else:
    print("LOGO folds limited to:", MAX_LOGO_FOLDS)

display(planned_df)

planned_df.to_csv(os.path.join(FAST_DL_OUT_DIR, "fast_dl_plan.csv"), index=False)

all_fast_summaries = []
all_fast_folds = []
all_fast_preds = []

t0 = time.time()

for i, row in planned_df.iterrows():
    spec = next(s for s in task_specs if s["task_name"] == row["task"])
    model_cfg = next(c for c in fast_model_configs if c["model_type"] == row["model"])

    print("\n" + "#" * 120)
    print(
        f"FAST DL RUN {i + 1}/{len(planned_df)} | "
        f"{row['task']} | {row['sensor_combo']} | "
        f"{row['model']} | seq_len={row['seq_len']} | k={row['k']} | seed={row['seed']}"
    )
    print("#" * 120)

    run_t0 = time.time()

    summary, fold_df, pred_df = train_one_fast_dl_run(
        spec=spec,
        sensor_combo=row["sensor_combo"],
        seq_len=int(row["seq_len"]),
        k_features=int(row["k"]),
        model_cfg=model_cfg,
        seed=int(row["seed"]),
    )

    run_minutes = (time.time() - run_t0) / 60

    if summary is None:
        print("Skipped or failed: no valid sequences.")
        continue

    summary["runtime_minutes"] = run_minutes

    all_fast_summaries.append(summary)

    if fold_df is not None and len(fold_df) > 0:
        fold_df["runtime_minutes_parent_run"] = run_minutes
        all_fast_folds.append(fold_df)

    if pred_df is not None and len(pred_df) > 0:
        all_fast_preds.append(pred_df)

    current = (
        pd.DataFrame(all_fast_summaries)
        .sort_values(["task", "sensor_combo", "macro_f1", "accuracy"],
                     ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    display(
        current
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .head(15)
        .round(4)
    )

    current.to_csv(
        os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_summary.csv"),
        index=False
    )

    if len(all_fast_folds) > 0:
        pd.concat(all_fast_folds, ignore_index=True).to_csv(
            os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_fold_metrics.csv"),
            index=False
        )

    if SAVE_FAST_DL_PREDICTIONS and len(all_fast_preds) > 0:
        pd.concat(all_fast_preds, ignore_index=True).to_csv(
            os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_predictions.csv"),
            index=False
        )

    done = len(all_fast_summaries)
    elapsed_min = (time.time() - t0) / 60
    avg_min = elapsed_min / max(done, 1)
    remaining = len(planned_df) - (i + 1)

    print(f"Run took: {run_minutes:.1f} min")
    print(f"Elapsed: {elapsed_min:.1f} min")
    print(f"Average per completed run: {avg_min:.1f} min")
    print(f"Estimated remaining: {remaining * avg_min:.1f} min")

# ================================================================
# SAVE BEST PER TASK/SENSOR
# ================================================================

if len(all_fast_summaries) > 0:
    combined_fast_dl = pd.DataFrame(all_fast_summaries)

    combined_fast_dl.to_csv(
        os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_summary.csv"),
        index=False
    )

    combined_fast_dl_best = (
        combined_fast_dl
        .sort_values(
            ["task", "sensor_combo", "macro_f1", "accuracy"],
            ascending=[True, True, False, False]
        )
        .groupby(["task", "sensor_combo"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_path = os.path.join(
        FAST_DL_OUT_DIR,
        "combined_fast_dl_no_elapsed_best_per_task_sensor.csv"
    )

    combined_fast_dl_best.to_csv(best_path, index=False)

    print("\n" + "=" * 100)
    print("FAST DL BESTS")
    print("=" * 100)
    display(combined_fast_dl_best.round(4))

    print("\nSaved summary folder:")
    print(FAST_DL_OUT_DIR)

else:
    print("No FAST DL summaries were created.")


# --- CELL 18 (code cell #13) ---
# ---- collect Part 1 into RESULTS (FIXED: in-memory, no stale globs) ----
RESULTS["part1_recognition"] = []          # drop anything stale

def _log_best(df, fam, keys):
    if df is None or len(df) == 0:
        print("  (nothing for", fam, ")"); return
    best = (df.sort_values(["macro_f1", "accuracy"], ascending=False)
              .groupby(keys, as_index=False).head(1))
    for _, r in best.iterrows():
        log_result("part1_recognition",
                   f"{r.get('task','?')} | {r.get('sensor_combo','?')} | {fam} | "
                   f"{r.get('model','?')} | k={r.get('k','?')}",
                   accuracy=round(float(r["accuracy"]), 4),
                   macro_f1=round(float(r["macro_f1"]), 4),
                   time_condition=r.get("time_condition", "no_elapsed"))

_g = globals()
if "combined_classical" in _g:
    _log_best(_g["combined_classical"], "classical", ["task", "sensor_combo", "time_condition"])
else:
    print("WARNING: combined_classical missing — re-run the classical cell")

if "combined_fast_dl" in _g:
    _log_best(_g["combined_fast_dl"], "deep", ["task", "sensor_combo"])
else:
    print("WARNING: combined_fast_dl missing — re-run the fast-DL cell")

print("\nPart 1 collected:", len(RESULTS["part1_recognition"]), "rows  (expect 7 sensors x 6 tasks x 3 regimes = 126)")
pd.DataFrame(RESULTS["part1_recognition"])


# --- CELL 19 (code cell #14) ---
# =====================================================================
# PER-GROUP VARIANCE FOR PART 1 (recognition) — new
# Reads only prediction files written by THIS run (mtime > RUN_START).
# =====================================================================
import glob

def _pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

# ---------- classical ----------
n_done = 0
for p in glob.glob(os.path.join(OUT_DIR, "**", "*_classical_predictions.csv"), recursive=True):
    if os.path.getmtime(p) < RUN_START:
        continue                                   # stale full-9 file, skip
    pr = _orig_read_csv(p)                          # results table, not raw data
    if len(pr) == 0: continue
    for (task, sens, tc), sub in pr.groupby(["task", "sensor_combo", "time_condition"]):
        row = combined_classical[(combined_classical.task == task) &
                                 (combined_classical.sensor_combo == sens) &
                                 (combined_classical.time_condition == tc)]
        if len(row) == 0: continue
        row = row.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        sel = sub[(sub.model == row["model"]) & (sub.k.astype(str) == str(row["k"]))]
        if len(sel) == 0: continue
        variance_report(sel.y_true, sel.y_pred, sel.group,
                        f"{task} | {sens} | classical {row['model']} k={row['k']} | {tc}",
                        "part1_variance")
        n_done += 1
print(f"\nclassical per-group reports: {n_done}")

# ---------- deep ----------
dlp = os.path.join(FAST_DL_OUT_DIR, "combined_fast_dl_no_elapsed_predictions.csv")
if os.path.exists(dlp) and os.path.getmtime(dlp) >= RUN_START:
    pr = _orig_read_csv(dlp)
    gc = _pick(pr, ["group", "test_group", "held_out_group", "fold_group"])
    yt = _pick(pr, ["y_true", "true", "y", "label"])
    yp = _pick(pr, ["y_pred", "pred", "prediction"])
    print("DL prediction columns ->", list(pr.columns)[:15], "| using", gc, yt, yp)
    if all([gc, yt, yp]):
        for (task, sens), sub in pr.groupby(["task", "sensor_combo"]):
            row = combined_fast_dl[(combined_fast_dl.task == task) &
                                   (combined_fast_dl.sensor_combo == sens)]
            if len(row) == 0: continue
            row = row.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
            sel = sub[sub.model_type == row["model_type"]] if "model_type" in sub.columns else sub
            if len(sel) == 0: continue
            variance_report(sel[yt], sel[yp], sel[gc],
                            f"{task} | {sens} | deep {row.get('model_type', '?')} | no_elapsed",
                            "part1_variance")
else:
    print("no fresh DL predictions — set SAVE_FAST_DL_PREDICTIONS = True and re-run the DL cell")

pd.DataFrame(RESULTS.get("part1_variance", []))


# --- CELL 21 (code cell #15) ---
# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

import os
import json
import math
import random
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

DATA_ROOT = '/content/drive/MyDrive/thesis/data'
CORE_OUT = os.path.join(DATA_ROOT, 'PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS')
os.makedirs(CORE_OUT, exist_ok=True)

RESUME_EXISTING = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

# Historical report seeds for token neural models.
TOKEN_SEEDS = [42, 1, 7]

# Historical report used population SD across these three seeds.
SEED_STD_DDOF = 0

# Group-level SD uses the conventional sample SD.
FOLD_STD_DDOF = 1

# In-fold sensor feature selection used in the old token notebook.
K_SELECT = 40

# Grammar orders for the updated Task 3 comparison.
GRAMMAR_ORDERS = [1, 2, 3, 5, 10]
HYBRID_ORDER = 3
HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)


# Version identifier included in corrected output/checkpoint names.
TASK3_PIPELINE_VERSION = 'final_v2_common_targets'
print('Task 3 pipeline version:', TASK3_PIPELINE_VERSION)
print('Output directory:', CORE_OUT)


# --- CELL 22 (code cell #16) ---
# ================================================================
# BUILD FULL-STATISTIC SIX-LABEL ACTIVITY TOKENS
# ================================================================

NORM = os.path.join(
    DATA_ROOT,
    'RQ3_LABEL_NORMALIZATION',
    'rq3_normalized_labels_full.csv',
)

FEATURE_CANDIDATES = [
    os.path.join(DATA_ROOT, 'INTERACTION_ENG3', 'interaction_eng3_features.csv'),
    os.path.join(DATA_ROOT, 'INTERACTION_OE10', 'interaction_oe10_10s.csv'),
]

MERGE6 = {
    'social_conversation': 'conversation',
    'task_conversation': 'conversation',
}


def first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_six_label_windows():
    labels = pd.read_csv(NORM)
    label_col = 'rq3_process_label'

    group_col = first_existing(labels.columns, ['group', 'group_id', 'session'])
    time_col = first_existing(
        labels.columns,
        ['window_start', 'win_start', 'start', 'time'],
    )

    if group_col is None or time_col is None or label_col not in labels.columns:
        raise ValueError('Could not identify group, time, or process-label columns.')

    labels = labels.dropna(subset=[group_col, time_col, label_col]).copy()
    labels[label_col] = labels[label_col].astype(str).replace(MERGE6)
    labels['_g'] = labels[group_col].astype(str)
    labels['_t'] = pd.to_numeric(labels[time_col], errors='coerce').round(1)

    feature_path = next(
        (path for path in FEATURE_CANDIDATES if os.path.exists(path)),
        None,
    )

    df = labels[['_g', '_t', label_col]].rename(
        columns={label_col: 'label'}
    )
    sensor_cols = []

    if feature_path is not None:
        features = pd.read_csv(feature_path)
        feature_group = first_existing(
            features.columns,
            ['group', 'group_id', 'session'],
        )
        feature_time = first_existing(
            features.columns,
            ['window_start', 'win_start', 'start', 'time'],
        )

        features['_g'] = features[feature_group].astype(str)
        features['_t'] = pd.to_numeric(
            features[feature_time], errors='coerce'
        ).round(1)

        excluded = {
            feature_group,
            feature_time,
            '_g',
            '_t',
            'group',
            'window_start',
            'window_end',
            'window_mid',
            'recognition_label',
            'label',
            'binary_label',
        }

        sensor_cols = []
        for column in features.columns:
            if column in excluded:
                continue
            numeric = pd.to_numeric(features[column], errors='coerce')
            if numeric.notna().sum() > 0:
                sensor_cols.append(column)

        feature_small = features[['_g', '_t'] + sensor_cols].copy()
        for column in sensor_cols:
            feature_small[column] = pd.to_numeric(
                feature_small[column], errors='coerce'
            )

        df = df.merge(
            feature_small.drop_duplicates(['_g', '_t']),
            on=['_g', '_t'],
            how='left',
        )

    df = (
        df.dropna(subset=['label'])
        .sort_values(['_g', '_t'])
        .reset_index(drop=True)
    )
    df['group'] = df['_g']
    return df, sensor_cols


STAT_NAMES = [
    'mean', 'std', 'min', 'max', 'range', 'median', 'iqr',
    'p10', 'p25', 'p75', 'p90', 'energy', 'rms', 'entropy',
]


def channel_statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {name: 0.0 for name in STAT_NAMES}

    q10, q25, q50, q75, q90 = np.percentile(
        values, [10, 25, 50, 75, 90]
    )

    if len(values) >= 8:
        power = np.abs(np.fft.rfft(values - values.mean())) ** 2
        power = power[1:]
        if power.sum() > 0 and len(power) > 1:
            probability = power / power.sum()
            entropy = float(
                -(probability * np.log(probability + 1e-12)).sum()
                / np.log(len(probability))
            )
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max()),
        'range': float(values.max() - values.min()),
        'median': float(q50),
        'iqr': float(q75 - q25),
        'p10': float(q10),
        'p25': float(q25),
        'p75': float(q75),
        'p90': float(q90),
        'energy': float(np.mean(values ** 2)),
        'rms': float(np.sqrt(np.mean(values ** 2))),
        'entropy': entropy,
    }


def build_fullstat_tokens(window_df, sensor_cols):
    token_rows = []

    for group, group_df in window_df.groupby('group'):
        group_df = group_df.sort_values('_t').reset_index(drop=True)
        segment_id = (group_df['label'] != group_df['label'].shift()).cumsum()

        for _, segment in group_df.groupby(segment_id):
            record = {
                'group': group,
                'label': segment['label'].iloc[0],
                'duration': float(len(segment)),
                'start_time': float(segment['_t'].iloc[0]),
            }

            for channel in sensor_cols:
                stats = channel_statistics(
                    pd.to_numeric(segment[channel], errors='coerce').values
                )
                for stat_name, value in stats.items():
                    record[f'{channel}__{stat_name}'] = value

            token_rows.append(record)

    tokens = pd.DataFrame(token_rows)
    tokens = tokens.sort_values(['group', 'start_time']).reset_index(drop=True)
    feature_cols = [
        column
        for column in tokens.columns
        if column not in {'group', 'label', 'start_time'}
    ]
    return tokens, feature_cols


TOKEN_PATH = os.path.join(CORE_OUT, 'activity_tokens_6label_fullstat.csv')

if RESUME_EXISTING and os.path.exists(TOKEN_PATH):
    print('Reloading existing activity-token table.')
    T = pd.read_csv(TOKEN_PATH)
    fcols = [
        column
        for column in T.columns
        if column not in {'group', 'label', 'start_time'}
    ]
else:
    windows_6label, raw_sensor_cols = load_six_label_windows()
    T, fcols = build_fullstat_tokens(windows_6label, raw_sensor_cols)
    T.to_csv(TOKEN_PATH, index=False)

print('tokens:', len(T))
print('groups:', T['group'].nunique())
print('classes:', sorted(T['label'].unique()))
print('token feature dimensions:', len(fcols))
print('\nToken counts by class:')
display(T['label'].value_counts().rename_axis('label').reset_index(name='tokens'))


# --- CELL 23 (code cell #17) ---
# =====================================================================
# BACK-OFF N-GRAM over activity tokens  (thesis Table 8.7 headline)
# LOGO over groups; h = 1,2,3,5
# =====================================================================
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, accuracy_score

tok = T.copy()
tok["gid"] = tok["group"].map(_gid)
print("tokens:", len(tok), "| groups:", sorted(tok["gid"].unique()))
print(tok["label"].value_counts().to_string())

ORDER_COL = "start_time" if "start_time" in tok.columns else tok.columns[0]

def seqs_by_group(df):
    out = {}
    for g, sub in df.groupby("gid"):
        s = sub.sort_values(ORDER_COL)
        out[g] = list(s["label"].astype(str).values)
    return out

SEQ = seqs_by_group(tok)

def fit_ngram(train_seqs, max_h):
    tabs = {h: defaultdict(Counter) for h in range(1, max_h + 1)}
    uni = Counter()
    for s in train_seqs:
        for i in range(len(s) - 1):
            uni[s[i + 1]] += 1
            for h in range(1, max_h + 1):
                if i - h + 1 >= 0:
                    tabs[h][tuple(s[i - h + 1:i + 1])][s[i + 1]] += 1
    return tabs, uni

def predict_ngram(hist, tabs, uni, max_h):
    for h in range(min(max_h, len(hist)), 0, -1):          # back off
        c = tabs[h].get(tuple(hist[-h:]))
        if c:
            return c.most_common(1)[0][0]
    return uni.most_common(1)[0][0] if uni else "conversation"

for H in [1, 2, 3, 5]:
    yt, yp, gg = [], [], []
    for held in sorted(SEQ):                                # LOGO
        train = [s for g, s in SEQ.items() if g != held]
        tabs, uni = fit_ngram(train, H)
        s = SEQ[held]
        for i in range(len(s) - 1):
            yt.append(s[i + 1]); gg.append(held)
            yp.append(predict_ngram(s[max(0, i - H + 1):i + 1], tabs, uni, H))
    variance_report(yt, yp, gg, f"n-gram back-off h={H}", "part2_grammar")


# --- CELL 25 (code cell #18) ---
# ================================================================
# CELL 1 - SETUP
# ================================================================

import os
import re
import glob
import json
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 250)

DATA_ROOT = "/content/drive/MyDrive/thesis/data"

NORM_DIR = os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION")
FULL_LABEL_PATH = os.path.join(NORM_DIR, "rq3_normalized_labels_full.csv")
META_PATH = os.path.join(NORM_DIR, "rq3_label_normalization_metadata.json")

OUT_DIR = os.path.join(DATA_ROOT, "RQ3_7LABEL_HISTORY_AWARE_PREDICTION")
os.makedirs(OUT_DIR, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("NORM_DIR:", NORM_DIR)
print("OUT_DIR:", OUT_DIR)


# --- CELL 26 (code cell #19) ---
# ================================================================
# CELL 2 - LOAD 7-LABEL NORMALIZED DATA
# ================================================================

if not os.path.exists(FULL_LABEL_PATH):
    raise FileNotFoundError(
        "Normalized label file not found. Run rq3_label_audit_and_normalization.ipynb first.\n"
        f"Missing: {FULL_LABEL_PATH}"
    )

labels_df = pd.read_csv(FULL_LABEL_PATH)

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
else:
    meta = {}

# Use the 7-label process vocabulary.
LABEL_COL = "rq3_process_label"

if LABEL_COL not in labels_df.columns:
    raise ValueError(f"{LABEL_COL} not found. Available columns: {labels_df.columns.tolist()}")

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

GROUP_COL_LABEL = meta.get("group_col", None)
TIME_COL_LABEL = meta.get("time_col", None)

if GROUP_COL_LABEL is None or GROUP_COL_LABEL not in labels_df.columns:
    GROUP_COL_LABEL = first_existing(labels_df.columns, ["group", "group_id", "session", "session_id"])

if TIME_COL_LABEL is None or TIME_COL_LABEL not in labels_df.columns:
    TIME_COL_LABEL = first_existing(labels_df.columns, ["window_start", "win_start", "start", "start_time", "window_mid", "time"])

if GROUP_COL_LABEL is None or TIME_COL_LABEL is None:
    raise ValueError("Could not detect group/time columns.")

labels_df[TIME_COL_LABEL] = pd.to_numeric(labels_df[TIME_COL_LABEL], errors="coerce")
labels_df = labels_df.dropna(subset=[GROUP_COL_LABEL, TIME_COL_LABEL, LABEL_COL]).copy()

labels_df[LABEL_COL] = labels_df[LABEL_COL].astype(str)

# ---- 6-LABEL MERGE (labels_df) ----
labels_df[LABEL_COL] = labels_df[LABEL_COL].replace(
    {"social_conversation": "conversation", "task_conversation": "conversation"})
labels_df["__group_key"] = labels_df[GROUP_COL_LABEL].astype(str)
labels_df["__time_key"] = labels_df[TIME_COL_LABEL].astype(float).round(3)

labels_df = labels_df.sort_values(["__group_key", "__time_key"]).reset_index(drop=True)

print("Loaded labels:", FULL_LABEL_PATH)
print("Shape:", labels_df.shape)
print("GROUP_COL_LABEL:", GROUP_COL_LABEL)
print("TIME_COL_LABEL:", TIME_COL_LABEL)
print("LABEL_COL:", LABEL_COL)

print("\n7-label distribution:")
display(labels_df[LABEL_COL].value_counts().reset_index().rename(columns={"index": LABEL_COL, LABEL_COL: "count"}))

print("\nLabels:", sorted(labels_df[LABEL_COL].unique()))


# --- CELL 27 (code cell #20) ---
# =====================================================================
# PERSISTENCE vs TRANSITION-ONLY  (thesis Table 8.2)
# =====================================================================
lab = labels_df.copy()
lab["gid"] = lab[GROUP_COL_LABEL].map(_gid)
lab = lab.dropna(subset=["gid", TIME_COL_LABEL, LABEL_COL])
lab = lab.sort_values(["gid", TIME_COL_LABEL])
print("label rows:", len(lab), "| groups:", sorted(lab["gid"].unique()))
print(lab[LABEL_COL].value_counts().to_string())

# build (current -> next) examples per group
ex = []
for g, sub in lab.groupby("gid"):
    v = sub[LABEL_COL].astype(str).values
    for i in range(len(v) - 1):
        ex.append({"gid": g, "cur": v[i], "nxt": v[i + 1], "is_trans": v[i] != v[i + 1]})
ex = pd.DataFrame(ex)
print(f"\nexamples: {len(ex)} | transitions: {int(ex.is_trans.sum())} "
      f"({100*ex.is_trans.mean():.1f}%)")

# --- 1) repeat-current, all windows
variance_report(ex.nxt, ex.cur, ex.gid, "repeat-current (all windows)", "part3_persistence")

# --- 2) no-self n-gram back-off (h=5), transition-only
from collections import defaultdict, Counter
tr = ex[ex.is_trans].copy()
yt, yp, gg = [], [], []
for held in sorted(ex.gid.unique()):
    tr_tab = defaultdict(Counter)
    for g, sub in lab[lab.gid != held].groupby("gid"):
        v = sub[LABEL_COL].astype(str).values
        for i in range(len(v) - 1):
            if v[i] != v[i + 1]:
                tr_tab[v[i]][v[i + 1]] += 1          # no-self: only real changes counted
    sub = tr[tr.gid == held]
    for _, r in sub.iterrows():
        c = tr_tab.get(r.cur)
        yt.append(r.nxt); gg.append(held)
        yp.append(c.most_common(1)[0][0] if c else r.cur)
if len(yt):
    variance_report(yt, yp, gg, "no-self n-gram (transition-only)", "part3_persistence")

# --- 3) repeat-current on transitions (always 0 by construction)
if len(tr):
    variance_report(tr.nxt, tr.cur, tr.gid, "repeat-current (transition-only)", "part3_persistence")


# --- CELL 29 (code cell #21) ---
fn = f"naive5_results_{TAG}.json"
with open(fn, "w") as fh:
    json.dump({"tag": TAG, "run_on": RUN_ON,
               "naive_groups": sorted(NAIVE_GROUPS), "results": RESULTS}, fh, indent=1)
print("saved ->", fn)
for part, rows in RESULTS.items():
    print(f"\n===== {part} ({len(rows)} rows) =====")
    display(pd.DataFrame(rows))

try:
    from google.colab import files
    files.download(fn)
except Exception as e:
    print("(download it manually from the file browser)", e)

