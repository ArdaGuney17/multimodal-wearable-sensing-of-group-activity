# --- CELL 1 (code cell #1) ---
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


# --- CELL 2 (code cell #2) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 3 (code cell #3) ---
import os, pandas as pd
pd.set_option("display.width", 220, "display.max_columns", 40)
R = "/content/drive/MyDrive/thesis/data"

FILES = {
 "8.7 token neural":      f"{R}/PUBLICATION_TASK3_FULL_COMPARISON/task3_token_neural_fold_metrics.csv",
 "8.7 long history":      f"{R}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/task3_grammar_long_history_common_targets_fold_metrics.csv",
 "8.7 legacy appendix":   f"{R}/PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/appendix_r_legacy_fold_metrics.csv",
 "8.2 history-aware":     f"{R}/RQ3_7LABEL_HISTORY_AWARE_PREDICTION/rq3_7label_history_aware_fold_std.csv",
 "8.3 segment forecast":  f"{R}/RQ3_7LABEL_SEGMENT_SENSOR_FORECAST/segment_sensor_forecast_fold_std.csv",
 "8.5 expanding prefix":  f"{R}/INTERACTION_ENG3/task3_expanding_prefix_segment_prediction_fold_std.csv",
}

for name, path in FILES.items():
    print("\n" + "=" * 100); print(name); print(path); print("=" * 100)
    if not os.path.exists(path):
        print("  YOK"); continue
    d = pd.read_csv(path)
    print("shape:", d.shape, "\ncolumns:", list(d.columns))
    print(d.to_string() if len(d) <= 60 else d.head(60).to_string())


# --- CELL 4 (code cell #4) ---
import pandas as pd
R = "/content/drive/MyDrive/thesis/data"
pd.set_option("display.width", 200)

# --- Tablo 8.2 icin eksik h=5 satirlari ---
d = pd.read_csv(f"{R}/RQ3_7LABEL_HISTORY_AWARE_PREDICTION/rq3_7label_history_aware_fold_std.csv")
want = [("logreg_sensor_history_only", 5, "all_windows"),
        ("ngram_markov_no_self_backoff_h5", 5, "transition_only"),
        ("logreg_label_plus_sensor_history", 5, "transition_only")]
print("=== 8.2 eksikler ===")
for m, h, sc in want:
    r = d[(d.model == m) & (d.history_len == h) & (d.eval_scope == sc)]
    print(r.to_string(index=False) if len(r) else f"YOK: {m} h={h} {sc}")

# --- Tablo 8.3 icin label-only baseline'lar nerede? ---
import glob, os
print("\n=== 8.3 baseline arama ===")
for p in glob.glob(f"{R}/RQ3_7LABEL_SEGMENT_SENSOR_FORECAST/*.csv"):
    print("\n", os.path.basename(p))
    dd = pd.read_csv(p)
    print("  columns:", list(dd.columns))
    for c in ["model", "forecast_model", "predictor", "baseline"]:
        if c in dd.columns:
            print(f"  {c}:", sorted(dd[c].dropna().unique().tolist()))


# --- CELL 5 (code cell #5) ---
# =====================================================================
# OPTI2_RELATIVE_ONLY nerede? Drive'da notebook + sonuç dosyası araması
# =====================================================================
import os, json, glob, re

ROOT = "/content/drive/MyDrive"          # gerekirse daralt: .../thesis
PATTERNS = ["OPTI2", "RELATIVE_ONLY", "relative_only", "0.8062", "0.8064"]

def peek(path, pats):
    try:
        with open(path, "r", errors="ignore") as fh:
            txt = fh.read()
    except Exception:
        return None
    hits = [p for p in pats if p in txt]
    return hits or None

print("=== NOTEBOOKS (.ipynb) ===")
nb_hits = []
for f in glob.glob(os.path.join(ROOT, "**", "*.ipynb"), recursive=True):
    h = peek(f, PATTERNS)
    if h:
        nb_hits.append((f, h))
        print(f"  {os.path.relpath(f, ROOT)}")
        print(f"      -> {h}")

print("\n=== PYTHON (.py) ===")
for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
    h = peek(f, PATTERNS)
    if h:
        print(f"  {os.path.relpath(f, ROOT)}  -> {h}")

print("\n=== RESULT FILES (.csv / .json) ===")
csv_hits = []
for ext in ("*.csv", "*.json"):
    for f in glob.glob(os.path.join(ROOT, "**", ext), recursive=True):
        if os.path.getsize(f) > 60_000_000:
            continue
        h = peek(f, PATTERNS)
        if h:
            csv_hits.append(f)
            print(f"  {os.path.relpath(f, ROOT)}  -> {h}")

print("\n=== FOLD-LEVEL DOSYALAR (per-fold metrik icerenler) ===")
for f in glob.glob(os.path.join(ROOT, "**", "*fold*"), recursive=True):
    if f.endswith((".csv", ".json")):
        print("  ", os.path.relpath(f, ROOT))

print(f"\nozet: {len(nb_hits)} notebook, {len(csv_hits)} sonuc dosyasi")


# --- CELL 6 (code cell #6) ---
# =====================================================================
# Optimized Task 1: per-fold skorlar + seq/seed sweep gercegi
# =====================================================================
import os, pandas as pd, numpy as np
from scipy import stats
pd.set_option("display.width", 200, "display.max_columns", 50)

BASE = "/content/drive/MyDrive/thesis/data/INTERACTION_BINARY_5S_SPECIALIZED_OE"

def show(path, title, n=40):
    print("\n" + "=" * 100); print(title); print("=" * 100)
    if not os.path.exists(path):
        print("  YOK:", path); return None
    d = pd.read_csv(path)
    print("shape:", d.shape, "\ncolumns:", list(d.columns))
    print(d.head(n).to_string())
    return d

# ---- 1) optimized Task 1: dokuz fold ----
fold = show(f"{BASE}/PUBLICATION_TASK1_OPTIMIZED_0806/task1_optimized_0806_fold_metrics.csv",
            "OPTIMIZED TASK 1 - FOLD METRICS")
summ = show(f"{BASE}/PUBLICATION_TASK1_OPTIMIZED_0806/task1_optimized_0806_summary_with_std.csv",
            "OPTIMIZED TASK 1 - SUMMARY")

if fold is not None:
    mcol = next((c for c in ["macro_f1","test_macro_f1","f1_macro"] if c in fold.columns), None)
    gcol = next((c for c in ["group","test_group","fold_group","fold"] if c in fold.columns), None)
    print("\n--- per-fold macro-F1 ---")
    if mcol and gcol:
        sub = fold[[gcol, mcol]].dropna().sort_values(gcol)
        print(sub.to_string(index=False))
        v = sub[mcol].values.astype(float)
        m, sd = v.mean(), v.std(ddof=1)
        lo, hi = stats.t.interval(0.95, len(v)-1, loc=m, scale=sd/np.sqrt(len(v)))
        print(f"\nn={len(v)}  mean={m:.4f}  SD={sd:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
        print("tezdeki deger: 0.781 +/- 0.086  [0.715, 0.847]")
    else:
        print("kolon tespiti basarisiz -> yukaridaki columns listesini bana gonder")

# ---- 2) seq / seed sweep: ne gercekten denendi? ----
ft = show(f"{BASE}/DEEP_SEQUENCE_FINE_TUNING_NO_ELAPSED/fine_tuned_deep_sequence_summary.csv",
          "FINE-TUNING SWEEP (seq / seed / k burada gorunur)")
ds = show(f"{BASE}/DEEP_SEQUENCE_MODELS_NO_ELAPSED/deep_sequence_no_elapsed_summary.csv",
          "BASELINE DEEP SEQUENCE SWEEP", n=25)

for name, d in [("fine_tuned", ft), ("baseline", ds)]:
    if d is None: continue
    print(f"\n--- {name}: benzersiz degerler ---")
    for c in ["seed","seq_len","sequence_length","k","k_features","model_type","feature_set","n_seeds"]:
        if c in d.columns:
            print(f"  {c:18s} -> {sorted(d[c].dropna().unique().tolist())[:12]}")


# --- CELL 7 (code cell #7) ---
import pandas as pd
d = pd.read_csv("/content/drive/MyDrive/thesis/data/"
                "PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS/"
                "task3_grammar_primary_common_targets_fold_metrics.csv")
print(d.shape, list(d.columns))
print(d.to_string())


# --- CELL 8 (code cell #8) ---
# =====================================================================
# THE ONLY SWITCH YOU CHANGE
# =====================================================================
RUN_ON = "all"          # "all" = original 9 groups | "naive" = 5 naive groups

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


# --- CELL 9 (code cell #9) ---
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


# --- CELL 10 (code cell #10) ---
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


# --- CELL 11 (code cell #11) ---
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


# --- CELL 12 (code cell #12) ---
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


# --- CELL 13 (code cell #13) ---
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


# --- CELL 15 (code cell #14) ---
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


# --- CELL 16 (code cell #15) ---
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


# --- CELL 18 (code cell #16) ---
# =====================================================================
# RUN EXACTLY THE 7 THESIS WINNERS  (6 deep-learning + 1 grammar)
# Calls train_one_fast_dl_run() directly -> 6 runs, no sweep.
# =====================================================================
SAVE_FAST_DL_PREDICTIONS = True
MAX_LOGO_FOLDS = None

# task -> (sensor_combo, model_type, seq_len, k, full9_A, full9_M)
WINNERS = {
 "interaction_vs_noninteraction":   ("OPTI",       "transformer", 18, 120, 0.751,  0.751),
 "conversation_vs_nonconversation": ("OPTI",       "transformer",  9, 120, 0.880,  0.845),
 "conversation_vs_building":        ("OPTI_XSENS", "bilstm",       9, 120, 0.881,  0.860),
 "conversation_vs_merging":         ("OE_OPTI",    "transformer",  9, 120, 0.879,  0.866),
 "merging_vs_building":             ("OPTI",       "transformer",  9, 120, 0.811,  0.738),
 "three_class_activity":            ("OE_OPTI",    "lstm",         9, 120, 0.7565, 0.6994),
}
SEED = 42

CFG_BY_TYPE = {c["model_type"]: c for c in ALL_MODEL_CONFIGS}
spec_by_name = {s["task_name"]: s for s in task_specs}

rows, all_preds = [], []
for task, (sens, mtype, seq, kf, f9a, f9m) in WINNERS.items():
    spec = spec_by_name.get(task)
    if spec is None:
        print(f"!! no spec for {task}"); continue
    if sens not in spec["combo_features"]:
        print(f"!! sensor {sens} not available for {task}"); continue

    print("\n" + "=" * 90)
    print(f"RUN  {task} | {sens} | {mtype} | seq={seq} | k={kf}")
    print("=" * 90)

    t0 = time.time()
    summary, fold_df, pred_df = train_one_fast_dl_run(
        spec, sens, seq, kf, CFG_BY_TYPE[mtype], SEED
    )
    if summary is None:
        print("   -> no result"); continue
    print(f"   done in {(time.time()-t0)/60:.1f} min")

    per = variance_report(pred_df["y_true"], pred_df["y_pred"], pred_df["group"],
                          f"{task} | {sens} | {mtype}", "winners")

    rows.append({
        "experiment": task, "sensors": sens, "model": mtype,
        "full9_A": f9a,  "naive5_A": round(summary["accuracy"], 4),
        "full9_M": f9m,  "naive5_M": round(summary["macro_f1"], 4),
        "dM": round(summary["macro_f1"] - f9m, 4),
        "fold_mean_M": round(float(per["macro_f1"].mean()), 4),
        "fold_sd_M":   round(float(per["macro_f1"].std(ddof=1)), 4),
        "n_folds": len(per),
    })
    all_preds.append(pred_df)

winners_df = pd.DataFrame(rows)
if len(all_preds):
    pd.concat(all_preds, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, f"winners_predictions_{TAG}.csv"), index=False)
winners_df.to_csv(os.path.join(OUT_DIR, f"winners_{TAG}.csv"), index=False)

print("\n===== 6 RECOGNITION WINNERS: full-9 vs naive-5 =====")
print(winners_df.to_string(index=False))


# --- CELL 20 (code cell #17) ---
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


# --- CELL 21 (code cell #18) ---
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


# --- CELL 22 (code cell #19) ---
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


# --- CELL 24 (code cell #20) ---
# =====================================================================
# FINAL COMPARISON TABLE  (adds experiment 7, the grammar winner)
# =====================================================================
g2 = [r for r in RESULTS.get("part2_grammar", [])
      if "h=2" in str(r.get("config", r.get("name", "")))]

final = winners_df.copy()
if g2:
    r = g2[0]
    final = pd.concat([final, pd.DataFrame([{
        "experiment": "next_activity (n-gram h=2)", "sensors": "labels only",
        "model": "backoff", "full9_A": 0.604, "naive5_A": round(r["pooled_acc"], 4),
        "full9_M": 0.499, "naive5_M": round(r["pooled_macro_f1"], 4),
        "dM": round(r["pooled_macro_f1"] - 0.499, 4),
        "fold_mean_M": r["fold_mean"], "fold_sd_M": r["fold_sd"],
        "n_folds": r["n_folds"],
    }])], ignore_index=True)
else:
    print("(run Part 2 to add the grammar row)")

final.to_csv(os.path.join(OUT_DIR, f"seven_winners_{TAG}.csv"), index=False)
print("===== SEVEN THESIS WINNERS: full-9 vs naive-5 =====")
print(final.to_string(index=False))
print("\nsaved ->", os.path.join(OUT_DIR, f"seven_winners_{TAG}.csv"))
final


# --- CELL 25 (code cell #21) ---
# =====================================================================
# ELAN DISCOVERY + GROUP DESCRIPTIVES (real session timings)
# Reads {DATA_ROOT}/group_*/elan/*.csv
# =====================================================================
import os, re, glob, numpy as np, pandas as pd

NAIVE, RESEARCHER = {2, 3, 5, 6, 10}, {1, 7, 8, 9}

files = sorted(glob.glob(os.path.join(DATA_ROOT, "group_*", "elan", "*.csv")))
print(f"found {len(files)} elan csv files\n")

by_group = {}
for f in files:
    m = re.search(r"group_(\d+)", f)
    if m:
        by_group.setdefault(int(m.group(1)), []).append(f)

for g in sorted(by_group):
    print(f"G{g}: {len(by_group[g])} file(s)")
    for f in by_group[g]:
        print("   ", os.path.basename(f))

# ---------- inspect the first file ----------
probe = files[0]
print("\n" + "=" * 100)
print("PROBE:", probe)
print("=" * 100)
for sep in [",", "\t", ";"]:
    try:
        d = _orig_read_csv(probe, sep=sep)
        if d.shape[1] > 1:
            print(f"sep={sep!r}  shape={d.shape}")
            print("columns:", list(d.columns))
            print(d.head(5).to_string())
            break
    except Exception as e:
        print(f"sep={sep!r} failed: {e}")

# also show a raw peek in case there is no header row
print("\n--- raw first 5 lines ---")
with open(probe) as fh:
    for i, line in enumerate(fh):
        if i >= 5: break
        print(repr(line[:200]))


# --- CELL 26 (code cell #22) ---
# =====================================================================
# GROUP DESCRIPTIVES FROM ELAN EXPORTS (real session timings)
# Schema: headerless, 9 cols
#   tier, blank, beg_hms, beg_s, end_hms, end_s, dur_hms, dur_s, label
# =====================================================================
import os, re, glob, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

NAIVE, RESEARCHER = {2, 3, 5, 6, 10}, {1, 7, 8, 9}
COLS = ["tier", "blank", "beg_hms", "beg_s", "end_hms", "end_s",
        "dur_hms", "dur_s", "label"]

# ---------- one canonical file per group ----------
files = sorted(glob.glob(os.path.join(DATA_ROOT, "group_*", "elan",
                                      "*_individual_build_renamed.csv")))
files = [f for f in files if "BACKUP" not in f]
chosen = {}
for f in files:
    g = int(re.search(r"group_(\d+)", f).group(1))
    chosen[g] = f
print("using:")
for g in sorted(chosen):
    print(f"  G{g:<3d} {os.path.basename(chosen[g])}")

# ---------- load ----------
frames = []
for g, f in sorted(chosen.items()):
    d = _orig_read_csv(f, header=None, names=COLS)
    for c in ["beg_s", "end_s", "dur_s"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["beg_s", "end_s", "label"])
    d["group"] = g
    d["cohort"] = "researcher" if g in RESEARCHER else "naive"
    frames.append(d)
ann = pd.concat(frames, ignore_index=True)
ann["dur_s"] = ann["dur_s"].fillna(ann["end_s"] - ann["beg_s"])

print(f"\nannotations={len(ann)}  groups={sorted(ann.group.unique())}")
print("\ntiers found:")
print(ann.groupby(["group", "tier"]).size().unstack(fill_value=0).to_string())
print(f"\ndistinct labels ({ann.label.nunique()}):")
print(ann.label.value_counts().to_string())

# ---------- per-group descriptives ----------
rows = []
for g, sub in ann.groupby("group"):
    session_s = float(sub["end_s"].max())
    # transitions computed WITHIN each tier, then summed
    trans, seg_durs = 0, []
    for t, s2 in sub.groupby("tier"):
        s2 = s2.sort_values("beg_s")
        labs = s2["label"].astype(str).values
        trans += int((labs[1:] != labs[:-1]).sum())
        seg_durs.extend(s2["dur_s"].tolist())
    p = sub.groupby("label")["dur_s"].sum()
    p = p / p.sum()
    rows.append({
        "group":            g,
        "cohort":           "researcher" if g in RESEARCHER else "naive",
        "session_min":      round(session_s / 60, 1),
        "annotated_min":    round(float(sub["dur_s"].sum()) / 60, 1),
        "n_annotations":    len(sub),
        "n_tiers":          sub["tier"].nunique(),
        "n_labels":         sub["label"].nunique(),
        "transitions":      trans,
        "trans_per_min":    round(trans / (session_s / 60), 2),
        "mean_ann_s":       round(float(np.mean(seg_durs)), 1),
        "median_ann_s":     round(float(np.median(seg_durs)), 1),
        "max_ann_s":        round(float(np.max(seg_durs)), 1),
        "entropy":          round(float(-(p * np.log2(p)).sum()), 3),
        "dominant_share":   round(float(p.max()), 3),
    })
desc = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

print("\n" + "=" * 110)
print("PER-GROUP DESCRIPTIVES (from ELAN)")
print("=" * 110)
print(desc.to_string(index=False))

# ---------- cohort comparison ----------
METRICS = ["session_min", "annotated_min", "n_annotations", "n_labels",
           "transitions", "trans_per_min", "mean_ann_s", "median_ann_s",
           "entropy", "dominant_share"]

print("\n" + "=" * 110)
print("COHORT MEAN (SD)")
print("=" * 110)
print(desc.groupby("cohort")[METRICS].agg(["mean", "std"]).round(2).to_string())

print("\n" + "=" * 110)
print("COHORT COMPARISON — n=4 researcher vs n=5 naive (DESCRIPTIVE ONLY, min p = 0.016)")
print("=" * 110)
print(f"{'metric':18s} {'researcher':>12s} {'naive':>12s} {'diff':>9s} {'U':>6s} {'p':>7s}")
for m in METRICS:
    a = desc.loc[desc.cohort == "researcher", m].astype(float)
    b = desc.loc[desc.cohort == "naive", m].astype(float)
    u, pv = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{m:18s} {a.mean():12.2f} {b.mean():12.2f} {a.mean()-b.mean():9.2f} {u:6.1f} {pv:7.3f}"
          + ("  <--" if pv < 0.05 else ""))

# ---------- time per activity ----------
print("\n" + "=" * 110)
print("MINUTES PER LABEL, PER GROUP")
print("=" * 110)
ph = (ann.groupby(["group", "label"])["dur_s"].sum().div(60)
      .unstack(fill_value=0).round(1))
ph.index = [f"G{i}" for i in ph.index]
print(ph.to_string())

print("\n--- share of annotated time (%) ---")
share = ph.div(ph.sum(axis=1), axis=0).mul(100).round(1)
print(share.to_string())

print("\n--- cohort mean share (%) ---")
s2 = share.copy()
s2["cohort"] = ["researcher" if int(i[1:]) in RESEARCHER else "naive" for i in s2.index]
print(s2.groupby("cohort").mean(numeric_only=True).round(1).to_string())

# ---------- annotation duration by label and cohort ----------
print("\n" + "=" * 110)
print("ANNOTATION DURATION (s) BY LABEL AND COHORT")
print("=" * 110)
print(ann.groupby(["label", "cohort"])["dur_s"]
        .agg(n="size", mean="mean", median="median", max="max").round(1).to_string())

# ---------- coverage vs the modelling table ----------
print("\n" + "=" * 110)
print("COVERAGE: ELAN session length vs windows used in modelling")
print("=" * 110)
try:
    lab = _orig_read_csv(os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION",
                                      "rq3_normalized_labels_full.csv"))
    lab["group"] = lab["group"].map(_gid)
    used = lab.groupby("group").size().mul(5).div(60).round(1)
    cov = desc.set_index("group")[["session_min"]].join(used.rename("modelled_min"))
    cov["coverage_%"] = (cov.modelled_min / cov.session_min * 100).round(1)
    print(cov.to_string())
    print(f"\noverall: {cov.modelled_min.sum():.1f} of {cov.session_min.sum():.1f} min "
          f"({cov.modelled_min.sum()/cov.session_min.sum()*100:.1f}%)")
except Exception as e:
    print("coverage check skipped:", e)

desc.to_csv("elan_group_descriptives.csv", index=False)
ann.to_csv("elan_annotations_all.csv", index=False)
print("\nsaved -> elan_group_descriptives.csv | elan_annotations_all.csv")
desc


# --- CELL 27 (code cell #23) ---
# ---- map 280 raw labels -> the 6 process categories (typo-robust stems) ----
import re
def norm_label(s):
    x = re.sub(r"[^a-z_+ ]", "", str(s).lower())
    x = x.split("+")[0].strip()            # keep primary of "A + B"
    if re.search(r"m[ei]r[gj]|merg", x):                       return "merging"
    if re.search(r"b[uıi]{1,2}ld|build", x) and "co_" in x:    return "co_building"
    if re.search(r"b[uıi]{1,2}ld|build", x):                   return "individual_build"
    if re.search(r"handover|hand[iı]ver|handıver", x):         return "object_handover"
    if re.search(r"convo|talk|dialou|conversation", x):        return "conversation"
    if re.search(r"[ai]nsp|isnp|inscp|checking|matching|mathi|mathc", x): return "inspection"
    if re.search(r"trav|travel|moving|carry|cary|deliver|approach|pick|put|plac|search|lift", x): return "moving_transport"
    if re.search(r"sync|clap", x):                             return "sync_marker"
    return "other"

ann["norm"] = ann["label"].map(norm_label)
print(ann["norm"].value_counts().to_string())
print("\nunmapped ('other') examples:")
print(ann.loc[ann.norm == "other", "label"].value_counts().head(20).to_string())

# recompute the two contested metrics on normalised labels
rows2 = []
for g, sub in ann.groupby("group"):
    sess = float(sub["end_s"].max()); tr = 0
    for t, s2 in sub.groupby("tier"):
        L = s2.sort_values("beg_s")["norm"].values
        tr += int((L[1:] != L[:-1]).sum())
    p = sub.groupby("norm")["dur_s"].sum(); p = p / p.sum()
    rows2.append({"group": g,
                  "cohort": "researcher" if g in RESEARCHER else "naive",
                  "session_min": round(sess / 60, 1),
                  "n_norm_labels": sub["norm"].nunique(),
                  "transitions": tr, "trans_per_min": round(tr / (sess / 60), 2),
                  "entropy": round(float(-(p * np.log2(p)).sum()), 3),
                  "dominant_share": round(float(p.max()), 3)})
d2 = pd.DataFrame(rows2).sort_values("group")
print("\n", d2.to_string(index=False))
print("\ncohort means:\n", d2.groupby("cohort").mean(numeric_only=True).round(2).to_string())
for m in ["session_min", "n_norm_labels", "transitions", "trans_per_min", "entropy", "dominant_share"]:
    a = d2.loc[d2.cohort == "researcher", m]; b = d2.loc[d2.cohort == "naive", m]
    u, pv = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{m:16s} researcher={a.mean():7.2f} naive={b.mean():7.2f} U={u:4.1f} p={pv:.3f}")

print("\n--- minutes per normalised category ---")
ph2 = ann.groupby(["group", "norm"])["dur_s"].sum().div(60).unstack(fill_value=0).round(1)
print(ph2.to_string())
print("\n--- cohort mean share (%) ---")
sh = ph2.div(ph2.sum(axis=1), axis=0).mul(100).round(1)
sh["cohort"] = ["researcher" if g in RESEARCHER else "naive" for g in sh.index]
print(sh.groupby("cohort").mean(numeric_only=True).round(1).to_string())

