"""Shared modelling code for Task 1 (interaction detection) and Task 2
(activity recognition). Both ported from near-identical Colab notebooks (see
docs/table_to_source_mapping.md) — the classical LOGO pipeline, DL sequence
pipeline, feature classification, and publication-table logic are ~verbatim
between them, so they live here once instead of twice.

Task-specific code (Task 1's exact-reproduction/naive5 paths; Task 2's task
list) stays in task1.py / task2.py.
"""

from __future__ import annotations

import gc
import os
import random
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SENSOR_COMBINATIONS = {
    "OE": ["OE"],
    "OPTI": ["OPTI"],
    "XSENS": ["XSENS"],
    "OE_OPTI": ["OE", "OPTI"],
    "OE_XSENS": ["OE", "XSENS"],
    "OPTI_XSENS": ["OPTI", "XSENS"],
    "OE_OPTI_XSENS": ["OE", "OPTI", "XSENS"],
}

BAD_TOKENS = [
    "label", "target", "class", "group", "window", "time", "elapsed",
    "pred", "prediction", "correct", "fold", "split", "index",
]

NAIVE_GROUPS = {2, 3, 5, 6, 10}


# =====================================================================
# Config
# =====================================================================

@dataclass
class Config:
    data_root: str = os.path.join(REPO_ROOT, "data", "processed")
    out_dir: str = ""  # filled in by each task's own default if not given
    run_tasks: list = field(default_factory=list)

    run_classical: bool = True
    run_dl: bool = True

    # REPORT_REPRODUCTION_MODE=True matches the thesis's reported classical
    # grid exactly (logreg + linearSVC at k=80/200). Set False for the wider
    # exploratory grid (adds RBF-SVC, RandomForest, ExtraTrees, more k).
    report_reproduction_mode: bool = True
    k_classical_full: list = field(default_factory=lambda: [40, 80, 120, 200, "all"])
    time_conditions: list = field(default_factory=lambda: ["no_elapsed", "with_elapsed"])

    run_dl_model_types: list = field(default_factory=lambda: ["lstm", "bilstm", "gru", "transformer"])
    k_dl: int = 120
    seeds: list = field(default_factory=lambda: [42])

    fast_max_epochs: int = 25
    fast_patience: int = 4
    fast_batch_size: int = 256
    fast_pred_batch_size: int = 1024

    max_logo_folds: int | None = None  # set e.g. 2 for a quick smoke test
    save_classical_predictions: bool = False
    save_fast_dl_predictions: bool = False

    # Resume/checkpoint support (Task 2's notebook made this a first-class
    # feature for surviving Colab disconnects on a long job; harmless to
    # offer for Task 1 too — off by default there since its grid is smaller).
    resume_existing: bool = False

    random_state: int = 42

    def activity_data_path(self) -> str:
        return os.path.join(self.data_root, "INTERACTION_ABLATIONS", "activity3_advanced_merged_10s_features.csv")

    def interaction_data_paths(self) -> list[str]:
        return [
            os.path.join(self.data_root, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "binary_5s_specialized_oe_merged_all_features.csv"),
            os.path.join(self.data_root, "INTERACTION_BINARY_5S_ADVANCED_FEATURES", "binary_5s_all_sensor_advanced_features.csv"),
        ]


# =====================================================================
# General helpers
# =====================================================================

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
            pd.to_numeric(df[start_col], errors="coerce") + pd.to_numeric(df[end_col], errors="coerce")
        ) / 2.0
    else:
        df["window_mid"] = pd.to_numeric(df[start_col], errors="coerce")

    df["elapsed_min"] = (df["window_mid"] - df.groupby(group_col)["window_mid"].transform("min")) / 60.0
    return df


def looks_bad_feature_name(c):
    cl = str(c).lower()
    return any(tok in cl for tok in BAD_TOKENS)


def is_oe_feature(c):
    c = str(c)
    if looks_bad_feature_name(c):
        return False
    return c.startswith(("oe__", "oe_", "oebest__", "ear_", "mag__", "mag_"))


def is_opti_feature(c):
    c = str(c)
    cl = c.lower()
    if looks_bad_feature_name(c):
        return False

    # Current advanced merged datasets normally use opti2_ / opti2__ prefixes.
    if c.startswith(("opti2_", "opti2__", "opti__", "opti_")):
        return True

    # Fallback for older unprefixed OptiTrack feature names. Kept conservative
    # to avoid catching OE/XSens features.
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
    if looks_bad_feature_name(c):
        return False
    return c.startswith(("xsens2__", "xsens2_", "xsens__", "xsens_"))


def clean_feature_list(dataframe, feats, label_cols=None, include_elapsed=False):
    label_cols = set(label_cols or [])
    bad_cols = set(label_cols) | {
        "group", "window_start", "window_end", "window_mid", "video_time_s", "time_s",
        "label", "target", "class", "activity", "activity_class", "general_class",
        "binary_label", "recognition_label", "task_label", "pred", "prediction", "correct",
    }

    cleaned = []
    for f in unique_feats(feats):
        if f not in dataframe.columns or f in bad_cols:
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


def make_classical_models(n_classes, random_state):
    return {
        "logreg_C1": LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced",
            solver="liblinear" if n_classes == 2 else "lbfgs",
            multi_class="auto", random_state=random_state,
        ),
        "linearSVC_C1": LinearSVC(
            C=1.0, class_weight="balanced", random_state=random_state,
            max_iter=10000, dual=False,
        ),
        "rbfSVC_C1_gscale": SVC(
            C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=random_state,
        ),
        "rf_leaf2": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "extraTrees_leaf1": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=1, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
    }


def build_pipeline(model, k, n_features):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    if k != "all":
        actual_k = min(int(k), n_features)
        steps.append(("select", SelectKBest(f_classif, k=actual_k)))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def metric_dict(y_true, y_pred, label_order, prefix=""):
    out = {
        prefix + "accuracy": accuracy_score(y_true, y_pred),
        prefix + "macro_f1": f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0),
        prefix + "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=label_order, zero_division=0)
    for lab, p, r, f, s in zip(label_order, pr, rc, f1, sup):
        safe = safe_name(lab)
        out[prefix + f"precision_{safe}"] = p
        out[prefix + f"recall_{safe}"] = r
        out[prefix + f"f1_{safe}"] = f
        out[prefix + f"support_{safe}"] = int(s)
    return out


# =====================================================================
# Load datasets and build task specs
# =====================================================================

def load_raw_frames(cfg: Config):
    """Load the activity and interaction feature CSVs. Returns
    (activity_df_raw, activity_window_s, interaction_df_raw, interaction_window_s).
    Both are loaded regardless of which task list is requested — matches the
    source notebooks' behavior (harmless extra I/O if only one side is used)."""
    activity_path = cfg.activity_data_path()
    if not os.path.exists(activity_path):
        raise FileNotFoundError(
            f"Could not find activity dataset:\n{activity_path}\n"
            "This is a feature-engineering output, not raw data — see "
            "docs/table_to_source_mapping.md for what stage produces it."
        )
    activity_df_raw = pd.read_csv(activity_path)
    activity_df_raw["recognition_label"] = activity_df_raw["recognition_label"].astype(str).str.strip()
    activity_df_raw = add_elapsed_min(activity_df_raw, "group", "window_start", "window_end")
    activity_window_s = infer_window_seconds(activity_df_raw, "window_start", "window_end", default=10.0)
    print("Loaded activity dataset:", activity_path, "shape:", activity_df_raw.shape)

    interaction_path = find_first_existing(cfg.interaction_data_paths())
    if interaction_path is None:
        print("WARNING: interaction binary dataset not found. Interaction task will be skipped.")
        return activity_df_raw, activity_window_s, None, 5.0

    interaction_df_raw = pd.read_csv(interaction_path)
    interaction_df_raw["binary_label"] = interaction_df_raw["binary_label"].astype(str).str.strip()
    interaction_df_raw = add_elapsed_min(interaction_df_raw, "group", "window_start", "window_end")
    interaction_window_s = infer_window_seconds(interaction_df_raw, "window_start", "window_end", default=5.0)
    print("Loaded interaction dataset:", interaction_path, "shape:", interaction_df_raw.shape)

    return activity_df_raw, activity_window_s, interaction_df_raw, interaction_window_s


def get_modality_features(df, label_cols=None):
    label_cols = label_cols or []
    oe_raw = [c for c in df.columns if is_oe_feature(c)]
    opti_raw = [c for c in df.columns if is_opti_feature(c)]
    xsens_raw = [c for c in df.columns if is_xsens_feature(c)]

    oe = clean_feature_list(df, oe_raw, label_cols=label_cols, include_elapsed=False)
    opti = clean_feature_list(df, [c for c in opti_raw if c not in oe], label_cols=label_cols, include_elapsed=False)
    xsens = clean_feature_list(df, [c for c in xsens_raw if c not in oe and c not in opti], label_cols=label_cols, include_elapsed=False)
    return {"OE": oe, "OPTI": opti, "XSENS": xsens}


def make_combo_feature_sets(df, modality_features, label_cols=None):
    combo_sets, combo_sets_elapsed = {}, {}
    for combo_name, modalities in SENSOR_COMBINATIONS.items():
        feats = []
        for m in modalities:
            feats.extend(modality_features.get(m, []))
        feats = clean_feature_list(df, feats, label_cols=label_cols, include_elapsed=False)
        feats_elapsed = clean_feature_list(
            df, feats + (["elapsed_min"] if "elapsed_min" in df.columns else []),
            label_cols=label_cols, include_elapsed=True,
        )
        combo_sets[combo_name] = feats
        combo_sets_elapsed[combo_name] = feats_elapsed
    return combo_sets, combo_sets_elapsed


# All 6 task variants the source notebooks define. Task 1's notebook only
# ever runs "interaction_vs_noninteraction"; Task 2's notebook runs the other
# 5 (see docs/thesis_reproduction_targets.md §5 for the Table 7.1-7.7 mapping:
# 2a=conversation_vs_nonconversation, 2b=conversation_vs_building,
# 2c=conversation_vs_merging, 2d=merging_vs_building, 2e=three_class_activity).
def prepare_task(task_name, activity_df_raw, activity_window_s, interaction_df_raw, interaction_window_s, out_dir):
    if task_name == "interaction_vs_noninteraction":
        if interaction_df_raw is None:
            return None
        df = interaction_df_raw.copy()
        label_col, labels = "binary_label", ["interaction", "non_interaction"]
        df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
        target_col = "task_label"
        df[target_col] = df[label_col]
        default_seq_lens = [6, 12, 18]  # 5s windows -> 30/60/90s
        window_s = interaction_window_s

    elif task_name in {
        "conversation_vs_nonconversation", "conversation_vs_building",
        "conversation_vs_merging", "merging_vs_building", "three_class_activity",
    }:
        df = activity_df_raw.copy()
        label_col = "recognition_label"
        target_col = "task_label"
        default_seq_lens = [3, 6, 9]  # 10s windows -> 30/60/90s
        window_s = activity_window_s
        if task_name == "conversation_vs_nonconversation":
            core = ["co_building", "co_merging", "conversation"]
            df = df[df[label_col].isin(core)].copy().reset_index(drop=True)
            df[target_col] = np.where(df[label_col] == "conversation", "conversation", "non_conversation")
            labels = ["conversation", "non_conversation"]
        else:
            labels = {
                "conversation_vs_building": ["conversation", "co_building"],
                "conversation_vs_merging": ["conversation", "co_merging"],
                "merging_vs_building": ["co_merging", "co_building"],
                "three_class_activity": ["co_building", "co_merging", "conversation"],
            }[task_name]
            df = df[df[label_col].isin(labels)].copy().reset_index(drop=True)
            df[target_col] = df[label_col]
    else:
        raise ValueError(f"Unknown task: {task_name}")

    if len(df) == 0:
        print("Skipping empty task:", task_name)
        return None

    modality_features = get_modality_features(df, label_cols=[label_col, target_col])
    combo_sets, combo_sets_elapsed = make_combo_feature_sets(df, modality_features, label_cols=[label_col, target_col])

    spec = {
        "task_name": task_name, "df": df, "label_col": label_col, "target_col": target_col,
        "label_order": labels, "group_col": "group", "start_col": "window_start", "end_col": "window_end",
        "window_seconds": window_s, "seq_lens": default_seq_lens,
        "modality_features": modality_features, "combo_features": combo_sets, "combo_features_elapsed": combo_sets_elapsed,
        "out_dir": os.path.join(out_dir, task_name),
    }
    os.makedirs(spec["out_dir"], exist_ok=True)
    return spec


def build_task_specs(cfg: Config):
    activity_df_raw, activity_window_s, interaction_df_raw, interaction_window_s = load_raw_frames(cfg)
    task_specs = []
    for task in cfg.run_tasks:
        spec = prepare_task(task, activity_df_raw, activity_window_s, interaction_df_raw, interaction_window_s, cfg.out_dir)
        if spec is not None:
            task_specs.append(spec)

    print("\n" + "=" * 100)
    print("PREPARED TASKS AND SENSOR COMBINATIONS")
    print("=" * 100)
    for spec in task_specs:
        print(f"\nTASK: {spec['task_name']}  rows={len(spec['df'])}  classes={spec['label_order']}")
        print("Combination counts, no elapsed:", {k: len(v) for k, v in spec["combo_features"].items()})
        counts = pd.DataFrame(
            [{"task": spec["task_name"], "sensor_combo": c,
              "n_no_elapsed": len(spec["combo_features"][c]),
              "n_with_elapsed": len(spec["combo_features_elapsed"][c])}
             for c in SENSOR_COMBINATIONS]
        )
        counts.to_csv(os.path.join(spec["out_dir"], f"{spec['task_name']}_feature_counts.csv"), index=False)
    return task_specs


# =====================================================================
# Classical LOGO runner
# =====================================================================

def run_classical_for_task_and_sensor(spec, sensor_combo, cfg: Config):
    task_name, df = spec["task_name"], spec["df"]
    y = df[spec["target_col"]].astype(str).values
    groups = df[spec["group_col"]].values
    label_order = spec["label_order"]
    n_classes = len(label_order)

    if cfg.report_reproduction_mode:
        classical_model_names = ["logreg_C1", "linearSVC_C1"]
        classical_k_values = [80, 200]
    else:
        classical_model_names = ["logreg_C1", "linearSVC_C1", "rbfSVC_C1_gscale", "rf_leaf2", "extraTrees_leaf1"]
        classical_k_values = cfg.k_classical_full

    all_models = make_classical_models(n_classes, cfg.random_state)
    models = {name: all_models[name] for name in classical_model_names if name in all_models}

    summary_rows, fold_rows, pred_rows = [], [], []
    logo = LeaveOneGroupOut()

    for time_condition in cfg.time_conditions:
        feats = spec["combo_features"][sensor_combo] if time_condition == "no_elapsed" else spec["combo_features_elapsed"][sensor_combo]
        if len(feats) == 0:
            print(f"Skipping {task_name} | {sensor_combo} | {time_condition}: no usable features")
            continue

        X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).values

        for model_name, model in models.items():
            for k in classical_k_values:
                if k != "all" and int(k) > len(feats):
                    continue
                print(f"Classical | {task_name} | {sensor_combo} | {time_condition} | {model_name} | k={k}")
                pipe = build_pipeline(model, k, len(feats))

                y_true_all, y_pred_all, group_all, row_index_all = [], [], [], []
                for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups), start=1):
                    if len(np.unique(y[tr_idx])) < 2:
                        continue
                    pipe_fold = clone(pipe)
                    pipe_fold.fit(X[tr_idx], y[tr_idx])
                    pred = pipe_fold.predict(X[te_idx])

                    fold_metric = metric_dict(y[te_idx], pred, label_order)
                    fold_rows.append({
                        "task": task_name, "sensor_combo": sensor_combo, "time_condition": time_condition,
                        "model": model_name, "k": k, "n_features": len(feats), "fold": fold,
                        "test_group": groups[te_idx][0], "n_test_rows": len(te_idx),
                        "accuracy": fold_metric["accuracy"], "macro_f1": fold_metric["macro_f1"],
                        "balanced_accuracy": fold_metric["balanced_accuracy"],
                    })
                    y_true_all.extend(y[te_idx]); y_pred_all.extend(pred)
                    group_all.extend(groups[te_idx]); row_index_all.extend(te_idx)

                if not y_true_all:
                    continue

                y_true_all, y_pred_all = np.asarray(y_true_all), np.asarray(y_pred_all)
                pooled = metric_dict(y_true_all, y_pred_all, label_order)
                pooled.update({
                    "task": task_name, "sensor_combo": sensor_combo, "time_condition": time_condition,
                    "model": model_name, "k": k, "n_features": len(feats), "n_rows_evaluated": len(y_true_all),
                })
                summary_rows.append(pooled)

                if cfg.save_classical_predictions:
                    for idx, g, yt, yp in zip(row_index_all, group_all, y_true_all, y_pred_all):
                        pred_rows.append({
                            "task": task_name, "sensor_combo": sensor_combo, "time_condition": time_condition,
                            "model": model_name, "k": k, "row_index": int(idx), "group": g,
                            "y_true": yt, "y_pred": yp, "correct": bool(yt == yp),
                        })

    summary, folds, preds = pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)
    if summary.empty:
        return summary, folds, preds, pd.DataFrame()

    group_keys = ["task", "sensor_combo", "time_condition", "model", "k", "n_features"]
    fold_stats = folds.groupby(group_keys, as_index=False).agg(
        fold_accuracy_mean=("accuracy", "mean"), fold_accuracy_std=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"), fold_macro_f1_std=("macro_f1", "std"),
        fold_balanced_accuracy_mean=("balanced_accuracy", "mean"), fold_balanced_accuracy_std=("balanced_accuracy", "std"),
        n_folds=("test_group", "nunique"),
    )
    summary = summary.merge(fold_stats, on=group_keys, how="left", validate="one_to_one")
    summary = summary.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

    # Match the report: select the best configuration by pooled macro-F1.
    best_per_condition = (
        summary.sort_values(["time_condition", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("time_condition", as_index=False).head(1).reset_index(drop=True)
    )

    out_dir = os.path.join(spec["out_dir"], sensor_combo)
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_summary_with_std.csv"), index=False)
    folds.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_fold_metrics.csv"), index=False)
    best_per_condition.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_best_per_condition_with_std.csv"), index=False)
    if cfg.save_classical_predictions and not preds.empty:
        preds.to_csv(os.path.join(out_dir, f"{task_name}_{sensor_combo}_classical_predictions.csv"), index=False)

    print(best_per_condition.round(4).to_string(index=False))
    return summary, folds, preds, best_per_condition


def run_all_classical(task_specs, cfg: Config):
    summary_path = os.path.join(cfg.out_dir, "combined_classical_summary_with_std.csv")
    folds_path = os.path.join(cfg.out_dir, "combined_classical_fold_metrics.csv")
    best_path = os.path.join(cfg.out_dir, "combined_classical_best_per_condition_with_std.csv")

    if cfg.resume_existing and all(os.path.exists(p) for p in [summary_path, folds_path, best_path]):
        print("Completed classical results found. Reloading them instead of re-running.")
        result = {
            "summary": pd.read_csv(summary_path),
            "folds": pd.read_csv(folds_path),
            "best": pd.read_csv(best_path),
        }
        print("Classical summary rows:", len(result["summary"]), "| fold rows:", len(result["folds"]), "| best-condition rows:", len(result["best"]))
        return result

    all_summaries, all_folds, all_best = [], [], []
    for spec in task_specs:
        for sensor_combo in SENSOR_COMBINATIONS:
            print("\n" + "=" * 110)
            print("CLASSICAL:", spec["task_name"], "|", sensor_combo)
            summary, folds, preds, best = run_classical_for_task_and_sensor(spec, sensor_combo, cfg)
            if not summary.empty:
                all_summaries.append(summary)
            if not folds.empty:
                all_folds.append(folds)
            if not best.empty:
                all_best.append(best)

    result = {}
    if all_summaries:
        result["summary"] = pd.concat(all_summaries, ignore_index=True)
        result["summary"].to_csv(summary_path, index=False)
    if all_folds:
        result["folds"] = pd.concat(all_folds, ignore_index=True)
        result["folds"].to_csv(folds_path, index=False)
    if all_best:
        result["best"] = pd.concat(all_best, ignore_index=True)
        result["best"].to_csv(best_path, index=False)
    return result


# =====================================================================
# DL model helpers
# =====================================================================

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, rnn_type="lstm", hidden_dim=64, num_layers=1, dropout=0.25, bidirectional=False):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


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
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.pos(self.input_proj(x))
        out = self.encoder(x)
        return self.head(out[:, -1, :])


ALL_MODEL_CONFIGS = [
    {"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "bilstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4},
]


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
    y, groups, starts = np.asarray(y), np.asarray(groups), np.asarray(starts, dtype=float)

    Xs, ys, gs, sts = [], [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(starts[idx])]
        if len(idx) < seq_len:
            continue
        for end_pos in range(seq_len - 1, len(idx)):
            win_idx = idx[end_pos - seq_len + 1 : end_pos + 1]
            Xs.append(X[win_idx]); ys.append(y[idx[end_pos]]); gs.append(g); sts.append(starts[idx[end_pos]])

    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.array([]), np.array([]), np.array([])
    return np.stack(Xs).astype(np.float32), np.array(ys), np.array(gs), np.array(sts)


def choose_validation_group(train_groups, y_all, groups_all):
    candidates = []
    for g in sorted(np.unique(train_groups)):
        mask = groups_all == g
        candidates.append((len(np.unique(y_all[mask])) >= 2, int(mask.sum()), g))
    return sorted(candidates, reverse=True)[0][2]


def torch_predict_fast(model, X, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            preds.extend(model(xb).argmax(1).cpu().numpy())
    return np.asarray(preds)


def choose_fast_seq_len(spec):
    return spec["seq_lens"][-1]  # FAST_DL_SEQ_CHOICE = "last" in the source notebooks


def train_one_fast_dl_run(spec, sensor_combo, seq_len, k_features, model_cfg, seed, cfg: Config):
    set_seed(seed)
    task_name, df = spec["task_name"], spec["df"]
    feature_list = spec["combo_features"][sensor_combo].copy()
    label_order = spec["label_order"]
    n_classes = len(label_order)
    if not feature_list:
        return None, None, None

    label_to_id = {label: i for i, label in enumerate(label_order)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    y_all = np.asarray([label_to_id[v] for v in df[spec["target_col"]].astype(str).values], dtype=int)
    groups_all = df[spec["group_col"]].values
    starts_all = pd.to_numeric(df[spec["start_col"]], errors="coerce").values
    X_raw = df[feature_list].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).values

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, fold_rows, pred_rows = [], [], [], []

    for fold, (trval_idx, te_idx) in enumerate(logo.split(X_raw, y_all, groups_all), start=1):
        if cfg.max_logo_folds is not None and fold > cfg.max_logo_folds:
            break
        if len(np.unique(y_all[trval_idx])) < 2:
            continue

        test_group = groups_all[te_idx][0]
        train_groups = np.unique(groups_all[trval_idx])
        val_group = choose_validation_group(train_groups, y_all, groups_all)
        val_mask = groups_all[trval_idx] == val_group
        val_idx, tr_idx = trval_idx[val_mask], trval_idx[~val_mask]
        if len(np.unique(y_all[tr_idx])) < 2:
            continue

        imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(X_raw[tr_idx]))
        Xval = scaler.transform(imputer.transform(X_raw[val_idx]))
        Xte = scaler.transform(imputer.transform(X_raw[te_idx]))

        actual_k = min(int(k_features), Xtr.shape[1])
        selector = SelectKBest(f_classif, k=actual_k)
        Xtr = selector.fit_transform(Xtr, y_all[tr_idx])
        Xval, Xte = selector.transform(Xval), selector.transform(Xte)

        Xtr_seq, ytr_seq, _, _ = make_sequences(Xtr, y_all[tr_idx], groups_all[tr_idx], starts_all[tr_idx], seq_len)
        Xval_seq, yval_seq, _, _ = make_sequences(Xval, y_all[val_idx], groups_all[val_idx], starts_all[val_idx], seq_len)
        Xte_seq, yte_seq, gte_seq, ste_seq = make_sequences(Xte, y_all[te_idx], groups_all[te_idx], starts_all[te_idx], seq_len)
        if len(Xtr_seq) == 0 or len(Xval_seq) == 0 or len(Xte_seq) == 0:
            continue

        model = build_model(model_cfg, input_dim=actual_k, n_classes=n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        train_loader = make_loader(Xtr_seq, ytr_seq, batch_size=cfg.fast_batch_size, shuffle=True)

        best_state, best_val_macro, patience_left, best_epoch = None, -np.inf, cfg.fast_patience, 0
        for epoch in range(1, cfg.fast_max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_pred = torch_predict_fast(model, Xval_seq, cfg.fast_pred_batch_size)
            val_macro = f1_score(yval_seq, val_pred, labels=np.arange(n_classes), average="macro", zero_division=0)
            if val_macro > best_val_macro:
                best_val_macro = val_macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left, best_epoch = cfg.fast_patience, epoch
            else:
                patience_left -= 1
            if patience_left <= 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(DEVICE)

        test_pred = torch_predict_fast(model, Xte_seq, cfg.fast_pred_batch_size)
        y_true_all.extend(yte_seq.tolist()); y_pred_all.extend(test_pred.tolist())

        fold_metric = metric_dict(yte_seq, test_pred, list(range(n_classes)))
        fold_rows.append({
            "task": task_name, "sensor_combo": sensor_combo, "time_condition": "no_elapsed",
            "model_type": model_cfg["model_type"], "seed": seed, "seq_len": seq_len,
            "context_seconds": seq_len * spec["window_seconds"], "k_features": actual_k,
            "n_features_before_select": len(feature_list), "fold": fold, "test_group": test_group,
            "n_sequences": len(Xte_seq), "best_epoch": best_epoch, "best_val_macro_f1": best_val_macro,
            "accuracy": fold_metric["accuracy"], "macro_f1": fold_metric["macro_f1"],
            "balanced_accuracy": fold_metric["balanced_accuracy"],
        })

        if cfg.save_fast_dl_predictions:
            for yt, yp, group, start in zip(yte_seq, test_pred, gte_seq, ste_seq):
                pred_rows.append({
                    "task": task_name, "sensor_combo": sensor_combo, "model_type": model_cfg["model_type"],
                    "seed": seed, "seq_len": seq_len, "group": group, "window_start": start,
                    "y_true": id_to_label[int(yt)], "y_pred": id_to_label[int(yp)], "correct": int(yt) == int(yp),
                })

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not y_true_all:
        return None, None, None

    y_true_labels = np.asarray([id_to_label[int(v)] for v in y_true_all])
    y_pred_labels = np.asarray([id_to_label[int(v)] for v in y_pred_all])
    summary = metric_dict(y_true_labels, y_pred_labels, label_order)
    summary.update({
        "task": task_name, "sensor_combo": sensor_combo, "time_condition": "no_elapsed",
        "model_type": model_cfg["model_type"], "seed": seed, "seq_len": seq_len,
        "context_seconds": seq_len * spec["window_seconds"], "k_features": int(k_features),
        "n_features_before_select": len(feature_list), "n_sequences_evaluated": len(y_true_labels),
        "max_epochs": cfg.fast_max_epochs, "patience": cfg.fast_patience,
    })
    return summary, pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def run_all_dl(task_specs, cfg: Config):
    fast_model_configs = [c for c in ALL_MODEL_CONFIGS if c["model_type"] in cfg.run_dl_model_types]
    out_dir = os.path.join(cfg.out_dir, "DL_NO_ELAPSED_ALL_ARCHITECTURES")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, "PER_CONFIGURATION_CHECKPOINTS")
    if cfg.resume_existing:
        os.makedirs(checkpoint_dir, exist_ok=True)

    planned = []
    for spec in task_specs:
        seq_len = choose_fast_seq_len(spec)
        for sensor_combo in SENSOR_COMBINATIONS:
            if not spec["combo_features"][sensor_combo]:
                continue
            for seed in cfg.seeds:
                for model_cfg in fast_model_configs:
                    planned.append({
                        "task": spec["task_name"], "sensor_combo": sensor_combo, "model": model_cfg["model_type"],
                        "seed": seed, "seq_len": seq_len,
                        "k": min(cfg.k_dl, len(spec["combo_features"][sensor_combo])),
                    })
    planned_df = pd.DataFrame(planned)
    planned_df.to_csv(os.path.join(out_dir, "dl_plan.csv"), index=False)

    all_summaries, all_folds, all_preds = [], [], []
    for run_index, row in planned_df.iterrows():
        spec = next(s for s in task_specs if s["task_name"] == row["task"])
        model_cfg = next(c for c in fast_model_configs if c["model_type"] == row["model"])

        run_key = f"{row['task']}__{row['sensor_combo']}__{row['model']}__seed{int(row['seed'])}__seq{int(row['seq_len'])}__k{int(row['k'])}"
        run_summary_path = os.path.join(checkpoint_dir, f"{run_key}__summary.csv")
        run_folds_path = os.path.join(checkpoint_dir, f"{run_key}__folds.csv")
        run_predictions_path = os.path.join(checkpoint_dir, f"{run_key}__predictions.csv")

        if cfg.resume_existing and os.path.exists(run_summary_path) and os.path.exists(run_folds_path):
            print(f"DL {run_index + 1}/{len(planned_df)} already completed: {run_key}. Reloading checkpoint.")
            summary = pd.read_csv(run_summary_path).iloc[0].to_dict()
            fold_df = pd.read_csv(run_folds_path)
            pred_df = pd.read_csv(run_predictions_path) if (cfg.save_fast_dl_predictions and os.path.exists(run_predictions_path)) else pd.DataFrame()
            all_summaries.append(summary)
            if not fold_df.empty:
                all_folds.append(fold_df)
            if cfg.save_fast_dl_predictions and not pred_df.empty:
                all_preds.append(pred_df)
            continue

        print("\n" + "=" * 110)
        print(f"DL {run_index + 1}/{len(planned_df)} | {row['task']} | {row['sensor_combo']} | {row['model']} | seed={row['seed']}")

        summary, fold_df, pred_df = train_one_fast_dl_run(
            spec, row["sensor_combo"], int(row["seq_len"]), int(row["k"]), model_cfg, int(row["seed"]), cfg,
        )
        if summary is None:
            continue
        all_summaries.append(summary)
        if not fold_df.empty:
            all_folds.append(fold_df)
        if cfg.save_fast_dl_predictions and not pred_df.empty:
            all_preds.append(pred_df)

        if cfg.resume_existing:
            # Checkpoint this configuration immediately so a later interruption
            # doesn't require rerunning it.
            pd.DataFrame([summary]).to_csv(run_summary_path, index=False)
            fold_df.to_csv(run_folds_path, index=False)
            if cfg.save_fast_dl_predictions and not pred_df.empty:
                pred_df.to_csv(run_predictions_path, index=False)

        pd.DataFrame(all_summaries).to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_summary_partial.csv"), index=False)
        if all_folds:
            pd.concat(all_folds, ignore_index=True).to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_folds_partial.csv"), index=False)

    result = {}
    if not all_summaries:
        return result

    combined = pd.DataFrame(all_summaries)
    combined_folds = pd.concat(all_folds, ignore_index=True)
    dl_group_keys = ["task", "sensor_combo", "time_condition", "model_type", "seed", "seq_len", "context_seconds", "k_features", "n_features_before_select"]
    dl_fold_stats = combined_folds.groupby(dl_group_keys, as_index=False).agg(
        fold_accuracy_mean=("accuracy", "mean"), fold_accuracy_std=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"), fold_macro_f1_std=("macro_f1", "std"),
        fold_balanced_accuracy_mean=("balanced_accuracy", "mean"), fold_balanced_accuracy_std=("balanced_accuracy", "std"),
        n_folds=("test_group", "nunique"),
    )
    combined = combined.merge(dl_fold_stats, on=dl_group_keys, how="left", validate="one_to_one")
    combined.to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_summary_with_std.csv"), index=False)
    combined_folds.to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_fold_metrics.csv"), index=False)

    combined_best = (
        combined.sort_values(["task", "sensor_combo", "macro_f1", "accuracy"], ascending=[True, True, False, False])
        .groupby(["task", "sensor_combo"], as_index=False).head(1).reset_index(drop=True)
    )
    combined_best.to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_best_per_task_sensor_with_std.csv"), index=False)
    print(combined_best.round(4).to_string(index=False))

    if len(cfg.seeds) > 1:
        seed_keys = ["task", "sensor_combo", "model_type", "seq_len", "context_seconds", "k_features", "n_features_before_select"]
        dl_seed_stats = combined.groupby(seed_keys, as_index=False).agg(
            seed_accuracy_mean=("accuracy", "mean"), seed_accuracy_std=("accuracy", "std"),
            seed_macro_f1_mean=("macro_f1", "mean"), seed_macro_f1_std=("macro_f1", "std"), n_seeds=("seed", "nunique"),
        )
        dl_seed_stats.to_csv(os.path.join(out_dir, "dl_seed_variability.csv"), index=False)

    if cfg.save_fast_dl_predictions and all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(out_dir, "combined_dl_no_elapsed_predictions.csv"), index=False)

    result["summary"] = combined
    result["folds"] = combined_folds
    result["best"] = combined_best
    return result


# =====================================================================
# Final three-regime comparison tables (Table 7.1-7.7 shape)
# =====================================================================

def build_publication_tables(classical_result, dl_result, out_dir):
    if "best" not in classical_result or "best" not in dl_result:
        print("Skipping publication tables: classical and/or DL results missing.")
        return None

    long_rows = []
    for _, row in classical_result["best"].iterrows():
        regime = "classical_no_elapsed" if row["time_condition"] == "no_elapsed" else "classical_with_elapsed"
        long_rows.append({
            "task": row["task"], "sensor_combo": row["sensor_combo"], "regime": regime, "model_family": "classical",
            "best_model": row["model"], "selection": f"k={row['k']}", "n_features": row["n_features"],
            "pooled_accuracy": row["accuracy"], "pooled_macro_f1": row["macro_f1"], "pooled_balanced_accuracy": row["balanced_accuracy"],
            "fold_accuracy_mean": row["fold_accuracy_mean"], "fold_accuracy_std": row["fold_accuracy_std"],
            "fold_macro_f1_mean": row["fold_macro_f1_mean"], "fold_macro_f1_std": row["fold_macro_f1_std"],
            "fold_balanced_accuracy_mean": row["fold_balanced_accuracy_mean"], "fold_balanced_accuracy_std": row["fold_balanced_accuracy_std"],
            "n_folds": row["n_folds"],
        })
    for _, row in dl_result["best"].iterrows():
        long_rows.append({
            "task": row["task"], "sensor_combo": row["sensor_combo"], "regime": "dl_no_elapsed", "model_family": "DL",
            "best_model": row["model_type"], "selection": f"seq={int(row['seq_len'])}, k={int(row['k_features'])}",
            "n_features": row["n_features_before_select"],
            "pooled_accuracy": row["accuracy"], "pooled_macro_f1": row["macro_f1"], "pooled_balanced_accuracy": row["balanced_accuracy"],
            "fold_accuracy_mean": row["fold_accuracy_mean"], "fold_accuracy_std": row["fold_accuracy_std"],
            "fold_macro_f1_mean": row["fold_macro_f1_mean"], "fold_macro_f1_std": row["fold_macro_f1_std"],
            "fold_balanced_accuracy_mean": row["fold_balanced_accuracy_mean"], "fold_balanced_accuracy_std": row["fold_balanced_accuracy_std"],
            "n_folds": row["n_folds"],
        })

    publication_long = pd.DataFrame(long_rows)
    for col, mean_col, std_col in [
        ("accuracy_mean_pm_std", "fold_accuracy_mean", "fold_accuracy_std"),
        ("macro_f1_mean_pm_std", "fold_macro_f1_mean", "fold_macro_f1_std"),
        ("balanced_accuracy_mean_pm_std", "fold_balanced_accuracy_mean", "fold_balanced_accuracy_std"),
    ]:
        publication_long[col] = publication_long.apply(lambda r, m=mean_col, s=std_col: f"{r[m]:.3f} ± {r[s]:.3f}", axis=1)

    regime_order = ["classical_no_elapsed", "classical_with_elapsed", "dl_no_elapsed"]
    publication_long["regime"] = pd.Categorical(publication_long["regime"], categories=regime_order, ordered=True)
    publication_long = publication_long.sort_values(["task", "sensor_combo", "regime"]).reset_index(drop=True)
    publication_long.to_csv(os.path.join(out_dir, "publication_three_regime_comparison_long.csv"), index=False)

    compact = publication_long.copy()
    compact["result_text"] = compact.apply(
        lambda r: (
            f"{r['best_model']} ({r['selection']}); pooled A={r['pooled_accuracy']:.3f}, "
            f"M={r['pooled_macro_f1']:.3f}, B={r['pooled_balanced_accuracy']:.3f}; "
            f"fold A={r['accuracy_mean_pm_std']}, M={r['macro_f1_mean_pm_std']}"
        ),
        axis=1,
    )
    publication_wide = compact.pivot(index=["task", "sensor_combo"], columns="regime", values="result_text").reset_index()

    winner_rows = (
        publication_long.sort_values(["task", "sensor_combo", "pooled_macro_f1", "pooled_accuracy"], ascending=[True, True, False, False])
        .groupby(["task", "sensor_combo"], as_index=False).head(1)
        [["task", "sensor_combo", "regime", "best_model", "pooled_macro_f1"]]
        .rename(columns={"regime": "best_regime_by_pooled_macro_f1", "best_model": "best_model_overall", "pooled_macro_f1": "best_pooled_macro_f1"})
    )
    publication_wide = publication_wide.merge(winner_rows, on=["task", "sensor_combo"], how="left")
    publication_wide.to_csv(os.path.join(out_dir, "publication_three_regime_comparison_wide.csv"), index=False)
    print(publication_wide.to_string(index=False))
    print("\nSaved final tables to:", out_dir)
    return publication_wide
