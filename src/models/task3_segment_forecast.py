"""Task 3, Appendix B — segment-level sensor-feature forecasting + decode.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "CELL 1 - SETUP"
through the held-out-group fold-std cell (source notebook's own "Appendix
B", cells 29-41). See docs/thesis_reproduction_targets.md §8.5 and
docs/table_to_source_mapping.md's Task 3 breakdown.

Pipeline: collapse consecutive same-label windows into activity *segments*
(same idea as task3_tokens.py's tokens, but rebuilt independently here from
the merged window+feature table, with mean-pooled — not full-statistic —
features per segment). For each segment, forecast the **next segment's**
feature vector from `history_len` past segment feature vectors (Ridge or
RandomForest regressor; labels are never a forecasting input), then decode
that forecast to a label with a LogisticRegression trained on real
(features -> label) pairs. An oracle variant decodes the *true* next
feature vector instead of the forecast, to show the ceiling — how much of
the gap is forecasting error vs. how discriminative segment features are
at all.

DISAMBIGUATION (resolves the open question in docs/table_to_source_mapping.md
about Table 8.3's source): the thesis's §8.5 "label Markov segment baseline"
(A=0.5106, M=0.2919, B=0.3775, n=235) is **not** the same number as Part
II's `segment_markov_h1` (task3_common_targets.py; 0.481/0.285) — those are
two different computations that happen to sound alike. `segment_markov_h1`
is Part II's corrected common-target evaluation (217 restricted targets).
The Table 8.3 baseline is *this* module's own `label_markov_segment_baseline`
(cell 38 below) — a first-order Markov table over ALL segment transitions
(no common-target restriction), confirmed by exact-number match against
docs/thesis_reproduction_targets.md §8.5. Table 8.3 as a whole is entirely
this module's output, not Part II's.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    mean_squared_error,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .task3_persistence import (
    _detect_group_time_cols,
    _is_result_file,
    _numeric_feature_count_sample,
    load_normalized_labels,
)

HISTORY_LENS = (1, 2, 3)
MAX_FEATURES = 80
MIN_SEGMENT_WINDOWS = 1
FORECAST_MODELS = ("ridge", "random_forest")

KNOWN_FEATURE_FILES_SUFFIXES = [
    os.path.join("INTERACTION_ENG3", "interaction_eng3_features.csv"),
    os.path.join("INTERACTION_ABLATIONS", "activity3_advanced_merged_10s_features.csv"),
    os.path.join("INTERACTION_ENG3", "eng3_recognition_5class_interaction_only_features.csv"),
]

COLUMN_DETECT_BAD_TOKENS = [
    "label", "target", "class", "state", "activity", "group", "session",
    "time", "start", "end", "window", "index", "row", "source_path",
]

# Published values this module's output should reproduce (thesis Table
# 8.3 — docs/thesis_reproduction_targets.md §8.5). Only these 5 rows are
# tabulated in the thesis (not a full model x history_len grid). Used
# only for the optional verification check in run_all().
REPORT_REFERENCE = {
    ("label_majority_segment_baseline", 1): {"n": 235, "accuracy": 0.3702, "macro_f1": 0.0901, "balanced_accuracy": 0.1667},
    ("label_markov_segment_baseline", 1): {"n": 235, "accuracy": 0.5106, "macro_f1": 0.2919, "balanced_accuracy": 0.3775},
    ("forecast_next_features_then_decode_ridge", 1): {"n": 235, "accuracy": 0.3021, "macro_f1": 0.2742, "balanced_accuracy": 0.3074},
    ("forecast_next_features_then_decode_random_forest", 2): {"n": 226, "accuracy": 0.3230, "macro_f1": 0.1679, "balanced_accuracy": 0.2060},
    ("oracle_decode_true_next_features_ridge", 2): {"n": 226, "accuracy": 0.3186, "macro_f1": 0.2753, "balanced_accuracy": 0.3167},
}


def select_and_merge_feature_file_required(data_root: str, labels_df: pd.DataFrame, label_col: str):
    """Cell 31. Same auto-selection scoring as task3_persistence's
    select_and_merge_feature_file, but Appendix B has no label-only
    fallback: a usable feature file is mandatory here (segments are
    defined by their feature vectors), so this raises instead."""
    import glob

    known_paths = [os.path.join(data_root, suffix) for suffix in KNOWN_FEATURE_FILES_SUFFIXES]
    recursive = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))
    auto = []
    for path in recursive:
        low = path.lower()
        name = os.path.basename(low)
        if _is_result_file(path):
            continue
        if any(tok in name for tok in ("feature", "features", "ml_ready", "merged", "advanced")) or "eng3" in low:
            auto.append(path)

    candidates = []
    for path in known_paths + auto:
        if os.path.exists(path) and path not in candidates:
            candidates.append(path)

    label_keys = labels_df[["__group_key", "__time_key"]].drop_duplicates()
    scores = []
    for path in candidates:
        n_features, group_col, time_col = _numeric_feature_count_sample(path)
        if group_col is None or time_col is None or n_features < 5:
            continue
        try:
            key_df = pd.read_csv(path, usecols=[group_col, time_col])
            key_df["__group_key"] = key_df[group_col].astype(str)
            key_df["__time_key"] = pd.to_numeric(key_df[time_col], errors="coerce").astype(float).round(3)
            key_df = key_df.dropna(subset=["__time_key"])
            overlap = (
                key_df[["__group_key", "__time_key"]].drop_duplicates()
                .merge(label_keys, on=["__group_key", "__time_key"], how="inner").shape[0]
            )
        except Exception:
            overlap = 0

        preference = 0
        low = path.lower()
        if "interaction_eng3" in low:
            preference += 500
        if "activity3_advanced_merged_10s_features" in low:
            preference += 300

        scores.append({
            "file_path": path, "group_col": group_col, "time_col": time_col,
            "numeric_feature_count_sample": n_features, "overlap_with_labels": overlap,
            "score": overlap * 10 + n_features + preference,
        })

    feature_score_df = pd.DataFrame(scores).sort_values("score", ascending=False).reset_index(drop=True)
    if len(feature_score_df) == 0 or feature_score_df.iloc[0]["overlap_with_labels"] == 0:
        raise RuntimeError("No usable feature file could be merged with labels (Appendix B requires one).")

    best = feature_score_df.iloc[0]
    feature_path, group_col, time_col = best["file_path"], best["group_col"], best["time_col"]
    print("\nSelected FEATURE_FILE:", feature_path)

    raw_feat = pd.read_csv(feature_path)
    raw_feat["__group_key"] = raw_feat[group_col].astype(str)
    raw_feat["__time_key"] = pd.to_numeric(raw_feat[time_col], errors="coerce").astype(float).round(3)

    label_keep_cols = ["__group_key", "__time_key", label_col]
    for c in ["raw_label", "rq3_compact_process_label", "rq3_conversation_binary"]:
        if c in labels_df.columns:
            label_keep_cols.append(c)

    data = raw_feat.merge(labels_df[label_keep_cols], on=["__group_key", "__time_key"], how="inner")
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[group_col, time_col, label_col]).copy()
    data[label_col] = data[label_col].astype(str).replace(
        {"social_conversation": "conversation", "task_conversation": "conversation"}
    )
    data = data.sort_values([group_col, time_col]).reset_index(drop=True)
    print("Merged data shape:", data.shape)

    return data, group_col, time_col, feature_score_df


def detect_statistical_sensor_features(data: pd.DataFrame, group_col: str, time_col: str, label_col: str) -> list[str]:
    """Cell 33."""
    meta_cols = {label_col, "raw_label", "rq3_process_label", "rq3_compact_process_label",
                 "rq3_conversation_binary", "__group_key", "__time_key", group_col, time_col}

    feature_cols = []
    for column in data.columns:
        if column in meta_cols:
            continue
        if any(token in column.lower() for token in COLUMN_DETECT_BAD_TOKENS):
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().mean() > 0.80:
            data[column] = numeric
            feature_cols.append(column)

    if len(feature_cols) < 5:
        raise RuntimeError("Too few numeric features detected.")

    return feature_cols


def build_segments(data: pd.DataFrame, group_col: str, time_col: str, label_col: str, feature_cols: list[str],
                    min_segment_windows: int = MIN_SEGMENT_WINDOWS) -> pd.DataFrame:
    """Cell 34. Collapse consecutive same-label windows into one segment
    row; each segment's feature vector is the **mean** of its member
    windows' feature vectors (unlike task3_tokens.py's 14-stat-per-channel
    tokens — this is a simpler mean-pooled representation, faithful to
    this specific notebook cell)."""
    segment_rows = []

    for g, sub in data.groupby(group_col):
        sub = sub.sort_values(time_col).reset_index(drop=True)
        labels = sub[label_col].astype(str).tolist()

        seg_id = 0
        start_idx = 0
        for i in range(1, len(sub) + 1):
            is_end = (i == len(sub)) or (labels[i] != labels[start_idx])
            if is_end:
                seg = sub.iloc[start_idx:i]
                label = labels[start_idx]
                nwin = len(seg)

                if nwin >= min_segment_windows:
                    rec = {
                        "group": g, "segment_index": seg_id, "label": label, "n_windows": nwin,
                        "start_time": seg[time_col].min(), "end_time": seg[time_col].max(),
                    }
                    for c in feature_cols:
                        rec[c] = pd.to_numeric(seg[c], errors="coerce").mean()
                    segment_rows.append(rec)
                    seg_id += 1

                start_idx = i

    segments = pd.DataFrame(segment_rows).dropna(subset=["group", "segment_index", "label"]).copy()
    segments["label"] = segments["label"].astype(str)
    return segments


def make_forecast_examples(segments: pd.DataFrame, history_len: int, used_features: list[str]):
    """Cell 35. One row per (group, segment i) with history_len-1 <= i <
    len(group)-1: X = concatenation of the last history_len segments'
    feature vectors (current first, then progressively older), Y_feat =
    the NEXT segment's feature vector, Y_label = the next segment's
    label (label is a forecasting target's decode input, never a
    forecasting input itself)."""
    rows, X_list, Y_feat_list, Y_label_list = [], [], [], []

    for g, sub in segments.groupby("group"):
        sub = sub.sort_values("segment_index").reset_index(drop=True)

        for i in range(history_len - 1, len(sub) - 1):
            rec = {
                "group": g, "pos": int(sub.loc[i, "segment_index"]),
                "current_label_for_analysis_only": sub.loc[i, "label"],
                "next_label": sub.loc[i + 1, "label"],
                "next_n_windows": sub.loc[i + 1, "n_windows"],
            }

            x_parts = []
            for lag in range(history_len):
                source_idx = i - lag
                x_parts.append(sub.loc[source_idx, used_features].astype(float).values)
                rec[f"n_windows_lag{lag}"] = sub.loc[source_idx, "n_windows"]

            X_list.append(np.concatenate(x_parts))
            Y_feat_list.append(sub.loc[i + 1, used_features].astype(float).values)
            Y_label_list.append(sub.loc[i + 1, "label"])
            rows.append(rec)

    X = np.vstack(X_list)
    Y_feat = np.vstack(Y_feat_list)
    Y_label = np.array(Y_label_list).astype(str)
    return pd.DataFrame(rows), X, Y_feat, Y_label


def _safe_log_loss(y_true, proba, labels):
    eps = 1e-12
    proba = np.clip(proba, eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return log_loss(y_true, proba, labels=labels)


def score_classification(y_true, y_pred, proba, model_name: str, history_len: int, label_order: list[str]) -> dict:
    """Cell 36."""
    out = {
        "model": model_name, "history_len": history_len, "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    if proba is not None:
        out["log_loss"] = _safe_log_loss(y_true, proba, label_order)
        top2 = np.argsort(proba, axis=1)[:, -2:]
        label_to_idx = {lab: i for i, lab in enumerate(label_order)}
        y_idx = np.array([label_to_idx[y] for y in y_true])
        out["top2_accuracy"] = np.mean([y_idx[i] in top2[i] for i in range(len(y_idx))])
    else:
        out["log_loss"] = np.nan
        out["top2_accuracy"] = np.nan

    return out


def _feature_rmse(y_true_feat, y_pred_feat) -> float:
    return float(np.sqrt(mean_squared_error(y_true_feat, y_pred_feat)))


def _mean_cosine_similarity(A, B) -> float:
    eps = 1e-12
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return float(np.mean(np.sum(A_norm * B_norm, axis=1)))


def _align_proba_to_label_order(clf, proba, label_order: list[str]) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(label_order)), dtype=float)
    for j, lab in enumerate(clf.classes_):
        if lab in label_order:
            aligned[:, label_order.index(lab)] = proba[:, j]
    aligned += 1e-12
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _make_decoder_classifier(n_used_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("selector", SelectKBest(f_classif, k=min(80, n_used_features))),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def run_forecast_experiments(examples_by_h: dict, label_order: list[str], forecast_models=FORECAST_MODELS):
    """Cell 37. For each history_len x forecast_model: LOGO CV where a
    regressor forecasts the next segment's feature vector from the past
    history_len vectors, a separately-trained LogisticRegression decodes
    features -> label, and both the "forecast then decode" model and the
    "oracle: decode the TRUE next features" ceiling are scored. Returns
    (summary_df, predictions_df)."""
    logo = LeaveOneGroupOut()
    summary_rows, prediction_rows = [], []

    for h, (ex, X_raw, Y_feat_raw, Y_label) in examples_by_h.items():
        groups = ex["group"].values
        print(f"\n{'#' * 100}\nHISTORY LENGTH: {h}\n{'#' * 100}")

        for forecast_model_name in forecast_models:
            all_y_true, all_pred_predfeat, all_pred_truefeat = [], [], []
            all_proba_predfeat, all_proba_truefeat = [], []
            all_feat_true, all_feat_pred = [], []
            all_groups, all_pos = [], []

            for tr_idx, te_idx in logo.split(X_raw, Y_label, groups):
                X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]
                Y_feat_train_raw, Y_feat_test_raw = Y_feat_raw[tr_idx], Y_feat_raw[te_idx]
                y_train, y_test = Y_label[tr_idx], Y_label[te_idx]

                metrics_raw_feature_imputer = SimpleImputer(strategy="median").fit(Y_feat_train_raw)

                x_scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
                y_scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
                X_train = x_scaler.fit_transform(X_train_raw)
                X_test = x_scaler.transform(X_test_raw)
                Y_feat_train = y_scaler.fit_transform(Y_feat_train_raw)

                if forecast_model_name == "ridge":
                    forecaster = MultiOutputRegressor(Ridge(alpha=10.0))
                elif forecast_model_name == "random_forest":
                    forecaster = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=42, n_jobs=-1)
                else:
                    raise ValueError(f"Unknown forecast model: {forecast_model_name}")

                forecaster.fit(X_train, Y_feat_train)
                Y_feat_pred_scaled = forecaster.predict(X_test)
                Y_feat_pred_raw = y_scaler.named_steps["scaler"].inverse_transform(Y_feat_pred_scaled)

                decoder = _make_decoder_classifier(Y_feat_raw.shape[1])
                decoder.fit(Y_feat_train_raw, y_train)
                clf = decoder.named_steps["clf"]

                proba_predfeat = _align_proba_to_label_order(clf, decoder.predict_proba(Y_feat_pred_raw), label_order)
                proba_truefeat = _align_proba_to_label_order(clf, decoder.predict_proba(Y_feat_test_raw), label_order)
                pred_predfeat = np.array(label_order)[np.argmax(proba_predfeat, axis=1)]
                pred_truefeat = np.array(label_order)[np.argmax(proba_truefeat, axis=1)]

                all_y_true.extend(y_test)
                all_pred_predfeat.extend(pred_predfeat)
                all_pred_truefeat.extend(pred_truefeat)
                all_proba_predfeat.append(proba_predfeat)
                all_proba_truefeat.append(proba_truefeat)
                all_feat_true.append(metrics_raw_feature_imputer.transform(Y_feat_test_raw))
                all_feat_pred.append(metrics_raw_feature_imputer.transform(Y_feat_pred_raw))
                all_groups.extend(groups[te_idx])
                all_pos.extend(ex.iloc[te_idx]["pos"].values)

            all_y_true = np.array(all_y_true)
            all_pred_predfeat, all_pred_truefeat = np.array(all_pred_predfeat), np.array(all_pred_truefeat)
            all_proba_predfeat, all_proba_truefeat = np.vstack(all_proba_predfeat), np.vstack(all_proba_truefeat)
            all_feat_true, all_feat_pred = np.vstack(all_feat_true), np.vstack(all_feat_pred)

            row_main = score_classification(all_y_true, all_pred_predfeat, all_proba_predfeat,
                                              f"forecast_next_features_then_decode_{forecast_model_name}", h, label_order)
            row_main["feature_rmse"] = _feature_rmse(all_feat_true, all_feat_pred)
            row_main["feature_cosine_similarity"] = _mean_cosine_similarity(all_feat_true, all_feat_pred)
            summary_rows.append(row_main)

            row_oracle = score_classification(all_y_true, all_pred_truefeat, all_proba_truefeat,
                                                f"oracle_decode_true_next_features_{forecast_model_name}", h, label_order)
            row_oracle["feature_rmse"] = 0.0
            row_oracle["feature_cosine_similarity"] = 1.0
            summary_rows.append(row_oracle)

            pred_df = pd.DataFrame({
                "history_len": h, "forecast_model": forecast_model_name, "group": all_groups, "pos": all_pos,
                "y_true": all_y_true, "pred_from_predicted_features": all_pred_predfeat,
                "pred_from_true_features_oracle": all_pred_truefeat,
            })
            for j, lab in enumerate(label_order):
                pred_df[f"proba_predfeat__{lab}"] = all_proba_predfeat[:, j]
                pred_df[f"proba_truefeat_oracle__{lab}"] = all_proba_truefeat[:, j]
            prediction_rows.append(pred_df)

    summary = pd.DataFrame(summary_rows).sort_values(["model", "history_len"]).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return summary, predictions


def run_label_segment_baselines(segments: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    """Cell 38. Label-only (no sensor features) first-order Markov and
    majority baselines over segment-to-segment transitions — this is the
    "label Markov segment baseline" that Table 8.3 reports (see module
    docstring's disambiguation note)."""
    def train_first_order_markov(train_df):
        table = {cur: sub["next_label"].value_counts().idxmax() for cur, sub in train_df.groupby("current_label")}
        return table, train_df["next_label"].value_counts().idxmax()

    def predict_first_order_markov(test_df, table, majority):
        return np.array([table.get(x, majority) for x in test_df["current_label"]])

    def predict_first_order_markov_no_self(test_df, train_df):
        majority = train_df["next_label"].value_counts().idxmax()
        table = {}
        for cur, sub in train_df.groupby("current_label"):
            chosen = next((lab for lab in sub["next_label"].value_counts().index if lab != cur), None)
            table[cur] = chosen if chosen is not None else majority
        return np.array([table.get(cur, majority) for cur in test_df["current_label"]])

    seg_rows = []
    for g, sub in segments.groupby("group"):
        sub = sub.sort_values("segment_index").reset_index(drop=True)
        for i in range(len(sub) - 1):
            seg_rows.append({"group": g, "pos": i, "current_label": sub.loc[i, "label"], "next_label": sub.loc[i + 1, "label"]})
    seg_ex = pd.DataFrame(seg_rows)

    logo = LeaveOneGroupOut()
    base_y, base_markov, base_markov_no_self, base_majority = [], [], [], []
    for tr_idx, te_idx in logo.split(seg_ex, seg_ex["next_label"], seg_ex["group"]):
        train, test = seg_ex.iloc[tr_idx], seg_ex.iloc[te_idx]
        table, majority = train_first_order_markov(train)
        base_y.extend(test["next_label"].astype(str).values)
        base_markov.extend(predict_first_order_markov(test, table, majority))
        base_markov_no_self.extend(predict_first_order_markov_no_self(test, train))
        base_majority.extend([majority] * len(test))

    base_y = np.array(base_y)
    return pd.DataFrame([
        score_classification(base_y, np.array(base_majority), None, "label_majority_segment_baseline", 1, label_order),
        score_classification(base_y, np.array(base_markov), None, "label_markov_segment_baseline", 1, label_order),
        score_classification(base_y, np.array(base_markov_no_self), None, "label_markov_no_self_segment_baseline", 1, label_order),
    ])


def compute_forecast_vs_oracle_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cell 39 (diagnostic, not part of Table 8.3 itself): for each
    (history_len, forecast_model), how much of the forecast's accuracy
    gap vs. the oracle ceiling is forecasting error vs. features being
    fundamentally hard to decode."""
    rows = []
    for (h, fm), g in predictions.groupby(["history_len", "forecast_model"]):
        yt = g["y_true"].values
        ypf = g["pred_from_predicted_features"].values
        yot = g["pred_from_true_features_oracle"].values

        mask_oracle_right = yot == yt
        rows.append({
            "history_len": h, "forecast_model": fm, "n": len(g),
            "forecast_acc": round(accuracy_score(yt, ypf), 3),
            "forecast_macroF1": round(f1_score(yt, ypf, average="macro", zero_division=0), 3),
            "oracle_acc": round(accuracy_score(yt, yot), 3),
            "forecast_vs_oracle_agreement": round(np.mean(ypf == yot), 3),
            "forecast_recovers_oracle_correct": round(np.mean(ypf[mask_oracle_right] == yt[mask_oracle_right]), 3) if mask_oracle_right.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["forecast_model", "history_len"])


def compute_segment_fold_std(predictions: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    """Cell 41. Held-out-group mean +/- SD for both the real forecast
    and the oracle-decode results."""
    rows = []
    for (history_len, forecast_model), subset in predictions.groupby(["history_len", "forecast_model"]):
        for prediction_column, result_name in [
            ("pred_from_predicted_features", "forecast_then_decode"),
            ("pred_from_true_features_oracle", "oracle_decode"),
        ]:
            for held_group, group_df in subset.groupby("group"):
                rows.append({
                    "history_len": history_len, "forecast_model": forecast_model, "result_type": result_name,
                    "held_group": held_group, "n": len(group_df),
                    "accuracy": accuracy_score(group_df["y_true"], group_df[prediction_column]),
                    "macro_f1": f1_score(group_df["y_true"], group_df[prediction_column], labels=label_order, average="macro", zero_division=0),
                })

    fold_metrics = pd.DataFrame(rows)
    return fold_metrics.groupby(["history_len", "forecast_model", "result_type"], as_index=False).agg(
        fold_accuracy_mean=("accuracy", "mean"), fold_accuracy_std=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"), fold_macro_f1_std=("macro_f1", "std"),
        n_folds=("held_group", "nunique"),
    )


def run_all(data_root: str, out_dir: str, history_lens=HISTORY_LENS, max_features: int = MAX_FEATURES,
            min_segment_windows: int = MIN_SEGMENT_WINDOWS, forecast_models=FORECAST_MODELS):
    """Orchestrates cells 30-41. Returns (summary, predictions,
    baseline_summary, agreement, fold_std, check)."""
    os.makedirs(out_dir, exist_ok=True)

    labels_df, _, _, label_col = load_normalized_labels(data_root)
    data, group_col, time_col, feature_score_df = select_and_merge_feature_file_required(data_root, labels_df, label_col)
    feature_score_df.to_csv(os.path.join(out_dir, "feature_file_selection_scores.csv"), index=False)

    feature_cols = detect_statistical_sensor_features(data, group_col, time_col, label_col)
    print("Detected numeric sensor/statistical features:", len(feature_cols))

    segments = build_segments(data, group_col, time_col, label_col, feature_cols, min_segment_windows=min_segment_windows)
    print("Number of segments:", len(segments))
    segments.to_csv(os.path.join(out_dir, "segment_level_feature_vectors.csv"), index=False)

    label_order = sorted(segments["label"].unique())
    used_features = feature_cols[:max_features]

    examples_by_h = {}
    for h in history_lens:
        ex, X, Y_feat, Y_label = make_forecast_examples(segments, h, used_features)
        examples_by_h[h] = (ex, X, Y_feat, Y_label)
        print(f"history_len={h}: {len(ex)} examples")
        ex.to_csv(os.path.join(out_dir, f"forecast_examples_h{h}.csv"), index=False)

    summary, predictions = run_forecast_experiments(examples_by_h, label_order, forecast_models=forecast_models)
    summary.to_csv(os.path.join(out_dir, "segment_sensor_forecast_summary.csv"), index=False)
    predictions.to_csv(os.path.join(out_dir, "segment_sensor_forecast_predictions.csv"), index=False)

    baseline_summary = run_label_segment_baselines(segments, label_order)
    baseline_summary.to_csv(os.path.join(out_dir, "label_segment_transition_baselines.csv"), index=False)

    agreement = compute_forecast_vs_oracle_agreement(predictions)

    fold_std = compute_segment_fold_std(predictions, label_order)
    fold_std.to_csv(os.path.join(out_dir, "segment_sensor_forecast_fold_std.csv"), index=False)

    real_summary = summary[~summary["model"].str.startswith("oracle_")].copy()
    if len(real_summary) > 0:
        best = real_summary.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
        print("\nBest real sensor-forecast model:\n", best)
        best_rows = predictions[(predictions["history_len"] == int(best["history_len"])) & (predictions["forecast_model"].apply(lambda x: x in best["model"]))]
        if len(best_rows) > 0:
            report_df = pd.DataFrame(classification_report(
                best_rows["y_true"].astype(str), best_rows["pred_from_predicted_features"].astype(str),
                labels=label_order, zero_division=0, output_dict=True,
            )).T
            report_df.to_csv(os.path.join(out_dir, "best_sensor_forecast_classification_report.csv"))

    # Table 8.3 rows combine both `summary` (forecast/oracle models) and
    # `baseline_summary` (label-only baselines) into one comparison.
    combined = pd.concat([summary, baseline_summary], ignore_index=True, sort=False)
    check_rows = []
    for (model, history_len), ref in REPORT_REFERENCE.items():
        row = combined[(combined["model"] == model) & (combined["history_len"] == history_len)]
        if row.empty:
            check_rows.append({"model": model, "history_len": history_len, "match": "MISSING"})
            continue
        row = row.iloc[0]
        match = (
            "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0006 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0006
                        and abs(row["balanced_accuracy"] - ref["balanced_accuracy"]) < 0.0006 and int(row["n"]) == ref["n"])
            else "DIFFERS"
        )
        check_rows.append({
            "model": model, "history_len": history_len, "n": int(row["n"]), "report_n": ref["n"],
            "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"],
            "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"],
            "match": match,
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.3")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "task3_segment_forecast_report_reproduction_check.csv"), index=False)

    n_exact = int((check["match"] == "EXACT").sum())
    print(f"\n{n_exact} of {len(check)} referenced Table 8.3 rows reproduce exactly.")

    return summary, predictions, baseline_summary, agreement, fold_std, check
