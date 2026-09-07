"""Task 3, Appendix A — window-level next-window prediction: the
"Persistence Problem" (all-window vs. transition-only evaluation).

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "CELL 1 - SETUP"
through "CELL 10 - COMPARE LABEL HISTORY VS SENSOR HISTORY" (source
notebook's own "Appendix A", cells 18-27). See
docs/thesis_reproduction_targets.md §8.4 and
docs/table_to_source_mapping.md's Task 3 breakdown.

Unlike the rest of Task 3 (which operates on *tokens* — one row per
contiguous same-label segment), this module operates directly on
*windows* (the 5s/10s classification unit, before any segment
collapsing). At each window t it predicts the label of window t+1, using
up to `history_len` past window labels (and, optionally, lagged sensor
features). This is what exposes "the persistence problem": most windows
repeat their own label, so naive baselines score very high overall but
score exactly zero once evaluation is restricted to windows where the
label actually changes (`is_transition`).

Table 8.2 (docs/thesis_reproduction_targets.md §8.4) reports a specific
handful of (model, history_len, eval_scope) combinations — not a full
grid — see REPORT_REFERENCE below, used only for the optional
verification check in run_all().

REPRODUCIBILITY NOTE (corrected 2026-09-06, real-data validation): the
source notebook's cells 17-18 ("---- 6-LABEL MERGE ----") *do* literally
collapse social/task conversation into a single "conversation" class
before evaluation, and an earlier version of this docstring took that at
face value and claimed Table 8.2's real vocabulary was 6-class despite its
"7-LABEL" name. That claim was wrong. Real-data validation against
docs/thesis_reproduction_targets.md's Table 8.2 (see
data/external/thesis_data/PUBLICATION_TASK3_CORRECTED_FINAL/
run_table_8_2_persistence_validation.py) shows the published numbers are
reproduced only when the merge is *not* applied — i.e. the true,
un-collapsed 7-class `rq3_process_label` vocabulary, matching Table 8.2's
own literal title ("Window-level 7-label next-window prediction"). With
the merge applied (this module's behavior prior to this fix): all 9
REPORT_REFERENCE rows differ from the published table (e.g.
repeat_current_label/h1/all_windows: 0.8865/0.7578 vs. published
0.8672/0.7425). With the merge removed: 6 of 9 rows match exactly, and the
remaining 3 (all sensor-feature logistic-regression models) are off by
0.0006-0.0041 — consistent with the "unpinned sklearn version" numerical
drift already documented elsewhere in this reproduction effort for other
logistic-regression-based tables, not a new discrepancy.

This means cells 17-18 of `task3_FINAL_V2_with_exact_report_reproduction.
ipynb` — despite being the notebook currently used as this port's
source-of-truth — do not reproduce the actual historical Table 8.2
computation for this specific step, the same kind of "current reference
notebook isn't quite the historical code path" situation already
documented for Table 8.7 (see task3_grammar.py's "Appendix R vs. Parts
II-IV" note and docs/table_to_source_mapping.md). Accordingly,
`load_normalized_labels()` and `select_and_merge_feature_file()` below
take an `apply_merge6` flag (default True, to preserve the already-exact
Table 8.3 behavior in task3_segment_forecast.py, which imports
`load_normalized_labels` from this module and *does* need the merge —
Table 8.3's own title says "6-label" and it reproduces exactly with the
merge applied). This module's own `run_all()` explicitly passes
`apply_merge6=False`.

MERGE6 itself is kept defined (unused by default here) since
task3_tokens.py and task3_segment_forecast.py both still need their own
equivalent merge for their own (validated-correct) published tables.

Airtight independent confirmation, not just the A/B statistical test above:
docs/thesis_reproduction_targets.md §8.2 ("Data representations — four
representations, exact example counts") lists this module's input as
representation #1, "Window-level history-aware data, **7-label**
next-window prediction" (n=2071, 275 transitions — matches this module's
own real-data numbers with apply_merge6=False exactly), as a *distinct*
representation from #2, "Segment-level **6-label** data" (244 segments —
what feeds Table 8.3/task3_segment_forecast.py and task3_tokens.py). The
thesis text itself draws this exact line between 7-class and 6-class
representations.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

HISTORY_LENS = (1, 2, 3, 5)
USE_FEATURE_LAGS = True
MAX_SENSOR_FEATURES = 80
FOLD_STD_DDOF = 1  # matches source notebook's pandas .agg('std') default

MERGE6 = {"social_conversation": "conversation", "task_conversation": "conversation"}

KNOWN_FEATURE_FILES_SUFFIXES = [
    os.path.join("INTERACTION_ENG3", "interaction_eng3_features.csv"),
    os.path.join("INTERACTION_ABLATIONS", "activity3_advanced_merged_10s_features.csv"),
    os.path.join("INTERACTION_ENG3", "eng3_recognition_5class_interaction_only_features.csv"),
]

RESULT_FILE_BAD_TOKENS = [
    "summary", "prediction", "predictions", "result", "results", "best",
    "fold_metrics", "aggregate", "importance", "confusion",
]

COLUMN_DETECT_BAD_TOKENS = [
    "label", "target", "class", "state", "activity", "group", "session",
    "time", "start", "end", "window", "index", "row", "source_path",
]

# Published values this module's output should reproduce (thesis Table
# 8.2 — docs/thesis_reproduction_targets.md §8.4). Table 8.2 is not a
# full grid: it reports one representative row per notable
# (model, history_len, eval_scope) combination, so this dict only has
# those specific keys. Used only for the optional verification check in
# run_all() — never to adjust behavior.
REPORT_REFERENCE = {
    ("repeat_current_label", 1, "all_windows"): {"n": 2071, "accuracy": 0.8672, "macro_f1": 0.7425, "balanced_accuracy": 0.7425},
    ("logreg_label_history_only", 1, "all_windows"): {"n": 2071, "accuracy": 0.8672, "macro_f1": 0.7425, "balanced_accuracy": 0.7425},
    ("ngram_markov_backoff_h1", 1, "all_windows"): {"n": 2071, "accuracy": 0.8629, "macro_f1": 0.6925, "balanced_accuracy": 0.6957},
    ("logreg_label_plus_sensor_history", 1, "all_windows"): {"n": 2071, "accuracy": 0.8160, "macro_f1": 0.6766, "balanced_accuracy": 0.7021},
    ("logreg_sensor_history_only", 5, "all_windows"): {"n": 2035, "accuracy": 0.3622, "macro_f1": 0.2592, "balanced_accuracy": 0.2887},
    ("ngram_markov_no_self_backoff_h5", 5, "transition_only"): {"n": 270, "accuracy": 0.4704, "macro_f1": 0.2402, "balanced_accuracy": 0.2801},
    ("logreg_sensor_history_only", 1, "transition_only"): {"n": 275, "accuracy": 0.2218, "macro_f1": 0.2184, "balanced_accuracy": 0.2595},
    ("logreg_label_plus_sensor_history", 5, "transition_only"): {"n": 270, "accuracy": 0.1296, "macro_f1": 0.1158, "balanced_accuracy": 0.1159},
    ("repeat_current_label", 1, "transition_only"): {"n": 275, "accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
}


def _first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _detect_group_time_cols(columns):
    group_col = _first_existing(columns, ["group", "group_id", "session", "session_id"])
    time_col = _first_existing(columns, ["window_start", "win_start", "start", "start_time", "window_mid", "time"])
    return group_col, time_col


def load_normalized_labels(data_root: str, apply_merge6: bool = True):
    """Cell 18 (label-loading half; cells 17/18 in the source notebook).
    Loads the `rq3_process_label` table and detects its group/time
    columns (metadata JSON first, falls back to name sniffing — mirrors
    the source notebook exactly).

    `apply_merge6` defaults to True to preserve the already-validated
    (real-data exact match) behavior of every OTHER caller of this
    function — task3_segment_forecast.py's Table 8.3, and the source
    notebook's own literal cells 17/29, all collapse social/task
    conversation into "conversation". This module's own run_all() is the
    one exception: it explicitly passes apply_merge6=False, because real-
    data validation shows Table 8.2 only reproduces with the true 7-class
    vocabulary — see this module's docstring "REPRODUCIBILITY NOTE"."""
    norm_dir = os.path.join(data_root, "RQ3_LABEL_NORMALIZATION")
    label_path = os.path.join(norm_dir, "rq3_normalized_labels_full.csv")
    meta_path = os.path.join(norm_dir, "rq3_label_normalization_metadata.json")

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Normalized label file not found:\n{label_path}\n"
            "This is a feature-engineering/label-cleaning stage output, not raw "
            "data — see docs/table_to_source_mapping.md for what produces it."
        )

    labels_df = pd.read_csv(label_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    label_col = "rq3_process_label"
    if label_col not in labels_df.columns:
        raise ValueError(f"{label_col} not found. Available columns: {labels_df.columns.tolist()}")

    group_col = meta.get("group_col")
    if group_col is None or group_col not in labels_df.columns:
        group_col = _first_existing(labels_df.columns, ["group", "group_id", "session", "session_id"])

    time_col = meta.get("time_col")
    if time_col is None or time_col not in labels_df.columns:
        time_col = _first_existing(labels_df.columns, ["window_start", "win_start", "start", "start_time", "window_mid", "time"])

    if group_col is None or time_col is None:
        raise ValueError("Could not detect group/time columns.")

    labels_df[time_col] = pd.to_numeric(labels_df[time_col], errors="coerce")
    labels_df = labels_df.dropna(subset=[group_col, time_col, label_col]).copy()
    labels_df[label_col] = labels_df[label_col].astype(str)
    if apply_merge6:
        labels_df[label_col] = labels_df[label_col].replace(MERGE6)
    labels_df["__group_key"] = labels_df[group_col].astype(str)
    labels_df["__time_key"] = labels_df[time_col].astype(float).round(3)
    labels_df = labels_df.sort_values(["__group_key", "__time_key"]).reset_index(drop=True)

    return labels_df, group_col, time_col, label_col


def _is_result_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(token in name for token in RESULT_FILE_BAD_TOKENS)


def _numeric_feature_count_sample(path: str):
    try:
        sample = pd.read_csv(path, nrows=200)
    except Exception:
        return 0, None, None

    group_col, time_col = _detect_group_time_cols(sample.columns)
    if group_col is None or time_col is None:
        return 0, group_col, time_col

    n_features = 0
    for column in sample.columns:
        if any(token in column.lower() for token in COLUMN_DETECT_BAD_TOKENS):
            continue
        numeric = pd.to_numeric(sample[column], errors="coerce")
        if numeric.notna().mean() > 0.80:
            n_features += 1

    return n_features, group_col, time_col


def select_and_merge_feature_file(data_root: str, labels_df: pd.DataFrame, label_col: str, apply_merge6: bool = True):
    """Cell 19. Auto-selects the best-overlapping sensor feature CSV
    (scored by row overlap with the label table's (group, time) keys,
    with a couple of hardcoded preference boosts — same scoring the
    source notebook uses), merges it against labels_df, and (by default)
    re-applies the 6-label merge to the merged frame's label column. Falls
    back to label-history-only (no sensor columns) if nothing usable is
    found. `apply_merge6` should match whatever was passed to
    load_normalized_labels() for the same labels_df — see this module's
    docstring "REPRODUCIBILITY NOTE" for why Table 8.2's own run_all()
    passes False here while Table 8.3's caller (task3_segment_forecast.py)
    needs the default True. Returns (data, group_col, time_col,
    feature_score_df)."""
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
        print("No usable feature file found. Running label-history models only.")
        data = labels_df.copy()
        group_col, time_col = "__group_key", "__time_key"
        # use the original (non-key) group/time columns for downstream sorting
        group_col = [c for c in labels_df.columns if c not in ("__group_key", "__time_key")][0]
    else:
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
        print("Merged data shape:", data.shape)

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[group_col, time_col, label_col]).copy()
    data[label_col] = data[label_col].astype(str)
    if apply_merge6:
        data[label_col] = data[label_col].replace(MERGE6)
    data = data.sort_values([group_col, time_col]).reset_index(drop=True)

    return data, group_col, time_col, feature_score_df


def detect_numeric_sensor_features(data: pd.DataFrame, group_col: str, time_col: str, label_col: str) -> list[str]:
    """Cell 20."""
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

    return feature_cols


def make_history_examples(data: pd.DataFrame, group_col: str, time_col: str, label_col: str,
                           history_len: int, sensor_cols: list[str], include_sensor_features: bool = True) -> pd.DataFrame:
    """Cell 21. One row per (group, window t) with history_len-1 <= t <
    len(group)-1: label_lag0..label_lag{history_len-1} (lag0 = current
    label y_t), next_label = y_{t+1}, plus optional lagged sensor
    features for the same window range."""
    rows = []
    for g, sub in data.groupby(group_col):
        sub = sub.sort_values(time_col).reset_index(drop=True)

        for i in range(history_len - 1, len(sub) - 1):
            rec = {
                "group": g, "pos": i, "time_t": sub.loc[i, time_col],
                "next_label": sub.loc[i + 1, label_col],
                "current_label": sub.loc[i, label_col],
                "is_transition": sub.loc[i, label_col] != sub.loc[i + 1, label_col],
            }

            history_labels = []
            for lag in range(history_len):
                lab = sub.loc[i - lag, label_col]
                rec[f"label_lag{lag}"] = lab
                history_labels.append(lab)

            rec["history_has_transition"] = int(len(set(history_labels)) > 1)
            rec["history_unique_labels"] = len(set(history_labels))
            rec["history_most_recent_run_length"] = 1
            for lag in range(1, history_len):
                if history_labels[lag] == history_labels[0]:
                    rec["history_most_recent_run_length"] += 1
                else:
                    break

            if include_sensor_features and len(sensor_cols) > 0:
                for lag in range(history_len):
                    for c in sensor_cols:
                        rec[f"{c}__lag{lag}"] = sub.loc[i - lag, c]

            rows.append(rec)

    return pd.DataFrame(rows)


def train_ngram_transition_model(train_df: pd.DataFrame, history_len: int):
    """Cell 22. For k=1..history_len, maps the last-k-labels context to the
    single most frequent next_label seen for that exact context in
    training. Back-off across k happens at prediction time."""
    models = {}
    majority = train_df["next_label"].value_counts().idxmax()

    for k in range(1, history_len + 1):
        table = defaultdict(Counter)
        for _, r in train_df.iterrows():
            key = tuple(r[f"label_lag{lag}"] for lag in range(k))
            table[key][r["next_label"]] += 1

        pred_table = {key: ctr.most_common(1)[0][0] for key, ctr in table.items()}
        models[k] = pred_table

    return models, majority


def predict_ngram_backoff(test_df: pd.DataFrame, models: dict, majority, history_len: int, force_no_self: bool = False) -> np.ndarray:
    preds = []
    for _, r in test_df.iterrows():
        pred = None
        for k in range(history_len, 0, -1):
            key = tuple(r[f"label_lag{lag}"] for lag in range(k))
            if key in models[k]:
                cand = models[k][key]
                if force_no_self and cand == r["current_label"]:
                    continue
                pred = cand
                break

        if pred is None:
            if force_no_self:
                all_candidates = []
                for table in models.values():
                    all_candidates.extend(list(table.values()))
                counts = Counter(all_candidates)
                for lab, _ in counts.most_common():
                    if lab != r["current_label"]:
                        pred = lab
                        break
            if pred is None:
                pred = majority

        preds.append(pred)

    return np.array(preds)


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_history_classifier(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Cell 23. One-hot label-history columns + median-imputed/robust-scaled
    numeric columns -> class-balanced multinomial logistic regression."""
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("cat", _make_ohe(), cat_cols))
    if len(num_cols) > 0:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
            num_cols,
        ))
    pre = ColumnTransformer(transformers=transformers)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", multi_class="auto", solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def score_predictions(y_true, y_pred, model_name: str, eval_scope: str, history_len: int, all_labels: list[str]) -> dict:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    if len(y_true) == 0:
        return {"model": model_name, "history_len": history_len, "eval_scope": eval_scope,
                "n": 0, "accuracy": np.nan, "macro_f1": np.nan, "balanced_accuracy": np.nan}

    return {
        "model": model_name, "history_len": history_len, "eval_scope": eval_scope, "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run_logo_history_experiments(examples_by_h: dict[int, pd.DataFrame], all_labels: list[str], sensor_cols_to_use: list[str]):
    """Cell 24. For each history length: LOGO over `group`, running the two
    naive baselines (repeat-current, train-majority), n-gram back-off
    (both with and without a "don't just repeat current label" override),
    and up to 3 logistic-regression variants (label-only, sensor-only,
    label+sensor) — whichever of the latter two are applicable depends on
    whether any lagged sensor columns exist for this history length.
    Returns (summary_df, predictions_df)."""
    logo = LeaveOneGroupOut()
    all_summary_rows = []
    all_prediction_rows = []

    for h, ex in examples_by_h.items():
        print(f"\n{'#' * 100}\nHISTORY LENGTH: {h}\n{'#' * 100}")

        label_cols = [f"label_lag{lag}" for lag in range(h)]
        label_summary_cols = ["history_has_transition", "history_unique_labels", "history_most_recent_run_length"]

        sensor_lag_cols = [c for c in ex.columns if "__lag" in c and c.split("__lag")[0] in sensor_cols_to_use]

        model_specs = [{"name": "logreg_label_history_only", "cat_cols": label_cols, "num_cols": label_summary_cols}]
        if len(sensor_lag_cols) > 0:
            model_specs.append({"name": "logreg_sensor_history_only", "cat_cols": [], "num_cols": sensor_lag_cols})
            model_specs.append({"name": "logreg_label_plus_sensor_history", "cat_cols": label_cols, "num_cols": label_summary_cols + sensor_lag_cols})

        y_all, repeat_all, majority_all, ngram_all, ngram_no_self_all = [], [], [], [], []
        trans_all, group_all, pos_all = [], [], []
        clf_preds = {spec["name"]: [] for spec in model_specs}

        for tr_idx, te_idx in logo.split(ex, ex["next_label"], ex["group"]):
            train, test = ex.iloc[tr_idx].copy(), ex.iloc[te_idx].copy()

            y_true = test["next_label"].astype(str).values
            y_all.extend(y_true)
            repeat_all.extend(test["current_label"].astype(str).values)
            majority_all.extend([train["next_label"].value_counts().idxmax()] * len(test))
            trans_all.extend(test["is_transition"].values)
            group_all.extend(test["group"].values)
            pos_all.extend(test["pos"].values)

            ngram_models, ngram_majority = train_ngram_transition_model(train, h)
            ngram_all.extend(predict_ngram_backoff(test, ngram_models, ngram_majority, h, force_no_self=False))
            ngram_no_self_all.extend(predict_ngram_backoff(test, ngram_models, ngram_majority, h, force_no_self=True))

            for spec in model_specs:
                pipe = build_history_classifier(spec["cat_cols"], spec["num_cols"])
                pipe.fit(train[spec["cat_cols"] + spec["num_cols"]], train["next_label"].astype(str))
                clf_preds[spec["name"]].extend(pipe.predict(test[spec["cat_cols"] + spec["num_cols"]]))

        y_all, trans_all = np.array(y_all), np.array(trans_all).astype(bool)
        group_all, pos_all = np.array(group_all), np.array(pos_all)

        pred_sources = {
            "repeat_current_label": np.array(repeat_all),
            "train_majority_next_label": np.array(majority_all),
            f"ngram_markov_backoff_h{h}": np.array(ngram_all),
            f"ngram_markov_no_self_backoff_h{h}": np.array(ngram_no_self_all),
        }
        for name, pred in clf_preds.items():
            pred_sources[name] = np.array(pred)

        for eval_scope, mask in [("all_windows", np.ones(len(y_all), dtype=bool)), ("transition_only", trans_all)]:
            for name, pred in pred_sources.items():
                all_summary_rows.append(score_predictions(y_all[mask], pred[mask], name, eval_scope, h, all_labels))

        pred_df = pd.DataFrame({"history_len": h, "group": group_all, "pos": pos_all, "y_true": y_all, "is_transition": trans_all})
        for name, pred in pred_sources.items():
            pred_df[name] = pred
        all_prediction_rows.append(pred_df)

    summary = pd.DataFrame(all_summary_rows).sort_values(["eval_scope", "macro_f1", "accuracy"], ascending=[True, False, False]).reset_index(drop=True)
    predictions = pd.concat(all_prediction_rows, ignore_index=True)
    return summary, predictions


def compute_fold_std(predictions: pd.DataFrame, all_labels: list[str]) -> pd.DataFrame:
    """Cell 27. Held-out-group mean +/- SD of accuracy/macro-F1 for every
    (history_len, model, eval_scope) combination — this is what fills in
    Table 8.2's "macro-F1 fold mean+/-SD [CI]" column."""
    rows = []
    for history_len, pred_subset in predictions.groupby("history_len"):
        model_columns = [c for c in pred_subset.columns if c not in {"history_len", "group", "pos", "y_true", "is_transition"}]

        for model_name in model_columns:
            for eval_scope, scope_df in [("all_windows", pred_subset), ("transition_only", pred_subset[pred_subset["is_transition"]])]:
                for held_group, group_df in scope_df.groupby("group"):
                    if group_df.empty:
                        continue
                    rows.append({
                        "history_len": history_len, "model": model_name, "eval_scope": eval_scope,
                        "held_group": held_group, "n": len(group_df),
                        "accuracy": accuracy_score(group_df["y_true"].astype(str), group_df[model_name].astype(str)),
                        "macro_f1": f1_score(group_df["y_true"].astype(str), group_df[model_name].astype(str), labels=all_labels, average="macro", zero_division=0),
                    })

    fold_metrics = pd.DataFrame(rows)
    fold_std = fold_metrics.groupby(["history_len", "model", "eval_scope"], as_index=False).agg(
        fold_accuracy_mean=("accuracy", "mean"), fold_accuracy_std=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"), fold_macro_f1_std=("macro_f1", "std"),
        n_folds=("held_group", "nunique"),
    )
    return fold_std


def run_all(data_root: str, out_dir: str, history_lens=HISTORY_LENS, max_sensor_features: int = MAX_SENSOR_FEATURES,
            use_feature_lags: bool = USE_FEATURE_LAGS):
    """Orchestrates cells 18-27: load labels, auto-select + merge a sensor
    feature file, build history examples for every history_len, run the
    full LOGO model comparison, compute fold mean+/-SD, and check the
    result against the published Table 8.2 rows. Returns
    (summary, predictions, fold_std, comparison, check)."""
    os.makedirs(out_dir, exist_ok=True)

    # apply_merge6=False: see this module's docstring "REPRODUCIBILITY NOTE"
    # (2026-09-06) -- Table 8.2's own title says "7-label", and real-data
    # validation confirms the published numbers only reproduce against the
    # true, un-collapsed 7-class rq3_process_label vocabulary, not the
    # 6-label social/task-conversation merge the source notebook's cells
    # 17-18 literally apply.
    labels_df, group_col_label, time_col_label, label_col = load_normalized_labels(data_root, apply_merge6=False)
    print("Loaded labels:", labels_df.shape, "| 7-label distribution:")
    print(labels_df[label_col].value_counts())

    data, group_col, time_col, feature_score_df = select_and_merge_feature_file(data_root, labels_df, label_col, apply_merge6=False)
    feature_score_df.to_csv(os.path.join(out_dir, "history_model_feature_file_scores.csv"), index=False)

    feature_cols = detect_numeric_sensor_features(data, group_col, time_col, label_col)
    print("Detected numeric sensor features:", len(feature_cols))
    pd.DataFrame({"feature": feature_cols}).to_csv(os.path.join(out_dir, "detected_numeric_features.csv"), index=False)

    sensor_cols_to_use = feature_cols[:max_sensor_features]
    all_labels = sorted(data[label_col].unique())

    examples_by_h = {}
    for h in history_lens:
        ex = make_history_examples(data, group_col, time_col, label_col, h, sensor_cols_to_use, include_sensor_features=use_feature_lags)
        examples_by_h[h] = ex
        print(f"history_len={h}: {len(ex)} examples, {int(ex['is_transition'].sum())} transitions")
        ex.to_csv(os.path.join(out_dir, f"history_examples_h{h}.csv"), index=False)

    summary, predictions = run_logo_history_experiments(examples_by_h, all_labels, sensor_cols_to_use)
    summary.to_csv(os.path.join(out_dir, "rq3_7label_history_aware_summary.csv"), index=False)
    predictions.to_csv(os.path.join(out_dir, "rq3_7label_history_aware_predictions.csv"), index=False)

    fold_std = compute_fold_std(predictions, all_labels)
    fold_std.to_csv(os.path.join(out_dir, "rq3_7label_history_aware_fold_std.csv"), index=False)

    comparison = summary[summary["model"].isin(
        ["logreg_label_history_only", "logreg_sensor_history_only", "logreg_label_plus_sensor_history"]
    )].sort_values(["eval_scope", "history_len", "macro_f1"], ascending=[True, True, False])
    comparison.to_csv(os.path.join(out_dir, "label_vs_sensor_history_comparison.csv"), index=False)

    print("\n" + "=" * 100)
    print("BEST MODELS (transition_only)")
    print("=" * 100)
    print(summary[summary["eval_scope"] == "transition_only"].head(10).round(4).to_string(index=False))

    # Table 8.2 combines pooled (n/A/M/B) numbers from `summary` with
    # fold mean+/-SD numbers from `fold_std` for the same
    # (model, history_len, eval_scope) triple.
    check_rows = []
    for (model, history_len, eval_scope), ref in REPORT_REFERENCE.items():
        row = summary[(summary["model"] == model) & (summary["history_len"] == history_len) & (summary["eval_scope"] == eval_scope)]
        if row.empty:
            check_rows.append({"model": model, "history_len": history_len, "eval_scope": eval_scope, "match": "MISSING"})
            continue
        row = row.iloc[0]
        match = (
            "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0006 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0006
                        and abs(row["balanced_accuracy"] - ref["balanced_accuracy"]) < 0.0006 and int(row["n"]) == ref["n"])
            else "DIFFERS"
        )
        check_rows.append({
            "model": model, "history_len": history_len, "eval_scope": eval_scope,
            "n": int(row["n"]), "report_n": ref["n"],
            "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"],
            "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"],
            "match": match,
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.2")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "task3_persistence_report_reproduction_check.csv"), index=False)

    n_exact = int((check["match"] == "EXACT").sum())
    print(f"\n{n_exact} of {len(check)} referenced Table 8.2 rows reproduce exactly.")

    return summary, predictions, fold_std, comparison, check
