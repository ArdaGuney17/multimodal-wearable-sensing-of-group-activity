"""Task 3, Part I — build six-label "activity tokens" from window-level labels
+ sensor features.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "GLOBAL
CONFIGURATION" / "BUILD FULL-STATISTIC SIX-LABEL ACTIVITY TOKENS" (source
notebook's own "Part I"). See docs/table_to_source_mapping.md and
notebooks_reference/task3_FINAL_V2_with_exact_report_reproduction_ANALYSIS.md
for the full porting notes.

A "token" = one contiguous run of consecutive same-label windows within a
group, collapsed into a single row: duration (in windows), start time, and
14 summary statistics (mean/std/min/max/range/median/iqr/p10/p25/p75/p90/
energy/rms/entropy) per numeric sensor channel. This is the representation
used by every Task 3 model (grammar, HMM, tokenized Transformer/LSTM).

IMPORTANT PORTING NOTE (from the source notebook, preserved here): in the
*original* Colab workflow, this token table was not actually rebuilt by this
notebook on a normal run — an earlier "Task 3 CORRECTED_FINAL" notebook (not
among the 3 provided for this port) had already produced a verified token
table, and this notebook's own cell 6 just copied it in and skipped straight
to reloading. The from-scratch build path below (`load_six_label_windows` /
`build_fullstat_tokens`) is real, executable code, and — as far as we know —
was never independently confirmed to be the path that actually produced the
original published Table 8.x numbers.

UPDATE (2026-09-05, Chapter 8 pilot validation): this from-scratch path was
run for real against `data/external/thesis_data/RQ3_LABEL_NORMALIZATION/
rq3_normalized_labels_full.csv` (label source) + `data/external/thesis_data/
INTERACTION_ENG3/interaction_eng3_features.csv` (sensor-feature source,
itself already validated exact from raw model_ready data by
`src/features/eng3_recognition_labels.py`'s `build_eng3_grid` — see
docs/session_2026-09-05_autonomous_progress.md) and diffed row-by-row,
column-by-column against the official
`PUBLICATION_TASK3_CORRECTED_FINAL/activity_tokens_6label_fullstat.csv`.
Result for Group 1: **exact match, 15/15 rows, 340/340 columns, 0 value
mismatches at 1e-6 tolerance** (see
`data/external/thesis_data/RAW_VALIDATION_FEATURES/
run_group1_task3_tokens_validation.py`). All 9 groups together also
reproduce the expected (244 tokens, 9 groups, 6 classes) shape used by
`get_or_build_tokens()`'s sanity check below. `feature_cols` (duration + 336
per-channel stats, excluding group/label/start_time) is 337-dimensional,
matching the source notebook's own "337-dimensional" markdown claim exactly
— that claim is confirmed, not stale. This does not prove this exact code
path is bit-for-bit what originally produced the historical Table 8.x
numbers (that provenance question is unresolved either way), but it does
confirm the ported mechanism is correct and fully reproducible from the
already-validated upstream artifacts, for Group 1 at least. Not yet
independently re-diffed per-group for groups 2/3/5/6/7/8/9/10 (only the
aggregate shape was checked for those).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The label-merge applied before tokenization: both conversation sub-types
# collapse into a single "conversation" class for the 6-label scheme.
MERGE6 = {
    "social_conversation": "conversation",
    "task_conversation": "conversation",
}

STAT_NAMES = [
    "mean", "std", "min", "max", "range", "median", "iqr",
    "p10", "p25", "p75", "p90", "energy", "rms", "entropy",
]

EXPECTED_TOKEN_COUNT = 244   # per docs/thesis_reproduction_targets.md §6.2
EXPECTED_GROUP_COUNT = 9
EXPECTED_CLASS_COUNT = 6


def first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def label_normalization_path(data_root: str) -> str:
    return os.path.join(data_root, "RQ3_LABEL_NORMALIZATION", "rq3_normalized_labels_full.csv")


def feature_candidate_paths(data_root: str) -> list[str]:
    return [
        os.path.join(data_root, "INTERACTION_ENG3", "interaction_eng3_features.csv"),
        os.path.join(data_root, "INTERACTION_OE10", "interaction_oe10_10s.csv"),
    ]


def load_six_label_windows(data_root: str):
    """Load window-level labels (merged to the 6-class scheme) and, if
    available, merge in numeric sensor features by (group, rounded time)."""
    norm_path = label_normalization_path(data_root)
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Label normalization output not found:\n{norm_path}\n"
            "This is a feature-engineering/label-cleaning stage output, not raw "
            "data — see docs/table_to_source_mapping.md for what produces it."
        )

    labels = pd.read_csv(norm_path)
    label_col = "rq3_process_label"

    group_col = first_existing(labels.columns, ["group", "group_id", "session"])
    time_col = first_existing(labels.columns, ["window_start", "win_start", "start", "time"])
    if group_col is None or time_col is None or label_col not in labels.columns:
        raise ValueError("Could not identify group, time, or process-label columns.")

    labels = labels.dropna(subset=[group_col, time_col, label_col]).copy()
    labels[label_col] = labels[label_col].astype(str).replace(MERGE6)
    labels["_g"] = labels[group_col].astype(str)
    labels["_t"] = pd.to_numeric(labels[time_col], errors="coerce").round(1)

    feature_path = next((p for p in feature_candidate_paths(data_root) if os.path.exists(p)), None)

    df = labels[["_g", "_t", label_col]].rename(columns={label_col: "label"})
    sensor_cols: list[str] = []

    if feature_path is not None:
        features = pd.read_csv(feature_path)
        feature_group = first_existing(features.columns, ["group", "group_id", "session"])
        feature_time = first_existing(features.columns, ["window_start", "win_start", "start", "time"])

        features["_g"] = features[feature_group].astype(str)
        features["_t"] = pd.to_numeric(features[feature_time], errors="coerce").round(1)

        excluded = {
            feature_group, feature_time, "_g", "_t", "group", "window_start", "window_end",
            "window_mid", "recognition_label", "label", "binary_label",
        }
        for column in features.columns:
            if column in excluded:
                continue
            if pd.to_numeric(features[column], errors="coerce").notna().sum() > 0:
                sensor_cols.append(column)

        feature_small = features[["_g", "_t"] + sensor_cols].copy()
        for column in sensor_cols:
            feature_small[column] = pd.to_numeric(feature_small[column], errors="coerce")

        df = df.merge(feature_small.drop_duplicates(["_g", "_t"]), on=["_g", "_t"], how="left")
    else:
        print(
            "WARNING: no sensor feature file found among:\n  "
            + "\n  ".join(feature_candidate_paths(data_root))
            + "\nTokens will carry label/timing only, no sensor statistics."
        )

    df = df.dropna(subset=["label"]).sort_values(["_g", "_t"]).reset_index(drop=True)
    df["group"] = df["_g"]
    return df, sensor_cols


def channel_statistics(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {name: 0.0 for name in STAT_NAMES}

    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])

    if len(values) >= 8:
        power = np.abs(np.fft.rfft(values - values.mean())) ** 2
        power = power[1:]
        if power.sum() > 0 and len(power) > 1:
            probability = power / power.sum()
            entropy = float(-(probability * np.log(probability + 1e-12)).sum() / np.log(len(probability)))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "max": float(values.max()), "range": float(values.max() - values.min()),
        "median": float(q50), "iqr": float(q75 - q25),
        "p10": float(q10), "p25": float(q25), "p75": float(q75), "p90": float(q90),
        "energy": float(np.mean(values ** 2)), "rms": float(np.sqrt(np.mean(values ** 2))),
        "entropy": entropy,
    }


def build_fullstat_tokens(window_df: pd.DataFrame, sensor_cols: list[str]):
    """Collapse consecutive same-label windows within each group into one
    "token" row (duration + per-channel statistics)."""
    token_rows = []

    for group, group_df in window_df.groupby("group"):
        group_df = group_df.sort_values("_t").reset_index(drop=True)
        segment_id = (group_df["label"] != group_df["label"].shift()).cumsum()

        for _, segment in group_df.groupby(segment_id):
            record = {
                "group": group,
                "label": segment["label"].iloc[0],
                "duration": float(len(segment)),
                "start_time": float(segment["_t"].iloc[0]),
            }
            for channel in sensor_cols:
                stats = channel_statistics(pd.to_numeric(segment[channel], errors="coerce").values)
                for stat_name, value in stats.items():
                    record[f"{channel}__{stat_name}"] = value
            token_rows.append(record)

    tokens = pd.DataFrame(token_rows)
    tokens = tokens.sort_values(["group", "start_time"]).reset_index(drop=True)
    feature_cols = [c for c in tokens.columns if c not in {"group", "label", "start_time"}]
    return tokens, feature_cols


def token_path(core_out: str) -> str:
    return os.path.join(core_out, "activity_tokens_6label_fullstat.csv")


def get_or_build_tokens(data_root: str, core_out: str, resume_existing: bool = True, verify_counts: bool = True):
    """Load the token table from `core_out` if it already exists (matches the
    source notebook's default "copy a verified table in" behavior — just
    without the copy-from-another-notebook step, since we don't have that
    notebook), otherwise build it from scratch via load_six_label_windows +
    build_fullstat_tokens.

    If verify_counts, warns (does not raise) when the result doesn't match
    the thesis's reported 244 tokens / 9 groups / 6 classes — a from-scratch
    build is not guaranteed to reproduce the historical table exactly, since
    we don't have the original "CORRECTED_FINAL" notebook that produced it.
    """
    path = token_path(core_out)

    if resume_existing and os.path.exists(path):
        print("Reloading existing activity-token table:", path)
        T = pd.read_csv(path)
        fcols = [c for c in T.columns if c not in {"group", "label", "start_time"}]
    else:
        print("Building activity tokens from scratch (UNVERIFIED path — see module docstring).")
        windows_6label, raw_sensor_cols = load_six_label_windows(data_root)
        T, fcols = build_fullstat_tokens(windows_6label, raw_sensor_cols)
        os.makedirs(core_out, exist_ok=True)
        T.to_csv(path, index=False)

    n_tokens, n_groups, n_classes = len(T), T["group"].nunique(), T["label"].nunique()
    print(f"tokens: {n_tokens} | groups: {n_groups} | classes: {n_classes} | token feature dims: {len(fcols)}")

    if verify_counts and (n_tokens != EXPECTED_TOKEN_COUNT or n_groups != EXPECTED_GROUP_COUNT or n_classes != EXPECTED_CLASS_COUNT):
        print(
            f"WARNING: token table shape ({n_tokens} tokens, {n_groups} groups, {n_classes} classes) "
            f"does not match the thesis's reported ({EXPECTED_TOKEN_COUNT}, {EXPECTED_GROUP_COUNT}, "
            f"{EXPECTED_CLASS_COUNT}). Downstream results built from this table should not be assumed "
            "to match published numbers — see this module's docstring."
        )

    return T, fcols
