"""Label cleaning / annotation normalization for Task 3 (Ch.4 §4.5-4.6,
Appendix A).

Ported from `THESIS_NOTEBOOKS/ML/rq3_label_audit_and_normalization.ipynb`.
Produces `RQ3_LABEL_NORMALIZATION/rq3_normalized_labels_full.csv` (+
`_compact.csv` and `rq3_label_normalization_metadata.json`) — the input
every Task 3 window-level module in this repo already expects
(`src/models/task3_tokens.py`, `task3_persistence.py`,
`task3_segment_forecast.py`, `task3_naive5.py`).

Pipeline: auto-discover candidate label-bearing CSVs already produced by
an earlier feature-engineering pass (scored, not hardcoded — the source
notebook's own scoring prefers `recognition_interaction_window_label_
inventory.csv`'s `dominant_normalized_label` column, which is exactly
what `task3_expanding_prefix.py` reads directly too), normalize the raw
ELAN label text (typo/spelling fixes + `+`/`,`/`;`-joined compound-label
splitting), map each atomic component to one of 8 coarse "process"
categories (`PROCESS_PRIORITY`), then collapse rare process labels
(< MIN_LABEL_WINDOWS occurrences) into "other".

Note the resulting `rq3_process_label` taxonomy (merging / building /
moving_transport / inspection / object_handover / task_conversation /
social_conversation / other) is exactly what task3_tokens.py's `MERGE6`
constant (`social_conversation`/`task_conversation` -> `conversation`)
and the rest of Task 3's 6-label scheme downstream of it are built on —
confirms this module is the actual upstream source of that vocabulary,
not an independent guess.

INPUT DEPENDENCY: needs at least one already-existing feature/label CSV
under `data_root` with a label-like column (see FILENAME_KEYWORDS /
LABEL_KEYWORDS) — i.e. this stage runs after some feature-engineering
pass has produced window-level labels, same as the rest of Task 3.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

FILENAME_KEYWORDS = [
    "ml_ready", "ml-ready", "activity", "advanced", "merged", "eng3",
    "label_inventory", "recognition", "interaction",
]
EXCLUDE_KEYWORDS = [
    "summary", "prediction", "predictions", "result", "results", "best",
    "plot", "explain", "importance", "confusion",
]
LABEL_KEYWORDS = ["label", "activity", "state", "target", "class", "dominant", "normalized", "recognition"]
LABEL_COLUMN_BAD_TOKENS = ["prob", "score", "confidence", "duration", "count", "n_", "num_", "ratio"]

# Meaningful Task 3 process grouping — individual and co- versions of the
# same activity are intentionally merged into one process category.
PROCESS_PRIORITY = [
    "merging", "building", "moving_transport", "inspection",
    "object_handover", "task_conversation", "social_conversation", "other",
]

MIN_LABEL_WINDOWS_DEFAULT = 20
SELECTED_RQ3_LABEL_COL_DEFAULT = "rq3_compact_process_label"  # fewer states -> safer for HMM than rq3_process_label


def find_candidate_csvs(data_root: str) -> list[str]:
    all_csvs = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))
    candidates = []
    for p in all_csvs:
        name = os.path.basename(p).lower()
        full = p.lower()
        has_good = any(k in name or k in full for k in FILENAME_KEYWORDS)
        has_bad = any(k in name or k in full for k in EXCLUDE_KEYWORDS)
        if has_good and not has_bad:
            candidates.append(p)
    return candidates


def looks_like_label_column(col) -> bool:
    c = str(col).lower()
    if any(k in c for k in LABEL_KEYWORDS):
        return not any(b in c for b in LABEL_COLUMN_BAD_TOKENS)
    return False


def detect_label_columns(candidate_csvs: list[str]) -> pd.DataFrame:
    """Samples the first 100 rows of every candidate CSV and records
    every column whose name looks label-like."""
    rows = []
    for path in candidate_csvs:
        try:
            sample = pd.read_csv(path, nrows=100)
        except Exception as e:
            print("Could not read:", path, "|", repr(e))
            continue

        label_cols = [c for c in sample.columns if looks_like_label_column(c)]
        if not label_cols:
            continue

        try:
            n_rows = sum(1 for _ in open(path, "rb")) - 1
        except Exception:
            n_rows = np.nan

        for c in label_cols:
            vals = sample[c].dropna().astype(str)
            vals = vals[vals.str.strip() != ""]
            rows.append({
                "file_path": path, "file_name": os.path.basename(path), "n_rows_estimate": n_rows,
                "label_column": c, "sample_n_unique": vals.nunique(),
                "sample_values": " | ".join(vals.value_counts().head(10).index.tolist()),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No label-like columns found in candidate CSVs. Try loosening FILENAME_KEYWORDS or LABEL_KEYWORDS.")
    return df.sort_values(["file_name", "label_column"]).reset_index(drop=True)


def build_label_inventory(label_columns_df: pd.DataFrame) -> pd.DataFrame:
    """Full raw-label value_counts for every detected (file, column) pair."""
    rows = []
    for _, row in label_columns_df.iterrows():
        path, col = row["file_path"], row["label_column"]
        try:
            df_tmp = pd.read_csv(path, usecols=[col])
        except Exception as e:
            print("Could not read column:", col, "from", path, "|", repr(e))
            continue

        vals = df_tmp[col].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        for label, count in vals.value_counts(dropna=False).items():
            rows.append({"file_path": path, "file_name": os.path.basename(path), "label_column": col, "raw_label": label, "count": int(count)})

    return pd.DataFrame(rows).sort_values(["file_name", "label_column", "count"], ascending=[True, True, False]).reset_index(drop=True)


def choose_main_label_source(label_columns_df: pd.DataFrame, override_file: str | None = None, override_col: str | None = None):
    """Scores each detected (file, column) candidate and returns the
    best (main_file, main_label_col). An explicit override always wins
    (mirrors the source notebook's commented-out manual-override
    lines)."""
    if override_file is not None and override_col is not None:
        return override_file, override_col

    scored = []
    for _, row in label_columns_df.iterrows():
        p, c = row["file_path"], row["label_column"]
        plow, clow = p.lower(), c.lower()

        score = 0
        if "recognition_interaction_window_label_inventory" in plow:
            score += 100
        if "activity3_advanced_merged_10s_features" in plow:
            score += 80
        if "ml_ready" in plow or "ml-ready" in plow:
            score += 60
        if "merged" in plow:
            score += 20
        if clow == "dominant_normalized_label":
            score += 50
        if "activity" in clow:
            score += 30
        if "target" in clow:
            score += 10

        scored.append((score, p, c))

    scored.sort(reverse=True)
    if not scored:
        raise RuntimeError("No label source candidates to choose from.")
    return scored[0][1], scored[0][2]


# ================================================================
# Label normalization
# ================================================================

_CLEAN_LABEL_REPLACEMENTS = {
    "object_handiver": "object_handover",
    "matching_pieces_with_target_image": "matching_pieces_to_target_image",
    "matching_pieces_to_image": "matching_pieces_to_target_image",
    "matching_pieces_with_image": "matching_pieces_to_target_image",
    "non_task_convo": "task_social_convo",
    "task_related_social_convo": "task_social_convo",
    "task_related_convo": "task_operational_convo",
}


def clean_label(x) -> str:
    if pd.isna(x):
        return "none"
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"__+", "_", s).strip("_")
    for old, new in _CLEAN_LABEL_REPLACEMENTS.items():
        s = s.replace(old, new)
    return s


def split_compound_label(x) -> list[str]:
    """Splits labels like 'task_operational_convo_+_co_inspecting_pieces'."""
    s = clean_label(x)
    parts = re.split(r"_\+_|\+|,|;", s)
    parts = [p.strip("_ ").strip() for p in parts if p.strip("_ ").strip()]
    return parts if parts else ["none"]


def component_to_process_label(component) -> str:
    c = clean_label(component)
    if c in ("none", "nan", ""):
        return "other"

    if "convo" in c or "conversation" in c:
        return "social_conversation" if ("social" in c or "non_task" in c) else "task_conversation"

    if "handover" in c or "delivering_piece" in c or "delivering_target_image" in c:
        return "object_handover"

    if any(tok in c for tok in ("building", "build", "subpiece_together", "placing_subpiece", "placing_piece", "starting_individual_build")):
        return "building"

    if "merging" in c or "merge" in c:
        return "merging"

    if any(tok in c for tok in ("inspect", "inspection", "matching", "searching", "target_image", "puzzle_pieces")):
        return "inspection"

    if any(tok in c for tok in ("moving", "tray", "carrying", "traveling", "approaching", "move", "transport")):
        return "moving_transport"

    if "synchronization" in c or "sync" in c:
        return "moving_transport"

    return "other"


def raw_to_process_label(raw_label) -> str:
    mapped = [component_to_process_label(p) for p in split_compound_label(raw_label)]
    for p in PROCESS_PRIORITY:
        if p in mapped:
            return p
    return "other"


def raw_to_conversation_binary(raw_label) -> str:
    mapped = [component_to_process_label(p) for p in split_compound_label(raw_label)]
    return "conversation" if ("task_conversation" in mapped or "social_conversation" in mapped) else "non_conversation"


def raw_to_compact_process_label(raw_label) -> str:
    p = raw_to_process_label(raw_label)
    if p in ("task_conversation", "social_conversation"):
        return "conversation"
    if p in ("building", "merging"):
        return p
    if p in ("inspection", "moving_transport", "object_handover"):
        return "preparation_or_transition"
    return "other"


def apply_normalization(df: pd.DataFrame, main_label_col: str) -> pd.DataFrame:
    df = df.copy()
    df["raw_label"] = df[main_label_col].astype(str).map(clean_label)
    df["rq3_process_label"] = df["raw_label"].map(raw_to_process_label)
    df["rq3_compact_process_label"] = df["raw_label"].map(raw_to_compact_process_label)
    df["rq3_conversation_binary"] = df["raw_label"].map(raw_to_conversation_binary)
    return df


def build_mapping_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["raw_label", "rq3_process_label", "rq3_compact_process_label", "rq3_conversation_binary"], as_index=False)
        .size().rename(columns={"size": "count"})
        .sort_values(["rq3_process_label", "count"], ascending=[True, False])
    )


def _find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def detect_group_time_columns(df: pd.DataFrame):
    group_col = _find_first_existing(df.columns, ["group", "group_id", "session", "session_id", "participant_group"])
    time_col = _find_first_existing(df.columns, ["window_start", "win_start", "start", "start_time", "timestamp", "time", "window_mid"])
    return group_col, time_col


def _collapse_repeats(seq: list) -> list:
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def check_temporal_sequences(df: pd.DataFrame, group_col: str, time_col: str, label_cols=("rq3_process_label", "rq3_compact_process_label")) -> dict[str, pd.DataFrame]:
    """Diagnostic only (not required for the saved output files): reports
    windows->segments collapse and the most common segment-to-segment
    transitions per label column. Returns {label_col: transition_counts_df}."""
    temp = df.copy()
    temp[time_col] = pd.to_numeric(temp[time_col], errors="coerce")
    temp = temp.sort_values([group_col, time_col])

    results = {}
    for label_col in label_cols:
        transition_rows = []
        for g, sub in temp.groupby(group_col):
            seq_seg = _collapse_repeats(sub[label_col].astype(str).tolist())
            for a, b in zip(seq_seg[:-1], seq_seg[1:]):
                transition_rows.append({"label_col": label_col, "group": g, "from": a, "to": b})

        trans_df = pd.DataFrame(transition_rows)
        if len(trans_df) > 0:
            trans_df = trans_df.groupby(["from", "to"], as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        results[label_col] = trans_df
    return results


def select_final_label(df: pd.DataFrame, selected_col: str = SELECTED_RQ3_LABEL_COL_DEFAULT, min_label_windows: int = MIN_LABEL_WINDOWS_DEFAULT):
    """Merges any value of `selected_col` occurring fewer than
    min_label_windows times into "other", storing the result as
    rq3_final_label. Returns (df, rare_labels_set)."""
    df = df.copy()
    counts = df[selected_col].value_counts()
    rare = set(counts[counts < min_label_windows].index)
    df["rq3_final_label"] = df[selected_col].where(~df[selected_col].isin(rare), "other")
    return df, rare


def run_all(data_root: str, out_dir: str | None = None, main_file_override: str | None = None,
            main_label_col_override: str | None = None, selected_rq3_label_col: str = SELECTED_RQ3_LABEL_COL_DEFAULT,
            min_label_windows: int = MIN_LABEL_WINDOWS_DEFAULT, run_temporal_check: bool = True):
    """Orchestrates the full pipeline and writes
    RQ3_LABEL_NORMALIZATION/rq3_normalized_labels_full.csv (+ _compact.csv
    + metadata json). Returns (df, mapping_table, metadata)."""
    out_dir = out_dir or os.path.join(data_root, "RQ3_LABEL_NORMALIZATION")
    os.makedirs(out_dir, exist_ok=True)

    candidates = find_candidate_csvs(data_root)
    print("Candidate usable/ML-ready CSV files:", len(candidates))

    label_columns_df = detect_label_columns(candidates)
    label_columns_df.to_csv(os.path.join(out_dir, "detected_label_columns_in_candidate_files.csv"), index=False)

    label_inventory = build_label_inventory(label_columns_df)
    label_inventory.to_csv(os.path.join(out_dir, "all_detected_raw_labels_inventory.csv"), index=False)

    main_file, main_label_col = choose_main_label_source(label_columns_df, main_file_override, main_label_col_override)
    print("Selected MAIN_FILE:", main_file, "| MAIN_LABEL_COL:", main_label_col)

    main_df = pd.read_csv(main_file)
    if main_label_col not in main_df.columns:
        raise ValueError(f"{main_label_col} not found in {main_file}.")

    df = apply_normalization(main_df, main_label_col)
    mapping_table = build_mapping_table(df)
    mapping_table.to_csv(os.path.join(out_dir, "rq3_raw_to_normalized_label_mapping.csv"), index=False)

    print("\nrq3_process_label distribution:")
    print(df["rq3_process_label"].value_counts())

    group_col, time_col = detect_group_time_columns(df)
    print("Detected GROUP_COL:", group_col, "| TIME_COL:", time_col)

    if run_temporal_check and group_col is not None and time_col is not None:
        transitions = check_temporal_sequences(df, group_col, time_col)
        for label_col, trans_df in transitions.items():
            if len(trans_df) > 0:
                trans_df.to_csv(os.path.join(out_dir, f"{label_col}_segment_transition_counts.csv"), index=False)

    df, rare = select_final_label(df, selected_rq3_label_col, min_label_windows)
    print(f"Rare '{selected_rq3_label_col}' values merged to 'other' (< {min_label_windows} windows):", sorted(rare))

    normalized_path = os.path.join(out_dir, "rq3_normalized_labels_full.csv")
    df.to_csv(normalized_path, index=False)
    print("Saved:", normalized_path)

    keep_cols = []
    for c in [group_col, time_col, main_label_col]:
        if c is not None and c in df.columns and c not in keep_cols:
            keep_cols.append(c)
    for c in ["raw_label", "rq3_process_label", "rq3_compact_process_label", "rq3_conversation_binary", "rq3_final_label"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    compact_path = os.path.join(out_dir, "rq3_normalized_labels_compact.csv")
    df[keep_cols].to_csv(compact_path, index=False)
    print("Saved:", compact_path)

    metadata = {
        "main_file": main_file, "main_label_col": main_label_col,
        "group_col": group_col, "time_col": time_col,
        "selected_rq3_label_col": selected_rq3_label_col, "final_label_col": "rq3_final_label",
        "out_dir": out_dir,
    }
    metadata_path = os.path.join(out_dir, "rq3_label_normalization_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved:", metadata_path)

    return df, mapping_table, metadata
