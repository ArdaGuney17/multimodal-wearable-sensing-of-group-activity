# --- CELL 1 (code cell #1) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 2 (code cell #2) ---

# ================================================================
# CELL 1 - SETUP
# ================================================================

import os
import re
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# Change this if your thesis folder is elsewhere.
DATA_ROOT = "/content/drive/MyDrive/thesis/data"

OUT_DIR = os.path.join(DATA_ROOT, "RQ3_LABEL_NORMALIZATION")
os.makedirs(OUT_DIR, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("OUT_DIR:", OUT_DIR)


# --- CELL 3 (code cell #3) ---

# ================================================================
# CELL 2 - FIND CANDIDATE CSV FILES
# ================================================================

FILENAME_KEYWORDS = [
    "ml_ready",
    "ml-ready",
    "activity",
    "advanced",
    "merged",
    "eng3",
    "label_inventory",
    "recognition",
    "interaction",
]

EXCLUDE_KEYWORDS = [
    "summary",
    "prediction",
    "predictions",
    "result",
    "results",
    "best",
    "plot",
    "explain",
    "importance",
    "confusion",
]

all_csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True))

candidate_csvs = []
for p in all_csvs:
    name = os.path.basename(p).lower()
    full = p.lower()

    has_good_keyword = any(k in name or k in full for k in FILENAME_KEYWORDS)
    has_bad_keyword = any(k in name or k in full for k in EXCLUDE_KEYWORDS)

    if has_good_keyword and not has_bad_keyword:
        candidate_csvs.append(p)

print("Total CSV files found:", len(all_csvs))
print("Candidate usable/ML-ready CSV files:", len(candidate_csvs))

for i, p in enumerate(candidate_csvs[:250], start=1):
    print(f"{i:03d}. {p}")


# --- CELL 4 (code cell #4) ---

# ================================================================
# CELL 3 - DETECT LABEL-LIKE COLUMNS IN CANDIDATE FILES
# ================================================================

LABEL_KEYWORDS = [
    "label",
    "activity",
    "state",
    "target",
    "class",
    "dominant",
    "normalized",
    "recognition",
]

def looks_like_label_column(col):
    c = str(col).lower()
    if any(k in c for k in LABEL_KEYWORDS):
        bad = ["prob", "score", "confidence", "duration", "count", "n_", "num_", "ratio"]
        if not any(b in c for b in bad):
            return True
    return False


label_column_rows = []

for path in candidate_csvs:
    try:
        sample = pd.read_csv(path, nrows=100)
    except Exception as e:
        print("Could not read:", path, "|", repr(e))
        continue

    label_cols = [c for c in sample.columns if looks_like_label_column(c)]

    if len(label_cols) == 0:
        continue

    try:
        n_rows = sum(1 for _ in open(path, "rb")) - 1
    except Exception:
        n_rows = np.nan

    for c in label_cols:
        vals = sample[c].dropna().astype(str)
        vals = vals[vals.str.strip() != ""]
        label_column_rows.append({
            "file_path": path,
            "file_name": os.path.basename(path),
            "n_rows_estimate": n_rows,
            "label_column": c,
            "sample_n_unique": vals.nunique(),
            "sample_values": " | ".join(vals.value_counts().head(10).index.tolist()),
        })

label_columns_df = pd.DataFrame(label_column_rows)

if len(label_columns_df) == 0:
    raise RuntimeError("No label-like columns found in candidate CSVs. Try loosening FILENAME_KEYWORDS or LABEL_KEYWORDS.")

label_columns_df = label_columns_df.sort_values(
    ["file_name", "label_column"]
).reset_index(drop=True)

print("Detected label-like columns:")
display(label_columns_df)

label_columns_df.to_csv(
    os.path.join(OUT_DIR, "detected_label_columns_in_candidate_files.csv"),
    index=False
)

print("Saved:", os.path.join(OUT_DIR, "detected_label_columns_in_candidate_files.csv"))


# --- CELL 5 (code cell #5) ---

# ================================================================
# CELL 4 - FULL LABEL INVENTORY ACROSS DETECTED FILES
# ================================================================

inventory_rows = []

for _, row in label_columns_df.iterrows():
    path = row["file_path"]
    col = row["label_column"]

    try:
        df_tmp = pd.read_csv(path, usecols=[col])
    except Exception as e:
        print("Could not read column:", col, "from", path, "|", repr(e))
        continue

    vals = df_tmp[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]

    vc = vals.value_counts(dropna=False)

    for label, count in vc.items():
        inventory_rows.append({
            "file_path": path,
            "file_name": os.path.basename(path),
            "label_column": col,
            "raw_label": label,
            "count": int(count),
        })

label_inventory = pd.DataFrame(inventory_rows)

label_inventory = label_inventory.sort_values(
    ["file_name", "label_column", "count"],
    ascending=[True, True, False]
).reset_index(drop=True)

print("Combined label inventory:")
display(label_inventory)

out_path = os.path.join(OUT_DIR, "all_detected_raw_labels_inventory.csv")
label_inventory.to_csv(out_path, index=False)

print("Saved:", out_path)


# --- CELL 6 (code cell #6) ---

# ================================================================
# CELL 5 - CHOOSE MAIN LABEL FILE / COLUMN
# ================================================================

preferred_candidates = []

for _, row in label_columns_df.iterrows():
    p = row["file_path"]
    c = row["label_column"]
    plow = p.lower()
    clow = c.lower()

    score = 0

    if "recognition_interaction_window_label_inventory" in plow:
        score += 100
    if "activity3_advanced_merged_10s_features" in plow:
        score += 80
    if "ml_ready" in plow or "ml-ready" in plow:
        score += 60
    if "merged" in plow:
        score += 20
    if "dominant_normalized_label" == clow:
        score += 50
    if "activity" in clow:
        score += 30
    if "target" in clow:
        score += 10

    preferred_candidates.append((score, p, c))

preferred_candidates = sorted(preferred_candidates, reverse=True)

print("Top candidate label sources:")
for score, p, c in preferred_candidates[:30]:
    print(f"score={score:03d} | col={c} | file={p}")

MAIN_FILE = preferred_candidates[0][1]
MAIN_LABEL_COL = preferred_candidates[0][2]

# OPTIONAL MANUAL OVERRIDE:
# MAIN_FILE = "/content/drive/MyDrive/thesis/data/INTERACTION_ENG3/recognition_interaction_window_label_inventory.csv"
# MAIN_LABEL_COL = "dominant_normalized_label"

print("\nSelected MAIN_FILE:")
print(MAIN_FILE)
print("Selected MAIN_LABEL_COL:")
print(MAIN_LABEL_COL)

main_df = pd.read_csv(MAIN_FILE)

print("\nMain file shape:", main_df.shape)
print("Main file columns:")
print(main_df.columns.tolist())

if MAIN_LABEL_COL not in main_df.columns:
    raise ValueError(f"{MAIN_LABEL_COL} not found in MAIN_FILE.")

print("\nRaw label distribution in selected source:")
display(main_df[MAIN_LABEL_COL].astype(str).value_counts().reset_index().rename(
    columns={"index": "raw_label", MAIN_LABEL_COL: "count"}
))


# --- CELL 7 (code cell #7) ---

# ================================================================
# CELL 6 - NORMALIZATION FUNCTIONS
# ================================================================

def clean_label(x):
    if pd.isna(x):
        return "none"

    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    s = s.strip("_")

    replacements = {
        "object_handiver": "object_handover",
        "matching_pieces_with_target_image": "matching_pieces_to_target_image",
        "matching_pieces_to_image": "matching_pieces_to_target_image",
        "matching_pieces_with_image": "matching_pieces_to_target_image",
        "non_task_convo": "task_social_convo",
        "task_related_social_convo": "task_social_convo",
        "task_related_convo": "task_operational_convo",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    return s


def split_compound_label(x):
    # Splits labels such as task_operational_convo_+_co_inspecting_pieces.
    s = clean_label(x)
    parts = re.split(r"_\+_|\+|,|;", s)
    parts = [p.strip("_ ").strip() for p in parts if p.strip("_ ").strip()]
    if len(parts) == 0:
        parts = ["none"]
    return parts


def component_to_process_label(component):
    # Meaningful Task 3 process grouping.
    # Individual and co versions of the same process are intentionally merged.
    c = clean_label(component)

    if c in ["none", "nan", ""]:
        return "other"

    if "convo" in c or "conversation" in c:
        if "social" in c or "non_task" in c:
            return "social_conversation"
        return "task_conversation"

    if "handover" in c or "delivering_piece" in c or "delivering_target_image" in c:
        return "object_handover"

    if (
        "building" in c
        or "build" in c
        or "subpiece_together" in c
        or "placing_subpiece" in c
        or "placing_piece" in c
        or "starting_individual_build" in c
    ):
        return "building"

    if "merging" in c or "merge" in c:
        return "merging"

    if (
        "inspect" in c
        or "inspection" in c
        or "matching" in c
        or "searching" in c
        or "target_image" in c
        or "puzzle_pieces" in c
    ):
        return "inspection"

    if (
        "moving" in c
        or "tray" in c
        or "carrying" in c
        or "traveling" in c
        or "approaching" in c
        or "move" in c
        or "transport" in c
    ):
        return "moving_transport"

    if "synchronization" in c or "sync" in c:
        return "moving_transport"

    return "other"


# Priority for compound labels.
# Example: task_operational_convo + co_inspecting_pieces.
# For temporal process prediction, we prioritize the physical task process if it is present.
PROCESS_PRIORITY = [
    "merging",
    "building",
    "moving_transport",
    "inspection",
    "object_handover",
    "task_conversation",
    "social_conversation",
    "other",
]


def raw_to_process_label(raw_label):
    parts = split_compound_label(raw_label)
    mapped = [component_to_process_label(p) for p in parts]

    for p in PROCESS_PRIORITY:
        if p in mapped:
            return p

    return "other"


def raw_to_conversation_binary(raw_label):
    parts = split_compound_label(raw_label)
    mapped = [component_to_process_label(p) for p in parts]
    return "conversation" if ("task_conversation" in mapped or "social_conversation" in mapped) else "non_conversation"


def raw_to_compact_process_label(raw_label):
    p = raw_to_process_label(raw_label)

    if p in ["task_conversation", "social_conversation"]:
        return "conversation"
    if p == "building":
        return "building"
    if p == "merging":
        return "merging"
    if p in ["inspection", "moving_transport", "object_handover"]:
        return "preparation_or_transition"
    return "other"


print("Normalization functions loaded.")


# --- CELL 8 (code cell #8) ---

# ================================================================
# CELL 7 - APPLY NORMALIZATION AND PRINT RAW -> GROUPED MAPPING
# ================================================================

df = main_df.copy()

df["raw_label"] = df[MAIN_LABEL_COL].astype(str).map(clean_label)
df["rq3_process_label"] = df["raw_label"].map(raw_to_process_label)
df["rq3_compact_process_label"] = df["raw_label"].map(raw_to_compact_process_label)
df["rq3_conversation_binary"] = df["raw_label"].map(raw_to_conversation_binary)

mapping_table = (
    df
    .groupby(["raw_label", "rq3_process_label", "rq3_compact_process_label", "rq3_conversation_binary"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
    .sort_values(["rq3_process_label", "count"], ascending=[True, False])
)

print("=" * 100)
print("RAW LABEL -> NORMALIZED TASK 3 LABEL MAPPING")
print("=" * 100)
display(mapping_table)

print("\nDistribution: rq3_process_label")
display(df["rq3_process_label"].value_counts().reset_index().rename(
    columns={"index": "rq3_process_label", "rq3_process_label": "count"}
))

print("\nDistribution: rq3_compact_process_label")
display(df["rq3_compact_process_label"].value_counts().reset_index().rename(
    columns={"index": "rq3_compact_process_label", "rq3_compact_process_label": "count"}
))

print("\nDistribution: rq3_conversation_binary")
display(df["rq3_conversation_binary"].value_counts().reset_index().rename(
    columns={"index": "rq3_conversation_binary", "rq3_conversation_binary": "count"}
))

mapping_path = os.path.join(OUT_DIR, "rq3_raw_to_normalized_label_mapping.csv")
mapping_table.to_csv(mapping_path, index=False)

print("\nSaved mapping:", mapping_path)


# --- CELL 9 (code cell #9) ---

# ================================================================
# CELL 8 - CHECK TEMPORAL SEQUENCES AND TRANSITIONS
# ================================================================

def find_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

GROUP_COL = find_first_existing(df.columns, ["group", "group_id", "session", "session_id", "participant_group"])
TIME_COL = find_first_existing(df.columns, ["window_start", "win_start", "start", "start_time", "timestamp", "time", "window_mid"])

print("Detected GROUP_COL:", GROUP_COL)
print("Detected TIME_COL:", TIME_COL)

# Manual override if needed:
# GROUP_COL = "group"
# TIME_COL = "window_start"

if GROUP_COL is None or TIME_COL is None:
    print("Could not detect group/time columns automatically.")
    print("Available columns:")
    print(df.columns.tolist())
else:
    temp = df.copy()
    temp[TIME_COL] = pd.to_numeric(temp[TIME_COL], errors="coerce")
    temp = temp.sort_values([GROUP_COL, TIME_COL])

    def collapse_repeats(seq):
        out = []
        for x in seq:
            if len(out) == 0 or out[-1] != x:
                out.append(x)
        return out

    for label_col in ["rq3_process_label", "rq3_compact_process_label"]:
        print("\n" + "=" * 100)
        print("SEGMENT SEQUENCES FOR:", label_col)
        print("=" * 100)

        seg_counts = []
        transition_rows = []

        for g, sub in temp.groupby(GROUP_COL):
            seq_win = sub[label_col].astype(str).tolist()
            seq_seg = collapse_repeats(seq_win)
            seg_counts.append(len(seq_seg))

            print(f"group {g}: {len(seq_win)} windows -> {len(seq_seg)} segments")
            print("  " + " -> ".join(seq_seg[:90]))

            for a, b in zip(seq_seg[:-1], seq_seg[1:]):
                transition_rows.append({
                    "label_col": label_col,
                    "group": g,
                    "from": a,
                    "to": b,
                })

        print("\nSegment count summary:")
        print(pd.Series(seg_counts).describe())

        trans_df = pd.DataFrame(transition_rows)
        if len(trans_df) > 0:
            trans_counts = (
                trans_df
                .groupby(["from", "to"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )

            print("\nMost common segment transitions:")
            display(trans_counts.head(60))

            trans_counts.to_csv(
                os.path.join(OUT_DIR, f"{label_col}_segment_transition_counts.csv"),
                index=False
            )


# --- CELL 10 (code cell #10) ---

# ================================================================
# CELL 9 - RARE LABEL CHECK AND FINAL LABEL CHOICE
# ================================================================

# Recommended first:
# rq3_compact_process_label = safer for HMM because it has fewer states.
# rq3_process_label = more detailed, but may have sparse states.

SELECTED_RQ3_LABEL_COL = "rq3_compact_process_label"
# SELECTED_RQ3_LABEL_COL = "rq3_process_label"

MIN_LABEL_WINDOWS = 20

counts = df[SELECTED_RQ3_LABEL_COL].value_counts()
rare = set(counts[counts < MIN_LABEL_WINDOWS].index)

df["rq3_final_label"] = df[SELECTED_RQ3_LABEL_COL].where(
    ~df[SELECTED_RQ3_LABEL_COL].isin(rare),
    "other"
)

print("Selected label column:", SELECTED_RQ3_LABEL_COL)
print("MIN_LABEL_WINDOWS:", MIN_LABEL_WINDOWS)
print("Rare labels merged to other:", sorted(rare))

print("\nFinal RQ3 label distribution:")
display(df["rq3_final_label"].value_counts().reset_index().rename(
    columns={"index": "rq3_final_label", "rq3_final_label": "count"}
))

print("\nFinal labels:")
print(sorted(df["rq3_final_label"].unique()))


# --- CELL 11 (code cell #11) ---

# ================================================================
# CELL 10 - SAVE NORMALIZED LABEL FILES
# ================================================================

normalized_path = os.path.join(OUT_DIR, "rq3_normalized_labels_full.csv")
df.to_csv(normalized_path, index=False)

print("Saved full normalized file:")
print(normalized_path)

keep_cols = []

for c in [GROUP_COL, TIME_COL, MAIN_LABEL_COL]:
    if c is not None and c in df.columns and c not in keep_cols:
        keep_cols.append(c)

for c in [
    "raw_label",
    "rq3_process_label",
    "rq3_compact_process_label",
    "rq3_conversation_binary",
    "rq3_final_label",
]:
    if c in df.columns and c not in keep_cols:
        keep_cols.append(c)

compact_path = os.path.join(OUT_DIR, "rq3_normalized_labels_compact.csv")
df[keep_cols].to_csv(compact_path, index=False)

print("\nSaved compact normalized file:")
print(compact_path)

metadata = {
    "main_file": MAIN_FILE,
    "main_label_col": MAIN_LABEL_COL,
    "group_col": GROUP_COL,
    "time_col": TIME_COL,
    "selected_rq3_label_col": SELECTED_RQ3_LABEL_COL,
    "final_label_col": "rq3_final_label",
    "out_dir": OUT_DIR,
}

metadata_path = os.path.join(OUT_DIR, "rq3_label_normalization_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved metadata:")
print(metadata_path)

print("\nDone.")

