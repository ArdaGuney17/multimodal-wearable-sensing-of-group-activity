# --- CELL 1 (code cell #1) ---

# ================================================================
# 0. CONFIGURATION
# ================================================================
import os
import re
import glob
import gc
import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# Mount Google Drive when running in Colab.
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    print("Google Drive mount skipped: not running in Colab.")

# Existing thesis data and model-ready files.
SOURCE_DATA_ROOT = "/content/drive/MyDrive/thesis/data"
RAW_DIR = os.path.join(
    SOURCE_DATA_ROOT,
    "ALL_MODEL_READY_FILES_IDENTITY_FIXED",
)

# Rebuilt outputs are written separately, so existing official files are safe.
OUTPUT_DATA_ROOT = os.path.join(
    SOURCE_DATA_ROOT,
    "FEATURE_GENERATOR_REBUILT",
)

# Existing official tables used only for verification.
REFERENCE_DATA_ROOT = SOURCE_DATA_ROOT

RUN_TASK1 = True
RUN_TASK2 = True
RUN_TASK3 = True


# Authoritative Task 1 label-grid input.
FEATURE_GENERATOR_INPUT_DIR = os.path.join(
    SOURCE_DATA_ROOT,
    "FEATURE_GENERATOR_INPUTS",
)
os.makedirs(FEATURE_GENERATOR_INPUT_DIR, exist_ok=True)

TASK1_LABEL_GRID_PATH = os.path.join(
    FEATURE_GENERATOR_INPUT_DIR,
    "task1_authoritative_5s_label_grid.csv",
)

TASK1_LABEL_GRID_SOURCE_CANDIDATES = [
    os.path.join(
        SOURCE_DATA_ROOT,
        "INTERACTION_BINARY_5S_SPECIALIZED_OE",
        "binary_5s_specialized_oe_merged_all_features.csv",
    ),
    os.path.join(
        SOURCE_DATA_ROOT,
        "INTERACTION_BINARY_5S_ADVANCED_FEATURES",
        "binary_5s_all_sensor_advanced_features.csv",
    ),
]

# The Task 3 token table needs this normalized annotation file.
RQ3_NORMALIZED_LABEL_PATH = os.path.join(
    SOURCE_DATA_ROOT,
    "RQ3_LABEL_NORMALIZATION",
    "rq3_normalized_labels_full.csv",
)

# The historical generic OE generator is unavailable.
# This switch builds a transparent reconstructed generic OE table.
BUILD_RECONSTRUCTED_GENERIC_OE = True

os.makedirs(OUTPUT_DATA_ROOT, exist_ok=True)

print("Raw model-ready folder:", RAW_DIR)
print("Rebuilt output root:", OUTPUT_DATA_ROOT)

if not os.path.isdir(RAW_DIR):
    raise FileNotFoundError(
        "Could not find the model-ready folder:\n"
        f"{RAW_DIR}"
    )


# --- CELL 3 (code cell #2) ---
# ================================================================
# A1. AUTHORITATIVE FIVE-SECOND BINARY WINDOW AND LABEL GRID
# ================================================================
GROUP_COL = "group"
START_COL = "window_start"
END_COL = "window_end"
LABEL_COL = "binary_label"

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 5.0


def group_number(value):
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


# The exact historical window-grid creation code was not retained.
# Preserve the authoritative labels/timing as a small standalone input.
if not os.path.exists(TASK1_LABEL_GRID_PATH):
    source_path = next(
        (
            path
            for path in TASK1_LABEL_GRID_SOURCE_CANDIDATES
            if os.path.exists(path)
        ),
        None,
    )

    if source_path is None:
        raise FileNotFoundError(
            "The authoritative Task 1 label grid is missing.\n"
            f"Expected: {TASK1_LABEL_GRID_PATH}\n\n"
            "Run once while either the official specialized or advanced "
            "Task 1 table is still available. The notebook will export only "
            "the timing, label and provenance columns into a small input CSV."
        )

    source = pd.read_csv(source_path, low_memory=False)

    preferred_grid_columns = [
        "group",
        "window_start",
        "window_end",
        "window_size_s",
        "sensor_coverage_s",
        "interaction_overlap_s",
        "interaction_ratio",
        "binary_label",
        "elapsed_min",
        "label_source_sensor",
        "label_source_path",
    ]

    grid_columns = [
        column
        for column in preferred_grid_columns
        if column in source.columns
    ]

    required = {"group", "window_start", "window_end", "binary_label"}
    if not required.issubset(grid_columns):
        raise ValueError(
            "The selected Task 1 source does not contain the required "
            "group, window_start, window_end and binary_label columns."
        )

    source[grid_columns].to_csv(TASK1_LABEL_GRID_PATH, index=False)
    print("Exported authoritative Task 1 label grid from:")
    print(source_path)
    print("Saved:", TASK1_LABEL_GRID_PATH)

base_windows = pd.read_csv(TASK1_LABEL_GRID_PATH, low_memory=False)
base_windows[LABEL_COL] = base_windows[LABEL_COL].astype(str).str.strip()
base_windows["group_num"] = base_windows[GROUP_COL].map(group_number)

# Fill only metadata that may be absent in a manually supplied grid.
if "window_size_s" not in base_windows.columns:
    base_windows["window_size_s"] = (
        pd.to_numeric(base_windows[END_COL], errors="coerce")
        - pd.to_numeric(base_windows[START_COL], errors="coerce")
    )

if "elapsed_min" not in base_windows.columns:
    midpoint = (
        pd.to_numeric(base_windows[START_COL], errors="coerce")
        + pd.to_numeric(base_windows[END_COL], errors="coerce")
    ) / 2.0
    base_windows["elapsed_min"] = (
        midpoint - midpoint.groupby(base_windows[GROUP_COL]).transform("min")
    ) / 60.0

task1_base_dir = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_BINARY_5S_ADVANCED_FEATURES",
)
task1_parts_dir = os.path.join(task1_base_dir, "_parts")
os.makedirs(task1_parts_dir, exist_ok=True)

LABELS_GRID_PATH = os.path.join(
    task1_parts_dir,
    "task1_window_labels_5s.csv",
)
base_windows.drop(columns=["group_num"], errors="ignore").to_csv(
    LABELS_GRID_PATH,
    index=False,
)

print("Task 1 authoritative label grid:", base_windows.shape)
display(base_windows[LABEL_COL].value_counts())
display(pd.crosstab(base_windows[GROUP_COL], base_windows[LABEL_COL]))


# --- CELL 4 (code cell #3) ---
# ================================================================
# SECTION 0b — advanced statistic expansion (shared by 0c/0d)
# Produces, per base signal x:
#   __mean __std __min __max __range __median __iqr __p10 __p90 __energy __rms
#   __first_half_mean __second_half_mean __half_delta __mad __slope
# ================================================================
def adv_stats(out, name, x, t=None):
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if fin.sum() == 0:
        for s in ["mean","std","min","max","range","median","iqr","p10","p90","energy","rms",
                  "first_half_mean","second_half_mean","half_delta","mad","slope"]:
            out[f"{name}__{s}"] = np.nan
        return
    v = x[fin]
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    out[f"{name}__mean"]   = float(v.mean())
    out[f"{name}__std"]    = float(v.std())
    out[f"{name}__min"]    = float(v.min())
    out[f"{name}__max"]    = float(v.max())
    out[f"{name}__range"]  = float(v.max() - v.min())
    out[f"{name}__median"] = float(q50)
    out[f"{name}__iqr"]    = float(q75 - q25)
    out[f"{name}__p10"]    = float(q10)
    out[f"{name}__p90"]    = float(q90)
    out[f"{name}__energy"] = float(np.sum(v ** 2))
    out[f"{name}__rms"]    = float(np.sqrt(np.mean(v ** 2)))
    h = len(x) // 2
    fh, sh = x[:h], x[h:]
    fh_m = float(np.nanmean(fh)) if np.isfinite(fh).any() else np.nan
    sh_m = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    out[f"{name}__first_half_mean"]  = fh_m
    out[f"{name}__second_half_mean"] = sh_m
    out[f"{name}__half_delta"] = sh_m - fh_m if np.isfinite(fh_m) and np.isfinite(sh_m) else np.nan
    out[f"{name}__mad"] = float(np.median(np.abs(v - q50)))
    if t is not None and fin.sum() > 2:
        tt = np.asarray(t, float)[fin]
        if np.ptp(tt) > 0:
            out[f"{name}__slope"] = float(np.polyfit(tt - tt[0], v, 1)[0])
        else:
            out[f"{name}__slope"] = np.nan
    else:
        out[f"{name}__slope"] = np.nan

def load_raw_file(kind, gnum):
    p = os.path.join(RAW_DIR, f"group_{gnum}_{kind}_model_ready.csv")
    if not os.path.exists(p):
        print(f"  WARNING: missing {p}"); return None
    r = pd.read_csv(p, low_memory=False)
    tcol = "video_time_s" if "video_time_s" in r.columns else "time_s"
    r["t"] = pd.to_numeric(r[tcol], errors="coerce")
    return r.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

def num(r, name, clean=1e6):
    if name in r.columns:
        v = pd.to_numeric(r[name], errors="coerce").to_numpy()
        return np.where(np.abs(v) < clean, v, np.nan)
    return np.full(len(r), np.nan)


# --- CELL 5 (code cell #4) ---

# ================================================================
# A2. TASK 1 OPTI2 FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR
#
# Uses the authoritative Task 1 windows, detects the actual landmark
# column names, and selects the best available model-ready source.
# ================================================================
OPTI2_5S_PATH = os.path.join(
    task1_parts_dir,
    "rebuild_opti2_5s.csv",
)
OPTI2_SOURCE_REPORT_PATH = os.path.join(
    task1_parts_dir,
    "rebuild_opti2_5s_source_report.csv",
)


def task1_standardize_columns(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def task1_find_landmark_mapping(frame):
    lower_to_original = {
        str(column).lower(): column
        for column in frame.columns
    }

    mapping = {}

    alternatives = {
        1: [
            ("landmark1_x", "landmark1_y", "landmark1_z"),
            ("p1_x", "p1_y", "p1_z"),
            ("participant1_x", "participant1_y", "participant1_z"),
            ("person1_x", "person1_y", "person1_z"),
        ],
        2: [
            ("landmark2_x", "landmark2_y", "landmark2_z"),
            ("p2_x", "p2_y", "p2_z"),
            ("participant2_x", "participant2_y", "participant2_z"),
            ("person2_x", "person2_y", "person2_z"),
        ],
        3: [
            ("landmark3_x", "landmark3_y", "landmark3_z"),
            ("p3_x", "p3_y", "p3_z"),
            ("participant3_x", "participant3_y", "participant3_z"),
            ("person3_x", "person3_y", "person3_z"),
        ],
    }

    for participant in (1, 2, 3):
        mapping[participant] = {
            "x": None,
            "y": None,
            "z": None,
        }

        for x_name, y_name, z_name in alternatives[participant]:
            if (
                x_name.lower() in lower_to_original
                and y_name.lower() in lower_to_original
                and z_name.lower() in lower_to_original
            ):
                mapping[participant] = {
                    "x": lower_to_original[x_name.lower()],
                    "y": lower_to_original[y_name.lower()],
                    "z": lower_to_original[z_name.lower()],
                }
                break

    return mapping


def task1_opti_candidate_paths(group):
    candidates = [
        os.path.join(
            RAW_DIR,
            f"group_{group}_optitrack_model_ready.csv",
        ),
        os.path.join(
            SOURCE_DATA_ROOT,
            "ALL_MODEL_READY_FILES",
            f"group_{group}_optitrack_model_ready.csv",
        ),
    ]

    return [
        path
        for path in candidates
        if os.path.exists(path)
    ]


def task1_load_opti_candidate(path):
    frame = pd.read_csv(path, low_memory=False)
    frame = task1_standardize_columns(frame)

    if "video_time_s" in frame.columns:
        time_column = "video_time_s"
    elif "time_s" in frame.columns:
        time_column = "time_s"
    else:
        return None

    frame["t"] = pd.to_numeric(
        frame[time_column],
        errors="coerce",
    )
    frame = (
        frame.dropna(subset=["t"])
        .sort_values("t")
        .reset_index(drop=True)
    )

    mapping = task1_find_landmark_mapping(frame)

    if any(
        mapping[participant][axis] is None
        for participant in (1, 2, 3)
        for axis in "xyz"
    ):
        return None

    return {
        "path": path,
        "frame": frame,
        "time_column": time_column,
        "mapping": mapping,
    }


def task1_opti_source_score(candidate, group_windows):
    if candidate is None:
        return -1

    frame = candidate["frame"]
    mapping = candidate["mapping"]
    times = frame["t"].to_numpy()
    score = 0

    for _, window in group_windows.iterrows():
        start = float(window[START_COL])
        end = float(window[END_COL])

        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, end, side="left")

        sub = frame.iloc[left:right]

        for participant in (1, 2, 3):
            for axis in "xyz":
                values = pd.to_numeric(
                    sub[mapping[participant][axis]],
                    errors="coerce",
                ).to_numpy()

                values = np.where(
                    np.abs(values) < 1e6,
                    values,
                    np.nan,
                )
                score += int(np.isfinite(values).sum())

    return score


def task1_choose_opti_source(group, group_windows):
    candidates = []

    for path in task1_opti_candidate_paths(group):
        candidate = task1_load_opti_candidate(path)

        if candidate is None:
            continue

        candidate["score"] = task1_opti_source_score(
            candidate,
            group_windows,
        )
        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No usable OptiTrack source found for group {group}"
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[0], candidates


if RUN_TASK1:
    rows = []
    source_rows = []

    for group in sorted(
        int(value)
        for value in base_windows["group_num"].dropna().unique()
    ):
        group_windows = (
            base_windows[base_windows["group_num"] == group]
            .sort_values(START_COL)
            .reset_index(drop=True)
        )

        chosen, candidates = task1_choose_opti_source(
            group,
            group_windows,
        )

        frame = chosen["frame"]
        mapping = chosen["mapping"]
        times = frame["t"].to_numpy()

        print(f"\nTask 1 OptiTrack group {group}")
        for candidate in candidates:
            print(
                " ",
                candidate["score"],
                "|",
                candidate["path"],
            )
        print(" Chosen:", chosen["path"])

        positions = {}

        for participant in (1, 2, 3):
            positions[participant] = np.stack(
                [
                    num(
                        frame,
                        mapping[participant][axis],
                    )
                    for axis in "xyz"
                ],
                axis=1,
            )

        dt = np.gradient(times)

        speeds = {
            participant: (
                np.linalg.norm(
                    np.gradient(
                        positions[participant],
                        axis=0,
                    ),
                    axis=1,
                )
                / np.where(dt > 0, dt, np.nan)
            )
            for participant in (1, 2, 3)
        }

        distances = {
            "p1_p2": np.linalg.norm(
                positions[1] - positions[2],
                axis=1,
            ),
            "p1_p3": np.linalg.norm(
                positions[1] - positions[3],
                axis=1,
            ),
            "p2_p3": np.linalg.norm(
                positions[2] - positions[3],
                axis=1,
            ),
        }

        distance_matrix = np.stack(
            list(distances.values()),
            axis=1,
        )

        nearest = np.nanmin(distance_matrix, axis=1)
        farthest = np.nanmax(distance_matrix, axis=1)
        all_pair = np.nanmean(distance_matrix, axis=1)
        pair_spread = np.nanstd(distance_matrix, axis=1)

        participant_stack = np.stack(
            [
                positions[1],
                positions[2],
                positions[3],
            ],
            axis=0,
        )

        centroid = np.nanmean(
            participant_stack,
            axis=0,
        )

        group_spread = np.sqrt(
            np.nanmean(
                np.sum(
                    (
                        participant_stack
                        - centroid[None, :, :]
                    )
                    ** 2,
                    axis=2,
                ),
                axis=0,
            )
        )

        all_speed = np.nanmean(
            np.stack(
                [
                    speeds[1],
                    speeds[2],
                    speeds[3],
                ],
                axis=0,
            ),
            axis=0,
        )

        triangle_area = 0.5 * np.linalg.norm(
            np.cross(
                positions[2] - positions[1],
                positions[3] - positions[1],
            ),
            axis=1,
        )

        source_rows.append(
            {
                "group": group,
                "chosen_path": chosen["path"],
                "score": chosen["score"],
                "p1_x": mapping[1]["x"],
                "p2_x": mapping[2]["x"],
                "p3_x": mapping[3]["x"],
            }
        )

        for _, window in group_windows.iterrows():
            start = float(window[START_COL])
            end = float(window[END_COL])
            mask = (times >= start) & (times < end)

            record = {
                GROUP_COL: window[GROUP_COL],
                START_COL: start,
                END_COL: end,
            }

            if mask.sum() >= 3:
                window_times = times[mask]

                for participant in (1, 2, 3):
                    for axis_index, axis in enumerate("xyz"):
                        adv_stats(
                            record,
                            f"opti2_p{participant}_{axis}",
                            positions[participant][mask, axis_index],
                            window_times,
                        )

                    adv_stats(
                        record,
                        f"opti2_p{participant}_speed",
                        speeds[participant][mask],
                        window_times,
                    )

                for name, values in distances.items():
                    adv_stats(
                        record,
                        f"opti2_{name}_dist",
                        values[mask],
                        window_times,
                    )

                adv_stats(
                    record,
                    "opti2_nearest_pair_dist",
                    nearest[mask],
                    window_times,
                )
                adv_stats(
                    record,
                    "opti2_farthest_pair_dist",
                    farthest[mask],
                    window_times,
                )
                adv_stats(
                    record,
                    "opti2_all_pair_dist",
                    all_pair[mask],
                    window_times,
                )
                adv_stats(
                    record,
                    "opti2_pair_dist_spread",
                    pair_spread[mask],
                    window_times,
                )
                adv_stats(
                    record,
                    "opti2_group_spread",
                    group_spread[mask],
                    window_times,
                )

                for axis_index, axis in enumerate("xyz"):
                    adv_stats(
                        record,
                        f"opti2_centroid_{axis}",
                        centroid[mask, axis_index],
                        window_times,
                    )

                adv_stats(
                    record,
                    "opti2_all_speed",
                    all_speed[mask],
                    window_times,
                )
                adv_stats(
                    record,
                    "opti2_triangle_area",
                    triangle_area[mask],
                    window_times,
                )

                for threshold, tag in [
                    (0.05, "0_05"),
                    (0.10, "0_10"),
                    (0.20, "0_20"),
                ]:
                    count = np.nansum(
                        np.stack(
                            [
                                speeds[participant][mask] > threshold
                                for participant in (1, 2, 3)
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )

                    record[
                        f"opti2_active_speed_count_gt_{tag}__mean"
                    ] = float(np.nanmean(count))

            rows.append(record)

        print(
            f"Task 1 OPTI2 rebuilt: group {group} | "
            f"windows={len(group_windows)}"
        )

    opti2_5s = pd.DataFrame(rows)
    opti2_source_report = pd.DataFrame(source_rows)

    opti2_5s.to_csv(OPTI2_5S_PATH, index=False)
    opti2_source_report.to_csv(
        OPTI2_SOURCE_REPORT_PATH,
        index=False,
    )

    print("Saved:", OPTI2_5S_PATH)
    print("Shape:", opti2_5s.shape)
    display(opti2_source_report)


# --- CELL 6 (code cell #5) ---

# ================================================================
# A3. TASK 1 XSENS2 FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR
#
# Keeps the historical simple 16-statistic feature schema, while
# selecting the best available raw source using the verified ENG7
# sanity-score procedure.
# ================================================================
XSENS2_5S_PATH = os.path.join(
    task1_parts_dir,
    "rebuild_xsens2_5s.csv",
)
XSENS2_SOURCE_REPORT_PATH = os.path.join(
    task1_parts_dir,
    "rebuild_xsens2_5s_source_report.csv",
)


def task1_xsens_candidate_paths(group):
    candidates = [
        os.path.join(
            RAW_DIR,
            f"group_{group}_xsens_model_ready.csv",
        ),
        os.path.join(
            SOURCE_DATA_ROOT,
            "ALL_MODEL_READY_FILES",
            f"group_{group}_xsens_model_ready.csv",
        ),
        os.path.join(
            SOURCE_DATA_ROOT,
            f"group_{group}",
            "xsens",
            "model_ready",
            f"group_{group}_xsens_model_ready.csv",
        ),
    ]

    return [
        path
        for path in candidates
        if os.path.exists(path)
    ]


def task1_load_xsens_candidate(path):
    frame = pd.read_csv(path, low_memory=False)
    frame = task1_standardize_columns(frame)

    if "video_time_s" in frame.columns:
        time_column = "video_time_s"
    elif "time_s" in frame.columns:
        time_column = "time_s"
    else:
        return None

    frame["t"] = pd.to_numeric(
        frame[time_column],
        errors="coerce",
    )

    frame = (
        frame.dropna(subset=["t"])
        .sort_values("t")
        .reset_index(drop=True)
    )

    return {
        "path": path,
        "frame": frame,
        "time_column": time_column,
    }


def task1_sane_xsens_values(values, kind):
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan

    if kind == "acc":
        values[np.abs(values) > 100.0] = np.nan
    elif kind == "gyr":
        values[np.abs(values) > 2000.0] = np.nan
    elif kind == "euler":
        values[np.abs(values) > 360.0] = np.nan

    return values


def task1_xsens_source_score(candidate, group_windows):
    if candidate is None:
        return -1

    frame = candidate["frame"]
    times = frame["t"].to_numpy()
    score = 0

    channels = [
        (
            f"p{participant}_{kind}_{axis}",
            kind,
        )
        for participant in (1, 2, 3)
        for kind in ("acc", "gyr", "euler")
        for axis in "xyz"
        if f"p{participant}_{kind}_{axis}" in frame.columns
    ]

    for _, window in group_windows.iterrows():
        start = float(window[START_COL])
        end = float(window[END_COL])

        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, end, side="left")

        sub = frame.iloc[left:right]

        for column, kind in channels:
            values = pd.to_numeric(
                sub[column],
                errors="coerce",
            ).to_numpy()

            values = task1_sane_xsens_values(
                values,
                kind,
            )
            score += int(np.isfinite(values).sum())

    return score


def task1_choose_xsens_source(group, group_windows):
    candidates = []

    for path in task1_xsens_candidate_paths(group):
        candidate = task1_load_xsens_candidate(path)

        if candidate is None:
            continue

        candidate["score"] = task1_xsens_source_score(
            candidate,
            group_windows,
        )
        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No usable Xsens source found for group {group}"
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[0], candidates


if RUN_TASK1:
    rows = []
    source_rows = []

    for group in sorted(
        int(value)
        for value in base_windows["group_num"].dropna().unique()
    ):
        group_windows = (
            base_windows[base_windows["group_num"] == group]
            .sort_values(START_COL)
            .reset_index(drop=True)
        )

        chosen, candidates = task1_choose_xsens_source(
            group,
            group_windows,
        )

        frame = chosen["frame"]
        times = frame["t"].to_numpy()

        print(f"\nTask 1 Xsens group {group}")
        for candidate in candidates:
            print(
                " ",
                candidate["score"],
                "|",
                candidate["path"],
            )
        print(" Chosen:", chosen["path"])

        channels = {}
        norms = {}
        availability = {}

        for participant in (1, 2, 3):
            for signal_type in ("acc", "gyr", "euler"):
                matrix = np.stack(
                    [
                        num(
                            frame,
                            f"p{participant}_{signal_type}_{axis}",
                            clean=1e6,
                        )
                        for axis in "xyz"
                    ],
                    axis=1,
                )

                for axis_index, axis in enumerate("xyz"):
                    channels[
                        f"p{participant}_{signal_type}_{axis}"
                    ] = matrix[:, axis_index]

                norms[
                    f"xsens2_norm_p{participant}_{signal_type}"
                ] = np.linalg.norm(
                    matrix,
                    axis=1,
                )

            availability[
                f"p{participant}_xsens_available"
            ] = np.isfinite(
                np.stack(
                    [
                        channels[f"p{participant}_acc_{axis}"]
                        for axis in "xyz"
                    ],
                    axis=1,
                )
            ).all(axis=1).astype(float)

        if "xsens_artifact_removed" in frame.columns:
            artifact = num(
                frame,
                "xsens_artifact_removed",
                clean=1e6,
            )
        else:
            artifact = np.zeros(len(frame))

        window_signal_names = (
            list(channels)
            + list(norms)
            + list(availability)
        )

        window_signal_matrix = np.column_stack(
            [
                channels.get(
                    name,
                    norms.get(
                        name,
                        availability.get(name),
                    ),
                )
                for name in window_signal_names
            ]
        )

        source_rows.append(
            {
                "group": group,
                "chosen_path": chosen["path"],
                "score": chosen["score"],
                "time_column": chosen["time_column"],
                "raw_rows": len(frame),
                "windows": len(group_windows),
            }
        )

        for _, window in group_windows.iterrows():
            start = float(window[START_COL])
            end = float(window[END_COL])
            mask = (times >= start) & (times < end)

            record = {
                GROUP_COL: window[GROUP_COL],
                START_COL: start,
                END_COL: end,
            }

            if mask.sum() >= 3:
                window_times = times[mask]

                for name, signal in channels.items():
                    adv_stats(
                        record,
                        f"xsens2__{name}",
                        signal[mask],
                        window_times,
                    )

                for name, signal in norms.items():
                    adv_stats(
                        record,
                        f"xsens2__{name}",
                        signal[mask],
                        window_times,
                    )

                for name, signal in availability.items():
                    adv_stats(
                        record,
                        f"xsens2__{name}",
                        signal[mask],
                        window_times,
                    )

                adv_stats(
                    record,
                    "xsens2__xsens_artifact_removed",
                    artifact[mask],
                    window_times,
                )

                window_values = window_signal_matrix[mask]
                available_by_channel = np.isfinite(
                    window_values
                ).any(axis=0)
                finite_values = window_values[
                    np.isfinite(window_values)
                ]

                record[
                    "xsens2__window_available_channels"
                ] = int(available_by_channel.sum())

                record["xsens2__window_mean_abs_all"] = (
                    float(np.mean(np.abs(finite_values)))
                    if finite_values.size
                    else np.nan
                )

                record["xsens2__window_energy_all"] = (
                    float(np.mean(finite_values ** 2))
                    if finite_values.size
                    else np.nan
                )

            rows.append(record)

        print(
            f"Task 1 XSENS2 rebuilt: group {group} | "
            f"windows={len(group_windows)}"
        )

    xsens2_5s = pd.DataFrame(rows)
    xsens2_source_report = pd.DataFrame(source_rows)

    xsens2_5s.to_csv(XSENS2_5S_PATH, index=False)
    xsens2_source_report.to_csv(
        XSENS2_SOURCE_REPORT_PATH,
        index=False,
    )

    print("Saved:", XSENS2_5S_PATH)
    print("Shape:", xsens2_5s.shape)
    display(xsens2_source_report)


# --- CELL 7 (code cell #6) ---

# ================================================================
# A2. RECONSTRUCTED GENERIC OPENEAREABLE FEATURES
#
# This replaces the missing historical generic OE generator.
# These columns are removed again by the verified SPECIAL_OE cell.
# ================================================================
GENERIC_OE_PATH = os.path.join(
    task1_parts_dir,
    "rebuild_generic_oe_5s.csv",
)

if RUN_TASK1 and BUILD_RECONSTRUCTED_GENERIC_OE:
    rows = []

    # Use the authoritative groups already present in the Task 1 grid.
    # load_raw_file() was defined in the preceding shared-helper cell.
    task1_groups = sorted(
        int(group)
        for group in pd.to_numeric(
            base_windows["group_num"],
            errors="coerce",
        ).dropna().unique()
    )

    for group in task1_groups:
        oe = load_raw_file("openearable", group)

        if oe is None:
            print(f"WARNING: no OpenEarable file for group {group}")
            continue

        group_windows = (
            base_windows[base_windows["group_num"] == group]
            .sort_values(START_COL)
            .reset_index(drop=True)
        )

        raw_channels = [
            column
            for column in oe.columns
            if re.match(
                r"^p[123]_(acc|gyro|mag)_[xyz]$",
                str(column),
            )
        ]

        for column in raw_channels:
            oe[column] = pd.to_numeric(
                oe[column],
                errors="coerce",
            )
            oe[column] = oe[column].where(
                oe[column].abs() < 1e6,
                np.nan,
            )

        t = oe["t"].to_numpy()

        for _, window in group_windows.iterrows():
            ws = float(window[START_COL])
            we = float(window[END_COL])
            mask = (t >= ws) & (t < we)

            record = {
                GROUP_COL: window[GROUP_COL],
                START_COL: ws,
                END_COL: we,
            }

            if mask.sum() >= 3:
                for column in raw_channels:
                    adv_stats(
                        record,
                        f"oe__{column}",
                        oe.loc[mask, column].to_numpy(),
                        t[mask],
                    )

            rows.append(record)

        print(
            f"Generic OE rebuilt: group {group} | "
            f"windows={len(group_windows)} | "
            f"raw_channels={len(raw_channels)}"
        )

        del oe
        gc.collect()

    generic_oe = pd.DataFrame(rows)
    generic_oe.to_csv(GENERIC_OE_PATH, index=False)

else:
    generic_oe = pd.DataFrame(
        columns=[GROUP_COL, START_COL, END_COL]
    )
    generic_oe.to_csv(GENERIC_OE_PATH, index=False)

print("Generic OE reconstruction:", generic_oe.shape)
print("Saved:", GENERIC_OE_PATH)


# --- CELL 8 (code cell #7) ---
# ================================================================
# A3. MERGE RECONSTRUCTED TASK 1 COMPATIBILITY BASE TABLE
# ================================================================
opti_5s = pd.read_csv(OPTI2_5S_PATH)
xsens_5s = pd.read_csv(XSENS2_5S_PATH)
generic_oe = pd.read_csv(GENERIC_OE_PATH)

merge_keys = [GROUP_COL, START_COL, END_COL]

task1_base = base_windows.drop(columns=["group_num"], errors="ignore").copy()

for part in [opti_5s, xsens_5s, generic_oe]:
    if len(part) == 0:
        continue
    task1_base = task1_base.merge(
        part.drop_duplicates(merge_keys),
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

TASK1_ADVANCED_PATH = os.path.join(
    task1_base_dir,
    "binary_5s_all_sensor_advanced_features.csv",
)
task1_base.to_csv(TASK1_ADVANCED_PATH, index=False)

print("Saved Task 1 compatibility base:")
print(TASK1_ADVANCED_PATH)
print("Shape:", task1_base.shape)
print(
    "Note: the old generic oe__ family is reconstructed and may not "
    "match the historical 1,860-column intermediate exactly."
)


# --- CELL 9 (code cell #8) ---
# ================================================================
# SPECIALIZED OE9/OE10-STYLE FEATURES FOR 5s BINARY TASK
#
# This fixes the problem:
#   old binary OE features = generic stats only
#   new binary OE features = specialized head/posture/turn/nod/spectral/jerk/sync features
#
# It:
#   1. Loads saved 5s binary dataset
#   2. Rebuilds specialized OE features for the exact same 5s windows
#   3. Drops the old generic oe__ features
#   4. Merges specialized OE features
#   5. Tests:
#        SPECIAL_OE
#        OPTI2_RELATIVE_ONLY
#        SPECIAL_OE + OPTI2_RELATIVE_ONLY
#        SPECIAL_OE + OPTI2_RELATIVE_ONLY + XSENS2
#
# Output folder:
# /content/drive/MyDrive/thesis/data/INTERACTION_BINARY_5S_SPECIALIZED_OE/
# ================================================================

import os
import re
import glob
import gc
import warnings
import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", None)

# ================================================================
# CONFIG
# ================================================================

DATA_ROOT = OUTPUT_DATA_ROOT

INPUT_DIR = RAW_DIR

OLD_BINARY_PATH = TASK1_ADVANCED_PATH

OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_BINARY_5S_SPECIALIZED_OE")
os.makedirs(OUT_DIR, exist_ok=True)

SPECIAL_OE_PATH = f"{OUT_DIR}/binary_5s_specialized_oe9_oe10_features.csv"
MERGED_SPECIAL_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_merged_all_features.csv"
SUMMARY_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_summary.csv"
PRED_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_predictions.csv"
BEST_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_best_per_condition.csv"
EFFECT_PATH = f"{OUT_DIR}/binary_5s_specialized_oe_effect.csv"

RESAMPLE_HZ = 25

NOD_BAND = (1.0, 3.0)
LOW_MOTION_BAND = (0.2, 1.0)
HIGH_MOTION_BAND = (3.0, 8.0)

MAG_LOW_BAND = (0.2, 1.0)
MAG_MID_BAND = (1.0, 3.0)
MAG_HIGH_BAND = (3.0, 8.0)

PAIRS = [(1, 2), (1, 3), (2, 3)]

EAR_ACC = lambda p: [f"p{p}_acc_{a}" for a in "xyz"]
EAR_GYR = lambda p: [f"p{p}_gyro_{a}" for a in "xyz"]
MAG_COLS = sum([[f"p{p}_mag_{a}" for a in "xyz"] for p in (1, 2, 3)], [])
ALL_OE_COLS = sum([EAR_ACC(p) + EAR_GYR(p) for p in (1, 2, 3)], []) + MAG_COLS

RANDOM_STATE = 42

K_LIST = [40, 80, 120, 200]

FEATURE_SETS_TO_RUN = [
    "SPECIAL_OE",
    "OPTI2_RELATIVE_ONLY",
    "SPECIAL_OE + OPTI2_RELATIVE_ONLY",
    "SPECIAL_OE + OPTI2_RELATIVE_ONLY + XSENS2",
]

TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]


# ================================================================
# LOAD OLD BINARY DATASET
# ================================================================

if not os.path.exists(OLD_BINARY_PATH):
    raise FileNotFoundError(
        f"Could not find old binary dataset:\n{OLD_BINARY_PATH}\n\n"
        "Run the earlier binary feature-generation cell first."
    )

base_df = pd.read_csv(OLD_BINARY_PATH)

print("=" * 100)
print("LOADED OLD 5s BINARY DATASET")
print("=" * 100)
print("Path:", OLD_BINARY_PATH)
print("Shape:", base_df.shape)

display(base_df["binary_label"].value_counts())

print("\nGroup x label counts:")
display(pd.crosstab(base_df["group"], base_df["binary_label"]))


# ================================================================
# IO HELPERS
# ================================================================

def discover_openearable(folder):
    found = {}

    for path in glob.glob(os.path.join(folder, "group_*_openearable_model_ready.csv")):
        m = re.search(r"group_(\d+)_openearable_model_ready", os.path.basename(path))

        if m:
            found[int(m.group(1))] = path

    return dict(sorted(found.items()))


def load_oe(path):
    df = pd.read_csv(path, low_memory=False)

    if "video_time_s" in df.columns:
        time_col = "video_time_s"
    elif "time_s" in df.columns:
        time_col = "time_s"
    else:
        raise ValueError(f"No video_time_s or time_s in {path}")

    df["t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    for c in ALL_OE_COLS:
        if c not in df.columns:
            df[c] = np.nan

        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e6, np.nan)

    return df


# ================================================================
# SIGNAL HELPERS
# ================================================================

def sinterp(grid, t, v):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    m = np.isfinite(t) & np.isfinite(v)

    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)

    out = np.interp(grid, t[m], v[m])
    out[(grid < np.nanmin(t[m])) | (grid > np.nanmax(t[m]))] = np.nan

    return out


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def safe_min(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def safe_range(x):
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).any():
        return np.nan

    return float(np.nanmax(x) - np.nanmin(x))


def safe_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.percentile(x, 75) - np.percentile(x, 25))


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.percentile(x, q))


def mad_diff(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return np.nan

    return float(np.nanmedian(np.abs(np.diff(x))))


def event_count(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) < 2:
        return 0

    return int(np.sum(np.diff(mask.astype(int)) == 1))


def spectral_entropy(sig, fs):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)
    P = np.abs(np.fft.rfft(s)) ** 2
    P = P[1:]

    if P.sum() <= 0:
        return np.nan

    p = P / P.sum()
    p = p[p > 0]

    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def band_ratio(sig, fs, lo, hi):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)

    f = np.fft.rfftfreq(len(s), 1 / fs)
    P = np.abs(np.fft.rfft(s)) ** 2

    total = P[1:].sum()

    if total <= 0:
        return np.nan

    return float(P[(f >= lo) & (f < hi)].sum() / total)


def corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() < 8:
        return np.nan

    a = a[m]
    b = b[m]

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def max_lag_corr(a, b, max_lag_steps):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    best = np.nan

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        c = corr_safe(aa, bb)

        if np.isfinite(c):
            if not np.isfinite(best) or abs(c) > abs(best):
                best = c

    return best


def aggregate_values(row, prefix, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_std"] = np.nan
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_max"] = np.nan
        row[f"{prefix}_range"] = np.nan
        return

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite))
    row[f"{prefix}_min"] = float(np.min(finite))
    row[f"{prefix}_max"] = float(np.max(finite))
    row[f"{prefix}_range"] = float(np.max(finite) - np.min(finite))


def circular_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))


# ================================================================
# SESSION BASELINE
# ================================================================

def session_baseline(oe):
    sess = {}

    for p in (1, 2, 3):
        acc = np.stack(
            [oe[f"p{p}_acc_{a}"].values for a in "xyz"],
            axis=1,
        )

        gyr = np.stack(
            [oe[f"p{p}_gyro_{a}"].values for a in "xyz"],
            axis=1,
        )

        acc_mag = np.linalg.norm(acc, axis=1)
        gyr_mag = np.linalg.norm(gyr, axis=1)

        pitch = np.arctan2(
            acc[:, 0],
            np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2),
        )

        roll = np.arctan2(
            acc[:, 1],
            np.sqrt(acc[:, 0] ** 2 + acc[:, 2] ** 2),
        )

        sess[p] = {
            "acc_mean": safe_mean(acc_mag),
            "acc_std": safe_std(acc_mag),
            "gyr_mean": safe_mean(gyr_mag),
            "gyr_std": safe_std(gyr_mag),
            "pitch_mean": safe_mean(pitch),
            "pitch_std": safe_std(pitch),
            "roll_mean": safe_mean(roll),
            "roll_std": safe_std(roll),
            "gyr_p75": safe_percentile(gyr_mag, 75),
            "gyr_p80": safe_percentile(gyr_mag, 80),
            "gyr_p90": safe_percentile(gyr_mag, 90),
            "acc_p75": safe_percentile(acc_mag, 75),
            "acc_p90": safe_percentile(acc_mag, 90),
        }

    return sess


# ================================================================
# SPECIALIZED OE9 + OE10-LIKE FEATURE EXTRACTION
# ================================================================

def extract_specialized_oe_features(oe, ws, we, sess):
    win_s = float(we - ws)
    n = max(8, int(round(win_s * RESAMPLE_HZ)))
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ

    row = {}

    acc = {}
    gyr = {}
    acc_mag = {}
    gyr_mag = {}
    acc_e = {}
    gyr_e = {}
    pitch = {}
    roll = {}
    vert = {}
    jerk = {}
    angular_jerk = {}

    # ------------------------------------------------------------
    # Per-person acc/gyro signals
    # ------------------------------------------------------------

    for p in (1, 2, 3):
        acc[p] = np.stack(
            [
                sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        gyr[p] = np.stack(
            [
                sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        acc_mag[p] = np.linalg.norm(acc[p], axis=1)
        gyr_mag[p] = np.linalg.norm(gyr[p], axis=1)

        acc_e[p] = (acc_mag[p] - sess[p]["acc_mean"]) / (sess[p]["acc_std"] + 1e-9)
        gyr_e[p] = (gyr_mag[p] - sess[p]["gyr_mean"]) / (sess[p]["gyr_std"] + 1e-9)

        pitch[p] = np.arctan2(
            acc[p][:, 0],
            np.sqrt(acc[p][:, 1] ** 2 + acc[p][:, 2] ** 2),
        )

        roll[p] = np.arctan2(
            acc[p][:, 1],
            np.sqrt(acc[p][:, 0] ** 2 + acc[p][:, 2] ** 2),
        )

        vert[p] = acc[p][:, 2]

        jerk[p] = np.r_[0, np.diff(acc_mag[p])] * fs
        angular_jerk[p] = np.r_[0, np.diff(gyr_mag[p])] * fs

    # ------------------------------------------------------------
    # Old ENG7-style explicit head features
    # ------------------------------------------------------------

    down_thr = -0.35

    down_fracs = []
    pitch_ranges = []
    switch_rates = []
    turn_rates = []
    nod_ratios = []

    for p in (1, 2, 3):
        down = pitch[p] < down_thr

        down_fracs.append(np.nanmean(down))
        pitch_ranges.append(safe_range(pitch[p]))
        switch_rates.append(event_count(down) / win_s)
        turn_rates.append(event_count(gyr_mag[p] > sess[p]["gyr_p80"]) / win_s)
        nod_ratios.append(band_ratio(vert[p], fs, *NOD_BAND))

    row["ear_head_down_fraction"] = safe_mean(down_fracs)
    row["ear_head_pitch_range_mean"] = safe_mean(pitch_ranges)
    row["ear_head_regime_switch_rate"] = safe_mean(switch_rates)
    row["ear_head_turn_event_rate"] = safe_mean(turn_rates)
    row["ear_head_nod_band_ratio"] = safe_mean(nod_ratios)

    head_active_binary = {
        p: (gyr_e[p] > 0).astype(float)
        for p in (1, 2, 3)
    }

    pair_corrs = []

    for a, b in PAIRS:
        c = corr_safe(head_active_binary[a], head_active_binary[b])
        if np.isfinite(c):
            pair_corrs.append(c)

    row["ear_head_activity_alternation"] = (
        float(-np.mean(pair_corrs)) if len(pair_corrs) else np.nan
    )

    # ------------------------------------------------------------
    # Rich OE9 per-person features aggregated across people
    # ------------------------------------------------------------

    per_person_feature_values = {}

    def collect(name, vals):
        per_person_feature_values[name] = vals
        aggregate_values(row, f"oe_{name}", vals)

    collect("acc_energy", [safe_mean(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_std", [safe_std(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_iqr", [safe_iqr(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_range", [safe_range(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_maddiff", [mad_diff(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_entropy", [spectral_entropy(acc_e[p], fs) for p in (1, 2, 3)])
    collect("acc_low_band", [band_ratio(acc_e[p], fs, *LOW_MOTION_BAND) for p in (1, 2, 3)])
    collect("acc_nod_band", [band_ratio(acc_e[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("acc_high_band", [band_ratio(acc_e[p], fs, *HIGH_MOTION_BAND) for p in (1, 2, 3)])

    collect("gyro_energy", [safe_mean(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_std", [safe_std(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_iqr", [safe_iqr(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_range", [safe_range(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_maddiff", [mad_diff(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_entropy", [spectral_entropy(gyr_e[p], fs) for p in (1, 2, 3)])
    collect("gyro_low_band", [band_ratio(gyr_e[p], fs, *LOW_MOTION_BAND) for p in (1, 2, 3)])
    collect("gyro_nod_band", [band_ratio(gyr_e[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("gyro_high_band", [band_ratio(gyr_e[p], fs, *HIGH_MOTION_BAND) for p in (1, 2, 3)])

    collect("pitch_mean", [safe_mean(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_std", [safe_std(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_iqr", [safe_iqr(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_range", [safe_range(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_maddiff", [mad_diff(pitch[p]) for p in (1, 2, 3)])

    collect("roll_mean", [safe_mean(roll[p]) for p in (1, 2, 3)])
    collect("roll_std", [safe_std(roll[p]) for p in (1, 2, 3)])
    collect("roll_iqr", [safe_iqr(roll[p]) for p in (1, 2, 3)])
    collect("roll_range", [safe_range(roll[p]) for p in (1, 2, 3)])
    collect("roll_maddiff", [mad_diff(roll[p]) for p in (1, 2, 3)])

    collect("jerk_abs_mean", [safe_mean(np.abs(jerk[p])) for p in (1, 2, 3)])
    collect("jerk_abs_std", [safe_std(np.abs(jerk[p])) for p in (1, 2, 3)])
    collect("angular_jerk_abs_mean", [safe_mean(np.abs(angular_jerk[p])) for p in (1, 2, 3)])
    collect("angular_jerk_abs_std", [safe_std(np.abs(angular_jerk[p])) for p in (1, 2, 3)])

    collect("down_fraction", [np.nanmean(pitch[p] < down_thr) for p in (1, 2, 3)])
    collect("up_fraction", [np.nanmean(pitch[p] >= down_thr) for p in (1, 2, 3)])
    collect("turn_rate_p75", [event_count(gyr_mag[p] > sess[p]["gyr_p75"]) / win_s for p in (1, 2, 3)])
    collect("turn_rate_p90", [event_count(gyr_mag[p] > sess[p]["gyr_p90"]) / win_s for p in (1, 2, 3)])
    collect("acc_burst_rate_p75", [event_count(acc_mag[p] > sess[p]["acc_p75"]) / win_s for p in (1, 2, 3)])
    collect("acc_burst_rate_p90", [event_count(acc_mag[p] > sess[p]["acc_p90"]) / win_s for p in (1, 2, 3)])

    # Extra explicit names so keyword check finds them
    collect("head_movement_frequency", [event_count(gyr_e[p] > 0) / win_s for p in (1, 2, 3)])
    collect("head_turn_frequency", [event_count(gyr_mag[p] > sess[p]["gyr_p80"]) / win_s for p in (1, 2, 3)])
    collect("head_nod_frequency_band", [band_ratio(vert[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("head_posture_switch_frequency", [event_count(pitch[p] < down_thr) / win_s for p in (1, 2, 3)])

    # ------------------------------------------------------------
    # Active-person-count features
    # ------------------------------------------------------------

    acc_active = np.stack([(acc_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    gyro_active = np.stack([(gyr_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    down_active = np.stack([(pitch[p] < down_thr).astype(int) for p in (1, 2, 3)], axis=0)

    acc_count = acc_active.sum(axis=0)
    gyro_count = gyro_active.sum(axis=0)
    down_count = down_active.sum(axis=0)

    for name, count in [
        ("acc_active_count", acc_count),
        ("gyro_active_count", gyro_count),
        ("down_count", down_count),
    ]:
        row[f"oe_{name}_mean"] = safe_mean(count)
        row[f"oe_{name}_std"] = safe_std(count)
        row[f"oe_{name}_exactly1_frac"] = float(np.mean(count == 1))
        row[f"oe_{name}_atleast2_frac"] = float(np.mean(count >= 2))
        row[f"oe_{name}_all3_frac"] = float(np.mean(count == 3))
        row[f"oe_{name}_switch_rate"] = event_count(np.r_[False, np.diff(count) != 0]) / win_s

    # ------------------------------------------------------------
    # Dominance / asymmetry
    # ------------------------------------------------------------

    for name, vals in per_person_feature_values.items():
        vals = np.asarray(vals, dtype=float)

        if np.isfinite(vals).sum() >= 2:
            row[f"oe_{name}_dominance_gap"] = float(np.nanmax(vals) - np.nanmedian(vals))
            row[f"oe_{name}_asymmetry"] = float(np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-9))
        else:
            row[f"oe_{name}_dominance_gap"] = np.nan
            row[f"oe_{name}_asymmetry"] = np.nan

    # ------------------------------------------------------------
    # Cross-person synchrony and lag
    # ------------------------------------------------------------

    pair_acc_corrs = []
    pair_gyro_corrs = []
    pair_pitch_corrs = []
    pair_acc_lagcorrs = []
    pair_gyro_lagcorrs = []

    max_lag_steps = int(1.0 * fs)

    for a, b in PAIRS:
        pair_acc_corrs.append(corr_safe(acc_e[a], acc_e[b]))
        pair_gyro_corrs.append(corr_safe(gyr_e[a], gyr_e[b]))
        pair_pitch_corrs.append(corr_safe(pitch[a], pitch[b]))

        pair_acc_lagcorrs.append(max_lag_corr(acc_e[a], acc_e[b], max_lag_steps))
        pair_gyro_lagcorrs.append(max_lag_corr(gyr_e[a], gyr_e[b], max_lag_steps))

    aggregate_values(row, "oe_pair_acc_corr", pair_acc_corrs)
    aggregate_values(row, "oe_pair_gyro_corr", pair_gyro_corrs)
    aggregate_values(row, "oe_pair_pitch_corr", pair_pitch_corrs)
    aggregate_values(row, "oe_pair_acc_lagcorr", pair_acc_lagcorrs)
    aggregate_values(row, "oe_pair_gyro_lagcorr", pair_gyro_lagcorrs)

    # ------------------------------------------------------------
    # First-half vs second-half change features
    # ------------------------------------------------------------

    half = len(grid) // 2

    for signal_name, signals in [
        ("acc_e", acc_e),
        ("gyro_e", gyr_e),
        ("pitch", pitch),
    ]:
        deltas = []

        for p in (1, 2, 3):
            x = signals[p]

            if len(x) >= 4:
                deltas.append(safe_mean(x[half:]) - safe_mean(x[:half]))
            else:
                deltas.append(np.nan)

        aggregate_values(row, f"oe_{signal_name}_half_delta", deltas)

    # ------------------------------------------------------------
    # OE10-like magnetometer features, if mag columns exist
    # ------------------------------------------------------------

    mag = {}
    mag_norm = {}
    heading = {}
    heading_raw = {}

    for p in (1, 2, 3):
        mag[p] = np.stack(
            [
                sinterp(grid, oe["t"].values, oe[f"p{p}_mag_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        mag_norm[p] = np.linalg.norm(mag[p], axis=1)
        heading_raw[p] = np.arctan2(mag[p][:, 1], mag[p][:, 0])
        heading[p] = np.unwrap(heading_raw[p])

    has_mag = any(np.isfinite(mag_norm[p]).sum() >= 8 for p in (1, 2, 3))

    if has_mag:
        collect("mag_magnitude_mean", [safe_mean(mag_norm[p]) for p in (1, 2, 3)])
        collect("mag_magnitude_std", [safe_std(mag_norm[p]) for p in (1, 2, 3)])
        collect("mag_magnitude_range", [safe_range(mag_norm[p]) for p in (1, 2, 3)])
        collect("mag_magnitude_iqr", [safe_iqr(mag_norm[p]) for p in (1, 2, 3)])
        collect("mag_magnitude_maddiff", [mad_diff(mag_norm[p]) for p in (1, 2, 3)])
        collect("mag_magnitude_entropy", [spectral_entropy(mag_norm[p], fs) for p in (1, 2, 3)])
        collect("mag_magnitude_low_band", [band_ratio(mag_norm[p], fs, *MAG_LOW_BAND) for p in (1, 2, 3)])
        collect("mag_magnitude_mid_band", [band_ratio(mag_norm[p], fs, *MAG_MID_BAND) for p in (1, 2, 3)])
        collect("mag_magnitude_high_band", [band_ratio(mag_norm[p], fs, *MAG_HIGH_BAND) for p in (1, 2, 3)])

        collect("mag_heading_std", [safe_std(heading[p]) for p in (1, 2, 3)])
        collect("mag_heading_range", [safe_range(heading[p]) for p in (1, 2, 3)])
        collect("mag_heading_maddiff", [mad_diff(heading[p]) for p in (1, 2, 3)])
        collect("mag_heading_turn_frequency", [event_count(np.abs(np.r_[0, np.diff(heading[p])]) > 0.05) / win_s for p in (1, 2, 3)])
        collect("mag_heading_low_band", [band_ratio(heading[p], fs, *MAG_LOW_BAND) for p in (1, 2, 3)])
        collect("mag_heading_mid_band", [band_ratio(heading[p], fs, *MAG_MID_BAND) for p in (1, 2, 3)])
        collect("mag_heading_high_band", [band_ratio(heading[p], fs, *MAG_HIGH_BAND) for p in (1, 2, 3)])

        pair_mag_corrs = []
        pair_heading_corrs = []
        pair_heading_diff_std = []
        pair_heading_diff_maddiff = []

        for a, b in PAIRS:
            pair_mag_corrs.append(corr_safe(mag_norm[a], mag_norm[b]))
            pair_heading_corrs.append(corr_safe(heading[a], heading[b]))

            hd = circular_diff(heading_raw[a], heading_raw[b])
            pair_heading_diff_std.append(safe_std(hd))
            pair_heading_diff_maddiff.append(mad_diff(hd))

        aggregate_values(row, "mag_pair_magnitude_corr", pair_mag_corrs)
        aggregate_values(row, "mag_pair_heading_corr", pair_heading_corrs)
        aggregate_values(row, "mag_pair_heading_diff_std", pair_heading_diff_std)
        aggregate_values(row, "mag_pair_heading_diff_maddiff", pair_heading_diff_maddiff)

    return row


# ================================================================
# BUILD SPECIALIZED OE FEATURES FOR EXACT BINARY WINDOWS
# ================================================================

files = discover_openearable(INPUT_DIR)

print("\n" + "=" * 100)
print("BUILDING SPECIALIZED OE FEATURES FOR 5s BINARY WINDOWS")
print("=" * 100)
print("Found OE files:", files)

rows = []

for group in sorted(base_df["group"].unique()):
    group = int(group)

    if group not in files:
        print(f"WARNING: no OpenEarable file for group {group}, skipping.")
        continue

    print("\nGroup", group, "|", os.path.basename(files[group]))

    oe = load_oe(files[group])
    sess = session_baseline(oe)

    gw = base_df[base_df["group"] == group][["group", "window_start", "window_end"]].copy()
    gw = gw.sort_values("window_start").reset_index(drop=True)

    print("Windows:", len(gw))

    for _, w in gw.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        row = extract_specialized_oe_features(oe, ws, we, sess)
        row["group"] = group
        row["window_start"] = ws
        row["window_end"] = we

        rows.append(row)

    del oe
    gc.collect()

special_oe_df = pd.DataFrame(rows)

special_oe_df.to_csv(SPECIAL_OE_PATH, index=False)

print("\n" + "=" * 100)
print("SAVED SPECIALIZED OE FEATURES")
print("=" * 100)
print(SPECIAL_OE_PATH)
print("Shape:", special_oe_df.shape)

special_cols = [
    c for c in special_oe_df.columns
    if c not in ["group", "window_start", "window_end"]
]

print("Specialized OE feature count:", len(special_cols))

keyword_cols = [
    c for c in special_cols
    if any(k in c.lower() for k in [
        "head", "movement", "freq", "frequency", "turn", "nod",
        "spectral", "entropy", "jerk", "sync", "lag", "mag"
    ])
]

print("Specialized keyword-matching columns:", len(keyword_cols))
print("\nFirst 100 keyword columns:")
for c in keyword_cols[:100]:
    print(c)


# ================================================================
# MERGE: DROP OLD GENERIC OE__ AND ADD SPECIALIZED OE
# ================================================================

generic_oe_cols = [c for c in base_df.columns if c.startswith("oe__")]

print("\nDropping old generic OE columns:", len(generic_oe_cols))

df = base_df.drop(columns=generic_oe_cols)

df = df.merge(
    special_oe_df,
    on=["group", "window_start", "window_end"],
    how="left",
)

df.to_csv(MERGED_SPECIAL_PATH, index=False)

print("\n" + "=" * 100)
print("MERGED SPECIALIZED OE DATASET")
print("=" * 100)
print(MERGED_SPECIAL_PATH)
print("Shape:", df.shape)

display(df["binary_label"].value_counts())


# --- CELL 11 (code cell #9) ---
# ============================================================
# BUILD INTERACTION_ENG3
# Successful ENG window grid + old ENG features + Xsens hand features
# No head features.
# Xsens does NOT define the window range.
# ============================================================

import os, glob, re, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

INPUT_DIR = RAW_DIR
OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3")

WINDOW_S, STRIDE_S, RESAMPLE_T = 5.0, 5.0, 64

GROUP_TIERS = [
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group",
]

ACC = [f"p{p}_acc_{a}" for p in (1, 2, 3) for a in "xyz"]
GYR = [f"p{p}_gyro_{a}" for p in (1, 2, 3) for a in "xyz"]
POS = [f"Participant{p}_{a}" for p in (1, 2, 3) for a in "xyz"]

XS_COLS = [f"p{p}_{s}_{a}" for p in (1, 2, 3) for s in ("acc", "gyr", "euler") for a in "xyz"]

os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Discover groups
# ------------------------------------------------------------

def discover(folder):
    groups = {}

    for path in glob.glob(os.path.join(folder, "group_*_model_ready.csv")):
        m = re.search(r"group_(\d+)_([a-z]+)_model_ready", os.path.basename(path))
        if m:
            g = int(m.group(1))
            sensor = m.group(2)
            groups.setdefault(g, {})[sensor] = path

    # ENG3 needs all three files available,
    # but window range will still be based only on OE + OptiTrack.
    groups = {
        g: paths
        for g, paths in sorted(groups.items())
        if all(s in paths for s in ("openearable", "optitrack", "xsens"))
    }

    return groups


# ------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------

def load(path, timecol, cols):
    df = pd.read_csv(path, low_memory=False)

    df["t"] = pd.to_numeric(df[timecol], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e6, np.nan)

    return df


def sinterp(grid, t, v):
    m = np.isfinite(v)
    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, t[m], v[m])


# ------------------------------------------------------------
# Xsens offset helper, same idea as ENG2
# ------------------------------------------------------------

def xoff(oe, xs):
    # If labels are not available, do not offset.
    if "label_Whole_Group" not in oe.columns or "label_Whole_Group" not in xs.columns:
        return 0.0

    lo = max(oe["t"].min(), 5)
    hi = oe["t"].max() - 5

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    grid = np.arange(lo, hi, 0.2)
    if len(grid) == 0:
        return 0.0

    def sample_labels(df, times):
        idx = np.clip(np.searchsorted(df["t"].values, times), 0, len(df) - 1)
        return df["label_Whole_Group"].fillna("NONE").astype(str).values[idx]

    ref = sample_labels(oe, grid)

    best_offset = 0.0
    best_score = -1.0

    for off in np.arange(0, 220, 0.5):
        score = np.mean(ref == sample_labels(xs, grid + off))
        if score > best_score:
            best_score = score
            best_offset = off

    return best_offset


# ------------------------------------------------------------
# Signal helpers
# ------------------------------------------------------------

def spec(sig, fs):
    sig = sig[np.isfinite(sig)]

    if len(sig) < 8:
        return np.nan, np.nan

    sig = sig - sig.mean()
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    P = np.abs(np.fft.rfft(sig)) ** 2

    if len(P) < 2 or P[1:].sum() <= 0:
        return np.nan, np.nan

    dom = f[1:][np.argmax(P[1:])]
    power = P[1:].sum()

    return float(dom), float(power)


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.mean(x))


def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.max(x))


# ------------------------------------------------------------
# OLD ENG features: exactly the successful ENG logic
# ------------------------------------------------------------

def old_eng_features_and_tensor(oe, ot, ws, we):
    grid = np.linspace(ws, we, RESAMPLE_T)
    fs = (RESAMPLE_T - 1) / (we - ws)

    # OptiTrack positions
    P = {
        p: np.stack([
            sinterp(grid, ot["t"].values, ot[f"Participant{p}_{a}"].values)
            for a in "xyz"
        ], axis=1)
        for p in (1, 2, 3)
    }

    # OpenEarable acceleration and gyro
    A = {
        p: np.stack([
            sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values)
            for a in "xyz"
        ], axis=1)
        for p in (1, 2, 3)
    }

    Gy = {
        p: np.stack([
            sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values)
            for a in "xyz"
        ], axis=1)
        for p in (1, 2, 3)
    }

    # Pairwise distances
    d = {
        (a, b): np.linalg.norm(P[a] - P[b], axis=1)
        for a, b in [(1, 2), (1, 3), (2, 3)]
    }

    D = np.stack(list(d.values()), axis=1)
    Dsort = np.sort(D, axis=1)

    # Speeds
    spd = {
        p: np.linalg.norm(np.gradient(P[p], axis=0), axis=1) * fs
        for p in (1, 2, 3)
    }

    cen = (P[1] + P[2] + P[3]) / 3
    cen_spd = np.linalg.norm(np.gradient(cen, axis=0), axis=1) * fs

    # Motion energy
    amag = {
        p: np.linalg.norm(A[p], axis=1)
        for p in (1, 2, 3)
    }

    gmag = {
        p: np.linalg.norm(Gy[p], axis=1)
        for p in (1, 2, 3)
    }

    aE = np.sort([np.nanstd(amag[p]) for p in (1, 2, 3)])
    gE = np.sort([np.nanstd(gmag[p]) for p in (1, 2, 3)])
    spdm = np.sort([np.nanmean(spd[p]) for p in (1, 2, 3)])

    def corr(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)

        if m.sum() < 3:
            return np.nan
        if np.nanstd(x[m]) < 1e-6 or np.nanstd(y[m]) < 1e-6:
            return 0.0

        return float(np.corrcoef(x[m], y[m])[0, 1])

    coord = np.nanmean([
        corr(amag[a], amag[b])
        for a, b in [(1, 2), (1, 3), (2, 3)]
    ])

    row = {
        "dist_close_mean": np.nanmean(Dsort[:, 0]),
        "dist_close_min":  np.nanmin(Dsort[:, 0]),
        "dist_mid_mean":   np.nanmean(Dsort[:, 1]),
        "dist_far_mean":   np.nanmean(Dsort[:, 2]),
        "dist_disp_mean":  np.nanmean(D),
        "dist_disp_std":   np.nanstd(D),

        "speed_min":       spdm[0],
        "speed_mid":       spdm[1],
        "speed_max":       spdm[2],
        "centroid_speed":  np.nanmean(cen_spd),

        "accE_min":        aE[0],
        "accE_mid":        aE[1],
        "accE_max":        aE[2],

        "gyrE_min":        gE[0],
        "gyrE_mid":        gE[1],
        "gyrE_max":        gE[2],

        "move_coord":      coord,
    }

    accmag_mean = np.nanmean(np.stack([amag[p] for p in (1, 2, 3)], axis=0), axis=0)
    accmag_max = np.nanmax(np.stack([amag[p] for p in (1, 2, 3)], axis=0), axis=0)
    gyromag_mean = np.nanmean(np.stack([gmag[p] for p in (1, 2, 3)], axis=0), axis=0)

    tensor = np.stack([
        Dsort[:, 0],
        Dsort[:, 1],
        Dsort[:, 2],
        cen_spd,
        accmag_mean,
        accmag_max,
        gyromag_mean,
    ], axis=1).astype(np.float32)

    return row, tensor


# ------------------------------------------------------------
# XSENS HAND FEATURES ONLY
# ------------------------------------------------------------

def xsens_hand_features(xs, ws, we):
    x = xs.iloc[
        np.searchsorted(xs["t"].values, ws):
        np.searchsorted(xs["t"].values, we)
    ]

    fx = max(len(x) / WINDOW_S, 1)
    grid = np.linspace(ws, we, RESAMPLE_T)

    xdom = []
    xpow = []
    ovar = []
    hmag = []

    for i in (1, 2, 3):
        acc_cols = [f"p{i}_acc_{a}" for a in "xyz"]
        eul_cols = [f"p{i}_euler_{a}" for a in "xyz"]

        acc = x[acc_cols].values if all(c in x.columns for c in acc_cols) else np.empty((0, 3))
        eul = x[eul_cols].values if all(c in x.columns for c in eul_cols) else np.empty((0, 3))

        if len(acc) >= 2:
            mag = np.linalg.norm(acc, axis=1)

            dom, power = spec(mag, fx)
            xdom.append(dom)
            xpow.append(power)

            if len(eul) >= 2:
                ovar.append(np.nanmean(np.nanstd(eul, axis=0)))

            valid = np.isfinite(mag) & np.isfinite(x["t"].values)
            if valid.sum() >= 2:
                hmag.append(np.interp(grid, x["t"].values[valid], mag[valid]))

    row = {
        "hand_freq_mean":  safe_mean(xdom),
        "hand_freq_max":   safe_max(xdom),
        "hand_power_mean": np.log1p(safe_mean(xpow)) if np.isfinite(safe_mean(xpow)) else np.nan,
        "hand_orient_var": safe_mean(ovar),
    }

    if len(hmag) == 3:
        def cr(a, b):
            if np.nanstd(a) < 1e-6 or np.nanstd(b) < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        row["hand_coord"] = np.nanmean([
            cr(hmag[0], hmag[1]),
            cr(hmag[0], hmag[2]),
            cr(hmag[1], hmag[2]),
        ])
    else:
        row["hand_coord"] = np.nan

    row["xsens_n_samples"] = len(x)
    row["xsens_available"] = 1 if len(x) >= 8 else 0

    return row


# ------------------------------------------------------------
# Build ENG3
# ------------------------------------------------------------

def build_eng3():
    rows = []
    tensors = []
    Y = []
    G = []

    group_counts = {}

    for group, paths in groups.items():
        print(f"\nBuilding group {group}")

        oe = load(paths["openearable"], "video_time_s", ACC + GYR)
        ot = load(paths["optitrack"], "video_time_s", POS)
        xs = load(paths["xsens"], "time_s", XS_COLS)

        # Align Xsens to video time as in ENG2.
        offset = xoff(oe, xs)
        xs["t"] = xs["t"] - offset

        print(f"  Xsens offset used: {offset:.2f}s")

        # IMPORTANT:
        # Same as original ENG.
        # Window range uses ONLY OpenEarable + OptiTrack.
        # Xsens does NOT shrink the interval.
        lo = max(oe["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), ot["t"].max())

        rt = oe["t"].values

        inter = np.zeros(len(oe), dtype=bool)

        for col in GROUP_TIERS:
            if col in oe.columns:
                inter |= (
                    oe[col].notna().values
                    & (oe[col].astype(str).str.strip() != "").values
                )

        n_before = len(Y)

        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S

            m = (rt >= ws) & (rt < we)
            if m.sum() == 0:
                continue

            label = "interaction" if inter[m].mean() >= 0.5 else "non_interaction"

            old_row, tensor = old_eng_features_and_tensor(oe, ot, ws, we)
            hand_row = xsens_hand_features(xs, ws, we)

            row = {}
            row.update(old_row)
            row.update(hand_row)

            row["group"] = group
            row["window_start"] = ws
            row["window_end"] = we

            rows.append(row)
            tensors.append(tensor)
            Y.append(label)
            G.append(group)

        group_counts[group] = len(Y) - n_before
        print(f"  windows: {group_counts[group]}")

        del oe, ot, xs
        gc.collect()

    F = pd.DataFrame(rows)
    F["label"] = Y

    X = np.stack(tensors)
    Y = np.array(Y)
    G = np.array(G)

    return F, X, Y, G, group_counts


groups = discover(INPUT_DIR)
print("Groups:", list(groups.keys()))

F3, X3, Y3, G3, group_counts = build_eng3()

# Save
F3.to_csv(f"{OUT_DIR}/interaction_eng3_features.csv", index=False)
np.savez_compressed(
    f"{OUT_DIR}/interaction_eng3_tensors.npz",
    X=X3,
    y=Y3,
    groups=G3
)

print("\n" + "=" * 80)
print("ENG3 SAVED")
print("=" * 80)
print("Feature file:", f"{OUT_DIR}/interaction_eng3_features.csv")
print("Tensor file:", f"{OUT_DIR}/interaction_eng3_tensors.npz")
print("Shape:", F3.shape)
print("Tensor shape:", X3.shape)
print("Label counts:", pd.Series(Y3).value_counts().to_dict())
print("Group counts:", group_counts)

print("\nFeatures:")
for c in F3.columns:
    if c not in ["group", "label", "window_start", "window_end"]:
        print(" -", c)


# --- CELL 12 (code cell #10) ---
# ============================================================
# RECOGNITION TASK — LABEL INVENTORY BEFORE GROUPING
# Goal:
#   Look at all pairwise + whole-group labels first.
#   Do NOT create non_interaction recognition class yet.
#   Decide grouping after seeing real labels.
# ============================================================

import os, glob, re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

INPUT_DIR = RAW_DIR
ENG3_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3", "interaction_eng3_features.csv")
OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3")

GROUP_TIERS = [
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group",
]

def discover(folder):
    groups = {}

    for path in glob.glob(os.path.join(folder, "group_*_model_ready.csv")):
        m = re.search(r"group_(\d+)_([a-z]+)_model_ready", os.path.basename(path))
        if m:
            g = int(m.group(1))
            sensor = m.group(2)
            groups.setdefault(g, {})[sensor] = path

    groups = {
        g: paths
        for g, paths in sorted(groups.items())
        if "openearable" in paths
    }

    return groups


def clean_label_value(v):
    if pd.isna(v):
        return None

    s = str(v).strip()

    if s == "" or s.lower() in ["nan", "none", "null"]:
        return None

    return s


def normalize_label(s):
    if s is None:
        return None

    s = str(s).strip().lower()
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)

    # common typo fixes, only for analysis
    s = s.replace("mering", "merging")
    s = s.replace("synchornizaion", "synchronization")
    s = s.replace("syncornaziton", "synchronization")
    s = s.replace("erarble", "earable")

    return s


def is_technical_label(label):
    lab = normalize_label(label)

    if lab is None:
        return False

    technical_keywords = [
        "sync",
        "synchronization",
        "earable",
        "drop",
        "dropping",
        "calibration",
    ]

    return any(k in lab for k in technical_keywords)


def load_openearable_labels(path):
    df = pd.read_csv(path, low_memory=False)

    df["t"] = pd.to_numeric(df["video_time_s"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    for col in GROUP_TIERS:
        if col not in df.columns:
            df[col] = np.nan

    return df[["t"] + GROUP_TIERS]


groups_paths = discover(INPUT_DIR)
eng3 = pd.read_csv(ENG3_PATH)

all_label_rows = []
window_label_rows = []

# ------------------------------------------------------------
# Part A — sample-level / duration-like inventory
# ------------------------------------------------------------

for group, paths in groups_paths.items():
    print(f"Reading group {group}")

    oe = load_openearable_labels(paths["openearable"])

    # Estimate sampling interval for approximate duration
    dt = np.nanmedian(np.diff(oe["t"].values))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1 / 50

    for tier in GROUP_TIERS:
        vals = oe[tier].apply(clean_label_value)

        for raw_label, count in vals.dropna().value_counts().items():
            all_label_rows.append({
                "group": group,
                "tier": tier,
                "tier_type": "whole_group" if tier == "label_Whole_Group" else "pairwise",
                "raw_label": raw_label,
                "normalized_label": normalize_label(raw_label),
                "is_technical_or_sync": is_technical_label(raw_label),
                "sample_count": int(count),
                "approx_duration_s": float(count * dt),
            })


label_inventory = pd.DataFrame(all_label_rows)

# ------------------------------------------------------------
# Part B — window-level inventory using ENG3 interaction windows only
# This tells us which labels appear in the actual recognition candidates.
# ------------------------------------------------------------

for group in sorted(eng3["group"].unique()):
    oe = load_openearable_labels(groups_paths[group]["openearable"])
    t = oe["t"].values

    group_windows = eng3[
        (eng3["group"] == group)
        & (eng3["label"] == "interaction")
    ].copy()

    for _, win in group_windows.iterrows():
        ws = float(win["window_start"])
        we = float(win["window_end"])

        m = (t >= ws) & (t < we)

        if m.sum() == 0:
            continue

        raw_counter = Counter()
        tier_counter = Counter()

        for tier in GROUP_TIERS:
            vals = oe.loc[m, tier].apply(clean_label_value).dropna()

            for raw_label in vals:
                raw_counter[raw_label] += 1
                tier_counter[tier] += 1

        if len(raw_counter) == 0:
            continue

        dominant_raw_label = raw_counter.most_common(1)[0][0]
        dominant_count = raw_counter.most_common(1)[0][1]

        window_label_rows.append({
            "group": group,
            "window_start": ws,
            "window_end": we,
            "binary_label": win["label"],
            "dominant_raw_label": dominant_raw_label,
            "dominant_normalized_label": normalize_label(dominant_raw_label),
            "dominant_is_technical_or_sync": is_technical_label(dominant_raw_label),
            "dominant_fraction_in_window": dominant_count / max(m.sum(), 1),
            "all_raw_labels_in_window": " | ".join(sorted(raw_counter.keys())),
            "source_tiers_in_window": " | ".join(sorted(tier_counter.keys())),
            "raw_label_counts": dict(raw_counter),
        })


window_label_inventory = pd.DataFrame(window_label_rows)

# ------------------------------------------------------------
# Summary tables
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ALL RAW LABELS FROM PAIRWISE + WHOLE_GROUP TIERS")
print("=" * 80)

raw_summary = (
    label_inventory
    .groupby(["normalized_label", "raw_label", "is_technical_or_sync"])
    .agg(
        sample_count=("sample_count", "sum"),
        approx_duration_s=("approx_duration_s", "sum"),
        groups_present=("group", lambda x: sorted(set(x))),
        tiers_present=("tier", lambda x: sorted(set(x))),
    )
    .reset_index()
    .sort_values("sample_count", ascending=False)
)

display(raw_summary)

print("\n" + "=" * 80)
print("NORMALIZED LABEL SUMMARY")
print("=" * 80)

norm_summary = (
    label_inventory
    .groupby(["normalized_label", "is_technical_or_sync"])
    .agg(
        sample_count=("sample_count", "sum"),
        approx_duration_s=("approx_duration_s", "sum"),
        n_raw_variants=("raw_label", "nunique"),
        raw_variants=("raw_label", lambda x: " | ".join(sorted(set(x)))),
        groups_present=("group", lambda x: sorted(set(x))),
        tiers_present=("tier", lambda x: sorted(set(x))),
    )
    .reset_index()
    .sort_values("sample_count", ascending=False)
)

display(norm_summary)

print("\n" + "=" * 80)
print("DOMINANT LABELS IN ENG3 INTERACTION WINDOWS")
print("=" * 80)

window_summary = (
    window_label_inventory
    .groupby(["dominant_normalized_label", "dominant_raw_label", "dominant_is_technical_or_sync"])
    .agg(
        window_count=("dominant_raw_label", "count"),
        mean_dominant_fraction=("dominant_fraction_in_window", "mean"),
        groups_present=("group", lambda x: sorted(set(x))),
        source_tiers=("source_tiers_in_window", lambda x: " | ".join(sorted(set(x)))),
    )
    .reset_index()
    .sort_values("window_count", ascending=False)
)

display(window_summary)

print("\n" + "=" * 80)
print("DOMINANT LABELS BY GROUP")
print("=" * 80)

display(
    pd.crosstab(
        window_label_inventory["group"],
        window_label_inventory["dominant_normalized_label"]
    )
)

print("\n" + "=" * 80)
print("TECHNICAL / SYNC LABELS")
print("=" * 80)

technical_labels = raw_summary[raw_summary["is_technical_or_sync"] == True]

if len(technical_labels) == 0:
    print("No technical/sync labels found.")
else:
    display(technical_labels)

print("\n" + "=" * 80)
print("NON-TECHNICAL LABELS ONLY")
print("=" * 80)

nontechnical_window_summary = window_summary[
    window_summary["dominant_is_technical_or_sync"] == False
]

display(nontechnical_window_summary)

# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

label_inventory_path = f"{OUT_DIR}/recognition_all_pairwise_wholegroup_raw_label_inventory.csv"
raw_summary_path = f"{OUT_DIR}/recognition_raw_label_summary.csv"
norm_summary_path = f"{OUT_DIR}/recognition_normalized_label_summary.csv"
window_inventory_path = f"{OUT_DIR}/recognition_interaction_window_label_inventory.csv"
window_summary_path = f"{OUT_DIR}/recognition_interaction_window_label_summary.csv"

label_inventory.to_csv(label_inventory_path, index=False)
raw_summary.to_csv(raw_summary_path, index=False)
norm_summary.to_csv(norm_summary_path, index=False)
window_label_inventory.to_csv(window_inventory_path, index=False)
window_summary.to_csv(window_summary_path, index=False)

print("\nSaved:")
print(label_inventory_path)
print(raw_summary_path)
print(norm_summary_path)
print(window_inventory_path)
print(window_summary_path)


# --- CELL 13 (code cell #11) ---
# ============================================================
# RECOGNITION TASK — CREATE FINAL 5-CLASS INTERACTION DATASET
#
# Goal:
#   Only use interaction windows.
#   Do NOT include non_interaction as a class.
#   Group raw pairwise + whole-group labels into 5 activity classes:
#       co_building
#       co_merging
#       co_inspection
#       conversation
#       object_transport
#
# Inputs expected from previous inventory step:
#   recognition_interaction_window_label_inventory.csv
#   recognition_raw_label_summary.csv
#
# Output:
#   eng3_recognition_5class_interaction_only_features.csv
#   eng3_recognition_5class_full_audit.csv
#   eng3_recognition_5class_label_mapping_audit.csv
# ============================================================

import os
import re
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3")

ENG3_PATH = f"{OUT_DIR}/interaction_eng3_features.csv"
WINDOW_LABEL_INVENTORY_PATH = f"{OUT_DIR}/recognition_interaction_window_label_inventory.csv"
RAW_LABEL_SUMMARY_PATH = f"{OUT_DIR}/recognition_raw_label_summary.csv"

eng3 = pd.read_csv(ENG3_PATH)
win_labels = pd.read_csv(WINDOW_LABEL_INVENTORY_PATH)

if os.path.exists(RAW_LABEL_SUMMARY_PATH):
    raw_summary = pd.read_csv(RAW_LABEL_SUMMARY_PATH)
else:
    raw_summary = None


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

def normalize_label(s):
    if pd.isna(s):
        return None

    s = str(s).strip().lower()
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)

    # typo normalization
    s = s.replace("mering", "merging")
    s = s.replace("handiver", "handover")
    s = s.replace("synchornizaion", "synchronization")
    s = s.replace("syncornaziton", "synchronization")
    s = s.replace("erarble", "earable")

    return s


def is_technical_or_sync(label):
    lab = normalize_label(label)

    if lab is None:
        return False

    technical_keywords = [
        "sync",
        "synchronization",
        "earable",
        "calibration",
        "drop",
        "clap_synchronization",
    ]

    return any(k in lab for k in technical_keywords)


# ------------------------------------------------------------
# Final 5-class mapping
# ------------------------------------------------------------

def map_to_5class(label):
    """
    Map raw/dominant label to final recognition class.

    Final classes:
        co_building
        co_merging
        co_inspection
        conversation
        object_transport

    Important:
        - non_interaction is NOT used.
        - sync/calibration labels are ignored.
        - mixed labels are mapped by priority:
              collaborative physical activity
              inspection
              object transport
              conversation
        - object_handover is included inside broader object_transport.
    """

    lab = normalize_label(label)

    if lab is None:
        return "ignore", "missing_label"

    # Ignore technical labels
    if is_technical_or_sync(lab):
        return "ignore", "technical_or_synchronization_label"

    # Ignore this because it is not a stable group activity class
    if lab == "starting_individual_build":
        return "ignore", "starting_marker_not_activity_class"

    # --------------------------------------------------------
    # 1. Co-building
    # --------------------------------------------------------
    if (
        "co_building" in lab
        or "building_subpiece" in lab
        or "building_piece" in lab
        or "building" in lab
        or "assembling" in lab
        or "assembly" in lab
    ):
        return "co_building", "building_or_assembly_label"

    # --------------------------------------------------------
    # 2. Co-merging
    # --------------------------------------------------------
    if (
        "co_merging" in lab
        or "merging" in lab
        or "merge" in lab
    ):
        return "co_merging", "merging_label"

    # --------------------------------------------------------
    # 3. Co-inspection
    # Includes inspecting pieces, inspecting target image,
    # and matching pieces to/with target image.
    # --------------------------------------------------------
    if (
        "co_inspecting" in lab
        or "co_inspection" in lab
        or "inspecting" in lab
        or "inspection" in lab
        or "matching_pieces_to_target_image" in lab
        or "matching_pieces_with_target_image" in lab
        or "matching_piece_to_target_image" in lab
        or "matching_piece_with_target_image" in lab
        or ("matching" in lab and "image" in lab)
        or ("target_image" in lab and "matching" in lab)
    ):
        return "co_inspection", "inspection_or_target_image_matching_label"

    # --------------------------------------------------------
    # 4. Object transport
    # This is broader than object_handover.
    # It includes handover, moving, carrying, placing,
    # delivering, presenting, tray/piece transport.
    # --------------------------------------------------------
    if (
        "object_handover" in lab
        or "handover" in lab
        or "moving_pieces" in lab
        or "placing_subpiece" in lab
        or "placing_piece" in lab
        or "carrying_tray" in lab
        or "tray" in lab
        or "traveling_between_units" in lab
        or "approaching_to" in lab
        or "delivering_piece" in lab
        or "delivering_target_image" in lab
        or "presenting_piece" in lab
        or "presenting_target_image" in lab
        or "piece_presentation" in lab
        or "searching_for_piece" in lab
    ):
        return "object_transport", "object_or_material_transport_label"

    # --------------------------------------------------------
    # 5. Conversation
    # --------------------------------------------------------
    if (
        "convo" in lab
        or "conversation" in lab
        or "discussion" in lab
        or "talking" in lab
    ):
        return "conversation", "conversation_label"

    # Anything else is too rare/ambiguous for the main recognizer
    return "ignore", "rare_or_unmapped_label"


# ------------------------------------------------------------
# Print initial label inventory information
# ------------------------------------------------------------

print("=" * 80)
print("INITIAL LABEL INVENTORY")
print("=" * 80)

if raw_summary is not None:
    n_raw = raw_summary["raw_label"].nunique()
    n_norm = raw_summary["normalized_label"].nunique()

    raw_summary_tmp = raw_summary.copy()
    mapped = raw_summary_tmp["raw_label"].apply(map_to_5class)
    raw_summary_tmp["mapped_class"] = [x[0] for x in mapped]
    raw_summary_tmp["mapping_reason"] = [x[1] for x in mapped]

    print("Raw pairwise/whole-group labels at beginning:", n_raw)
    print("Normalized pairwise/whole-group labels at beginning:", n_norm)

    print("\nMapped sample-level/duration-level label summary:")
    display(
        raw_summary_tmp
        .groupby("mapped_class")
        .agg(
            n_raw_labels=("raw_label", "nunique"),
            sample_count=("sample_count", "sum"),
            approx_duration_s=("approx_duration_s", "sum"),
        )
        .sort_values("sample_count", ascending=False)
    )

else:
    print("Raw label summary file not found, skipping raw-level inventory summary.")

print("\nDominant raw labels in ENG3 interaction windows:", win_labels["dominant_raw_label"].nunique())
print("Dominant normalized labels in ENG3 interaction windows:", win_labels["dominant_normalized_label"].nunique())


# ------------------------------------------------------------
# Keep interaction windows only
# ------------------------------------------------------------

eng3_interaction = eng3[eng3["label"] == "interaction"].copy()

print("\n" + "=" * 80)
print("ENG3 INTERACTION WINDOWS")
print("=" * 80)
print("Total ENG3 rows:", len(eng3))
print("Interaction windows:", len(eng3_interaction))
print("Non-interaction windows excluded:", len(eng3) - len(eng3_interaction))


# ------------------------------------------------------------
# Merge dominant annotation labels onto ENG3 interaction windows
# ------------------------------------------------------------

merge_cols = ["group", "window_start", "window_end"]

needed_cols = [
    "group",
    "window_start",
    "window_end",
    "dominant_raw_label",
    "dominant_normalized_label",
    "dominant_fraction_in_window",
    "all_raw_labels_in_window",
    "source_tiers_in_window",
    "raw_label_counts",
]

missing_cols = [c for c in needed_cols if c not in win_labels.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Missing columns in window label inventory: {missing_cols}")

rec = eng3_interaction.merge(
    win_labels[needed_cols],
    on=merge_cols,
    how="left"
)

# Apply mapping
mapped = rec["dominant_raw_label"].apply(map_to_5class)
rec["recognition_label"] = [x[0] for x in mapped]
rec["mapping_reason"] = [x[1] for x in mapped]

# Full audit before filtering
rec_full = rec.copy()

# Final clean recognition dataset
rec5 = rec[rec["recognition_label"] != "ignore"].copy()

# Safety checks
assert "non_interaction" not in rec5["recognition_label"].unique()
assert set(rec5["label"].unique()) == {"interaction"}


# ------------------------------------------------------------
# Create mapping audit tables
# ------------------------------------------------------------

mapping_audit_window = (
    rec_full
    .groupby([
        "recognition_label",
        "mapping_reason",
        "dominant_normalized_label",
        "dominant_raw_label"
    ])
    .agg(
        window_count=("dominant_raw_label", "count"),
        mean_dominant_fraction=("dominant_fraction_in_window", "mean"),
        groups_present=("group", lambda x: sorted(set(x))),
        source_tiers=("source_tiers_in_window", lambda x: " | ".join(sorted(set(map(str, x)))))
    )
    .reset_index()
    .sort_values(["recognition_label", "window_count"], ascending=[True, False])
)

if raw_summary is not None:
    raw_mapping_audit = raw_summary.copy()
    mapped_raw = raw_mapping_audit["raw_label"].apply(map_to_5class)
    raw_mapping_audit["mapped_class"] = [x[0] for x in mapped_raw]
    raw_mapping_audit["mapping_reason"] = [x[1] for x in mapped_raw]
else:
    raw_mapping_audit = pd.DataFrame()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

rec5_path = f"{OUT_DIR}/eng3_recognition_5class_interaction_only_features.csv"
rec_full_path = f"{OUT_DIR}/eng3_recognition_5class_full_audit.csv"
mapping_audit_path = f"{OUT_DIR}/eng3_recognition_5class_label_mapping_audit.csv"
raw_mapping_audit_path = f"{OUT_DIR}/eng3_recognition_5class_raw_label_mapping_audit.csv"

rec5.to_csv(rec5_path, index=False)
rec_full.to_csv(rec_full_path, index=False)
mapping_audit_window.to_csv(mapping_audit_path, index=False)

if raw_summary is not None:
    raw_mapping_audit.to_csv(raw_mapping_audit_path, index=False)


# ------------------------------------------------------------
# Print final summaries
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL 5-CLASS RECOGNITION DATASET")
print("=" * 80)

print("Interaction windows before filtering:", len(rec_full))
print("Usable recognition windows after grouping:", len(rec5))
print("Ignored windows:", len(rec_full) - len(rec5))

print("\nFinal recognition class counts:")
print(rec5["recognition_label"].value_counts())

print("\nFinal recognition class counts by group:")
display(pd.crosstab(rec5["group"], rec5["recognition_label"]))

print("\nClass percentage:")
print((rec5["recognition_label"].value_counts(normalize=True) * 100).round(2))

print("\nIgnored dominant labels:")
ignored = rec_full[rec_full["recognition_label"] == "ignore"]

if len(ignored) == 0:
    print("No ignored labels.")
else:
    print(
        ignored["dominant_raw_label"]
        .value_counts()
        .head(50)
        .to_string()
    )

print("\n" + "=" * 80)
print("WINDOW-LEVEL LABEL MAPPING AUDIT")
print("=" * 80)
display(mapping_audit_window)

print("\n" + "=" * 80)
print("RAW LABEL MAPPING AUDIT")
print("=" * 80)

if raw_summary is not None:
    display(
        raw_mapping_audit[
            [
                "raw_label",
                "normalized_label",
                "mapped_class",
                "mapping_reason",
                "sample_count",
                "approx_duration_s",
                "groups_present",
                "tiers_present",
            ]
        ].sort_values(["mapped_class", "sample_count"], ascending=[True, False])
    )
else:
    print("Raw mapping audit unavailable because recognition_raw_label_summary.csv was not found.")

print("\nSaved:")
print(rec5_path)
print(rec_full_path)
print(mapping_audit_path)

if raw_summary is not None:
    print(raw_mapping_audit_path)


# --- CELL 14 (code cell #12) ---

# ================================================================
# B1. CREATE THE THREE-CLASS CORE RECOGNITION TABLE
# ================================================================
ENG3_OUT_DIR = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_ENG3",
)
five_class_path = os.path.join(
    ENG3_OUT_DIR,
    "eng3_recognition_5class_interaction_only_features.csv",
)
core_path = os.path.join(
    ENG3_OUT_DIR,
    "eng3_recognition_3class_core_features.csv",
)

core_classes = [
    "co_building",
    "co_merging",
    "conversation",
]

recognition_5class = pd.read_csv(five_class_path)
recognition_core = recognition_5class[
    recognition_5class["recognition_label"].isin(core_classes)
].copy().reset_index(drop=True)

recognition_core.to_csv(core_path, index=False)

print("Saved three-class recognition table:")
print(core_path)
print("Shape:", recognition_core.shape)
display(recognition_core["recognition_label"].value_counts())


# --- CELL 15 (code cell #13) ---
# ================================================================
# BUILD INTERACTION_ENG7 — activity-discriminating, TRANSFER-INVARIANT
# head & hand features for co_building / co_merging / conversation.
#
# Design principle: every per-person signal is self-normalised by that
# person's OWN session statistics before thresholding, and features are
# ANGLES / FRACTIONS / COUNTS / RATIOS / ALTERNATION — quantities that do
# not depend on who wears the sensor (the thing that killed ENG3-6 on LOGO).
#
# Four families:
#   (1) HEAD POSTURE REGIME   (ear acc -> pitch angle)
#   (2) HEAD GAZE EVENTS       (ear gyro -> turns; vertical band -> nods)
#   (3) HAND RHYTHM/REGIME     (wrist acc -> active-fraction, burstiness, entropy)
#   (4) CROSS-PERSON STRUCTURE (alternation, coupling, handover, role-split)
#
# WINDOW_S parameterised (default 10s). Re-merge to recognition labels with
# the ENG6->recognition cell (nearest-window).
# ================================================================
import os, glob, re, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

INPUT_DIR = RAW_DIR
OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG7")
WINDOW_S, STRIDE_S = 10.0, 10.0
RESAMPLE_HZ = 25                 # grid for event/rhythm detection
NOD_BAND   = (1.0, 3.0)          # Hz, head-nod / listening band
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_TIERS=["label_Participant1_Participant2","label_Participant1_Participant3",
             "label_Participant2_Participant3","label_Whole_Group"]
PAIRS=[(1,2),(1,3),(2,3)]
EAR_ACC =lambda p:[f"p{p}_acc_{a}"  for a in "xyz"]
EAR_GYR =lambda p:[f"p{p}_gyro_{a}" for a in "xyz"]
WR_ACC  =lambda p:[f"p{p}_acc_{a}"  for a in "xyz"]
POS=[f"Participant{p}_{a}" for p in(1,2,3) for a in"xyz"]

# ---------------- IO ----------------
def discover(folder):
    g={}
    for path in glob.glob(os.path.join(folder,"group_*_model_ready.csv")):
        m=re.search(r"group_(\d+)_([a-z]+)_model_ready",os.path.basename(path))
        if m: g.setdefault(int(m.group(1)),{})[m.group(2)]=path
    return {k:v for k,v in sorted(g.items()) if all(s in v for s in("openearable","optitrack","xsens"))}
def load(path,timecol,cols):
    df=pd.read_csv(path,low_memory=False)
    df["t"]=pd.to_numeric(df[timecol],errors="coerce")
    df=df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    for c in cols:
        if c not in df.columns: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors="coerce"); df[c]=df[c].where(df[c].abs()<1e6,np.nan)
    return df
def xoff(oe,xs):
    if "label_Whole_Group" not in oe.columns or "label_Whole_Group" not in xs.columns: return 0.0
    lo=max(oe["t"].min(),5); hi=oe["t"].max()-5
    if not(np.isfinite(lo) and np.isfinite(hi)) or hi<=lo: return 0.0
    grid=np.arange(lo,hi,0.2)
    if len(grid)==0: return 0.0
    def samp(df,tt):
        idx=np.clip(np.searchsorted(df["t"].values,tt),0,len(df)-1)
        return df["label_Whole_Group"].fillna("NONE").astype(str).values[idx]
    ref=samp(oe,grid); bo,bs=0.0,-1.0
    for off in np.arange(0,220,0.5):
        s=np.mean(ref==samp(xs,grid+off))
        if s>bs: bs,bo=s,off
    return bo
def sinterp(grid,t,v):
    m=np.isfinite(v)
    if m.sum()<2: return np.full_like(grid,np.nan,dtype=float)
    return np.interp(grid,t[m],v[m])

# ---------------- helpers ----------------
def band_ratio(sig, fs, lo, hi):
    s=sig[np.isfinite(sig)]
    if len(s)<8: return np.nan
    s=s-s.mean(); f=np.fft.rfftfreq(len(s),1/fs); P=np.abs(np.fft.rfft(s))**2
    tot=P[1:].sum()
    if tot<=0: return np.nan
    return float(P[(f>=lo)&(f<hi)].sum()/tot)
def spectral_entropy(sig, fs):
    s=sig[np.isfinite(sig)]
    if len(s)<8: return np.nan
    s=s-s.mean(); P=np.abs(np.fft.rfft(s))**2; P=P[1:]
    if P.sum()<=0: return np.nan
    p=P/P.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/np.log(len(p)))   # 0=tonal, 1=broadband
def active_mask(sig, thr):           # sig already self-normalised (z); thr in std units
    return sig>thr
def event_count(mask):               # rising edges
    m=mask.astype(int)
    return int(np.sum(np.diff(m)==1))
def burstiness(mask):                # CV of inter-event gaps (Fano-like)
    idx=np.where(np.diff(mask.astype(int))==1)[0]
    if len(idx)<3: return np.nan
    gaps=np.diff(idx)
    return float(np.std(gaps)/ (np.mean(gaps)+1e-9))

# ---------------- per-window features ----------------
def features(oe, xs, ot, ws, we, sess):
    gh=RESAMPLE_HZ; grid=np.linspace(ws,we,int(WINDOW_S*gh))
    row={}
    # per-person signals on grid (self-normalised using SESSION baseline in `sess`)
    head_pitch={}; head_yawrate={}; head_vert={}; hand_e={}; head_e={}
    for p in (1,2,3):
        # ---- (1) head pitch ANGLE (gravity), invariant ----
        ac=np.stack([sinterp(grid,oe["t"].values,oe[f"p{p}_acc_{a}"].values) for a in "xyz"],1)
        pitch=np.arctan2(ac[:,0],np.sqrt(ac[:,1]**2+ac[:,2]**2))
        head_pitch[p]=pitch
        head_vert[p]=ac[:,2]                       # vertical accel for nod band
        # ---- (2) head yaw rate (gyro magnitude as turn signal) ----
        gy=np.stack([sinterp(grid,oe["t"].values,oe[f"p{p}_gyro_{a}"].values) for a in "xyz"],1)
        head_yawrate[p]=np.linalg.norm(gy,axis=1)
        # head/hand energy (self-normalised by session std)
        he=np.linalg.norm(ac,axis=1)
        head_e[p]=(he-sess[p]["head_mean"])/ (sess[p]["head_std"]+1e-9)
        wa=np.stack([sinterp(grid,xs["t"].values,xs[f"p{p}_acc_{a}"].values) for a in "xyz"],1)
        we_=np.linalg.norm(wa,axis=1)
        hand_e[p]=(we_-sess[p]["hand_mean"])/ (sess[p]["hand_std"]+1e-9)

    # ===== FAMILY 1: head posture regime =====
    down_thr=-0.35  # rad (~ -20 deg) head tilted down
    row["ear_head_down_fraction"]=float(np.nanmean([np.nanmean(head_pitch[p]<down_thr) for p in(1,2,3)]))
    row["ear_head_pitch_range_mean"]=float(np.nanmean([np.nanmax(head_pitch[p])-np.nanmin(head_pitch[p]) for p in(1,2,3)]))
    # level crossings of the down threshold = switching between down/up regimes
    row["ear_head_regime_switch_rate"]=float(np.nanmean([event_count(head_pitch[p]<down_thr)/WINDOW_S for p in(1,2,3)]))

    # ===== FAMILY 2: gaze events =====
    # turn events: peaks in yaw rate above each person's own session 80th pct
    turn_rates=[]; nod_ratios=[]
    for p in (1,2,3):
        thr=sess[p]["yaw_p80"]
        turn_rates.append(event_count(head_yawrate[p]>thr)/WINDOW_S)
        nod_ratios.append(band_ratio(head_vert[p],gh,*NOD_BAND))
    row["ear_head_turn_event_rate"]=float(np.nanmean(turn_rates))
    row["ear_head_nod_band_ratio"]=float(np.nanmean([r for r in nod_ratios if np.isfinite(r)])) if any(np.isfinite(nod_ratios)) else np.nan

    # ===== FAMILY 3: hand rhythm / regime =====
    afrac=[]; burst=[]; entropy=[]; intermit=[]
    for p in (1,2,3):
        m=active_mask(hand_e[p],0.0)               # active = above own session mean
        afrac.append(float(np.mean(m)))
        burst.append(burstiness(m))
        wa_mag=hand_e[p]
        entropy.append(spectral_entropy(wa_mag,gh))
        intermit.append(float(np.mean(~m)/(np.mean(m)+1e-9)))
    row["wrist_hand_active_fraction"]=float(np.nanmean(afrac))
    row["wrist_hand_burstiness"]=float(np.nanmean([b for b in burst if np.isfinite(b)])) if any(np.isfinite(burst)) else np.nan
    row["wrist_hand_spectral_entropy"]=float(np.nanmean([e for e in entropy if np.isfinite(e)])) if any(np.isfinite(entropy)) else np.nan
    row["wrist_hand_intermittency"]=float(np.nanmean(intermit))

    # ===== FAMILY 4: cross-person structure =====
    # head-activity alternation: anti-correlation of who's-active-when (turn-taking)
    Hact={p:(head_e[p]>0).astype(float) for p in (1,2,3)}
    altcorr=[]
    for a,b in PAIRS:
        if np.std(Hact[a])>1e-9 and np.std(Hact[b])>1e-9:
            altcorr.append(np.corrcoef(Hact[a],Hact[b])[0,1])
    row["ear_head_activity_alternation"]=float(-np.nanmean(altcorr)) if altcorr else np.nan  # +ve = alternate

    # hand coupling events: near-simultaneous hand bursts across people (<= ~0.4s)
    Hand_on={p:active_mask(hand_e[p],0.5) for p in (1,2,3)}
    simul=np.stack([Hand_on[p] for p in (1,2,3)],0).sum(0)
    row["wrist_hand_coupling_rate"]=float(np.mean(simul>=2))
    # handover: anti-phase wrist spikes (one rising while other falling) per pair
    ho=[]
    for a,b in PAIRS:
        da=np.diff(hand_e[a]); db=np.diff(hand_e[b])
        ho.append(float(np.mean((da>0.3)&(db<-0.3) | (da<-0.3)&(db>0.3))))
    row["wrist_handover_event_rate"]=float(np.nanmean(ho))
    # role split: some people head-active/hand-idle (talk) vs hand-active/head-idle (work)
    talk={p:(head_e[p]>0)&(hand_e[p]<0) for p in (1,2,3)}
    work={p:(hand_e[p]>0)&(head_e[p]<0) for p in (1,2,3)}
    talkfrac=np.mean([np.mean(talk[p]) for p in (1,2,3)])
    workfrac=np.mean([np.mean(work[p]) for p in (1,2,3)])
    row["xmod_role_split_index"]=float(abs(talkfrac-workfrac))
    row["xmod_talk_fraction"]=float(talkfrac)
    row["xmod_work_fraction"]=float(workfrac)

    # ===== proximity baseline (for comparison) =====
    Pg={p:np.stack([sinterp(grid,ot["t"].values,ot[f"Participant{p}_{a}"].values) for a in "xyz"],1) for p in (1,2,3)}
    D=np.stack([np.linalg.norm(Pg[a]-Pg[b],axis=1) for a,b in PAIRS],1); Ds=np.sort(D,1)
    row["opti_nearest_pair_dist_mean"]=float(np.nanmean(Ds[:,0]))
    row["opti_all_pairs_dist_mean"]=float(np.nanmean(D))
    row["opti_all_pairs_dist_std"]=float(np.nanstd(D))
    return row

# ---------------- session baselines (for self-normalisation) ----------------
def session_baseline(oe, xs):
    sess={}
    for p in (1,2,3):
        he=np.linalg.norm(np.stack([oe[f"p{p}_acc_{a}"].values for a in "xyz"],1),axis=1)
        yaw=np.linalg.norm(np.stack([oe[f"p{p}_gyro_{a}"].values for a in "xyz"],1),axis=1)
        ha=np.linalg.norm(np.stack([xs[f"p{p}_acc_{a}"].values for a in "xyz"],1),axis=1)
        sess[p]=dict(head_mean=np.nanmean(he), head_std=np.nanstd(he),
                     hand_mean=np.nanmean(ha), hand_std=np.nanstd(ha),
                     yaw_p80=np.nanpercentile(yaw[np.isfinite(yaw)],80) if np.isfinite(yaw).any() else np.inf)
    return sess

# ---------------- build ----------------
def build():
    rows,Y,G=[],[],[]; counts={}
    for group,paths in groups.items():
        print(f"\nGroup {group}")
        oe=load(paths["openearable"],"video_time_s",sum([EAR_ACC(p)+EAR_GYR(p) for p in(1,2,3)],[]))
        ot=load(paths["optitrack"],"video_time_s",POS)
        xs=load(paths["xsens"],"time_s",sum([WR_ACC(p) for p in(1,2,3)],[])); xs["t"]=xs["t"]-xoff(oe,xs)
        sess=session_baseline(oe,xs)
        lo=max(oe["t"].min(),ot["t"].min()); hi=min(oe["t"].max(),ot["t"].max())
        rt=oe["t"].values
        inter=np.zeros(len(oe),bool)
        for col in GROUP_TIERS:
            if col in oe.columns:
                inter|=(oe[col].notna().values & (oe[col].astype(str).str.strip()!="").values)
        n0=len(Y)
        for ws in np.arange(lo,hi-WINDOW_S+1e-9,STRIDE_S):
            we=ws+WINDOW_S; m=(rt>=ws)&(rt<we)
            if m.sum()==0: continue
            label="interaction" if inter[m].mean()>=0.5 else "non_interaction"
            row=features(oe,xs,ot,ws,we,sess)
            row["group"]=group; row["window_start"]=ws; row["window_end"]=we
            rows.append(row); Y.append(label); G.append(group)
        counts[group]=len(Y)-n0; print(f"  windows: {counts[group]}")
        del oe,ot,xs; gc.collect()
    F=pd.DataFrame(rows); F["label"]=Y
    return F,np.array(Y),np.array(G),counts

groups=discover(INPUT_DIR)
print("Groups:",list(groups.keys()),"| WINDOW_S =",WINDOW_S)
F7,Y7,G7,counts=build()
tag=f"{int(WINDOW_S)}s"
F7.to_csv(f"{OUT_DIR}/interaction_eng7_{tag}.csv",index=False)
print("\nSAVED:",f"{OUT_DIR}/interaction_eng7_{tag}.csv | shape",F7.shape)
print("\nFeatures:")
for c in F7.columns:
    if c not in("group","window_start","window_end","label"): print("  -",c)


# --- CELL 16 (code cell #14) ---
# ================================================================
# BUILD OE9 — RICH OPENEAREABLE-ONLY FEATURES
#
# Goal:
#   Maximize 3-class recognition using ONLY OpenEarable acc/gyro.
#
# Compared with old ENG7/OE8:
#   - keeps old ear_ features for comparison
#   - adds richer invariant oe_ features:
#       per-person distribution summaries
#       pitch/roll/head-posture dynamics
#       gyro/turn dynamics
#       nod/spectral bands
#       jerk/change features
#       active-person-count features
#       dominance/asymmetry features
#       cross-person synchrony and lag features
#
# Output:
#   /content/drive/MyDrive/thesis/data/INTERACTION_OE9/interaction_oe9_10s.csv
# ================================================================

import os
import glob
import re
import gc
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ---------------- Paths / settings ----------------

INPUT_DIR = RAW_DIR
OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_OE9")

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_S = 10.0
STRIDE_S = 10.0
RESAMPLE_HZ = 25

NOD_BAND = (1.0, 3.0)
LOW_MOTION_BAND = (0.2, 1.0)
HIGH_MOTION_BAND = (3.0, 8.0)

PAIRS = [(1, 2), (1, 3), (2, 3)]

EAR_ACC = lambda p: [f"p{p}_acc_{a}" for a in "xyz"]
EAR_GYR = lambda p: [f"p{p}_gyro_{a}" for a in "xyz"]

ALL_OE_COLS = sum([EAR_ACC(p) + EAR_GYR(p) for p in (1, 2, 3)], [])


# ================================================================
# IO
# ================================================================

def discover_openearable(folder):
    found = {}

    for path in glob.glob(os.path.join(folder, "group_*_openearable_model_ready.csv")):
        m = re.search(r"group_(\d+)_openearable_model_ready", os.path.basename(path))

        if m:
            found[int(m.group(1))] = path

    return dict(sorted(found.items()))


def load_oe(path):
    df = pd.read_csv(path, low_memory=False)

    if "video_time_s" in df.columns:
        time_col = "video_time_s"
    elif "time_s" in df.columns:
        time_col = "time_s"
    else:
        raise ValueError("No video_time_s or time_s column found.")

    df["t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    for c in ALL_OE_COLS:
        if c not in df.columns:
            df[c] = np.nan

        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e6, np.nan)

    return df


# ================================================================
# Signal helpers
# ================================================================

def sinterp(grid, t, v):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    m = np.isfinite(t) & np.isfinite(v)

    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)

    return np.interp(grid, t[m], v[m])


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def safe_min(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def safe_range(x):
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).any():
        return np.nan

    return float(np.nanmax(x) - np.nanmin(x))


def safe_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.percentile(x, 75) - np.percentile(x, 25))


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.percentile(x, q))


def mad_diff(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return np.nan

    return float(np.nanmedian(np.abs(np.diff(x))))


def event_count(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) < 2:
        return 0

    return int(np.sum(np.diff(mask.astype(int)) == 1))


def spectral_entropy(sig, fs):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)
    P = np.abs(np.fft.rfft(s)) ** 2
    P = P[1:]

    if P.sum() <= 0:
        return np.nan

    p = P / P.sum()
    p = p[p > 0]

    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def band_ratio(sig, fs, lo, hi):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)
    f = np.fft.rfftfreq(len(s), 1 / fs)
    P = np.abs(np.fft.rfft(s)) ** 2

    total = P[1:].sum()

    if total <= 0:
        return np.nan

    return float(P[(f >= lo) & (f < hi)].sum() / total)


def corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() < 8:
        return np.nan

    a = a[m]
    b = b[m]

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def max_lag_corr(a, b, max_lag_steps):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    best = np.nan

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        c = corr_safe(aa, bb)

        if np.isfinite(c):
            if not np.isfinite(best) or abs(c) > abs(best):
                best = c

    return best


def aggregate_values(row, prefix, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_std"] = np.nan
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_max"] = np.nan
        row[f"{prefix}_range"] = np.nan
        return

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite))
    row[f"{prefix}_min"] = float(np.min(finite))
    row[f"{prefix}_max"] = float(np.max(finite))
    row[f"{prefix}_range"] = float(np.max(finite) - np.min(finite))


# ================================================================
# Session baselines
# ================================================================

def session_baseline(oe):
    sess = {}

    for p in (1, 2, 3):
        acc = np.stack(
            [oe[f"p{p}_acc_{a}"].values for a in "xyz"],
            axis=1,
        )

        gyr = np.stack(
            [oe[f"p{p}_gyro_{a}"].values for a in "xyz"],
            axis=1,
        )

        acc_mag = np.linalg.norm(acc, axis=1)
        gyr_mag = np.linalg.norm(gyr, axis=1)

        pitch = np.arctan2(
            acc[:, 0],
            np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2),
        )

        roll = np.arctan2(
            acc[:, 1],
            np.sqrt(acc[:, 0] ** 2 + acc[:, 2] ** 2),
        )

        sess[p] = {
            "acc_mean": safe_mean(acc_mag),
            "acc_std": safe_std(acc_mag),
            "gyr_mean": safe_mean(gyr_mag),
            "gyr_std": safe_std(gyr_mag),
            "pitch_mean": safe_mean(pitch),
            "pitch_std": safe_std(pitch),
            "roll_mean": safe_mean(roll),
            "roll_std": safe_std(roll),
            "gyr_p75": safe_percentile(gyr_mag, 75),
            "gyr_p80": safe_percentile(gyr_mag, 80),
            "gyr_p90": safe_percentile(gyr_mag, 90),
            "acc_p75": safe_percentile(acc_mag, 75),
            "acc_p90": safe_percentile(acc_mag, 90),
        }

    return sess


# ================================================================
# Feature extraction
# ================================================================

def extract_oe9_features(oe, ws, we, sess):
    n = int(WINDOW_S * RESAMPLE_HZ)
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ

    row = {}

    acc = {}
    gyr = {}
    acc_mag = {}
    gyr_mag = {}
    acc_e = {}
    gyr_e = {}
    pitch = {}
    roll = {}
    vert = {}
    jerk = {}
    angular_jerk = {}

    # ------------------------------------------------------------
    # Per-person signals
    # ------------------------------------------------------------

    for p in (1, 2, 3):
        acc[p] = np.stack(
            [
                sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        gyr[p] = np.stack(
            [
                sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        acc_mag[p] = np.linalg.norm(acc[p], axis=1)
        gyr_mag[p] = np.linalg.norm(gyr[p], axis=1)

        acc_e[p] = (acc_mag[p] - sess[p]["acc_mean"]) / (sess[p]["acc_std"] + 1e-9)
        gyr_e[p] = (gyr_mag[p] - sess[p]["gyr_mean"]) / (sess[p]["gyr_std"] + 1e-9)

        pitch[p] = np.arctan2(
            acc[p][:, 0],
            np.sqrt(acc[p][:, 1] ** 2 + acc[p][:, 2] ** 2),
        )

        roll[p] = np.arctan2(
            acc[p][:, 1],
            np.sqrt(acc[p][:, 0] ** 2 + acc[p][:, 2] ** 2),
        )

        vert[p] = acc[p][:, 2]

        jerk[p] = np.r_[0, np.diff(acc_mag[p])] * fs
        angular_jerk[p] = np.r_[0, np.diff(gyr_mag[p])] * fs

    # ------------------------------------------------------------
    # Old ENG7-style ear_ features
    # ------------------------------------------------------------

    down_thr = -0.35

    down_fracs = []
    pitch_ranges = []
    switch_rates = []
    turn_rates = []
    nod_ratios = []

    for p in (1, 2, 3):
        down = pitch[p] < down_thr

        down_fracs.append(np.nanmean(down))
        pitch_ranges.append(safe_range(pitch[p]))
        switch_rates.append(event_count(down) / WINDOW_S)

        turn_rates.append(event_count(gyr_mag[p] > sess[p]["gyr_p80"]) / WINDOW_S)
        nod_ratios.append(band_ratio(vert[p], fs, *NOD_BAND))

    row["ear_head_down_fraction"] = safe_mean(down_fracs)
    row["ear_head_pitch_range_mean"] = safe_mean(pitch_ranges)
    row["ear_head_regime_switch_rate"] = safe_mean(switch_rates)
    row["ear_head_turn_event_rate"] = safe_mean(turn_rates)
    row["ear_head_nod_band_ratio"] = safe_mean(nod_ratios)

    head_active_binary = {
        p: (gyr_e[p] > 0).astype(float)
        for p in (1, 2, 3)
    }

    pair_corrs = []

    for a, b in PAIRS:
        c = corr_safe(head_active_binary[a], head_active_binary[b])
        if np.isfinite(c):
            pair_corrs.append(c)

    row["ear_head_activity_alternation"] = (
        float(-np.mean(pair_corrs)) if len(pair_corrs) else np.nan
    )

    # ------------------------------------------------------------
    # Rich per-person OE features, aggregated across people
    # ------------------------------------------------------------

    per_person_feature_values = {}

    def collect(name, vals):
        per_person_feature_values[name] = vals
        aggregate_values(row, f"oe_{name}", vals)

    collect("acc_energy", [safe_mean(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_std", [safe_std(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_iqr", [safe_iqr(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_range", [safe_range(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_energy_maddiff", [mad_diff(acc_e[p]) for p in (1, 2, 3)])
    collect("acc_entropy", [spectral_entropy(acc_e[p], fs) for p in (1, 2, 3)])
    collect("acc_low_band", [band_ratio(acc_e[p], fs, *LOW_MOTION_BAND) for p in (1, 2, 3)])
    collect("acc_nod_band", [band_ratio(acc_e[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("acc_high_band", [band_ratio(acc_e[p], fs, *HIGH_MOTION_BAND) for p in (1, 2, 3)])

    collect("gyro_energy", [safe_mean(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_std", [safe_std(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_iqr", [safe_iqr(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_range", [safe_range(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_energy_maddiff", [mad_diff(gyr_e[p]) for p in (1, 2, 3)])
    collect("gyro_entropy", [spectral_entropy(gyr_e[p], fs) for p in (1, 2, 3)])
    collect("gyro_low_band", [band_ratio(gyr_e[p], fs, *LOW_MOTION_BAND) for p in (1, 2, 3)])
    collect("gyro_nod_band", [band_ratio(gyr_e[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("gyro_high_band", [band_ratio(gyr_e[p], fs, *HIGH_MOTION_BAND) for p in (1, 2, 3)])

    collect("pitch_mean", [safe_mean(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_std", [safe_std(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_iqr", [safe_iqr(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_range", [safe_range(pitch[p]) for p in (1, 2, 3)])
    collect("pitch_maddiff", [mad_diff(pitch[p]) for p in (1, 2, 3)])

    collect("roll_mean", [safe_mean(roll[p]) for p in (1, 2, 3)])
    collect("roll_std", [safe_std(roll[p]) for p in (1, 2, 3)])
    collect("roll_iqr", [safe_iqr(roll[p]) for p in (1, 2, 3)])
    collect("roll_range", [safe_range(roll[p]) for p in (1, 2, 3)])
    collect("roll_maddiff", [mad_diff(roll[p]) for p in (1, 2, 3)])

    collect("jerk_abs_mean", [safe_mean(np.abs(jerk[p])) for p in (1, 2, 3)])
    collect("jerk_abs_std", [safe_std(np.abs(jerk[p])) for p in (1, 2, 3)])
    collect("angular_jerk_abs_mean", [safe_mean(np.abs(angular_jerk[p])) for p in (1, 2, 3)])
    collect("angular_jerk_abs_std", [safe_std(np.abs(angular_jerk[p])) for p in (1, 2, 3)])

    collect("down_fraction", [np.nanmean(pitch[p] < down_thr) for p in (1, 2, 3)])
    collect("up_fraction", [np.nanmean(pitch[p] >= down_thr) for p in (1, 2, 3)])
    collect("turn_rate_p75", [event_count(gyr_mag[p] > sess[p]["gyr_p75"]) / WINDOW_S for p in (1, 2, 3)])
    collect("turn_rate_p90", [event_count(gyr_mag[p] > sess[p]["gyr_p90"]) / WINDOW_S for p in (1, 2, 3)])
    collect("acc_burst_rate_p75", [event_count(acc_mag[p] > sess[p]["acc_p75"]) / WINDOW_S for p in (1, 2, 3)])
    collect("acc_burst_rate_p90", [event_count(acc_mag[p] > sess[p]["acc_p90"]) / WINDOW_S for p in (1, 2, 3)])

    # ------------------------------------------------------------
    # Active-person-count features
    # ------------------------------------------------------------

    acc_active = np.stack([(acc_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    gyro_active = np.stack([(gyr_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    down_active = np.stack([(pitch[p] < down_thr).astype(int) for p in (1, 2, 3)], axis=0)

    acc_count = acc_active.sum(axis=0)
    gyro_count = gyro_active.sum(axis=0)
    down_count = down_active.sum(axis=0)

    for name, count in [
        ("acc_active_count", acc_count),
        ("gyro_active_count", gyro_count),
        ("down_count", down_count),
    ]:
        row[f"oe_{name}_mean"] = safe_mean(count)
        row[f"oe_{name}_std"] = safe_std(count)
        row[f"oe_{name}_exactly1_frac"] = float(np.mean(count == 1))
        row[f"oe_{name}_atleast2_frac"] = float(np.mean(count >= 2))
        row[f"oe_{name}_all3_frac"] = float(np.mean(count == 3))
        row[f"oe_{name}_switch_rate"] = event_count(np.r_[False, np.diff(count) != 0]) / WINDOW_S

    # ------------------------------------------------------------
    # Dominance / asymmetry features
    # ------------------------------------------------------------

    for name, vals in per_person_feature_values.items():
        vals = np.asarray(vals, dtype=float)

        if np.isfinite(vals).sum() >= 2:
            row[f"oe_{name}_dominance_gap"] = float(np.nanmax(vals) - np.nanmedian(vals))
            row[f"oe_{name}_asymmetry"] = float(np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-9))
        else:
            row[f"oe_{name}_dominance_gap"] = np.nan
            row[f"oe_{name}_asymmetry"] = np.nan

    # ------------------------------------------------------------
    # Cross-person synchrony and lag
    # ------------------------------------------------------------

    pair_acc_corrs = []
    pair_gyro_corrs = []
    pair_pitch_corrs = []
    pair_acc_lagcorrs = []
    pair_gyro_lagcorrs = []

    max_lag_steps = int(1.0 * fs)

    for a, b in PAIRS:
        pair_acc_corrs.append(corr_safe(acc_e[a], acc_e[b]))
        pair_gyro_corrs.append(corr_safe(gyr_e[a], gyr_e[b]))
        pair_pitch_corrs.append(corr_safe(pitch[a], pitch[b]))

        pair_acc_lagcorrs.append(max_lag_corr(acc_e[a], acc_e[b], max_lag_steps))
        pair_gyro_lagcorrs.append(max_lag_corr(gyr_e[a], gyr_e[b], max_lag_steps))

    aggregate_values(row, "oe_pair_acc_corr", pair_acc_corrs)
    aggregate_values(row, "oe_pair_gyro_corr", pair_gyro_corrs)
    aggregate_values(row, "oe_pair_pitch_corr", pair_pitch_corrs)
    aggregate_values(row, "oe_pair_acc_lagcorr", pair_acc_lagcorrs)
    aggregate_values(row, "oe_pair_gyro_lagcorr", pair_gyro_lagcorrs)

    # ------------------------------------------------------------
    # First-half vs second-half change features
    # ------------------------------------------------------------

    half = len(grid) // 2

    for signal_name, signals in [
        ("acc_e", acc_e),
        ("gyro_e", gyr_e),
        ("pitch", pitch),
    ]:
        deltas = []

        for p in (1, 2, 3):
            x = signals[p]

            if len(x) >= 4:
                deltas.append(safe_mean(x[half:]) - safe_mean(x[:half]))
            else:
                deltas.append(np.nan)

        aggregate_values(row, f"oe_{signal_name}_half_delta", deltas)

    return row


# ================================================================
# Build dataset
# ================================================================

def build_oe9():
    files = discover_openearable(INPUT_DIR)

    print("=" * 100)
    print("BUILDING OE9")
    print("=" * 100)
    print("Groups:", list(files.keys()))
    print("Window:", WINDOW_S, "Stride:", STRIDE_S)

    rows = []
    counts = {}

    for group, path in files.items():
        print("\nGroup", group, "|", os.path.basename(path))

        oe = load_oe(path)
        sess = session_baseline(oe)

        lo = float(oe["t"].min())
        hi = float(oe["t"].max())

        n0 = len(rows)

        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S

            row = extract_oe9_features(oe, ws, we, sess)
            row["group"] = group
            row["window_start"] = float(ws)
            row["window_end"] = float(we)

            rows.append(row)

        counts[group] = len(rows) - n0
        print("  windows:", counts[group])

        del oe
        gc.collect()

    out = pd.DataFrame(rows)

    return out, counts


F9, counts = build_oe9()

OUT_PATH = f"{OUT_DIR}/interaction_oe9_{int(WINDOW_S)}s.csv"
F9.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 100)
print("SAVED OE9")
print("=" * 100)
print(OUT_PATH)
print("Shape:", F9.shape)

print("\nNumber of old ear_ features:", len([c for c in F9.columns if c.startswith("ear_")]))
print("Number of new oe_ features:", len([c for c in F9.columns if c.startswith("oe_")]))

print("\nRows by group:")
print(pd.Series(counts).to_string())


# --- CELL 17 (code cell #15) ---
# ================================================================
# BUILD OE10 — ADD MAGNETOMETER FEATURES TO EXISTING OE9
#
# Input:
#   OE9 acc+gyro features:
#   /content/drive/MyDrive/thesis/data/INTERACTION_OE9/interaction_oe9_10s.csv
#
# Raw OpenEarable files contain:
#   p1_mag_x/y/z
#   p2_mag_x/y/z
#   p3_mag_x/y/z
#
# Output:
#   /content/drive/MyDrive/thesis/data/INTERACTION_OE10/interaction_oe10_10s.csv
#
# Features added:
#   mag_...
# ================================================================

import os
import glob
import re
import gc
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

INPUT_DIR = RAW_DIR
OE9_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_OE9", "interaction_oe9_10s.csv")
OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_OE10")

os.makedirs(OUT_DIR, exist_ok=True)

RESAMPLE_HZ = 25
LOW_BAND = (0.2, 1.0)
MID_BAND = (1.0, 3.0)
HIGH_BAND = (3.0, 8.0)

PAIRS = [(1, 2), (1, 3), (2, 3)]

MAG_COLS = sum([[f"p{p}_mag_{a}" for a in "xyz"] for p in (1, 2, 3)], [])


# ================================================================
# Helpers
# ================================================================

def discover_openearable(folder):
    files = {}

    for path in glob.glob(os.path.join(folder, "group_*_openearable_model_ready.csv")):
        m = re.search(r"group_(\d+)_openearable_model_ready", os.path.basename(path))

        if m:
            files[int(m.group(1))] = path

    return dict(sorted(files.items()))


def load_mag(path):
    df = pd.read_csv(path, low_memory=False)

    if "video_time_s" in df.columns:
        time_col = "video_time_s"
    elif "time_s" in df.columns:
        time_col = "time_s"
    else:
        raise ValueError("No usable time column found.")

    df["t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    for c in MAG_COLS:
        if c not in df.columns:
            df[c] = np.nan

        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e9, np.nan)

    return df


def sinterp(grid, t, v):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    m = np.isfinite(t) & np.isfinite(v)

    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)

    return np.interp(grid, t[m], v[m])


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def safe_range(x):
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).any():
        return np.nan

    return float(np.nanmax(x) - np.nanmin(x))


def safe_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.percentile(x, 75) - np.percentile(x, 25))


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.percentile(x, q))


def mad_diff(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3:
        return np.nan

    return float(np.nanmedian(np.abs(np.diff(x))))


def event_count(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) < 2:
        return 0

    return int(np.sum(np.diff(mask.astype(int)) == 1))


def spectral_entropy(sig, fs):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)
    P = np.abs(np.fft.rfft(s)) ** 2
    P = P[1:]

    if P.sum() <= 0:
        return np.nan

    p = P / P.sum()
    p = p[p > 0]

    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def band_ratio(sig, fs, lo, hi):
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]

    if len(s) < 8:
        return np.nan

    s = s - np.mean(s)
    f = np.fft.rfftfreq(len(s), 1 / fs)
    P = np.abs(np.fft.rfft(s)) ** 2

    total = P[1:].sum()

    if total <= 0:
        return np.nan

    return float(P[(f >= lo) & (f < hi)].sum() / total)


def corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() < 8:
        return np.nan

    a = a[m]
    b = b[m]

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def circular_diff(a, b):
    """
    Difference between two angles in radians, wrapped to [-pi, pi].
    """
    return np.angle(np.exp(1j * (a - b)))


def max_lag_corr(a, b, max_lag_steps):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    best = np.nan

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b

        c = corr_safe(aa, bb)

        if np.isfinite(c):
            if not np.isfinite(best) or abs(c) > abs(best):
                best = c

    return best


def aggregate(row, prefix, vals):
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]

    if len(finite) == 0:
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_std"] = np.nan
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_max"] = np.nan
        row[f"{prefix}_range"] = np.nan
        return

    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite))
    row[f"{prefix}_min"] = float(np.min(finite))
    row[f"{prefix}_max"] = float(np.max(finite))
    row[f"{prefix}_range"] = float(np.max(finite) - np.min(finite))


# ================================================================
# Session-level magnetometer baseline
# ================================================================

def mag_session_baseline(df):
    sess = {}

    for p in (1, 2, 3):
        M = np.stack(
            [df[f"p{p}_mag_{a}"].values for a in "xyz"],
            axis=1,
        )

        mag_mag = np.linalg.norm(M, axis=1)
        heading = np.unwrap(np.arctan2(M[:, 1], M[:, 0]))

        t = df["t"].values
        dt = np.diff(t)
        dh = np.diff(heading)

        good = np.isfinite(dt) & (dt > 0) & (dt < 1.0) & np.isfinite(dh)

        if good.sum() > 10:
            heading_vel = dh[good] / dt[good]
        else:
            heading_vel = np.array([])

        dm = np.diff(mag_mag)
        good_m = np.isfinite(dt) & (dt > 0) & (dt < 1.0) & np.isfinite(dm)

        if good_m.sum() > 10:
            mag_vel = dm[good_m] / dt[good_m]
        else:
            mag_vel = np.array([])

        sess[p] = {
            "mag_mean": safe_mean(mag_mag),
            "mag_std": safe_std(mag_mag),
            "mag_p75": safe_percentile(mag_mag, 75),
            "heading_vel_p75": safe_percentile(np.abs(heading_vel), 75),
            "heading_vel_p90": safe_percentile(np.abs(heading_vel), 90),
            "mag_vel_abs_p75": safe_percentile(np.abs(mag_vel), 75),
            "mag_vel_abs_p90": safe_percentile(np.abs(mag_vel), 90),
        }

    return sess


# ================================================================
# Window-level magnetometer features
# ================================================================

def extract_mag_features(df, ws, we, sess):
    dur = float(we - ws)

    if dur <= 0:
        return {}

    n = max(8, int(dur * RESAMPLE_HZ))
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ

    row = {}

    M = {}
    mag_mag = {}
    mag_e = {}
    heading = {}
    heading_vel = {}
    mag_vel = {}
    horizontal_strength = {}

    for p in (1, 2, 3):
        M[p] = np.stack(
            [
                sinterp(grid, df["t"].values, df[f"p{p}_mag_{a}"].values)
                for a in "xyz"
            ],
            axis=1,
        )

        mag_mag[p] = np.linalg.norm(M[p], axis=1)

        mag_e[p] = (
            (mag_mag[p] - sess[p]["mag_mean"])
            / (sess[p]["mag_std"] + 1e-9)
        )

        heading[p] = np.unwrap(np.arctan2(M[p][:, 1], M[p][:, 0]))

        heading_vel[p] = np.r_[0, np.diff(heading[p])] * fs
        mag_vel[p] = np.r_[0, np.diff(mag_e[p])] * fs

        horizontal_strength[p] = np.sqrt(M[p][:, 0] ** 2 + M[p][:, 1] ** 2)

    # ------------------------------------------------------------
    # Per-person magnitude features
    # ------------------------------------------------------------

    def collect(name, vals):
        aggregate(row, f"mag_{name}", vals)

    collect("magnitude_z_mean", [safe_mean(mag_e[p]) for p in (1, 2, 3)])
    collect("magnitude_z_std", [safe_std(mag_e[p]) for p in (1, 2, 3)])
    collect("magnitude_z_iqr", [safe_iqr(mag_e[p]) for p in (1, 2, 3)])
    collect("magnitude_z_range", [safe_range(mag_e[p]) for p in (1, 2, 3)])
    collect("magnitude_z_maddiff", [mad_diff(mag_e[p]) for p in (1, 2, 3)])

    collect("magnitude_vel_abs_mean", [safe_mean(np.abs(mag_vel[p])) for p in (1, 2, 3)])
    collect("magnitude_vel_abs_std", [safe_std(np.abs(mag_vel[p])) for p in (1, 2, 3)])

    collect("horizontal_strength_mean", [safe_mean(horizontal_strength[p]) for p in (1, 2, 3)])
    collect("horizontal_strength_std", [safe_std(horizontal_strength[p]) for p in (1, 2, 3)])
    collect("horizontal_strength_range", [safe_range(horizontal_strength[p]) for p in (1, 2, 3)])

    # ------------------------------------------------------------
    # Heading/orientation features
    # ------------------------------------------------------------

    collect("heading_std", [safe_std(heading[p]) for p in (1, 2, 3)])
    collect("heading_range", [safe_range(heading[p]) for p in (1, 2, 3)])
    collect("heading_maddiff", [mad_diff(heading[p]) for p in (1, 2, 3)])
    collect("heading_vel_abs_mean", [safe_mean(np.abs(heading_vel[p])) for p in (1, 2, 3)])
    collect("heading_vel_abs_std", [safe_std(np.abs(heading_vel[p])) for p in (1, 2, 3)])
    collect("heading_vel_abs_range", [safe_range(np.abs(heading_vel[p])) for p in (1, 2, 3)])

    collect(
        "heading_turn_rate_p75",
        [
            event_count(np.abs(heading_vel[p]) > sess[p]["heading_vel_p75"]) / dur
            for p in (1, 2, 3)
        ],
    )

    collect(
        "heading_turn_rate_p90",
        [
            event_count(np.abs(heading_vel[p]) > sess[p]["heading_vel_p90"]) / dur
            for p in (1, 2, 3)
        ],
    )

    collect(
        "mag_burst_rate_p75",
        [
            event_count(np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p75"]) / dur
            for p in (1, 2, 3)
        ],
    )

    collect(
        "mag_burst_rate_p90",
        [
            event_count(np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p90"]) / dur
            for p in (1, 2, 3)
        ],
    )

    # ------------------------------------------------------------
    # Spectral features
    # ------------------------------------------------------------

    collect("magnitude_entropy", [spectral_entropy(mag_e[p], fs) for p in (1, 2, 3)])
    collect("magnitude_low_band", [band_ratio(mag_e[p], fs, *LOW_BAND) for p in (1, 2, 3)])
    collect("magnitude_mid_band", [band_ratio(mag_e[p], fs, *MID_BAND) for p in (1, 2, 3)])
    collect("magnitude_high_band", [band_ratio(mag_e[p], fs, *HIGH_BAND) for p in (1, 2, 3)])

    collect("heading_vel_entropy", [spectral_entropy(heading_vel[p], fs) for p in (1, 2, 3)])
    collect("heading_vel_low_band", [band_ratio(heading_vel[p], fs, *LOW_BAND) for p in (1, 2, 3)])
    collect("heading_vel_mid_band", [band_ratio(heading_vel[p], fs, *MID_BAND) for p in (1, 2, 3)])
    collect("heading_vel_high_band", [band_ratio(heading_vel[p], fs, *HIGH_BAND) for p in (1, 2, 3)])

    # ------------------------------------------------------------
    # Active-person-count features from magnetometer
    # ------------------------------------------------------------

    heading_active = np.stack(
        [
            (np.abs(heading_vel[p]) > sess[p]["heading_vel_p75"]).astype(int)
            for p in (1, 2, 3)
        ],
        axis=0,
    )

    mag_active = np.stack(
        [
            (np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p75"]).astype(int)
            for p in (1, 2, 3)
        ],
        axis=0,
    )

    for name, count in [
        ("heading_active_count", heading_active.sum(axis=0)),
        ("mag_active_count", mag_active.sum(axis=0)),
    ]:
        row[f"mag_{name}_mean"] = safe_mean(count)
        row[f"mag_{name}_std"] = safe_std(count)
        row[f"mag_{name}_exactly1_frac"] = float(np.mean(count == 1))
        row[f"mag_{name}_atleast2_frac"] = float(np.mean(count >= 2))
        row[f"mag_{name}_all3_frac"] = float(np.mean(count == 3))
        row[f"mag_{name}_switch_rate"] = event_count(np.r_[False, np.diff(count) != 0]) / dur

    # ------------------------------------------------------------
    # Cross-person magnetometer features
    # ------------------------------------------------------------

    pair_mag_corrs = []
    pair_heading_vel_corrs = []
    pair_heading_diffs_mean = []
    pair_heading_diffs_std = []
    pair_mag_lagcorrs = []
    pair_heading_vel_lagcorrs = []

    max_lag_steps = int(1.0 * fs)

    for a, b in PAIRS:
        pair_mag_corrs.append(corr_safe(mag_e[a], mag_e[b]))
        pair_heading_vel_corrs.append(corr_safe(heading_vel[a], heading_vel[b]))

        hdiff = np.abs(circular_diff(heading[a], heading[b]))

        pair_heading_diffs_mean.append(safe_mean(hdiff))
        pair_heading_diffs_std.append(safe_std(hdiff))

        pair_mag_lagcorrs.append(max_lag_corr(mag_e[a], mag_e[b], max_lag_steps))
        pair_heading_vel_lagcorrs.append(max_lag_corr(heading_vel[a], heading_vel[b], max_lag_steps))

    aggregate(row, "mag_pair_magnitude_corr", pair_mag_corrs)
    aggregate(row, "mag_pair_heading_vel_corr", pair_heading_vel_corrs)
    aggregate(row, "mag_pair_heading_diff_mean", pair_heading_diffs_mean)
    aggregate(row, "mag_pair_heading_diff_std", pair_heading_diffs_std)
    aggregate(row, "mag_pair_magnitude_lagcorr", pair_mag_lagcorrs)
    aggregate(row, "mag_pair_heading_vel_lagcorr", pair_heading_vel_lagcorrs)

    return row


# ================================================================
# Main build
# ================================================================

oe9 = pd.read_csv(OE9_PATH).copy()
files = discover_openearable(INPUT_DIR)

print("=" * 100)
print("BUILDING OE10 = OE9 + MAGNETOMETER")
print("=" * 100)

print("OE9 shape:", oe9.shape)
print("Groups:", sorted(oe9["group"].unique()))

mag_rows = []

for group, gwin in oe9.groupby("group"):
    group = int(group)

    print("\nGroup", group)

    if group not in files:
        print("  missing raw OpenEarable file")
        continue

    raw = load_mag(files[group])
    sess = mag_session_baseline(raw)

    print(
        f"  raw time: {raw['t'].min():.3f} -> {raw['t'].max():.3f} | "
        f"windows: {len(gwin)}"
    )

    for _, w in gwin.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        feats = extract_mag_features(raw, ws, we, sess)

        feats["group"] = group
        feats["window_start"] = ws
        feats["window_end"] = we

        mag_rows.append(feats)

    del raw
    gc.collect()

mag_df = pd.DataFrame(mag_rows)

oe10 = oe9.merge(
    mag_df,
    on=["group", "window_start", "window_end"],
    how="left",
)

OUT_PATH = f"{OUT_DIR}/interaction_oe10_10s.csv"
oe10.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 100)
print("SAVED OE10")
print("=" * 100)

print(OUT_PATH)
print("OE10 shape:", oe10.shape)
print("Mag features:", len([c for c in oe10.columns if c.startswith("mag_")]))
print("Old ear_ features:", len([c for c in oe10.columns if c.startswith("ear_")]))
print("OE9 oe_ features:", len([c for c in oe10.columns if c.startswith("oe_")]))


# --- CELL 18 (code cell #16) ---
# ================================================================
# OPTI2 — RICH OPTITRACK FEATURE GENERATION, ROBUST VERSION
#
# This version automatically chooses the better OptiTrack source file:
#   1) ALL_MODEL_READY_FILES_IDENTITY_FIXED
#   2) ALL_MODEL_READY_FILES
#
# It chooses the file with more usable landmark coordinate data
# inside the actual 10s windows.
#
# Output:
#   /content/drive/MyDrive/thesis/data/INTERACTION_OPTI2/interaction_opti2_10s.csv
# ================================================================

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# Paths
# ================================================================

ENG7_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG7", "interaction_eng7_10s.csv")
REC_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3", "eng3_recognition_3class_core_features.csv")

OPTI_DIR_ID_FIXED = RAW_DIR
OPTI_DIR_OLD = os.path.join(SOURCE_DATA_ROOT, "ALL_MODEL_READY_FILES")

OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_OPTI2")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = f"{OUT_DIR}/interaction_opti2_10s.csv"
SOURCE_REPORT_PATH = f"{OUT_DIR}/opti2_source_selection_report.csv"

CORE = ["co_building", "co_merging", "conversation"]
GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]


# ================================================================
# Helpers
# ================================================================

def get_num_col(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy()
    return np.full(len(df), np.nan)


def safe_stats(out, prefix, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    stats = ["mean", "std", "min", "max", "range", "median", "iqr", "p10", "p90"]

    if len(arr) == 0:
        for s in stats:
            out[f"{prefix}_{s}"] = np.nan
        return

    out[f"{prefix}_mean"] = np.nanmean(arr)
    out[f"{prefix}_std"] = np.nanstd(arr)
    out[f"{prefix}_min"] = np.nanmin(arr)
    out[f"{prefix}_max"] = np.nanmax(arr)
    out[f"{prefix}_range"] = np.nanmax(arr) - np.nanmin(arr)
    out[f"{prefix}_median"] = np.nanmedian(arr)
    out[f"{prefix}_iqr"] = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
    out[f"{prefix}_p10"] = np.nanpercentile(arr, 10)
    out[f"{prefix}_p90"] = np.nanpercentile(arr, 90)


def safe_fraction(out, prefix, mask):
    mask = np.asarray(mask)
    if len(mask) == 0:
        out[prefix] = np.nan
    else:
        out[prefix] = np.mean(mask)


def row_nanmin(A):
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmin(A[mask], axis=1)
    return out


def row_nanmax(A):
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmax(A[mask], axis=1)
    return out


def row_nanmean(A):
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmean(A[mask], axis=1)
    return out


def row_nanstd(A):
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanstd(A[mask], axis=1)
    return out


def centroid_from_stack(A):
    A = np.asarray(A, dtype=float)
    n, people, dim = A.shape
    out = np.full((n, dim), np.nan)

    for i in range(n):
        valid_people = np.all(np.isfinite(A[i]), axis=1)
        if valid_people.any():
            out[i] = np.nanmean(A[i, valid_people, :], axis=0)

    return out


def compute_speed(pos, t):
    pos = np.asarray(pos, dtype=float)
    t = np.asarray(t, dtype=float)

    valid = np.all(np.isfinite(pos), axis=1) & np.isfinite(t)
    pos = pos[valid]
    t = t[valid]

    if len(t) < 3:
        return np.array([])

    dt = np.diff(t)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)

    good = np.isfinite(dt) & np.isfinite(dp) & (dt > 1e-6)

    if good.sum() == 0:
        return np.array([])

    speed = dp[good] / dt[good]
    speed = speed[np.isfinite(speed)]

    # Remove clear tracking jumps.
    speed = speed[speed < 10.0]

    return speed


def triangle_area_2d(p1, p2, p3):
    return 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )


def add_recognition_labels(base_df, rec_df):
    labels = []

    for _, w in base_df.iterrows():
        sub = rec_df[
            (rec_df["group"] == w["group"])
            & (rec_df["mid"] >= w["window_start"])
            & (rec_df["mid"] < w["window_end"])
        ]

        labels.append(sub["recognition_label"].mode().iat[0] if len(sub) else np.nan)

    out = base_df.copy()
    out["recognition_label"] = labels
    return out


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_landmark_mapping(df):
    """
    Returns mapping:
      1 -> {"x": col, "y": col, "z": col}
      2 -> ...
      3 -> ...
    Handles common naming possibilities.
    """

    cols = list(df.columns)
    lower_to_original = {c.lower(): c for c in cols}

    mapping = {}

    candidates = {
        1: [
            ("landmark1_x", "landmark1_y", "landmark1_z"),
            ("p1_x", "p1_y", "p1_z"),
            ("participant1_x", "participant1_y", "participant1_z"),
            ("person1_x", "person1_y", "person1_z"),
        ],
        2: [
            ("landmark2_x", "landmark2_y", "landmark2_z"),
            ("p2_x", "p2_y", "p2_z"),
            ("participant2_x", "participant2_y", "participant2_z"),
            ("person2_x", "person2_y", "person2_z"),
        ],
        3: [
            ("landmark3_x", "landmark3_y", "landmark3_z"),
            ("p3_x", "p3_y", "p3_z"),
            ("participant3_x", "participant3_y", "participant3_z"),
            ("person3_x", "person3_y", "person3_z"),
        ],
    }

    for i in [1, 2, 3]:
        mapping[i] = {"x": None, "y": None, "z": None}

        for xname, yname, zname in candidates[i]:
            if (
                xname.lower() in lower_to_original
                and yname.lower() in lower_to_original
                and zname.lower() in lower_to_original
            ):
                mapping[i]["x"] = lower_to_original[xname.lower()]
                mapping[i]["y"] = lower_to_original[yname.lower()]
                mapping[i]["z"] = lower_to_original[zname.lower()]
                break

    return mapping


def find_available_col(df, i):
    candidates = [
        f"landmark{i}_available",
        f"p{i}_available",
        f"participant{i}_available",
        f"person{i}_available",
    ]

    lower_to_original = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]

    return None


def load_candidate_file(path):
    raw = pd.read_csv(path, low_memory=False)
    raw = standardize_columns(raw)

    time_col = "video_time_s" if "video_time_s" in raw.columns else "time_s"

    if time_col not in raw.columns:
        return None, None, None

    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    mapping = find_landmark_mapping(raw)

    return raw, time_col, mapping


def coordinate_score_for_windows(raw, time_col, mapping, base_g, max_windows=20):
    """
    Score candidate file by counting finite coordinate values in actual windows.
    """
    if raw is None:
        return -1

    t_raw = raw[time_col].to_numpy()
    score = 0

    sample = base_g.head(max_windows).copy()

    for _, w in sample.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        start_idx = np.searchsorted(t_raw, ws, side="left")
        end_idx = np.searchsorted(t_raw, we, side="left")

        sub = raw.iloc[start_idx:end_idx]

        for i in [1, 2, 3]:
            for axis in ["x", "y", "z"]:
                col = mapping[i][axis]
                if col is not None and col in sub.columns:
                    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy()
                    score += np.isfinite(vals).sum()

    return int(score)


def choose_best_opti_source(group, base_g):
    candidates = []

    paths = [
        f"{OPTI_DIR_ID_FIXED}/group_{group}_optitrack_model_ready.csv",
        f"{OPTI_DIR_OLD}/group_{group}_optitrack_model_ready.csv",
    ]

    for path in paths:
        if not os.path.exists(path):
            continue

        raw, time_col, mapping = load_candidate_file(path)

        score = coordinate_score_for_windows(raw, time_col, mapping, base_g)

        candidates.append({
            "path": path,
            "raw": raw,
            "time_col": time_col,
            "mapping": mapping,
            "coordinate_score": score,
        })

    if len(candidates) == 0:
        raise FileNotFoundError(f"No OptiTrack file found for group {group}")

    candidates = sorted(candidates, key=lambda x: x["coordinate_score"], reverse=True)

    return candidates[0], candidates


# ================================================================
# Window feature computation
# ================================================================

def compute_window_features(sub, group, ws, we, mapping):
    out = {
        "group": group,
        "window_start": ws,
        "window_end": we,
    }

    if len(sub) == 0:
        return out

    sub = standardize_columns(sub)

    time_col = "video_time_s" if "video_time_s" in sub.columns else "time_s"
    t = get_num_col(sub, time_col)

    # ------------------------------------------------------------
    # Availability / tracking quality
    # ------------------------------------------------------------

    for i in [1, 2, 3]:
        avail_col = find_available_col(sub, i)

        if avail_col is not None:
            vals = get_num_col(sub, avail_col)
            out[f"opti2_lm{i}_available_frac"] = np.nanmean(vals) if np.isfinite(vals).sum() else np.nan
        else:
            xcol = mapping[i]["x"]
            ycol = mapping[i]["y"]
            zcol = mapping[i]["z"]

            if xcol is not None and ycol is not None and zcol is not None:
                valid = (
                    np.isfinite(get_num_col(sub, xcol))
                    & np.isfinite(get_num_col(sub, ycol))
                    & np.isfinite(get_num_col(sub, zcol))
                )
                out[f"opti2_lm{i}_available_frac"] = np.mean(valid)
            else:
                out[f"opti2_lm{i}_available_frac"] = np.nan

    if "active_clean_landmarks" in sub.columns:
        active = get_num_col(sub, "active_clean_landmarks")
    else:
        active = np.zeros(len(sub))
        for i in [1, 2, 3]:
            xcol = mapping[i]["x"]
            ycol = mapping[i]["y"]
            zcol = mapping[i]["z"]

            if xcol is not None and ycol is not None and zcol is not None:
                valid = (
                    np.isfinite(get_num_col(sub, xcol))
                    & np.isfinite(get_num_col(sub, ycol))
                    & np.isfinite(get_num_col(sub, zcol))
                )
                active += valid.astype(float)

    safe_stats(out, "opti2_active_landmarks", active)
    safe_fraction(out, "opti2_all3_available_frac", active >= 3)
    safe_fraction(out, "opti2_atleast2_available_frac", active >= 2)
    safe_fraction(out, "opti2_atleast1_available_frac", active >= 1)

    # ------------------------------------------------------------
    # Positions and individual movement
    # ------------------------------------------------------------

    P = {}

    for i in [1, 2, 3]:
        xcol = mapping[i]["x"]
        ycol = mapping[i]["y"]
        zcol = mapping[i]["z"]

        x = get_num_col(sub, xcol) if xcol is not None else np.full(len(sub), np.nan)
        y = get_num_col(sub, ycol) if ycol is not None else np.full(len(sub), np.nan)
        z = get_num_col(sub, zcol) if zcol is not None else np.full(len(sub), np.nan)

        pos3 = np.column_stack([x, y, z])
        pos2 = np.column_stack([x, z])

        P[i] = {
            "x": x,
            "y": y,
            "z": z,
            "pos3": pos3,
            "pos2": pos2,
        }

        safe_stats(out, f"opti2_lm{i}_x", x)
        safe_stats(out, f"opti2_lm{i}_y", y)
        safe_stats(out, f"opti2_lm{i}_z", z)

        speed2 = compute_speed(pos2, t)
        speed3 = compute_speed(pos3, t)

        safe_stats(out, f"opti2_lm{i}_speed2d", speed2)
        safe_stats(out, f"opti2_lm{i}_speed3d", speed3)

        safe_fraction(out, f"opti2_lm{i}_moving2d_frac_005", speed2 > 0.05)
        safe_fraction(out, f"opti2_lm{i}_moving2d_frac_010", speed2 > 0.10)
        safe_fraction(out, f"opti2_lm{i}_moving2d_frac_020", speed2 > 0.20)

    # ------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------

    pair_names = [(1, 2), (1, 3), (2, 3)]

    all_pair_d2 = []
    all_pair_d3 = []

    for a, b in pair_names:
        valid2 = (
            np.all(np.isfinite(P[a]["pos2"]), axis=1)
            & np.all(np.isfinite(P[b]["pos2"]), axis=1)
        )

        valid3 = (
            np.all(np.isfinite(P[a]["pos3"]), axis=1)
            & np.all(np.isfinite(P[b]["pos3"]), axis=1)
        )

        d2 = np.linalg.norm(P[a]["pos2"] - P[b]["pos2"], axis=1)
        d3 = np.linalg.norm(P[a]["pos3"] - P[b]["pos3"], axis=1)

        d2[~valid2] = np.nan
        d3[~valid3] = np.nan

        safe_stats(out, f"opti2_pair{a}{b}_dist2d", d2)
        safe_stats(out, f"opti2_pair{a}{b}_dist3d", d3)

        safe_fraction(out, f"opti2_pair{a}{b}_valid_frac", np.isfinite(d2))

        if len(t) > 2:
            dt = np.diff(t)

            dd2 = np.abs(np.diff(d2))
            dd3 = np.abs(np.diff(d3))

            good2 = np.isfinite(dd2) & np.isfinite(dt) & (dt > 1e-6)
            good3 = np.isfinite(dd3) & np.isfinite(dt) & (dt > 1e-6)

            d2_change = dd2[good2] / dt[good2]
            d3_change = dd3[good3] / dt[good3]

            d2_change = d2_change[np.isfinite(d2_change) & (d2_change < 10.0)]
            d3_change = d3_change[np.isfinite(d3_change) & (d3_change < 10.0)]
        else:
            d2_change = np.array([])
            d3_change = np.array([])

        safe_stats(out, f"opti2_pair{a}{b}_dist2d_change_rate", d2_change)
        safe_stats(out, f"opti2_pair{a}{b}_dist3d_change_rate", d3_change)

        all_pair_d2.append(d2)
        all_pair_d3.append(d3)

    all_pair_d2 = np.vstack(all_pair_d2).T
    all_pair_d3 = np.vstack(all_pair_d3).T

    safe_stats(out, "opti2_all_pairs_dist2d_flat", all_pair_d2.ravel())
    safe_stats(out, "opti2_all_pairs_dist3d_flat", all_pair_d3.ravel())

    nearest2 = row_nanmin(all_pair_d2)
    farthest2 = row_nanmax(all_pair_d2)
    mean2 = row_nanmean(all_pair_d2)
    std2 = row_nanstd(all_pair_d2)
    range2 = farthest2 - nearest2

    nearest3 = row_nanmin(all_pair_d3)
    farthest3 = row_nanmax(all_pair_d3)
    mean3 = row_nanmean(all_pair_d3)
    std3 = row_nanstd(all_pair_d3)
    range3 = farthest3 - nearest3

    safe_stats(out, "opti2_nearest_pair_dist2d", nearest2)
    safe_stats(out, "opti2_farthest_pair_dist2d", farthest2)
    safe_stats(out, "opti2_all_pairs_dist2d_mean_over_time", mean2)
    safe_stats(out, "opti2_all_pairs_dist2d_std_over_time", std2)
    safe_stats(out, "opti2_all_pairs_dist2d_range_over_time", range2)

    safe_stats(out, "opti2_nearest_pair_dist3d", nearest3)
    safe_stats(out, "opti2_farthest_pair_dist3d", farthest3)
    safe_stats(out, "opti2_all_pairs_dist3d_mean_over_time", mean3)
    safe_stats(out, "opti2_all_pairs_dist3d_std_over_time", std3)
    safe_stats(out, "opti2_all_pairs_dist3d_range_over_time", range3)

    # ------------------------------------------------------------
    # Centroid and spread
    # ------------------------------------------------------------

    pos2_stack = np.stack([P[1]["pos2"], P[2]["pos2"], P[3]["pos2"]], axis=1)
    pos3_stack = np.stack([P[1]["pos3"], P[2]["pos3"], P[3]["pos3"]], axis=1)

    centroid2 = centroid_from_stack(pos2_stack)
    centroid3 = centroid_from_stack(pos3_stack)

    safe_stats(out, "opti2_centroid2d_x", centroid2[:, 0])
    safe_stats(out, "opti2_centroid2d_z", centroid2[:, 1])

    safe_stats(out, "opti2_centroid3d_x", centroid3[:, 0])
    safe_stats(out, "opti2_centroid3d_y", centroid3[:, 1])
    safe_stats(out, "opti2_centroid3d_z", centroid3[:, 2])

    centroid_speed2 = compute_speed(centroid2, t)
    centroid_speed3 = compute_speed(centroid3, t)

    safe_stats(out, "opti2_centroid_speed2d", centroid_speed2)
    safe_stats(out, "opti2_centroid_speed3d", centroid_speed3)

    dist_to_centroid2 = np.linalg.norm(pos2_stack - centroid2[:, None, :], axis=2)
    dist_to_centroid3 = np.linalg.norm(pos3_stack - centroid3[:, None, :], axis=2)

    safe_stats(out, "opti2_spread2d_mean", row_nanmean(dist_to_centroid2))
    safe_stats(out, "opti2_spread2d_std", row_nanstd(dist_to_centroid2))
    safe_stats(out, "opti2_spread2d_max", row_nanmax(dist_to_centroid2))

    safe_stats(out, "opti2_spread3d_mean", row_nanmean(dist_to_centroid3))
    safe_stats(out, "opti2_spread3d_std", row_nanstd(dist_to_centroid3))
    safe_stats(out, "opti2_spread3d_max", row_nanmax(dist_to_centroid3))

    # ------------------------------------------------------------
    # Formation triangle
    # ------------------------------------------------------------

    all3_valid_2d = (
        np.all(np.isfinite(P[1]["pos2"]), axis=1)
        & np.all(np.isfinite(P[2]["pos2"]), axis=1)
        & np.all(np.isfinite(P[3]["pos2"]), axis=1)
    )

    area2 = np.full(len(sub), np.nan)
    perimeter2 = np.full(len(sub), np.nan)

    if all3_valid_2d.sum() > 0:
        p1 = P[1]["pos2"][all3_valid_2d]
        p2 = P[2]["pos2"][all3_valid_2d]
        p3 = P[3]["pos2"][all3_valid_2d]

        area_vals = triangle_area_2d(p1, p2, p3)

        d12 = np.linalg.norm(p1 - p2, axis=1)
        d13 = np.linalg.norm(p1 - p3, axis=1)
        d23 = np.linalg.norm(p2 - p3, axis=1)

        perimeter_vals = d12 + d13 + d23

        area2[all3_valid_2d] = area_vals
        perimeter2[all3_valid_2d] = perimeter_vals

    compactness = area2 / ((perimeter2 ** 2) + 1e-9)

    safe_stats(out, "opti2_triangle_area2d", area2)
    safe_stats(out, "opti2_triangle_perimeter2d", perimeter2)
    safe_stats(out, "opti2_triangle_compactness2d", compactness)

    # ------------------------------------------------------------
    # Movement asymmetry
    # ------------------------------------------------------------

    speed_means = []

    for i in [1, 2, 3]:
        speed = compute_speed(P[i]["pos2"], t)
        speed_means.append(np.nanmean(speed) if len(speed) else np.nan)

    speed_means = np.asarray(speed_means, dtype=float)

    safe_stats(out, "opti2_individual_speed2d_mean_across_people", speed_means)

    return out


# ================================================================
# Load base windows from ENG7
# ================================================================

eng7 = pd.read_csv(ENG7_PATH).copy()

rec = pd.read_csv(REC_PATH)[
    ["group", "window_start", "window_end", "recognition_label"]
].copy()

rec = rec[rec["recognition_label"].isin(CORE)].copy()
rec["mid"] = (rec["window_start"] + rec["window_end"]) / 2.0

eng7 = add_recognition_labels(eng7, rec)

eng7 = (
    eng7[eng7["recognition_label"].isin(CORE)]
    .dropna(subset=["recognition_label"])
    .reset_index(drop=True)
)

eng7["window_mid"] = (eng7["window_start"] + eng7["window_end"]) / 2.0
group_start = eng7.groupby("group")["window_mid"].transform("min")
eng7["elapsed_min"] = (eng7["window_mid"] - group_start) / 60.0

old_eng7_opti = [c for c in eng7.columns if c.startswith("opti_")]

print("=" * 100)
print("BASE WINDOWS")
print("=" * 100)
print("Shape:", eng7.shape)
print("Class counts:")
print(eng7["recognition_label"].value_counts().to_string())

print("\nOld ENG7 proximity columns:")
for c in old_eng7_opti:
    print(" ", c)


# ================================================================
# Generate features
# ================================================================

all_rows = []
source_rows = []

for group in GROUPS:
    print("\n" + "=" * 100)
    print(f"Processing group {group}")
    print("=" * 100)

    base_g = eng7[eng7["group"] == group].copy().reset_index(drop=True)

    chosen, candidates = choose_best_opti_source(group, base_g)

    raw = chosen["raw"]
    time_col = chosen["time_col"]
    mapping = chosen["mapping"]

    print("Candidate source scores:")
    for cand in candidates:
        print(" ", cand["coordinate_score"], "|", cand["path"])

    print("\nChosen:", chosen["path"])
    print("Coordinate score:", chosen["coordinate_score"])
    print("Time column:", time_col)
    print("Mapping:")
    print(mapping)

    t_raw = raw[time_col].to_numpy()

    print("Raw rows:", len(raw))
    print("Windows:", len(base_g))
    print("Raw time range:", np.nanmin(t_raw), "to", np.nanmax(t_raw))
    print("Window range:", base_g["window_start"].min(), "to", base_g["window_end"].max())

    source_rows.append({
        "group": group,
        "chosen_path": chosen["path"],
        "coordinate_score": chosen["coordinate_score"],
        "time_col": time_col,
        "lm1_x": mapping[1]["x"],
        "lm1_y": mapping[1]["y"],
        "lm1_z": mapping[1]["z"],
        "lm2_x": mapping[2]["x"],
        "lm2_y": mapping[2]["y"],
        "lm2_z": mapping[2]["z"],
        "lm3_x": mapping[3]["x"],
        "lm3_y": mapping[3]["y"],
        "lm3_z": mapping[3]["z"],
    })

    for idx, w in base_g.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        start_idx = np.searchsorted(t_raw, ws, side="left")
        end_idx = np.searchsorted(t_raw, we, side="left")

        sub = raw.iloc[start_idx:end_idx].copy()

        feat = compute_window_features(
            sub=sub,
            group=group,
            ws=ws,
            we=we,
            mapping=mapping,
        )

        feat["recognition_label"] = w["recognition_label"]
        feat["elapsed_min"] = w["elapsed_min"]

        for c in old_eng7_opti:
            feat[c] = w[c]

        all_rows.append(feat)

        if (idx + 1) % 50 == 0:
            print(f"  completed {idx + 1}/{len(base_g)} windows", flush=True)


# ================================================================
# Save
# ================================================================

opti2 = pd.DataFrame(all_rows)
source_report = pd.DataFrame(source_rows)

all_nan_cols = [c for c in opti2.columns if opti2[c].isna().all()]

if all_nan_cols:
    print("\nDropping all-NaN columns:", len(all_nan_cols))
    opti2 = opti2.drop(columns=all_nan_cols)

opti2.to_csv(OUT_PATH, index=False)
source_report.to_csv(SOURCE_REPORT_PATH, index=False)

opti2_features = [c for c in opti2.columns if c.startswith("opti2_")]
old_prox_features = [c for c in opti2.columns if c.startswith("opti_") and not c.startswith("opti2_")]

print("\n" + "=" * 100)
print("OPTI2 SAVED")
print("=" * 100)
print("Path:", OUT_PATH)
print("Shape:", opti2.shape)

print("\nSource report:")
display(source_report)

print("\nClass counts:")
print(opti2["recognition_label"].value_counts().to_string())

print("\nNumber of OPTI2 rich features:", len(opti2_features))
print("Number of old ENG7 proximity features kept:", len(old_prox_features))

print("\nFirst 100 OPTI2 features:")
for c in opti2_features[:100]:
    print(" ", c)

print("\nOld ENG7 proximity columns kept:")
for c in old_prox_features:
    print(" ", c)

print("\nSaved source selection report:")
print(SOURCE_REPORT_PATH)


# --- CELL 19 (code cell #17) ---
# ================================================================
# XSENS2 — RICH XSENS FEATURE GENERATION
#
# Builds rich 10s Xsens features from model-ready Xsens files.
#
# Raw Xsens columns expected:
#   p1_euler_x/y/z, p1_acc_x/y/z, p1_gyr_x/y/z
#   p2_euler_x/y/z, p2_acc_x/y/z, p2_gyr_x/y/z
#   p3_euler_x/y/z, p3_acc_x/y/z, p3_gyr_x/y/z
#
# Features:
#   - acceleration energy
#   - dynamic acceleration
#   - gyroscope energy
#   - jerk / angular jerk
#   - Euler posture/range/rate
#   - movement burst fractions
#   - active participant count
#   - cross-person synchrony
#   - movement asymmetry
#   - spectral entropy / band ratios
#
# Output:
#   /content/drive/MyDrive/thesis/data/INTERACTION_XSENS2/interaction_xsens2_10s.csv
# ================================================================

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# Paths
# ================================================================

ENG7_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG7", "interaction_eng7_10s.csv")
REC_PATH = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3", "eng3_recognition_3class_core_features.csv")

XSENS_DIR_ID_FIXED = RAW_DIR
XSENS_DIR_OLD = os.path.join(SOURCE_DATA_ROOT, "ALL_MODEL_READY_FILES")
ROOT = SOURCE_DATA_ROOT

OUT_DIR = os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_XSENS2")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = f"{OUT_DIR}/interaction_xsens2_10s.csv"
SOURCE_REPORT_PATH = f"{OUT_DIR}/xsens2_source_selection_report.csv"

CORE = ["co_building", "co_merging", "conversation"]
GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]


# ================================================================
# Helpers
# ================================================================

def get_num_col(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy()
    return np.full(len(df), np.nan)


def safe_stats(out, prefix, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    stats = ["mean", "std", "min", "max", "range", "median", "iqr", "p10", "p90"]

    if len(arr) == 0:
        for s in stats:
            out[f"{prefix}_{s}"] = np.nan
        return

    out[f"{prefix}_mean"] = np.nanmean(arr)
    out[f"{prefix}_std"] = np.nanstd(arr)
    out[f"{prefix}_min"] = np.nanmin(arr)
    out[f"{prefix}_max"] = np.nanmax(arr)
    out[f"{prefix}_range"] = np.nanmax(arr) - np.nanmin(arr)
    out[f"{prefix}_median"] = np.nanmedian(arr)
    out[f"{prefix}_iqr"] = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
    out[f"{prefix}_p10"] = np.nanpercentile(arr, 10)
    out[f"{prefix}_p90"] = np.nanpercentile(arr, 90)


def safe_fraction(out, prefix, mask):
    mask = np.asarray(mask)

    if len(mask) == 0:
        out[prefix] = np.nan
    else:
        out[prefix] = np.mean(mask)


def add_recognition_labels(base_df, rec_df):
    labels = []

    for _, w in base_df.iterrows():
        sub = rec_df[
            (rec_df["group"] == w["group"])
            & (rec_df["mid"] >= w["window_start"])
            & (rec_df["mid"] < w["window_end"])
        ]

        labels.append(sub["recognition_label"].mode().iat[0] if len(sub) else np.nan)

    out = base_df.copy()
    out["recognition_label"] = labels
    return out


def clean_signal(x, kind):
    """
    Removes impossible corrupted values.
    Xsens here seems to be:
      acc roughly m/s^2
      gyr roughly deg/s
      euler roughly degrees

    Some files contain huge corrupt values, so we mask them.
    """
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan

    if kind == "acc":
        x[np.abs(x) > 100.0] = np.nan

    elif kind == "gyr":
        x[np.abs(x) > 2000.0] = np.nan

    elif kind == "euler":
        x[np.abs(x) > 360.0] = np.nan

    return x


def unwrap_degrees(x):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)

    valid = np.isfinite(x)

    if valid.sum() < 2:
        return out

    out[valid] = np.rad2deg(np.unwrap(np.deg2rad(x[valid])))

    return out


def vector_norm(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    out = np.full(len(x), np.nan)
    out[valid] = np.sqrt(x[valid] ** 2 + y[valid] ** 2 + z[valid] ** 2)

    return out


def derivative_norm(x, y, z, t, max_value=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(t)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    t = t[valid]

    if len(t) < 3:
        return np.array([])

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    good = np.isfinite(dt) & (dt > 1e-6)

    if good.sum() == 0:
        return np.array([])

    val = np.sqrt(dx[good] ** 2 + dy[good] ** 2 + dz[good] ** 2) / dt[good]
    val = val[np.isfinite(val)]

    if max_value is not None:
        val = val[val < max_value]

    return val


def derivative_1d(x, t, max_value=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    valid = np.isfinite(x) & np.isfinite(t)

    x = x[valid]
    t = t[valid]

    if len(t) < 3:
        return np.array([])

    dt = np.diff(t)
    dx = np.diff(x)

    good = np.isfinite(dt) & (dt > 1e-6)

    if good.sum() == 0:
        return np.array([])

    val = np.abs(dx[good]) / dt[good]
    val = val[np.isfinite(val)]

    if max_value is not None:
        val = val[val < max_value]

    return val


def interp_nan(y):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n == 0:
        return y

    idx = np.arange(n)
    valid = np.isfinite(y)

    if valid.sum() < 3:
        return None

    y2 = y.copy()
    y2[~valid] = np.interp(idx[~valid], idx[valid], y[valid])

    return y2


def spectral_features(out, prefix, y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    valid_t = np.isfinite(t)

    if valid_t.sum() < 16:
        out[f"{prefix}_spec_entropy"] = np.nan
        out[f"{prefix}_spec_low_ratio"] = np.nan
        out[f"{prefix}_spec_mid_ratio"] = np.nan
        out[f"{prefix}_spec_high_ratio"] = np.nan
        return

    y2 = interp_nan(y)

    if y2 is None or len(y2) < 16:
        out[f"{prefix}_spec_entropy"] = np.nan
        out[f"{prefix}_spec_low_ratio"] = np.nan
        out[f"{prefix}_spec_mid_ratio"] = np.nan
        out[f"{prefix}_spec_high_ratio"] = np.nan
        return

    t2 = t[np.isfinite(t)]
    dt = np.nanmedian(np.diff(t2))

    if not np.isfinite(dt) or dt <= 0:
        out[f"{prefix}_spec_entropy"] = np.nan
        out[f"{prefix}_spec_low_ratio"] = np.nan
        out[f"{prefix}_spec_mid_ratio"] = np.nan
        out[f"{prefix}_spec_high_ratio"] = np.nan
        return

    fs = 1.0 / dt

    y2 = y2 - np.nanmean(y2)

    if np.nanstd(y2) < 1e-12:
        out[f"{prefix}_spec_entropy"] = 0.0
        out[f"{prefix}_spec_low_ratio"] = 0.0
        out[f"{prefix}_spec_mid_ratio"] = 0.0
        out[f"{prefix}_spec_high_ratio"] = 0.0
        return

    fft = np.fft.rfft(y2)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(y2), d=dt)

    # remove DC
    power = power[1:]
    freqs = freqs[1:]

    total = np.nansum(power)

    if total <= 1e-12:
        out[f"{prefix}_spec_entropy"] = 0.0
        out[f"{prefix}_spec_low_ratio"] = 0.0
        out[f"{prefix}_spec_mid_ratio"] = 0.0
        out[f"{prefix}_spec_high_ratio"] = 0.0
        return

    p = power / total
    p = p[p > 0]

    entropy = -np.sum(p * np.log(p)) / np.log(len(p)) if len(p) > 1 else 0.0

    low = np.sum(power[(freqs >= 0.1) & (freqs < 0.5)]) / total
    mid = np.sum(power[(freqs >= 0.5) & (freqs < 2.0)]) / total
    high = np.sum(power[(freqs >= 2.0) & (freqs < 6.0)]) / total

    out[f"{prefix}_spec_entropy"] = entropy
    out[f"{prefix}_spec_low_ratio"] = low
    out[f"{prefix}_spec_mid_ratio"] = mid
    out[f"{prefix}_spec_high_ratio"] = high


def corr_feature(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)

    if valid.sum() < 10:
        return np.nan

    aa = a[valid]
    bb = b[valid]

    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return np.nan

    return np.corrcoef(aa, bb)[0, 1]


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def candidate_paths_for_group(group):
    paths = [
        f"{XSENS_DIR_ID_FIXED}/group_{group}_xsens_model_ready.csv",
        f"{XSENS_DIR_OLD}/group_{group}_xsens_model_ready.csv",
        f"{ROOT}/group_{group}/xsens/model_ready/group_{group}_xsens_model_ready.csv",
    ]

    return [p for p in paths if os.path.exists(p)]


def load_xsens_file(path):
    raw = pd.read_csv(path, low_memory=False)
    raw = standardize_columns(raw)

    # Prefer video_time_s when available because ENG7 windows are video-time based.
    if "video_time_s" in raw.columns:
        time_col = "video_time_s"
    elif "time_s" in raw.columns:
        time_col = "time_s"
    else:
        return None, None

    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    return raw, time_col


def sanity_score_for_source(raw, time_col, base_g):
    """
    Source score = number of sane finite sensor values inside the actual windows.
    This avoids selecting a file with corrupted huge values or poor overlap.
    """
    if raw is None or time_col is None:
        return -1

    needed = []

    for p in [1, 2, 3]:
        for kind in ["euler", "acc", "gyr"]:
            for axis in ["x", "y", "z"]:
                c = f"p{p}_{kind}_{axis}"
                if c in raw.columns:
                    needed.append(c)

    if len(needed) == 0:
        return -1

    t_raw = raw[time_col].to_numpy()
    score = 0

    for _, w in base_g.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        start_idx = np.searchsorted(t_raw, ws, side="left")
        end_idx = np.searchsorted(t_raw, we, side="left")

        sub = raw.iloc[start_idx:end_idx]

        if len(sub) == 0:
            continue

        for c in needed:
            if "_acc_" in c:
                kind = "acc"
            elif "_gyr_" in c:
                kind = "gyr"
            else:
                kind = "euler"

            x = clean_signal(get_num_col(sub, c), kind)
            score += np.isfinite(x).sum()

    return int(score)


def choose_best_xsens_source(group, base_g):
    rows = []

    for path in candidate_paths_for_group(group):
        raw, time_col = load_xsens_file(path)

        score = sanity_score_for_source(raw, time_col, base_g)

        rows.append({
            "path": path,
            "raw": raw,
            "time_col": time_col,
            "sanity_score": score,
        })

    if len(rows) == 0:
        raise FileNotFoundError(f"No Xsens file found for group {group}")

    rows = sorted(rows, key=lambda r: r["sanity_score"], reverse=True)

    return rows[0], rows


# ================================================================
# Main window feature function
# ================================================================

def compute_xsens_window_features(sub, group, ws, we, time_col):
    out = {
        "group": group,
        "window_start": ws,
        "window_end": we,
    }

    if len(sub) == 0:
        return out

    sub = standardize_columns(sub)

    t = get_num_col(sub, time_col)

    participant_data = {}

    # ------------------------------------------------------------
    # Per-participant features
    # ------------------------------------------------------------

    for p in [1, 2, 3]:
        prefix = f"xsens2_p{p}"

        # Availability
        avail_col = f"p{p}_xsens_available"

        if avail_col in sub.columns:
            avail = get_num_col(sub, avail_col)
            out[f"{prefix}_available_frac"] = np.nanmean(avail) if np.isfinite(avail).sum() else np.nan
        else:
            # infer availability from acc/gyr/euler finite values
            possible_cols = [
                f"p{p}_acc_x", f"p{p}_acc_y", f"p{p}_acc_z",
                f"p{p}_gyr_x", f"p{p}_gyr_y", f"p{p}_gyr_z",
                f"p{p}_euler_x", f"p{p}_euler_y", f"p{p}_euler_z",
            ]

            finite_any = np.zeros(len(sub), dtype=bool)

            for c in possible_cols:
                if c in sub.columns:
                    kind = "acc" if "_acc_" in c else "gyr" if "_gyr_" in c else "euler"
                    finite_any |= np.isfinite(clean_signal(get_num_col(sub, c), kind))

            out[f"{prefix}_available_frac"] = np.mean(finite_any)

        # Raw signals
        acc_x = clean_signal(get_num_col(sub, f"p{p}_acc_x"), "acc")
        acc_y = clean_signal(get_num_col(sub, f"p{p}_acc_y"), "acc")
        acc_z = clean_signal(get_num_col(sub, f"p{p}_acc_z"), "acc")

        gyr_x = clean_signal(get_num_col(sub, f"p{p}_gyr_x"), "gyr")
        gyr_y = clean_signal(get_num_col(sub, f"p{p}_gyr_y"), "gyr")
        gyr_z = clean_signal(get_num_col(sub, f"p{p}_gyr_z"), "gyr")

        eul_x = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_x"), "euler"))
        eul_y = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_y"), "euler"))
        eul_z = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_z"), "euler"))

        # Axis-level stats
        for axis_name, arr in [
            ("acc_x", acc_x), ("acc_y", acc_y), ("acc_z", acc_z),
            ("gyr_x", gyr_x), ("gyr_y", gyr_y), ("gyr_z", gyr_z),
            ("euler_x", eul_x), ("euler_y", eul_y), ("euler_z", eul_z),
        ]:
            safe_stats(out, f"{prefix}_{axis_name}", arr)

        # Vector magnitudes
        acc_norm = vector_norm(acc_x, acc_y, acc_z)
        gyr_norm = vector_norm(gyr_x, gyr_y, gyr_z)

        # Dynamic acceleration: remove window median per axis
        acc_x_dyn = acc_x - np.nanmedian(acc_x)
        acc_y_dyn = acc_y - np.nanmedian(acc_y)
        acc_z_dyn = acc_z - np.nanmedian(acc_z)

        acc_dyn_norm = vector_norm(acc_x_dyn, acc_y_dyn, acc_z_dyn)

        # Jerk and angular jerk
        jerk_norm = derivative_norm(acc_x_dyn, acc_y_dyn, acc_z_dyn, t, max_value=500.0)
        angular_jerk_norm = derivative_norm(gyr_x, gyr_y, gyr_z, t, max_value=10000.0)

        # Euler rate
        eul_rate_x = derivative_1d(eul_x, t, max_value=1000.0)
        eul_rate_y = derivative_1d(eul_y, t, max_value=1000.0)
        eul_rate_z = derivative_1d(eul_z, t, max_value=1000.0)

        # For same-length euler-rate proxy, use gradient style on unwrapped euler
        eul_rate_norm = derivative_norm(eul_x, eul_y, eul_z, t, max_value=2000.0)

        safe_stats(out, f"{prefix}_acc_norm", acc_norm)
        safe_stats(out, f"{prefix}_acc_dyn_norm", acc_dyn_norm)
        safe_stats(out, f"{prefix}_gyr_norm", gyr_norm)
        safe_stats(out, f"{prefix}_jerk_norm", jerk_norm)
        safe_stats(out, f"{prefix}_angular_jerk_norm", angular_jerk_norm)

        safe_stats(out, f"{prefix}_euler_rate_x", eul_rate_x)
        safe_stats(out, f"{prefix}_euler_rate_y", eul_rate_y)
        safe_stats(out, f"{prefix}_euler_rate_z", eul_rate_z)
        safe_stats(out, f"{prefix}_euler_rate_norm", eul_rate_norm)

        # Activity/burst fractions
        safe_fraction(out, f"{prefix}_acc_dyn_gt_0p5_frac", acc_dyn_norm > 0.5)
        safe_fraction(out, f"{prefix}_acc_dyn_gt_1p0_frac", acc_dyn_norm > 1.0)
        safe_fraction(out, f"{prefix}_acc_dyn_gt_2p0_frac", acc_dyn_norm > 2.0)

        safe_fraction(out, f"{prefix}_gyr_gt_30_frac", gyr_norm > 30.0)
        safe_fraction(out, f"{prefix}_gyr_gt_60_frac", gyr_norm > 60.0)
        safe_fraction(out, f"{prefix}_gyr_gt_100_frac", gyr_norm > 100.0)

        # Spectral features
        spectral_features(out, f"{prefix}_acc_dyn_norm", acc_dyn_norm, t)
        spectral_features(out, f"{prefix}_gyr_norm", gyr_norm, t)

        participant_data[p] = {
            "acc_dyn_norm": acc_dyn_norm,
            "gyr_norm": gyr_norm,
            "acc_norm": acc_norm,
            "euler_x": eul_x,
            "euler_y": eul_y,
            "euler_z": eul_z,
            "acc_dyn_mean": np.nanmean(acc_dyn_norm),
            "gyr_mean": np.nanmean(gyr_norm),
            "jerk_mean": np.nanmean(jerk_norm) if len(jerk_norm) else np.nan,
            "angular_jerk_mean": np.nanmean(angular_jerk_norm) if len(angular_jerk_norm) else np.nan,
        }

    # ------------------------------------------------------------
    # Group-level active counts
    # ------------------------------------------------------------

    acc_active = []
    gyr_active = []

    for p in [1, 2, 3]:
        acc_active.append(participant_data[p]["acc_dyn_norm"] > 1.0)
        gyr_active.append(participant_data[p]["gyr_norm"] > 60.0)

    acc_active = np.vstack(acc_active).T
    gyr_active = np.vstack(gyr_active).T

    acc_count = np.sum(acc_active, axis=1)
    gyr_count = np.sum(gyr_active, axis=1)

    safe_stats(out, "xsens2_acc_active_count", acc_count)
    safe_stats(out, "xsens2_gyr_active_count", gyr_count)

    for k in [0, 1, 2, 3]:
        safe_fraction(out, f"xsens2_acc_active_count_exactly{k}_frac", acc_count == k)
        safe_fraction(out, f"xsens2_gyr_active_count_exactly{k}_frac", gyr_count == k)

    safe_fraction(out, "xsens2_acc_active_count_atleast2_frac", acc_count >= 2)
    safe_fraction(out, "xsens2_acc_active_count_all3_frac", acc_count == 3)

    safe_fraction(out, "xsens2_gyr_active_count_atleast2_frac", gyr_count >= 2)
    safe_fraction(out, "xsens2_gyr_active_count_all3_frac", gyr_count == 3)

    # ------------------------------------------------------------
    # Cross-person synchrony and asymmetry
    # ------------------------------------------------------------

    pair_values = {
        "acc_dyn_corr": [],
        "gyr_corr": [],
        "acc_dyn_absdiff_mean": [],
        "gyr_absdiff_mean": [],
    }

    for a, b in [(1, 2), (1, 3), (2, 3)]:
        a_acc = participant_data[a]["acc_dyn_norm"]
        b_acc = participant_data[b]["acc_dyn_norm"]

        a_gyr = participant_data[a]["gyr_norm"]
        b_gyr = participant_data[b]["gyr_norm"]

        acc_corr = corr_feature(a_acc, b_acc)
        gyr_corr = corr_feature(a_gyr, b_gyr)

        acc_absdiff = np.abs(a_acc - b_acc)
        gyr_absdiff = np.abs(a_gyr - b_gyr)

        out[f"xsens2_pair{a}{b}_acc_dyn_corr"] = acc_corr
        out[f"xsens2_pair{a}{b}_gyr_corr"] = gyr_corr

        safe_stats(out, f"xsens2_pair{a}{b}_acc_dyn_absdiff", acc_absdiff)
        safe_stats(out, f"xsens2_pair{a}{b}_gyr_absdiff", gyr_absdiff)

        pair_values["acc_dyn_corr"].append(acc_corr)
        pair_values["gyr_corr"].append(gyr_corr)
        pair_values["acc_dyn_absdiff_mean"].append(np.nanmean(acc_absdiff))
        pair_values["gyr_absdiff_mean"].append(np.nanmean(gyr_absdiff))

    for name, vals in pair_values.items():
        safe_stats(out, f"xsens2_pair_summary_{name}", vals)

    # Movement asymmetry across participants
    for name in ["acc_dyn_mean", "gyr_mean", "jerk_mean", "angular_jerk_mean"]:
        vals = [participant_data[p][name] for p in [1, 2, 3]]
        safe_stats(out, f"xsens2_person_summary_{name}", vals)

    return out


# ================================================================
# Load base windows and labels
# ================================================================

eng7 = pd.read_csv(ENG7_PATH).copy()

rec = pd.read_csv(REC_PATH)[
    ["group", "window_start", "window_end", "recognition_label"]
].copy()

rec = rec[rec["recognition_label"].isin(CORE)].copy()
rec["mid"] = (rec["window_start"] + rec["window_end"]) / 2.0

eng7 = add_recognition_labels(eng7, rec)

eng7 = (
    eng7[eng7["recognition_label"].isin(CORE)]
    .dropna(subset=["recognition_label"])
    .reset_index(drop=True)
)

eng7["window_mid"] = (eng7["window_start"] + eng7["window_end"]) / 2.0
group_start = eng7.groupby("group")["window_mid"].transform("min")
eng7["elapsed_min"] = (eng7["window_mid"] - group_start) / 60.0

print("=" * 100)
print("BASE WINDOWS")
print("=" * 100)
print("Shape:", eng7.shape)
print("Class counts:")
print(eng7["recognition_label"].value_counts().to_string())


# ================================================================
# Generate XSENS2 features
# ================================================================

all_rows = []
source_rows = []

for group in GROUPS:
    print("\n" + "=" * 100)
    print(f"Processing group {group}")
    print("=" * 100)

    base_g = eng7[eng7["group"] == group].copy().reset_index(drop=True)

    chosen, candidates = choose_best_xsens_source(group, base_g)

    raw = chosen["raw"]
    time_col = chosen["time_col"]

    print("Candidate source scores:")
    for cand in candidates:
        print(" ", cand["sanity_score"], "|", cand["time_col"], "|", cand["path"])

    print("\nChosen:", chosen["path"])
    print("Sanity score:", chosen["sanity_score"])
    print("Time column:", time_col)

    t_raw = raw[time_col].to_numpy()

    print("Raw rows:", len(raw))
    print("Windows:", len(base_g))
    print("Raw time range:", np.nanmin(t_raw), "to", np.nanmax(t_raw))
    print("Window range:", base_g["window_start"].min(), "to", base_g["window_end"].max())

    source_rows.append({
        "group": group,
        "chosen_path": chosen["path"],
        "time_col": time_col,
        "sanity_score": chosen["sanity_score"],
        "raw_rows": len(raw),
        "windows": len(base_g),
        "raw_time_min": np.nanmin(t_raw),
        "raw_time_max": np.nanmax(t_raw),
        "window_min": base_g["window_start"].min(),
        "window_max": base_g["window_end"].max(),
    })

    for idx, w in base_g.iterrows():
        ws = float(w["window_start"])
        we = float(w["window_end"])

        start_idx = np.searchsorted(t_raw, ws, side="left")
        end_idx = np.searchsorted(t_raw, we, side="left")

        sub = raw.iloc[start_idx:end_idx].copy()

        feat = compute_xsens_window_features(
            sub=sub,
            group=group,
            ws=ws,
            we=we,
            time_col=time_col,
        )

        feat["recognition_label"] = w["recognition_label"]
        feat["elapsed_min"] = w["elapsed_min"]

        all_rows.append(feat)

        if (idx + 1) % 50 == 0:
            print(f"  completed {idx + 1}/{len(base_g)} windows", flush=True)


# ================================================================
# Save
# ================================================================

xsens2 = pd.DataFrame(all_rows)
source_report = pd.DataFrame(source_rows)

# Drop all-NaN columns
all_nan_cols = [c for c in xsens2.columns if xsens2[c].isna().all()]

if all_nan_cols:
    print("\nDropping all-NaN columns:", len(all_nan_cols))
    xsens2 = xsens2.drop(columns=all_nan_cols)

xsens2.to_csv(OUT_PATH, index=False)
source_report.to_csv(SOURCE_REPORT_PATH, index=False)

xsens2_features = [c for c in xsens2.columns if c.startswith("xsens2_")]

print("\n" + "=" * 100)
print("XSENS2 SAVED")
print("=" * 100)
print("Path:", OUT_PATH)
print("Shape:", xsens2.shape)

print("\nSource report:")
display(source_report)

print("\nClass counts:")
print(xsens2["recognition_label"].value_counts().to_string())

print("\nNumber of XSENS2 rich features:", len(xsens2_features))

print("\nFirst 100 XSENS2 features:")
for c in xsens2_features[:100]:
    print(" ", c)

print("\nSaved source selection report:")
print(SOURCE_REPORT_PATH)


# --- CELL 20 (code cell #18) ---
# ================================================================
# B7. SAFE TASK 2 ADVANCED MERGE
# Uses XSENS2 as the labelled 992-row base and left-merges OE10 and OPTI2.
# ================================================================
import os
import numpy as np
import pandas as pd

CORE = ["co_building", "co_merging", "conversation"]

OE10_PATH = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_OE10",
    "interaction_oe10_10s.csv",
)
OPTI2_PATH = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_OPTI2",
    "interaction_opti2_10s.csv",
)
XSENS2_PATH = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_XSENS2",
    "interaction_xsens2_10s.csv",
)

ADVANCED_DATA_PATH = os.path.join(
    OUTPUT_DATA_ROOT,
    "INTERACTION_ABLATIONS",
    "activity3_advanced_merged_10s_features.csv",
)

# PART 1 — SAFE ADVANCED MERGED DATASET CREATION
# ================================================================

def unique_feats(feats):
    return list(dict.fromkeys(feats))


def add_merge_keys(df, decimals):
    out = df.copy()
    out["_group_key"] = pd.to_numeric(out["group"], errors="coerce").astype(int)
    out["_ws_key"] = pd.to_numeric(out["window_start"], errors="coerce").round(decimals)
    out["_we_key"] = pd.to_numeric(out["window_end"], errors="coerce").round(decimals)
    return out


def merge_features_left_safe(base, other, other_feature_cols, label):
    """
    Left-merge features onto labelled base.
    Important:
      - Never drops base rows.
      - If matching fails, features become NaN but base remains.
      - This avoids the previous Shape: (0, 1543) problem.
    """
    other_feature_cols = unique_feats(other_feature_cols)

    other_small = other[["group", "window_start", "window_end"] + other_feature_cols].copy()

    best_merge = None
    best_round = None
    best_matched = -1

    for decimals in [6, 5, 4, 3, 2, 1]:
        a = add_merge_keys(base, decimals)
        b = add_merge_keys(other_small, decimals)

        b = b.drop_duplicates(subset=["_group_key", "_ws_key", "_we_key"], keep="first")
        b[f"_matched_{label}"] = 1

        merged_try = a.merge(
            b.drop(columns=["group", "window_start", "window_end"]),
            on=["_group_key", "_ws_key", "_we_key"],
            how="left",
        )

        matched = int(merged_try[f"_matched_{label}"].fillna(0).sum())
        print(f"{label} merge round={decimals} | matched={matched}/{len(base)}")

        if matched > best_matched:
            best_matched = matched
            best_round = decimals
            best_merge = merged_try

    out = best_merge.copy()

    helper_cols = [
        "_group_key",
        "_ws_key",
        "_we_key",
        f"_matched_{label}",
    ]

    out = out.drop(columns=[c for c in helper_cols if c in out.columns])

    print(f"Best {label} merge rounding:", best_round)
    print(f"Matched {label}:", best_matched, "/", len(base))
    print("Shape after safe left merge:", out.shape)

    return out, best_matched


def is_xsens_quality_feature(c):
    c = str(c).lower()
    return "available_frac" in c


def is_xsens_raw_euler_posture(c):
    c = str(c).lower()
    return (
        "_euler_x_" in c
        or "_euler_y_" in c
        or "_euler_z_" in c
    )


def is_opti2_quality_feature(c):
    c = str(c).lower()
    return (
        "available" in c
        or "active_landmarks" in c
        or "valid_frac" in c
        or "atleast" in c
        or "all3_available" in c
    )


def is_opti2_absolute_position_feature(c):
    c = str(c).lower()

    if any(pattern in c for pattern in [
        "_lm1_x_", "_lm1_y_", "_lm1_z_",
        "_lm2_x_", "_lm2_y_", "_lm2_z_",
        "_lm3_x_", "_lm3_y_", "_lm3_z_",
    ]):
        return True

    if any(pattern in c for pattern in [
        "centroid2d_x",
        "centroid2d_z",
        "centroid3d_x",
        "centroid3d_y",
        "centroid3d_z",
    ]):
        return True

    return False


def create_safe_advanced_dataset():
    print("=" * 100)
    print("CREATING SAFE ADVANCED 3-CLASS MERGED DATASET")
    print("=" * 100)

    for p in [OE10_PATH, OPTI2_PATH, XSENS2_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing source file:\n{p}")

    oe10 = pd.read_csv(OE10_PATH).copy()
    opti2 = pd.read_csv(OPTI2_PATH).copy()
    xsens2 = pd.read_csv(XSENS2_PATH).copy()

    print("Raw source shapes:")
    print("OE10:", oe10.shape)
    print("OPTI2:", opti2.shape)
    print("XSENS2:", xsens2.shape)

    if "recognition_label" not in xsens2.columns:
        raise KeyError("XSENS2 must contain recognition_label, but it does not.")

    if "recognition_label" not in opti2.columns:
        raise KeyError("OPTI2 must contain recognition_label, but it does not.")

    # Use XSENS2 as labelled base.
    xsens2 = xsens2[xsens2["recognition_label"].isin(CORE)].dropna(subset=["recognition_label"]).reset_index(drop=True)
    opti2 = opti2[opti2["recognition_label"].isin(CORE)].dropna(subset=["recognition_label"]).reset_index(drop=True)

    if len(xsens2) == 0:
        raise ValueError("XSENS2 filtered to 0 rows. Cannot continue.")

    print("\nFiltered labelled base:")
    print("XSENS2:", xsens2.shape)
    print("OPTI2:", opti2.shape)
    print("\nXSENS2 class counts:")
    print(xsens2["recognition_label"].value_counts().to_string())

    # ------------------------------------------------------------
    # Base columns and XSENS features
    # ------------------------------------------------------------

    base_cols = ["group", "window_start", "window_end", "recognition_label"]

    if "elapsed_min" in xsens2.columns:
        base_cols.append("elapsed_min")

    xsens_features = [c for c in xsens2.columns if c.startswith("xsens2_")]

    df_adv = xsens2[base_cols + xsens_features].copy()

    # ------------------------------------------------------------
    # OE features, without using OE recognition_label
    # ------------------------------------------------------------

    old_ear = [c for c in oe10.columns if c.startswith("ear_")]
    oe9_features = [c for c in oe10.columns if c.startswith("oe_")]
    mag_features = [c for c in oe10.columns if c.startswith("mag_")]

    oe_motion = [
        c for c in old_ear + oe9_features
        if (
            "acc_" in c
            or "gyro_" in c
            or "jerk" in c
            or "turn" in c
        )
    ]

    mag_magnitude = [
        c for c in mag_features
        if (
            "magnitude" in c
            or "horizontal" in c
            or "mag_active" in c
        )
    ]

    oe_best_original = unique_feats(oe_motion + mag_magnitude)

    oe10_small = oe10[["group", "window_start", "window_end"] + oe_best_original].copy()

    oe_rename_map = {c: f"oe__{c}" for c in oe_best_original}
    oe10_small = oe10_small.rename(columns=oe_rename_map)

    oe_best = [oe_rename_map[c] for c in oe_best_original]

    print("\nOE feature count before merge:", len(oe_best))

    df_adv, oe_matched = merge_features_left_safe(
        base=df_adv,
        other=oe10_small,
        other_feature_cols=oe_best,
        label="oe10",
    )

    # ------------------------------------------------------------
    # OPTI2 features
    # ------------------------------------------------------------

    opti2_features = [c for c in opti2.columns if c.startswith("opti2_")]

    old_eng7_prox = [
        c for c in [
            "opti_nearest_pair_dist_mean",
            "opti_all_pairs_dist_mean",
            "opti_all_pairs_dist_std",
        ]
        if c in opti2.columns
    ]

    opti_feature_cols = unique_feats(opti2_features + old_eng7_prox)

    print("\nOPTI2 feature count before merge:", len(opti_feature_cols))

    df_adv, opti_matched = merge_features_left_safe(
        base=df_adv,
        other=opti2,
        other_feature_cols=opti_feature_cols,
        label="opti2",
    )

    # Remove helper columns
    df_adv = df_adv.drop(columns=[c for c in df_adv.columns if c.startswith("_")], errors="ignore")

    if len(df_adv) == 0:
        raise ValueError("Advanced dataset has 0 rows. Stop.")

    os.makedirs(os.path.dirname(ADVANCED_DATA_PATH), exist_ok=True)
    df_adv.to_csv(ADVANCED_DATA_PATH, index=False)

    print("\n" + "=" * 100)
    print("SAVED SAFE ADVANCED 3-CLASS MERGED DATASET")
    print("=" * 100)
    print("Path:", ADVANCED_DATA_PATH)
    print("Shape:", df_adv.shape)

    print("\nClass counts:")
    print(df_adv["recognition_label"].value_counts().to_string())

    print("\nRaw feature counts in saved file:")
    print("OE features:", len([c for c in df_adv.columns if c.startswith("oe__")]))
    print("OPTI2 features:", len([c for c in df_adv.columns if c.startswith("opti2_")]))
    print("XSENS2 features:", len([c for c in df_adv.columns if c.startswith("xsens2_")]))
    print("Elapsed present:", "elapsed_min" in df_adv.columns)

    print("\nMerge matches:")
    print("OE matched:", oe_matched, "/", len(xsens2))
    print("OPTI2 matched:", opti_matched, "/", len(xsens2))

    return df_adv


# Always recreate because previous file was broken.
if os.path.exists(ADVANCED_DATA_PATH):
    os.remove(ADVANCED_DATA_PATH)
    print("Deleted old advanced file:")
    print(ADVANCED_DATA_PATH)

df = create_safe_advanced_dataset()


# --- CELL 22 (code cell #19) ---
# ================================================================
# BUILD FULL-STATISTIC SIX-LABEL ACTIVITY TOKENS
# ================================================================

NORM = RQ3_NORMALIZED_LABEL_PATH

FEATURE_CANDIDATES = [
    os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_ENG3", "interaction_eng3_features.csv"),
    os.path.join(OUTPUT_DATA_ROOT, "INTERACTION_OE10", "interaction_oe10_10s.csv"),
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


TOKEN_PATH = os.path.join(OUTPUT_DATA_ROOT, "PUBLICATION_TASK3_CORRECTED_FINAL", "activity_tokens_6label_fullstat.csv")

os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)

if False and os.path.exists(TOKEN_PATH):
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


# --- CELL 24 (code cell #20) ---

# ================================================================
# D1. VERIFY REBUILT OUTPUTS AGAINST OFFICIAL TABLES
# ================================================================
GENERATED_FILES = {
    "task1_advanced_base": os.path.join(
        OUTPUT_DATA_ROOT,
        "INTERACTION_BINARY_5S_ADVANCED_FEATURES",
        "binary_5s_all_sensor_advanced_features.csv",
    ),
    "task1_specialized": os.path.join(
        OUTPUT_DATA_ROOT,
        "INTERACTION_BINARY_5S_SPECIALIZED_OE",
        "binary_5s_specialized_oe_merged_all_features.csv",
    ),
    "eng3": os.path.join(
        OUTPUT_DATA_ROOT,
        "INTERACTION_ENG3",
        "interaction_eng3_features.csv",
    ),
    "task2_advanced": os.path.join(
        OUTPUT_DATA_ROOT,
        "INTERACTION_ABLATIONS",
        "activity3_advanced_merged_10s_features.csv",
    ),
    "task3_tokens": os.path.join(
        OUTPUT_DATA_ROOT,
        "PUBLICATION_TASK3_CORRECTED_FINAL",
        "activity_tokens_6label_fullstat.csv",
    ),
}

REFERENCE_FILES = {
    "task1_advanced_base": os.path.join(
        REFERENCE_DATA_ROOT,
        "INTERACTION_BINARY_5S_ADVANCED_FEATURES",
        "binary_5s_all_sensor_advanced_features.csv",
    ),
    "task1_specialized": os.path.join(
        REFERENCE_DATA_ROOT,
        "INTERACTION_BINARY_5S_SPECIALIZED_OE",
        "binary_5s_specialized_oe_merged_all_features.csv",
    ),
    "eng3": os.path.join(
        REFERENCE_DATA_ROOT,
        "INTERACTION_ENG3",
        "interaction_eng3_features.csv",
    ),
    "task2_advanced": os.path.join(
        REFERENCE_DATA_ROOT,
        "INTERACTION_ABLATIONS",
        "activity3_advanced_merged_10s_features.csv",
    ),
    "task3_tokens": os.path.join(
        REFERENCE_DATA_ROOT,
        "PUBLICATION_TASK3_CORRECTED_FINAL",
        "activity_tokens_6label_fullstat.csv",
    ),
}


def schema_and_value_report(name, generated_path, reference_path):
    report = {
        "name": name,
        "generated_path": generated_path,
        "reference_path": reference_path,
        "generated_exists": os.path.exists(generated_path),
        "reference_exists": os.path.exists(reference_path),
    }

    if not report["generated_exists"]:
        report["status"] = "generated file missing"
        return report

    generated = pd.read_csv(generated_path, low_memory=False)
    report["generated_rows"] = len(generated)
    report["generated_columns"] = len(generated.columns)

    if not report["reference_exists"]:
        report["status"] = "generated; no reference available"
        return report

    reference = pd.read_csv(reference_path, low_memory=False)
    report["reference_rows"] = len(reference)
    report["reference_columns"] = len(reference.columns)

    generated_cols = set(generated.columns)
    reference_cols = set(reference.columns)
    common = sorted(generated_cols & reference_cols)

    report["common_columns"] = len(common)
    report["only_generated_columns"] = len(generated_cols - reference_cols)
    report["only_reference_columns"] = len(reference_cols - generated_cols)

    key_candidates = [
        ["group", "window_start", "window_end"],
        ["group", "start_time", "label"],
        ["group", "label"],
    ]
    keys = next(
        (
            key
            for key in key_candidates
            if all(c in generated.columns for c in key)
            and all(c in reference.columns for c in key)
        ),
        None,
    )

    if keys:
        gk = generated[keys].copy()
        rk = reference[keys].copy()
        for frame in [gk, rk]:
            for column in keys:
                if "time" in column or "window" in column:
                    frame[column] = pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    ).round(4)
                else:
                    frame[column] = frame[column].astype(str)
        report["matching_key_rows"] = len(
            gk.drop_duplicates().merge(
                rk.drop_duplicates(),
                on=keys,
                how="inner",
            )
        )

    numeric_common = []
    for column in common:
        if column in {
            "group", "window_start", "window_end", "start_time",
            "label", "binary_label", "recognition_label",
        }:
            continue
        a = pd.to_numeric(generated[column], errors="coerce")
        b = pd.to_numeric(reference[column], errors="coerce")
        if a.notna().sum() > 0 and b.notna().sum() > 0:
            numeric_common.append(column)

    correlations = []
    if keys and numeric_common:
        left = generated[keys + numeric_common].copy()
        right = reference[keys + numeric_common].copy()

        for frame in [left, right]:
            for column in keys:
                if "time" in column or "window" in column:
                    frame[column] = pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    ).round(4)
                else:
                    frame[column] = frame[column].astype(str)

        merged = left.merge(
            right,
            on=keys,
            suffixes=("_generated", "_reference"),
        )

        for column in numeric_common[:500]:
            a = pd.to_numeric(
                merged[f"{column}_generated"],
                errors="coerce",
            ).to_numpy()
            b = pd.to_numeric(
                merged[f"{column}_reference"],
                errors="coerce",
            ).to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10:
                continue
            if np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
                continue
            correlations.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))

    if correlations:
        report["median_common_feature_correlation"] = float(
            np.nanmedian(correlations)
        )
        report["features_correlation_ge_0_99"] = int(
            np.sum(np.asarray(correlations) >= 0.99)
        )
        report["correlations_checked"] = len(correlations)

    report["status"] = "compared"

    # Save exact schema differences for diagnosis.
    diff_dir = os.path.join(OUTPUT_DATA_ROOT, "MANIFEST", "COLUMN_DIFFS")
    os.makedirs(diff_dir, exist_ok=True)
    pd.DataFrame({"column": sorted(generated_cols - reference_cols)}).to_csv(
        os.path.join(diff_dir, f"{name}__only_generated.csv"), index=False
    )
    pd.DataFrame({"column": sorted(reference_cols - generated_cols)}).to_csv(
        os.path.join(diff_dir, f"{name}__only_reference.csv"), index=False
    )

    return report


verification_rows = [
    schema_and_value_report(
        name,
        GENERATED_FILES[name],
        REFERENCE_FILES[name],
    )
    for name in GENERATED_FILES
]

verification_df = pd.DataFrame(verification_rows)

# The historical generic Task 1 base is intentionally a compatibility reconstruction.
verification_df["expected_exact"] = verification_df["name"].ne("task1_advanced_base")
display(verification_df)

manifest_dir = os.path.join(OUTPUT_DATA_ROOT, "MANIFEST")
os.makedirs(manifest_dir, exist_ok=True)

verification_path = os.path.join(
    manifest_dir,
    "feature_generator_verification.csv",
)
verification_df.to_csv(verification_path, index=False)

manifest_rows = []
for name, path in GENERATED_FILES.items():
    if not os.path.exists(path):
        continue
    frame = pd.read_csv(path, low_memory=False)
    manifest_rows.append(
        {
            "dataset": name,
            "path": path,
            "rows": len(frame),
            "columns": len(frame.columns),
        }
    )

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = os.path.join(
    manifest_dir,
    "generated_dataset_manifest.csv",
)
manifest_df.to_csv(manifest_path, index=False)

print("Verification report:", verification_path)
print("Dataset manifest:", manifest_path)
display(manifest_df)


# --- CELL 25 (code cell #21) ---
from pathlib import Path

root = Path("/content/drive/MyDrive/thesis")

search_terms = [
    "binary_5s_all_sensor_advanced_features.csv",
    "xsens2__window_energy_all",
    "xsens2__p1_acc_x__mean",
    "opti2_group_spread",
    "opti2_triangle_area",
    "adv_stats(",
]

extensions = {".ipynb", ".py", ".txt"}

for path in root.rglob("*"):
    if path.suffix.lower() not in extensions:
        continue

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    matches = [
        term for term in search_terms
        if term in text
    ]

    if matches:
        print("\nFILE:", path)
        print("MATCHES:", matches)


# --- CELL 26 (code cell #22) ---
from pathlib import Path
import json
import zipfile
import re

ROOT = Path("/content/drive/MyDrive")

SEARCH_TERMS = [
    "INTERACTION_BINARY_5S_ADVANCED_FEATURES",
    "binary_5s_all_sensor_advanced_features.csv",
]

# Patterns giving stronger evidence that a notebook created the folder/file
CREATION_PATTERNS = [
    r"os\.makedirs",
    r"\.mkdir\(",
    r"Path\(",
    r"to_csv\(",
    r"OUTPUT_DIR",
    r"OUT_DIR",
    r"output_dir",
    r"save_dir",
]

results = []


def extract_notebook_text(raw_bytes):
    """Extract code and markdown text from an ipynb file."""
    try:
        notebook = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return None

    cells = notebook.get("cells", [])
    extracted = []

    for cell_number, cell in enumerate(cells):
        source = cell.get("source", [])

        if isinstance(source, list):
            source = "".join(source)

        extracted.append({
            "cell_number": cell_number,
            "cell_type": cell.get("cell_type", "unknown"),
            "text": source,
        })

    return extracted


def inspect_notebook(notebook_name, raw_bytes, location_type):
    cells = extract_notebook_text(raw_bytes)

    if cells is None:
        return

    for cell in cells:
        text = cell["text"]

        matched_terms = [
            term for term in SEARCH_TERMS
            if term.lower() in text.lower()
        ]

        if not matched_terms:
            continue

        creation_matches = [
            pattern for pattern in CREATION_PATTERNS
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]

        # Stronger score for folder creation or file writing
        score = 1

        if "INTERACTION_BINARY_5S_ADVANCED_FEATURES" in matched_terms:
            score += 2

        if "binary_5s_all_sensor_advanced_features.csv" in matched_terms:
            score += 2

        if creation_matches:
            score += 3

        if "to_csv" in text:
            score += 5

        if "makedirs" in text or ".mkdir(" in text:
            score += 5

        # Short useful snippet
        lines = text.splitlines()
        relevant_lines = []

        for line_number, line in enumerate(lines):
            if any(term.lower() in line.lower() for term in SEARCH_TERMS):
                start = max(0, line_number - 4)
                end = min(len(lines), line_number + 6)
                relevant_lines.extend(lines[start:end])

        snippet = "\n".join(dict.fromkeys(relevant_lines))

        results.append({
            "score": score,
            "notebook": notebook_name,
            "location_type": location_type,
            "cell_number": cell["cell_number"],
            "cell_type": cell["cell_type"],
            "matched_terms": matched_terms,
            "creation_patterns": creation_matches,
            "snippet": snippet[:4000],
        })


# -------------------------------------------------
# 1. Search normal notebooks
# -------------------------------------------------
print("Searching normal notebooks...")

for path in ROOT.rglob("*.ipynb"):
    try:
        inspect_notebook(
            notebook_name=str(path),
            raw_bytes=path.read_bytes(),
            location_type="normal notebook",
        )
    except Exception as error:
        print("Could not inspect:", path, error)


# -------------------------------------------------
# 2. Search notebooks stored inside ZIP files
# -------------------------------------------------
print("Searching notebooks inside ZIP files...")

for zip_path in ROOT.rglob("*.zip"):
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".ipynb"):
                    continue

                try:
                    raw_bytes = archive.read(member)

                    inspect_notebook(
                        notebook_name=f"{zip_path} :: {member}",
                        raw_bytes=raw_bytes,
                        location_type="inside ZIP",
                    )
                except Exception:
                    pass

    except zipfile.BadZipFile:
        pass
    except Exception as error:
        print("Could not inspect ZIP:", zip_path, error)


# -------------------------------------------------
# Show ranked results
# -------------------------------------------------
results = sorted(
    results,
    key=lambda item: (-item["score"], item["notebook"])
)

print("\n" + "=" * 100)
print(f"FOUND {len(results)} MATCHING CELLS")
print("=" * 100)

for index, result in enumerate(results, start=1):
    print(f"\nRESULT {index}")
    print("-" * 100)
    print("SCORE:", result["score"])
    print("NOTEBOOK:", result["notebook"])
    print("LOCATION:", result["location_type"])
    print("CELL:", result["cell_number"], result["cell_type"])
    print("MATCHED:", result["matched_terms"])
    print("CREATION INDICATORS:", result["creation_patterns"])
    print("\nSNIPPET:\n")
    print(result["snippet"])

