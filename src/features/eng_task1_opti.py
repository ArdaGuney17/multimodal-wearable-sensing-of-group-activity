"""Feature engineering, Task 1 OptiTrack (OPTI2) half — the historical
simple 16-statistic OptiTrack feature family that feeds
`binary_5s_all_sensor_advanced_features.csv` (the Task 1 *compatibility
base* table; OE's specialized features are merged on top of a copy of this
same base to produce the actual Task 1 model input,
`binary_5s_specialized_oe_merged_all_features.csv` — see
`src/features/eng_task1_oe.py`).

Ported from `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`
(`notebooks_reference/master_feature_generator_task1_task2_task3_CORRECTED_
V4_ANALYSIS.md`). Scope is deliberately restricted to the OptiTrack-only
piece of the 5s Task 1 rebuild — no Xsens (XSENS2, already ported in
`eng_task1_xsens.py`, untouched here), no OpenEarable (already ported in
`eng_task1_oe.py`, untouched here).

**Circularity note** — same situation as `eng_task1_oe.py`/
`eng_task1_xsens.py`: the window/label grid (`group`/`window_start`/
`window_end`/`binary_label`, + a few metadata columns) is bootstrapped by
copying those columns out of an already-existing official Task 1 output CSV
(CELL 3, "A1. AUTHORITATIVE FIVE-SECOND BINARY WINDOW AND LABEL GRID" —
genuinely circular, not re-derived). `load_base_windows` is reused directly
from `eng_task1_oe.py` rather than reimplemented, since it is the exact same
cell. What IS independently computed here is every OPTI2 feature VALUE:
`build_opti2_5s` recomputes all 25 position/speed/distance/centroid/spread/
triangle-area signals via `adv_stats` from the raw
`group_{g}_optitrack_model_ready.csv` file — the window boundaries are
reused, the numbers inside each window are not.

Ported pieces, by source cell:
  - `adv_stats`, `num` — CELL 4 ("SECTION 0b — advanced statistic
    expansion"), the shared helpers cell. Checked line-by-line against
    `eng_task1_oe.py`'s copies (which come from the identical source cell)
    and found byte-identical, so they are imported from there rather than
    duplicated.
  - `task1_standardize_columns`, `task1_find_landmark_mapping`,
    `task1_opti_candidate_paths`, `task1_load_opti_candidate`,
    `task1_opti_source_score`, `task1_choose_opti_source`, `build_opti2_5s`
    — CELL 3 in this notebook's actual cell order ("A2. TASK 1 OPTI2
    FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR", the 4th code cell /
    `code_cells[3]` when indexing only code cells; ~12.9KB, the largest of
    the three ENG3-family Task 1 cells). This IS source-aware, confirmed by
    reading the cell itself: unlike XSENS2's fixed `p{p}_{kind}_{axis}`
    column names, OPTI2's raw marker/landmark columns vary in naming across
    the thesis's model-ready files, so `task1_find_landmark_mapping` tries
    four candidate per-participant naming schemes in order
    (`landmark{p}_{x,y,z}`, `p{p}_{x,y,z}`, `participant{p}_{x,y,z}`,
    `person{p}_{x,y,z}`, matched case-insensitively against the frame's
    actual columns) and picks the first fully-present triple per
    participant — real Group 1 data uses `Participant{p}_x/_y/_z`
    (capitalized), which the case-insensitive `participant{p}_x` alternative
    matches. `task1_opti_candidate_paths` lists up to two possible raw file
    locations per group (`RAW_DIR`, and a flat `ALL_MODEL_READY_FILES`
    fallback under `SOURCE_DATA_ROOT` — one fewer fallback than XSENS2's
    three, since OPTI2 has no per-group nested `_old`/re-processed-variant
    location in the source notebook). Candidates are loaded, scored by
    counting finite in-window position samples after a `|v|<1e6` corrupt-
    value guard (`task1_opti_source_score`), and the highest-scoring one is
    picked (`task1_choose_opti_source` — matches the function name flagged
    in the analysis doc). For the chosen source, the main body computes:
    (a) per-participant xyz position + speed (`opti2_p{p}_{x,y,z}`,
    `opti2_p{p}_speed`, speed = `|gradient(position)| / gradient(t)`), (b)
    pairwise Euclidean distances for all 3 pairs (`opti2_p{a}_p{b}_dist`),
    (c) nearest/farthest/mean/std-of-pairwise-distance
    (`opti2_nearest_pair_dist`, `opti2_farthest_pair_dist`,
    `opti2_all_pair_dist`, `opti2_pair_dist_spread`), (d) 3-participant
    centroid position + RMS group spread around it
    (`opti2_centroid_{x,y,z}`, `opti2_group_spread`), (e) mean-of-3
    all-participant speed (`opti2_all_speed`), (f) triangle area via the
    cross product of two edge vectors (`opti2_triangle_area`) — 25 signals
    total (12 per-participant pos/speed + 3 pairwise distances + 5
    nearest/farthest/mean/spread/group-spread + 3 centroid axes + 1
    all-speed + 1 triangle area), each run through `adv_stats` (16 stats
    each = 400 columns), plus 3 non-`adv_stats` window-level scalars: the
    mean count of participants whose speed exceeds 0.05/0.10/0.20 m/s
    (`opti2_active_speed_count_gt_{0_05,0_10,0_20}__mean`). 25*16+3 = 403,
    matching the official file's 403 `opti2_*` columns exactly.
  - `load_base_windows` / `build_task1_opti_features` — the driver.
    `load_base_windows` is imported directly from `eng_task1_oe.py` (CELL
    3's grid-bootstrap, identical for OE/Xsens/OptiTrack). `build_task1_
    opti_features` loads the grid, computes `build_opti2_5s` for the
    requested groups, then merges the OPTI2-specific slice of CELL 6 ("A3.
    MERGE RECONSTRUCTED TASK 1 COMPATIBILITY BASE TABLE"): a plain
    left-merge of the OPTI2 table onto the label grid on
    `["group","window_start","window_end"]` — like XSENS2, OPTI2 has only
    one feature table and nothing gets dropped.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.eng_task1_oe import (
    END_COL,
    GROUP_COL,
    START_COL,
    adv_stats,
    load_base_windows,
    num,
)

# task1_opti_source_score's corrupt-value guard (CELL 3's literal constant,
# same threshold as `num`'s default `clean` parameter).
OPTI_CLEAN = 1e6


# ================================================================
# A2. Task 1 OPTI2 features — source-aware five-second generator
# ================================================================

def task1_standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of CELL 3's `task1_standardize_columns`."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def task1_find_landmark_mapping(frame: pd.DataFrame) -> dict:
    """Verbatim port of CELL 3's `task1_find_landmark_mapping`. Tries, in
    order, four naming schemes per participant (`landmark{p}_*`,
    `p{p}_*`, `participant{p}_*`, `person{p}_*`), matched
    case-insensitively against the frame's actual column names, and picks
    the first fully-present xyz triple per participant."""
    lower_to_original = {str(column).lower(): column for column in frame.columns}

    mapping: dict = {}

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
        mapping[participant] = {"x": None, "y": None, "z": None}

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


def task1_opti_candidate_paths(raw_dir: str, group: int, source_data_root: str | None = None) -> list[str]:
    """Verbatim port of CELL 3's `task1_opti_candidate_paths`. The
    notebook's `RAW_DIR` maps to this module's `raw_dir` argument; the
    notebook's `SOURCE_DATA_ROOT`-relative `ALL_MODEL_READY_FILES/`
    fallback only applies when `source_data_root` is supplied — omitted,
    only the primary `raw_dir` candidate is checked."""
    candidates = [os.path.join(raw_dir, f"group_{group}_optitrack_model_ready.csv")]

    if source_data_root:
        candidates.append(
            os.path.join(source_data_root, "ALL_MODEL_READY_FILES", f"group_{group}_optitrack_model_ready.csv")
        )

    return [path for path in candidates if os.path.exists(path)]


def task1_load_opti_candidate(path: str) -> dict | None:
    """Verbatim port of CELL 3's `task1_load_opti_candidate`."""
    frame = pd.read_csv(path, low_memory=False)
    frame = task1_standardize_columns(frame)

    if "video_time_s" in frame.columns:
        time_column = "video_time_s"
    elif "time_s" in frame.columns:
        time_column = "time_s"
    else:
        return None

    frame["t"] = pd.to_numeric(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    mapping = task1_find_landmark_mapping(frame)

    if any(mapping[participant][axis] is None for participant in (1, 2, 3) for axis in "xyz"):
        return None

    return {"path": path, "frame": frame, "time_column": time_column, "mapping": mapping}


def task1_opti_source_score(candidate: dict | None, group_windows: pd.DataFrame) -> int:
    """Verbatim port of CELL 3's `task1_opti_source_score`."""
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
                values = pd.to_numeric(sub[mapping[participant][axis]], errors="coerce").to_numpy()
                values = np.where(np.abs(values) < OPTI_CLEAN, values, np.nan)
                score += int(np.isfinite(values).sum())

    return score


def task1_choose_opti_source(
    raw_dir: str, group: int, group_windows: pd.DataFrame, source_data_root: str | None = None
) -> tuple[dict, list[dict]]:
    """Verbatim port of CELL 3's `task1_choose_opti_source`."""
    candidates = []

    for path in task1_opti_candidate_paths(raw_dir, group, source_data_root):
        candidate = task1_load_opti_candidate(path)
        if candidate is None:
            continue
        candidate["score"] = task1_opti_source_score(candidate, group_windows)
        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No usable OptiTrack source found for group {group}")

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates[0], candidates


def build_opti2_5s(
    raw_dir: str,
    base_windows: pd.DataFrame,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of CELL 3's `if RUN_TASK1:` body — the actual OPTI2 5s
    feature rebuild. Returns `(opti2_5s, opti2_source_report)`."""
    rows = []
    source_rows = []

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )

    for group in groups:
        group_windows = (
            base_windows[base_windows["group_num"] == group].sort_values(START_COL).reset_index(drop=True)
        )

        chosen, candidates = task1_choose_opti_source(raw_dir, group, group_windows, source_data_root)

        frame = chosen["frame"]
        mapping = chosen["mapping"]
        times = frame["t"].to_numpy()

        print(f"\nTask 1 OptiTrack group {group}")
        for candidate in candidates:
            print(" ", candidate["score"], "|", candidate["path"])
        print(" Chosen:", chosen["path"])

        positions = {}

        for participant in (1, 2, 3):
            positions[participant] = np.stack(
                [num(frame, mapping[participant][axis]) for axis in "xyz"], axis=1
            )

        dt = np.gradient(times)

        speeds = {
            participant: (
                np.linalg.norm(np.gradient(positions[participant], axis=0), axis=1)
                / np.where(dt > 0, dt, np.nan)
            )
            for participant in (1, 2, 3)
        }

        distances = {
            "p1_p2": np.linalg.norm(positions[1] - positions[2], axis=1),
            "p1_p3": np.linalg.norm(positions[1] - positions[3], axis=1),
            "p2_p3": np.linalg.norm(positions[2] - positions[3], axis=1),
        }

        distance_matrix = np.stack(list(distances.values()), axis=1)

        nearest = np.nanmin(distance_matrix, axis=1)
        farthest = np.nanmax(distance_matrix, axis=1)
        all_pair = np.nanmean(distance_matrix, axis=1)
        pair_spread = np.nanstd(distance_matrix, axis=1)

        participant_stack = np.stack([positions[1], positions[2], positions[3]], axis=0)

        centroid = np.nanmean(participant_stack, axis=0)

        group_spread = np.sqrt(
            np.nanmean(np.sum((participant_stack - centroid[None, :, :]) ** 2, axis=2), axis=0)
        )

        all_speed = np.nanmean(np.stack([speeds[1], speeds[2], speeds[3]], axis=0), axis=0)

        triangle_area = 0.5 * np.linalg.norm(
            np.cross(positions[2] - positions[1], positions[3] - positions[1]), axis=1
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

            record = {GROUP_COL: window[GROUP_COL], START_COL: start, END_COL: end}

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

                    adv_stats(record, f"opti2_p{participant}_speed", speeds[participant][mask], window_times)

                for name, values in distances.items():
                    adv_stats(record, f"opti2_{name}_dist", values[mask], window_times)

                adv_stats(record, "opti2_nearest_pair_dist", nearest[mask], window_times)
                adv_stats(record, "opti2_farthest_pair_dist", farthest[mask], window_times)
                adv_stats(record, "opti2_all_pair_dist", all_pair[mask], window_times)
                adv_stats(record, "opti2_pair_dist_spread", pair_spread[mask], window_times)
                adv_stats(record, "opti2_group_spread", group_spread[mask], window_times)

                for axis_index, axis in enumerate("xyz"):
                    adv_stats(record, f"opti2_centroid_{axis}", centroid[mask, axis_index], window_times)

                adv_stats(record, "opti2_all_speed", all_speed[mask], window_times)
                adv_stats(record, "opti2_triangle_area", triangle_area[mask], window_times)

                for threshold, tag in [(0.05, "0_05"), (0.10, "0_10"), (0.20, "0_20")]:
                    count = np.nansum(
                        np.stack(
                            [speeds[participant][mask] > threshold for participant in (1, 2, 3)], axis=0
                        ),
                        axis=0,
                    )

                    record[f"opti2_active_speed_count_gt_{tag}__mean"] = float(np.nanmean(count))

            rows.append(record)

        print(f"Task 1 OPTI2 rebuilt: group {group} | windows={len(group_windows)}")

    opti2_5s = pd.DataFrame(rows)
    opti2_source_report = pd.DataFrame(source_rows)

    return opti2_5s, opti2_source_report


# ================================================================
# Driver
# ================================================================

def build_task1_opti_features(
    raw_dir: str,
    official_grid_path: str,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> dict:
    """Loads the official window/label grid (`eng_task1_oe.load_base_
    windows`, CELL 3's bootstrap), computes the OPTI2 5s feature table for
    `groups` (default: every group present in the grid), and merges it
    onto the grid the way CELL 6's "A3. MERGE RECONSTRUCTED TASK 1
    COMPATIBILITY BASE TABLE" does for the OPTI2 slice: a plain left-merge
    on `["group","window_start","window_end"]`.

    Returns a dict with `base_windows`, `opti2_5s`, `opti2_source_report`,
    and `merged` (grid columns + OPTI2 columns).
    """
    base_windows = load_base_windows(official_grid_path)

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )
    else:
        base_windows = base_windows[base_windows["group_num"].isin(groups)].reset_index(drop=True)

    opti2_5s, opti2_source_report = build_opti2_5s(
        raw_dir, base_windows, groups=groups, source_data_root=source_data_root
    )

    merge_keys = [GROUP_COL, START_COL, END_COL]

    task1_base = base_windows.drop(columns=["group_num"], errors="ignore").copy()
    if len(opti2_5s) > 0:
        task1_base = task1_base.merge(
            opti2_5s.drop_duplicates(merge_keys), on=merge_keys, how="left", validate="one_to_one"
        )

    return {
        "base_windows": base_windows,
        "opti2_5s": opti2_5s,
        "opti2_source_report": opti2_source_report,
        "merged": task1_base,
    }
