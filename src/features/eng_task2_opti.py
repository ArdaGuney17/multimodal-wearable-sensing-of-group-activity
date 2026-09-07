"""Feature engineering, Task 2 OPTI2 family — the rich 10s OptiTrack
feature table ("OPTI2 — RICH OPTITRACK FEATURE GENERATION, ROBUST
VERSION", CELL 15 of `master_feature_generator_task1_task2_task3_
CORRECTED_V4.ipynb`) that `eng_task2_merge.py` (CELL 17) left-merges onto
the XSENS2 base.

DIFFERENT, richer feature set from Task 1's `eng_task1_opti.py` (that
module's own CELL 3 is "the historical simple" 5s-window OPTI2 generator
with fixed landmark-column detection). This CELL 15 ("robust version")
adds: automatic source-file *scoring* by counting finite coordinates
inside the actual windows (not just "does the column exist"), pairwise
2D/3D distance CHANGE RATES, row-wise nearest/farthest/mean/std/range
summaries across the 3 pairwise distances, 2D/3D centroid + spread
(distance-to-centroid) stats, a 2D formation triangle (area/perimeter/
compactness), and a cross-person speed-asymmetry summary — none of which
`eng_task1_opti.py` computes. Ported fresh rather than reusing anything
from that module.

Ported pieces, all from CELL 15:
  - `safe_stats`, `safe_fraction`, `row_nanmin/max/mean/std`,
    `centroid_from_stack`, `compute_speed`, `triangle_area_2d`,
    `standardize_columns` — shared numeric helpers.
  - `find_landmark_mapping`, `find_available_col`, `load_candidate_file`,
    `coordinate_score_for_windows`, `choose_best_opti_source` —
    source-aware file + landmark-column selection (restricted, like
    `eng_task1_opti.py`'s and `eng_task2_xsens.py`'s analogous functions,
    to the single `raw_dir` candidate when no `source_data_root` is
    supplied).
  - `compute_window_features` — the actual `opti2_*` feature function.
  - `build_task2_opti_features` — the driver: for each group, choose the
    best OptiTrack source + landmark mapping, slice each base window by
    raw-timestamp index, compute the per-window feature dict, and copy
    across the 3 ENG7 proximity columns (`opti_nearest_pair_dist_mean`,
    `opti_all_pairs_dist_mean`, `opti_all_pairs_dist_std`) already present
    on `base_windows` — CELL 15 carries these through verbatim from its
    ENG7 input rather than recomputing them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CORE = ["co_building", "co_merging", "conversation"]

PAIR_NAMES = [(1, 2), (1, 3), (2, 3)]


# ================================================================
# Shared numeric helpers (verbatim from CELL 15)
# ================================================================

def get_num_col(df: pd.DataFrame, col) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy()
    return np.full(len(df), np.nan)


def safe_stats(out: dict, prefix: str, arr) -> None:
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


def safe_fraction(out: dict, prefix: str, mask) -> None:
    mask = np.asarray(mask)
    out[prefix] = np.nan if len(mask) == 0 else np.mean(mask)


def row_nanmin(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmin(A[mask], axis=1)
    return out


def row_nanmax(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmax(A[mask], axis=1)
    return out


def row_nanmean(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanmean(A[mask], axis=1)
    return out


def row_nanstd(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    out = np.full(A.shape[0], np.nan)
    mask = np.isfinite(A).any(axis=1)
    out[mask] = np.nanstd(A[mask], axis=1)
    return out


def centroid_from_stack(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    n, people, dim = A.shape
    out = np.full((n, dim), np.nan)
    for i in range(n):
        valid_people = np.all(np.isfinite(A[i]), axis=1)
        if valid_people.any():
            out[i] = np.nanmean(A[i, valid_people, :], axis=0)
    return out


def compute_speed(pos, t) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    t = np.asarray(t, dtype=float)

    valid = np.all(np.isfinite(pos), axis=1) & np.isfinite(t)
    pos, t = pos[valid], t[valid]
    if len(t) < 3:
        return np.array([])

    dt = np.diff(t)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    good = np.isfinite(dt) & np.isfinite(dp) & (dt > 1e-6)
    if good.sum() == 0:
        return np.array([])

    speed = dp[good] / dt[good]
    speed = speed[np.isfinite(speed)]
    speed = speed[speed < 10.0]  # remove clear tracking jumps
    return speed


def triangle_area_2d(p1, p2, p3) -> np.ndarray:
    return 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ================================================================
# Source-aware landmark-column detection + file selection
# ================================================================

def find_landmark_mapping(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    lower_to_original = {c.lower(): c for c in cols}
    mapping: dict = {}

    candidates = {
        1: [("landmark1_x", "landmark1_y", "landmark1_z"), ("p1_x", "p1_y", "p1_z"),
            ("participant1_x", "participant1_y", "participant1_z"), ("person1_x", "person1_y", "person1_z")],
        2: [("landmark2_x", "landmark2_y", "landmark2_z"), ("p2_x", "p2_y", "p2_z"),
            ("participant2_x", "participant2_y", "participant2_z"), ("person2_x", "person2_y", "person2_z")],
        3: [("landmark3_x", "landmark3_y", "landmark3_z"), ("p3_x", "p3_y", "p3_z"),
            ("participant3_x", "participant3_y", "participant3_z"), ("person3_x", "person3_y", "person3_z")],
    }

    for i in (1, 2, 3):
        mapping[i] = {"x": None, "y": None, "z": None}
        for xname, yname, zname in candidates[i]:
            if xname.lower() in lower_to_original and yname.lower() in lower_to_original and zname.lower() in lower_to_original:
                mapping[i]["x"] = lower_to_original[xname.lower()]
                mapping[i]["y"] = lower_to_original[yname.lower()]
                mapping[i]["z"] = lower_to_original[zname.lower()]
                break

    return mapping


def find_available_col(df: pd.DataFrame, i: int):
    candidates = [f"landmark{i}_available", f"p{i}_available", f"participant{i}_available", f"person{i}_available"]
    lower_to_original = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    return None


def load_candidate_file(path: str):
    raw = pd.read_csv(path, low_memory=False)
    raw = standardize_columns(raw)

    time_col = "video_time_s" if "video_time_s" in raw.columns else "time_s"
    if time_col not in raw.columns:
        return None, None, None

    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    mapping = find_landmark_mapping(raw)
    return raw, time_col, mapping


def coordinate_score_for_windows(raw, time_col, mapping, base_g: pd.DataFrame, max_windows: int = 20) -> int:
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

        for i in (1, 2, 3):
            for axis in ("x", "y", "z"):
                col = mapping[i][axis]
                if col is not None and col in sub.columns:
                    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy()
                    score += np.isfinite(vals).sum()

    return int(score)


def candidate_paths_for_group(raw_dir: str, group: int, source_data_root: str | None = None) -> list[str]:
    candidates = [os.path.join(raw_dir, f"group_{group}_optitrack_model_ready.csv")]
    if source_data_root:
        candidates.append(os.path.join(source_data_root, "ALL_MODEL_READY_FILES", f"group_{group}_optitrack_model_ready.csv"))
    return candidates


def choose_best_opti_source(raw_dir: str, group: int, base_g: pd.DataFrame, source_data_root: str | None = None):
    candidates = []

    for path in candidate_paths_for_group(raw_dir, group, source_data_root):
        if not os.path.exists(path):
            continue

        raw, time_col, mapping = load_candidate_file(path)
        score = coordinate_score_for_windows(raw, time_col, mapping, base_g)
        candidates.append({"path": path, "raw": raw, "time_col": time_col, "mapping": mapping, "coordinate_score": score})

    if len(candidates) == 0:
        raise FileNotFoundError(f"No OptiTrack file found for group {group}")

    candidates = sorted(candidates, key=lambda x: x["coordinate_score"], reverse=True)
    return candidates[0], candidates


# ================================================================
# Per-window feature computation (CELL 15's `compute_window_features`
# — verbatim)
# ================================================================

def compute_window_features(sub: pd.DataFrame, group: int, ws: float, we: float, mapping: dict) -> dict:
    out = {"group": group, "window_start": ws, "window_end": we}

    if len(sub) == 0:
        return out

    sub = standardize_columns(sub)
    time_col = "video_time_s" if "video_time_s" in sub.columns else "time_s"
    t = get_num_col(sub, time_col)

    # ---------------- Availability / tracking quality ----------------
    for i in (1, 2, 3):
        avail_col = find_available_col(sub, i)
        if avail_col is not None:
            vals = get_num_col(sub, avail_col)
            out[f"opti2_lm{i}_available_frac"] = np.nanmean(vals) if np.isfinite(vals).sum() else np.nan
        else:
            xcol, ycol, zcol = mapping[i]["x"], mapping[i]["y"], mapping[i]["z"]
            if xcol is not None and ycol is not None and zcol is not None:
                valid = np.isfinite(get_num_col(sub, xcol)) & np.isfinite(get_num_col(sub, ycol)) & np.isfinite(get_num_col(sub, zcol))
                out[f"opti2_lm{i}_available_frac"] = np.mean(valid)
            else:
                out[f"opti2_lm{i}_available_frac"] = np.nan

    if "active_clean_landmarks" in sub.columns:
        active = get_num_col(sub, "active_clean_landmarks")
    else:
        active = np.zeros(len(sub))
        for i in (1, 2, 3):
            xcol, ycol, zcol = mapping[i]["x"], mapping[i]["y"], mapping[i]["z"]
            if xcol is not None and ycol is not None and zcol is not None:
                valid = np.isfinite(get_num_col(sub, xcol)) & np.isfinite(get_num_col(sub, ycol)) & np.isfinite(get_num_col(sub, zcol))
                active += valid.astype(float)

    safe_stats(out, "opti2_active_landmarks", active)
    safe_fraction(out, "opti2_all3_available_frac", active >= 3)
    safe_fraction(out, "opti2_atleast2_available_frac", active >= 2)
    safe_fraction(out, "opti2_atleast1_available_frac", active >= 1)

    # ---------------- Positions and individual movement ----------------
    P = {}
    for i in (1, 2, 3):
        xcol, ycol, zcol = mapping[i]["x"], mapping[i]["y"], mapping[i]["z"]
        x = get_num_col(sub, xcol) if xcol is not None else np.full(len(sub), np.nan)
        y = get_num_col(sub, ycol) if ycol is not None else np.full(len(sub), np.nan)
        z = get_num_col(sub, zcol) if zcol is not None else np.full(len(sub), np.nan)

        pos3 = np.column_stack([x, y, z])
        pos2 = np.column_stack([x, z])
        P[i] = {"x": x, "y": y, "z": z, "pos3": pos3, "pos2": pos2}

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

    # ---------------- Pairwise distances ----------------
    all_pair_d2, all_pair_d3 = [], []

    for a, b in PAIR_NAMES:
        valid2 = np.all(np.isfinite(P[a]["pos2"]), axis=1) & np.all(np.isfinite(P[b]["pos2"]), axis=1)
        valid3 = np.all(np.isfinite(P[a]["pos3"]), axis=1) & np.all(np.isfinite(P[b]["pos3"]), axis=1)

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

    nearest2, farthest2 = row_nanmin(all_pair_d2), row_nanmax(all_pair_d2)
    mean2, std2 = row_nanmean(all_pair_d2), row_nanstd(all_pair_d2)
    range2 = farthest2 - nearest2

    nearest3, farthest3 = row_nanmin(all_pair_d3), row_nanmax(all_pair_d3)
    mean3, std3 = row_nanmean(all_pair_d3), row_nanstd(all_pair_d3)
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

    # ---------------- Centroid and spread ----------------
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

    # ---------------- Formation triangle ----------------
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

    # ---------------- Movement asymmetry ----------------
    speed_means = []
    for i in (1, 2, 3):
        speed = compute_speed(P[i]["pos2"], t)
        speed_means.append(np.nanmean(speed) if len(speed) else np.nan)

    safe_stats(out, "opti2_individual_speed2d_mean_across_people", np.asarray(speed_means, dtype=float))

    return out


# ================================================================
# Driver
# ================================================================

def build_task2_opti_features(
    raw_dir: str,
    base_windows: pd.DataFrame,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> pd.DataFrame:
    """Verbatim port of CELL 15's main loop. `base_windows` must have
    `group`/`window_start`/`window_end`/`recognition_label`/`elapsed_min`
    columns, plus (if present) the 3 `opti_*` ENG7 proximity columns,
    which are copied through unchanged onto every output row exactly as
    CELL 15 does (`for c in old_eng7_opti: feat[c] = w[c]`)."""
    if groups is None:
        groups = sorted(int(g) for g in base_windows["group"].unique())

    old_eng7_opti = [c for c in base_windows.columns if c.startswith("opti_")]

    all_rows = []

    for group in groups:
        base_g = base_windows[base_windows["group"] == group].copy().reset_index(drop=True)
        if len(base_g) == 0:
            continue

        chosen, _candidates = choose_best_opti_source(raw_dir, group, base_g, source_data_root)
        raw = chosen["raw"]
        time_col = chosen["time_col"]
        mapping = chosen["mapping"]
        t_raw = raw[time_col].to_numpy()

        for _, w in base_g.iterrows():
            ws = float(w["window_start"])
            we = float(w["window_end"])

            start_idx = np.searchsorted(t_raw, ws, side="left")
            end_idx = np.searchsorted(t_raw, we, side="left")
            sub = raw.iloc[start_idx:end_idx].copy()

            feat = compute_window_features(sub=sub, group=group, ws=ws, we=we, mapping=mapping)
            feat["recognition_label"] = w["recognition_label"]
            feat["elapsed_min"] = w["elapsed_min"]

            for c in old_eng7_opti:
                feat[c] = w[c]

            all_rows.append(feat)

        del raw

    opti2 = pd.DataFrame(all_rows)

    # NOTE: as in eng_task2_xsens.py, CELL 15's all-NaN-column drop is a
    # multi-group decision — deliberately not replicated for a
    # single-group rebuild (see that module's docstring for the same
    # reasoning).

    return opti2
