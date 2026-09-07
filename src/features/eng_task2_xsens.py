"""Feature engineering, Task 2 XSENS2 family — the rich 10s Xsens
feature table ("XSENS2 — RICH XSENS FEATURE GENERATION", CELL 16 of
`master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`) that
`eng_task2_merge.py` (CELL 17, "B7. SAFE TASK 2 ADVANCED MERGE") uses as
the labelled 992-row base of `activity3_advanced_merged_10s_features.csv`.

This is a DIFFERENT, richer feature set from Task 1's `eng_task1_xsens.py`
(that module's own docstring calls Task 1's XSENS2 "the historical simple
16-statistic feature schema" — CELL 4 of this same notebook, a 5s-window
generator). This module ports CELL 16 fresh rather than reusing anything
from `eng_task1_xsens.py`: the per-axis/per-signal stat helper
(`safe_stats`, 9 stats: mean/std/min/max/range/median/iqr/p10/p90) is
structurally similar to `adv_stats` but is its own, smaller function
(no energy/rms/half-delta/mad/slope), and the signal-cleaning /
derivative / spectral-feature machinery (`clean_signal`, `unwrap_degrees`,
`derivative_norm`, `derivative_1d`, `spectral_features`, `corr_feature`)
has no Task-1 analogue at all (Task 1's Xsens port never computes jerk,
Euler rate, or per-window FFT band ratios).

Ported pieces, all from CELL 16:
  - `safe_stats`, `safe_fraction`, `clean_signal`, `unwrap_degrees`,
    `vector_norm`, `derivative_norm`, `derivative_1d`, `interp_nan`,
    `spectral_features`, `corr_feature`, `standardize_columns` — the
    shared numeric helpers.
  - `candidate_paths_for_group`, `load_xsens_file`,
    `sanity_score_for_source`, `choose_best_xsens_source` — source-aware
    file selection, restricted (like `eng_task1_xsens.py`'s analogous
    functions) to the single `raw_dir` candidate when no
    `source_data_root` is supplied (this repo's downloaded fixtures only
    ever provide one candidate path per group).
  - `compute_xsens_window_features` — the actual `xsens2_*` feature
    function (per-participant acc/gyr/euler stats + dynamic acceleration,
    jerk, Euler rate, burst fractions, spectral features; group-level
    active-participant counts; cross-person correlation/absdiff/asymmetry
    summaries).
  - `build_task2_xsens_features` — the driver: for each group, choose the
    best Xsens source, slice each base window by index (`np.searchsorted`
    on the *raw* Xsens timestamps — CELL 16 does NOT resample Xsens onto
    a fixed-rate grid the way OE9/OE10/Task 1's ports do), and compute the
    per-window feature dict.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CORE = ["co_building", "co_merging", "conversation"]


# ================================================================
# Shared numeric helpers (verbatim from CELL 16)
# ================================================================

def get_num_col(df: pd.DataFrame, col: str) -> np.ndarray:
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
    if len(mask) == 0:
        out[prefix] = np.nan
    else:
        out[prefix] = np.mean(mask)


def clean_signal(x, kind: str) -> np.ndarray:
    """Removes impossible corrupted values. Verbatim port."""
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan

    if kind == "acc":
        x[np.abs(x) > 100.0] = np.nan
    elif kind == "gyr":
        x[np.abs(x) > 2000.0] = np.nan
    elif kind == "euler":
        x[np.abs(x) > 360.0] = np.nan

    return x


def unwrap_degrees(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return out
    out[valid] = np.rad2deg(np.unwrap(np.deg2rad(x[valid])))
    return out


def vector_norm(x, y, z) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    out = np.full(len(x), np.nan)
    out[valid] = np.sqrt(x[valid] ** 2 + y[valid] ** 2 + z[valid] ** 2)
    return out


def derivative_norm(x, y, z, t, max_value=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(t)
    x, y, z, t = x[valid], y[valid], z[valid], t[valid]

    if len(t) < 3:
        return np.array([])

    dt = np.diff(t)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    good = np.isfinite(dt) & (dt > 1e-6)
    if good.sum() == 0:
        return np.array([])

    val = np.sqrt(dx[good] ** 2 + dy[good] ** 2 + dz[good] ** 2) / dt[good]
    val = val[np.isfinite(val)]
    if max_value is not None:
        val = val[val < max_value]
    return val


def derivative_1d(x, t, max_value=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    valid = np.isfinite(x) & np.isfinite(t)
    x, t = x[valid], t[valid]

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


def spectral_features(out: dict, prefix: str, y, t) -> None:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    valid_t = np.isfinite(t)
    keys = [f"{prefix}_spec_entropy", f"{prefix}_spec_low_ratio", f"{prefix}_spec_mid_ratio", f"{prefix}_spec_high_ratio"]

    if valid_t.sum() < 16:
        for k in keys:
            out[k] = np.nan
        return

    y2 = interp_nan(y)
    if y2 is None or len(y2) < 16:
        for k in keys:
            out[k] = np.nan
        return

    t2 = t[np.isfinite(t)]
    dt = np.nanmedian(np.diff(t2))
    if not np.isfinite(dt) or dt <= 0:
        for k in keys:
            out[k] = np.nan
        return

    fs = 1.0 / dt
    y2 = y2 - np.nanmean(y2)

    if np.nanstd(y2) < 1e-12:
        for k in keys:
            out[k] = 0.0
        return

    fft = np.fft.rfft(y2)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(y2), d=dt)

    power = power[1:]
    freqs = freqs[1:]
    total = np.nansum(power)

    if total <= 1e-12:
        for k in keys:
            out[k] = 0.0
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


def corr_feature(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return np.nan
    aa, bb = a[valid], b[valid]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return np.nan
    return np.corrcoef(aa, bb)[0, 1]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ================================================================
# Source-aware file selection (restricted to raw_dir when
# source_data_root is omitted — mirrors eng_task1_xsens.py's pattern)
# ================================================================

def candidate_paths_for_group(raw_dir: str, group: int, source_data_root: str | None = None) -> list[str]:
    candidates = [os.path.join(raw_dir, f"group_{group}_xsens_model_ready.csv")]
    if source_data_root:
        candidates.append(os.path.join(source_data_root, "ALL_MODEL_READY_FILES", f"group_{group}_xsens_model_ready.csv"))
        candidates.append(os.path.join(source_data_root, f"group_{group}", "xsens", "model_ready", f"group_{group}_xsens_model_ready.csv"))
    return [p for p in candidates if os.path.exists(p)]


def load_xsens_file(path: str):
    raw = pd.read_csv(path, low_memory=False)
    raw = standardize_columns(raw)

    if "video_time_s" in raw.columns:
        time_col = "video_time_s"
    elif "time_s" in raw.columns:
        time_col = "time_s"
    else:
        return None, None

    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return raw, time_col


def sanity_score_for_source(raw, time_col, base_g: pd.DataFrame) -> int:
    if raw is None or time_col is None:
        return -1

    needed = []
    for p in (1, 2, 3):
        for kind in ("euler", "acc", "gyr"):
            for axis in ("x", "y", "z"):
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


def choose_best_xsens_source(raw_dir: str, group: int, base_g: pd.DataFrame, source_data_root: str | None = None):
    rows = []
    for path in candidate_paths_for_group(raw_dir, group, source_data_root):
        raw, time_col = load_xsens_file(path)
        score = sanity_score_for_source(raw, time_col, base_g)
        rows.append({"path": path, "raw": raw, "time_col": time_col, "sanity_score": score})

    if len(rows) == 0:
        raise FileNotFoundError(f"No Xsens file found for group {group}")

    rows = sorted(rows, key=lambda r: r["sanity_score"], reverse=True)
    return rows[0], rows


# ================================================================
# Per-window feature computation (CELL 16's `compute_xsens_window_
# features` — verbatim)
# ================================================================

def compute_xsens_window_features(sub: pd.DataFrame, group: int, ws: float, we: float, time_col: str) -> dict:
    out = {"group": group, "window_start": ws, "window_end": we}

    if len(sub) == 0:
        return out

    sub = standardize_columns(sub)
    t = get_num_col(sub, time_col)

    participant_data = {}

    for p in (1, 2, 3):
        prefix = f"xsens2_p{p}"

        avail_col = f"p{p}_xsens_available"
        if avail_col in sub.columns:
            avail = get_num_col(sub, avail_col)
            out[f"{prefix}_available_frac"] = np.nanmean(avail) if np.isfinite(avail).sum() else np.nan
        else:
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

        acc_x = clean_signal(get_num_col(sub, f"p{p}_acc_x"), "acc")
        acc_y = clean_signal(get_num_col(sub, f"p{p}_acc_y"), "acc")
        acc_z = clean_signal(get_num_col(sub, f"p{p}_acc_z"), "acc")

        gyr_x = clean_signal(get_num_col(sub, f"p{p}_gyr_x"), "gyr")
        gyr_y = clean_signal(get_num_col(sub, f"p{p}_gyr_y"), "gyr")
        gyr_z = clean_signal(get_num_col(sub, f"p{p}_gyr_z"), "gyr")

        eul_x = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_x"), "euler"))
        eul_y = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_y"), "euler"))
        eul_z = unwrap_degrees(clean_signal(get_num_col(sub, f"p{p}_euler_z"), "euler"))

        for axis_name, arr in [
            ("acc_x", acc_x), ("acc_y", acc_y), ("acc_z", acc_z),
            ("gyr_x", gyr_x), ("gyr_y", gyr_y), ("gyr_z", gyr_z),
            ("euler_x", eul_x), ("euler_y", eul_y), ("euler_z", eul_z),
        ]:
            safe_stats(out, f"{prefix}_{axis_name}", arr)

        acc_norm = vector_norm(acc_x, acc_y, acc_z)
        gyr_norm = vector_norm(gyr_x, gyr_y, gyr_z)

        acc_x_dyn = acc_x - np.nanmedian(acc_x)
        acc_y_dyn = acc_y - np.nanmedian(acc_y)
        acc_z_dyn = acc_z - np.nanmedian(acc_z)
        acc_dyn_norm = vector_norm(acc_x_dyn, acc_y_dyn, acc_z_dyn)

        jerk_norm = derivative_norm(acc_x_dyn, acc_y_dyn, acc_z_dyn, t, max_value=500.0)
        angular_jerk_norm = derivative_norm(gyr_x, gyr_y, gyr_z, t, max_value=10000.0)

        eul_rate_x = derivative_1d(eul_x, t, max_value=1000.0)
        eul_rate_y = derivative_1d(eul_y, t, max_value=1000.0)
        eul_rate_z = derivative_1d(eul_z, t, max_value=1000.0)
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

        safe_fraction(out, f"{prefix}_acc_dyn_gt_0p5_frac", acc_dyn_norm > 0.5)
        safe_fraction(out, f"{prefix}_acc_dyn_gt_1p0_frac", acc_dyn_norm > 1.0)
        safe_fraction(out, f"{prefix}_acc_dyn_gt_2p0_frac", acc_dyn_norm > 2.0)

        safe_fraction(out, f"{prefix}_gyr_gt_30_frac", gyr_norm > 30.0)
        safe_fraction(out, f"{prefix}_gyr_gt_60_frac", gyr_norm > 60.0)
        safe_fraction(out, f"{prefix}_gyr_gt_100_frac", gyr_norm > 100.0)

        spectral_features(out, f"{prefix}_acc_dyn_norm", acc_dyn_norm, t)
        spectral_features(out, f"{prefix}_gyr_norm", gyr_norm, t)

        participant_data[p] = {
            "acc_dyn_norm": acc_dyn_norm,
            "gyr_norm": gyr_norm,
            "acc_norm": acc_norm,
            "euler_x": eul_x, "euler_y": eul_y, "euler_z": eul_z,
            "acc_dyn_mean": np.nanmean(acc_dyn_norm),
            "gyr_mean": np.nanmean(gyr_norm),
            "jerk_mean": np.nanmean(jerk_norm) if len(jerk_norm) else np.nan,
            "angular_jerk_mean": np.nanmean(angular_jerk_norm) if len(angular_jerk_norm) else np.nan,
        }

    acc_active = []
    gyr_active = []
    for p in (1, 2, 3):
        acc_active.append(participant_data[p]["acc_dyn_norm"] > 1.0)
        gyr_active.append(participant_data[p]["gyr_norm"] > 60.0)

    acc_active = np.vstack(acc_active).T
    gyr_active = np.vstack(gyr_active).T
    acc_count = acc_active.sum(axis=1)
    gyr_count = gyr_active.sum(axis=1)

    safe_stats(out, "xsens2_acc_active_count", acc_count)
    safe_stats(out, "xsens2_gyr_active_count", gyr_count)

    for k in (0, 1, 2, 3):
        safe_fraction(out, f"xsens2_acc_active_count_exactly{k}_frac", acc_count == k)
        safe_fraction(out, f"xsens2_gyr_active_count_exactly{k}_frac", gyr_count == k)

    safe_fraction(out, "xsens2_acc_active_count_atleast2_frac", acc_count >= 2)
    safe_fraction(out, "xsens2_acc_active_count_all3_frac", acc_count == 3)
    safe_fraction(out, "xsens2_gyr_active_count_atleast2_frac", gyr_count >= 2)
    safe_fraction(out, "xsens2_gyr_active_count_all3_frac", gyr_count == 3)

    pair_values = {"acc_dyn_corr": [], "gyr_corr": [], "acc_dyn_absdiff_mean": [], "gyr_absdiff_mean": []}

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

    for name in ("acc_dyn_mean", "gyr_mean", "jerk_mean", "angular_jerk_mean"):
        vals = [participant_data[p][name] for p in (1, 2, 3)]
        safe_stats(out, f"xsens2_person_summary_{name}", vals)

    return out


# ================================================================
# Driver
# ================================================================

def build_task2_xsens_features(
    raw_dir: str,
    base_windows: pd.DataFrame,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> pd.DataFrame:
    """Verbatim port of CELL 16's main loop. `base_windows` must have
    `group`/`window_start`/`window_end`/`recognition_label`/`elapsed_min`
    columns (the output of `eng_task2_grid.build_task2_base_windows`).
    Returns a DataFrame with `group`/`window_start`/`window_end`, every
    `xsens2_*` feature column, `recognition_label`, `elapsed_min`."""
    if groups is None:
        groups = sorted(int(g) for g in base_windows["group"].unique())

    all_rows = []

    for group in groups:
        base_g = base_windows[base_windows["group"] == group].copy().reset_index(drop=True)
        if len(base_g) == 0:
            continue

        chosen, _candidates = choose_best_xsens_source(raw_dir, group, base_g, source_data_root)
        raw = chosen["raw"]
        time_col = chosen["time_col"]
        t_raw = raw[time_col].to_numpy()

        for _, w in base_g.iterrows():
            ws = float(w["window_start"])
            we = float(w["window_end"])

            start_idx = np.searchsorted(t_raw, ws, side="left")
            end_idx = np.searchsorted(t_raw, we, side="left")
            sub = raw.iloc[start_idx:end_idx].copy()

            feat = compute_xsens_window_features(sub=sub, group=group, ws=ws, we=we, time_col=time_col)
            feat["recognition_label"] = w["recognition_label"]
            feat["elapsed_min"] = w["elapsed_min"]

            all_rows.append(feat)

        del raw

    xsens2 = pd.DataFrame(all_rows)

    # NOTE: CELL 16 drops all-NaN columns here, but that's a decision made
    # after building ALL 9 groups together — for a single-group rebuild
    # (this module's scope) dropping columns that happen to be all-NaN for
    # just this group would silently diverge from the official file's own
    # (multi-group) column set. Left un-dropped deliberately; the
    # validation script compares only columns the official file actually
    # has.

    return xsens2
