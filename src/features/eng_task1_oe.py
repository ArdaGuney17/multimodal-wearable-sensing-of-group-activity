"""Feature engineering, Task 1 OpenEarable (OE) half — the two OE-only
feature families that feed `binary_5s_specialized_oe_merged_all_features.csv`,
Task 1's actual 5s-binary-interaction model input (Ch.5, and Ch.7 Tables
7.9/7.10's shared upstream file for the SPECIAL_OE feature set).

Ported from `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`
(`notebooks_reference/master_feature_generator_task1_task2_task3_CORRECTED_
V4_ANALYSIS.md`). Scope is deliberately restricted to the OpenEarable-only
pieces — no OptiTrack (OPTI2), no Xsens (XSENS2); those are out of scope
for this port (see docs/table_to_source_mapping.md, "ENG3 family" row).

**Circularity note (why this is legitimate to port).** The notebook's own
CELL 3 ("A1. AUTHORITATIVE FIVE-SECOND BINARY WINDOW AND LABEL GRID")
admits the original from-scratch window/label-grid generator was lost, and
bootstraps `base_windows` (group, window_start, window_end, binary_label,
...) by copying those columns straight out of an already-existing official
Task 1 output CSV. That part genuinely is circular and is NOT re-derived
here — `load_base_windows()` below does the same thing openly: it reads
group/window_start/window_end/binary_label (+ a few metadata columns) from
the official file. What is NOT circular, and is what this module actually
computes, is every OE feature VALUE: both `build_generic_oe_5s` (CELL 6,
"A2. RECONSTRUCTED GENERIC OPENEAREABLE FEATURES") and
`build_specialized_oe_5s` (CELL 8, "SPECIALIZED OE9/OE10-STYLE FEATURES
FOR 5s BINARY TASK") independently recompute every feature column from
raw `group_{g}_openearable_model_ready.csv` data — the window boundaries
are reused, but the numbers inside each window are not. The notebook's own
CELL 24 ("D1. VERIFY REBUILT OUTPUTS AGAINST OFFICIAL TABLES") checks this
by diffing its rebuilt output against the official file; this module's
companion validation script
(`data/external/thesis_data/RAW_VALIDATION_FEATURES/
run_group1_oe_features_validation.py`) does the same thing independently.

Ported pieces, by source cell:
  - `adv_stats`, `load_raw_file`, `num` — CELL 4 ("SECTION 0b — advanced
    statistic expansion"), the shared helpers cell. Only the parts the OE
    cells actually use are kept (no OPTI2/XSENS2-specific bits).
  - `build_generic_oe_5s` — CELL 6 ("A2. RECONSTRUCTED GENERIC
    OPENEAREABLE FEATURES"): the `oe__<channel>__<stat>` 16-stat-per-raw-
    channel family (`adv_stats` applied to each of the 27
    `p{1,2,3}_{acc,gyro,mag}_{x,y,z}` raw channels). These columns are
    dropped again by the final merge below, exactly as CELL 9 does in the
    notebook — kept here only for architectural parity/traceability with
    the source, and because CELL 9's own diff (drop generic, add
    specialized) is part of what this module's driver reproduces.
  - `session_baseline`, `extract_specialized_oe_features`,
    `build_specialized_oe_5s` — CELL 8 ("SPECIALIZED OE9/OE10-STYLE
    FEATURES FOR 5s BINARY TASK"), the actual Task 1 OE feature set: head
    posture/turn/nod features, rich self-normalized per-person acc/gyro
    stats (energy/entropy/band-ratio/pitch/roll/jerk), active-person
    counts, cross-person correlation/lag-correlation, first/second-half
    deltas, and (if present) magnetometer heading/magnitude features.
    This cell is a NEAR-duplicate of `oe9_oe10_features.py`'s
    `extract_oe9_features`/`extract_mag_features` (ported earlier from a
    different source notebook for Table 7.9's 10s windows) but is NOT
    byte-identical: this cell (a) uses variable-length windows sized from
    `we - ws` with `round()` and a `max(8, ...)` floor rather than a fixed
    `WINDOW_S`, (b) folds the magnetometer features into the SAME
    function/window loop rather than a separate merge pass, and (c) uses
    a `sinterp` that NaNs any grid point outside the raw signal's own time
    range — `oe_signal_common.sinterp` does not do that clamp. Per the
    task's own instruction, these differences are real (not just
    stylistic), so this module ports CELL 8's `extract_specialized_oe_
    features` as its own function with its own local `_sinterp`, rather
    than silently reusing `oe9_oe10_features`'s version. All the smaller
    stat helpers below (`safe_mean/std/min/max/range/iqr/percentile`,
    `mad_diff`, `event_count`, `spectral_entropy`, `band_ratio`,
    `corr_safe`, `max_lag_corr`, `circular_diff`, `aggregate_values`) were
    checked line-by-line against `oe_signal_common.py`'s copies and found
    identical, so those ARE reused directly.
  - `load_base_windows` / `build_task1_oe_features` — the driver. Loads
    the window/label grid from the official file (CELL 3's approach,
    acknowledged as the bootstrapped part above), loads one group's raw
    OpenEarable file, computes both feature families, then merges them the
    way CELL 9 does: drop the generic `oe__*` columns, left-merge the
    specialized OE table on `["group","window_start","window_end"]`.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from src.features.oe_signal_common import (
    aggregate_values,
    band_ratio,
    circular_diff,
    corr_safe,
    event_count,
    mad_diff,
    max_lag_corr,
    safe_iqr,
    safe_mean,
    safe_percentile,
    safe_range,
    safe_std,
    spectral_entropy,
)

GROUP_COL = "group"
START_COL = "window_start"
END_COL = "window_end"
LABEL_COL = "binary_label"

RESAMPLE_HZ = 25

NOD_BAND = (1.0, 3.0)
LOW_MOTION_BAND = (0.2, 1.0)
HIGH_MOTION_BAND = (3.0, 8.0)

MAG_LOW_BAND = (0.2, 1.0)
MAG_MID_BAND = (1.0, 3.0)
MAG_HIGH_BAND = (3.0, 8.0)

DOWN_THR = -0.35  # rad, ~-20 deg head tilted down
PAIRS = [(1, 2), (1, 3), (2, 3)]

EAR_ACC = lambda p: [f"p{p}_acc_{a}" for a in "xyz"]  # noqa: E731
EAR_GYR = lambda p: [f"p{p}_gyro_{a}" for a in "xyz"]  # noqa: E731
MAG_COLS = sum([[f"p{p}_mag_{a}" for a in "xyz"] for p in (1, 2, 3)], [])
ALL_OE_COLS = sum([EAR_ACC(p) + EAR_GYR(p) for p in (1, 2, 3)], []) + MAG_COLS

# Official Task 1 grid files the notebook's CELL 3 bootstraps from
# (checked in this order; the first one found is used).
TASK1_LABEL_GRID_SOURCE_CANDIDATES = [
    "INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv",
    "INTERACTION_BINARY_5S_ADVANCED_FEATURES/binary_5s_all_sensor_advanced_features.csv",
]

PREFERRED_GRID_COLUMNS = [
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


def group_number(value) -> int | None:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


# ================================================================
# Shared helpers (CELL 4, "SECTION 0b" — only the OE-relevant parts)
# ================================================================

def adv_stats(out: dict, name: str, x, t=None) -> None:
    """Writes 16 stats (`{name}__mean` ... `{name}__slope`) into `out`.
    Verbatim port of CELL 4's `adv_stats`."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if fin.sum() == 0:
        for s in [
            "mean", "std", "min", "max", "range", "median", "iqr", "p10", "p90",
            "energy", "rms", "first_half_mean", "second_half_mean", "half_delta",
            "mad", "slope",
        ]:
            out[f"{name}__{s}"] = np.nan
        return
    v = x[fin]
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    out[f"{name}__mean"] = float(v.mean())
    out[f"{name}__std"] = float(v.std())
    out[f"{name}__min"] = float(v.min())
    out[f"{name}__max"] = float(v.max())
    out[f"{name}__range"] = float(v.max() - v.min())
    out[f"{name}__median"] = float(q50)
    out[f"{name}__iqr"] = float(q75 - q25)
    out[f"{name}__p10"] = float(q10)
    out[f"{name}__p90"] = float(q90)
    out[f"{name}__energy"] = float(np.sum(v ** 2))
    out[f"{name}__rms"] = float(np.sqrt(np.mean(v ** 2)))
    h = len(x) // 2
    fh, sh = x[:h], x[h:]
    fh_m = float(np.nanmean(fh)) if np.isfinite(fh).any() else np.nan
    sh_m = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    out[f"{name}__first_half_mean"] = fh_m
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


def load_raw_file(raw_dir: str, kind: str, gnum: int) -> pd.DataFrame | None:
    """Verbatim port of CELL 4's `load_raw_file`."""
    p = os.path.join(raw_dir, f"group_{gnum}_{kind}_model_ready.csv")
    if not os.path.exists(p):
        print(f"  WARNING: missing {p}")
        return None
    r = pd.read_csv(p, low_memory=False)
    tcol = "video_time_s" if "video_time_s" in r.columns else "time_s"
    r["t"] = pd.to_numeric(r[tcol], errors="coerce")
    return r.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def num(r: pd.DataFrame, name: str, clean: float = 1e6) -> np.ndarray:
    """Verbatim port of CELL 4's `num`."""
    if name in r.columns:
        v = pd.to_numeric(r[name], errors="coerce").to_numpy()
        return np.where(np.abs(v) < clean, v, np.nan)
    return np.full(len(r), np.nan)


# ================================================================
# A2. Reconstructed generic OpenEarable features (CELL 6)
# ================================================================

def build_generic_oe_5s(raw_dir: str, base_windows: pd.DataFrame, groups: list[int] | None = None) -> pd.DataFrame:
    """Verbatim port of CELL 6. For every raw `p[123]_(acc|gyro|mag)_[xyz]`
    channel, applies `adv_stats` inside each window -> `oe__<channel>__<stat>`
    columns. These are dropped again by `build_task1_oe_features` (mirrors
    CELL 9's drop-generic/add-specialized merge), kept only for parity with
    the source notebook's own A2/A3 architecture."""
    rows = []

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )

    for group in groups:
        oe = load_raw_file(raw_dir, "openearable", group)
        if oe is None:
            print(f"WARNING: no OpenEarable file for group {group}")
            continue

        group_windows = (
            base_windows[base_windows["group_num"] == group].sort_values(START_COL).reset_index(drop=True)
        )

        raw_channels = [c for c in oe.columns if re.match(r"^p[123]_(acc|gyro|mag)_[xyz]$", str(c))]

        for column in raw_channels:
            oe[column] = pd.to_numeric(oe[column], errors="coerce")
            oe[column] = oe[column].where(oe[column].abs() < 1e6, np.nan)

        t = oe["t"].to_numpy()

        for _, window in group_windows.iterrows():
            ws = float(window[START_COL])
            we = float(window[END_COL])
            mask = (t >= ws) & (t < we)

            record = {GROUP_COL: window[GROUP_COL], START_COL: ws, END_COL: we}

            if mask.sum() >= 3:
                for column in raw_channels:
                    adv_stats(record, f"oe__{column}", oe.loc[mask, column].to_numpy(), t[mask])

            rows.append(record)

        del oe

    return pd.DataFrame(rows)


# ================================================================
# Specialized OE9/OE10-style features for the 5s binary task (CELL 8)
# ================================================================

def _sinterp(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Local copy of CELL 8's `sinterp`. Differs from
    `oe_signal_common.sinterp` by NaN-ing grid points outside the raw
    signal's own [min(t), max(t)] range after interpolation — that clamp
    is present in this notebook cell but not in the ENG7-notebook copy
    `oe_signal_common.sinterp` was consolidated from."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    out = np.interp(grid, t[m], v[m])
    out[(grid < np.nanmin(t[m])) | (grid > np.nanmax(t[m]))] = np.nan
    return out


def discover_openearable(folder: str) -> dict:
    """Verbatim port of CELL 8's `discover_openearable`."""
    import glob

    found = {}
    for path in glob.glob(os.path.join(folder, "group_*_openearable_model_ready.csv")):
        m = re.search(r"group_(\d+)_openearable_model_ready", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def load_oe(path: str) -> pd.DataFrame:
    """Verbatim port of CELL 8's `load_oe`."""
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


def session_baseline(oe: pd.DataFrame) -> dict:
    """Verbatim port of CELL 8's `session_baseline`."""
    sess = {}
    for p in (1, 2, 3):
        acc = np.stack([oe[f"p{p}_acc_{a}"].values for a in "xyz"], axis=1)
        gyr = np.stack([oe[f"p{p}_gyro_{a}"].values for a in "xyz"], axis=1)
        acc_mag = np.linalg.norm(acc, axis=1)
        gyr_mag = np.linalg.norm(gyr, axis=1)
        pitch = np.arctan2(acc[:, 0], np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2))
        roll = np.arctan2(acc[:, 1], np.sqrt(acc[:, 0] ** 2 + acc[:, 2] ** 2))
        sess[p] = {
            "acc_mean": safe_mean(acc_mag), "acc_std": safe_std(acc_mag),
            "gyr_mean": safe_mean(gyr_mag), "gyr_std": safe_std(gyr_mag),
            "pitch_mean": safe_mean(pitch), "pitch_std": safe_std(pitch),
            "roll_mean": safe_mean(roll), "roll_std": safe_std(roll),
            "gyr_p75": safe_percentile(gyr_mag, 75), "gyr_p80": safe_percentile(gyr_mag, 80),
            "gyr_p90": safe_percentile(gyr_mag, 90),
            "acc_p75": safe_percentile(acc_mag, 75), "acc_p90": safe_percentile(acc_mag, 90),
        }
    return sess


def extract_specialized_oe_features(oe: pd.DataFrame, ws: float, we: float, sess: dict) -> dict:
    """Verbatim port of CELL 8's `extract_specialized_oe_features` — the
    ~300-line specialized OE9/OE10-style feature function. Handles
    variable-length windows (`win_s = we - ws`, `n = max(8, round(win_s *
    RESAMPLE_HZ))`), unlike `oe9_oe10_features.extract_oe9_features`'s
    fixed 10s windows."""
    win_s = float(we - ws)
    n = max(8, int(round(win_s * RESAMPLE_HZ)))
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ

    row: dict = {}

    acc, gyr, acc_mag, gyr_mag, acc_e, gyr_e = {}, {}, {}, {}, {}, {}
    pitch, roll, vert, jerk, angular_jerk = {}, {}, {}, {}, {}

    for p in (1, 2, 3):
        acc[p] = np.stack([_sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values) for a in "xyz"], axis=1)
        gyr[p] = np.stack([_sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values) for a in "xyz"], axis=1)

        acc_mag[p] = np.linalg.norm(acc[p], axis=1)
        gyr_mag[p] = np.linalg.norm(gyr[p], axis=1)

        acc_e[p] = (acc_mag[p] - sess[p]["acc_mean"]) / (sess[p]["acc_std"] + 1e-9)
        gyr_e[p] = (gyr_mag[p] - sess[p]["gyr_mean"]) / (sess[p]["gyr_std"] + 1e-9)

        pitch[p] = np.arctan2(acc[p][:, 0], np.sqrt(acc[p][:, 1] ** 2 + acc[p][:, 2] ** 2))
        roll[p] = np.arctan2(acc[p][:, 1], np.sqrt(acc[p][:, 0] ** 2 + acc[p][:, 2] ** 2))
        vert[p] = acc[p][:, 2]

        jerk[p] = np.r_[0, np.diff(acc_mag[p])] * fs
        angular_jerk[p] = np.r_[0, np.diff(gyr_mag[p])] * fs

    down_thr = DOWN_THR

    down_fracs, pitch_ranges, switch_rates, turn_rates, nod_ratios = [], [], [], [], []

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

    head_active_binary = {p: (gyr_e[p] > 0).astype(float) for p in (1, 2, 3)}

    pair_corrs = []
    for a, b in PAIRS:
        c = corr_safe(head_active_binary[a], head_active_binary[b])
        if np.isfinite(c):
            pair_corrs.append(c)

    row["ear_head_activity_alternation"] = float(-np.mean(pair_corrs)) if len(pair_corrs) else np.nan

    per_person_feature_values: dict = {}

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

    collect("head_movement_frequency", [event_count(gyr_e[p] > 0) / win_s for p in (1, 2, 3)])
    collect("head_turn_frequency", [event_count(gyr_mag[p] > sess[p]["gyr_p80"]) / win_s for p in (1, 2, 3)])
    collect("head_nod_frequency_band", [band_ratio(vert[p], fs, *NOD_BAND) for p in (1, 2, 3)])
    collect("head_posture_switch_frequency", [event_count(pitch[p] < down_thr) / win_s for p in (1, 2, 3)])

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

    for name, vals in per_person_feature_values.items():
        vals = np.asarray(vals, dtype=float)
        if np.isfinite(vals).sum() >= 2:
            row[f"oe_{name}_dominance_gap"] = float(np.nanmax(vals) - np.nanmedian(vals))
            row[f"oe_{name}_asymmetry"] = float(np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-9))
        else:
            row[f"oe_{name}_dominance_gap"] = np.nan
            row[f"oe_{name}_asymmetry"] = np.nan

    pair_acc_corrs, pair_gyro_corrs, pair_pitch_corrs = [], [], []
    pair_acc_lagcorrs, pair_gyro_lagcorrs = [], []

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

    half = len(grid) // 2

    for signal_name, signals in [("acc_e", acc_e), ("gyro_e", gyr_e), ("pitch", pitch)]:
        deltas = []
        for p in (1, 2, 3):
            x = signals[p]
            if len(x) >= 4:
                deltas.append(safe_mean(x[half:]) - safe_mean(x[:half]))
            else:
                deltas.append(np.nan)
        aggregate_values(row, f"oe_{signal_name}_half_delta", deltas)

    mag, mag_norm, heading, heading_raw = {}, {}, {}, {}

    for p in (1, 2, 3):
        mag[p] = np.stack([_sinterp(grid, oe["t"].values, oe[f"p{p}_mag_{a}"].values) for a in "xyz"], axis=1)
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
        collect(
            "mag_heading_turn_frequency",
            [event_count(np.abs(np.r_[0, np.diff(heading[p])]) > 0.05) / win_s for p in (1, 2, 3)],
        )
        collect("mag_heading_low_band", [band_ratio(heading[p], fs, *MAG_LOW_BAND) for p in (1, 2, 3)])
        collect("mag_heading_mid_band", [band_ratio(heading[p], fs, *MAG_MID_BAND) for p in (1, 2, 3)])
        collect("mag_heading_high_band", [band_ratio(heading[p], fs, *MAG_HIGH_BAND) for p in (1, 2, 3)])

        pair_mag_corrs, pair_heading_corrs = [], []
        pair_heading_diff_std, pair_heading_diff_maddiff = [], []

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


def build_specialized_oe_5s(raw_dir: str, base_df: pd.DataFrame, groups: list[int] | None = None) -> pd.DataFrame:
    """Verbatim port of CELL 8's "BUILD SPECIALIZED OE FEATURES FOR EXACT
    BINARY WINDOWS" section. `base_df` needs `group`/`window_start`/
    `window_end` columns (the official grid, or any grid with those three
    columns)."""
    files = discover_openearable(raw_dir)

    if groups is None:
        groups = sorted(int(g) for g in base_df["group"].unique())

    rows = []

    for group in groups:
        if group not in files:
            print(f"WARNING: no OpenEarable file for group {group}, skipping.")
            continue

        oe = load_oe(files[group])
        sess = session_baseline(oe)

        gw = base_df[base_df["group"] == group][["group", "window_start", "window_end"]].copy()
        gw = gw.sort_values("window_start").reset_index(drop=True)

        for _, w in gw.iterrows():
            ws = float(w["window_start"])
            we = float(w["window_end"])

            row = extract_specialized_oe_features(oe, ws, we, sess)
            row["group"] = group
            row["window_start"] = ws
            row["window_end"] = we

            rows.append(row)

        del oe

    return pd.DataFrame(rows)


# ================================================================
# Driver
# ================================================================

def load_base_windows(official_grid_path: str) -> pd.DataFrame:
    """Port of CELL 3's grid-bootstrap logic, minus the "write it to a
    small standalone CSV" step (not needed here — we just read it back
    into memory). Reads group/window_start/window_end/binary_label (+
    available metadata columns) straight from the official Task 1 file.
    This is the acknowledged non-independent part of the pipeline — see
    the module docstring."""
    source = pd.read_csv(official_grid_path, low_memory=False)

    grid_columns = [c for c in PREFERRED_GRID_COLUMNS if c in source.columns]
    required = {"group", "window_start", "window_end", "binary_label"}
    if not required.issubset(grid_columns):
        raise ValueError(
            "The selected Task 1 source does not contain the required "
            "group, window_start, window_end and binary_label columns."
        )

    base_windows = source[grid_columns].copy()
    base_windows[LABEL_COL] = base_windows[LABEL_COL].astype(str).str.strip()
    base_windows["group_num"] = base_windows[GROUP_COL].map(group_number)
    return base_windows


def build_task1_oe_features(raw_dir: str, official_grid_path: str, groups: list[int] | None = None) -> dict:
    """Loads the official window/label grid, computes both OE feature
    families for `groups` (default: every group present in the grid),
    and merges them the way CELL 9 does: drop the generic `oe__*`
    columns from a `base_windows + generic_oe` merge, then left-merge the
    specialized OE table on `["group","window_start","window_end"]`.

    Returns a dict with `generic_oe`, `specialized_oe`, and `merged`
    (the final table — grid columns + specialized OE columns only).
    """
    base_windows = load_base_windows(official_grid_path)

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )
    else:
        base_windows = base_windows[base_windows["group_num"].isin(groups)].reset_index(drop=True)

    generic_oe = build_generic_oe_5s(raw_dir, base_windows, groups=groups)
    specialized_oe = build_specialized_oe_5s(raw_dir, base_windows, groups=groups)

    merge_keys = [GROUP_COL, START_COL, END_COL]

    task1_base = base_windows.drop(columns=["group_num"], errors="ignore").copy()
    if len(generic_oe) > 0:
        task1_base = task1_base.merge(
            generic_oe.drop_duplicates(merge_keys), on=merge_keys, how="left", validate="one_to_one"
        )

    generic_oe_cols = [c for c in task1_base.columns if c.startswith("oe__")]
    merged = task1_base.drop(columns=generic_oe_cols)
    merged = merged.merge(specialized_oe, on=merge_keys, how="left")

    return {"base_windows": base_windows, "generic_oe": generic_oe, "specialized_oe": specialized_oe, "merged": merged}
