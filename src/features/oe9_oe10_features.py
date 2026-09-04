"""Feature engineering, OE9/OE10 family — the rich, self-normalized
OpenEarable acc/gyro (+ optional magnetometer) feature tables that supply
"OE9 motion + MAG magnitude" (Table 7.9 row 1, 305 features) and the
`oe_best` feature set every other Table 7.9 row extends (Ch.7 §7.5.2).

Ported from `07_feature_engineering_ENG7_activity_invariant.ipynb`
(`notebooks_reference/07_feature_engineering_ENG7_activity_invariant_
EXECUTED_CODE_ONLY.py`):
  - CELL 10 (lines 389-1038) — `build_oe9`/`extract_oe9_features`: 10s/10s
    window grid built straight from each group's raw
    `group_{g}_openearable_model_ready.csv`. Every per-person acc/gyro
    signal is self-normalized by that person's own session mean/std
    before thresholding (the source notebook's stated design goal —
    features that don't depend on who wears the sensor). Legacy `ear_`
    features (old ENG7-style head posture/gaze) are kept alongside the
    richer new `oe_` features (energy/entropy/band-ratio/pitch/roll/
    jerk/active-count/dominance/cross-person-correlation), exactly as in
    the source.
  - CELL 15 (lines 2050-2621) — `extract_mag_features`: magnetometer-only
    features (`mag_*`), computed the same way from the same raw
    OpenEarable files, left-merged onto the OE9 table on
    `["group","window_start","window_end"]` to produce OE10.

Magic numbers preserved verbatim: `WINDOW_S=STRIDE_S=10.0`,
`RESAMPLE_HZ=25`, `NOD_BAND=(1.0,3.0)`, `LOW_MOTION_BAND=(0.2,1.0)`,
`HIGH_MOTION_BAND=(3.0,8.0)`, `MAG_LOW_BAND=(0.2,1.0)`,
`MAG_MID_BAND=(1.0,3.0)`, `MAG_HIGH_BAND=(3.0,8.0)`, head-down pitch
threshold `-0.35` rad, `max_lag_steps = int(1.0*fs)`.

Neither builder reads OptiTrack or Xsens — both are OpenEarable-only by
design (that's the point of "OE9"/"OE10": maximize 3-class recognition
using only ear-worn acc/gyro/magnetometer). Group discovery is therefore
single-sensor (`oe_signal_common.discover_single_sensor_files`), not
`eng2_features.discover_model_ready_groups`'s "all 3 sensors" rule.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.oe_signal_common import (
    aggregate_values,
    band_ratio,
    circular_diff,
    corr_safe,
    discover_single_sensor_files,
    event_count,
    mad_diff,
    max_lag_corr,
    safe_iqr,
    safe_mean,
    safe_percentile,
    safe_range,
    safe_std,
    sinterp,
    spectral_entropy,
)

WINDOW_S = 10.0
STRIDE_S = 10.0
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
ALL_OE_COLS = sum([EAR_ACC(p) + EAR_GYR(p) for p in (1, 2, 3)], [])
MAG_COLS = sum([[f"p{p}_mag_{a}" for a in "xyz"] for p in (1, 2, 3)], [])


# ================================================================
# IO
# ================================================================

def _load_oe(path: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    time_col = "video_time_s" if "video_time_s" in df.columns else ("time_s" if "time_s" in df.columns else None)
    if time_col is None:
        raise ValueError(f"No video_time_s or time_s column found in {path}.")
    df["t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e6, np.nan)
    return df


def _load_mag(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    time_col = "video_time_s" if "video_time_s" in df.columns else ("time_s" if "time_s" in df.columns else None)
    if time_col is None:
        raise ValueError(f"No usable time column found in {path}.")
    df["t"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    for c in MAG_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e9, np.nan)
    return df


# ================================================================
# OE9 — session baselines + per-window features (CELL 10)
# ================================================================

def session_baseline_oe9(oe: pd.DataFrame) -> dict:
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
            "gyr_p75": safe_percentile(gyr_mag, 75), "gyr_p80": safe_percentile(gyr_mag, 80), "gyr_p90": safe_percentile(gyr_mag, 90),
            "acc_p75": safe_percentile(acc_mag, 75), "acc_p90": safe_percentile(acc_mag, 90),
        }
    return sess


def extract_oe9_features(oe: pd.DataFrame, ws: float, we: float, sess: dict) -> dict:
    """Verbatim port of CELL 10's `extract_oe9_features`."""
    n = int(WINDOW_S * RESAMPLE_HZ)
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ
    row: dict = {}

    acc, gyr, acc_mag, gyr_mag, acc_e, gyr_e = {}, {}, {}, {}, {}, {}
    pitch, roll, vert, jerk, angular_jerk = {}, {}, {}, {}, {}

    for p in (1, 2, 3):
        acc[p] = np.stack([sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values) for a in "xyz"], axis=1)
        gyr[p] = np.stack([sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values) for a in "xyz"], axis=1)
        acc_mag[p] = np.linalg.norm(acc[p], axis=1)
        gyr_mag[p] = np.linalg.norm(gyr[p], axis=1)
        acc_e[p] = (acc_mag[p] - sess[p]["acc_mean"]) / (sess[p]["acc_std"] + 1e-9)
        gyr_e[p] = (gyr_mag[p] - sess[p]["gyr_mean"]) / (sess[p]["gyr_std"] + 1e-9)
        pitch[p] = np.arctan2(acc[p][:, 0], np.sqrt(acc[p][:, 1] ** 2 + acc[p][:, 2] ** 2))
        roll[p] = np.arctan2(acc[p][:, 1], np.sqrt(acc[p][:, 0] ** 2 + acc[p][:, 2] ** 2))
        vert[p] = acc[p][:, 2]
        jerk[p] = np.r_[0, np.diff(acc_mag[p])] * fs
        angular_jerk[p] = np.r_[0, np.diff(gyr_mag[p])] * fs

    down_fracs, pitch_ranges, switch_rates, turn_rates, nod_ratios = [], [], [], [], []
    for p in (1, 2, 3):
        down = pitch[p] < DOWN_THR
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

    head_active_binary = {p: (gyr_e[p] > 0).astype(float) for p in (1, 2, 3)}
    pair_corrs = [c for a, b in PAIRS if np.isfinite(c := corr_safe(head_active_binary[a], head_active_binary[b]))]
    row["ear_head_activity_alternation"] = float(-np.mean(pair_corrs)) if pair_corrs else np.nan

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

    collect("down_fraction", [np.nanmean(pitch[p] < DOWN_THR) for p in (1, 2, 3)])
    collect("up_fraction", [np.nanmean(pitch[p] >= DOWN_THR) for p in (1, 2, 3)])
    collect("turn_rate_p75", [event_count(gyr_mag[p] > sess[p]["gyr_p75"]) / WINDOW_S for p in (1, 2, 3)])
    collect("turn_rate_p90", [event_count(gyr_mag[p] > sess[p]["gyr_p90"]) / WINDOW_S for p in (1, 2, 3)])
    collect("acc_burst_rate_p75", [event_count(acc_mag[p] > sess[p]["acc_p75"]) / WINDOW_S for p in (1, 2, 3)])
    collect("acc_burst_rate_p90", [event_count(acc_mag[p] > sess[p]["acc_p90"]) / WINDOW_S for p in (1, 2, 3)])

    acc_active = np.stack([(acc_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    gyro_active = np.stack([(gyr_e[p] > 0).astype(int) for p in (1, 2, 3)], axis=0)
    down_active = np.stack([(pitch[p] < DOWN_THR).astype(int) for p in (1, 2, 3)], axis=0)

    for name, count in [("acc_active_count", acc_active.sum(axis=0)), ("gyro_active_count", gyro_active.sum(axis=0)), ("down_count", down_active.sum(axis=0))]:
        row[f"oe_{name}_mean"] = safe_mean(count)
        row[f"oe_{name}_std"] = safe_std(count)
        row[f"oe_{name}_exactly1_frac"] = float(np.mean(count == 1))
        row[f"oe_{name}_atleast2_frac"] = float(np.mean(count >= 2))
        row[f"oe_{name}_all3_frac"] = float(np.mean(count == 3))
        row[f"oe_{name}_switch_rate"] = event_count(np.r_[False, np.diff(count) != 0]) / WINDOW_S

    for name, vals in per_person_feature_values.items():
        vals = np.asarray(vals, dtype=float)
        if np.isfinite(vals).sum() >= 2:
            row[f"oe_{name}_dominance_gap"] = float(np.nanmax(vals) - np.nanmedian(vals))
            row[f"oe_{name}_asymmetry"] = float(np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-9))
        else:
            row[f"oe_{name}_dominance_gap"] = np.nan
            row[f"oe_{name}_asymmetry"] = np.nan

    pair_acc_corrs, pair_gyro_corrs, pair_pitch_corrs, pair_acc_lagcorrs, pair_gyro_lagcorrs = [], [], [], [], []
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
            deltas.append(safe_mean(x[half:]) - safe_mean(x[:half]) if len(x) >= 4 else np.nan)
        aggregate_values(row, f"oe_{signal_name}_half_delta", deltas)

    return row


def build_oe9(data_root: str, groups: dict | None = None) -> pd.DataFrame:
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    files = discover_single_sensor_files(input_dir, "openearable") if groups is None else groups

    rows = []
    for group, path in files.items():
        oe = _load_oe(path, ALL_OE_COLS)
        sess = session_baseline_oe9(oe)
        lo, hi = float(oe["t"].min()), float(oe["t"].max())
        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S
            row = extract_oe9_features(oe, ws, we, sess)
            row["group"] = group
            row["window_start"] = float(ws)
            row["window_end"] = float(we)
            rows.append(row)
        del oe

    return pd.DataFrame(rows)


# ================================================================
# OE10 = OE9 + magnetometer (CELL 15)
# ================================================================

def mag_session_baseline(df: pd.DataFrame) -> dict:
    sess = {}
    for p in (1, 2, 3):
        M = np.stack([df[f"p{p}_mag_{a}"].values for a in "xyz"], axis=1)
        mag_mag = np.linalg.norm(M, axis=1)
        heading = np.unwrap(np.arctan2(M[:, 1], M[:, 0]))
        t = df["t"].values
        dt, dh = np.diff(t), np.diff(heading)
        good = np.isfinite(dt) & (dt > 0) & (dt < 1.0) & np.isfinite(dh)
        heading_vel = dh[good] / dt[good] if good.sum() > 10 else np.array([])
        dm = np.diff(mag_mag)
        good_m = np.isfinite(dt) & (dt > 0) & (dt < 1.0) & np.isfinite(dm)
        mag_vel = dm[good_m] / dt[good_m] if good_m.sum() > 10 else np.array([])
        sess[p] = {
            "mag_mean": safe_mean(mag_mag), "mag_std": safe_std(mag_mag), "mag_p75": safe_percentile(mag_mag, 75),
            "heading_vel_p75": safe_percentile(np.abs(heading_vel), 75), "heading_vel_p90": safe_percentile(np.abs(heading_vel), 90),
            "mag_vel_abs_p75": safe_percentile(np.abs(mag_vel), 75), "mag_vel_abs_p90": safe_percentile(np.abs(mag_vel), 90),
        }
    return sess


def extract_mag_features(df: pd.DataFrame, ws: float, we: float, sess: dict) -> dict:
    """Verbatim port of CELL 15's `extract_mag_features`."""
    dur = float(we - ws)
    if dur <= 0:
        return {}
    n = max(8, int(dur * RESAMPLE_HZ))
    grid = np.linspace(ws, we, n, endpoint=False)
    fs = RESAMPLE_HZ
    row: dict = {}

    M, mag_mag, mag_e, heading, heading_vel, mag_vel, horizontal_strength = {}, {}, {}, {}, {}, {}, {}
    for p in (1, 2, 3):
        M[p] = np.stack([sinterp(grid, df["t"].values, df[f"p{p}_mag_{a}"].values) for a in "xyz"], axis=1)
        mag_mag[p] = np.linalg.norm(M[p], axis=1)
        mag_e[p] = (mag_mag[p] - sess[p]["mag_mean"]) / (sess[p]["mag_std"] + 1e-9)
        heading[p] = np.unwrap(np.arctan2(M[p][:, 1], M[p][:, 0]))
        heading_vel[p] = np.r_[0, np.diff(heading[p])] * fs
        mag_vel[p] = np.r_[0, np.diff(mag_e[p])] * fs
        horizontal_strength[p] = np.sqrt(M[p][:, 0] ** 2 + M[p][:, 1] ** 2)

    def collect(name, vals):
        aggregate_values(row, f"mag_{name}", vals)

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

    collect("heading_std", [safe_std(heading[p]) for p in (1, 2, 3)])
    collect("heading_range", [safe_range(heading[p]) for p in (1, 2, 3)])
    collect("heading_maddiff", [mad_diff(heading[p]) for p in (1, 2, 3)])
    collect("heading_vel_abs_mean", [safe_mean(np.abs(heading_vel[p])) for p in (1, 2, 3)])
    collect("heading_vel_abs_std", [safe_std(np.abs(heading_vel[p])) for p in (1, 2, 3)])
    collect("heading_vel_abs_range", [safe_range(np.abs(heading_vel[p])) for p in (1, 2, 3)])
    collect("heading_turn_rate_p75", [event_count(np.abs(heading_vel[p]) > sess[p]["heading_vel_p75"]) / dur for p in (1, 2, 3)])
    collect("heading_turn_rate_p90", [event_count(np.abs(heading_vel[p]) > sess[p]["heading_vel_p90"]) / dur for p in (1, 2, 3)])
    collect("mag_burst_rate_p75", [event_count(np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p75"]) / dur for p in (1, 2, 3)])
    collect("mag_burst_rate_p90", [event_count(np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p90"]) / dur for p in (1, 2, 3)])

    collect("magnitude_entropy", [spectral_entropy(mag_e[p], fs) for p in (1, 2, 3)])
    collect("magnitude_low_band", [band_ratio(mag_e[p], fs, *MAG_LOW_BAND) for p in (1, 2, 3)])
    collect("magnitude_mid_band", [band_ratio(mag_e[p], fs, *MAG_MID_BAND) for p in (1, 2, 3)])
    collect("magnitude_high_band", [band_ratio(mag_e[p], fs, *MAG_HIGH_BAND) for p in (1, 2, 3)])
    collect("heading_vel_entropy", [spectral_entropy(heading_vel[p], fs) for p in (1, 2, 3)])
    collect("heading_vel_low_band", [band_ratio(heading_vel[p], fs, *MAG_LOW_BAND) for p in (1, 2, 3)])
    collect("heading_vel_mid_band", [band_ratio(heading_vel[p], fs, *MAG_MID_BAND) for p in (1, 2, 3)])
    collect("heading_vel_high_band", [band_ratio(heading_vel[p], fs, *MAG_HIGH_BAND) for p in (1, 2, 3)])

    heading_active = np.stack([(np.abs(heading_vel[p]) > sess[p]["heading_vel_p75"]).astype(int) for p in (1, 2, 3)], axis=0)
    mag_active = np.stack([(np.abs(mag_vel[p]) > sess[p]["mag_vel_abs_p75"]).astype(int) for p in (1, 2, 3)], axis=0)
    for name, count in [("heading_active_count", heading_active.sum(axis=0)), ("mag_active_count", mag_active.sum(axis=0))]:
        row[f"mag_{name}_mean"] = safe_mean(count)
        row[f"mag_{name}_std"] = safe_std(count)
        row[f"mag_{name}_exactly1_frac"] = float(np.mean(count == 1))
        row[f"mag_{name}_atleast2_frac"] = float(np.mean(count >= 2))
        row[f"mag_{name}_all3_frac"] = float(np.mean(count == 3))
        row[f"mag_{name}_switch_rate"] = event_count(np.r_[False, np.diff(count) != 0]) / dur

    pair_mag_corrs, pair_heading_vel_corrs, pair_heading_diffs_mean, pair_heading_diffs_std = [], [], [], []
    pair_mag_lagcorrs, pair_heading_vel_lagcorrs = [], []
    max_lag_steps = int(1.0 * fs)
    for a, b in PAIRS:
        pair_mag_corrs.append(corr_safe(mag_e[a], mag_e[b]))
        pair_heading_vel_corrs.append(corr_safe(heading_vel[a], heading_vel[b]))
        hdiff = np.abs(circular_diff(heading[a], heading[b]))
        pair_heading_diffs_mean.append(safe_mean(hdiff))
        pair_heading_diffs_std.append(safe_std(hdiff))
        pair_mag_lagcorrs.append(max_lag_corr(mag_e[a], mag_e[b], max_lag_steps))
        pair_heading_vel_lagcorrs.append(max_lag_corr(heading_vel[a], heading_vel[b], max_lag_steps))

    aggregate_values(row, "mag_pair_magnitude_corr", pair_mag_corrs)
    aggregate_values(row, "mag_pair_heading_vel_corr", pair_heading_vel_corrs)
    aggregate_values(row, "mag_pair_heading_diff_mean", pair_heading_diffs_mean)
    aggregate_values(row, "mag_pair_heading_diff_std", pair_heading_diffs_std)
    aggregate_values(row, "mag_pair_magnitude_lagcorr", pair_mag_lagcorrs)
    aggregate_values(row, "mag_pair_heading_vel_lagcorr", pair_heading_vel_lagcorrs)

    return row


def build_oe10(data_root: str, oe9: pd.DataFrame | None = None, groups: dict | None = None) -> pd.DataFrame:
    """Builds OE9 (if not already given) then left-merges the magnetometer
    feature table onto it on `["group","window_start","window_end"]`."""
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    if oe9 is None:
        oe9 = build_oe9(data_root, groups=groups)

    files = discover_single_sensor_files(input_dir, "openearable") if groups is None else groups

    mag_rows = []
    for group, gwin in oe9.groupby("group"):
        group = int(group)
        if group not in files:
            continue
        raw = _load_mag(files[group])
        sess = mag_session_baseline(raw)
        for _, w in gwin.iterrows():
            feats = extract_mag_features(raw, float(w["window_start"]), float(w["window_end"]), sess)
            feats["group"] = group
            feats["window_start"] = float(w["window_start"])
            feats["window_end"] = float(w["window_end"])
            mag_rows.append(feats)
        del raw

    mag_df = pd.DataFrame(mag_rows)
    return oe9.merge(mag_df, on=["group", "window_start", "window_end"], how="left")


def run_all(data_root: str, out_dir: str | None = None):
    """Builds and saves OE9 (`interaction_oe9_10s.csv`) and OE10
    (`interaction_oe10_10s.csv`) under `{data_root}/INTERACTION_OE9`
    and `{data_root}/INTERACTION_OE10` — same filenames the source
    notebook uses. Returns (oe9_df, oe10_df)."""
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Model-ready input not found:\n{input_dir}\n"
            "This is the output of the (not-yet-ported) global-cleaning stage "
            "— see docs/table_to_source_mapping.md."
        )

    files = discover_single_sensor_files(input_dir, "openearable")
    if not files:
        raise RuntimeError(f"No group_*_openearable_model_ready.csv found under {input_dir}")

    oe9_dir = out_dir or os.path.join(data_root, "INTERACTION_OE9")
    oe10_dir = out_dir or os.path.join(data_root, "INTERACTION_OE10")
    os.makedirs(oe9_dir, exist_ok=True)
    os.makedirs(oe10_dir, exist_ok=True)

    oe9 = build_oe9(data_root, groups=files)
    oe9_path = os.path.join(oe9_dir, "interaction_oe9_10s.csv")
    oe9.to_csv(oe9_path, index=False)
    print(f"Saved OE9: {oe9_path} | shape={oe9.shape}")

    oe10 = build_oe10(data_root, oe9=oe9, groups=files)
    oe10_path = os.path.join(oe10_dir, "interaction_oe10_10s.csv")
    oe10.to_csv(oe10_path, index=False)
    print(f"Saved OE10: {oe10_path} | shape={oe10.shape}")

    return oe9, oe10
