"""Feature engineering, ENG2 family — proximity/head/hand engineered
features built directly from `ALL_MODEL_READY_FILES_IDENTITY_FIXED` (the
confirmed output of the global-cleaning stage).

Ported from `GAR_Complete_Selfcontained.ipynb`, cells "CELL 1 (v2) —
INTERACTION" and "CELL 1 — RECOGNITION v2" (source notebook's own
naming). See notebooks_reference/GAR_Complete_Selfcontained_ANALYSIS.md
for the full porting notes — most importantly the **circularity
verdict**: these two builders are the one piece of Chapter-5-adjacent
feature-generation code in this project confirmed to be genuinely
re-derivable from raw `model_ready` files (they compute their own window
grid from raw timestamps and their own labels from raw `label_*`
columns), unlike `master_feature_generator_task1_task2_task3_
CORRECTED_V4.ipynb`, which was found to bootstrap its window grid from
an already-existing historical Task 1 output CSV.

IMPORTANT NAMING NOTE: "ENG2" is a different, smaller, earlier-vintage
feature family than "ENG3" (`interaction_eng3_features.csv`,
`eng3_recognition_3class_core_features.csv`, etc.), which is what
`src/models/task1.py`, `task2.py`, and the `task3_*.py` modules already
expect. Porting this module does not, by itself, reproduce the thesis's
headline Chapter 7/8 result tables — see docs/table_to_source_mapping.md
for the standing decision to port ENG2 now (proven non-circular) and
revisit ENG3's own non-circular source separately.

Also includes 3 "mutual-gaze" columns (`head_antiface`, `head_colinear`,
`head_facing_min`) — a magnetometer-heading-alignment proxy for "who is
facing whom," not true eye-gaze tracking. These are **not** part of the
thesis's documented feature set (absent from
docs/thesis_reproduction_targets.md); kept here as an explicitly-flagged
notebook-only addition, not silently presented as thesis-derived.
"""

from __future__ import annotations

import gc
import glob
import os
import re

import numpy as np
import pandas as pd

WINDOW_S = 5.0
STRIDE_S = 5.0
RESAMPLE_T = 64

OE_COLS = [f"p{p}_{s}_{a}" for p in (1, 2, 3) for s in ("acc", "gyro", "mag") for a in "xyz"]
XS_COLS = [f"p{p}_{s}_{a}" for p in (1, 2, 3) for s in ("acc", "gyr", "euler") for a in "xyz"]
POS_COLS = [f"Participant{p}_{a}" for p in (1, 2, 3) for a in "xyz"]

GROUP_TIERS = [
    "label_Participant1_Participant2", "label_Participant1_Participant3",
    "label_Participant2_Participant3", "label_Whole_Group",
]
LABEL_TIER = "label_Whole_Group"
RECOGNITION_KEEP = {"conversation", "co_building", "co_merging"}

CLASS_SETS = {
    "co_building": {"co_building_subpiece", "co_building_piece", "building_subpiece_together", "building_subpiece"},
    "co_merging": {"co_merging_subpiece", "merging_subpiece"},
    "conversation": {"task_operational_convo", "task_related_convo", "task_social_convo", "task_related_social_convo", "non_task_convo"},
}


def map_recognition_label(label) -> str:
    """Maps a raw (possibly `+`/`|`/`/`-joined) ELAN label to one of
    {"co_building", "co_merging", "conversation"}, or "" if none match."""
    if pd.isna(label) or str(label).strip() == "":
        return ""
    parts = [str(p).strip().lower() for p in re.split(r"[+|/]", str(label))]
    for cls, names in CLASS_SETS.items():
        if any(p in names for p in parts):
            return cls
    return ""


def discover_model_ready_groups(input_dir: str) -> dict:
    """Globs `group_*_model_ready.csv` under input_dir, groups by
    (group_id, sensor_name), and keeps only groups that have all three
    of openearable/xsens/optitrack. No group list is hardcoded — this is
    what makes the builders below genuinely raw-derived rather than
    tied to a fixed roster."""
    groups: dict = {}
    for path in glob.glob(os.path.join(input_dir, "group_*_model_ready.csv")):
        m = re.search(r"group_(\d+)_([a-z]+)_model_ready", os.path.basename(path))
        if m:
            groups.setdefault(int(m.group(1)), {})[m.group(2)] = path
    return {k: v for k, v in sorted(groups.items()) if all(s in v for s in ("openearable", "xsens", "optitrack"))}


def load_sensor_csv(path: str, timecol: str, cols: list[str]) -> pd.DataFrame:
    """Loads one model_ready CSV, coerces the time column to numeric
    `"t"`, sorts by time, and nulls out sentinel/garbage values
    (`abs() >= 1e6`) in every requested feature column."""
    df = pd.read_csv(path, low_memory=False)
    df["t"] = pd.to_numeric(df[timecol], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c].abs() < 1e6, np.nan)
    return df


def find_xsens_time_offset(oe: pd.DataFrame, xs: pd.DataFrame) -> float:
    """Brute-force cross-sensor time-offset search: samples
    `label_Whole_Group` on both streams over a shared 0.2s probe grid,
    then finds the offset in [0, 220)s (0.5s step) that maximizes label
    agreement between OpenEarable and Xsens."""
    probe = np.arange(max(oe["t"].min(), 5), oe["t"].max() - 5, 0.2)

    def sample(d, times):
        idx = np.clip(np.searchsorted(d["t"].values, times), 0, len(d) - 1)
        return d["label_Whole_Group"].fillna("NONE").astype(str).values[idx]

    reference = sample(oe, probe)
    best = (0, -1)
    for offset in np.arange(0, 220, 0.5):
        agreement = np.mean(reference == sample(xs, probe + offset))
        if agreement > best[1]:
            best = (offset, agreement)
    return best[0]


def spectral_summary(sig: np.ndarray, fs: float):
    """FFT-based spectral summary of a 1D signal: (dominant_freq,
    spectral_centroid, total_power), excluding the DC bin. Returns
    (0,0,0) if fewer than 8 finite samples. Callers that only want the
    2-value (dominant_freq, power) form used by the source notebook's
    recognition builder can just ignore the centroid."""
    sig = sig[np.isfinite(sig)]
    if len(sig) < 8:
        return 0.0, 0.0, 0.0
    sig = sig - sig.mean()
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    if len(power) < 2 or power[1:].sum() <= 0:
        return 0.0, 0.0, float(power.sum())
    dominant = freqs[1:][np.argmax(power[1:])]
    centroid = (freqs[1:] * power[1:]).sum() / power[1:].sum()
    return float(dominant), float(centroid), float(power[1:].sum())


def circular_mean_degrees(angles: np.ndarray) -> float:
    angles = angles[np.isfinite(angles)]
    if len(angles) == 0:
        return 0.0
    r = np.deg2rad(angles)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a[m]) < 3 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _head_gaze_block(o: pd.DataFrame, fo: float):
    """Shared head-orientation / "mutual gaze" computation used by both
    builders. Returns a dict of row fields plus the raw acc-magnitude
    signals (for move_coord) and per-participant motion-energy grid
    signal (recognition's tensor needs the latter; interaction doesn't).
    See module docstring for the gaze-proxy caveat."""
    pitch, headg, hdom, hpow = [], [], [], []
    acc_stds, gyr_stds = [], []
    acc_mag_by_participant: dict[int, np.ndarray] = {}

    for i in (1, 2, 3):
        acc = o[[f"p{i}_acc_{a}" for a in "xyz"]].values
        mag = o[[f"p{i}_mag_{a}" for a in "xyz"]].values
        gyr = o[[f"p{i}_gyro_{a}" for a in "xyz"]].values
        if len(acc):
            pitch.append(np.nanmean(np.degrees(np.arctan2(acc[:, 0], np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2)))))
            headg.append(circular_mean_degrees(np.degrees(np.arctan2(mag[:, 1], mag[:, 0]))))

            acc_mag = np.linalg.norm(acc, axis=1)
            gyr_mag = np.linalg.norm(gyr, axis=1)
            dominant, _, power = spectral_summary(gyr_mag, fo)
            hdom.append(dominant)
            hpow.append(power)

            acc_stds.append(np.nanstd(acc_mag))
            gyr_stds.append(np.nanstd(gyr_mag))
            acc_mag_by_participant[i] = acc_mag

    row = {"head_pitch_mean": np.nanmean(pitch) if pitch else 0.0}

    if len(headg) == 3:
        pair_cos = np.array([np.cos(np.deg2rad(headg[a] - headg[b])) for a, b in [(0, 1), (0, 2), (1, 2)]])
    else:
        pair_cos = np.zeros(3)
    row["head_facing"] = float(np.mean(pair_cos)) if len(headg) == 3 else 0.0
    row["head_antiface"] = float(np.mean(np.clip(-pair_cos, 0, 1)))
    row["head_colinear"] = float(np.mean(np.abs(pair_cos)))
    row["head_facing_min"] = float(np.min(pair_cos))

    row["head_freq_mean"] = np.mean(hdom) if hdom else 0.0
    row["head_freq_max"] = np.max(hdom) if hdom else 0.0
    row["head_power_mean"] = np.log1p(np.mean(hpow)) if hpow else 0.0

    aE = np.sort(acc_stds) if acc_stds else [0.0, 0.0, 0.0]
    gE = np.sort(gyr_stds) if gyr_stds else [0.0, 0.0, 0.0]
    row["accE_min"], row["accE_mid"], row["accE_max"] = aE[0], aE[1], aE[2]
    row["gyrE_min"], row["gyrE_mid"], row["gyrE_max"] = gE[0], gE[1], gE[2]

    if all(i in acc_mag_by_participant and len(acc_mag_by_participant[i]) > 0 for i in (1, 2, 3)):
        row["move_coord"] = np.mean([
            _safe_correlation(acc_mag_by_participant[1], acc_mag_by_participant[2]),
            _safe_correlation(acc_mag_by_participant[1], acc_mag_by_participant[3]),
            _safe_correlation(acc_mag_by_participant[2], acc_mag_by_participant[3]),
        ])
    else:
        row["move_coord"] = 0.0

    return row


# ================================================================
# Interaction builder (binary interaction / non_interaction)
# ================================================================

def _interaction_window_features(oe: pd.DataFrame, xs: pd.DataFrame, ot: pd.DataFrame, ws: float, we: float):
    o = oe.iloc[np.searchsorted(oe["t"].values, ws):np.searchsorted(oe["t"].values, we)]
    x = xs.iloc[np.searchsorted(xs["t"].values, ws):np.searchsorted(xs["t"].values, we)]
    p = ot.iloc[np.searchsorted(ot["t"].values, ws):np.searchsorted(ot["t"].values, we)]

    fo = max(len(o) / WINDOW_S, 1)
    fx = max(len(x) / WINDOW_S, 1)
    grid = np.linspace(ws, we, RESAMPLE_T)
    row: dict = {}

    # A. Optitrack position & proximity
    P = {i: p[[f"Participant{i}_{a}" for a in "xyz"]].values for i in (1, 2, 3)}
    pair_dist = {(a, b): np.linalg.norm(P[a] - P[b], axis=1) for a, b in [(1, 2), (1, 3), (2, 3)]}
    D = np.stack(list(pair_dist.values()), 1)
    Ds = np.sort(D, 1) if len(p) > 0 else np.zeros((1, 3))

    with np.errstate(all="ignore"):
        row["dist_close_mean"] = np.nanmean(Ds[:, 0])
        row["dist_close_min"] = np.nanmin(Ds[:, 0])
        row["dist_mid_mean"] = np.nanmean(Ds[:, 1])
        row["dist_far_mean"] = np.nanmean(Ds[:, 2])
        row["dist_disp_mean"] = np.nanmean(D)
        row["dist_disp_std"] = np.nanstd(D)

    centroid = (P[1] + P[2] + P[3]) / 3
    row["centroid_speed"] = np.nanmean(np.linalg.norm(np.gradient(centroid, axis=0), axis=1)) * (len(p) / WINDOW_S) if len(p) > 1 else 0.0

    # B. Per-participant speeds
    if len(p) > 1:
        fps_p = len(p) / WINDOW_S
        speed = {i: np.linalg.norm(np.gradient(P[i], axis=0), axis=1) * fps_p for i in (1, 2, 3)}
        speed_sorted = np.sort([np.nanmean(speed[i]) for i in (1, 2, 3)])
    else:
        speed_sorted = [0.0, 0.0, 0.0]
    row["speed_min"], row["speed_mid"], row["speed_max"] = speed_sorted[0], speed_sorted[1], speed_sorted[2]

    # C. Head orientation / motion energy / mutual-gaze proxy
    row.update(_head_gaze_block(o, fo))

    # D. Xsens hand kinematics
    xdom, xpow, ovar, hand_mag_grid = [], [], [], []
    for i in (1, 2, 3):
        acc = x[[f"p{i}_acc_{a}" for a in "xyz"]].values
        eul = x[[f"p{i}_euler_{a}" for a in "xyz"]].values
        if len(acc):
            mag = np.linalg.norm(acc, axis=1)
            dominant, _, power = spectral_summary(mag, fx)
            xdom.append(dominant)
            xpow.append(power)
            ovar.append(np.nanmean(np.nanstd(eul, axis=0)))
            hand_mag_grid.append(np.interp(grid, x["t"].values, mag))

    row["hand_freq_mean"] = np.mean(xdom) if xdom else 0.0
    row["hand_freq_max"] = np.max(xdom) if xdom else 0.0
    row["hand_power_mean"] = np.log1p(np.mean(xpow)) if xpow else 0.0
    row["hand_orient_var"] = np.mean(ovar) if ovar else 0.0
    if len(hand_mag_grid) == 3:
        row["hand_coord"] = np.mean([
            _safe_correlation(hand_mag_grid[0], hand_mag_grid[1]),
            _safe_correlation(hand_mag_grid[0], hand_mag_grid[2]),
            _safe_correlation(hand_mag_grid[1], hand_mag_grid[2]),
        ])
    else:
        row["hand_coord"] = 0.0

    # 3-channel sequence tensor (pairwise distances) for sequence models
    if len(p) > 1:
        pt = p["t"].values
        tensor = np.stack([np.interp(grid, pt, Ds[:, k]) for k in range(3)], 1).astype(np.float32)
    else:
        tensor = np.zeros((RESAMPLE_T, 3), np.float32)

    row = {k: (0.0 if (v is None or not np.isfinite(v)) else v) for k, v in row.items()}
    return row, tensor


def build_interaction_features(input_dir: str, groups: dict | None = None):
    """Cell 4. Fixed 5s/5s sliding-window binary interaction/non_interaction
    features, derived entirely from raw model_ready CSVs. Returns
    (features_df, tensor_array, y_array, group_array)."""
    if groups is None:
        groups = discover_model_ready_groups(input_dir)
    print("Groups:", list(groups))

    feats_, tens, Y, G = [], [], [], []
    for g, paths in groups.items():
        oe = load_sensor_csv(paths["openearable"], "video_time_s", OE_COLS)
        xs = load_sensor_csv(paths["xsens"], "time_s", XS_COLS)
        ot = load_sensor_csv(paths["optitrack"], "video_time_s", POS_COLS)
        xs["t"] -= find_xsens_time_offset(oe, xs)

        lo = max(oe["t"].min(), xs["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), xs["t"].max(), ot["t"].max())
        rt = oe["t"].values

        interacting = np.zeros(len(oe), dtype=bool)
        for col in GROUP_TIERS:
            if col in oe.columns:
                interacting |= oe[col].notna().values & (oe[col].astype(str).str.strip() != "").values

        n0 = len(Y)
        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S
            m = (rt >= ws) & (rt < we)
            if m.sum() == 0:
                continue
            label = "interaction" if interacting[m].mean() >= 0.5 else "non_interaction"
            row, tensor = _interaction_window_features(oe, xs, ot, ws, we)
            row["group"] = g
            feats_.append(row); tens.append(tensor); Y.append(label); G.append(g)
        print(f"  group {g}: {len(Y) - n0} windows")
        del oe, xs, ot
        gc.collect()

    return pd.DataFrame(feats_).assign(label=Y), np.stack(tens), np.array(Y), np.array(G)


# ================================================================
# Recognition builder (co_building / co_merging / conversation)
# ================================================================

def _recognition_window_features(oe, xs, ot, ws, we, home):
    o = oe.iloc[np.searchsorted(oe["t"].values, ws):np.searchsorted(oe["t"].values, we)]
    x = xs.iloc[np.searchsorted(xs["t"].values, ws):np.searchsorted(xs["t"].values, we)]
    p = ot.iloc[np.searchsorted(ot["t"].values, ws):np.searchsorted(ot["t"].values, we)]

    fo = max(len(o) / WINDOW_S, 1)
    fx = max(len(x) / WINDOW_S, 1)
    grid = np.linspace(ws, we, RESAMPLE_T)
    row: dict = {}

    # 1. Proximity
    P = {i: p[[f"Participant{i}_{a}" for a in "xyz"]].values for i in (1, 2, 3)}
    pair_dist = {(a, b): np.linalg.norm(P[a] - P[b], axis=1) for a, b in [(1, 2), (1, 3), (2, 3)]}
    D = np.stack(list(pair_dist.values()), 1)
    Ds = np.sort(D, 1) if len(p) > 0 else np.zeros((1, 3))
    row["dist_close_mean"] = np.nanmean(Ds[:, 0])
    row["dist_mid_mean"] = np.nanmean(Ds[:, 1])
    row["dist_far_mean"] = np.nanmean(Ds[:, 2])
    row["dist_disp_std"] = np.nanstd(D)

    centroid = (P[1] + P[2] + P[3]) / 3
    row["centroid_speed"] = np.nanmean(np.linalg.norm(np.gradient(centroid, axis=0), axis=1)) * (len(p) / WINDOW_S) if len(p) > 1 else 0.0

    # 2. Speeds
    if len(p) > 1:
        fps_p = len(p) / WINDOW_S
        speed = {i: np.linalg.norm(np.gradient(P[i], axis=0), axis=1) * fps_p for i in (1, 2, 3)}
        speed_sorted = np.sort([np.nanmean(speed[i]) for i in (1, 2, 3)])
    else:
        speed_sorted = [0.0, 0.0, 0.0]
    row["speed_min"], row["speed_mid"], row["speed_max"] = speed_sorted[0], speed_sorted[1], speed_sorted[2]

    # 3. Location / "home" (session-median position) cues
    far_safe = row["dist_far_mean"] if np.isfinite(row["dist_far_mean"]) else 0.0
    from_home = []
    for i in (1, 2, 3):
        if len(P[i]) > 0 and np.isfinite(home[i]).all():
            from_home.append(np.nanmean(np.linalg.norm(P[i] - home[i], axis=1)))
    if from_home:
        row["fromhome_mean"] = float(np.mean(from_home))
        row["fromhome_min"] = float(np.min(from_home))
        row["converge_ratio"] = float(np.min(from_home) / (far_safe + 0.5))
    else:
        row["fromhome_mean"] = row["fromhome_min"] = row["converge_ratio"] = 0.0

    center_dists = []
    for i in (1, 2, 3):
        if len(P[i]) > 0:
            center_dists.append(np.nanmean(np.sqrt(P[i][:, 0] ** 2 + P[i][:, 2] ** 2)))
    row["center_dist_centroid"] = np.nanmean(np.sqrt(centroid[:, 0] ** 2 + centroid[:, 2] ** 2)) if len(p) > 0 else 0.0
    row["center_dist_min"] = float(np.min(center_dists)) if center_dists else 0.0

    # 4. Head / mutual-gaze proxy + motion energy (also collect a per-window grid signal for the tensor)
    head_row = _head_gaze_block(o, fo)
    row.update(head_row)
    head_energy_grid = []
    for i in (1, 2, 3):
        gyr = o[[f"p{i}_gyro_{a}" for a in "xyz"]].values
        if len(gyr):
            head_energy_grid.append(np.interp(grid, o["t"].values, np.linalg.norm(gyr, axis=1)))

    # 5. Xsens hand kinematics
    xdom, xpow, ovar, hand_grid = [], [], [], []
    for i in (1, 2, 3):
        acc = x[[f"p{i}_acc_{a}" for a in "xyz"]].values
        eul = x[[f"p{i}_euler_{a}" for a in "xyz"]].values
        if len(acc) > 0:
            mag = np.linalg.norm(acc, axis=1)
            dominant, _, power = spectral_summary(mag, fx)
            xdom.append(dominant)
            xpow.append(power)
            ovar.append(np.nanmean(np.nanstd(eul, axis=0)))
            hand_grid.append(np.interp(grid, x["t"].values, mag))

    row["hand_freq_mean"] = np.mean(xdom) if xdom else 0.0
    row["hand_freq_max"] = np.max(xdom) if xdom else 0.0
    row["hand_power_mean"] = np.log1p(np.mean(xpow)) if xpow else 0.0
    row["hand_orient_var"] = np.mean(ovar) if ovar else 0.0
    if len(hand_grid) == 3:
        row["hand_coord"] = np.mean([
            _safe_correlation(hand_grid[0], hand_grid[1]),
            _safe_correlation(hand_grid[0], hand_grid[2]),
            _safe_correlation(hand_grid[1], hand_grid[2]),
        ])
    else:
        row["hand_coord"] = 0.0

    # 5-channel sequence tensor: 3 pairwise-distance channels + mean head-energy + mean hand-energy
    dist_channels = [np.interp(grid, p["t"].values, Ds[:, k]) if len(p) > 1 else np.zeros(RESAMPLE_T) for k in range(3)]
    head_channel = np.mean(head_energy_grid, axis=0) if len(head_energy_grid) == 3 else np.zeros(RESAMPLE_T)
    hand_channel = np.mean(hand_grid, axis=0) if len(hand_grid) == 3 else np.zeros(RESAMPLE_T)
    tensor = np.stack(dist_channels + [head_channel, hand_channel], 1).astype(np.float32)

    row = {k: (0.0 if (v is None or not np.isfinite(v)) else v) for k, v in row.items()}
    return row, tensor


def build_recognition_features(input_dir: str, mode: str, groups: dict | None = None):
    """Cell 6. mode="fixed": uniform 5s/5s sliding-window grid, same as
    the interaction builder. mode="labeled": windows are the raw
    contiguous same-mapped-class segments instead (run-length encoded),
    kept only if >= 0.5s. Both restrict to windows whose majority-vote
    label is in RECOGNITION_KEEP. Returns (features_df, tensor_array,
    y_array, group_array)."""
    if mode not in ("fixed", "labeled"):
        raise ValueError(f"mode must be 'fixed' or 'labeled', got {mode!r}")

    if groups is None:
        groups = discover_model_ready_groups(input_dir)
    print("Groups:", list(groups))

    feats_, tens, Y, G = [], [], [], []
    for g, paths in groups.items():
        oe = load_sensor_csv(paths["openearable"], "video_time_s", OE_COLS)
        xs = load_sensor_csv(paths["xsens"], "time_s", XS_COLS)
        ot = load_sensor_csv(paths["optitrack"], "video_time_s", POS_COLS)
        xs["t"] -= find_xsens_time_offset(oe, xs)

        lo = max(oe["t"].min(), xs["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), xs["t"].max(), ot["t"].max())
        rt = oe["t"].values

        P_full = {i: ot[[f"Participant{i}_{a}" for a in "xyz"]].values for i in (1, 2, 3)}
        home = {i: np.nanmedian(P_full[i], axis=0) for i in (1, 2, 3)}

        unique_labels = {u: map_recognition_label(u) for u in oe[LABEL_TIER].dropna().astype(str).unique()}
        mapped = oe[LABEL_TIER].astype(str).map(unique_labels).fillna("").values
        valid = np.isin(mapped, list(RECOGNITION_KEEP))

        if mode == "fixed":
            windows = [(w, w + WINDOW_S) for w in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S)]
        else:
            codes = pd.Series(mapped).astype("category").cat.codes.values
            change_points = np.where(np.diff(codes) != 0)[0] + 1
            bounds = [0] + list(change_points) + [len(oe)]
            windows = []
            for i in range(len(bounds) - 1):
                st, et = rt[bounds[i]], rt[min(bounds[i + 1], len(rt) - 1)]
                if et - st >= 0.5 and st >= lo and et <= hi:
                    windows.append((st, et))

        n0 = len(Y)
        for ws, we in windows:
            m = (rt >= ws) & (rt < we) & valid
            if m.sum() == 0:
                continue
            label = pd.Series(mapped[m]).mode().iloc[0]
            row, tensor = _recognition_window_features(oe, xs, ot, ws, we, home)
            row["group"] = g
            feats_.append(row); tens.append(tensor); Y.append(label); G.append(g)
        print(f"  group {g} [{mode}]: {len(Y) - n0} windows")
        del oe, xs, ot
        gc.collect()

    return pd.DataFrame(feats_).assign(label=Y), np.stack(tens), np.array(Y), np.array(G)


def run_all(data_root: str, out_dir_interaction: str | None = None, out_dir_recognition: str | None = None):
    """Builds and saves both feature families (interaction, and
    recognition in both 'fixed' and 'labeled' modes) from
    `{data_root}/ALL_MODEL_READY_FILES_IDENTITY_FIXED`. Mirrors the
    source notebook's own output layout (INTERACTION_ENG2/,
    RECOGNITION_ENG2/) by default."""
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Model-ready input not found:\n{input_dir}\n"
            "This is the output of the (not-yet-ported) global-cleaning stage "
            "— see docs/table_to_source_mapping.md."
        )

    out_dir_interaction = out_dir_interaction or os.path.join(data_root, "INTERACTION_ENG2")
    out_dir_recognition = out_dir_recognition or os.path.join(data_root, "RECOGNITION_ENG2")
    os.makedirs(out_dir_interaction, exist_ok=True)
    os.makedirs(out_dir_recognition, exist_ok=True)

    groups = discover_model_ready_groups(input_dir)
    if not groups:
        raise RuntimeError(f"No group has all 3 sensors' model_ready CSVs present under {input_dir}")

    F, X, Y, G = build_interaction_features(input_dir, groups=groups)
    F.to_csv(os.path.join(out_dir_interaction, "interaction_eng_features.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir_interaction, "interaction_eng_tensors.npz"), X=X, y=Y, groups=G)
    print(f"\ninteraction: {X.shape} | {pd.Series(Y).value_counts().to_dict()}")

    results = {"interaction": (F, X, Y, G)}
    for mode in ("fixed", "labeled"):
        F, X, Y, G = build_recognition_features(input_dir, mode, groups=groups)
        F.to_csv(os.path.join(out_dir_recognition, f"recognition_{mode}_features.csv"), index=False)
        np.savez_compressed(os.path.join(out_dir_recognition, f"recognition_{mode}_tensors.npz"), X=X, y=Y, groups=G)
        print(f"recognition[{mode}]: {X.shape} | {pd.Series(Y).value_counts().to_dict()}")
        results[f"recognition_{mode}"] = (F, X, Y, G)

    return results
