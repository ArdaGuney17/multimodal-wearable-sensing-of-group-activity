"""Feature engineering, ENG3 family — the 3-class recognition-label table
(`eng3_recognition_3class_core_features.csv`) that Table 7.9's feature
builders join their windows against (Ch.7 §7.5.2).

Ported from `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`
(see notebooks_reference/master_feature_generator_task1_task2_task3_
CORRECTED_V4_CODE_ONLY.py and its _ANALYSIS.md):
  - CELL 11 (lines 2087-2552) — 5s/5s sliding-window grid built straight
    from raw `group_{g}_{sensor}_model_ready.csv` files (window range from
    OpenEarable + OptiTrack only, Xsens does not shrink it), binary
    interaction/non_interaction label (majority vote of whether any of
    the 4 `label_*` annotation-tier columns is non-empty), and the legacy
    17-feature `old_eng_features_and_tensor` proximity/speed/motion block.
    This CELL 11 window-grid + binary-label part is confirmed **non-
    circular** — same verdict as `eng2_features.py`'s builders, and
    unrelated to the still-circular ENG3 bootstrap flagged elsewhere in
    docs/table_to_source_mapping.md for the *other* ENG3 outputs (Task
    1/2/3's `interaction_eng3_features.csv` consumption). This module
    only reproduces CELL 11's window grid + the recognition-label chain
    (CELLs 12-14), not the whole ENG3 family.
  - CELL 12 (lines 2553-2870) — per-window dominant raw annotation label
    (majority vote across the 4 label tiers, restricted to windows
    already labelled "interaction").
  - CELL 13 (lines 2871-3291) — `map_to_5class`: priority-ordered keyword
    mapping of the dominant raw label into {co_building, co_merging,
    co_inspection, conversation, object_transport, ignore}. Ported
    verbatim, including the `if`-chain order (first match wins) and the
    embedded `normalize_label`/`is_technical_or_sync` typo/exclusion
    logic CELL 13 uses internally (its own copy, not CELL 12's — CELL 12
    computes a separate, unused-downstream normalized-label audit column
    with slightly different typo fixes; only CELL 13's copy actually
    feeds the final classification, so that's the one kept here).
  - CELL 14 (lines 3292-3328) — filters the 5-class table down to the 3
    core classes {co_building, co_merging, conversation}.

UPDATE (2026-09-05 validation session): the deliberate scope cut mentioned
in earlier revisions of this docstring (skipping Xsens hand-kinematics
features and the full `interaction_eng3_features.csv` window/label grid,
keeping only the derived 3-class recognition table) has been undone.
Comparing against the real official files
(`data/external/thesis_data/INTERACTION_ENG3/interaction_eng3_features.csv`,
28 columns, ALL windows; `eng3_recognition_3class_core_features.csv`, 36
columns) showed both are needed: `interaction_eng3_features.csv` is
CELL 8's own direct output (17 `old_eng_features_and_tensor` columns + 7
Xsens `xsens_hand_features` columns + group/window_start/window_end/
label, for every window including non_interaction ones), and
`eng3_recognition_3class_core_features.csv` is that same 28-column table
restricted to interaction windows, left-merged with 6 more label-
inventory audit columns from CELL 9 Part B (`dominant_raw_label`,
`dominant_normalized_label`, `dominant_fraction_in_window`,
`all_raw_labels_in_window`, `source_tiers_in_window`,
`raw_label_counts`), then mapped/filtered via CELLs 10-11
(`recognition_label`, `mapping_reason`). Both are now built here:
`build_eng3_grid` (CELL 8) and `build_recognition_table` (CELL 8's grid
+ CELLs 9-11's label chain, fused into one per-group pass instead of the
notebook's 4 separate save/reload round-trips through intermediate CSVs).

One genuine, confirmed-harmless discrepancy kept faithfully: CELL 9's
own `normalize_label` (used only to compute the `dominant_normalized_label`
*audit* column) is missing the `"handiver"->"handover"` typo fix that
CELL 10/13's copy (used for the actual `map_to_5class` classification)
has — the notebook itself has two non-identical `normalize_label`
copies. Kept as two separate functions here (`normalize_label_audit` vs
`normalize_label`) rather than silently unifying them. Group 1's real
data never hits a `"handiver"` label, so this has zero observable effect
on Group 1's validation, but the distinction is preserved for
correctness/fidelity beyond Group 1.

Group discovery still requires all 3 sensor files to be present per
group (via `eng2_features.discover_model_ready_groups`), matching CELL
8's own `discover()` filter byte-for-byte.
"""

from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.features.eng2_features import discover_model_ready_groups, find_xsens_time_offset, load_sensor_csv
from src.features.oe_signal_common import sinterp

WINDOW_S = 5.0
STRIDE_S = 5.0
RESAMPLE_T = 64

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

CORE_CLASSES = ["co_building", "co_merging", "conversation"]


# ================================================================
# CELL 11 — legacy 17-feature proximity/speed/motion block
# ================================================================

def old_eng_features_and_tensor(oe: pd.DataFrame, ot: pd.DataFrame, ws: float, we: float):
    """Verbatim port of CELL 11's `old_eng_features_and_tensor`. Returns
    (row, tensor); callers that don't need the 7-channel sequence tensor
    (this module doesn't) can discard it. This is the source of the 10
    `OPTITRACK_FEATURES` names `task_oe9_recognition.py` pulls out by
    hand: dist_close_mean/close_min/mid_mean/far_mean/disp_mean/disp_std,
    speed_min/mid/max, centroid_speed."""
    grid = np.linspace(ws, we, RESAMPLE_T)
    fs = (RESAMPLE_T - 1) / (we - ws)

    P = {p: np.stack([sinterp(grid, ot["t"].values, ot[f"Participant{p}_{a}"].values) for a in "xyz"], axis=1) for p in (1, 2, 3)}
    A = {p: np.stack([sinterp(grid, oe["t"].values, oe[f"p{p}_acc_{a}"].values) for a in "xyz"], axis=1) for p in (1, 2, 3)}
    Gy = {p: np.stack([sinterp(grid, oe["t"].values, oe[f"p{p}_gyro_{a}"].values) for a in "xyz"], axis=1) for p in (1, 2, 3)}

    d = {(a, b): np.linalg.norm(P[a] - P[b], axis=1) for a, b in [(1, 2), (1, 3), (2, 3)]}
    D = np.stack(list(d.values()), axis=1)
    Dsort = np.sort(D, axis=1)

    spd = {p: np.linalg.norm(np.gradient(P[p], axis=0), axis=1) * fs for p in (1, 2, 3)}
    cen = (P[1] + P[2] + P[3]) / 3
    cen_spd = np.linalg.norm(np.gradient(cen, axis=0), axis=1) * fs

    amag = {p: np.linalg.norm(A[p], axis=1) for p in (1, 2, 3)}
    gmag = {p: np.linalg.norm(Gy[p], axis=1) for p in (1, 2, 3)}

    aE = np.sort([np.nanstd(amag[p]) for p in (1, 2, 3)])
    gE = np.sort([np.nanstd(gmag[p]) for p in (1, 2, 3)])
    spdm = np.sort([np.nanmean(spd[p]) for p in (1, 2, 3)])

    def corr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        if np.nanstd(x[m]) < 1e-6 or np.nanstd(y[m]) < 1e-6:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0, 1])

    coord = np.nanmean([corr(amag[a], amag[b]) for a, b in [(1, 2), (1, 3), (2, 3)]])

    row = {
        "dist_close_mean": np.nanmean(Dsort[:, 0]),
        "dist_close_min": np.nanmin(Dsort[:, 0]),
        "dist_mid_mean": np.nanmean(Dsort[:, 1]),
        "dist_far_mean": np.nanmean(Dsort[:, 2]),
        "dist_disp_mean": np.nanmean(D),
        "dist_disp_std": np.nanstd(D),
        "speed_min": spdm[0],
        "speed_mid": spdm[1],
        "speed_max": spdm[2],
        "centroid_speed": np.nanmean(cen_spd),
        "accE_min": aE[0],
        "accE_mid": aE[1],
        "accE_max": aE[2],
        "gyrE_min": gE[0],
        "gyrE_mid": gE[1],
        "gyrE_max": gE[2],
        "move_coord": coord,
    }

    accmag_mean = np.nanmean(np.stack([amag[p] for p in (1, 2, 3)], axis=0), axis=0)
    accmag_max = np.nanmax(np.stack([amag[p] for p in (1, 2, 3)], axis=0), axis=0)
    gyromag_mean = np.nanmean(np.stack([gmag[p] for p in (1, 2, 3)], axis=0), axis=0)
    tensor = np.stack([Dsort[:, 0], Dsort[:, 1], Dsort[:, 2], cen_spd, accmag_mean, accmag_max, gyromag_mean], axis=1).astype(np.float32)

    return row, tensor


# ================================================================
# CELL 8 (tail) — Xsens hand-kinematics features
# ================================================================

def _spec(sig: np.ndarray, fs: float):
    """Verbatim port of CELL 8's `spec` — FFT dominant-frequency + total
    power of a 1D signal, excluding the DC bin."""
    sig = np.asarray(sig, dtype=float)
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


def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.mean(x))


def _safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(np.max(x))


def xsens_hand_features(xs: pd.DataFrame, ws: float, we: float) -> dict:
    """Verbatim port of CELL 8's `xsens_hand_features` — per-window hand
    kinematics from the Xsens wrist sensors (already video-time-aligned
    by the caller via `eng2_features.find_xsens_time_offset`): dominant
    hand-motion frequency, log-power, orientation variance, 3-way
    hand-magnitude coordination, and 2 bookkeeping columns
    (`xsens_n_samples`, `xsens_available`)."""
    x = xs.iloc[np.searchsorted(xs["t"].values, ws):np.searchsorted(xs["t"].values, we)]
    fx = max(len(x) / WINDOW_S, 1)
    grid = np.linspace(ws, we, RESAMPLE_T)

    xdom, xpow, ovar, hmag = [], [], [], []
    for i in (1, 2, 3):
        acc_cols = [f"p{i}_acc_{a}" for a in "xyz"]
        eul_cols = [f"p{i}_euler_{a}" for a in "xyz"]
        acc = x[acc_cols].values if all(c in x.columns for c in acc_cols) else np.empty((0, 3))
        eul = x[eul_cols].values if all(c in x.columns for c in eul_cols) else np.empty((0, 3))

        if len(acc) >= 2:
            mag = np.linalg.norm(acc, axis=1)
            dom, power = _spec(mag, fx)
            xdom.append(dom)
            xpow.append(power)
            if len(eul) >= 2:
                ovar.append(np.nanmean(np.nanstd(eul, axis=0)))
            valid = np.isfinite(mag) & np.isfinite(x["t"].values)
            if valid.sum() >= 2:
                hmag.append(np.interp(grid, x["t"].values[valid], mag[valid]))

    row = {
        "hand_freq_mean": _safe_mean(xdom),
        "hand_freq_max": _safe_max(xdom),
        "hand_power_mean": np.log1p(_safe_mean(xpow)) if np.isfinite(_safe_mean(xpow)) else np.nan,
        "hand_orient_var": _safe_mean(ovar),
    }

    if len(hmag) == 3:
        def cr(a, b):
            if np.nanstd(a) < 1e-6 or np.nanstd(b) < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        row["hand_coord"] = np.nanmean([cr(hmag[0], hmag[1]), cr(hmag[0], hmag[2]), cr(hmag[1], hmag[2])])
    else:
        row["hand_coord"] = np.nan

    row["xsens_n_samples"] = len(x)
    row["xsens_available"] = 1 if len(x) >= 8 else 0
    return row


# ================================================================
# CELL 8 — full ENG3 window/label grid (ALL windows, both labels)
# ================================================================

def build_eng3_grid(input_dir: str, groups: dict | None = None) -> pd.DataFrame:
    """Verbatim port of CELL 8's `build_eng3` main loop: 5s/5s window
    grid over every group, both `interaction`/`non_interaction` windows
    kept, 17 `old_eng_features_and_tensor` columns + 7 `xsens_hand_
    features` columns per row. Equivalent of the real
    `interaction_eng3_features.csv` (28 columns: 17+7+group+window_start+
    window_end+label)."""
    if groups is None:
        groups = discover_model_ready_groups(input_dir)

    rows = []
    for g, paths in groups.items():
        oe = load_sensor_csv(paths["openearable"], "video_time_s", ACC + GYR)
        ot = load_sensor_csv(paths["optitrack"], "video_time_s", POS)
        xs = load_sensor_csv(paths["xsens"], "time_s", XS_COLS)
        xs = xs.copy()
        xs["t"] = xs["t"] - find_xsens_time_offset(oe, xs)

        # Window range uses ONLY OpenEarable + OptiTrack (CELL 8's own comment).
        lo = max(oe["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), ot["t"].max())
        rt = oe["t"].values

        inter = np.zeros(len(oe), dtype=bool)
        for col in GROUP_TIERS:
            if col in oe.columns:
                inter |= oe[col].notna().values & (oe[col].astype(str).str.strip() != "").values

        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S
            m = (rt >= ws) & (rt < we)
            if m.sum() == 0:
                continue
            label = "interaction" if inter[m].mean() >= 0.5 else "non_interaction"
            old_row, _tensor = old_eng_features_and_tensor(oe, ot, ws, we)
            hand_row = xsens_hand_features(xs, ws, we)
            row = {**old_row, **hand_row, "group": g, "window_start": ws, "window_end": we, "label": label}
            rows.append(row)

        del oe, ot, xs

    return pd.DataFrame(rows)


# ================================================================
# CELL 9 (Part B) — dominant raw label + label-inventory audit columns
# per interaction window
# ================================================================

def clean_label_value(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def normalize_label_audit(s):
    """CELL 9's own (slightly different, audit-column-only) copy of
    `normalize_label` — missing the `"handiver"->"handover"` fix that
    CELL 10/13's copy (`normalize_label` below) has. Feeds only the
    `dominant_normalized_label` output column, never the classification
    itself. See module docstring."""
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.replace("mering", "merging")
    s = s.replace("synchornizaion", "synchronization")
    s = s.replace("syncornaziton", "synchronization")
    s = s.replace("erarble", "earable")
    return s


def window_label_inventory(oe: pd.DataFrame, ws: float, we: float, group_tiers: list[str]) -> dict | None:
    """Verbatim port of CELL 9 Part B's per-window inventory computation
    (the loop body that builds `window_label_rows`, restricted here to a
    single window instead of iterating `eng3["group"].unique()` — same
    per-window Counter logic, no cross-window state). Returns None if the
    window has no OE samples or no label values at all (CELL 9 skips
    those windows via `continue`), matching the notebook's own left-merge
    semantics (unmatched interaction windows keep NaN dominant_* fields)."""
    t = oe["t"].values
    m = (t >= ws) & (t < we)
    if m.sum() == 0:
        return None

    raw_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    for tier in group_tiers:
        if tier not in oe.columns:
            continue
        vals = oe.loc[m, tier].apply(clean_label_value).dropna()
        for raw_label in vals:
            raw_counter[raw_label] += 1
            tier_counter[tier] += 1

    if len(raw_counter) == 0:
        return None

    dominant_raw_label, dominant_count = raw_counter.most_common(1)[0]
    return {
        "dominant_raw_label": dominant_raw_label,
        "dominant_normalized_label": normalize_label_audit(dominant_raw_label),
        "dominant_fraction_in_window": dominant_count / max(int(m.sum()), 1),
        "all_raw_labels_in_window": " | ".join(sorted(raw_counter.keys())),
        "source_tiers_in_window": " | ".join(sorted(tier_counter.keys())),
        "raw_label_counts": dict(raw_counter),
    }


# ================================================================
# CELL 13 — normalize_label / is_technical_or_sync / map_to_5class
# ================================================================

def normalize_label(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.replace("mering", "merging")
    s = s.replace("handiver", "handover")
    s = s.replace("synchornizaion", "synchronization")
    s = s.replace("syncornaziton", "synchronization")
    s = s.replace("erarble", "earable")
    return s


def is_technical_or_sync(label) -> bool:
    lab = normalize_label(label)
    if lab is None:
        return False
    technical_keywords = ["sync", "synchronization", "earable", "calibration", "drop", "clap_synchronization"]
    return any(k in lab for k in technical_keywords)


def map_to_5class(label):
    """Verbatim port of CELL 13's `map_to_5class` — priority-ordered
    keyword mapping (first match wins): co_building > co_merging >
    co_inspection > object_transport > conversation > ignore."""
    lab = normalize_label(label)

    if lab is None:
        return "ignore", "missing_label"
    if is_technical_or_sync(lab):
        return "ignore", "technical_or_synchronization_label"
    if lab == "starting_individual_build":
        return "ignore", "starting_marker_not_activity_class"

    if ("co_building" in lab or "building_subpiece" in lab or "building_piece" in lab
            or "building" in lab or "assembling" in lab or "assembly" in lab):
        return "co_building", "building_or_assembly_label"

    if "co_merging" in lab or "merging" in lab or "merge" in lab:
        return "co_merging", "merging_label"

    if ("co_inspecting" in lab or "co_inspection" in lab or "inspecting" in lab or "inspection" in lab
            or "matching_pieces_to_target_image" in lab or "matching_pieces_with_target_image" in lab
            or "matching_piece_to_target_image" in lab or "matching_piece_with_target_image" in lab
            or ("matching" in lab and "image" in lab) or ("target_image" in lab and "matching" in lab)):
        return "co_inspection", "inspection_or_target_image_matching_label"

    if ("object_handover" in lab or "handover" in lab or "moving_pieces" in lab or "placing_subpiece" in lab
            or "placing_piece" in lab or "carrying_tray" in lab or "tray" in lab or "traveling_between_units" in lab
            or "approaching_to" in lab or "delivering_piece" in lab or "delivering_target_image" in lab
            or "presenting_piece" in lab or "presenting_target_image" in lab or "piece_presentation" in lab
            or "searching_for_piece" in lab):
        return "object_transport", "object_or_material_transport_label"

    if "convo" in lab or "conversation" in lab or "discussion" in lab or "talking" in lab:
        return "conversation", "conversation_label"

    return "ignore", "rare_or_unmapped_label"


# ================================================================
# Build: CELL 8 window grid + CELLs 9-11 label chain, one pass/group
# ================================================================

def build_recognition_table(input_dir: str, groups: dict | None = None) -> pd.DataFrame:
    """Builds the 5s/5s window grid (CELL 8, all 28 columns: 17
    `old_eng_features_and_tensor` + 7 `xsens_hand_features` +
    group/window_start/window_end/label), keeps interaction windows,
    computes each one's label-inventory audit columns (CELL 9 Part B),
    maps the dominant label to the 5-class taxonomy (CELL 10) and keeps
    only the 3 core classes (CELL 11) — all in one per-group pass instead
    of the notebook's 4 separate save/reload round-trips. Returns a
    36-column DataFrame matching the real
    `eng3_recognition_3class_core_features.csv` exactly (28 CELL-8
    columns + `dominant_raw_label`, `dominant_normalized_label`,
    `dominant_fraction_in_window`, `all_raw_labels_in_window`,
    `source_tiers_in_window`, `raw_label_counts`, `recognition_label`,
    `mapping_reason`)."""
    if groups is None:
        groups = discover_model_ready_groups(input_dir)

    rows = []
    for g, paths in groups.items():
        oe = load_sensor_csv(paths["openearable"], "video_time_s", ACC + GYR)
        ot = load_sensor_csv(paths["optitrack"], "video_time_s", POS)
        xs = load_sensor_csv(paths["xsens"], "time_s", XS_COLS)
        xs = xs.copy()
        xs["t"] = xs["t"] - find_xsens_time_offset(oe, xs)

        lo = max(oe["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), ot["t"].max())
        rt = oe["t"].values

        inter = np.zeros(len(oe), dtype=bool)
        for col in GROUP_TIERS:
            if col in oe.columns:
                inter |= oe[col].notna().values & (oe[col].astype(str).str.strip() != "").values

        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S
            m = (rt >= ws) & (rt < we)
            if m.sum() == 0:
                continue
            label = "interaction" if inter[m].mean() >= 0.5 else "non_interaction"
            if label != "interaction":
                continue  # CELL 10 restricts to interaction windows before mapping

            inventory = window_label_inventory(oe, ws, we, GROUP_TIERS)
            if inventory is None:
                continue  # unmatched window — CELL 10's left-merge would drop the mapping too
            recognition_label, mapping_reason = map_to_5class(inventory["dominant_raw_label"])
            if recognition_label not in CORE_CLASSES:
                continue  # CELL 11: keep only the 3 core classes

            old_row, _tensor = old_eng_features_and_tensor(oe, ot, ws, we)
            hand_row = xsens_hand_features(xs, ws, we)
            row = {
                **old_row, **hand_row,
                "group": g, "window_start": ws, "window_end": we, "label": label,
                **inventory,
                "recognition_label": recognition_label, "mapping_reason": mapping_reason,
            }
            rows.append(row)

        del oe, ot, xs

    return pd.DataFrame(rows)


def run_all(data_root: str, out_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds and saves both real ENG3 outputs under
    `{data_root}/INTERACTION_ENG3/`: `interaction_eng3_features.csv`
    (CELL 8's full window/label grid, all windows) and
    `eng3_recognition_3class_core_features.csv` (CELLs 8-11's 3-class
    recognition table) — same filenames the source notebook uses, so
    downstream modules (`task_oe9_recognition.py`) can read them exactly
    like the source notebook does. Returns (full_grid, recognition_core)."""
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Model-ready input not found:\n{input_dir}\n"
            "This is the output of the (not-yet-ported) global-cleaning stage "
            "— see docs/table_to_source_mapping.md."
        )

    out_dir = out_dir or os.path.join(data_root, "INTERACTION_ENG3")
    os.makedirs(out_dir, exist_ok=True)

    groups = discover_model_ready_groups(input_dir)
    if not groups:
        raise RuntimeError(f"No group has all 3 sensors' model_ready CSVs present under {input_dir}")

    full_grid = build_eng3_grid(input_dir, groups=groups)
    full_path = os.path.join(out_dir, "interaction_eng3_features.csv")
    full_grid.to_csv(full_path, index=False)
    print(f"Saved full ENG3 window/label grid: {full_path}")
    print("Shape:", full_grid.shape)

    recognition_core = build_recognition_table(input_dir, groups=groups)
    out_path = os.path.join(out_dir, "eng3_recognition_3class_core_features.csv")
    recognition_core.to_csv(out_path, index=False)

    print(f"Saved three-class recognition table: {out_path}")
    print("Shape:", recognition_core.shape)
    if len(recognition_core):
        print(recognition_core["recognition_label"].value_counts().to_string())

    return full_grid, recognition_core
