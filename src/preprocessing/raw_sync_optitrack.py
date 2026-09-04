"""Raw OptiTrack sensor sync + ELAN label transfer (Ch.4 Tables 4.1-4.4's
upstream stage, OptiTrack half — see `raw_sync_oe_xsens.py` for the
OpenEarable/Xsens half of the same stage).

Ported from `OPTI_TRACK_PROCESSING.ipynb` — see
notebooks_reference/OPTI_TRACK_PROCESSING_ANALYSIS.md for the full
per-group trace this module is built from (exact cell numbers, exact
algorithms quoted verbatim, exact per-group constants, exact output
filenames).

SCOPE (see docs/table_to_source_mapping.md's "Raw sensor sync & cleaning"
row): starts from `group_{g}/optitrack/optitrack_final/group_{g}_
optitrack_cleaned_combined_240hz.csv` — the source notebook's own raw
marker reconstruction pass (Hungarian-algorithm nearest-neighbor tracking
plus hand-curated per-group/per-take tracklet-stitching `CHAINS` dicts) is
explicitly OUT OF SCOPE, per an explicit decision documented in that doc
row: it is the single least-automatable step in the whole project (a
literal mapping of raw Motive "Unlabeled NNNN" marker-track IDs to
participant identities, unique per group and per take, not derivable from
a rule). The already-cleaned/combined file is treated as a fixed
intermediate input, the same way `global_cleaning.py` already treats the
`*_labeled*.csv` files it consumes.

All 9 groups (1,2,3,5,6,7,8,9,10) are covered — every group's final sync
shift in the source notebook's saved run was a **human-confirmed
constant** (a value read off a plot and hand-transcribed), not something
the candidate-peak detector selected unattended (see
OPTI_TRACK_PROCESSING_ANALYSIS.md section 1). This module preserves both:
the real, reusable candidate-detection algorithm
(`compute_optitrack_movement`/`auto_detect_optitrack_sync`, useful for
re-verification or re-deriving a shift for new data), and the per-group
constants actually used to produce every one of the 9 groups' real
`*_optitrack_labeled.csv` files (`GROUP_SYNC_CONFIG` below) — matching the
same "preserve the human decision as literal config" precedent already
used throughout this repo (`global_cleaning.py`'s `SELECTED_FILE_NAMES`/
`PARTICIPANT_POSITION_MAP`, `raw_sync_oe_xsens.py`'s per-group search
windows).

ANONYMIZATION: unlike `raw_sync_oe_xsens.py`, this stage never touches a
real-name file at all — its ELAN input is always the already-anonymized
`Group_N_individual_build_renamed.csv` (produced by
`raw_sync_oe_xsens.py`'s anonymization step / already present in the real
Drive data as a pre-existing file), and the raw OptiTrack Motive exports
contain only 3D marker positions, never participant-identifying text. See
OPTI_TRACK_PROCESSING_ANALYSIS.md section 4 (confirmed clean).

LABEL TEXT: this module transfers ELAN labels onto OptiTrack rows as
**raw free text**, exactly as the source notebook does — it does not
re-implement the source's own small per-group typo-correction dict
(`normalize_label`), since this repo already has a canonical, more
thorough typo/synonym normalizer for exactly this kind of text
(`src/preprocessing/labels.py`, ported from
`rq3_label_audit_and_normalization.ipynb`). Downstream consumers that need
cleaned label text should run this module's output through that module,
not expect it pre-cleaned here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group_4 excluded (camera failure) — canonical everywhere in this repo

TIERS = [
    "Participant1", "Participant2", "Participant3",
    "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3",
    "Whole_Group",
]

# Headerless ELAN export layout — identical positional schema to
# raw_sync_oe_xsens.RAW_ELAN_COLS (verbatim from OPTI_TRACK_PROCESSING.ipynb's
# own `load_elan_annotations`, which reads via header=None and indexes
# iloc[:,0]/[:,3]/[:,5]/[:,7]/[:,8] — the same tier/begin_s/end_s/dur_s/label
# positions).
RAW_ELAN_COLS = ["tier", "blank", "begin_hms", "begin_s", "end_hms", "end_s", "dur_hms", "dur_s", "label"]

TIME_COL = "time_s"
OUT_TIME_COL = "video_time_s"


# ================================================================
# Per-group sync configuration (verbatim constants from
# OPTI_TRACK_PROCESSING_ANALYSIS.md section 1's per-group table)
# ================================================================

@dataclass
class OptitrackSyncConfig:
    group: int
    method: str                          # "single" | "two_point"
    anchor_tier: str = "Whole_Group"      # which ELAN tier the sync label must belong to
    sync_time: float | None = None                 # single-point: OPTITRACK_SYNC_TIME / _MANUAL
    first_sync_time: float | None = None           # two-point / single-first-only: OPTITRACK_FIRST_SYNC_TIME
    last_sync_time: float | None = None            # two-point only: OPTITRACK_LAST_SYNC_TIME
    use_first_only: bool = False                    # group 5: only the first ELAN sync interval is usable


GROUP_SYNC_CONFIG: dict[int, OptitrackSyncConfig] = {
    # Group 10: single-point, human-confirmed OPTITRACK_SYNC_TIME_MANUAL=35.95.
    10: OptitrackSyncConfig(group=10, method="single", sync_time=35.95),

    # Group 9: two-point linear drift correction.
    9: OptitrackSyncConfig(group=9, method="two_point",
                            first_sync_time=23.141667, last_sync_time=3606.117833),

    # Group 8: two-point linear.
    8: OptitrackSyncConfig(group=8, method="two_point",
                            first_sync_time=111.125, last_sync_time=2312.608333),

    # Group 7: two-point linear.
    7: OptitrackSyncConfig(group=7, method="two_point",
                            first_sync_time=71.119, last_sync_time=2977.631667),

    # Group 6: two-point linear.
    6: OptitrackSyncConfig(group=6, method="two_point",
                            first_sync_time=32.108333, last_sync_time=1367.339833),

    # Group 5: single-point, first sync interval only (the final ELAN sync
    # occurs after the OptiTrack recording ends, so drift can't be checked).
    5: OptitrackSyncConfig(group=5, method="single", sync_time=113.5875, use_first_only=True),

    # Group 3: single-point (only one ELAN synchronization_move exists at all).
    3: OptitrackSyncConfig(group=3, method="single", sync_time=301.704167),

    # Group 2: single-point, anchored on Participant1's tier specifically
    # (not all 3 members performed the sync move).
    2: OptitrackSyncConfig(group=2, method="single", sync_time=199.883333, anchor_tier="Participant1"),

    # Group 1: single-point (only one ELAN synchronization_move available).
    1: OptitrackSyncConfig(group=1, method="single", sync_time=1829.856833),
}


# ================================================================
# ELAN read + sync-interval location
# ================================================================

def elan_renamed_path(data_root: str, group: int) -> str:
    """Path to the already-anonymized ELAN table this stage consumes
    (produced by raw_sync_oe_xsens.py's anonymization step / already
    present as real Drive data)."""
    return os.path.join(data_root, f"group_{group}", "elan", f"Group_{group}_individual_build_renamed.csv")


def read_renamed_elan(path: str) -> pd.DataFrame:
    """Headerless CSV, fixed 9-column layout — see RAW_ELAN_COLS."""
    df = pd.read_csv(path, header=None, names=RAW_ELAN_COLS, dtype={"blank": str})
    return df[["tier", "begin_s", "end_s", "label"]].copy()


def find_sync_rows(elan_df: pd.DataFrame, anchor_tier: str = "Whole_Group") -> pd.DataFrame:
    """All ELAN rows whose label text matches the sync-anchor regex
    ("sync|synchron", case-insensitive substring — verbatim from
    `find_elan_sync_interval`), preferring `anchor_tier` when any of its
    rows match, else falling back to every matching row regardless of
    tier. Sorted by begin_s."""
    mask = elan_df["label"].astype(str).str.lower().str.contains("sync|synchron", regex=True, na=False)
    candidates = elan_df[mask].copy()
    if candidates.empty:
        raise ValueError("no synchronization label found in ELAN file")

    tier_matches = candidates[candidates["tier"].astype(str).str.lower() == anchor_tier.lower()]
    chosen_pool = tier_matches if len(tier_matches) > 0 else candidates
    return chosen_pool.sort_values("begin_s").reset_index(drop=True)


def sync_row_mid(row: pd.Series) -> float:
    return (float(row["begin_s"]) + float(row["end_s"])) / 2.0


# ================================================================
# OptiTrack-side movement-score peak candidate detection
# (real, reusable algorithm — see module docstring re: which value is
# actually used for each group's shift)
# ================================================================

def compute_optitrack_movement(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Per-landmark 3D displacement magnitude, smoothed ~0.5s
    (rolling(120, center=True, min_periods=1) @ 240Hz), then combined
    across the 3 landmarks (tolerant=mean skipna, NaN if <2 available;
    strict=min skipna=False) and smoothed again ~1s
    (rolling(240, center=True, min_periods=1)). Verbatim from
    OPTI_TRACK_PROCESSING_ANALYSIS.md section 1."""
    out = df.copy()
    movement_cols = []
    for i in (1, 2, 3):
        cols = [f"landmark{i}_{ax}" for ax in "xyz"]
        if not all(c in out.columns for c in cols):
            continue
        d = out[cols].diff()
        mag = np.sqrt((d ** 2).sum(axis=1))
        out[f"landmark{i}_movement"] = mag
        smooth_col = f"landmark{i}_movement_smooth"
        out[smooth_col] = mag.rolling(120, center=True, min_periods=1).mean()
        movement_cols.append(smooth_col)

    if not movement_cols:
        raise ValueError("no landmark{1,2,3}_{x,y,z} columns found for movement computation")

    available_count = out[movement_cols].notna().sum(axis=1)
    tolerant = out[movement_cols].mean(axis=1, skipna=True)
    tolerant = tolerant.where(available_count >= 2, np.nan)
    strict = out[movement_cols].min(axis=1, skipna=False)

    out["sync_movement_score_tolerant"] = tolerant
    out["sync_movement_score_strict"] = strict
    out["sync_movement_score_tolerant_smooth"] = tolerant.rolling(240, center=True, min_periods=1).mean()
    out["sync_movement_score_strict_smooth"] = strict.rolling(240, center=True, min_periods=1).mean()
    return out


def auto_detect_optitrack_sync(df_movement: pd.DataFrame, time_col: str = TIME_COL,
                                take_col: str = "take", boundary_exclusion_s: float = 2.0,
                                bin_s: float = 2.0, top_n: int = 20) -> pd.DataFrame:
    """Top-N candidate peak times, ranked by the smoothed tolerant
    movement score, one candidate per `bin_s`-second bin, excluding a
    window around each take's start (avoids false peaks from
    recording start/stop transients). Verbatim binning/exclusion logic
    from OPTI_TRACK_PROCESSING_ANALYSIS.md section 1. This is the real
    candidate-generation algorithm; it is NOT what produces any of the 9
    groups' actual shift in this module's `run_all()` (see module
    docstring) — provided for re-verification / re-deriving a shift on
    new data."""
    cand = df_movement[df_movement["sync_movement_score_tolerant_smooth"].notna()].copy()
    if take_col in cand.columns:
        take_starts = cand.groupby(take_col)[time_col].min().to_numpy()
        for t in take_starts:
            cand = cand[~((cand[time_col] >= t - boundary_exclusion_s) & (cand[time_col] <= t + boundary_exclusion_s))]

    cand["_time_bin"] = (cand[time_col] // bin_s).astype(int)
    peaks = (
        cand.sort_values("sync_movement_score_tolerant_smooth", ascending=False)
            .groupby("_time_bin", as_index=False).head(1)
            .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
            .head(top_n)
            .drop(columns="_time_bin")
            .reset_index(drop=True)
    )
    return peaks


# ================================================================
# Shift computation + application
# ================================================================

def compute_single_point_shift(elan_df: pd.DataFrame, cfg: OptitrackSyncConfig) -> tuple[float, float]:
    """SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID (verbatim, line 13671
    of OPTI_TRACK_PROCESSING.ipynb). Returns (shift, elan_sync_mid)."""
    rows = find_sync_rows(elan_df, cfg.anchor_tier)
    chosen = rows.iloc[0] if cfg.use_first_only or cfg.method == "single" else rows.iloc[0]
    elan_mid = sync_row_mid(chosen)
    if cfg.sync_time is None:
        raise ValueError(f"group {cfg.group}: method='single' requires sync_time")
    return cfg.sync_time - elan_mid, elan_mid


def compute_two_point_alignment(opti_first: float, elan_first: float, opti_last: float, elan_last: float) -> tuple[float, float]:
    """Linear a*t+b fit mapping OptiTrack time -> ELAN/video time from two
    confirmed sync points. Verbatim from OPTI_TRACK_PROCESSING_ANALYSIS.md
    section 1."""
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b


def compute_two_point_shift(elan_df: pd.DataFrame, cfg: OptitrackSyncConfig) -> tuple[float, float]:
    """Returns (a, b) for video_time_s = a*time_s + b, using the earliest
    and latest ELAN sync-row midpoints paired with the two hand-confirmed
    OptiTrack-side sync instants (cfg.first_sync_time/last_sync_time)."""
    rows = find_sync_rows(elan_df, cfg.anchor_tier)
    if len(rows) < 2:
        raise ValueError(f"group {cfg.group}: method='two_point' needs >=2 ELAN sync rows, found {len(rows)}")
    elan_first = sync_row_mid(rows.iloc[0])
    elan_last = sync_row_mid(rows.iloc[-1])
    if cfg.first_sync_time is None or cfg.last_sync_time is None:
        raise ValueError(f"group {cfg.group}: method='two_point' requires first_sync_time and last_sync_time")
    return compute_two_point_alignment(cfg.first_sync_time, elan_first, cfg.last_sync_time, elan_last)


def apply_sync(optitrack_df: pd.DataFrame, elan_df: pd.DataFrame, cfg: OptitrackSyncConfig,
                time_col: str = TIME_COL) -> tuple[pd.DataFrame, dict]:
    """Adds `video_time_s` to optitrack_df per cfg.method. Returns
    (df_with_video_time, alignment_metadata)."""
    out = optitrack_df.copy()
    if cfg.method == "single":
        shift, elan_mid = compute_single_point_shift(elan_df, cfg)
        out[OUT_TIME_COL] = out[time_col] - shift
        meta = {"method": "single", "shift_s": shift, "elan_sync_mid_s": elan_mid}
    elif cfg.method == "two_point":
        a, b = compute_two_point_shift(elan_df, cfg)
        out[OUT_TIME_COL] = a * out[time_col] + b
        meta = {"method": "two_point", "alignment_a": a, "alignment_b": b}
    else:
        raise ValueError(f"unknown sync method {cfg.method!r}")
    return out, meta


# ================================================================
# Label transfer
# ================================================================

def fast_assign_labels(optitrack_df: pd.DataFrame, elan_df: pd.DataFrame,
                        time_col: str = OUT_TIME_COL) -> pd.DataFrame:
    """Transfers each ELAN tier's interval labels onto every OptiTrack row
    whose `time_col` falls inside that interval, one `label_{tier}` text
    column per tier. Uses a merge-asof (nearest preceding segment start,
    kept only if the row's time also falls before that segment's end) —
    a simplification of the source's own per-row interval-membership
    scan that does not reproduce the rare case of two overlapping-in-time
    annotation rows on the *same* tier (which the source concatenates
    with " + "); single-match is exact for the overwhelming majority of
    non-overlapping ELAN annotations."""
    out = optitrack_df.copy()
    for tier in TIERS:
        seg = (elan_df[elan_df["tier"] == tier][["begin_s", "end_s", "label"]]
               .dropna(subset=["begin_s"]).sort_values("begin_s").reset_index(drop=True))
        if seg.empty:
            out[f"label_{tier}"] = ""
            continue
        left = out[[time_col]].reset_index().sort_values(time_col)
        merged = pd.merge_asof(left, seg, left_on=time_col, right_on="begin_s", direction="backward")
        merged = merged.set_index("index").sort_index()
        label = merged["label"].where(merged[time_col] <= merged["end_s"], "")
        out[f"label_{tier}"] = label.fillna("").astype(str)
    return out


def labeling_summary(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Per-tier labeled-row-count/percentage/unique-label summary,
    mirroring `make_labeling_summary`."""
    n = len(labeled_df)
    rows = []
    for tier in TIERS:
        col = f"label_{tier}"
        if col not in labeled_df.columns:
            continue
        non_empty = labeled_df[col].astype(str).str.len() > 0
        rows.append({
            "tier": tier,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100) if n else 0.0,
            "n_unique_labels": int(labeled_df.loc[non_empty, col].nunique()),
        })
    return pd.DataFrame(rows)


# ================================================================
# Orchestration
# ================================================================

def combined_path(data_root: str, group: int) -> str:
    return os.path.join(data_root, f"group_{group}", "optitrack", "optitrack_final",
                         f"group_{group}_optitrack_cleaned_combined_240hz.csv")


def output_dir(data_root: str, group: int, out_dir: str | None = None) -> str:
    root = out_dir or data_root
    return os.path.join(root, f"group_{group}", "optitrack", "optitrack_labeled")


def process_group(data_root: str, group: int, out_dir: str | None = None,
                   optitrack_df: pd.DataFrame | None = None,
                   elan_df: pd.DataFrame | None = None) -> dict:
    """Runs sync + label-transfer for one group: read the already-cleaned/
    combined OptiTrack CSV (or use the caller-supplied `optitrack_df`,
    e.g. synthetic test data) and the already-anonymized ELAN table (or
    `elan_df`), compute this group's configured shift, attach video_time_s
    and per-tier labels, and write `group_{g}_optitrack_labeled.csv` +
    `group_{g}_optitrack_labeling_summary.csv`."""
    cfg = GROUP_SYNC_CONFIG[group]

    if optitrack_df is None:
        optitrack_df = pd.read_csv(combined_path(data_root, group))
    if elan_df is None:
        elan_df = read_renamed_elan(elan_renamed_path(data_root, group))

    shifted, meta = apply_sync(optitrack_df, elan_df, cfg)
    labeled = fast_assign_labels(shifted, elan_df)
    summary = labeling_summary(labeled)

    out_root = output_dir(data_root, group, out_dir)
    os.makedirs(out_root, exist_ok=True)
    labeled_path = os.path.join(out_root, f"group_{group}_optitrack_labeled.csv")
    summary_path = os.path.join(out_root, f"group_{group}_optitrack_labeling_summary.csv")
    labeled.to_csv(labeled_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {"labeled_path": labeled_path, "summary_path": summary_path, "alignment": meta}


def run_all(data_root: str, out_dir: str | None = None) -> dict:
    """Runs process_group() for every group in GROUPS. Groups that fail
    (e.g. missing input files) record an "error" key instead of raising,
    so one bad group doesn't abort the rest."""
    results = {}
    for group in GROUPS:
        try:
            results[group] = process_group(data_root, group, out_dir=out_dir)
        except Exception as e:
            results[group] = {"error": str(e)}
    return results
