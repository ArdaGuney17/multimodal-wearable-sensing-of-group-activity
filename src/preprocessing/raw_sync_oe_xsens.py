"""Raw OpenEarable + Xsens sensor sync, ELAN labeling, and peak-alignment
shift (Ch.4 Tables 4.1-4.4's upstream stage).

Ported from `FINAL_ARDA_THESIS.ipynb` — see
notebooks_reference/FINAL_ARDA_THESIS_ANALYSIS.md for the full per-group
trace this module is built from (exact cell numbers, exact algorithms
quoted verbatim, exact per-group constants, exact output filenames).

SCOPE (see docs/table_to_source_mapping.md's "Raw sensor sync & cleaning"
row): all 9 groups (1,2,3,5,6,7,8,9,10), OpenEarable (-> 50Hz wide grid) +
Xsens (-> ~30Hz wide grid) sensor merge, raw ELAN read + anonymization,
`individual_build` gap-fill, label attachment, and the peak-alignment sync
shift. The shift machinery (GROUP_SYNC_CONFIG + compute_peak_shift) covers
all 9 groups, but run_all()'s file-producing path only *writes* a shifted
variant for the (group, sensor) pairs `src/preprocessing/global_cleaning.
py`'s current SELECTED_FILE_NAMES actually selects a shifted/cleaned
variant for — as of 2026-09-04 that's groups 7, 8, 9, 10 (both sensors),
plus group 6's xsens artifact-cleaning-without-shift. Every other
(group, sensor) gets the plain unshifted `_labeled.csv` file
global_cleaning.py actually selects. Callers who want a shifted variant for
a group not wired into run_all() can call compute_peak_shift()/
process_group() directly — the search-window parameters for all 9 groups
are preserved below regardless of whether run_all() writes that file.

ANONYMIZATION (see docs/data_provenance.md and the task brief that produced
this module): this stage is the one place in the repo that reads raw,
real-participant-name ELAN exports (`Group_N.csv`), by design — unavoidable,
it's the anonymization step's own source stage. Per that constraint, the
real-name -> ParticipantN NAME mapping is treated as **external data the
caller supplies**: load_name_map() reads a local, gitignored JSON file
under data/ (never committed), and callers may pass a name_map dict
directly instead. No real participant name is embedded as a Python literal
anywhere in this module, unlike the source notebook's own hardcoded
per-group `NAME` dicts — NAME dicts are used here as data, not narration.
Every other per-group constant below (search windows, occurrence rules,
device offsets) is preserved verbatim from FINAL_ARDA_THESIS_ANALYSIS.md
since none of it is real-participant-identifying data.

Column-naming note: the source notebook's raw OE/Xsens per-stream CSVs and
their exact time-column name were not fully re-quoted in the analysis
(only the merge algorithm and grid constants were). This port assumes a
"time_s" (seconds, float) time column on every raw per-stream file and
"time_s"-keyed wide grids after merging — the smoke test builds synthetic
data on that assumption. If real raw files use a different time-column
name/unit, adjust `load_oe_participant_streams`/`load_xsens_participant`
accordingly; the merge/label/shift algorithms below don't otherwise depend
on that choice.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group_4 excluded (camera failure) — canonical everywhere in this repo
PARTICIPANTS = (1, 2, 3)

TIERS = [
    "Participant1", "Participant2", "Participant3",
    "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3",
    "Whole_Group",
]

SYNC_LABEL_COL = "label_Whole_Group"

# OpenEarable merge grid — identical across every group's OE-merge cell (verbatim constants).
OE_STREAMS = ["acc", "gyro", "mgnt", "bone_acc", "baro", "env_temp", "skin_temp", "ppg"]
OE_SLOW_STREAMS = {"skin_temp"}  # wider tolerance stream, ~32Hz native rate
OE_GRID_STEP_S = 0.02   # GRID_US = 20_000 -> 50 Hz
OE_TOL_FAST_S = 0.012   # TOL_FAST_US = 12_000
OE_TOL_SLOW_S = 0.040   # TOL_SLOW_US = 40_000

XSENS_HZ = 30  # native rate; master grid = Participant 1's clean ticks intersected with the other 2

# Group 3's two-part video concatenation offset: real wall-clock gap between
# PART1_START_CLOCK="12:21:50" and PART2_START_CLOCK="12:46:56" (no participant names involved).
GROUP3_PART2_OFFSET_S = 1506.0

ACC_PREFIX_TEMPLATE = {"openearable": "p{n}_acc_", "xsens": "p{n}_Acc_"}
ACC_AXIS_SUFFIXES = {"openearable": ("x", "y", "z"), "xsens": ("X", "Y", "Z")}


# ================================================================
# Per-group sync configuration (verbatim constants from
# FINAL_ARDA_THESIS_ANALYSIS.md section 1's per-group table)
# ================================================================

@dataclass
class OeXsensSyncConfig:
    group: int
    sync_labels: tuple[str, ...] = ()
    match_mode: str = "exact"                       # "exact" | "substring"
    occurrence: str = "longest"                      # "longest" | "first" | "last"
    expected_mid_range: tuple[float, float] | None = None
    search_mode: str = "relative"                     # "relative"|"absolute"|"manual_range"|"mid_window"|"last10pct"|"none"
    search_before_s: float = 15.0
    search_after_s: float = 5.0
    search_left_s: float | None = None
    search_right_s: float | None = None
    peak_users: tuple[int, ...] = (1, 2, 3)
    smooth_window: int = 25
    device_start_offset_s: float = 0.0                # applied to xsens only (Group 7's XSENS_TO_VIDEO_OFFSET_S)
    apply_shift_openearable: bool = False              # whether run_all() writes the shifted OE file
    apply_shift_xsens: bool = False                    # whether run_all() writes the shifted Xsens file
    clean_xsens_artifacts: bool = False                # Group 6 only


GROUP_SYNC_CONFIG: dict[int, OeXsensSyncConfig] = {
    # Group 1: last-10%-of-recording peak search; shift was computed by the
    # notebook but NOT adopted in its own final file selection -> unshifted.
    1: OeXsensSyncConfig(group=1, sync_labels=("synchronizaiton_move",), search_mode="last10pct"),

    # Group 2: no shift cell exists at all in the source notebook.
    2: OeXsensSyncConfig(group=2, search_mode="none"),

    # Group 3: peak search around the sync label; exact before/after window
    # wasn't re-quoted in the analysis for this specific group (only "peak
    # search around the sync label" plus a follow-up refinement cell), so
    # this reuses Group 5's documented window as the representative default.
    # Computed but not adopted in the final selection -> unshifted.
    3: OeXsensSyncConfig(group=3, sync_labels=("synchronizaiton_move",), search_mode="relative",
                          search_before_s=15.0, search_after_s=5.0),

    # Group 5: SEARCH_BEFORE_SYNC_S=15, SEARCH_AFTER_SYNC_S=5, PEAK_USER=3 only.
    # Computed but not adopted in the final selection -> unshifted.
    5: OeXsensSyncConfig(group=5, sync_labels=("clap_synchronizaiton_move", "synchronizaiton_move"),
                          search_mode="relative", search_before_s=15.0, search_after_s=5.0, peak_users=(3,)),

    # Group 6: substring match on "synchron" (not exact), manually hand-picked
    # numeric windows read off a plot: TARGET_SYNC_MIDDLE_RANGE=(1360,1375),
    # SEARCH_WINDOW=(1356,1361.5), SMOOTH_WIN=5. Artifact-cleaned but NOT
    # shifted in the final selection -> clean_xsens_artifacts on, shift off.
    6: OeXsensSyncConfig(group=6, sync_labels=("synchron",), match_mode="substring",
                          expected_mid_range=(1360.0, 1375.0), search_mode="manual_range",
                          search_left_s=1356.0, search_right_s=1361.5, smooth_window=5,
                          clean_xsens_artifacts=True),

    # Group 7: hardcoded device-start offset (XSENS_TO_VIDEO_OFFSET_S=59.0,
    # xsens only) applied before the peak shift. SHIFTED files ARE selected.
    7: OeXsensSyncConfig(group=7, sync_labels=("synchronizaiton_move",), search_mode="relative",
                          search_before_s=15.0, search_after_s=5.0, device_start_offset_s=59.0,
                          apply_shift_openearable=True, apply_shift_xsens=True),

    # Group 8: "SYNC EVENT 0" — first occurrence of possibly several sync
    # label segments. SHIFTED files ARE selected.
    8: OeXsensSyncConfig(group=8, sync_labels=("synchronizaiton_move",), occurrence="first",
                          search_mode="relative", search_before_s=15.0, search_after_s=5.0,
                          apply_shift_openearable=True, apply_shift_xsens=True),

    # Group 9: the official ELAN sync tag was found not to correspond to the
    # actual motion burst; an "early common peak" absolute window overrides
    # it (SEARCH_LEFT=50, SEARCH_RIGHT=70). SHIFTED_EARLY_PEAK files ARE selected.
    9: OeXsensSyncConfig(group=9, sync_labels=("synchronizaiton_move", "synchronization_move"),
                          search_mode="absolute", search_left_s=50.0, search_right_s=70.0,
                          apply_shift_openearable=True, apply_shift_xsens=True),

    # Group 10: latest of multiple sync occurrences, search_start=mid-5,
    # search_end=mid+35, restricted to USERS_FOR_SYNC=[1,3] (P2 short/unreliable).
    # SHIFTED files ARE selected.
    10: OeXsensSyncConfig(group=10, sync_labels=("synchronizaiton_move",), occurrence="last",
                           search_mode="mid_window", search_before_s=5.0, search_after_s=35.0,
                           peak_users=(1, 3), apply_shift_openearable=True, apply_shift_xsens=True),
}

# Plain unshifted output filenames (what global_cleaning.py selects for
# every (group, sensor) NOT in apply_shift_*=True above). Verbatim from
# FINAL_ARDA_THESIS_ANALYSIS.md / global_cleaning.py's SELECTED_FILE_NAMES.
UNSHIFTED_FILENAMES: dict[tuple[int, str], str] = {
    (g, sensor): f"group_{g}_{sensor}_labeled.csv" for g in GROUPS for sensor in ("openearable", "xsens")
}
UNSHIFTED_FILENAMES[(6, "xsens")] = "group_6_xsens_labeled_cleaned.csv"

# Shifted output filenames — one entry per (group, sensor) that HAS a shift
# cell in the source notebook, whether or not run_all() currently writes it
# (per the "still be able to produce it, just not wired in" scope note).
# Verbatim filenames from FINAL_ARDA_THESIS_ANALYSIS.md's outline section.
SHIFTED_FILENAMES: dict[tuple[int, str], str] = {
    (1, "xsens"): "group_1_xsens_labeled_shifted_by_last10_peak.csv",
    (1, "openearable"): "group_1_openearable_labeled_shifted_by_last10_peak.csv",
    (3, "openearable"): "group_3_openearable_labeled_shifted_by_sync_peak.csv",
    (3, "xsens"): "group_3_xsens_labeled_shifted_by_sync_peak.csv",
    (5, "xsens"): "group_5_xsens_labeled_shifted_by_clap_sync_peak.csv",
    (5, "openearable"): "group_5_openearable_labeled_shifted_by_sync_peak.csv",
    (6, "xsens"): "group_6_xsens_labeled_cleaned_SHIFTED.csv",
    (7, "openearable"): "group_7_openearable_labeled_SHIFTED.csv",
    (7, "xsens"): "group_7_xsens_labeled_SHIFTED.csv",
    (8, "openearable"): "group_8_openearable_labeled_SHIFTED.csv",
    (8, "xsens"): "group_8_xsens_labeled_SHIFTED.csv",
    (9, "openearable"): "group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv",
    (9, "xsens"): "group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv",
    (10, "openearable"): "group_10_openearable_labeled_SHIFTED.csv",
    (10, "xsens"): "group_10_xsens_labeled_SHIFTED.csv",
}


# ================================================================
# Name-map loading (anonymization boundary — see module docstring)
# ================================================================

def load_name_map(data_root: str, group: int) -> dict:
    """Loads {real_name: 'ParticipantN'} from a local, gitignored JSON file
    at {data_root}/group_{g}/elan/name_map.json. This file is never
    committed to the repo and this function never embeds a real name as a
    Python literal — callers may instead pass a name_map dict directly to
    process_group()/run_group() from whatever private local source they
    maintain."""
    path = os.path.join(data_root, f"group_{group}", "elan", "name_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rename_tier(tier_value: str, name_map: dict) -> str:
    """Maps a real-name (or name-pair, e.g. 'Bas_Arda') raw ELAN tier value
    to its canonical ParticipantN / ParticipantN_ParticipantM form. Tiers
    already canonical (Whole_Group, or already a ParticipantN value) pass
    through unchanged."""
    if tier_value in name_map:
        return name_map[tier_value]
    if tier_value == "Whole_Group" or tier_value in name_map.values():
        return tier_value
    parts = re.split(r"[_,]+", tier_value)
    mapped = [name_map.get(p, p) for p in parts]

    def sort_key(x):
        m = re.search(r"\d+", x)
        return int(m.group()) if m else 999

    mapped = sorted(dict.fromkeys(mapped), key=sort_key)
    return "_".join(mapped)


def anonymize_elan(elan_df: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    df = elan_df.copy()
    df["tier"] = df["tier"].astype(str).apply(lambda t: rename_tier(t, name_map))
    return df


# ================================================================
# Raw ELAN read, individual_build gap-fill, Group-3 concatenation
# ================================================================

def raw_elan_path(data_root: str, group: int) -> str:
    return os.path.join(data_root, f"group_{group}", "elan", f"Group_{group}.csv")


RAW_ELAN_COLS = ["tier", "blank", "begin_hms", "begin_s", "end_hms", "end_s", "dur_hms", "dur_s", "label"]


def read_raw_elan(path: str) -> pd.DataFrame:
    """Reads the raw (real-name) ELAN export. **Headerless** CSV, fixed
    9-column layout (verbatim from FINAL_ARDA_THESIS.ipynb CELL 10:
    `pd.read_csv(ELAN_PATH, header=None, names=COLS, ...)`, COLS =
    RAW_ELAN_COLS below) — not a headered tier/begin_s/end_s/label file.
    Returns just the 4 columns the rest of this module needs; the exact
    idiom quoted in FINAL_ARDA_THESIS_ANALYSIS.md,
    `elan[elan.tier==t][["begin_s","end_s","label"]]`, operates on this
    already-parsed frame, not on the raw file's own (nonexistent) header
    row."""
    df = pd.read_csv(path, header=None, names=RAW_ELAN_COLS, dtype={"blank": str})
    return df[["tier", "begin_s", "end_s", "label"]].copy()


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _involved_tiers(person: str, all_tiers) -> list[str]:
    """Which raw tier values count as this person being "busy", for the
    individual_build gap computation. Verbatim rule from CELL 10's
    per-group `INVOLVING` dict (e.g. `"Arda": ["Arda", "Arda_Rachel",
    "Bas_Arda", "Whole_Group"]`) generalized so it doesn't need a
    hardcoded, real-name-keyed dict per group: a tier counts if it *is*
    the person's own solo tier, is "Whole_Group", or is an underscore-
    joined pair tier that has the person as one of its two components
    (either order, matching the source's own "Arda_Rachel"/"Bas_Arda"
    asymmetric naming). Operates purely on whatever tier strings are
    actually present in this group's raw ELAN — no participant name is a
    Python literal here."""
    involved = {person, "Whole_Group"}
    for t in all_tiers:
        parts = t.split("_")
        if len(parts) == 2 and person in parts:
            involved.add(t)
    return sorted(involved)


def apply_individual_build(elan_df: pd.DataFrame, cutoffs_s: dict) -> pd.DataFrame:
    """Mirrors CELL 10's per-participant CUTOFF gap synthesis: for every
    person in cutoffs_s, finds every interval in [0, cutoff] where that
    person is not already "busy" — annotated on their own solo tier, a
    pair tier they're part of, or Whole_Group, per `_involved_tiers` —
    and fills each such gap with a synthetic "individual_build" segment
    on that person's own solo tier (the pre-collaboration solo-assembly
    phase). cutoffs_s is keyed by whatever raw tier value the caller's
    raw ELAN uses for that person (real name or otherwise) — this
    function does not care what the keys mean beyond that. Apply this
    BEFORE anonymize_elan(), matching the source notebook's own cell
    order (CELL 10 before CELL 12)."""
    if not cutoffs_s:
        return elan_df.copy()

    all_tiers = elan_df["tier"].dropna().unique().tolist()
    rows = []
    for person, cutoff in cutoffs_s.items():
        tiers = _involved_tiers(person, all_tiers)
        existing = elan_df[elan_df["tier"].isin(tiers) & (elan_df["begin_s"] < cutoff)]
        intervals = _merge_intervals(
            [(float(s), float(min(e, cutoff))) for s, e in zip(existing["begin_s"], existing["end_s"])]
        )
        cursor = 0.0
        for s, e in intervals:
            if s > cursor:
                rows.append({"tier": person, "begin_s": cursor, "end_s": s, "label": "individual_build"})
            cursor = max(cursor, e)
        if cursor < cutoff:
            rows.append({"tier": person, "begin_s": cursor, "end_s": cutoff, "label": "individual_build"})

    if not rows:
        return elan_df.copy()
    gap_df = pd.DataFrame(rows)
    return pd.concat([elan_df, gap_df], ignore_index=True).sort_values(["tier", "begin_s"]).reset_index(drop=True)


def shift_elan_times(elan_df: pd.DataFrame, offset_s: float, time_cols=("begin_s", "end_s")) -> pd.DataFrame:
    df = elan_df.copy()
    for c in time_cols:
        df[c] = df[c] + offset_s
    return df


def concatenate_group3_parts(part1_df: pd.DataFrame, part2_df: pd.DataFrame,
                              offset_s: float = GROUP3_PART2_OFFSET_S) -> pd.DataFrame:
    """Mirrors CELL 39 (GROUP 3 - STEP 0): Part 2's ELAN timestamps are
    shifted by the real wall-clock gap between the two recordings' start
    times before the two tables are stacked (the session's video recording
    was split into two files). Group 3 is otherwise identical to every
    other group once concatenated."""
    part2_shifted = shift_elan_times(part2_df, offset_s)
    return pd.concat([part1_df, part2_shifted], ignore_index=True).sort_values("begin_s").reset_index(drop=True)


# ================================================================
# Sensor merge — OpenEarable (50Hz wide grid) / Xsens (~30Hz wide grid)
# ================================================================

def oe_raw_paths(data_root: str, group: int, participant: int) -> dict:
    base = os.path.join(data_root, f"group_{group}", "openearable", f"Participant{participant}")
    return {stream: os.path.join(base, f"Participant{participant}_{stream}.csv") for stream in OE_STREAMS}


def xsens_raw_path(data_root: str, group: int, participant: int) -> str:
    return os.path.join(data_root, f"group_{group}", "xsens", f"Participant{participant}.csv")


def load_oe_participant_streams(data_root: str, group: int, participant: int, time_col: str = "time_s") -> dict:
    streams = {}
    for stream, path in oe_raw_paths(data_root, group, participant).items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if time_col not in df.columns:
                raise ValueError(f"{path}: missing time column {time_col!r}")
            streams[stream] = df
    return streams


def _merge_asof_onto_grid(df_stream: pd.DataFrame, time_col: str, value_cols: list, grid: np.ndarray, tol_s: float) -> pd.DataFrame:
    stream = df_stream[[time_col] + value_cols].dropna(subset=[time_col]).sort_values(time_col)
    grid_df = pd.DataFrame({time_col: grid})
    merged = pd.merge_asof(grid_df, stream, on=time_col, direction="nearest", tolerance=tol_s)
    return merged[value_cols]


def merge_openearable_participant(streams: dict, time_col: str = "time_s", grid_step_s: float = OE_GRID_STEP_S,
                                   tol_fast_s: float = OE_TOL_FAST_S, tol_slow_s: float = OE_TOL_SLOW_S,
                                   slow_streams=OE_SLOW_STREAMS) -> pd.DataFrame:
    """One participant's raw per-stream OpenEarable files (acc/gyro/mgnt/
    bone_acc/baro/env_temp/skin_temp/ppg) merge-asof'd onto a shared 50Hz
    grid, GRID_US=20_000 (grid_step_s=0.02), with a wider tolerance for the
    ~32Hz skin_temp stream (TOL_SLOW_US=40_000) vs. every other stream
    (TOL_FAST_US=12_000)."""
    starts = [df[time_col].min() for df in streams.values() if len(df)]
    ends = [df[time_col].max() for df in streams.values() if len(df)]
    if not starts:
        raise ValueError("no non-empty OpenEarable streams to merge")
    start, end = max(starts), min(ends)
    grid = np.arange(start, end, grid_step_s)

    out = pd.DataFrame({time_col: grid})
    for stream_name, df_stream in streams.items():
        value_cols = [c for c in df_stream.columns if c != time_col]
        tol = tol_slow_s if stream_name in slow_streams else tol_fast_s
        merged_vals = _merge_asof_onto_grid(df_stream, time_col, value_cols, grid, tol)
        for c in value_cols:
            out[f"{stream_name}_{c}"] = merged_vals[c].values
    return out


def merge_openearable_group(participant_streams: dict, time_col: str = "time_s", **kwargs) -> pd.DataFrame:
    """Wide p1_*/p2_*/p3_*-prefixed 50Hz table on the intersected overlap
    window across all 3 participants. participant_streams: {participant:
    {stream_name: df}}."""
    per_participant = {
        p: merge_openearable_participant(streams, time_col=time_col, **kwargs)
        for p, streams in participant_streams.items()
    }
    starts = [df[time_col].min() for df in per_participant.values()]
    ends = [df[time_col].max() for df in per_participant.values()]
    start, end = max(starts), min(ends)

    out = None
    for p, df in per_participant.items():
        df_win = df[(df[time_col] >= start) & (df[time_col] <= end)].reset_index(drop=True)
        df_win = df_win.rename(columns={c: f"p{p}_{c}" for c in df_win.columns if c != time_col})
        out = df_win if out is None else pd.merge(out, df_win, on=time_col, how="inner")
    return out


def load_xsens_participant(data_root: str, group: int, participant: int, time_col: str = "time_s") -> pd.DataFrame:
    path = xsens_raw_path(data_root, group, participant)
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"{path}: missing time column {time_col!r}")
    return df


def merge_xsens_group(participant_frames: dict, time_col: str = "time_s") -> pd.DataFrame:
    """Merges each participant's Euler/Acc/Gyr columns onto Participant 1's
    native ~30Hz time grid, using only ticks common to all 3 participants
    (rows with any NaN from a non-P1 participant's nearest-match are
    dropped) — no synthetic fixed-rate grid is built, unlike OpenEarable's
    50Hz grid."""
    p1 = participant_frames[1].sort_values(time_col).reset_index(drop=True)
    grid = p1[time_col].to_numpy()
    out = p1.rename(columns={c: f"p1_{c}" for c in p1.columns if c != time_col})

    for p, df in participant_frames.items():
        if p == 1:
            continue
        df_sorted = df.sort_values(time_col)
        value_cols = [c for c in df_sorted.columns if c != time_col]
        merged = pd.merge_asof(pd.DataFrame({time_col: grid}), df_sorted, on=time_col, direction="nearest")
        for c in value_cols:
            out[f"p{p}_{c}"] = merged[c].values

    non_time_cols = [c for c in out.columns if c != time_col]
    return out.dropna(subset=non_time_cols, how="any").reset_index(drop=True)


def build_oe_merged_grid(data_root: str, group: int, time_col: str = "time_s") -> pd.DataFrame:
    participant_streams = {p: load_oe_participant_streams(data_root, group, p, time_col) for p in PARTICIPANTS}
    return merge_openearable_group(participant_streams, time_col=time_col)


def build_xsens_merged_grid(data_root: str, group: int, time_col: str = "time_s") -> pd.DataFrame:
    participant_frames = {p: load_xsens_participant(data_root, group, p, time_col) for p in PARTICIPANTS}
    return merge_xsens_group(participant_frames, time_col=time_col)


# ================================================================
# Label attachment (merge-asof, per tier)
# ================================================================

def attach_tier_label(sensor_df: pd.DataFrame, elan_df: pd.DataFrame, tier: str, time_col: str = "time_s") -> pd.Series:
    seg = (elan_df[elan_df["tier"] == tier][["begin_s", "end_s", "label"]]
           .dropna(subset=["begin_s"]).sort_values("begin_s").reset_index(drop=True))
    if seg.empty:
        return pd.Series([""] * len(sensor_df), index=sensor_df.index)

    left = sensor_df[[time_col]].reset_index().sort_values(time_col)
    merged = pd.merge_asof(left, seg, left_on=time_col, right_on="begin_s", direction="backward")
    merged = merged.set_index("index").sort_index()
    label = merged["label"].where(merged[time_col] <= merged["end_s"], "")
    return label.fillna("").astype(str)


def attach_all_labels(sensor_df: pd.DataFrame, elan_df: pd.DataFrame, time_col: str = "time_s") -> pd.DataFrame:
    df = sensor_df.copy()
    for tier in TIERS:
        df[f"label_{tier}"] = attach_tier_label(df, elan_df, tier, time_col)
    return df


# ================================================================
# Group 6 xsens artifact cleaning
# ================================================================

def clean_extreme_artifacts(df: pd.DataFrame, value_cols: list, z_thresh: float = 6.0):
    """Approximate re-implementation of CELL 80's "clean/cure extreme
    artifacts" step: NaNs out values that are an extreme outlier
    (|z-score| > z_thresh) within value_cols. FINAL_ARDA_THESIS_ANALYSIS.md
    documents only the row count (82) and provenance ("these are chosen
    according to your image" — read off a plot) for the original
    selection, not an exact reusable criterion — this is a faithful,
    generic approximation of the same intent (strip sensor-glitch spikes
    before syncing), not a byte-exact reproduction of which 82 rows."""
    df = df.copy()
    removed = 0
    for col in value_cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mu, sigma = np.nanmean(x), np.nanstd(x)
        if not np.isfinite(sigma) or sigma == 0:
            continue
        mask = np.abs((x - mu) / sigma) > z_thresh
        removed += int(mask.sum())
        df.loc[mask, col] = np.nan
    return df, removed


# ================================================================
# Peak-alignment sync shift
# ================================================================

def find_label_segments(df: pd.DataFrame, time_col: str, label_col: str) -> list:
    """Collapses a text label column into (start, end, label) runs of
    consecutive equal non-empty values."""
    labels = df[label_col].fillna("").astype(str).to_numpy()
    times = df[time_col].to_numpy()
    segments = []
    cur_label, seg_start = None, None
    for i, lab in enumerate(labels):
        if lab != cur_label:
            if cur_label not in (None, ""):
                segments.append((seg_start, times[i - 1], cur_label))
            cur_label, seg_start = lab, times[i]
    if cur_label not in (None, "") and len(times):
        segments.append((seg_start, times[-1], cur_label))
    return segments


def combined_acc_signal(df: pd.DataFrame, participants, prefix_template: str, axis_suffixes=("x", "y", "z")) -> np.ndarray:
    mags = []
    for p in participants:
        cols = [f"{prefix_template.format(n=p)}{ax}" for ax in axis_suffixes]
        if not all(c in df.columns for c in cols):
            continue
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        mags.append(np.sqrt((vals ** 2).sum(axis=1)))
    if not mags:
        return np.full(len(df), np.nan)
    return np.nanmean(np.vstack(mags), axis=0)


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(signal).rolling(window, center=True, min_periods=1).mean().to_numpy()


def find_peak_time(df: pd.DataFrame, time_col: str, signal: np.ndarray, search_left: float, search_right: float) -> float:
    times = df[time_col].to_numpy()
    mask = (times >= search_left) & (times <= search_right)
    idx = np.where(mask)[0]
    if len(idx) == 0 or not np.isfinite(signal[idx]).any():
        raise ValueError(f"no finite signal samples in search window [{search_left}, {search_right}]")
    best = idx[np.nanargmax(signal[idx])]
    return float(times[best])


def resolve_search_window(cfg: OeXsensSyncConfig, df: pd.DataFrame, time_col: str,
                           sync_start: float, sync_end: float, sync_mid: float) -> tuple:
    if cfg.search_mode == "relative":
        return sync_start - cfg.search_before_s, sync_end + cfg.search_after_s
    if cfg.search_mode in ("absolute", "manual_range"):
        return cfg.search_left_s, cfg.search_right_s
    if cfg.search_mode == "mid_window":
        return sync_mid - cfg.search_before_s, sync_mid + cfg.search_after_s
    if cfg.search_mode == "last10pct":
        start_idx = int(len(df) * 0.90)
        return float(df[time_col].iloc[start_idx]), float(df[time_col].iloc[-1])
    raise ValueError(f"unknown search_mode {cfg.search_mode!r}")


def compute_peak_shift(labeled_df: pd.DataFrame, cfg: OeXsensSyncConfig, sensor: str,
                        time_col: str = "time_s", label_col: str = SYNC_LABEL_COL):
    """Core algorithm (FINAL_ARDA_THESIS_ANALYSIS.md section 1, quoted from
    CELL 66): locate the sync-anchor ELAN label segment on the sensor grid's
    already-attached label_Whole_Group column, take its midpoint, find the
    highest peak of a (participant-averaged) accelerometer-magnitude signal
    within a per-group search window, and return offset_s = peak_time -
    sync_mid (the amount every ELAN label segment should be shifted by).
    For xsens, cfg.device_start_offset_s (Group 7's hardcoded device-start
    correction) is added on top. Returns (offset_s, sync_mid, peak_time)."""
    if cfg.search_mode == "none":
        raise ValueError(f"group {cfg.group} has no sync mechanism (search_mode='none')")

    segments = find_label_segments(labeled_df, time_col, label_col)
    if cfg.match_mode == "substring":
        matches = [s for s in segments if any(sl.lower() in s[2].lower() for sl in cfg.sync_labels)]
    else:
        matches = [s for s in segments if s[2] in cfg.sync_labels]

    if cfg.expected_mid_range is not None:
        lo, hi = cfg.expected_mid_range
        ranged = [s for s in matches if lo <= (s[0] + s[1]) / 2 <= hi]
        if ranged:
            matches = ranged

    if not matches:
        raise ValueError(f"group {cfg.group}/{sensor}: no sync-label segment found matching {cfg.sync_labels!r}")

    if cfg.occurrence == "first":
        chosen = min(matches, key=lambda s: s[0])
    elif cfg.occurrence == "last":
        chosen = max(matches, key=lambda s: s[0])
    else:
        chosen = max(matches, key=lambda s: s[1] - s[0])

    sync_start, sync_end, _ = chosen
    sync_mid = (sync_start + sync_end) / 2
    left, right = resolve_search_window(cfg, labeled_df, time_col, sync_start, sync_end, sync_mid)

    signal = combined_acc_signal(labeled_df, cfg.peak_users, ACC_PREFIX_TEMPLATE[sensor], ACC_AXIS_SUFFIXES[sensor])
    signal = smooth_signal(signal, cfg.smooth_window)
    peak_time = find_peak_time(labeled_df, time_col, signal, left, right)

    offset_s = peak_time - sync_mid
    if sensor == "xsens":
        offset_s += cfg.device_start_offset_s
    return offset_s, sync_mid, peak_time


# ================================================================
# Orchestration
# ================================================================

def output_path(data_root: str, group: int, sensor: str, filename: str, out_dir: str | None = None) -> str:
    root = out_dir or data_root
    return os.path.join(root, f"group_{group}", sensor, f"{sensor}_labeled", filename)


def process_group(data_root: str, group: int, name_map: dict, out_dir: str | None = None,
                   cutoffs_s: dict | None = None, elan_df: pd.DataFrame | None = None) -> dict:
    """Runs merge -> individual_build -> anonymize -> label-attach ->
    (Group 6 only) artifact-clean -> (where GROUP_SYNC_CONFIG wires it in)
    peak-shift, for both sensors of one group. Always writes the plain
    unshifted (or, Group 6 xsens, cleaned-but-unshifted) file; additionally
    writes the shifted file when cfg.apply_shift_{sensor} is True.

    `elan_df` lets a caller inject an already-loaded raw ELAN table (e.g.
    Group 3's concatenate_group3_parts() result, or a synthetic table in
    tests) instead of reading raw_elan_path() from disk.
    """
    cfg = GROUP_SYNC_CONFIG[group]

    if elan_df is None:
        elan_df = read_raw_elan(raw_elan_path(data_root, group))
    if cutoffs_s:
        elan_df = apply_individual_build(elan_df, cutoffs_s)
    elan_df = anonymize_elan(elan_df, name_map)

    oe_grid = build_oe_merged_grid(data_root, group)
    xsens_grid = build_xsens_merged_grid(data_root, group)

    results = {}
    for sensor, grid in (("openearable", oe_grid), ("xsens", xsens_grid)):
        labeled = attach_all_labels(grid, elan_df)
        artifact_removed = None
        artifact_cols = None
        if sensor == "xsens" and cfg.clean_xsens_artifacts:
            artifact_cols = [c for c in labeled.columns if re.search(r"_acc_[xyz]$", c, re.IGNORECASE)]
            labeled, artifact_removed = clean_extreme_artifacts(labeled, artifact_cols)

        plain_name = UNSHIFTED_FILENAMES[(group, sensor)]
        plain_path = output_path(data_root, group, sensor, plain_name, out_dir)
        os.makedirs(os.path.dirname(plain_path), exist_ok=True)
        labeled.to_csv(plain_path, index=False)
        results[sensor] = {"labeled_path": plain_path, "artifact_rows_removed": artifact_removed}

        apply_shift = cfg.apply_shift_openearable if sensor == "openearable" else cfg.apply_shift_xsens
        if cfg.search_mode != "none":
            try:
                offset_s, sync_mid, peak_time = compute_peak_shift(labeled, cfg, sensor)
                results[sensor]["shift_info"] = {"offset_s": offset_s, "sync_mid": sync_mid, "peak_time": peak_time}
                if apply_shift:
                    shifted_elan = shift_elan_times(elan_df, offset_s)
                    shifted_labeled = attach_all_labels(grid, shifted_elan)
                    if artifact_cols is not None:
                        shifted_labeled, _ = clean_extreme_artifacts(shifted_labeled, artifact_cols)
                    shifted_name = SHIFTED_FILENAMES[(group, sensor)]
                    shifted_path = output_path(data_root, group, sensor, shifted_name, out_dir)
                    shifted_labeled.to_csv(shifted_path, index=False)
                    results[sensor]["shifted_path"] = shifted_path
            except ValueError as e:
                results[sensor]["shift_error"] = str(e)

    return results


def run_all(data_root: str, out_dir: str | None = None, name_maps: dict | None = None,
            cutoffs: dict | None = None) -> dict:
    """Runs process_group() for every group in GROUPS. name_maps/cutoffs:
    {group: name_map_dict} / {group: cutoffs_s_dict}; if name_maps is None,
    load_name_map(data_root, group) is used per group. Groups that fail
    (e.g. missing raw input files) record an "error" key instead of raising,
    so one bad group doesn't abort the rest."""
    name_maps = name_maps or {}
    cutoffs = cutoffs or {}
    results = {}
    for group in GROUPS:
        try:
            name_map = name_maps.get(group) or load_name_map(data_root, group)
            results[group] = process_group(data_root, group, name_map, out_dir=out_dir, cutoffs_s=cutoffs.get(group))
        except Exception as e:
            results[group] = {"error": str(e)}
    return results
