"""Raw OptiTrack sensor sync + ELAN label transfer (Ch.4 Tables 4.1-4.4's
upstream stage, OptiTrack half — see `raw_sync_oe_xsens.py` for the
OpenEarable/Xsens half of the same stage).

Ported from `OPTI_TRACK_PROCESSING.ipynb` — see
notebooks_reference/OPTI_TRACK_PROCESSING_ANALYSIS.md for the full
per-group trace this module is built from (exact cell numbers, exact
algorithms quoted verbatim, exact per-group constants, exact output
filenames).

SCOPE: `process_group()`/`run_all()` below start from `group_{g}/
optitrack/optitrack_final/group_{g}_optitrack_cleaned_combined_240hz.csv`
as a fixed intermediate input (the same convention `global_cleaning.py`
uses for its own `*_labeled*.csv` inputs) — they do NOT call the raw
marker reconstruction functions themselves; a caller that wants the full
raw-Motive-export-to-labeled-CSV chain must call
`reconstruct_markers_group{1,2,3,5,7,8,9,10}()` first and pass its output
in as `optitrack_df`. UPDATE (2026-09-10/11 + 2026-09-11): the raw marker
reconstruction pass itself — originally believed out of scope, "the
single least-automatable step in the whole project" (a literal mapping of
raw Motive "Unlabeled NNNN" marker-track IDs to participant identities,
unique per group and per take, not derivable from a rule) — has since
been ported and validated for ALL 9 groups that have an OptiTrack
recording (Group 1 first, then 2/3/5/7/8/9/10 bit-for-bit exact against
real raw Motive takes; Group 6 added 2026-09-11, see below — every study
group with an OptiTrack recording is now covered; group_4 has none,
camera failure, excluded everywhere in this repo). See
`reconstruct_markers_group1()`'s docstring below for the Group-1 trace
and `reconstruct_markers_multi_take()`'s for the shared mechanism the
other 8 groups turned out to genuinely share.

CORRECTION (2026-09-11): earlier revisions of this module and of
`docs/table_to_source_mapping.md` stated as fact that "Group 6 has no
OptiTrack recording." That claim was never independently verified and
was wrong — real raw Motive take files exist
(data/raw/group_6/optitrack/Arda_Group-6_Take_{1,2,3}.csv), Group 6 has
its own full section in OPTI_TRACK_PROCESSING.ipynb (CELL 22-26 raw
cleaning, CELL 71-73 sync/labeling), and the official Task 2 reference
CSV has 82 real, non-null Group 6 `opti2_*` rows. `reconstruct_markers_
group6()` below is the now-ported ground truth; do not resurrect the old
"no recording" claim anywhere else in this repo.
`scripts/reproduce_pipeline.py`'s `sync` stage still bridges OptiTrack
from the pre-reconstructed fixture rather than calling these functions —
wiring that up is a follow-up, not yet done as of this update.

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

import csv
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
# Raw marker-track reconstruction (Group 1) — ported from
# OPTI_TRACK_PROCESSING.ipynb's own actually-authoritative Group-1 path
# for producing group_1_optitrack_cleaned_combined_240hz.csv from the
# TRUE raw Motive export (raw 3D marker tracks like "Unlabeled 1682",
# no stable participant identity).
#
# Traced by reading every OptiTrack Group-1 cell in
# notebooks_reference/OPTI_TRACK_PROCESSING_CODE_ONLY.py IN ORDER (not
# grepped, not assumed): the notebook actually contains SIX distinct
# Group-1 reconstruction/stitching cells, not three. The first three —
# "FAST FIRST PASS 3-LANDMARK RECONSTRUCTION" (MAX_ASSIGN_DIST=0.35,
# writes optitrack_cleaned/), "CONSERVATIVE 3-LANDMARK RECONSTRUCTION"
# (MAX_ASSIGN_DIST=0.18, writes optitrack_cleaned_conservative/), and
# "BALANCED 3-LANDMARK TRACKER" (MAX_ASSIGN_DIST=0.32,
# MAX_REACQUIRE_DIST=0.55, writes optitrack_cleaned_balanced/) — are all
# scipy.optimize.linear_sum_assignment (Hungarian-algorithm) nearest-
# neighbor trackers, but NONE of their three output directories is ever
# read by any later cell in the notebook; they are abandoned exploratory
# variants, not the authoritative path, confirmed by grepping every
# `OUT_DIR =`/`IN_DIR =` assignment in the whole Group-1 section and by
# the fact that the real, already-validated
# group_1_optitrack_cleaned_combined_240hz.csv (see
# RAW_VALIDATION/group_1_optitrack/.../optitrack_final/) has the column
# set frame,time_s,landmark{1,2,3}_{x,y,z,source},
# landmark{1,2,3}_available,active_clean_landmarks,take,
# time_s_original — with a `_source` provenance column (a "+"-joined
# list of literal raw marker names) that only the manual-chains path
# below produces; the Hungarian-tracker cells never write a `_source`
# column at all.
#
# The actually-authoritative chain is:
#   "MANUAL TRACKLET STITCHING" (first hand-picked CHAINS dict, writes
#     optitrack_cleaned_manual_stitched/)
#   -> "FIND CANDIDATE FRAGMENTS FOR MISSING GAPS" (diagnostic-only
#     inspection of unused marker fragments near existing chain gaps;
#     produces no file consumed by any later cell)
#   -> "UPDATED MANUAL TRACKLET STITCHING" (CHAINS revised after the gap
#     inspection above — the version ported below — writes
#     optitrack_cleaned_manual_stitched_UPDATED/)
#   -> "COMBINE TAKE 1 + TAKE 2" — its own `IN_DIR` is set to exactly the
#     "UPDATED" cell's `OUT_DIR`, and its own `OUT_DIR`/`out_path` is
#     exactly group_1/optitrack_final/
#     group_1_optitrack_cleaned_combined_240hz.csv (verbatim, no
#     ambiguity — this cell's file path IS the target filename).
#
# So the real per-(take, landmark) marker reconstruction here is NOT a
# distance-based assignment algorithm at all — it is a hand-curated
# allowlist of literal raw marker names per landmark per take
# (`GROUP1_CHAINS` below), selected by the notebook's author from the
# tracklet-timeline/successor-candidate plots the two inspection cells
# produce. This matches this module's own pre-existing docstring
# characterization ("a literal mapping of raw Motive 'Unlabeled NNNN'
# marker-track IDs to participant identities, unique per group and per
# take, not derivable from a rule") — confirmed here for Group 1
# specifically, cell-by-cell, rather than assumed.
# ================================================================

RAW_MOTIVE_HEADER_ROWS = 7  # meta/name/id/axis header rows before the frame data in a raw Motive export

# Verbatim from CELL 8 "OPTITRACK GROUP 1 - UPDATED MANUAL TRACKLET
# STITCHING"'s own `CHAINS` dict — the revision actually consumed by the
# combine cell (superseding CELL 6's earlier, narrower `CHAINS`, which
# never reaches the combined file).
GROUP1_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1669", "Unlabeled 1682", "Unlabeled 1726"],
        "landmark2": [
            "Unlabeled 1670", "Unlabeled 1687", "Unlabeled 1691", "Unlabeled 1704",
            "Unlabeled 1710", "Unlabeled 1718", "Unlabeled 1735",
        ],
        "landmark3": [
            "Unlabeled 1668", "Unlabeled 1674", "Unlabeled 1673", "Unlabeled 1678",
            "Unlabeled 1680", "Unlabeled 1683", "Unlabeled 1696", "Unlabeled 1703",
            "Unlabeled 1705", "Unlabeled 1711", "Unlabeled 1717", "Unlabeled 1732",
            "Unlabeled 1733",
        ],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1820"],
        "landmark2": ["Unlabeled 1823"],
        "landmark3": ["Unlabeled 1826", "Unlabeled 1829", "Unlabeled 1835"],
    },
}

# Verbatim from CELL 9 "OPTITRACK GROUP 1 - COMBINE TAKE 1 + TAKE 2"'s own
# comment: hardcoded from the two takes' real Motive "Capture Start Time"
# metadata values (10:23:10.716 -> 10:46:16.552), not re-derived at
# runtime from the raw files themselves.
GROUP1_TAKE2_OFFSET_S = 1385.836


def read_motive_long_and_frame_time(path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Reads one raw Motive/OptiTrack long-format take export: 7
    metadata/header rows (`RAW_MOTIVE_HEADER_ROWS`), then one row per
    exported frame with `frame`, `time_s`, then one (x, y, z) column
    triple per marker starting at column index 2. Returns
    `(meta, frame_time, long_df)`: `meta` is the parsed first header row
    (key/value pairs), `frame_time` is the take's full (frame, time_s)
    grid, and `long_df` is one row per (frame, marker) restricted to
    frames where that marker's x/y/z were all present. Verbatim from
    CELL 8's `read_motive_long_and_frame_time`."""
    header_rows = []
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(RAW_MOTIVE_HEADER_ROWS):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=RAW_MOTIVE_HEADER_ROWS, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({"frame": frame.astype("Int64"), "time_s": time_s}).dropna().copy()
    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []
    for info in marker_infos:
        c = info["start_col"]
        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]
        present = xyz.notna().all(axis=1)
        if present.sum() == 0:
            continue
        parts.append(pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        }))

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)
    return meta, frame_time, long_df


def build_stitched_clean(frame_time: pd.DataFrame, long_df: pd.DataFrame,
                          chains: dict[str, list[str]]) -> pd.DataFrame:
    """For each landmark (in `chains`' own key order — landmark1,
    landmark2, landmark3), averages the x/y/z of every listed raw marker
    name present on a given frame (more than one simultaneously-present
    listed marker -> mean; provenance recorded as a "+"-joined sorted
    list of the marker names actually used, in `<landmark>_source`),
    left-merged onto the take's full frame/time_s grid so every frame is
    kept even when no listed marker is present. Verbatim from CELL 8's
    `build_stitched_clean`."""
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)
    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()
        lm_by_frame = (
            lm_df.groupby("frame", as_index=False)
            .agg(x=("x", "mean"), y=("y", "mean"), z=("z", "mean"),
                 source_markers=("marker_name", lambda x: "+".join(sorted(set(x)))))
        )
        clean = clean.merge(lm_by_frame, on="frame", how="left")
        clean = clean.rename(columns={
            "x": f"{landmark_name}_x", "y": f"{landmark_name}_y", "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source",
        })

    for i in (1, 2, 3):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)
    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)
    return clean


def smooth_short_gaps(clean: pd.DataFrame, limit: int = 10, window: int = 5) -> pd.DataFrame:
    """Per axis, per landmark: linearly interpolate gaps of up to
    `limit` consecutive missing frames (both directions), then apply a
    centered rolling mean of `window` frames (min_periods=1). Verbatim
    from CELL 8's `smooth_short_gaps` (same defaults, same order of
    operations — interpolate THEN rolling-mean, applied to every frame
    including already-non-missing ones)."""
    out = clean.copy()
    for i in (1, 2, 3):
        for axis in ("x", "y", "z"):
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col].interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1).mean()
            )
    for i in (1, 2, 3):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)
    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)
    return out


def reconstruct_markers_group1(take1_path: str, take2_path: str,
                                take2_offset_s: float = GROUP1_TAKE2_OFFSET_S) -> pd.DataFrame:
    """Full Group-1 raw-marker reconstruction, from the two raw Motive
    take exports straight through to the same shape as
    `group_1_optitrack_cleaned_combined_240hz.csv`: for each take, reads
    the raw long-format export, stitches Landmark1/2/3 from
    `GROUP1_CHAINS` (the notebook's hand-curated, gap-inspection-revised
    marker-name allowlist), smooths short gaps, then concatenates take_2
    (shifted by `take2_offset_s`, the real inter-take capture-start gap)
    after take_1 and re-sorts by `time_s`, recomputing per-row landmark
    availability on the combined frame exactly as the source does.
    Verbatim from CELL 8 ("UPDATED MANUAL TRACKLET STITCHING") + CELL 9
    ("COMBINE TAKE 1 + TAKE 2"); column order matches the real fixture
    exactly (frame, time_s, landmark{1,2,3}_{x,y,z,source},
    landmark{1,2,3}_available, active_clean_landmarks, take,
    time_s_original)."""
    take_paths = {"take_1": take1_path, "take_2": take2_path}
    stitched = {}
    for take_name, path in take_paths.items():
        _, frame_time, long_df = read_motive_long_and_frame_time(path)
        clean = build_stitched_clean(frame_time, long_df, GROUP1_CHAINS[take_name])
        stitched[take_name] = smooth_short_gaps(clean)

    take1 = stitched["take_1"].copy()
    take2 = stitched["take_2"].copy()
    take1["take"] = "take_1"
    take2["take"] = "take_2"
    take1["time_s_original"] = take1["time_s"]
    take2["time_s_original"] = take2["time_s"]
    take1["time_s"] = take1["time_s_original"]
    take2["time_s"] = take2["time_s_original"] + take2_offset_s

    combined = pd.concat([take1, take2], ignore_index=True)
    combined = combined.sort_values("time_s").reset_index(drop=True)

    for i in (1, 2, 3):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)
    combined["active_clean_landmarks"] = combined[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)
    return combined


# ================================================================
# Raw marker-track reconstruction (Groups 2, 3, 5, 6, 7, 8, 9, 10) --
# ported the same way as Group 1 above: each group's OWN cells in
# OPTI_TRACK_PROCESSING.ipynb were read in order (not assumed to share
# Group 1's structure). Confirmed per-group, cell-by-cell, against
# notebooks_reference/OPTI_TRACK_PROCESSING_CODE_ONLY.py:
#
#   Group 6  (CELL 22-26, #20-24), added 2026-09-11 -- previously,
#     wrongly, believed to have no OptiTrack recording at all (see the
#     module docstring's "Group 6 has no OptiTrack recording" claim,
#     now known false and left as a corrected historical note there).
#     Genuinely different from every other group in one respect: THREE
#     raw Motive take files exist locally
#     (data/raw/group_6/optitrack/Arda_Group-6_Take_{1,2,3}.csv), but
#     only 2 are actually used. CELL 22 "FILE INTEGRITY / TIMESTAMP
#     AUDIT" is diagnostic-only. CELL 23 "CHECK IF TAKE 3 IS DUPLICATE
#     PREFIX OF TAKE 2" empirically compares Take 3 row-for-row against
#     the first len(Take 3) rows of Take 2 (frame/time equality +
#     numeric max-abs-diff over every column) and finds them identical
#     (Take 3 is an exact duplicate prefix of Take 2, "same timing,
#     same values" -- the cell's own printed recommendation is "Take 3
#     is an exact duplicate prefix of Take 2. Exclude Take 3."). CELL 24
#     "START INSPECTION" and CELL 25 "MANUAL TRACKLET STITCHING" (single
#     pass, no update/revision cell -- unlike Groups 9/10) both then
#     explicitly restrict `TAKE_FILES`/`TAKE_PATHS` to Take 1 and Take 2
#     only, with an explicit comment ("Take 3 is excluded because it is
#     an exact duplicate prefix of Take 2."). CELL 26 "COMBINE TAKE 1-2"
#     concatenates just those two stitched takes with a literal
#     TAKE_OFFSETS_S (take_1=0.000, take_2=907.869, from the two takes'
#     real Motive "Capture Start Time" metadata: 2026-04-23 16:25:43.206
#     -> 16:40:51.075) and adds one optitrack_quality_note (take_2 only)
#     -- confirmed by reading CELL 26 directly, not assumed from any
#     other group's shape. So Group 6 combines Groups 2/3/5's "single
#     stitching cell, no update pass" shape with Groups 7/8's "has
#     optitrack_quality_note" shape -- a genuinely new combination, not
#     identical to any single prior group. Mechanically it still fits
#     `reconstruct_markers_multi_take()` exactly (2 takes instead of
#     4-6, same helpers, same offset/quality-note style) -- ported as
#     `reconstruct_markers_group6()` below, called with only
#     take_1/take_2 paths (the caller must NOT pass a take_3 path; doing
#     so would silently double-count Take 3's frames, since nothing in
#     `reconstruct_markers_multi_take()` itself knows to deduplicate
#     it).
#   Group 2  (CELL 11-13,  #11-13): single "MANUAL TRACKLET STITCHING"
#     cell (no update/gap-inspection revision) -> "COMBINE TAKE 1-5".
#     5 takes. No optitrack_quality_note column in the real output
#     (confirmed against the real fixture's header).
#   Group 3  (CELL 14-16,  #14-16): same shape as Group 2. 5 takes.
#     No optitrack_quality_note column.
#   Group 5  (CELL 18-20,  #17-19): same shape. 5 takes. No
#     optitrack_quality_note column.
#   Group 7  (CELL 28-31,  #25-28): "MANUAL TRACKLET STITCHING" -> "GAP
#     CANDIDATE INSPECTION FOR CURRENT STITCHING" (diagnostic only --
#     writes to a DIFFERENT OUT_DIR, optitrack_gap_candidate_inspection/,
#     never read by the combine cell's own IN_DIR, confirmed directly) ->
#     "COMBINE TAKE 1-4", which adds an `optitrack_quality_note` text
#     column (verified present in the real fixture's header). 4 takes.
#   Group 8  (CELL 33-36,  #29-32): same shape as Group 7 (one stitching
#     cell + one diagnostic-only gap-inspection cell) -> "COMBINE TAKE
#     1-5" with optitrack_quality_note (3 per-take notes). 5 takes.
#   Group 9  (CELL 38-43,  #33-38): "MANUAL TRACKLET STITCHING" (first
#     pass -- take_5 chains explicitly flagged "approximate" in the
#     cell's own comment, inspection output was truncated) -> two
#     diagnostic-only cells ("FOCUSED MARKER SUMMARY FOR TAKE 5 AND
#     TAKE 6", "FOCUSED GAP CANDIDATE INSPECTION") -> "MANUAL TRACKLET
#     STITCHING UPDATED", which writes to the SAME OUT_DIR as the
#     first-pass cell (overwriting it, confirmed by comparing both
#     cells' OUT_DIR strings) -- this revised CHAINS (take_2 landmark3
#     +Unlabeled 2532, take_5 landmark1 +Unlabeled 2908, take_6
#     unchanged) is what the combine cell actually consumes. -> "COMBINE
#     TAKE 1-6" with optitrack_quality_note (3 per-take notes). 6 takes.
#   Group 10 (CELL 45-50,  #39-44): same update pattern as Group 9
#     ("MANUAL TRACKLET STITCHING" -> diagnostic "FOCUSED GAP CANDIDATE
#     INSPECTION" -> "MANUAL TRACKLET STITCHING UPDATED", same OUT_DIR,
#     supersedes) -> "COMBINE TAKE 1-3" -- which then appears TWICE in
#     immediate succession (CELL 49 #43 and CELL 50 #44); diffed
#     directly line-by-line, confirmed byte-identical (same
#     TAKE_OFFSETS_S, same quality-note text, same out_path) -- an
#     accidental cell re-run left in the saved notebook, not a
#     divergent revision; harmless since it's a deterministic rerun.
#     optitrack_quality_note present (2 per-take notes). 3 takes.
#
# Genuinely shared mechanism across all 8 groups (verified by reading
# every cell, not assumed): each group's own
# `read_motive_long_and_frame_time`/`build_stitched_clean`/
# `smooth_short_gaps` are byte-identical to Group 1's (and to each
# other's -- checked directly, same defaults limit=10/window=5),
# applied per-take, then concatenated with a LITERAL per-take
# capture-start offset (`TAKE_OFFSETS_S`, hardcoded from each take's
# real Motive "Capture Start Time" metadata) and combined-file
# availability recomputed -- this offset style differs from Group 1's
# own "single TAKE2_OFFSET_S added to take_2 only" shape (Group 1 only
# has 2 takes; every other group's combine cell instead hardcodes one
# absolute offset per take, take_1's always 0.000). Only the per-take
# marker-name CHAINS (unique per group/take, hand-curated) and the
# offsets/quality-notes differ across groups -- ported below as one
# shared `reconstruct_markers_multi_take()` parameterized by those, with
# per-group config in `GROUP_RECONSTRUCTION_CONFIG` and thin per-group
# wrapper functions for the naming convention used elsewhere in this
# module.
# ================================================================

GROUP2_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1163"],
        "landmark2": ["Unlabeled 1164", "Unlabeled 1178"],
        "landmark3": ["Unlabeled 1165", "Unlabeled 1168", "Unlabeled 1169", "Unlabeled 1170",
                      "Unlabeled 1181", "Unlabeled 1182"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1259"],
        "landmark2": ["Unlabeled 1257"],
        "landmark3": ["Unlabeled 1256", "Unlabeled 1260", "Unlabeled 1265", "Unlabeled 1268"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 1307"],
        "landmark2": ["Unlabeled 1308", "Unlabeled 1312", "Unlabeled 1314"],
        "landmark3": ["Unlabeled 1309", "Unlabeled 1310", "Unlabeled 1311", "Unlabeled 1313"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 1353", "Unlabeled 1356", "Unlabeled 1361", "Unlabeled 1363"],
        "landmark2": ["Unlabeled 1354", "Unlabeled 1357", "Unlabeled 1362"],
        "landmark3": ["Unlabeled 1352", "Unlabeled 1358"],
    },
    "take_5": {
        "landmark1": ["Unlabeled 1386"],
        "landmark2": ["Unlabeled 1387"],
        "landmark3": ["Unlabeled 1388", "Unlabeled 1392"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 13):
# Take 1: 2026-04-21 14:18:05.206 / Take 2: 14:28:08.543 / Take 3: 14:38:16.803 /
# Take 4: 14:48:22.744 / Take 5: 14:58:28.830
GROUP2_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 603.337, "take_3": 1211.597,
                          "take_4": 1817.538, "take_5": 2423.624}

GROUP3_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1202"],
        "landmark2": ["Unlabeled 1201", "Unlabeled 1227"],
        "landmark3": ["Unlabeled 1203", "Unlabeled 1225", "Unlabeled 1228", "Unlabeled 1230"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1287"],
        "landmark2": ["Unlabeled 1288"],
        "landmark3": ["Unlabeled 1286", "Unlabeled 1308"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 1352"],
        "landmark2": ["Unlabeled 1351", "Unlabeled 1360", "Unlabeled 1363", "Unlabeled 1366",
                      "Unlabeled 1367", "Unlabeled 1369", "Unlabeled 1370"],
        "landmark3": ["Unlabeled 1353", "Unlabeled 1357"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 1460", "Unlabeled 1493"],
        "landmark2": ["Unlabeled 1462", "Unlabeled 1478", "Unlabeled 1487", "Unlabeled 1492",
                      "Unlabeled 1499"],
        "landmark3": ["Unlabeled 1461", "Unlabeled 1466", "Unlabeled 1468", "Unlabeled 1476",
                      "Unlabeled 1485", "Unlabeled 1486"],
    },
    "take_5": {
        "landmark1": ["Unlabeled 1520"],
        "landmark2": ["Unlabeled 1521"],
        "landmark3": ["Unlabeled 1522"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 16):
# Take 1: 2026-04-22 14:33:14.209 / Take 2: 14:43:30.161 / Take 3: 14:53:39.149 /
# Take 4: 15:03:56.360 / Take 5: 15:16:01.878
GROUP3_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 615.952, "take_3": 1224.940,
                          "take_4": 1842.151, "take_5": 2567.669}

GROUP5_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1317"],
        "landmark2": ["Unlabeled 1318", "Unlabeled 1322", "Unlabeled 1339", "Unlabeled 1340",
                      "Unlabeled 1341", "Unlabeled 1344", "Unlabeled 1345", "Unlabeled 1348",
                      "Unlabeled 1350", "Unlabeled 1351"],
        "landmark3": ["Unlabeled 1319", "Unlabeled 1320", "Unlabeled 1323", "Unlabeled 1327",
                      "Unlabeled 1329", "Unlabeled 1331", "Unlabeled 1338"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1412"],
        "landmark2": ["Unlabeled 1413"],
        "landmark3": ["Unlabeled 1411", "Unlabeled 1414", "Unlabeled 1415", "Unlabeled 1416",
                      "Unlabeled 1417", "Unlabeled 1418", "Unlabeled 1419", "Unlabeled 1420",
                      "Unlabeled 1425", "Unlabeled 1426", "Unlabeled 1427", "Unlabeled 1428",
                      "Unlabeled 1429", "Unlabeled 1430", "Unlabeled 1431", "Unlabeled 1433",
                      "Unlabeled 1434", "Unlabeled 1435", "Unlabeled 1438", "Unlabeled 1439"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 1522"],
        "landmark2": ["Unlabeled 1523", "Unlabeled 1551"],
        "landmark3": ["Unlabeled 1524", "Unlabeled 1527", "Unlabeled 1530", "Unlabeled 1533",
                      "Unlabeled 1534", "Unlabeled 1535", "Unlabeled 1536", "Unlabeled 1537",
                      "Unlabeled 1538", "Unlabeled 1539", "Unlabeled 1540", "Unlabeled 1541",
                      "Unlabeled 1542", "Unlabeled 1549", "Unlabeled 1554", "Unlabeled 1556"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 1644", "Unlabeled 1668", "Unlabeled 1672", "Unlabeled 1677",
                      "Unlabeled 1678"],
        "landmark2": ["Unlabeled 1645", "Unlabeled 1647", "Unlabeled 1654", "Unlabeled 1667"],
        "landmark3": ["Unlabeled 1646", "Unlabeled 1657", "Unlabeled 1662", "Unlabeled 1666",
                      "Unlabeled 1674", "Unlabeled 1675", "Unlabeled 1680"],
    },
    "take_5": {
        "landmark1": ["Unlabeled 1751"],
        "landmark2": ["Unlabeled 1752"],
        "landmark3": ["Unlabeled 1754", "Unlabeled 1756"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 20):
# Take 1: 2026-04-23 13:20:45.737 / Take 2: 13:33:20.518 / Take 3: 13:43:29.474 /
# Take 4: 13:53:41.450 / Take 5: 14:07:08.483
GROUP5_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 754.781, "take_3": 1363.737,
                          "take_4": 1975.713, "take_5": 2782.746}

# Group 6: CELL 25 "MANUAL TRACKLET STITCHING" -- single pass, no update
# revision. Only take_1/take_2 (Take 3 excluded: CELL 23 empirically
# confirmed it is an exact duplicate prefix of Take 2 -- identical
# frame/time and numeric values over its full length -- so it carries
# no additional data and must NOT be passed to
# `reconstruct_markers_group6()`).
GROUP6_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1289", "Unlabeled 1348", "Unlabeled 1369"],
        "landmark2": ["Unlabeled 1288", "Unlabeled 1290", "Unlabeled 1305", "Unlabeled 1320",
                      "Unlabeled 1321", "Unlabeled 1338"],
        "landmark3": ["Unlabeled 1293", "Unlabeled 1303", "Unlabeled 1307", "Unlabeled 1334",
                      "Unlabeled 1344", "Unlabeled 1349", "Unlabeled 1360", "Unlabeled 1362",
                      "Unlabeled 1363", "Unlabeled 1374", "Unlabeled 1378", "Unlabeled 1385"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 2086", "Unlabeled 2108", "Unlabeled 2130", "Unlabeled 2135",
                      "Unlabeled 2137", "Unlabeled 2139", "Unlabeled 2144", "Unlabeled 2174"],
        "landmark2": ["Unlabeled 2084", "Unlabeled 2088", "Unlabeled 2090", "Unlabeled 2093",
                      "Unlabeled 2096", "Unlabeled 2098", "Unlabeled 2103", "Unlabeled 2109",
                      "Unlabeled 2111", "Unlabeled 2115", "Unlabeled 2118", "Unlabeled 2120",
                      "Unlabeled 2127"],
        "landmark3": ["Unlabeled 2085", "Unlabeled 2100", "Unlabeled 2106", "Unlabeled 2110",
                      "Unlabeled 2116", "Unlabeled 2117", "Unlabeled 2123", "Unlabeled 2126",
                      "Unlabeled 2148", "Unlabeled 2177"],
    },
}
# Real offset from Take 1 capture start (verbatim comment, CELL 26):
# Take 1: 2026-04-23 16:25:43.206 / Take 2: 16:40:51.075 -> 907.869s.
GROUP6_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 907.869}
GROUP6_QUALITY_NOTES = {
    "take_2": "Take 2 has weak Landmark 3 availability; large missing interval not force-filled.",
}

GROUP7_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1216"],
        "landmark2": ["Unlabeled 1218"],
        "landmark3": ["Unlabeled 1220", "Unlabeled 1221", "Unlabeled 1226"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1406"],
        "landmark2": ["Unlabeled 1404", "Unlabeled 1408", "Unlabeled 1414", "Unlabeled 1416",
                      "Unlabeled 1433"],
        "landmark3": ["Unlabeled 1405", "Unlabeled 1411", "Unlabeled 1412", "Unlabeled 1417",
                      "Unlabeled 1425", "Unlabeled 1430"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 1483", "Unlabeled 1494", "Unlabeled 1496"],
        "landmark2": ["Unlabeled 1484", "Unlabeled 1491", "Unlabeled 1499"],
        "landmark3": ["Unlabeled 1482", "Unlabeled 1492", "Unlabeled 1495", "Unlabeled 1497",
                      "Unlabeled 1502", "Unlabeled 1503"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 1289", "Unlabeled 1303", "Unlabeled 1304", "Unlabeled 1307",
                      "Unlabeled 1308", "Unlabeled 1320", "Unlabeled 1333", "Unlabeled 1337",
                      "Unlabeled 1338"],
        "landmark2": ["Unlabeled 1291", "Unlabeled 1293", "Unlabeled 1298", "Unlabeled 1315",
                      "Unlabeled 1324", "Unlabeled 1328", "Unlabeled 1329"],
        "landmark3": ["Unlabeled 1292", "Unlabeled 1294", "Unlabeled 1295", "Unlabeled 1301",
                      "Unlabeled 1302", "Unlabeled 1314", "Unlabeled 1316", "Unlabeled 1334",
                      "Unlabeled 1339", "Unlabeled 1342"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 31):
# Take 1: 2026-04-29 12:35:49.872 / Take 2: 12:46:00.195 / Take 3: 12:56:17.972 /
# Take 4: 13:07:50.337
GROUP7_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 610.323, "take_3": 1228.100,
                          "take_4": 1920.465}
GROUP7_QUALITY_NOTES = {
    "take_2": "Take 2 has weak Landmark 2 availability; large gap was not force-filled.",
}

GROUP8_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 1657"],
        "landmark2": ["Unlabeled 1658", "Unlabeled 1678", "Unlabeled 1692"],
        "landmark3": ["Unlabeled 1655", "Unlabeled 1662", "Unlabeled 1664", "Unlabeled 1665",
                      "Unlabeled 1672", "Unlabeled 1675", "Unlabeled 1676", "Unlabeled 1679",
                      "Unlabeled 1680", "Unlabeled 1681", "Unlabeled 1682", "Unlabeled 1684",
                      "Unlabeled 1685", "Unlabeled 1687", "Unlabeled 1688", "Unlabeled 1689",
                      "Unlabeled 1690", "Unlabeled 1691"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1782"],
        "landmark2": ["Unlabeled 1783", "Unlabeled 1790", "Unlabeled 1802", "Unlabeled 1823"],
        "landmark3": ["Unlabeled 1781", "Unlabeled 1784", "Unlabeled 1785", "Unlabeled 1786",
                      "Unlabeled 1787", "Unlabeled 1789", "Unlabeled 1791", "Unlabeled 1792",
                      "Unlabeled 1793", "Unlabeled 1796", "Unlabeled 1797", "Unlabeled 1798",
                      "Unlabeled 1801", "Unlabeled 1808", "Unlabeled 1809", "Unlabeled 1810",
                      "Unlabeled 1811", "Unlabeled 1813", "Unlabeled 1814", "Unlabeled 1815",
                      "Unlabeled 1817", "Unlabeled 1821", "Unlabeled 1822"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 1992", "Unlabeled 2015", "Unlabeled 2053"],
        "landmark2": ["Unlabeled 1991", "Unlabeled 1994", "Unlabeled 1995", "Unlabeled 2011",
                      "Unlabeled 2043", "Unlabeled 2062", "Unlabeled 2063", "Unlabeled 2074"],
        "landmark3": ["Unlabeled 1993", "Unlabeled 1996", "Unlabeled 1997", "Unlabeled 1998",
                      "Unlabeled 1999", "Unlabeled 2000", "Unlabeled 2004", "Unlabeled 2007",
                      "Unlabeled 2008", "Unlabeled 2020", "Unlabeled 2021", "Unlabeled 2022",
                      "Unlabeled 2023", "Unlabeled 2024", "Unlabeled 2025", "Unlabeled 2027",
                      "Unlabeled 2030", "Unlabeled 2031", "Unlabeled 2033", "Unlabeled 2034",
                      "Unlabeled 2035", "Unlabeled 2036", "Unlabeled 2037", "Unlabeled 2042",
                      "Unlabeled 2044", "Unlabeled 2045", "Unlabeled 2046", "Unlabeled 2047",
                      "Unlabeled 2048", "Unlabeled 2050", "Unlabeled 2051", "Unlabeled 2055",
                      "Unlabeled 2056", "Unlabeled 2060", "Unlabeled 2064", "Unlabeled 2068",
                      "Unlabeled 2075"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 2149"],
        "landmark2": ["Unlabeled 2142", "Unlabeled 2147", "Unlabeled 2153", "Unlabeled 2155",
                      "Unlabeled 2161", "Unlabeled 2162", "Unlabeled 2173", "Unlabeled 2174"],
        "landmark3": ["Unlabeled 2143", "Unlabeled 2144", "Unlabeled 2146", "Unlabeled 2148",
                      "Unlabeled 2150", "Unlabeled 2151", "Unlabeled 2158", "Unlabeled 2160"],
    },
    "take_5": {
        "landmark1": ["Unlabeled 2186"],
        "landmark2": ["Unlabeled 2187", "Unlabeled 2189"],
        "landmark3": ["Unlabeled 2185", "Unlabeled 2188"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 36):
# Take 1: 2026-04-30 13:45:54.738 / Take 2: 13:56:41.954 / Take 3: 14:06:57.794 /
# Take 4: 14:20:35.288 / Take 5: 14:28:25.432
GROUP8_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 647.216, "take_3": 1263.056,
                          "take_4": 2080.550, "take_5": 2550.694}
GROUP8_QUALITY_NOTES = {
    "take_2": "Take 2 has moderate Landmark 3 availability; weak gaps were not force-filled.",
    "take_3": "Take 3 has weak Landmark 2 availability; long gap was not force-filled.",
    "take_4": "Take 4 has moderate Landmark 1 and Landmark 3 availability; risky candidates were not added.",
}

# Group 9: CELL 42 ("UPDATED") CHAINS -- supersedes CELL 39's first pass
# (see docstring above). take_5 gains Unlabeled 2908 on landmark1; take_2
# gains Unlabeled 2532 on landmark3; take_6 unchanged.
GROUP9_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 2310", "Unlabeled 2331"],
        "landmark2": ["Unlabeled 2311", "Unlabeled 2317", "Unlabeled 2334", "Unlabeled 2338",
                      "Unlabeled 2343", "Unlabeled 2345", "Unlabeled 2358", "Unlabeled 2373"],
        "landmark3": ["Unlabeled 2335", "Unlabeled 2344", "Unlabeled 2354", "Unlabeled 2359",
                      "Unlabeled 2371", "Unlabeled 2372"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 2479"],
        "landmark2": ["Unlabeled 2481", "Unlabeled 2482", "Unlabeled 2493", "Unlabeled 2495",
                      "Unlabeled 2497", "Unlabeled 2503", "Unlabeled 2525"],
        "landmark3": ["Unlabeled 2485", "Unlabeled 2498", "Unlabeled 2501", "Unlabeled 2505",
                      "Unlabeled 2507", "Unlabeled 2508", "Unlabeled 2511", "Unlabeled 2512",
                      "Unlabeled 2513", "Unlabeled 2514", "Unlabeled 2516", "Unlabeled 2517",
                      "Unlabeled 2518", "Unlabeled 2519", "Unlabeled 2521", "Unlabeled 2524",
                      "Unlabeled 2528", "Unlabeled 2530", "Unlabeled 2532", "Unlabeled 2533"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 2605", "Unlabeled 2613", "Unlabeled 2635"],
        "landmark2": ["Unlabeled 2604"],
        "landmark3": ["Unlabeled 2610", "Unlabeled 2611", "Unlabeled 2612", "Unlabeled 2616",
                      "Unlabeled 2617", "Unlabeled 2618", "Unlabeled 2620", "Unlabeled 2621",
                      "Unlabeled 2623", "Unlabeled 2624", "Unlabeled 2625", "Unlabeled 2626",
                      "Unlabeled 2627", "Unlabeled 2630"],
    },
    "take_4": {
        "landmark1": ["Unlabeled 2732", "Unlabeled 2759", "Unlabeled 2768", "Unlabeled 2771"],
        "landmark2": ["Unlabeled 2731", "Unlabeled 2763", "Unlabeled 2767", "Unlabeled 2770"],
        "landmark3": ["Unlabeled 2733", "Unlabeled 2743", "Unlabeled 2744", "Unlabeled 2747",
                      "Unlabeled 2752", "Unlabeled 2754", "Unlabeled 2758", "Unlabeled 2769",
                      "Unlabeled 2774"],
    },
    "take_5": {
        "landmark1": ["Unlabeled 2904", "Unlabeled 2908", "Unlabeled 2910", "Unlabeled 2920",
                      "Unlabeled 2925", "Unlabeled 2929"],
        "landmark2": ["Unlabeled 2903", "Unlabeled 2916", "Unlabeled 2922", "Unlabeled 2924"],
        "landmark3": ["Unlabeled 2905", "Unlabeled 2906", "Unlabeled 2907", "Unlabeled 2912",
                      "Unlabeled 2914", "Unlabeled 2915", "Unlabeled 2919"],
    },
    "take_6": {
        "landmark1": ["Unlabeled 3031"],
        "landmark2": ["Unlabeled 3032", "Unlabeled 3060", "Unlabeled 3062"],
        "landmark3": ["Unlabeled 3033", "Unlabeled 3061"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 43):
# Take 1: 2026-04-30 15:42:50.664 / Take 2: 15:53:22.735 / Take 3: 16:04:25.584 /
# Take 4: 16:15:00.610 / Take 5: 16:30:24.858 / Take 6: 16:41:25.186
GROUP9_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 632.071, "take_3": 1294.920,
                          "take_4": 1929.946, "take_5": 2854.194, "take_6": 3514.522}
GROUP9_QUALITY_NOTES = {
    "take_1": "Take 1 has weak Landmark 3 availability; large missing interval was not force-filled.",
    "take_2": "Take 2 has weak/moderate Landmark 3 availability; only safe short candidate was added.",
    "take_6": "Take 6 has weak Landmark 1 availability; no safe successor candidate was found.",
}

# Group 10: CELL 48 ("UPDATED") CHAINS -- supersedes CELL 46's first pass.
# take_1 landmark2 gains 3115/3119/3123/3124/3129; take_2 landmark1 gains
# 3187/3188; take_3 landmark1 unchanged (no safe candidate found).
GROUP10_CHAINS: dict[str, dict[str, list[str]]] = {
    "take_1": {
        "landmark1": ["Unlabeled 3113", "Unlabeled 3122", "Unlabeled 3128"],
        "landmark2": ["Unlabeled 3115", "Unlabeled 3119", "Unlabeled 3120", "Unlabeled 3123",
                      "Unlabeled 3124", "Unlabeled 3125", "Unlabeled 3126", "Unlabeled 3127",
                      "Unlabeled 3129", "Unlabeled 3130", "Unlabeled 3132", "Unlabeled 3136"],
        "landmark3": ["Unlabeled 3118", "Unlabeled 3131", "Unlabeled 3133", "Unlabeled 3134",
                      "Unlabeled 3135"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 3176", "Unlabeled 3186", "Unlabeled 3187", "Unlabeled 3188",
                      "Unlabeled 3193"],
        "landmark2": ["Unlabeled 3177", "Unlabeled 3180"],
        "landmark3": ["Unlabeled 3178", "Unlabeled 3183", "Unlabeled 3184", "Unlabeled 3185",
                      "Unlabeled 3190", "Unlabeled 3192"],
    },
    "take_3": {
        "landmark1": ["Unlabeled 3251", "Unlabeled 3256"],
        "landmark2": ["Unlabeled 3252", "Unlabeled 3254", "Unlabeled 3257", "Unlabeled 3258",
                      "Unlabeled 3260"],
        "landmark3": ["Unlabeled 3253", "Unlabeled 3259", "Unlabeled 3261", "Unlabeled 3266",
                      "Unlabeled 3267", "Unlabeled 3268"],
    },
}
# Real offsets from Take 1 capture start (verbatim comment, CELL 49/50 --
# byte-identical in both cells):
# Take 1: 2026-04-30 17:43:36.632 / Take 2: 17:53:46.260 / Take 3: 18:04:00.673
GROUP10_TAKE_OFFSETS_S = {"take_1": 0.000, "take_2": 609.628, "take_3": 1224.041}
GROUP10_QUALITY_NOTES = {
    "take_2": "Take 2 Landmark 1 was improved using short candidates; medium confidence around the stitched gap.",
    "take_3": "Take 3 has weak Landmark 1 availability; no safe continuation candidate was found.",
}


def reconstruct_markers_multi_take(take_paths: dict[str, str], chains: dict[str, dict[str, list[str]]],
                                    take_offsets_s: dict[str, float],
                                    quality_notes: dict[str, str] | None = None) -> pd.DataFrame:
    """Shared reconstruction mechanism for every group EXCEPT Group 1
    (whose combine step instead adds one offset to a second take only,
    see `reconstruct_markers_group1`): for each take in `take_paths`
    (processed in dict order, matching each group's own `TAKE_PATHS`),
    reads the raw long-format Motive export, stitches Landmark1/2/3 from
    `chains[take_name]`, smooths short gaps, then concatenates every
    take with its own literal `take_offsets_s[take_name]` added to
    time_s (real per-take Motive capture-start offsets, take_1 always
    0.000) and re-sorts by time_s, recomputing per-row landmark
    availability on the combined frame. If `quality_notes` is given, adds
    an `optitrack_quality_note` text column (empty string by default,
    the given note text on rows whose `take` is a key of the dict) --
    verbatim from each group's own "COMBINE TAKE" cell; omitted
    entirely for groups whose real output has no such column (2, 3, 5)."""
    stitched = {}
    for take_name, path in take_paths.items():
        _, frame_time, long_df = read_motive_long_and_frame_time(path)
        clean = build_stitched_clean(frame_time, long_df, chains[take_name])
        stitched[take_name] = smooth_short_gaps(clean)

    parts = []
    for take_name, df in stitched.items():
        df = df.copy()
        df["take"] = take_name
        df["time_s_original"] = df["time_s"]
        df["time_s"] = df["time_s_original"] + take_offsets_s[take_name]
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("time_s").reset_index(drop=True)

    for i in (1, 2, 3):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)
    combined["active_clean_landmarks"] = combined[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    if quality_notes is not None:
        combined["optitrack_quality_note"] = ""
        for take_name, note in quality_notes.items():
            combined.loc[combined["take"] == take_name, "optitrack_quality_note"] = note

    return combined


def reconstruct_markers_group2(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 2, 5 takes. Verbatim from CELL 12 + CELL 13."""
    return reconstruct_markers_multi_take(take_paths, GROUP2_CHAINS, GROUP2_TAKE_OFFSETS_S)


def reconstruct_markers_group3(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 3, 5 takes. Verbatim from CELL 15 + CELL 16."""
    return reconstruct_markers_multi_take(take_paths, GROUP3_CHAINS, GROUP3_TAKE_OFFSETS_S)


def reconstruct_markers_group5(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 5, 5 takes. Verbatim from CELL 19 + CELL 20."""
    return reconstruct_markers_multi_take(take_paths, GROUP5_CHAINS, GROUP5_TAKE_OFFSETS_S)


def reconstruct_markers_group6(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 6, 2 takes (take_1, take_2 only). Verbatim from CELL 25 +
    CELL 26. Take 3's raw file exists on disk
    (data/raw/group_6/optitrack/Arda_Group-6_Take_3.csv) but is NOT part
    of the reconstruction: CELL 23 empirically confirmed it is an exact
    duplicate prefix of Take 2 (identical frame/time and numeric values
    over its full length), and both CELL 24 and CELL 25 explicitly
    exclude it. Callers must pass only take_1/take_2 paths -- passing a
    take_3 entry would not reproduce the source notebook's output (it
    never ran Take 3 through the stitcher at all)."""
    return reconstruct_markers_multi_take(take_paths, GROUP6_CHAINS, GROUP6_TAKE_OFFSETS_S,
                                           GROUP6_QUALITY_NOTES)


def reconstruct_markers_group7(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 7, 4 takes. Verbatim from CELL 29 + CELL 31 (CELL 30's
    "GAP CANDIDATE INSPECTION" is diagnostic-only, confirmed not read by
    CELL 31's IN_DIR)."""
    return reconstruct_markers_multi_take(take_paths, GROUP7_CHAINS, GROUP7_TAKE_OFFSETS_S,
                                           GROUP7_QUALITY_NOTES)


def reconstruct_markers_group8(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 8, 5 takes. Verbatim from CELL 34 + CELL 36 (CELL 35's gap
    inspection is diagnostic-only, same pattern as Group 7)."""
    return reconstruct_markers_multi_take(take_paths, GROUP8_CHAINS, GROUP8_TAKE_OFFSETS_S,
                                           GROUP8_QUALITY_NOTES)


def reconstruct_markers_group9(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 9, 6 takes. Verbatim from CELL 42 (the "UPDATED" chains
    that supersede CELL 39's first pass, same OUT_DIR) + CELL 43."""
    return reconstruct_markers_multi_take(take_paths, GROUP9_CHAINS, GROUP9_TAKE_OFFSETS_S,
                                           GROUP9_QUALITY_NOTES)


def reconstruct_markers_group10(take_paths: dict[str, str]) -> pd.DataFrame:
    """Group 10, 3 takes. Verbatim from CELL 48 (the "UPDATED" chains
    that supersede CELL 46's first pass, same OUT_DIR) + CELL 49/50
    (byte-identical duplicate combine cell, diffed directly)."""
    return reconstruct_markers_multi_take(take_paths, GROUP10_CHAINS, GROUP10_TAKE_OFFSETS_S,
                                           GROUP10_QUALITY_NOTES)


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

    # Group 9: two-point linear drift correction. OPTI_TRACK_PROCESSING.ipynb
    # cell 61 ("GROUP 9 - FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT
    # SYNC")'s own find_sync_rows does NOT restrict by tier at all -- it
    # takes the globally-earliest and globally-latest sync-labeled rows
    # across every tier. Group 9's real ELAN file has exactly 2 sync rows:
    # the first on Whole_Group (12.180-15.300s), the last on Participant1
    # (3594.727-3598.364s) -- see cell 59's own comment ("Beginning sync:
    # Whole_Group ... / End sync: Participant1 ..."), confirmed empirically
    # against Group_9_individual_build_renamed.csv. This module's
    # find_sync_rows *prefers* anchor_tier when it has >=1 match instead of
    # only when it has all matches, so the dataclass default
    # anchor_tier="Whole_Group" would wrongly restrict to the single
    # Whole_Group row and then raise (two_point needs >=2 rows). Setting
    # anchor_tier to a tier with zero sync-labeled rows in Group 9's real
    # data forces the intended fallback-to-all-candidates branch, exactly
    # reproducing the notebook's tier-agnostic global first/last selection.
    9: OptitrackSyncConfig(group=9, method="two_point", anchor_tier="Participant2",
                            first_sync_time=23.141667, last_sync_time=3606.117833),

    # Group 8: two-point linear. OPTI_TRACK_PROCESSING_ANALYSIS.md's per-group
    # table (CELL 63-65 / #54-56, "GROUP 8") gives OPTITRACK_FIRST_SYNC_TIME=
    # 111.125 / OPTITRACK_LAST_SYNC_TIME=2312.608333 verbatim -- matches
    # below exactly, no change needed. Empirically verified against the real
    # Group_8_individual_build_renamed.csv (RAW_VALIDATION session, 2026-09-07):
    # exactly 2 sync-like rows exist, BOTH on tier Whole_Group
    # (123.381-127.476s and 2325.546-2328.636s) -- so unlike Group 9, no
    # anchor_tier override is needed; the dataclass default
    # anchor_tier="Whole_Group" already selects both rows directly (same
    # situation as Group 7, see below).
    8: OptitrackSyncConfig(group=8, method="two_point",
                            first_sync_time=111.125, last_sync_time=2312.608333),

    # Group 7: two-point linear. OPTI_TRACK_PROCESSING_ANALYSIS.md's per-group
    # table (CELL 67-69 / #57-59) gives OPTITRACK_FIRST_SYNC_TIME=71.119 /
    # OPTITRACK_LAST_SYNC_TIME=2977.631667 verbatim -- matches below exactly.
    # Empirically verified against the real Group_7_individual_build_renamed
    # .csv (RAW_VALIDATION session, 2026-09-07): exactly 2 sync-like rows
    # exist, BOTH on tier Whole_Group (69.430-72.808s and
    # 2974.818-2979.273s) -- no anchor_tier override needed, unlike Group 9.
    7: OptitrackSyncConfig(group=7, method="two_point",
                            first_sync_time=71.119, last_sync_time=2977.631667),

    # Group 6: two-point linear. OPTI_TRACK_PROCESSING_ANALYSIS.md's per-group
    # table (CELL 71-73 / #60-62) gives OPTITRACK_FIRST_SYNC_TIME=32.108333 /
    # OPTITRACK_LAST_SYNC_TIME=1367.339833 verbatim -- matches below exactly,
    # re-verified directly against CELL 73's own literal assignment (2026-09-11).
    # Empirically re-verified against the real Group_6_individual_build_renamed
    # .csv: exactly 2 sync-like rows exist, BOTH on tier Whole_Group
    # (29.988-31.926s, midpoint 30.957, matching CELL 72's own
    # ELAN_FIRST_SYNC_MID=30.957; 1367.455-1371.091s, midpoint 1369.273,
    # matching CELL 72's own ELAN_LAST_SYNC_MID=1369.273) -- no anchor_tier
    # override needed, same situation as Groups 7/8 (unlike Group 9).
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
