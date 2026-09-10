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
variant for — as of 2026-09-05 that's groups 5 (xsens only), 6 (xsens,
cleaned+shifted), 7, 8, 9, 10 (both sensors). Every other
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

Column-naming note (fixed 2026-09-04, verified against real Group 1 raw
data + FINAL_ARDA_THESIS.ipynb directly): the raw OE per-stream files are
headerless — column 0 is `timestamp_us` (absolute Unix epoch microseconds,
int64), followed by the stream's named value columns, followed by a
trailing all-zero flag column that gets dropped (see `OE_SCHEMA`,
`load_sensor_stream`). There is no `time_s` column on the raw files at all
— an earlier version of this module assumed one and failed immediately
against real data. Xsens raw per-participant files are headered and use
`SampleTimeFine` (a device-relative tick counter, no shared epoch across
devices/participants) as their time column; the merged grid derives its
own device-relative `time_s` from it (see `merge_xsens_group`).

Video-relative time (`video_time_s`) is then derived from each sensor's
device-relative time via a per-group, hardcoded, human-confirmed constant
— `GROUP_SYNC_CONFIG[g].video_start_us` for OE (an absolute epoch: the
moment that group's video recording started) and
`GROUP_SYNC_CONFIG[g].xsens_to_video_offset_s` for Xsens (a simple additive
offset in seconds) — both read verbatim from the notebook's own per-group
labeling cells (CELL 8 for Group 1 OE, CELL 9 for Group 1 Xsens, etc.),
re-verified directly against the notebook by grepping every
`VIDEO_START_US =`/`XSENS_TO_VIDEO_OFFSET_S =` literal across all 9 groups.
This conversion is a separate, EARLIER step than the peak-search shift
(`compute_peak_shift`/`GROUP_SYNC_CONFIG`'s search-window fields) — the
peak search always operates on the already-video-aligned `video_time_s`
axis (confirmed by directly reading Group 1, 3, and 7's OE+Xsens labeling
AND shift cells: CELLs 8/9/17 for Group 1, 34/35/39 for Group 3, 76/79 for
Group 7 — Group 7's CELL 79 in particular computes its shift delta purely
from `video_time_s`, identically for OE and Xsens, with no extra device
offset added on top). Because of this, the old `device_start_offset_s`
field (an ad-hoc "+correction inside compute_peak_shift", the previous
port's attempt to compensate for the *absent* video_time_s conversion) is
now redundant and has been removed — see `compute_peak_shift`'s docstring.
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

# OpenEarable raw per-stream value-column schema — verbatim `SCHEMA` from
# FINAL_ARDA_THESIS.ipynb CELL 3 (and identical in every other group's
# OE-merge cell). NOTE the "mgnt" stream's raw filename stem does NOT match
# its value-column names: they're "mag_x"/"mag_y"/"mag_z", not "mgnt_*" —
# this was a bug in the previous port (which derived column names from the
# stream/filename stem) as well as the missing timestamp_us fix.
OE_SCHEMA: dict[str, list[str]] = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}
OE_STREAMS = list(OE_SCHEMA)  # raw filename stems, in the notebook's own column order
OE_WINDOW_STREAMS = ("acc", "gyro", "mgnt")  # session-window anchor trio (clips e.g. bone_acc warm-up)
OE_SLOW_STREAMS = {"skin_temp"}  # wider tolerance stream, ~32Hz native rate

# OpenEarable merge grid — identical across every group's OE-merge cell (verbatim constants).
OE_GRID_US = 20_000       # 50 Hz target grid
OE_TOL_FAST_US = 12_000   # match tolerance for 50Hz streams
OE_TOL_SLOW_US = 40_000   # wider tolerance for the ~32Hz skin_temp

# Xsens raw per-participant value-column schema — verbatim `SENSOR_COLS`
# from FINAL_ARDA_THESIS.ipynb CELL 5. Merged/output column names are these,
# lower-cased (e.g. "Acc_X" -> "p1_acc_x").
XSENS_SENSOR_COLS = ["Euler_X", "Euler_Y", "Euler_Z", "Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
XSENS_HZ = 30  # native rate; master grid = Participant 1's clean SampleTimeFine ticks intersected with the other 2

# Group 3's two-part video concatenation offset: real wall-clock gap between
# PART1_START_CLOCK="12:21:50" and PART2_START_CLOCK="12:46:56" (no participant names involved).
GROUP3_PART2_OFFSET_S = 1506.0

# Both sensors' merged acc columns are lower_case ("p{n}_acc_x") — Xsens's
# raw "Acc_X" etc. get lower-cased by merge_xsens_group (verbatim CELL 5:
# `.rename(columns={c: f"{{pre}}_{{c.lower()}}" ...})`). An earlier version
# of this constant assumed xsens kept "p{n}_Acc_X" (mixed-case) column
# names, which never matched merge_xsens_group's actual output — fixed
# alongside the video_time_s bug (2026-09-04), since it silently broke
# compute_peak_shift's xsens signal lookup (combined_acc_signal found no
# matching columns -> all-NaN signal -> "no finite signal samples").
ACC_PREFIX_TEMPLATE = {"openearable": "p{n}_acc_", "xsens": "p{n}_acc_"}
ACC_AXIS_SUFFIXES = {"openearable": ("x", "y", "z"), "xsens": ("x", "y", "z")}


# ================================================================
# Per-group sync configuration (verbatim constants from
# FINAL_ARDA_THESIS_ANALYSIS.md section 1's per-group table)
# ================================================================

@dataclass
class OeXsensSyncConfig:
    group: int
    # Video-time alignment (a separate, EARLIER step than the peak-search
    # shift below — see module docstring). video_start_us is an absolute
    # Unix-epoch-microseconds constant (when THIS group's video recording
    # started); xsens_to_video_offset_s is a simple additive offset in
    # seconds. Both read verbatim from the notebook's per-group labeling
    # cells (grepped `VIDEO_START_US =`/`XSENS_TO_VIDEO_OFFSET_S =` across
    # all 9 groups' cells, 2026-09-04) — there is no algorithmic way to
    # derive either; they come from a human noting the video's real
    # wall-clock start time / the Xsens-vs-video start gap.
    video_start_us: int = 0
    xsens_to_video_offset_s: float = 0.0
    sync_labels: tuple[str, ...] = ()
    match_mode: str = "exact"                       # "exact" | "substring"
    occurrence: str = "longest"                      # "longest" | "first" | "last"
    expected_mid_range: tuple[float, float] | None = None
    # Group 9 only (see compute_peak_shift(), re-diagnosed 2026-09-07 against
    # the real downloaded raw files): CELL 118 ("GROUP 9 - OPTIONAL SHIFT
    # USING EARLY COMMON PEAK", notebooks_reference/FINAL_ARDA_THESIS_CODE_ONLY.py)
    # does NOT derive SYNC_MID from find_label_segments() over the already-
    # labeled grid at all -- it hardcodes it as a literal Python constant,
    # SYNC_MID = (12.180 + 15.300) / 2 = 13.740, straight from the official
    # ELAN Whole_Group `synchronization_move` row. This matters because, for
    # the real Group 9 raw files, EVERY OpenEarable stream (all 3
    # participants) and the Xsens continuous grid (offset 29.0s) only start
    # recording ~29-36s into video_time -- i.e. strictly AFTER the official
    # sync segment's own end (15.300s) -- so that segment can never actually
    # appear in label_Whole_Group on either sensor's merged grid, and the
    # normal find_label_segments()-based lookup in compute_peak_shift() (the
    # path every other group, including the grid-repaint half of Group 9's
    # own shift, correctly uses) raises "no sync-label segment found"
    # (confirmed empirically: video_time_s min is 32.805s for the P1/P2-
    # anchored OE grid, 29.0s for the P2/P3-anchored Xsens continuous grid).
    # None (default) preserves the existing grid-based lookup for every other
    # group. A float here makes compute_peak_shift() skip that lookup
    # entirely and use this value as sync_mid directly (sync_start=sync_end=
    # this value too, which is harmless for Group 9 since its search_mode is
    # "absolute" and never reads sync_start/sync_end/sync_mid via
    # resolve_search_window). The shift's OTHER half -- shift_labeled_frame()
    # re-painting label_* columns from the grid's own (start,end) segment
    # boundaries -- is unaffected and stays grid-based, matching CELL 118's
    # own shift_label_column()/shift_all_labels() verbatim.
    sync_mid_override: float | None = None
    search_mode: str = "relative"                     # "relative"|"absolute"|"manual_range"|"mid_window"|"last10pct"|"none"
    search_before_s: float = 15.0
    search_after_s: float = 5.0
    search_left_s: float | None = None
    search_right_s: float | None = None
    # Per-sensor overrides for search_before_s/search_after_s (used only by
    # "relative"/"mid_window" search modes). None (the default) means "use
    # the shared search_before_s/search_after_s above, identically for both
    # sensors" — this is what every group except Group 7 does. Group 7's own
    # CELL 79 (FINAL_ARDA_THESIS.ipynb, "GROUP 7 - SHIFT OPENEARAMBLE AND
    # XSENS LABELS USING SYNC PEAK") is the one confirmed case with distinct
    # per-sensor windows: OE before=5/after=25, Xsens before=5/after=30 (its
    # own CONFIG["oe"]/CONFIG["xsens"] dict, read verbatim). Added so that
    # case can be represented without changing behavior for any other group.
    search_before_s_openearable: float | None = None
    search_after_s_openearable: float | None = None
    search_before_s_xsens: float | None = None
    search_after_s_xsens: float | None = None
    peak_users: tuple[int, ...] = (1, 2, 3)
    smooth_window: int = 25
    # Per-sensor smooth_window overrides (Groups 9/10 only — both groups'
    # shift cells apply a DIFFERENT rolling-mean smoothing window per sensor
    # before peak-finding: SMOOTH_OE=10, SMOOTH_XSENS=5, verbatim from CELL
    # 118 "GROUP 9 - OPTIONAL SHIFT USING EARLY COMMON PEAK" and CELL 132
    # "GROUP 10 - FINAL SYNC SHIFT USING LATE EVENT ONLY", re-read 2026-09-06).
    # None (default) means "use the shared smooth_window above for both
    # sensors" -- every other group's existing behavior, unchanged.
    smooth_window_openearable: int | None = None
    smooth_window_xsens: int | None = None
    apply_shift_openearable: bool = False              # whether run_all() writes the shifted OE file
    apply_shift_xsens: bool = False                    # whether run_all() writes the shifted Xsens file
    clean_xsens_artifacts: bool = False                # Groups 6 and 10
    # Which artifact-cleaning algorithm clean_xsens_artifacts=True dispatches
    # to (see clean_extreme_artifacts()/clean_extreme_artifacts_group10()
    # below) -- "group6" (default, unchanged behavior) or "group10".
    xsens_cleaning_style: str = "group6"
    # Group 6: the cleaned frame IS the plain/unshifted selected file
    # (UNSHIFTED_FILENAMES[(6,"xsens")]="...cleaned.csv") -- cleaning output
    # is written as the plain file and then shifted from there. Group 10:
    # verbatim from CELL 131/132, re-read 2026-09-06 -- cleaning happens as a
    # SEPARATE later step that only ever feeds the peak-shift/SHIFTED output;
    # the plain/unshifted selected file (group_10_xsens_labeled.csv) stays
    # UNCLEANED. False (Group 10) means "clean only for shift/peak-search,
    # write the plain file uncleaned"; True (Group 6, default) means "clean
    # writes the plain file too".
    xsens_cleaning_writes_plain: bool = True
    # Group 7 only (see merge_openearable_group()/build_oe_merged_grid()):
    # Participant3's OE recording stopped early (~744s vs P1/P2's ~2988-3001s).
    # FINAL_ARDA_THESIS.ipynb's "GROUP 7 OPENEARAMBLE WIDE MERGE" cell (notebook
    # code-cell index 71, re-read verbatim 2026-09-06) anchors the shared
    # 50Hz grid's start/end to ONLY `ANCHOR_PARTICIPANTS = ["Participant1",
    # "Participant2"]`, not the 3-way window intersection every other group
    # uses (which would truncate the grid to P3's short ~744s window and never
    # reach the real sync event at ~2974.8-2979.3s video-time). All 3
    # participants are still merge_asof'd onto that P1/P2-anchored grid --
    # P3 simply goes NaN past its own recording's end, exactly like every
    # other stream tolerance-miss. None (default) means "anchor to every
    # participant present", i.e. every other group's existing behavior,
    # unchanged.
    oe_anchor_participants: tuple[int, ...] | None = None
    # Xsens merge algorithm dispatch (see build_xsens_merged_grid()):
    #  - "generic" (default): merge_xsens_group() -- exact-SampleTimeFine
    #    intersection join on P1's own ticks. Every group except 7/8/9/10.
    #  - "group7_wraparound": merge_xsens_group_group7() -- per-participant
    #    SampleTimeFine 32-bit wraparound fix (`(SampleTimeFine - first_tick)
    #    % 2**32`, THEN /1e6) + P1-anchored merge_asof(nearest, tolerance=
    #    0.02), with cell 72's own p3_ratio>=0.85 ALL3-vs-P1P2 branch. Real
    #    Group 7/8 Xsens files have wrapped/reset SampleTimeFine counters the
    #    generic exact-match join can't reconcile across participants.
    #    Verbatim from cell 72 ("GROUP 7 XSENS MERGE - AFTER UPDATED FILES")
    #    and Group 8's structurally-identical CELL 97 ("GROUP 8 XSENS MERGE -
    #    WRAP-SAFE VERSION", re-read verbatim 2026-09-06 -- Group 8's own cell
    #    hardcodes end=min(durations.values()) with no branch, which is what
    #    cell 72's branch reduces to whenever P3's ratio is >=0.85, i.e.
    #    whenever P3 hasn't dropped out early like Group 7's did).
    #  - "continuous_grid": merge_xsens_group_continuous_grid() -- a THIRD,
    #    genuinely different algorithm needed for Groups 9 and 10 (CELL 110
    #    "GROUP 9 XSENS MERGE - CONTINUOUS GRID VERSION" / CELL 123 "GROUP 10
    #    XSENS MERGE - CONTINUOUS P1/P3 ANCHOR VERSION", both re-read verbatim
    #    2026-09-06): a UNIFORM time_s grid (step 1/30 s, not P1's own real
    #    sample ticks) from 0 to `end` = min duration across
    #    `xsens_continuous_grid_anchor_participants` (Group 9: P2/P3 duration
    #    min; Group 10: P3's own duration alone), with every participant
    #    (wraparound-fixed) merge_asof'd (nearest, tolerance=0.02) onto that
    #    grid. This is the file each group's real downstream labeling cell
    #    actually reads (`..._merged_30hz_CONTINUOUS.csv`) -- the plain
    #    "..._merged_30hz.csv" that a naive port of CELL 108/122's own
    #    wrap-safe-but-non-continuous merge would produce is a dead-end
    #    intermediate the notebook computes but never labels from.
    xsens_merge_mode: str = "generic"
    # Only used when xsens_merge_mode == "continuous_grid" (see above).
    xsens_continuous_grid_anchor_participants: tuple[int, ...] = (1, 2, 3)
    shift_end_inclusive: bool = False                  # CELL 17 (Group 1) uses <= on the shifted upper
                                                         # bound; CELL 79 (Group 7) uses strict < — a real,
                                                         # confirmed inconsistency between the notebook's own
                                                         # shift cells, not a porting error. Defaults to the
                                                         # CELL 79 convention (matches attach_tier_label's own
                                                         # strict <); Group 1 overrides to True to match CELL 17.
    # Group 6 only (see compute_peak_shift_from_elan()/shift_labeled_frame_from_elan(),
    # dispatched from process_group()): FINAL_ARDA_THESIS.ipynb's Group 6 Xsens
    # shift cell (notebook code-cell index 68, "GROUP 6 XSENS SHIFT
    # CALCULATION", re-read verbatim 2026-09-06) does NOT use this module's
    # shared shift_labeled_frame()/shift_label_column() algorithm (re-derive
    # segment boundaries from the already-painted, grid-quantized label
    # column, then re-paint a shifted copy) the way Group 1's CELL 17 and
    # Group 7's CELL 79 do. Instead cell 68: (1) locates the sync segment's
    # begin_s/end_s/mid straight off the ORIGINAL ELAN table's own row (full
    # real-valued precision), not off the grid-quantized label column that
    # find_label_segments()/compute_peak_shift() use; (2) shifts the ELAN
    # table's begin_s/end_s by the computed offset; (3) drops every label_*
    # column and re-runs the exact same merge_asof(direction="backward")/
    # strict-< labeling logic used to build the original (unshifted) label
    # columns, now against the shifted ELAN times. Confirmed empirically
    # against the real Group 6 Xsens fixture (run_group6_validation.py):
    # shift_labeled_frame() (grid re-paint) leaves 3-53 mismatched rows per
    # tier out of 59,544; shifting elan_df's own begin_s/end_s and
    # re-running attach_all_labels() against it reproduces the fixture
    # EXACTLY (0/59,544 mismatches on all 7 label_* columns) -- the mismatch
    # was NOT in shift_labeled_frame()'s re-paint step itself but in BOTH
    # that AND compute_peak_shift()'s use of the grid-quantized sync
    # segment for sync_mid (off by up to one Xsens grid step, ~0.03s,
    # relative to the ELAN's true, real-valued mid) -- fixing only the
    # re-paint step (keeping grid-quantized sync_mid) still left 0-11
    # boundary-row mismatches per tier; both pieces together are needed.
    # False (default) keeps every other group on the generic
    # compute_peak_shift()/shift_labeled_frame() path, unchanged.
    shift_from_elan: bool = False


# Per-group video_start_us (OE) / xsens_to_video_offset_s (Xsens) — grepped
# verbatim from the notebook's own literal assignments (2026-09-04): every
# `VIDEO_START_US = ...` and `XSENS_TO_VIDEO_OFFSET_S = ...` /
# `XSENS_START_OFFSET_FROM_VIDEO_S = ...` cell across all 9 groups. Re-verified
# by full cell reads for groups 1 (CELLs 8-9), 3 (CELLs 34-35), and 7 (CELLs
# 76, 79) specifically.
_VIDEO_START_US = {
    1: 1776759287000000,   # CELL 8, literal — 2026-04-21 08:14:47 UTC
    2: 1776773065000000,   # CELL 24, literal
    3: int(pd.Timestamp("2026-04-22 12:21:50", tz="UTC").timestamp() * 1_000_000),   # CELL 34
    5: int(pd.Timestamp("2026-04-23 11:08:44", tz="UTC").timestamp() * 1_000_000),   # CELL 47
    6: int(pd.Timestamp("2026-04-23 14:17:43", tz="UTC").timestamp() * 1_000_000),   # CELL 63
    7: int(pd.Timestamp("2026-04-29 10:24:00", tz="UTC").timestamp() * 1_000_000),   # CELL 75
    8: int(pd.Timestamp("2026-04-30 11:35:00", tz="UTC").timestamp() * 1_000_000),   # CELL 85
    9: int(pd.Timestamp("2026-04-30 13:32:00", tz="UTC").timestamp() * 1_000_000),   # CELL 97
    10: int(pd.Timestamp("2026-04-30 15:33:00", tz="UTC").timestamp() * 1_000_000),  # CELL 109
}
_XSENS_TO_VIDEO_OFFSET_S = {
    1: -25.0,   # CELL 9 (XSENS_START_OFFSET_FROM_VIDEO_S) — Xsens started 25s BEFORE video
    2: 206.0,   # CELL 25
    3: 0.0,     # CELL 35 — Part 2 offset already folded into the concatenated ELAN
    5: 206.0,   # CELL 48
    6: -485.0,  # CELL 64
    7: 59.0,    # CELL 76
    8: -17.0,   # CELL 86
    9: 29.0,    # CELL 98
    10: 8.0,    # CELL 110
}

GROUP_SYNC_CONFIG: dict[int, OeXsensSyncConfig] = {
    # Group 1: last-10%-of-recording peak search; shift was computed by the
    # notebook but NOT adopted in its own final file selection -> unshifted.
    1: OeXsensSyncConfig(group=1, video_start_us=_VIDEO_START_US[1], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[1],
                          sync_labels=("synchronizaiton_move",), search_mode="last10pct", shift_end_inclusive=True),

    # Group 2: no shift cell exists at all in the source notebook.
    2: OeXsensSyncConfig(group=2, video_start_us=_VIDEO_START_US[2], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[2],
                          search_mode="none"),

    # Group 3: verbatim from FINAL_ARDA_THESIS.ipynb CELLs 40-41 (re-read
    # directly, 2026-09-05 -- the previous version of this entry both
    # mis-spelled sync_labels and guessed the search window from Group 5).
    # Two real, confirmed findings from that read:
    #   (1) Group 3's own sync label is spelled "synchronaztion_move" -- a
    #       DIFFERENT typo from every other group's "synchronizaiton_move"
    #       (CELL 37 even has an explicit comment "# exact spelling in Group
    #       3"; confirmed identically in CELLs 37/38/39/40/41). The old
    #       "synchronizaiton_move" literal here never matches Group 3's real
    #       ELAN text, so compute_peak_shift() always raised "no sync-label
    #       segment found" for this group -- harmless today (apply_shift_*
    #       stays False below, so it only ever populated shift_error, never
    #       blocked run_all()), but wrong; fixed to the real spelling.
    #   (2) CELL 40 ("GROUP 3 - SHIFT BOTH OPENEARAMBLE AND XSENS LABELS")
    #       uses SEARCH_BEFORE_SYNC_S=5, SEARCH_AFTER_SYNC_S=10 (not 15/5) --
    #       fixed to match. This is Step A of a genuinely bespoke, two-stage,
    #       per-sensor-asymmetric shift that this module's single
    #       compute_peak_shift()-per-sensor design does NOT reproduce, even
    #       with these corrected numbers (reported, not fixed -- a Group-7-
    #       style structural quirk, out of this task's fix scope):
    #         Step A (CELL 40): computes ONE offset from OpenEarable's OWN
    #         combined_acc_mag peak (search window -5s/+10s around the OE
    #         sync-label midpoint) and applies that SAME offset to BOTH
    #         sensors' labels -> group_3_{openearable,xsens}_labeled_
    #         shifted_by_sync_peak.csv.
    #         Step B (CELL 41, "SHIFT XSENS LABELS LEFT USING XSENS LOCAL
    #         PEAK"): a further, Xsens-ONLY refinement -- re-loads the
    #         ORIGINAL unshifted xsens_labeled.csv (discarding Step A's
    #         xsens output entirely), computes a SECOND, independent offset
    #         from Xsens's OWN acc peak (search window -12s/+4s), and writes
    #         a THIRD file, group_3_xsens_labeled_shifted_by_xsens_sync_
    #         peak.csv, which SHIFTED_FILENAMES below does not represent at
    #         all. OpenEarable never gets an Xsens-informed refinement.
    #       None of this affects the validated/adopted pipeline output:
    #       global_cleaning.py's SELECTED_FILE_NAMES picks the UNSHIFTED
    #       group_3_{openearable,xsens}_labeled.csv for both sensors (cells
    #       34/35's own direct output, before either shift cell runs), and
    #       apply_shift_openearable/apply_shift_xsens correctly stay False
    #       below -- confirmed against the real fixture in
    #       run_group3_validation.py.
    3: OeXsensSyncConfig(group=3, video_start_us=_VIDEO_START_US[3], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[3],
                          sync_labels=("synchronaztion_move",), search_mode="relative",
                          search_before_s=5.0, search_after_s=10.0),

    # Group 5: verbatim from FINAL_ARDA_THESIS.ipynb CELL 54 ("GROUP 5 - SHIFT
    # XSENS LABELS USING clap_synchronizaiton_move") and CELL 55 ("GROUP 5 -
    # SHIFT OPENEARAMBLE LABELS USING synchronizaiton_move"), re-read directly
    # 2026-09-05. Confirmed verbatim: SEARCH_BEFORE_SYNC_S=15/SEARCH_AFTER_SYNC_S=5
    # and PEAK_USER=3-only for Xsens (CELL 54); VIDEO_START_US/
    # XSENS_TO_VIDEO_OFFSET_S=206.0 (CELLs 47-48) both re-verified exact.
    #
    # Fixed here: apply_shift_xsens is now True (was False). The previous
    # comment ("computed but not adopted -> unshifted") was wrong for Xsens:
    # global_cleaning.py's SELECTED_FILE_NAMES[(5,"xsens")] selects
    # `group_5_xsens_labeled_shifted_by_clap_sync_peak.csv` (confirmed against
    # the real fixture on Drive, and against cell 3's own SELECTED_FILES dict)
    # -- i.e. the Xsens shift genuinely IS adopted downstream, so run_all()
    # must write that file or global_cleaning.py's selection is dangling.
    # apply_shift_openearable correctly stays False: OE's shift cell (CELL 55)
    # exists and computes an offset, but global_cleaning.py selects the plain
    # `group_5_openearable_labeled.csv` (unshifted) for OE -- matching this
    # session's task brief note "XSens sync checked. OE sync weaker."
    #
    # Known non-blocking discrepancy (reported, not fully fixed): CELL 54's
    # Xsens shift matches ONLY "clap_synchronizaiton_move"; CELL 55's OE shift
    # matches ONLY "synchronizaiton_move" -- two DIFFERENT single labels, one
    # per sensor. This module's `sync_labels` field is one shared tuple passed
    # to compute_peak_shift() for BOTH sensors (OR'd together, then
    # occurrence="longest" tie-breaks), not a per-sensor single label -- there
    # is no per-sensor sync_labels override field (unlike search_before_s/
    # search_after_s below, which Group 7 already needed). For Group 5's real
    # ELAN, "clap_synchronizaiton_move" (4.839s, begin_s=4066.419) is longer
    # than "synchronizaiton_move" (4.548s, begin_s=4085.484), so "longest"
    # happens to pick the CORRECT (clap) segment for Xsens by coincidence of
    # durations -- verified exact against the real fixture below. But the same
    # shared tuple+"longest" would ALSO pick the clap segment (not
    # "synchronizaiton_move") for OE's compute_peak_shift(), which is WRONG
    # per CELL 55 -- harmless today only because apply_shift_openearable=False
    # (OE's shift_info is computed but never written to a selected file), the
    # same kind of latent bug as Group 3's redundant/unused shift logic. Not
    # fixed: adding a per-sensor sync_labels field would be a structural
    # change beyond this task's "constant correction" scope.
    #
    # search_after_s_openearable=15.0 (below) IS a genuine constant fix: CELL
    # 55 uses SEARCH_BEFORE_SYNC_S=15/SEARCH_AFTER_SYNC_S=15 for OE, not the
    # shared 15/5 that CELL 54 uses for Xsens -- confirmed by direct read.
    # Wired in via the existing per-sensor override fields (same mechanism
    # Group 7 already uses) since apply_shift_openearable=False means this
    # only affects the reported (unused) OE shift_info, never a written file.
    #
    # smooth_window=1 (FIXED 2026-09-06; was left at the dataclass default of
    # 25, a real bug): re-read CELL 54 (Xsens) and CELL 55 (OE) verbatim end
    # to end -- neither applies ANY rolling-mean smoothing before peak search.
    # CELL 54's `find_xsens_peak_near_clap()` calls `signal =
    # acc_mag_xsens(df, PEAK_USER)` (raw per-sample sqrt(x^2+y^2+z^2)) directly
    # into `np.nanargmax(signal[mask])`; CELL 55's `find_oe_peak_near_sync()`
    # does the same with `combined_acc_mag_oe(df)` (`np.mean(...)`, no
    # `.rolling()` anywhere in either cell). This module's compute_peak_shift()
    # unconditionally calls `smooth_signal(signal, cfg.smooth_window)` before
    # `find_peak_time()`, so leaving smooth_window at its default of 25 (a
    # visualization-only constant elsewhere in the notebook, same trap
    # documented on Group 7's own smooth_window=1 override above) silently
    # smoothed a signal CELL 54/55 never smooth. Confirmed against the real
    # Group 5 fixture: raw-signal peak_time=4062.8919s (offset_s=-5.9334s,
    # reproduces `group_5_xsens_labeled_shifted_by_clap_sync_peak.csv` exactly)
    # vs smooth_window=25's peak_time=4062.5919s (offset_s=-6.2335s, a 0.3s
    # error) -- small in isolation, but large enough to misclassify ~1-2% of
    # rows (up to 1201/121045 on label_Participant2) wherever the shifted
    # timeline crosses one of the many short, closely-spaced individual_build/
    # real-annotation segment boundaries in Group 5's first ~50 minutes. This
    # was the actual, sole cause of this session's task-brief-reported "Group
    # 5 individual_build cutoffs" gap -- see the session log for why that
    # original diagnosis (a missing `cutoffs_s` dict) does not hold: Group 5's
    # own `elan/Group_5.csv` RAW_VALIDATION fixture is verified byte-identical
    # to `data/external/thesis_data/ELAN_RENAMED/
    # Group_5_individual_build_renamed.csv` (`diff <(sort ...) <(sort ...)`
    # -> empty), i.e. it is CELL 45/46's OWN individual_build-filled,
    # Participant-renamed output, not the true pre-individual_build raw ELAN
    # -- apply_individual_build()/cutoffs_s is correctly never invoked by
    # run_group5_validation.py, and doing so now would double-apply the gap
    # already baked into that fixture. apply_individual_build() itself,
    # CUTOFF={"Mintan":2965,"Adarsh":2047,"Ali":2047} (CELL 45, real
    # participant names -- confirmed verbatim, NOT wired in anywhere per this
    # module's own anonymization-boundary rule: no real name may be a Python
    # literal in this module) and the person->ParticipantN NAME map (CELL 46)
    # remain correctly unused for Group 5's real-data validation path.
    5: OeXsensSyncConfig(group=5, video_start_us=_VIDEO_START_US[5], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[5],
                          sync_labels=("clap_synchronizaiton_move", "synchronizaiton_move"),
                          search_mode="relative", search_before_s=15.0, search_after_s=5.0, peak_users=(3,),
                          search_after_s_openearable=15.0, smooth_window=1,
                          apply_shift_xsens=True),

    # Group 6: substring match on "synchron" (not exact), manually hand-picked
    # numeric windows read off a plot: TARGET_SYNC_MIDDLE_RANGE=(1360,1375),
    # SEARCH_WINDOW=(1356,1361.5), SMOOTH_WIN=5 -- all re-verified verbatim
    # 2026-09-05 directly against FINAL_ARDA_THESIS.ipynb's own Group 6 shift
    # cell (notebook code-cell index 68, "GROUP 6 XSENS SHIFT CALCULATION").
    #
    # apply_shift_xsens=True (FIXED 2026-09-05; was False here, a real bug):
    # that same cell 68 reads IN_PATH="...xsens_labeled_cleaned.csv" (the
    # artifact-cleaned-but-unshifted file) and writes
    # OUT_PATH="...xsens_labeled_cleaned_SHIFTED.csv" -- i.e. the notebook
    # DOES compute and save a real cleaned+SHIFTED Xsens file for Group 6,
    # and global_cleaning.py's SELECTED_FILE_NAMES[(6,"xsens")] selects
    # exactly that shifted filename (group_6_xsens_labeled_cleaned_SHIFTED.
    # csv, cross-checked against Global_Cleaning_Before_Model.ipynb cell 3's
    # own SELECTED_FILES dict). The old comment here ("Artifact-cleaned but
    # NOT shifted in the final selection") was wrong -- run_all() with
    # apply_shift_xsens=False would never have written the file
    # global_cleaning.py actually needs, identical in kind to the Group 5
    # apply_shift_xsens bug fixed earlier this session.
    #
    # shift_from_elan=True (FIXED 2026-09-06; this module previously had no
    # way to represent it, a real bug): see the shift_from_elan field
    # docstring above -- cell 68 sources the sync segment's begin_s/end_s/mid
    # from the ORIGINAL ELAN table directly and re-derives every label_*
    # column via a fresh merge_asof against the SHIFTED ELAN times, not via
    # shift_labeled_frame()'s grid-quantized re-paint (that path is correct
    # for Groups 1/7's own cells, confirmed verbatim, but not for Group 6's).
    6: OeXsensSyncConfig(group=6, video_start_us=_VIDEO_START_US[6], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[6],
                          sync_labels=("synchron",), match_mode="substring",
                          expected_mid_range=(1360.0, 1375.0), search_mode="manual_range",
                          search_left_s=1356.0, search_right_s=1361.5, smooth_window=5,
                          clean_xsens_artifacts=True, apply_shift_xsens=True,
                          shift_from_elan=True),

    # Group 7: SHIFTED files ARE selected. (The old device_start_offset_s=59.0
    # "extra correction inside compute_peak_shift" is gone — that number was
    # actually this same XSENS_TO_VIDEO_OFFSET_S, now applied once, upfront,
    # via video_start_us/xsens_to_video_offset_s below, exactly matching
    # notebook CELL 79 which computes its shift purely from video_time_s.)
    #
    # Search-window constants below are verbatim from FINAL_ARDA_THESIS.ipynb
    # CELL 79 ("GROUP 7 - SHIFT OPENEARAMBLE AND XSENS LABELS USING SYNC
    # PEAK"), re-read directly 2026-09-04 (previous port's shared 15/5 window
    # was wrong — a real, confirmed bug, left unfixed until now):
    #   CONFIG = {
    #       "oe":    {"search_before_s": 5, "search_after_s": 25},  # "visible OE peak is after the label, around 2990-2993"
    #       "xsens": {"search_before_s": 5, "search_after_s": 30},  # "visible XSens peak is after the label, around 91s"
    #   }
    # find_sync_segment() sorts matching sync-label segments by start time
    # ascending and takes segments[0] — the EARLIEST occurrence, i.e.
    # occurrence="first" (comment: "Use the first available segment in each
    # file"; also notes OE only sees the later of two ELAN sync occurrences
    # because OE's own recording starts around 80s into the session, while
    # Xsens sees the earlier one — both still resolve to "first" within
    # their own respective label set). process_shift() computes the peak
    # directly on the raw (unsmoothed) combined_acc_mag signal — no rolling-
    # mean smoothing step anywhere in this cell, unlike the visualization-
    # only cells elsewhere in the notebook that use SMOOTH_WINDOW_OE=25/
    # SMOOTH_WINDOW_XSENS=15 for plotting — so smooth_window=1 here makes
    # smooth_signal()'s rolling(window=1) a verified no-op, matching CELL 79
    # exactly rather than silently smoothing with the dataclass's default of
    # 25. match_mode/occurrence-tiebreak/expected_mid_range/peak_users are
    # otherwise unchanged (exact match on "synchronizaiton_move"; USERS =
    # [1, 2, 3]).
    7: OeXsensSyncConfig(group=7, video_start_us=_VIDEO_START_US[7], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[7],
                          sync_labels=("synchronizaiton_move",), search_mode="relative",
                          occurrence="first",
                          search_before_s_openearable=5.0, search_after_s_openearable=25.0,
                          search_before_s_xsens=5.0, search_after_s_xsens=30.0,
                          smooth_window=1,
                          apply_shift_openearable=True, apply_shift_xsens=True,
                          oe_anchor_participants=(1, 2),
                          xsens_merge_mode="group7_wraparound"),

    # Group 8: verbatim from FINAL_ARDA_THESIS.ipynb CELL 104 ("GROUP 8 -
    # SHIFT OPENEARAMBLE AND XSENS LABELS USING SYNC EVENT 0"), re-read
    # directly 2026-09-06 (previous version of this entry was wrong on 3
    # separate constants -- never independently notebook-verified before).
    # Real Whole_Group `synchronizaiton_move` ELAN segments: exactly 2, at
    # mid~125.4s and mid~2327.1s (confirmed from the notebook's own stored
    # cell output). Despite the cell's misleading comment ("We use event 0
    # because...") the code literally sets TARGET_EVENT_INDEX=1, i.e. it
    # picks the SECOND (later) segment -- with only 2 segments total this is
    # equivalent to occurrence="last" (fixed here; was "first", which would
    # have picked the wrong/early segment at ~125s instead of ~2327s).
    # SEARCH_BEFORE_S=3/SEARCH_AFTER_S=30 (fixed; was the generic default
    # 15/5, never actually verified for this group). combined_acc_mag() is
    # used directly with no rolling-mean smoothing step (same as Group 7's
    # CELL 79) -- smooth_window=1 added (fixed; was defaulting to 25, which
    # would have silently smoothed the signal before peak-finding). Also:
    # Group 8's own Xsens merge cell (CELL 97, "GROUP 8 XSENS MERGE -
    # WRAP-SAFE VERSION") is structurally identical to Group 7's wraparound-
    # safe merge (per-participant SampleTimeFine 32-bit wraparound fix +
    # P1-anchored merge_asof(nearest, tolerance=0.02s)) -- just without
    # Group 7's P3-early-dropout branch, since Group 8's 3 participants all
    # have comparable durations (p3_ratio ~1 >= the 0.85 threshold, so
    # merge_xsens_group_group7()'s existing branch logic reduces to exactly
    # this group's simpler all-three-overlap grid). xsens_merge_mode=
    # "group7_wraparound" added (fixed; was "generic", meaning run_all()
    # would have used the exact-SampleTimeFine-intersection join instead --
    # untested against real data before this read, likely would have
    # produced a materially different/degenerate merge for this group's real
    # Xsens files). SHIFTED files ARE selected (global_cleaning.py).
    8: OeXsensSyncConfig(group=8, video_start_us=_VIDEO_START_US[8], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[8],
                          sync_labels=("synchronizaiton_move",), occurrence="last",
                          search_mode="relative", search_before_s=3.0, search_after_s=30.0,
                          smooth_window=1, xsens_merge_mode="group7_wraparound",
                          apply_shift_openearable=True, apply_shift_xsens=True),

    # Group 9: verbatim from FINAL_ARDA_THESIS.ipynb, re-read directly
    # 2026-09-06 (never independently notebook-verified before -- this entry
    # previously only got the shift-search-window part right by luck).
    # Real findings, all confirmed against the notebook's own stored cell
    # outputs (not assumed):
    #  - OE merge (CELL 107, "GROUP 9 OPENEARAMBLE WIDE MERGE"): Participant3
    #    stopped early -> ANCHOR_PARTICIPANTS=[P1,P2], same mechanism as
    #    Group 7. oe_anchor_participants=(1,2) added (fixed; was None/default
    #    3-way intersection, which would truncate the OE grid to P3's short
    #    window and miss the group's real ~2300s+ content entirely).
    #  - Xsens merge is NOT the generic exact-tick join NOR Group 7/8's
    #    wraparound+P1-anchor join -- CELL 114 ("GROUP 9 - LABEL XSENS USING
    #    VIDEO TIME") reads `group_9_xsens_merged_30hz_CONTINUOUS.csv`, the
    #    output of a THIRD, distinct algorithm (CELL 110, "GROUP 9 XSENS
    #    MERGE - CONTINUOUS GRID VERSION"): a uniform 1/30s time_s grid from
    #    0 to end=min(P2, P3 durations) -- explicitly NOT P1 (P1 has internal
    #    gaps this version preserves as NaN rather than silently dropping) --
    #    with all 3 participants merge_asof'd (nearest, tolerance=0.02) onto
    #    it. The plain wrap-safe `..._merged_30hz.csv` CELL 108 also computes
    #    is a dead-end the real pipeline never labels from.
    #    xsens_merge_mode="continuous_grid",
    #    xsens_continuous_grid_anchor_participants=(2, 3) added (fixed; was
    #    the generic exact-tick join, which would have produced a completely
    #    different row count/grid for this group's real data).
    #  - The "official" Whole_Group `synchronization_move` ELAN row at
    #    [12.180, 15.300]s (CELL 116's patch target) was confirmed already
    #    present in the real ELAN fixture ("Row already exists" in the
    #    notebook's own stored output) -- no ELAN-patching code needed here,
    #    read_raw_elan() already surfaces it as-is.
    #  - Shift (CELL 118, "GROUP 9 - OPTIONAL SHIFT USING EARLY COMMON PEAK"):
    #    the official sync label doesn't line up with the real motion burst,
    #    so SYNC_MID is the official label's own midpoint (13.740s -- what
    #    the module's normal find-the-real-segment-then-take-its-midpoint
    #    logic already reproduces, since only one real segment matches) but
    #    the peak search is an ABSOLUTE window (SEARCH_LEFT=50/RIGHT=70,
    #    already correct below) with PER-SENSOR smoothing before peak-finding
    #    (smooth_win=10 for OE, 5 for Xsens) -- smooth_window_openearable=10/
    #    smooth_window_xsens=5 added (fixed; was the shared default 25 for
    #    both, which is wrong for both sensors).
    9: OeXsensSyncConfig(group=9, video_start_us=_VIDEO_START_US[9], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[9],
                          sync_labels=("synchronizaiton_move", "synchronization_move"),
                          search_mode="absolute", search_left_s=50.0, search_right_s=70.0,
                          smooth_window_openearable=10, smooth_window_xsens=5,
                          oe_anchor_participants=(1, 2),
                          xsens_merge_mode="continuous_grid",
                          xsens_continuous_grid_anchor_participants=(2, 3),
                          sync_mid_override=13.740,
                          apply_shift_openearable=True, apply_shift_xsens=True),

    # Group 10: verbatim from FINAL_ARDA_THESIS.ipynb, re-read directly
    # 2026-09-06 (previously only the shift search-window/occurrence/
    # peak_users part -- CELL 132 -- had been notebook-verified; the merge
    # and cleaning stages had not, both turned out to need real fixes).
    #  - OE merge (CELL 121, "GROUP 10 OPENEARAMBLE WIDE MERGE"):
    #    Participant2 is short/weak -> ANCHOR_PARTICIPANTS=[P1,P3].
    #    oe_anchor_participants=(1, 3) added (fixed; was None/default 3-way
    #    intersection, which would truncate the grid to P2's shorter window).
    #  - Xsens merge: same "continuous_grid" family as Group 9 (CELL 123,
    #    "GROUP 10 XSENS MERGE - CONTINUOUS P1/P3 ANCHOR VERSION") but with a
    #    SINGLE anchor participant -- end = Participant3's own full duration
    #    alone (comment: "Use P3 clean full duration... keeps ~25.45 min
    #    instead of cutting to P2's 15.28 min"), not a min-of-two like Group
    #    9. xsens_merge_mode="continuous_grid",
    #    xsens_continuous_grid_anchor_participants=(3,) added (fixed; was
    #    the generic exact-tick join).
    #  - Xsens artifact cleaning: CELL 131 ("GROUP 10 XSENS CLEANING")
    #    is a THIRD, distinct algorithm from Group 6's CELL 80 -- flat
    #    independent per-axis thresholds (ACC>200, GYR>700, EULER>360, plus a
    #    catch-all >1e6) applied directly to the merged/labeled frame's 27
    #    sensor columns, ALL of a flagged row's sensor columns set to NaN
    #    (not just the offending triple), and critically NO interpolation
    #    step afterward (unlike Group 6's `.interpolate(limit=5)`). It is
    #    also a LATER, separate step whose output
    #    (group_10_xsens_labeled_cleaned.csv) only ever feeds the peak-shift/
    #    SHIFTED file (CELL 132's XSENS_IN) -- the plain/unshifted selected
    #    file (group_10_xsens_labeled.csv, UNSHIFTED_FILENAMES default) stays
    #    UNCLEANED, unlike Group 6 where cleaning writes the plain file
    #    itself. clean_xsens_artifacts=True, xsens_cleaning_style="group10",
    #    xsens_cleaning_writes_plain=False added (fixed; was
    #    clean_xsens_artifacts=False entirely -- the SHIFTED file's Xsens
    #    values would have come from uncleaned data, and any 1e30-style raw
    #    artifact would have silently dominated the peak search).
    #  - Shift (CELL 132): occurrence="last"/search_mode="mid_window"/
    #    before=5/after=35/peak_users=(1,3) were already correct (verified
    #    against USERS_FOR_SYNC=[1,3], search_start=mid-5/search_end=mid+35,
    #    choose_late_sync_segment()'s sorted-ascending-then-[-1]). Per-sensor
    #    smoothing (smooth_win=10 OE / 5 Xsens, same values as Group 9) was
    #    missing -- smooth_window_openearable=10/smooth_window_xsens=5 added.
    10: OeXsensSyncConfig(group=10, video_start_us=_VIDEO_START_US[10], xsens_to_video_offset_s=_XSENS_TO_VIDEO_OFFSET_S[10],
                           sync_labels=("synchronizaiton_move",), occurrence="last",
                           search_mode="mid_window", search_before_s=5.0, search_after_s=35.0,
                           smooth_window_openearable=10, smooth_window_xsens=5,
                           peak_users=(1, 3),
                           oe_anchor_participants=(1, 3),
                           xsens_merge_mode="continuous_grid",
                           xsens_continuous_grid_anchor_participants=(3,),
                           clean_xsens_artifacts=True, xsens_cleaning_style="group10",
                           xsens_cleaning_writes_plain=False,
                           apply_shift_openearable=True, apply_shift_xsens=True),
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
    maintain.

    Returns {} (not an error) when the file doesn't exist — that's the
    expected, normal case for anyone running this against the published
    anonymized dataset, where ELAN tiers already read "Participant1" etc.
    rename_tier() already passes already-canonical ParticipantN values
    through unchanged, so an empty map is a correct no-op here, not a
    degraded fallback. A real name_map.json is only needed to process the
    original, non-anonymized ELAN files directly (never published, kept
    private by the thesis author)."""
    path = os.path.join(data_root, f"group_{group}", "elan", "name_map.json")
    if not os.path.exists(path):
        return {}
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
    return {stream: os.path.join(base, f"Participant{participant}_{stream}.csv") for stream in OE_SCHEMA}


def xsens_raw_path(data_root: str, group: int, participant: int) -> str:
    return os.path.join(data_root, f"group_{group}", "xsens", f"Participant{participant}.csv")


def load_sensor_stream(path: str, value_names: list) -> pd.DataFrame:
    """Loads one raw OpenEarable per-stream CSV. Headerless: column 0 is
    `timestamp_us` (absolute Unix epoch microseconds), followed by
    len(value_names) named value columns, followed by a trailing all-zero
    flag column that is dropped. Verbatim from FINAL_ARDA_THESIS.ipynb
    CELL 3's `load_sensor()` (identical in every other group's OE-merge
    cell)."""
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(value_names)]
    df.columns = ["timestamp_us"] + value_names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)
    return df.drop_duplicates("timestamp_us").sort_values("timestamp_us").reset_index(drop=True)


def load_oe_participant_streams(data_root: str, group: int, participant: int) -> dict:
    """{stream_name: df} for one participant's raw OpenEarable streams that
    exist on disk (acc/gyro/mgnt/bone_acc/baro/env_temp/skin_temp/ppg)."""
    streams = {}
    for stream, path in oe_raw_paths(data_root, group, participant).items():
        if os.path.exists(path):
            streams[stream] = load_sensor_stream(path, OE_SCHEMA[stream])
    return streams


def _oe_participant_window(streams: dict, window_streams=OE_WINDOW_STREAMS) -> tuple:
    """Session window (t0, t1) in timestamp_us from the acc/gyro/mgnt trio
    — this also clips e.g. bone_acc's warm-up samples recorded minutes
    early. Verbatim from the notebook's per-participant `windows[p]`."""
    imu = pd.concat([streams[k]["timestamp_us"] for k in window_streams if k in streams])
    return int(imu.min()), int(imu.max())


def merge_openearable_group(participant_streams: dict, grid_us: int = OE_GRID_US,
                             tol_fast_us: int = OE_TOL_FAST_US, tol_slow_us: int = OE_TOL_SLOW_US,
                             slow_streams=OE_SLOW_STREAMS,
                             anchor_participants: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Wide p1_*/p2_*/p3_*-prefixed 50Hz table on the overlap window across
    all 3 participants (latest participant start -> earliest participant
    end, each participant's own window from `_oe_participant_window`).
    Verbatim from the notebook's per-group "WIDE MERGE" cell (e.g. CELL 4
    for Group 1): ONE shared `timestamp_us` grid; each participant's each
    stream is merge_asof'd (nearest, per-stream tolerance) directly onto
    that shared grid — NOT built as separate per-participant grids and
    then inner-joined (which would silently drop rows on any sub-tolerance
    cross-participant timestamp misalignment, since raw timestamps are
    independent absolute epochs, not aligned to a common reference).
    participant_streams: {participant: {stream_name: df}}.

    `anchor_participants`: which participants' windows bound the shared
    grid's start/end (default None = every participant present, i.e. the
    full 3-way intersection every group except Group 7 uses). Group 7's
    Participant3 OE recording stops early, so its config passes (1, 2) here
    (verbatim `ANCHOR_PARTICIPANTS` from FINAL_ARDA_THESIS.ipynb's "GROUP 7
    OPENEARAMBLE WIDE MERGE" cell) — every participant (P3 included) is
    still merge_asof'd onto that grid below; only the grid's own start/end
    bounds are restricted to the anchor set."""
    windows = {p: _oe_participant_window(streams) for p, streams in participant_streams.items()}
    anchor = anchor_participants if anchor_participants is not None else list(windows.keys())
    start = max(windows[p][0] for p in anchor)
    end = min(windows[p][1] for p in anchor)
    grid = np.arange(start, end + grid_us, grid_us, dtype=np.int64)
    grid_df = pd.DataFrame({"timestamp_us": grid})

    out = grid_df.copy()
    out.insert(1, "datetime_utc", pd.to_datetime(out["timestamp_us"], unit="us", utc=True))

    for p, streams in participant_streams.items():
        for stream_name, df_stream in streams.items():
            value_cols = OE_SCHEMA[stream_name]
            df_w = df_stream[(df_stream.timestamp_us >= start - grid_us) & (df_stream.timestamp_us <= end + grid_us)]
            tol = tol_slow_us if stream_name in slow_streams else tol_fast_us
            m = pd.merge_asof(grid_df, df_w, on="timestamp_us", direction="nearest", tolerance=tol)
            for c in value_cols:
                out[f"p{p}_{c}"] = m[c].values
    return out


def load_xsens_participant(data_root: str, group: int, participant: int) -> pd.DataFrame:
    """Loads one participant's raw, headered Xsens CSV. Verbatim from
    FINAL_ARDA_THESIS.ipynb CELL 5's `load()`: drops the trailing
    whitespace-only column (from a trailing comma), strips column-name
    whitespace, coerces SampleTimeFine/PacketCounter/sensor columns
    numeric, and keeps SampleTimeFine as nullable Int64 — a device-relative
    tick counter (no shared epoch across devices/participants, unlike OE's
    `timestamp_us`)."""
    path = xsens_raw_path(data_root, group, participant)
    d = pd.read_csv(path)
    d = d.loc[:, [c for c in d.columns if c.strip() != ""]]
    d.columns = [c.strip() for c in d.columns]
    for c in ["SampleTimeFine", "PacketCounter"] + XSENS_SENSOR_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")
    return d


def merge_xsens_group(participant_frames: dict) -> pd.DataFrame:
    """Master ~30Hz grid = Participant 1's own clean SampleTimeFine ticks,
    intersected with the span common to all 3 participants (drops e.g. a
    participant's corrupted/out-of-range rows). `time_s` is derived as
    (SampleTimeFine - start)/1e6 rounded to 4 decimals, where `start` is
    the shared-span start (device-relative — NOT yet video-relative; see
    `add_xsens_video_time`). Verbatim from FINAL_ARDA_THESIS.ipynb CELL 5.
    participant_frames: {participant: df} from `load_xsens_participant`."""
    p1 = participant_frames[1].dropna(subset=["SampleTimeFine"])
    p1_ticks = set(p1["SampleTimeFine"].tolist())

    spans = []
    for df in participant_frames.values():
        valid = df[df["SampleTimeFine"].isin(p1_ticks)]
        spans.append((int(valid["SampleTimeFine"].min()), int(valid["SampleTimeFine"].max())))
    start = max(s[0] for s in spans)
    end = min(s[1] for s in spans)

    master = (p1[(p1["SampleTimeFine"] >= start) & (p1["SampleTimeFine"] <= end)]
              [["SampleTimeFine"]].sort_values("SampleTimeFine").reset_index(drop=True))
    master["time_s"] = ((master["SampleTimeFine"].astype(np.int64) - start) / 1e6).round(4)

    wide = master.copy()
    for p, df in participant_frames.items():
        pre = f"p{p}"
        sub = (df[["SampleTimeFine"] + XSENS_SENSOR_COLS]
               .dropna(subset=["SampleTimeFine"]).drop_duplicates("SampleTimeFine")
               .rename(columns={c: f"{pre}_{c.lower()}" for c in XSENS_SENSOR_COLS}))
        wide = wide.merge(sub, on="SampleTimeFine", how="left")
    return wide


# Group 7 only — SampleTimeFine 32-bit wraparound fix + P1-anchored
# tolerance merge. Verbatim from FINAL_ARDA_THESIS.ipynb's "GROUP 7 XSENS
# MERGE - AFTER UPDATED FILES" cell (notebook code-cell index 72, re-read
# verbatim 2026-09-06). Real Group 7 Xsens files have a wrapped/reset
# SampleTimeFine tick counter that merge_xsens_group()'s exact-tick
# intersection join can't reconcile across participants — this fixes it
# per-participant (each participant's own first row's raw tick becomes that
# participant's zero) and joins via merge_asof(nearest, tolerance=0.02s)
# instead of an exact-value join.
XSENS_WRAP_MOD = 2 ** 32           # cell 72's `MOD`
XSENS_ASOF_TOLERANCE_S = 0.02      # cell 72's `tolerance=0.02` in merge_asof
XSENS_P3_FULL_RATIO_THRESHOLD = 0.85  # cell 72's `if p3_ratio >= 0.85: ... else: ...`


def load_xsens_participant_wraparound(data_root: str, group: int, participant: int) -> pd.DataFrame:
    """Loads one participant's raw Xsens CSV with the 32-bit SampleTimeFine
    wraparound fix applied. Verbatim from cell 72's `load_xsens_fixed()`:
    coerce columns numeric, drop rows with no SampleTimeFine (keeping
    original row order), take the FIRST remaining row's raw tick (in
    original file order, not chronological order) as `first_tick`, then
    `xsens_time_s_fixed = ((SampleTimeFine - first_tick) % 2**32) / 1e6` —
    a per-participant-zeroed, wraparound-corrected device-relative
    time in seconds. Sorted/deduplicated on that computed column afterward
    (not on raw SampleTimeFine), matching the notebook exactly."""
    path = xsens_raw_path(data_root, group, participant)
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + XSENS_SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])
    df["xsens_time_s_fixed"] = ((df["SampleTimeFine"].astype(np.int64) - first_tick) % XSENS_WRAP_MOD) / 1e6

    return (df.sort_values("xsens_time_s_fixed")
              .drop_duplicates("xsens_time_s_fixed")
              .reset_index(drop=True))


def merge_xsens_group_group7(participant_frames: dict,
                              p3_full_ratio_threshold: float = XSENS_P3_FULL_RATIO_THRESHOLD,
                              tolerance_s: float = XSENS_ASOF_TOLERANCE_S) -> pd.DataFrame:
    """Group-7-only Xsens merge. Verbatim from cell 72: master grid = the
    P1 participant's OWN wraparound-fixed timeline (real per-sample ticks,
    NOT a synthetic uniform grid like OE's), truncated to `end`; `end` is
    the P1/P2 overlap unless Participant3's own recorded duration is at
    least `p3_full_ratio_threshold` (0.85) of that overlap, in which case
    the full 3-way overlap is used instead (cell 72's `p3_ratio` branch —
    for real Group 7 data P3 stops at ~744s vs P1/P2's ~2988-3001s, so the
    ratio is ~0.25 and the P1/P2-anchored branch is always the one taken).
    Each participant (P1 included) is then merge_asof'd (nearest,
    tolerance=0.02s) onto that P1 timeline — not an exact-tick join.
    participant_frames: {participant: df} from
    `load_xsens_participant_wraparound`, keyed 1/2/3."""
    durations = {p: df["xsens_time_s_fixed"].max() for p, df in participant_frames.items()}
    p1p2_end = min(durations[1], durations[2])
    all3_end = min(durations.values())
    p3_ratio = durations[3] / p1p2_end
    end = all3_end if p3_ratio >= p3_full_ratio_threshold else p1p2_end

    p1 = participant_frames[1]
    master = (p1[(p1["xsens_time_s_fixed"] >= 0) & (p1["xsens_time_s_fixed"] <= end)]
              [["xsens_time_s_fixed"]].sort_values("xsens_time_s_fixed").reset_index(drop=True))
    master["time_s"] = master["xsens_time_s_fixed"].round(4)
    wide = master[["time_s"]].copy()

    for p, df in participant_frames.items():
        pre = f"p{p}"
        sub = (df[["xsens_time_s_fixed"] + XSENS_SENSOR_COLS]
               .rename(columns={c: f"{pre}_{c.lower()}" for c in XSENS_SENSOR_COLS}))
        wide = pd.merge_asof(
            wide.sort_values("time_s"),
            sub.sort_values("xsens_time_s_fixed"),
            left_on="time_s", right_on="xsens_time_s_fixed",
            direction="nearest", tolerance=tolerance_s,
        )
        wide = wide.drop(columns=["xsens_time_s_fixed"])
    return wide


XSENS_CONTINUOUS_GRID_S = 1.0 / 30.0   # CELL 110/123's `GRID_S = 1 / 30`
XSENS_CONTINUOUS_TOL_S = 0.02          # CELL 110/123's `TOL_S = 0.02`


def merge_xsens_group_continuous_grid(participant_frames: dict, anchor_participants: tuple,
                                       grid_s: float = XSENS_CONTINUOUS_GRID_S,
                                       tolerance_s: float = XSENS_CONTINUOUS_TOL_S) -> pd.DataFrame:
    """Groups 9/10-only Xsens merge. Verbatim from CELL 110 ("GROUP 9 XSENS
    MERGE - CONTINUOUS GRID VERSION") / CELL 123 ("GROUP 10 XSENS MERGE -
    CONTINUOUS P1/P3 ANCHOR VERSION"): a UNIFORM `time_s` grid (step
    `grid_s`, NOT any participant's own real sample ticks — unlike
    merge_xsens_group()/merge_xsens_group_group7()) from 0 to
    `end = min(durations[p] for p in anchor_participants)`, with EVERY
    participant present in `participant_frames` (not just the anchors)
    merge_asof'd (nearest, tolerance=`tolerance_s`) onto that grid — gaps in
    any one participant's own coverage simply go NaN at that grid point
    rather than truncating or dropping time. Group 9: anchor_participants=
    (2, 3) (P1 has internal gaps this preserves as NaN). Group 10:
    anchor_participants=(3,) (a single participant's own full duration —
    "Use P3 clean full duration... instead of cutting to P2's shorter
    duration"). participant_frames: {participant: df} from
    `load_xsens_participant_wraparound`, keyed 1/2/3 — reused as-is even
    though this algorithm doesn't need the wraparound fix's tick-count
    columns beyond `xsens_time_s_fixed`, which is treated here as the plain
    per-participant device-relative `time_s` CELL 110/123 compute inline."""
    durations = {p: df["xsens_time_s_fixed"].max() for p, df in participant_frames.items()}
    end = min(durations[p] for p in anchor_participants)

    grid = pd.DataFrame({"time_s": np.round(np.arange(0, end + grid_s, grid_s), 4)})
    wide = grid.copy()

    for p, df in participant_frames.items():
        pre = f"p{p}"
        sub = (df[["xsens_time_s_fixed"] + XSENS_SENSOR_COLS]
               .rename(columns={c: f"{pre}_{c.lower()}" for c in XSENS_SENSOR_COLS})
               .rename(columns={"xsens_time_s_fixed": "time_s"}))
        wide = pd.merge_asof(
            wide.sort_values("time_s"),
            sub.sort_values("time_s"),
            on="time_s",
            direction="nearest", tolerance=tolerance_s,
        )
    return wide


def build_oe_merged_grid(data_root: str, group: int) -> pd.DataFrame:
    participant_streams = {p: load_oe_participant_streams(data_root, group, p) for p in PARTICIPANTS}
    cfg = GROUP_SYNC_CONFIG.get(group)
    anchor = cfg.oe_anchor_participants if cfg is not None else None
    return merge_openearable_group(participant_streams, anchor_participants=anchor)


def build_xsens_merged_grid(data_root: str, group: int) -> pd.DataFrame:
    cfg = GROUP_SYNC_CONFIG.get(group)
    mode = cfg.xsens_merge_mode if cfg is not None else "generic"
    if mode == "group7_wraparound":
        participant_frames = {p: load_xsens_participant_wraparound(data_root, group, p) for p in PARTICIPANTS}
        return merge_xsens_group_group7(participant_frames)
    if mode == "continuous_grid":
        participant_frames = {p: load_xsens_participant_wraparound(data_root, group, p) for p in PARTICIPANTS}
        return merge_xsens_group_continuous_grid(participant_frames, cfg.xsens_continuous_grid_anchor_participants)
    participant_frames = {p: load_xsens_participant(data_root, group, p) for p in PARTICIPANTS}
    return merge_xsens_group(participant_frames)


# ================================================================
# Video-relative time (video_time_s) — a separate, EARLIER step than the
# peak-search shift below. See module docstring for the full rationale;
# this is the step that was entirely missing before this fix.
# ================================================================

def _move_col_after(df: pd.DataFrame, col: str, after: str) -> pd.DataFrame:
    cols = list(df.columns)
    cols.remove(col)
    idx = cols.index(after) + 1 if after in cols else 1
    cols.insert(idx, col)
    return df[cols]


def add_oe_video_time(oe_grid: pd.DataFrame, video_start_us: int) -> pd.DataFrame:
    """video_time_s = (timestamp_us - video_start_us) / 1e6 — verbatim from
    every group's OE labeling cell (e.g. CELL 8 for Group 1). Column is
    placed right after datetime_utc, matching the notebook's own ordering."""
    df = oe_grid.copy()
    df["video_time_s"] = (df["timestamp_us"].astype(np.int64) - video_start_us) / 1e6
    return _move_col_after(df, "video_time_s", "datetime_utc")


def add_xsens_video_time(xsens_grid: pd.DataFrame, xsens_to_video_offset_s: float) -> pd.DataFrame:
    """video_time_s = time_s + xsens_to_video_offset_s — verbatim from every
    group's Xsens labeling cell (e.g. CELL 9 for Group 1, CELL 76 for Group
    7). Column is placed right after time_s."""
    df = xsens_grid.copy()
    df["video_time_s"] = df["time_s"] + xsens_to_video_offset_s
    return _move_col_after(df, "video_time_s", "time_s")


# ================================================================
# Label attachment (merge-asof, per tier)
# ================================================================

def attach_tier_label(sensor_df: pd.DataFrame, elan_df: pd.DataFrame, tier: str, time_col: str = "video_time_s") -> pd.Series:
    """merge_asof-backward label lookup — verbatim from every group's
    labeling cell (e.g. CELL 8/CELL 9): a row's label is the most recent
    tier segment whose begin_s is <= the row's time, provided the row's
    time is still STRICTLY before that segment's end_s (`m["video_time_s"]
    < m["_end"]`, not <=)."""
    seg = (elan_df[elan_df["tier"] == tier][["begin_s", "end_s", "label"]]
           .dropna(subset=["begin_s"]).sort_values("begin_s").reset_index(drop=True))
    if seg.empty:
        return pd.Series([""] * len(sensor_df), index=sensor_df.index)

    left = sensor_df[[time_col]].reset_index().sort_values(time_col)
    merged = pd.merge_asof(left, seg, left_on=time_col, right_on="begin_s", direction="backward")
    merged = merged.set_index("index").sort_index()
    label = merged["label"].where(merged[time_col] < merged["end_s"], "")
    return label.fillna("").astype(str)


def attach_all_labels(sensor_df: pd.DataFrame, elan_df: pd.DataFrame, time_col: str = "video_time_s") -> pd.DataFrame:
    df = sensor_df.copy()
    for tier in TIERS:
        df[f"label_{tier}"] = attach_tier_label(df, elan_df, tier, time_col)
    return df


# ================================================================
# Group 6 xsens artifact cleaning
# ================================================================

XSENS_ARTIFACT_ACC_LIMIT = 500.0      # m/s^2 -- generous, keeps real motion
XSENS_ARTIFACT_GYR_LIMIT = 500.0      # generous
XSENS_ARTIFACT_EULER_LIMIT = 10000.0  # very generous, catches absurd corruption (e.g. 1e31)
XSENS_ARTIFACT_INTERP_LIMIT = 5       # samples; ~0.17s at 30Hz


def clean_extreme_artifacts(df: pd.DataFrame, participants=("p1", "p2", "p3")):
    """Verbatim port of CELL 80 ("GROUP 6 - CLEAN / CURE XSENS EXTREME
    ARTIFACTS — Removes impossible values like 1e31 and saves a clean copy"),
    re-read directly from FINAL_ARDA_THESIS.ipynb 2026-09-06.

    A previous version of this function approximated the step as a z-score
    threshold (|z| > 6.0) applied independently to only the 9 `*_acc_[xyz]`
    columns, with no interpolation (NaN'd cells left as NaN). That was WRONG
    — CELL 80's real algorithm:
      - Column set: for each of p1/p2/p3, the 9 columns
        {euler,acc,gyr}_{x,y,z} — 27 sensor columns total (present-only).
      - Per participant, per sensor-type triple (euler / acc / gyr): a row is
        an artifact for that triple if ANY of its 3 axis values exceeds a
        FLAT absolute-value threshold (ACC_LIMIT=500.0, GYR_LIMIT=500.0,
        EULER_LIMIT=10000.0), not a z-score — ALL 3 axis columns of that
        triple/participant are then set to NaN for that row (not just the
        offending axis).
      - The row-level OR of every participant/triple's bad-mask is exposed as
        an `xsens_artifact_removed` boolean column (CELL 80's own output
        column name), not just a returned count.
      - Finally ALL 27 sensor columns are interpolated TOGETHER via
        `.interpolate(method="linear", limit=5, limit_direction="both")` —
        i.e. NaN'd cells are filled in (short gaps only, <=5 consecutive
        samples in either direction), not left as NaN.

    Returns (df_with_cleaned_cols_and_artifact_flag, n_artifact_rows).
    """
    df = df.copy()

    sensor_cols = []
    for p in participants:
        sensor_cols += [
            f"{p}_euler_x", f"{p}_euler_y", f"{p}_euler_z",
            f"{p}_acc_x", f"{p}_acc_y", f"{p}_acc_z",
            f"{p}_gyr_x", f"{p}_gyr_y", f"{p}_gyr_z",
        ]
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    for c in sensor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    artifact_mask = pd.Series(False, index=df.index)

    for p in participants:
        acc_cols = [c for c in (f"{p}_acc_x", f"{p}_acc_y", f"{p}_acc_z") if c in df.columns]
        gyr_cols = [c for c in (f"{p}_gyr_x", f"{p}_gyr_y", f"{p}_gyr_z") if c in df.columns]
        euler_cols = [c for c in (f"{p}_euler_x", f"{p}_euler_y", f"{p}_euler_z") if c in df.columns]

        if acc_cols:
            bad_acc = df[acc_cols].abs().gt(XSENS_ARTIFACT_ACC_LIMIT).any(axis=1)
            df.loc[bad_acc, acc_cols] = np.nan
            artifact_mask = artifact_mask | bad_acc

        if gyr_cols:
            bad_gyr = df[gyr_cols].abs().gt(XSENS_ARTIFACT_GYR_LIMIT).any(axis=1)
            df.loc[bad_gyr, gyr_cols] = np.nan
            artifact_mask = artifact_mask | bad_gyr

        if euler_cols:
            bad_euler = df[euler_cols].abs().gt(XSENS_ARTIFACT_EULER_LIMIT).any(axis=1)
            df.loc[bad_euler, euler_cols] = np.nan
            artifact_mask = artifact_mask | bad_euler

    df["xsens_artifact_removed"] = artifact_mask

    if sensor_cols:
        df[sensor_cols] = df[sensor_cols].interpolate(
            method="linear", limit=XSENS_ARTIFACT_INTERP_LIMIT, limit_direction="both"
        )

    return df, int(artifact_mask.sum())


GROUP10_ACC_LIMIT = 200.0
GROUP10_GYR_LIMIT = 700.0
GROUP10_EULER_LIMIT = 360.0
GROUP10_EXTREME_LIMIT = 1e6


def clean_extreme_artifacts_group10(df: pd.DataFrame, participants=("p1", "p2", "p3")):
    """Verbatim port of CELL 131 ("GROUP 10 XSENS CLEANING — Remove
    impossible numeric artifacts before sync decision"), re-read directly
    from FINAL_ARDA_THESIS.ipynb 2026-09-06. Deliberately NOT the same
    algorithm as clean_extreme_artifacts() (Group 6's CELL 80):
      - A SINGLE row-level mask shared across every sensor column (not one
        mask per participant/axis-triple): a row is bad if ANY sensor column
        of ANY type exceeds its OWN flat threshold (ACC=200, GYR=700,
        EULER=360), OR any sensor column exceeds a catch-all 1e6 regardless
        of type.
      - Every one of the 27 sensor columns is set to NaN for a bad row (not
        just the offending participant/triple).
      - NO interpolation afterward — NaN'd cells stay NaN (unlike Group 6's
        `.interpolate(limit=5)`).
    Returns (df_with_cleaned_cols, n_artifact_rows) — no boolean flag column
    is added (CELL 131 doesn't add one either, unlike CELL 80's
    `xsens_artifact_removed`)."""
    df = df.copy()

    sensor_cols = []
    for p in participants:
        sensor_cols += [
            f"{p}_euler_x", f"{p}_euler_y", f"{p}_euler_z",
            f"{p}_acc_x", f"{p}_acc_y", f"{p}_acc_z",
            f"{p}_gyr_x", f"{p}_gyr_y", f"{p}_gyr_z",
        ]
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    for c in sensor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    artifact_mask = pd.Series(False, index=df.index)
    for c in sensor_cols:
        vals = df[c]
        if "_acc_" in c:
            artifact_mask |= vals.abs() > GROUP10_ACC_LIMIT
        elif "_gyr_" in c:
            artifact_mask |= vals.abs() > GROUP10_GYR_LIMIT
        elif "_euler_" in c:
            artifact_mask |= vals.abs() > GROUP10_EULER_LIMIT
    for c in sensor_cols:
        artifact_mask |= df[c].abs() > GROUP10_EXTREME_LIMIT

    if sensor_cols:
        df.loc[artifact_mask, sensor_cols] = np.nan

    return df, int(artifact_mask.sum())


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


def shift_label_column(labeled_df: pd.DataFrame, time_col: str, label_col: str, offset_s: float,
                        end_inclusive: bool = False) -> pd.Series:
    """Re-derives one shifted label column FROM THE ALREADY-LABELED GRID
    (not from the original ELAN table): finds each existing segment's
    (start, end) via find_label_segments, shifts both boundaries by
    offset_s, and re-paints via a boolean range mask over the (unchanged)
    grid's time_col. This is the notebook's own actual shift algorithm —
    verbatim from CELL 17's `shift_label_column`/`shift_all_labels` (Group
    1) and CELL 79's `shift_label_column`/`shift_all_labels` (Group 7).

    This is NOT the same as `shift_elan_times()` + re-running
    `attach_all_labels()` from the original ELAN begin_s/end_s — that
    alternative was tried first and empirically does NOT reproduce the
    Group 1 real-data fixture (small numbers of boundary-row mismatches,
    a few dozen out of ~60k rows, one per label segment): re-attaching
    from the original real-valued ELAN times ignores that the notebook's
    two-pass shift is grid-quantized (it re-extracts segment boundaries
    from the already-painted grid, at the grid's own resolution, before
    shifting) rather than working from full-precision ELAN times.
    `end_inclusive` selects `<=` vs strict `<` on the shifted upper bound
    — CELL 17 (Group 1) uses `<=`; CELL 79 (Group 7) uses `<`, a real,
    confirmed inconsistency between the notebook's own cells."""
    times = labeled_df[time_col].to_numpy()
    shifted = np.full(len(labeled_df), "", dtype=object)
    for start, end, label in find_label_segments(labeled_df, time_col, label_col):
        new_start, new_end = start + offset_s, end + offset_s
        if end_inclusive:
            mask = (times >= new_start) & (times <= new_end)
        else:
            mask = (times >= new_start) & (times < new_end)
        shifted[mask] = label
    return pd.Series(shifted, index=labeled_df.index).astype(str)


def shift_labeled_frame(labeled_df: pd.DataFrame, time_col: str, offset_s: float,
                         end_inclusive: bool = False) -> pd.DataFrame:
    """Applies shift_label_column() to every tier's label_* column."""
    df = labeled_df.copy()
    for tier in TIERS:
        col = f"label_{tier}"
        if col in df.columns:
            df[col] = shift_label_column(df, time_col, col, offset_s, end_inclusive)
    return df


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
                           sync_start: float, sync_end: float, sync_mid: float,
                           sensor: str | None = None) -> tuple:
    # Per-sensor before/after override (Group 7 only — see the dataclass
    # field docstrings); falls back to the shared search_before_s/
    # search_after_s when unset or when sensor is None, so every other
    # group's behavior is unchanged.
    before = cfg.search_before_s
    after = cfg.search_after_s
    if sensor is not None:
        override_before = getattr(cfg, f"search_before_s_{sensor}", None)
        override_after = getattr(cfg, f"search_after_s_{sensor}", None)
        if override_before is not None:
            before = override_before
        if override_after is not None:
            after = override_after

    if cfg.search_mode == "relative":
        return sync_start - before, sync_end + after
    if cfg.search_mode in ("absolute", "manual_range"):
        return cfg.search_left_s, cfg.search_right_s
    if cfg.search_mode == "mid_window":
        return sync_mid - before, sync_mid + after
    if cfg.search_mode == "last10pct":
        start_idx = int(len(df) * 0.90)
        return float(df[time_col].iloc[start_idx]), float(df[time_col].iloc[-1])
    raise ValueError(f"unknown search_mode {cfg.search_mode!r}")


def compute_peak_shift(labeled_df: pd.DataFrame, cfg: OeXsensSyncConfig, sensor: str,
                        time_col: str = "video_time_s", label_col: str = SYNC_LABEL_COL):
    """Core algorithm (FINAL_ARDA_THESIS_ANALYSIS.md section 1, quoted from
    CELL 66): locate the sync-anchor ELAN label segment on the sensor grid's
    already-attached label_Whole_Group column, take its midpoint, find the
    highest peak of a (participant-averaged) accelerometer-magnitude signal
    within a per-group search window, and return offset_s = peak_time -
    sync_mid (the amount every ELAN label segment should be shifted by).

    `labeled_df` must already be on the `video_time_s` axis (i.e. labeled
    via `attach_all_labels` after `add_oe_video_time`/`add_xsens_video_time`)
    — this is confirmed against the notebook's own Group 7 shift cell
    (CELL 79), which computes its shift delta identically for OE and Xsens,
    purely from `video_time_s`, with NO extra device-offset term added on
    top. (An earlier version of this function added a `device_start_offset_s`
    for xsens here — that was compensating for the *absence* of the
    video_time_s conversion in the previous, broken port; now that the
    conversion happens upfront via `add_xsens_video_time`, adding it again
    here would double-count Group 7's shift. Removed.) Returns (offset_s,
    sync_mid, peak_time)."""
    if cfg.search_mode == "none":
        raise ValueError(f"group {cfg.group} has no sync mechanism (search_mode='none')")

    if cfg.sync_mid_override is not None:
        # Group 9 only -- see OeXsensSyncConfig.sync_mid_override's docstring.
        # The official sync segment never appears on this group's merged
        # sensor grid (both sensors start recording after it ends), so the
        # normal find_label_segments()-based lookup below is skipped entirely.
        sync_start = sync_end = sync_mid = cfg.sync_mid_override
    else:
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
    left, right = resolve_search_window(cfg, labeled_df, time_col, sync_start, sync_end, sync_mid, sensor=sensor)

    signal = combined_acc_signal(labeled_df, cfg.peak_users, ACC_PREFIX_TEMPLATE[sensor], ACC_AXIS_SUFFIXES[sensor])
    smooth_window = getattr(cfg, f"smooth_window_{sensor}", None)
    if smooth_window is None:
        smooth_window = cfg.smooth_window
    signal = smooth_signal(signal, smooth_window)
    peak_time = find_peak_time(labeled_df, time_col, signal, left, right)

    offset_s = peak_time - sync_mid
    return offset_s, sync_mid, peak_time


# ================================================================
# Group 6 only — shift-from-original-ELAN alternative (see
# OeXsensSyncConfig.shift_from_elan's docstring for the full diagnosis).
# Dispatched from process_group() only when cfg.shift_from_elan is True;
# every other group keeps using compute_peak_shift()/shift_labeled_frame()
# unchanged.
# ================================================================

def find_sync_segment_from_elan(elan_df: pd.DataFrame, cfg: OeXsensSyncConfig, tier: str = "Whole_Group"):
    """Locates the sync-anchor segment's (begin_s, end_s, mid) straight off
    the ORIGINAL (pre-shift) ELAN table's own row values — verbatim from
    FINAL_ARDA_THESIS.ipynb's Group 6 shift cell (notebook code-cell index
    68): filters `tier` rows whose label matches `cfg.sync_labels` (exact or
    substring per `cfg.match_mode`), further filters to `cfg.expected_mid_range`
    when set (falling back to the unfiltered match set if nothing lands in
    range, matching every other group's compute_peak_shift() behavior), then
    sorts by mid ASCENDING and takes the first row (cell 68's own
    `target_sync.sort_values("mid").iloc[0]`). This does NOT use
    `cfg.occurrence` — cell 68 never varies its own tie-break, unlike the
    generic grid-based path other groups use.

    This is a genuinely different data source than find_label_segments()'s
    grid-quantized (start, end, label) tuples: the ELAN table's begin_s/
    end_s/mid are full real-valued precision, while the grid-quantized
    values are snapped to the sensor's own sample times (~0.033s steps for
    Xsens). Confirmed empirically against the real Group 6 fixture: using
    this ELAN-sourced mid (together with shift_labeled_frame_from_elan()
    below) is what closes the last 0-11 boundary-row-per-tier gap that
    remains even after switching only the re-paint step to
    shift_labeled_frame_from_elan()."""
    wg = elan_df[elan_df["tier"] == tier].copy()
    wg["mid"] = (wg["begin_s"] + wg["end_s"]) / 2

    if cfg.match_mode == "substring":
        matches = wg[wg["label"].astype(str).apply(
            lambda lab: any(sl.lower() in lab.lower() for sl in cfg.sync_labels))]
    else:
        matches = wg[wg["label"].astype(str).isin(cfg.sync_labels)]

    if cfg.expected_mid_range is not None:
        lo, hi = cfg.expected_mid_range
        ranged = matches[matches["mid"].between(lo, hi)]
        if len(ranged):
            matches = ranged

    if len(matches) == 0:
        raise ValueError(f"group {cfg.group}: no sync-label segment found matching "
                          f"{cfg.sync_labels!r} in tier {tier!r} of the ELAN table")

    row = matches.sort_values("mid").iloc[0]
    return float(row["begin_s"]), float(row["end_s"]), float(row["mid"])


def compute_peak_shift_from_elan(elan_df: pd.DataFrame, labeled_df: pd.DataFrame, cfg: OeXsensSyncConfig,
                                  sensor: str, time_col: str = "video_time_s"):
    """Group-6-only counterpart to compute_peak_shift(): identical peak-search
    machinery (resolve_search_window/combined_acc_signal/smooth_signal/
    find_peak_time), but the sync segment's start/end/mid come from
    find_sync_segment_from_elan() (the ELAN table itself) rather than from
    find_label_segments() run over the already-labeled grid. Returns
    (offset_s, sync_mid, peak_time), same shape as compute_peak_shift()."""
    if cfg.search_mode == "none":
        raise ValueError(f"group {cfg.group} has no sync mechanism (search_mode='none')")

    sync_start, sync_end, sync_mid = find_sync_segment_from_elan(elan_df, cfg)
    left, right = resolve_search_window(cfg, labeled_df, time_col, sync_start, sync_end, sync_mid, sensor=sensor)

    signal = combined_acc_signal(labeled_df, cfg.peak_users, ACC_PREFIX_TEMPLATE[sensor], ACC_AXIS_SUFFIXES[sensor])
    smooth_window = getattr(cfg, f"smooth_window_{sensor}", None)
    if smooth_window is None:
        smooth_window = cfg.smooth_window
    signal = smooth_signal(signal, smooth_window)
    peak_time = find_peak_time(labeled_df, time_col, signal, left, right)

    offset_s = peak_time - sync_mid
    return offset_s, sync_mid, peak_time


def shift_labeled_frame_from_elan(labeled_df: pd.DataFrame, elan_df: pd.DataFrame, offset_s: float,
                                   time_col: str = "video_time_s") -> pd.DataFrame:
    """Group-6-only counterpart to shift_labeled_frame(): shifts the
    ORIGINAL ELAN table's begin_s/end_s by offset_s (via the existing
    shift_elan_times()) and re-runs attach_all_labels() against the
    (sensor-column-unchanged) already-labeled/cleaned `labeled_df` — verbatim
    from cell 68's "REGENERATE LABEL COLUMNS ON XSENS USING SHIFTED ELAN"
    section (drops every label_* column, then re-does the same
    merge_asof(direction="backward")/strict-< labeling as the original
    unshifted attach, now against the shifted ELAN times). attach_all_labels()
    overwrites label_* columns in place, so no explicit drop step is needed
    here. NOT the same as shift_labeled_frame(), which re-derives segment
    boundaries from the already-painted (grid-quantized) label columns
    instead of from the original, full-precision ELAN times."""
    elan_shifted = shift_elan_times(elan_df, offset_s)
    return attach_all_labels(labeled_df, elan_shifted, time_col=time_col)


# ================================================================
# Orchestration
# ================================================================

def output_path(data_root: str, group: int, sensor: str, filename: str, out_dir: str | None = None) -> str:
    root = out_dir or data_root
    return os.path.join(root, f"group_{group}", sensor, f"{sensor}_labeled", filename)


def process_group(data_root: str, group: int, name_map: dict, out_dir: str | None = None,
                   cutoffs_s: dict | None = None, elan_df: pd.DataFrame | None = None) -> dict:
    """Runs merge -> video_time_s conversion -> individual_build ->
    anonymize -> label-attach -> (Group 6 only) artifact-clean -> (where
    GROUP_SYNC_CONFIG wires it in) peak-shift, for both sensors of one
    group. The video_time_s conversion (add_oe_video_time/
    add_xsens_video_time, using cfg.video_start_us/
    cfg.xsens_to_video_offset_s) happens BEFORE label attachment, since
    ELAN's begin_s/end_s are video-relative — label attachment and the
    peak-search shift both then operate on video_time_s, not raw device
    time. Always writes the plain unshifted (or, Group 6 xsens,
    cleaned-but-unshifted) file; additionally writes the shifted file when
    cfg.apply_shift_{sensor} is True. `results[sensor]["shift_info"]` is
    populated whenever cfg.search_mode != "none", regardless of
    apply_shift_{sensor} — a caller that wants the shifted labeled frame for
    a group where run_all() doesn't write it can still get it from
    shift_info via shift_labeled_frame(), e.g. for real-data validation of
    Group 1's xsens shift (apply_shift_xsens=False there, but the shift
    itself is still fully computed). The shift itself is applied via
    shift_labeled_frame() (re-deriving each label segment's boundaries from
    the already-labeled grid and re-painting shifted ones), matching the
    notebook's own CELL 17/CELL 79 shift algorithm — NOT by shifting the
    original ELAN times and re-running attach_all_labels(), which was tried
    first and does not reproduce the real-data fixture (see
    shift_label_column's docstring).

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

    oe_grid = add_oe_video_time(build_oe_merged_grid(data_root, group), cfg.video_start_us)
    xsens_grid = add_xsens_video_time(build_xsens_merged_grid(data_root, group), cfg.xsens_to_video_offset_s)

    results = {}
    for sensor, grid in (("openearable", oe_grid), ("xsens", xsens_grid)):
        labeled = attach_all_labels(grid, elan_df)
        artifact_removed = None
        # `shift_source` is what the peak-search/shift step actually reads.
        # Group 6 (xsens_cleaning_writes_plain=True, default): cleaning
        # output also becomes the plain/unshifted file, so shift_source ==
        # labeled either way. Group 10 (xsens_cleaning_writes_plain=False):
        # the plain file stays uncleaned; only shift_source is cleaned (CELL
        # 131/132 — cleaning is a later, separate step that only ever feeds
        # the peak-shift/SHIFTED output, not the plain labeled file).
        shift_source = labeled
        if sensor == "xsens" and cfg.clean_xsens_artifacts:
            clean_fn = clean_extreme_artifacts_group10 if cfg.xsens_cleaning_style == "group10" else clean_extreme_artifacts
            cleaned, artifact_removed = clean_fn(labeled)
            shift_source = cleaned
            if cfg.xsens_cleaning_writes_plain:
                labeled = cleaned

        plain_name = UNSHIFTED_FILENAMES[(group, sensor)]
        plain_path = output_path(data_root, group, sensor, plain_name, out_dir)
        os.makedirs(os.path.dirname(plain_path), exist_ok=True)
        labeled.to_csv(plain_path, index=False)
        results[sensor] = {"labeled_path": plain_path, "artifact_rows_removed": artifact_removed}

        apply_shift = cfg.apply_shift_openearable if sensor == "openearable" else cfg.apply_shift_xsens
        if cfg.search_mode != "none":
            try:
                if cfg.shift_from_elan:
                    offset_s, sync_mid, peak_time = compute_peak_shift_from_elan(elan_df, shift_source, cfg, sensor)
                else:
                    offset_s, sync_mid, peak_time = compute_peak_shift(shift_source, cfg, sensor)
                results[sensor]["shift_info"] = {"offset_s": offset_s, "sync_mid": sync_mid, "peak_time": peak_time}
                if apply_shift:
                    if cfg.shift_from_elan:
                        shifted_labeled = shift_labeled_frame_from_elan(shift_source, elan_df, offset_s,
                                                                          time_col="video_time_s")
                    else:
                        shifted_labeled = shift_labeled_frame(shift_source, "video_time_s", offset_s,
                                                                end_inclusive=cfg.shift_end_inclusive)
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
