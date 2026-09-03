# Analysis: Arda_Thesis_Data_Preprocessing_FIXED_v4.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\Arda_Thesis_Data_Preprocessing_FIXED_v4.ipynb`

## Cell counts

Total cells: 30
Code cells: 23
Markdown cells: 7

## Import statements (deduplicated)

```python
from google.colab import drive
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
```

## Markdown headers / outline (in order)

- (cell 1, H1) `# Fixed notebook version` — a change-log describing the fixes applied in this "_v4" pass: (1) `label_df()` now does sync-safe `individual_building` filling and will not fill it during any `sync_move` interval; (2) Groups 2, 3, and 5 now correctly pass `collab_cutoff` into `label_df()`; (3) Group 2 tier mapping now accepts both underscore and space variants; (4) Group 3 tier mapping now includes the missing `Participant1_Participant2` pair variants; (5) notebook outputs were cleared so old warning/output values do not confuse the rerun.
- (cell 3, plain text, not a markdown header) `Group - 1`
- (cell 7, plain text, not a markdown header) `Group - 2`
- (cell 10, plain text, not a markdown header) `Group - 3`
- (cell 15, plain text, not a markdown header) `Group - 5`
- (cell 22, plain text, not a markdown header) `GROUP_6`
- (cell 28, H2) `## Final validation`

Note: only cells 1 and 28 use actual Markdown `#`/`##` syntax. The five "Group - N" / "GROUP_6" cells are unstyled plain text used as informal section dividers between per-group processing blocks.

## Hardcoded file paths found in code

All paths are rooted at `/content/drive/MyDrive/thesis/data/` (Google Drive, mounted in cell 0 via `drive.mount('/content/drive')`). The directory layout expected per group is:

```
group_N/
  elan/
    Group_N.csv                 (raw ELAN export, or Group_N_Part1.csv / Group_N_Part2.csv for group_3)
    Group_N_clean.csv           (written by this notebook)
  openearable/
    <participant>/
      <participant>_acc.csv
      <participant>_gyro.csv
      <participant>_mgnt.csv
      <participant>_baro.csv
      <participant>_bone_acc.csv
      <participant>_ppg.csv
      <participant>_skin_temp.csv
      <participant>_env_temp.csv
  xsens/
    <participant>.csv
  group_N_oe_labelled.csv       (written by this notebook)
  group_N_xsens_labelled.csv    (written by this notebook)
```

This is **raw, per-sensor-stream data directly from data collection** (individual OpenEarable CSVs per channel, individual Xsens CSVs per participant, raw ELAN tier-export CSVs) — it is **not** reading any output of a `sensor_sync_fixed.ipynb`-style notebook. There is no "already synchronized" merged input file anywhere in this notebook; synchronization (video/Xsens clock offset alignment) is computed inline in this same notebook via hardcoded `VIDEO_START_UTC` / `XSENS_START_UTC` datetime constants per group (see Hyperparameters section and Summary). This directly contradicts the assumption that sensor synchronization is handled entirely upstream by a separate notebook — at least for the offset/alignment step, it is done here.

Inputs read (exact literal paths / patterns, by group):

- **Group 1**: `/content/drive/MyDrive/thesis/data/group_1/group1_xsens_labelled.csv` (cell 4 — reads a file this same notebook only creates in cell 6, see "Potentially broken" below), `/content/drive/MyDrive/thesis/data/group_1/elan/Group_1.csv` (raw ELAN, `header=None`, 9 positional columns `tier,_,t_start_hms,t_start_s,t_end_hms,t_end_s,duration_hms,duration_s,label`), `BASE + "elan/Group_1_clean.csv"` where `BASE = "/content/drive/MyDrive/thesis/data/group_1/"`, `BASE + f"openearable/{p}/"` for `p in ["arda","bas","rachel"]` (via `load_oe_all`), `BASE + "xsens/"` (via `load_xsens`, reads `{p}.csv"`).
- **Group 2**: `/content/drive/MyDrive/thesis/data/group_2/elan/Group_2.csv` (raw ELAN), `BASE + "elan/Group_2_clean.csv"` with `BASE = "/content/drive/MyDrive/thesis/data/group_2/"`, `BASE + f"openearable/{p}/"` and `BASE + "xsens/"` for `p in ["Participant1","Participant2","Participant3"]`.
- **Group 3**: `/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_Part1.csv` and `.../Group_3_Part2.csv` (raw ELAN split into two parts, concatenated with a `PART2_OFFSET = 1506.0`s time shift applied to Part 2), `BASE + "elan/Group_3_clean.csv"` with `BASE = "/content/drive/MyDrive/thesis/data/group_3/"`, `BASE + f"openearable/{p}/"` and `BASE + "xsens/"` for `p in ["Participant1","Participant2","Participant3"]`, plus a diagnostic-only read of `.../group_3/xsens/Participant1.csv` twice (cells 16-17).
- **Group 5**: `/content/drive/MyDrive/thesis/data/group_5/xsens/Participant1.csv` (diagnostic-only, cells 16-17 — note these are misfiled between "Group - 3" and "Group - 5" markdown dividers; see below), `/content/drive/MyDrive/thesis/data/group_5/elan/Group_5.csv` (raw ELAN), `BASE + "elan/Group_5_clean.csv"` with `BASE = "/content/drive/MyDrive/thesis/data/group_5/"`, `BASE + f"openearable/{p}/"` and `BASE + "xsens/"` for `p in ["Participant1","Participant2","Participant3"]`.
- **Group 6**: `BASE + "openearable/{p}/{p}_acc.csv"` (diagnostic timestamp check), `/content/drive/MyDrive/thesis/data/group_6/elan/Group_6.csv` (raw ELAN, diagnostic read), `/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_clean.csv` (cell 25 — re-reads *Group 1's* clean ELAN mid-way through Group 6 processing, purely a diagnostic sanity check unrelated to Group 6), `BASE + "elan/Group_6.csv"` (raw, re-read for the real cleaning pass) and written to `ELAN_FILE = BASE + "elan/Group_6_clean.csv"` with `BASE = "/content/drive/MyDrive/thesis/data/group_6/"`, `BASE + f"openearable/{p}/"` and `BASE + "xsens/"` for `p in ["Participant1","Participant2","Participant3"]`.
- **Cross-group validation** (cells 20, 21, 29): re-reads each group's own `group_N_oe_labelled.csv` / `group_N_xsens_labelled.csv` outputs (paths hardcoded per group in a `groups = {...}` / `GROUP_VALIDATION = {...}` dict).

Outputs written (exact literal paths / naming convention):

- Per-group cleaned ELAN: `{BASE}elan/Group_{N}_clean.csv` — one CSV per group, columns `tier, t_start_hms(raw only), t_start_s, t_end_hms(raw only), t_end_s, duration_hms(raw only), duration_s(raw only), label`, index not written (`index=False`).
- Per-group, per-sensor labelled data: `{BASE}group{N}_oe_labelled.csv` and `{BASE}group{N}_xsens_labelled.csv` — **one file per (group, sensor)** combination, i.e. two output files per group (not one merged file, and not one file per group). Written with `index=False, na_rep=""`. Each is the full-resolution sensor stream (one row per original sample) with one extra string column appended per ELAN tier (e.g. `Participant1`, `Participant1_Participant2`, `Whole_Group`), containing the activity label active at that timestamp or `""` if none.
- Final cross-group validation summary: `/content/drive/MyDrive/thesis/data/all_groups_validation_summary.csv` (cell 29) — one row per (group, sensor, tier) combination with columns `group, sensor, file, tier, rows, filled_rows, sync_rows_any_tier, individual_building_sync_overlap, missing_tier, expected_zero_ok, top_values`.

No windowing/feature-engineering output files (no fixed-size window CSVs, no `*_features.csv`, no `*_5s_*` / `*_10s_*` style names) are produced anywhere in this notebook — see Summary.

## Function / class definitions

All function definitions live in a single cell (cell 2); every other code cell is a top-level, group-specific script (no functions/classes defined there).

- cell 2, function `load_oe(path, col_names)` — reads one headerless OpenEarable sensor CSV, assigns `col_names`, drops rows where `ts` or the second named column fail numeric coercion, casts all columns except `flag` to `float`, and returns it sorted by `ts`.
- cell 2, function `load_oe_all(base_path, p)` — loads all 8 OpenEarable channel files for participant `p` (`acc, gyro, mgnt, baro, bone_acc, ppg, skin_temp, env_temp`) via `load_oe`, then `pd.merge_asof`s each onto the accelerometer's `ts` grid using `direction="nearest", tolerance=25000` (25,000 raw ts units — ts is later divided by `1e6` to get seconds, so this tolerance is 25 ms), tags the merged frame with `participant=p`.
- cell 2, function `load_xsens(base_path, participants, xsens_offset)` — for each participant, reads their Xsens CSV, drops the stray `Unnamed: 11` column, coerces `Acc_X/Y/Z, Gyr_X/Y/Z, PacketCounter` to numeric (dropping non-numeric rows), drops rows where `Acc_X==Acc_Y==Acc_Z==0` (treated as sensor dropout), sorts by `PacketCounter`, detects a `PacketCounter` jump `>1000` (~33s gap at an assumed 30Hz) and truncates everything after the first such jump (prints a `WARNING: {p} Xsens has jump — trimming`), computes `t_local = (PacketCounter - PacketCounter.iloc[0]) / 30.0` (hardcoded 30Hz sample rate assumption), concatenates all participants, and finally computes `t_video = t_local + xsens_offset` per participant to align onto the shared video timeline.
- cell 2, function `_interval_mask(times, intervals)` — returns a boolean Series that is `True` wherever each value in `times` falls inside any closed `[t_start_s, t_end_s]` interval in `intervals`; used to protect `sync_move` intervals from later label overwrites.
- cell 2, function `normalize_elan_columns(elan)` — strips/stringifies the `tier` and `label` columns and coerces `t_start_s`/`t_end_s` to numeric.
- cell 2, function `label_df(df, time_col, elan, all_tiers, collab_cutoff=None)` — the core per-timestamp labeling routine. For each tier in `all_tiers`, it uses `pd.merge_asof(..., direction="backward")` to find the most recent ELAN interval start at or before each sensor timestamp, then keeps that label only if the timestamp is inside the half-open interval `[t_start_s, t_end_s)`; otherwise the label is `""`. For individual-participant tiers (tier name has no `_` and isn't `Whole_Group`) with a matching `collab_cutoff` entry, any still-empty label before the cutoff time is auto-filled with `"individual_building"` — **except** where the timestamp falls inside a `sync_move` interval (from any other tier), which is explicitly protected from this auto-fill (this is the "FIXED" sync-safe behavior the notebook's own markdown cell announces). Prints a per-tier count of labelled rows.
- cell 2, function `validate_label_output(df, all_tiers, name="dataset")` — prints the dataframe shape, lists any `all_tiers` columns missing from `df`, and prints a per-tier count of non-empty labels; a lightweight post-hoc sanity check, not used for filtering/dropping data.

## Hyperparameter-looking constants (verbatim)

There are no window-size, window-stride, or feature-window constants anywhere in this notebook (confirmed via full-text search for `window`, `stride`, `resample`, `rolling` — zero matches). The "hyperparameter"-shaped constants that do exist are sensor-synchronization/cleaning thresholds and per-group timing offsets:

- `xsens_offset` sample rate assumption (inside `load_xsens`): `df["t_local"] = (df["PacketCounter"] - df["PacketCounter"].iloc[0]) / 30.0` — hardcoded 30 Hz Xsens sample rate.
- Jump/gap detection threshold (inside `load_xsens`): `jumps = pc_diff[pc_diff > 1000]  # more than ~33s gap at 30Hz`.
- `merge_asof` OE-channel alignment tolerance (inside `load_oe_all`): `tolerance=25000` (raw ts units ≈ 25 ms, since `ts` is later divided by `1e6` for seconds).
- `t_video = (oe["ts"] / 1e6) - VIDEO_START_UTC` — the `1e6` divisor implies OE `ts` is in microseconds (Unix epoch microseconds).
- Half-open label interval rule (inside `label_df`): `valid = labelled["t_end_s"].notna() & (labelled[time_col] < labelled["t_end_s"])`.
- Per-group video/Xsens synchronization offsets (all `datetime(...).timestamp()` constants, one triplet per group):
  - Group 1: `VIDEO_START_UTC = datetime(2026, 4, 21, 8, 14, 47, tzinfo=timezone.utc)`, `XSENS_START_UTC = datetime(2026, 4, 21, 8, 14, 22, tzinfo=timezone.utc)`.
  - Group 2: `VIDEO_START_UTC = datetime(2026, 4, 21, 12, 4, 25, ...)`, `XSENS_START_UTC = datetime(2026, 4, 21, 12, 7, 51, ...)`.
  - Group 3: `VIDEO_START_UTC = datetime(2026, 4, 22, 12, 21, 50, ...)`, `XSENS_START_UTC = datetime(2026, 4, 22, 12, 23, 31, ...)`, plus `PART2_OFFSET = 1506.0` (seconds added to Group 3 ELAN Part 2 timestamps before concatenation).
  - Group 5: `VIDEO_START_UTC = datetime(2026, 4, 23, 11, 8, 44, ...)`, `XSENS_START_UTC = datetime(2026, 4, 23, 11, 12, 10, ...)`.
  - Group 6: `VIDEO_START_UTC = datetime(2026, 4, 23, 14, 17, 43, ...)`, `XSENS_START_UTC = datetime(2026, 4, 23, 14, 9, 38, ...)`.
  - In every group, `XSENS_OFFSET = XSENS_START_UTC - VIDEO_START_UTC`.
- Per-group `collab_cutoff` dicts (seconds; the time after which "individual_building" gap-filling stops for that participant):
  - Group 1: `{"arda": 1420.0, "bas": 1160.0, "rachel": 1360.0}` (also defined identically, capitalized, earlier in cell 5).
  - Group 2: `{"Participant1": 1742.0, "Participant2": 1742.0, "Participant3": 1742.0}`.
  - Group 3: `{"Participant1": 2495.0, "Participant2": 2495.0, "Participant3": 2065.0}` — note Participant3's cutoff differs from the other two (group-specific asymmetry, intentional per the data, not obviously a bug).
  - Group 5: `{"Participant1": 2965.0, "Participant2": 2047.0, "Participant3": 2047.0}`.
  - Group 6: `{"Participant1": 752.0, "Participant2": 1030.0, "Participant3": 752.0}`.
- `MAX_32 = 4294967296` (2^32) — a `SampleTimeFine` uint32-overflow correction constant used only in the Group 5 diagnostic cell 17 (see "Potentially broken" — this correction is computed and printed but never applied to the actual pipeline; `load_xsens` uses `PacketCounter`/30Hz, not `SampleTimeFine`, for timing).
- Jump threshold in that same diagnostic-only cell: `diff[diff > 1e8]` on `SampleTimeFine`.
- Label/annotation columns: the per-timestamp label columns are named after ELAN tiers, not a single fixed "label" column — e.g. `ALL_TIERS = ["Participant1","Participant2","Participant3","Participant1_Participant2","Participant1_Participant3","Participant2_Participant3","Whole_Group"]` (groups 2/3/5/6) or `["Arda","Bas","Rachel","Bas_Arda","Arda_Rachel","Bas_Rachel","Whole_Group"]` (group 1). The literal string label used for the auto-filled gap category is `"individual_building"`; the literal sync marker label is `"sync_move"`.
- No sample-rate, missing-data-interpolation-limit, or outlier-threshold constants exist for the continuous IMU/PPG/barometer/temperature channels themselves — see Summary for what "cleaning" actually means here.

## Potentially broken / disabled cells

- **Cell 4 reads a file that cell 6 (later in the same notebook) is the one that creates.** Cell 4 does `xs = pd.read_csv(".../group_1/group1_xsens_labelled.csv", ...)` and prints Rachel's row count/time range, but `group1_xsens_labelled.csv` is only written at the end of cell 6 (`xs.to_csv(BASE + "group1_xsens_labelled.csv", ...)`). This only works if a stale copy of that file already exists on disk from a previous run — cell 4 is effectively validating a leftover artifact from an earlier (possibly pre-"FIXED") run, not the output of this notebook's own current run. Worth deciding during porting whether this check is even meaningful, or should be moved after cell 6 / dropped.
- **Execution-order evidence suggests Group 1 and the final validation cells were never actually re-run after the "FIXED" changes were made.** Cells 0, 2, 4, 5, 6 (all of Group 1's processing) and cells 20, 21, 29 (the cross-group / final validation cells) all show `execution_count: None` in the saved `.ipynb`. Cells 4, 5, 6 still carry output text from some prior run (e.g. "OE arda: 99929 rows..."), but cells 20, 21, 29 have **zero** stored outputs — meaning, in the session that produced this saved file, they were not executed at all. Meanwhile Groups 2, 3, 5, 6 (cells 8-9, 11-14, 16-19, 23-27) all carry real, sequential execution counts (6-28). Net effect: there is no evidence in this file that Group 1's labelled CSVs reflect the sync-safe `label_df()` fix described in cell 1's own markdown changelog, and no evidence the final validation summary (`all_groups_validation_summary.csv`) was ever actually produced/checked against the fixed data. This should be re-verified against whatever is actually on disk in Google Drive before trusting Group 1's labelled files as "fixed".
- **Cells ran out of order within the Group 3 block.** Cell 14 (`execution_count=18`, the "Whole_Group labels... setup interval no longer mislabeled" sanity check that reads `group3_oe_labelled.csv`/`group3_xsens_labelled.csv`) has a *lower* execution count than cell 13 (`execution_count=19`, the cell that actually regenerates and writes those same two files). This means cell 14's "OK" validation, as stored in this notebook, ran against an **older** version of the Group 3 labelled files, not the output of the cell 13 run that appears just above it. The validation result shown cannot be trusted as confirming the final Group 3 output; it should be re-run after cell 13 to be meaningful.
- **Group 5 diagnostic cells (16, 17) are misplaced/orphaned.** They sit between the "Group - 3" markdown divider (cell 10) and the "Group - 5" markdown divider (cell 15), but both read `/content/drive/MyDrive/thesis/data/group_5/xsens/Participant1.csv` — i.e. they are Group 5 diagnostics that were pasted into the Group 3 section. Cell 17 additionally computes a `SampleTimeFine`/`PacketCounter`-overflow ("`MAX_32`") correction and prints detected jumps, but this correction is **never used** by the real pipeline — `load_xsens()` (cell 2) times Xsens data purely from `PacketCounter / 30.0`, not from the overflow-corrected `SampleTimeFine`. This looks like leftover exploratory/debug code that never got wired into the actual cleaning logic (and never got moved to the correct section).
- **Cell 25 is an unrelated Group 1 diagnostic dropped into the middle of Group 6 processing.** Between the Group 6 diagnostic cells (23, 24) and the real Group 6 cleaning cell (26), cell 25 re-reads `/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_clean.csv` and re-prints Group 1's `sync_move` rows/cutoffs — unrelated to Group 6 and almost certainly a copy-paste/scratch cell left in place.
- **Groups 4, 7, 8, and 9 are entirely absent.** A full-text search of the code for `group_4`, `group4`, `group_7`, `group_8`, `group_9` (and no-underscore variants) returns zero matches. Only groups **1, 2, 3, 5, and 6** are processed anywhere in this notebook — 5 groups, not 9. If the study design has 9 groups total, the remaining four groups' cleaning/labeling is not present in this notebook at all and must live elsewhere (or not exist yet).
- **No windowing logic anywhere.** Despite the filename ("Data_Preprocessing") and the task framing that this notebook implements windowing (thesis §4.5-4.9) after sensor synchronization, this notebook contains **no** fixed-size time-windowing step, no stride/overlap parameter, no majority-vote or dominant-label window-labeling rule, and no resampling to a fixed sample grid. It stops at per-timestamp label attachment on the raw, native-sample-rate sensor streams. It also performs the sensor-timeline synchronization itself (via `VIDEO_START_UTC`/`XSENS_START_UTC` offsets), rather than consuming an already-synchronized upstream file. This is a fundamental mismatch with the assumed pipeline stage and is the most important finding of this analysis — see Summary.
- **Group-specific tier/typo/compound-label maps are large, hand-maintained, and not shared across groups.** Each group's cell independently redefines its own `tier_map`, `typo_map`, and `compound_map` dictionaries (dozens of entries each, mostly correcting misspelled ELAN annotation labels like `"synchronaziton_move"` → `"sync_move"`). These are not bugs, but they are extensive, error-prone, per-group special-casing that a ported module would need to externalize (e.g. into per-group config/YAML) rather than hardcode in Python, both for maintainability and to make it obvious the four missing groups have no equivalent mapping yet.
- **Group 1 uses named-participant tiers, Groups 2/3/5/6 use generic `ParticipantN` tiers.** Group 1's `ALL_TIERS` is `["Arda","Bas","Rachel","Bas_Arda","Arda_Rachel","Bas_Rachel","Whole_Group"]` while all other groups use `["Participant1","Participant2","Participant3","Participant1_Participant2","Participant1_Participant3","Participant2_Participant3","Whole_Group"]`. Downstream code/joins that assume the generic `ParticipantN` naming will not work unmodified on Group 1's output files.
- No `TODO`/`FIXME`/`XXX`/`HACK` comments were found. No `!pip install` cells. No stored error/exception outputs in any cell (`errors=0` for all 23 code cells).

## Summary

This notebook is **not** a windowing/feature-engineering notebook — it is a per-group raw-sensor loading, timeline synchronization, and ELAN-annotation-to-timestamp labeling notebook, and it appears to be doing (part of) the job the task description assumed was handled entirely by a separate `sensor_sync_fixed.ipynb`. For each of five groups (`group_1`, `group_2`, `group_3`, `group_5`, `group_6` — **groups 4, 7, 8, 9 are not present anywhere in this file**), it: (1) reads a raw, headerless ELAN tier-export CSV (`elan/Group_N.csv`, or `Group_N_Part1.csv`/`Part2.csv` for group 3) and cleans it — stripping whitespace, applying a large hand-built per-group `typo_map` to fix misspelled activity labels, applying a `tier_map` to normalize participant/pair tier names, expanding "`label1 + label2`" compound annotations into separate rows via `compound_map`, and synthesizing `"individual_building"` filler rows for any time gap (before a per-participant `collab_cutoff` timestamp) not already covered by an explicit annotation or a `sync_move` interval — writing the result to `elan/Group_N_clean.csv`; (2) loads the raw per-participant OpenEarable channel CSVs (`acc/gyro/mgnt/baro/bone_acc/ppg/skin_temp/env_temp`, one file per channel per participant) and Xsens IMU CSVs (one file per participant), and (3) attaches, per raw sensor sample, a label for every ELAN tier by nearest-preceding-interval lookup (`merge_asof` + half-open `[start,end)` containment) via `label_df()`. "Dirty" data here means: (a) non-numeric/malformed rows in the raw OE or Xsens CSVs (dropped via `pd.to_numeric(..., errors="coerce")` + filtering out NaN), (b) all-zero-accelerometer Xsens rows, treated as sensor dropout and dropped, (c) a single large `PacketCounter` jump (>1000 counts, ~33s at the assumed fixed 30Hz Xsens rate) taken to mean a corrupted/restarted recording, after which all subsequent Xsens rows for that participant are discarded, and (d) misspelled/inconsistent ELAN annotation text, handled by the typo/tier/compound maps described above. There is **no numeric outlier removal, no missing-data interpolation, and no resampling to a common/fixed sample rate** anywhere in this notebook — gaps and bad samples are simply dropped, never filled or interpolated, and each sensor stream keeps its own native, uneven sampling. There is likewise **no windowing step of any kind**: no window-size or stride constant, no majority-vote or dominant-label rule for assigning one label to a window, and no fixed-size feature matrix is produced. The actual outputs are two full-resolution, per-sample CSVs per group — `group{N}_oe_labelled.csv` and `group{N}_xsens_labelled.csv` (`index=False, na_rep=""`) — each the original sensor rows plus one extra text column per ELAN tier holding the active label (or `""`) at that exact timestamp; a per-group cleaned ELAN CSV (`elan/Group_N_clean.csv`); and, intended as a final cross-check, one merged validation CSV (`all_groups_validation_summary.csv`) — though the execution-count evidence in this saved notebook (see "Potentially broken") suggests that final validation cell, and all of Group 1's processing, were never actually run in the session that produced this saved state, so their correctness relative to the "FIXED" `label_df()` cannot be confirmed from this file alone. The five groups are **not** handled through one shared parametrized function — each group's ELAN-cleaning block is a copy-pasted, independently hand-edited script cell with its own `tier_map`/`typo_map`/`compound_map`/`collab_cutoff`/`VIDEO_START_UTC`/`XSENS_START_UTC` constants (only the OE/Xsens loading and the core `label_df` labeling logic are shared, via the functions in cell 2); Group 1 also uses named participant tiers (`Arda`, `Bas`, `Rachel`, ...) while the other four groups use generic `Participant1/2/3` tiers, and Group 3 additionally splits its ELAN source into two time-shifted parts. For anyone porting this into `src/preprocessing/`: the reusable, group-agnostic core is exactly the seven functions in cell 2 (`load_oe`, `load_oe_all`, `load_xsens`, `_interval_mask`, `normalize_elan_columns`, `label_df`, `validate_label_output`); everything else is per-group configuration (tier maps, typo maps, compound maps, collab cutoffs, video/Xsens UTC start times, participant lists, base paths) that should be externalized into a per-group config structure (e.g. a dict or YAML keyed by group id) rather than copy-pasted code, with the four missing groups (4, 7, 8, 9) either sourced from elsewhere or flagged as not-yet-processed. Critically, this notebook produces **labelled, per-sample, native-rate sensor data** — it is the necessary input to a windowing step, not the windowing step itself; the actual fixed-size-window-with-attached-label logic (window size, stride/overlap, and the majority-vote/dominant-label rule the task description expects) is not implemented anywhere in this file and must be located elsewhere (likely in whatever notebook actually produced the `binary_5s_*` / `activity3_*_10s_*` feature CSVs referenced by the downstream `task1_full_comparison_*` notebook already analyzed).
