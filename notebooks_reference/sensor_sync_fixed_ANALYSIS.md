# Analysis: sensor_sync_fixed.ipynb

Source notebook: `C:\Users\Arda\Desktop\multimodal-group-activity-recognition\notebooks_reference\sensor_sync_fixed.ipynb`

## Cell counts

Total cells: 23
Code cells: 13
Markdown cells: 10

## Import statements (deduplicated)

```python
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample_poly
from math import gcd
from pathlib import Path
from google.colab import drive
import os
```

Note: `import os` appears twice (cell 3 as `import os, glob`, and again as a bare `import os` at the top of cell 7) — harmless duplicate, not a bug.

## Markdown headers / outline (in order)

- (cell 1, H1) Multi-Sensor Synchronization Pipeline — subtitle "Xsens DOT (30 Hz) + OpenEarable → ELAN-annotated group CSV"
  - (cell 1, H3) Two-step approach (ASCII diagram of STEP 1 / STEP 2 data flow)
- (cell 2, H2) Cell 0 — Imports & Configuration
- (cell 5, H2) Cell 1 — Helper Functions
- (cell 8, H2) Cell 2 — Load Raw Data & Sanity Check
- (cell 12, H2) Cell 3 — Sync Peak Detection & Visual Confirmation
- (cell 14, H2) Cell 3b — Manual Peak Override *(skip if Cell 3 looks correct)*
- (cell 16, H2) STEP 1 — Per-Participant Synchronization
  - (cell 16, H3) Cell 4 — Align & Resample Each Participant
- (cell 18, H2) Cell 5 — Verify Step 1 Alignment
- (cell 20, H2) STEP 2 — Group Synchronization
  - (cell 20, H3) Cell 6 — Merge All 3 Participants into One Wide CSV
- (cell 22, H2) Cell 7 — Final Summary & Label Distribution

(Note: these markdown "Cell N" labels are the notebook author's own internal step-numbering scheme, distinct from the actual Jupyter cell index used in this document — e.g. the markdown header "Cell 4" precedes actual notebook cell 16.)

## Hardcoded file paths found in code

**Input:**
- `GROUP_DIR = '/content/drive/MyDrive/thesis/data/group_1'` (cell 3) — root folder for a single group; hardcoded to `group_1`, must be hand-edited to point at a different group's data.
- `/content/drive` — Colab mount point (cell 4, `drive.mount('/content/drive')`).
- ELAN export: `glob.glob(os.path.join(GROUP_DIR, 'elan', '*.csv'))[0]` (cell 9) — reads the **first** `*.csv` found under `{GROUP_DIR}/elan/`, headerless, 9 fixed columns.
- Xsens: `glob.glob(os.path.join(GROUP_DIR, 'xsens', f'{name}*.csv'))` or, if empty, `glob.glob(os.path.join(GROUP_DIR, 'xsens', f'{name_cap}*.csv'))` (cell 9) — one CSV per participant under `{GROUP_DIR}/xsens/`, filename must start with the participant's lowercase or capitalized first name (e.g. `arda*.csv` or `Arda*.csv`).
- OpenEarable participant folder: `glob.glob(os.path.join(GROUP_DIR, 'openearable', name))` or `glob.glob(os.path.join(GROUP_DIR, 'openearable', name_cap))` (cell 9) — expects a subfolder `{GROUP_DIR}/openearable/{arda|Arda}/` etc.
- OpenEarable per-sensor files inside that folder: `glob.glob(os.path.join(oe_dir, f'*_{sensor_key}.csv'))` (cell 9) for `sensor_key` in `{acc, gyro, mgnt, baro, ppg, env_temp, skin_temp, bone_acc}` — i.e. files matching `*_acc.csv`, `*_gyro.csv`, `*_mgnt.csv`, `*_baro.csv`, `*_ppg.csv`, `*_env_temp.csv`, `*_skin_temp.csv`, `*_bone_acc.csv` inside the participant's OpenEarable folder. Headerless CSVs; last column is dropped on load (`df.iloc[:, :-1]`), suggesting a trailing junk/checksum column in the raw export.

**Output:**
- `OUTPUT_DIR = 'output/group_1'` (cell 3) — relative path, auto-created with `os.makedirs(OUTPUT_DIR, exist_ok=True)`.
- `os.path.join(OUTPUT_DIR, 'sync_peak_confirmation.png')` (cell 13) — diagnostic plot of detected sync peaks per participant/device.
- `os.path.join(OUTPUT_DIR, f'{name}_aligned.csv')` (cell 17) — one per-participant aligned CSV, e.g. `arda_aligned.csv`, `rachel_aligned.csv`, `bas_aligned.csv`.
- `os.path.join(OUTPUT_DIR, 'step1_alignment_check.png')` (cell 19) — verification plot overlaying all participants' acc-magnitude on the shared ELAN timeline.
- `os.path.join(OUTPUT_DIR, 'group_synchronized.csv')` (cell 21) — final merged, wide-format, all-participants-plus-labels CSV for the group.

No output paths are namespaced by group number (e.g. no `f'group_{n}_synchronized.csv'`) — the group identity is implicit only in `OUTPUT_DIR` (`'output/group_1'`), which must be edited by hand per run.

## Function / class definitions

- cell 6, function `load_oe_sensor(filepath)` — reads a headerless OpenEarable sensor CSV, drops the last column (trailing junk), renames columns to `time_us` + `v0..vN`, coerces values to numeric, nulls any value with `abs() > 1e6` (comment: "← lowered from 1e10"), interpolates and drops remaining NaNs, casts `time_us` to int64, and returns `(df, hz)` where `hz` is derived from the median of consecutive `time_us` diffs (`1e6 / median(diff)`).
- cell 6, function `load_xsens(filepath)` — reads an Xsens DOT CSV (`skipinitialspace=True`), drops `Unnamed:` columns, coerces all non-`PacketCounter` columns to numeric, **fixes 32-bit `SampleTimeFine` timestamp wrap-around** by detecting backward jumps (`ts[i] < ts[i-1] - WRAP/2`) and adding `2**32` to everything after the wrap, nulls corrupt `Acc_X/Y/Z` spikes (`abs() > 1e6`) and interpolates, computes `time_rel_sec` relative to the first sample, and returns `(df, hz)` (hz from median diff of the unwrapped `SampleTimeFine`).
- cell 6, function `load_elan(filepath)` — reads a headerless ELAN export CSV and assigns fixed column names `['participant', 'tier', 'begin_hms', 'begin_sec', 'end_hms', 'end_sec', 'dur_hms', 'dur_sec', 'label']`, strips whitespace from `label`.
- cell 6, function `xs_acc_mag(df)` — Euclidean norm of `Acc_X, Acc_Y, Acc_Z` (Xsens accelerometer magnitude).
- cell 6, function `oe_acc_mag(df)` — Euclidean norm of `v0, v1, v2` (OpenEarable accelerometer magnitude, first 3 value columns of the `acc` sensor file).
- cell 6, function `resample_array(data, orig_hz, target_hz)` — poly-phase resample via `scipy.signal.resample_poly`, computing `up/down` factors from `gcd(round(orig_hz), target_hz)`; works on 1-D or 2-D (`n_samples x n_channels`) arrays (column-wise for 2-D).
- cell 6, function `interp_to_grid(signal, src_time, dst_time)` — linear interpolation (`np.interp`) of a signal from its own time base onto an arbitrary destination time grid; 1-D or 2-D.
- cell 6, function `get_elan_sync_time(elan_df)` — finds the ELAN row where `label == SYNC_LABEL` (i.e. `'synchronizaiton_move'`) and returns the **midpoint** of that annotation's `begin_sec`/`end_sec` as the single sync-anchor timestamp; raises `ValueError` if the label is not found.
- cell 6, function `assign_labels(time_arr, elan_df, participant_filter)` — for each ELAN annotation row where `participant == participant_filter`, sets the corresponding label string on every sample of `time_arr` that falls within `[begin_sec, end_sec]`; returns an object array of strings (empty string where unlabeled).
- cell 13, function `detect_sync_peak(time_sec, acc_mag, search_window=SYNC_SEARCH_WINDOW_SEC)` — restricts the signal to the **last `search_window` seconds** of the recording, runs `scipy.signal.find_peaks(a_win, height=PEAK_MIN_HEIGHT, distance=10)` to find candidate "sit-up" peaks, and picks the candidate with the highest acceleration magnitude (falling back to the absolute max of the window if no peak clears the height threshold); returns `(best_peak_time_sec, all_candidate_times)`.

No classes are defined anywhere in the notebook. All other code (cells 4, 7, 9, 10, 11, 15, 17, 19, 21, 23) is top-level procedural script logic (loops over `PARTICIPANTS`, direct `print`/plotting/`to_csv` calls) rather than reusable functions — this is the main thing a port to `src/preprocessing/` would need to restructure into callables.

## Hyperparameter-looking constants (verbatim)

- `GROUP_DIR    = '/content/drive/MyDrive/thesis/data/group_1'` — single-group root path, hand-edited per run.
- `OUTPUT_DIR   = 'output/group_1'` — single-group output path, hand-edited per run.
- `PARTICIPANTS = ['arda', 'rachel', 'bas']` — hardcoded participant first names for group 1 only.
- `TARGET_HZ    = 30` — common output sampling rate (Hz) all sensors are resampled/interpolated to.
- `SYNC_LABEL   = 'synchronizaiton_move'` — exact ELAN annotation label used as the sync anchor; comment explicitly says "typo preserved" (i.e. the misspelling is intentional/required to match the ELAN files).
- `SYNC_SEARCH_WINDOW_SEC = 300` — only the last 300s of each recording is searched for the sync (sit-up) peak.
- `PEAK_MIN_HEIGHT = 14` — m/s² acceleration-magnitude threshold for `scipy.signal.find_peaks`.
- `distance=10` — hardcoded (not a named constant) minimum-sample spacing passed to `find_peaks` inside `detect_sync_peak`.
- `OE_SENSORS = {'acc': [...], 'gyro': [...], 'mgnt': [...], 'baro': [...], 'ppg': [...], 'env_temp': [...], 'skin_temp': [...], 'bone_acc': [...]}` — maps OpenEarable file-suffix keys to output column-name lists.
- `ELAN_PAIR_TIERS = ['Bas_Rachel', 'Bas_Arda', 'Arda_Rachel']` — dyad tier names expected in the ELAN export's `tier`/`participant` field.
- `ELAN_GROUP_TIER = 'Whole_Group'` — whole-group tier name.
- `PLOT_WINDOW = 60` — seconds shown either side of the detected peak in the confirmation plot (cell 13).
- `MANUAL_OVERRIDES = {'arda': {'xs': None, 'oe': None}, 'rachel': {'xs': None, 'oe': None}, 'bas': {'xs': None, 'oe': None}}` (cell 15) — per-participant, per-device manual sync-peak override slots; **all currently `None`** (no manual corrections applied for this run). This is the only "correction" mechanism in the notebook and it is empty/inert as saved.
- `1e6` — magnitude threshold used twice for spike/corruption detection (OE sensor values in `load_oe_sensor`, Xsens `Acc_X/Y/Z` in `load_xsens`); comment on the OE one says it was "lowered from 1e10" in a prior iteration.
- `1e6` (again) — microseconds-to-seconds conversion divisor used throughout (`/ 1e6`), not a tunable but worth noting as a repeated magic number.

**No group-specific numeric offset table exists.** There is no `NAIVE_GROUPS`-style set, no per-group dict of manual time-offset corrections, and no code branch keyed on a group id/number anywhere in the notebook (confirmed via search — no `if group ==`, `elif group`, `GROUP_ID`, `group_id` patterns found). The only "group" identity in the code is the single hardcoded `GROUP_DIR = '.../group_1'` string.

## Potentially broken / disabled cells

- **No group-loop / batch mode — must be manually re-run per group.** Despite the notebook being described as operating across a 9-group study, the code contains no loop over groups and no per-group config table. `GROUP_DIR`, `OUTPUT_DIR`, and `PARTICIPANTS` (cell 3) must be hand-edited and the entire notebook re-executed top-to-bottom once per group. This is the single most important structural fact for any porting effort: the "pipeline" as written is a **single-group, human-in-the-loop notebook**, not a batch script.
- **`PARTICIPANTS` list is not generically supported everywhere.** Most loops (`for name in PARTICIPANTS`) are written generically and would work with any list length, and the confirmation-plot cell (cell 13, `plt.subplots(len(PARTICIPANTS), 2, ...)`) is properly parameterized. However, the STEP-1 verification plot (cell 19) **hardcodes 4 subplot rows**: `fig, axes = plt.subplots(4, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [2, 1, 1, 1]})` and then indexes `axes[i + 1]` in a loop over `PARTICIPANTS` — this only works correctly for exactly 3 participants. If a future group has a different participant count, this cell will raise an `IndexError` (too many participants) or silently leave blank axes (too few). Worth flagging if any of the 9 groups have team sizes other than 3.
- **`MANUAL_OVERRIDES` scaffold is empty/unused in this saved run.** All three participants' `xs`/`oe` override slots are `None` (cell 15), meaning the auto-detected sync peaks from `detect_sync_peak` (cell 13) were accepted as-is for group 1. The mechanism exists (and is documented with an example comment: `#   'rachel': {'xs': 1854.0, 'oe': None}`) but was not exercised in this saved copy — a future reader should not assume any manual correction was actually needed/applied for group 1, and should expect to re-derive/re-enter overrides by hand for other groups if their auto-detected peaks look wrong on the confirmation plot.
- **Deliberate typo relied upon as data contract.** `SYNC_LABEL = 'synchronizaiton_move'` — the comment explicitly says the typo is "preserved" to match the actual ELAN annotation text. `get_elan_sync_time()` raises `ValueError` if this exact (misspelled) string isn't found in a group's ELAN export. Fragile: if any of the 9 groups' ELAN files used the correctly-spelled label, or a different label altogether, this cell fails hard for that group.
- **Hard cross-cell dependencies, no runtime guards.** Unlike the other reference notebook analyzed previously (which added explicit `RuntimeError` checks for `globals()` before reusing state from earlier cells), this notebook has **no defensive checks** — cells 9, 10, 11, 13, 15, 17, 19, 21, 23 all assume prior cells ran successfully and reference variables (`raw`, `elan_df`, `elan_sync_t`, `sync_peaks`, `aligned`, `group_df`, etc.) defined earlier with plain Python `NameError` as the only failure mode if run out of order or after a kernel restart.
- **Magic-number spike thresholds not derived from a documented spec.** The `1e6` cutoffs in `load_oe_sensor` and `load_xsens`, and `PEAK_MIN_HEIGHT = 14` (m/s²) in `detect_sync_peak`, all look empirically tuned (the `1e6`/"lowered from 1e10" comment is explicit evidence of prior tuning). These should be validated against all 9 groups' data during porting rather than assumed universal.
- **No `TODO`/`FIXME`/commented-out logic blocks and no per-group `if`/`elif` special-casing found anywhere** in the code (confirmed by pattern search) — contrary to what the task brief anticipated, this notebook does **not** contain per-group manual-correction constants; the correction mechanism is the (currently empty) `MANUAL_OVERRIDES` dict described above, and any true per-group corrections would exist only in *other, separately-run copies* of this notebook (one per group), not in this file.
- **Diagnostic-only cells kept in the run.** Cells 7 (raw `os.walk` directory print), 10 (quick per-participant Acc_X range print), and 11 (diagnostic "File load verification" with heuristic warnings like "ARDA XSENS: acc near zero") are pure print/sanity-check cells with no persisted output — safe to drop or convert to logging/assertions in a ported module, but they encode useful implicit validation logic (e.g. `if xs_mag.max() < 1`, `if oe_t[-1] > 4000`, `if oe_mag.max() > 100`) that should probably become real assertions/tests in the port rather than being discarded.

## Summary

This notebook is a **single-group, semi-manual synchronization pipeline** for one recording session (group 1 only, as saved: `GROUP_DIR = '/content/drive/MyDrive/thesis/data/group_1'`, `PARTICIPANTS = ['arda', 'rachel', 'bas']`). It expects, per group, a directory tree of `{GROUP_DIR}/elan/*.csv` (one headerless ELAN export, 9 fixed columns: participant, tier, begin_hms, begin_sec, end_hms, end_sec, dur_hms, dur_sec, label), `{GROUP_DIR}/xsens/{participant_name}*.csv` (one Xsens DOT CSV per participant, name-prefixed, case-flexible), and `{GROUP_DIR}/openearable/{participant_name}/` (one subfolder per participant containing per-sensor headerless CSVs named `*_acc.csv`, `*_gyro.csv`, `*_mgnt.csv`, `*_baro.csv`, `*_ppg.csv`, `*_env_temp.csv`, `*_skin_temp.csv`, `*_bone_acc.csv`). **The synchronization algorithm is a single-event, physical "sync gesture" (a sit-up) detected via peak-picking, not cross-correlation:** each device's accelerometer-magnitude signal is searched in its last `SYNC_SEARCH_WINDOW_SEC` (300s) for peaks above `PEAK_MIN_HEIGHT` (14 m/s²) using `scipy.signal.find_peaks(..., distance=10)`, the highest such peak is taken as that device's sync event time, and the ELAN video timeline's own annotated "synchronizaiton_move" interval midpoint is the ground-truth anchor (`elan_sync_t`). Each device's own detected peak time is then subtracted from `elan_sync_t` to get a scalar per-device, per-participant time offset (`xs_off`/`oe_off` = `elan_sync_t − detected_peak_time`), which is simply added to that device's relative time axis to place it on the shared ELAN timeline — i.e. offset correction is a single constant shift per participant per device, not a continuous/dynamic re-sync. A manual-override dict (`MANUAL_OVERRIDES`, cell 15) lets a human replace any auto-detected peak with an eyeballed value from the confirmation plot, but in this saved copy all overrides are `None` (unused). After offsetting, each participant's Xsens (9 channels: 3 Euler + 3 Acc + 3 Gyro) and OpenEarable sensors (up to 8 sensor groups) are poly-phase/linearly resampled onto a common `TARGET_HZ` (30 Hz) grid spanning the overlap of all that participant's devices, ELAN labels are attached (individual, relevant pair, and whole-group tiers), and the result is written to `{OUTPUT_DIR}/{name}_aligned.csv` — this is STEP 1 (per-participant). STEP 2 then intersects the three participants' aligned time ranges into one shared window, re-interpolates every participant's sensor columns onto that shared 30 Hz grid with participant-name-prefixed column names (e.g. `arda_xs_acc_x`, `rachel_oe_acc_y`), re-attaches all ELAN label tiers (per-participant, per-dyad, whole-group), and writes one final wide CSV, `{OUTPUT_DIR}/group_synchronized.csv`, plus two diagnostic PNGs (`sync_peak_confirmation.png`, `step1_alignment_check.png`) for visual QA. **It does not handle all 9 groups uniformly in one run** — there is no group loop, no per-group config table, and no per-group correction constants; the group identity is entirely encoded in the hand-edited `GROUP_DIR`/`OUTPUT_DIR`/`PARTICIPANTS` at the top, meaning this exact notebook file was presumably copied/re-run (or its top cell hand-edited and re-executed) once per group, and any per-group manual peak corrections that were needed would live only in those other run instances, not in this artifact. For porting to `src/preprocessing/`, the useful reusable units are the nine functions in cell 6 plus `detect_sync_peak` in cell 13 (all already fairly clean and dependency-light: pandas/numpy/scipy only); everything else (cells 7, 9–11, 15, 17, 19, 21, 23) is inline script logic that should become a parameterized function/class (e.g. `sync_group(group_dir, participants, output_dir, target_hz=30, sync_label=..., manual_overrides=None)`) accepting the group/participant list as arguments instead of module-level constants, ideally looped over all 9 groups' participant rosters (which are not recorded anywhere in this notebook and must be sourced elsewhere, e.g. a per-group manifest). Key risk areas to validate across all 9 groups during the port: whether `SYNC_LABEL`'s exact misspelling and `ELAN_PAIR_TIERS`/`ELAN_GROUP_TIER` naming conventions hold for every group's ELAN export, whether every group has exactly 3 participants (cell 19's hardcoded 4-row plot assumes this), whether the `1e6` spike-threshold and `PEAK_MIN_HEIGHT=14` constants generalize, and whether any group's auto-detected sync peak was wrong and silently accepted (since `MANUAL_OVERRIDES` for group 1 is empty, there is no evidence in this file of how/when a human should distrust the auto-detection).
