"""Feature engineering, Task 1 Xsens (XSENS2) half — the historical simple
16-statistic Xsens feature family that feeds
`binary_5s_all_sensor_advanced_features.csv` (the Task 1 *compatibility
base* table; OE's specialized features are merged on top of a copy of this
same base to produce the actual Task 1 model input,
`binary_5s_specialized_oe_merged_all_features.csv` — see
`src/features/eng_task1_oe.py`).

Ported from `master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb`
(`notebooks_reference/master_feature_generator_task1_task2_task3_CORRECTED_
V4_ANALYSIS.md`). Scope is deliberately restricted to the Xsens-only piece
of the 5s Task 1 rebuild — no OptiTrack (OPTI2, out of scope: raw file size
~230MB, a separate task), no OpenEarable (already ported, untouched here).

**Circularity note** — same situation as `eng_task1_oe.py`: the window/
label grid (`group`/`window_start`/`window_end`/`binary_label`, + a few
metadata columns) is bootstrapped by copying those columns out of an
already-existing official Task 1 output CSV (CELL 3, "A1. AUTHORITATIVE
FIVE-SECOND BINARY WINDOW AND LABEL GRID" — genuinely circular, not
re-derived). `load_base_windows` is reused directly from `eng_task1_oe.py`
rather than reimplemented, since it is the exact same cell. What IS
independently computed here is every XSENS2 feature VALUE: `build_xsens2_5s`
recomputes all 40 raw/derived-channel `adv_stats` blocks from the raw
`group_{g}_xsens_model_ready.csv` file(s) — the window boundaries are
reused, the numbers inside each window are not.

Ported pieces, by source cell:
  - `adv_stats`, `num` — CELL 4 ("SECTION 0b — advanced statistic
    expansion"), the shared helpers cell. Checked line-by-line against
    `eng_task1_oe.py`'s copies (which come from the identical source cell)
    and found byte-identical, so they are imported from there rather than
    duplicated.
  - `task1_standardize_columns`, `task1_xsens_candidate_paths`,
    `task1_load_xsens_candidate`, `task1_sane_xsens_values`,
    `task1_xsens_source_score`, `task1_choose_xsens_source`,
    `build_xsens2_5s` — CELL 6 in this notebook's actual cell order ("A3.
    TASK 1 XSENS2 FEATURES — SOURCE-AWARE FIVE-SECOND GENERATOR", the
    5th code cell / `code_cells[4]` when indexing only code cells). This
    IS source-aware, confirmed by reading the cell itself (not assumed
    from the OPTI2 cell's analogous docstring): `task1_xsens_candidate_
    paths` lists up to three possible raw file locations per group
    (`RAW_DIR`, a flat `ALL_MODEL_READY_FILES` fallback, and a per-group
    nested `group_{g}/xsens/model_ready/` fallback — the third is exactly
    the kind of `_old`/re-processed-variant fallback the task brief asked
    to check for), loads whichever candidates exist, scores each by
    counting finite in-window sensor samples after `task1_sane_xsens_
    values` clips insane values (|acc|>100, |gyr|>2000, |euler|>360 to
    NaN), and picks the highest-scoring one (`task1_choose_xsens_source`).
    `task1_standardize_columns` (trivial column-name-strip) is technically
    defined in the *preceding* OPTI2 cell (`code_cells[3]`) in the
    notebook, since Python name resolution just needs it defined before
    this cell runs at execution time — reimplemented verbatim here as a
    one-line helper rather than importing anything from an OPTI2 module,
    since OPTI2 is explicitly out of scope for this port. For each chosen
    source, the main body builds 3 feature families per `adv_stats` call:
    (a) the 27 raw per-axis `p{1,2,3}_{acc,gyr,euler}_{x,y,z}` channels
    directly (`xsens2__p{p}_{kind}_{axis}__*`), (b) 9 per-participant
    vector norms of those same triples (`xsens2__xsens2_norm_p{p}_{kind}
    __*`), (c) 3 per-participant binary "acc data present" availability
    flags (`xsens2__p{p}_xsens_available__*`), plus a 40th `adv_stats`
    call on `xsens_artifact_removed` (zeros if that raw column is absent —
    it is absent from the real Group 1 fixture) and 3 window-level scalar
    summaries (`xsens2__window_available_channels`, `xsens2__window_mean_
    abs_all`, `xsens2__window_energy_all`) computed directly from the
    stacked signal matrix, not via `adv_stats`. This is the "historical
    simple 16-statistic schema" (`adv_stats` writes exactly 16 stats per
    signal) the task brief refers to — a much smaller/flatter feature set
    than OE's specialized ~300-line function, with no per-window resampling
    (features are computed straight from the raw, irregularly-sampled
    in-window rows via `adv_stats`'s own `t`-based slope fit, exactly like
    OE's *generic*, not *specialized*, family).
  - `load_base_windows` / `build_task1_xsens_features` — the driver.
    `load_base_windows` is imported directly from `eng_task1_oe.py` (CELL
    3's grid-bootstrap, identical for both OE and Xsens halves).
    `build_task1_xsens_features` loads the grid, computes `build_xsens2_5s`
    for the requested groups, then merges the XSENS2-specific slice of
    CELL 8 ("A3. MERGE RECONSTRUCTED TASK 1 COMPATIBILITY BASE TABLE"): a
    plain left-merge of the XSENS2 table onto the label grid on
    `["group","window_start","window_end"]` — unlike OE's specialized/
    generic dance, XSENS2 has only one feature table and nothing gets
    dropped.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.eng_task1_oe import (
    END_COL,
    GROUP_COL,
    START_COL,
    adv_stats,
    load_base_windows,
    num,
)

# task1_sane_xsens_values' clip thresholds (CELL 6's literal constants).
XSENS_ACC_CLEAN = 100.0
XSENS_GYR_CLEAN = 2000.0
XSENS_EULER_CLEAN = 360.0


# ================================================================
# A3. Task 1 XSENS2 features — source-aware five-second generator
# ================================================================

def task1_standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port (column-strip only; defined in the preceding OPTI2
    cell in the source notebook, reimplemented here since OPTI2 is out of
    scope for this module)."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def task1_xsens_candidate_paths(raw_dir: str, group: int, source_data_root: str | None = None) -> list[str]:
    """Verbatim port of CELL 6's `task1_xsens_candidate_paths`. The
    notebook's `RAW_DIR` maps to this module's `raw_dir` argument; the
    notebook's `SOURCE_DATA_ROOT`-relative fallbacks (`ALL_MODEL_READY_
    FILES/`, `group_{g}/xsens/model_ready/`) only apply when
    `source_data_root` is supplied — omitted, only the primary `raw_dir`
    candidate is checked."""
    candidates = [os.path.join(raw_dir, f"group_{group}_xsens_model_ready.csv")]

    if source_data_root:
        candidates.append(
            os.path.join(source_data_root, "ALL_MODEL_READY_FILES", f"group_{group}_xsens_model_ready.csv")
        )
        candidates.append(
            os.path.join(
                source_data_root, f"group_{group}", "xsens", "model_ready", f"group_{group}_xsens_model_ready.csv"
            )
        )

    return [path for path in candidates if os.path.exists(path)]


def task1_load_xsens_candidate(path: str) -> dict | None:
    """Verbatim port of CELL 6's `task1_load_xsens_candidate`."""
    frame = pd.read_csv(path, low_memory=False)
    frame = task1_standardize_columns(frame)

    if "video_time_s" in frame.columns:
        time_column = "video_time_s"
    elif "time_s" in frame.columns:
        time_column = "time_s"
    else:
        return None

    frame["t"] = pd.to_numeric(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    return {"path": path, "frame": frame, "time_column": time_column}


def task1_sane_xsens_values(values, kind: str) -> np.ndarray:
    """Verbatim port of CELL 6's `task1_sane_xsens_values`."""
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan

    if kind == "acc":
        values[np.abs(values) > XSENS_ACC_CLEAN] = np.nan
    elif kind == "gyr":
        values[np.abs(values) > XSENS_GYR_CLEAN] = np.nan
    elif kind == "euler":
        values[np.abs(values) > XSENS_EULER_CLEAN] = np.nan

    return values


def task1_xsens_source_score(candidate: dict | None, group_windows: pd.DataFrame) -> int:
    """Verbatim port of CELL 6's `task1_xsens_source_score`."""
    if candidate is None:
        return -1

    frame = candidate["frame"]
    times = frame["t"].to_numpy()
    score = 0

    channels = [
        (f"p{participant}_{kind}_{axis}", kind)
        for participant in (1, 2, 3)
        for kind in ("acc", "gyr", "euler")
        for axis in "xyz"
        if f"p{participant}_{kind}_{axis}" in frame.columns
    ]

    for _, window in group_windows.iterrows():
        start = float(window[START_COL])
        end = float(window[END_COL])

        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, end, side="left")

        sub = frame.iloc[left:right]

        for column, kind in channels:
            values = pd.to_numeric(sub[column], errors="coerce").to_numpy()
            values = task1_sane_xsens_values(values, kind)
            score += int(np.isfinite(values).sum())

    return score


def task1_choose_xsens_source(
    raw_dir: str, group: int, group_windows: pd.DataFrame, source_data_root: str | None = None
) -> tuple[dict, list[dict]]:
    """Verbatim port of CELL 6's `task1_choose_xsens_source`."""
    candidates = []

    for path in task1_xsens_candidate_paths(raw_dir, group, source_data_root):
        candidate = task1_load_xsens_candidate(path)
        if candidate is None:
            continue
        candidate["score"] = task1_xsens_source_score(candidate, group_windows)
        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No usable Xsens source found for group {group}")

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates[0], candidates


def build_xsens2_5s(
    raw_dir: str,
    base_windows: pd.DataFrame,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of CELL 6's `if RUN_TASK1:` body — the actual XSENS2
    5s feature rebuild. Returns `(xsens2_5s, xsens2_source_report)`."""
    rows = []
    source_rows = []

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )

    for group in groups:
        group_windows = (
            base_windows[base_windows["group_num"] == group].sort_values(START_COL).reset_index(drop=True)
        )

        chosen, candidates = task1_choose_xsens_source(raw_dir, group, group_windows, source_data_root)

        frame = chosen["frame"]
        times = frame["t"].to_numpy()

        print(f"\nTask 1 Xsens group {group}")
        for candidate in candidates:
            print(" ", candidate["score"], "|", candidate["path"])
        print(" Chosen:", chosen["path"])

        channels: dict = {}
        norms: dict = {}
        availability: dict = {}

        for participant in (1, 2, 3):
            for signal_type in ("acc", "gyr", "euler"):
                matrix = np.stack(
                    [num(frame, f"p{participant}_{signal_type}_{axis}", clean=1e6) for axis in "xyz"], axis=1
                )

                for axis_index, axis in enumerate("xyz"):
                    channels[f"p{participant}_{signal_type}_{axis}"] = matrix[:, axis_index]

                norms[f"xsens2_norm_p{participant}_{signal_type}"] = np.linalg.norm(matrix, axis=1)

            availability[f"p{participant}_xsens_available"] = (
                np.isfinite(
                    np.stack([channels[f"p{participant}_acc_{axis}"] for axis in "xyz"], axis=1)
                )
                .all(axis=1)
                .astype(float)
            )

        if "xsens_artifact_removed" in frame.columns:
            artifact = num(frame, "xsens_artifact_removed", clean=1e6)
        else:
            artifact = np.zeros(len(frame))

        window_signal_names = list(channels) + list(norms) + list(availability)

        window_signal_matrix = np.column_stack(
            [channels.get(name, norms.get(name, availability.get(name))) for name in window_signal_names]
        )

        source_rows.append(
            {
                "group": group,
                "chosen_path": chosen["path"],
                "score": chosen["score"],
                "time_column": chosen["time_column"],
                "raw_rows": len(frame),
                "windows": len(group_windows),
            }
        )

        for _, window in group_windows.iterrows():
            start = float(window[START_COL])
            end = float(window[END_COL])
            mask = (times >= start) & (times < end)

            record = {GROUP_COL: window[GROUP_COL], START_COL: start, END_COL: end}

            if mask.sum() >= 3:
                window_times = times[mask]

                for name, signal in channels.items():
                    adv_stats(record, f"xsens2__{name}", signal[mask], window_times)

                for name, signal in norms.items():
                    adv_stats(record, f"xsens2__{name}", signal[mask], window_times)

                for name, signal in availability.items():
                    adv_stats(record, f"xsens2__{name}", signal[mask], window_times)

                adv_stats(record, "xsens2__xsens_artifact_removed", artifact[mask], window_times)

                window_values = window_signal_matrix[mask]
                available_by_channel = np.isfinite(window_values).any(axis=0)
                finite_values = window_values[np.isfinite(window_values)]

                record["xsens2__window_available_channels"] = int(available_by_channel.sum())
                record["xsens2__window_mean_abs_all"] = (
                    float(np.mean(np.abs(finite_values))) if finite_values.size else np.nan
                )
                record["xsens2__window_energy_all"] = (
                    float(np.mean(finite_values**2)) if finite_values.size else np.nan
                )

            rows.append(record)

        print(f"Task 1 XSENS2 rebuilt: group {group} | windows={len(group_windows)}")

    xsens2_5s = pd.DataFrame(rows)
    xsens2_source_report = pd.DataFrame(source_rows)

    return xsens2_5s, xsens2_source_report


# ================================================================
# Driver
# ================================================================

def build_task1_xsens_features(
    raw_dir: str,
    official_grid_path: str,
    groups: list[int] | None = None,
    source_data_root: str | None = None,
) -> dict:
    """Loads the official window/label grid (`eng_task1_oe.load_base_
    windows`, CELL 3's bootstrap), computes the XSENS2 5s feature table
    for `groups` (default: every group present in the grid), and merges
    it onto the grid the way CELL 8's "A3. MERGE RECONSTRUCTED TASK 1
    COMPATIBILITY BASE TABLE" does for the XSENS2 slice: a plain left-merge
    on `["group","window_start","window_end"]`.

    Returns a dict with `base_windows`, `xsens2_5s`, `xsens2_source_
    report`, and `merged` (grid columns + XSENS2 columns).
    """
    base_windows = load_base_windows(official_grid_path)

    if groups is None:
        groups = sorted(
            int(g) for g in pd.to_numeric(base_windows["group_num"], errors="coerce").dropna().unique()
        )
    else:
        base_windows = base_windows[base_windows["group_num"].isin(groups)].reset_index(drop=True)

    xsens2_5s, xsens2_source_report = build_xsens2_5s(
        raw_dir, base_windows, groups=groups, source_data_root=source_data_root
    )

    merge_keys = [GROUP_COL, START_COL, END_COL]

    task1_base = base_windows.drop(columns=["group_num"], errors="ignore").copy()
    if len(xsens2_5s) > 0:
        task1_base = task1_base.merge(
            xsens2_5s.drop_duplicates(merge_keys), on=merge_keys, how="left", validate="one_to_one"
        )

    return {
        "base_windows": base_windows,
        "xsens2_5s": xsens2_5s,
        "xsens2_source_report": xsens2_source_report,
        "merged": task1_base,
    }
