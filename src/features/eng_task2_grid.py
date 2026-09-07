"""Feature engineering, Task 2 base window/label grid — the 10s window
grid + `recognition_label` attachment that CELLs 15 ("OPTI2 — RICH
OPTITRACK FEATURE GENERATION, ROBUST VERSION") and 16 ("XSENS2 — RICH
XSENS FEATURE GENERATION") of `master_feature_generator_task1_task2_
task3_CORRECTED_V4.ipynb` both build identically before computing their
own per-sensor feature families.

Ported from CELLs 15/16's shared grid-prep block (both cells contain a
byte-identical copy of `add_recognition_labels` and the
`eng7 = pd.read_csv(ENG7_PATH) ... eng7["elapsed_min"] = ...` sequence):
  - `add_recognition_labels` — for every row of a base 10s-window grid,
    finds all 3-class-core recognition-label rows (5s windows, from
    `eng3_recognition_labels.build_recognition_table`) whose midpoint
    falls inside that 10s window, and assigns the mode label. Windows
    with no matching 5s row get `NaN` (dropped by the caller).
  - `build_task2_base_windows` — wraps the full grid-prep sequence:
    filter the recognition table to the 3 core classes, attach labels
    to the ENG7 grid, drop unlabelled/non-core windows, and compute
    `elapsed_min` (minutes since each group's first surviving window).

**Non-circularity note.** Unlike Task 1's `load_base_windows` (which
reads an official pre-built CSV because the notebook's own from-scratch
window/label generator was lost), both inputs here ARE reproducible from
raw `model_ready` data by already-ported modules in this repo:
`eng7_proximity_features.build_eng7_proximity` (the 10s window grid,
CELL 12's own logic) and `eng3_recognition_labels.build_recognition_table`
(the 5s recognition-label table, CELLs 8-11's logic reproduced from a
different source notebook but functionally identical — see that module's
own docstring). This module's driver, `build_task2_base_windows`, accepts
either those two rebuilt DataFrames or the two official CSVs
(`interaction_eng7_10s.csv`, `eng3_recognition_3class_core_features.csv`)
interchangeably — same columns, same join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CORE_CLASSES = ["co_building", "co_merging", "conversation"]


def add_recognition_labels(base_df: pd.DataFrame, rec_df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of CELLs 15/16's `add_recognition_labels`. `rec_df`
    must already have a `mid` column (window midpoint) and be pre-filtered
    to the 3 core classes — matches both cells' call sites exactly."""
    labels = []
    for _, w in base_df.iterrows():
        sub = rec_df[
            (rec_df["group"] == w["group"])
            & (rec_df["mid"] >= w["window_start"])
            & (rec_df["mid"] < w["window_end"])
        ]
        labels.append(sub["recognition_label"].mode().iat[0] if len(sub) else np.nan)

    out = base_df.copy()
    out["recognition_label"] = labels
    return out


def build_task2_base_windows(eng7_df: pd.DataFrame, rec_df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of CELLs 15/16's shared grid-prep sequence:
    1. Filter the recognition table to the 3 core classes, compute `mid`.
    2. Attach `recognition_label` to every ENG7 10s window by midpoint
       containment (`add_recognition_labels`).
    3. Drop windows without a core-class label.
    4. Compute `elapsed_min` = minutes since each group's earliest
       surviving window's midpoint.

    Returns the ENG7 grid (all its own columns, e.g. the 3 `opti_*`
    proximity columns if present) plus `recognition_label` and
    `elapsed_min`.
    """
    rec = rec_df[["group", "window_start", "window_end", "recognition_label"]].copy()
    rec = rec[rec["recognition_label"].isin(CORE_CLASSES)].copy()
    rec["mid"] = (rec["window_start"] + rec["window_end"]) / 2.0

    eng7 = add_recognition_labels(eng7_df, rec)
    eng7 = (
        eng7[eng7["recognition_label"].isin(CORE_CLASSES)]
        .dropna(subset=["recognition_label"])
        .reset_index(drop=True)
    )

    eng7["window_mid"] = (eng7["window_start"] + eng7["window_end"]) / 2.0
    group_start = eng7.groupby("group")["window_mid"].transform("min")
    eng7["elapsed_min"] = (eng7["window_mid"] - group_start) / 60.0

    return eng7
