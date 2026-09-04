"""Feature engineering, ENG7 family — the 3-column native OptiTrack
proximity baseline used by Table 7.9 rows 3-4 ("OE best + ENG7
proximity[+elapsed]", Ch.7 §7.5.2).

Ported from `07_feature_engineering_ENG7_activity_invariant.ipynb`'s
CELL 6 (`notebooks_reference/07_feature_engineering_ENG7_activity_
invariant_EXECUTED_CODE_ONLY.py`, lines 55-291): `BUILD INTERACTION_ENG7`
builds a 10s/10s sliding-window grid and computes 4 transfer-invariant
head/hand feature families PLUS a 3-column proximity baseline
(`opti_nearest_pair_dist_mean`, `opti_all_pairs_dist_mean`,
`opti_all_pairs_dist_std`) in one `features()` call per window.

DELIBERATE SCOPE CUT (documented, not silent): only the proximity block
is ported here — per the task brief, Table 7.9 needs just these 3
`opti_*` columns, and the other 3 families (head posture, gaze events,
hand rhythm, cross-person structure) are self-contained numbered blocks
inside the same `features()` function that are cheap to skip outright
(they don't feed the proximity computation or vice versa). Skipping them
also means Xsens is never loaded here (only the head/hand families need
it) — the window range (`lo`/`hi`) still matches CELL 6 exactly, since
CELL 6 itself computes it from OpenEarable + OptiTrack only
(`lo=max(oe["t"].min(),ot["t"].min())`), not Xsens. Group discovery still
requires all 3 sensor files to be present per group (via `eng2_features.
discover_model_ready_groups`), matching CELL 6's own `discover()` filter,
even though Xsens is never opened — keeps the group roster identical to
the source notebook's.

The interaction/non_interaction binary label CELL 6 also computes per
window is dropped here too — it's written into `interaction_eng7_10s.csv`
but never read by any of the Table 7.9 cells (19/20/22), which instead
join windows to `eng3_recognition_3class_core_features.csv`'s
recognition_label by window midpoint (see task_oe9_recognition.py).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.eng2_features import discover_model_ready_groups, load_sensor_csv
from src.features.oe_signal_common import sinterp

WINDOW_S = 10.0
STRIDE_S = 10.0
RESAMPLE_HZ = 25

POS = [f"Participant{p}_{a}" for p in (1, 2, 3) for a in "xyz"]
PAIRS = [(1, 2), (1, 3), (2, 3)]


def proximity_window(ot: pd.DataFrame, ws: float, we: float) -> dict:
    """Verbatim port of CELL 6's proximity block (the tail of `features()`,
    lines 234-239): resamples each participant's OptiTrack position onto
    a `WINDOW_S * RESAMPLE_HZ`-point grid, computes the 3 sorted pairwise
    distances, and returns the 3 `opti_*` summary columns."""
    grid = np.linspace(ws, we, int(WINDOW_S * RESAMPLE_HZ))
    Pg = {p: np.stack([sinterp(grid, ot["t"].values, ot[f"Participant{p}_{a}"].values) for a in "xyz"], axis=1) for p in (1, 2, 3)}
    D = np.stack([np.linalg.norm(Pg[a] - Pg[b], axis=1) for a, b in PAIRS], axis=1)
    Ds = np.sort(D, axis=1)
    return {
        "opti_nearest_pair_dist_mean": float(np.nanmean(Ds[:, 0])),
        "opti_all_pairs_dist_mean": float(np.nanmean(D)),
        "opti_all_pairs_dist_std": float(np.nanstd(D)),
    }


def build_eng7_proximity(input_dir: str, groups: dict | None = None) -> pd.DataFrame:
    """Builds the 10s/10s window grid and the 3-column proximity baseline
    for every group. Returns a DataFrame with `group`, `window_start`,
    `window_end`, and the 3 `opti_*` columns."""
    if groups is None:
        groups = discover_model_ready_groups(input_dir)

    rows = []
    for g, paths in groups.items():
        oe = load_sensor_csv(paths["openearable"], "video_time_s", [])
        ot = load_sensor_csv(paths["optitrack"], "video_time_s", POS)

        lo = max(oe["t"].min(), ot["t"].min())
        hi = min(oe["t"].max(), ot["t"].max())
        rt = oe["t"].values

        for ws in np.arange(lo, hi - WINDOW_S + 1e-9, STRIDE_S):
            we = ws + WINDOW_S
            m = (rt >= ws) & (rt < we)
            if m.sum() == 0:
                continue
            row = proximity_window(ot, ws, we)
            row["group"] = g
            row["window_start"] = ws
            row["window_end"] = we
            rows.append(row)

        del oe, ot

    return pd.DataFrame(rows)


def run_all(data_root: str, out_dir: str | None = None) -> pd.DataFrame:
    """Builds and saves the ENG7 proximity table under
    `{data_root}/INTERACTION_ENG7/interaction_eng7_proximity_10s.csv`
    (a leaner, proximity-only sibling of CELL 6's own
    `interaction_eng7_10s.csv` — see module docstring for the scope cut)."""
    input_dir = os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Model-ready input not found:\n{input_dir}\n"
            "This is the output of the (not-yet-ported) global-cleaning stage "
            "— see docs/table_to_source_mapping.md."
        )

    out_dir = out_dir or os.path.join(data_root, "INTERACTION_ENG7")
    os.makedirs(out_dir, exist_ok=True)

    groups = discover_model_ready_groups(input_dir)
    if not groups:
        raise RuntimeError(f"No group has all 3 sensors' model_ready CSVs present under {input_dir}")

    eng7 = build_eng7_proximity(input_dir, groups=groups)
    out_path = os.path.join(out_dir, "interaction_eng7_proximity_10s.csv")
    eng7.to_csv(out_path, index=False)

    print(f"Saved ENG7 proximity table: {out_path}")
    print("Shape:", eng7.shape)
    return eng7
