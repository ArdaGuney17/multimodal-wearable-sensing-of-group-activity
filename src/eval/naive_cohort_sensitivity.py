"""Naive-cohort sensitivity — Tables 7.11-7.13 (Ch.7 §7.6).

Unlike most of this repo's other Task 1/2/3 ports, this module is not a
line-by-line port of one notebook cell range. All 3 `after_GL` notebooks
investigated for these tables (`seven_winners_naive5.ipynb`,
`naive5_best_per_task.ipynb`, `naive5_headline_rerun.ipynb` — see their
`notebooks_reference/*_ANALYSIS.md`) turned out to have **no clean saved
execution evidence of actual naive-5 numbers**: one was persisted with
`RUN_ON="all"` despite its name, another's saved outputs stop before any
of its naive-cohort cells ran, the third shows no saved output for its
later cells either. There is nothing genuinely verified to port verbatim.

Instead, this module **re-derives the same computation from the
already-ported, already-verified full-cohort pipeline**
(`src/models/common.py`, shared by `task1.py`/`task2.py`) — restricted to
`NAIVE_GROUPS` via the same explicit dataframe-mask pattern
`task1.py::run_naive5_reproduction` and `task3_naive5.py` already use,
not the source notebooks' `pd.read_csv` monkeypatch. Table 7.12's
`HEADLINE_CONFIGS` (below) is nonetheless cross-verified against
`naive5_best_per_task.ipynb`'s own hardcoded `BEST` dict, which matches
Table 7.12's Config/Full9-A/Full9-M columns digit-for-digit — so even
though that notebook's own run never completed, its config table is
trustworthy as "which (sensor_combo, model_type, seq_len, k) is each
task's headline."

Three tables:
  - Table 7.11 (`run_table_7_11`): OptiTrack-feature-family-only grid,
    full9 vs naive5 — best classical config per (task, time_condition)
    plus best-of-BiLSTM/Transformer deep, no-elapsed, k=120.
  - Table 7.12 (`run_headline_comparison`): the single headline
    (sensor, model) config per task, full9 vs naive5, with per-group
    fold mean±SD+CI (t8=2.306 for 9 folds, t4=2.776 for 5).
  - Table 7.13 is exactly the per-group breakdown of the same run
    (returned alongside 7.12 by `run_headline_comparison`).

Task 1's "(optimised)" row (`OPTI2_RELATIVE_ONLY` exact reproduction) is
NOT run by this module — that already has its own dedicated,
higher-fidelity implementation in `task1.py`'s `run_exact_reproduction`/
`run_naive5_reproduction`, using a richer feature set this module's
generic `common.py` pipeline doesn't recover. This module covers the
other 6 rows: Task 1 (ablation) + Tasks 2a-2e.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.common import (
    ALL_MODEL_CONFIGS,
    Config,
    NAIVE_GROUPS,
    build_task_specs,
    choose_fast_seq_len,
    run_classical_for_task_and_sensor,
    train_one_fast_dl_run,
)

# 95% CI critical t-values, keyed by fold count (thesis uses t8=2.306 for
# the 9-group full cohort, t4=2.776 for the 5-group naive subset).
T_CRITICAL_BY_N_FOLDS = {9: 2.306, 5: 2.776}

# task_name -> (sensor_combo, model_type, seq_len, k_features) — the
# headline config for each of Table 7.12's 6 non-"optimised" rows.
HEADLINE_CONFIGS = {
    "interaction_vs_noninteraction": ("OPTI", "transformer", 18, 120),
    "conversation_vs_nonconversation": ("OPTI", "transformer", 9, 120),
    "conversation_vs_building": ("OPTI_XSENS", "bilstm", 9, 120),
    "conversation_vs_merging": ("OE_OPTI", "transformer", 9, 120),
    "merging_vs_building": ("OPTI", "transformer", 9, 120),
    "three_class_activity": ("OE_OPTI", "lstm", 9, 120),
}

DEEP_CANDIDATE_MODEL_TYPES = ("bilstm", "transformer")  # Table 7.11's "Deep no-elapsed" column

# Published values (docs/thesis_reproduction_targets.md §7.6). Used only
# for the optional verification checks below.
TABLE_7_12_REFERENCE = {
    "interaction_vs_noninteraction": {"full9_A": 0.751, "full9_M": 0.751, "naive5_A": 0.627, "naive5_M": 0.626, "dM": -0.125},
    "conversation_vs_nonconversation": {"full9_A": 0.880, "full9_M": 0.845, "naive5_A": 0.883, "naive5_M": 0.825, "dM": -0.020},
    "conversation_vs_building": {"full9_A": 0.881, "full9_M": 0.860, "naive5_A": 0.741, "naive5_M": 0.706, "dM": -0.154},
    "conversation_vs_merging": {"full9_A": 0.879, "full9_M": 0.866, "naive5_A": 0.672, "naive5_M": 0.587, "dM": -0.279},
    "merging_vs_building": {"full9_A": 0.811, "full9_M": 0.738, "naive5_A": 0.609, "naive5_M": 0.403, "dM": -0.335},
    "three_class_activity": {"full9_A": 0.756, "full9_M": 0.699, "naive5_A": 0.540, "naive5_M": 0.513, "dM": -0.186},
}

TABLE_7_13_REFERENCE = {
    "interaction_vs_noninteraction": {"G2": 0.666, "G3": 0.829, "G5": 0.474, "G6": 0.407, "G10": 0.519, "pooled": 0.626},
    "conversation_vs_nonconversation": {"G2": 0.882, "G3": 0.922, "G5": 0.612, "G6": 0.664, "G10": 0.715, "pooled": 0.825},
    "conversation_vs_building": {"G2": 0.882, "G3": 0.931, "G5": 0.389, "G6": 0.561, "G10": 0.725, "pooled": 0.706},
    "conversation_vs_merging": {"G2": 0.438, "G3": 0.432, "G5": 0.132, "G6": 0.930, "G10": 0.921, "pooled": 0.587},
    "merging_vs_building": {"G2": 0.172, "G3": 0.443, "G5": 0.423, "G6": 0.662, "G10": 0.313, "pooled": 0.403},
    "three_class_activity": {"G2": 0.263, "G3": 0.766, "G5": 0.406, "G6": 0.452, "G10": 0.464, "pooled": 0.513},
}

# task -> {classical_no_elapsed: (full9_A, full9_M, naive5_A, naive5_M), ...}
TABLE_7_11_REFERENCE = {
    "interaction_vs_noninteraction": {"classical_no_elapsed": (0.729, 0.729, 0.713, 0.713), "classical_with_elapsed": (0.750, 0.750, 0.730, 0.727), "deep_no_elapsed": (0.751, 0.751, 0.610, 0.609)},
    "conversation_vs_nonconversation": {"classical_no_elapsed": (0.842, 0.819, 0.885, 0.857), "classical_with_elapsed": (0.861, 0.843, 0.885, 0.861), "deep_no_elapsed": (0.880, 0.845, 0.901, 0.860)},
    "conversation_vs_building": {"classical_no_elapsed": (0.830, 0.817, 0.839, 0.819), "classical_with_elapsed": (0.848, 0.838, 0.854, 0.837), "deep_no_elapsed": (0.875, 0.852, 0.798, 0.757)},
    "conversation_vs_merging": {"classical_no_elapsed": (0.827, 0.797, 0.904, 0.882), "classical_with_elapsed": (0.845, 0.814, 0.932, 0.918), "deep_no_elapsed": (0.833, 0.828, 0.869, 0.867)},
    "merging_vs_building": {"classical_no_elapsed": (0.730, 0.653, 0.867, 0.693), "classical_with_elapsed": (0.723, 0.640, 0.870, 0.710), "deep_no_elapsed": (0.811, 0.738, 0.819, 0.591)},
    "three_class_activity": {"classical_no_elapsed": (0.695, 0.653, 0.740, 0.644), "classical_with_elapsed": (0.727, 0.628, 0.787, 0.671), "deep_no_elapsed": (0.767, 0.695, 0.677, 0.558)},
}


def _gid(g) -> int:
    digits = "".join(ch for ch in str(g) if ch.isdigit())
    return int(digits) if digits else -1


def restrict_to_naive_groups(spec: dict) -> dict:
    """Explicit dataframe-mask restriction (not a monkeypatch) — matches
    task1.py::run_naive5_reproduction / task3_naive5.py's pattern."""
    spec = dict(spec)
    df = spec["df"]
    mask = df[spec["group_col"]].map(_gid).isin(NAIVE_GROUPS)
    spec["df"] = df.loc[mask].reset_index(drop=True)
    return spec


def fold_mean_sd_ci(values: np.ndarray):
    """Returns (mean, sd, ci_lo, ci_hi) using the fold-count-matched
    t-critical value (2.306 for 9 folds, 2.776 for 5); falls back to a
    normal-approximation z=1.96 for any other fold count."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else float("nan")
        return m, 0.0, m, m
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    t = T_CRITICAL_BY_N_FOLDS.get(n, 1.96)
    margin = t * sd / np.sqrt(n)
    return mean, sd, mean - margin, mean + margin


def _model_cfg_for(model_type: str) -> dict:
    return next(m for m in ALL_MODEL_CONFIGS if m["model_type"] == model_type)


def run_headline_comparison(task_specs: list[dict], cfg: Config, out_dir: str):
    """Table 7.12 + Table 7.13. Returns (table_7_12, table_7_13)."""
    os.makedirs(out_dir, exist_ok=True)
    rows_712, rows_713 = [], []

    for spec in task_specs:
        task = spec["task_name"]
        if task not in HEADLINE_CONFIGS:
            continue
        sensor_combo, model_type, seq_len, k = HEADLINE_CONFIGS[task]
        model_cfg = _model_cfg_for(model_type)

        print(f"\n[Table 7.12/7.13] {task} | {sensor_combo} | {model_type} | seq={seq_len} | k={k}")
        summary9, folds9, _ = train_one_fast_dl_run(spec, sensor_combo, seq_len, k, model_cfg, seed=42, cfg=cfg)
        spec5 = restrict_to_naive_groups(spec)
        summary5, folds5, _ = train_one_fast_dl_run(spec5, sensor_combo, seq_len, k, model_cfg, seed=42, cfg=cfg)

        if summary9 is None or summary5 is None:
            print(f"  skipped (no evaluable folds for {task})")
            continue

        m9, sd9, lo9, hi9 = fold_mean_sd_ci(folds9["macro_f1"].values)
        m5, sd5, lo5, hi5 = fold_mean_sd_ci(folds5["macro_f1"].values)

        rows_712.append({
            "task": task, "sensor_combo": sensor_combo, "model_type": model_type, "seq_len": seq_len, "k": k,
            "full9_A": summary9["accuracy"], "full9_M": summary9["macro_f1"],
            "full9_fold_mean": m9, "full9_fold_sd": sd9, "full9_ci_lo": lo9, "full9_ci_hi": hi9, "full9_n_folds": len(folds9),
            "naive5_A": summary5["accuracy"], "naive5_M": summary5["macro_f1"],
            "naive5_fold_mean": m5, "naive5_fold_sd": sd5, "naive5_ci_lo": lo5, "naive5_ci_hi": hi5, "naive5_n_folds": len(folds5),
            "dM": summary5["macro_f1"] - summary9["macro_f1"],
        })

        per_group = folds5.set_index("test_group")["macro_f1"].to_dict()
        row713 = {"task": task}
        for g in sorted(NAIVE_GROUPS):
            row713[f"G{g}"] = per_group.get(g, np.nan)
        row713["pooled"] = summary5["macro_f1"]
        rows_713.append(row713)

    table_7_12 = pd.DataFrame(rows_712)
    table_7_13 = pd.DataFrame(rows_713)
    table_7_12.to_csv(os.path.join(out_dir, "table_7_12_headline_naive5_comparison.csv"), index=False)
    table_7_13.to_csv(os.path.join(out_dir, "table_7_13_per_group_breakdown.csv"), index=False)

    print("\nTABLE 7.12 (headline config, full9 vs naive5)")
    print(table_7_12.round(4).to_string(index=False))
    print("\nTABLE 7.13 (per-group breakdown, naive5)")
    print(table_7_13.round(4).to_string(index=False))

    return table_7_12, table_7_13


def run_table_7_11(task_specs: list[dict], cfg: Config, out_dir: str, deep_seq_choice=choose_fast_seq_len):
    """Table 7.11: OptiTrack-only grid, full9 vs naive5. For each task:
    best classical config per time_condition (via run_classical_for_task_and_sensor's
    own best-by-macro-F1 selection) + best-of-{bilstm,transformer} deep,
    no-elapsed, k=120."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for spec in task_specs:
        task = spec["task_name"]
        print(f"\n[Table 7.11] {task} | OPTI")
        row = {"task": task}

        _, _, _, best9 = run_classical_for_task_and_sensor(spec, "OPTI", cfg)
        spec5 = restrict_to_naive_groups(spec)
        _, _, _, best5 = run_classical_for_task_and_sensor(spec5, "OPTI", cfg)

        for time_condition in cfg.time_conditions:
            b9 = best9[best9["time_condition"] == time_condition] if len(best9) else best9
            b5 = best5[best5["time_condition"] == time_condition] if len(best5) else best5
            if len(b9):
                row[f"classical_{time_condition}_full9_A"] = b9.iloc[0]["accuracy"]
                row[f"classical_{time_condition}_full9_M"] = b9.iloc[0]["macro_f1"]
            if len(b5):
                row[f"classical_{time_condition}_naive5_A"] = b5.iloc[0]["accuracy"]
                row[f"classical_{time_condition}_naive5_M"] = b5.iloc[0]["macro_f1"]

        seq_len = deep_seq_choice(spec)
        best_deep9, best_deep5 = None, None
        for model_type in DEEP_CANDIDATE_MODEL_TYPES:
            model_cfg = _model_cfg_for(model_type)
            s9, _, _ = train_one_fast_dl_run(spec, "OPTI", seq_len, 120, model_cfg, seed=42, cfg=cfg)
            if s9 is not None and (best_deep9 is None or s9["macro_f1"] > best_deep9["macro_f1"]):
                best_deep9 = s9
            s5, _, _ = train_one_fast_dl_run(spec5, "OPTI", seq_len, 120, model_cfg, seed=42, cfg=cfg)
            if s5 is not None and (best_deep5 is None or s5["macro_f1"] > best_deep5["macro_f1"]):
                best_deep5 = s5

        if best_deep9 is not None:
            row["deep_no_elapsed_full9_A"] = best_deep9["accuracy"]
            row["deep_no_elapsed_full9_M"] = best_deep9["macro_f1"]
        if best_deep5 is not None:
            row["deep_no_elapsed_naive5_A"] = best_deep5["accuracy"]
            row["deep_no_elapsed_naive5_M"] = best_deep5["macro_f1"]

        rows.append(row)

    table_7_11 = pd.DataFrame(rows)
    table_7_11.to_csv(os.path.join(out_dir, "table_7_11_full9_vs_naive5_opti.csv"), index=False)
    print("\nTABLE 7.11 (OptiTrack feature family, full9 vs naive5)")
    print(table_7_11.round(4).to_string(index=False))
    return table_7_11


def check_against_reference(table_7_12: pd.DataFrame, table_7_13: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """Verification check against docs/thesis_reproduction_targets.md
    §7.6. A looser tolerance than the exact-reproduction checks elsewhere
    in this repo (0.02, not 0.0006-0.0015) because these are stochastic
    DL runs (fresh model + optimizer per LOGO fold, no fixed-seed exact-
    reproduction guarantee the way task1.py's run_exact_reproduction has)
    — a DIFFERS verdict here does not necessarily mean a porting bug."""
    rows = []
    t12 = table_7_12.set_index("task") if len(table_7_12) else table_7_12
    for task, ref in TABLE_7_12_REFERENCE.items():
        if task not in t12.index if len(table_7_12) else True:
            rows.append({"table": "7.12", "task": task, "match": "MISSING"})
            continue
        row = t12.loc[task]
        match = "EXACT" if all(abs(row[k] - ref[k]) < tol for k in ("full9_M", "naive5_M")) else "DIFFERS"
        rows.append({"table": "7.12", "task": task, "naive5_M": round(row["naive5_M"], 3), "report_naive5_M": ref["naive5_M"], "match": match})

    t13 = table_7_13.set_index("task") if len(table_7_13) else table_7_13
    for task, ref in TABLE_7_13_REFERENCE.items():
        if not len(table_7_13) or task not in t13.index:
            rows.append({"table": "7.13", "task": task, "match": "MISSING"})
            continue
        row = t13.loc[task]
        match = "EXACT" if abs(row["pooled"] - ref["pooled"]) < tol else "DIFFERS"
        rows.append({"table": "7.13", "task": task, "pooled": round(row["pooled"], 3), "report_pooled": ref["pooled"], "match": match})

    return pd.DataFrame(rows)


def run_all(cfg: Config, out_dir: str | None = None):
    """Orchestrates Tables 7.11-7.13. `cfg.run_tasks` should include all 6
    recognition tasks (interaction_vs_noninteraction +
    conversation_vs_nonconversation/conversation_vs_building/
    conversation_vs_merging/merging_vs_building/three_class_activity)."""
    out_dir = out_dir or os.path.join(cfg.data_root, "NAIVE_COHORT_SENSITIVITY")
    os.makedirs(out_dir, exist_ok=True)

    task_specs = build_task_specs(cfg)

    table_7_11 = run_table_7_11(task_specs, cfg, out_dir)
    table_7_12, table_7_13 = run_headline_comparison(task_specs, cfg, out_dir)
    check = check_against_reference(table_7_12, table_7_13)

    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md §7.6")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "naive_cohort_sensitivity_report_reproduction_check.csv"), index=False)

    return table_7_11, table_7_12, table_7_13, check
