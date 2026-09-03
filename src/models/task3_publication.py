"""Task 3, Part IV — merge Part II (deterministic) and Part III (neural)
into the final publication tables.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cell "FINAL CORRECTED
CORE PUBLICATION TABLES" (source notebook's own "Part IV"). Deterministic and
neural models are kept in separate panels because their headline metric
bases differ (pooled LOGO score vs. mean across 3 seeds) — this is the
source notebook's own explicit design choice, preserved here rather than
force-merged into one ranking.

Table 8.4 = Panel A's `segment_markov_h1` row + Panel B's 3 neural rows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def build_core_tables(grammar_summary: pd.DataFrame, neural_summary: pd.DataFrame, long_history_summary: pd.DataFrame, out_dir: str):
    """Returns (deterministic_table, neural_table, combined_descriptive_table).
    grammar_summary / neural_summary / long_history_summary are the outputs of
    task3_common_targets.run_primary_panel / task3_neural.run_all_seeds /
    task3_common_targets.run_long_history_panel respectively."""

    # Panel A: deterministic grammar and sensor models.
    deterministic_table = grammar_summary.sort_values(["pooled_macro_f1", "pooled_accuracy"], ascending=False).reset_index(drop=True)

    # Panel B: neural models.
    neural_table = neural_summary.copy()
    neural_table["model"] = neural_table["model_kind"] + "_" + neural_table["feature_setting"]
    neural_table = neural_table.sort_values("seed_macro_f1_mean", ascending=False).reset_index(drop=True)

    # Descriptive combined table (convenience only — does not rank across
    # panels, since their headline metric bases are not directly comparable).
    combined_rows = []
    for _, row in deterministic_table.iterrows():
        combined_rows.append({
            "panel": "A_deterministic", "model": row["model"],
            "headline_metric_basis": "pooled LOGO score; group mean ± SD reported separately",
            "headline_accuracy": row["pooled_accuracy"], "headline_macro_f1": row["pooled_macro_f1"], "headline_balanced_accuracy": row["pooled_balanced_accuracy"],
            "group_accuracy_mean": row["fold_accuracy_mean"], "group_accuracy_std": row["fold_accuracy_std"],
            "group_macro_f1_mean": row["fold_macro_f1_mean"], "group_macro_f1_std": row["fold_macro_f1_std"],
            "group_balanced_accuracy_mean": row["fold_balanced_accuracy_mean"], "group_balanced_accuracy_std": row["fold_balanced_accuracy_std"],
            "seed_accuracy_mean": np.nan, "seed_accuracy_std": np.nan, "seed_macro_f1_mean": np.nan, "seed_macro_f1_std": np.nan,
        })
    for _, row in neural_table.iterrows():
        combined_rows.append({
            "panel": "B_neural", "model": row["model"],
            "headline_metric_basis": "mean ± SD of pooled results across three seeds",
            "headline_accuracy": row["seed_accuracy_mean"], "headline_macro_f1": row["seed_macro_f1_mean"], "headline_balanced_accuracy": row["seed_balanced_accuracy_mean"],
            "group_accuracy_mean": row["group_accuracy_mean"], "group_accuracy_std": row["group_accuracy_std"],
            "group_macro_f1_mean": row["group_macro_f1_mean"], "group_macro_f1_std": row["group_macro_f1_std"],
            "group_balanced_accuracy_mean": row["group_balanced_accuracy_mean"], "group_balanced_accuracy_std": row["group_balanced_accuracy_std"],
            "seed_accuracy_mean": row["seed_accuracy_mean"], "seed_accuracy_std": row["seed_accuracy_std"],
            "seed_macro_f1_mean": row["seed_macro_f1_mean"], "seed_macro_f1_std": row["seed_macro_f1_std"],
        })
    combined_table = pd.DataFrame(combined_rows)

    os.makedirs(out_dir, exist_ok=True)
    deterministic_table.to_csv(os.path.join(out_dir, "task3_core_FINAL_V2_PANEL_A_deterministic_common_targets.csv"), index=False)
    neural_table.to_csv(os.path.join(out_dir, "task3_core_FINAL_V2_PANEL_B_neural_common_targets.csv"), index=False)
    long_history_summary.to_csv(os.path.join(out_dir, "task3_core_FINAL_V2_PANEL_C_long_history_sensitivity.csv"), index=False)
    combined_table.to_csv(os.path.join(out_dir, "task3_core_FINAL_V2_descriptive_combined_table.csv"), index=False)

    print("\nPANEL A — DETERMINISTIC COMMON-TARGET COMPARISON")
    print(deterministic_table.round(4).to_string(index=False))
    print("\nBest deterministic model by pooled macro-F1:", deterministic_table.iloc[0]["model"])

    print("\nPANEL B — LEAKAGE-FREE NEURAL COMPARISON (-> Table 8.4)")
    print(neural_table.round(4).to_string(index=False))
    print("\nBest neural model by three-seed mean macro-F1:", neural_table.iloc[0]["model"])

    print("\nSaved all corrected core Task 3 tables to:", out_dir)
    return deterministic_table, neural_table, combined_table
