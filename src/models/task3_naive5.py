"""Task 3 — naive-5-cohort sensitivity rerun (Table 8.8).

Table 8.8 restricts two already-ported Task 3 pipelines to the 5 fully
naive groups {2, 3, 5, 6, 10} (no researcher participation — see
docs/data_provenance.md): the activity-token back-off grammar
(task3_grammar.run_ngram_backoff, upper block: 116 tokens vs. 244 for the
full cohort) and the window-level persistence analysis
(task3_persistence, lower block: 947 5-second-label windows, 111
transitions, vs. 2071/275 for the full cohort).

Unlike Task 1's `run_naive5_reproduction` (task1.py), no dedicated
`naive5_*.ipynb` notebook was fetched for Task 3 (`thesis/after_GL/
seven_winners_naive5.ipynb`, `naive5_best_per_task.ipynb`,
`naive5_headline_rerun.ipynb` — see docs/table_to_source_mapping.md).
Rather than re-fetch those from Drive, this module follows the same
approach Task 1's naive-5 rerun already uses: **the naive-cohort table is
a straightforward group-restricted rerun of the already-ported full-cohort
pipeline**, not a separately-coded computation. Table 8.8's own numbers
support this: the reported token count (116) is a plausible subset of the
full 244, and the grammar model definitions match task3_grammar.py's
exactly. Treat results from this module as **unverified against the
original naive5 notebooks specifically** (only against the pipeline code
we do have) until/unless those notebooks are fetched and diffed.

The thesis text itself flags one open question this module deliberately
does NOT resolve by fiat: "n-gram back-off, h=1" and "no-self n-gram,
transitions" are reported as numerically identical
(0.190/0.278/0.234/0.121/0.074) — this module runs both computations
faithfully and independently (task3_grammar's ngram_backoff_h1 vs.
task3_persistence's ngram_markov_no_self_backoff_h1) rather than
special-casing them to match; whether they coincide is left to be an
observation, not an assumption.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import NAIVE_GROUPS
from . import task3_grammar
from . import task3_persistence

T4_5FOLD = 2.776  # 5 naive groups, 4 df — docs/thesis_reproduction_targets.md §4

# Published values this module's output should reproduce (thesis Table
# 8.8 — docs/thesis_reproduction_targets.md §8.10). Used only for the
# optional verification check in run_all().
REPORT_REFERENCE = {
    "ngram_backoff_h1": {"G2": 0.190, "G3": 0.278, "G5": 0.234, "G6": 0.121, "G10": 0.074, "mean": 0.179, "std": 0.083},
    "ngram_backoff_h2": {"G2": 0.335, "G3": 0.492, "G5": 0.492, "G6": 0.366, "G10": 0.417, "mean": 0.420, "std": 0.072},
    "ngram_backoff_h3": {"G2": 0.335, "G3": 0.468, "G5": 0.440, "G6": 0.188, "G10": 0.306, "mean": 0.347, "std": 0.112},
    "ngram_backoff_h5": {"G2": 0.334, "G3": 0.362, "G5": 0.451, "G6": 0.205, "G10": 0.306, "mean": 0.331, "std": 0.090},
    "repeat_current_all_windows": {"G2": 0.488, "G3": 0.661, "G5": 0.774, "G6": 0.582, "G10": 0.759, "mean": 0.653, "std": 0.121},
    "no_self_ngram_transitions": {"G2": 0.190, "G3": 0.278, "G5": 0.234, "G6": 0.121, "G10": 0.074, "mean": 0.179, "std": 0.083},
    "repeat_current_transitions": {"G2": 0.0, "G3": 0.0, "G5": 0.0, "G6": 0.0, "G10": 0.0, "mean": 0.0, "std": 0.0},
}


def _gid(g) -> int:
    """Extracts the integer group id from a value that may already be an
    int, or a string like 'group_2' / '2' — same pattern task1.py's
    naive-5 rerun uses, needed because group columns are inconsistently
    typed/named across the original pipeline's outputs."""
    digits = "".join(ch for ch in str(g) if ch.isdigit())
    return int(digits) if digits else -1


def _mean_std_ci(values: np.ndarray):
    m, sd = float(values.mean()), float(values.std(ddof=1))
    lo, hi = m - T4_5FOLD * sd / np.sqrt(len(values)), m + T4_5FOLD * sd / np.sqrt(len(values))
    return m, sd, lo, hi


def run_naive5_grammar(T: pd.DataFrame, fcols: list[str], out_dir: str, hist_lens=(1, 2, 3, 5)) -> pd.DataFrame:
    """Table 8.8 upper block: restricts the 6-label token table to the 5
    naive groups and reruns task3_grammar's n-gram back-off sweep."""
    naive_mask = T["group"].map(_gid).isin(NAIVE_GROUPS)
    T_naive = T.loc[naive_mask].reset_index(drop=True)
    print(f"tokens: {len(T)} -> {len(T_naive)} (naive-5 subset) | groups: {sorted(T_naive['group'].map(_gid).unique())}")

    rows, fold_rows = [], []
    # group_as_text=False (2026-09-06 bug fix, real-data validation): the
    # opposite of task3_grammar.py's own default, which Table 8.7's
    # full-9-group cohort genuinely needs (see that module's docstring).
    # For the naive-5 subset, text-sorting the 5 group ids as strings
    # ('10','2','3','5','6') changes the back-off Counter's training-group
    # insertion order specifically for the G6-held-out fold (only fold
    # where the naive-5 group values' text vs. int sort order differs),
    # flipping most_common() tie-breaks and shifting every ngram_backoff_h*
    # row's G6 macro-F1 by ~0.012-0.013 vs. the published Table 8.8 value.
    # With group_as_text=False, G6 reproduces published exactly at all 4
    # history lengths and G2/G3/G5/G10 are unaffected (verified identical
    # to 4 decimals either way) -- see task3_grammar.run_ngram_backoff's
    # own docstring and docs/table_to_source_mapping.md's "naive-5 rerun"
    # row for the full A/B evidence.
    task3_grammar.run_ngram_backoff(T_naive, rows, fold_rows, hist_lens=hist_lens, group_as_text=False)
    fold_df = pd.DataFrame(fold_rows)

    out_rows = []
    for h in hist_lens:
        model = f"ngram_backoff_h{h}"
        sub = fold_df[fold_df["model"] == model].copy()
        sub["group_id"] = sub["held_group"].map(_gid)
        sub = sub.sort_values("group_id")
        values = sub["macro_f1"].to_numpy(dtype=float)
        mean, std, lo, hi = _mean_std_ci(values)
        out_rows.append({
            "configuration": model,
            **{f"G{g}": float(sub.loc[sub['group_id'] == g, 'macro_f1'].iloc[0]) if (sub['group_id'] == g).any() else np.nan for g in sorted(NAIVE_GROUPS)},
            "mean": mean, "std": std, "ci_lo": lo, "ci_hi": hi, "n_groups": len(values),
        })

    grammar_df = pd.DataFrame(out_rows)
    os.makedirs(out_dir, exist_ok=True)
    grammar_df.to_csv(os.path.join(out_dir, "task3_naive5_grammar.csv"), index=False)
    return grammar_df


def run_naive5_persistence(data_root: str, out_dir: str) -> pd.DataFrame:
    """Table 8.8 lower block: restricts the window-level (group, time)
    table to the 5 naive groups and reruns task3_persistence's
    history_len=1 comparison (only repeat-current and no-self n-gram are
    needed for Table 8.8's 3 lower-block rows)."""
    # apply_merge6=True (kept explicit, NOT False): task3_persistence.py's
    # own run_all() (Table 8.2, full cohort) needs apply_merge6=False
    # because that table's own real-data validation only reproduces
    # against the true 7-class rq3_process_label vocabulary (see that
    # module's "REPRODUCIBILITY NOTE"). It is tempting to assume Table
    # 8.8's lower block -- described as "rerunning" that same Appendix A
    # pipeline on a 5-group subset -- needs the same apply_merge6=False.
    # Real-data validation shows that assumption is WRONG for this table:
    # with apply_merge6=False the naive-5 history-example table has 118
    # transitions, but docs/thesis_reproduction_targets.md's own prose
    # states the naive subset has "111 transitions in 942 ... examples" --
    # a number reproduced bit-exactly only with apply_merge6=True (6-class
    # MERGE6 collapse), which also reproduces 3 of 5 groups'
    # repeat_current_all_windows macro-F1 exactly (G3, G5, G6), vs. 0 of 5
    # exact under apply_merge6=False. See
    # data/external/thesis_data/PUBLICATION_TASK3_CORRECTED_FINAL/
    # run_table_8_8_naive5_validation.py and
    # docs/table_to_source_mapping.md's "naive-5 rerun" row for the full
    # A/B comparison. So Table 8.8's lower block apparently used the
    # 6-class vocabulary despite Table 8.2 (the full-cohort version of the
    # same pipeline) needing the 7-class one -- an inconsistency in the
    # original thesis computation, not a bug in this port.
    labels_df, _, _, label_col = task3_persistence.load_normalized_labels(data_root, apply_merge6=True)
    data, group_col, time_col, _ = task3_persistence.select_and_merge_feature_file(data_root, labels_df, label_col, apply_merge6=True)

    naive_mask = data[group_col].map(_gid).isin(NAIVE_GROUPS)
    data_naive = data.loc[naive_mask].reset_index(drop=True)
    print(f"windows: {len(data)} -> {len(data_naive)} (naive-5 subset) | groups: {sorted(data_naive[group_col].map(_gid).unique())}")

    all_labels = sorted(data_naive[label_col].unique())

    # No sensor features needed for Table 8.8's lower block (label-only
    # persistence rows), so skip detect_numeric_sensor_features entirely.
    ex = task3_persistence.make_history_examples(data_naive, group_col, time_col, label_col, history_len=1, sensor_cols=[], include_sensor_features=False)
    print(f"history examples (h=1): {len(ex)} | transitions: {int(ex['is_transition'].sum())}")

    _, predictions = task3_persistence.run_logo_history_experiments({1: ex}, all_labels, sensor_cols_to_use=[])
    predictions["group_id"] = predictions["group"].map(_gid)

    def _per_group_macro_f1(mask, model_col):
        # Bug fix (2026-09-06, real-data validation): each held-out group's
        # macro-F1 must be scored against that GROUP'S OWN locally-occurring
        # label set (the union of its y_true and its predictions) -- NOT the
        # `all_labels` vocabulary pooled across all 5 naive groups. Passing
        # the pooled vocabulary silently scores 0 for any class this
        # specific group never actually has (e.g. G2 never has "merging",
        # G10 never has "inspection"), which artificially drags that
        # group's macro-F1 down even though its own predictions are
        # correct. Confirmed empirically: G2's repeat_current_all_windows
        # macro-F1 is 0.4877 with the local label set vs. 0.4065 with the
        # pooled one -- the published Table 8.8 value is 0.488, an exact
        # match only for the local computation (same for G10: local=0.7588
        # vs. published 0.759, pooled=0.6324). G3/G5/G6 already contain
        # every naive-5 class in their own windows, so local == pooled for
        # them and they were unaffected either way -- which is exactly why
        # only G2/G10 showed this discrepancy. This matches sklearn's own
        # f1_score default (average="macro" with no explicit `labels`
        # auto-derives the label set from that call's own y_true/y_pred),
        # i.e. the natural per-group implementation, rather than the
        # cross-group-consistent `labels=all_labels` this function
        # previously (incorrectly) reused from the pooled full-cohort
        # scoring convention. See docs/table_to_source_mapping.md's
        # "naive-5 rerun" row for the full A/B evidence.
        subset = predictions.loc[mask]
        out = {}
        for g, gdf in subset.groupby("group_id"):
            local_labels = sorted(set(gdf["y_true"].astype(str).unique()) | set(gdf[model_col].astype(str).unique()))
            out[g] = task3_persistence.score_predictions(gdf["y_true"], gdf[model_col], model_col, "n/a", 1, local_labels)["macro_f1"]
        return out

    all_mask = np.ones(len(predictions), dtype=bool)
    transition_mask = predictions["is_transition"].to_numpy(dtype=bool)

    rows_spec = [
        ("repeat_current_all_windows", all_mask, "repeat_current_label"),
        ("no_self_ngram_transitions", transition_mask, "ngram_markov_no_self_backoff_h1"),
        ("repeat_current_transitions", transition_mask, "repeat_current_label"),
    ]

    out_rows = []
    for name, mask, model_col in rows_spec:
        per_group = _per_group_macro_f1(mask, model_col)
        values = np.array([per_group[g] for g in sorted(NAIVE_GROUPS) if g in per_group], dtype=float)
        mean, std, lo, hi = _mean_std_ci(values) if len(values) > 1 else (float(values.mean()) if len(values) else np.nan, 0.0, np.nan, np.nan)
        out_rows.append({
            "configuration": name,
            **{f"G{g}": per_group.get(g, np.nan) for g in sorted(NAIVE_GROUPS)},
            "mean": mean, "std": std, "ci_lo": lo, "ci_hi": hi, "n_groups": len(values),
        })

    persistence_df = pd.DataFrame(out_rows)
    os.makedirs(out_dir, exist_ok=True)
    persistence_df.to_csv(os.path.join(out_dir, "task3_naive5_persistence.csv"), index=False)
    return persistence_df


def run_all(data_root: str, core_out: str, out_dir: str, hist_lens=(1, 2, 3, 5)):
    """Runs both blocks and assembles Table 8.8. Returns (combined_df, check_df).
    core_out is where task3_tokens.get_or_build_tokens looks for/writes the
    activity-token table (same as the rest of Task 3)."""
    from . import task3_tokens

    T, fcols = task3_tokens.get_or_build_tokens(data_root, core_out, resume_existing=True, verify_counts=False)
    grammar_df = run_naive5_grammar(T, fcols, out_dir, hist_lens=hist_lens)
    persistence_df = run_naive5_persistence(data_root, out_dir)

    combined = pd.concat([grammar_df, persistence_df], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(os.path.join(out_dir, "task3_naive5_table_8_8.csv"), index=False)

    print("\n" + "=" * 100)
    print("TABLE 8.8 — TASK 3 ON THE FIVE NAIVE GROUPS")
    print("=" * 100)
    print(combined.round(3).to_string(index=False))

    check_rows = []
    for config, ref in REPORT_REFERENCE.items():
        row = combined[combined["configuration"] == config]
        if row.empty:
            check_rows.append({"configuration": config, "match": "MISSING"})
            continue
        row = row.iloc[0]
        per_group_ok = all(
            pd.notna(row.get(f"G{g}", np.nan)) and abs(row.get(f"G{g}") - ref[f"G{g}"]) < 0.0015
            for g in sorted(NAIVE_GROUPS)
        )
        mean_ok = abs(row["mean"] - ref["mean"]) < 0.0015
        match = "EXACT" if (per_group_ok and mean_ok) else "DIFFERS"
        check_rows.append({
            "configuration": config, "mean": round(row["mean"], 3), "report_mean": ref["mean"],
            "std": round(row["std"], 3), "report_std": ref["std"], "match": match,
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.8")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "task3_naive5_report_reproduction_check.csv"), index=False)

    n_exact = int((check["match"] == "EXACT").sum())
    print(f"\n{n_exact} of {len(check)} referenced Table 8.8 rows reproduce exactly.")
    print(
        "\nNote: 'n-gram back-off, h=1' and 'no-self n-gram, transitions' are reported as "
        "numerically identical in the thesis — this run computes them independently (see module "
        "docstring) rather than assuming that; compare the two rows above yourself."
    )

    return combined, check
