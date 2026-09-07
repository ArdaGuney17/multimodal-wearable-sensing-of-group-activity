"""Feature engineering, Task 2 final merge — "B7. SAFE TASK 2 ADVANCED
MERGE" (CELL 17 of `master_feature_generator_task1_task2_task3_
CORRECTED_V4.ipynb`): the recipe that actually produces
`activity3_advanced_merged_10s_features.csv`.

**Merge recipe, as read directly from CELL 17 (not assumed):**
  1. Base = XSENS2 (`interaction_xsens2_10s.csv`), filtered to the 3 core
     recognition classes. Base columns kept: `group`, `window_start`,
     `window_end`, `recognition_label`, `elapsed_min` (if present), and
     every `xsens2_*` column — verbatim, no renaming.
  2. OE10 features (`interaction_oe10_10s.csv`) are NOT merged wholesale.
     Only a curated subset is selected:
       - `ear_*` and `oe_*` columns containing "acc_", "gyro_", "jerk",
         or "turn" (motion-related legacy + rich features).
       - `mag_*` columns containing "magnitude", "horizontal", or
         "mag_active".
     That subset is then renamed `{col: f"oe__{col}"}` (double
     underscore) and left-merged onto the XSENS2 base by a fuzzy
     rounded-key join (see `merge_features_left_safe` below) on
     `["group","window_start","window_end"]`.
  3. OPTI2 features (`interaction_opti2_10s.csv`) are merged wholesale:
     every `opti2_*` column, plus 3 specific legacy ENG7 proximity
     columns if present (`opti_nearest_pair_dist_mean`,
     `opti_all_pairs_dist_mean`, `opti_all_pairs_dist_std`) — no
     renaming this time. Same fuzzy rounded-key left-merge.
  4. `cell 8`'s ENG3 5s window/label grid is NOT a literal dependency of
     this merge — confirmed by reading this cell: it reads OE10/OPTI2/
     XSENS2 CSVs directly and never touches `interaction_eng3_features.csv`
     itself. (The 3-class recognition labels DO trace back to that
     lineage, but one level removed — through XSENS2/OPTI2's own
     `eng3_recognition_3class_core_features.csv` join, ported in
     `eng_task2_grid.py` — not through this merge cell.)

**The fuzzy rounded-key merge (`merge_features_left_safe`).** For each
candidate rounding precision in `[6, 5, 4, 3, 2, 1]` decimals, rounds
both sides' `window_start`/`window_end` to that many decimals and left-
merges on `(group, rounded_start, rounded_end)`; whichever precision
yields the most matched rows against the base is kept. This exists in
the source notebook as a defensive measure against float round-trip
noise across the 3 independently-built 10s window grids — in a from-
-raw-data rebuild where all 3 tables are built from the SAME base window
grid (`eng_task2_grid.build_task2_base_windows`), the 6-decimal round
should match every row exactly, but the search is kept verbatim (rather
than assuming that) so any discrepancy surfaces as a printed match count
instead of silently mismatching.
"""

from __future__ import annotations

import pandas as pd

CORE = ["co_building", "co_merging", "conversation"]


def unique_feats(feats: list[str]) -> list[str]:
    return list(dict.fromkeys(feats))


def add_merge_keys(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    out["_group_key"] = pd.to_numeric(out["group"], errors="coerce").astype(int)
    out["_ws_key"] = pd.to_numeric(out["window_start"], errors="coerce").round(decimals)
    out["_we_key"] = pd.to_numeric(out["window_end"], errors="coerce").round(decimals)
    return out


def merge_features_left_safe(base: pd.DataFrame, other: pd.DataFrame, other_feature_cols: list[str], label: str):
    """Verbatim port of CELL 17's `merge_features_left_safe`. Never drops
    base rows; tries rounding precisions 6..1 and keeps whichever matches
    the most base rows. Returns (merged_df, best_matched_count)."""
    other_feature_cols = unique_feats(other_feature_cols)
    other_small = other[["group", "window_start", "window_end"] + other_feature_cols].copy()

    best_merge = None
    best_round = None
    best_matched = -1

    for decimals in [6, 5, 4, 3, 2, 1]:
        a = add_merge_keys(base, decimals)
        b = add_merge_keys(other_small, decimals)

        b = b.drop_duplicates(subset=["_group_key", "_ws_key", "_we_key"], keep="first")
        b[f"_matched_{label}"] = 1

        merged_try = a.merge(
            b.drop(columns=["group", "window_start", "window_end"]),
            on=["_group_key", "_ws_key", "_we_key"],
            how="left",
        )

        matched = int(merged_try[f"_matched_{label}"].fillna(0).sum())
        print(f"{label} merge round={decimals} | matched={matched}/{len(base)}")

        if matched > best_matched:
            best_matched = matched
            best_round = decimals
            best_merge = merged_try

    out = best_merge.copy()
    helper_cols = ["_group_key", "_ws_key", "_we_key", f"_matched_{label}"]
    out = out.drop(columns=[c for c in helper_cols if c in out.columns])

    print(f"Best {label} merge rounding:", best_round)
    print(f"Matched {label}:", best_matched, "/", len(base))
    print("Shape after safe left merge:", out.shape)

    return out, best_matched


def select_oe_best_columns(oe10: pd.DataFrame) -> list[str]:
    """Verbatim port of CELL 17's OE feature-selection logic (the
    `oe_motion`/`mag_magnitude`/`oe_best_original` block)."""
    old_ear = [c for c in oe10.columns if c.startswith("ear_")]
    oe9_features = [c for c in oe10.columns if c.startswith("oe_")]
    mag_features = [c for c in oe10.columns if c.startswith("mag_")]

    oe_motion = [
        c for c in old_ear + oe9_features
        if ("acc_" in c or "gyro_" in c or "jerk" in c or "turn" in c)
    ]

    mag_magnitude = [
        c for c in mag_features
        if ("magnitude" in c or "horizontal" in c or "mag_active" in c)
    ]

    return unique_feats(oe_motion + mag_magnitude)


def create_safe_advanced_dataset(oe10: pd.DataFrame, opti2: pd.DataFrame, xsens2: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of CELL 17's `create_safe_advanced_dataset`, taking
    the 3 already-built feature tables as DataFrames instead of reading
    them from CSV paths."""
    print("=" * 100)
    print("CREATING SAFE ADVANCED 3-CLASS MERGED DATASET")
    print("=" * 100)

    oe10 = oe10.copy()
    opti2 = opti2.copy()
    xsens2 = xsens2.copy()

    print("Raw source shapes:")
    print("OE10:", oe10.shape)
    print("OPTI2:", opti2.shape)
    print("XSENS2:", xsens2.shape)

    if "recognition_label" not in xsens2.columns:
        raise KeyError("XSENS2 must contain recognition_label, but it does not.")
    if "recognition_label" not in opti2.columns:
        raise KeyError("OPTI2 must contain recognition_label, but it does not.")

    xsens2 = xsens2[xsens2["recognition_label"].isin(CORE)].dropna(subset=["recognition_label"]).reset_index(drop=True)
    opti2 = opti2[opti2["recognition_label"].isin(CORE)].dropna(subset=["recognition_label"]).reset_index(drop=True)

    if len(xsens2) == 0:
        raise ValueError("XSENS2 filtered to 0 rows. Cannot continue.")

    print("\nFiltered labelled base:")
    print("XSENS2:", xsens2.shape)
    print("OPTI2:", opti2.shape)
    print("\nXSENS2 class counts:")
    print(xsens2["recognition_label"].value_counts().to_string())

    base_cols = ["group", "window_start", "window_end", "recognition_label"]
    if "elapsed_min" in xsens2.columns:
        base_cols.append("elapsed_min")

    xsens_features = [c for c in xsens2.columns if c.startswith("xsens2_")]
    df_adv = xsens2[base_cols + xsens_features].copy()

    # ---------------- OE features ----------------
    oe_best_original = select_oe_best_columns(oe10)
    oe10_small = oe10[["group", "window_start", "window_end"] + oe_best_original].copy()
    oe_rename_map = {c: f"oe__{c}" for c in oe_best_original}
    oe10_small = oe10_small.rename(columns=oe_rename_map)
    oe_best = [oe_rename_map[c] for c in oe_best_original]

    print("\nOE feature count before merge:", len(oe_best))

    df_adv, oe_matched = merge_features_left_safe(base=df_adv, other=oe10_small, other_feature_cols=oe_best, label="oe10")

    # ---------------- OPTI2 features ----------------
    opti2_features = [c for c in opti2.columns if c.startswith("opti2_")]
    old_eng7_prox = [
        c for c in ["opti_nearest_pair_dist_mean", "opti_all_pairs_dist_mean", "opti_all_pairs_dist_std"]
        if c in opti2.columns
    ]
    opti_feature_cols = unique_feats(opti2_features + old_eng7_prox)

    print("\nOPTI2 feature count before merge:", len(opti_feature_cols))

    df_adv, opti_matched = merge_features_left_safe(base=df_adv, other=opti2, other_feature_cols=opti_feature_cols, label="opti2")

    df_adv = df_adv.drop(columns=[c for c in df_adv.columns if c.startswith("_")], errors="ignore")

    if len(df_adv) == 0:
        raise ValueError("Advanced dataset has 0 rows. Stop.")

    print("\n" + "=" * 100)
    print("SAFE ADVANCED 3-CLASS MERGED DATASET (in-memory)")
    print("=" * 100)
    print("Shape:", df_adv.shape)
    print("\nClass counts:")
    print(df_adv["recognition_label"].value_counts().to_string())
    print("\nRaw feature counts:")
    print("OE features:", len([c for c in df_adv.columns if c.startswith("oe__")]))
    print("OPTI2 features:", len([c for c in df_adv.columns if c.startswith("opti2_")]))
    print("XSENS2 features:", len([c for c in df_adv.columns if c.startswith("xsens2_")]))
    print("Elapsed present:", "elapsed_min" in df_adv.columns)
    print("\nMerge matches:")
    print("OE matched:", oe_matched, "/", len(xsens2))
    print("OPTI2 matched:", opti_matched, "/", len(xsens2))

    return df_adv
