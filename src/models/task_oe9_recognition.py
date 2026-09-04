"""Table 7.9 — Developed OpenEarable three-class results (Ch.7 §7.5.2).

Ported from `07_feature_engineering_ENG7_activity_invariant.ipynb`'s
CELLs 17/19/20/22 (`notebooks_reference/07_feature_engineering_ENG7_
activity_invariant_EXECUTED_CODE_ONLY.py`; see notebooks_reference/
07_feature_engineering_ENG7_activity_invariant_EXECUTED_ANALYSIS.md's
"Table 7.9 / 7.10 source mapping" section for the full cell-by-cell
trace). Builds all 4 feature sets and runs the fixed RBF-SVC/LOGO
evaluator every row of Table 7.9 uses:

  SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced",
      random_state=42)
  Pipeline: SimpleImputer(median) -> RobustScaler -> SVC (no SelectKBest
      — every row reports its full feature count, not a selected subset)
  LeaveOneGroupOut over the 9 study groups [1,2,3,5,6,7,8,9,10]
      (group 4 excluded — camera failure, documented elsewhere in this
      repo's docs).

Feature sets (CELL 17's `oe_best = unique_feats(motion + mag_magnitude)`
is the shared 305-feature base every other row extends):
  1. "OE9 motion + MAG magnitude" (305) — `oe_best` alone.
  2. "OE best + manual OptiTrack" (315) — + the 10 hand-picked
     `OPTITRACK_FEATURES` names (CELL 19), which are legacy ENG3 5s
     proximity/speed columns (`eng3_recognition_labels.py`'s
     `old_eng_features_and_tensor`), averaged onto each 10s OE10 window
     (`sub[c].mean()` over the ENG3 5s sub-windows inside it) and
     prefixed `opti_`.
  3. "OE best + ENG7 proximity" (308) — + the native 3-column `opti_*`
     proximity baseline from `eng7_proximity_features.py` (CELL 6),
     merged onto OE10 via a robust rounded-timestamp join (CELL 20's
     `add_merge_keys`, best-of-6-decimals-place match).
  4. "OE best + ENG7 proximity + elapsed" (309) — + a freshly computed,
     NON-normalized `elapsed_min` (CELL 22): minutes since that group's
     first ENG7 window, not clipped to [0,1], not using the session end.

INPUT DEPENDENCY: this module reads 3 already-built feature tables
(does not build them itself) — `eng3_recognition_labels.run_all`'s
`eng3_recognition_3class_core_features.csv`, `eng7_proximity_features.
run_all`'s `interaction_eng7_proximity_10s.csv`, and `oe9_oe10_features.
run_all`'s `interaction_oe10_10s.csv`. Run those three first (or pass
already-built DataFrames directly to `run_all`/the builder functions).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from src.features.oe_signal_common import clean_feature_list, unique_feats

CORE_CLASSES = ["co_building", "co_merging", "conversation"]
LOGO_GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group 4 excluded — camera failure

# The notebook's own hand-curated OptiTrack/spatial feature list (CELL 19)
# — the single most load-bearing artifact for faithful reproduction, per
# the source analysis doc. Copied verbatim; do not regenerate from a rule.
OPTITRACK_FEATURES = [
    "dist_close_mean",
    "dist_close_min",
    "dist_mid_mean",
    "dist_far_mean",
    "dist_disp_mean",
    "dist_disp_std",
    "speed_min",
    "speed_mid",
    "speed_max",
    "centroid_speed",
]

# Published values this module's output should reproduce (thesis Table
# 7.9 — docs/thesis_reproduction_targets.md §7.5.2). Used only for the
# optional verification check in check_against_reference().
REPORT_REFERENCE = {
    "OE9 motion + MAG magnitude": {"n_features": 305, "accuracy": 0.6340, "macro_f1": 0.5920, "balanced_accuracy": 0.6260},
    "OE best + manual OptiTrack": {"n_features": 315, "accuracy": 0.6620, "macro_f1": 0.6150, "balanced_accuracy": 0.6420},
    "OE best + ENG7 proximity": {"n_features": 308, "accuracy": 0.6710, "macro_f1": 0.6270, "balanced_accuracy": 0.6530},
    "OE best + ENG7 proximity + elapsed": {"n_features": 309, "accuracy": 0.6850, "macro_f1": 0.6400, "balanced_accuracy": 0.6640},
}


def make_svc_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42),
    )


# ================================================================
# CELL 17's shared `oe_best` definition
# ================================================================

def oe_best_features(df: pd.DataFrame) -> list[str]:
    """`oe_best = unique_feats(motion + mag_magnitude)` — CELLs 17/19/20/22
    all define this identically from an OE10-shaped DataFrame's own
    `ear_`/`oe_`/`mag_` columns."""
    old_ear = [c for c in df.columns if c.startswith("ear_")]
    oe9_features = [c for c in df.columns if c.startswith("oe_")]
    mag_features = [c for c in df.columns if c.startswith("mag_")]

    motion = [c for c in old_ear + oe9_features if ("acc_" in c or "gyro_" in c or "jerk" in c or "turn" in c)]
    mag_magnitude = [c for c in mag_features if ("magnitude" in c or "horizontal" in c or "mag_active" in c)]

    return unique_feats(motion + mag_magnitude)


# ================================================================
# Window <-> recognition-label joins (shared by CELLs 17/19/20/22)
# ================================================================

def _label_windows_by_midpoint(base_df: pd.DataFrame, rec: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """For each row of `base_df` (group/window_start/window_end), finds
    the `rec` rows whose window midpoint falls inside it, sets
    `recognition_label` to their majority vote, and (if `extra_cols` is
    given) also averages each of those columns into `opti_{col}` — CELL
    17's plain label join and CELL 19's label+opti-average join are the
    same operation, generalized here into one function."""
    extra_cols = extra_cols or []
    matched = []
    for _, w in base_df.iterrows():
        sub = rec[(rec["group"] == w["group"]) & (rec["mid"] >= w["window_start"]) & (rec["mid"] < w["window_end"])]
        out = {"group": w["group"], "window_start": w["window_start"], "window_end": w["window_end"]}
        if len(sub) == 0:
            out["recognition_label"] = np.nan
            for c in extra_cols:
                out[f"opti_{c}"] = np.nan
        else:
            out["recognition_label"] = sub["recognition_label"].mode().iat[0]
            for c in extra_cols:
                out[f"opti_{c}"] = pd.to_numeric(sub[c], errors="coerce").mean()
        matched.append(out)
    return pd.DataFrame(matched)


def _add_merge_keys(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    out["_group_key"] = out["group"].astype(int)
    out["_ws_key"] = out["window_start"].round(decimals)
    out["_we_key"] = out["window_end"].round(decimals)
    return out


def _robust_merge(eng7: pd.DataFrame, oe10_small: pd.DataFrame) -> pd.DataFrame:
    """CELL 20/22's `add_merge_keys` best-of-6-decimal-roundings merge:
    ENG7 and OE10 are independently built 10s window grids, so exact
    float equality on window_start/window_end isn't guaranteed — try
    rounding to 6..1 decimals and keep whichever matches the most rows."""
    best_merge, best_round, best_matched = None, None, -1
    for decimals in [6, 5, 4, 3, 2, 1]:
        e = _add_merge_keys(eng7, decimals)
        o = _add_merge_keys(oe10_small, decimals)
        o["_matched_oe10"] = 1
        merged_try = e.merge(o.drop(columns=["group", "window_start", "window_end"]), on=["_group_key", "_ws_key", "_we_key"], how="left")
        matched = int(merged_try["_matched_oe10"].fillna(0).sum())
        if matched > best_matched:
            best_matched, best_merge, best_round = matched, merged_try, decimals
    merged = best_merge[best_merge["_matched_oe10"] == 1].reset_index(drop=True)
    return merged


# ================================================================
# The 4 feature-set builders (one per Table 7.9 row)
# ================================================================

def build_row1_oe9_motion_mag(oe10: pd.DataFrame, rec: pd.DataFrame):
    """Row 1: "OE9 motion + MAG magnitude" (CELL 17)."""
    labels = _label_windows_by_midpoint(oe10[["group", "window_start", "window_end"]], rec)
    df = oe10.merge(labels, on=["group", "window_start", "window_end"], how="left")
    df = df[df["recognition_label"].isin(CORE_CLASSES)].dropna(subset=["recognition_label"]).reset_index(drop=True)
    feats = oe_best_features(df)
    return df, feats


def build_row2_manual_optitrack(oe10: pd.DataFrame, rec: pd.DataFrame):
    """Row 2: "OE best + manual OptiTrack" (CELL 19)."""
    available_opti = [c for c in OPTITRACK_FEATURES if c in rec.columns]
    labels = _label_windows_by_midpoint(oe10[["group", "window_start", "window_end"]], rec, extra_cols=available_opti)
    df = oe10.merge(labels, on=["group", "window_start", "window_end"], how="left")
    df = df[df["recognition_label"].isin(CORE_CLASSES)].dropna(subset=["recognition_label"]).reset_index(drop=True)

    oe_best = oe_best_features(df)
    opti_features = [f"opti_{c}" for c in available_opti]
    feats = unique_feats(oe_best + opti_features)
    return df, feats


def _label_eng7(eng7: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    labels = _label_windows_by_midpoint(eng7[["group", "window_start", "window_end"]], rec)
    out = eng7.merge(labels, on=["group", "window_start", "window_end"], how="left")
    return out[out["recognition_label"].isin(CORE_CLASSES)].dropna(subset=["recognition_label"]).reset_index(drop=True)


def _renamed_oe_best(oe10: pd.DataFrame):
    oe_best = oe_best_features(oe10)
    oe10_small = oe10[["group", "window_start", "window_end"] + oe_best].copy()
    rename_map = {c: f"oebest__{c}" for c in oe_best}
    oe10_small = oe10_small.rename(columns=rename_map)
    return oe10_small, [rename_map[c] for c in oe_best]


def build_row3_eng7_proximity(eng7: pd.DataFrame, oe10: pd.DataFrame, rec: pd.DataFrame):
    """Row 3: "OE best + ENG7 proximity" (CELL 20)."""
    eng7_labeled = _label_eng7(eng7, rec)
    eng7_opti = [c for c in eng7_labeled.columns if c.startswith("opti_")]

    oe10_small, oe_best_renamed = _renamed_oe_best(oe10)
    merged = _robust_merge(eng7_labeled, oe10_small)

    feats = unique_feats(oe_best_renamed + eng7_opti)
    return merged, feats


def build_row4_eng7_proximity_elapsed(eng7: pd.DataFrame, oe10: pd.DataFrame, rec: pd.DataFrame):
    """Row 4: "OE best + ENG7 proximity + elapsed" (CELL 22)."""
    eng7_labeled = _label_eng7(eng7, rec)
    eng7_opti = [c for c in eng7_labeled.columns if c.startswith("opti_")]

    eng7_labeled = eng7_labeled.copy()
    eng7_labeled["window_mid"] = (eng7_labeled["window_start"] + eng7_labeled["window_end"]) / 2.0
    group_start = eng7_labeled.groupby("group")["window_mid"].transform("min")
    eng7_labeled["elapsed_min"] = (eng7_labeled["window_mid"] - group_start) / 60.0

    oe10_small, oe_best_renamed = _renamed_oe_best(oe10)
    merged = _robust_merge(eng7_labeled, oe10_small)

    feats = unique_feats(oe_best_renamed + eng7_opti + ["elapsed_min"])
    return merged, feats


# ================================================================
# LOGO evaluator (fixed RBF-SVC, no SelectKBest)
# ================================================================

def run_svc_logo_eval(df: pd.DataFrame, feats: list[str], label_col: str = "recognition_label", group_col: str = "group", logo_groups: list[int] | None = None):
    """Fixed SVC(C=1, gamma='scale', rbf)/LOGO evaluator every Table 7.9
    row uses. Restricts to `logo_groups` (default LOGO_GROUPS, i.e. group
    4 excluded) before splitting. Returns (pooled_result_dict,
    fold_metrics_df) — pooled_result carries `accuracy`/`macro_f1`/
    `balanced_accuracy` (Table 7.9's A/M/B columns, pooled across all
    LOGO folds) plus `fold_accuracy_mean/std`, `fold_macro_f1_mean/std`
    (Table 7.9's "fold mean±SD (A/M)" column, ddof=1 across the 9
    per-fold scores — same pattern as task1.py's
    `run_exact_reproduction`)."""
    logo_groups = logo_groups if logo_groups is not None else LOGO_GROUPS
    df = df[df[group_col].isin(logo_groups)].reset_index(drop=True)

    used_feats = clean_feature_list(df, feats)
    X = df[used_feats].apply(pd.to_numeric, errors="coerce").values
    y = df[label_col].values
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    yt_all, yp_all, fold_rows = [], [], []

    for fold, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        test_group = groups[te][0]
        clf = make_svc_pipeline()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])

        yt_all.extend(y[te].tolist())
        yp_all.extend(pred.tolist())
        fold_rows.append({
            "fold": fold, "test_group": test_group, "n_test": len(te),
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
        })

    yt_all, yp_all = np.array(yt_all), np.array(yp_all)
    fold_df = pd.DataFrame(fold_rows)

    result = {
        "n_features": len(used_feats),
        "accuracy": accuracy_score(yt_all, yp_all),
        "macro_f1": f1_score(yt_all, yp_all, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yt_all, yp_all),
        "fold_accuracy_mean": fold_df["accuracy"].mean(), "fold_accuracy_std": fold_df["accuracy"].std(ddof=1),
        "fold_macro_f1_mean": fold_df["macro_f1"].mean(), "fold_macro_f1_std": fold_df["macro_f1"].std(ddof=1),
        "n_folds": fold_df["test_group"].nunique(),
    }
    return result, fold_df


def check_against_reference(results: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """Verification check against docs/thesis_reproduction_targets.md
    Table 7.9 — mirrors `naive_cohort_sensitivity.check_against_
    reference`'s pattern. A looser tolerance than task1.py's exact-
    reproduction checks (0.02, not 0.0006) since real-data numbers here
    also depend on the not-yet-independently-verified ENG3/ENG7/OE9/OE10
    feature-generation chain, not just the fixed SVC."""
    rows = []
    r = results.set_index("condition") if len(results) else results
    for condition, ref in REPORT_REFERENCE.items():
        if not len(results) or condition not in r.index:
            rows.append({"condition": condition, "match": "MISSING"})
            continue
        row = r.loc[condition]
        match = "EXACT" if all(abs(row[k] - ref[k]) < tol for k in ("accuracy", "macro_f1", "balanced_accuracy")) else "DIFFERS"
        rows.append({
            "condition": condition, "n_features": int(row["n_features"]), "report_n_features": ref["n_features"],
            "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"],
            "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"],
            "balanced_accuracy": round(row["balanced_accuracy"], 4), "report_balanced_accuracy": ref["balanced_accuracy"],
            "match": match,
        })
    return pd.DataFrame(rows)


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_root: str, out_dir: str | None = None, eng3=None, eng7=None, oe10=None):
    """Runs all 4 Table 7.9 rows. Reads `eng3_recognition_3class_core_
    features.csv`, `interaction_eng7_proximity_10s.csv`, and
    `interaction_oe10_10s.csv` from their default `run_all()` output
    locations unless already-built DataFrames are passed in directly
    (used by the smoke test to skip disk round-trips). Returns
    (results_df, check_df, fold_metrics_by_condition)."""
    out_dir = out_dir or os.path.join(data_root, "PUBLICATION_TABLE_7_9")
    os.makedirs(out_dir, exist_ok=True)

    if eng3 is None:
        eng3_path = os.path.join(data_root, "INTERACTION_ENG3", "eng3_recognition_3class_core_features.csv")
        if not os.path.exists(eng3_path):
            raise FileNotFoundError(f"ENG3 recognition-label table not found:\n{eng3_path}\nRun eng3_recognition_labels.run_all() first.")
        eng3 = pd.read_csv(eng3_path)
    if eng7 is None:
        eng7_path = os.path.join(data_root, "INTERACTION_ENG7", "interaction_eng7_proximity_10s.csv")
        if not os.path.exists(eng7_path):
            raise FileNotFoundError(f"ENG7 proximity table not found:\n{eng7_path}\nRun eng7_proximity_features.run_all() first.")
        eng7 = pd.read_csv(eng7_path)
    if oe10 is None:
        oe10_path = os.path.join(data_root, "INTERACTION_OE10", "interaction_oe10_10s.csv")
        if not os.path.exists(oe10_path):
            raise FileNotFoundError(f"OE10 table not found:\n{oe10_path}\nRun oe9_oe10_features.run_all() first.")
        oe10 = pd.read_csv(oe10_path)

    rec = eng3[eng3["recognition_label"].isin(CORE_CLASSES)].copy()
    rec["mid"] = (rec["window_start"] + rec["window_end"]) / 2.0

    result_rows = []
    fold_metrics = {}

    df1, feats1 = build_row1_oe9_motion_mag(oe10, rec)
    r1, f1df = run_svc_logo_eval(df1, feats1)
    result_rows.append({"condition": "OE9 motion + MAG magnitude", **r1})
    fold_metrics["OE9 motion + MAG magnitude"] = f1df

    df2, feats2 = build_row2_manual_optitrack(oe10, rec)
    r2, f2df = run_svc_logo_eval(df2, feats2)
    result_rows.append({"condition": "OE best + manual OptiTrack", **r2})
    fold_metrics["OE best + manual OptiTrack"] = f2df

    df3, feats3 = build_row3_eng7_proximity(eng7, oe10, rec)
    r3, f3df = run_svc_logo_eval(df3, feats3)
    result_rows.append({"condition": "OE best + ENG7 proximity", **r3})
    fold_metrics["OE best + ENG7 proximity"] = f3df

    df4, feats4 = build_row4_eng7_proximity_elapsed(eng7, oe10, rec)
    r4, f4df = run_svc_logo_eval(df4, feats4)
    result_rows.append({"condition": "OE best + ENG7 proximity + elapsed", **r4})
    fold_metrics["OE best + ENG7 proximity + elapsed"] = f4df

    results = pd.DataFrame(result_rows)
    print("\nTABLE 7.9 — DEVELOPED OPENEARABLE THREE-CLASS RESULTS")
    print(results.round(4).to_string(index=False))

    check = check_against_reference(results)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 7.9")
    print(check.to_string(index=False))

    results.to_csv(os.path.join(out_dir, "table_7_9_summary.csv"), index=False)
    check.to_csv(os.path.join(out_dir, "table_7_9_report_reproduction_check.csv"), index=False)
    for condition, fdf in fold_metrics.items():
        safe_name = condition.lower().replace(" ", "_").replace("+", "plus")
        fdf.to_csv(os.path.join(out_dir, f"table_7_9_fold_metrics_{safe_name}.csv"), index=False)

    return results, check, fold_metrics
