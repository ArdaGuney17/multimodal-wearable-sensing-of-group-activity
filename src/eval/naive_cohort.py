"""Researcher-participant vs. naive-cohort descriptive comparison (Ch.10
§10.5, Table 10.1).

Ported from `thesis/after_GL/seven_winners_naive5.ipynb`, notebook cells
27-28 (raw ELAN loading + the `norm_label()` typo-robust label-category
mapper). Of the three `after_GL` notebooks investigated for this repo's
naive-cohort tables, this is the only one whose Table 10.1 computation is
**confirmed correct against the published numbers**: its own cell 27
(un-normalized) does NOT match Table 10.1 (e.g. G1 transitions=198 vs.
the published 134), but its cell 28 — which first maps the 280 raw ELAN
label strings onto 8 process categories via `norm_label()` before
counting transitions — reproduces every column of Table 10.1 exactly
(verified during porting: G1 session_min=30.5, transitions=134,
trans_per_min=4.39, dominant_share=0.607, all matching
docs/thesis_reproduction_targets.md's Table 10.1 verbatim). The source
notebook's own cell 28 never persisted this correct output to a file
(only printed to stdout) — this module fixes that.

INPUT: raw, headerless ELAN annotation exports, one per group, matched by
the `*_individual_build_renamed.csv` naming convention (9 columns: tier,
blank, begin_hms, begin_s, end_hms, end_s, duration_hms, duration_s,
label) — the same canonical per-group ELAN file selection used elsewhere
in this project's anonymization checklist (docs/data_provenance.md).
Session length and transition counts are computed directly from these
raw annotation boundaries, NOT from the 5-second modelling windows.

Note on "Transitions per min": this is the TOTAL transition count across
all annotation tiers, divided by session minutes (matches Table 10.1's
column directly) — not the same as the per-tier-normalized "changes/min"
figure quoted in the thesis's surrounding prose (§10.5), which further
divides by the tier count (7, or 6 for group 3).

Naive-cohort membership (NAIVE_GROUPS / RESEARCHER_GROUPS) matches
`src/models/common.py`'s NAIVE_GROUPS exactly — the 5 groups {2,3,5,6,10}
with no researcher participant, vs. the 4 groups {1,7,8,9} where the
thesis author served as third participant.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

NAIVE_GROUPS = {2, 3, 5, 6, 10}
RESEARCHER_GROUPS = {1, 7, 8, 9}

ELAN_COLS = ["tier", "blank", "beg_hms", "beg_s", "end_hms", "end_s", "dur_hms", "dur_s", "label"]

COMPARISON_METRICS = ["session_min", "n_norm_labels", "transitions", "trans_per_min", "entropy", "dominant_share"]


def find_group_elan_files(data_root: str) -> dict[int, str]:
    """One canonical file per group: `group_{g}/elan/*_individual_build_renamed.csv`,
    excluding any BACKUP variant."""
    files = sorted(glob.glob(os.path.join(data_root, "group_*", "elan", "*_individual_build_renamed.csv")))
    files = [f for f in files if "BACKUP" not in f]

    chosen: dict[int, str] = {}
    for f in files:
        m = re.search(r"group_(\d+)", f)
        if m:
            chosen[int(m.group(1))] = f
    return chosen


def load_annotations(data_root: str) -> pd.DataFrame:
    """Loads every group's raw ELAN export into one long DataFrame with
    `group`/`cohort` columns attached."""
    chosen = find_group_elan_files(data_root)
    if not chosen:
        raise FileNotFoundError(f"No *_individual_build_renamed.csv files found under {data_root}/group_*/elan/")

    frames = []
    for g, f in sorted(chosen.items()):
        d = pd.read_csv(f, header=None, names=ELAN_COLS)
        for c in ["beg_s", "end_s", "dur_s"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["beg_s", "end_s", "label"])
        d["group"] = g
        d["cohort"] = "researcher" if g in RESEARCHER_GROUPS else "naive"
        frames.append(d)

    ann = pd.concat(frames, ignore_index=True)
    ann["dur_s"] = ann["dur_s"].fillna(ann["end_s"] - ann["beg_s"])
    return ann


def norm_label(s) -> str:
    """Typo-robust regex mapper from the ~280 raw ELAN label strings onto
    8 process categories (+ "other"). This is the exact function that
    makes the downstream transition counts and dominant-share numbers
    match Table 10.1 — the un-normalized raw label text does not."""
    x = re.sub(r"[^a-z_+ ]", "", str(s).lower())
    x = x.split("+")[0].strip()  # keep primary of "A + B"

    if re.search(r"m[ei]r[gj]|merg", x):
        return "merging"
    if re.search(r"b[uıi]{1,2}ld|build", x) and "co_" in x:
        return "co_building"
    if re.search(r"b[uıi]{1,2}ld|build", x):
        return "individual_build"
    if re.search(r"handover|hand[iı]ver|handıver", x):
        return "object_handover"
    if re.search(r"convo|talk|dialou|conversation", x):
        return "conversation"
    if re.search(r"[ai]nsp|isnp|inscp|checking|matching|mathi|mathc", x):
        return "inspection"
    if re.search(r"trav|travel|moving|carry|cary|deliver|approach|pick|put|plac|search|lift", x):
        return "moving_transport"
    if re.search(r"sync|clap", x):
        return "sync_marker"
    return "other"


def compute_group_descriptives(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-group Table 10.1 row: session length, total cross-tier
    transitions (and per-minute rate), category-share entropy/dominant
    share, and individual-build time share — all computed on
    `norm_label()`-normalized labels."""
    ann = ann.copy()
    ann["norm"] = ann["label"].map(norm_label)

    rows = []
    for g, sub in ann.groupby("group"):
        session_s = float(sub["end_s"].max())

        transitions = 0
        for _, tier_sub in sub.groupby("tier"):
            labels = tier_sub.sort_values("beg_s")["norm"].values
            transitions += int((labels[1:] != labels[:-1]).sum())

        category_minutes = sub.groupby("norm")["dur_s"].sum() / 60
        category_share = category_minutes / category_minutes.sum()

        rows.append({
            "group": g,
            "cohort": "researcher" if g in RESEARCHER_GROUPS else "naive",
            "session_min": round(session_s / 60, 1),
            "n_norm_labels": sub["norm"].nunique(),
            "transitions": transitions,
            "trans_per_min": round(transitions / (session_s / 60), 2),
            "individual_build_pct": round(float(category_share.get("individual_build", 0.0) * 100), 1),
            "entropy": round(float(-(category_share * np.log2(category_share)).sum()), 3),
            "dominant_share": round(float(category_share.max()), 3),
        })

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_cohort_comparison(desc: pd.DataFrame, metrics=COMPARISON_METRICS) -> pd.DataFrame:
    """Two-sided Mann-Whitney U test per metric, researcher (n=4) vs.
    naive (n=5) cohorts. p-values are "for completeness only" (smallest
    attainable p at this sample size is 0.016) — same framing as the
    thesis text."""
    rows = []
    for m in metrics:
        a = desc.loc[desc["cohort"] == "researcher", m].astype(float)
        b = desc.loc[desc["cohort"] == "naive", m].astype(float)
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "metric": m, "researcher_mean": round(a.mean(), 3), "naive_mean": round(b.mean(), 3),
            "diff": round(a.mean() - b.mean(), 3), "U": u, "p": round(p, 3),
        })
    return pd.DataFrame(rows)


def run_all(data_root: str, out_dir: str | None = None):
    """Orchestrates the full Table 10.1 pipeline and saves
    group_descriptives_normalized.csv + cohort_comparison.csv (the
    source notebook's own correct-but-unsaved cell 28 output). Returns
    (desc, comparison)."""
    out_dir = out_dir or os.path.join(data_root, "TABLE_10_1_COHORT_COMPARISON")
    os.makedirs(out_dir, exist_ok=True)

    ann = load_annotations(data_root)
    print(f"annotations={len(ann)} groups={sorted(ann['group'].unique())}")

    desc = compute_group_descriptives(ann)
    print("\nTABLE 10.1 — PER-GROUP DESCRIPTIVES (normalized labels)")
    print(desc.to_string(index=False))

    comparison = compute_cohort_comparison(desc)
    print("\nCOHORT COMPARISON (researcher n=4 vs. naive n=5, descriptive only)")
    print(comparison.to_string(index=False))

    desc.to_csv(os.path.join(out_dir, "group_descriptives_normalized.csv"), index=False)
    comparison.to_csv(os.path.join(out_dir, "cohort_comparison.csv"), index=False)
    print("\nSaved:", out_dir)

    return desc, comparison
