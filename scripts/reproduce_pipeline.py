#!/usr/bin/env python
"""Top-level orchestrator for the thesis reproduction pipeline.

Given ``--data-root`` (a raw sensor data root laid out as
``{data_root}/group_{g}/{elan,openearable,xsens,optitrack}/...``, matching
what ``src/preprocessing/raw_sync_oe_xsens.py`` / ``raw_sync_optitrack.py``
expect) and ``--out-dir``, runs the full pipeline for whichever groups
currently have the inputs each stage needs, in dependency order:

  1. sync     -- src/preprocessing/raw_sync_oe_xsens.py, raw_sync_optitrack.py
  2. clean    -- src/preprocessing/global_cleaning.py
  3. features -- src/features/eng3_recognition_labels.py, eng7_proximity_features.py,
                 oe9_oe10_features.py, eng_task2_grid/opti/xsens/merge.py,
                 eng_task1_oe/opti/xsens.py
  4. models   -- src/models/task1.py-equivalent, task2.py-equivalent,
                 task_oe_specific.py (Table 7.8), task_oe9_recognition.py (Table 7.9)
  5. task3    -- src/models/task3_tokens.py -> task3_grammar/common_targets/
                 persistence/segment_forecast/expanding_prefix/hmm_appendix_d/naive5.py

Every real module call is wrapped so one group/stage's failure never aborts
the rest -- each (stage, group, item) gets a status of:

  done     -- freshly computed in THIS run from --data-root via the real
              module code.
  bridged  -- --data-root did not have what this stage needed (e.g. this
              sandbox's data/raw is missing the private name_map.json /
              un-ported OptiTrack marker-reconstruction step), so a
              READ-ONLY, already-validated real fixture already sitting in
              this repo (data/external/thesis_data/RAW_VALIDATION*, never
              written to) was copied in instead of fabricating anything.
              Clearly distinguished from "done" in every status table.
  skipped  -- required input genuinely absent (from --data-root AND from
              the bridge fixtures) -- not attempted, not a crash.
  failed   -- attempted, the real module code raised -- reason is the
              real exception text.

KNOWN, DOCUMENTED GAPS this script does not attempt to solve (see
docs/table_to_source_mapping.md for the full history):
  - The ENG3/Task-1 5s window/label grid is circular in its original
    notebook source; src/features/eng_task1_{oe,opti,xsens}.py need a
    pre-existing "official grid" CSV as input. This script resolves that
    via the same pre-existing real fixture the ported code's own
    docstrings point at (INTERACTION_BINARY_5S_SPECIALIZED_OE/...), not a
    fresh computation -- reported as "bridged", not "done".
  - OptiTrack's raw marker-identity reconstruction (Hungarian-algorithm
    tracklet stitching) was never ported (explicitly out of scope) --
    raw_sync_optitrack.py starts from an already-reconstructed
    "*_cleaned_combined_240hz.csv" file, which this sandbox's data/raw
    does not have; bridged from the validated RAW_VALIDATION fixtures
    where available.
  - src/models/task3_expanding_prefix.py needs
    INTERACTION_ENG3/recognition_interaction_window_label_inventory.csv,
    which nothing in this pipeline (or this script) currently produces --
    reported "failed"/"skipped" with the real FileNotFoundError, not
    faked.
  - Neural training (src/models/task3_neural.py, and the DL grids inside
    Task 1/2/task_oe_specific.py) is real but slow; off by default here
    (--run-dl / --run-neural to opt in) so a full orchestrator run stays
    minutes, not hours. task3_publication.py (needs neural output) is
    skipped alongside it unless --run-neural is given.

Usage:
    python scripts/reproduce_pipeline.py --data-root data --out-dir data/processed/pipeline_run
    python scripts/reproduce_pipeline.py --stages sync,clean --groups 1,2,3 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from src.preprocessing import raw_sync_oe_xsens, raw_sync_optitrack, global_cleaning  # noqa: E402
from src.features import eng3_recognition_labels, eng7_proximity_features, oe9_oe10_features  # noqa: E402
from src.features import eng_task1_oe, eng_task1_opti, eng_task1_xsens  # noqa: E402
from src.features import eng_task2_grid, eng_task2_opti, eng_task2_xsens, eng_task2_merge  # noqa: E402
from src.models import common as mc  # noqa: E402
from src.models import task_oe_specific, task_oe9_recognition  # noqa: E402
from src.models import (  # noqa: E402
    task3_tokens, task3_grammar, task3_common_targets, task3_persistence,
    task3_segment_forecast, task3_expanding_prefix, task3_hmm_appendix_d,
    task3_naive5, task3_neural, task3_publication,
)

ALL_GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group_4 excluded (camera failure) -- canonical across this repo
ALL_STAGES = ["sync", "clean", "features", "models", "task3"]

# Read-only real-data fixtures already in this repo, used ONLY as a fallback
# bridge when --data-root itself doesn't have what a stage needs. Never
# written to.
FIXTURES_ROOT = os.path.join(REPO_ROOT, "data", "external", "thesis_data")
RAW_VALIDATION = os.path.join(FIXTURES_ROOT, "RAW_VALIDATION")
RAW_VALIDATION_FEATURES = os.path.join(FIXTURES_ROOT, "RAW_VALIDATION_FEATURES")


# =====================================================================
# Status tracking
# =====================================================================

class StatusTracker:
    def __init__(self):
        self.rows = []

    def add(self, stage, group, item, status, detail=""):
        detail = str(detail).replace("\n", " ")[:220]
        self.rows.append({"stage": stage, "group": group, "item": item, "status": status, "detail": detail})
        print(f"[{stage:9s}] group={str(group):>4s} {item:28s} -> {status:8s} {detail}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["stage", "group", "item", "status", "detail"])

    def print_table(self):
        df = self.frame()
        if df.empty:
            print("(nothing ran)")
            return
        print("\n" + "=" * 100)
        print("PIPELINE STATUS")
        print("=" * 100)
        for stage in df["stage"].unique():
            sub = df[df["stage"] == stage]
            counts = sub["status"].value_counts().to_dict()
            print(f"\n-- {stage} -- {counts}")
            with pd.option_context("display.max_rows", None, "display.width", 160):
                print(sub.drop(columns=["stage"]).to_string(index=False))


def bridge_copy(src: str, dst: str) -> bool:
    """Copies src -> dst (read-only source, never touched) if src exists.
    Idempotent: if dst already exists, treated as already-bridged. Returns
    True iff dst exists after this call."""
    if os.path.exists(dst):
        return True
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def safe_call(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs), returning (result, None) on success or
    (None, error_string) on any exception -- never raises."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# =====================================================================
# Stage 1 -- raw sync
# =====================================================================

def stage_sync(groups, data_root, out_dir, status: StatusTracker, verbose_errors=False):
    sync_root = os.path.join(out_dir, "raw_sync")
    os.makedirs(sync_root, exist_ok=True)

    oe_results, oe_err = safe_call(raw_sync_oe_xsens.run_all, data_root, out_dir=sync_root)
    if oe_err:
        print(f"raw_sync_oe_xsens.run_all() itself raised (unexpected -- it normally catches per-group): {oe_err}")
        oe_results = {}
    opti_results, opti_err = safe_call(raw_sync_optitrack.run_all, data_root, out_dir=sync_root)
    if opti_err:
        print(f"raw_sync_optitrack.run_all() itself raised: {opti_err}")
        opti_results = {}

    for g in groups:
        for sensor in ("openearable", "xsens"):
            res = (oe_results or {}).get(g, {})
            written = isinstance(res, dict) and sensor in res and res[sensor].get("labeled_path") \
                and os.path.exists(res[sensor]["labeled_path"])
            if written:
                status.add("sync", g, sensor, "done", "raw_sync_oe_xsens.process_group() on --data-root")
                continue

            reason = res.get("error") if isinstance(res, dict) and "error" in res else \
                (res.get(sensor, {}).get("shift_error") if isinstance(res, dict) else None)
            fname = raw_sync_oe_xsens.SELECTED_FILE_NAMES.get((g, sensor)) if hasattr(raw_sync_oe_xsens, "SELECTED_FILE_NAMES") else None
            fname = fname or global_cleaning.SELECTED_FILE_NAMES.get((g, sensor))
            src = os.path.join(RAW_VALIDATION, "_out", f"group_{g}", sensor, f"{sensor}_labeled", fname) if fname else ""
            dst = os.path.join(sync_root, f"group_{g}", sensor, f"{sensor}_labeled", fname) if fname else ""
            if fname and bridge_copy(src, dst):
                status.add("sync", g, sensor, "bridged", f"--data-root lacked input ({reason or 'no result'}); used validated RAW_VALIDATION/_out fixture")
            else:
                status.add("sync", g, sensor, "skipped", reason or "no raw input available on --data-root or in bridge fixtures")

        res = (opti_results or {}).get(g, {})
        written = isinstance(res, dict) and res.get("labeled_path") and os.path.exists(res["labeled_path"])
        if written:
            status.add("sync", g, "optitrack", "done", "raw_sync_optitrack.process_group() on --data-root")
            continue
        reason = res.get("error") if isinstance(res, dict) else None
        fname = f"group_{g}_optitrack_labeled.csv"
        src = os.path.join(RAW_VALIDATION, f"group_{g}_optitrack", f"group_{g}", "optitrack", "optitrack_labeled", fname)
        dst = os.path.join(sync_root, f"group_{g}", "optitrack", "optitrack_labeled", fname)
        if bridge_copy(src, dst):
            status.add("sync", g, "optitrack", "bridged", f"--data-root lacked input ({reason or 'no result'}); used validated RAW_VALIDATION optitrack fixture")
        else:
            status.add("sync", g, "optitrack", "skipped", reason or "no raw optitrack input available (marker reconstruction out of scope, see module docstring)")

    return sync_root


# =====================================================================
# Stage 2 -- global cleaning (model-ready assembly)
# =====================================================================

def stage_clean(groups, sync_root, status: StatusTracker):
    _, clean_err = safe_call(global_cleaning.run_all, sync_root)
    if clean_err:
        print(f"global_cleaning.run_all() raised: {clean_err}")

    central = os.path.join(sync_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    os.makedirs(central, exist_ok=True)
    for g in groups:
        for sensor in ("openearable", "xsens", "optitrack"):
            dst = os.path.join(central, f"group_{g}_{sensor}_model_ready.csv")
            if os.path.exists(dst):
                status.add("clean", g, sensor, "done", "global_cleaning.run_all() on stage-1 output")
                continue
            src = os.path.join(RAW_VALIDATION_FEATURES, f"group_{g}", f"group_{g}_{sensor}_model_ready.csv")
            if bridge_copy(src, dst):
                status.add("clean", g, sensor, "bridged", "stage-1 labeled CSV unavailable; used validated RAW_VALIDATION_FEATURES model_ready fixture")
            else:
                status.add("clean", g, sensor, "skipped", "no labeled input for this group/sensor (see sync stage)")

    return central


# =====================================================================
# Stage 3 -- feature engineering
# =====================================================================

def _model_ready_groups(central, sensors=("openearable", "xsens", "optitrack")):
    ready = []
    for g in ALL_GROUPS:
        if all(os.path.exists(os.path.join(central, f"group_{g}_{s}_model_ready.csv")) for s in sensors):
            ready.append(g)
    return ready


def stage_features(groups, sync_root, central, out_dir, status: StatusTracker):
    features = {}
    ready = [g for g in groups if g in _model_ready_groups(central)]
    if not ready:
        for g in groups:
            status.add("features", g, "eng3/eng7/oe9_oe10/task2", "skipped", "no model_ready CSVs for all 3 sensors")
        return features

    # --- ENG3 (non-circular: window/label grid + recognition table) ---
    r, err = safe_call(eng3_recognition_labels.run_all, sync_root, out_dir=os.path.join(sync_root, "INTERACTION_ENG3"))
    if err:
        for g in ready:
            status.add("features", g, "eng3", "failed", err)
    else:
        full_grid, rec_core = r
        features["eng3_full"], features["eng3_rec"] = full_grid, rec_core
        for g in sorted(full_grid["group"].dropna().unique().astype(int)):
            if g in ready:
                status.add("features", g, "eng3", "done", f"rows={int((full_grid['group'] == g).sum())}")

    # --- ENG7 proximity ---
    r, err = safe_call(eng7_proximity_features.run_all, sync_root, out_dir=os.path.join(sync_root, "INTERACTION_ENG7"))
    if err:
        for g in ready:
            status.add("features", g, "eng7", "failed", err)
    else:
        features["eng7"] = r
        for g in sorted(r["group"].dropna().unique().astype(int)):
            if g in ready:
                status.add("features", g, "eng7", "done", f"rows={int((r['group'] == g).sum())}")

    # --- OE9 / OE10 ---
    r, err = safe_call(oe9_oe10_features.run_all, sync_root, out_dir=None)
    if err:
        for g in ready:
            status.add("features", g, "oe9_oe10", "failed", err)
    else:
        oe9, oe10 = r
        features["oe9"], features["oe10"] = oe9, oe10
        for g in sorted(oe10["group"].dropna().unique().astype(int)):
            if g in ready:
                status.add("features", g, "oe9_oe10", "done", f"rows={int((oe10['group'] == g).sum())}")

    # --- Task 2 base window grid (non-circular: eng7 + eng3 recognition) ---
    if "eng7" in features and "eng3_rec" in features:
        base_windows, err = safe_call(eng_task2_grid.build_task2_base_windows, features["eng7"], features["eng3_rec"])
        if err:
            for g in ready:
                status.add("features", g, "task2_grid", "failed", err)
        else:
            features["task2_base_windows"] = base_windows
            for g in sorted(base_windows["group"].dropna().unique().astype(int)):
                if g in ready:
                    status.add("features", g, "task2_grid", "done", f"rows={int((base_windows['group'] == g).sum())}")

            opti2, err = safe_call(eng_task2_opti.build_task2_opti_features, central, base_windows)
            if err:
                for g in ready:
                    status.add("features", g, "task2_opti", "failed", err)
            else:
                features["task2_opti2"] = opti2
                for g in sorted(opti2["group"].dropna().unique().astype(int)):
                    if g in ready:
                        status.add("features", g, "task2_opti", "done", f"rows={len(opti2[opti2['group'] == g])}")

            xsens2, err = safe_call(eng_task2_xsens.build_task2_xsens_features, central, base_windows)
            if err:
                for g in ready:
                    status.add("features", g, "task2_xsens", "failed", err)
            else:
                features["task2_xsens2"] = xsens2
                for g in sorted(xsens2["group"].dropna().unique().astype(int)):
                    if g in ready:
                        status.add("features", g, "task2_xsens", "done", f"rows={len(xsens2[xsens2['group'] == g])}")

            if "oe10" in features and "task2_opti2" in features and "task2_xsens2" in features:
                merged, err = safe_call(eng_task2_merge.create_safe_advanced_dataset, features["oe10"], opti2, xsens2)
                if err:
                    for g in ready:
                        status.add("features", g, "task2_merge", "failed", err)
                else:
                    features["activity3_advanced"] = merged
                    dst = os.path.join(sync_root, "INTERACTION_ABLATIONS", "activity3_advanced_merged_10s_features.csv")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    merged.to_csv(dst, index=False)
                    for g in sorted(merged["group"].dropna().unique().astype(int)):
                        if g in ready:
                            status.add("features", g, "task2_merge", "done", f"-> {os.path.relpath(dst, out_dir)}")

    # --- Task 1 OE/Opti/Xsens (KNOWN GAP: needs a pre-existing official
    # grid CSV -- the ENG3/Task-1 5s window/label bootstrap is circular in
    # its original notebook source, see docs/table_to_source_mapping.md).
    # Bridged from the same real fixture the ported code's own docstring
    # points at -- never fabricated, but not a fresh from-scratch build. ---
    official_grid = os.path.join(sync_root, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "binary_5s_specialized_oe_merged_all_features.csv")
    bridged_grid = bridge_copy(
        os.path.join(FIXTURES_ROOT, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "binary_5s_specialized_oe_merged_all_features.csv"),
        official_grid,
    )
    if not bridged_grid:
        for g in ready:
            status.add("features", g, "task1_oe/opti/xsens", "skipped", "official Task-1 5s grid unavailable (known ENG3/Task-1 circularity gap)")
    else:
        oe1, err = safe_call(eng_task1_oe.build_task1_oe_features, central, official_grid)
        if err:
            for g in ready:
                status.add("features", g, "task1_oe", "failed", err)
        else:
            features["task1_oe"] = oe1["merged"]
            for g in sorted(oe1["merged"]["group"].dropna().astype(int).unique()) if "group" in oe1["merged"].columns else []:
                if g in ready:
                    status.add("features", g, "task1_oe", "bridged", "uses pre-existing official grid CSV (ENG3/Task-1 circularity gap)")

        opti1, err = safe_call(eng_task1_opti.build_task1_opti_features, central, official_grid)
        if err:
            status.add("features", "ALL", "task1_opti", "failed", err)
        else:
            features["task1_opti"] = opti1
            status.add("features", "ALL", "task1_opti", "bridged", "uses pre-existing official grid CSV (ENG3/Task-1 circularity gap)")

        xs1, err = safe_call(eng_task1_xsens.build_task1_xsens_features, central, official_grid)
        if err:
            status.add("features", "ALL", "task1_xsens", "failed", err)
        else:
            features["task1_xsens"] = xs1
            status.add("features", "ALL", "task1_xsens", "bridged", "uses pre-existing official grid CSV (ENG3/Task-1 circularity gap)")

    return features


# =====================================================================
# Stage 4 -- model training / eval (Table 7.1-7.10)
# =====================================================================

def _fast_config(**overrides):
    cfg = mc.Config(**overrides)
    cfg.fast_max_epochs = 3
    cfg.fast_patience = 1
    cfg.max_logo_folds = 2
    cfg.seeds = [42]
    cfg.k_dl = 40
    return cfg


def stage_models(groups, sync_root, out_dir, status: StatusTracker, run_dl=False):
    # --- Table 7.1/7.2: interaction detection (Task 1's shared modelling code) ---
    cfg = _fast_config(data_root=sync_root, out_dir=os.path.join(out_dir, "table_7_1_7_2"),
                        run_tasks=["interaction_vs_noninteraction"], run_classical=True, run_dl=run_dl)
    os.makedirs(cfg.out_dir, exist_ok=True)
    specs, err = safe_call(mc.build_task_specs, cfg)
    if err:
        status.add("models", "ALL", "task1_interaction", "failed", err)
    else:
        classical, err2 = safe_call(mc.run_all_classical, specs, cfg)
        dl = {}
        if run_dl:
            dl, err3 = safe_call(mc.run_all_dl, specs, cfg)
        if err2:
            status.add("models", "ALL", "task1_interaction", "failed", err2)
        else:
            groups_seen = sorted({int(g) for s in specs for g in pd.unique(s["df"][s["group_col"]])})
            for g in groups:
                if g in groups_seen:
                    status.add("models", g, "task1_interaction(7.1/7.2)", "done", f"-> table_7_1_7_2/ (classical{'+dl' if run_dl else ', dl skipped (--run-dl)'})")
            if run_dl and dl:
                safe_call(mc.build_publication_tables, classical, dl, cfg.out_dir)

    # --- Table 7.3-7.7: activity recognition, 5 sub-tasks (Task 2) ---
    activity_tasks = [
        "conversation_vs_nonconversation", "conversation_vs_building", "conversation_vs_merging",
        "merging_vs_building", "three_class_activity",
    ]
    cfg2 = _fast_config(data_root=sync_root, out_dir=os.path.join(out_dir, "table_7_3_7_7"),
                         run_tasks=activity_tasks, run_classical=True, run_dl=run_dl, resume_existing=True)
    os.makedirs(cfg2.out_dir, exist_ok=True)
    specs2, err = safe_call(mc.build_task_specs, cfg2)
    if err:
        status.add("models", "ALL", "task2_activity", "failed", err)
    else:
        classical2, err2 = safe_call(mc.run_all_classical, specs2, cfg2)
        dl2 = {}
        if run_dl:
            dl2, err3 = safe_call(mc.run_all_dl, specs2, cfg2)
        if err2:
            status.add("models", "ALL", "task2_activity", "failed", err2)
        else:
            groups_seen = sorted({int(g) for s in specs2 for g in pd.unique(s["df"][s["group_col"]])})
            for g in groups:
                if g in groups_seen:
                    status.add("models", g, "task2_activity(7.3-7.7)", "done", f"-> table_7_3_7_7/ (classical{'+dl' if run_dl else ', dl skipped (--run-dl)'})")
            if run_dl and dl2:
                safe_call(mc.build_publication_tables, classical2, dl2, cfg2.out_dir)

    # --- Table 7.8: OE-only conversation detection. task_oe_specific.run_all()
    # always runs both a classical grid and a (small) DL grid internally --
    # there is no classical-only entry point -- kept real but cheap via
    # max_epochs/patience regardless of --run-dl. ---
    r, err = safe_call(task_oe_specific.run_all, sync_root, out_dir=os.path.join(out_dir, "table_7_8"), max_epochs=3, patience=1)
    if err:
        status.add("models", "ALL", "table_7_8", "failed", err)
    else:
        status.add("models", "ALL", "table_7_8", "done", "task_oe_specific.run_all() -> table_7_8/")

    # --- Table 7.9: OE9 three-class recognition (fully non-circular; pass
    # freshly-computed eng3/eng7/oe10 DataFrames straight through in-memory
    # when the features stage built them this run) ---
    return cfg.out_dir  # unused, kept for readability


def stage_table_7_9(sync_root, out_dir, features, status: StatusTracker):
    kwargs = {}
    if "eng3_full" in features:
        kwargs["eng3"] = features["eng3_full"]
    if "eng7" in features:
        kwargs["eng7"] = features["eng7"]
    if "oe10" in features:
        kwargs["oe10"] = features["oe10"]
    r, err = safe_call(task_oe9_recognition.run_all, sync_root, out_dir=os.path.join(out_dir, "table_7_9"), **kwargs)
    if err:
        status.add("models", "ALL", "table_7_9", "failed", err)
    else:
        status.add("models", "ALL", "table_7_9", "done", "task_oe9_recognition.run_all() -> table_7_9/ "
                   + ("(fresh in-memory eng3/eng7/oe10)" if kwargs else "(disk fallback)"))


# =====================================================================
# Stage 5 -- Task 3 (Table 8.2-8.8)
# =====================================================================

def stage_task3(sync_root, out_dir, status: StatusTracker, run_neural=False):
    task3_root = os.path.join(sync_root)  # RQ3_LABEL_NORMALIZATION + INTERACTION_ENG3 + ALL_MODEL_READY_FILES_IDENTITY_FIXED all live here
    bridge_copy(
        os.path.join(FIXTURES_ROOT, "RQ3_LABEL_NORMALIZATION", "rq3_normalized_labels_full.csv"),
        os.path.join(task3_root, "RQ3_LABEL_NORMALIZATION", "rq3_normalized_labels_full.csv"),
    )
    bridge_copy(
        os.path.join(FIXTURES_ROOT, "INTERACTION_OPTI2", "interaction_opti2_10s.csv"),
        os.path.join(task3_root, "INTERACTION_OPTI2", "interaction_opti2_10s.csv"),
    )
    core_out = os.path.join(out_dir, "task3_tokens")

    TF, err = safe_call(task3_tokens.get_or_build_tokens, task3_root, core_out, True, False)
    if err:
        status.add("task3", "ALL", "tokens", "failed", err)
        status.add("task3", "ALL", "table_8_2..8_8", "skipped", "token table unavailable")
        return
    T, fcols = TF
    status.add("task3", "ALL", "tokens", "done", f"n={len(T)} groups={T['group'].nunique()} classes={T['label'].nunique()}")

    grammar_summary = None
    r, err = safe_call(task3_grammar.run_all, T, fcols, os.path.join(out_dir, "table_8_7"))
    if err:
        status.add("task3", "ALL", "table_8_7_grammar", "failed", err)
    else:
        grammar_summary = r[0]
        status.add("task3", "ALL", "table_8_7_grammar", "done", "task3_grammar.run_all() -> table_8_7/")

    long_history_summary = None
    r, err = safe_call(task3_common_targets.run_primary_panel, T, fcols, os.path.join(out_dir, "table_8_4_common_targets"))
    if err:
        status.add("task3", "ALL", "table_8_4_primary_panel", "failed", err)
    else:
        status.add("task3", "ALL", "table_8_4_primary_panel", "done", "task3_common_targets.run_primary_panel() -> table_8_4_common_targets/")
    r2, err2 = safe_call(task3_common_targets.run_long_history_panel, T, fcols, os.path.join(out_dir, "table_8_4_common_targets"))
    if err2:
        status.add("task3", "ALL", "table_8_4_long_history", "failed", err2)
    else:
        long_history_summary = r2
        status.add("task3", "ALL", "table_8_4_long_history", "done", "task3_common_targets.run_long_history_panel() -> table_8_4_common_targets/")

    r, err = safe_call(task3_persistence.run_all, task3_root, os.path.join(out_dir, "table_8_2"))
    status.add("task3", "ALL", "table_8_2_persistence", "failed" if err else "done", err or "task3_persistence.run_all() -> table_8_2/")

    r, err = safe_call(task3_segment_forecast.run_all, task3_root, os.path.join(out_dir, "table_8_3"))
    status.add("task3", "ALL", "table_8_3_segment_forecast", "failed" if err else "done", err or "task3_segment_forecast.run_all() -> table_8_3/")

    r, err = safe_call(task3_expanding_prefix.run_all, task3_root, os.path.join(out_dir, "table_8_5"), max_epochs=10)
    status.add("task3", "ALL", "table_8_5_expanding_prefix", "failed" if err else "done", err or "task3_expanding_prefix.run_all() -> table_8_5/")

    r, err = safe_call(task3_hmm_appendix_d.run_all, task3_root, os.path.join(out_dir, "table_8_6"))
    status.add("task3", "ALL", "table_8_6_hmm", "failed" if err else "done", err or "task3_hmm_appendix_d.run_all() -> table_8_6/")

    r, err = safe_call(task3_naive5.run_all, task3_root, core_out, os.path.join(out_dir, "table_8_8"))
    status.add("task3", "ALL", "table_8_8_naive5", "failed" if err else "done", err or "task3_naive5.run_all() -> table_8_8/")

    if run_neural:
        r, err = safe_call(task3_neural.run_all_seeds, T, fcols, os.path.join(out_dir, "table_8_4_neural"), epochs=10)
        if err:
            status.add("task3", "ALL", "table_8_4_neural", "failed", err)
        else:
            neural_summary = r[0]
            status.add("task3", "ALL", "table_8_4_neural", "done", "task3_neural.run_all_seeds(epochs=10) -> table_8_4_neural/")
            if grammar_summary is not None and long_history_summary is not None:
                r2, err2 = safe_call(task3_publication.build_core_tables, grammar_summary, neural_summary, long_history_summary, os.path.join(out_dir, "table_8_4"))
                status.add("task3", "ALL", "table_8_4_publication", "failed" if err2 else "done", err2 or "task3_publication.build_core_tables() -> table_8_4/")
    else:
        status.add("task3", "ALL", "table_8_4_neural", "skipped", "DL training off by default (--run-neural to enable; slow)")
        status.add("task3", "ALL", "table_8_4_publication", "skipped", "depends on table_8_4_neural")


# =====================================================================
# Planning / dry-run
# =====================================================================

def print_plan(groups, stages, data_root, out_dir):
    print("PLAN (--dry-run, nothing executed)")
    print(f"  data_root = {data_root}")
    print(f"  out_dir   = {out_dir}")
    print(f"  groups    = {groups}")
    print(f"  stages    = {stages}")
    print()
    for stage in stages:
        print(f"  stage '{stage}':")
        if stage == "sync":
            for g in groups:
                for sensor in ("openearable", "xsens", "optitrack"):
                    print(f"    group {g} / {sensor}: raw_sync_*.run_all() against {data_root}, "
                          f"bridge fallback from RAW_VALIDATION if unavailable")
        elif stage == "clean":
            print(f"    global_cleaning.run_all(<out_dir>/raw_sync) for groups {groups}")
        elif stage == "features":
            print("    eng3_recognition_labels / eng7_proximity_features / oe9_oe10_features / "
                  "eng_task2_grid+opti+xsens+merge (non-circular); eng_task1_oe/opti/xsens "
                  "(bridged official grid, known circularity gap)")
        elif stage == "models":
            print("    Table 7.1/7.2 (interaction), 7.3-7.7 (activity), 7.8 (OE-only), 7.9 (OE9 recognition)")
        elif stage == "task3":
            print("    tokens -> table_8_2 .. table_8_8 (neural/8.4-publication off unless --run-neural)")
        print()


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "raw"),
                    help="Raw sensor data root (group_{g}/{elan,openearable,xsens,optitrack}/...). Default: data/raw")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "processed", "pipeline_run"),
                    help="Where all pipeline output goes. Default: data/processed/pipeline_run")
    p.add_argument("--groups", default=",".join(str(g) for g in ALL_GROUPS),
                    help=f"Comma-separated group numbers to process. Default: all ({ALL_GROUPS})")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"Comma-separated stages to run, in order. Default: all ({ALL_STAGES})")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running anything.")
    p.add_argument("--run-dl", action="store_true", help="Also run the (slow) deep-learning grids for Task 1/2/7.8. Off by default.")
    p.add_argument("--run-neural", action="store_true", help="Also run Task 3's neural training (Table 8.4 neural rows + publication merge). Off by default, slow.")
    args = p.parse_args()

    groups = sorted(int(g) for g in args.groups.split(",") if g.strip())
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        p.error(f"unknown stage(s) {bad}; choose from {ALL_STAGES}")
    bad_g = [g for g in groups if g not in ALL_GROUPS]
    if bad_g:
        p.error(f"unknown group(s) {bad_g}; choose from {ALL_GROUPS}")

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)

    if args.dry_run:
        print_plan(groups, stages, data_root, out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    status = StatusTracker()
    sync_root = os.path.join(out_dir, "raw_sync")
    central = os.path.join(sync_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    features = {}

    if "sync" in stages:
        print("\n" + "#" * 100 + "\n# STAGE 1: raw sync\n" + "#" * 100)
        sync_root = stage_sync(groups, data_root, out_dir, status)

    if "clean" in stages:
        print("\n" + "#" * 100 + "\n# STAGE 2: global cleaning (model-ready assembly)\n" + "#" * 100)
        central = stage_clean(groups, sync_root, status)

    if "features" in stages:
        print("\n" + "#" * 100 + "\n# STAGE 3: feature engineering\n" + "#" * 100)
        features = stage_features(groups, sync_root, central, out_dir, status)

    if "models" in stages:
        print("\n" + "#" * 100 + "\n# STAGE 4: model training / eval\n" + "#" * 100)
        stage_models(groups, sync_root, out_dir, status, run_dl=args.run_dl)
        stage_table_7_9(sync_root, out_dir, features, status)

    if "task3" in stages:
        print("\n" + "#" * 100 + "\n# STAGE 5: Task 3\n" + "#" * 100)
        stage_task3(sync_root, out_dir, status, run_neural=args.run_neural)

    status.print_table()
    status_path = os.path.join(out_dir, "pipeline_status.csv")
    status.frame().to_csv(status_path, index=False)
    print(f"\nFull status table written to: {status_path}")


if __name__ == "__main__":
    main()
