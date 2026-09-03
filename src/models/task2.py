"""Task 2 — Group Activity Recognition (5 sub-tasks 2a-2e).

Ported from the Colab notebook `github notebooks/actual ones/
task2_full_comparison_RESUMABLE_with_std.ipynb` (see
docs/table_to_source_mapping.md). Shares essentially all its modelling logic
with Task 1 — that shared code lives in src/models/common.py. This file is
just Task 2's task list and its resume-by-default CLI default (the source
notebook made checkpoint/resume a first-class feature for surviving Colab
disconnects on this longer job; src/models/common.py implements it generically
for both tasks, see Config.resume_existing).

Maps to thesis Ch.7 §7.4 (Tables 7.3-7.7) — confirmed against
docs/thesis_reproduction_targets.md §5, all 5 sub-tasks including the
three-class one (2e, the "main multi-class task" per the thesis):

    notebook task name                   thesis sub-task   table
    conversation_vs_nonconversation      2a                 7.3
    conversation_vs_building             2b                 7.4
    conversation_vs_merging              2c                 7.5
    merging_vs_building                  2d                 7.6
    three_class_activity                 2e (three-class)   7.7

INPUT DEPENDENCY: like Task 1, this consumes a pre-computed feature CSV
(INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv) — not raw
data. See src/models/task1.py's module docstring for the same caveat.
"""

from __future__ import annotations

import argparse
import os

from src.models.common import (
    DEVICE,
    REPO_ROOT,
    Config,
    build_publication_tables,
    build_task_specs,
    run_all_classical,
    run_all_dl,
)

ACTIVITY_TASKS = [
    "conversation_vs_nonconversation",
    "conversation_vs_building",
    "conversation_vs_merging",
    "merging_vs_building",
    "three_class_activity",
]


def _default_config(**overrides) -> Config:
    overrides.setdefault("run_tasks", list(ACTIVITY_TASKS))
    overrides.setdefault("resume_existing", True)  # matches the source notebook's RESUME_EXISTING=True
    cfg = Config(**overrides)
    if not cfg.out_dir:
        cfg.out_dir = os.path.join(cfg.data_root, "PUBLICATION_TASK2_FULL_COMPARISON")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "processed"), help="Folder containing the feature CSVs (see module docstring).")
    parser.add_argument("--out-dir", default=None, help="Output folder for results (default: <data-root>/PUBLICATION_TASK2_FULL_COMPARISON).")
    parser.add_argument("--tasks", nargs="+", default=None, choices=ACTIVITY_TASKS, help="Restrict to a subset of the 5 sub-tasks (default: all 5).")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--full-grid", action="store_true", help="Use the wider exploratory classical grid instead of the report-reproduction one.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume/checkpoint reuse (default: on, matching the source notebook's RESUME_EXISTING=True).")
    parser.add_argument("--max-logo-folds", type=int, default=None, help="Cap LOGO folds for a quick smoke test (e.g. 2).")
    args = parser.parse_args()

    cfg = _default_config(
        data_root=args.data_root,
        out_dir=args.out_dir or "",
        run_tasks=args.tasks or list(ACTIVITY_TASKS),
        run_classical=not args.skip_classical,
        run_dl=not args.skip_dl,
        report_reproduction_mode=not args.full_grid,
        resume_existing=not args.no_resume,
        max_logo_folds=args.max_logo_folds,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Device:", DEVICE)
    print("Data root:", cfg.data_root)
    print("Output dir:", cfg.out_dir)
    print("Tasks:", cfg.run_tasks)
    print("Resume existing:", cfg.resume_existing)

    if not (cfg.run_classical or cfg.run_dl):
        return

    task_specs = build_task_specs(cfg)
    classical_result = run_all_classical(task_specs, cfg) if cfg.run_classical else {}
    dl_result = run_all_dl(task_specs, cfg) if cfg.run_dl else {}
    if cfg.run_classical and cfg.run_dl:
        build_publication_tables(classical_result, dl_result, cfg.out_dir)


if __name__ == "__main__":
    main()
