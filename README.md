# Multimodal Wearable Sensing of Group Activity

Reproduction repository for the MSc thesis *"Multimodal Wearable Sensing of Group Activity:
Understanding, Recognizing and Predicting Collaborative Group Activities from Wearable Sensor
Data"* (Arda Güney, MSc Interaction Technology, University of Twente, 2026).

> **Status: work in progress.** This repo is being rebuilt from the original (uncurated) Colab
> notebook history into a clean, scripted pipeline. The README below describes the intended end
> state; see [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for the honest,
> per-table status of what's actually ported and verified so far. Nothing here claims to reproduce
> a number until that table marks it ✅.
>
> **[See the full reproduction scorecard →](docs/reproduction_comparison.html)** — every
> accuracy/macro-F1 figure this code produces, checked row-by-row against the published thesis
> numbers across all 9 validated tables (54 results checked, 42 bit-exact, 12 close and explained,
> 0 failures). GitHub shows that link as raw source, not a styled page — download it and open it
> in a browser, or enable GitHub Pages for this repo, to see it rendered.

## What this reproduces

Three sensing modalities — ear-worn (OpenEarable 2.0), wrist-worn (Xsens DOT), and ceiling-mounted
optical tracking (OptiTrack) — recorded from groups of three participants performing a
collaborative puzzle-assembly task, used to train and evaluate models for:

1. **Task 1 — Interaction detection** (binary: interacting vs. not)
2. **Task 2 — Group activity recognition** (conversation / co-building / co-merging, 5 sub-tasks 2a–2e)
3. **Task 3 — Next-activity forecasting** (exploratory anticipation of upcoming collaboration state)

All results use leave-one-group-out (LOGO) evaluation across 9 groups (group 4 was recorded but
excluded — camera failure made its annotation unusable), with a secondary sensitivity analysis on
the 5 groups that never included a researcher as a participant. Full experimental design,
methodology, and every target number are documented in
[`docs/thesis_reproduction_targets.md`](docs/thesis_reproduction_targets.md).

## Repository structure

```
data/
  raw/          # Raw per-group sensor recordings + ELAN annotations. NOT committed to git —
                 # archived on Zenodo (see docs/data_provenance.md) and pulled locally by
                 # `python -m src.data.download`.
  processed/    # Synced, windowed, feature-engineered outputs. NOT committed (.gitignore) —
                 # produced by running src/preprocessing and src/features.
  external/     # Any third-party reference data, if needed.
src/
  preprocessing/  # Sync, cleaning, annotation normalization, windowing
  features/       # Feature engineering per sensor family (Ch.5)
  models/         # Task 1 / 2 / 3 training + evaluation
  eval/           # LOGO evaluation harness, naive-cohort sensitivity analysis, metrics
results/
  tables/       # Generated tables (small CSVs, committed as plain git — human-diffable),
                 # meant to be compared directly against docs/thesis_reproduction_targets.md
  figures/      # Generated figures (not committed — regenerate locally)
docs/
  thesis_reproduction_targets.md   # Ground-truth spec: every table, exact numbers, methodology
  drive_source_inventory.md        # Inventory of the original (messy) Colab/Drive history
  table_to_source_mapping.md       # Per-table status: which script reproduces which table
notebooks_reference/  # Original notebooks kept for reference during porting only —
                       # not the source of truth once a table is marked ✅ in the mapping doc
```

## Setup

```bash
git clone <repo-url>
cd multimodal-group-activity-recognition
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Data

Raw sensor data (~2.5GB, all 9 groups' OptiTrack/Xsens/OpenEarable/ELAN recordings, anonymized —
see [`docs/data_provenance.md`](docs/data_provenance.md) for exactly what's excluded and why) is
**not** committed to this repository. It's hosted in two places:

- **Now (interim)**: a public Google Drive folder —
  [multimodal-group-activity-recognition-raw-data](https://drive.google.com/drive/folders/1c0LOZ98eEX9RyOY7iio6Rl_NOwFgxTf2).
- **Eventually**: a permanent, DOI-citable Zenodo record, once published alongside the thesis paper.

Fetch it with:

```bash
python -m src.data.download                    # pulls from the current interim source (Drive)
python -m src.data.download --source zenodo --record-id <id>   # once Zenodo is live
```

Either way it downloads + extracts into `data/raw/` and verifies checksums.

See [`docs/drive_source_inventory.md`](docs/drive_source_inventory.md) for what each raw
subfolder contains and [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for
known data issues (e.g. a mislabeled folder) still being verified.

### Reproducing a table

Each pipeline stage is a plain Python script/module (no notebooks in the critical path), intended
to run in order:

```bash
python -m src.preprocessing.run_all      # raw -> synced, cleaned, windowed
python -m src.features.run_all           # windowed -> feature tables (Ch.5)
python -m src.models.task1               # -> results/tables/table_7_1.csv, table_7_2.csv
python -m src.models.task2               # -> results/tables/table_7_3.csv ... table_7_7.csv
python -m src.models.task_oe_specific    # -> results/tables/table_7_8.csv ... table_7_10.csv
python -m src.eval.naive_cohort          # -> results/tables/table_7_11.csv ... table_7_13.csv, table_10_1.csv
python -m src.models.task3               # -> results/tables/table_8_1.csv ... table_8_8.csv
```

(These entry points are the target interface — see the mapping doc for which actually exist yet.)

### Reproducing the pipeline

`scripts/reproduce_pipeline.py` is the actual working orchestrator today — a single entry point
that runs the full chain (raw sync → cleaning → feature engineering → model training/eval → Task 3)
against whatever groups currently have the inputs each stage needs, in dependency order. It never
lets one group's or stage's failure abort the rest: each `(stage, group, item)` gets a status of
`done` (freshly computed from `--data-root`), `bridged` (a validated real fixture already in this
repo was used because `--data-root` lacked that input — see the script's own docstring for exactly
which gaps this covers), `skipped` (input genuinely absent), or `failed` (the real module code
raised; the actual exception is recorded). A full `pipeline_status.csv` is written to `--out-dir`
at the end.

```bash
# See the plan without running anything
python scripts/reproduce_pipeline.py --dry-run

# Quick check on a few known-good groups/stages
python scripts/reproduce_pipeline.py --groups 1,2,3 --stages sync,clean --dry-run
python scripts/reproduce_pipeline.py --groups 1,2,3 --stages sync,clean --out-dir data/processed/pipeline_run_test

# Full run, all groups and stages (classical models only — slow DL/neural grids off by default)
python scripts/reproduce_pipeline.py --data-root data/raw --out-dir data/processed/pipeline_run
```

Key flags (see `python scripts/reproduce_pipeline.py --help` for the full list):

- `--data-root` — raw sensor data root, laid out as `{data_root}/group_{g}/{elan,openearable,xsens,optitrack}/...`. Default: `data/raw`.
- `--out-dir` — where all pipeline output (synced/cleaned files, features, model tables, `pipeline_status.csv`) is written. Default: `data/processed/pipeline_run`.
- `--groups` — comma-separated group numbers to process. Default: all of `1,2,3,5,6,7,8,9,10` (group 4 is excluded everywhere — camera failure made its annotation unusable).
- `--stages` — comma-separated stages to run, in order: `sync,clean,features,models,task3`. Default: all five.
- `--dry-run` — print the plan (which groups/stages/modules would run) and exit without executing anything.
- `--run-dl` / `--run-neural` — opt in to the slow deep-learning/neural training grids (off by default so a routine run stays in the minutes-not-hours range).

## Known limitations of this reproduction

- Some intermediate files from the original project were lost; where a stage cannot be exactly
  reproduced from the base raw data, this is documented explicitly in the mapping doc rather than
  silently papered over.
- The original notebook history contains many duplicate/superseded versions of the same analysis;
  `docs/drive_source_inventory.md` records which version was treated as canonical and why.
- Numbers that don't match the thesis after porting are marked ⚠️ in the mapping doc with the
  observed discrepancy, not silently adjusted to match.

## Citation

If you use this dataset or pipeline, please cite the thesis (full citation to be added on
submission/archival).

## License

Code in this repository is released under the MIT License (see `LICENSE`). Data usage terms may
differ — see `docs/data_provenance.md`.
