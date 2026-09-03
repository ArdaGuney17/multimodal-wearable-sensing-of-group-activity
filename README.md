# Multimodal Wearable Sensing of Group Activity

Reproduction repository for the MSc thesis *"Multimodal Wearable Sensing of Group Activity:
Understanding, Recognizing and Predicting Collaborative Group Activities from Wearable Sensor
Data"* (Arda Güney, MSc Interaction Technology, University of Twente, 2026).

> **Status: work in progress.** This repo is being rebuilt from the original (uncurated) Colab
> notebook history into a clean, scripted pipeline. The README below describes the intended end
> state; see [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for the honest,
> per-table status of what's actually ported and verified so far. Nothing here claims to reproduce
> a number until that table marks it ✅.

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
  raw/          # Raw per-group sensor recordings + ELAN annotations (Git LFS). The only
                 # data actually committed — everything below is regenerated.
  processed/    # Synced, windowed, feature-engineered outputs. NOT committed (.gitignore) —
                 # produced by running src/preprocessing and src/features.
  external/     # Any third-party reference data, if needed (Git LFS).
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
git lfs install
git clone <repo-url>
cd multimodal-group-activity-recognition
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Data

Raw sensor data is tracked via Git LFS in `data/raw/`. If you cloned without LFS pulling
automatically, run `git lfs pull`. See
[`docs/drive_source_inventory.md`](docs/drive_source_inventory.md) for what each raw subfolder
contains and [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for known data
issues (e.g. a mislabeled folder) still being verified.

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
