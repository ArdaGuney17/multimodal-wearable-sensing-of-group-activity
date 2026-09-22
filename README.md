# Multimodal Wearable Sensing of Group Activity

Reproduction repository for the MSc thesis *"Multimodal Wearable Sensing of Group Activity:
Understanding, Recognizing and Predicting Collaborative Group Activities from Wearable Sensor
Data"* (Arda Güney, MSc Interaction Technology, University of Twente, 2026).

[![License: MIT](https://img.shields.io/badge/license-MIT-0f7f88)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB)](requirements.txt)
[![Status: work in progress](https://img.shields.io/badge/status-work_in_progress-e69138)](docs/table_to_source_mapping.md)
[![Reproduction check](https://img.shields.io/badge/reproduction-134_rows_checked%2C_92%25_within_3pp-2ea44f)](docs/reproduction_ledger.md)
[![Last verified](https://img.shields.io/badge/clean--room_verified-2026--09--21-0f7f88)](docs/QUICKSTART.md)

> [!IMPORTANT]
> This repo is being rebuilt from the original (uncurated) Colab notebook history into a clean,
> scripted pipeline. The README below describes the intended end state; see
> [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for the honest, per-table
> status of what's actually ported and verified so far. Nothing here claims to reproduce a number
> until that table marks it ✅.

> [!TIP]
> **New here? Start with these two pages** — both verified end-to-end 2026-09-21 in a genuine
> clean-room clone (fresh venv, real public download, zero local shortcuts):
> - 🚀 **[Quickstart guide](docs/QUICKSTART.md)** — every command from `git clone` to real result
>   tables, tested exactly as written.
> - 📊 **[Reproduction Ledger](docs/reproduction_ledger.md)** — all 134 published numbers this
>   code produces, checked line by line against the thesis, with a status badge (✅/🟡/🔴) per
>   row. Renders directly on GitHub. Prefer a nicer look? Download
>   **[reproduction_ledger.html](docs/reproduction_ledger.html)** and open it in a browser instead
>   (GitHub shows `.html` as raw source, not styled).

## Contents

- [What this reproduces](#what-this-reproduces)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Data](#data)
- [Reproducing the pipeline](#reproducing-the-pipeline)
- [Checking your results](#checking-your-results)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

## What this reproduces

Three sensing modalities — ear-worn (OpenEarable 2.0), wrist-worn (Xsens DOT), and ceiling-mounted
optical tracking (OptiTrack) — recorded from groups of three participants performing a
collaborative puzzle-assembly task, used to train and evaluate models for three tasks:

| | Task | What it predicts | Sensors used |
|---|---|---|---|
| 🤝 | **Task 1 — Interaction detection** | Binary: interacting vs. not | OE, OptiTrack, Xsens, and combinations |
| 🗣️ | **Task 2 — Group activity recognition** | Conversation / co-building / co-merging (5 sub-tasks, 2a–2e) | OE, OptiTrack, Xsens, and combinations |
| 🔮 | **Task 3 — Next-activity forecasting** | Exploratory anticipation of the upcoming collaboration state | Activity-token sequences + raw sensor features |

All results use leave-one-group-out (LOGO) evaluation across 9 groups (group 4 was recorded but
excluded — camera failure made its annotation unusable), with a secondary sensitivity analysis on
the 5 groups that never included a researcher as a participant. Full experimental design,
methodology, and every target number are documented in
[`docs/thesis_reproduction_targets.md`](docs/thesis_reproduction_targets.md).

## Repository structure

```
data/
  raw/          # Raw per-group sensor recordings + ELAN annotations. NOT committed to git —
                 # hosted on Google Drive for now, eventually Zenodo (see "Data" section below
                 # and docs/data_provenance.md) — pulled locally by `python -m src.data.download`.
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
  QUICKSTART.md                     # Step-by-step terminal walkthrough, clone to results
  reproduction_ledger.md            # Every result, line by line, vs. the thesis (GitHub-native)
  reproduction_ledger.html          # Same data, styled — download + open in a browser
  thesis_reproduction_targets.md    # Ground-truth spec: every table, exact numbers, methodology
  drive_source_inventory.md         # Inventory of the original (messy) Colab/Drive history
  table_to_source_mapping.md        # Per-table status: which script reproduces which table
notebooks_reference/  # Original notebooks kept for reference during porting only —
                       # not the source of truth once a table is marked ✅ in the mapping doc
```

## Setup

> [!TIP]
> Prefer a full walkthrough? **[docs/QUICKSTART.md](docs/QUICKSTART.md)** has every command below,
> plus real observed timings and disk usage, tested from a throwaway clone.

```bash
git clone https://github.com/ArdaGuney17/multimodal-wearable-sensing-of-group-activity.git
cd multimodal-wearable-sensing-of-group-activity
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu  # see warning below
pip install -r requirements.txt
```

> [!WARNING]
> `pip install -r requirements.txt` **will fail on `torch`** on its own — the pin is a `+cpu`
> local version specifier that plain PyPI doesn't serve. Install torch from its own CPU index
> first (as above), *then* run the full install — torch will already be satisfied, so it just
> resolves everything else.

### 📦 Data

Raw sensor data (~2.5GB, all 9 groups' OptiTrack/Xsens/OpenEarable/ELAN recordings, anonymized —
see [`docs/data_provenance.md`](docs/data_provenance.md) for exactly what's excluded and why) is
**not** committed to this repository. It's hosted in two places:

| Source | Status |
|---|---|
| 🔗 [Public Google Drive folder](https://drive.google.com/drive/folders/1c0LOZ98eEX9RyOY7iio6Rl_NOwFgxTf2) | **Interim** — used today |
| 📦 Zenodo (DOI-citable) | **Planned** — once published alongside the thesis paper |

Fetch it with:

```bash
python -m src.data.download                    # pulls from the current interim source (Drive)
python -m src.data.download --source zenodo --record-id <id>   # once Zenodo is live
```

Either way it downloads + extracts into `data/raw/` and verifies each file's SHA-256 checksum as
it goes. If your connection drops partway through, just run the same command again — it resumes
per-group instead of restarting the whole download.

See [`docs/drive_source_inventory.md`](docs/drive_source_inventory.md) for what each raw
subfolder contains and [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) for
known data issues (e.g. a mislabeled folder) still being verified.

## Reproducing the pipeline

`scripts/reproduce_pipeline.py` is the one real, tested entry point — a single orchestrator that
runs the full chain in dependency order against whatever groups currently have the inputs each
stage needs. It never lets one group's or stage's failure abort the rest.

| Stage | Icon | Produces |
|---|---|---|
| `sync` | 🔄 | Raw OpenEarable/Xsens/OptiTrack streams aligned to a shared video-time axis, ELAN labels attached |
| `clean` | 🧹 | Final per-group `model_ready.csv` files (one per sensor per group) — the "model-ready files" |
| `features` | 🧮 | Every downstream feature table (ENG3/ENG7/OE9/OE10, Task 2's 10s-window features, Task 1's 5s window/label grid) |
| `models` | 🤖 | Task 1 (`table_7_1_7_2/`) and Task 2 (`table_7_3_7_7/`) results, plus Table 7.8/7.9 |
| `task3` | 🔮 | The full Chapter 8 suite — activity tokens, then Tables 8.2, 8.3, 8.5, 8.6, 8.7, 8.8 |

Each `(stage, group, item)` gets a real status: `done` (freshly computed), `bridged` (a validated
fixture was used because your `--data-root` was missing that specific input), `skipped` (input
genuinely absent), or `failed` (the real exception text, never faked) — written to
`pipeline_status.csv` at the end.

```bash
# See the plan without running anything
python scripts/reproduce_pipeline.py --dry-run

# Quick check on a few known-good groups/stages
python scripts/reproduce_pipeline.py --groups 1,2,3 --stages sync,clean --dry-run

# Full run, all groups and stages (classical models only — slow DL/neural grids off by default)
python scripts/reproduce_pipeline.py --data-root data/raw --out-dir data/processed/pipeline_run
```

<details>
<summary><strong>⚙️ Key flags</strong> (<code>--help</code> for the full list)</summary>

| Flag | Meaning | Default |
|---|---|---|
| `--data-root` | Raw sensor data root, laid out as `{data_root}/group_{g}/{elan,openearable,xsens,optitrack}/...` | `data/raw` |
| `--out-dir` | Where all pipeline output lands | `data/processed/pipeline_run` |
| `--groups` | Comma-separated group numbers to process | `1,2,3,5,6,7,8,9,10` (group 4 always excluded) |
| `--stages` | Comma-separated stages, in order | `sync,clean,features,models,task3` |
| `--dry-run` | Print the plan and exit without executing | off |
| `--run-dl` / `--run-neural` | Opt into the slow deep-learning/neural training grids | off |

</details>

A couple of individual modules also have their own `python -m` entry point for running just one
piece against already-built feature files: `python -m src.models.task1` and
`python -m src.models.task2` (`--help` on either for options).

## Checking your results

Once you've run `models,task3`, set your output next to the published numbers:

- 📊 **Fastest path:** open **[docs/reproduction_ledger.md](docs/reproduction_ledger.md)** — it
  already has this repo's own clean-room run compared line-by-line against every table (7.2
  through 8.8), with a status badge (✅ exact / 🟡 close / 🔴 notable, traced) per row. Prefer a
  styled, downloadable version? See
  **[reproduction_ledger.html](docs/reproduction_ledger.html)**.
- Task 3's modules additionally print a `REPRODUCTION CHECK AGAINST
  docs/thesis_reproduction_targets.md` block straight to the terminal as they run.
- For Task 1/2 by hand: `table_7_1_7_2/.../*_best_per_condition_with_std.csv` or
  `table_7_3_7_7/combined_classical_best_per_condition_with_std.csv` next to the matching table in
  [`docs/thesis_reproduction_targets.md`](docs/thesis_reproduction_targets.md).

> [!NOTE]
> **What to actually expect**, from a real from-scratch run against the public data only: Task 2
> and Task 3's label/token-sequence models reproduce closely (most conditions within 0.01–0.03 of
> target). A few XSENS-involving Task 2 conditions and any Task 3 model that consumes raw sensor
> feature values show a somewhat larger gap — both are known, already-investigated effects
> (library version drift; one specific Group 3 sensor-shift precision gap), not something wrong
> with your run. Task 1's headline Transformer number does not currently match the historical
> 0.8064 published figure (this reproduction gets ~0.73–0.75) — a genuinely open gap.

## Known limitations

- Some intermediate files from the original project were lost; where a stage cannot be exactly
  reproduced from the base raw data, this is documented explicitly in
  [`docs/table_to_source_mapping.md`](docs/table_to_source_mapping.md) rather than silently
  papered over.
- The original notebook history contains many duplicate/superseded versions of the same analysis;
  [`docs/drive_source_inventory.md`](docs/drive_source_inventory.md) records which version was
  treated as canonical and why.
- Numbers that don't match the thesis after porting are marked ⚠️ in the mapping doc with the
  observed discrepancy, not silently adjusted to match.

## Citation

If you use this dataset or pipeline, please cite the thesis (full citation to be added on
submission/archival).

## License

Code in this repository is released under the [MIT License](LICENSE). Data usage terms may
differ — see [`docs/data_provenance.md`](docs/data_provenance.md).
