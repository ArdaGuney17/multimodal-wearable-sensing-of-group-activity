# 🚀 Quickstart: raw data → synced/model-ready files → results

A step-by-step terminal walkthrough, from a bare `git clone` to actual result tables.

> [!NOTE]
> Every command below was run for real, from scratch, in a throwaway clone with no
> pre-existing local files (**2026-09-21**) — this isn't the aspirational interface, it's
> what actually happened.

**Checklist:** ☐ Clone → ☐ Virtual env → ☐ Install deps → ☐ Download data → ☐ Sync/clean/features → ☐ Run models → ☐ Compare results

## ⏱️ Time & disk budget

| Step | Expect | Notes |
|---|---|---|
| 📦 Install dependencies | 5–10 min | — |
| ⬇️ Download raw data | 15–60 min | ~2.5GB; Drive throttles large files to ~0.5–3MB/s, varies a lot run to run |
| 🔄 Sync → clean → features | 15–30 min | — |
| 🤖 Run models | 20–40 min | classical models only — `--run-dl`/`--run-neural` adds hours, not minutes |
| 💾 Peak disk usage | 10–15GB free | recommended headroom for a full run |

---

## 1️⃣ Clone and enter the repo

```bash
git clone https://github.com/ArdaGuney17/multimodal-wearable-sensing-of-group-activity.git
cd multimodal-wearable-sensing-of-group-activity
```

## 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 3️⃣ Install dependencies

> [!WARNING]
> `pip install -r requirements.txt` **will fail on `torch`** on its own — the pin is a `+cpu`
> local version specifier that plain PyPI doesn't serve. Install torch from its own CPU index
> first, *then* the rest:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

*(The second command re-resolves everything; torch is already satisfied so it just adds the rest.)*

## 4️⃣ Download the raw sensor data

```bash
python -m src.data.download
```

Pulls all 9 groups (~2.5GB total) from the public Drive archive into `data/raw/`, verifying each
file's SHA-256 checksum as it goes.

> [!TIP]
> If your connection drops partway through, just run the exact same command again — it skips
> whatever's already downloaded and only fetches what's missing (no `--force` needed unless you
> want to force a full re-download).

## 5️⃣ Run sync → clean → features

*Produces the synced / model-ready files.*

```bash
python scripts/reproduce_pipeline.py --stages sync,clean,features --data-root data/raw --out-dir data/processed/pipeline_run
```

| Stage | What it does |
|---|---|
| 🔄 `sync` | Aligns each group's raw OpenEarable/Xsens/OptiTrack streams to a shared video-time axis and attaches ELAN annotation labels |
| 🧹 `clean` | Assembles the final per-group `model_ready.csv` files (one per sensor per group) — the "model-ready files" step |
| 🧮 `features` | Builds every downstream feature table (ENG3/ENG7/OE9/OE10, Task 2's merged 10s-window features, Task 1's 5s window/label grid) |

**Output lands under `--out-dir`:**

| Path | Contents |
|---|---|
| `raw_sync/ALL_MODEL_READY_FILES_IDENTITY_FIXED/` | The model-ready files themselves — `group_{g}_{sensor}_model_ready.csv`, 27 files: 9 groups × 3 sensors |
| `raw_sync/INTERACTION_ABLATIONS/activity3_advanced_merged_10s_features.csv` | Task 2's feature table |
| `raw_sync/INTERACTION_BINARY_5S_SPECIALIZED_OE/binary_5s_specialized_oe_merged_all_features.csv` | Task 1's feature table |
| `pipeline_status.csv` | One row per `(stage, group, item)` with a real status: `done` (freshly computed), `bridged` (a validated fixture was used because your `data-root` was missing that specific input), `skipped`, or `failed` (with the real exception text, never faked) |

> [!TIP]
> **Sanity check before committing to a full run:** add `--groups 1,2,3 --dry-run` to see the plan
> without executing anything, or drop `--dry-run` to actually run just those 3 groups as a quick
> smoke test.

## 6️⃣ Run the models

*Produces the actual result tables.*

```bash
python scripts/reproduce_pipeline.py --stages models,task3 --data-root data/raw --out-dir data/processed/pipeline_run
```

Reuses the `raw_sync/` output from step 5 under the same `--out-dir` — don't delete it between
steps 5 and 6.

| Stage | What it runs |
|---|---|
| 🤖 `models` | Task 1 (`table_7_1_7_2/`) and Task 2 (`table_7_3_7_7/`), classical grids across all 7 sensor-combination conditions, plus Table 7.8/7.9 (`table_7_8/`, `table_7_9/`) |
| 🔮 `task3` | The full Chapter 8 suite — tokens, then Tables 8.2, 8.3, 8.5, 8.6, 8.7 (the thesis's own declared headline result), 8.8 |

Each output folder has both raw per-condition CSVs and a `combined_classical_best_per_condition_
with_std.csv`-style summary. Task 3's modules additionally print a `REPRODUCTION CHECK AGAINST
docs/thesis_reproduction_targets.md` block straight to the terminal, comparing every row against
the published numbers live as it runs.

## 7️⃣ Compare your results against the published thesis numbers

📊 **Fastest path:** open **[reproduction_ledger.md](reproduction_ledger.md)** (or the styled
**[reproduction_ledger.html](reproduction_ledger.html)**) — it already has a from-scratch run
compared line by line against every table.

To compare your own run by hand: [`thesis_reproduction_targets.md`](thesis_reproduction_targets.md)
has the ground-truth numbers for every table, section by section.
- For **Task 3**, you don't need to compare by hand — the reproduction-check output from step 6
  already does it.
- For **Task 1/2**, open `table_7_1_7_2/.../*_best_per_condition_with_std.csv` or
  `table_7_3_7_7/combined_classical_best_per_condition_with_std.csv` next to the matching table in
  `thesis_reproduction_targets.md`.

> [!NOTE]
> **What to actually expect**, from a real, from-scratch run against the public data only, no
> local shortcuts:
> - Task 2 and Task 3's label/token-sequence models reproduce closely (most conditions within
>   0.01–0.03 of target).
> - A few XSENS-involving Task 2 conditions and any Task 3 model that consumes raw sensor feature
>   values will show a somewhat larger gap — both are known, already-investigated effects (library
>   version drift; one specific Group 3 sensor-shift precision gap), not something wrong with your
>   run.
> - Task 1's headline Transformer number does not currently match the historical 0.8064 published
>   figure (this reproduction gets ~0.73–0.75, depending on environment) — a genuinely open gap,
>   not something this guide's steps will close.

---

## ⚙️ Options worth knowing about

| Flag | Effect |
|---|---|
| `--groups 1,2,3,5,6,7,8,9,10` | Restrict to specific groups (group 4 is excluded everywhere — camera failure made its annotation unusable) |
| `--run-dl` | Also run the deep-learning grids for Task 1/2/7.8 (slow; off by default) |
| `--run-neural` | Also run Task 3's neural training, Table 8.4's neural rows + publication merge (slow; off by default) |
| `--help` | Full flag reference — `python scripts/reproduce_pipeline.py --help` |

## 🚨 If something looks wrong

Check `pipeline_status.csv` first — every `(stage, group, item)` combination has an honest status
and, for anything that failed, the real exception text.

> [!CAUTION]
> If a status says `bridged` and you ran against a full 9-group `data/raw` from step 4, that's
> unexpected — please open an issue with the relevant `pipeline_status.csv` rows.
