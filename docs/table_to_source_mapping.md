# Thesis table → pipeline stage → original Drive source

This is the working map between what the thesis reports (`thesis_reproduction_targets.md`)
and where the corresponding logic currently lives in the messy Drive/Colab history
(`drive_source_inventory.md`). It exists so that porting a notebook into `src/` is a
lookup, not a re-investigation, and so gaps are visible instead of silently skipped.

Status legend: ⬜ not started · 🟨 in progress · ✅ ported + numbers verified against thesis · ⚠️ ported but numbers don't match / needs review

| Stage | Thesis target(s) | Best Drive source | Repo destination | Status |
|---|---|---|---|---|
| Raw sensor sync & cleaning | Ch.4 (Tables 4.1–4.4) | `PRE_PROCESSING/Arda_Thesis_Data_Preprocessing_FIXED_v4.ipynb`, `PRE_PROCESSING/sensor_sync_fixed.ipynb`, `PRE_PROCESSING/OPTI_TRACK_PROCESSINGipynb` (56MB, malformed filename), `PRE_PROCESSING/Global_Cleaning_Before_Model.ipynb` | `src/preprocessing/` | ⬜ |
| Label cleaning / annotation normalization | Ch.4 §4.5–4.6, Appendix A | `ML/rq3_label_audit_and_normalization.ipynb`, `GENERATOR_SOURCES/rq3_label_audit_and_normalization_generator_cells.txt`, `thesis/final_label_inventory/*.csv` | `src/preprocessing/labels.py` | ⬜ |
| Windowing & model-ready assembly | Ch.4 §4.7–4.9 (Table 4.4) | scattered across `PRE_PROCESSING/*` + `data/model_ready_reports/`, `data/ALL_MODEL_READY_FILES_IDENTITY_FIXED/` (outputs already exist — treat as regression fixtures) | `src/preprocessing/windowing.py` | ⬜ |
| Feature engineering (all families) | Ch.5 (Tables 5.1–5.4) | `github notebooks/master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb` (newest of 4 versions — V2/V3/base are superseded), `CLEAN_ML/0{4,5,6,7}_feature_engineering_ENG*.ipynb` | `src/features/` | ⬜ |
| Task 1 — interaction detection | Ch.7 §7.3 (Tables 7.1, 7.2) | `github notebooks/actual ones/task1_full_comparison_classical_elapsed_dl_with_std.ipynb` | `src/models/task1.py` | 🟨 ported + smoke-tested on synthetic data; **not yet run against real thesis numbers** — needs the actual feature CSVs (`binary_5s_specialized_oe_merged_all_features.csv` etc.), which come from the not-yet-ported feature-engineering stage |
| Task 2a–2e — activity recognition | Ch.7 §7.4 (Tables 7.3–7.7) | `github notebooks/actual ones/task2_full_comparison_RESUMABLE_with_std.ipynb` | `src/models/task2.py` (+ shared `src/models/common.py`) | 🟨 ported + smoke-tested (incl. the 2e three-class task specifically) on synthetic data; **not yet run against real thesis numbers** — same feature-CSV dependency as Task 1 |
| Dedicated OpenEarable experiments (OE9, SPECIAL_OE) | Ch.7 §7.5 (Tables 7.8–7.10) | `THESIS_NOTEBOOKS/ML/oe_conversation_nonconversation_experiment.ipynb`, `data/INTERACTION_OE9`, `INTERACTION_OE8`, `INTERACTION_OE10` | `src/models/task_oe_specific.py` | ⬜ |
| Naive-cohort / researcher-participation sensitivity | Ch.7 §7.6 (Tables 7.11–7.13), Ch.8 §8.10 (Table 8.8) | `thesis/after_GL/seven_winners_naive5.ipynb`, `naive5_best_per_task.ipynb`, `naive5_headline_rerun.ipynb` — **newest files in all of Drive (Aug 17), same day as final thesis submission — highest-trust source for headline numbers** | `src/eval/naive_cohort.py` | ⬜ |
| Task 3 — next-activity forecasting (all of Ch.8) | Ch.8 (Tables 8.1–8.8) | `github notebooks/actual ones/task3_FINAL_V2_with_exact_report_reproduction.ipynb` — see breakdown below, this one notebook is big enough to need its own table | see below | 🟨 **every piece ported + smoke-tested on synthetic data** (Parts I–IV, Appendices A–D, naive-5); none yet run against real feature/label CSVs — that's the feature-engineering-stage blocker described elsewhere in this doc |
| Table 10.1 (cohort comparison) | Ch.10 §10.5 | derived from the naive5 vs. full-9 evaluation outputs above, no separate notebook identified yet | `src/eval/naive_cohort.py` (shared with 7.11–7.13) | ⬜ |

## Task 3 breakdown (source: `task3_FINAL_V2_with_exact_report_reproduction.ipynb`, 43 code cells)

This one notebook covers all of Table 8.1–8.8, organized into a "mandatory core" (Parts I–IV) plus
four "optional" appendices plus a section the notebook itself calls "Appendix R" — **that last
name is misleading for reproduction purposes, see the finding below.**

**Important finding (verified against the actual code, not just the thesis text):** the notebook's
own docs frame "Appendix R" as a disposable check — reproducing an old, "methodologically
superseded" evaluation protocol against the corrected Parts II–IV pipeline elsewhere in the same
notebook. That framing is backwards for us: the thesis's own text (§8.9, "THE WINNING RESULT")
describes exactly this computation — n-gram/HMM/hybrid over all 244 tokens starting from position
1, not the "common-target" 217-target restriction Parts II–IV use — as what actually produced
**Table 8.7, the thesis's headline Task 3 result.** Confirmed by matching numbers: "Appendix R"'s
own `REPORT_REFERENCE` dict (`ngram_backoff_h2: (0.604, 0.499)`, etc.) is *exactly* Table 8.7's
published values. There's also a genuine reproducibility trap baked into the source code: it only
hits those exact numbers when the token `group` column is treated as **text**, not int — the
notebook's own comment records having verified both (`int order -> h=2 0.596/0.460 ; text order ->
0.604/0.499`) and chose text deliberately for tie-breaking reasons. Ported faithfully, quirk and
all, in `src/models/task3_grammar.py`'s docstring and `_prepare_text_group_tokens()`.

| Notebook part | Produces | Repo destination | Status |
|---|---|---|---|
| Part I (cell 8) — build six-label activity tokens | input to everything else in Ch.8 | `src/models/task3_tokens.py` | 🟨 ported (dual-path: reload existing table, or build from scratch — the from-scratch path is **unverified**, the original workflow copied in a pre-built table from a 4th notebook we don't have, see module docstring) |
| "Appendix R" (cells 50–54) — n-gram/HMM/hybrid grammar over all 244 tokens | **Table 8.7 — THE WINNING RESULT** | `src/models/task3_grammar.py` | 🟨 ported + smoke-tested on synthetic data; not yet run against the real token table |
| Part II (cell 10) — corrected common-target grammar/sensor/hybrid (217 targets) | `segment_markov_h1` feeds **both** Table 8.4's "Segment Markov" row AND Table 8.7's "First-order segment Markov" row (verified: both 0.481/0.285). The other 4 models' common-target numbers aren't directly tabulated — narrative-only, motivating the task3_grammar.py follow-up (§8.9 "the earlier comparison") | `src/models/task3_common_targets.py` | 🟨 ported + smoke-tested |
| Part III (cell 12) — leakage-free neural (Transformer/LSTM) on 217 common targets, 3 seeds | Table 8.4 (Transformer labels-only / Transformer+sensors / LSTM+sensors rows) | `src/models/task3_neural.py` | 🟨 ported + smoke-tested (reduced config); cross-checked target-count match against Part II |
| Part IV (cell 15) — merge II+III into publication tables | assembles the final Table 8.4-shaped panel (Segment Markov from Part II + 3 neural rows from Part III) | `src/models/task3_publication.py` | ✅ ported + smoke-tested — **Task 3 core pipeline (Parts I–IV) complete** |
| Appendix A (cells 18–27) — window-level 7-label history, all-window vs. transition-only | Table 8.2 (§8.4 "Persistence Problem") | `src/models/task3_persistence.py` | 🟨 ported + smoke-tested on synthetic data; not yet run against real feature/label CSVs. Naming quirk found in source code: despite the cell range calling this "7-label" throughout, it collapses social/task conversation into one class before evaluating, same as everywhere else in Task 3 — actual vocabulary evaluated is 6-class, not 7. See module docstring. |
| Appendix B (cells 29–41) — segment-level sensor-feature forecast + decode | Table 8.3 — **now disambiguated**: entirely this module's own output, not Part II's. Its `label_markov_segment_baseline` (0.5106/0.2919/0.3775, n=235) exactly matches the thesis's "Label Markov segment baseline" row; Part II's `segment_markov_h1` (0.481/0.285) is a different, unrelated computation (common-target-restricted) that merely sounds similar | `src/models/task3_segment_forecast.py` | 🟨 ported + smoke-tested on synthetic data; not yet run against real feature/label CSVs |
| Appendix C (cells 43–44) — expanding-prefix fine-13 vs. coarse-4 next-segment | Table 8.5 | `src/models/task3_expanding_prefix.py` | 🟨 ported + smoke-tested on synthetic data (reduced epoch count for speed); not yet run against real feature/inventory CSVs. Reads its own input pair (`INTERACTION_ENG3/recognition_interaction_window_label_inventory.csv` + `interaction_eng3_features.csv`), not the RQ3_LABEL_NORMALIZATION output Appendix A/B use |
| Appendix D (cells 46–48) — HMM / 5-class collective-state, causal vs. Viterbi | Table 8.6 | `src/models/task3_hmm_appendix_d.py` | 🟨 ported + smoke-tested on synthetic data; not yet run against real feature/model-ready CSVs. Table's 7 rows are assembled from 3 separately self-contained notebook cells (kept as 3 functions, not force-merged) — cell 46 needs the optional `hmmlearn` dependency (added to requirements.txt), cells 47/48 read the raw `ALL_MODEL_READY_FILES_IDENTITY_FIXED` per-group CSVs directly rather than the RQ3_LABEL_NORMALIZATION output |
| naive-5 rerun (grammar + persistence) | Table 8.8 | `src/models/task3_naive5.py` | 🟨 ported + smoke-tested on synthetic data. No dedicated naive5 notebook was fetched — follows Task 1's `run_naive5_reproduction` pattern instead: rerun the already-ported task3_grammar.py (upper block) and task3_persistence.py (lower block) restricted to `NAIVE_GROUPS`. Deliberately does **not** force-match the thesis's flagged "h=1 back-off = no-self-transition" coincidence — computes both independently and reports whether they agree |
| Cell 56 (large model-ready data inspection) | nothing — leftover exploration tooling | intentionally not porting | — |

## Known duplication to resolve before porting (see `drive_source_inventory.md` §4 for full detail)

- `github notebooks/` root has 5 generations of `01_interaction_vs_noninteraction*` and 4 of
  `master_feature_generator*` — **use `actual ones/` subfolder + `_CORRECTED_V4`, ignore the rest.**
- `CLEAN_ML/` (ironically) has the worst duplication: `01_interaction_detection.ipynb` ×3,
  `02_activity_recognition.ipynb` ×2 (byte-identical), `03_activity_prediction.ipynb` ×2
  (byte-identical). Diff before picking one.
- `PRE_PROCESSING/` has 4 overlapping `sensor_sync*` variants and 3 generations of
  `Arda_Thesis_Data_Preprocessing*` — start from `_FIXED_v4`.
- `thesis/output/group_3` contains files misnamed `group5_oe_labelled...csv` — **verify by hand**
  whether this is genuinely group 3's data before trusting it as a source.

## Open questions to resolve while porting (see `thesis_reproduction_targets.md` §8.2/§8.3)

- `OE10` feature set is named in the thesis prose but never independently tabulated — check
  whether the actual notebook code defines it even though no results table reports it alone.
- Table 4.2's duplicate `video_time_s` row (conflicting descriptions) needs resolving against
  actual notebook code, not just the PDF text.
- Table 7.8's GRU row uses a `seq=3, 30s` windowing convention that doesn't match the `seq=9/18`
  pattern used everywhere else — confirm in the OE-conversation notebook whether this is real or
  a transcription artifact.
- Session-duration figures differ between Table 3.1 (raw recording) and Table 10.1 (ELAN
  timeline) — pick one convention per pipeline stage and document which.
