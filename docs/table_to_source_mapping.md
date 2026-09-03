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
| Task 2a–2e — activity recognition | Ch.7 §7.4 (Tables 7.3–7.7) | `github notebooks/actual ones/task2_full_comparison_RESUMABLE_with_std.ipynb` | `src/models/task2.py` | ⬜ |
| Dedicated OpenEarable experiments (OE9, SPECIAL_OE) | Ch.7 §7.5 (Tables 7.8–7.10) | `THESIS_NOTEBOOKS/ML/oe_conversation_nonconversation_experiment.ipynb`, `data/INTERACTION_OE9`, `INTERACTION_OE8`, `INTERACTION_OE10` | `src/models/task_oe_specific.py` | ⬜ |
| Naive-cohort / researcher-participation sensitivity | Ch.7 §7.6 (Tables 7.11–7.13), Ch.8 §8.10 (Table 8.8) | `thesis/after_GL/seven_winners_naive5.ipynb`, `naive5_best_per_task.ipynb`, `naive5_headline_rerun.ipynb` — **newest files in all of Drive (Aug 17), same day as final thesis submission — highest-trust source for headline numbers** | `src/eval/naive_cohort.py` | ⬜ |
| Task 3 — next-activity forecasting (all of Ch.8) | Ch.8 (Tables 8.1–8.8) | `github notebooks/actual ones/task3_FINAL_V2_with_exact_report_reproduction.ipynb` — name explicitly claims exact report reproduction | `src/models/task3.py` | ⬜ |
| Table 10.1 (cohort comparison) | Ch.10 §10.5 | derived from the naive5 vs. full-9 evaluation outputs above, no separate notebook identified yet | `src/eval/naive_cohort.py` (shared with 7.11–7.13) | ⬜ |

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
