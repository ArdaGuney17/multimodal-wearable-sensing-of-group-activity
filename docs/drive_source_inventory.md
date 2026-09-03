# Google Drive Inventory — Thesis Trees

Generated 2026-09-03. Owner: ardagney05@gmail.com. Listing/metadata only (titles, ids, mimeType, size, modifiedTime) — no file contents were read. `.ipynb_checkpoints` folders were detected but **not** enumerated (Colab autosave clutter) — only their presence is noted.

Two root trees were inventoried:
- **thesis** = `1yqCwyfJsMz1gGKwBqodRTtwGK6EBBUr3`
- **THESIS_NOTEBOOKS** = `1lWaPul53Zj685972o9tN5JdEtZUtI7j3`

---

## 0. HIGH VALUE: `thesis/github notebooks/` (full contents)

Folder id `1tKzlBfUkK8G4xqBAt0lK30exrI3zeSmm`. This looks like a **working "cleaned up for GitHub" staging area for notebooks**, mixing several generations of the same pipelines. It is **not** a clean, de-duplicated final set — it still contains multiple superseded versions of the same notebook side by side.

### Direct files (16 notebooks)

| Title | id | Size | Modified |
|---|---|---|---|
| master_feature_generator_task1_task2_task3_CORRECTED_V4.ipynb | 1LdBmQy57UxiH6iDxsWyHfYL14gvd4T8h | 944,844 B | 2026-07-23 |
| master_feature_generator_task1_task2_task3_CORRECTED_V3.ipynb | 1guqzQHNLsFPM4gGZPDzTQOLsTO0ORZOK | 894,230 B | 2026-07-23 |
| master_feature_generator_task1_task2_task3_CORRECTED_V2.ipynb | 1uPRlYeQXUBpuIAdX8czmkRcCKcwqFDS2 | 247,500 B | 2026-07-22 |
| master_feature_generator_task1_task2_task3.ipynb | 1OCaQINdkLEThT4_b9E8rZFXEi4ec2ZMq | 844,145 B | 2026-07-22 |
| 00_build_feature_tables.ipynb | 1YQlZFr5BAii0mrsa9J1wEGmNN9L3E8NJ | 793,524 B | 2026-07-23 |
| feature_inventory.ipynb | 1X40LgHyxTpOGjUfF3YGSIbxQ_WS3ejAp | 20,598 B | 2026-07-23 |
| 00_feature_table_repair_and_verification.ipynb | 1SwXqliH0apycgXRHC7s6JWSCR-IzAz0L | 45,042 B | 2026-07-23 |
| 02_activity_recognition_5s_v2.ipynb | 1Gb8jqm0U431Yj1O46PeHWSx8I7dPvbpR | 740,556 B | 2026-07-23 |
| task1_RAW_SENSOR_DL_publication.ipynb | 1JdkhBAf3RaGpKAW0ZkWTb8ygps4REA0y | 63,296 B | 2026-07-22 |
| task3_FINAL_V2_common_target_publication_notebook.ipynb | 132BuftAx2TULnNAFcQhvA47o17jOfukX | 1,108,932 B | 2026-07-21 |
| task3_CORRECTED_FINAL_publication_notebook.ipynb | 1G2IwKHFBtowZzTiTOvJcJYfWJU1zYsft | 1,044,975 B | 2026-07-21 |
| task3_full_publication_RESUMABLE_with_std.ipynb | 11_PLCfvQMVlD2krHVrRX-2U-jNiIguYD | 1,028,125 B | 2026-07-21 |
| 01_interaction_vs_noninteraction_v2.ipynb | 1OCGfXcaDv31e4hFGf4P64J6K8Sexe0Rz | 246,266 B | 2026-07-20 |
| 02_activity_recognition_5s.ipynb | 1JlUFeBH-bKBWWRo6r7bzIZ9oHRenpPSX | 105,435 B | 2026-07-19 |
| 01_interaction_vs_noninteraction_FIXED.ipynb | 1lgde5wpMl7mXT_rQAPDUavT0S1pRF8fF | 177,254 B | 2026-07-17 |
| 01_interaction_vs_noninteraction.ipynb | 1iIMuSXwL3Jz8jN2_zRT1CLoYZoB_ZuvU | 156,324 B | 2026-07-16 |
| 01_interaction_vs_noninteraction.ipynb | 1Xvc1TBwrWnasJHfhL-u6yIFbCSA-2nLT | 50,405 B | 2026-07-16 |
| 01_interaction_vs_noninteraction.ipynb | 1XMFvMvWjGP6i4B5NoxlbASwW83NjOrHw | 78,218 B | 2026-07-16 |

**Flags:** `01_interaction_vs_noninteraction.ipynb` exists **3 times** in this same folder (different ids, sizes 50KB/78KB/156KB) plus two more revisions (`_FIXED`, `_v2`) — 5 generations of the same notebook coexist. `master_feature_generator_task1_task2_task3*.ipynb` exists in **4 versions** (base, V2, V3, V4) in the same folder. This folder needs de-duplication before anything is pushed to GitHub — keep only the newest/`_CORRECTED_V4` / `_v2` / `_FIXED` variants and delete the superseded ones.

### Subfolder: `actual ones` (id `1rI5z_EISjui_MJ1j2ejDeV1j-pdqDeKX`)

A separate 3-notebook set, apparently the *actually final* publication notebooks (name suggests "ignore the above, these are the real ones"):

| Title | id | Size | Modified |
|---|---|---|---|
| task1_full_comparison_classical_elapsed_dl_with_std.ipynb | 1jH0jfDZGg-1V1gEbdEgUJJW6Y0R_akMY | 320,553 B | 2026-08-13 |
| task3_FINAL_V2_with_exact_report_reproduction.ipynb | 15MQgbqeqyU_dPTtpzmQRF7WKYH2KLYft | 1,159,785 B | 2026-07-22 |
| task2_full_comparison_RESUMABLE_with_std.ipynb | 1p5IVGij9mh0TSmCuXG0cc9LzfSQ7PD8q | 840,186 B | 2026-07-20 |

These are the most recently modified notebooks in the whole "github notebooks" tree (task1 one modified 2026-08-13, the newest timestamp seen anywhere under `github notebooks`), and one per task (1/2/3) — this subfolder is the best candidate for "the real, final, cite-able" notebook set. No further subfolders under `actual ones`.

**Bottom line for GitHub cleanup:** use `github notebooks/actual ones/` (3 files) as the canonical publication notebooks; the 16 files directly in `github notebooks/` are an uncurated working history full of duplicate `01_interaction_vs_noninteraction*` and `master_feature_generator*` versions.

---

## 1. `thesis/` root — top-level folders

All 9 folders below are direct children of `1yqCwyfJsMz1gGKwBqodRTtwGK6EBBUr3` (confirmed exhaustive listing, no pagination needed):

| Folder | id |
|---|---|
| after_GL | 19kAOUxnOBF4lIkEPHSxKUd8kpSn2FPwd |
| github notebooks | 1tKzlBfUkK8G4xqBAt0lK30exrI3zeSmm |
| GENERATOR_SOURCES | 1aG4K1ZSui1UFqFEWLb1bvwEdtqf7fz_f |
| data | 1a9vaiEdgkG-FBquFY3ySzfyWUoL1eMWH |
| final_label_inventory | 1tNbGu3jMTw0od3v0eMwZC41rOFZsYJRq |
| final_visualizations_participant_lines | 13bvR0c58lR_hmRJgsw63FpaEbE0xQxMe |
| final_visualizations_multisensor | 1lqvGeC0WJb7tfc22-wiD_PjCszVsz6SM |
| final_visualizations | 1CjBbVpaykLv0Oc9qUHZl7Q5k6x0owD18 |
| output | 1v-SZH-sMg6FqM67Ns9R-c4T6QBgJvFGm |

### after_GL (3 notebooks)
- seven_winners_naive5.ipynb — 1KwMaHt0yj3DnYakomO7qnAwqARjPGqYC — 653,448 B — modified 2026-08-17
- naive5_best_per_task.ipynb — 1ZqSqr9EMAKSFUDEgbseTGmHGcNUNuyD1 — 133,493 B — modified 2026-08-12
- naive5_headline_rerun.ipynb — 1WWzrowhgHcnWUpMKoekU9LfcnI4pEE5p — 2,050,616 B — modified 2026-08-12

These have the latest modification dates of anything in the whole `thesis` tree (2026-08-17), i.e. `after_GL` is the most recently active folder — likely post-"GL" (some late-stage analysis, e.g. "after Group Leader/final review") follow-up work.

### GENERATOR_SOURCES (2 files — plain-text notebook-cell dumps, per task instructions not read)
- rq3_label_audit_and_normalization_generator_cells.txt — 1DbBsszKXgnOF-XuOOg6Lw3eDRXrRk4xr — 1,550 B
- THESIS_NEW_ML_generator_cells.txt — 1ChTA5Aj2VpIqM_MqMv7CDKgme7qt0jk3 — 111,224 B

### final_label_inventory (3 CSVs)
- broad_category_summary.csv — 1D8dquCPK7o1CtIP3yAMdv1qpJqzuV0bo — 2,143 B
- unique_label_list.csv — 1B0l9kN64tYBRPQlkMkCi_eI18zVNkixc — 34,543 B
- full_label_inventory.csv — 1O3QbFLC8LG8wJx4aMiBblkHJHRqyqJef — 336,468 B

### final_visualizations_participant_lines (10 files)
1 CSV (`final_group_file_summary_acc_mag.csv`, 3,653 B) + 9 PNGs, one per group (`group_1`…`group_10`, **skipping group_4**): `group_{1,2,3,5,6,7,8,9,10}_final_acc_mag_participant_lines.png`, each ~1.0–1.3 MB. Same pattern of group numbering confirms **group_4 does not exist** anywhere in the visualization outputs either.

### final_visualizations_multisensor (20 PNGs)
Two PNGs per group (`_OE_multisensor_participants.png` and `_XSens_multisensor_participants.png`) for groups 1,2,3,5,6,7,8,9,10 (again no group_4) — sizes range ~1.0–2.5 MB.

### final_visualizations (10 files)
1 CSV (`final_group_file_summary.csv`, 3,795 B) + 9 PNGs `group_{1,2,3,5,6,7,8,9,10}_final_visual_summary.png` (~680–830 KB each).

### output (5 subfolders — group_1, group_2, group_3, group_5, group_6 only)
Only these 5 groups exist under `thesis/output/`; **group_4, group_7, group_8, group_9, group_10 are absent** here (unlike `data/`, which has raw folders for all groups except group_4 — see §2). Each present group folder holds exactly 2 large labelled CSVs:

| Folder | Files |
|---|---|
| group_1 (1D90NvDZokD9YCMRLXRLsUYQXQjHJYsHO) | group1_xsens_labelled_ALL_SHIFTED_IDENTITY_CORRECTED.csv (52.0 MB), group1_oe_labelled_ALL_SHIFTED_IDENTITY_CORRECTED.csv (73.0 MB) |
| group_2 (1j3xE8FmPdsd6ak7XjjrpWdCpOlDPPYqm) | group2_oe_labelled_NEW_SYNC1_SHIFTED (1).csv (95.3 MB), group2_xsens_labelled_NEW_SYNC1_SHIFTED (1).csv (70.5 MB) |
| group_3 (1HWlN3X24OIoDWpqWEZHXjRCgIt6-W-xD) | group5_oe_labelled (1).csv (136.4 MB), group5_xsens_labelled (1).csv (103.7 MB) |
| group_5 (1rEZk_sIjnncMOrS8zqTCLL3a2Wt0Jw6P) | **empty — 0 files** (confirmed on repeat query) |
| group_6 (1j84dhr_DNr4qLVmwURYCFner26xr6CjN) | **empty — 0 files** (confirmed on repeat query) |

**Flag / anomaly:** the two files inside `output/group_3` are literally named `group5_oe_labelled (1).csv` / `group5_xsens_labelled (1).csv` — i.e. **group_3's folder contains files named "group5"**. Either a mislabeled folder or a copy-paste-from-group_5 naming mistake. Worth checking by hand. Also note the `" (1)"` suffix on group_2 and group_3's files, suggesting these are re-uploaded duplicates of an original `groupN_..._labelled...csv` (Drive auto-renamed on conflict) — the un-suffixed original may or may not still exist elsewhere. **`output/group_5` and `output/group_6` are both genuinely empty folders (0 files, confirmed on repeat query)** — likely created as placeholders that were never populated, or their labelled CSVs were moved/renamed elsewhere (possibly explaining the misnamed "group5" files sitting in `group_3` above).

---

## 2. `thesis/data/` — the base data tree (recursively explored)

`data` (`1a9vaiEdgkG-FBquFY3ySzfyWUoL1eMWH`) is by far the largest branch: **68 direct subfolders** plus a handful of loose files, many subfolders 1–3 levels deep. Full recursive enumeration of every leaf file was not practical inside token limits for the largest, most repetitive branches (flagged below); everything else was listed exhaustively.

### 2.1 Loose files directly in `data/`
- ALL_RESULTS_WITH_STD_V3.csv — 1kE-Geh5uuIz2imHuZkvJEXKOA63kjYkm — 253,386 B
- ALL_RESULTS_WITH_STD_V2.csv — 1k3kN9LuEWGzeYhv9NVtb_d2ootukZbv5 — 46,516 B
- ALL_RESULTS_WITH_STD.csv — 1WpFgzwcStW4UWUN9K4vSArCn4fOEXzj8 — 90,425 B
- all_elan_unique_labels_for_cleaning.csv — 1Id7sIDnxU8DULoUYQ_prFwISnl25l730 — 14,508 B

(V/V2/V3 progression of the same results file — keep only V3 unless history is needed.)

### 2.2 Raw per-group sensor data — `data/group_1` … `data/group_10` (no group_4)
`data/group_1` (id `1KqT_124TdwNlHB10VbDKMtpw25gNQY8S`) is the **true raw-data root** for a group and has this structure (confirmed by direct listing):
```
group_1/
  optitrack/           (id 1Ll4Pe2O5jQU7F4NuSO9Yy0vG7Rkfqz7a)
    model_ready/ (subfolder, not further expanded)
    optitrack_labeled/ (subfolder, not further expanded)
    optitrack_final/ (subfolder, not further expanded)
    Arda_Thesis_Group-1_Take_2.csv   32.4 MB
    Arda_Thesis_Group-1_Take_1.csv   96.6 MB
  xsens/                (id 1LbmOkzgeVdAPzsjkgNTReJ06AWwj7EQX)
    model_ready/ (subfolder)
    xsens_labeled/ (subfolder)
    Participant1.csv   12.0 MB
    Participant2.csv   12.1 MB
    Participant3.csv   11.9 MB
  openearable/          (id 1BiD9oL4A4Sw_7_Dw9kIrkdiIld9NsiyP)
    Participant1/, Participant2/, Participant3/ (per-participant subfolders, not expanded)
    model_ready/ (subfolder)
    openearable_labeled/ (subfolder)
  elan/                 (id 178v347p-l4I7hjWUQdBL3MtA4RlntJdq)
    Group_1_individual_build_renamed.csv   28,707 B
    .ipynb_checkpoints/  <- Colab clutter, NOT enumerated
```
`data/group_10` (id `1YVpqEDgPZ2GwU-UyjDCb77kRDjFIJt-f`) has the **same 4-way structure** (`optitrack/`, `elan/`, `optitrack_labeled/`, `xsens/`, `openearable/`), confirming this is a fixed per-group raw-data template.

`data/group_2, 3, 5, 6, 7, 8, 9` (ids `10awdLk9LMhZpbBf4S9pI3FQI9SamUX2v`, `1kmsFk5kAYVsgLfnei22EmyJRT2OgCF8J`, `1chA2D_XvtMBeSVicCLF0YvN-pL5hjri6`, `1ncaQpLixGDkNg3e9RUvq4t-cfmsgiwtg`, `17r1p6ACnhXyvLvOjJsMN8q3m0gI7tPv9`, `1EZHsoPEsh7qwfU11I_99Sl_ta26eC2gF`, `10UNeP_M29DofIou4TW-xbPS8tHy_VmyF`) were **not individually re-expanded** past this point (time/size budget) — by strong structural analogy to group_1 and group_10 they almost certainly follow the identical `optitrack/xsens/openearable/elan` raw-sensor layout with 3 participants each. **group_4 is confirmed absent** (not present in this listing, nor in `output/`, nor in the visualization folders — it does not exist anywhere in the tree).

`data/output/group_1` (id `1TWpj2Fv5XTND7YnO_omua8xj_IELfaia`) — a *second*, separate "output" folder nested inside `data/` (distinct from `thesis/output/`) — only `group_1` exists under it (not expanded further).

### 2.3 Model-ready / feature-engineering experiment folders in `data/`
These are individually named experiment folders, each typically holding a handful of CSVs (predictions/summary/fold_metrics) plus occasionally an explainability PNG. Full per-folder contents were captured; sizes mostly range from a few KB (summary tables) to tens of MB (raw prediction dumps). Representative full listing highlights:

- **FEATURE_GENERATOR_INPUTS** (1 file): task1_authoritative_5s_label_grid.csv — 801,997 B
- **FEATURE_GENERATOR_REBUILT_v3** (7 subfolders, not expanded further): INTERACTION_OPTI2, INTERACTION_OE10, INTERACTION_OE9, INTERACTION_ENG7, INTERACTION_ENG3, INTERACTION_BINARY_5S_SPECIALIZED_OE, INTERACTION_BINARY_5S_ADVANCED_FEATURES
- **FEATURE_GENERATOR_REBUILT** (11 subfolders, not expanded further): INTERACTION_BINARY_5S_ADVANCED_FEATURES, MANIFEST, PUBLICATION_TASK3_CORRECTED_FINAL, INTERACTION_ABLATIONS, INTERACTION_XSENS2, INTERACTION_OPTI2, INTERACTION_OE10, INTERACTION_OE9, INTERACTION_ENG7, INTERACTION_ENG3, INTERACTION_BINARY_5S_SPECIALIZED_OE
- **OE_CONVERSATION_NONCONVERSATION** (18 CSV/PNG files): explain_*_family_summary.csv, plot_*_permutation.png / _coefficients.png, oe_conv_nonconv_*_predictions/summary/fold_metrics.csv (largest: oe_conv_nonconv_classical_predictions.csv, 4.9 MB)
- **INTERACTION_OE9** (3 files): interaction_oe9_10s.csv (13.2 MB), oe9_focused_search_predictions.csv (9.5 MB), oe9_focused_search_summary.csv
- **ALL_MODEL_READY_FILES_IDENTITY_FIXED** (16 files + `_inspection_report/` subfolder not expanded): group_{1,2,3,5,6,7,8,9,10}_optitrack/xsens/openearable_model_ready.csv, largest group_9_optitrack_model_ready.csv = **508 MB**; also IDENTITY_FIXED_manifest.csv, IDENTITY_FIXED_optitrack_column_renaming.csv
- **PUBLICATION_TASK1_RAW_SENSOR_DL** (2 subfolders, not expanded): CHECKPOINTS/, RAW_WINDOW_CACHE_5S_25HZ/
- **PUBLICATION_TASK3_FINAL_V2_COMMON_TARGETS** (18 CSVs + TOKEN_NEURAL_CHECKPOINTS_FINAL_V2_COMMON_TARGETS/ subfolder not expanded): appendix_r_*, task3_core_FINAL_V2_*, task3_token_neural_FINAL_V2_*, task3_grammar_*, activity_tokens_6label_fullstat.csv
- **PUBLICATION_TASK3_CORRECTED_FINAL** (14 CSVs + TOKEN_NEURAL_CHECKPOINTS_LEAKAGE_FREE_V1/ subfolder not expanded): task3_core_CORRECTED_*, task3_token_neural_LEAKAGE_FREE_*, task3_grammar_*
- **PUBLICATION_TASK3_FULL_COMPARISON** (17 CSVs + TOKEN_NEURAL_CHECKPOINTS/ subfolder not expanded): task3_grammar_*, task3_token_neural_*, task3_core_publication_comparison_with_std.csv
- **INTERACTION_BINARY_5S_ADVANCED_FEATURES** (10 CSVs, sizes up to 126 MB: binary_5s_all_sensor_advanced_features.csv)
- **PUBLICATION_TASK2_FULL_COMPARISON** (9 items: 2 CSVs + 7 subfolders — DL_NO_ELAPSED_ALL_ARCHITECTURES/, three_class_activity/, merging_vs_building/, conversation_vs_merging/, conversation_vs_building/, conversation_vs_nonconversation/ — none expanded further)
- **PUBLICATION_TASK1_FULL_COMPARISON** (5 CSVs + DL_NO_ELAPSED_ALL_ARCHITECTURES/, interaction_vs_noninteraction/ subfolders not expanded)
- **PUB_TASK2_ACTIVITY_10S** (7 CSVs, largest task2_merged_all_features_10s_AUGMENTED.csv 25.7 MB)
- **PUB_TASK2_ACTIVITY_5S** (7 CSVs, largest task2_merged_all_features_5s_AUGMENTED.csv 51.1 MB)
- **PUB_TASK1_INTERACTION** (9 items: 8 CSVs + `raw64_cache/` subfolder + `.ipynb_checkpoints/` [clutter, not enumerated]); largest binary_5s_specialized_oe_merged_all_features_AUGMENTED.csv 119.1 MB
- **RQ3_ACTIVITY_TOKEN_FULLSTAT** (7 files, token-level HMM/Markov comparison CSVs + 1 PNG)
- **ALL_SENSOR_MULTI_TASK_ABLATIONS** (8 loose files incl. `seven_winners_full9.csv`, `winners_full9.csv`, `winners_predictions_full9.csv`, `seven_winners_naive5.csv`, `winners_naive5.csv`, `winners_predictions_naive5.csv`, `combined_classical_best_per_condition.csv`, `combined_classical_summary.csv` + 8 subfolders: FAST_DL_EXPLAINABILITY, FAST_DL_BILSTM_TRANSFORMER_NO_ELAPSED, FAST_DL_NO_ELAPSED, three_class_activity, merging_vs_building, conversation_vs_merging, conversation_vs_building, conversation_vs_nonconversation, interaction_vs_noninteraction (none expanded further)
- **ACTIVITY_3CLASS_DEEP_SEQUENCE_ADVANCED_MULTIMODAL_NO_ELAPSED / _ADVANCED_NO_ELAPSED / _NO_ELAPSED** (5, 5, 5 CSVs respectively — predictions/summary/fold_metrics/aggregate_by_setting/best_per_feature_set, near-identical naming pattern across the three variants)
- **INTERACTION_BINARY_5S_MODEL_READY_FIXED / _MODEL_READY** (4 and 3 CSVs — label_column_report, base_windows, source_report, +features file)
- **INTERACTION_BINARY_5S** — folder too large to enumerate in one call (result exceeded tool size limit); contains at minimum the same label/report/features CSV family as the _FIXED variant, likely plus large sensor feature CSVs. Not fully listed.
- **INTERACTION_ABLATIONS** (3 CSVs, largest activity3_advanced_merged_10s_features.csv 24.5 MB)
- **INTERACTION_XSENS2** (4 CSVs: xsens2_multimodal_search_predictions/summary, xsens2_source_selection_report, interaction_xsens2_10s.csv)
- **INTERACTION_OPTI2** (4 CSVs: opti2_rich_search_predictions/summary, interaction_opti2_10s.csv, opti2_source_selection_report.csv)
- **INTERACTION_OE10** (24 files: interaction_oe10_10s.csv 19.8 MB, oe_best_plus_*_predictions/summary.csv (4 variants) + matching `_used_features.txt`, oe10_mag_focused_search_*, oe10_missing_motion_mag_search_*, + 2 subfolders `final_feature_lists/`, `explainability_best_model/` not expanded)
- **INTERACTION_ENG7_CONTEXT** (3 CSVs)
- **LABEL_DIAGNOSTICS** (24 CSVs — group-pair/merging/label-coverage diagnostic tables, e.g. `raw_group_pair_core_*`, `merging_related_*`, `lost_*`, `raw_model_ready_*`)
- **INTERACTION_OE8** (3 files: interaction_oe8_10s.csv 8.3 MB, oe8_FAST_logreg_rf_predictions/summary)
- **INTERACTION_ENG7** (4 CSVs)
- **INTERACTION_ENG6** (1 CSV: interaction_eng6_5s.csv, 55.7 MB)
- **INTERACTION_ENG5** (1 CSV: interaction_eng5_coupling_10s.csv)
- **INTERACTION_ENG4** (1 CSV: interaction_eng4_features.csv)
- **INTERACTION_ENG_30S** (1 CSV: interaction_eng30_features.csv)
- **RECOGNITION_5CLASS** (2 files: recognition_fixed_tensors_5class.npz, recognition_fixed_features_5class.csv)
- **RECOGNITION_ENG** (5 files: recognition_labeled/fixed features.csv + .npz tensors + recognition_eng_results_logo.csv)
- **INTERACTION_ENG3** — folder too large to enumerate in one call (result exceeded tool size limit). Not fully listed.
- **INTERACTION_ENG_ALL** — queried, returned **0 files** (empty folder).
- **RECOGNITION_ENG2** (4 files: interaction_eng_results(_logo).csv, interaction_eng_tensors.npz, interaction_eng_features.csv)
- **INTERACTION_ENG** (4 files, same naming family as RECOGNITION_ENG2)
- **INTERACTION_ENG2** (5 files incl. sensor_ablation_summary.csv, sensor_ablation_per_group.csv)
- **RECOGNITION_WINDOWS** (5 files: recognition_results.csv + labeled/fixed features+tensors)
- **INTERACTION_WINDOWS** (5 files: interaction_results.csv, interaction_labeled/fixed features.csv, interaction_labeled/fixed tensors.npz — fixed tensor is 81.6 MB)
- **ML_OUTPUT** — empty (0 files)
- **ML_DATASETS** (26 CSVs — model1/model2/model2B/model3/model3A/model3B token datasets, fused_multimodal_activity_tokens*, sync_time_* diagnostics; several files ~15 MB)
- **ALL_MODEL_READY_FILES** (26 files: group_{1,2,3,5,6,7,8,9,10}_{optitrack,xsens,openearable}_model_ready.csv (up to 508 MB for group_9 optitrack) + ALL_MODEL_READY_FILES_manifest.csv + ALL_MODEL_READY_FILES_missing.csv (1 byte — effectively empty/placeholder))
- **ML_MODELS** (24 subfolders, model1/model2/model2B/model3A/model3B variant training-run folders, e.g. `model1_interaction_transformer`, `model3B_masked_5class_bidirectional_transformer_LOGO`, `extra_conversation_subtype_*` — none expanded further; likely contain checkpoints/metrics per run)
- **model_ready_reports** (9 CSVs: FINAL_model_ready_*, model_ready_processing_report.csv, model_ready_missing_or_skipped_files.csv (1 byte, effectively empty))
- **optitrack_final_summary** (4 items: optitrack_supervisor_table.csv, optitrack_all_groups_summary.csv, optitrack_all_groups_take_level_summary.csv, + `plots/` subfolder not expanded)
- **RQ3_ACTIVITY_TOKEN_RICH, RQ3_ACTIVITY_TOKEN_SEQ, RQ3_FULLHISTORY_TRANSFORMER, RQ3_6LABEL_HMM, RQ3_7LABEL_HISTORY_AWARE_PREDICTION, RQ3_7LABEL_SEGMENT_SENSOR_FORECAST, RQ3_TEMPORAL_PREDICTION_7_PROCESS_LABELS, RQ3_TEMPORAL_PREDICTION_NORMALIZED_FEATURE_MERGED, RQ3_LABEL_NORMALIZATION** — discovered as top-level `data/` folders but **not individually expanded** in this pass (time budget); by naming they parallel the RQ3_* notebooks in `THESIS_NOTEBOOKS/ML/` §3.1 and almost certainly hold the corresponding CSV outputs for each RQ3 modelling variant.

**Bottom line for `data/`:** it is the authoritative base-data root. Raw per-participant sensor CSVs live under `data/group_N/{optitrack,xsens,openearable,elan}/`, N ∈ {1,2,3,5,6,7,8,9,10} (**no group_4, confirmed**). Everything else in `data/` is derived: per-experiment feature/prediction/summary CSV bundles (dozens of folders, mostly 1–20 files each) and the large `ALL_MODEL_READY_FILES` / `ALL_MODEL_READY_FILES_IDENTITY_FIXED` consolidated per-group model-ready CSVs (biggest single files in the whole tree, up to ~508 MB).

---

## 3. `THESIS_NOTEBOOKS/` root

Direct children of `1lWaPul53Zj685972o9tN5JdEtZUtI7j3`:

| Folder | id |
|---|---|
| ML | 1-K8okjrfHsW8CmNGTFd1W0wesumlLC6H |
| CLEAN_ML | 1_Xl940ENnTlrkRT8Pbjkq1_W9puex2Vk |
| PRE_PROCESSING | 1J6yxanegUQkFM8s-3kzlY2VTTTeOdIbh |

### 3.1 ML (23 notebooks, flat folder, no subfolders)

| Title | id | Size | Modified |
|---|---|---|---|
| all_sensor_multitask_ablation_notebook_with_merging_building.ipynb | 1YFhFO6w4uDwP45yvLWP5F5EUecVajb_c | 7,940,830 B | 2026-08-11 |
| rq3_activity_token_fullstat.ipynb | 1CXaoGbcYyrErZEqfOCCas_bTvfk74ABS | 281,187 B | 2026-07-16 |
| rq3_activity_token_enriched.ipynb | 1fxTxLg1kFZ9CiWcgAmTYXQ_OcykXjSCk | 22,098 B | 2026-07-06 |
| rq3_activity_token_bigger_models.ipynb | 1ju7cc7HBE2N1DaBfUMgN6RYPaYN9CphQ | 18,949 B | 2026-07-06 |
| rq3_activity_token_sequence_prediction.ipynb | 1IKw8iMO3t9VFTMFSh2TnjC5fyz6hR862 | 17,878 B | 2026-07-06 |
| rq3_fullhistory_transformer_prediction.ipynb | 15BS9BxKMaj_EqciDD02T03zcq6jtwewm | 19,346 B | 2026-07-06 |
| rq3_6label_segment_sensor_forecasting.ipynb | 1w1ySAZJ2zsGNQi6IV-rXJZB8rkEZMKtI | 125,569 B | 2026-07-06 |
| rq3_6label_history_aware_prediction.ipynb | 19UN5Sli4UwcxgBef0lPrCkPOAcv2NxJ8 | 183,314 B | 2026-07-06 |
| rq3_6label_hmm_prediction_and_smoothing.ipynb | 1FMgwdU7xCg3fckccn6wbjOVdsH-Y5Cd- | 19,734 B | 2026-07-06 |
| oe_conversation_nonconversation_experiment.ipynb | 1HLc04mPH_av-5u8KzekfANoQk5L3lbK9 | 1,644,274 B | 2026-07-04 |
| rq3_7label_segment_sensor_forecasting (1).ipynb | 14yENbIMCBdIWB21sG81POxx1XsUCNyPt | 127,887 B | 2026-07-06 |
| rq3_7label_history_aware_prediction.ipynb | 12SZBcbTiZSMdt-WAevydKqbuqy4M71Ui | 154,116 B | 2026-07-06 |
| THESIS_NEW_ML.ipynb | 12GEgdOjD6KOekAMOasdK1AmuXEc6IXpV | 5,299,686 B | 2026-06-25 |
| **Untitled10.ipynb** | 1L_kpgz2kJs-EE-Ny60HbV5JNccukrfU1 | 324 B | 2026-06-22 |
| classic_all_sensor_multitask_ablation_notebook_with_merging_building.ipynb | 1Tay88HGMABrt6ebvmLnzlCh9EyHgT1kY | 109,115 B | 2026-07-05 |
| rq3_temporal_prediction_7_process_labels.ipynb | 1EbYCNNmxUsf1riXoDWrC7DK9E12qQLQk | 155,437 B | 2026-07-05 |
| rq3_temporal_prediction_normalized_feature_merged.ipynb | 1XxDZe89nWbKxPgFahHZArGN_sQVdNtY0 | 132,857 B | 2026-07-05 |
| rq3_label_audit_and_normalization.ipynb | 167RIbcj76QNetYhA75iRCABtAgs1pTx0 | 177,754 B | 2026-07-05 |
| TASK3.ipynb | 1rXOQtMYSbvHgG-ZioVxoekVJz66CKLyN | 2,588,835 B | 2026-07-01 |
| GAR_Complete_Selfcontained.ipynb | 1VN7WZn8Ohf2hdLfbiHTZjWQvSb3gYbO- | 674,969 B | 2026-06-23 |
| MACHINE_LEARNING_MODELS.ipynb | 1JKIdSTMGuFQNir_XNn6EM6xrv7xf5jpt | 7,774,281 B | 2026-06-12 |
| GAR_Final_Analysis.ipynb | 16Kx69kxyawgnDr-yC9gmoppkTXwOgp5N | 490,144 B | 2026-06-22 |

**Flags:** `Untitled10.ipynb` (324 bytes — essentially an empty/blank notebook) is stray clutter. `GAR_Final_Analysis.ipynb` was not in the caller's "already known" list — newly confirmed. `rq3_7label_segment_sensor_forecasting (1).ipynb` has the Drive `" (1)"` conflict suffix, suggesting a duplicate-upload artifact (the un-suffixed original was not found in this folder — it may have been renamed/replaced).

### 3.2 CLEAN_ML (14 files, flat folder) — heavy duplication

| Title | id | Size | Modified |
|---|---|---|---|
| 07_feature_engineering_ENG7_activity_invariant (1).ipynb | 1KN7UoBv-wWt41ASgVGqe29xHOuOa20LW | 27,251 B | 2026-06-27 |
| 07_feature_engineering_ENG7_activity_invariant (1).ipynb | 1eR2T--SgWMD2Cx7d3Spj9ZfZwH2YWMFT | 9,090,019 B | 2026-07-01 |
| 07_feature_engineering_ENG7_activity_invariant.ipynb | 1bSuUavPow8if9u6H8T9RcINTFypHiw-K | 186,880 B | 2026-06-27 |
| 06_feature_engineering_ENG6_classic_togetherness.ipynb | 1LL29aWiW_Tta4UwZfI6oRVsiyGA7iIaF | 43,082 B | 2026-06-27 |
| 05_feature_engineering_ENG5_coupling.ipynb | 1mQjeHw5ihDdjAXOSxpTBNLCJGScLBeUC | 26,438 B | 2026-06-27 |
| 04_feature_engineering_ENG4.ipynb | 1kFNapDoYA3UO1UIqIZ1_SBojQn-0hZ0u | 33,476 B | 2026-06-27 |
| 01_interaction_detection.ipynb | 1Nj8I8DuyWNyyPuI6AUppTVgulGls7wSV | 96,449 B | 2026-06-27 |
| 01_interaction_detection.ipynb | 1aOP9LXV5CQd1AIuO6sboR1v9QC3c9-Uv | 82,179 B | 2026-06-27 |
| 01_interaction_detection.ipynb | 1Gj3WRO1M_rsFO7ZfJFtnqtp8K42R-F3V | 85,586 B | 2026-06-27 |
| 03_activity_prediction.ipynb | 1MRWXEJZENYRQf13phzEhVSNrWnZPbSeP | 194,834 B | 2026-06-27 |
| 03_activity_prediction.ipynb | 12HqQHgwU2VQ5t90wD9E6KdTFVapAhjJd | 194,834 B | 2026-06-27 |
| 02_activity_recognition.ipynb | 1U0mjfqYp8X7D9sAMV1NVRy6ipqYbwGrF | 165,013 B | 2026-06-27 |
| 02_activity_recognition.ipynb | 1kInEREhFChd1LyhZYKOpxbIOqyLWztqI | 165,013 B | 2026-06-27 |
| **Untitled** | 1yBmz8u4jGjVanNasKyxJmjy_vPx-YtKe | 306 B | 2026-06-27 |

**Flags (heavy duplication in this folder):**
- `01_interaction_detection.ipynb` exists **3 times** (3 different ids/sizes).
- `02_activity_recognition.ipynb` exists **2 times**, and both copies are **byte-identical in size** (165,013 B) — likely a true duplicate upload, not a revision.
- `03_activity_prediction.ipynb` exists **2 times**, also identical size (194,834 B) — likely exact duplicates.
- `07_feature_engineering_ENG7_activity_invariant (1).ipynb` exists **twice** with wildly different sizes (27 KB vs 9.09 MB) despite the identical "(1)" name — the 9 MB one almost certainly has heavy embedded output/data and is a different save than the 27 KB one; needs manual disambiguation.
- **`Untitled`** (306 B, no extension) is a stray blank notebook — clutter.
- Given the "CLEAN_ML" name, this folder is ironically the messiest of the three THESIS_NOTEBOOKS subfolders.

### 3.3 PRE_PROCESSING (8 notebooks, flat folder) — duplication + odd filename

| Title | id | Size | Modified |
|---|---|---|---|
| Global_Cleaning_Before_Model.ipynb | 1V-u_sdeb_5OvNbZomwBQIwcyjJeCzuah | 561,295 B | 2026-06-11 |
| **OPTI_TRACK_PROCESSINGipynb** | 1h238-RElP5LeAUoVN_1-N4XFqkDl3cBG | 56,522,741 B | 2026-06-10 |
| Arda_Thesis_Data_Preprocessing.ipynb | 1RDpDdaTLeU9T64bB6cmGK2m_E0zpQ3Ej | 329,326 B | 2026-06-02 |
| Arda_Thesis_Data_Preprocessing_FIXED_v4.ipynb | 1gHos_oJEFFR57enJw6_gAS7GJuyp3t2c | 329,763 B | 2026-06-01 |
| Arda_Thesis_Data_Preprocessing_FIXED.ipynb | 1EvsR6lnom42X-nrIterpvxPSRb7MKDer | 68,582 B | 2026-06-01 |
| sensor_sync_fixed (1).ipynb | 12hnrtqxWKHuw6_aJLAIl_iU5iU6OCIyr | 336,321 B | 2026-05-28 |
| sensor_sync_fixed (1).ipynb | 19NoNLSZ_KG7py-FdxU_iEnbv7vQjMyvj | 997,789 B | 2026-05-26 |
| sensor_sync_fixed.ipynb | 1jdAxF5F28MmJ_izbq5yB6QHSCeCuXsMT | 997,584 B | 2026-05-26 |
| sensor_sync (1).ipynb | 1i32vnj5fZD2L9KlqHnGx0MU3qgImBiVz | 994,677 B | 2026-05-26 |

**Flags:**
- `OPTI_TRACK_PROCESSINGipynb` — **missing the dot before "ipynb"** in the title (should presumably be `OPTI_TRACK_PROCESSING.ipynb`); `fileExtension` is empty string, confirming Drive doesn't recognize it as a proper `.ipynb`. This is also the single largest notebook found anywhere in either tree (56.5 MB — confirms the caller's prior note).
- `sensor_sync_fixed (1).ipynb` exists **twice** (336 KB and 998 KB) alongside `sensor_sync_fixed.ipynb` (998 KB, same size as one of the "(1)" copies) and `sensor_sync (1).ipynb` — 4 near-identical sensor-sync notebook variants in one folder, prime candidates for cleanup.
- `Arda_Thesis_Data_Preprocessing*.ipynb` has 3 generations (base, `_FIXED`, `_FIXED_v4`) — keep `_FIXED_v4` only.

---

## 4. Overall red flags / cleanup candidates

1. **`github notebooks/`** (root files, not the `actual ones` subfolder) — 5 generations of `01_interaction_vs_noninteraction*.ipynb` and 4 generations of `master_feature_generator_task1_task2_task3*.ipynb` coexist. Use `github notebooks/actual ones/` (3 files, most recently modified) as canonical instead.
2. **`CLEAN_ML/`** — despite its name, has the worst duplication: `01_interaction_detection.ipynb` ×3, `02_activity_recognition.ipynb` ×2 (identical size), `03_activity_prediction.ipynb` ×2 (identical size), plus a stray blank `Untitled` notebook and a same-named-but-different-size `(1)` pair.
3. **`PRE_PROCESSING/`** — `OPTI_TRACK_PROCESSINGipynb` has a malformed filename (missing `.` before extension) and is the largest notebook in the whole inventory (56.5 MB); 4 overlapping `sensor_sync*` variants.
4. **`ML/`** — stray blank `Untitled10.ipynb` (324 B).
5. **`thesis/output/group_3`** — contains files literally named `group5_oe_labelled (1).csv` / `group5_xsens_labelled (1).csv`, i.e. group_3's folder holds group_5-named files. Likely a copy/rename mistake — worth a manual check.
6. **`" (1)"` Drive-conflict-suffix files** appear repeatedly across the tree (`github notebooks` "actual ones" is clean, but `ML/rq3_7label_segment_sensor_forecasting (1).ipynb`, `CLEAN_ML/07_feature_engineering_ENG7_activity_invariant (1).ipynb` ×2, `PRE_PROCESSING/sensor_sync_fixed (1).ipynb` ×2, `PRE_PROCESSING/sensor_sync (1).ipynb`, `data/output/group_2` and `group_3` model CSVs) — these mark re-uploads that collided with an existing file of the same name; several coexist with a same-named non-suffixed file of a different size, meaning both are genuinely different content, not simple dupes.
7. **1-byte placeholder files**: `data/ALL_MODEL_READY_FILES/ALL_MODEL_READY_FILES_missing.csv` and `data/model_ready_reports/model_ready_missing_or_skipped_files.csv` are both exactly 1 byte — effectively empty (no missing files to report), not real data.
8. **`.ipynb_checkpoints/`** clutter folders were seen under `data/group_1/elan/` and `data/PUB_TASK1_INTERACTION/` and `data/group_1/xsens/elan`-equivalent paths — present but intentionally not enumerated per instructions.

## 5. `group_4` finding

**Confirmed: `group_4` does not exist anywhere in either tree.** Checked explicitly in:
- `thesis/data/` (direct children list — groups 1,2,3,5,6,7,8,9,10, group_4 absent)
- `thesis/output/` (groups 1,2,3,5,6 only)
- `thesis/final_visualizations`, `final_visualizations_multisensor`, `final_visualizations_participant_lines` (all use groups 1,2,3,5,6,7,8,9,10)

The group numbering consistently skips from 3 to 5 everywhere in the Drive structure — this is very likely an intentional exclusion (e.g. group_4's session/recording was discarded or excluded from the study), not a listing/pagination artifact.

---

## 6. Not fully enumerated (time/size budget) — candidates for a follow-up pass

- `data/INTERACTION_BINARY_5S` and `data/INTERACTION_ENG3` — individual `search_files` calls exceeded the tool's response size limit; contents unknown beyond what's inferable from sibling `_FIXED`/`_MODEL_READY` folders.
- `data/group_2, 3, 5, 6, 7, 8, 9` raw sensor subfolders — not individually expanded (assumed identical `optitrack/xsens/openearable/elan` template to group_1 and group_10, which were confirmed).
- `data/RQ3_ACTIVITY_TOKEN_RICH`, `RQ3_ACTIVITY_TOKEN_SEQ`, `RQ3_FULLHISTORY_TRANSFORMER`, `RQ3_6LABEL_HMM`, `RQ3_7LABEL_HISTORY_AWARE_PREDICTION`, `RQ3_7LABEL_SEGMENT_SENSOR_FORECAST`, `RQ3_TEMPORAL_PREDICTION_7_PROCESS_LABELS`, `RQ3_TEMPORAL_PREDICTION_NORMALIZED_FEATURE_MERGED`, `RQ3_LABEL_NORMALIZATION` — discovered as top-level `data/` folder names only, contents not queried.
- `data/ML_MODELS/*` (24 training-run subfolders) and various `*CHECKPOINTS*` / `raw64_cache` / `_inspection_report` subfolders — noted to exist but not expanded (expected to contain many near-identical checkpoint/metric files per run).
