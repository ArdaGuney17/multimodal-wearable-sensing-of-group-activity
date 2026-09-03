"""Global cleaning / model-ready assembly (Ch.4 §4.7-4.9).

Ported from `PRE_PROCESSING/Global_Cleaning_Before_Model.ipynb`, cells
"FINAL MODEL-READY PROCESSING WITH MANUALLY SELECTED FILES" (cell 3),
"FIX GROUP 1 XSENS EXTRA LABEL COLUMNS" (cell 5), "FINAL SANITY CHECK FOR
ALL MODEL-READY FILES" (cell 4, read-only), "COPY ALL MODEL-READY FILES
INTO ONE CENTRAL FOLDER" (cell 6), and "ROBUST FIX OPTITRACK PARTICIPANT
IDENTITY" (cell 7 — the notebook's own final cell). See
notebooks_reference/Global_Cleaning_Before_Model_ANALYSIS.md for the full
porting notes.

INPUT DEPENDENCY / KNOWN GAP (see docs/table_to_source_mapping.md): this
stage's input is a set of already time-synced, already ELAN-labeled
per-group per-sensor CSVs (`{data_root}/group_{g}/{sensor}/{sensor}_labeled/
<filename>`), hand-selected below in SELECTED_FILE_NAMES. The code that
actually produced most of those files (sync/label attachment for groups
2,3,6,7,8,9,10) is not present in any notebook found in Drive — only
group 1's sync code (`sensor_sync_fixed.ipynb`) was recoverable. Those
`*_labeled*.csv` files themselves still exist in the original Drive data
and are treated here as a **fixed intermediate input**, the same way raw
sensor data would be, rather than something this pipeline re-derives.

Not actually windowed: despite the docs' Ch.4 §4.7-4.9 heading
("windowing"), this stage produces per-sample (native-rate) output —
every "model_ready" row is still one original sensor sample. Windowing
into fixed 5s/10s segments happens downstream, in the feature-engineering
stage (src/features/).

Pipeline (four steps, run in this order by run_all()):
  1. build_model_ready_files() — cleans label text (typo/spelling
     normalization via GLOBAL_LABEL_REPLACEMENTS) and trims each sensor
     file to its first..last labeled row range.
  2. fix_group1_xsens_extra_label_columns() — a genuine, narrow,
     group-1-only patch (drops 3 stray non-canonical label columns
     discovered via the sanity check; preserved as a named step, not
     silently generalized to other groups).
  3. copy_to_central_folder() — flattens the per-group/per-sensor
     model_ready files into one folder, ALL_MODEL_READY_FILES/.
  4. apply_optitrack_identity_fix() — OptiTrack's raw rigid-body/marker
     prefixes are arbitrary per session; infers each one's physical
     left/middle/right position from its mean x-coordinate and uses
     PARTICIPANT_POSITION_MAP (a hand-built per-group seating chart) to
     rename columns to ParticipantN. OpenEarable/Xsens files pass through
     unchanged (already correctly participant-attributed from ELAN).
     Writes the final ALL_MODEL_READY_FILES_IDENTITY_FIXED/ this whole
     project's downstream code (src/features/eng2_features.py and
     several src/models/task3_*.py modules) consumes.
"""

from __future__ import annotations

import os
import re
import shutil

import numpy as np
import pandas as pd

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group_4 excluded (camera failure) — canonical everywhere in this repo
SENSORS = ["openearable", "xsens", "optitrack"]

CANONICAL_LABEL_COLS = [
    "label_Participant1", "label_Participant2", "label_Participant3",
    "label_Participant1_Participant2", "label_Participant1_Participant3",
    "label_Participant2_Participant3", "label_Whole_Group",
]

# Human-curated "winning" input file per (group, sensor) — each group
# needed a different upstream time-sync correction method (clap sync,
# last-10-peak, sync-peak, early-peak), chosen by hand; a human picked
# which corrected variant to trust. All live under the uniform
# `{data_root}/group_{g}/{sensor}/{sensor}_labeled/` directory.
SELECTED_FILE_NAMES = {
    (1, "openearable"): "group_1_openearable_labeled.csv",
    (1, "xsens"): "group_1_xsens_labeled_shifted_by_last10_peak.csv",
    (1, "optitrack"): "group_1_optitrack_labeled.csv",

    (2, "openearable"): "group_2_openearable_labeled.csv",
    (2, "xsens"): "group_2_xsens_labeled.csv",
    (2, "optitrack"): "group_2_optitrack_labeled.csv",

    (3, "openearable"): "group_3_openearable_labeled_shifted_by_sync_peak.csv",
    (3, "xsens"): "group_3_xsens_labeled.csv",
    (3, "optitrack"): "group_3_optitrack_labeled.csv",

    (5, "openearable"): "group_5_openearable_labeled.csv",
    (5, "xsens"): "group_5_xsens_labeled_shifted_by_clap_sync_peak.csv",
    (5, "optitrack"): "group_5_optitrack_labeled.csv",

    (6, "openearable"): "group_6_openearable_labeled.csv",
    (6, "xsens"): "group_6_xsens_labeled_cleaned_SHIFTED.csv",
    (6, "optitrack"): "group_6_optitrack_labeled.csv",

    (7, "openearable"): "group_7_openearable_labeled_SHIFTED.csv",
    (7, "xsens"): "group_7_xsens_labeled_SHIFTED.csv",
    (7, "optitrack"): "group_7_optitrack_labeled.csv",

    (8, "openearable"): "group_8_openearable_labeled_SHIFTED.csv",
    (8, "xsens"): "group_8_xsens_labeled_SHIFTED.csv",
    (8, "optitrack"): "group_8_optitrack_labeled.csv",

    (9, "openearable"): "group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv",
    (9, "xsens"): "group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv",
    (9, "optitrack"): "group_9_optitrack_labeled.csv",

    (10, "openearable"): "group_10_openearable_labeled_SHIFTED.csv",
    (10, "xsens"): "group_10_xsens_labeled_SHIFTED.csv",
    (10, "optitrack"): "group_10_optitrack_labeled.csv",
}

# Hand-built per-group seating chart — which physical position
# (left/middle/right, by mean OptiTrack x-coordinate) each canonical
# ParticipantN occupied during that session. Required for the OptiTrack
# identity fix; has no generic derivation, must stay a literal per-group
# table.
PARTICIPANT_POSITION_MAP = {
    1: {"Participant1": "left", "Participant2": "right", "Participant3": "middle"},
    2: {"Participant1": "right", "Participant2": "left", "Participant3": "middle"},
    3: {"Participant1": "middle", "Participant2": "right", "Participant3": "left"},
    5: {"Participant1": "middle", "Participant2": "right", "Participant3": "left"},
    6: {"Participant1": "middle", "Participant2": "left", "Participant3": "right"},
    7: {"Participant1": "right", "Participant2": "left", "Participant3": "middle"},
    8: {"Participant1": "right", "Participant2": "middle", "Participant3": "left"},
    9: {"Participant1": "left", "Participant2": "right", "Participant3": "middle"},
    10: {"Participant1": "left", "Participant2": "middle", "Participant3": "right"},
}

# The ~150-entry dictionary embedded in the notebook's own cell 3 (the
# "FINAL" manually-selected-files cell) — this is the one that actually
# determines the shipped label text (it runs after, and overwrites the
# output of, an earlier automatic-selection cell with a larger ~210-entry
# dict; see the module docstring and the notebook's own ANALYSIS.md for
# why the smaller dict, not the larger one, is canonical here).
GLOBAL_LABEL_REPLACEMENTS = {
    # Synchronization
    "synchronaziton_move": "synchronization_move",
    "synchronaztion_move": "synchronization_move",
    "synchronizaiton_move": "synchronization_move",
    "synchronisation_move": "synchronization_move",
    "clap_synchronizaiton_move": "clap_synchronization_move",
    "droping_erarble_syncornaziton_move": "dropping_earable_synchronization_move",
    "khalil_synchornizaion_move": "participant1_synchronization_move",

    # Conversation
    "task_operaitona_convo": "task_operational_convo",
    "task_operaitonal_convo": "task_operational_convo",
    "task_operaitonal_talk": "task_operational_convo",
    "task_operatinal_convo": "task_operational_convo",
    "task_operation_convo": "task_operational_convo",
    "task_operationa_convo": "task_operational_convo",
    "task_operational convo": "task_operational_convo",
    "task_operational_convo,": "task_operational_convo",
    "task_operatşonal_convo": "task_operational_convo",
    "task_opraitonal_convo": "task_operational_convo",
    "task_operational_talk": "task_operational_convo",
    "task_realted_convo": "task_related_convo",
    "task_related_dialouge": "task_related_convo",
    "task_socail_convo": "task_social_convo",
    "task_social_talk": "task_social_convo",
    "task_social_sensor": "task_social_convo",

    # Individual / co-building
    "individual_bu": "individual_build",
    "individual_building": "individual_build",
    "start_of_individual_build": "starting_individual_build",

    "cbuilding_subpiece": "co_building_subpiece",
    "cobuilding_subpiece": "co_building_subpiece",
    "co_buiilding_subpiece": "co_building_subpiece",
    "co_building_sub_part": "co_building_subpiece",
    "co_building_sub_piece": "co_building_subpiece",
    "co_building_sub_pieec": "co_building_subpiece",
    "co_building_subpart": "co_building_subpiece",
    "co_building_the_subpiece": "co_building_subpiece",
    "co_builidng_sub_piece": "co_building_subpiece",
    "co_builidng_subpiece": "co_building_subpiece",
    "co_bulding_subpiece": "co_building_subpiece",
    "co_building_pieces": "co_building_piece",
    "co_building_puzzle": "co_building_piece",
    "buılding_subpiece_together": "building_subpiece_together",

    # Merging
    "merging_sub_piece": "merging_subpiece",
    "merging_sub_pieces": "merging_subpiece",
    "merging_subpieces": "merging_subpiece",
    "mering_sub_pieces": "merging_subpiece",
    "mering_subpieces": "merging_subpiece",
    "co_merging_pieces": "co_merging_subpiece",
    "co_merging_sub_pieces": "co_merging_subpiece",
    "co_merging_subpieces": "co_merging_subpiece",

    # Inspecting
    ",,nspecting_pieces": "inspecting_pieces",
    "inscpecting_pieces": "inspecting_pieces",
    "inspecitng_pieces": "inspecting_pieces",
    "inspectingpiece": "inspecting_piece",
    "inspecitng_piece": "inspecting_piece",
    "inscpecing_other_pieces": "inspecting_other_pieces",
    "inspecitng_other_pieces": "inspecting_other_pieces",
    "inspecitng_other_puzlle_pieces": "inspecting_other_puzzle_pieces",
    "inspecitng_other_unit": "inspecting_other_units",
    "inspecitng_other_units": "inspecting_other_units",
    "inspecitng_other_untis": "inspecting_other_units",
    "inspecting_other_unit": "inspecting_other_units",
    "inspecting_others_unit": "inspecting_other_units",
    "inscpeing_target_image": "inspecting_target_image",
    "inspecitng_target_image": "inspecting_target_image",
    "inspecting_target_imge": "inspecting_target_image",
    "isnpecting_setup": "inspecting_setup",
    "co_inspceitng_image": "co_inspecting_image",
    "co_inspecitng_image": "co_inspecting_image",
    "co_inspecitng_pieces": "co_inspecting_pieces",

    # Matching
    "matchin_pieces_with_target_image": "matching_pieces_with_target_image",
    "mathicng_pieces_to_target_image": "matching_pieces_to_target_image",
    "mathing_pieces_with_image": "matching_pieces_with_image",
    "matching_image_to_pieces": "matching_pieces_to_image",
    "piece_and_image_matching": "matching_pieces_to_target_image",

    # Carrying
    "carriyng_thray_to_central table": "carrying_tray_to_central_table",
    "carrying_the_tray": "carrying_tray",
    "caryying_the_tray": "carrying_tray",
    "caryying_tray": "carrying_tray",
    "carrying_the_tray_to_central_table": "carrying_tray_to_central_table",
    "carrying_tray_to_centraltable": "carrying_tray_to_central_table",
    "carying_subpiece_with_tray": "carrying_subpiece_with_tray",

    # Delivering
    "delivering_pieceBA": "delivering_piece",
    "delivering_pieceRB": "delivering_piece",
    "delivering_pieceS": "delivering_piece",
    "delivering_piece_toS": "delivering_piece",
    "delivering_target_imageRB": "delivering_target_image",
    "delivering_trayBR": "delivering_tray",

    # Object handover
    "object_handıver": "object_handover",
    "pbject_handover": "object_handover",

    # Picking up
    "pickign_up_pice": "picking_up_piece",
    "pickinf_up_pieces": "picking_up_pieces",
    "pickinup_piece": "picking_up_piece",
    "pickinupiece": "picking_up_piece",
    "picking_up_pieceS": "picking_up_piece",
    "picking_up_piece_fS": "picking_up_piece",
    "picking_up_piecefS": "picking_up_piece",
    "picking_up_target_i": "picking_up_target_image",
    "picking_up_target_imageS": "picking_up_target_image",
    "picking_up_target_image_fR": "picking_up_target_image",
    "picking_up_targetimage": "picking_up_target_image",
    "picking_up_traget_image": "picking_up_target_image",
    "pickinig_up_tray": "picking_up_tray",
    "pıcking_up_image": "picking_up_image",

    # Placing / putting down
    "placing_sub_piece_to_tray": "placing_subpiece_to_tray",
    "placing_subpieces_to_tray": "placing_subpiece_to_tray",
    "placing_pieces_to_tray": "placing_subpiece_to_tray",
    "placing_subpiece_to_centraltable": "placing_subpiece_to_central_table",
    "placing_subpiece_to_centraltable_from_tray": "placing_subpiece_to_central_table_from_tray",
    "placing_subpiece_from_tray_to_centraltable": "placing_subpiece_from_tray_to_central_table",
    "placing_subpiece_from_tray_to_table": "placing_subpiece_from_tray_to_central_table",
    "puting_down_piece": "putting_down_piece",
    "putting_doen_piece": "putting_down_piece",
    "putting_down_piece_fS": "putting_down_piece",
    "putting_doen_target_image": "putting_down_target_image",
    "putting_down_tagret_image": "putting_down_target_image",
    "putting_doen_tray": "putting_down_tray",

    # Moving / traveling
    "moving_between_units": "traveling_between_units",
    "taveling_between_units": "traveling_between_units",
    "travel_between_units": "traveling_between_units",
    "travelin_between_units": "traveling_between_units",
    "traveling_betweein_units": "traveling_between_units",
    "traveling_between units": "traveling_between_units",
    "traveling_between__units": "traveling_between_units",
    "traveling_between_untis": "traveling_between_units",
    "traveling_betwen_units": "traveling_between_units",
    "traveling_betwwen_units": "traveling_between_units",
    "traveling_beween_units": "traveling_between_units",
    "trvaeling_between_units": "traveling_between_units",
    "trveling_between_units": "traveling_between_units",
    "co_traveking_between_units": "co_traveling_between_units",
    "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",

    "moving_pieces_to_central_table_from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_pieces_to_central_table_with_tray": "moving_pieces_from_tray_to_central_table",
    "moving_pieces_to_table__from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_subpiece_from_tray_to_central_table": "moving_pieces_from_tray_to_central_table",
    "moving_subpiece_to_central_table_from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_subpieces_from_tray_to_table": "moving_pieces_from_tray_to_central_table",
    "co_moving_pieces_from_tray_to_central_tabel": "co_moving_pieces_from_tray_to_central_table",
    "co_moving_the_pieces_to_tray": "co_moving_pieces_to_tray",
    "co_moving_subpiece_to_tray": "co_moving_pieces_to_tray",
    "moving_subpiece_to_tray": "moving_pieces_to_tray",
    "moving_subpieces_to_tray": "moving_pieces_to_tray",
    "moving_subpiece_from_tray_to_table": "moving_pieces_from_tray_to_central_table",

    # Searching / pointing / guide / approach
    "searchin_for_image": "searching_for_image",
    "searching_target_image": "searching_for_target_image",
    "search_for_target_image": "searching_for_target_image",
    "searching_piece": "searching_for_piece",
    "pointing_thowards_target_image": "pointing_to_target_image",
    "pointing_to_somethign": "pointing_to_something",
    "actiavely_using_guide_image": "actively_using_guide_image",
    "approcahing_to_piece": "approaching_to_piece",
}


# ================================================================
# Label normalization
# ================================================================

def normalize_single_label(label) -> str:
    if pd.isna(label):
        return ""
    label = str(label).strip()
    if label == "" or label.lower() in ("nan", "none", "null"):
        return ""

    label = label.replace("ı", "i").replace("İ", "I")
    label = re.sub(r"\s+", "_", label)
    label = label.strip(" _;,.")
    label = re.sub(r"_+", "_", label)

    label = GLOBAL_LABEL_REPLACEMENTS.get(label, label)

    label = label.replace("ı", "i").replace("İ", "I")
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"_+", "_", label)
    label = label.strip(" _;,.")
    label = GLOBAL_LABEL_REPLACEMENTS.get(label, label)

    return label


def normalize_label_cell(cell) -> str:
    if pd.isna(cell):
        return ""
    text = str(cell).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return ""

    text = text.replace("|", "+").replace("＋", "+")
    parts = [p.strip() for p in text.split("+")]

    cleaned: list[str] = []
    for p in parts:
        c = normalize_single_label(p)
        if "+" in c:
            for sp in (normalize_single_label(sp.strip()) for sp in c.split("+")):
                if sp != "" and sp not in cleaned:
                    cleaned.append(sp)
        elif c != "" and c not in cleaned:
            cleaned.append(c)

    return " + ".join(cleaned)


def get_label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_")]


def collect_unique_labels(df: pd.DataFrame, label_cols: list[str]) -> set:
    unique = set()
    for col in label_cols:
        for v in df[col].dropna().astype(str):
            if v.strip() == "":
                continue
            for p in str(v).replace("|", "+").split("+"):
                p = p.strip()
                if p:
                    unique.add(p)
    return unique


def clean_label_columns(df: pd.DataFrame):
    df = df.copy()
    label_cols = get_label_columns(df)
    before_unique = collect_unique_labels(df, label_cols)
    for col in label_cols:
        df[col] = df[col].apply(normalize_label_cell)
    after_unique = collect_unique_labels(df, label_cols)
    return df, before_unique, after_unique


def trim_to_labeled_range(df: pd.DataFrame):
    """Trims to the row range [first labeled row, last labeled row]
    (inclusive), across all label_* columns. Returns (trimmed_df,
    first_idx, last_idx, removed_before, removed_after); first_idx/
    last_idx are None if no label columns or no labeled rows exist."""
    label_cols = get_label_columns(df)
    if len(label_cols) == 0:
        return df.copy(), None, None, 0, 0

    labels_text = df[label_cols].fillna("").astype(str)
    any_label = labels_text.apply(lambda row: any(str(x).strip() != "" for x in row), axis=1)
    labeled_indices = np.where(any_label.to_numpy())[0]
    if len(labeled_indices) == 0:
        return df.copy(), None, None, 0, 0

    first_idx, last_idx = int(labeled_indices[0]), int(labeled_indices[-1])
    trimmed = df.iloc[first_idx:last_idx + 1].copy()
    return trimmed, first_idx, last_idx, first_idx, len(df) - last_idx - 1


def get_time_info(df: pd.DataFrame) -> dict:
    possible_cols = ["video_time_s", "time_s", "timestamp", "timestamp_s", "elapsed_time_s", "time"]
    out = {}
    for c in possible_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                out[f"{c}_min"] = float(vals.min())
                out[f"{c}_max"] = float(vals.max())
    return out


def add_model_ready_metadata(df: pd.DataFrame, group: int, sensor: str, source_path: str, first_idx, last_idx) -> pd.DataFrame:
    df = df.copy()
    df["model_ready_group"] = group
    df["model_ready_sensor"] = sensor
    df["model_ready_source_file"] = os.path.basename(source_path)
    df["model_ready_source_path"] = source_path
    df["model_ready_trim_first_original_row"] = first_idx
    df["model_ready_trim_last_original_row"] = last_idx
    return df


# ================================================================
# Step 1 (cell 3) — build per-group/per-sensor model_ready files
# ================================================================

def selected_file_path(data_root: str, group: int, sensor: str) -> str:
    filename = SELECTED_FILE_NAMES[(group, sensor)]
    return os.path.join(data_root, f"group_{group}", sensor, f"{sensor}_labeled", filename)


def model_ready_path(data_root: str, group: int, sensor: str) -> str:
    return os.path.join(data_root, f"group_{group}", sensor, "model_ready", f"group_{group}_{sensor}_model_ready.csv")


def build_model_ready_files(data_root: str, report_dir: str | None = None):
    """Cleans label text + trims to labeled range for every (group,
    sensor) in SELECTED_FILE_NAMES; writes each to model_ready_path().
    Returns (report_df, missing_df)."""
    report_dir = report_dir or os.path.join(data_root, "model_ready_reports")
    os.makedirs(report_dir, exist_ok=True)

    reports, missing_or_skipped = [], []

    for (group, sensor), _ in SELECTED_FILE_NAMES.items():
        input_path = selected_file_path(data_root, group, sensor)
        print(f"GROUP {group} | SENSOR {sensor.upper()} | input: {input_path}")

        if not os.path.exists(input_path):
            print("  MISSING")
            missing_or_skipped.append({"group": group, "sensor": sensor, "status": "missing_selected_file", "input_path": input_path})
            continue

        df = pd.read_csv(input_path, low_memory=False)
        original_rows, original_cols = len(df), len(df.columns)
        label_cols = get_label_columns(df)

        if len(label_cols) == 0:
            print("  ERROR: no label columns")
            missing_or_skipped.append({"group": group, "sensor": sensor, "status": "no_label_columns", "input_path": input_path})
            continue

        df_clean, before_unique, after_unique = clean_label_columns(df)
        df_trimmed, first_idx, last_idx, removed_before, removed_after = trim_to_labeled_range(df_clean)

        if first_idx is None:
            print("  ERROR: no labeled rows")
            missing_or_skipped.append({"group": group, "sensor": sensor, "status": "no_labeled_rows", "input_path": input_path})
            continue

        df_trimmed = add_model_ready_metadata(df_trimmed, group, sensor, input_path, first_idx, last_idx)

        out_path = model_ready_path(data_root, group, sensor)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_trimmed.to_csv(out_path, index=False)
        print(f"  saved: {out_path} | {df_trimmed.shape}")

        label_cols_after = get_label_columns(df_trimmed)
        labels_text = df_trimmed[label_cols_after].fillna("").astype(str)
        any_label_after = labels_text.apply(lambda row: any(str(x).strip() != "" for x in row), axis=1)

        report_row = {
            "group": group, "sensor": sensor, "status": "saved", "input_path": input_path, "output_path": out_path,
            "original_rows": original_rows, "original_cols": original_cols,
            "final_rows": len(df_trimmed), "final_cols": len(df_trimmed.columns),
            "removed_before": removed_before, "removed_after": removed_after,
            "first_original_row": first_idx, "last_original_row": last_idx,
            "label_columns_count": len(label_cols_after),
            "unique_labels_before_cleaning": len(before_unique), "unique_labels_after_cleaning": len(after_unique),
            "rows_with_any_label_after_trim": int(any_label_after.sum()),
            "pct_rows_with_any_label_after_trim": float(any_label_after.mean() * 100),
        }
        for k, v in get_time_info(df).items():
            report_row[f"original_{k}"] = v
        for k, v in get_time_info(df_trimmed).items():
            report_row[f"trimmed_{k}"] = v
        reports.append(report_row)

    report_df = pd.DataFrame(reports)
    missing_df = pd.DataFrame(missing_or_skipped)
    report_df.to_csv(os.path.join(report_dir, "FINAL_manual_selected_model_ready_processing_report.csv"), index=False)
    missing_df.to_csv(os.path.join(report_dir, "FINAL_manual_selected_missing_or_skipped_files.csv"), index=False)
    return report_df, missing_df


# ================================================================
# Step 2 (cell 5) — group-1/xsens-only patch
# ================================================================

def fix_group1_xsens_extra_label_columns(data_root: str):
    """Drops stray non-canonical label_p1/label_p2/label_p3 columns
    found only in group 1's xsens model_ready file (discovered via the
    sanity check — a genuine, narrow, group-1-only quirk, not a general
    rule). No-op if the file doesn't exist yet or doesn't have those
    columns."""
    path = model_ready_path(data_root, 1, "xsens")
    if not os.path.exists(path):
        print(f"skip (not built yet): {path}")
        return

    drop_cols = ["label_p1", "label_p2", "label_p3"]
    df = pd.read_csv(path, low_memory=False)
    existing = [c for c in drop_cols if c in df.columns]
    if not existing:
        print("nothing to drop")
        return

    df = df.drop(columns=existing)
    df.to_csv(path, index=False)
    print(f"dropped {existing} from {path}")


# ================================================================
# Step 3 (cell 4) — read-only sanity check
# ================================================================

def sanity_check_model_ready_files(data_root: str, report_dir: str | None = None) -> pd.DataFrame:
    """Verifies every expected (group, sensor) model_ready file exists,
    carries all 7 canonical label columns and no unexpected extras, and
    has at least one labeled row. Does not modify anything."""
    report_dir = report_dir or os.path.join(data_root, "model_ready_reports")
    os.makedirs(report_dir, exist_ok=True)

    rows = []
    for group in GROUPS:
        for sensor in SENSORS:
            path = model_ready_path(data_root, group, sensor)
            if not os.path.exists(path):
                rows.append({"group": group, "sensor": sensor, "status": "missing_file", "path": path})
                continue

            df = pd.read_csv(path, low_memory=False)
            all_label_cols = get_label_columns(df)
            missing_canonical = [c for c in CANONICAL_LABEL_COLS if c not in df.columns]
            extra_label_cols = [c for c in all_label_cols if c not in CANONICAL_LABEL_COLS]
            canonical_present = [c for c in CANONICAL_LABEL_COLS if c in df.columns]

            if canonical_present:
                labels_text = df[canonical_present].fillna("").astype(str)
                any_label = labels_text.apply(lambda row: any(str(x).strip() != "" for x in row), axis=1)
                rows_with_label, pct_with_label = int(any_label.sum()), float(any_label.mean() * 100)
            else:
                rows_with_label, pct_with_label = 0, 0.0

            status = "ok"
            if missing_canonical:
                status = "missing_canonical_label_cols"
            if extra_label_cols:
                status = "has_extra_label_cols"
            if len(df) == 0:
                status = "empty_file"
            if rows_with_label == 0:
                status = "no_labeled_rows"

            row = {
                "group": group, "sensor": sensor, "status": status, "path": path,
                "rows": len(df), "cols": len(df.columns),
                "canonical_label_cols_present_count": len(canonical_present),
                "missing_canonical_label_cols": " | ".join(missing_canonical),
                "extra_label_cols": " | ".join(extra_label_cols),
                "rows_with_any_canonical_label": rows_with_label,
                "pct_rows_with_any_canonical_label": pct_with_label,
            }
            row.update(get_time_info(df))
            rows.append(row)

    sanity_df = pd.DataFrame(rows)
    sanity_df.to_csv(os.path.join(report_dir, "FINAL_model_ready_sanity_check.csv"), index=False)
    return sanity_df


# ================================================================
# Step 4 (cell 6) — flatten into one central folder
# ================================================================

def copy_to_central_folder(data_root: str, dest_dir: str | None = None) -> pd.DataFrame:
    dest_dir = dest_dir or os.path.join(data_root, "ALL_MODEL_READY_FILES")
    os.makedirs(dest_dir, exist_ok=True)

    copied_rows, missing_rows = [], []
    for group in GROUPS:
        for sensor in SENSORS:
            src_path = model_ready_path(data_root, group, sensor)
            dst_filename = f"group_{group}_{sensor}_model_ready.csv"
            dst_path = os.path.join(dest_dir, dst_filename)

            if not os.path.exists(src_path):
                missing_rows.append({"group": group, "sensor": sensor, "source_path": src_path, "destination_path": dst_path, "status": "missing"})
                continue

            shutil.copy2(src_path, dst_path)
            copied_rows.append({"group": group, "sensor": sensor, "filename": dst_filename, "source_path": src_path, "destination_path": dst_path, "status": "copied"})

    copied_df, missing_df = pd.DataFrame(copied_rows), pd.DataFrame(missing_rows)
    copied_df.to_csv(os.path.join(dest_dir, "ALL_MODEL_READY_FILES_manifest.csv"), index=False)
    missing_df.to_csv(os.path.join(dest_dir, "ALL_MODEL_READY_FILES_missing.csv"), index=False)
    print(f"copied {len(copied_df)}, missing {len(missing_df)} -> {dest_dir}")
    return copied_df


# ================================================================
# Step 5 (cell 7) — OptiTrack participant-identity fix
# ================================================================

def detect_coordinate_prefixes(df: pd.DataFrame) -> list[str]:
    prefixes = set()
    for col in df.columns:
        if col.startswith("label_") or col.startswith("model_ready_"):
            continue
        m = re.match(r"(.+)_([xyzXYZ])$", col)
        if m and m.group(2).lower() == "x":
            prefixes.add(m.group(1))

    valid = []
    cols = set(df.columns)
    for p in sorted(prefixes):
        has_x = f"{p}_x" in cols or f"{p}_X" in cols
        has_y = f"{p}_y" in cols or f"{p}_Y" in cols
        has_z = f"{p}_z" in cols or f"{p}_Z" in cols
        if has_x and (has_y or has_z):
            valid.append(p)
    return valid


def get_x_col(df: pd.DataFrame, prefix: str):
    for c in (f"{prefix}_x", f"{prefix}_X"):
        if c in df.columns:
            return c
    return None


def infer_left_middle_right_robust(df: pd.DataFrame, prefixes: list[str], search_rows: int = 100000, valid_sample_rows: int = 5000):
    """For each prefix, finds its first valid (non-NaN) x value within
    the first search_rows rows, then averages the first
    valid_sample_rows valid x-values as that landmark's representative
    x position. Sorts the 3 prefixes ascending by mean x to assign
    physical left/middle/right. Raises ValueError if fewer than exactly
    3 prefixes end up with valid x data."""
    x_means, first_valid_indices = {}, {}
    for p in prefixes:
        x_col = get_x_col(df, p)
        if x_col is None:
            continue
        vals = pd.to_numeric(df[x_col].head(min(search_rows, len(df))), errors="coerce")
        valid_vals = vals.dropna()
        if len(valid_vals) == 0:
            continue
        first_valid_indices[p] = int(valid_vals.index[0])
        x_means[p] = float(valid_vals.head(valid_sample_rows).mean())

    if len(x_means) != 3:
        raise ValueError(f"Expected 3 OptiTrack coordinate prefixes with valid x values, found {len(x_means)}: {x_means}. First valid indices: {first_valid_indices}")

    sorted_by_x = sorted(x_means.items(), key=lambda kv: kv[1])
    position_to_prefix = {"left": sorted_by_x[0][0], "middle": sorted_by_x[1][0], "right": sorted_by_x[2][0]}
    return position_to_prefix, x_means, first_valid_indices


def build_prefix_rename_map(position_to_prefix: dict, participant_position_map: dict) -> dict:
    return {position_to_prefix[position]: participant for participant, position in participant_position_map.items()}


def rename_optitrack_feature_columns(df: pd.DataFrame, prefix_to_participant: dict):
    rename_cols = {}
    for col in df.columns:
        if col.startswith("label_"):
            continue
        for old_prefix, participant in prefix_to_participant.items():
            if col == old_prefix:
                rename_cols[col] = participant
            elif col.startswith(old_prefix + "_"):
                rename_cols[col] = f"{participant}_{col[len(old_prefix) + 1:]}"
    return df.rename(columns=rename_cols), rename_cols


def check_canonical_labels(df: pd.DataFrame):
    all_label_cols = get_label_columns(df)
    missing = [c for c in CANONICAL_LABEL_COLS if c not in df.columns]
    extra = [c for c in all_label_cols if c not in CANONICAL_LABEL_COLS]
    return all_label_cols, missing, extra


def apply_optitrack_identity_fix(data_root: str, input_dir: str | None = None, output_dir: str | None = None) -> pd.DataFrame:
    """Cell 7 — the notebook's own final cell, and the one that produces
    ALL_MODEL_READY_FILES_IDENTITY_FIXED/, the input every downstream
    src/features/ and src/models/task3_*.py module expects. Only
    OptiTrack files are transformed (column rename); openearable/xsens
    pass through via shutil.copy2. On any failure to safely detect
    exactly 3 coordinate prefixes, the file is copied unchanged and the
    manifest's `status` column records why — callers should check that
    column rather than assume renaming happened."""
    input_dir = input_dir or os.path.join(data_root, "ALL_MODEL_READY_FILES")
    output_dir = output_dir or os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED")
    os.makedirs(output_dir, exist_ok=True)

    report_rows, rename_rows = [], []

    for group in GROUPS:
        for sensor in SENSORS:
            filename = f"group_{group}_{sensor}_model_ready.csv"
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)

            if not os.path.exists(src_path):
                report_rows.append({"group": group, "sensor": sensor, "status": "missing_source", "source_path": src_path, "destination_path": dst_path})
                continue

            if sensor != "optitrack":
                shutil.copy2(src_path, dst_path)
                df_check = pd.read_csv(dst_path, nrows=5, low_memory=False)
                all_label_cols, missing_labels, extra_labels = check_canonical_labels(df_check)
                report_rows.append({
                    "group": group, "sensor": sensor, "status": "copied_unchanged", "source_path": src_path, "destination_path": dst_path,
                    "label_cols_count": len(all_label_cols), "missing_label_cols": " | ".join(missing_labels), "extra_label_cols": " | ".join(extra_labels),
                })
                continue

            df = pd.read_csv(src_path, low_memory=False)
            prefixes = detect_coordinate_prefixes(df)

            if len(prefixes) != 3:
                print(f"WARNING group {group}/{sensor}: could not detect exactly 3 coordinate prefixes ({prefixes}) — copying unchanged")
                shutil.copy2(src_path, dst_path)
                report_rows.append({
                    "group": group, "sensor": sensor, "status": "optitrack_prefix_detection_failed_copied_unchanged",
                    "source_path": src_path, "destination_path": dst_path, "detected_prefixes": " | ".join(prefixes),
                })
                continue

            try:
                position_to_prefix, x_means, first_valid_indices = infer_left_middle_right_robust(df, prefixes)
                prefix_to_participant = build_prefix_rename_map(position_to_prefix, PARTICIPANT_POSITION_MAP[group])
                df_fixed, rename_cols = rename_optitrack_feature_columns(df, prefix_to_participant)
                all_label_cols, missing_labels, extra_labels = check_canonical_labels(df_fixed)
                df_fixed.to_csv(dst_path, index=False)

                report_rows.append({
                    "group": group, "sensor": sensor, "status": "identity_fixed", "source_path": src_path, "destination_path": dst_path,
                    "detected_prefixes": " | ".join(prefixes), "first_valid_indices": str(first_valid_indices), "x_means": str(x_means),
                    "position_to_prefix": str(position_to_prefix), "prefix_to_participant": str(prefix_to_participant),
                    "renamed_columns_count": len(rename_cols), "label_cols_count": len(all_label_cols),
                    "missing_label_cols": " | ".join(missing_labels), "extra_label_cols": " | ".join(extra_labels),
                })
                for old_col, new_col in rename_cols.items():
                    rename_rows.append({"group": group, "old_column": old_col, "new_column": new_col})

            except Exception as e:
                print(f"ERROR group {group}/{sensor} during identity fixing: {e} — copying unchanged")
                shutil.copy2(src_path, dst_path)
                report_rows.append({
                    "group": group, "sensor": sensor, "status": "identity_fix_failed_copied_unchanged",
                    "source_path": src_path, "destination_path": dst_path, "detected_prefixes": " | ".join(prefixes), "error": str(e),
                })

    report_df = pd.DataFrame(report_rows)
    pd.DataFrame(rename_rows).to_csv(os.path.join(output_dir, "IDENTITY_FIXED_optitrack_column_renaming.csv"), index=False)
    report_df.to_csv(os.path.join(output_dir, "IDENTITY_FIXED_manifest.csv"), index=False)

    n_failed = int((~report_df["status"].isin(["copied_unchanged", "identity_fixed"])).sum()) if len(report_df) else 0
    print(f"identity fix: {len(report_df)} files processed, {n_failed} not cleanly fixed (see IDENTITY_FIXED_manifest.csv status column)")
    return report_df


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_root: str):
    """Runs all 5 steps in order. Returns a dict of each step's report."""
    build_report, build_missing = build_model_ready_files(data_root)
    fix_group1_xsens_extra_label_columns(data_root)
    sanity_df = sanity_check_model_ready_files(data_root)
    copy_report = copy_to_central_folder(data_root)
    identity_report = apply_optitrack_identity_fix(data_root)

    return {
        "build_report": build_report, "build_missing": build_missing,
        "sanity": sanity_df, "copy_report": copy_report, "identity_report": identity_report,
    }
