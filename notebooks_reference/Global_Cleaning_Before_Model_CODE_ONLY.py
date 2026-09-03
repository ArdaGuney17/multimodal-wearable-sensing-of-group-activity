# --- CELL 0 (code cell #1) ---
# ============================================================
# INSPECT ALL ELAN LABELS ACROSS GROUPS
# ============================================================

import os
import glob
import pandas as pd
import re

BASE = "/content/drive/MyDrive/thesis/data"
GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]

all_labels = []

def find_elan_file(group):
    candidates = glob.glob(
        f"{BASE}/group_{group}/**/*Group_{group}*individual*build*renamed*.csv",
        recursive=True
    )

    candidates = [
        p for p in candidates
        if "backup" not in os.path.basename(p).lower()
    ]

    if len(candidates) == 0:
        return None

    return sorted(candidates, key=len)[0]

for group in GROUPS:
    path = find_elan_file(group)

    print("\n" + "="*80)
    print(f"GROUP {group}")
    print("="*80)

    if path is None:
        print("No ELAN file found.")
        continue

    print("ELAN:", path)

    raw = pd.read_csv(path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "group": group,
        "tier": raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
        "label_raw": raw.iloc[:, 8].astype(str).str.strip(),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["label_raw"].notna()].copy()
    elan = elan[elan["label_raw"].str.lower() != "nan"].copy()
    elan = elan[elan["label_raw"].str.strip() != ""].copy()

    all_labels.append(elan)

all_elan_labels = pd.concat(all_labels, ignore_index=True)

# Split combined labels with +
rows = []

for _, row in all_elan_labels.iterrows():
    raw_label = row["label_raw"]

    parts = [p.strip() for p in str(raw_label).split("+")]

    for p in parts:
        if p != "":
            rows.append({
                "group": row["group"],
                "tier": row["tier"],
                "label": p,
                "full_cell": raw_label
            })

label_df = pd.DataFrame(rows)

label_counts = (
    label_df
    .groupby("label")
    .agg(
        count=("label", "size"),
        groups=("group", lambda x: ", ".join(map(str, sorted(set(x))))),
        tiers=("tier", lambda x: " | ".join(sorted(set(x))))
    )
    .reset_index()
    .sort_values(["label"])
)

print("\n" + "="*80)
print("UNIQUE ATOMIC ELAN LABELS")
print("="*80)
display(label_counts)

out_path = f"{BASE}/all_elan_unique_labels_for_cleaning.csv"
label_counts.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)


# --- CELL 1 (code cell #2) ---
# ============================================================
# GLOBAL CLEAN + TRIM ALL LABELED SENSOR FILES FOR MODEL INPUT
#
# Sensors:
# - OpenEarable
# - Xsens
# - OptiTrack
#
# For each group and sensor:
# 1. Find latest labeled file
# 2. Clean/globalize label names
# 3. Trim rows before first labeled row and after last labeled row
# 4. Save to model_ready folder
#
# Originals are NOT overwritten.
# ============================================================

import os
import re
import glob
import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

BASE = "/content/drive/MyDrive/thesis/data"

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
SENSORS = ["openearable", "xsens", "optitrack"]

REPORT_DIR = f"{BASE}/model_ready_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# GLOBAL LABEL DICTIONARY BASED ON ALL ELAN LABELS
# ============================================================

GLOBAL_LABEL_REPLACEMENTS = {
    # --------------------------------------------------------
    # Synchronization
    # --------------------------------------------------------
    "synchronaziton_move": "synchronization_move",
    "synchronaztion_move": "synchronization_move",
    "synchronizaiton_move": "synchronization_move",
    "synchronization_move": "synchronization_move",
    "synchronisation_move": "synchronization_move",
    "synch_motion": "synchronization_move",
    "sync_motion": "synchronization_move",
    "synch_move": "synchronization_move",
    "sync_move": "synchronization_move",
    "clap_synchronizaiton_move": "clap_synchronization_move",
    "clap_synchronization_move": "clap_synchronization_move",
    "droping_erarble_syncornaziton_move": "dropping_earable_synchronization_move",
    "dropping_earable_synchronization_move": "dropping_earable_synchronization_move",
    "khalil_synchornizaion_move": "participant1_synchronization_move",
    "participant1_synchronization_move": "participant1_synchronization_move",

    # --------------------------------------------------------
    # Conversation labels
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Individual building
    # --------------------------------------------------------
    "individual_bu": "individual_build",
    "individual_building": "individual_build",
    "individual_build": "individual_build",
    "start_of_individual_build": "starting_individual_build",

    # --------------------------------------------------------
    # Co-building / building
    # --------------------------------------------------------
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
    "co_building_subpiece": "co_building_subpiece",

    "co_building_pieces": "co_building_piece",
    "co_building_puzzle": "co_building_piece",

    "buılding_subpiece_together": "building_subpiece_together",
    "building_subpiece_together": "building_subpiece_together",

    # --------------------------------------------------------
    # Merging
    # --------------------------------------------------------
    "merging_sub_piece": "merging_subpiece",
    "merging_sub_pieces": "merging_subpiece",
    "merging_subpieces": "merging_subpiece",
    "merging_subpiece": "merging_subpiece",
    "mering_sub_pieces": "merging_subpiece",
    "mering_subpieces": "merging_subpiece",

    "co_merging_pieces": "co_merging_subpiece",
    "co_merging_sub_pieces": "co_merging_subpiece",
    "co_merging_subpieces": "co_merging_subpiece",

    # --------------------------------------------------------
    # Inspecting
    # --------------------------------------------------------
    ",,nspecting_pieces": "inspecting_pieces",
    "inscpecting_pieces": "inspecting_pieces",
    "inspecitng_pieces": "inspecting_pieces",
    "inspecting_pieces": "inspecting_pieces",

    "inspectingpiece": "inspecting_piece",
    "inspecitng_piece": "inspecting_piece",
    "inspecting_piece": "inspecting_piece",

    "inscpecing_other_pieces": "inspecting_other_pieces",
    "inspecitng_other_pieces": "inspecting_other_pieces",
    "inspecting_other_pieces": "inspecting_other_pieces",

    "inspecitng_other_puzlle_pieces": "inspecting_other_puzzle_pieces",

    "inspecitng_other_unit": "inspecting_other_units",
    "inspecitng_other_units": "inspecting_other_units",
    "inspecitng_other_untis": "inspecting_other_units",
    "inspecting_other_unit": "inspecting_other_units",
    "inspecting_others_unit": "inspecting_other_units",
    "inspecting_other_units": "inspecting_other_units",

    "inscpeing_target_image": "inspecting_target_image",
    "inspecitng_target_image": "inspecting_target_image",
    "inspecting_target_imge": "inspecting_target_image",
    "inspecting_target_image": "inspecting_target_image",

    "isnpecting_setup": "inspecting_setup",
    "inspecting_setup": "inspecting_setup",

    "co_inspceitng_image": "co_inspecting_image",
    "co_inspecitng_image": "co_inspecting_image",
    "co_inspecting_image": "co_inspecting_image",

    "co_inspecitng_pieces": "co_inspecting_pieces",
    "co_inspecting_pieces": "co_inspecting_pieces",

    "co_inspecting_target_image": "co_inspecting_target_image",
    "inspecting_subpiece": "inspecting_subpiece",

    # --------------------------------------------------------
    # Matching / checking
    # --------------------------------------------------------
    "matchin_pieces_with_target_image": "matching_pieces_with_target_image",
    "mathicng_pieces_to_target_image": "matching_pieces_to_target_image",
    "mathing_pieces_with_image": "matching_pieces_with_image",
    "matching_image_to_pieces": "matching_pieces_to_image",
    "piece_and_image_matching": "matching_pieces_to_target_image",

    # --------------------------------------------------------
    # Carrying
    # --------------------------------------------------------
    "carriyng_thray_to_central table": "carrying_tray_to_central_table",
    "carrying_the_tray": "carrying_tray",
    "caryying_the_tray": "carrying_tray",
    "caryying_tray": "carrying_tray",
    "carrying_tray": "carrying_tray",

    "carrying_the_tray_to_central_table": "carrying_tray_to_central_table",
    "carrying_tray_to_centraltable": "carrying_tray_to_central_table",
    "carrying_tray_to_central_table": "carrying_tray_to_central_table",

    "carying_subpiece_with_tray": "carrying_subpiece_with_tray",
    "carrying_piece_by_tray": "carrying_piece_by_tray",
    "carrying_piece_to_table": "carrying_piece_to_table",
    "carrying_object": "carrying_object",
    "carrying_target_image": "carrying_target_image",

    # --------------------------------------------------------
    # Delivering
    # --------------------------------------------------------
    "delivering_pieceBA": "delivering_piece",
    "delivering_pieceRB": "delivering_piece",
    "delivering_pieceS": "delivering_piece",
    "delivering_piece_toS": "delivering_piece",
    "delivering_piece": "delivering_piece",

    "delivering_target_imageRB": "delivering_target_image",
    "delivering_target_image": "delivering_target_image",

    "delivering_trayBR": "delivering_tray",
    "delivering_tray": "delivering_tray",

    # --------------------------------------------------------
    # Object handover
    # --------------------------------------------------------
    "object_handıver": "object_handover",
    "object_handover": "object_handover",
    "pbject_handover": "object_handover",

    # --------------------------------------------------------
    # Picking up
    # --------------------------------------------------------
    "pickign_up_pice": "picking_up_piece",
    "pickinf_up_pieces": "picking_up_pieces",
    "pickinup_piece": "picking_up_piece",
    "pickinupiece": "picking_up_piece",

    "picking_up_pieceS": "picking_up_piece",
    "picking_up_piece_fS": "picking_up_piece",
    "picking_up_piecefS": "picking_up_piece",
    "picking_up_piece": "picking_up_piece",

    "picking_up_target_i": "picking_up_target_image",
    "picking_up_target_imageS": "picking_up_target_image",
    "picking_up_target_image_fR": "picking_up_target_image",
    "picking_up_targetimage": "picking_up_target_image",
    "picking_up_traget_image": "picking_up_target_image",
    "picking_up_target_image": "picking_up_target_image",

    "pickinig_up_tray": "picking_up_tray",
    "picking_up_tray": "picking_up_tray",

    "pıcking_up_image": "picking_up_image",
    "picking_up_image": "picking_up_image",

    # --------------------------------------------------------
    # Placing / putting down
    # --------------------------------------------------------
    "placing_sub_piece_to_tray": "placing_subpiece_to_tray",
    "placing_subpieces_to_tray": "placing_subpiece_to_tray",
    "placing_pieces_to_tray": "placing_subpiece_to_tray",
    "placing_subpiece_to_tray": "placing_subpiece_to_tray",

    "placing_subpiece_to_centraltable": "placing_subpiece_to_central_table",
    "placing_subpiece_to_central_table": "placing_subpiece_to_central_table",

    "placing_subpiece_to_centraltable_from_tray": "placing_subpiece_to_central_table_from_tray",
    "placing_subpiece_to_central_table_from_tray": "placing_subpiece_to_central_table_from_tray",

    "placing_subpiece_from_tray_to_centraltable": "placing_subpiece_from_tray_to_central_table",
    "placing_subpiece_from_tray_to_table": "placing_subpiece_from_tray_to_central_table",
    "placing_subpiece_from_tray_to_central_table": "placing_subpiece_from_tray_to_central_table",

    "co_placing_subpiece_to_tray": "co_placing_subpiece_to_tray",

    "puting_down_piece": "putting_down_piece",
    "putting_doen_piece": "putting_down_piece",
    "putting_down_piece_fS": "putting_down_piece",
    "putting_down_piece": "putting_down_piece",

    "putting_doen_target_image": "putting_down_target_image",
    "putting_down_tagret_image": "putting_down_target_image",
    "putting_down_target_image": "putting_down_target_image",

    "putting_doen_tray": "putting_down_tray",
    "putting_down_tray": "putting_down_tray",

    # --------------------------------------------------------
    # Moving / traveling
    # --------------------------------------------------------
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
    "traveling_between_units": "traveling_between_units",

    "co_traveking_between_units": "co_traveling_between_units",
    "co_traveling_between_units": "co_traveling_between_units",

    "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",

    "moving_pieces_from_tray_to_central_table": "moving_pieces_from_tray_to_central_table",
    "moving_pieces_to_central_table_from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_pieces_to_central_table_with_tray": "moving_pieces_from_tray_to_central_table",
    "moving_pieces_to_table__from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_subpiece_from_tray_to_central_table": "moving_pieces_from_tray_to_central_table",
    "moving_subpiece_to_central_table_from_tray": "moving_pieces_from_tray_to_central_table",
    "moving_subpieces_from_tray_to_table": "moving_pieces_from_tray_to_central_table",

    "co_moving_pieces_from_tray_to_central_tabel": "co_moving_pieces_from_tray_to_central_table",
    "co_moving_pieces_from_tray_to_central_table": "co_moving_pieces_from_tray_to_central_table",

    "co_moving_the_pieces_to_tray": "co_moving_pieces_to_tray",
    "co_moving_subpiece_to_tray": "co_moving_pieces_to_tray",
    "co_moving_pieces_to_tray": "co_moving_pieces_to_tray",

    "moving_subpiece_to_tray": "moving_pieces_to_tray",
    "moving_subpieces_to_tray": "moving_pieces_to_tray",
    "moving_pieces_to_tray": "moving_pieces_to_tray",

    "moving_to_piece": "moving_to_piece",
    "moving_to_tray": "moving_to_tray",
    "moving_subpiece_from_tray_to_table": "moving_pieces_from_tray_to_central_table",

    # --------------------------------------------------------
    # Searching
    # --------------------------------------------------------
    "searchin_for_image": "searching_for_image",
    "searching_target_image": "searching_for_target_image",
    "search_for_target_image": "searching_for_target_image",
    "searching_for_target_image": "searching_for_target_image",
    "searching_piece": "searching_for_piece",
    "searching_for_piece": "searching_for_piece",

    # --------------------------------------------------------
    # Pointing / presenting
    # --------------------------------------------------------
    "pointing_thowards_target_image": "pointing_to_target_image",
    "pointing_to_target_image": "pointing_to_target_image",
    "pointing_to_somethign": "pointing_to_something",
    "pointing_to_something": "pointing_to_something",

    # --------------------------------------------------------
    # Guide / target image
    # --------------------------------------------------------
    "actiavely_using_guide_image": "actively_using_guide_image",
    "actively_using_guide_image": "actively_using_guide_image",

    # --------------------------------------------------------
    # Approaching
    # --------------------------------------------------------
    "approcahing_to_piece": "approaching_to_piece",
    "approaching_to_piece": "approaching_to_piece",
    "approaching_to_central_table": "approaching_to_central_table",

    # --------------------------------------------------------
    # Helping
    # --------------------------------------------------------
    "helping_task_mate": "helping_task_mate",
}


# ============================================================
# LABEL NORMALIZATION FUNCTIONS
# ============================================================

def normalize_single_label(label):
    if pd.isna(label):
        return ""

    label = str(label).strip()

    if label == "" or label.lower() in ["nan", "none", "null"]:
        return ""

    # Normalize Turkish characters used accidentally in labels
    label = label.replace("ı", "i")
    label = label.replace("İ", "I")

    # Standardize whitespace to underscores
    label = re.sub(r"\s+", "_", label)

    # Strip punctuation around label
    label = label.strip(" _;,.")

    # Collapse repeated underscores
    label = re.sub(r"_+", "_", label)

    # First dictionary pass
    label = GLOBAL_LABEL_REPLACEMENTS.get(label, label)

    # Cleanup again after replacement
    label = label.replace("ı", "i")
    label = label.replace("İ", "I")
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"_+", "_", label)
    label = label.strip(" _;,.")

    # Second dictionary pass, because cleanup may reveal a key
    label = GLOBAL_LABEL_REPLACEMENTS.get(label, label)

    return label


def normalize_label_cell(cell):
    if pd.isna(cell):
        return ""

    text = str(cell).strip()

    if text == "" or text.lower() in ["nan", "none", "null"]:
        return ""

    # Standardize separators
    text = text.replace("|", "+")
    text = text.replace("＋", "+")

    parts = [p.strip() for p in text.split("+")]

    cleaned = []

    for p in parts:
        c = normalize_single_label(p)

        # If replacement itself creates combined label, split again
        if "+" in c:
            subparts = [normalize_single_label(sp.strip()) for sp in c.split("+")]
            for sp in subparts:
                if sp != "" and sp not in cleaned:
                    cleaned.append(sp)
        else:
            if c != "" and c not in cleaned:
                cleaned.append(c)

    return " + ".join(cleaned)


# ============================================================
# FILE FINDING
# ============================================================

def path_score(path, group, sensor):
    """
    Lower score is better.
    We prefer:
    - correct labeled folders
    - exact group/sensor file names
    - not model_ready
    - not summary
    - not backup/old
    """
    p = path.lower()
    name = os.path.basename(p)

    score = 0

    # Strong exclusions are handled before scoring, but score still helps.
    if "model_ready" in p:
        score += 10000
    if "summary" in name:
        score += 10000
    if "labeling_summary" in name:
        score += 10000
    if "backup" in name:
        score += 5000
    if "old" in name:
        score += 1000

    # Prefer labeled folders/files
    if "labeled" in p or "labelled" in p:
        score -= 100

    if sensor in p:
        score -= 50

    if f"group_{group}" in p or f"group{group}" in p:
        score -= 50

    # Sensor-specific preferences
    if sensor == "optitrack":
        if f"group_{group}_optitrack_labeled.csv" in name:
            score -= 1000
        if "/optitrack/optitrack_labeled/" in p:
            score -= 500

    if sensor == "openearable":
        if "openearable" in p:
            score -= 300
        if "_oe_" in name or "oe_labeled" in name or "oe_labelled" in name:
            score -= 200
        if "acc" in name or "accelerometer" in name:
            score += 100  # raw accel files are less likely to be final labeled files

    if sensor == "xsens":
        if "xsens" in p:
            score -= 300
        if "xsens_labeled" in name or "xsens_labelled" in name:
            score -= 300

    # Prefer shorter path if score tie
    score += len(path) / 10000

    return score


def find_labeled_file(group, sensor):
    """
    Finds best labeled CSV for a group/sensor.
    This is intentionally robust because OpenEarable/Xsens filenames varied.
    """

    patterns = []

    if sensor == "optitrack":
        patterns = [
            f"{BASE}/group_{group}/optitrack/optitrack_labeled/group_{group}_optitrack_labeled.csv",
            f"{BASE}/group_{group}/**/*optitrack*labeled*.csv",
            f"{BASE}/group_{group}/**/*optitrack*labelled*.csv",
        ]

    elif sensor == "openearable":
        patterns = [
            f"{BASE}/group_{group}/openearable/**/*labeled*.csv",
            f"{BASE}/group_{group}/openearable/**/*labelled*.csv",
            f"{BASE}/group_{group}/**/*openearable*labeled*.csv",
            f"{BASE}/group_{group}/**/*openearable*labelled*.csv",
            f"{BASE}/group_{group}/**/*oe*labeled*.csv",
            f"{BASE}/group_{group}/**/*oe*labelled*.csv",
            f"{BASE}/group_{group}/**/group{group}_oe*.csv",
            f"{BASE}/group_{group}/**/group_{group}_oe*.csv",
        ]

    elif sensor == "xsens":
        patterns = [
            f"{BASE}/group_{group}/xsens/**/*labeled*.csv",
            f"{BASE}/group_{group}/xsens/**/*labelled*.csv",
            f"{BASE}/group_{group}/**/*xsens*labeled*.csv",
            f"{BASE}/group_{group}/**/*xsens*labelled*.csv",
            f"{BASE}/group_{group}/**/group{group}_xsens*.csv",
            f"{BASE}/group_{group}/**/group_{group}_xsens*.csv",
        ]

    candidates = []

    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))

    # Keep only CSVs
    candidates = [p for p in candidates if p.lower().endswith(".csv")]

    # Hard exclusions
    candidates = [
        p for p in candidates
        if "model_ready" not in p.lower()
        and "summary" not in os.path.basename(p).lower()
        and "labeling_summary" not in os.path.basename(p).lower()
        and "all_elan_unique_labels" not in os.path.basename(p).lower()
    ]

    if len(candidates) == 0:
        return None, []

    candidates_sorted = sorted(
        candidates,
        key=lambda p: path_score(p, group, sensor)
    )

    return candidates_sorted[0], candidates_sorted


# ============================================================
# DATA HELPERS
# ============================================================

def get_label_columns(df):
    return [c for c in df.columns if c.startswith("label_")]


def clean_label_columns(df):
    df = df.copy()
    label_cols = get_label_columns(df)

    before_unique = set()

    for col in label_cols:
        vals = df[col].dropna().astype(str)
        for v in vals:
            if v.strip() != "":
                for part in str(v).replace("|", "+").split("+"):
                    part = part.strip()
                    if part:
                        before_unique.add(part)

    for col in label_cols:
        df[col] = df[col].apply(normalize_label_cell)

    after_unique = set()

    for col in label_cols:
        vals = df[col].dropna().astype(str)
        for v in vals:
            if v.strip() != "":
                for part in str(v).replace("|", "+").split("+"):
                    part = part.strip()
                    if part:
                        after_unique.add(part)

    return df, before_unique, after_unique


def trim_to_labeled_range(df):
    """
    Trim dataframe from first row with any label to last row with any label.
    """
    label_cols = get_label_columns(df)

    if len(label_cols) == 0:
        return df.copy(), None, None, 0, 0

    labels_text = df[label_cols].fillna("").astype(str)

    any_label = labels_text.apply(
        lambda row: any(str(x).strip() != "" for x in row),
        axis=1
    )

    labeled_indices = np.where(any_label.to_numpy())[0]

    if len(labeled_indices) == 0:
        return df.copy(), None, None, 0, 0

    first_idx = int(labeled_indices[0])
    last_idx = int(labeled_indices[-1])

    trimmed = df.iloc[first_idx:last_idx + 1].copy()

    removed_before = first_idx
    removed_after = len(df) - last_idx - 1

    return trimmed, first_idx, last_idx, removed_before, removed_after


def add_model_ready_metadata(df, group, sensor, source_path, first_idx, last_idx):
    df = df.copy()

    df["model_ready_group"] = group
    df["model_ready_sensor"] = sensor
    df["model_ready_source_file"] = os.path.basename(source_path)

    if first_idx is None:
        df["model_ready_trim_first_original_row"] = ""
        df["model_ready_trim_last_original_row"] = ""
    else:
        df["model_ready_trim_first_original_row"] = first_idx
        df["model_ready_trim_last_original_row"] = last_idx

    return df


def get_time_info(df):
    """
    Return useful timing columns if present.
    """
    possible_time_cols = [
        "video_time_s",
        "time_s",
        "time",
        "timestamp",
        "timestamp_s",
        "elapsed_time_s",
        "seconds",
    ]

    result = {}

    for c in possible_time_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                result[f"{c}_min"] = float(vals.min())
                result[f"{c}_max"] = float(vals.max())

    return result


# ============================================================
# MAIN PROCESS
# ============================================================

all_reports = []
missing_or_skipped = []
unique_label_reports = []

print("=" * 100)
print("STARTING GLOBAL MODEL-READY CLEANING")
print("=" * 100)

for group in GROUPS:
    for sensor in SENSORS:

        print("\n" + "=" * 100)
        print(f"GROUP {group} | SENSOR: {sensor.upper()}")
        print("=" * 100)

        in_path, all_candidates = find_labeled_file(group, sensor)

        if in_path is None:
            print("No labeled file found.")

            missing_or_skipped.append({
                "group": group,
                "sensor": sensor,
                "status": "missing_labeled_file",
                "selected_input_path": "",
                "candidate_count": 0,
                "candidates": "",
            })
            continue

        print("Selected input:")
        print(in_path)

        if len(all_candidates) > 1:
            print("\nOther candidates found:")
            for c in all_candidates[1:6]:
                print(" -", c)

        try:
            df = pd.read_csv(in_path, low_memory=False)
        except Exception as e:
            print("ERROR reading file:", e)

            missing_or_skipped.append({
                "group": group,
                "sensor": sensor,
                "status": f"read_error: {e}",
                "selected_input_path": in_path,
                "candidate_count": len(all_candidates),
                "candidates": " | ".join(all_candidates),
            })
            continue

        original_rows = len(df)
        original_cols = len(df.columns)
        label_cols = get_label_columns(df)

        print("Original shape:", df.shape)
        print("Label columns:", label_cols)

        if len(label_cols) == 0:
            print("WARNING: No label columns found. Skipping.")

            missing_or_skipped.append({
                "group": group,
                "sensor": sensor,
                "status": "no_label_columns",
                "selected_input_path": in_path,
                "candidate_count": len(all_candidates),
                "candidates": " | ".join(all_candidates),
            })
            continue

        # Clean labels
        df_clean, before_unique, after_unique = clean_label_columns(df)

        # Trim to first-last labeled row
        df_trimmed, first_idx, last_idx, removed_before, removed_after = trim_to_labeled_range(df_clean)

        if first_idx is None:
            print("WARNING: File has label columns but no labeled rows. Skipping save.")

            missing_or_skipped.append({
                "group": group,
                "sensor": sensor,
                "status": "no_labeled_rows",
                "selected_input_path": in_path,
                "candidate_count": len(all_candidates),
                "candidates": " | ".join(all_candidates),
            })
            continue

        # Add metadata
        df_trimmed = add_model_ready_metadata(
            df=df_trimmed,
            group=group,
            sensor=sensor,
            source_path=in_path,
            first_idx=first_idx,
            last_idx=last_idx
        )

        # Save
        out_dir = f"{BASE}/group_{group}/{sensor}/model_ready"
        os.makedirs(out_dir, exist_ok=True)

        out_path = f"{out_dir}/group_{group}_{sensor}_model_ready.csv"

        print("Trim first original row:", first_idx)
        print("Trim last original row :", last_idx)
        print("Removed before:", removed_before)
        print("Removed after :", removed_after)
        print("Final shape:", df_trimmed.shape)
        print("Saving:")
        print(out_path)

        df_trimmed.to_csv(out_path, index=False)

        # Label coverage after trim
        label_cols_after = get_label_columns(df_trimmed)
        labels_text = df_trimmed[label_cols_after].fillna("").astype(str)

        any_label_after = labels_text.apply(
            lambda row: any(str(x).strip() != "" for x in row),
            axis=1
        )

        time_info_original = get_time_info(df)
        time_info_trimmed = get_time_info(df_trimmed)

        report_row = {
            "group": group,
            "sensor": sensor,
            "status": "saved",
            "input_path": in_path,
            "output_path": out_path,
            "original_rows": original_rows,
            "original_cols": original_cols,
            "final_rows": len(df_trimmed),
            "final_cols": len(df_trimmed.columns),
            "removed_before": removed_before,
            "removed_after": removed_after,
            "first_original_row": first_idx,
            "last_original_row": last_idx,
            "label_columns_count": len(label_cols_after),
            "unique_labels_before_cleaning": len(before_unique),
            "unique_labels_after_cleaning": len(after_unique),
            "rows_with_any_label_after_trim": int(any_label_after.sum()),
            "pct_rows_with_any_label_after_trim": float(any_label_after.mean() * 100),
        }

        for k, v in time_info_original.items():
            report_row[f"original_{k}"] = v

        for k, v in time_info_trimmed.items():
            report_row[f"trimmed_{k}"] = v

        all_reports.append(report_row)

        # Store unique labels before/after for detailed checking
        for lab in sorted(before_unique):
            unique_label_reports.append({
                "group": group,
                "sensor": sensor,
                "stage": "before_cleaning",
                "label": lab,
            })

        for lab in sorted(after_unique):
            unique_label_reports.append({
                "group": group,
                "sensor": sensor,
                "stage": "after_cleaning",
                "label": lab,
            })


# ============================================================
# SAVE REPORTS
# ============================================================

report_df = pd.DataFrame(all_reports)
missing_df = pd.DataFrame(missing_or_skipped)
unique_labels_df = pd.DataFrame(unique_label_reports)

report_path = f"{REPORT_DIR}/model_ready_processing_report.csv"
missing_path = f"{REPORT_DIR}/model_ready_missing_or_skipped_files.csv"
unique_labels_path = f"{REPORT_DIR}/model_ready_unique_labels_before_after.csv"

report_df.to_csv(report_path, index=False)
missing_df.to_csv(missing_path, index=False)
unique_labels_df.to_csv(unique_labels_path, index=False)

print("\n" + "=" * 100)
print("GLOBAL MODEL-READY CLEANING DONE")
print("=" * 100)

print("\nSaved processing report:")
print(report_path)

print("\nSaved missing/skipped report:")
print(missing_path)

print("\nSaved unique labels before/after report:")
print(unique_labels_path)

print("\nSuccessful files:")
display(report_df)

print("\nMissing/skipped files:")
display(missing_df)

print("\nUnique labels report preview:")
display(unique_labels_df.head(50))


# --- CELL 2 (code cell #3) ---
# ============================================================
# FAST LIST ALL CANDIDATE LABELED SENSOR FILES
#
# This DOES NOT process files.
# It only lists candidate paths and basic header info.
# Much faster than full CSV reading.
# ============================================================

import os
import glob
import pandas as pd

BASE = "/content/drive/MyDrive/thesis/data"

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
SENSORS = ["openearable", "xsens", "optitrack"]

REPORT_DIR = f"{BASE}/model_ready_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def get_candidate_patterns(group, sensor):
    if sensor == "openearable":
        return [
            f"{BASE}/group_{group}/openearable/**/*.csv",
            f"{BASE}/group_{group}/**/*openearable*.csv",
            f"{BASE}/group_{group}/**/*oe*.csv",
        ]

    if sensor == "xsens":
        return [
            f"{BASE}/group_{group}/xsens/**/*.csv",
            f"{BASE}/group_{group}/**/*xsens*.csv",
        ]

    if sensor == "optitrack":
        return [
            f"{BASE}/group_{group}/optitrack/**/*.csv",
            f"{BASE}/group_{group}/**/*optitrack*.csv",
        ]

    return []


def detect_flags(path):
    p = path.lower()
    name = os.path.basename(p)

    flags = []

    for key in [
        "labeled",
        "labelled",
        "shifted",
        "sync_peak",
        "last10_peak",
        "clap",
        "early_peak",
        "cleaned",
        "gap_flag",
        "merged",
        "continuous",
        "model_ready",
        "backup",
    ]:
        if key in p:
            flags.append(key)

    if "summary" in name:
        flags.append("summary")

    return ", ".join(flags)


def inspect_csv_header_only(path):
    info = {
        "read_ok": False,
        "sample_rows_read": 0,
        "cols": None,
        "label_cols_count": None,
        "label_cols": "",
        "time_cols": "",
        "sample_video_time_s_min": None,
        "sample_video_time_s_max": None,
        "sample_time_s_min": None,
        "sample_time_s_max": None,
        "has_labels": False,
        "file_size_mb": None,
        "error": "",
    }

    try:
        info["file_size_mb"] = os.path.getsize(path) / (1024 * 1024)

        df = pd.read_csv(path, nrows=5, low_memory=False)

        info["read_ok"] = True
        info["sample_rows_read"] = len(df)
        info["cols"] = len(df.columns)

        label_cols = [c for c in df.columns if c.startswith("label_")]
        info["label_cols_count"] = len(label_cols)
        info["label_cols"] = " | ".join(label_cols)
        info["has_labels"] = len(label_cols) > 0

        time_cols = []

        for c in ["video_time_s", "time_s", "timestamp", "timestamp_s", "elapsed_time_s", "time"]:
            if c in df.columns:
                time_cols.append(c)

                vals = pd.to_numeric(df[c], errors="coerce")

                if vals.notna().any():
                    info[f"sample_{c}_min"] = float(vals.min())
                    info[f"sample_{c}_max"] = float(vals.max())

        info["time_cols"] = " | ".join(time_cols)

    except Exception as e:
        info["error"] = str(e)

    return info


all_rows = []

for group in GROUPS:
    for sensor in SENSORS:

        print(f"Scanning group {group} | {sensor}...")

        patterns = get_candidate_patterns(group, sensor)
        candidates = []

        for pat in patterns:
            candidates.extend(glob.glob(pat, recursive=True))

        candidates = list(dict.fromkeys(candidates))

        candidates = [p for p in candidates if p.lower().endswith(".csv")]

        candidates = [
            p for p in candidates
            if "model_ready" not in p.lower()
            and "summary" not in os.path.basename(p).lower()
            and "labeling_summary" not in os.path.basename(p).lower()
            and "all_elan_unique_labels" not in os.path.basename(p).lower()
        ]

        for path in sorted(candidates):
            info = inspect_csv_header_only(path)

            all_rows.append({
                "group": group,
                "sensor": sensor,
                "filename": os.path.basename(path),
                "flags": detect_flags(path),
                "path": path,
                **info
            })


candidate_df = pd.DataFrame(all_rows)

preferred_cols = [
    "group",
    "sensor",
    "filename",
    "flags",
    "file_size_mb",
    "cols",
    "label_cols_count",
    "has_labels",
    "time_cols",
    "sample_video_time_s_min",
    "sample_video_time_s_max",
    "sample_time_s_min",
    "sample_time_s_max",
    "path",
    "label_cols",
    "read_ok",
    "error",
]

candidate_df = candidate_df[[c for c in preferred_cols if c in candidate_df.columns]]

candidate_df = candidate_df.sort_values(
    by=["group", "sensor", "has_labels", "flags", "filename"],
    ascending=[True, True, False, True, True]
).reset_index(drop=True)

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 300)

display(candidate_df)

out_path = f"{REPORT_DIR}/all_candidate_labeled_sensor_files_FAST.csv"
candidate_df.to_csv(out_path, index=False)

print("\nSaved fast candidate list:")
print(out_path)


# --- CELL 3 (code cell #4) ---
# ============================================================
# FINAL MODEL-READY PROCESSING WITH MANUALLY SELECTED FILES
#
# Actions:
# 1. Use manually selected OpenEarable, Xsens, OptiTrack files
# 2. Clean/globalize label names
# 3. Trim from first labeled row to last labeled row
# 4. Save final model-ready files
#
# Originals are NOT overwritten.
# ============================================================

import os
import re
import pandas as pd
import numpy as np

BASE = "/content/drive/MyDrive/thesis/data"
REPORT_DIR = f"{BASE}/model_ready_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# MANUALLY SELECTED FINAL INPUT FILES
# ============================================================

SELECTED_FILES = {
    (1, "openearable"): f"{BASE}/group_1/openearable/openearable_labeled/group_1_openearable_labeled.csv",
    (1, "xsens"):       f"{BASE}/group_1/xsens/xsens_labeled/group_1_xsens_labeled_shifted_by_last10_peak.csv",
    (1, "optitrack"):   f"{BASE}/group_1/optitrack/optitrack_labeled/group_1_optitrack_labeled.csv",

    (2, "openearable"): f"{BASE}/group_2/openearable/openearable_labeled/group_2_openearable_labeled.csv",
    (2, "xsens"):       f"{BASE}/group_2/xsens/xsens_labeled/group_2_xsens_labeled.csv",
    (2, "optitrack"):   f"{BASE}/group_2/optitrack/optitrack_labeled/group_2_optitrack_labeled.csv",

    (3, "openearable"): f"{BASE}/group_3/openearable/openearable_labeled/group_3_openearable_labeled_shifted_by_sync_peak.csv",
    (3, "xsens"):       f"{BASE}/group_3/xsens/xsens_labeled/group_3_xsens_labeled.csv",
    (3, "optitrack"):   f"{BASE}/group_3/optitrack/optitrack_labeled/group_3_optitrack_labeled.csv",

    (5, "openearable"): f"{BASE}/group_5/openearable/openearable_labeled/group_5_openearable_labeled.csv",
    (5, "xsens"):       f"{BASE}/group_5/xsens/xsens_labeled/group_5_xsens_labeled_shifted_by_clap_sync_peak.csv",
    (5, "optitrack"):   f"{BASE}/group_5/optitrack/optitrack_labeled/group_5_optitrack_labeled.csv",

    (6, "openearable"): f"{BASE}/group_6/openearable/openearable_labeled/group_6_openearable_labeled.csv",
    (6, "xsens"):       f"{BASE}/group_6/xsens/xsens_labeled/group_6_xsens_labeled_cleaned_SHIFTED.csv",
    (6, "optitrack"):   f"{BASE}/group_6/optitrack/optitrack_labeled/group_6_optitrack_labeled.csv",

    (7, "openearable"): f"{BASE}/group_7/openearable/openearable_labeled/group_7_openearable_labeled_SHIFTED.csv",
    (7, "xsens"):       f"{BASE}/group_7/xsens/xsens_labeled/group_7_xsens_labeled_SHIFTED.csv",
    (7, "optitrack"):   f"{BASE}/group_7/optitrack/optitrack_labeled/group_7_optitrack_labeled.csv",

    (8, "openearable"): f"{BASE}/group_8/openearable/openearable_labeled/group_8_openearable_labeled_SHIFTED.csv",
    (8, "xsens"):       f"{BASE}/group_8/xsens/xsens_labeled/group_8_xsens_labeled_SHIFTED.csv",
    (8, "optitrack"):   f"{BASE}/group_8/optitrack/optitrack_labeled/group_8_optitrack_labeled.csv",

    (9, "openearable"): f"{BASE}/group_9/openearable/openearable_labeled/group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv",
    (9, "xsens"):       f"{BASE}/group_9/xsens/xsens_labeled/group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv",
    (9, "optitrack"):   f"{BASE}/group_9/optitrack/optitrack_labeled/group_9_optitrack_labeled.csv",

    (10, "openearable"): f"{BASE}/group_10/openearable/openearable_labeled/group_10_openearable_labeled_SHIFTED.csv",
    (10, "xsens"):       f"{BASE}/group_10/xsens/xsens_labeled/group_10_xsens_labeled_SHIFTED.csv",
    (10, "optitrack"):   f"{BASE}/group_10/optitrack/optitrack_labeled/group_10_optitrack_labeled.csv",
}

# ============================================================
# GLOBAL LABEL CLEANING DICTIONARY
# ============================================================

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

# ============================================================
# FUNCTIONS
# ============================================================

def normalize_single_label(label):
    if pd.isna(label):
        return ""

    label = str(label).strip()

    if label == "" or label.lower() in ["nan", "none", "null"]:
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


def normalize_label_cell(cell):
    if pd.isna(cell):
        return ""

    text = str(cell).strip()

    if text == "" or text.lower() in ["nan", "none", "null"]:
        return ""

    text = text.replace("|", "+").replace("＋", "+")
    parts = [p.strip() for p in text.split("+")]

    cleaned = []

    for p in parts:
        c = normalize_single_label(p)

        if "+" in c:
            subparts = [normalize_single_label(sp.strip()) for sp in c.split("+")]
            for sp in subparts:
                if sp != "" and sp not in cleaned:
                    cleaned.append(sp)
        else:
            if c != "" and c not in cleaned:
                cleaned.append(c)

    return " + ".join(cleaned)


def get_label_columns(df):
    return [c for c in df.columns if c.startswith("label_")]


def collect_unique_labels(df, label_cols):
    unique = set()

    for col in label_cols:
        for v in df[col].dropna().astype(str):
            if v.strip() == "":
                continue
            parts = str(v).replace("|", "+").split("+")
            for p in parts:
                p = p.strip()
                if p:
                    unique.add(p)

    return unique


def clean_label_columns(df):
    df = df.copy()
    label_cols = get_label_columns(df)

    before_unique = collect_unique_labels(df, label_cols)

    for col in label_cols:
        df[col] = df[col].apply(normalize_label_cell)

    after_unique = collect_unique_labels(df, label_cols)

    return df, before_unique, after_unique


def trim_to_labeled_range(df):
    label_cols = get_label_columns(df)

    if len(label_cols) == 0:
        return df.copy(), None, None, 0, 0

    labels_text = df[label_cols].fillna("").astype(str)

    any_label = labels_text.apply(
        lambda row: any(str(x).strip() != "" for x in row),
        axis=1
    )

    labeled_indices = np.where(any_label.to_numpy())[0]

    if len(labeled_indices) == 0:
        return df.copy(), None, None, 0, 0

    first_idx = int(labeled_indices[0])
    last_idx = int(labeled_indices[-1])

    trimmed = df.iloc[first_idx:last_idx + 1].copy()

    removed_before = first_idx
    removed_after = len(df) - last_idx - 1

    return trimmed, first_idx, last_idx, removed_before, removed_after


def get_time_info(df):
    possible_cols = [
        "video_time_s",
        "time_s",
        "timestamp",
        "timestamp_s",
        "elapsed_time_s",
        "time",
    ]

    out = {}

    for c in possible_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                out[f"{c}_min"] = float(vals.min())
                out[f"{c}_max"] = float(vals.max())

    return out


def add_model_ready_metadata(df, group, sensor, source_path, first_idx, last_idx):
    df = df.copy()

    df["model_ready_group"] = group
    df["model_ready_sensor"] = sensor
    df["model_ready_source_file"] = os.path.basename(source_path)
    df["model_ready_source_path"] = source_path
    df["model_ready_trim_first_original_row"] = first_idx
    df["model_ready_trim_last_original_row"] = last_idx

    return df


# ============================================================
# MAIN PROCESS
# ============================================================

reports = []
missing_or_skipped = []
unique_label_rows = []

print("=" * 100)
print("FINAL MANUAL-SELECTION MODEL-READY PROCESSING")
print("=" * 100)

for (group, sensor), input_path in SELECTED_FILES.items():

    print("\n" + "=" * 100)
    print(f"GROUP {group} | SENSOR: {sensor.upper()}")
    print("=" * 100)

    print("Input:")
    print(input_path)

    if not os.path.exists(input_path):
        print("ERROR: file does not exist.")

        missing_or_skipped.append({
            "group": group,
            "sensor": sensor,
            "status": "missing_selected_file",
            "input_path": input_path,
        })
        continue

    df = pd.read_csv(input_path, low_memory=False)

    original_rows = len(df)
    original_cols = len(df.columns)
    label_cols = get_label_columns(df)

    print("Original shape:", df.shape)
    print("Label columns:", label_cols)

    if len(label_cols) == 0:
        print("ERROR: no label columns found. Skipping.")

        missing_or_skipped.append({
            "group": group,
            "sensor": sensor,
            "status": "no_label_columns",
            "input_path": input_path,
        })
        continue

    df_clean, before_unique, after_unique = clean_label_columns(df)

    df_trimmed, first_idx, last_idx, removed_before, removed_after = trim_to_labeled_range(df_clean)

    if first_idx is None:
        print("ERROR: no labeled rows found. Skipping.")

        missing_or_skipped.append({
            "group": group,
            "sensor": sensor,
            "status": "no_labeled_rows",
            "input_path": input_path,
        })
        continue

    df_trimmed = add_model_ready_metadata(
        df=df_trimmed,
        group=group,
        sensor=sensor,
        source_path=input_path,
        first_idx=first_idx,
        last_idx=last_idx,
    )

    out_dir = f"{BASE}/group_{group}/{sensor}/model_ready"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/group_{group}_{sensor}_model_ready.csv"

    print("Trim first original row:", first_idx)
    print("Trim last original row :", last_idx)
    print("Removed before:", removed_before)
    print("Removed after :", removed_after)
    print("Final shape:", df_trimmed.shape)
    print("Unique labels before cleaning:", len(before_unique))
    print("Unique labels after cleaning :", len(after_unique))
    print("Saving:")
    print(out_path)

    df_trimmed.to_csv(out_path, index=False)

    label_cols_after = get_label_columns(df_trimmed)
    labels_text = df_trimmed[label_cols_after].fillna("").astype(str)

    any_label_after = labels_text.apply(
        lambda row: any(str(x).strip() != "" for x in row),
        axis=1
    )

    report_row = {
        "group": group,
        "sensor": sensor,
        "status": "saved",
        "input_path": input_path,
        "output_path": out_path,
        "original_rows": original_rows,
        "original_cols": original_cols,
        "final_rows": len(df_trimmed),
        "final_cols": len(df_trimmed.columns),
        "removed_before": removed_before,
        "removed_after": removed_after,
        "first_original_row": first_idx,
        "last_original_row": last_idx,
        "label_columns_count": len(label_cols_after),
        "unique_labels_before_cleaning": len(before_unique),
        "unique_labels_after_cleaning": len(after_unique),
        "rows_with_any_label_after_trim": int(any_label_after.sum()),
        "pct_rows_with_any_label_after_trim": float(any_label_after.mean() * 100),
    }

    original_time_info = get_time_info(df)
    trimmed_time_info = get_time_info(df_trimmed)

    for k, v in original_time_info.items():
        report_row[f"original_{k}"] = v

    for k, v in trimmed_time_info.items():
        report_row[f"trimmed_{k}"] = v

    reports.append(report_row)

    for lab in sorted(before_unique):
        unique_label_rows.append({
            "group": group,
            "sensor": sensor,
            "stage": "before_cleaning",
            "label": lab,
        })

    for lab in sorted(after_unique):
        unique_label_rows.append({
            "group": group,
            "sensor": sensor,
            "stage": "after_cleaning",
            "label": lab,
        })

# ============================================================
# SAVE REPORTS
# ============================================================

report_df = pd.DataFrame(reports)
missing_df = pd.DataFrame(missing_or_skipped)
unique_labels_df = pd.DataFrame(unique_label_rows)

report_path = f"{REPORT_DIR}/FINAL_manual_selected_model_ready_processing_report.csv"
missing_path = f"{REPORT_DIR}/FINAL_manual_selected_missing_or_skipped_files.csv"
unique_labels_path = f"{REPORT_DIR}/FINAL_manual_selected_unique_labels_before_after.csv"

report_df.to_csv(report_path, index=False)
missing_df.to_csv(missing_path, index=False)
unique_labels_df.to_csv(unique_labels_path, index=False)

print("\n" + "=" * 100)
print("FINAL MODEL-READY PROCESSING DONE")
print("=" * 100)

print("\nSaved processing report:")
print(report_path)

print("\nSaved missing/skipped report:")
print(missing_path)

print("\nSaved unique labels report:")
print(unique_labels_path)

print("\nSuccessful files:")
display(report_df)

print("\nMissing/skipped files:")
display(missing_df)

print("\nUnique labels after cleaning preview:")
display(unique_labels_df[unique_labels_df["stage"] == "after_cleaning"].head(100))


# --- CELL 4 (code cell #5) ---
# ============================================================
# FINAL SANITY CHECK FOR ALL MODEL-READY FILES
#
# This cell does NOT modify files.
# It checks:
# - all expected model_ready files exist
# - canonical label columns exist
# - unexpected/extra label columns
# - row counts
# - time ranges
# - label coverage
# ============================================================

import os
import pandas as pd
import numpy as np

BASE = "/content/drive/MyDrive/thesis/data"
REPORT_DIR = f"{BASE}/model_ready_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
SENSORS = ["openearable", "xsens", "optitrack"]

CANONICAL_LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group",
]

def get_expected_path(group, sensor):
    return f"{BASE}/group_{group}/{sensor}/model_ready/group_{group}_{sensor}_model_ready.csv"


def get_time_range(df):
    time_cols = [
        "video_time_s",
        "time_s",
        "timestamp",
        "timestamp_s",
        "elapsed_time_s",
        "time",
    ]

    out = {}

    for c in time_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                out[f"{c}_min"] = float(vals.min())
                out[f"{c}_max"] = float(vals.max())
            else:
                out[f"{c}_min"] = np.nan
                out[f"{c}_max"] = np.nan

    return out


def label_coverage(df, label_cols):
    labels_text = df[label_cols].fillna("").astype(str)

    any_label = labels_text.apply(
        lambda row: any(str(x).strip() != "" for x in row),
        axis=1
    )

    return int(any_label.sum()), float(any_label.mean() * 100)


def collect_unique_labels(df, label_cols):
    unique = set()

    for col in label_cols:
        vals = df[col].fillna("").astype(str)

        for v in vals:
            if v.strip() == "":
                continue

            parts = str(v).replace("|", "+").split("+")

            for p in parts:
                p = p.strip()
                if p:
                    unique.add(p)

    return sorted(unique)


reports = []
unique_label_rows = []

print("=" * 100)
print("FINAL MODEL-READY SANITY CHECK")
print("=" * 100)

for group in GROUPS:
    for sensor in SENSORS:

        path = get_expected_path(group, sensor)

        print("\n" + "=" * 100)
        print(f"GROUP {group} | SENSOR: {sensor.upper()}")
        print("=" * 100)

        print("Path:")
        print(path)

        if not os.path.exists(path):
            print("MISSING FILE")

            reports.append({
                "group": group,
                "sensor": sensor,
                "status": "missing_file",
                "path": path,
            })
            continue

        df = pd.read_csv(path, low_memory=False)

        all_label_cols = [c for c in df.columns if c.startswith("label_")]

        missing_canonical = [c for c in CANONICAL_LABEL_COLS if c not in df.columns]
        extra_label_cols = [c for c in all_label_cols if c not in CANONICAL_LABEL_COLS]

        canonical_present = [c for c in CANONICAL_LABEL_COLS if c in df.columns]

        if len(canonical_present) > 0:
            rows_with_label, pct_with_label = label_coverage(df, canonical_present)
            unique_labels = collect_unique_labels(df, canonical_present)
        else:
            rows_with_label, pct_with_label = 0, 0.0
            unique_labels = []

        time_info = get_time_range(df)

        status = "ok"

        if len(missing_canonical) > 0:
            status = "missing_canonical_label_cols"

        if len(extra_label_cols) > 0:
            status = "has_extra_label_cols"

        if len(df) == 0:
            status = "empty_file"

        if rows_with_label == 0:
            status = "no_labeled_rows"

        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("Canonical label columns present:", len(canonical_present), "/ 7")
        print("Missing canonical label columns:", missing_canonical)
        print("Extra label columns:", extra_label_cols)
        print("Rows with any canonical label:", rows_with_label)
        print("Pct rows with any canonical label:", round(pct_with_label, 3))
        print("Unique canonical labels:", len(unique_labels))
        print("Status:", status)

        report_row = {
            "group": group,
            "sensor": sensor,
            "status": status,
            "path": path,
            "rows": len(df),
            "cols": len(df.columns),
            "all_label_cols_count": len(all_label_cols),
            "canonical_label_cols_present_count": len(canonical_present),
            "missing_canonical_label_cols": " | ".join(missing_canonical),
            "extra_label_cols": " | ".join(extra_label_cols),
            "rows_with_any_canonical_label": rows_with_label,
            "pct_rows_with_any_canonical_label": pct_with_label,
            "unique_canonical_labels_count": len(unique_labels),
        }

        for k, v in time_info.items():
            report_row[k] = v

        reports.append(report_row)

        for lab in unique_labels:
            unique_label_rows.append({
                "group": group,
                "sensor": sensor,
                "label": lab,
            })


sanity_df = pd.DataFrame(reports)
unique_df = pd.DataFrame(unique_label_rows)

sanity_path = f"{REPORT_DIR}/FINAL_model_ready_sanity_check.csv"
unique_path = f"{REPORT_DIR}/FINAL_model_ready_unique_canonical_labels.csv"

sanity_df.to_csv(sanity_path, index=False)
unique_df.to_csv(unique_path, index=False)

print("\n" + "=" * 100)
print("SANITY CHECK DONE")
print("=" * 100)

print("\nSaved sanity report:")
print(sanity_path)

print("\nSaved unique canonical labels report:")
print(unique_path)

print("\nSanity report:")
display(sanity_df)

print("\nFiles with warnings:")
warnings_df = sanity_df[sanity_df["status"] != "ok"].copy()
display(warnings_df)

print("\nUnique labels preview:")
display(unique_df.head(100))


# --- CELL 5 (code cell #6) ---
# ============================================================
# FIX GROUP 1 XSENS EXTRA LABEL COLUMNS
#
# Removes non-canonical duplicate label columns:
# - label_p1
# - label_p2
# - label_p3
#
# Keeps canonical labels:
# - label_Participant1
# - label_Participant2
# - label_Participant3
# - label_Participant1_Participant2
# - label_Participant1_Participant3
# - label_Participant2_Participant3
# - label_Whole_Group
# ============================================================

import os
import pandas as pd

BASE = "/content/drive/MyDrive/thesis/data"

path = f"{BASE}/group_1/xsens/model_ready/group_1_xsens_model_ready.csv"

DROP_COLS = ["label_p1", "label_p2", "label_p3"]

print("Loading:")
print(path)

df = pd.read_csv(path, low_memory=False)

print("Before shape:", df.shape)
print("Before label columns:")
print([c for c in df.columns if c.startswith("label_")])

existing_drop_cols = [c for c in DROP_COLS if c in df.columns]

print("Dropping:")
print(existing_drop_cols)

df = df.drop(columns=existing_drop_cols)

print("After shape:", df.shape)
print("After label columns:")
print([c for c in df.columns if c.startswith("label_")])

# overwrite only the model_ready file, not the original source
df.to_csv(path, index=False)

print("\nDONE. Cleaned Group 1 Xsens model-ready file saved:")
print(path)


# --- CELL 6 (code cell #7) ---
# ============================================================
# COPY ALL MODEL-READY FILES INTO ONE CENTRAL FOLDER
#
# Source:
# /content/drive/MyDrive/thesis/data/group_X/sensor/model_ready/...
#
# Destination:
# /content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES/
#
# This copies files. It does NOT delete or move originals.
# ============================================================

import os
import shutil
import pandas as pd

BASE = "/content/drive/MyDrive/thesis/data"

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
SENSORS = ["openearable", "xsens", "optitrack"]

DEST_DIR = f"{BASE}/ALL_MODEL_READY_FILES"
os.makedirs(DEST_DIR, exist_ok=True)

copied_rows = []
missing_rows = []

for group in GROUPS:
    for sensor in SENSORS:
        src_path = f"{BASE}/group_{group}/{sensor}/model_ready/group_{group}_{sensor}_model_ready.csv"
        dst_filename = f"group_{group}_{sensor}_model_ready.csv"
        dst_path = os.path.join(DEST_DIR, dst_filename)

        print("=" * 80)
        print(f"Group {group} | {sensor}")
        print("Source:", src_path)
        print("Destination:", dst_path)

        if not os.path.exists(src_path):
            print("MISSING SOURCE FILE")
            missing_rows.append({
                "group": group,
                "sensor": sensor,
                "source_path": src_path,
                "destination_path": dst_path,
                "status": "missing",
            })
            continue

        shutil.copy2(src_path, dst_path)

        # Quick check
        df_head = pd.read_csv(dst_path, nrows=5, low_memory=False)
        label_cols = [c for c in df_head.columns if c.startswith("label_")]

        copied_rows.append({
            "group": group,
            "sensor": sensor,
            "filename": dst_filename,
            "source_path": src_path,
            "destination_path": dst_path,
            "label_cols_count": len(label_cols),
            "status": "copied",
        })

        print("COPIED")

copied_df = pd.DataFrame(copied_rows)
missing_df = pd.DataFrame(missing_rows)

copied_report_path = f"{DEST_DIR}/ALL_MODEL_READY_FILES_manifest.csv"
missing_report_path = f"{DEST_DIR}/ALL_MODEL_READY_FILES_missing.csv"

copied_df.to_csv(copied_report_path, index=False)
missing_df.to_csv(missing_report_path, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("Central folder:")
print(DEST_DIR)

print("\nCopied files:", len(copied_df))
print("Missing files:", len(missing_df))

print("\nManifest saved:")
print(copied_report_path)

print("\nMissing report saved:")
print(missing_report_path)

display(copied_df)
display(missing_df)


# --- CELL 7 (code cell #8) ---
# ============================================================
# ROBUST FIX OPTITRACK PARTICIPANT IDENTITY
#
# Fixes previous issue:
# Some groups may have NaN/missing values for one landmark at the beginning.
# This version searches for the first valid rows where each landmark has x values.
#
# Input:
# /content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES
#
# Output:
# /content/drive/MyDrive/thesis/data/ALL_MODEL_READY_FILES_IDENTITY_FIXED
# ============================================================

import os
import re
import shutil
import pandas as pd
import numpy as np

BASE = "/content/drive/MyDrive/thesis/data"

INPUT_DIR = f"{BASE}/ALL_MODEL_READY_FILES"
OUTPUT_DIR = f"{BASE}/ALL_MODEL_READY_FILES_IDENTITY_FIXED"

os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
SENSORS = ["openearable", "xsens", "optitrack"]

PARTICIPANT_POSITION_MAP = {
    1:  {"Participant1": "left",   "Participant2": "right",  "Participant3": "middle"},
    2:  {"Participant1": "right",  "Participant2": "left",   "Participant3": "middle"},
    3:  {"Participant1": "middle", "Participant2": "right",  "Participant3": "left"},
    5:  {"Participant1": "middle", "Participant2": "right",  "Participant3": "left"},
    6:  {"Participant1": "middle", "Participant2": "left",   "Participant3": "right"},
    7:  {"Participant1": "right",  "Participant2": "left",   "Participant3": "middle"},
    8:  {"Participant1": "right",  "Participant2": "middle", "Participant3": "left"},
    9:  {"Participant1": "left",   "Participant2": "right",  "Participant3": "middle"},
    10: {"Participant1": "left",   "Participant2": "middle", "Participant3": "right"},
}

CANONICAL_LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group",
]

def detect_coordinate_prefixes(df):
    prefixes = set()

    for col in df.columns:
        if col.startswith("label_") or col.startswith("model_ready_"):
            continue

        m = re.match(r"(.+)_([xyzXYZ])$", col)
        if m:
            prefix = m.group(1)
            coord = m.group(2).lower()

            if coord == "x":
                prefixes.add(prefix)

    valid_prefixes = []

    for p in sorted(prefixes):
        cols = set(df.columns)

        has_x = f"{p}_x" in cols or f"{p}_X" in cols
        has_y = f"{p}_y" in cols or f"{p}_Y" in cols
        has_z = f"{p}_z" in cols or f"{p}_Z" in cols

        if has_x and (has_y or has_z):
            valid_prefixes.append(p)

    return valid_prefixes


def get_x_col(df, prefix):
    for c in [f"{prefix}_x", f"{prefix}_X"]:
        if c in df.columns:
            return c
    return None


def infer_left_middle_right_robust(df, prefixes, search_rows=100000, valid_sample_rows=5000):
    """
    Robust method:
    - Looks inside the first search_rows rows.
    - For each landmark, finds its first valid x values.
    - Uses the mean of the first valid_sample_rows valid x values.
    - This avoids failing when one landmark is NaN at the very beginning.
    """

    x_means = {}
    first_valid_indices = {}

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
        raise ValueError(
            f"Expected 3 OptiTrack coordinate prefixes with valid x values, "
            f"found {len(x_means)}: {x_means}. "
            f"First valid indices: {first_valid_indices}"
        )

    sorted_by_x = sorted(x_means.items(), key=lambda kv: kv[1])

    position_to_prefix = {
        "left": sorted_by_x[0][0],
        "middle": sorted_by_x[1][0],
        "right": sorted_by_x[2][0],
    }

    return position_to_prefix, x_means, first_valid_indices


def build_prefix_rename_map(position_to_prefix, participant_position_map):
    prefix_to_participant = {}

    for participant, position in participant_position_map.items():
        prefix = position_to_prefix[position]
        prefix_to_participant[prefix] = participant

    return prefix_to_participant


def rename_optitrack_feature_columns(df, prefix_to_participant):
    rename_cols = {}

    for col in df.columns:
        if col.startswith("label_"):
            continue

        for old_prefix, participant in prefix_to_participant.items():

            if col == old_prefix:
                rename_cols[col] = participant

            elif col.startswith(old_prefix + "_"):
                suffix = col[len(old_prefix) + 1:]
                rename_cols[col] = f"{participant}_{suffix}"

    df2 = df.rename(columns=rename_cols)

    return df2, rename_cols


def check_canonical_labels(df):
    all_label_cols = [c for c in df.columns if c.startswith("label_")]
    missing = [c for c in CANONICAL_LABEL_COLS if c not in df.columns]
    extra = [c for c in all_label_cols if c not in CANONICAL_LABEL_COLS]

    return all_label_cols, missing, extra


report_rows = []
rename_rows = []

print("=" * 100)
print("ROBUST IDENTITY-FIXED MODEL-READY FOLDER CREATION")
print("=" * 100)

for group in GROUPS:
    for sensor in SENSORS:

        filename = f"group_{group}_{sensor}_model_ready.csv"

        src_path = os.path.join(INPUT_DIR, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)

        print("\n" + "=" * 100)
        print(f"GROUP {group} | SENSOR: {sensor.upper()}")
        print("=" * 100)

        print("Source:")
        print(src_path)

        print("Destination:")
        print(dst_path)

        if not os.path.exists(src_path):
            print("MISSING SOURCE")

            report_rows.append({
                "group": group,
                "sensor": sensor,
                "status": "missing_source",
                "source_path": src_path,
                "destination_path": dst_path,
            })
            continue

        if sensor != "optitrack":
            shutil.copy2(src_path, dst_path)

            df_check = pd.read_csv(dst_path, nrows=5, low_memory=False)
            all_label_cols, missing_labels, extra_labels = check_canonical_labels(df_check)

            print("Copied unchanged.")

            report_rows.append({
                "group": group,
                "sensor": sensor,
                "status": "copied_unchanged",
                "source_path": src_path,
                "destination_path": dst_path,
                "label_cols_count": len(all_label_cols),
                "missing_label_cols": " | ".join(missing_labels),
                "extra_label_cols": " | ".join(extra_labels),
            })

            continue

        df = pd.read_csv(src_path, low_memory=False)

        prefixes = detect_coordinate_prefixes(df)

        print("Detected coordinate prefixes:")
        print(prefixes)

        if len(prefixes) != 3:
            print("WARNING: could not safely detect exactly 3 coordinate prefixes. File copied unchanged.")

            shutil.copy2(src_path, dst_path)

            report_rows.append({
                "group": group,
                "sensor": sensor,
                "status": "optitrack_prefix_detection_failed_copied_unchanged",
                "source_path": src_path,
                "destination_path": dst_path,
                "detected_prefixes": " | ".join(prefixes),
            })

            continue

        try:
            position_to_prefix, x_means, first_valid_indices = infer_left_middle_right_robust(
                df,
                prefixes,
                search_rows=100000,
                valid_sample_rows=5000
            )

            print("First valid x row per prefix:")
            print(first_valid_indices)

            print("Mean x from first valid samples:")
            print(x_means)

            print("Detected position -> OptiTrack prefix:")
            print(position_to_prefix)

            participant_position_map = PARTICIPANT_POSITION_MAP[group]
            prefix_to_participant = build_prefix_rename_map(position_to_prefix, participant_position_map)

            print("Final OptiTrack prefix -> participant mapping:")
            print(prefix_to_participant)

            df_fixed, rename_cols = rename_optitrack_feature_columns(df, prefix_to_participant)

            all_label_cols, missing_labels, extra_labels = check_canonical_labels(df_fixed)

            df_fixed.to_csv(dst_path, index=False)

            print("Saved identity-fixed OptiTrack file.")
            print("Renamed feature columns:", len(rename_cols))
            print("Label columns unchanged:", all_label_cols)

            report_rows.append({
                "group": group,
                "sensor": sensor,
                "status": "identity_fixed",
                "source_path": src_path,
                "destination_path": dst_path,
                "detected_prefixes": " | ".join(prefixes),
                "first_valid_indices": str(first_valid_indices),
                "x_means": str(x_means),
                "position_to_prefix": str(position_to_prefix),
                "prefix_to_participant": str(prefix_to_participant),
                "renamed_columns_count": len(rename_cols),
                "label_cols_count": len(all_label_cols),
                "missing_label_cols": " | ".join(missing_labels),
                "extra_label_cols": " | ".join(extra_labels),
            })

            for old_col, new_col in rename_cols.items():
                rename_rows.append({
                    "group": group,
                    "old_column": old_col,
                    "new_column": new_col,
                })

        except Exception as e:
            print("ERROR during identity fixing:")
            print(e)
            print("File copied unchanged for safety.")

            shutil.copy2(src_path, dst_path)

            report_rows.append({
                "group": group,
                "sensor": sensor,
                "status": "identity_fix_failed_copied_unchanged",
                "source_path": src_path,
                "destination_path": dst_path,
                "detected_prefixes": " | ".join(prefixes),
                "error": str(e),
            })


report_df = pd.DataFrame(report_rows)
rename_df = pd.DataFrame(rename_rows)

report_path = os.path.join(OUTPUT_DIR, "IDENTITY_FIXED_manifest.csv")
rename_path = os.path.join(OUTPUT_DIR, "IDENTITY_FIXED_optitrack_column_renaming.csv")

report_df.to_csv(report_path, index=False)
rename_df.to_csv(rename_path, index=False)

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print("Identity-fixed folder:")
print(OUTPUT_DIR)

print("\nManifest saved:")
print(report_path)

print("\nOptiTrack renaming report saved:")
print(rename_path)

print("\nSummary:")
display(report_df)

print("\nWarnings / failures:")
display(report_df[~report_df["status"].isin(["copied_unchanged", "identity_fixed"])])

print("\nOptiTrack column renaming preview:")
display(rename_df.head(100))

