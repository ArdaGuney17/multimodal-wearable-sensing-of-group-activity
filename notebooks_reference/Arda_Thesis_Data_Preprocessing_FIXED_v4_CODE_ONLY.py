# --- CELL 0 (code cell #1) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 2 (code cell #2) ---
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def load_oe(path, col_names):
    df = pd.read_csv(path, header=None, names=col_names)
    df = df[pd.to_numeric(df["ts"], errors="coerce").notna()].copy()
    df = df[pd.to_numeric(df[col_names[1]], errors="coerce").notna()].copy()
    df = df.astype({c: float for c in col_names if c != "flag"})
    return df.sort_values("ts").reset_index(drop=True)

def load_oe_all(base_path, p):
    acc       = load_oe(base_path + f"{p}_acc.csv",       ["ts","ax","ay","az","flag"])
    gyro      = load_oe(base_path + f"{p}_gyro.csv",      ["ts","gx","gy","gz","flag"])
    mgnt      = load_oe(base_path + f"{p}_mgnt.csv",      ["ts","mx","my","mz","flag"])
    baro      = load_oe(base_path + f"{p}_baro.csv",      ["ts","baro","flag"])
    bone_acc  = load_oe(base_path + f"{p}_bone_acc.csv",  ["ts","bx","by","bz","flag"])
    ppg       = load_oe(base_path + f"{p}_ppg.csv",       ["ts","ppg_red","ppg_ir","ppg_green","ppg_ambient","flag"])
    skin_temp = load_oe(base_path + f"{p}_skin_temp.csv", ["ts","skin_temp","flag"])
    env_temp  = load_oe(base_path + f"{p}_env_temp.csv",  ["ts","env_temp","flag"])

    merged = acc[["ts","ax","ay","az"]].copy()
    for df, cols in [
        (gyro,      ["ts","gx","gy","gz"]),
        (mgnt,      ["ts","mx","my","mz"]),
        (baro,      ["ts","baro"]),
        (bone_acc,  ["ts","bx","by","bz"]),
        (ppg,       ["ts","ppg_red","ppg_ir","ppg_green","ppg_ambient"]),
        (skin_temp, ["ts","skin_temp"]),
        (env_temp,  ["ts","env_temp"]),
    ]:
        merged = pd.merge_asof(merged, df[cols], on="ts", direction="nearest", tolerance=25000)
    merged["participant"] = p
    return merged

def load_xsens(base_path, participants, xsens_offset):
    xs_all = []
    for p in participants:
        df = pd.read_csv(base_path + f"{p}.csv", low_memory=False)
        df["participant"] = p

        df = df.drop(columns=["Unnamed: 11"], errors="ignore")
        for col in ["Acc_X","Acc_Y","Acc_Z","Gyr_X","Gyr_Y","Gyr_Z","PacketCounter"]:
            df = df[pd.to_numeric(df[col], errors="coerce").notna()]
            df[col] = df[col].astype(float)
        df = df[~((df["Acc_X"]==0) & (df["Acc_Y"]==0) & (df["Acc_Z"]==0))]

        # sort by PacketCounter — always correct order, never overflows
        df = df.sort_values("PacketCounter").reset_index(drop=True)

        # check for real gaps in PacketCounter
        pc_diff = df["PacketCounter"].diff()
        jumps = pc_diff[pc_diff > 1000]  # more than ~33s gap at 30Hz
        if len(jumps) > 0:
            print(f"WARNING: {p} Xsens has jump — trimming")
            df = df[df.index < jumps.index[0]].reset_index(drop=True)

        # compute local time from PacketCounter at 30Hz
        df["t_local"] = (df["PacketCounter"] - df["PacketCounter"].iloc[0]) / 30.0

        duration = df["t_local"].iloc[-1]
        print(f"Xsens {p}: {len(df)} rows, duration {duration:.1f}s")
        xs_all.append(df)

    xs = pd.concat(xs_all, ignore_index=True)
    xs = xs.sort_values(["participant","PacketCounter"]).reset_index(drop=True)

    for p in participants:
        mask = xs["participant"] == p
        xs.loc[mask, "t_video"] = xs.loc[mask, "t_local"] + xsens_offset

    xs = xs.sort_values(["t_video","participant"]).reset_index(drop=True)
    return xs

def _interval_mask(times, intervals):
    """Return True where each time value falls inside any [start, end] interval."""
    mask = pd.Series(False, index=times.index)
    if intervals is None or len(intervals) == 0:
        return mask
    for start, end in intervals[["t_start_s", "t_end_s"]].dropna().itertuples(index=False):
        mask |= (times >= start) & (times <= end)  # closed protection for sync boundaries
    return mask

def normalize_elan_columns(elan):
    """Make sure ELAN timing columns are numeric and text columns are clean strings."""
    elan = elan.copy()
    elan["tier"] = elan["tier"].astype(str).str.strip()
    elan["label"] = elan["label"].astype(str).str.strip()
    elan["t_start_s"] = pd.to_numeric(elan["t_start_s"], errors="coerce")
    elan["t_end_s"] = pd.to_numeric(elan["t_end_s"], errors="coerce")
    return elan

def label_df(df, time_col, elan, all_tiers, collab_cutoff=None):
    """
    Add one label column per ELAN tier.

    Fixes included:
    1. sync_move protection: individual_building is never inserted during sync intervals.
    2. half-open label intervals [start, end), plus closed sync protection at boundaries.
    3. numeric safety: ELAN start/end times are converted to numeric.
    """
    elan = normalize_elan_columns(elan)
    sync_intervals = elan.loc[elan["label"].eq("sync_move"), ["t_start_s", "t_end_s"]].copy()

    # Sort sensor times once. This avoids repeated expensive sorting inside every tier.
    t_sorted = df[[time_col]].copy().reset_index().sort_values(time_col)

    for tier in all_tiers:
        tier_rows = (
            elan[elan["tier"] == tier][["t_start_s", "t_end_s", "label"]]
            .dropna(subset=["t_start_s", "t_end_s", "label"])
            .sort_values("t_start_s")
            .reset_index(drop=True)
        )

        is_individual = "_" not in tier and tier != "Whole_Group"
        cutoff = None
        if is_individual and collab_cutoff:
            for p_key, p_cutoff in collab_cutoff.items():
                if p_key.lower() == tier.lower():
                    cutoff = p_cutoff
                    break

        if len(tier_rows) == 0:
            result = pd.Series("", index=df.index, dtype="object")
        else:
            labelled = pd.merge_asof(
                t_sorted,
                tier_rows,
                left_on=time_col,
                right_on="t_start_s",
                direction="backward"
            )

            # Vectorized replacement: keep label only when time is inside the matched interval.
            valid = labelled["t_end_s"].notna() & (labelled[time_col] < labelled["t_end_s"])  # half-open interval: [start, end)
            labelled["label"] = np.where(valid, labelled["label"], "")
            result = labelled.set_index("index")["label"].reindex(df.index)
            result = result.fillna("").astype(str)

        if is_individual and cutoff is not None:
            # Use closed sync protection so boundary samples are not auto-filled as individual_building.
            sync_mask = _interval_mask(df[time_col], sync_intervals)

            # Extra safety: if individual_building came from pre-cleaned ELAN rows, remove it during sync.
            result.loc[sync_mask & (result == "individual_building")] = ""

            mask = (result == "") & (df[time_col] < cutoff) & (~sync_mask)
            result.loc[mask] = "individual_building"

        df[tier] = result.fillna("").replace("nan", "")
        print(f"  {tier}: {(df[tier] != '').sum()} labelled rows")

    return df

def validate_label_output(df, all_tiers, name="dataset"):
    """Small sanity check after labelling."""
    print(f"\nValidation for {name}: {df.shape}")
    missing = [c for c in all_tiers if c not in df.columns]
    print("Missing tier columns:", missing if missing else "none")
    for tier in all_tiers:
        if tier in df.columns:
            print(f"{tier}: {(df[tier] != '').sum()} labelled rows")


# --- CELL 4 (code cell #3) ---
xs = pd.read_csv("/content/drive/MyDrive/thesis/data/group_1/group1_xsens_labelled.csv", low_memory=False)
print(f"Rachel rows: {len(xs[xs['participant']=='rachel'])}")
print(f"Rachel t_video: {xs[xs['participant']=='rachel']['t_video'].min():.1f} to {xs[xs['participant']=='rachel']['t_video'].max():.1f}s")


# --- CELL 5 (code cell #4) ---
import pandas as pd

elan_raw = pd.read_csv("/content/drive/MyDrive/thesis/data/group_1/elan/Group_1.csv", header=None,
                       names=["tier","_","t_start_hms","t_start_s",
                              "t_end_hms","t_end_s","duration_hms","duration_s","label"])

tier_map = {
    "Arda": "Arda", "Bas": "Bas", "Rachel": "Rachel",
    "Bas_Arda": "Bas_Arda", "Arda_Rachel": "Arda_Rachel",
    "Bas_Rachel": "Bas_Rachel", "Whole_Group": "Whole_Group",
}
elan_raw["tier"] = elan_raw["tier"].str.strip()

typo_map = {
    "travel_between_units": "traveling_between_units",
    "traveling_betwen_units": "traveling_between_units",
    "trveling_between_units": "traveling_between_units",
    "moving_between_units": "traveling_between_units",
    "inspecting_other_puzzle_pieces": "inspecting_other_pieces",
    "inspecitng_other_puzlle_pieces": "inspecting_other_pieces",
    "inspecting_puzzle_pieces": "inspecting_other_pieces",
    "inspecitng_pieces": "inspecting_pieces",
    "merging_subpieces": "merging_subpiece",
    "merging_sub_pieces": "merging_subpiece",
    "carying_subpiece_with_tray": "carrying_subpiece_with_tray",
    "putting_doen_tray": "putting_down_tray",
    "task_operatinal_convo": "task_operational_convo",
    "task_operation_convo": "task_operational_convo",
    "task_realted_convo": "task_operational_convo",
    "task_related_dialouge": "task_operational_convo",
    "task_related_social_convo": "task_social_convo",
    "picking_up_pieces": "picking_up_piece",
    "buılding_subpiece_together": "building_subpiece_together",
    "searching_for_piece": "searching_piece",
    "delivering_pieceBA": "delivering_piece",
    "delivering_pieceRB": "delivering_piece",
    "delivering_trayBR": "delivering_piece",
    "delivering_target_imageRB": "delivering_piece",
    "picking_up_target_image_fR": "lifting_target_image",
    "returning_target_image_tR": "returning_target_image",
    "synchronaziton_move": "sync_move",
    "synchronizaiton_move": "sync_move",
}
elan_raw["label"] = elan_raw["label"].str.strip().replace(typo_map)

compound_map = {
    "inspecting_other_pieces +traveling_between units": ["inspecting_other_pieces","traveling_between_units"],
    "inspecting_target_image + task_operational_convo": ["inspecting_target_image","task_operational_convo"],
    "object_handover + task_operational_convo": ["object_handover","task_operational_convo"],
    "traveling_between_units + inspecting_target_image": ["traveling_between_units","inspecting_target_image"],
}
expanded = []
for _, row in elan_raw.iterrows():
    label = str(row["label"]).strip()
    if label in compound_map:
        for l in compound_map[label]:
            new_row = row.copy(); new_row["label"] = l; expanded.append(new_row)
    else:
        expanded.append(row)
elan_raw = pd.DataFrame(expanded).reset_index(drop=True)

collab_cutoff = {"Arda": 1420.0, "Bas": 1160.0, "Rachel": 1360.0}
sync_intervals = elan_raw[elan_raw["label"] == "sync_move"][["t_start_s","t_end_s"]]

new_rows = []
for participant, cutoff in collab_cutoff.items():
    p_elan = elan_raw[elan_raw["tier"] == participant].sort_values("t_start_s")
    covered = []
    for _, row in p_elan.iterrows():
        start = max(0, row["t_start_s"]); end = min(cutoff, row["t_end_s"])
        if start < end: covered.append([start, end])
    for _, row in sync_intervals.iterrows():
        start = max(0, row["t_start_s"]); end = row["t_end_s"]
        if start < end: covered.append([start, end])
    merged_iv = []
    for start, end in sorted(covered):
        if merged_iv and start <= merged_iv[-1][1]: merged_iv[-1][1] = max(merged_iv[-1][1], end)
        else: merged_iv.append([start, end])
    cursor = 0.0
    for start, end in merged_iv:
        if start > cursor:
            new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": start, "label": "individual_building"})
        cursor = max(cursor, end)
    if cursor < cutoff:
        new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": cutoff, "label": "individual_building"})

elan = pd.concat([elan_raw, pd.DataFrame(new_rows)], ignore_index=True).sort_values(["tier","t_start_s"]).reset_index(drop=True)
elan.to_csv("/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_clean.csv", index=False)
print(f"Group 1 ELAN: {len(elan)} rows, {elan['label'].nunique()} labels")
print("sync_move rows:", (elan['label']=='sync_move').sum())


# --- CELL 6 (code cell #5) ---
BASE            = "/content/drive/MyDrive/thesis/data/group_1/"
ELAN_FILE       = BASE + "elan/Group_1_clean.csv"
PARTICIPANTS    = ["arda", "bas", "rachel"]
VIDEO_START_UTC = datetime(2026, 4, 21, 8, 14, 47, tzinfo=timezone.utc).timestamp()
XSENS_START_UTC = datetime(2026, 4, 21, 8, 14, 22, tzinfo=timezone.utc).timestamp()
XSENS_OFFSET    = XSENS_START_UTC - VIDEO_START_UTC

ALL_TIERS = ["Arda", "Bas", "Rachel",
             "Bas_Arda", "Arda_Rachel", "Bas_Rachel", "Whole_Group"]

collab_cutoff = {"arda": 1420.0, "bas": 1160.0, "rachel": 1360.0}

elan = pd.read_csv(ELAN_FILE)

# OE
oe_all = []
for p in PARTICIPANTS:
    merged = load_oe_all(BASE + f"openearable/{p}/", p)
    print(f"OE {p}: {len(merged)} rows")
    oe_all.append(merged)
oe = pd.concat(oe_all, ignore_index=True).sort_values(["ts","participant"]).reset_index(drop=True)
oe["t_video"] = (oe["ts"] / 1e6) - VIDEO_START_UTC

# Xsens
xs = load_xsens(BASE + "xsens/", PARTICIPANTS, XSENS_OFFSET)

print(f"\nOE t_video: {oe['t_video'].min():.1f} to {oe['t_video'].max():.1f}s")
print(f"Xsens t_video: {xs['t_video'].min():.1f} to {xs['t_video'].max():.1f}s")

# label
print("\nLabelling OE:")
oe = label_df(oe, "t_video", elan, ALL_TIERS, collab_cutoff)
print("\nLabelling Xsens:")
xs = label_df(xs, "t_video", elan, ALL_TIERS, collab_cutoff)

# clean NaN → empty string in tier columns
for tier in ALL_TIERS:
    oe[tier] = oe[tier].fillna("").replace("nan", "")
    xs[tier] = xs[tier].fillna("").replace("nan", "")

oe.to_csv(BASE + "group1_oe_labelled.csv", index=False, na_rep="")
xs.to_csv(BASE + "group1_xsens_labelled.csv", index=False, na_rep="")
print(f"\nSaved — OE: {oe.shape}, Xsens: {xs.shape}")


# --- CELL 8 (code cell #6) ---
elan_raw = pd.read_csv("/content/drive/MyDrive/thesis/data/group_2/elan/Group_2.csv", header=None,
                       names=["tier","_","t_start_hms","t_start_s",
                              "t_end_hms","t_end_s","duration_hms","duration_s","label"])

tier_map = {
    "Khalil": "Participant1", "Arda": "Participant2", "Shizin": "Participant3",

    # Pair tiers — include both underscore and space variants, both name orders.
    "Arda_Khalil": "Participant1_Participant2",
    "Khalil_Arda": "Participant1_Participant2",
    "Arda Khalil": "Participant1_Participant2",
    "Khalil Arda": "Participant1_Participant2",

    "Arda_Shizin": "Participant2_Participant3",
    "Shizin_Arda": "Participant2_Participant3",
    "Arda Shizin": "Participant2_Participant3",
    "Shizin Arda": "Participant2_Participant3",

    "Shizin_Khalil": "Participant1_Participant3",
    "Khalil_Shizin": "Participant1_Participant3",
    "Shizin Khalil": "Participant1_Participant3",
    "Khalil Shizin": "Participant1_Participant3",

    "Whole_Group": "Whole_Group",
    "Whole Group": "Whole_Group",
    "whole group": "Whole_Group",
}
elan_raw["tier"] = elan_raw["tier"].str.strip().replace(tier_map)

typo_map = {
    "travel_between_units": "traveling_between_units",
    "traveling_betwen_units": "traveling_between_units",
    "travelin_between_units": "traveling_between_units",
    "putting_doen_piece": "putting_down_piece",
    "puting_down_piece": "putting_down_piece",
    "putting_down_piece_fS": "putting_down_piece",
    "putting_down_object": "putting_down_piece",
    "inspecitng_other_pieces": "inspecting_other_pieces",
    "inspecitng_target_image": "inspecting_target_image",
    "inspecting_piece": "inspecting_pieces",
    "co_building_subpiece": "co_building_subpiece",
    "co_building_sub_piece": "co_building_subpiece",
    "co_building_subpart": "co_building_subpiece",
    "co_building_sub_part": "co_building_subpiece",
    "co_building_the_subpiece": "co_building_subpiece",
    "co_builidng_subpiece": "co_building_subpiece",
    "mering_sub_pieces": "merging_subpiece",
    "mering_subpieces": "merging_subpiece",
    "merging_subpieces": "merging_subpiece",
    "carrying_tray_to_centraltable": "carrying_tray_to_table",
    "carrying_tray_to_central_table": "carrying_tray_to_table",
    "placing_sub_piece_to_tray": "placing_subpiece_to_tray",
    "placing_subpieces_to_tray": "placing_subpiece_to_tray",
    "placing_subpiece_to_centraltable": "placing_subpiece_to_tray",
    "placing_subpiece_to_centraltable_from_tray": "placing_subpiece_to_tray",
    "placing_subpiece_from_tray_to_centraltable": "placing_subpiece_to_tray",
    "co_placing_subpiece_to_tray": "placing_subpiece_to_tray",
    "task_operational_talk": "task_operational_convo",
    "matching_pieces_with_image": "matching_pieces_to_target_image",
    "pointing_thowards_target_image": "presenting_target_image",
    "picking_up_pieceS": "picking_up_piece",
    "picking_up_piecefS": "picking_up_piece",
    "picking_up_piece_fS": "picking_up_piece",
    "picking_up_target_imageS": "picking_up_target_image",
    "delivering_pieceS": "delivering_piece",
    "delivering_piece_toS": "delivering_piece",
    "start_of_individual_build": "sync_move",
    "khalil_synchornizaion_move": "sync_move",
    "droping_erarble_syncornaziton_move": "sync_move",
    "synchronaziton_move": "sync_move",
    "synchronizaiton_move": "sync_move",
}
elan_raw["label"] = elan_raw["label"].str.strip().replace(typo_map)

compound_map = {
    "picking_up_piece + traveling_between_units": ["picking_up_piece","traveling_between_units"],
    "putting_down_piece + traveling_between_units": ["putting_down_piece","traveling_between_units"],
    "inspecting_target_image + inspecting_pieces": ["inspecting_target_image","inspecting_pieces"],
    "inspecting_pieces + inspecting_target_image": ["inspecting_pieces","inspecting_target_image"],
    "inspecting_target_image + matching_pieces_to_target_image": ["inspecting_target_image","matching_pieces_to_target_image"],
    "traveling_between_units + inspecting_other_pieces": ["traveling_between_units","inspecting_other_pieces"],
    "inspecitng_pieces + task_operational_convo": ["inspecting_pieces","task_operational_convo"],
}
expanded = []
for _, row in elan_raw.iterrows():
    label = str(row["label"]).strip()
    if label in compound_map:
        for l in compound_map[label]:
            new_row = row.copy(); new_row["label"] = l; expanded.append(new_row)
    else:
        expanded.append(row)
elan_raw = pd.DataFrame(expanded).reset_index(drop=True)

collab_cutoff = {"Participant1": 1742.0, "Participant2": 1742.0, "Participant3": 1742.0}
sync_intervals = elan_raw[elan_raw["label"] == "sync_move"][["t_start_s","t_end_s"]]

new_rows = []
for participant, cutoff in collab_cutoff.items():
    p_elan = elan_raw[elan_raw["tier"] == participant].sort_values("t_start_s")
    covered = []
    for _, row in p_elan.iterrows():
        start = max(0, row["t_start_s"]); end = min(cutoff, row["t_end_s"])
        if start < end: covered.append([start, end])
    for _, row in sync_intervals.iterrows():
        start = max(0, row["t_start_s"]); end = row["t_end_s"]
        if start < end: covered.append([start, end])
    merged_iv = []
    for start, end in sorted(covered):
        if merged_iv and start <= merged_iv[-1][1]: merged_iv[-1][1] = max(merged_iv[-1][1], end)
        else: merged_iv.append([start, end])
    cursor = 0.0
    for start, end in merged_iv:
        if start > cursor:
            new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": start, "label": "individual_building"})
        cursor = max(cursor, end)
    if cursor < cutoff:
        new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": cutoff, "label": "individual_building"})

elan = pd.concat([elan_raw, pd.DataFrame(new_rows)], ignore_index=True).sort_values(["tier","t_start_s"]).reset_index(drop=True)
elan.to_csv("/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_clean.csv", index=False)
print(f"Group 2 ELAN: {len(elan)} rows, {elan['label'].nunique()} labels")
print("sync_move rows:", (elan['label']=='sync_move').sum())


# --- CELL 9 (code cell #7) ---
BASE            = "/content/drive/MyDrive/thesis/data/group_2/"
ELAN_FILE       = BASE + "elan/Group_2_clean.csv"
PARTICIPANTS    = ["Participant1", "Participant2", "Participant3"]
VIDEO_START_UTC = datetime(2026, 4, 21, 12, 4, 25, tzinfo=timezone.utc).timestamp()
XSENS_START_UTC = datetime(2026, 4, 21, 12, 7, 51, tzinfo=timezone.utc).timestamp()
XSENS_OFFSET    = XSENS_START_UTC - VIDEO_START_UTC

ALL_TIERS = ["Participant1", "Participant2", "Participant3",
             "Participant1_Participant2", "Participant1_Participant3",
             "Participant2_Participant3", "Whole_Group"]

collab_cutoff = {"Participant1": 1742.0, "Participant2": 1742.0, "Participant3": 1742.0}

elan = pd.read_csv(ELAN_FILE)

# Robust protection: if Group_2_clean.csv still contains name-based tiers,
# map them again here so pair/whole-group labels do not become zero.
GROUP2_TIER_MAP = {
    "Khalil": "Participant1", "Arda": "Participant2", "Shizin": "Participant3",
    "Arda_Khalil": "Participant1_Participant2",
    "Khalil_Arda": "Participant1_Participant2",
    "Arda Khalil": "Participant1_Participant2",
    "Khalil Arda": "Participant1_Participant2",
    "Arda_Shizin": "Participant2_Participant3",
    "Shizin_Arda": "Participant2_Participant3",
    "Arda Shizin": "Participant2_Participant3",
    "Shizin Arda": "Participant2_Participant3",
    "Shizin_Khalil": "Participant1_Participant3",
    "Khalil_Shizin": "Participant1_Participant3",
    "Shizin Khalil": "Participant1_Participant3",
    "Khalil Shizin": "Participant1_Participant3",
    "Whole_Group": "Whole_Group",
    "Whole Group": "Whole_Group",
    "whole group": "Whole_Group",
}
elan = normalize_elan_columns(elan)
elan["tier"] = elan["tier"].replace(GROUP2_TIER_MAP)

missing_tiers = sorted(set(ALL_TIERS) - set(elan["tier"].unique()))
if missing_tiers:
    print("WARNING: These tiers are not present in Group 2 ELAN after mapping:", missing_tiers)


oe_all = []
for p in PARTICIPANTS:
    merged = load_oe_all(BASE + f"openearable/{p}/", p)
    print(f"OE {p}: {len(merged)} rows")
    oe_all.append(merged)
oe = pd.concat(oe_all, ignore_index=True).sort_values(["ts","participant"]).reset_index(drop=True)
oe["t_video"] = (oe["ts"] / 1e6) - VIDEO_START_UTC

xs = load_xsens(BASE + "xsens/", PARTICIPANTS, XSENS_OFFSET)

print(f"\nOE t_video: {oe['t_video'].min():.1f} to {oe['t_video'].max():.1f}s")
print(f"Xsens t_video: {xs['t_video'].min():.1f} to {xs['t_video'].max():.1f}s")

print("\nLabelling OE:")
oe = label_df(oe, "t_video", elan, ALL_TIERS, collab_cutoff)
print("\nLabelling Xsens:")
xs = label_df(xs, "t_video", elan, ALL_TIERS, collab_cutoff)

# clean NaN → empty string in tier columns
for tier in ALL_TIERS:
    oe[tier] = oe[tier].fillna("").replace("nan", "")
    xs[tier] = xs[tier].fillna("").replace("nan", "")

oe.to_csv(BASE + "group2_oe_labelled.csv", index=False, na_rep="")
xs.to_csv(BASE + "group2_xsens_labelled.csv", index=False, na_rep="")
print(f"\nSaved — OE: {oe.shape}, Xsens: {xs.shape}")
validate_label_output(oe, ALL_TIERS, "Group 2 OE")
validate_label_output(xs, ALL_TIERS, "Group 2 Xsens")


# --- CELL 11 (code cell #8) ---
import pandas as pd
import numpy as np

p1 = pd.read_csv(
    "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_Part1.csv",
    header=None,
    names=[
        "tier", "_", "t_start_hms", "t_start_s",
        "t_end_hms", "t_end_s", "duration_hms", "duration_s", "label"
    ]
)

p2 = pd.read_csv(
    "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_Part2.csv",
    header=None,
    names=[
        "tier", "_", "t_start_hms", "t_start_s",
        "t_end_hms", "t_end_s", "duration_hms", "duration_s", "label"
    ]
)

# Group 3 Part 2 starts later in the full experiment timeline
PART2_OFFSET = 1506.0
p2["t_start_s"] = pd.to_numeric(p2["t_start_s"], errors="coerce") + PART2_OFFSET
p2["t_end_s"]   = pd.to_numeric(p2["t_end_s"], errors="coerce") + PART2_OFFSET

elan_raw = pd.concat([p1, p2], ignore_index=True)

# Clean basic columns
elan_raw["tier"] = elan_raw["tier"].astype(str).str.strip()
elan_raw["label"] = elan_raw["label"].astype(str).str.strip()
elan_raw["t_start_s"] = pd.to_numeric(elan_raw["t_start_s"], errors="coerce")
elan_raw["t_end_s"] = pd.to_numeric(elan_raw["t_end_s"], errors="coerce")

# Tier mapping
tier_map = {
    "chinyu": "Participant1",
    "long": "Participant2",
    "egemen": "Participant3",

    # Chinyu–Long had no interaction in Group 3, but keep these mappings if a row exists
    "chinyu long": "Participant1_Participant2",
    "long chinyu": "Participant1_Participant2",

    "egemen chinyu": "Participant1_Participant3",
    "chinyu egemen": "Participant1_Participant3",

    "egemen long": "Participant2_Participant3",
    "long egemen": "Participant2_Participant3",

    "whole group": "Whole_Group",
    "Whole Group": "Whole_Group",
    "Whole_Group": "Whole_Group",
}

elan_raw["tier"] = elan_raw["tier"].replace(tier_map)

# Label typo mapping
# IMPORTANT:
# inspecting_setup should stay inspecting_setup.
# It should NOT be converted into inspecting_target_image.
typo_map = {
    "traveling_betwwen_units": "traveling_between_units",
    "traveling_between_unit": "traveling_between_units",

    "inspecting_piece": "inspecting_pieces",
    "inspecitng_pieces": "inspecting_pieces",

    "picking_up_targetimage": "picking_up_target_image",

    "approaching_to_central_table": "traveling_between_units",
    "approaching_to_single_units": "traveling_between_units",

    # Correct setup labels
    "inspecting_setup": "inspecting_setup",
    "isnpecting_setup": "inspecting_setup",

    "co_building_subpiece": "co_building_subpiece",
    "co_building_sub_piece": "co_building_subpiece",

    "merging_sub_piece": "merging_subpiece",
    "mering_subpieces": "merging_subpiece",

    "moving_subpiece_from_tray_to_central_table": "moving_pieces_to_central_table_from_tray",
    "moving_subpiece_to_central_table_from_tray": "moving_pieces_to_central_table_from_tray",
    "moving_subpiece_to_tray": "placing_subpiece_to_tray",
    "moving_tray_to_central_table": "carrying_tray_to_table",

    "mathicng_pieces_to_target_image": "matching_pieces_to_target_image",

    "task_operational_talk": "task_operational_convo",
    "task_operatşonal_convo": "task_operational_convo",

    "pointing_to_piece": "presenting_piece",

    "synchronaziton_move": "sync_move",
    "synchronizaiton_move": "sync_move",
    "synchronaztion_move": "sync_move",
    "synchronaztion_move": "sync_move",
}

elan_raw["label"] = elan_raw["label"].replace(typo_map)

# Compound labels
# IMPORTANT:
# "task_operational_convo + isnpecting_setup" should become:
# task_operational_convo + inspecting_setup
# NOT inspecting_target_image.
compound_map = {
    "approaching_to_central_table + task_operational_convo": [
        "traveling_between_units",
        "task_operational_convo"
    ],

    "task_operational_convo + isnpecting_setup": [
        "task_operational_convo",
        "inspecting_setup"
    ],

    "task_operational_convo + inspecting_setup": [
        "task_operational_convo",
        "inspecting_setup"
    ],

    "picking_up_target_image + traveling_between_units": [
        "picking_up_target_image",
        "traveling_between_units"
    ],

    "object_handover + task_operational_talk": [
        "object_handover",
        "task_operational_convo"
    ],

    "object_handover + task_operational_convo": [
        "object_handover",
        "task_operational_convo"
    ],

    "putting_down_target_image + traveling_between_units": [
        "putting_down_target_image",
        "traveling_between_units"
    ],

    "putting_down_target_image + traveling_betwwen_units": [
        "putting_down_target_image",
        "traveling_between_units"
    ],

    "traveling_between_units + putting_down_piece": [
        "traveling_between_units",
        "putting_down_piece"
    ],
}

expanded = []

for _, row in elan_raw.iterrows():
    label = str(row["label"]).strip()

    if label in compound_map:
        for new_label in compound_map[label]:
            new_row = row.copy()
            new_row["label"] = new_label
            expanded.append(new_row)
    else:
        expanded.append(row)

elan_raw = pd.DataFrame(expanded).reset_index(drop=True)

# Individual-building gap filling
collab_cutoff = {
    "Participant1": 2495.0,
    "Participant2": 2495.0,
    "Participant3": 2065.0,
}

sync_intervals = elan_raw[
    elan_raw["label"] == "sync_move"
][["t_start_s", "t_end_s"]]

new_rows = []

for participant, cutoff in collab_cutoff.items():
    p_elan = elan_raw[elan_raw["tier"] == participant].sort_values("t_start_s")

    covered = []

    for _, row in p_elan.iterrows():
        start = max(0, row["t_start_s"])
        end = min(cutoff, row["t_end_s"])

        if start < end:
            covered.append([start, end])

    # Also protect sync_move intervals from being filled as individual_building
    for _, row in sync_intervals.iterrows():
        start = max(0, row["t_start_s"])
        end = row["t_end_s"]

        if start < end:
            covered.append([start, end])

    merged_iv = []

    for start, end in sorted(covered):
        if merged_iv and start <= merged_iv[-1][1]:
            merged_iv[-1][1] = max(merged_iv[-1][1], end)
        else:
            merged_iv.append([start, end])

    cursor = 0.0

    for start, end in merged_iv:
        if start > cursor:
            new_rows.append({
                "tier": participant,
                "t_start_s": cursor,
                "t_end_s": start,
                "label": "individual_building"
            })

        cursor = max(cursor, end)

    if cursor < cutoff:
        new_rows.append({
            "tier": participant,
            "t_start_s": cursor,
            "t_end_s": cutoff,
            "label": "individual_building"
        })

elan = (
    pd.concat([elan_raw, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "t_start_s"])
    .reset_index(drop=True)
)

elan.to_csv(
    "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_clean.csv",
    index=False
)

print(f"Group 3 ELAN: {len(elan)} rows, {elan['label'].nunique()} labels")
print("sync_move rows:", (elan["label"] == "sync_move").sum())

print("\nWhole_Group labels around setup part:")
print(
    elan[
        (elan["tier"] == "Whole_Group") &
        (elan["t_start_s"] < 300)
    ][["tier", "t_start_s", "t_end_s", "label"]].to_string(index=False)
)


# --- CELL 12 (code cell #9) ---
import pandas as pd

for part, path in [
    ("Part 1", "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_Part1.csv"),
    ("Part 2", "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_Part2.csv"),
]:
    df = pd.read_csv(path, header=None,
                     names=["tier","_","t_start_hms","t_start_s",
                            "t_end_hms","t_end_s","duration_hms","duration_s","label"])
    print(f"=== {part} ===")
    print(f"Rows: {len(df)}")
    print("All labels:")
    for l in sorted(df["label"].dropna().str.strip().unique().tolist()):
        print(f"  {l}")
    print()


# --- CELL 13 (code cell #10) ---
BASE            = "/content/drive/MyDrive/thesis/data/group_3/"
ELAN_FILE       = BASE + "elan/Group_3_clean.csv"
PARTICIPANTS    = ["Participant1", "Participant2", "Participant3"]
VIDEO_START_UTC = datetime(2026, 4, 22, 12, 21, 50, tzinfo=timezone.utc).timestamp()
XSENS_START_UTC = datetime(2026, 4, 22, 12, 23, 31, tzinfo=timezone.utc).timestamp()
XSENS_OFFSET    = XSENS_START_UTC - VIDEO_START_UTC

ALL_TIERS = ["Participant1", "Participant2", "Participant3",
             "Participant1_Participant2", "Participant1_Participant3",
             "Participant2_Participant3", "Whole_Group"]

collab_cutoff = {"Participant1": 2495.0, "Participant2": 2495.0, "Participant3": 2065.0}

elan = pd.read_csv(ELAN_FILE)

oe_all = []
for p in PARTICIPANTS:
    merged = load_oe_all(BASE + f"openearable/{p}/", p)
    print(f"OE {p}: {len(merged)} rows")
    oe_all.append(merged)
oe = pd.concat(oe_all, ignore_index=True).sort_values(["ts","participant"]).reset_index(drop=True)
oe["t_video"] = (oe["ts"] / 1e6) - VIDEO_START_UTC

xs = load_xsens(BASE + "xsens/", PARTICIPANTS, XSENS_OFFSET)

print(f"\nOE t_video: {oe['t_video'].min():.1f} to {oe['t_video'].max():.1f}s")
print(f"Xsens t_video: {xs['t_video'].min():.1f} to {xs['t_video'].max():.1f}s")

print("\nLabelling OE:")
oe = label_df(oe, "t_video", elan, ALL_TIERS, collab_cutoff)
print("\nLabelling Xsens:")
xs = label_df(xs, "t_video", elan, ALL_TIERS, collab_cutoff)

# clean NaN → empty string in tier columns
for tier in ALL_TIERS:
    oe[tier] = oe[tier].fillna("").replace("nan", "")
    xs[tier] = xs[tier].fillna("").replace("nan", "")

oe.to_csv(BASE + "group3_oe_labelled.csv", index=False, na_rep="")
xs.to_csv(BASE + "group3_xsens_labelled.csv", index=False, na_rep="")
print(f"\nSaved — OE: {oe.shape}, Xsens: {xs.shape}")


# --- CELL 14 (code cell #11) ---
import pandas as pd

BASE = "/content/drive/MyDrive/thesis/data/group_3/"

for name, path in {
    "OE": BASE + "group3_oe_labelled.csv",
    "Xsens": BASE + "group3_xsens_labelled.csv",
}.items():
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)

    print(f"\n{name}")
    print("Whole_Group labels between 80s and 115s:")
    print(
        df[
            (df["t_video"] >= 80) &
            (df["t_video"] <= 115) &
            (df["Whole_Group"] != "")
        ][["t_video", "participant", "Whole_Group"]]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )

    print("\nWhole_Group value counts around early setup:")
    print(
        df[
            (df["t_video"] >= 80) &
            (df["t_video"] <= 115)
        ]["Whole_Group"].value_counts().head(10)
    )

    wrong = df[
        (df["t_video"] >= 80) &
        (df["t_video"] <= 115) &
        (df["Whole_Group"] == "inspecting_target_image")
    ]

    if len(wrong) > 0:
        print("WARNING: still has inspecting_target_image in setup interval")
    else:
        print("OK: setup interval is no longer mislabeled as inspecting_target_image")


# --- CELL 16 (code cell #12) ---
df = pd.read_csv("/content/drive/MyDrive/thesis/data/group_5/xsens/Participant1.csv", low_memory=False)
print(df["PacketCounter"].head(10).tolist())
print(df["PacketCounter"].tail(10).tolist())
print(f"PacketCounter min: {df['PacketCounter'].min()}, max: {df['PacketCounter'].max()}")
print(f"SampleTimeFine first 5: {df['SampleTimeFine'].head(5).tolist()}")
print(f"SampleTimeFine last 5: {df['SampleTimeFine'].tail(5).tolist()}")


# --- CELL 17 (code cell #13) ---
import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/thesis/data/group_5/xsens/Participant1.csv", low_memory=False)
df["SampleTimeFine"] = pd.to_numeric(df["SampleTimeFine"], errors="coerce")
df["PacketCounter"] = pd.to_numeric(df["PacketCounter"], errors="coerce")
df = df.dropna(subset=["SampleTimeFine","PacketCounter"])
df["SampleTimeFine"] = df["SampleTimeFine"].astype(float)
df = df.sort_values("PacketCounter").reset_index(drop=True)

# apply overflow fix
MAX_32 = 4294967296
stf = df["SampleTimeFine"].values.copy()
offset = 0
for i in range(1, len(stf)):
    if stf[i] + offset < stf[i-1] + offset - 1e8:
        offset += MAX_32
    stf[i] += offset
df["SampleTimeFine"] = stf

# find jumps
diff = df["SampleTimeFine"].diff()
jumps = diff[diff > 1e8]
print(f"Jumps after fix: {len(jumps)}")
for idx, val in jumps.items():
    print(f"  idx={idx}, gap={val/1e6:.1f}s")
    print(f"  PacketCounter around jump: {df.loc[idx-1,'PacketCounter']} → {df.loc[idx,'PacketCounter']}")
    print(f"  SampleTimeFine: {df.loc[idx-1,'SampleTimeFine']:.0f} → {df.loc[idx,'SampleTimeFine']:.0f}")


# --- CELL 18 (code cell #14) ---
elan_raw = pd.read_csv("/content/drive/MyDrive/thesis/data/group_5/elan/Group_5.csv", header=None,
                       names=["tier","_","t_start_hms","t_start_s",
                              "t_end_hms","t_end_s","duration_hms","duration_s","label"])

tier_map = {
    "Mintan": "Participant1", "Adarsh": "Participant2", "Ali": "Participant3",
    "Adarsh Mintan": "Participant1_Participant2",
    "Mintan Ali": "Participant1_Participant3",
    "Ali Adarsh": "Participant2_Participant3",
    "Whole Group": "Whole_Group",
}
elan_raw["tier"] = elan_raw["tier"].str.strip().replace(tier_map)

typo_map = {
    "traveling_between__units": "traveling_between_units",
    "traveling_between_untis": "traveling_between_units",
    "traveling_betwen_units": "traveling_between_units",
    "trvaeling_between_units": "traveling_between_units",
    "approcahing_to_piece": "traveling_between_units",
    "moving_to_single_units": "traveling_between_units",
    "pickinf_up_pieces": "picking_up_piece",
    "picking_up_pieces": "picking_up_piece",
    "pıcking_up_image": "picking_up_target_image",
    "picking_up_image": "picking_up_target_image",
    "putting_doen_tray": "putting_down_tray",
    "putting_down_image": "putting_down_target_image",
    "putting_down_tagret_image": "putting_down_target_image",
    "inscpecing_other_pieces": "inspecting_other_pieces",
    "looking_target_image": "inspecting_target_image",
    "matchin_pieces_with_target_image": "matching_pieces_to_target_image",
    "matching_image_to_pieces": "matching_pieces_to_target_image",
    "matching_pieces_to_image": "matching_pieces_to_target_image",
    "matching_pieces_with_image": "matching_pieces_to_target_image",
    "matching_pieces_with_target_image": "matching_pieces_to_target_image",
    "mathing_pieces_with_image": "matching_pieces_to_target_image",
    "merging_sub_pieces": "merging_subpiece",
    "moving_pieces_from_tray_to_central_table": "moving_pieces_to_central_table_from_tray",
    "moving_pieces_to_table__from_tray": "moving_pieces_to_central_table_from_tray",
    "moving_subpieces_to_tray": "moving_pieces_to_tray",
    "search_for_target_image": "searching_target_image",
    "searchin_for_image": "searching_target_image",
    "searching_for_image": "searching_target_image",
    "searching_for_target_image": "searching_target_image",
    "searching_tray": "searching_piece",
    "task_operaitonal_convo": "task_operational_convo",
    "pointing_to_piece": "presenting_piece",
    "replacing_image": "returning_target_image",
    "co_buiilding_subpiece": "co_building_subpiece",
    "co_building_pieces": "co_building_subpiece",
    "co_building_sub_piece": "co_building_subpiece",
    "co_building_sub_pieec": "co_building_subpiece",
    "co_builidng_sub_piece": "co_building_subpiece",
    "co_builidng_subpiece": "co_building_subpiece",
    "co_merging_sub_pieces": "co_merging_subpiece",
    "co_merging_subpieces": "co_merging_subpiece",
    "co_inspceitng_image": "co_inspecting_image",
    "co_inspecitng_image": "co_inspecting_image",
    "clap_synchronizaiton_move": "sync_move",
    "synchronaziton_move": "sync_move",
    "synchronizaiton_move": "sync_move",
}
elan_raw["label"] = elan_raw["label"].str.strip().replace(typo_map)

compound_map = {
    "co_building_subpiece + task_operational_convo": ["co_building_subpiece","task_operational_convo"],
    "co_inspecting_target_image + task_operational_convo": ["co_inspecting_image","task_operational_convo"],
    "object_handover + co_inspecting_image": ["object_handover","co_inspecting_image"],
    "object_handover + task_operational convo": ["object_handover","task_operational_convo"],
    "object_handover + task_operational_convo": ["object_handover","task_operational_convo"],
    "picking_up_image +traveling_between_units": ["picking_up_target_image","traveling_between_units"],
    "picking_up_target_image + pickinig_up_tray": ["picking_up_target_image","picking_up_tray"],
    "picking_up_target_image + traveling_between_units": ["picking_up_target_image","traveling_between_units"],
    "task_operational_convo + inspecting_subpiece": ["task_operational_convo","inspecting_other_pieces"],
    "traveling_between_units + presenting_target_image": ["traveling_between_units","presenting_target_image"],
}
expanded = []
for _, row in elan_raw.iterrows():
    label = str(row["label"]).strip()
    if label in compound_map:
        for l in compound_map[label]:
            new_row = row.copy(); new_row["label"] = l; expanded.append(new_row)
    else:
        expanded.append(row)
elan_raw = pd.DataFrame(expanded).reset_index(drop=True)

collab_cutoff = {"Participant1": 2965.0, "Participant2": 2047.0, "Participant3": 2047.0}
sync_intervals = elan_raw[elan_raw["label"] == "sync_move"][["t_start_s","t_end_s"]]

new_rows = []
for participant, cutoff in collab_cutoff.items():
    p_elan = elan_raw[elan_raw["tier"] == participant].sort_values("t_start_s")
    covered = []
    for _, row in p_elan.iterrows():
        start = max(0, row["t_start_s"]); end = min(cutoff, row["t_end_s"])
        if start < end: covered.append([start, end])
    for _, row in sync_intervals.iterrows():
        start = max(0, row["t_start_s"]); end = row["t_end_s"]
        if start < end: covered.append([start, end])
    merged_iv = []
    for start, end in sorted(covered):
        if merged_iv and start <= merged_iv[-1][1]: merged_iv[-1][1] = max(merged_iv[-1][1], end)
        else: merged_iv.append([start, end])
    cursor = 0.0
    for start, end in merged_iv:
        if start > cursor:
            new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": start, "label": "individual_building"})
        cursor = max(cursor, end)
    if cursor < cutoff:
        new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": cutoff, "label": "individual_building"})

elan = pd.concat([elan_raw, pd.DataFrame(new_rows)], ignore_index=True).sort_values(["tier","t_start_s"]).reset_index(drop=True)
elan.to_csv("/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_clean.csv", index=False)
print(f"Group 5 ELAN: {len(elan)} rows, {elan['label'].nunique()} labels")
print("sync_move rows:", (elan['label']=='sync_move').sum())


# --- CELL 19 (code cell #15) ---
BASE            = "/content/drive/MyDrive/thesis/data/group_5/"
ELAN_FILE       = BASE + "elan/Group_5_clean.csv"
PARTICIPANTS    = ["Participant1", "Participant2", "Participant3"]
VIDEO_START_UTC = datetime(2026, 4, 23, 11, 8, 44, tzinfo=timezone.utc).timestamp()
XSENS_START_UTC = datetime(2026, 4, 23, 11, 12, 10, tzinfo=timezone.utc).timestamp()
XSENS_OFFSET    = XSENS_START_UTC - VIDEO_START_UTC

ALL_TIERS = ["Participant1", "Participant2", "Participant3",
             "Participant1_Participant2", "Participant1_Participant3",
             "Participant2_Participant3", "Whole_Group"]

collab_cutoff = {"Participant1": 2965.0, "Participant2": 2047.0, "Participant3": 2047.0}

elan = pd.read_csv(ELAN_FILE)

oe_all = []
for p in PARTICIPANTS:
    merged = load_oe_all(BASE + f"openearable/{p}/", p)
    print(f"OE {p}: {len(merged)} rows")
    oe_all.append(merged)
oe = pd.concat(oe_all, ignore_index=True).sort_values(["ts","participant"]).reset_index(drop=True)
oe["t_video"] = (oe["ts"] / 1e6) - VIDEO_START_UTC

xs = load_xsens(BASE + "xsens/", PARTICIPANTS, XSENS_OFFSET)

print(f"\nOE t_video: {oe['t_video'].min():.1f} to {oe['t_video'].max():.1f}s")
print(f"Xsens t_video: {xs['t_video'].min():.1f} to {xs['t_video'].max():.1f}s")

print("\nLabelling OE:")
oe = label_df(oe, "t_video", elan, ALL_TIERS, collab_cutoff)
print("\nLabelling Xsens:")
xs = label_df(xs, "t_video", elan, ALL_TIERS, collab_cutoff)

# clean NaN → empty string in tier columns
for tier in ALL_TIERS:
    oe[tier] = oe[tier].fillna("").replace("nan", "")
    xs[tier] = xs[tier].fillna("").replace("nan", "")

oe.to_csv(BASE + "group5_oe_labelled.csv", index=False, na_rep="")
xs.to_csv(BASE + "group5_xsens_labelled.csv", index=False, na_rep="")
print(f"\nSaved — OE: {oe.shape}, Xsens: {xs.shape}")


# --- CELL 20 (code cell #16) ---
import pandas as pd

groups = {
    "group_1": {
        "oe":    "/content/drive/MyDrive/thesis/data/group_1/group1_oe_labelled.csv",
        "xsens": "/content/drive/MyDrive/thesis/data/group_1/group1_xsens_labelled.csv",
        "tiers": ["Arda","Bas","Rachel","Bas_Arda","Arda_Rachel","Bas_Rachel","Whole_Group"],
    },
    "group_2": {
        "oe":    "/content/drive/MyDrive/thesis/data/group_2/group2_oe_labelled.csv",
        "xsens": "/content/drive/MyDrive/thesis/data/group_2/group2_xsens_labelled.csv",
        "tiers": ["Participant1","Participant2","Participant3",
                  "Participant1_Participant2","Participant1_Participant3",
                  "Participant2_Participant3","Whole_Group"],
    },
    "group_3": {
        "oe":    "/content/drive/MyDrive/thesis/data/group_3/group3_oe_labelled.csv",
        "xsens": "/content/drive/MyDrive/thesis/data/group_3/group3_xsens_labelled.csv",
        "tiers": ["Participant1","Participant2","Participant3",
                  "Participant1_Participant2","Participant1_Participant3",
                  "Participant2_Participant3","Whole_Group"],
    },
    "group_5": {
        "oe":    "/content/drive/MyDrive/thesis/data/group_5/group5_oe_labelled.csv",
        "xsens": "/content/drive/MyDrive/thesis/data/group_5/group5_xsens_labelled.csv",
        "tiers": ["Participant1","Participant2","Participant3",
                  "Participant1_Participant2","Participant1_Participant3",
                  "Participant2_Participant3","Whole_Group"],
    },
}

for group, info in groups.items():
    print(f"\n{'='*40}")
    print(f"{group.upper()}")
    for sensor, path in [("OE", info["oe"]), ("Xsens", info["xsens"])]:
        df = pd.read_csv(path, low_memory=False, keep_default_na=False)
        tiers = info["tiers"]

        # 1. sync_move present
        sync_counts = {t: (df[t] == "sync_move").sum() for t in tiers}
        sync_total  = sum(sync_counts.values())

        # 2. any NaN in tier columns
        nan_counts  = {t: df[t].isna().sum() for t in tiers}
        nan_total   = sum(nan_counts.values())

        # 3. individual_building overlapping sync_move rows
        overlap = 0
        for t in tiers:
            if "_" not in t and t != "Whole_Group":  # individual tiers only
                # find rows where another tier has sync_move and this tier has individual_building
                sync_mask = df[[x for x in tiers if x != t]].apply(
                    lambda col: col == "sync_move", axis=0).any(axis=1)
                ib_mask = df[t] == "individual_building"
                overlap += (sync_mask & ib_mask).sum()

        print(f"  {sensor}: shape={df.shape} | "
              f"sync_move_rows={sync_total} | "
              f"NaNs={nan_total} | "
              f"individual_building+sync_overlap={overlap}")
        if sync_total == 0:
            print(f"    ⚠️  NO sync_move found — needs reprocessing")
        if nan_total > 0:
            print(f"    ⚠️  NaNs: {nan_counts}")
        if overlap > 0:
            print(f"    ⚠️  overlap in tiers: check individual_building gap filling")


# --- CELL 21 (code cell #17) ---
import pandas as pd

for group, path, tiers in [
    ("group_2", "/content/drive/MyDrive/thesis/data/group_2/group2_xsens_labelled.csv",
     ["Participant1","Participant2","Participant3"]),
    ("group_3", "/content/drive/MyDrive/thesis/data/group_3/group3_xsens_labelled.csv",
     ["Participant1","Participant2","Participant3"]),
]:
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    all_tiers = [c for c in df.columns if "Participant" in c or c == "Whole_Group"]
    for t in tiers:
        sync_mask = df[[x for x in all_tiers if x != t]].apply(
            lambda col: col == "sync_move", axis=0).any(axis=1)
        ib_mask = df[t] == "individual_building"
        overlap = df[sync_mask & ib_mask]
        if len(overlap) > 0:
            print(f"{group} — {t}: {len(overlap)} overlap rows")
            print(f"  t_video range: {overlap['t_video'].min():.2f} to {overlap['t_video'].max():.2f}s")
            print(f"  sync_move in tiers: {[x for x in all_tiers if x != t and (overlap[x]=='sync_move').any()]}")


# --- CELL 23 (code cell #18) ---
from datetime import datetime, timezone
import pandas as pd

BASE = "/content/drive/MyDrive/thesis/data/group_6/openearable/"
for p in ["Participant1", "Participant2", "Participant3"]:
    df = pd.read_csv(f"{BASE}{p}/{p}_acc.csv", header=None)
    ts = float(df.iloc[0, 0]) / 1e6
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    print(f"{p}: {dt.strftime('%H:%M:%S')} UTC")


# --- CELL 24 (code cell #19) ---
elan = pd.read_csv("/content/drive/MyDrive/thesis/data/group_6/elan/Group_6.csv", header=None)
print("Tiers:", elan[0].unique().tolist())
print("\nLabels:")
for l in sorted(elan[8].dropna().str.strip().unique().tolist()):
    print(f"  {l}")


# --- CELL 25 (code cell #20) ---
elan = pd.read_csv("/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_clean.csv")

# check what's annotated around the sync move time
sync = elan[elan["label"] == "sync_move"]
print("Sync move rows:")
print(sync[["tier","t_start_s","t_end_s","label"]].to_string())

print("\nCollab cutoffs were: arda=1420, bas=1160, rachel=1360")
print("Sync move after all cutoffs?", sync["t_start_s"].min() > 1420)


# --- CELL 26 (code cell #21) ---
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BASE         = "/content/drive/MyDrive/thesis/data/group_6/"
ELAN_FILE    = BASE + "elan/Group_6_clean.csv"
PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

VIDEO_START_UTC = datetime(2026, 4, 23, 14, 17, 43, tzinfo=timezone.utc).timestamp()
XSENS_START_UTC = datetime(2026, 4, 23, 14,  9, 38, tzinfo=timezone.utc).timestamp()
XSENS_OFFSET    = XSENS_START_UTC - VIDEO_START_UTC

ALL_TIERS = ["Participant1", "Participant2", "Participant3",
             "Participant1_Participant2", "Participant1_Participant3",
             "Participant2_Participant3", "Whole_Group"]

collab_cutoff = {
    "Participant1": 752.0,
    "Participant2": 1030.0,
    "Participant3": 752.0,
}

# ── ELAN cleaning ─────────────────────────────────────────────────────────────
elan_raw = pd.read_csv(BASE + "elan/Group_6.csv", header=None,
                       names=["tier","_","t_start_hms","t_start_s",
                              "t_end_hms","t_end_s","duration_hms","duration_s","label"])

tier_map = {
    "Kas":         "Participant1",
    "Gozluk":      "Participant2",
    "Mark":        "Participant3",
    "Gozluk Kas":  "Participant1_Participant2",
    "Kas Mark":    "Participant1_Participant3",
    "Mark Gozluk": "Participant2_Participant3",
    "Whole Group": "Whole_Group",
}
elan_raw["tier"] = elan_raw["tier"].str.strip().replace(tier_map)

typo_map = {
    "traveling_betwen_units":                       "traveling_between_units",
    "inspecitng_other_unit":                        "inspecting_other_units",
    "inspecitng_other_units":                       "inspecting_other_units",
    "inspecting_others_unit":                       "inspecting_other_units",
    "inspecting_piece":                             "inspecting_pieces",
    "pickinup_piece":                               "picking_up_piece",
    "picking_up_pieces":                            "picking_up_piece",
    "caryying_tray":                                "carrying_tray",
    "co_building_sub_piece":                        "co_building_subpiece",
    "merging_sub_pieces":                           "merging_subpiece",
    "moving_pieces_from_table_to_tray":             "moving_pieces_to_tray",
    "moving_pieces_to_tray_from_table":             "moving_pieces_to_tray",
    "moving_pieces_from_tray_to_central_table":     "moving_pieces_to_central_table_from_tray",
    "building_subpiece":                            "individual_building",
    "searching_for_target_image":                   "searching_target_image",
    "task_operaitonal_convo":                       "task_operational_convo",
}
elan_raw["label"] = elan_raw["label"].str.strip().replace(typo_map)

compound_map = {
    "object_handover + task_operational_convo":                 ["object_handover","task_operational_convo"],
    "picking_up_target_image + traveling_between_units":        ["picking_up_target_image","traveling_between_units"],
    "putting_down_target_image + traveling_between_units":      ["putting_down_target_image","traveling_between_units"],
    "task_operational_convo + matching_pieces_to_target_image": ["task_operational_convo","matching_pieces_to_target_image"],
    "traveling_between_units + inspecting_other_units":         ["traveling_between_units","inspecting_other_units"],
    "traveling_between_units + putting_down_piece":             ["traveling_between_units","putting_down_piece"],
}
expanded = []
for _, row in elan_raw.iterrows():
    label = str(row["label"]).strip()
    if label in compound_map:
        for l in compound_map[label]:
            new_row = row.copy(); new_row["label"] = l; expanded.append(new_row)
    else:
        expanded.append(row)
elan_raw = pd.DataFrame(expanded).reset_index(drop=True)

sync_map = {
    "synchronaziton_move":               "sync_move",
    "synchronizaiton_move":              "sync_move",
    "clap_synchronizaiton_move":         "sync_move",
    "khalil_synchornizaion_move":        "sync_move",
    "droping_erarble_syncornaziton_move":"sync_move",
}
elan_raw["label"] = elan_raw["label"].replace(sync_map)

# individual_building gaps with sync_move exclusion
sync_intervals = elan_raw[elan_raw["label"] == "sync_move"][["t_start_s","t_end_s"]]

new_rows = []
for participant, cutoff in collab_cutoff.items():
    p_elan = elan_raw[elan_raw["tier"] == participant].sort_values("t_start_s")
    covered = []
    for _, row in p_elan.iterrows():
        start = max(0, row["t_start_s"]); end = min(cutoff, row["t_end_s"])
        if start < end: covered.append([start, end])
    for _, row in sync_intervals.iterrows():
        start = max(0, row["t_start_s"]); end = row["t_end_s"]
        if start < end: covered.append([start, end])
    merged_iv = []
    for start, end in sorted(covered):
        if merged_iv and start <= merged_iv[-1][1]: merged_iv[-1][1] = max(merged_iv[-1][1], end)
        else: merged_iv.append([start, end])
    cursor = 0.0
    for start, end in merged_iv:
        if start > cursor:
            new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": start, "label": "individual_building"})
        cursor = max(cursor, end)
    if cursor < cutoff:
        new_rows.append({"tier": participant, "t_start_s": cursor, "t_end_s": cutoff, "label": "individual_building"})

elan = pd.concat([elan_raw, pd.DataFrame(new_rows)], ignore_index=True).sort_values(["tier","t_start_s"]).reset_index(drop=True)
elan.to_csv(ELAN_FILE, index=False)
print(f"ELAN saved: {len(elan)} rows, {elan['label'].nunique()} unique labels")
print(sorted(elan["label"].dropna().unique().tolist()))

# ── OE ────────────────────────────────────────────────────────────────────────
oe_all = []
for p in PARTICIPANTS:
    merged = load_oe_all(BASE + f"openearable/{p}/", p)
    print(f"OE {p}: {len(merged)} rows")
    oe_all.append(merged)
oe = pd.concat(oe_all, ignore_index=True).sort_values(["ts","participant"]).reset_index(drop=True)
oe["t_video"] = (oe["ts"] / 1e6) - VIDEO_START_UTC

# ── Xsens ─────────────────────────────────────────────────────────────────────
xs = load_xsens(BASE + "xsens/", PARTICIPANTS, XSENS_OFFSET)

print(f"\nOE t_video:    {oe['t_video'].min():.1f} to {oe['t_video'].max():.1f}s")
print(f"Xsens t_video: {xs['t_video'].min():.1f} to {xs['t_video'].max():.1f}s")

# ── label ─────────────────────────────────────────────────────────────────────
print("\nLabelling OE:")
oe = label_df(oe, "t_video", elan, ALL_TIERS, collab_cutoff)
print("\nLabelling Xsens:")
xs = label_df(xs, "t_video", elan, ALL_TIERS, collab_cutoff)

for tier in ALL_TIERS:
    oe[tier] = oe[tier].fillna("").replace("nan", "")
    xs[tier] = xs[tier].fillna("").replace("nan", "")

oe.to_csv(BASE + "group6_oe_labelled.csv", index=False, na_rep="")
xs.to_csv(BASE + "group6_xsens_labelled.csv", index=False, na_rep="")
print(f"\nSaved group6_oe_labelled.csv    {oe.shape}")
print(f"Saved group6_xsens_labelled.csv {xs.shape}")


# --- CELL 27 (code cell #22) ---
import pandas as pd

BASE  = "/content/drive/MyDrive/thesis/data/group_6/"
TIERS = ["Participant1","Participant2","Participant3",
         "Participant1_Participant2","Participant1_Participant3",
         "Participant2_Participant3","Whole_Group"]

for sensor, path in [
    ("OE",    BASE + "group6_oe_labelled.csv"),
    ("Xsens", BASE + "group6_xsens_labelled.csv"),
]:
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    print(f"=== GROUP 6 {sensor} ===")
    print(f"Shape:        {df.shape}")
    print(f"Participants: {df['participant'].value_counts().to_dict()}")
    print(f"t_video:      {df['t_video'].min():.1f} to {df['t_video'].max():.1f}s")

    # NaNs
    nan_total = sum(df[t].isna().sum() for t in TIERS)
    print(f"NaNs:         {nan_total}")

    # sync_move
    sync_counts = {t: (df[t]=='sync_move').sum() for t in TIERS}
    print(f"sync_move:    {sync_counts}")

    # individual_building + sync_move overlap
    overlap = 0
    for t in ["Participant1","Participant2","Participant3"]:
        sync_mask = df[[x for x in TIERS if x != t]].apply(
            lambda col: col == "sync_move", axis=0).any(axis=1)
        ib_mask = df[t] == "individual_building"
        overlap += (sync_mask & ib_mask).sum()
    print(f"IB+sync overlap: {overlap}")

    # filled rows per tier
    print("Filled per tier:")
    for t in TIERS:
        filled = (df[t] != '').sum()
        top2   = df[df[t]!=''][t].value_counts().head(2).to_dict()
        print(f"  {t}: {filled} rows — top: {top2}")

    print()


# --- CELL 29 (code cell #23) ---
# Final sanity check for all processed groups
# Run this after reprocessing Groups 1, 2, 3, 5, and 6.
#
# This validator treats tier columns as STRING LABEL columns, not numeric 0/1 columns.
# Empty cells are read as "", so counts are based on non-empty labels.

import pandas as pd
import json

GROUP_VALIDATION = {
    "group_1": {
        "base": "/content/drive/MyDrive/thesis/data/group_1/",
        "files": {"OE": "group1_oe_labelled.csv", "Xsens": "group1_xsens_labelled.csv"},
        "tiers": ["Arda", "Bas", "Rachel", "Bas_Arda", "Arda_Rachel", "Bas_Rachel", "Whole_Group"],
        "individual": ["Arda", "Bas", "Rachel"],
        "expected_zero_ok": [],
    },
    "group_2": {
        "base": "/content/drive/MyDrive/thesis/data/group_2/",
        "files": {"OE": "group2_oe_labelled.csv", "Xsens": "group2_xsens_labelled.csv"},
        "tiers": ["Participant1", "Participant2", "Participant3", "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3", "Whole_Group"],
        "individual": ["Participant1", "Participant2", "Participant3"],
        "expected_zero_ok": [],
    },
    "group_3": {
        "base": "/content/drive/MyDrive/thesis/data/group_3/",
        "files": {"OE": "group3_oe_labelled.csv", "Xsens": "group3_xsens_labelled.csv"},
        "tiers": ["Participant1", "Participant2", "Participant3", "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3", "Whole_Group"],
        "individual": ["Participant1", "Participant2", "Participant3"],
        # Chinyu–Long had no interaction, so this tier being empty is valid.
        "expected_zero_ok": ["Participant1_Participant2"],
    },
    "group_5": {
        "base": "/content/drive/MyDrive/thesis/data/group_5/",
        "files": {"OE": "group5_oe_labelled.csv", "Xsens": "group5_xsens_labelled.csv"},
        "tiers": ["Participant1", "Participant2", "Participant3", "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3", "Whole_Group"],
        "individual": ["Participant1", "Participant2", "Participant3"],
        "expected_zero_ok": [],
    },
    "group_6": {
        "base": "/content/drive/MyDrive/thesis/data/group_6/",
        "files": {"OE": "group6_oe_labelled.csv", "Xsens": "group6_xsens_labelled.csv"},
        "tiers": ["Participant1", "Participant2", "Participant3", "Participant1_Participant2", "Participant1_Participant3", "Participant2_Participant3", "Whole_Group"],
        "individual": ["Participant1", "Participant2", "Participant3"],
        "expected_zero_ok": [],
    },
}

validation_rows = []

for group, info in GROUP_VALIDATION.items():
    print(f"\n{'='*60}\n{group.upper()}")
    for sensor, fname in info["files"].items():
        path = info["base"] + fname
        df = pd.read_csv(path, low_memory=False, keep_default_na=False)

        tiers = [t for t in info["tiers"] if t in df.columns]
        missing_tiers = [t for t in info["tiers"] if t not in df.columns]

        sync_mask = df[tiers].eq("sync_move").any(axis=1) if tiers else pd.Series(False, index=df.index)
        sync_rows = int(sync_mask.sum())

        ib_sync_overlap = 0
        for t in info["individual"]:
            if t in df.columns:
                ib_sync_overlap += int(((df[t] == "individual_building") & sync_mask).sum())

        print(f"{sensor}: shape={df.shape}, sync_rows={sync_rows}, IB+sync_overlap={ib_sync_overlap}")

        if missing_tiers:
            print(f"  WARNING missing tier columns: {missing_tiers}")

        if ib_sync_overlap > 0:
            print("  WARNING: individual_building overlaps with sync_move. Re-run preprocessing with the fixed label_df().")

        print("  Filled rows per tier:")
        for t in info["tiers"]:
            if t not in df.columns:
                filled = None
                top_values = {}
            else:
                filled = int((df[t] != "").sum())
                top_values = df.loc[df[t] != "", t].value_counts().head(5).to_dict()

            expected_zero = t in info.get("expected_zero_ok", [])
            if filled == 0 and not expected_zero:
                print(f"    WARNING {t}: 0 rows")
            elif filled == 0 and expected_zero:
                print(f"    {t}: 0 rows (expected/OK)")
            else:
                print(f"    {t}: {filled} rows — top: {top_values}")

            validation_rows.append({
                "group": group,
                "sensor": sensor,
                "file": fname,
                "tier": t,
                "rows": len(df),
                "filled_rows": filled,
                "sync_rows_any_tier": sync_rows,
                "individual_building_sync_overlap": ib_sync_overlap,
                "missing_tier": t in missing_tiers,
                "expected_zero_ok": expected_zero,
                "top_values": json.dumps(top_values, ensure_ascii=False),
            })

validation_df = pd.DataFrame(validation_rows)
validation_out = "/content/drive/MyDrive/thesis/data/all_groups_validation_summary.csv"
validation_df.to_csv(validation_out, index=False)
print(f"\nSaved validation summary to: {validation_out}")

