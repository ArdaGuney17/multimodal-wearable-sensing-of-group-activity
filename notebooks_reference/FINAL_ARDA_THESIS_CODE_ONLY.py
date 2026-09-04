# --- CELL 0 (code cell #1) ---
# ============================================================
# FINAL SUPERVISOR VISUALIZATION DASHBOARD - PARTICIPANT LINES
# Same clean style as initial version, but:
# - P1/P2/P3 shown as separate lines
# - Change SIGNAL_TYPE to "acc_mag" or "gyro_mag"
# - Saves one figure per group
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIG
# ============================================================

BASE = "/content/drive/MyDrive/thesis/data"

OUT_DIR = "/content/drive/MyDrive/thesis/final_visualizations_participant_lines"
os.makedirs(OUT_DIR, exist_ok=True)

# Change this:
# "acc_mag"  -> accelerometer magnitude
# "gyro_mag" -> gyroscope magnitude
SIGNAL_TYPE = "acc_mag"

GROUP_FILES = {
    1: {
        "oe":    f"{BASE}/group_1/openearable_labeled/group_1_openearable_labeled.csv",
        "xsens": f"{BASE}/group_1/xsens_labeled/group_1_xsens_labeled.csv",
        "note": "Standard labeled files."
    },
    2: {
        "oe":    f"{BASE}/group_2/openearable_labeled/group_2_openearable_labeled.csv",
        "xsens": f"{BASE}/group_2/xsens_labeled/group_2_xsens_labeled.csv",
        "note": "Standard labeled files."
    },
    3: {
        "oe":    f"{BASE}/group_3/openearable_labeled/group_3_openearable_labeled.csv",
        "xsens": f"{BASE}/group_3/xsens_labeled/group_3_xsens_labeled.csv",
        "note": "Timing checked and aligned."
    },
    5: {
        "oe":    f"{BASE}/group_5/openearable_labeled/group_5_openearable_labeled.csv",
        "xsens": f"{BASE}/group_5/xsens_labeled/group_5_xsens_labeled.csv",
        "note": "XSens sync checked. OE sync weaker."
    },
    6: {
        "oe":    f"{BASE}/group_6/openearable_labeled/group_6_openearable_labeled.csv",
        "xsens": f"{BASE}/group_6/xsens_labeled/group_6_xsens_labeled_cleaned.csv",
        "note": "XSens cleaned. 82 artifact rows removed/set to NaN."
    },
    7: {
        "oe":    f"{BASE}/group_7/openearable_labeled/group_7_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_7/xsens_labeled/group_7_xsens_labeled_SHIFTED.csv",
        "note": "Shifted. OE +14.880s, XSens +20.350s. P3 OE partial."
    },
    8: {
        "oe":    f"{BASE}/group_8/openearable_labeled/group_8_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_8/xsens_labeled/group_8_xsens_labeled_SHIFTED.csv",
        "note": "Shifted. OE +27.210s, XSens +28.234s."
    },
    9: {
        "oe":    f"{BASE}/group_9/openearable_labeled/group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv",
        "xsens": f"{BASE}/group_9/xsens_labeled/group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv",
        "note": "Shifted using early common peak. OE +46.725s, XSens +48.093s. OE P3 partial; XSens shorter."
    },
    10: {
        "oe":    f"{BASE}/group_10/openearable_labeled/group_10_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_10/xsens_labeled/group_10_xsens_labeled_SHIFTED.csv",
        "note": "XSens cleaned. Shifted late sync. OE +31.970s, XSens +26.650s. P2 partial."
    },
}

TIME_COL = "video_time_s"
USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Whole_Group"
]

AVAILABILITY_COLS = [
    "p3_oe_available",
    "p2_oe_available",
    "p1_xsens_available",
    "p2_xsens_available",
    "p3_xsens_available"
]

FIGSIZE = (22, 10)
SMOOTH_OE = 25
SMOOTH_XSENS = 15
LABEL_ALPHA = 0.16
TEXT_EVERY_N_SEGMENTS = 8


# ============================================================
# HELPERS
# ============================================================

def load_file(path):
    if not os.path.exists(path):
        print(f"WARNING: file not found: {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def moving_average(x, win):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            cols = [f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"]
        elif signal_type == "gyro_mag":
            cols = [f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"]
        else:
            raise ValueError("SIGNAL_TYPE must be 'acc_mag' or 'gyro_mag'")

    elif source == "xsens":
        if signal_type == "acc_mag":
            cols = [f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"]
        elif signal_type == "gyro_mag":
            cols = [f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"]
        else:
            raise ValueError("SIGNAL_TYPE must be 'acc_mag' or 'gyro_mag'")

    else:
        raise ValueError("source must be oe or xsens")

    if not all(c in df.columns for c in cols):
        return np.full(len(df), np.nan)

    x = pd.to_numeric(df[cols[0]], errors="coerce")
    y = pd.to_numeric(df[cols[1]], errors="coerce")
    z = pd.to_numeric(df[cols[2]], errors="coerce")

    return np.sqrt(x**2 + y**2 + z**2)


def robust_ylim(signals, lower=1, upper=99):
    values = []

    for s in signals:
        s = np.asarray(s)
        values.extend(s[np.isfinite(s)])

    values = np.asarray(values)

    if len(values) == 0:
        return None

    lo, hi = np.percentile(values, [lower, upper])

    if lo == hi:
        return lo - 1, hi + 1

    margin = (hi - lo) * 0.2
    return lo - margin, hi + margin


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def make_label_colors(df_oe, df_xs):
    all_labels = []

    for df in [df_oe, df_xs]:
        if df is None:
            continue

        for col in LABEL_COLS:
            if col in df.columns:
                vals = df[col].dropna().astype(str).unique().tolist()
                vals = [v for v in vals if v != ""]
                all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {
        lab: cmap(i % 20)
        for i, lab in enumerate(unique_labels)
    }

    return label_colors, unique_labels


def add_label_spans(ax, df, label_colors):
    segment_counter = 0

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        segments = get_label_segments(df, TIME_COL, col)

        for s, e, lab in segments:
            if lab == "":
                continue

            ax.axvspan(
                s,
                e,
                color=label_colors.get(lab, "gray"),
                alpha=LABEL_ALPHA
            )

            if col == "label_Whole_Group" and segment_counter % TEXT_EVERY_N_SEGMENTS == 0:
                mid = (s + e) / 2
                y_top = ax.get_ylim()[1]
                ax.text(
                    mid,
                    y_top,
                    lab,
                    rotation=90,
                    fontsize=6,
                    ha="center",
                    va="top"
                )

            segment_counter += 1


def add_availability_shading(ax, df, time):
    y0, y1 = ax.get_ylim()

    for col in AVAILABILITY_COLS:
        if col in df.columns:
            unavailable = ~df[col].astype(bool).values

            ax.fill_between(
                time,
                y0,
                y1,
                where=unavailable,
                alpha=0.08,
                label=f"{col} unavailable"
            )


def summarize_file(df, group, source, path, note):
    if df is None:
        return {
            "group": group,
            "source": source,
            "rows": None,
            "start_s": None,
            "end_s": None,
            "duration_min": None,
            "whole_group_pct": None,
            "availability_flags": "",
            "path": path,
            "note": note
        }

    duration_min = (df[TIME_COL].max() - df[TIME_COL].min()) / 60

    whole_pct = None
    if "label_Whole_Group" in df.columns:
        whole_pct = df["label_Whole_Group"].notna().mean() * 100

    flags = []

    for col in AVAILABILITY_COLS:
        if col in df.columns:
            flags.append(f"{col}: {df[col].mean()*100:.1f}%")

    return {
        "group": group,
        "source": source,
        "rows": len(df),
        "start_s": round(df[TIME_COL].min(), 2),
        "end_s": round(df[TIME_COL].max(), 2),
        "duration_min": round(duration_min, 2),
        "whole_group_pct": round(whole_pct, 1) if whole_pct is not None else None,
        "availability_flags": " | ".join(flags),
        "path": path,
        "note": note
    }


def plot_group(group, oe_df, xs_df, note):
    label_colors, unique_labels = make_label_colors(oe_df, xs_df)

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=False)

    # --------------------------------------------------------
    # OpenEarable
    # --------------------------------------------------------
    if oe_df is not None:
        t = oe_df[TIME_COL].values

        signals = []
        for user in USERS:
            sig = get_signal(oe_df, user, "oe", SIGNAL_TYPE)
            sig = moving_average(sig, SMOOTH_OE)
            signals.append(sig)

            axes[0].plot(
                t,
                sig,
                linewidth=0.9,
                label=f"P{user}"
            )

        ylim = robust_ylim(signals)
        if ylim is not None:
            axes[0].set_ylim(ylim)

        add_label_spans(axes[0], oe_df, label_colors)
        add_availability_shading(axes[0], oe_df, t)

        axes[0].set_title(f"Group {group} OpenEarable | {SIGNAL_TYPE} | {note}")
        axes[0].set_ylabel(SIGNAL_TYPE)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="upper right", ncol=3)

    else:
        axes[0].set_title(f"Group {group} OpenEarable missing")

    # --------------------------------------------------------
    # XSens
    # --------------------------------------------------------
    if xs_df is not None:
        t = xs_df[TIME_COL].values

        signals = []
        for user in USERS:
            sig = get_signal(xs_df, user, "xsens", SIGNAL_TYPE)
            sig = moving_average(sig, SMOOTH_XSENS)
            signals.append(sig)

            axes[1].plot(
                t,
                sig,
                linewidth=0.9,
                label=f"P{user}"
            )

        ylim = robust_ylim(signals)
        if ylim is not None:
            axes[1].set_ylim(ylim)

        add_label_spans(axes[1], xs_df, label_colors)
        add_availability_shading(axes[1], xs_df, t)

        axes[1].set_title(f"Group {group} XSens | {SIGNAL_TYPE}")
        axes[1].set_ylabel(SIGNAL_TYPE)
        axes[1].set_xlabel("Video time (seconds)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right", ncol=3)

    else:
        axes[1].set_title(f"Group {group} XSens missing")

    # --------------------------------------------------------
    # Label legend
    # --------------------------------------------------------
    if len(unique_labels) > 0:
        patches = [
            Patch(
                facecolor=label_colors[lab],
                alpha=LABEL_ALPHA,
                label=lab
            )
            for lab in unique_labels[:30]
        ]

        fig.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            title="Labels"
        )

    plt.tight_layout()

    out_path = os.path.join(
        OUT_DIR,
        f"group_{group}_final_{SIGNAL_TYPE}_participant_lines.png"
    )

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved figure ->", out_path)


# ============================================================
# RUN
# ============================================================

summary_rows = []

for group, cfg in GROUP_FILES.items():
    print("\n" + "="*90)
    print(f"GROUP {group}")
    print("="*90)

    oe_df = load_file(cfg["oe"])
    xs_df = load_file(cfg["xsens"])

    summary_rows.append(
        summarize_file(oe_df, group, "OpenEarable", cfg["oe"], cfg["note"])
    )

    summary_rows.append(
        summarize_file(xs_df, group, "XSens", cfg["xsens"], cfg["note"])
    )

    plot_group(group, oe_df, xs_df, cfg["note"])

summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUT_DIR,
    f"final_group_file_summary_{SIGNAL_TYPE}.csv"
)

summary.to_csv(summary_path, index=False)

print("\n" + "="*90)
print("FINAL SUMMARY TABLE")
print("="*90)

display(summary)

print("\nSaved summary CSV ->", summary_path)
print("Saved figures directory ->", OUT_DIR)


# --- CELL 1 (code cell #2) ---
# ============================================================
# FINAL LABEL INVENTORY ACROSS ALL FINAL FILES
# Groups: 1,2,3,5,6,7,8,9,10
# Produces:
# - full_label_inventory.csv
# - unique_label_list.csv
# ============================================================

import os
import pandas as pd
import numpy as np

BASE = "/content/drive/MyDrive/thesis/data"
OUT_DIR = "/content/drive/MyDrive/thesis/final_label_inventory"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_FILES = {
    1: {
        "oe":    f"{BASE}/group_1/openearable_labeled/group_1_openearable_labeled.csv",
        "xsens": f"{BASE}/group_1/xsens_labeled/group_1_xsens_labeled.csv",
    },
    2: {
        "oe":    f"{BASE}/group_2/openearable_labeled/group_2_openearable_labeled.csv",
        "xsens": f"{BASE}/group_2/xsens_labeled/group_2_xsens_labeled.csv",
    },
    3: {
        "oe":    f"{BASE}/group_3/openearable_labeled/group_3_openearable_labeled.csv",
        "xsens": f"{BASE}/group_3/xsens_labeled/group_3_xsens_labeled.csv",
    },
    5: {
        "oe":    f"{BASE}/group_5/openearable_labeled/group_5_openearable_labeled.csv",
        "xsens": f"{BASE}/group_5/xsens_labeled/group_5_xsens_labeled.csv",
    },
    6: {
        "oe":    f"{BASE}/group_6/openearable_labeled/group_6_openearable_labeled.csv",
        "xsens": f"{BASE}/group_6/xsens_labeled/group_6_xsens_labeled_cleaned.csv",
    },
    7: {
        "oe":    f"{BASE}/group_7/openearable_labeled/group_7_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_7/xsens_labeled/group_7_xsens_labeled_SHIFTED.csv",
    },
    8: {
        "oe":    f"{BASE}/group_8/openearable_labeled/group_8_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_8/xsens_labeled/group_8_xsens_labeled_SHIFTED.csv",
    },
    9: {
        "oe":    f"{BASE}/group_9/openearable_labeled/group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv",
        "xsens": f"{BASE}/group_9/xsens_labeled/group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv",
    },
    10: {
        "oe":    f"{BASE}/group_10/openearable_labeled/group_10_openearable_labeled_SHIFTED.csv",
        "xsens": f"{BASE}/group_10/xsens_labeled/group_10_xsens_labeled_SHIFTED.csv",
    },
}

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group",
]

TIER_TYPE = {
    "label_Participant1": "individual",
    "label_Participant2": "individual",
    "label_Participant3": "individual",
    "label_Participant1_Participant2": "dyad",
    "label_Participant1_Participant3": "dyad",
    "label_Participant2_Participant3": "dyad",
    "label_Whole_Group": "whole_group",
}


def broad_category(label):
    """
    Rough first-pass grouping.
    We can refine this manually after seeing the inventory.
    """
    lab = str(label).lower().strip()

    if lab == "" or lab == "nan":
        return "empty"

    if "individual_build" in lab or "starting_individual_build" in lab:
        return "individual_build"

    if "sync" in lab or "synch" in lab or "synchron" in lab:
        return "synchronization"

    if "social" in lab or "non_task" in lab:
        return "social/non-task conversation"

    if "operational" in lab or "operaitonal" in lab or "operat" in lab or "talk" in lab or "convo" in lab:
        return "task-related communication"

    if "travel" in lab or "approach" in lab or "carrying" in lab or "cary" in lab:
        return "movement / transition"

    if "target_image" in lab or "guide_image" in lab or "image" in lab:
        return "target/guide image interaction"

    if "inspect" in lab or "inspec" in lab or "matching" in lab or "search" in lab or "pointing" in lab:
        return "inspection / search / matching"

    if "handover" in lab or "handıver" in lab:
        return "object handover"

    if "tray" in lab:
        return "tray handling"

    if "merge" in lab or "mering" in lab or "merving" in lab:
        return "merging pieces"

    if "co_build" in lab or "cobuild" in lab or "building_subpiece" in lab or "build" in lab:
        return "building / co-building"

    if "pick" in lab or "putting_down" in lab or "presenting" in lab or "deliver" in lab or "placing" in lab:
        return "object manipulation"

    return "other / needs review"


rows = []

for group, files in GROUP_FILES.items():
    for source, path in files.items():

        if not os.path.exists(path):
            print(f"Missing file: Group {group} {source} -> {path}")
            continue

        df = pd.read_csv(path, low_memory=False)

        total_rows = len(df)

        for col in LABEL_COLS:
            if col not in df.columns:
                continue

            vc = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
            )

            for label, count in vc.items():
                if label == "" or label.lower() == "nan":
                    continue

                rows.append({
                    "group": group,
                    "source": source,
                    "tier_column": col,
                    "tier_type": TIER_TYPE.get(col, "unknown"),
                    "label": label,
                    "broad_category": broad_category(label),
                    "rows": int(count),
                    "percent_of_file": count / total_rows * 100,
                    "file_rows": total_rows,
                    "file_path": path
                })


inventory = pd.DataFrame(rows)

# Sort by group/source/tier/label
inventory = inventory.sort_values(
    ["group", "source", "tier_type", "tier_column", "label"]
).reset_index(drop=True)

# Unique label summary
unique_summary = (
    inventory
    .groupby(["label", "broad_category"], as_index=False)
    .agg(
        total_rows=("rows", "sum"),
        appearances=("label", "size"),
        groups=("group", lambda x: ", ".join(map(str, sorted(set(x))))),
        sources=("source", lambda x: ", ".join(sorted(set(x)))),
        tier_types=("tier_type", lambda x: ", ".join(sorted(set(x)))),
        tier_columns=("tier_column", lambda x: ", ".join(sorted(set(x))))
    )
    .sort_values(["broad_category", "total_rows"], ascending=[True, False])
    .reset_index(drop=True)
)

# Broad category summary
category_summary = (
    inventory
    .groupby(["broad_category", "tier_type"], as_index=False)
    .agg(
        total_rows=("rows", "sum"),
        unique_labels=("label", "nunique"),
        appearances=("label", "size"),
        groups=("group", lambda x: ", ".join(map(str, sorted(set(x)))))
    )
    .sort_values(["broad_category", "tier_type"])
    .reset_index(drop=True)
)

inventory_path = os.path.join(OUT_DIR, "full_label_inventory.csv")
unique_path = os.path.join(OUT_DIR, "unique_label_list.csv")
category_path = os.path.join(OUT_DIR, "broad_category_summary.csv")

inventory.to_csv(inventory_path, index=False)
unique_summary.to_csv(unique_path, index=False)
category_summary.to_csv(category_path, index=False)

print("================================================")
print("LABEL INVENTORY COMPLETE")
print("================================================")
print("Full inventory saved to:")
print(inventory_path)

print("\nUnique label list saved to:")
print(unique_path)

print("\nBroad category summary saved to:")
print(category_path)

print("\nTotal unique raw labels:")
print(unique_summary["label"].nunique())

print("\nBroad category summary:")
display(category_summary)

print("\nTop 50 labels by total rows:")
display(
    unique_summary
    .sort_values("total_rows", ascending=False)
    .head(50)
)

print("\nLabels needing review:")
display(
    unique_summary[
        unique_summary["broad_category"] == "other / needs review"
    ].sort_values("total_rows", ascending=False)
)


# --- CELL 2 (code cell #3) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 4 (code cell #4) ---
import pandas as pd, numpy as np, os

# ---- the only two lines you might edit ----
BASE   = "/content/drive/MyDrive/thesis/data/group_1/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_1/openearable_merged"
# -------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)
PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

# value columns per sensor (the trailing all-zero flag column is dropped)
SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}
GRID_US     = 20_000   # 50 Hz target grid
TOL_FAST_US = 12_000   # match tolerance for 50 Hz streams (~half the 20 ms period)
TOL_SLOW_US = 40_000   # wider tolerance for the ~32 Hz skin_temp

def load_sensor(path, names):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :1 + len(names)]            # keep timestamp + values, drop flag
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)
    return (df.drop_duplicates("timestamp_us")
              .sort_values("timestamp_us")
              .reset_index(drop=True))

def merge_participant(participant):
    pdir = os.path.join(BASE, participant)
    streams = {k: load_sensor(os.path.join(pdir, f"{participant}_{k}.csv"), v)
               for k, v in SCHEMA.items()}

    # session window from the acc/gyro/mag trio that start together
    # (this also clips the bone_acc warm-up samples recorded minutes early)
    imu = pd.concat([streams[k]["timestamp_us"] for k in ("acc", "gyro", "mgnt")])
    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({"timestamp_us": np.arange(t0, t1 + GRID_US, GRID_US, dtype=np.int64)})
    out = grid.copy()
    for name, df in streams.items():
        df = df[(df.timestamp_us >= t0 - GRID_US) & (df.timestamp_us <= t1 + GRID_US)]
        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US
        m = pd.merge_asof(grid, df, on="timestamp_us", direction="nearest", tolerance=tol)
        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(1, "datetime_utc", pd.to_datetime(out["timestamp_us"], unit="us", utc=True))
    return out

for p in PARTICIPANTS:
    out  = merge_participant(p)
    path = os.path.join(OUTDIR, f"{p}_openearable_merged_50hz.csv")
    out.to_csv(path, index=False)
    miss = out.iloc[:, 2:].isna().mean().mean() * 100
    print(f"{p}: {len(out):>6} rows, {out.shape[1]} cols, avg-missing {miss:.1f}%  ->  {path}")


# --- CELL 6 (code cell #5) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_1/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/openearable_merged"
LOCAL_OUT = "/content/openearable_merged"
# ----------------
os.makedirs(LOCAL_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]
PREFIX = {"Participant1": "p1", "Participant2": "p2", "Participant3": "p3"}
SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}
GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000   # 50 Hz grid; match tolerances

def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)
    return (df.drop_duplicates("timestamp_us")
              .sort_values("timestamp_us").reset_index(drop=True))

# load every participant's 8 raw streams + find each one's session window
all_streams, windows = {}, {}
for p in PARTICIPANTS:
    streams = {k: load_sensor(os.path.join(BASE, p, f"{p}_{k}.csv"), v)
               for k, v in SCHEMA.items()}
    all_streams[p] = streams
    imu = pd.concat([streams[k]["timestamp_us"] for k in ("acc", "gyro", "mgnt")])
    windows[p] = (int(imu.min()), int(imu.max()))

# shared grid = overlap window (latest start -> earliest end) so all three are present
start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())
grid  = pd.DataFrame({"timestamp_us": np.arange(start, end + GRID_US, GRID_US, dtype=np.int64)})

wide = grid.copy()
wide.insert(1, "datetime_utc", pd.to_datetime(wide["timestamp_us"], unit="us", utc=True))
for p in PARTICIPANTS:
    pre = PREFIX[p]
    for name, df in all_streams[p].items():
        df = df[(df.timestamp_us >= start - GRID_US) & (df.timestamp_us <= end + GRID_US)]
        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST
        m = pd.merge_asof(grid, df, on="timestamp_us", direction="nearest", tolerance=tol)
        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values
for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA          # filled later from ELAN

# write locally, then copy to Drive
local_path = os.path.join(LOCAL_OUT, "group_1_openearable_merged_50hz.csv")
wide.to_csv(local_path, index=False)
print("shape:", wide.shape, "| overlap",
      f"{(end-start)/1e6:.1f}s ->", local_path)

os.makedirs(DRIVE_OUT, exist_ok=True)
dst = os.path.join(DRIVE_OUT, "group_1_openearable_merged_50hz.csv")
shutil.copy2(local_path, dst)
print("copied ->", dst)


# --- CELL 8 (code cell #6) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_1/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/xsens_merged"
LOCAL_OUT = "/content/xsens_merged"
# ----------------
os.makedirs(LOCAL_OUT, exist_ok=True)

PREFIX = {"Participant1": "p1", "Participant2": "p2", "Participant3": "p3"}
SENSOR_COLS = ["Euler_X","Euler_Y","Euler_Z","Acc_X","Acc_Y","Acc_Z","Gyr_X","Gyr_Y","Gyr_Z"]

def load(p):
    d = pd.read_csv(os.path.join(BASE, f"{p}.csv"))
    d = d.loc[:, [c for c in d.columns if c.strip() != ""]]   # drop trailing empty col
    d.columns = [c.strip() for c in d.columns]
    for c in ["SampleTimeFine", "PacketCounter"] + SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")
    return d

raw = {p: load(p) for p in PREFIX}

# master grid = P1's clean SampleTimeFine (the superset that contains every valid tick)
p1 = raw["Participant1"].dropna(subset=["SampleTimeFine"])
p1_ticks = set(p1["SampleTimeFine"].tolist())

# common span across all three (uses only ticks that exist on P1's grid -> drops P3's corrupted row)
spans = []
for d in raw.values():
    valid = d[d["SampleTimeFine"].isin(p1_ticks)]
    spans.append((int(valid["SampleTimeFine"].min()), int(valid["SampleTimeFine"].max())))
start = max(s[0] for s in spans)
end   = min(s[1] for s in spans)

master = (p1[(p1["SampleTimeFine"] >= start) & (p1["SampleTimeFine"] <= end)]
          [["SampleTimeFine"]].sort_values("SampleTimeFine").reset_index(drop=True))
master["time_s"] = ((master["SampleTimeFine"].astype(np.int64) - start) / 1e6).round(4)

wide = master.copy()
for p, d in raw.items():
    pre = PREFIX[p]
    sub = (d[["SampleTimeFine"] + SENSOR_COLS]
           .dropna(subset=["SampleTimeFine"]).drop_duplicates("SampleTimeFine")
           .rename(columns={c: f"{pre}_{c.lower()}" for c in SENSOR_COLS}))
    wide = wide.merge(sub, on="SampleTimeFine", how="left")
for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA          # filled later from ELAN

local_path = os.path.join(LOCAL_OUT, "group_1_xsens_merged_30hz.csv")
wide.to_csv(local_path, index=False)
print("shape:", wide.shape, "| dur:", round(wide['time_s'].iloc[-1], 1), "s ->", local_path)

os.makedirs(DRIVE_OUT, exist_ok=True)
dst = os.path.join(DRIVE_OUT, "group_1_xsens_merged_30hz.csv")
shutil.copy2(local_path, dst)
print("copied ->", dst)


# --- CELL 10 (code cell #7) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1.csv"   # adjust if it lives elsewhere
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_with_individual_build.csv"
LOCAL_OUT = "/content/Group_1_with_individual_build.csv"
# ---- config you might edit ----
CUTOFF      = {"Arda": 23*60+40, "Bas": 19*60+20, "Rachel": 22*60+40}   # P1, P2, P3 (seconds)
START       = 0.0          # individual_build begins at video 0
MIN_GAP_S   = 0.0          # set e.g. 0.5 to drop slivers shorter than this
# --------------------------------
INVOLVING = {
    "Arda":   ["Arda", "Arda_Rachel", "Bas_Arda", "Whole_Group"],
    "Bas":    ["Bas", "Bas_Rachel", "Bas_Arda", "Whole_Group"],
    "Rachel": ["Rachel", "Bas_Rachel", "Arda_Rachel", "Whole_Group"],
}
COLS = ["tier","blank","begin_hms","begin_s","end_hms","end_s","dur_hms","dur_s","label"]

df = pd.read_csv(ELAN_PATH, header=None, names=COLS, dtype={"blank": str})

def merge_iv(ivs):
    ivs = sorted(ivs); out = []
    for s, e in ivs:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out

def gaps(busy, start, end):
    busy = merge_iv([[max(s, start), min(e, end)] for s, e in busy if e > start and s < end])
    g = []; cur = start
    for s, e in busy:
        if s > cur: g.append((cur, s))
        cur = max(cur, e)
    if cur < end: g.append((cur, end))
    return g

def hms(x):
    h = int(x // 3600); m = int((x % 3600) // 60); s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

new_rows = []
for person, tiers in INVOLVING.items():
    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()
    for s, e in gaps(busy, START, CUTOFF[person]):
        d = e - s
        if d < MIN_GAP_S or d <= 0:
            continue
        new_rows.append({"tier": person, "blank": "", "begin_hms": hms(s), "begin_s": round(s, 3),
                         "end_hms": hms(e), "end_s": round(e, 3), "dur_hms": hms(d), "dur_s": round(d, 3),
                         "label": "individual_build"})

both = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).sort_values(["tier", "begin_s"])
fmt = lambda x: str(round(float(x), 3))
lines = [f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
         for r in both.itertuples()]
open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")
print(f"{len(df)} -> {len(both)} annotations ({len(new_rows)} individual_build added)")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)
print("copied ->", DRIVE_OUT)


# --- CELL 12 (code cell #8) ---
import pandas as pd, shutil

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_1_individual_build_renamed.csv"
# ----------------
NAME = {"Arda": "Participant1", "Bas": "Participant2", "Rachel": "Participant3"}
COLS = ["tier","blank","begin_hms","begin_s","end_hms","end_s","dur_hms","dur_s","label"]

def rename_tier(t):
    parts = t.split("_")
    if all(p in NAME for p in parts):                  # individual or dyad tier built from names
        mapped = sorted((NAME[p] for p in parts),
                        key=lambda x: int(x.replace("Participant", "")))
        return "_".join(mapped)                        # e.g. Bas_Arda -> Participant1_Participant2
    return t                                           # Whole_Group left unchanged

df = pd.read_csv(IN_PATH, header=None, names=COLS, dtype={"blank": str})
df["tier"] = df["tier"].map(rename_tier)
df = df.sort_values(["tier", "begin_s"])

fmt = lambda x: str(round(float(x), 3))
lines = [f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
         f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
         for r in df.itertuples()]
open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)
print("tiers now:", sorted(df["tier"].unique()))
print("copied ->", DRIVE_OUT)


# --- CELL 14 (code cell #9) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_1/openearable_merged/group_1_openearable_merged_50hz.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/openearable_labeled/group_1_openearable_labeled.csv"
LOCAL_OUT = "/content/group_1_openearable_labeled.csv"
# ---- sync anchor ----
VIDEO_START_US = 1776759287000000   # 2026-04-21 08:14:47 UTC  (video started 10:14:47 CEST)
# -------------------------
ECOLS = ["tier","blank","begin_hms","begin_s","end_hms","end_s","dur_hms","dur_s","label"]
TIERS = ["Participant1","Participant2","Participant3",
         "Participant1_Participant2","Participant1_Participant3",
         "Participant2_Participant3","Whole_Group"]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe   = pd.read_csv(OE_PATH)

oe["video_time_s"] = (oe["timestamp_us"].astype(np.int64) - VIDEO_START_US) / 1e6
oe = oe.drop(columns=[c for c in ["label_p1","label_p2","label_p3"] if c in oe.columns])

# put video_time_s right after datetime_utc
cols = list(oe.columns); cols.remove("video_time_s")
cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
oe = oe[cols]

for t in TIERS:
    sub = (elan[elan.tier==t][["begin_s","end_s","label"]]
           .sort_values("begin_s").reset_index(drop=True)
           .rename(columns={"begin_s":"video_time_s","label":"_lab","end_s":"_end"}))
    col = f"label_{t}"
    if len(sub) == 0:
        oe[col] = pd.NA; continue
    m = pd.merge_asof(oe[["video_time_s"]].reset_index(),
                      sub, on="video_time_s", direction="backward")
    valid = (m["video_time_s"] < m["_end"]) & m["_lab"].notna()
    oe[col] = m["_lab"].where(valid, other=pd.NA).values

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
print("shape:", oe.shape)
for t in TIERS:
    n = oe[f"label_{t}"].notna().sum()
    print(f"  label_{t}: {n} rows ({n/len(oe)*100:.1f}%)")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)
print("copied ->", DRIVE_OUT)


# --- CELL 16 (code cell #10) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_1/xsens_merged/group_1_xsens_merged_30hz.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1/xsens_labeled/group_1_xsens_labeled.csv"
LOCAL_OUT = "/content/group_1_xsens_labeled.csv"
# ----------------

# ---- timing ----
# Group 1:
# Video start: 10:14:47
# XSens start: 10:14:22
# XSens starts 25 sec BEFORE video
XSENS_START_OFFSET_FROM_VIDEO_S = -25.0
# ----------------

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

# ============================================================
# Load files
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

xsens = pd.read_csv(XSENS_PATH)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")

# ============================================================
# Create video-relative time for XSens
# ============================================================

# XSens time_s starts from XSens recording start.
# Since XSens started 25 sec before video:
# video_time_s = time_s - 25
xsens["video_time_s"] = xsens["time_s"] + XSENS_START_OFFSET_FROM_VIDEO_S

# remove old placeholder label columns if they exist
xsens = xsens.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in xsens.columns],
    errors="ignore"
)

# put video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# ============================================================
# Add labels from ELAN to XSens timeline
# ============================================================

for t in TIERS:
    sub = (
        elan[elan.tier == t][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "label": "_lab",
                "end_s": "_end"
            }
        )
    )

    col = f"label_{t}"

    if len(sub) == 0:
        xsens[col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[col] = m["_lab"].where(valid, other=pd.NA).values

# ============================================================
# Save
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

xsens.to_csv(LOCAL_OUT, index=False)

print("shape:", xsens.shape)

print("\nXSens time range:")
print("time_s start      :", xsens["time_s"].min())
print("time_s end        :", xsens["time_s"].max())
print("video_time_s start:", xsens["video_time_s"].min())
print("video_time_s end  :", xsens["video_time_s"].max())

print("\nLabel coverage:")
for t in TIERS:
    n = xsens[f"label_{t}"].notna().sum()
    print(f"  label_{t}: {n} rows ({n / len(xsens) * 100:.1f}%)")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\ncopied ->", DRIVE_OUT)


# --- CELL 17 (code cell #11) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_label_segments(df, time_col, label_col):
    """
    Finds continuous segments for a label column.
    Returns: [(start_time, end_time, label_name), ...]
    """
    segments = []

    if label_col not in df.columns:
        return segments

    labels = df[label_col].fillna("").astype(str)

    in_segment = False
    start_time = None
    current_label = None

    for i in range(len(df)):
        label = labels.iloc[i]
        time = df[time_col].iloc[i]

        if label != "" and not in_segment:
            in_segment = True
            start_time = time
            current_label = label

        elif label != "" and in_segment and label != current_label:
            end_time = df[time_col].iloc[i - 1]
            segments.append((start_time, end_time, current_label))

            start_time = time
            current_label = label

        elif label == "" and in_segment:
            end_time = df[time_col].iloc[i - 1]
            segments.append((start_time, end_time, current_label))

            in_segment = False
            start_time = None
            current_label = None

    if in_segment:
        end_time = df[time_col].iloc[-1]
        segments.append((start_time, end_time, current_label))

    return segments


def plot_three_users_with_all_labels(
    csv_path,
    time_col="video_time_s",
    signal_type="acc_mag",
    users=(1, 2, 3),
    start_time=None,
    end_time=None,
    only_label=None,
    figsize=(20, 10),
    alpha=0.25
):
    """
    Plot 3 users' signals and show individual, pair, and whole-group labels.

    Parameters
    ----------
    csv_path : str
        Path to labeled CSV.

    time_col : str
        Time column, usually "video_time_s".

    signal_type : str
        Options:
        "acc_mag", "gyro_mag", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"

    only_label : str or None
        If None, shows all labels.
        If a label name is given, only that label is highlighted.
        Example: only_label="synchronizaiton_move"
    """

    df = pd.read_csv(csv_path, low_memory=False)

    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found in dataframe.")

    # Create magnitude columns if needed
    for u in users:
        if signal_type == "acc_mag":
            x, y, z = f"p{u}_acc_x", f"p{u}_acc_y", f"p{u}_acc_z"
            df[f"p{u}_acc_mag"] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

        elif signal_type == "gyro_mag":
            x, y, z = f"p{u}_gyro_x", f"p{u}_gyro_y", f"p{u}_gyro_z"
            df[f"p{u}_gyro_mag"] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    # Optional time zoom
    if start_time is not None:
        df = df[df[time_col] >= start_time]

    if end_time is not None:
        df = df[df[time_col] <= end_time]

    df = df.reset_index(drop=True)

    # All label columns
    label_cols = [
        "label_Participant1",
        "label_Participant2",
        "label_Participant3",
        "label_Participant1_Participant2",
        "label_Participant1_Participant3",
        "label_Participant2_Participant3",
        "label_Whole_Group"
    ]

    label_cols = [col for col in label_cols if col in df.columns]

    # Collect unique labels
    all_labels = []

    for col in label_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        all_labels.extend(values)

    unique_labels = sorted(set(all_labels))

    if only_label is not None:
        unique_labels = [only_label]

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    label_colors = {
        label: cmap(i) for i, label in enumerate(unique_labels)
    }

    fig, axes = plt.subplots(len(users), 1, figsize=figsize, sharex=True)

    if len(users) == 1:
        axes = [axes]

    for ax, u in zip(axes, users):

        if signal_type in ["acc_mag", "gyro_mag"]:
            signal_col = f"p{u}_{signal_type}"
        else:
            signal_col = f"p{u}_{signal_type}"

        if signal_col not in df.columns:
            raise ValueError(f"{signal_col} not found in dataframe.")

        ax.plot(
            df[time_col],
            df[signal_col],
            linewidth=0.8,
            label=f"Participant {u} - {signal_col}"
        )

        # Plot labels from all label columns
        for label_col in label_cols:
            segments = get_label_segments(df, time_col, label_col)

            for start, end, label in segments:

                if only_label is not None and label != only_label:
                    continue

                if label not in label_colors:
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors[label],
                    alpha=alpha
                )

                mid = (start + end) / 2
                y_top = ax.get_ylim()[1]

                ax.text(
                    mid,
                    y_top,
                    f"{label_col}: {label}",
                    rotation=90,
                    fontsize=7,
                    ha="center",
                    va="top"
                )

        ax.set_title(f"Participant {u} - {signal_col}")
        ax.set_ylabel(signal_col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels"
    )

    plt.tight_layout()
    plt.show()


# --- CELL 18 (code cell #12) ---
plot_three_users_with_all_labels(
    csv_path="group_1_openearable_labeled.csv",
    signal_type="gyro_mag",
    only_label="synchronizaiton_move"
)


# --- CELL 19 (code cell #13) ---
# ============================================================
# VISUALIZE SHIFTED SYNC LABEL FOR ALL 3 PARTICIPANTS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"

CSV_PATH = os.path.join(
    DRIVE_OUT,
    "group_1_openearable_labeled_shifted_by_last10_peak.csv"
)

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"   # exact spelling in your file

USERS = [1, 2, 3]

SMOOTH_WINDOW = 25
ZOOM_MARGIN = 25


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def acc_mag(df, user):
    x = f"p{user}_acc_x"
    y = f"p{user}_acc_y"
    z = f"p{user}_acc_z"

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def moving_average(signal, window=25):
    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def get_label_segments(df, time_col, label_col):
    """
    Returns continuous label regions:
    [(start_time, end_time, label_name), ...]
    """

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def find_sync_segment(df, time_col, label_col, sync_label):
    segments = get_label_segments(df, time_col, label_col)

    sync_segments = [
        seg for seg in segments
        if seg[2] == sync_label
    ]

    if len(sync_segments) == 0:
        raise ValueError(f"Could not find '{sync_label}' in '{label_col}'")

    # If more than one exists, use the longest one
    sync_segments = sorted(
        sync_segments,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return sync_segments[0]


def plot_shifted_sync_all_participants(
    csv_path,
    time_col="video_time_s",
    sync_label_col="label_Whole_Group",
    sync_label="synchronizaiton_move",
    users=[1, 2, 3],
    smooth_window=25,
    zoom_margin=25
):
    df = pd.read_csv(csv_path, low_memory=False)

    times = df[time_col].values

    # Find shifted synchronization label segment
    sync_start, sync_end, _ = find_sync_segment(
        df,
        time_col,
        sync_label_col,
        sync_label
    )

    sync_middle = (sync_start + sync_end) / 2

    print("================================================")
    print("SHIFTED SYNC LABEL LOCATION")
    print("================================================")
    print(f"Sync label start  : {sync_start:.3f} s")
    print(f"Sync label end    : {sync_end:.3f} s")
    print(f"Sync label middle : {sync_middle:.3f} s")
    print(f"Duration          : {sync_end - sync_start:.3f} s")
    print("================================================")

    # Compute individual acc magnitudes
    acc_signals = {}

    for user in users:
        raw = acc_mag(df, user)
        smooth = moving_average(raw, smooth_window)

        acc_signals[user] = {
            "raw": raw,
            "smooth": smooth
        }

    # Combined signal
    combined_raw = np.mean(
        [acc_signals[user]["raw"] for user in users],
        axis=0
    )

    combined_smooth = moving_average(combined_raw, smooth_window)

    # Zoom window
    left = sync_start - zoom_margin
    right = sync_end + zoom_margin

    mask = (times >= left) & (times <= right)

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # --------------------------------------------------------
    # Combined signal
    # --------------------------------------------------------
    axes[0].plot(
        times[mask],
        combined_smooth[mask],
        linewidth=1.0,
        label="Combined acc magnitude smoothed"
    )

    axes[0].axvspan(
        sync_start,
        sync_end,
        alpha=0.25,
        label="Shifted synchronization label"
    )

    axes[0].axvline(
        sync_middle,
        linestyle="--",
        linewidth=1.5,
        label="Sync label middle"
    )

    axes[0].set_title("Combined acceleration magnitude")
    axes[0].set_ylabel("Combined acc mag")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # --------------------------------------------------------
    # Participant 1, 2, 3
    # --------------------------------------------------------
    for i, user in enumerate(users, start=1):

        axes[i].plot(
            times[mask],
            acc_signals[user]["smooth"][mask],
            linewidth=1.0,
            label=f"Participant {user} acc magnitude smoothed"
        )

        axes[i].axvspan(
            sync_start,
            sync_end,
            alpha=0.25,
            label="Shifted synchronization label"
        )

        axes[i].axvline(
            sync_middle,
            linestyle="--",
            linewidth=1.5,
            label="Sync label middle"
        )

        axes[i].set_title(f"Participant {user}")
        axes[i].set_ylabel(f"p{user} acc mag")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_shifted_checked = plot_shifted_sync_all_participants(
    csv_path=CSV_PATH,
    time_col=TIME_COL,
    sync_label_col=SYNC_LABEL_COL,
    sync_label=SYNC_LABEL,
    users=USERS,
    smooth_window=SMOOTH_WINDOW,
    zoom_margin=ZOOM_MARGIN
)


# --- CELL 20 (code cell #14) ---
# ============================================================
# GROUP 1 XSENS - INSPECT SYNC LABEL FOR ALL 3 PARTICIPANTS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"

CSV_PATH = os.path.join(
    DRIVE_OUT,
    "xsens_labeled",
    "group_1_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"   # exact spelling from Group 1

USERS = [1, 2, 3]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

SMOOTH_WINDOW = 15
ZOOM_MARGIN = 25

ROBUST_CLIP = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window=15):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_clip_for_plot(signal, lower_p=1, upper_p=99):
    """
    Only clips for visualization.
    Does not change the CSV or dataframe.
    """
    signal = np.asarray(signal, dtype=float)
    finite = signal[np.isfinite(signal)]

    if len(finite) == 0:
        return signal

    low = np.percentile(finite, lower_p)
    high = np.percentile(finite, upper_p)

    return np.clip(signal, low, high)


def xsens_acc_mag(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def xsens_gyro_mag(df, user):
    return np.sqrt(
        df[f"p{user}_gyr_x"]**2 +
        df[f"p{user}_gyr_y"]**2 +
        df[f"p{user}_gyr_z"]**2
    )


def get_signal(df, user, signal_type="acc_mag"):
    if signal_type == "acc_mag":
        return xsens_acc_mag(df, user)

    elif signal_type == "gyro_mag":
        return xsens_gyro_mag(df, user)

    else:
        raise ValueError("SIGNAL_TYPE must be 'acc_mag' or 'gyro_mag'")


def get_label_segments(df, time_col, label_col):
    """
    Returns continuous label regions:
    [(start_time, end_time, label_name), ...]
    """

    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in dataframe.")

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def find_sync_segment(df, time_col, label_col, sync_label):
    segments = get_label_segments(df, time_col, label_col)

    sync_segments = [
        seg for seg in segments
        if seg[2] == sync_label
    ]

    if len(sync_segments) == 0:
        print("Available Whole_Group labels:")
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find '{sync_label}' in '{label_col}'")

    # if multiple exist, use longest
    sync_segments = sorted(
        sync_segments,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return sync_segments[0]


def plot_group1_xsens_sync(
    csv_path,
    time_col="video_time_s",
    sync_label_col="label_Whole_Group",
    sync_label="synchronizaiton_move",
    users=[1, 2, 3],
    signal_type="acc_mag",
    smooth_window=15,
    zoom_margin=25,
    robust_clip=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    times = df[time_col].values

    # Find synchronization label segment
    sync_start, sync_end, found_label = find_sync_segment(
        df,
        time_col,
        sync_label_col,
        sync_label
    )

    sync_middle = (sync_start + sync_end) / 2

    print("================================================")
    print("GROUP 1 XSENS SYNC LABEL LOCATION")
    print("================================================")
    print(f"Sync label       : {found_label}")
    print(f"Sync label start : {sync_start:.3f} s")
    print(f"Sync label end   : {sync_end:.3f} s")
    print(f"Sync middle      : {sync_middle:.3f} s")
    print(f"Duration         : {sync_end - sync_start:.3f} s")
    print("================================================")

    # Compute signals
    signals = {}

    for user in users:
        sig = get_signal(df, user, signal_type)
        sig = moving_average(sig, smooth_window)

        if robust_clip:
            sig = robust_clip_for_plot(
                sig,
                LOWER_PERCENTILE,
                UPPER_PERCENTILE
            )

        signals[user] = sig

    combined = np.mean(
        [signals[user] for user in users],
        axis=0
    )

    # Zoom window
    left = sync_start - zoom_margin
    right = sync_end + zoom_margin

    mask = (times >= left) & (times <= right)

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # Combined signal
    axes[0].plot(
        times[mask],
        combined[mask],
        linewidth=1.0,
        label=f"Combined XSens {signal_type}"
    )

    axes[0].axvspan(
        sync_start,
        sync_end,
        alpha=0.25,
        label=sync_label
    )

    axes[0].axvline(
        sync_middle,
        linestyle="--",
        linewidth=1.5,
        label="sync label middle"
    )

    axes[0].set_title(f"Group 1 XSens - Combined {signal_type}")
    axes[0].set_ylabel(signal_type)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Participant signals
    for i, user in enumerate(users, start=1):

        axes[i].plot(
            times[mask],
            signals[user][mask],
            linewidth=1.0,
            label=f"Participant {user} XSens {signal_type}"
        )

        axes[i].axvspan(
            sync_start,
            sync_end,
            alpha=0.25,
            label=sync_label
        )

        axes[i].axvline(
            sync_middle,
            linestyle="--",
            linewidth=1.5,
            label="sync label middle"
        )

        axes[i].set_title(f"Group 1 XSens - Participant {user}")
        axes[i].set_ylabel(f"p{user}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_group1_xsens_sync = plot_group1_xsens_sync(
    csv_path=CSV_PATH,
    time_col=TIME_COL,
    sync_label_col=SYNC_LABEL_COL,
    sync_label=SYNC_LABEL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW,
    zoom_margin=ZOOM_MARGIN,
    robust_clip=ROBUST_CLIP
)


# --- CELL 21 (code cell #15) ---
# ============================================================
# FULL VISUALIZATION OF ALL LABELS ON 3 PARTICIPANT SIGNALS
# using latest shifted output file
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"

CSV_PATH = os.path.join(
    DRIVE_OUT,
    "group_1_openearable_labeled_shifted_by_last10_peak.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

SIGNAL_TYPE = "acc_mag"   # options: "acc_mag", "gyro_mag", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"

SMOOTH_WINDOW = 25        # set to 1 if you do not want smoothing

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.18
TEXT_EVERY_N_SEGMENTS = 1   # increase to 2 or 3 if text becomes too crowded


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def magnitude(df, user, signal_type):
    if signal_type == "acc_mag":
        x = f"p{user}_acc_x"
        y = f"p{user}_acc_y"
        z = f"p{user}_acc_z"
        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    elif signal_type == "gyro_mag":
        x = f"p{user}_gyro_x"
        y = f"p{user}_gyro_y"
        z = f"p{user}_gyro_z"
        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    else:
        col = f"p{user}_{signal_type}"
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        return df[col].values


def moving_average(signal, window=25):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def get_label_segments(df, time_col, label_col):
    """
    Finds continuous labeled regions.

    Returns:
    [(start_time, end_time, label_name), ...]
    """

    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []

    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def plot_full_three_participants_all_labels(
    csv_path,
    time_col="video_time_s",
    users=[1, 2, 3],
    signal_type="acc_mag",
    label_cols=None,
    smooth_window=25,
    figsize=(24, 12),
    label_alpha=0.18,
    text_every_n_segments=1
):
    df = pd.read_csv(csv_path, low_memory=False)

    if label_cols is None:
        label_cols = [
            "label_Participant1",
            "label_Participant2",
            "label_Participant3",
            "label_Participant1_Participant2",
            "label_Participant1_Participant3",
            "label_Participant2_Participant3",
            "label_Whole_Group"
        ]

    existing_label_cols = [
        col for col in label_cols
        if col in df.columns
    ]

    if len(existing_label_cols) == 0:
        raise ValueError("No label columns found.")

    times = df[time_col].values

    # --------------------------------------------------------
    # Collect all unique labels
    # --------------------------------------------------------
    all_labels = []

    for col in existing_label_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]
        all_labels.extend(values)

    unique_labels = sorted(set(all_labels))

    print("================================================")
    print("LABEL SUMMARY")
    print("================================================")
    print(f"File: {csv_path}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print("Label columns used:")
    for col in existing_label_cols:
        print(" -", col)
    print("================================================")

    # --------------------------------------------------------
    # Color map
    # --------------------------------------------------------
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        len(users),
        1,
        figsize=figsize,
        sharex=True
    )

    if len(users) == 1:
        axes = [axes]

    for ax, user in zip(axes, users):

        sig = magnitude(df, user, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        # Draw all label segments
        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, time_col, label_col)

            for start, end, label in segments:

                if label == "":
                    continue

                color = label_colors.get(label, "gray")

                ax.axvspan(
                    start,
                    end,
                    color=color,
                    alpha=label_alpha
                )

                # Write text label
                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=6,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user} {signal_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")

    # --------------------------------------------------------
    # Legend outside plot
    # --------------------------------------------------------
    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_full_labels = plot_full_three_participants_all_labels(
    csv_path=CSV_PATH,
    time_col=TIME_COL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    label_cols=LABEL_COLS,
    smooth_window=SMOOTH_WINDOW,
    figsize=FIGSIZE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS
)


# --- CELL 22 (code cell #16) ---
# ============================================================
# GROUP 1 XSENS
# FULL VISUALIZATION OF ALL LABELS ON 3 PARTICIPANT SIGNALS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"

CSV_PATH = os.path.join(
    DRIVE_OUT,
    "xsens_labeled",
    "group_1_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

# options:
# "acc_mag", "gyro_mag",
# "acc_x", "acc_y", "acc_z",
# "gyr_x", "gyr_y", "gyr_z",
# "euler_x", "euler_y", "euler_z"
SIGNAL_TYPE = "acc_mag"

SMOOTH_WINDOW = 15

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.16
TEXT_EVERY_N_SEGMENTS = 3

# Important for XSens because sometimes there are extreme outliers
ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window=15):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def xsens_signal(df, user, signal_type):
    """
    XSens columns are like:
    p1_acc_x, p1_acc_y, p1_acc_z
    p1_gyr_x, p1_gyr_y, p1_gyr_z
    p1_euler_x, p1_euler_y, p1_euler_z
    """

    if signal_type == "acc_mag":
        x = f"p{user}_acc_x"
        y = f"p{user}_acc_y"
        z = f"p{user}_acc_z"

        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    elif signal_type == "gyro_mag":
        x = f"p{user}_gyr_x"
        y = f"p{user}_gyr_y"
        z = f"p{user}_gyr_z"

        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    else:
        col = f"p{user}_{signal_type}"

        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

        return df[col].values


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]

    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)

    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_label_segments(df, time_col, label_col):
    """
    Finds continuous labeled regions.

    Returns:
    [(start_time, end_time, label_name), ...]
    """

    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []

    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def print_signal_diagnostics(df, users, signal_type):
    print("================================================")
    print("XSENS SIGNAL DIAGNOSTICS")
    print("================================================")

    for user in users:
        sig = xsens_signal(df, user, signal_type)
        clean = sig[np.isfinite(sig)]

        print(f"Participant {user} - {signal_type}")
        print(f"  min   : {np.min(clean):.6g}")
        print(f"  p1    : {np.percentile(clean, 1):.6g}")
        print(f"  median: {np.median(clean):.6g}")
        print(f"  p99   : {np.percentile(clean, 99):.6g}")
        print(f"  max   : {np.max(clean):.6g}")
        print("------------------------------------------------")


def plot_full_xsens_three_participants_all_labels(
    csv_path,
    time_col="video_time_s",
    users=[1, 2, 3],
    signal_type="acc_mag",
    label_cols=None,
    smooth_window=15,
    figsize=(24, 12),
    label_alpha=0.16,
    text_every_n_segments=3,
    robust_ylim_enabled=True,
    lower_percentile=1,
    upper_percentile=99
):
    df = pd.read_csv(csv_path, low_memory=False)

    if label_cols is None:
        label_cols = [
            "label_Participant1",
            "label_Participant2",
            "label_Participant3",
            "label_Participant1_Participant2",
            "label_Participant1_Participant3",
            "label_Participant2_Participant3",
            "label_Whole_Group"
        ]

    existing_label_cols = [
        col for col in label_cols
        if col in df.columns
    ]

    if len(existing_label_cols) == 0:
        raise ValueError("No label columns found.")

    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found in file.")

    times = df[time_col].values

    print_signal_diagnostics(df, users, signal_type)

    # --------------------------------------------------------
    # Collect all unique labels
    # --------------------------------------------------------

    all_labels = []

    for col in existing_label_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]
        all_labels.extend(values)

    unique_labels = sorted(set(all_labels))

    print("================================================")
    print("LABEL SUMMARY")
    print("================================================")
    print(f"File: {csv_path}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print("Label columns used:")
    for col in existing_label_cols:
        print(" -", col)
    print("================================================")

    # --------------------------------------------------------
    # Color map
    # --------------------------------------------------------

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        len(users),
        1,
        figsize=figsize,
        sharex=True
    )

    if len(users) == 1:
        axes = [axes]

    for ax, user in zip(axes, users):

        sig = xsens_signal(df, user, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - XSens {signal_type}"
        )

        if robust_ylim_enabled:
            ylim = robust_ylim(
                sig,
                lower_p=lower_percentile,
                upper_p=upper_percentile
            )

            if ylim is not None:
                ax.set_ylim(ylim)

        # Draw all label segments
        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, time_col, label_col)

            for start, end, label in segments:

                if label == "":
                    continue

                color = label_colors.get(label, "gray")

                ax.axvspan(
                    start,
                    end,
                    color=color,
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"Group 1 XSens - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user} {signal_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    # --------------------------------------------------------
    # Legend outside plot
    # --------------------------------------------------------

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_group1_xsens_full_labels = plot_full_xsens_three_participants_all_labels(
    csv_path=CSV_PATH,
    time_col=TIME_COL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    label_cols=LABEL_COLS,
    smooth_window=SMOOTH_WINDOW,
    figsize=FIGSIZE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_ylim_enabled=ROBUST_YLIM,
    lower_percentile=LOWER_PERCENTILE,
    upper_percentile=UPPER_PERCENTILE
)


# --- CELL 23 (code cell #17) ---
import pandas as pd
import numpy as np
import os

# ============================================================
# GROUP 1 XSENS LABEL COVERAGE DIAGNOSTIC
# ============================================================

XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_1/xsens_labeled/group_1_xsens_labeled.csv"
ELAN_PATH  = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"

TIME_COL = "video_time_s"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

xsens = pd.read_csv(XSENS_PATH, low_memory=False)
elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

print("================================================")
print("XSENS TIME RANGE")
print("================================================")
print("time_s range:")
print(xsens["time_s"].min(), "->", xsens["time_s"].max())

print("\nvideo_time_s range:")
print(xsens[TIME_COL].min(), "->", xsens[TIME_COL].max())
print("duration:", xsens[TIME_COL].max() - xsens[TIME_COL].min())

print("\n================================================")
print("ELAN TIME RANGE")
print("================================================")
print("ELAN begin/end:")
print(elan["begin_s"].min(), "->", elan["end_s"].max())
print("ELAN duration:", elan["end_s"].max() - elan["begin_s"].min())

print("\n================================================")
print("OVERLAP BETWEEN XSENS AND ELAN")
print("================================================")
overlap_start = max(xsens[TIME_COL].min(), elan["begin_s"].min())
overlap_end = min(xsens[TIME_COL].max(), elan["end_s"].max())

print("overlap:", overlap_start, "->", overlap_end)
print("overlap duration:", overlap_end - overlap_start)

if overlap_end <= overlap_start:
    print("WARNING: No overlap between XSens video_time_s and ELAN labels.")
else:
    print("Overlap exists.")

print("\n================================================")
print("LABEL COVERAGE PER COLUMN")
print("================================================")
for col in LABEL_COLS:
    if col in xsens.columns:
        n = xsens[col].notna().sum()
        pct_total = n / len(xsens) * 100

        in_overlap = xsens[(xsens[TIME_COL] >= overlap_start) & (xsens[TIME_COL] <= overlap_end)]
        n_overlap = in_overlap[col].notna().sum()
        pct_overlap = n_overlap / len(in_overlap) * 100 if len(in_overlap) > 0 else 0

        print(f"{col}:")
        print(f"  labeled rows total   : {n} / {len(xsens)} = {pct_total:.1f}%")
        print(f"  labeled rows overlap : {n_overlap} / {len(in_overlap)} = {pct_overlap:.1f}%")

print("\n================================================")
print("FIRST/LAST LABELED TIME PER COLUMN")
print("================================================")
for col in LABEL_COLS:
    if col in xsens.columns:
        sub = xsens[xsens[col].notna()]
        if len(sub) == 0:
            print(f"{col}: no labels")
        else:
            print(
                f"{col}: {sub[TIME_COL].min():.3f}s -> {sub[TIME_COL].max():.3f}s"
            )

print("\n================================================")
print("WHOLE GROUP LABELS")
print("================================================")
if "label_Whole_Group" in xsens.columns:
    print(xsens["label_Whole_Group"].dropna().value_counts())


# --- CELL 25 (code cell #18) ---
# ============================================================
# XSENS + SEPARATE LABEL FILE
# Align center of synchronizaiton_move label with highest ACC peak
# in the last 10% of XSens data
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"
os.makedirs(DRIVE_OUT, exist_ok=True)

XSENS_CSV_PATH = "/content/drive/MyDrive/thesis/data/group_1/xsens_merged/group_1_xsens_merged_30hz.csv"
LABEL_CSV_PATH = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"

OUTPUT_CSV_PATH = os.path.join(
    DRIVE_OUT,
    "group_1_xsens_labeled_shifted_by_last10_peak.csv"
)

TIME_COL = "time_s"

SYNC_LABEL = "synchronizaiton_move"
SYNC_LABEL_COL = "label_Whole_Group"

USERS = [1, 2, 3]

SMOOTH_WINDOW = 25
ZOOM_MARGIN = 25

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]


# ============================================================
# BASIC SIGNAL FUNCTIONS
# ============================================================

def acc_mag(df, user):
    x = f"p{user}_acc_x"
    y = f"p{user}_acc_y"
    z = f"p{user}_acc_z"

    if not all(col in df.columns for col in [x, y, z]):
        raise ValueError(f"Missing acceleration columns for participant {user}")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def moving_average(signal, window=25):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def combined_acc_signal(df, users=[1, 2, 3], smooth_window=25):
    mags = []

    for user in users:
        mags.append(acc_mag(df, user))

    combined_raw = np.mean(mags, axis=0)
    combined_smooth = moving_average(combined_raw, smooth_window)

    return combined_raw, combined_smooth


# ============================================================
# LOAD LABEL FILE AND PAINT LABELS ON XSENS TIME GRID
# ============================================================

def load_annotation_file(label_csv_path):
    """
    Your label file does not have a normal header.
    Structure:
    column 0 = participant / pair / whole group
    column 3 = start time in seconds
    column 5 = end time in seconds
    column 8 = label name
    """

    ann = pd.read_csv(label_csv_path, header=None, low_memory=False)

    ann = ann[[0, 3, 5, 8]].copy()
    ann.columns = ["entity", "start_s", "end_s", "label"]

    ann["start_s"] = pd.to_numeric(ann["start_s"], errors="coerce")
    ann["end_s"] = pd.to_numeric(ann["end_s"], errors="coerce")
    ann["label"] = ann["label"].astype(str)
    ann["entity"] = ann["entity"].astype(str)

    ann = ann.dropna(subset=["start_s", "end_s", "label"])

    return ann


def entity_to_label_column(entity):
    mapping = {
        "Participant1": "label_Participant1",
        "Participant2": "label_Participant2",
        "Participant3": "label_Participant3",
        "Participant1_Participant2": "label_Participant1_Participant2",
        "Participant1_Participant3": "label_Participant1_Participant3",
        "Participant2_Participant3": "label_Participant2_Participant3",
        "Whole_Group": "label_Whole_Group",
    }

    return mapping.get(entity, None)


def add_labels_to_signal_file(signal_df, annotation_df, time_col="time_s"):
    """
    Creates label columns in the XSens signal dataframe
    using the separate annotation file.
    """

    df = signal_df.copy()
    times = df[time_col].values

    for col in LABEL_COLS:
        df[col] = ""

    for _, row in annotation_df.iterrows():
        label_col = entity_to_label_column(row["entity"])

        if label_col is None:
            continue

        start_s = row["start_s"]
        end_s = row["end_s"]
        label = row["label"]

        mask = (times >= start_s) & (times <= end_s)
        df.loc[mask, label_col] = label

    return df


# ============================================================
# LABEL SEGMENT FUNCTIONS
# ============================================================

def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def find_sync_segment(df, time_col, label_col, sync_label):
    segments = get_label_segments(df, time_col, label_col)

    sync_segments = [
        seg for seg in segments
        if seg[2] == sync_label
    ]

    if len(sync_segments) == 0:
        raise ValueError(
            f"Could not find '{sync_label}' inside '{label_col}'. "
            f"Check spelling or label file."
        )

    sync_segments = sorted(
        sync_segments,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return sync_segments[0]


def shift_label_column(df, time_col, label_col, offset_sec):
    times = df[time_col].values
    shifted = np.array([""] * len(df), dtype=object)

    segments = get_label_segments(df, time_col, label_col)

    for start_s, end_s, label in segments:
        new_start = start_s + offset_sec
        new_end = end_s + offset_sec

        mask = (times >= new_start) & (times <= new_end)
        shifted[mask] = label

    return shifted


def shift_all_labels(df, time_col, label_cols, offset_sec):
    df_shifted = df.copy()

    for col in label_cols:
        if col in df_shifted.columns:
            df_shifted[col] = shift_label_column(
                df_shifted,
                time_col,
                col,
                offset_sec
            )

    return df_shifted


# ============================================================
# PEAK DETECTION
# ============================================================

def find_highest_peak_in_last_10_percent(df, time_col, users=[1, 2, 3], smooth_window=25):
    times = df[time_col].values

    _, combined_smooth = combined_acc_signal(
        df,
        users=users,
        smooth_window=smooth_window
    )

    start_idx = int(len(df) * 0.90)

    last_signal = combined_smooth[start_idx:]

    peak_idx_local = np.argmax(last_signal)
    peak_idx_global = start_idx + peak_idx_local

    peak_time = times[peak_idx_global]
    peak_value = combined_smooth[peak_idx_global]

    return peak_time, peak_value, peak_idx_global, combined_smooth


# ============================================================
# PLOTTING
# ============================================================

def plot_xsens_alignment_all_participants(
    df_before,
    df_after,
    time_col,
    old_sync_start,
    old_sync_end,
    new_sync_start,
    new_sync_end,
    peak_time,
    combined_smooth,
    users=[1, 2, 3],
    zoom_margin=25,
    smooth_window=25
):
    times = df_before[time_col].values

    left = min(old_sync_start, new_sync_start, peak_time) - zoom_margin
    right = max(old_sync_end, new_sync_end, peak_time) + zoom_margin

    mask = (times >= left) & (times <= right)

    old_middle = (old_sync_start + old_sync_end) / 2
    new_middle = (new_sync_start + new_sync_end) / 2

    p_smooth = {}

    for user in users:
        p_smooth[user] = moving_average(
            acc_mag(df_before, user),
            smooth_window
        )

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # Combined
    axes[0].plot(
        times[mask],
        combined_smooth[mask],
        linewidth=1.0,
        label="Combined XSens acc magnitude smoothed"
    )

    axes[0].axvspan(
        old_sync_start,
        old_sync_end,
        alpha=0.20,
        label="Original sync label"
    )

    axes[0].axvspan(
        new_sync_start,
        new_sync_end,
        alpha=0.20,
        label="Shifted sync label"
    )

    axes[0].axvline(
        peak_time,
        linestyle="--",
        linewidth=1.5,
        label="Highest acc peak in last 10%"
    )

    axes[0].axvline(
        old_middle,
        linestyle=":",
        linewidth=1.3,
        label="Original sync middle"
    )

    axes[0].axvline(
        new_middle,
        linestyle=":",
        linewidth=1.3,
        label="Shifted sync middle"
    )

    axes[0].set_title("Combined XSens acceleration magnitude")
    axes[0].set_ylabel("Combined acc mag")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Participants
    for i, user in enumerate(users, start=1):
        axes[i].plot(
            times[mask],
            p_smooth[user][mask],
            linewidth=1.0,
            label=f"Participant {user} XSens acc magnitude smoothed"
        )

        axes[i].axvspan(
            new_sync_start,
            new_sync_end,
            alpha=0.25,
            label="Shifted sync label"
        )

        axes[i].axvline(
            peak_time,
            linestyle="--",
            linewidth=1.5,
            label="Highest peak"
        )

        axes[i].set_title(f"Participant {user}")
        axes[i].set_ylabel(f"p{user} acc mag")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

# 1. Load XSens signal file
xsens_df = pd.read_csv(XSENS_CSV_PATH, low_memory=False)

# 2. Load annotation file
annotation_df = load_annotation_file(LABEL_CSV_PATH)

print("Loaded annotation labels:")
print(annotation_df["label"].value_counts().head(15))

# 3. Add labels to XSens dataframe
xsens_labeled = add_labels_to_signal_file(
    xsens_df,
    annotation_df,
    time_col=TIME_COL
)

# 4. Find original synchronization label
sync_start, sync_end, _ = find_sync_segment(
    xsens_labeled,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

sync_middle = (sync_start + sync_end) / 2

# 5. Find highest acceleration peak in last 10%
peak_time, peak_value, peak_idx, combined_smooth = find_highest_peak_in_last_10_percent(
    xsens_labeled,
    TIME_COL,
    users=USERS,
    smooth_window=SMOOTH_WINDOW
)

# 6. Calculate offset
offset_sec = peak_time - sync_middle

print("================================================")
print("XSENS LABEL ALIGNMENT")
print("================================================")
print(f"Original sync start        : {sync_start:.3f} s")
print(f"Original sync end          : {sync_end:.3f} s")
print(f"Original sync middle       : {sync_middle:.3f} s")
print("------------------------------------------------")
print(f"Highest peak time last 10% : {peak_time:.3f} s")
print(f"Highest peak value         : {peak_value:.3f}")
print("------------------------------------------------")
print(f"Offset                     : {offset_sec:.3f} s")
if offset_sec > 0:
    print("Labels will move LATER.")
elif offset_sec < 0:
    print("Labels will move EARLIER.")
else:
    print("No shift needed.")
print("================================================")

# 7. Shift all labels
xsens_shifted = shift_all_labels(
    xsens_labeled,
    TIME_COL,
    LABEL_COLS,
    offset_sec
)

# 8. Confirm new sync segment
new_sync_start, new_sync_end, _ = find_sync_segment(
    xsens_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

new_sync_middle = (new_sync_start + new_sync_end) / 2

print("After shifting:")
print(f"New sync start          : {new_sync_start:.3f} s")
print(f"New sync end            : {new_sync_end:.3f} s")
print(f"New sync middle         : {new_sync_middle:.3f} s")
print(f"Difference from peak    : {new_sync_middle - peak_time:.6f} s")

# 9. Save output
xsens_shifted.to_csv(OUTPUT_CSV_PATH, index=False)

print("================================================")
print("Saved shifted XSens file to:")
print(OUTPUT_CSV_PATH)
print("================================================")

# 10. Visual check
plot_xsens_alignment_all_participants(
    df_before=xsens_labeled,
    df_after=xsens_shifted,
    time_col=TIME_COL,
    old_sync_start=sync_start,
    old_sync_end=sync_end,
    new_sync_start=new_sync_start,
    new_sync_end=new_sync_end,
    peak_time=peak_time,
    combined_smooth=combined_smooth,
    users=USERS,
    zoom_margin=ZOOM_MARGIN,
    smooth_window=SMOOTH_WINDOW
)


# --- CELL 26 (code cell #19) ---
# ============================================================
# CLEAN FULL XSENS VISUALIZATION WITH ALL LABELS
# Fixes p1/p3 flat-line issue by using robust y-axis scaling
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# CONFIG
# ============================================================

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_1"

CSV_PATH = os.path.join(
    DRIVE_OUT,
    "group_1_xsens_labeled_shifted_by_last10_peak.csv"
)

TIME_COL = "time_s"

USERS = [1, 2, 3]

SIGNAL_TYPE = "acc_mag"   # "acc_mag", "gyro_mag", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"

SMOOTH_WINDOW = 15

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.12

TEXT_EVERY_N_SEGMENTS = 4   # increase if still too crowded

# Important part:
# This only affects visualization, not the actual data.
ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window=15):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def signal_for_user(df, user, signal_type="acc_mag"):
    if signal_type == "acc_mag":
        x = f"p{user}_acc_x"
        y = f"p{user}_acc_y"
        z = f"p{user}_acc_z"

        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    elif signal_type == "gyro_mag":
        x = f"p{user}_gyro_x"
        y = f"p{user}_gyro_y"
        z = f"p{user}_gyro_z"

        return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    else:
        col = f"p{user}_{signal_type}"

        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

        return df[col].values


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]

    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)

    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def print_signal_diagnostics(df, users, signal_type):
    print("================================================")
    print("SIGNAL DIAGNOSTICS")
    print("================================================")

    for user in users:
        sig = signal_for_user(df, user, signal_type)
        clean = sig[np.isfinite(sig)]

        print(f"Participant {user} - {signal_type}")
        print(f"  min   : {np.min(clean):.6g}")
        print(f"  p1    : {np.percentile(clean, 1):.6g}")
        print(f"  median: {np.median(clean):.6g}")
        print(f"  p99   : {np.percentile(clean, 99):.6g}")
        print(f"  max   : {np.max(clean):.6g}")
        print("------------------------------------------------")


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================

def plot_clean_full_xsens_all_labels(
    csv_path,
    time_col="time_s",
    users=[1, 2, 3],
    signal_type="acc_mag",
    smooth_window=15,
    label_cols=None,
    label_alpha=0.12,
    figsize=(24, 12),
    text_every_n_segments=4,
    robust_ylim_enabled=True,
    lower_percentile=1,
    upper_percentile=99
):
    df = pd.read_csv(csv_path, low_memory=False)

    if label_cols is None:
        label_cols = [
            "label_Participant1",
            "label_Participant2",
            "label_Participant3",
            "label_Participant1_Participant2",
            "label_Participant1_Participant3",
            "label_Participant2_Participant3",
            "label_Whole_Group"
        ]

    existing_label_cols = [
        col for col in label_cols
        if col in df.columns
    ]

    times = df[time_col].values

    print_signal_diagnostics(df, users, signal_type)

    # --------------------------------------------------------
    # Collect unique labels
    # --------------------------------------------------------

    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    print("================================================")
    print("LABEL SUMMARY")
    print("================================================")
    print(f"Unique labels found: {len(unique_labels)}")
    print("Label columns used:")
    for col in existing_label_cols:
        print(" -", col)
    print("================================================")

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        len(users),
        1,
        figsize=figsize,
        sharex=True
    )

    if len(users) == 1:
        axes = [axes]

    for ax, user in zip(axes, users):

        sig_raw = signal_for_user(df, user, signal_type)
        sig = moving_average(sig_raw, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        # Set robust y-axis before adding text
        if robust_ylim_enabled:
            ylim = robust_ylim(
                sig,
                lower_p=lower_percentile,
                upper_p=upper_percentile
            )

            if ylim is not None:
                ax.set_ylim(ylim)

        # Add labels
        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, time_col, label_col)

            for start, end, label in segments:

                if label == "":
                    continue

                color = label_colors.get(label, "gray")

                ax.axvspan(
                    start,
                    end,
                    color=color,
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user} {signal_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_xsens_clean_full = plot_clean_full_xsens_all_labels(
    csv_path=CSV_PATH,
    time_col=TIME_COL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW,
    label_cols=LABEL_COLS,
    label_alpha=LABEL_ALPHA,
    figsize=FIGSIZE,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_ylim_enabled=ROBUST_YLIM,
    lower_percentile=LOWER_PERCENTILE,
    upper_percentile=UPPER_PERCENTILE
)


# --- CELL 29 (code cell #20) ---
import pandas as pd, numpy as np, os

# ---- Group 2 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_2/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_2/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

# value columns per sensor
# trailing all-zero / flag column is dropped
SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000   # 50 Hz target grid
TOL_FAST_US = 12_000   # tolerance for 50 Hz streams
TOL_SLOW_US = 40_000   # wider tolerance for slower streams, especially skin_temp


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + value columns, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    # this clips possible warm-up samples from other streams
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():
        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 30 (code cell #21) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_2/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_2"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load every participant's 8 raw streams
# and find each participant's session window
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    # session window based on acc / gyro / mgnt
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Shared grid = overlap window
# latest participant start -> earliest participant end
# ============================================================

start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge every participant and every sensor onto shared 50 Hz grid
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_2_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_2_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 31 (code cell #22) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_2/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_2"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]


def load(p):
    path = os.path.join(BASE, f"{p}.csv")

    d = pd.read_csv(path)

    # drop trailing empty columns if they exist
    d = d.loc[:, [c for c in d.columns if str(c).strip() != ""]]

    # clean column names
    d.columns = [str(c).strip() for c in d.columns]

    # convert important columns to numeric
    for c in ["SampleTimeFine", "PacketCounter"] + SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")

    return d


# ============================================================
# Load raw XSens files
# ============================================================

raw = {
    p: load(p)
    for p in PREFIX
}

for p, d in raw.items():
    print(
        p,
        "| rows:", len(d),
        "| SampleTimeFine valid:", d["SampleTimeFine"].notna().sum()
    )


# ============================================================
# Master grid = Participant1 clean SampleTimeFine
# ============================================================

p1 = raw["Participant1"].dropna(subset=["SampleTimeFine"])
p1_ticks = set(p1["SampleTimeFine"].tolist())


# ============================================================
# Common span across all 3 participants
# using only ticks that exist on P1 grid
# ============================================================

spans = []

for p, d in raw.items():
    valid = d[d["SampleTimeFine"].isin(p1_ticks)]

    span = (
        int(valid["SampleTimeFine"].min()),
        int(valid["SampleTimeFine"].max())
    )

    spans.append(span)

    print(
        f"{p} span:",
        span[0],
        "->",
        span[1],
        "| duration:",
        round((span[1] - span[0]) / 1e6, 1),
        "s"
    )


start = max(s[0] for s in spans)
end   = min(s[1] for s in spans)


# ============================================================
# Build master timeline
# ============================================================

master = (
    p1[
        (p1["SampleTimeFine"] >= start) &
        (p1["SampleTimeFine"] <= end)
    ][["SampleTimeFine"]]
    .sort_values("SampleTimeFine")
    .reset_index(drop=True)
)

master["time_s"] = (
    (master["SampleTimeFine"].astype(np.int64) - start) / 1e6
).round(4)


# ============================================================
# Merge all participants onto master grid
# ============================================================

wide = master.copy()

for p, d in raw.items():
    pre = PREFIX[p]

    sub = (
        d[["SampleTimeFine"] + SENSOR_COLS]
        .dropna(subset=["SampleTimeFine"])
        .drop_duplicates("SampleTimeFine")
        .rename(columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        })
    )

    wide = wide.merge(
        sub,
        on="SampleTimeFine",
        how="left"
    )


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_2_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| dur:",
    round(wide["time_s"].iloc[-1], 1),
    "s ->",
    local_path
)

os.makedirs(DRIVE_OUT, exist_ok=True)

dst = os.path.join(
    DRIVE_OUT,
    "group_2_xsens_merged_30hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing check
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 32 (code cell #23) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_with_individual_build.csv"
LOCAL_OUT = "/content/Group_2_with_individual_build.csv"
# ----------------

PEOPLE = ["Khalil", "Arda", "Shizin"]

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())

# ============================================================
# Involving tiers
# Handles:
# Khalil
# Arda Khalil
# Shizin Khalil
# Whole Group
# ============================================================

INVOLVING = {}

for person in PEOPLE:
    tiers = []

    for tier in all_tiers:
        tier_parts = tier.split()

        if tier == person:
            tiers.append(tier)

        elif tier == "Whole Group":
            tiers.append(tier)

        elif person in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

# ============================================================
# Cutoff per person
# ============================================================

CUTOFF = {}

for person, tiers in INVOLVING.items():
    involved_rows = df[df["tier"].isin(tiers)]

    if len(involved_rows) == 0:
        raise ValueError(f"No annotations found for {person}")

    CUTOFF[person] = float(involved_rows["end_s"].max())

print("\nCutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ============================================================
# Add individual_build
# ============================================================

new_rows = []

for person, tiers in INVOLVING.items():
    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):
        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })

both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

print(
    f"\n{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("copied ->", DRIVE_OUT)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)


# --- CELL 33 (code cell #24) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_2_individual_build_renamed.csv"
# ----------------

# Group 2 mapping
NAME = {
    "Khalil": "Participant1",
    "Arda": "Participant2",
    "Shizin": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

def rename_tier(t):
    t = str(t).strip()

    if t == "Whole Group":
        return "Whole_Group"

    if t == "Whole_Group":
        return "Whole_Group"

    # handles both "Arda Khalil" and "Arda_Khalil"
    if "_" in t:
        parts = t.split("_")
    else:
        parts = t.split()

    if all(p in NAME for p in parts):
        mapped = sorted(
            [NAME[p] for p in parts],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)


# --- CELL 34 (code cell #25) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_2/openearable_merged/group_2_openearable_merged_50hz.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/openearable_labeled/group_2_openearable_labeled.csv"
LOCAL_OUT = "/content/group_2_openearable_labeled.csv"

# ---- sync anchor ----
# Group 2 video start:
# 2026-04-21 12:04:25 UTC = 14:04:25 Netherlands local
VIDEO_START_US = 1776773065000000
# ---------------------

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

oe = pd.read_csv(OE_PATH)

# make numeric
elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"]   = pd.to_numeric(elan["end_s"], errors="coerce")

# create video-relative time from OpenEarable timestamp
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder label columns if they exist
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

# put video_time_s right after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]


# ============================================================
# Add labels from ELAN to OpenEarable timeline
# ============================================================

for t in TIERS:
    sub = (
        elan[elan.tier == t][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "label": "_lab",
                "end_s": "_end"
            }
        )
    )

    col = f"label_{t}"

    if len(sub) == 0:
        oe[col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[col] = m["_lab"].where(valid, other=pd.NA).values


# ============================================================
# Save
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

oe.to_csv(LOCAL_OUT, index=False)

print("shape:", oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nLabel coverage:")
for t in TIERS:
    n = oe[f"label_{t}"].notna().sum()
    print(f"  label_{t}: {n} rows ({n / len(oe) * 100:.1f}%)")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\ncopied ->", DRIVE_OUT)


# --- CELL 35 (code cell #26) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_2/xsens_merged/group_2_xsens_merged_30hz.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_2/xsens_labeled/group_2_xsens_labeled.csv"
LOCAL_OUT = "/content/group_2_xsens_labeled.csv"
# ----------------

# ---- timing ----
# Video start: 14:04:25
# XSens start: 14:07:51
# Difference: 3 min 26 sec = 206 sec
XSENS_START_OFFSET_FROM_VIDEO_S = 206.0
# ----------------

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

# ============================================================
# Load files
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

xsens = pd.read_csv(XSENS_PATH)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")

# ============================================================
# Create video-relative time for XSens
# ============================================================

# XSens time_s starts from XSens recording start.
# To compare it with ELAN video labels, convert it to video time.
xsens["video_time_s"] = xsens["time_s"] + XSENS_START_OFFSET_FROM_VIDEO_S

# remove old placeholder label columns if they exist
xsens = xsens.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in xsens.columns],
    errors="ignore"
)

# Put video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# ============================================================
# Add labels from ELAN to XSens timeline
# ============================================================

for t in TIERS:
    sub = (
        elan[elan.tier == t][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "label": "_lab",
                "end_s": "_end"
            }
        )
    )

    col = f"label_{t}"

    if len(sub) == 0:
        xsens[col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[col] = m["_lab"].where(valid, other=pd.NA).values

# ============================================================
# Save
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

xsens.to_csv(LOCAL_OUT, index=False)

print("shape:", xsens.shape)

print("\nXSens time range:")
print("time_s start      :", xsens["time_s"].min())
print("time_s end        :", xsens["time_s"].max())
print("video_time_s start:", xsens["video_time_s"].min())
print("video_time_s end  :", xsens["video_time_s"].max())

print("\nLabel coverage:")
for t in TIERS:
    n = xsens[f"label_{t}"].notna().sum()
    print(f"  label_{t}: {n} rows ({n / len(xsens) * 100:.1f}%)")

shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\ncopied ->", DRIVE_OUT)


# --- CELL 36 (code cell #27) ---
import pandas as pd

OE_PATH = "/content/drive/MyDrive/thesis/data/group_2/openearable_labeled/group_2_openearable_labeled.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_2/xsens_labeled/group_2_xsens_labeled.csv"

oe = pd.read_csv(OE_PATH, low_memory=False)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

print("OpenEarable Participant1 labels:")
print(oe["label_Participant1"].dropna().unique())

print("\nXSens Participant1 labels:")
print(xsens["label_Participant1"].dropna().unique())


# --- CELL 37 (code cell #28) ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_2"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_2_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_2_xsens_labeled.csv"
)

TIME_COL = "video_time_s"
EVENT_COL = "label_Participant1"

TARGET_EVENT = "droping_erarble_syncornaziton_move"

USERS = [1, 2, 3]

ZOOM_MARGIN = 25
SMOOTH_OE = 25
SMOOTH_XSENS = 15


def moving_average(x, window):
    if window <= 1:
        return x

    return (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def acc_mag(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def robust_clip_for_plot(x, low_p=1, high_p=99):
    """
    Only clips for visualization.
    Does not change the dataframe.
    """
    x = np.asarray(x, dtype=float)

    finite = x[np.isfinite(x)]

    if len(finite) == 0:
        return x

    low = np.percentile(finite, low_p)
    high = np.percentile(finite, high_p)

    return np.clip(x, low, high)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_target_event(df, target_event):
    segments = get_label_segments(df, TIME_COL, EVENT_COL)

    matched = [
        seg for seg in segments
        if seg[2] == target_event
    ]

    if len(matched) == 0:
        print("Available Participant1 labels:")
        print(df[EVENT_COL].dropna().unique())
        raise ValueError(f"Could not find event: {target_event}")

    return matched


def plot_specific_event(
    csv_path,
    title,
    target_event,
    smooth_window,
    zoom_margin=25,
    robust=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    events = find_target_event(df, target_event)

    print("\n================================================")
    print(title)
    print("Target event:", target_event)
    print("================================================")

    for i, (s, e, lab) in enumerate(events):
        print(f"{i}: {s:.3f}s -> {e:.3f}s | duration={e-s:.3f}s")

    # Use first occurrence
    event_start, event_end, event_label = events[0]
    event_middle = (event_start + event_end) / 2

    times = df[TIME_COL].values

    left = event_start - zoom_margin
    right = event_end + zoom_margin

    mask = (times >= left) & (times <= right)

    signals = {}

    for user in USERS:
        sig = acc_mag(df, user)
        sig = moving_average(sig, smooth_window)

        if robust:
            sig = robust_clip_for_plot(sig, 1, 99)

        signals[user] = sig

    combined = np.mean(
        [signals[user] for user in USERS],
        axis=0
    )

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # Combined
    axes[0].plot(
        times[mask],
        combined[mask],
        linewidth=1.0,
        label="Combined acc_mag"
    )

    axes[0].axvspan(
        event_start,
        event_end,
        alpha=0.25,
        label=event_label
    )

    axes[0].axvline(
        event_middle,
        linestyle="--",
        linewidth=1.5,
        label="event middle"
    )

    axes[0].set_title(f"{title} - Combined acc_mag")
    axes[0].set_ylabel("combined")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Participants
    for i, user in enumerate(USERS, start=1):
        axes[i].plot(
            times[mask],
            signals[user][mask],
            linewidth=1.0,
            label=f"Participant {user} acc_mag"
        )

        axes[i].axvspan(
            event_start,
            event_end,
            alpha=0.25,
            label=event_label
        )

        axes[i].axvline(
            event_middle,
            linestyle="--",
            linewidth=1.5,
            label="event middle"
        )

        axes[i].set_title(f"{title} - Participant {user}")
        axes[i].set_ylabel(f"p{user}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return df, events


df_oe_drop, oe_drop_events = plot_specific_event(
    csv_path=OE_PATH,
    title="Group 2 OpenEarable",
    target_event=TARGET_EVENT,
    smooth_window=SMOOTH_OE,
    zoom_margin=ZOOM_MARGIN,
    robust=True
)

df_xsens_drop, xsens_drop_events = plot_specific_event(
    csv_path=XSENS_PATH,
    title="Group 2 XSens",
    target_event=TARGET_EVENT,
    smooth_window=SMOOTH_XSENS,
    zoom_margin=ZOOM_MARGIN,
    robust=True
)


# --- CELL 39 (code cell #29) ---
# ============================================================
# GROUP 3 - STEP 0
# Concatenate ELAN Part 1 + Part 2 using real video start times
# ============================================================

import pandas as pd
import os
import shutil
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

PART1_PATH = os.path.join(
    GROUP_DIR,
    "elan",
    "Group_3_Part1.csv"
)

PART2_PATH = os.path.join(
    GROUP_DIR,
    "elan",
    "Group_3_Part2.csv"
)

LOCAL_OUT = "/content/Group_3_concatenated.csv"

DRIVE_OUT = os.path.join(
    GROUP_DIR,
    "elan",
    "Group_3_concatenated.csv"
)


# ============================================================
# REAL VIDEO START TIMES
# ============================================================

PART1_START_CLOCK = "12:21:50"
PART2_START_CLOCK = "12:46:56"

# Part 2 offset relative to Part 1 video timeline
fmt_clock = "%H:%M:%S"

part1_start_dt = datetime.strptime(PART1_START_CLOCK, fmt_clock)
part2_start_dt = datetime.strptime(PART2_START_CLOCK, fmt_clock)

PART2_OFFSET_S = (part2_start_dt - part1_start_dt).total_seconds()

print("Part 2 offset:", PART2_OFFSET_S, "seconds")


# ============================================================
# ELAN COLUMNS
# ============================================================

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


# ============================================================
# HELPERS
# ============================================================

def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def load_elan(path):
    df = pd.read_csv(
        path,
        header=None,
        names=COLS,
        dtype={"blank": str}
    )

    df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
    df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
    df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

    df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

    return df


# ============================================================
# LOAD PARTS
# ============================================================

part1 = load_elan(PART1_PATH)
part2 = load_elan(PART2_PATH)


# ============================================================
# SHIFT PART 2 BY REAL CLOCK OFFSET
# ============================================================

part1_fixed = part1.copy()
part2_shifted = part2.copy()

part2_shifted["begin_s"] = part2_shifted["begin_s"] + PART2_OFFSET_S
part2_shifted["end_s"] = part2_shifted["end_s"] + PART2_OFFSET_S
part2_shifted["dur_s"] = part2_shifted["end_s"] - part2_shifted["begin_s"]

# recompute hms columns
part1_fixed["begin_hms"] = part1_fixed["begin_s"].apply(hms)
part1_fixed["end_hms"] = part1_fixed["end_s"].apply(hms)
part1_fixed["dur_hms"] = part1_fixed["dur_s"].apply(hms)

part2_shifted["begin_hms"] = part2_shifted["begin_s"].apply(hms)
part2_shifted["end_hms"] = part2_shifted["end_s"].apply(hms)
part2_shifted["dur_hms"] = part2_shifted["dur_s"].apply(hms)


# ============================================================
# CONCATENATE
# ============================================================

combined = (
    pd.concat([part1_fixed, part2_shifted], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)


# ============================================================
# SAVE IN SAME ELAN-STYLE CSV FORMAT
# ============================================================

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in combined.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)


# ============================================================
# CHECK OUTPUT
# ============================================================

print("================================================")
print("GROUP 3 ELAN CONCATENATION COMPLETE")
print("================================================")

print("Part 1 start clock:", PART1_START_CLOCK)
print("Part 2 start clock:", PART2_START_CLOCK)
print("Part 2 offset     :", PART2_OFFSET_S, "seconds")

print("\nRows:")
print("Part 1   :", len(part1))
print("Part 2   :", len(part2))
print("Combined :", len(combined))

print("\nTime ranges:")
print(f"Part 1 original       : {part1['begin_s'].min():.3f} -> {part1['end_s'].max():.3f}")
print(f"Part 2 original       : {part2['begin_s'].min():.3f} -> {part2['end_s'].max():.3f}")
print(f"Part 2 shifted        : {part2_shifted['begin_s'].min():.3f} -> {part2_shifted['end_s'].max():.3f}")
print(f"Combined full range   : {combined['begin_s'].min():.3f} -> {combined['end_s'].max():.3f}")

print("\nSaved to:")
print(DRIVE_OUT)

print("\nTiers:")
for t in sorted(combined["tier"].unique()):
    print(" -", t)

print("================================================")


# --- CELL 40 (code cell #30) ---
import pandas as pd, numpy as np, os

# ---- Group 3 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_3/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_3/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

# value columns per sensor
# trailing all-zero / flag column is dropped
SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000   # 50 Hz target grid
TOL_FAST_US = 12_000   # tolerance for 50 Hz streams
TOL_SLOW_US = 40_000   # wider tolerance for slower streams


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + value columns, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    # this clips possible warm-up samples from other streams
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 41 (code cell #31) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_3/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_3"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load every participant's 8 raw streams
# and find each participant's session window
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    # session window based on acc / gyro / mgnt
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Shared grid = overlap window
# latest participant start -> earliest participant end
# ============================================================

start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge every participant and every sensor onto shared 50 Hz grid
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_3_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_3_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 42 (code cell #32) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_3/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_3"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]


def load(p):
    path = os.path.join(BASE, f"{p}.csv")

    d = pd.read_csv(path, low_memory=False)

    # drop trailing empty columns if they exist
    d = d.loc[:, [c for c in d.columns if str(c).strip() != ""]]

    # clean column names
    d.columns = [str(c).strip() for c in d.columns]

    # convert important columns to numeric
    for c in ["SampleTimeFine", "PacketCounter"] + SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")

    return d


# ============================================================
# Load raw XSens files
# ============================================================

raw = {
    p: load(p)
    for p in PREFIX
}

for p, d in raw.items():
    print(
        p,
        "| rows:", len(d),
        "| SampleTimeFine valid:", d["SampleTimeFine"].notna().sum()
    )


# ============================================================
# Master grid = Participant1 clean SampleTimeFine
# ============================================================

p1 = raw["Participant1"].dropna(subset=["SampleTimeFine"])
p1_ticks = set(p1["SampleTimeFine"].tolist())


# ============================================================
# Common span across all 3 participants
# using only ticks that exist on P1 grid
# ============================================================

spans = []

for p, d in raw.items():
    valid = d[d["SampleTimeFine"].isin(p1_ticks)]

    span = (
        int(valid["SampleTimeFine"].min()),
        int(valid["SampleTimeFine"].max())
    )

    spans.append(span)

    print(
        f"{p} span:",
        span[0],
        "->",
        span[1],
        "| duration:",
        round((span[1] - span[0]) / 1e6, 1),
        "s"
    )


start = max(s[0] for s in spans)
end   = min(s[1] for s in spans)


# ============================================================
# Build master timeline
# ============================================================

master = (
    p1[
        (p1["SampleTimeFine"] >= start) &
        (p1["SampleTimeFine"] <= end)
    ][["SampleTimeFine"]]
    .sort_values("SampleTimeFine")
    .reset_index(drop=True)
)

master["time_s"] = (
    (master["SampleTimeFine"].astype(np.int64) - start) / 1e6
).round(4)


# ============================================================
# Merge all participants onto master grid
# ============================================================

wide = master.copy()

for p, d in raw.items():
    pre = PREFIX[p]

    sub = (
        d[["SampleTimeFine"] + SENSOR_COLS]
        .dropna(subset=["SampleTimeFine"])
        .drop_duplicates("SampleTimeFine")
        .rename(columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        })
    )

    wide = wide.merge(
        sub,
        on="SampleTimeFine",
        how="left"
    )


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_3_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| dur:",
    round(wide["time_s"].iloc[-1], 1),
    "s ->",
    local_path
)

os.makedirs(DRIVE_OUT, exist_ok=True)

dst = os.path.join(
    DRIVE_OUT,
    "group_3_xsens_merged_30hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing check
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 43 (code cell #33) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_concatenated.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_with_individual_build.csv"
LOCAL_OUT = "/content/Group_3_with_individual_build.csv"
# ----------------

# ---- Group 3 participant mapping ----
# chinyu = Participant1
# long   = Participant2
# egemen = Participant3
PEOPLE = ["chinyu", "long", "egemen"]

# ---- manual individual_build cutoff times ----
CUTOFF = {
    "chinyu": 41 * 60 + 35,   # 2495 s
    "long":   41 * 60 + 35,   # 2495 s
    "egemen": 34 * 60 + 25,   # 2065 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


# ============================================================
# Helper: normalize names for safer matching
# ============================================================

def norm(x):
    return str(x).strip().lower()


# ============================================================
# Involving tiers
# Includes:
# own tier + dyads containing person + Whole Group / Whole_Group
# Handles both spaces and underscores
# ============================================================

INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


# ============================================================
# Interval helpers
# ============================================================

def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ============================================================
# Add individual_build
# ============================================================

new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)


# ============================================================
# Save same ELAN-style CSV
# Still uses real names; next cell will rename to Participant1/2/3
# ============================================================

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 3 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 44 (code cell #34) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_3_individual_build_renamed.csv"
# ----------------

# Group 3 mapping
NAME = {
    "chinyu": "Participant1",
    "long": "Participant2",
    "egemen": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    # handles both "egemen chinyu" and "egemen_chinyu"
    parts = t_lower.replace("_", " ").split()

    if all(p in NAME for p in parts):
        mapped = sorted(
            [NAME[p] for p in parts],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 45 (code cell #35) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 3 - LABEL OPENEARAMBLE USING CONCATENATED ELAN
# ============================================================

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_3/openearable_merged/group_3_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/openearable_labeled/group_3_openearable_labeled.csv"
LOCAL_OUT = "/content/group_3_openearable_labeled.csv"
# ----------------

# ============================================================
# VIDEO START TIME
# ============================================================

# Group 3 Part 1 video start:
# 2026-04-22 12:21:50 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-22 12:21:50", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)


# ============================================================
# COLUMNS
# ============================================================

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]


# ============================================================
# LOAD
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

oe = pd.read_csv(
    OE_PATH,
    low_memory=False
)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# ============================================================
# CREATE VIDEO TIME FOR OPENEARAMBLE
# ============================================================

oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder label columns
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

# remove old full label columns if rerunning
oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# put video_time_s right after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]


# ============================================================
# ADD ELAN LABELS TO OPENEARAMBLE TIMELINE
# ============================================================

for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values


# ============================================================
# SAVE
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)


# ============================================================
# CHECK
# ============================================================

print("================================================")
print("GROUP 3 OPENEARAMBLE LABELED")
print("================================================")

print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 46 (code cell #36) ---
# ============================================================
# GROUP 3 - LABEL XSENS CORRECTLY USING CONCATENATED VIDEO TIME
# Group 3 start = 2026-04-22 12:21:50 UTC
# Part 2 ELAN already shifted by 1506s
# Therefore:
# video_time_s = time_s
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil


# ============================================================
# PATHS
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_individual_build_renamed.csv"

XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_3/xsens_merged/group_3_xsens_merged_30hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_3/xsens_labeled/group_3_xsens_labeled.csv"

LOCAL_OUT = "/content/group_3_xsens_labeled.csv"


# ============================================================
# TIMING
# ============================================================

# XSens starts at the same reference as Group 3 video Part 1.
# Part 2 offset is already included in the concatenated ELAN file.
XSENS_TO_VIDEO_OFFSET_S = 0.0


# ============================================================
# COLUMNS
# ============================================================

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]


# ============================================================
# LOAD FILES
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

xsens = pd.read_csv(
    XSENS_PATH,
    low_memory=False
)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")


# ============================================================
# CREATE VIDEO-RELATIVE TIME FOR XSENS
# ============================================================

xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder label columns if they exist
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

# remove old full label columns if rerunning the cell
xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s right after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]


# ============================================================
# LABEL XSENS USING ELAN VIDEO TIME
# ============================================================

for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values


# ============================================================
# SAVE
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)


# ============================================================
# CHECK OUTPUT
# ============================================================

print("================================================")
print("GROUP 3 XSENS LABELED SUCCESSFULLY")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("ELAN start:", elan["begin_s"].min())
print("ELAN end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 47 (code cell #37) ---
# ============================================================
# GROUP 3 - VISUAL SANITY CHECK
# Full label visualization for OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_3_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 4

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]

    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)

    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    """
    source: "oe" or "xsens"
    signal_type: "acc_mag" or "gyro_mag"
    """

    if source == "oe":

        if signal_type == "acc_mag":
            x = f"p{user}_acc_x"
            y = f"p{user}_acc_y"
            z = f"p{user}_acc_z"

        elif signal_type == "gyro_mag":
            x = f"p{user}_gyro_x"
            y = f"p{user}_gyro_y"
            z = f"p{user}_gyro_z"

        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":

        if signal_type == "acc_mag":
            x = f"p{user}_acc_x"
            y = f"p{user}_acc_y"
            z = f"p{user}_acc_z"

        elif signal_type == "gyro_mag":
            x = f"p{user}_gyr_x"
            y = f"p{user}_gyr_y"
            z = f"p{user}_gyr_z"

        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be 'oe' or 'xsens'")

    missing = [c for c in [x, y, z] if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns for participant {user}: {missing}")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_cols=None,
    label_alpha=0.14,
    text_every_n_segments=4,
    robust_y=True,
    figsize=(24, 12)
):
    df = pd.read_csv(csv_path, low_memory=False)

    if label_cols is None:
        label_cols = LABEL_COLS

    existing_label_cols = [
        col for col in label_cols
        if col in df.columns
    ]

    if len(existing_label_cols) == 0:
        raise ValueError("No label columns found.")

    if TIME_COL not in df.columns:
        raise ValueError(f"{TIME_COL} not found.")

    times = df[TIME_COL].values

    print_label_summary(df, title)

    # collect unique labels
    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    fig, axes = plt.subplots(
        len(USERS),
        1,
        figsize=figsize,
        sharex=True
    )

    if len(USERS) == 1:
        axes = [axes]

    for ax, user in zip(axes, USERS):

        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(
                sig,
                lower_p=LOWER_PERCENTILE,
                upper_p=UPPER_PERCENTILE
            )

            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:

                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN FULL VISUAL CHECKS
# ============================================================

df_group3_oe_full = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 3 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_cols=LABEL_COLS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM,
    figsize=FIGSIZE
)

df_group3_xsens_full = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 3 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_cols=LABEL_COLS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM,
    figsize=FIGSIZE
)


# --- CELL 48 (code cell #38) ---
# ============================================================
# GROUP 3 - ZOOM AROUND SYNCHRONIZATION MOVE
# OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_3_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronaztion_move"   # exact spelling in Group 3

USERS = [1, 2, 3]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

ZOOM_MARGIN = 25

ROBUST_CLIP = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_clip_for_plot(signal, lower_p=1, upper_p=99):
    signal = np.asarray(signal, dtype=float)
    finite = signal[np.isfinite(signal)]

    if len(finite) == 0:
        return signal

    low = np.percentile(finite, lower_p)
    high = np.percentile(finite, upper_p)

    return np.clip(signal, low, high)


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x = f"p{user}_acc_x"
            y = f"p{user}_acc_y"
            z = f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x = f"p{user}_gyro_x"
            y = f"p{user}_gyro_y"
            z = f"p{user}_gyro_z"
        else:
            raise ValueError("Use 'acc_mag' or 'gyro_mag'.")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x = f"p{user}_acc_x"
            y = f"p{user}_acc_y"
            z = f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x = f"p{user}_gyr_x"
            y = f"p{user}_gyr_y"
            z = f"p{user}_gyr_z"
        else:
            raise ValueError("Use 'acc_mag' or 'gyro_mag'.")

    else:
        raise ValueError("source must be 'oe' or 'xsens'.")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def find_sync_segment(df, time_col, label_col, sync_label):
    segments = get_label_segments(df, time_col, label_col)

    sync_segments = [
        seg for seg in segments
        if seg[2] == sync_label
    ]

    if len(sync_segments) == 0:
        print("Available labels in", label_col)
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find label: {sync_label}")

    # If more than one, use longest segment
    sync_segments = sorted(
        sync_segments,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return sync_segments[0]


def plot_sync_zoom(
    csv_path,
    source,
    title,
    time_col="video_time_s",
    sync_label_col="label_Whole_Group",
    sync_label="synchronaztion_move",
    users=[1, 2, 3],
    signal_type="acc_mag",
    smooth_window=25,
    zoom_margin=25,
    robust_clip=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    sync_start, sync_end, found_label = find_sync_segment(
        df,
        time_col,
        sync_label_col,
        sync_label
    )

    sync_middle = (sync_start + sync_end) / 2

    print("================================================")
    print(title)
    print("SYNC LABEL ZOOM")
    print("================================================")
    print(f"Label    : {found_label}")
    print(f"Start    : {sync_start:.3f} s")
    print(f"End      : {sync_end:.3f} s")
    print(f"Middle   : {sync_middle:.3f} s")
    print(f"Duration : {sync_end - sync_start:.3f} s")
    print("================================================")

    times = df[time_col].values

    left = sync_start - zoom_margin
    right = sync_end + zoom_margin

    mask = (times >= left) & (times <= right)

    signals = {}

    for user in users:
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        if robust_clip:
            sig = robust_clip_for_plot(
                sig,
                LOWER_PERCENTILE,
                UPPER_PERCENTILE
            )

        signals[user] = sig

    combined = np.mean(
        [signals[user] for user in users],
        axis=0
    )

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # Combined
    axes[0].plot(
        times[mask],
        combined[mask],
        linewidth=1.0,
        label=f"Combined {signal_type}"
    )

    axes[0].axvspan(
        sync_start,
        sync_end,
        alpha=0.25,
        label=found_label
    )

    axes[0].axvline(
        sync_middle,
        linestyle="--",
        linewidth=1.5,
        label="sync middle"
    )

    axes[0].set_title(f"{title} - Combined {signal_type}")
    axes[0].set_ylabel("combined")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Participants
    for i, user in enumerate(users, start=1):

        axes[i].plot(
            times[mask],
            signals[user][mask],
            linewidth=1.0,
            label=f"Participant {user} {signal_type}"
        )

        axes[i].axvspan(
            sync_start,
            sync_end,
            alpha=0.25,
            label=found_label
        )

        axes[i].axvline(
            sync_middle,
            linestyle="--",
            linewidth=1.5,
            label="sync middle"
        )

        axes[i].set_title(f"{title} - Participant {user}")
        axes[i].set_ylabel(f"p{user}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_g3_oe_sync = plot_sync_zoom(
    csv_path=OE_PATH,
    source="oe",
    title="Group 3 OpenEarable",
    time_col=TIME_COL,
    sync_label_col=SYNC_LABEL_COL,
    sync_label=SYNC_LABEL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    zoom_margin=ZOOM_MARGIN,
    robust_clip=ROBUST_CLIP
)

df_g3_xsens_sync = plot_sync_zoom(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 3 XSens",
    time_col=TIME_COL,
    sync_label_col=SYNC_LABEL_COL,
    sync_label=SYNC_LABEL,
    users=USERS,
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    zoom_margin=ZOOM_MARGIN,
    robust_clip=ROBUST_CLIP
)


# --- CELL 49 (code cell #39) ---
# ============================================================
# GROUP 3 - RAW ZOOM AROUND SYNC REGION
# No smoothing, no robust clipping
# To check whether the rectangular peak is real or visualization artifact
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_3_openearable_labeled.csv"
)

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronaztion_move"

USERS = [1, 2, 3]


def acc_mag(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


df = pd.read_csv(OE_PATH, low_memory=False)

segments = get_label_segments(df, TIME_COL, SYNC_LABEL_COL)

sync_segments = [
    s for s in segments
    if s[2] == SYNC_LABEL
]

if len(sync_segments) == 0:
    print(df[SYNC_LABEL_COL].dropna().unique())
    raise ValueError("Sync label not found")

sync_start, sync_end, _ = sync_segments[0]
sync_mid = (sync_start + sync_end) / 2

# Manually inspect wider area
left = sync_start - 10
right = sync_end + 15

times = df[TIME_COL].values
mask = (times >= left) & (times <= right)

signals = {}
for u in USERS:
    signals[u] = acc_mag(df, u)

combined = np.mean([signals[u] for u in USERS], axis=0)

fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

axes[0].plot(times[mask], combined[mask], linewidth=0.8, label="Combined raw acc_mag")
axes[0].axvspan(sync_start, sync_end, alpha=0.25, label=SYNC_LABEL)
axes[0].axvline(sync_mid, linestyle="--", label="label middle")
axes[0].set_title("Group 3 OpenEarable - Raw combined acc_mag")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

for i, u in enumerate(USERS, start=1):
    axes[i].plot(times[mask], signals[u][mask], linewidth=0.8, label=f"P{u} raw acc_mag")
    axes[i].axvspan(sync_start, sync_end, alpha=0.25, label=SYNC_LABEL)
    axes[i].axvline(sync_mid, linestyle="--", label="label middle")
    axes[i].set_title(f"Participant {u}")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[-1].set_xlabel("Video time (seconds)")
plt.tight_layout()
plt.show()

# Find highest raw peak in this local region
local_idx = np.where(mask)[0]
peak_idx = local_idx[np.nanargmax(combined[mask])]
peak_time = times[peak_idx]

print("================================================")
print("SYNC VS LOCAL PEAK")
print("================================================")
print(f"Sync start : {sync_start:.3f}")
print(f"Sync end   : {sync_end:.3f}")
print(f"Sync middle: {sync_mid:.3f}")
print(f"Local peak : {peak_time:.3f}")
print(f"Difference peak - sync_middle: {peak_time - sync_mid:.3f} seconds")
print("================================================")


# --- CELL 50 (code cell #40) ---
# ============================================================
# GROUP 3 - XSENS RAW ZOOM AROUND SYNC REGION
# No smoothing, no robust clipping
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronaztion_move"

USERS = [1, 2, 3]


# ============================================================
# HELPERS
# ============================================================

def acc_mag_xsens(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(XSENS_PATH, low_memory=False)

segments = get_label_segments(df, TIME_COL, SYNC_LABEL_COL)

sync_segments = [
    s for s in segments
    if s[2] == SYNC_LABEL
]

if len(sync_segments) == 0:
    print("Available Whole_Group labels:")
    print(df[SYNC_LABEL_COL].dropna().unique())
    raise ValueError("Sync label not found.")

# use longest sync segment if there are multiple
sync_segments = sorted(
    sync_segments,
    key=lambda x: x[1] - x[0],
    reverse=True
)

sync_start, sync_end, _ = sync_segments[0]
sync_mid = (sync_start + sync_end) / 2


# ============================================================
# ZOOM WINDOW
# ============================================================

left = sync_start - 10
right = sync_end + 15

times = df[TIME_COL].values
mask = (times >= left) & (times <= right)


# ============================================================
# SIGNALS
# ============================================================

signals = {}

for user in USERS:
    signals[user] = acc_mag_xsens(df, user)

combined = np.mean(
    [signals[user] for user in USERS],
    axis=0
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

axes[0].plot(
    times[mask],
    combined[mask],
    linewidth=0.8,
    label="Combined raw XSens acc_mag"
)

axes[0].axvspan(
    sync_start,
    sync_end,
    alpha=0.25,
    label=SYNC_LABEL
)

axes[0].axvline(
    sync_mid,
    linestyle="--",
    label="label middle"
)

axes[0].set_title("Group 3 XSens - Raw combined acc_mag")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

for i, user in enumerate(USERS, start=1):
    axes[i].plot(
        times[mask],
        signals[user][mask],
        linewidth=0.8,
        label=f"P{user} raw XSens acc_mag"
    )

    axes[i].axvspan(
        sync_start,
        sync_end,
        alpha=0.25,
        label=SYNC_LABEL
    )

    axes[i].axvline(
        sync_mid,
        linestyle="--",
        label="label middle"
    )

    axes[i].set_title(f"Participant {user}")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[-1].set_xlabel("Video time (seconds)")

plt.tight_layout()
plt.show()


# ============================================================
# LOCAL PEAK CHECK
# ============================================================

local_idx = np.where(mask)[0]

peak_idx = local_idx[np.nanargmax(combined[mask])]
peak_time = times[peak_idx]

print("================================================")
print("XSENS SYNC VS LOCAL PEAK")
print("================================================")
print(f"Sync start : {sync_start:.3f}")
print(f"Sync end   : {sync_end:.3f}")
print(f"Sync middle: {sync_mid:.3f}")
print(f"Local peak : {peak_time:.3f}")
print(f"Difference peak - sync_middle: {peak_time - sync_mid:.3f} seconds")
print("================================================")


# --- CELL 51 (code cell #41) ---
# ============================================================
# GROUP 3 - SHIFT BOTH OPENEARAMBLE AND XSENS LABELS
# Based on OpenEarable raw local peak around synchronaztion_move
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_3_openearable_labeled.csv"
)

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_3_openearable_labeled_shifted_by_sync_peak.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled_shifted_by_sync_peak.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronaztion_move"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# Search for the movement peak in this window around the sync label
SEARCH_BEFORE_SYNC_S = 5
SEARCH_AFTER_SYNC_S = 10

# Plot window
ZOOM_MARGIN = 15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def acc_mag(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def combined_acc_mag(df):
    mags = []

    for user in USERS:
        mags.append(acc_mag(df, user))

    return np.mean(mags, axis=0)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []

    active_label = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def find_label_segment(df, time_col, label_col, target_label):
    segments = get_label_segments(df, time_col, label_col)

    matched = [
        seg for seg in segments
        if seg[2] == target_label
    ]

    if len(matched) == 0:
        print("Available labels:")
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find label: {target_label}")

    # If multiple, use longest
    matched = sorted(
        matched,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return matched[0]


def find_local_peak_near_sync(df):
    times = df[TIME_COL].values

    sync_start, sync_end, _ = find_label_segment(
        df,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    sync_middle = (sync_start + sync_end) / 2

    signal = combined_acc_mag(df)

    search_left = sync_start - SEARCH_BEFORE_SYNC_S
    search_right = sync_end + SEARCH_AFTER_SYNC_S

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in local search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    offset_sec = peak_time - sync_middle

    return {
        "sync_start": sync_start,
        "sync_end": sync_end,
        "sync_middle": sync_middle,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "offset_sec": offset_sec
    }


def shift_label_column(df, time_col, label_col, offset_sec):
    times = df[time_col].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, time_col, label_col)

    for start, end, label in segments:
        new_start = start + offset_sec
        new_end = end + offset_sec

        mask = (times >= new_start) & (times < new_end)
        shifted[mask] = label

    return shifted


def shift_all_labels(df, label_cols, offset_sec):
    shifted_df = df.copy()

    for col in label_cols:
        if col in shifted_df.columns:
            shifted_df[col] = shift_label_column(
                shifted_df,
                TIME_COL,
                col,
                offset_sec
            )

    return shifted_df


def plot_before_after(df_before, df_after, title, source_name, offset_info):
    times = df_before[TIME_COL].values

    sig_before = combined_acc_mag(df_before)

    old_start = offset_info["sync_start"]
    old_end = offset_info["sync_end"]
    old_mid = offset_info["sync_middle"]
    peak_time = offset_info["peak_time"]

    new_start, new_end, _ = find_label_segment(
        df_after,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    new_mid = (new_start + new_end) / 2

    left = min(old_start, new_start, peak_time) - ZOOM_MARGIN
    right = max(old_end, new_end, peak_time) + ZOOM_MARGIN

    mask = (times >= left) & (times <= right)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    # Before
    axes[0].plot(
        times[mask],
        sig_before[mask],
        linewidth=0.8,
        label=f"{source_name} combined raw acc_mag"
    )

    axes[0].axvspan(
        old_start,
        old_end,
        alpha=0.25,
        label="Original sync label"
    )

    axes[0].axvline(
        old_mid,
        linestyle="--",
        label="Original sync middle"
    )

    axes[0].axvline(
        peak_time,
        linestyle="--",
        label="Detected local peak"
    )

    axes[0].set_title(f"{title} - Before shifting")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # After
    axes[1].plot(
        times[mask],
        sig_before[mask],
        linewidth=0.8,
        label=f"{source_name} combined raw acc_mag"
    )

    axes[1].axvspan(
        new_start,
        new_end,
        alpha=0.25,
        label="Shifted sync label"
    )

    axes[1].axvline(
        new_mid,
        linestyle="--",
        label="Shifted sync middle"
    )

    axes[1].axvline(
        peak_time,
        linestyle="--",
        label="Detected local peak"
    )

    axes[1].set_title(f"{title} - After shifting")
    axes[1].set_xlabel("Video time (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# ============================================================
# LOAD FILES
# ============================================================

oe = pd.read_csv(OE_IN, low_memory=False)
xsens = pd.read_csv(XSENS_IN, low_memory=False)


# ============================================================
# FIND OFFSET FROM OPENEARAMBLE
# ============================================================

offset_info = find_local_peak_near_sync(oe)

offset_sec = offset_info["offset_sec"]

print("================================================")
print("GROUP 3 LABEL SHIFT BASED ON OPENEARAMBLE SYNC PEAK")
print("================================================")
print(f"Original sync start  : {offset_info['sync_start']:.3f} s")
print(f"Original sync end    : {offset_info['sync_end']:.3f} s")
print(f"Original sync middle : {offset_info['sync_middle']:.3f} s")
print(f"Detected peak time   : {offset_info['peak_time']:.3f} s")
print(f"Detected peak value  : {offset_info['peak_value']:.3f}")
print("------------------------------------------------")
print(f"Offset to apply      : {offset_sec:.3f} s")

if offset_sec > 0:
    print("Labels will move LATER.")
elif offset_sec < 0:
    print("Labels will move EARLIER.")
else:
    print("No shift needed.")

print("================================================")


# ============================================================
# SHIFT BOTH FILES
# ============================================================

oe_shifted = shift_all_labels(
    oe,
    LABEL_COLS,
    offset_sec
)

xsens_shifted = shift_all_labels(
    xsens,
    LABEL_COLS,
    offset_sec
)


# ============================================================
# SAVE
# ============================================================

oe_shifted.to_csv(OE_OUT, index=False)
xsens_shifted.to_csv(XSENS_OUT, index=False)

print("\nSaved shifted OpenEarable:")
print(OE_OUT)

print("\nSaved shifted XSens:")
print(XSENS_OUT)


# ============================================================
# VERIFY NEW SYNC POSITION
# ============================================================

oe_new_start, oe_new_end, _ = find_label_segment(
    oe_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

xs_new_start, xs_new_end, _ = find_label_segment(
    xsens_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

print("\n================================================")
print("CHECK SHIFTED SYNC POSITION")
print("================================================")
print("OpenEarable shifted sync:")
print(f"{oe_new_start:.3f} -> {oe_new_end:.3f}, middle={(oe_new_start + oe_new_end)/2:.3f}")

print("\nXSens shifted sync:")
print(f"{xs_new_start:.3f} -> {xs_new_end:.3f}, middle={(xs_new_start + xs_new_end)/2:.3f}")

print("\nDetected OE peak:")
print(f"{offset_info['peak_time']:.3f}")

print("================================================")


# ============================================================
# PLOT BEFORE / AFTER
# ============================================================

plot_before_after(
    df_before=oe,
    df_after=oe_shifted,
    title="Group 3 OpenEarable",
    source_name="OpenEarable",
    offset_info=offset_info
)

plot_before_after(
    df_before=xsens,
    df_after=xsens_shifted,
    title="Group 3 XSens",
    source_name="XSens",
    offset_info=offset_info
)


# --- CELL 52 (code cell #42) ---
# ============================================================
# GROUP 3 - SHIFT XSENS LABELS LEFT USING XSENS LOCAL PEAK
# OpenEarable already looks good, so this only fixes XSens.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_3"

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_3_xsens_labeled_shifted_by_xsens_sync_peak.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronaztion_move"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# For XSens, the peak is before the label, so search wider on the left
SEARCH_BEFORE_SYNC_S = 12
SEARCH_AFTER_SYNC_S = 4

ZOOM_MARGIN = 18


# ============================================================
# HELPERS
# ============================================================

def acc_mag_xsens(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def combined_acc_mag_xsens(df):
    return np.mean(
        [acc_mag_xsens(df, user) for user in USERS],
        axis=0
    )


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active is None:
            active = label
            start_idx = i

        elif label != "" and active is not None and label != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = label
            start_idx = i

        elif label == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_label_segment(df, time_col, label_col, target_label):
    segments = get_label_segments(df, time_col, label_col)

    matched = [
        seg for seg in segments
        if seg[2] == target_label
    ]

    if len(matched) == 0:
        print("Available labels:")
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find label: {target_label}")

    # use longest if multiple
    matched = sorted(
        matched,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return matched[0]


def find_xsens_local_peak(df):
    times = df[TIME_COL].values

    sync_start, sync_end, _ = find_label_segment(
        df,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    sync_middle = (sync_start + sync_end) / 2

    signal = combined_acc_mag_xsens(df)

    search_left = sync_start - SEARCH_BEFORE_SYNC_S
    search_right = sync_end + SEARCH_AFTER_SYNC_S

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    offset_sec = peak_time - sync_middle

    return {
        "sync_start": sync_start,
        "sync_end": sync_end,
        "sync_middle": sync_middle,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "offset_sec": offset_sec
    }


def shift_label_column(df, time_col, label_col, offset_sec):
    times = df[time_col].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, time_col, label_col)

    for start, end, label in segments:
        new_start = start + offset_sec
        new_end = end + offset_sec

        mask = (times >= new_start) & (times < new_end)
        shifted[mask] = label

    return shifted


def shift_all_labels(df, label_cols, offset_sec):
    shifted_df = df.copy()

    for col in label_cols:
        if col in shifted_df.columns:
            shifted_df[col] = shift_label_column(
                shifted_df,
                TIME_COL,
                col,
                offset_sec
            )

    return shifted_df


def plot_xsens_before_after(df_before, df_after, offset_info):
    times = df_before[TIME_COL].values
    signal = combined_acc_mag_xsens(df_before)

    old_start = offset_info["sync_start"]
    old_end = offset_info["sync_end"]
    old_mid = offset_info["sync_middle"]
    peak_time = offset_info["peak_time"]

    new_start, new_end, _ = find_label_segment(
        df_after,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    new_mid = (new_start + new_end) / 2

    left = min(old_start, new_start, peak_time) - ZOOM_MARGIN
    right = max(old_end, new_end, peak_time) + ZOOM_MARGIN

    mask = (times >= left) & (times <= right)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axes[0].plot(
        times[mask],
        signal[mask],
        linewidth=0.8,
        label="XSens combined raw acc_mag"
    )

    axes[0].axvspan(
        old_start,
        old_end,
        alpha=0.25,
        label="Original sync label"
    )

    axes[0].axvline(
        old_mid,
        linestyle="--",
        label="Original sync middle"
    )

    axes[0].axvline(
        peak_time,
        linestyle="--",
        label="Detected XSens local peak"
    )

    axes[0].set_title("Group 3 XSens - Before shifting")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(
        times[mask],
        signal[mask],
        linewidth=0.8,
        label="XSens combined raw acc_mag"
    )

    axes[1].axvspan(
        new_start,
        new_end,
        alpha=0.25,
        label="Shifted sync label"
    )

    axes[1].axvline(
        new_mid,
        linestyle="--",
        label="Shifted sync middle"
    )

    axes[1].axvline(
        peak_time,
        linestyle="--",
        label="Detected XSens local peak"
    )

    axes[1].set_title("Group 3 XSens - After shifting LEFT/RIGHT based on XSens")
    axes[1].set_xlabel("Video time (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

xsens = pd.read_csv(XSENS_IN, low_memory=False)

offset_info = find_xsens_local_peak(xsens)
offset_sec = offset_info["offset_sec"]

print("================================================")
print("GROUP 3 XSENS LABEL SHIFT BASED ON XSENS LOCAL PEAK")
print("================================================")
print(f"Original sync start  : {offset_info['sync_start']:.3f} s")
print(f"Original sync end    : {offset_info['sync_end']:.3f} s")
print(f"Original sync middle : {offset_info['sync_middle']:.3f} s")
print(f"Detected XSens peak  : {offset_info['peak_time']:.3f} s")
print(f"Peak value           : {offset_info['peak_value']:.3f}")
print("------------------------------------------------")
print(f"Offset to apply      : {offset_sec:.3f} s")

if offset_sec < 0:
    print("Labels will move EARLIER / LEFT.")
elif offset_sec > 0:
    print("Labels will move LATER / RIGHT.")
else:
    print("No shift needed.")

print("================================================")

xsens_shifted = shift_all_labels(
    xsens,
    LABEL_COLS,
    offset_sec
)

xsens_shifted.to_csv(XSENS_OUT, index=False)

print("\nSaved corrected shifted XSens:")
print(XSENS_OUT)

new_start, new_end, _ = find_label_segment(
    xsens_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

print("\nShifted sync label:")
print(f"{new_start:.3f} -> {new_end:.3f}")
print(f"Shifted middle: {(new_start + new_end) / 2:.3f}")
print(f"Detected peak : {offset_info['peak_time']:.3f}")
print(f"Difference    : {(new_start + new_end) / 2 - offset_info['peak_time']:.6f} s")

plot_xsens_before_after(
    xsens,
    xsens_shifted,
    offset_info
)


# --- CELL 54 (code cell #43) ---
import pandas as pd, numpy as np, os

# ---- Group 5 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_5/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_5/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

# value columns per sensor
# trailing flag column is dropped
SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000   # 50 Hz target grid
TOL_FAST_US = 12_000   # tolerance for 50 Hz streams
TOL_SLOW_US = 40_000   # wider tolerance for slower streams


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + value columns, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    # this clips possible warm-up samples from other streams
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 55 (code cell #44) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_5/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_5"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load every participant's 8 raw streams
# and find each participant's session window
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Shared grid = overlap window
# latest participant start -> earliest participant end
# ============================================================

start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge every participant and every sensor onto shared 50 Hz grid
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_5_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_5_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 56 (code cell #45) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_5/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_5"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]


def load(p):
    path = os.path.join(BASE, f"{p}.csv")

    d = pd.read_csv(path, low_memory=False)

    # drop trailing empty columns if they exist
    d = d.loc[:, [c for c in d.columns if str(c).strip() != ""]]

    # clean column names
    d.columns = [str(c).strip() for c in d.columns]

    # convert important columns to numeric
    for c in ["SampleTimeFine", "PacketCounter"] + SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")

    return d


# ============================================================
# Load raw XSens files
# ============================================================

raw = {
    p: load(p)
    for p in PREFIX
}

for p, d in raw.items():
    print(
        p,
        "| rows:", len(d),
        "| SampleTimeFine valid:", d["SampleTimeFine"].notna().sum()
    )


# ============================================================
# Master grid = Participant1 clean SampleTimeFine
# ============================================================

p1 = raw["Participant1"].dropna(subset=["SampleTimeFine"])
p1_ticks = set(p1["SampleTimeFine"].tolist())


# ============================================================
# Common span across all 3 participants
# using only ticks that exist on P1 grid
# ============================================================

spans = []

for p, d in raw.items():
    valid = d[d["SampleTimeFine"].isin(p1_ticks)]

    span = (
        int(valid["SampleTimeFine"].min()),
        int(valid["SampleTimeFine"].max())
    )

    spans.append(span)

    print(
        f"{p} span:",
        span[0],
        "->",
        span[1],
        "| duration:",
        round((span[1] - span[0]) / 1e6, 1),
        "s"
    )


start = max(s[0] for s in spans)
end   = min(s[1] for s in spans)


# ============================================================
# Build master timeline
# ============================================================

master = (
    p1[
        (p1["SampleTimeFine"] >= start) &
        (p1["SampleTimeFine"] <= end)
    ][["SampleTimeFine"]]
    .sort_values("SampleTimeFine")
    .reset_index(drop=True)
)

master["time_s"] = (
    (master["SampleTimeFine"].astype(np.int64) - start) / 1e6
).round(4)


# ============================================================
# Merge all participants onto master grid
# ============================================================

wide = master.copy()

for p, d in raw.items():
    pre = PREFIX[p]

    sub = (
        d[["SampleTimeFine"] + SENSOR_COLS]
        .dropna(subset=["SampleTimeFine"])
        .drop_duplicates("SampleTimeFine")
        .rename(columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        })
    )

    wide = wide.merge(
        sub,
        on="SampleTimeFine",
        how="left"
    )


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_5_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| dur:",
    round(wide["time_s"].iloc[-1], 1),
    "s ->",
    local_path
)

os.makedirs(DRIVE_OUT, exist_ok=True)

dst = os.path.join(
    DRIVE_OUT,
    "group_5_xsens_merged_30hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing check
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 57 (code cell #46) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_with_individual_build.csv"
LOCAL_OUT = "/content/Group_5_with_individual_build.csv"
# ----------------

# ---- Group 5 participant mapping ----
# Mintan = Participant1
# Adarsh = Participant2
# Ali    = Participant3
PEOPLE = ["Mintan", "Adarsh", "Ali"]

# ---- manual individual_build cutoff times ----
CUTOFF = {
    "Mintan": 49 * 60 + 25,   # 2965 s
    "Adarsh": 34 * 60 + 7,    # 2047 s
    "Ali":    34 * 60 + 7,    # 2047 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


# ============================================================
# Helper: normalize names for safer matching
# ============================================================

def norm(x):
    return str(x).strip().lower()


# ============================================================
# Involving tiers
# Includes:
# own tier + dyads containing person + Whole Group / Whole_Group
# Handles both spaces and underscores
# ============================================================

INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        # handles "Mintan Adarsh", "Mintan_Adarsh", etc.
        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


# ============================================================
# Interval helpers
# ============================================================

def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ============================================================
# Add individual_build rows
# ============================================================

new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)


# ============================================================
# Save same ELAN-style CSV
# Still uses real names; next cell will rename to Participant1/2/3
# ============================================================

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 5 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 58 (code cell #47) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_5_individual_build_renamed.csv"
# ----------------

# Group 5 mapping
NAME = {
    "Mintan": "Participant1",
    "Adarsh": "Participant2",
    "Ali": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    # handles both "Adarsh Mintan" and "Adarsh_Mintan"
    parts_original = t.replace("_", " ").split()

    # map case-insensitively
    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 59 (code cell #48) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 5 - LABEL OPENEARAMBLE USING VIDEO TIME
# ============================================================

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_5/openearable_merged/group_5_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/openearable_labeled/group_5_openearable_labeled.csv"
LOCAL_OUT = "/content/group_5_openearable_labeled.csv"
# ----------------

# Group 5 video start:
# 2026-04-23 11:08:44 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-23 11:08:44", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

# ============================================================
# LOAD
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

oe = pd.read_csv(
    OE_PATH,
    low_memory=False
)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# ============================================================
# CREATE VIDEO TIME FOR OPENEARAMBLE
# ============================================================

oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder labels
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

# remove old full label columns if rerunning
oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# place video_time_s right after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# ============================================================
# ADD ELAN LABELS TO OPENEARAMBLE TIMELINE
# ============================================================

for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# ============================================================
# SAVE
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

# ============================================================
# CHECK
# ============================================================

print("================================================")
print("GROUP 5 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 60 (code cell #49) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 5 - LABEL XSENS USING VIDEO TIME
# XSens starts 206s after video
# Therefore: video_time_s = time_s + 206
# ============================================================

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_5/xsens_merged/group_5_xsens_merged_30hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_5/xsens_labeled/group_5_xsens_labeled.csv"
LOCAL_OUT = "/content/group_5_xsens_labeled.csv"
# ----------------

XSENS_TO_VIDEO_OFFSET_S = 206.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

# ============================================================
# LOAD
# ============================================================

elan = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=ECOLS
)

xsens = pd.read_csv(
    XSENS_PATH,
    low_memory=False
)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# ============================================================
# CREATE VIDEO-RELATIVE TIME FOR XSENS
# ============================================================

xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder labels
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

# remove old full label columns if rerunning
xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s right after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# ============================================================
# ADD ELAN LABELS TO XSENS TIMELINE
# ============================================================

for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# ============================================================
# SAVE
# ============================================================

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)

xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

# ============================================================
# CHECK
# ============================================================

print("================================================")
print("GROUP 5 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 61 (code cell #50) ---
# ============================================================
# GROUP 5 - VISUAL SANITY CHECK
# Full label visualization for OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_5"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_5_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_5_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 5

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]

    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be 'oe' or 'xsens'")

    missing = [c for c in [x, y, z] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for participant {user}: {missing}")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active_label = None
    start_idx = None

    for i, label in enumerate(labels):
        if label != "" and active_label is None:
            active_label = label
            start_idx = i

        elif label != "" and active_label is not None and label != active_label:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = label
            start_idx = i

        elif label == "" and active_label is not None:
            segments.append((times[start_idx], times[i - 1], active_label))
            active_label = None
            start_idx = None

    if active_label is not None:
        segments.append((times[start_idx], times[-1], active_label))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_cols=None,
    label_alpha=0.14,
    text_every_n_segments=5,
    robust_y=True,
    figsize=(24, 12)
):
    df = pd.read_csv(csv_path, low_memory=False)

    if label_cols is None:
        label_cols = LABEL_COLS

    existing_label_cols = [col for col in label_cols if col in df.columns]

    if len(existing_label_cols) == 0:
        raise ValueError("No label columns found.")

    if TIME_COL not in df.columns:
        raise ValueError(f"{TIME_COL} not found.")

    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(len(USERS), 1, figsize=figsize, sharex=True)

    if len(USERS) == 1:
        axes = [axes]

    for ax, user in zip(axes, USERS):

        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN FULL VISUAL CHECKS
# ============================================================

df_group5_oe_full = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 5 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_cols=LABEL_COLS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM,
    figsize=FIGSIZE
)

df_group5_xsens_full = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 5 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_cols=LABEL_COLS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM,
    figsize=FIGSIZE
)


# --- CELL 62 (code cell #51) ---
import pandas as pd

XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_5/xsens_labeled/group_5_xsens_labeled.csv"

df_xsens = pd.read_csv(XSENS_PATH, low_memory=False)

print(df_xsens["video_time_s"].diff().describe())

print("\nLargest time gaps:")
print(df_xsens["video_time_s"].diff().nlargest(20))


# --- CELL 63 (code cell #52) ---
gap = df_xsens["video_time_s"].diff()
biggest_gap_idx = gap.idxmax()

print("Biggest gap index:", biggest_gap_idx)
print("Gap size:", gap.loc[biggest_gap_idx], "seconds")

print("\nBefore / after gap:")
display(
    df_xsens.loc[
        biggest_gap_idx-5:biggest_gap_idx+5,
        ["SampleTimeFine", "time_s", "video_time_s"]
    ]
)

print("\nGap starts after:")
print(df_xsens.loc[biggest_gap_idx-1, "video_time_s"])

print("\nGap ends at:")
print(df_xsens.loc[biggest_gap_idx, "video_time_s"])


# --- CELL 64 (code cell #53) ---
import pandas as pd
import numpy as np
import os

XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_5/xsens_labeled/group_5_xsens_labeled.csv"

OUT_PATH = "/content/drive/MyDrive/thesis/data/group_5/xsens_labeled/group_5_xsens_labeled_with_gap_flag.csv"

df = pd.read_csv(XSENS_PATH, low_memory=False)

df["dt_s"] = df["video_time_s"].diff()

# Mark rows where a new continuous segment starts.
# Normal XSens is ~0.033s, so anything >1s is a real break.
df["new_segment_after_gap"] = df["dt_s"] > 1.0

# Assign continuous segment IDs
df["segment_id"] = df["new_segment_after_gap"].cumsum()

df.to_csv(OUT_PATH, index=False)

print("Saved:")
print(OUT_PATH)

print("\nSegments:")
print(df.groupby("segment_id")["video_time_s"].agg(["min", "max", "count"]))

print("\nDetected gaps:")
print(df[df["new_segment_after_gap"]][["video_time_s", "dt_s", "segment_id"]])


# --- CELL 65 (code cell #54) ---
# ============================================================
# GROUP 5 - ZOOM AROUND SYNC ACTIONS
# OpenEarable + XSens
# Labels:
# - clap_synchronizaiton_move
# - synchronizaiton_move
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_5"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_5_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_5_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"

SYNC_LABELS = [
    "clap_synchronizaiton_move",
    "synchronizaiton_move"
]

USERS = [1, 2, 3]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

ZOOM_MARGIN = 25

# Raw view is better for deciding shift
SMOOTH = False
SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

ROBUST_CLIP = False


# ============================================================
# HELPERS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_clip_for_plot(signal, lower_p=1, upper_p=99):
    signal = np.asarray(signal, dtype=float)
    finite = signal[np.isfinite(signal)]

    if len(finite) == 0:
        return signal

    low = np.percentile(finite, lower_p)
    high = np.percentile(finite, upper_p)

    return np.clip(signal, low, high)


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_segments_for_label(df, label_name):
    segments = get_label_segments(df, TIME_COL, SYNC_LABEL_COL)

    return [
        seg for seg in segments
        if seg[2] == label_name
    ]


def plot_sync_label(
    csv_path,
    source,
    title,
    sync_label,
    smooth=False,
    smooth_window=15,
    robust_clip=False,
    zoom_margin=25
):
    df = pd.read_csv(csv_path, low_memory=False)

    segments = find_segments_for_label(df, sync_label)

    print("\n================================================")
    print(title)
    print("Label:", sync_label)
    print("================================================")

    if len(segments) == 0:
        print("Label not found.")
        print("Available Whole_Group labels:")
        print(df[SYNC_LABEL_COL].dropna().unique())
        return None

    for i, (s, e, lab) in enumerate(segments):
        print(f"{i}: {s:.3f}s -> {e:.3f}s | dur={e-s:.3f}s")

    # If multiple, plot all
    for event_i, (sync_start, sync_end, lab) in enumerate(segments):

        sync_mid = (sync_start + sync_end) / 2

        times = df[TIME_COL].values

        left = sync_start - zoom_margin
        right = sync_end + zoom_margin

        mask = (times >= left) & (times <= right)

        signals = {}

        for user in USERS:
            sig = get_signal(df, user, source, SIGNAL_TYPE)

            if smooth:
                sig = moving_average(sig, smooth_window)

            if robust_clip:
                sig = robust_clip_for_plot(sig)

            signals[user] = sig

        combined = np.mean(
            [signals[user] for user in USERS],
            axis=0
        )

        # local peak in the zoom window
        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = times[peak_idx]
        peak_value = combined[peak_idx]
        offset = peak_time - sync_mid

        print(
            f"Event {event_i} local peak: {peak_time:.3f}s | "
            f"value={peak_value:.3f} | peak - label_middle = {offset:.3f}s"
        )

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(
            times[mask],
            combined[mask],
            linewidth=0.8,
            label=f"Combined {source} {SIGNAL_TYPE}"
        )

        axes[0].axvspan(
            sync_start,
            sync_end,
            alpha=0.25,
            label=sync_label
        )

        axes[0].axvline(
            sync_mid,
            linestyle="--",
            label="label middle"
        )

        axes[0].axvline(
            peak_time,
            linestyle="--",
            label="local peak"
        )

        axes[0].set_title(f"{title} | {sync_label} | event {event_i} | Combined")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        for i, user in enumerate(USERS, start=1):
            axes[i].plot(
                times[mask],
                signals[user][mask],
                linewidth=0.8,
                label=f"P{user} {source} {SIGNAL_TYPE}"
            )

            axes[i].axvspan(
                sync_start,
                sync_end,
                alpha=0.25,
                label=sync_label
            )

            axes[i].axvline(
                sync_mid,
                linestyle="--",
                label="label middle"
            )

            axes[i].axvline(
                peak_time,
                linestyle="--",
                label="local peak"
            )

            axes[i].set_title(f"{title} | Participant {user}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper left")

        axes[-1].set_xlabel("Video time (seconds)")

        plt.tight_layout()
        plt.show()

    return df


# ============================================================
# RUN
# ============================================================

for label in SYNC_LABELS:

    plot_sync_label(
        csv_path=OE_PATH,
        source="oe",
        title="Group 5 OpenEarable",
        sync_label=label,
        smooth=SMOOTH,
        smooth_window=SMOOTH_WINDOW_OE,
        robust_clip=ROBUST_CLIP,
        zoom_margin=ZOOM_MARGIN
    )

    plot_sync_label(
        csv_path=XSENS_PATH,
        source="xsens",
        title="Group 5 XSens",
        sync_label=label,
        smooth=SMOOTH,
        smooth_window=SMOOTH_WINDOW_XSENS,
        robust_clip=ROBUST_CLIP,
        zoom_margin=ZOOM_MARGIN
    )


# --- CELL 66 (code cell #55) ---
# ============================================================
# GROUP 5 - SHIFT XSENS LABELS USING clap_synchronizaiton_move
# Peak detected from Participant 3 XSens acc_mag
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_5"

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_5_xsens_labeled.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_5_xsens_labeled_shifted_by_clap_sync_peak.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "clap_synchronizaiton_move"

# We use Participant 3 because the clap/sync peak is clearest there
PEAK_USER = 3

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# Search around the label.
# Since the peak looks earlier than the current label, search more before.
SEARCH_BEFORE_SYNC_S = 15
SEARCH_AFTER_SYNC_S = 5

ZOOM_MARGIN = 25


# ============================================================
# HELPERS
# ============================================================

def acc_mag_xsens(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active is None:
            active = label
            start_idx = i

        elif label != "" and active is not None and label != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = label
            start_idx = i

        elif label == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_label_segment(df, time_col, label_col, target_label):
    segments = get_label_segments(df, time_col, label_col)

    matched = [
        seg for seg in segments
        if seg[2] == target_label
    ]

    if len(matched) == 0:
        print("Available Whole_Group labels:")
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find label: {target_label}")

    # if multiple, use longest
    matched = sorted(
        matched,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return matched[0]


def find_xsens_peak_near_clap(df):
    times = df[TIME_COL].values

    sync_start, sync_end, _ = find_label_segment(
        df,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    sync_middle = (sync_start + sync_end) / 2

    # Use P3 signal for peak detection
    signal = acc_mag_xsens(df, PEAK_USER)

    search_left = sync_start - SEARCH_BEFORE_SYNC_S
    search_right = sync_end + SEARCH_AFTER_SYNC_S

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in local search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    offset_sec = peak_time - sync_middle

    return {
        "sync_start": sync_start,
        "sync_end": sync_end,
        "sync_middle": sync_middle,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "offset_sec": offset_sec
    }


def shift_label_column(df, time_col, label_col, offset_sec):
    times = df[time_col].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, time_col, label_col)

    for start, end, label in segments:
        new_start = start + offset_sec
        new_end = end + offset_sec

        mask = (times >= new_start) & (times < new_end)
        shifted[mask] = label

    return shifted


def shift_all_labels(df, label_cols, offset_sec):
    shifted_df = df.copy()

    for col in label_cols:
        if col in shifted_df.columns:
            shifted_df[col] = shift_label_column(
                shifted_df,
                TIME_COL,
                col,
                offset_sec
            )

    return shifted_df


def plot_xsens_before_after(df_before, df_after, offset_info):
    times = df_before[TIME_COL].values

    old_start = offset_info["sync_start"]
    old_end = offset_info["sync_end"]
    old_mid = offset_info["sync_middle"]
    peak_time = offset_info["peak_time"]

    new_start, new_end, _ = find_label_segment(
        df_after,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    new_mid = (new_start + new_end) / 2

    left = min(old_start, new_start, peak_time) - ZOOM_MARGIN
    right = max(old_end, new_end, peak_time) + ZOOM_MARGIN

    mask = (times >= left) & (times <= right)

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

    # combined signal for view
    combined = np.mean(
        [acc_mag_xsens(df_before, u) for u in USERS],
        axis=0
    )

    axes[0].plot(
        times[mask],
        combined[mask],
        linewidth=0.8,
        label="Combined XSens raw acc_mag"
    )

    axes[0].axvspan(old_start, old_end, alpha=0.20, label="Original clap sync label")
    axes[0].axvspan(new_start, new_end, alpha=0.20, label="Shifted clap sync label")
    axes[0].axvline(old_mid, linestyle="--", label="Original middle")
    axes[0].axvline(new_mid, linestyle="--", label="Shifted middle")
    axes[0].axvline(peak_time, linestyle="--", label=f"P{PEAK_USER} detected peak")

    axes[0].set_title("Group 5 XSens - Combined before/after label shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    for i, user in enumerate(USERS, start=1):
        sig = acc_mag_xsens(df_before, user)

        axes[i].plot(
            times[mask],
            sig[mask],
            linewidth=0.8,
            label=f"P{user} XSens raw acc_mag"
        )

        axes[i].axvspan(old_start, old_end, alpha=0.20, label="Original label")
        axes[i].axvspan(new_start, new_end, alpha=0.20, label="Shifted label")
        axes[i].axvline(old_mid, linestyle="--", label="Original middle")
        axes[i].axvline(new_mid, linestyle="--", label="Shifted middle")
        axes[i].axvline(peak_time, linestyle="--", label=f"P{PEAK_USER} detected peak")

        axes[i].set_title(f"Participant {user}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Video time (seconds)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

xsens = pd.read_csv(XSENS_IN, low_memory=False)

offset_info = find_xsens_peak_near_clap(xsens)
offset_sec = offset_info["offset_sec"]

print("================================================")
print("GROUP 5 XSENS LABEL SHIFT BASED ON clap_synchronizaiton_move")
print("================================================")
print(f"Peak detection participant : P{PEAK_USER}")
print(f"Original clap start        : {offset_info['sync_start']:.3f} s")
print(f"Original clap end          : {offset_info['sync_end']:.3f} s")
print(f"Original clap middle       : {offset_info['sync_middle']:.3f} s")
print(f"Detected P{PEAK_USER} peak : {offset_info['peak_time']:.3f} s")
print(f"Peak value                 : {offset_info['peak_value']:.3f}")
print("------------------------------------------------")
print(f"Offset to apply            : {offset_sec:.3f} s")

if offset_sec < 0:
    print("Labels will move EARLIER / LEFT.")
elif offset_sec > 0:
    print("Labels will move LATER / RIGHT.")
else:
    print("No shift needed.")

print("================================================")

xsens_shifted = shift_all_labels(
    xsens,
    LABEL_COLS,
    offset_sec
)

xsens_shifted.to_csv(XSENS_OUT, index=False)

print("\nSaved shifted XSens file:")
print(XSENS_OUT)

new_start, new_end, _ = find_label_segment(
    xsens_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

print("\nShifted clap sync:")
print(f"{new_start:.3f} -> {new_end:.3f}")
print(f"Shifted middle: {(new_start + new_end) / 2:.3f}")
print(f"Detected peak : {offset_info['peak_time']:.3f}")
print(f"Difference    : {(new_start + new_end) / 2 - offset_info['peak_time']:.6f} s")

plot_xsens_before_after(
    xsens,
    xsens_shifted,
    offset_info
)


# --- CELL 67 (code cell #56) ---
# ============================================================
# GROUP 5 - SHIFT OPENEARAMBLE LABELS USING synchronizaiton_move
# Peak detected from OpenEarable combined acc_mag
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_5"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_5_openearable_labeled.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_5_openearable_labeled_shifted_by_sync_peak.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# Search around sync label
# Adjust if the detected peak is not the one you want
SEARCH_BEFORE_SYNC_S = 15
SEARCH_AFTER_SYNC_S = 15

ZOOM_MARGIN = 25


# ============================================================
# HELPERS
# ============================================================

def acc_mag_oe(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )


def combined_acc_mag_oe(df):
    return np.mean(
        [acc_mag_oe(df, user) for user in USERS],
        axis=0
    )


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, label in enumerate(labels):

        if label != "" and active is None:
            active = label
            start_idx = i

        elif label != "" and active is not None and label != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = label
            start_idx = i

        elif label == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_label_segment(df, time_col, label_col, target_label):
    segments = get_label_segments(df, time_col, label_col)

    matched = [
        seg for seg in segments
        if seg[2] == target_label
    ]

    if len(matched) == 0:
        print("Available Whole_Group labels:")
        print(df[label_col].dropna().unique())
        raise ValueError(f"Could not find label: {target_label}")

    # if multiple, use longest
    matched = sorted(
        matched,
        key=lambda x: x[1] - x[0],
        reverse=True
    )

    return matched[0]


def find_oe_peak_near_sync(df):
    times = df[TIME_COL].values

    sync_start, sync_end, _ = find_label_segment(
        df,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    sync_middle = (sync_start + sync_end) / 2

    signal = combined_acc_mag_oe(df)

    search_left = sync_start - SEARCH_BEFORE_SYNC_S
    search_right = sync_end + SEARCH_AFTER_SYNC_S

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in local search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    offset_sec = peak_time - sync_middle

    return {
        "sync_start": sync_start,
        "sync_end": sync_end,
        "sync_middle": sync_middle,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "offset_sec": offset_sec
    }


def shift_label_column(df, time_col, label_col, offset_sec):
    times = df[time_col].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, time_col, label_col)

    for start, end, label in segments:
        new_start = start + offset_sec
        new_end = end + offset_sec

        mask = (times >= new_start) & (times < new_end)
        shifted[mask] = label

    return shifted


def shift_all_labels(df, label_cols, offset_sec):
    shifted_df = df.copy()

    for col in label_cols:
        if col in shifted_df.columns:
            shifted_df[col] = shift_label_column(
                shifted_df,
                TIME_COL,
                col,
                offset_sec
            )

    return shifted_df


def plot_oe_before_after(df_before, df_after, offset_info):
    times = df_before[TIME_COL].values
    signal = combined_acc_mag_oe(df_before)

    old_start = offset_info["sync_start"]
    old_end = offset_info["sync_end"]
    old_mid = offset_info["sync_middle"]
    peak_time = offset_info["peak_time"]

    new_start, new_end, _ = find_label_segment(
        df_after,
        TIME_COL,
        SYNC_LABEL_COL,
        SYNC_LABEL
    )

    new_mid = (new_start + new_end) / 2

    left = min(old_start, new_start, peak_time) - ZOOM_MARGIN
    right = max(old_end, new_end, peak_time) + ZOOM_MARGIN

    mask = (times >= left) & (times <= right)

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

    axes[0].plot(
        times[mask],
        signal[mask],
        linewidth=0.8,
        label="Combined OpenEarable raw acc_mag"
    )

    axes[0].axvspan(old_start, old_end, alpha=0.20, label="Original sync label")
    axes[0].axvspan(new_start, new_end, alpha=0.20, label="Shifted sync label")
    axes[0].axvline(old_mid, linestyle="--", label="Original middle")
    axes[0].axvline(new_mid, linestyle="--", label="Shifted middle")
    axes[0].axvline(peak_time, linestyle="--", label="Detected OE local peak")

    axes[0].set_title("Group 5 OpenEarable - Combined before/after label shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    for i, user in enumerate(USERS, start=1):
        sig = acc_mag_oe(df_before, user)

        axes[i].plot(
            times[mask],
            sig[mask],
            linewidth=0.8,
            label=f"P{user} OpenEarable raw acc_mag"
        )

        axes[i].axvspan(old_start, old_end, alpha=0.20, label="Original label")
        axes[i].axvspan(new_start, new_end, alpha=0.20, label="Shifted label")
        axes[i].axvline(old_mid, linestyle="--", label="Original middle")
        axes[i].axvline(new_mid, linestyle="--", label="Shifted middle")
        axes[i].axvline(peak_time, linestyle="--", label="Detected OE local peak")

        axes[i].set_title(f"Participant {user}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Video time (seconds)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

oe = pd.read_csv(OE_IN, low_memory=False)

offset_info = find_oe_peak_near_sync(oe)
offset_sec = offset_info["offset_sec"]

print("================================================")
print("GROUP 5 OPENEARAMBLE LABEL SHIFT BASED ON synchronizaiton_move")
print("================================================")
print(f"Original sync start  : {offset_info['sync_start']:.3f} s")
print(f"Original sync end    : {offset_info['sync_end']:.3f} s")
print(f"Original sync middle : {offset_info['sync_middle']:.3f} s")
print(f"Detected OE peak     : {offset_info['peak_time']:.3f} s")
print(f"Peak value           : {offset_info['peak_value']:.3f}")
print("------------------------------------------------")
print(f"Offset to apply      : {offset_sec:.3f} s")

if offset_sec < 0:
    print("Labels will move EARLIER / LEFT.")
elif offset_sec > 0:
    print("Labels will move LATER / RIGHT.")
else:
    print("No shift needed.")

print("================================================")

oe_shifted = shift_all_labels(
    oe,
    LABEL_COLS,
    offset_sec
)

oe_shifted.to_csv(OE_OUT, index=False)

print("\nSaved shifted OpenEarable file:")
print(OE_OUT)

new_start, new_end, _ = find_label_segment(
    oe_shifted,
    TIME_COL,
    SYNC_LABEL_COL,
    SYNC_LABEL
)

print("\nShifted sync:")
print(f"{new_start:.3f} -> {new_end:.3f}")
print(f"Shifted middle: {(new_start + new_end) / 2:.3f}")
print(f"Detected peak : {offset_info['peak_time']:.3f}")
print(f"Difference    : {(new_start + new_end) / 2 - offset_info['peak_time']:.6f} s")

plot_oe_before_after(
    oe,
    oe_shifted,
    offset_info
)


# --- CELL 68 (code cell #57) ---
import pandas as pd

OE_PATH = "/content/drive/MyDrive/thesis/data/group_5/openearable_labeled/group_5_openearable_labeled.csv"

df = pd.read_csv(OE_PATH, low_memory=False)

TIME_COL = "video_time_s"
LABEL_COL = "label_Whole_Group"

sync_labels = [
    "clap_synchronizaiton_move",
    "synchronizaiton_move"
]

def get_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments

segments = get_segments(df, TIME_COL, LABEL_COL)

print("Whole_Group sync segments:")
for s, e, lab in segments:
    if lab in sync_labels:
        print(f"{lab}: {s:.3f}s -> {e:.3f}s | dur={e-s:.3f}s | middle={(s+e)/2:.3f}s")


# --- CELL 69 (code cell #58) ---
import numpy as np
import matplotlib.pyplot as plt

def acc_mag_oe(df, user):
    return np.sqrt(
        df[f"p{user}_acc_x"]**2 +
        df[f"p{user}_acc_y"]**2 +
        df[f"p{user}_acc_z"]**2
    )

def plot_oe_sync(label_name, margin=25):
    segments = [
        (s, e, lab)
        for s, e, lab in get_segments(df, TIME_COL, LABEL_COL)
        if lab == label_name
    ]

    if len(segments) == 0:
        print("Not found:", label_name)
        return

    for idx, (s, e, lab) in enumerate(segments):
        mid = (s + e) / 2
        t = df[TIME_COL].values
        mask = (t >= s - margin) & (t <= e + margin)

        p1 = acc_mag_oe(df, 1)
        p2 = acc_mag_oe(df, 2)
        p3 = acc_mag_oe(df, 3)
        combined = (p1 + p2 + p3) / 3

        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = t[peak_idx]

        print(f"\n{label_name} event {idx}")
        print(f"label: {s:.3f} -> {e:.3f}, middle={mid:.3f}")
        print(f"local peak: {peak_time:.3f}, peak-middle={peak_time-mid:.3f}s")

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(t[mask], combined[mask], label="combined OE acc_mag")
        axes[0].axvspan(s, e, alpha=0.25, label=label_name)
        axes[0].axvline(mid, linestyle="--", label="label middle")
        axes[0].axvline(peak_time, linestyle="--", label="local peak")
        axes[0].set_title(f"Group 5 OpenEarable | {label_name} | combined")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for i, sig in enumerate([p1, p2, p3], start=1):
            axes[i].plot(t[mask], sig[mask], label=f"P{i} OE acc_mag")
            axes[i].axvspan(s, e, alpha=0.25, label=label_name)
            axes[i].axvline(mid, linestyle="--", label="label middle")
            axes[i].axvline(peak_time, linestyle="--", label="local peak")
            axes[i].set_title(f"Participant {i}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        axes[-1].set_xlabel("Video time (seconds)")
        plt.tight_layout()
        plt.show()

plot_oe_sync("clap_synchronizaiton_move")
plot_oe_sync("synchronizaiton_move")


# --- CELL 71 (code cell #59) ---
import pandas as pd, numpy as np, os

# ---- Group 6 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_6/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_6/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + values, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 72 (code cell #60) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_6/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_6"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load raw streams and find each participant window
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Shared grid = overlap window
# ============================================================

start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge all participants and sensors
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_6_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_6_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 73 (code cell #61) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_6/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_6"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]


def load(p):
    path = os.path.join(BASE, f"{p}.csv")

    d = pd.read_csv(path, low_memory=False)

    # drop trailing empty columns if they exist
    d = d.loc[:, [c for c in d.columns if str(c).strip() != ""]]

    # clean column names
    d.columns = [str(c).strip() for c in d.columns]

    # convert important columns to numeric
    for c in ["SampleTimeFine", "PacketCounter"] + SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["SampleTimeFine"] = d["SampleTimeFine"].astype("Int64")

    return d


# ============================================================
# Load raw XSens files
# ============================================================

raw = {
    p: load(p)
    for p in PREFIX
}

for p, d in raw.items():
    print(
        p,
        "| rows:", len(d),
        "| SampleTimeFine valid:", d["SampleTimeFine"].notna().sum()
    )


# ============================================================
# Master grid = Participant1 clean SampleTimeFine
# ============================================================

p1 = raw["Participant1"].dropna(subset=["SampleTimeFine"])
p1_ticks = set(p1["SampleTimeFine"].tolist())


# ============================================================
# Common span across all 3 participants
# using only ticks that exist on P1 grid
# ============================================================

spans = []

for p, d in raw.items():
    valid = d[d["SampleTimeFine"].isin(p1_ticks)]

    span = (
        int(valid["SampleTimeFine"].min()),
        int(valid["SampleTimeFine"].max())
    )

    spans.append(span)

    print(
        f"{p} span:",
        span[0],
        "->",
        span[1],
        "| duration:",
        round((span[1] - span[0]) / 1e6, 1),
        "s"
    )


start = max(s[0] for s in spans)
end   = min(s[1] for s in spans)


# ============================================================
# Build master timeline
# ============================================================

master = (
    p1[
        (p1["SampleTimeFine"] >= start) &
        (p1["SampleTimeFine"] <= end)
    ][["SampleTimeFine"]]
    .sort_values("SampleTimeFine")
    .reset_index(drop=True)
)

master["time_s"] = (
    (master["SampleTimeFine"].astype(np.int64) - start) / 1e6
).round(4)


# ============================================================
# Merge all participants onto master grid
# ============================================================

wide = master.copy()

for p, d in raw.items():
    pre = PREFIX[p]

    sub = (
        d[["SampleTimeFine"] + SENSOR_COLS]
        .dropna(subset=["SampleTimeFine"])
        .drop_duplicates("SampleTimeFine")
        .rename(columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        })
    )

    wide = wide.merge(
        sub,
        on="SampleTimeFine",
        how="left"
    )


# ============================================================
# Empty label columns, filled later from ELAN
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save locally, then copy to Drive
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_6_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "shape:", wide.shape,
    "| dur:",
    round(wide["time_s"].iloc[-1], 1),
    "s ->",
    local_path
)

os.makedirs(DRIVE_OUT, exist_ok=True)

dst = os.path.join(
    DRIVE_OUT,
    "group_6_xsens_merged_30hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Quick missing check
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")


# --- CELL 74 (code cell #62) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_with_individual_build.csv"
LOCAL_OUT = "/content/Group_6_with_individual_build.csv"
# ----------------

# ---- Group 6 participant mapping ----
# Kas    = Participant1
# Gozluk = Participant2
# Mark   = Participant3
PEOPLE = ["Kas", "Gozluk", "Mark"]

# ---- manual individual_build cutoff times ----
CUTOFF = {
    "Kas":    12 * 60 + 32,   # 752 s
    "Gozluk": 17 * 60 + 10,   # 1030 s
    "Mark":   12 * 60 + 32,   # 752 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


# ============================================================
# Helper: normalize names
# ============================================================

def norm(x):
    return str(x).strip().lower()


# ============================================================
# Involving tiers
# Includes:
# own tier + dyads containing person + Whole Group / Whole_Group
# Handles spaces and underscores
# ============================================================

INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


# ============================================================
# Interval helpers
# ============================================================

def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ============================================================
# Add individual_build rows
# ============================================================

new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)


# ============================================================
# Save ELAN-style CSV
# Still uses real names; next cell will rename to Participant1/2/3
# ============================================================

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 6 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 75 (code cell #63) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_6_individual_build_renamed.csv"
# ----------------

# Group 6 mapping
NAME = {
    "Kas": "Participant1",
    "Gozluk": "Participant2",
    "Mark": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    # handles both "Gozluk Kas" and "Gozluk_Kas"
    parts_original = t.replace("_", " ").split()

    # case-insensitive mapping
    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 76 (code cell #64) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 6 - LABEL OPENEARAMBLE USING VIDEO TIME
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_6/openearable_merged/group_6_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/openearable_labeled/group_6_openearable_labeled.csv"
LOCAL_OUT = "/content/group_6_openearable_labeled.csv"

# Group 6 video start:
# 2026-04-23 14:17:43 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-23 14:17:43", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe = pd.read_csv(OE_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# OpenEarable absolute timestamp -> video-relative time
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder / full label columns if rerunning
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# put video_time_s after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# add labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 6 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 77 (code cell #65) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 6 - LABEL XSENS USING VIDEO TIME
# XSens started 485s before video
# Therefore: video_time_s = time_s - 485
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_6/xsens_merged/group_6_xsens_merged_30hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled.csv"
LOCAL_OUT = "/content/group_6_xsens_labeled.csv"

XSENS_TO_VIDEO_OFFSET_S = -485.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# XSens relative time -> video-relative time
xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder / full label columns if rerunning
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# put video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# add labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 6 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 78 (code cell #66) ---
# ============================================================
# GROUP 6 - VISUAL SANITY CHECK
# Full labels + sync label list for OpenEarable and XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_6"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_6_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_6_xsens_labeled.csv"
)


# ============================================================
# CONFIG
# ============================================================

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"   # "acc_mag" or "gyro_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 4

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


# ============================================================
# HELPERS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal

    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]

    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be 'oe' or 'xsens'")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_alpha=0.14,
    text_every_n_segments=4,
    robust_y=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    existing_label_cols = [
        col for col in LABEL_COLS
        if col in df.columns
    ]

    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(len(USERS), 1, figsize=FIGSIZE, sharex=True)

    for ax, user in zip(axes, USERS):
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(facecolor=label_colors[label], alpha=label_alpha, label=label)
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# RUN
# ============================================================

df_g6_oe = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 6 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

df_g6_xsens = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 6 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

print("\nOpenEarable Whole_Group labels:")
print(df_g6_oe["label_Whole_Group"].dropna().unique())

print("\nXSens Whole_Group labels:")
print(df_g6_xsens["label_Whole_Group"].dropna().unique())


# --- CELL 79 (code cell #67) ---
# ============================================================
# GROUP 6 - ZOOM AROUND SYNC ACTIONS
# OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_6"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_6_openearable_labeled.csv"
)

XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled_cleaned.csv"

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"

SYNC_LABELS = [
    "synchronaziton_move",
    "synchronizaiton_move"
]

USERS = [1, 2, 3]
SIGNAL_TYPE = "acc_mag"

ZOOM_MARGIN = 25


def get_signal(df, user, source):
    if source == "oe":
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    elif source == "xsens":
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def plot_sync_label(csv_path, source, title, sync_label):
    df = pd.read_csv(csv_path, low_memory=False)

    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == sync_label
    ]

    print("\n================================================")
    print(title)
    print("Label:", sync_label)
    print("================================================")

    if len(segments) == 0:
        print("Label not found.")
        print("Available Whole_Group labels:")
        print(df[SYNC_LABEL_COL].dropna().unique())
        return None

    for i, (s, e, lab) in enumerate(segments):
        print(f"{i}: {s:.3f}s -> {e:.3f}s | dur={e-s:.3f}s | middle={(s+e)/2:.3f}s")

    for event_i, (sync_start, sync_end, lab) in enumerate(segments):

        sync_mid = (sync_start + sync_end) / 2

        times = df[TIME_COL].values
        left = sync_start - ZOOM_MARGIN
        right = sync_end + ZOOM_MARGIN

        mask = (times >= left) & (times <= right)

        signals = {}
        for user in USERS:
            signals[user] = get_signal(df, user, source)

        combined = np.mean(
            [signals[user] for user in USERS],
            axis=0
        )

        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = times[peak_idx]
        peak_value = combined[peak_idx]
        offset = peak_time - sync_mid

        print(
            f"Event {event_i} local peak: {peak_time:.3f}s | "
            f"value={peak_value:.3f} | peak - label_middle = {offset:.3f}s"
        )

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(
            times[mask],
            combined[mask],
            linewidth=0.8,
            label=f"Combined {source} acc_mag"
        )

        axes[0].axvspan(sync_start, sync_end, alpha=0.25, label=sync_label)
        axes[0].axvline(sync_mid, linestyle="--", label="label middle")
        axes[0].axvline(peak_time, linestyle="--", label="local peak")
        axes[0].set_title(f"{title} | {sync_label} | combined")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        for i, user in enumerate(USERS, start=1):
            axes[i].plot(
                times[mask],
                signals[user][mask],
                linewidth=0.8,
                label=f"P{user} {source} acc_mag"
            )

            axes[i].axvspan(sync_start, sync_end, alpha=0.25, label=sync_label)
            axes[i].axvline(sync_mid, linestyle="--", label="label middle")
            axes[i].axvline(peak_time, linestyle="--", label="local peak")
            axes[i].set_title(f"{title} | Participant {user}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper left")

        axes[-1].set_xlabel("Video time (seconds)")
        plt.tight_layout()
        plt.show()

    return df


for label in SYNC_LABELS:

    plot_sync_label(
        csv_path=OE_PATH,
        source="oe",
        title="Group 6 OpenEarable",
        sync_label=label
    )

    plot_sync_label(
        csv_path=XSENS_PATH,
        source="xsens",
        title="Group 6 XSens",
        sync_label=label
    )


# --- CELL 80 (code cell #68) ---
# ============================================================
# GROUP 6 - CLEAN / CURE XSENS EXTREME ARTIFACTS
# Removes impossible values like 1e31 and saves a clean copy
# ============================================================

import pandas as pd
import numpy as np
import os

# ============================================================
# PATHS
# ============================================================

IN_PATH = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled.csv"

OUT_PATH = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled_cleaned.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(IN_PATH, low_memory=False)


# ============================================================
# SENSOR COLUMNS
# ============================================================

SENSOR_COLS = []

for p in ["p1", "p2", "p3"]:
    SENSOR_COLS += [
        f"{p}_euler_x", f"{p}_euler_y", f"{p}_euler_z",
        f"{p}_acc_x", f"{p}_acc_y", f"{p}_acc_z",
        f"{p}_gyr_x", f"{p}_gyr_y", f"{p}_gyr_z",
    ]

SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for c in SENSOR_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ============================================================
# BEFORE CLEANING CHECK
# ============================================================

print("================================================")
print("BEFORE CLEANING - largest absolute values")
print("================================================")

max_abs = df[SENSOR_COLS].abs().max().sort_values(ascending=False)
print(max_abs.head(20))


# ============================================================
# CLEANING RULES
# ============================================================

# Very generous thresholds.
# These mainly catch impossible corruption such as 1e31.
ACC_LIMIT = 500.0       # m/s^2, very high, keeps real motion
GYR_LIMIT = 500.0       # generous
EULER_LIMIT = 10000.0   # very generous, catches absurd corruption

artifact_mask = pd.Series(False, index=df.index)

for p in ["p1", "p2", "p3"]:

    acc_cols = [f"{p}_acc_x", f"{p}_acc_y", f"{p}_acc_z"]
    gyr_cols = [f"{p}_gyr_x", f"{p}_gyr_y", f"{p}_gyr_z"]
    euler_cols = [f"{p}_euler_x", f"{p}_euler_y", f"{p}_euler_z"]

    acc_cols = [c for c in acc_cols if c in df.columns]
    gyr_cols = [c for c in gyr_cols if c in df.columns]
    euler_cols = [c for c in euler_cols if c in df.columns]

    # artifact rows per sensor type
    if acc_cols:
        bad_acc = df[acc_cols].abs().gt(ACC_LIMIT).any(axis=1)
        df.loc[bad_acc, acc_cols] = np.nan
        artifact_mask = artifact_mask | bad_acc

    if gyr_cols:
        bad_gyr = df[gyr_cols].abs().gt(GYR_LIMIT).any(axis=1)
        df.loc[bad_gyr, gyr_cols] = np.nan
        artifact_mask = artifact_mask | bad_gyr

    if euler_cols:
        bad_euler = df[euler_cols].abs().gt(EULER_LIMIT).any(axis=1)
        df.loc[bad_euler, euler_cols] = np.nan
        artifact_mask = artifact_mask | bad_euler


df["xsens_artifact_removed"] = artifact_mask


# ============================================================
# INTERPOLATE ONLY SHORT GAPS
# ============================================================

# This fills only very short corruption gaps.
# limit=5 means max 5 consecutive samples.
# At 30 Hz, 5 samples is about 0.17 seconds.
df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(
    method="linear",
    limit=5,
    limit_direction="both"
)


# ============================================================
# AFTER CLEANING CHECK
# ============================================================

print("\n================================================")
print("CLEANING SUMMARY")
print("================================================")

print("Artifact rows detected:", int(df["xsens_artifact_removed"].sum()))
print("Artifact row percentage:", df["xsens_artifact_removed"].mean() * 100, "%")

print("\nAfter cleaning - largest absolute values:")
max_abs_after = df[SENSOR_COLS].abs().max().sort_values(ascending=False)
print(max_abs_after.head(20))


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUT_PATH, index=False)

print("\nSaved cleaned XSens file:")
print(OUT_PATH)


# --- CELL 81 (code cell #69) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# PATHS
# ============================================================
XSENS_IN  = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled_cleaned.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_individual_build_renamed.csv"
OUT_PATH  = "/content/drive/MyDrive/thesis/data/group_6/xsens_labeled/group_6_xsens_labeled_cleaned_SHIFTED.csv"

# ============================================================
# SETTINGS
# These are chosen according to your image
# ============================================================
TARGET_SYNC_MIDDLE_RANGE = (1360, 1375)   # choose the later sync event
SEARCH_WINDOW            = (1356, 1361.5) # search for the left-side peak here
SMOOTH_WIN               = 5              # light smoothing for peak detection

# ============================================================
# CONSTANTS
# ============================================================
ECOLS = ["tier","blank","begin_hms","begin_s","end_hms","end_s","dur_hms","dur_s","label"]
TIERS = [
    "Participant1", "Participant2", "Participant3",
    "Participant1_Participant2", "Participant1_Participant3",
    "Participant2_Participant3", "Whole_Group"
]

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(XSENS_IN, low_memory=False)
elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)

# numeric safety
for c in ["begin_s", "end_s", "dur_s"]:
    elan[c] = pd.to_numeric(elan[c], errors="coerce")

# ============================================================
# BUILD COMBINED XSENS ACC MAGNITUDE
# ============================================================
for p in ["p1", "p2", "p3"]:
    df[f"{p}_acc_mag_tmp"] = np.sqrt(
        pd.to_numeric(df[f"{p}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"{p}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"{p}_acc_z"], errors="coerce")**2
    )

df["combined_acc_mag_tmp"] = df[[f"{p}_acc_mag_tmp" for p in ["p1", "p2", "p3"]]].mean(axis=1)
df["combined_acc_mag_smooth_tmp"] = (
    df["combined_acc_mag_tmp"]
    .rolling(SMOOTH_WIN, center=True, min_periods=1)
    .mean()
)

# ============================================================
# FIND THE TARGET SYNC SEGMENT IN ELAN
# ============================================================
wg = elan[elan["tier"] == "Whole_Group"].copy()
wg["mid"] = (wg["begin_s"] + wg["end_s"]) / 2

sync_rows = wg[wg["label"].astype(str).str.contains("synchron", case=False, na=False)].copy()

if len(sync_rows) == 0:
    raise ValueError("No Whole_Group sync label found in ELAN.")

target_sync = sync_rows[sync_rows["mid"].between(*TARGET_SYNC_MIDDLE_RANGE)].copy()

if len(target_sync) == 0:
    print("Available sync candidates:")
    print(sync_rows[["begin_s", "end_s", "mid", "label"]])
    raise ValueError("No sync event found inside TARGET_SYNC_MIDDLE_RANGE.")

# If multiple exist, use the first one in that range
target_sync = target_sync.sort_values("mid").iloc[0]

orig_begin = float(target_sync["begin_s"])
orig_end   = float(target_sync["end_s"])
orig_mid   = float(target_sync["mid"])
orig_label = str(target_sync["label"])

# ============================================================
# FIND LOCAL PEAK IN THE LEFT SEARCH WINDOW
# ============================================================
win = df[df["video_time_s"].between(*SEARCH_WINDOW)].copy()

if len(win) == 0:
    raise ValueError("No XSens samples inside SEARCH_WINDOW.")

peak_idx = win["combined_acc_mag_smooth_tmp"].idxmax()
peak_time = float(df.loc[peak_idx, "video_time_s"])
peak_val  = float(df.loc[peak_idx, "combined_acc_mag_smooth_tmp"])

# delta < 0 means shift LEFT
delta_s = peak_time - orig_mid

print("================================================")
print("GROUP 6 XSENS SHIFT CALCULATION")
print("================================================")
print(f"Target sync label : {orig_label}")
print(f"Original sync     : {orig_begin:.3f}s -> {orig_end:.3f}s")
print(f"Original middle   : {orig_mid:.3f}s")
print(f"Detected peak     : {peak_time:.3f}s")
print(f"Peak value        : {peak_val:.3f}")
print(f"Shift delta       : {delta_s:.3f}s")
print("(negative = shift LEFT)")
print("================================================")

# ============================================================
# SHIFT ELAN INTERVALS
# ============================================================
elan_shifted = elan.copy()
elan_shifted["begin_s_shift"] = elan_shifted["begin_s"] + delta_s
elan_shifted["end_s_shift"]   = elan_shifted["end_s"] + delta_s

# ============================================================
# REGENERATE LABEL COLUMNS ON XSENS USING SHIFTED ELAN
# ============================================================
df_out = df.copy()

# remove old label columns first
old_label_cols = [c for c in df_out.columns if c.startswith("label_")]
df_out = df_out.drop(columns=old_label_cols, errors="ignore")

for t in TIERS:
    sub = (
        elan_shifted[elan_shifted["tier"] == t][["begin_s_shift", "end_s_shift", "label"]]
        .sort_values("begin_s_shift")
        .reset_index(drop=True)
        .rename(columns={"begin_s_shift": "video_time_s", "end_s_shift": "_end", "label": "_lab"})
    )

    col = f"label_{t}"

    if len(sub) == 0:
        df_out[col] = pd.NA
        continue

    m = pd.merge_asof(
        df_out[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (m["video_time_s"] < m["_end"]) & m["_lab"].notna()
    df_out[col] = m["_lab"].where(valid, other=pd.NA).values

# ============================================================
# DROP TEMP HELPER COLUMNS BEFORE SAVING
# ============================================================
drop_tmp = [c for c in df_out.columns if c.endswith("_tmp")]
df_save = df_out.drop(columns=drop_tmp, errors="ignore")

# ============================================================
# SAVE
# ============================================================
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df_save.to_csv(OUT_PATH, index=False)

print("\nSaved shifted XSens file to:")
print(OUT_PATH)

# ============================================================
# QUICK BEFORE/AFTER VISUAL CHECK
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# BEFORE
axes[0].plot(df["video_time_s"], df["combined_acc_mag_smooth_tmp"], lw=1, label="Combined xsens acc_mag")
axes[0].axvspan(orig_begin, orig_end, alpha=0.25, label="Original sync label")
axes[0].axvline(orig_mid, ls="--", label="Original middle")
axes[0].axvline(peak_time, ls="--", label="Detected peak")
axes[0].set_title("Before shifting")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# AFTER
new_begin = orig_begin + delta_s
new_end   = orig_end + delta_s
new_mid   = orig_mid + delta_s

axes[1].plot(df["video_time_s"], df["combined_acc_mag_smooth_tmp"], lw=1, label="Combined xsens acc_mag")
axes[1].axvspan(new_begin, new_end, alpha=0.25, label="Shifted sync label")
axes[1].axvline(new_mid, ls="--", label="Shifted middle")
axes[1].axvline(peak_time, ls="--", label="Detected peak")
axes[1].set_title("After shifting")
axes[1].legend(loc="upper left")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel("Video time (seconds)")

plt.tight_layout()
plt.show()


# --- CELL 83 (code cell #70) ---
import pandas as pd, numpy as np, os

# ---- Group 7 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_7/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_7/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000   # 50 Hz grid
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + sensor values, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 84 (code cell #71) ---
import pandas as pd, numpy as np, os

BASE   = "/content/drive/MyDrive/thesis/data/group_7/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_7/openearable_merged"

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 85 (code cell #72) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 7 OPENEARAMBLE WIDE MERGE
# Participant3 recording is shorter/interrupted.
# Therefore timeline is based on Participant1 + Participant2 overlap.
# Participant3 is merged where available and NaN after it stops.
# ============================================================

BASE      = "/content/drive/MyDrive/thesis/data/group_7/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_7"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

ANCHOR_PARTICIPANTS = ["Participant1", "Participant2"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load raw streams and find windows
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Main timeline based only on Participant1 + Participant2
# This avoids cutting everything to short Participant3 recording.
# ============================================================

start = max(windows[p][0] for p in ANCHOR_PARTICIPANTS)
end   = min(windows[p][1] for p in ANCHOR_PARTICIPANTS)

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge all participants onto P1/P2 timeline
# P3 will naturally become NaN after its recording stops.
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# ============================================================
# Empty label columns
# ============================================================

for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_7_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "\nshape:", wide.shape,
    "| P1/P2 anchor overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_7_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(20))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")

print("\nParticipant-level missing average:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")


# --- CELL 86 (code cell #73) ---
# ============================================================
# GROUP 7 XSENS MERGE - AFTER UPDATED FILES
# Handles SampleTimeFine wrap correctly
# Uses all three participants if they are now full-length
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_7/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_7_updated"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["xsens_time_s_fixed"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("xsens_time_s_fixed")
          .drop_duplicates("xsens_time_s_fixed")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD + DIAGNOSE UPDATED FILES
# ============================================================

raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["xsens_time_s_fixed"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("fixed time start:", df["xsens_time_s_fixed"].min())
    print("fixed time end  :", df["xsens_time_s_fixed"].max())
    print("duration min    :", df["xsens_time_s_fixed"].max() / 60)
    print("median dt       :", diffs.median())
    print("max gap         :", diffs.max())
    print("gaps > 1s       :", (diffs > 1).sum())


# ============================================================
# DECIDE MASTER END
# If P3 is now full-length, this will use all 3 overlap.
# If P3 is still short, it will warn and use P1/P2 anchor.
# ============================================================

durations = {
    p: raw[p]["xsens_time_s_fixed"].max()
    for p in PARTICIPANTS
}

print("\nDurations:")
for p, d in durations.items():
    print(p, f"{d:.2f}s = {d/60:.2f} min")

p1p2_end = min(durations["Participant1"], durations["Participant2"])
all3_end = min(durations.values())

p3_ratio = durations["Participant3"] / p1p2_end

if p3_ratio >= 0.85:
    print("\nParticipant3 looks full enough. Using ALL THREE overlap.")
    end = all3_end
    OUT_NAME = "group_7_xsens_merged_30hz_UPDATED_ALL3.csv"
else:
    print("\nWARNING: Participant3 is still much shorter. Using P1/P2 anchor; P3 will be NaN after it stops.")
    print("P3/P1P2 duration ratio:", p3_ratio)
    end = p1p2_end
    OUT_NAME = "group_7_xsens_merged_30hz_UPDATED_P1P2_ANCHOR.csv"


# ============================================================
# MASTER GRID = P1 timeline up to chosen end
# ============================================================

p1 = raw["Participant1"].copy()

master = p1[
    (p1["xsens_time_s_fixed"] >= 0) &
    (p1["xsens_time_s_fixed"] <= end)
][["xsens_time_s_fixed"]].copy()

master = master.sort_values("xsens_time_s_fixed").reset_index(drop=True)
master["time_s"] = master["xsens_time_s_fixed"].round(4)

wide = master[["time_s"]].copy()


# ============================================================
# MERGE ALL PARTICIPANTS
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["xsens_time_s_fixed"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("xsens_time_s_fixed"),
        left_on="time_s",
        right_on="xsens_time_s_fixed",
        direction="nearest",
        tolerance=0.02
    )

    wide = wide.drop(columns=["xsens_time_s_fixed"])


for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# SAVE
# ============================================================

local_path = os.path.join(LOCAL_OUT, OUT_NAME)
drive_path = os.path.join(DRIVE_OUT, OUT_NAME)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 7 XSENS UPDATED MERGE SAVED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")


# --- CELL 87 (code cell #74) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_with_individual_build.csv"
LOCAL_OUT = "/content/Group_7_with_individual_build.csv"
# ----------------

# Canonical people
PEOPLE = ["Kadın", "Arda", "Adam"]

# Aliases for spelling differences in ELAN
ALIASES = {
    "Kadın": ["Kadın", "Kadin"],
    "Arda": ["Arda"],
    "Adam": ["Adam"],
}

# Manual cutoffs
CUTOFF = {
    "Kadın": 42 * 60 + 54,   # 2574 s
    "Arda":  17 * 60 + 21,   # 1041 s
    "Adam":  30 * 60 + 22,   # 1822 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


def norm(x):
    return str(x).strip().lower()


def tier_contains_person(tier, person):
    tier_norm = norm(tier)
    tier_parts = tier_norm.replace("_", " ").split()

    aliases = [norm(a) for a in ALIASES[person]]

    # exact own tier
    if tier_norm in aliases:
        return True

    # dyad tier containing alias
    for alias in aliases:
        if alias in tier_parts:
            return True

    return False


# ============================================================
# Involving tiers
# own tier + dyads containing person + Whole Group
# Handles Kadın / Kadin issue
# ============================================================

INVOLVING = {}

for person in PEOPLE:
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        if tier_contains_person(tier, person):
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ============================================================
# Add individual_build rows
# ============================================================

new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)


# ============================================================
# Save
# ============================================================

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 7 individual_build COMPLETE - CORRECTED")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 88 (code cell #75) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_7_individual_build_renamed.csv"
# ----------------

# Group 7 mapping
NAME = {
    "Kadın": "Participant1",
    "Kadin": "Participant1",
    "Arda": "Participant2",
    "Adam": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    parts_original = t.replace("_", " ").split()

    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 89 (code cell #76) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 7 - LABEL OPENEARAMBLE USING VIDEO TIME
# OE uses P1/P2-anchor merged file because P3 OE stopped early
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_7/openearable_merged/group_7_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/openearable_labeled/group_7_openearable_labeled.csv"
LOCAL_OUT = "/content/group_7_openearable_labeled.csv"

# Group 7 video start:
# 2026-04-29 10:24:00 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-29 10:24:00", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe = pd.read_csv(OE_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# absolute OE time -> video-relative time
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder / label columns if rerunning
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# place video_time_s after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# useful availability flag for P3 OE
oe["p3_oe_available"] = oe["p3_acc_x"].notna()

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 7 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nP3 OpenEarable availability:")
print("available rows:", oe["p3_oe_available"].sum(), "/", len(oe))
print("available %   :", oe["p3_oe_available"].mean() * 100)

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 90 (code cell #77) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 7 - LABEL XSENS USING VIDEO TIME
# XSens started 59s after video
# Therefore: video_time_s = time_s + 59
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_7/xsens_merged/group_7_xsens_merged_30hz_UPDATED_ALL3.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_7/xsens_labeled/group_7_xsens_labeled.csv"
LOCAL_OUT = "/content/group_7_xsens_labeled.csv"

XSENS_TO_VIDEO_OFFSET_S = 59.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# XSens relative time -> video-relative time
xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder / label columns if rerunning
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 7 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 91 (code cell #78) ---
# ============================================================
# GROUP 7 - VISUAL SANITY CHECK
# Full labels + Whole_Group label list for OpenEarable and XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_7"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_7_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_7_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 5

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


def moving_average(signal, window):
    if window <= 1:
        return signal
    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]
    if len(clean) == 0:
        return None
    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15
    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")
    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    if "p3_oe_available" in df.columns:
        print("\nP3 OpenEarable availability:")
        print(df["p3_oe_available"].value_counts(dropna=False))
        print("available %:", df["p3_oe_available"].mean() * 100)

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_alpha=0.14,
    text_every_n_segments=5,
    robust_y=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    existing_label_cols = [col for col in LABEL_COLS if col in df.columns]

    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []
    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(len(USERS), 1, figsize=FIGSIZE, sharex=True)

    for ax, user in zip(axes, USERS):
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]
                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(facecolor=label_colors[label], alpha=label_alpha, label=label)
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


df_g7_oe = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 7 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

df_g7_xsens = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 7 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

print("\nOpenEarable Whole_Group labels:")
print(df_g7_oe["label_Whole_Group"].dropna().unique())

print("\nXSens Whole_Group labels:")
print(df_g7_xsens["label_Whole_Group"].dropna().unique())


# --- CELL 92 (code cell #79) ---
# ============================================================
# GROUP 7 - ZOOM AROUND synchronizaiton_move
# OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_7"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_7_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_7_xsens_labeled.csv"
)

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

USERS = [1, 2, 3]
ZOOM_MARGIN = 25


def get_signal(df, user, source):
    if source == "oe":
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    else:
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i
        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i
        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def plot_sync(csv_path, source, title):
    df = pd.read_csv(csv_path, low_memory=False)

    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == SYNC_LABEL
    ]

    print("\n================================================")
    print(title)
    print("================================================")

    if len(segments) == 0:
        print("No synchronizaiton_move found.")
        print(df[SYNC_LABEL_COL].dropna().unique())
        return

    for event_i, (s, e, lab) in enumerate(segments):
        mid = (s + e) / 2
        print(f"event {event_i}: {s:.3f}s -> {e:.3f}s | mid={mid:.3f}s | dur={e-s:.3f}s")

        times = df[TIME_COL].values
        mask = (times >= s - ZOOM_MARGIN) & (times <= e + ZOOM_MARGIN)

        signals = {}
        for u in USERS:
            signals[u] = get_signal(df, u, source)

        combined = np.nanmean([signals[u] for u in USERS], axis=0)

        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = times[peak_idx]

        print(f"local peak: {peak_time:.3f}s | peak - middle = {peak_time - mid:.3f}s")

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(times[mask], combined[mask], linewidth=0.8, label=f"Combined {source} acc_mag")
        axes[0].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
        axes[0].axvline(mid, linestyle="--", label="label middle")
        axes[0].axvline(peak_time, linestyle="--", label="local peak")
        axes[0].set_title(f"{title} | Combined | event {event_i}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        for i, u in enumerate(USERS, start=1):
            axes[i].plot(times[mask], signals[u][mask], linewidth=0.8, label=f"P{u} {source} acc_mag")
            axes[i].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
            axes[i].axvline(mid, linestyle="--", label="label middle")
            axes[i].axvline(peak_time, linestyle="--", label="local peak")
            axes[i].set_title(f"{title} | Participant {u}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper left")

        axes[-1].set_xlabel("Video time (seconds)")
        plt.tight_layout()
        plt.show()


plot_sync(OE_PATH, "oe", "Group 7 OpenEarable")
plot_sync(XSENS_PATH, "xsens", "Group 7 XSens")


# --- CELL 93 (code cell #80) ---
# ============================================================
# GROUP 7 - SHIFT OPENEARAMBLE AND XSENS LABELS USING SYNC PEAK
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_7"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_7_openearable_labeled.csv"
)

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_7_xsens_labeled.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_7_openearable_labeled_SHIFTED.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_7_xsens_labeled_SHIFTED.csv"
)


TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]


# ============================================================
# CONFIG
# Search windows chosen according to your plots
# ============================================================

CONFIG = {
    "oe": {
        "in_path": OE_IN,
        "out_path": OE_OUT,
        "title": "Group 7 OpenEarable",
        # visible OE peak is after the label, around 2990-2993
        "search_after_s": 25,
        "search_before_s": 5,
    },
    "xsens": {
        "in_path": XSENS_IN,
        "out_path": XSENS_OUT,
        "title": "Group 7 XSens",
        # visible XSens peak is after the label, around 91s
        "search_after_s": 30,
        "search_before_s": 5,
    }
}


# ============================================================
# HELPERS
# ============================================================

def acc_mag(df, user, source):
    if source == "oe":
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    elif source == "xsens":
        x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(
        pd.to_numeric(df[x], errors="coerce")**2 +
        pd.to_numeric(df[y], errors="coerce")**2 +
        pd.to_numeric(df[z], errors="coerce")**2
    )


def combined_acc_mag(df, source):
    sigs = []

    for user in USERS:
        sig = acc_mag(df, user, source)

        # For Group 7 OE, P3 is missing after early stop.
        # nanmean handles that safely.
        sigs.append(sig)

    return np.nanmean(np.vstack(sigs), axis=0)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_sync_segment(df):
    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == SYNC_LABEL
    ]

    if len(segments) == 0:
        print("Available Whole_Group labels:")
        print(df[SYNC_LABEL_COL].dropna().unique())
        raise ValueError(f"No {SYNC_LABEL} found.")

    # For Group 7:
    # OE only sees the late sync because OE starts around 80s.
    # XSens sees the early sync.
    # Use the first available segment in each file.
    segments = sorted(segments, key=lambda x: x[0])

    return segments[0]


def shift_label_column(df, label_col, delta_s):
    times = df[TIME_COL].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, TIME_COL, label_col)

    for s, e, lab in segments:
        new_s = s + delta_s
        new_e = e + delta_s

        mask = (times >= new_s) & (times < new_e)
        shifted[mask] = lab

    return shifted


def shift_all_labels(df, delta_s):
    out = df.copy()

    for col in LABEL_COLS:
        if col in out.columns:
            out[col] = shift_label_column(out, col, delta_s)

    return out


def process_shift(source):
    cfg = CONFIG[source]

    df = pd.read_csv(cfg["in_path"], low_memory=False)

    times = df[TIME_COL].values
    signal = combined_acc_mag(df, source)

    sync_start, sync_end, _ = find_sync_segment(df)
    sync_mid = (sync_start + sync_end) / 2

    search_left = sync_start - cfg["search_before_s"]
    search_right = sync_end + cfg["search_after_s"]

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    delta_s = peak_time - sync_mid

    print("\n================================================")
    print(cfg["title"])
    print("SHIFT CALCULATION")
    print("================================================")
    print(f"Sync label start : {sync_start:.3f}s")
    print(f"Sync label end   : {sync_end:.3f}s")
    print(f"Sync label middle: {sync_mid:.3f}s")
    print(f"Detected peak    : {peak_time:.3f}s")
    print(f"Peak value       : {peak_value:.3f}")
    print(f"Shift delta      : {delta_s:.3f}s")
    if delta_s > 0:
        print("Labels move RIGHT / later.")
    elif delta_s < 0:
        print("Labels move LEFT / earlier.")
    else:
        print("No shift.")
    print("================================================")

    shifted = shift_all_labels(df, delta_s)
    shifted.to_csv(cfg["out_path"], index=False)

    print("Saved shifted file:")
    print(cfg["out_path"])

    # Re-find shifted sync segment
    new_sync_start, new_sync_end, _ = find_sync_segment(shifted)
    new_sync_mid = (new_sync_start + new_sync_end) / 2

    # Plot before/after
    left = min(sync_start, new_sync_start, peak_time) - 20
    right = max(sync_end, new_sync_end, peak_time) + 20
    view = (times >= left) & (times <= right)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axes[0].plot(times[view], signal[view], linewidth=0.8, label=f"{source} combined acc_mag")
    axes[0].axvspan(sync_start, sync_end, alpha=0.25, label="Original sync label")
    axes[0].axvline(sync_mid, linestyle="--", label="Original middle")
    axes[0].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[0].set_title(f"{cfg['title']} - Before shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(times[view], signal[view], linewidth=0.8, label=f"{source} combined acc_mag")
    axes[1].axvspan(new_sync_start, new_sync_end, alpha=0.25, label="Shifted sync label")
    axes[1].axvline(new_sync_mid, linestyle="--", label="Shifted middle")
    axes[1].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[1].set_title(f"{cfg['title']} - After shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return shifted, delta_s


# ============================================================
# RUN
# ============================================================

oe_shifted, oe_delta = process_shift("oe")
xsens_shifted, xsens_delta = process_shift("xsens")

print("\n================================================")
print("GROUP 7 FINAL SHIFT SUMMARY")
print("================================================")
print(f"OpenEarable shift: {oe_delta:.3f}s")
print(f"XSens shift      : {xsens_delta:.3f}s")
print("================================================")


# --- CELL 95 (code cell #81) ---
import pandas as pd, numpy as np, os

# ---- Group 8 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_8/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_8/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + values, drop trailing flag column
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mgnt
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 96 (code cell #82) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
BASE      = "/content/drive/MyDrive/thesis/data/group_8/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_8"
# ----------------

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load raw streams and find each participant window
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Shared all-three overlap grid
# ============================================================

start = max(w[0] for w in windows.values())
end   = min(w[1] for w in windows.values())

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge all participants and sensors
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# Empty label columns
for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# Save
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_8_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "\nshape:", wide.shape,
    "| all-three overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_8_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(15))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")

print("\nParticipant-level missing average:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")


# --- CELL 97 (code cell #83) ---
# ============================================================
# GROUP 8 XSENS MERGE - WRAP-SAFE VERSION
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_8/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_8"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["xsens_time_s_fixed"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("xsens_time_s_fixed")
          .drop_duplicates("xsens_time_s_fixed")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD + DIAGNOSE
# ============================================================

raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["xsens_time_s_fixed"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("fixed time start:", df["xsens_time_s_fixed"].min())
    print("fixed time end  :", df["xsens_time_s_fixed"].max())
    print("duration min    :", df["xsens_time_s_fixed"].max() / 60)
    print("median dt       :", diffs.median())
    print("max gap         :", diffs.max())
    print("gaps > 1s       :", (diffs > 1).sum())


# ============================================================
# MASTER GRID = all-three overlap using P1 timeline
# ============================================================

durations = {
    p: raw[p]["xsens_time_s_fixed"].max()
    for p in PARTICIPANTS
}

print("\nDurations:")
for p, d in durations.items():
    print(p, f"{d:.2f}s = {d/60:.2f} min")

end = min(durations.values())

p1 = raw["Participant1"].copy()

master = p1[
    (p1["xsens_time_s_fixed"] >= 0) &
    (p1["xsens_time_s_fixed"] <= end)
][["xsens_time_s_fixed"]].copy()

master = master.sort_values("xsens_time_s_fixed").reset_index(drop=True)
master["time_s"] = master["xsens_time_s_fixed"].round(4)

wide = master[["time_s"]].copy()


# ============================================================
# MERGE ALL PARTICIPANTS
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["xsens_time_s_fixed"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("xsens_time_s_fixed"),
        left_on="time_s",
        right_on="xsens_time_s_fixed",
        direction="nearest",
        tolerance=0.02
    )

    wide = wide.drop(columns=["xsens_time_s_fixed"])


# Empty label columns
for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# SAVE
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_8_xsens_merged_30hz.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_8_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 8 XSENS MERGED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nTop missing columns:")
print(wide.isna().mean().sort_values(ascending=False).head(20))


# --- CELL 98 (code cell #84) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_with_individual_build.csv"
LOCAL_OUT = "/content/Group_8_with_individual_build.csv"
# ----------------

PEOPLE = ["Arda", "Ewoud", "Jennifer"]

CUTOFF = {
    "Arda":     18 * 60 + 1,    # 1081 s
    "Ewoud":    24 * 60 + 40,   # 1480 s
    "Jennifer": 24 * 60 + 50,   # 1490 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


def norm(x):
    return str(x).strip().lower()


INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 8 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 99 (code cell #85) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_8_individual_build_renamed.csv"
# ----------------

NAME = {
    "Arda": "Participant1",
    "Ewoud": "Participant2",
    "Jennifer": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    parts_original = t.replace("_", " ").split()

    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 100 (code cell #86) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 8 - LABEL OPENEARAMBLE USING VIDEO TIME
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_8/openearable_merged/group_8_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/openearable_labeled/group_8_openearable_labeled.csv"
LOCAL_OUT = "/content/group_8_openearable_labeled.csv"

# Group 8 video start:
# 2026-04-30 11:35:00 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-30 11:35:00", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe = pd.read_csv(OE_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# absolute OE time -> video-relative time
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder / label columns if rerunning
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# place video_time_s after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 8 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 101 (code cell #87) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 8 - LABEL XSENS USING VIDEO TIME
# XSens started 17s before video
# Therefore: video_time_s = time_s - 17
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_8/xsens_merged/group_8_xsens_merged_30hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_8/xsens_labeled/group_8_xsens_labeled.csv"
LOCAL_OUT = "/content/group_8_xsens_labeled.csv"

XSENS_TO_VIDEO_OFFSET_S = -17.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# XSens relative time -> video-relative time
xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder / label columns if rerunning
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 8 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 102 (code cell #88) ---
# ============================================================
# GROUP 8 - VISUAL SANITY CHECK
# Full labels + Whole_Group label list for OpenEarable and XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_8"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_8_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_8_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 5

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


def moving_average(signal, window):
    if window <= 1:
        return signal
    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]
    if len(clean) == 0:
        return None
    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15
    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")
    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_alpha=0.14,
    text_every_n_segments=5,
    robust_y=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    existing_label_cols = [col for col in LABEL_COLS if col in df.columns]
    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []
    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(len(USERS), 1, figsize=FIGSIZE, sharex=True)

    for ax, user in zip(axes, USERS):
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]
                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(facecolor=label_colors[label], alpha=label_alpha, label=label)
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


df_g8_oe = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 8 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

df_g8_xsens = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 8 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

print("\nOpenEarable Whole_Group labels:")
print(df_g8_oe["label_Whole_Group"].dropna().unique())

print("\nXSens Whole_Group labels:")
print(df_g8_xsens["label_Whole_Group"].dropna().unique())


# --- CELL 103 (code cell #89) ---
# ============================================================
# GROUP 8 - ZOOM AROUND synchronizaiton_move
# OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_8"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_8_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_8_xsens_labeled.csv"
)

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

USERS = [1, 2, 3]
ZOOM_MARGIN = 25


def get_signal(df, user):
    x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    return np.sqrt(
        pd.to_numeric(df[x], errors="coerce")**2 +
        pd.to_numeric(df[y], errors="coerce")**2 +
        pd.to_numeric(df[z], errors="coerce")**2
    )


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def plot_sync(csv_path, source_name):
    df = pd.read_csv(csv_path, low_memory=False)

    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == SYNC_LABEL
    ]

    print("\n================================================")
    print(f"Group 8 {source_name}")
    print("================================================")

    if len(segments) == 0:
        print("No synchronizaiton_move found.")
        print(df[SYNC_LABEL_COL].dropna().unique())
        return

    for event_i, (s, e, lab) in enumerate(segments):
        mid = (s + e) / 2
        print(f"event {event_i}: {s:.3f}s -> {e:.3f}s | mid={mid:.3f}s | dur={e-s:.3f}s")

        times = df[TIME_COL].values
        mask = (times >= s - ZOOM_MARGIN) & (times <= e + ZOOM_MARGIN)

        signals = {u: get_signal(df, u) for u in USERS}
        combined = np.nanmean([signals[u] for u in USERS], axis=0)

        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = times[peak_idx]

        print(f"local peak: {peak_time:.3f}s | peak - middle = {peak_time - mid:.3f}s")

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(times[mask], combined[mask], linewidth=0.8, label=f"Combined {source_name} acc_mag")
        axes[0].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
        axes[0].axvline(mid, linestyle="--", label="label middle")
        axes[0].axvline(peak_time, linestyle="--", label="local peak")
        axes[0].set_title(f"Group 8 {source_name} | Combined | event {event_i}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        for i, u in enumerate(USERS, start=1):
            axes[i].plot(times[mask], signals[u][mask], linewidth=0.8, label=f"P{u} {source_name} acc_mag")
            axes[i].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
            axes[i].axvline(mid, linestyle="--", label="label middle")
            axes[i].axvline(peak_time, linestyle="--", label="local peak")
            axes[i].set_title(f"Group 8 {source_name} | Participant {u}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper left")

        axes[-1].set_xlabel("Video time (seconds)")
        plt.tight_layout()
        plt.show()


plot_sync(OE_PATH, "OpenEarable")
plot_sync(XSENS_PATH, "XSens")


# --- CELL 104 (code cell #90) ---
# ============================================================
# GROUP 8 - SHIFT OPENEARAMBLE AND XSENS LABELS USING SYNC EVENT 0
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_8"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_8_openearable_labeled.csv"
)

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_8_xsens_labeled.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_8_openearable_labeled_SHIFTED.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_8_xsens_labeled_SHIFTED.csv"
)

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# We use event 0 because it is visually clearer for both sensors.
TARGET_EVENT_INDEX = 1

# Search mainly AFTER the label, because the real sync peak is to the right.
SEARCH_BEFORE_S = 3
SEARCH_AFTER_S  = 30


def acc_mag(df, user):
    x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
    return np.sqrt(
        pd.to_numeric(df[x], errors="coerce")**2 +
        pd.to_numeric(df[y], errors="coerce")**2 +
        pd.to_numeric(df[z], errors="coerce")**2
    )


def combined_acc_mag(df):
    sigs = [acc_mag(df, user) for user in USERS]
    return np.nanmean(np.vstack(sigs), axis=0)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def find_sync_segment(df, event_index=0):
    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == SYNC_LABEL
    ]

    if len(segments) == 0:
        print("Available Whole_Group labels:")
        print(df[SYNC_LABEL_COL].dropna().unique())
        raise ValueError(f"No {SYNC_LABEL} found.")

    segments = sorted(segments, key=lambda x: x[0])

    print("Available sync events:")
    for i, (s, e, lab) in enumerate(segments):
        print(f"{i}: {s:.3f}s -> {e:.3f}s | mid={(s+e)/2:.3f}s | dur={e-s:.3f}s")

    if event_index >= len(segments):
        raise ValueError(f"TARGET_EVENT_INDEX={event_index} but only {len(segments)} sync events exist.")

    return segments[event_index]


def shift_label_column(df, label_col, delta_s):
    times = df[TIME_COL].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, TIME_COL, label_col)

    for s, e, lab in segments:
        new_s = s + delta_s
        new_e = e + delta_s

        mask = (times >= new_s) & (times < new_e)
        shifted[mask] = lab

    return shifted


def shift_all_labels(df, delta_s):
    out = df.copy()

    for col in LABEL_COLS:
        if col in out.columns:
            out[col] = shift_label_column(out, col, delta_s)

    return out


def process_shift(in_path, out_path, title):
    df = pd.read_csv(in_path, low_memory=False)

    times = df[TIME_COL].values
    signal = combined_acc_mag(df)

    sync_start, sync_end, _ = find_sync_segment(df, TARGET_EVENT_INDEX)
    sync_mid = (sync_start + sync_end) / 2

    search_left = sync_start - SEARCH_BEFORE_S
    search_right = sync_end + SEARCH_AFTER_S

    mask = (times >= search_left) & (times <= search_right)

    if mask.sum() == 0:
        raise ValueError("No samples found in search window.")

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(signal[mask])]

    peak_time = times[peak_idx]
    peak_value = signal[peak_idx]

    delta_s = peak_time - sync_mid

    print("\n================================================")
    print(title)
    print("SHIFT CALCULATION")
    print("================================================")
    print(f"Sync label start : {sync_start:.3f}s")
    print(f"Sync label end   : {sync_end:.3f}s")
    print(f"Sync label middle: {sync_mid:.3f}s")
    print(f"Detected peak    : {peak_time:.3f}s")
    print(f"Peak value       : {peak_value:.3f}")
    print(f"Shift delta      : {delta_s:.3f}s")
    if delta_s > 0:
        print("Labels move RIGHT / later.")
    elif delta_s < 0:
        print("Labels move LEFT / earlier.")
    else:
        print("No shift.")
    print("================================================")

    shifted = shift_all_labels(df, delta_s)
    shifted.to_csv(out_path, index=False)

    print("Saved shifted file:")
    print(out_path)

    # Re-find shifted sync segment
    new_sync_start, new_sync_end, _ = find_sync_segment(shifted, TARGET_EVENT_INDEX)
    new_sync_mid = (new_sync_start + new_sync_end) / 2

    left = min(sync_start, new_sync_start, peak_time) - 15
    right = max(sync_end, new_sync_end, peak_time) + 15
    view = (times >= left) & (times <= right)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axes[0].plot(times[view], signal[view], linewidth=0.8, label="combined acc_mag")
    axes[0].axvspan(sync_start, sync_end, alpha=0.25, label="Original sync label")
    axes[0].axvline(sync_mid, linestyle="--", label="Original middle")
    axes[0].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[0].set_title(f"{title} - Before shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(times[view], signal[view], linewidth=0.8, label="combined acc_mag")
    axes[1].axvspan(new_sync_start, new_sync_end, alpha=0.25, label="Shifted sync label")
    axes[1].axvline(new_sync_mid, linestyle="--", label="Shifted middle")
    axes[1].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[1].set_title(f"{title} - After shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return shifted, delta_s


oe_shifted, oe_delta = process_shift(
    OE_IN,
    OE_OUT,
    "Group 8 OpenEarable"
)

xsens_shifted, xsens_delta = process_shift(
    XSENS_IN,
    XSENS_OUT,
    "Group 8 XSens"
)

print("\n================================================")
print("GROUP 8 FINAL SHIFT SUMMARY")
print("================================================")
print(f"OpenEarable shift: {oe_delta:.3f}s")
print(f"XSens shift      : {xsens_delta:.3f}s")
print("================================================")


# --- CELL 106 (code cell #91) ---
import pandas as pd, numpy as np, os

# ---- Group 9 paths ----
BASE   = "/content/drive/MyDrive/thesis/data/group_9/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_9/openearable_merged"
# -----------------------

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0, t1 = int(imu.min()), int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for name, df in streams.items():

        df = df[
            (df.timestamp_us >= t0 - GRID_US) &
            (df.timestamp_us <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if name == "skin_temp" else TOL_FAST_US

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            out[c] = m[c].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for p in PARTICIPANTS:
    out = merge_participant(p)

    path = os.path.join(
        OUTDIR,
        f"{p}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    miss = out.iloc[:, 2:].isna().mean().mean() * 100

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    print(
        f"{p}: {len(out):>6} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {miss:.1f}%  ->  {path}"
    )


# --- CELL 107 (code cell #92) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 9 OPENEARAMBLE WIDE MERGE
# Participant3 recording stopped early.
# Timeline is based on Participant1 + Participant2 overlap.
# Participant3 is merged where available and NaN after it stops.
# ============================================================

BASE      = "/content/drive/MyDrive/thesis/data/group_9/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_9"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]
ANCHOR_PARTICIPANTS = ["Participant1", "Participant2"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US, TOL_FAST, TOL_SLOW = 20_000, 12_000, 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load raw streams and find windows
# ============================================================

all_streams, windows = {}, {}

for p in PARTICIPANTS:
    streams = {
        k: load_sensor(
            os.path.join(BASE, p, f"{p}_{k}.csv"),
            v
        )
        for k, v in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Main timeline based only on P1 + P2
# ============================================================

start = max(windows[p][0] for p in ANCHOR_PARTICIPANTS)
end   = min(windows[p][1] for p in ANCHOR_PARTICIPANTS)

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge all participants onto P1/P2 timeline
# P3 naturally becomes NaN after its recording ends.
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for name, df in all_streams[p].items():

        df = df[
            (df.timestamp_us >= start - GRID_US) &
            (df.timestamp_us <= end + GRID_US)
        ]

        tol = TOL_SLOW if name == "skin_temp" else TOL_FAST

        m = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for c in SCHEMA[name]:
            wide[f"{pre}_{c}"] = m[c].values


# Empty label columns
for pre in ("p1", "p2", "p3"):
    wide[f"label_{pre}"] = pd.NA


# Useful availability flag
wide["p3_oe_available"] = wide["p3_acc_x"].notna()


# ============================================================
# Save
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_9_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)

print(
    "\nshape:", wide.shape,
    "| P1/P2 anchor overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

dst = os.path.join(
    DRIVE_OUT,
    "group_9_openearable_merged_50hz.csv"
)

shutil.copy2(local_path, dst)

print("copied ->", dst)


# ============================================================
# Missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(20))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")

print("\nParticipant-level missing average:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nP3 OpenEarable availability:")
print("available rows:", wide["p3_oe_available"].sum(), "/", len(wide))
print("available %   :", wide["p3_oe_available"].mean() * 100)


# --- CELL 108 (code cell #93) ---
# ============================================================
# GROUP 9 XSENS MERGE - WRAP-SAFE VERSION
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_9/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_9"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["xsens_time_s_fixed"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("xsens_time_s_fixed")
          .drop_duplicates("xsens_time_s_fixed")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD + DIAGNOSE
# ============================================================

raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["xsens_time_s_fixed"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("fixed time start:", df["xsens_time_s_fixed"].min())
    print("fixed time end  :", df["xsens_time_s_fixed"].max())
    print("duration min    :", df["xsens_time_s_fixed"].max() / 60)
    print("median dt       :", diffs.median())
    print("max gap         :", diffs.max())
    print("gaps > 1s       :", (diffs > 1).sum())


# ============================================================
# MASTER GRID = all-three overlap using P1 timeline
# If P3 is also short in XSens, we will adjust after seeing output.
# ============================================================

durations = {
    p: raw[p]["xsens_time_s_fixed"].max()
    for p in PARTICIPANTS
}

print("\nDurations:")
for p, d in durations.items():
    print(p, f"{d:.2f}s = {d/60:.2f} min")

end = min(durations.values())

p1 = raw["Participant1"].copy()

master = p1[
    (p1["xsens_time_s_fixed"] >= 0) &
    (p1["xsens_time_s_fixed"] <= end)
][["xsens_time_s_fixed"]].copy()

master = master.sort_values("xsens_time_s_fixed").reset_index(drop=True)
master["time_s"] = master["xsens_time_s_fixed"].round(4)

wide = master[["time_s"]].copy()


# ============================================================
# MERGE ALL PARTICIPANTS
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["xsens_time_s_fixed"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("xsens_time_s_fixed"),
        left_on="time_s",
        right_on="xsens_time_s_fixed",
        direction="nearest",
        tolerance=0.02
    )

    wide = wide.drop(columns=["xsens_time_s_fixed"])


for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# SAVE
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_9_xsens_merged_30hz.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_9_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 9 XSENS MERGED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nTop missing columns:")
print(wide.isna().mean().sort_values(ascending=False).head(20))


# --- CELL 109 (code cell #94) ---
import pandas as pd
import numpy as np
import os

BASE = "/content/drive/MyDrive/thesis/data/group_9/xsens"

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32

def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")
    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])
    df["xsens_time_s_fixed"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("xsens_time_s_fixed")
          .drop_duplicates("xsens_time_s_fixed")
          .reset_index(drop=True)
    )

    return df

p1 = load_xsens_fixed("Participant1")

p1["dt_s"] = p1["xsens_time_s_fixed"].diff()

print("P1 big gaps:")
print(
    p1[p1["dt_s"] > 1][
        ["xsens_time_s_fixed", "dt_s"]
    ]
)

for idx in p1.index[p1["dt_s"] > 1]:
    print("\nGap around index:", idx)
    display(
        p1.loc[idx-5:idx+5, ["PacketCounter", "SampleTimeFine", "xsens_time_s_fixed", "dt_s"]]
    )


# --- CELL 110 (code cell #95) ---
# ============================================================
# GROUP 9 XSENS MERGE - CONTINUOUS GRID VERSION
# Preserves gaps as NaN instead of silently dropping time.
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_9/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_9_continuous"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32
GRID_S = 1 / 30
TOL_S = 0.02


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["time_s"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("time_s")
          .drop_duplicates("time_s")
          .reset_index(drop=True)
    )

    return df


raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["time_s"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("time start:", df["time_s"].min())
    print("time end  :", df["time_s"].max())
    print("duration min:", df["time_s"].max() / 60)
    print("median dt:", diffs.median())
    print("max gap:", diffs.max())
    print("gaps > 1s:", (diffs > 1).sum())


durations = {p: raw[p]["time_s"].max() for p in PARTICIPANTS}

print("\nDurations:")
for p, d in durations.items():
    print(p, f"{d:.2f}s = {d/60:.2f} min")

# Strategy:
# P2 and P3 are clean and same length, so use their common span.
# P1 is merged onto this grid, with NaN during its internal gaps.
end = min(durations["Participant2"], durations["Participant3"])

grid = pd.DataFrame({
    "time_s": np.round(np.arange(0, end + GRID_S, GRID_S), 4)
})

wide = grid.copy()

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["time_s"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=TOL_S
    )


for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA

# Gap / availability flags
for pre in ["p1", "p2", "p3"]:
    wide[f"{pre}_xsens_available"] = wide[f"{pre}_acc_x"].notna()


local_path = os.path.join(
    LOCAL_OUT,
    "group_9_xsens_merged_30hz_CONTINUOUS.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_9_xsens_merged_30hz_CONTINUOUS.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 9 XSENS CONTINUOUS MERGE SAVED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_") and not c.endswith("_available")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nAvailability:")
for pre in ["p1", "p2", "p3"]:
    print(pre, f"{wide[f'{pre}_xsens_available'].mean() * 100:.2f}%")

print("\nTop missing columns:")
print(wide.isna().mean().sort_values(ascending=False).head(20))


# --- CELL 111 (code cell #96) ---
import pandas as pd, numpy as np, os, shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_with_individual_build.csv"
LOCAL_OUT = "/content/Group_9_with_individual_build.csv"
# ----------------

PEOPLE = ["Arda", "Roy", "Concetta"]

CUTOFF = {
    "Arda":     26 * 60 + 50,   # 1610 s
    "Roy":      39 * 60 + 10,   # 2350 s
    "Concetta": 41 * 60 + 20,   # 2480 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


def norm(x):
    return str(x).strip().lower()


INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 9 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 112 (code cell #97) ---
import pandas as pd, shutil, os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_9_individual_build_renamed.csv"
# ----------------

NAME = {
    "Arda": "Participant1",
    "Roy": "Participant2",
    "Concetta": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    parts_original = t.replace("_", " ").split()

    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 113 (code cell #98) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 9 - LABEL OPENEARAMBLE USING VIDEO TIME
# OE uses P1/P2 anchor because P3 OE stopped early
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_9/openearable_merged/group_9_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/openearable_labeled/group_9_openearable_labeled.csv"
LOCAL_OUT = "/content/group_9_openearable_labeled.csv"

# Group 9 video start:
# 2026-04-30 13:32:00 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-30 13:32:00", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe = pd.read_csv(OE_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# absolute OE time -> video-relative time
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder / label columns if rerunning
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# place video_time_s after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# keep P3 OE availability flag if present; otherwise create it
if "p3_oe_available" not in oe.columns:
    oe["p3_oe_available"] = oe["p3_acc_x"].notna()

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 9 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nP3 OpenEarable availability:")
print("available rows:", oe["p3_oe_available"].sum(), "/", len(oe))
print("available %   :", oe["p3_oe_available"].mean() * 100)

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 114 (code cell #99) ---
import pandas as pd, numpy as np, os, shutil

# ============================================================
# GROUP 9 - LABEL XSENS USING VIDEO TIME
# XSens started 29s after video
# Therefore: video_time_s = time_s + 29
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_9/xsens_merged/group_9_xsens_merged_30hz_CONTINUOUS.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_9/xsens_labeled/group_9_xsens_labeled.csv"
LOCAL_OUT = "/content/group_9_xsens_labeled.csv"

XSENS_TO_VIDEO_OFFSET_S = 29.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# XSens relative time -> video-relative time
xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder / label columns if rerunning
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# availability flags are already in merged file; keep them
# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 9 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nXSens availability:")
for pre in ["p1", "p2", "p3"]:
    col = f"{pre}_xsens_available"
    if col in xsens.columns:
        print(pre, f"{xsens[col].mean() * 100:.2f}%")

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 115 (code cell #100) ---
# ============================================================
# GROUP 9 - VISUAL SANITY CHECK
# Full labels + Whole_Group label list for OpenEarable and XSens
# Handles:
# - OE P3 early cutoff via p3_oe_available
# - XSens shorter total duration + availability flags
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_9"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_9_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_9_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 6

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


def moving_average(signal, window):
    if window <= 1:
        return signal
    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]
    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(
        pd.to_numeric(df[x], errors="coerce")**2 +
        pd.to_numeric(df[y], errors="coerce")**2 +
        pd.to_numeric(df[z], errors="coerce")**2
    )


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    print("\nTime range:")
    print("start:", df[TIME_COL].min())
    print("end  :", df[TIME_COL].max())
    print("dur  :", df[TIME_COL].max() - df[TIME_COL].min())

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    if "p3_oe_available" in df.columns:
        print("\nP3 OpenEarable availability:")
        print(df["p3_oe_available"].value_counts(dropna=False))
        print("available %:", df["p3_oe_available"].mean() * 100)

    for pre in ["p1", "p2", "p3"]:
        col = f"{pre}_xsens_available"
        if col in df.columns:
            print(f"\n{pre} XSens availability:")
            print(df[col].value_counts(dropna=False))
            print("available %:", df[col].mean() * 100)

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_alpha=0.14,
    text_every_n_segments=6,
    robust_y=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    existing_label_cols = [
        col for col in LABEL_COLS
        if col in df.columns
    ]

    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    fig, axes = plt.subplots(len(USERS), 1, figsize=FIGSIZE, sharex=True)

    for ax, user in zip(axes, USERS):
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


df_g9_oe = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 9 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

df_g9_xsens = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 9 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

print("\nOpenEarable Whole_Group labels:")
print(df_g9_oe["label_Whole_Group"].dropna().unique())

print("\nXSens Whole_Group labels:")
print(df_g9_xsens["label_Whole_Group"].dropna().unique())


# --- CELL 116 (code cell #101) ---
import pandas as pd
import os
import shutil

# ============================================================
# GROUP 9 - PATCH MISSING WHOLE_GROUP SYNCHRONIZATION ROW
# ============================================================

RAW_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9.csv"

WITH_BUILD_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_with_individual_build.csv"
RENAMED_PATH    = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

missing_raw_row = {
    "tier": "Whole Group",
    "blank": "",
    "begin_hms": "00:00:12.180",
    "begin_s": 12.180,
    "end_hms": "00:00:15.300",
    "end_s": 15.300,
    "dur_hms": "00:00:03.120",
    "dur_s": 3.120,
    "label": "synchronization_move"
}

missing_renamed_row = {
    "tier": "Whole_Group",
    "blank": "",
    "begin_hms": "00:00:12.180",
    "begin_s": 12.180,
    "end_hms": "00:00:15.300",
    "end_s": 15.300,
    "dur_hms": "00:00:03.120",
    "dur_s": 3.120,
    "label": "synchronization_move"
}


def load_elan(path):
    df = pd.read_csv(path, header=None, names=COLS, dtype={"blank": str})
    df["tier"] = df["tier"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
    df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
    df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")
    return df


def save_elan(df, path):
    df = df.sort_values(["tier", "begin_s", "end_s", "label"]).reset_index(drop=True)

    fmt = lambda x: str(round(float(x), 3))

    lines = [
        f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
        f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
        for r in df.itertuples()
    ]

    backup = path.replace(".csv", "_BACKUP_before_sync_patch.csv")
    shutil.copy2(path, backup)

    open(path, "w").write("\n".join(lines) + "\n")

    print("backup saved ->", backup)
    print("patched saved ->", path)


def row_exists(df, tier, begin_s, end_s, label):
    return (
        (df["tier"] == tier) &
        (df["begin_s"].round(3) == round(begin_s, 3)) &
        (df["end_s"].round(3) == round(end_s, 3)) &
        (df["label"] == label)
    ).any()


# ============================================================
# Patch with_individual_build
# ============================================================

df_build = load_elan(WITH_BUILD_PATH)

if not row_exists(
    df_build,
    "Whole Group",
    12.180,
    15.300,
    "synchronization_move"
):
    df_build = pd.concat(
        [df_build, pd.DataFrame([missing_raw_row])],
        ignore_index=True
    )
    print("Added missing row to Group_9_with_individual_build.csv")
else:
    print("Row already exists in Group_9_with_individual_build.csv")

save_elan(df_build, WITH_BUILD_PATH)


# ============================================================
# Patch renamed file
# ============================================================

df_renamed = load_elan(RENAMED_PATH)

if not row_exists(
    df_renamed,
    "Whole_Group",
    12.180,
    15.300,
    "synchronization_move"
):
    df_renamed = pd.concat(
        [df_renamed, pd.DataFrame([missing_renamed_row])],
        ignore_index=True
    )
    print("Added missing row to Group_9_individual_build_renamed.csv")
else:
    print("Row already exists in Group_9_individual_build_renamed.csv")

save_elan(df_renamed, RENAMED_PATH)


# ============================================================
# Verify
# ============================================================

print("\nVerification - sync/start labels in renamed:")
df_check = load_elan(RENAMED_PATH)

mask = df_check["label"].str.lower().str.contains(
    "sync|synch|synchron|start",
    regex=True,
    na=False
)

print(
    df_check.loc[mask, ["tier", "begin_s", "end_s", "dur_s", "label"]]
    .sort_values(["begin_s", "tier"])
    .to_string(index=False)
)

print("\nAnnotation counts per tier after patch:")
print(df_check.groupby("tier").size())


# --- CELL 117 (code cell #102) ---
# ============================================================
# GROUP 9 - EARLY SENSOR PEAK CHECK
# Compare OpenEarable and XSens in the first 2 minutes
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_9"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_9_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_9_xsens_labeled.csv"
)

TIME_COL = "video_time_s"
USERS = [1, 2, 3]

# first 2 minutes of shared available video-time region
START_S = 0
END_S = 160

SMOOTH_OE = 10
SMOOTH_XSENS = 5


def acc_mag(df, user):
    return np.sqrt(
        pd.to_numeric(df[f"p{user}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_z"], errors="coerce")**2
    )


def smooth(x, win):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def combined_acc(df):
    sigs = [acc_mag(df, u) for u in USERS]
    return np.nanmean(np.vstack(sigs), axis=0)


def robust_ylim(x, p1=1, p2=99):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    lo, hi = np.percentile(x, [p1, p2])
    margin = (hi - lo) * 0.2
    return lo - margin, hi + margin


oe = pd.read_csv(OE_PATH, low_memory=False)
xs = pd.read_csv(XSENS_PATH, low_memory=False)

# shared region inside first 2 minutes
shared_start = max(START_S, oe[TIME_COL].min(), xs[TIME_COL].min())
shared_end = min(END_S, oe[TIME_COL].max(), xs[TIME_COL].max())

print("================================================")
print("GROUP 9 EARLY SHARED WINDOW")
print("================================================")
print("OE time range   :", oe[TIME_COL].min(), "->", oe[TIME_COL].max())
print("XSens time range:", xs[TIME_COL].min(), "->", xs[TIME_COL].max())
print("Shared window   :", shared_start, "->", shared_end)
print("================================================")

oe_t = oe[TIME_COL].values
xs_t = xs[TIME_COL].values

oe_mask = (oe_t >= shared_start) & (oe_t <= shared_end)
xs_mask = (xs_t >= shared_start) & (xs_t <= shared_end)

oe_combined = smooth(combined_acc(oe), SMOOTH_OE)
xs_combined = smooth(combined_acc(xs), SMOOTH_XSENS)

# local peaks
oe_local_idx = np.where(oe_mask)[0]
xs_local_idx = np.where(xs_mask)[0]

oe_peak_idx = oe_local_idx[np.nanargmax(oe_combined[oe_mask])]
xs_peak_idx = xs_local_idx[np.nanargmax(xs_combined[xs_mask])]

oe_peak_t = oe_t[oe_peak_idx]
xs_peak_t = xs_t[xs_peak_idx]

print("OE strongest early peak   :", oe_peak_t)
print("XSens strongest early peak:", xs_peak_t)
print("XSens peak - OE peak      :", xs_peak_t - oe_peak_t, "s")

# ============================================================
# Plot combined signals together
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

axes[0].plot(oe_t[oe_mask], oe_combined[oe_mask], linewidth=0.9, label="OpenEarable combined acc_mag")
axes[0].axvline(oe_peak_t, linestyle="--", label=f"OE peak {oe_peak_t:.2f}s")
axes[0].axvline(xs_peak_t, linestyle="--", label=f"XSens peak {xs_peak_t:.2f}s")
axes[0].set_title("Group 9 OpenEarable - first shared 2 minutes")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper left")
ylim = robust_ylim(oe_combined[oe_mask])
if ylim:
    axes[0].set_ylim(ylim)

axes[1].plot(xs_t[xs_mask], xs_combined[xs_mask], linewidth=0.9, label="XSens combined acc_mag")
axes[1].axvline(oe_peak_t, linestyle="--", label=f"OE peak {oe_peak_t:.2f}s")
axes[1].axvline(xs_peak_t, linestyle="--", label=f"XSens peak {xs_peak_t:.2f}s")
axes[1].set_title("Group 9 XSens - first shared 2 minutes")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper left")
ylim = robust_ylim(xs_combined[xs_mask])
if ylim:
    axes[1].set_ylim(ylim)

axes[-1].set_xlabel("Video time (seconds)")
plt.tight_layout()
plt.show()

# ============================================================
# Plot each participant separately
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(20, 12), sharex=True)

for i, u in enumerate(USERS):
    oe_sig = smooth(acc_mag(oe, u), SMOOTH_OE)
    xs_sig = smooth(acc_mag(xs, u), SMOOTH_XSENS)

    axes[i, 0].plot(oe_t[oe_mask], oe_sig[oe_mask], linewidth=0.8, label=f"OE P{u}")
    axes[i, 0].axvline(oe_peak_t, linestyle="--", label="OE combined peak")
    axes[i, 0].axvline(xs_peak_t, linestyle="--", label="XSens combined peak")
    axes[i, 0].set_title(f"OpenEarable Participant {u}")
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend(loc="upper left")
    ylim = robust_ylim(oe_sig[oe_mask])
    if ylim:
        axes[i, 0].set_ylim(ylim)

    axes[i, 1].plot(xs_t[xs_mask], xs_sig[xs_mask], linewidth=0.8, label=f"XSens P{u}")
    axes[i, 1].axvline(oe_peak_t, linestyle="--", label="OE combined peak")
    axes[i, 1].axvline(xs_peak_t, linestyle="--", label="XSens combined peak")
    axes[i, 1].set_title(f"XSens Participant {u}")
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend(loc="upper left")
    ylim = robust_ylim(xs_sig[xs_mask])
    if ylim:
        axes[i, 1].set_ylim(ylim)

axes[-1, 0].set_xlabel("Video time (seconds)")
axes[-1, 1].set_xlabel("Video time (seconds)")
plt.tight_layout()
plt.show()


# --- CELL 118 (code cell #103) ---
# ============================================================
# GROUP 9 - OPTIONAL SHIFT USING EARLY COMMON PEAK
# Align ELAN Whole_Group synchronization_move at 12.180-15.300s
# to detected early sensor peak around ~60s.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_9"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_9_openearable_labeled.csv"
)

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_9_xsens_labeled.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_9_openearable_labeled_SHIFTED_EARLY_PEAK.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_9_xsens_labeled_SHIFTED_EARLY_PEAK.csv"
)

TIME_COL = "video_time_s"

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

USERS = [1, 2, 3]

# Official ELAN sync annotation
SYNC_START = 12.180
SYNC_END   = 15.300
SYNC_MID   = (SYNC_START + SYNC_END) / 2

# Search around the visible common early burst
SEARCH_LEFT = 50
SEARCH_RIGHT = 70


def acc_mag(df, user):
    return np.sqrt(
        pd.to_numeric(df[f"p{user}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_z"], errors="coerce")**2
    )


def combined_acc(df):
    sigs = [acc_mag(df, u) for u in USERS]
    return np.nanmean(np.vstack(sigs), axis=0)


def smooth(x, win=5):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def shift_label_column(df, label_col, delta_s):
    times = df[TIME_COL].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, TIME_COL, label_col)

    for s, e, lab in segments:
        new_s = s + delta_s
        new_e = e + delta_s

        mask = (times >= new_s) & (times < new_e)
        shifted[mask] = lab

    return shifted


def shift_all_labels(df, delta_s):
    out = df.copy()

    for col in LABEL_COLS:
        if col in out.columns:
            out[col] = shift_label_column(out, col, delta_s)

    return out


def process_file(in_path, out_path, title, smooth_win):
    df = pd.read_csv(in_path, low_memory=False)

    times = df[TIME_COL].values
    sig = smooth(combined_acc(df), smooth_win)

    mask = (times >= SEARCH_LEFT) & (times <= SEARCH_RIGHT)

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(sig[mask])]

    peak_time = times[peak_idx]
    delta_s = peak_time - SYNC_MID

    print("\n================================================")
    print(title)
    print("EARLY PEAK SHIFT CALCULATION")
    print("================================================")
    print(f"Official sync start : {SYNC_START:.3f}s")
    print(f"Official sync end   : {SYNC_END:.3f}s")
    print(f"Official sync middle: {SYNC_MID:.3f}s")
    print(f"Detected early peak : {peak_time:.3f}s")
    print(f"Shift delta         : {delta_s:.3f}s")
    print("Labels move RIGHT / later.")
    print("================================================")

    shifted = shift_all_labels(df, delta_s)
    shifted.to_csv(out_path, index=False)

    print("Saved shifted file:")
    print(out_path)

    # Plot before/after around early burst
    view = (times >= 35) & (times <= 85)

    shifted_sync_start = SYNC_START + delta_s
    shifted_sync_end = SYNC_END + delta_s
    shifted_sync_mid = SYNC_MID + delta_s

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axes[0].plot(times[view], sig[view], linewidth=0.9, label="combined acc_mag")
    axes[0].axvspan(SYNC_START, SYNC_END, alpha=0.25, label="Original sync label")
    axes[0].axvline(SYNC_MID, linestyle="--", label="Original sync middle")
    axes[0].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[0].set_title(f"{title} - Before shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(times[view], sig[view], linewidth=0.9, label="combined acc_mag")
    axes[1].axvspan(shifted_sync_start, shifted_sync_end, alpha=0.25, label="Shifted sync label")
    axes[1].axvline(shifted_sync_mid, linestyle="--", label="Shifted sync middle")
    axes[1].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[1].set_title(f"{title} - After shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return shifted, delta_s, peak_time


oe_shifted, oe_delta, oe_peak = process_file(
    OE_IN,
    OE_OUT,
    "Group 9 OpenEarable",
    smooth_win=10
)

xsens_shifted, xsens_delta, xsens_peak = process_file(
    XSENS_IN,
    XSENS_OUT,
    "Group 9 XSens",
    smooth_win=5
)

print("\n================================================")
print("GROUP 9 EARLY PEAK SHIFT SUMMARY")
print("================================================")
print(f"OpenEarable detected peak: {oe_peak:.3f}s")
print(f"XSens detected peak      : {xsens_peak:.3f}s")
print(f"OpenEarable shift        : {oe_delta:.3f}s")
print(f"XSens shift              : {xsens_delta:.3f}s")
print(f"XSens peak - OE peak     : {xsens_peak - oe_peak:.3f}s")
print("================================================")


# --- CELL 120 (code cell #104) ---
import pandas as pd
import numpy as np
import os

# ============================================================
# GROUP 10 - OPENEARAMBLE PER-PARTICIPANT MERGE
# ============================================================

BASE   = "/content/drive/MyDrive/thesis/data/group_10/openearable"
OUTDIR = "/content/drive/MyDrive/thesis/data/group_10/openearable_merged"

os.makedirs(OUTDIR, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US     = 20_000   # 50 Hz
TOL_FAST_US = 12_000
TOL_SLOW_US = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None)

    # keep timestamp + sensor values, drop trailing flag/status column if present
    df = df.iloc[:, :1 + len(names)]

    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


def merge_participant(participant):
    pdir = os.path.join(BASE, participant)

    streams = {
        sensor_name: load_sensor(
            os.path.join(pdir, f"{participant}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    # session window from acc / gyro / mag
    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    t0 = int(imu.min())
    t1 = int(imu.max())

    grid = pd.DataFrame({
        "timestamp_us": np.arange(
            t0,
            t1 + GRID_US,
            GRID_US,
            dtype=np.int64
        )
    })

    out = grid.copy()

    for sensor_name, df in streams.items():

        df = df[
            (df["timestamp_us"] >= t0 - GRID_US) &
            (df["timestamp_us"] <= t1 + GRID_US)
        ]

        tol = TOL_SLOW_US if sensor_name == "skin_temp" else TOL_FAST_US

        merged = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for col in SCHEMA[sensor_name]:
            out[col] = merged[col].values

    out.insert(
        1,
        "datetime_utc",
        pd.to_datetime(out["timestamp_us"], unit="us", utc=True)
    )

    return out


for participant in PARTICIPANTS:
    out = merge_participant(participant)

    path = os.path.join(
        OUTDIR,
        f"{participant}_openearable_merged_50hz.csv"
    )

    out.to_csv(path, index=False)

    duration_s = (
        out["timestamp_us"].max() - out["timestamp_us"].min()
    ) / 1e6

    avg_missing = out.iloc[:, 2:].isna().mean().mean() * 100

    print(
        f"{participant}: {len(out):>7} rows, "
        f"{out.shape[1]} cols, "
        f"duration {duration_s:.1f}s, "
        f"avg-missing {avg_missing:.1f}%  ->  {path}"
    )


# --- CELL 121 (code cell #105) ---
import pandas as pd
import numpy as np
import os
import shutil

# ============================================================
# GROUP 10 OPENEARAMBLE WIDE MERGE
# Participant2 is shorter / weaker.
# Timeline is based on Participant1 + Participant3 overlap.
# Participant2 is merged where available and NaN after it stops.
# ============================================================

BASE      = "/content/drive/MyDrive/thesis/data/group_10/openearable"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/openearable_merged"
LOCAL_OUT = "/content/openearable_merged_group_10"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]
ANCHOR_PARTICIPANTS = ["Participant1", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SCHEMA = {
    "acc":       ["acc_x", "acc_y", "acc_z"],
    "gyro":      ["gyro_x", "gyro_y", "gyro_z"],
    "mgnt":      ["mag_x", "mag_y", "mag_z"],
    "bone_acc":  ["bone_acc_x", "bone_acc_y", "bone_acc_z"],
    "baro":      ["baro_pa"],
    "env_temp":  ["env_temp_c"],
    "skin_temp": ["skin_temp_c"],
    "ppg":       ["ppg_0", "ppg_1", "ppg_2", "ppg_3"],
}

GRID_US = 20_000
TOL_FAST = 12_000
TOL_SLOW = 40_000


def load_sensor(path, names):
    df = pd.read_csv(path, header=None).iloc[:, :1 + len(names)]
    df.columns = ["timestamp_us"] + names
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)

    return (
        df.drop_duplicates("timestamp_us")
          .sort_values("timestamp_us")
          .reset_index(drop=True)
    )


# ============================================================
# Load streams and find windows
# ============================================================

all_streams = {}
windows = {}

for p in PARTICIPANTS:
    streams = {
        sensor_name: load_sensor(
            os.path.join(BASE, p, f"{p}_{sensor_name}.csv"),
            value_cols
        )
        for sensor_name, value_cols in SCHEMA.items()
    }

    all_streams[p] = streams

    imu = pd.concat([
        streams[k]["timestamp_us"]
        for k in ("acc", "gyro", "mgnt")
    ])

    windows[p] = (int(imu.min()), int(imu.max()))

    print(
        f"{p} window:",
        pd.to_datetime(windows[p][0], unit="us", utc=True),
        "->",
        pd.to_datetime(windows[p][1], unit="us", utc=True),
        f"duration {(windows[p][1] - windows[p][0]) / 1e6:.1f}s"
    )


# ============================================================
# Main timeline based on P1 + P3
# ============================================================

start = max(windows[p][0] for p in ANCHOR_PARTICIPANTS)
end   = min(windows[p][1] for p in ANCHOR_PARTICIPANTS)

grid = pd.DataFrame({
    "timestamp_us": np.arange(
        start,
        end + GRID_US,
        GRID_US,
        dtype=np.int64
    )
})

wide = grid.copy()

wide.insert(
    1,
    "datetime_utc",
    pd.to_datetime(wide["timestamp_us"], unit="us", utc=True)
)


# ============================================================
# Merge all participants onto P1/P3 timeline
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    for sensor_name, df in all_streams[p].items():

        df = df[
            (df["timestamp_us"] >= start - GRID_US) &
            (df["timestamp_us"] <= end + GRID_US)
        ]

        tol = TOL_SLOW if sensor_name == "skin_temp" else TOL_FAST

        merged = pd.merge_asof(
            grid,
            df,
            on="timestamp_us",
            direction="nearest",
            tolerance=tol
        )

        for col in SCHEMA[sensor_name]:
            wide[f"{pre}_{col}"] = merged[col].values


# Empty label placeholders
for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# Useful availability flag for shorter participant
wide["p2_oe_available"] = wide["p2_acc_x"].notna()


# ============================================================
# Save
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_10_openearable_merged_50hz.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_10_openearable_merged_50hz.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print(
    "\nshape:", wide.shape,
    "| P1/P3 anchor overlap",
    f"{(end - start) / 1e6:.1f}s ->",
    local_path
)

print("copied ->", drive_path)


# ============================================================
# Missing summary
# ============================================================

miss = wide.iloc[:, 2:].isna().mean().sort_values(ascending=False)

print("\nTop missing columns:")
print(miss.head(20))

print("\nAverage missing:")
print(f"{wide.iloc[:, 2:].isna().mean().mean() * 100:.2f}%")

print("\nParticipant-level missing average:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nP2 OpenEarable availability:")
print("available rows:", wide["p2_oe_available"].sum(), "/", len(wide))
print("available %   :", wide["p2_oe_available"].mean() * 100)


# --- CELL 122 (code cell #106) ---
# ============================================================
# GROUP 10 XSENS MERGE - WRAP-SAFE VERSION
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_10/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_10"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["xsens_time_s_fixed"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("xsens_time_s_fixed")
          .drop_duplicates("xsens_time_s_fixed")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD + DIAGNOSE
# ============================================================

raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["xsens_time_s_fixed"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("fixed time start:", df["xsens_time_s_fixed"].min())
    print("fixed time end  :", df["xsens_time_s_fixed"].max())
    print("duration min    :", df["xsens_time_s_fixed"].max() / 60)
    print("median dt       :", diffs.median())
    print("max gap         :", diffs.max())
    print("gaps > 1s       :", (diffs > 1).sum())


# ============================================================
# MASTER GRID = all-three overlap using P1 timeline
# If one participant has gaps/short duration, we adjust after output.
# ============================================================

durations = {
    p: raw[p]["xsens_time_s_fixed"].max()
    for p in PARTICIPANTS
}

print("\nDurations:")
for p, d in durations.items():
    print(p, f"{d:.2f}s = {d/60:.2f} min")

end = min(durations.values())

p1 = raw["Participant1"].copy()

master = p1[
    (p1["xsens_time_s_fixed"] >= 0) &
    (p1["xsens_time_s_fixed"] <= end)
][["xsens_time_s_fixed"]].copy()

master = master.sort_values("xsens_time_s_fixed").reset_index(drop=True)
master["time_s"] = master["xsens_time_s_fixed"].round(4)

wide = master[["time_s"]].copy()


# ============================================================
# MERGE ALL PARTICIPANTS
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["xsens_time_s_fixed"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("xsens_time_s_fixed"),
        left_on="time_s",
        right_on="xsens_time_s_fixed",
        direction="nearest",
        tolerance=0.02
    )

    wide = wide.drop(columns=["xsens_time_s_fixed"])


for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# ============================================================
# SAVE
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_10_xsens_merged_30hz.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_10_xsens_merged_30hz.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 10 XSENS MERGED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nTop missing columns:")
print(wide.isna().mean().sort_values(ascending=False).head(20))


# --- CELL 123 (code cell #107) ---
# ============================================================
# GROUP 10 XSENS MERGE - CONTINUOUS P1/P3 ANCHOR VERSION
# P2 is short, so P2 becomes NaN after it stops.
# P1 has gaps/extra chunks, so gaps are preserved as NaN.
# Timeline is based on P3 clean duration.
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil

BASE      = "/content/drive/MyDrive/thesis/data/group_10/xsens"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/xsens_merged"
LOCAL_OUT = "/content/xsens_merged_group_10_continuous"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

PREFIX = {
    "Participant1": "p1",
    "Participant2": "p2",
    "Participant3": "p3"
}

SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32
GRID_S = 1 / 30
TOL_S = 0.02


def load_xsens_fixed(participant):
    path = os.path.join(BASE, f"{participant}.csv")

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["PacketCounter", "SampleTimeFine"] + SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SampleTimeFine"]).copy()
    df["SampleTimeFine"] = df["SampleTimeFine"].astype(np.int64)

    first_tick = int(df["SampleTimeFine"].iloc[0])

    df["time_s"] = (
        (df["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    df = (
        df.sort_values("time_s")
          .drop_duplicates("time_s")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD + DIAGNOSE
# ============================================================

raw = {}

for p in PARTICIPANTS:
    df = load_xsens_fixed(p)
    raw[p] = df

    diffs = df["time_s"].diff()

    print("\n" + "="*60)
    print(p)
    print("="*60)
    print("rows:", len(df))
    print("time start:", df["time_s"].min())
    print("time end  :", df["time_s"].max())
    print("duration min:", df["time_s"].max() / 60)
    print("median dt:", diffs.median())
    print("max gap:", diffs.max())
    print("gaps > 1s:", (diffs > 1).sum())


# ============================================================
# Use P3 clean full duration as main timeline.
# This keeps ~25.45 min instead of cutting to P2's 15.28 min.
# ============================================================

end = raw["Participant3"]["time_s"].max()

grid = pd.DataFrame({
    "time_s": np.round(np.arange(0, end + GRID_S, GRID_S), 4)
})

wide = grid.copy()


# ============================================================
# Merge all participants onto continuous grid
# ============================================================

for p in PARTICIPANTS:
    pre = PREFIX[p]

    sub = raw[p][["time_s"] + SENSOR_COLS].copy()

    sub = sub.rename(
        columns={
            c: f"{pre}_{c.lower()}"
            for c in SENSOR_COLS
        }
    )

    wide = pd.merge_asof(
        wide.sort_values("time_s"),
        sub.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=TOL_S
    )


# Empty labels
for pre in ["p1", "p2", "p3"]:
    wide[f"label_{pre}"] = pd.NA


# Availability flags
for pre in ["p1", "p2", "p3"]:
    wide[f"{pre}_xsens_available"] = wide[f"{pre}_acc_x"].notna()


# ============================================================
# SAVE
# ============================================================

local_path = os.path.join(
    LOCAL_OUT,
    "group_10_xsens_merged_30hz_CONTINUOUS.csv"
)

drive_path = os.path.join(
    DRIVE_OUT,
    "group_10_xsens_merged_30hz_CONTINUOUS.csv"
)

wide.to_csv(local_path, index=False)
shutil.copy2(local_path, drive_path)

print("\n" + "="*60)
print("GROUP 10 XSENS CONTINUOUS MERGE SAVED")
print("="*60)
print("shape:", wide.shape)
print("duration:", wide["time_s"].max(), "s =", wide["time_s"].max()/60, "min")
print("saved ->", drive_path)

print("\nParticipant-level missing:")
for pre in ["p1", "p2", "p3"]:
    cols = [c for c in wide.columns if c.startswith(pre + "_") and not c.endswith("_available")]
    print(pre, f"{wide[cols].isna().mean().mean() * 100:.2f}%")

print("\nAvailability:")
for pre in ["p1", "p2", "p3"]:
    print(pre, f"{wide[f'{pre}_xsens_available'].mean() * 100:.2f}%")

print("\nTop missing columns:")
print(wide.isna().mean().sort_values(ascending=False).head(20))


# --- CELL 124 (code cell #108) ---
import pandas as pd
import numpy as np
import os
import shutil

# ---- paths ----
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_with_individual_build.csv"
LOCAL_OUT = "/content/Group_10_with_individual_build.csv"
# ----------------

PEOPLE = ["Rick", "Lennart", "Ali"]

CUTOFF = {
    "Rick":    22 * 60 + 40,   # 1360 s
    "Lennart": 19 * 60 + 20,   # 1160 s
    "Ali":     22 * 60 + 20,   # 1340 s
}

START = 0.0
MIN_GAP_S = 0.0

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

df = pd.read_csv(
    ELAN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].astype(str).str.strip()
df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.dropna(subset=["tier", "begin_s", "end_s", "label"]).copy()

print("Original tiers:")
print(sorted(df["tier"].unique()))

all_tiers = sorted(df["tier"].dropna().unique())


def norm(x):
    return str(x).strip().lower()


INVOLVING = {}

for person in PEOPLE:
    person_norm = norm(person)
    tiers = []

    for tier in all_tiers:
        tier_norm = norm(tier)

        if tier_norm in ["whole group", "whole_group"]:
            tiers.append(tier)
            continue

        tier_parts = tier_norm.replace("_", " ").split()

        if tier_norm == person_norm:
            tiers.append(tier)

        elif person_norm in tier_parts:
            tiers.append(tier)

    INVOLVING[person] = tiers

print("\nInvolving tiers:")
for person, tiers in INVOLVING.items():
    print(person, "->", tiers)

print("\nManual cutoffs:")
for person, cutoff in CUTOFF.items():
    print(f"{person}: {cutoff:.3f}s = {cutoff/60:.2f} min")


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []

    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def gaps(busy, start, end):
    busy = merge_iv([
        [max(s, start), min(e, end)]
        for s, e in busy
        if e > start and s < end
    ])

    g = []
    cur = start

    for s, e in busy:
        if s > cur:
            g.append((cur, s))
        cur = max(cur, e)

    if cur < end:
        g.append((cur, end))

    return g


def hms(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


new_rows = []

for person, tiers in INVOLVING.items():

    busy = df[df.tier.isin(tiers)][["begin_s", "end_s"]].values.tolist()

    for s, e in gaps(busy, START, CUTOFF[person]):

        d = e - s

        if d < MIN_GAP_S or d <= 0:
            continue

        new_rows.append({
            "tier": person,
            "blank": "",
            "begin_hms": hms(s),
            "begin_s": round(s, 3),
            "end_hms": hms(e),
            "end_s": round(e, 3),
            "dur_hms": hms(d),
            "dur_s": round(d, 3),
            "label": "individual_build"
        })


both = (
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    .sort_values(["tier", "begin_s"])
    .reset_index(drop=True)
)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in both.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("\n================================================")
print("GROUP 10 individual_build COMPLETE")
print("================================================")
print(
    f"{len(df)} -> {len(both)} annotations "
    f"({len(new_rows)} individual_build added)"
)

print("\nindividual_build count per person:")
print(
    both[both["label"] == "individual_build"]
    .groupby("tier")
    .size()
)

print("\nSaved to:")
print(DRIVE_OUT)
print("================================================")


# --- CELL 125 (code cell #109) ---
import pandas as pd
import shutil
import os

# ---- paths ----
IN_PATH   = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_with_individual_build.csv"
DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_individual_build_renamed.csv"
LOCAL_OUT = "/content/Group_10_individual_build_renamed.csv"
# ----------------

NAME = {
    "Rick": "Participant1",
    "Lennart": "Participant2",
    "Ali": "Participant3"
}

COLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]


def rename_tier(t):
    t = str(t).strip()
    t_lower = t.lower()

    if t_lower in ["whole group", "whole_group"]:
        return "Whole_Group"

    parts_original = t.replace("_", " ").split()

    name_lookup = {
        k.lower(): v
        for k, v in NAME.items()
    }

    parts_lower = [
        p.lower()
        for p in parts_original
    ]

    if all(p in name_lookup for p in parts_lower):
        mapped = sorted(
            [name_lookup[p] for p in parts_lower],
            key=lambda x: int(x.replace("Participant", ""))
        )
        return "_".join(mapped)

    return t


df = pd.read_csv(
    IN_PATH,
    header=None,
    names=COLS,
    dtype={"blank": str}
)

df["tier"] = df["tier"].map(rename_tier)

df["begin_s"] = pd.to_numeric(df["begin_s"], errors="coerce")
df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
df["dur_s"] = pd.to_numeric(df["dur_s"], errors="coerce")

df = df.sort_values(["tier", "begin_s"]).reset_index(drop=True)

fmt = lambda x: str(round(float(x), 3))

lines = [
    f'"{r.tier}","",{r.begin_hms},{fmt(r.begin_s)},{r.end_hms},{fmt(r.end_s)},'
    f'{r.dur_hms},{fmt(r.dur_s)},"{r.label}"'
    for r in df.itertuples()
]

open(LOCAL_OUT, "w").write("\n".join(lines) + "\n")

os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("tiers now:")
for t in sorted(df["tier"].unique()):
    print(" -", t)

print("\ncopied ->", DRIVE_OUT)

print("\nannotation counts per tier:")
print(df.groupby("tier").size())


# --- CELL 126 (code cell #110) ---
import pandas as pd
import numpy as np
import os
import shutil

# ============================================================
# GROUP 10 - LABEL OPENEARAMBLE USING VIDEO TIME
# OE uses P1/P3 anchor because P2 OE is partial
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_individual_build_renamed.csv"
OE_PATH   = "/content/drive/MyDrive/thesis/data/group_10/openearable_merged/group_10_openearable_merged_50hz.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/openearable_labeled/group_10_openearable_labeled.csv"
LOCAL_OUT = "/content/group_10_openearable_labeled.csv"

# Group 10 video start:
# 2026-04-30 15:33:00 UTC
VIDEO_START_US = int(pd.Timestamp("2026-04-30 15:33:00", tz="UTC").timestamp() * 1_000_000)

print("VIDEO_START_US:", VIDEO_START_US)

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
oe = pd.read_csv(OE_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

# absolute OE time -> video-relative time
oe["video_time_s"] = (
    oe["timestamp_us"].astype(np.int64) - VIDEO_START_US
) / 1e6

# remove old placeholder / label columns if rerunning
oe = oe.drop(
    columns=[c for c in ["label_p1", "label_p2", "label_p3"] if c in oe.columns],
    errors="ignore"
)

oe = oe.drop(
    columns=[f"label_{t}" for t in TIERS if f"label_{t}" in oe.columns],
    errors="ignore"
)

# place video_time_s after datetime_utc
cols = list(oe.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

if "datetime_utc" in cols:
    cols.insert(cols.index("datetime_utc") + 1, "video_time_s")
else:
    cols.insert(1, "video_time_s")

oe = oe[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        oe[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        oe[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    oe[label_col] = m["_lab"].where(valid, other=pd.NA).values

# keep P2 OE availability flag if present; otherwise create it
if "p2_oe_available" not in oe.columns:
    oe["p2_oe_available"] = oe["p2_acc_x"].notna()

# save
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
oe.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 10 OPENEARAMBLE LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(oe.shape)

print("\nVideo time range:")
print("start:", oe["video_time_s"].min())
print("end  :", oe["video_time_s"].max())
print("dur  :", oe["video_time_s"].max() - oe["video_time_s"].min())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nP2 OpenEarable availability:")
print("available rows:", oe["p2_oe_available"].sum(), "/", len(oe))
print("available %   :", oe["p2_oe_available"].mean() * 100)

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = oe[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(oe) * 100:.1f}%)")

print("================================================")


# --- CELL 127 (code cell #111) ---
import pandas as pd
import numpy as np
import os
import shutil

# ============================================================
# GROUP 10 - LABEL XSENS USING VIDEO TIME
# XSens started 8s after video
# Therefore: video_time_s = time_s + 8
# ============================================================

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_individual_build_renamed.csv"
XSENS_PATH = "/content/drive/MyDrive/thesis/data/group_10/xsens_merged/group_10_xsens_merged_30hz_CONTINUOUS.csv"

DRIVE_OUT = "/content/drive/MyDrive/thesis/data/group_10/xsens_labeled/group_10_xsens_labeled.csv"
LOCAL_OUT = "/content/group_10_xsens_labeled.csv"

XSENS_TO_VIDEO_OFFSET_S = 8.0

ECOLS = [
    "tier", "blank",
    "begin_hms", "begin_s",
    "end_hms", "end_s",
    "dur_hms", "dur_s",
    "label"
]

TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group"
]

elan = pd.read_csv(ELAN_PATH, header=None, names=ECOLS)
xsens = pd.read_csv(XSENS_PATH, low_memory=False)

elan["begin_s"] = pd.to_numeric(elan["begin_s"], errors="coerce")
elan["end_s"] = pd.to_numeric(elan["end_s"], errors="coerce")
elan["dur_s"] = pd.to_numeric(elan["dur_s"], errors="coerce")

xsens["time_s"] = pd.to_numeric(xsens["time_s"], errors="coerce")

# XSens relative time -> video-relative time
xsens["video_time_s"] = xsens["time_s"] + XSENS_TO_VIDEO_OFFSET_S

# remove old placeholder / label columns if rerunning
xsens = xsens.drop(
    columns=[
        c for c in ["label_p1", "label_p2", "label_p3"]
        if c in xsens.columns
    ],
    errors="ignore"
)

xsens = xsens.drop(
    columns=[
        f"label_{t}" for t in TIERS
        if f"label_{t}" in xsens.columns
    ],
    errors="ignore"
)

# place video_time_s after time_s
cols = list(xsens.columns)

if "video_time_s" in cols:
    cols.remove("video_time_s")

cols.insert(cols.index("time_s") + 1, "video_time_s")
xsens = xsens[cols]

# apply labels
for tier in TIERS:

    sub = (
        elan[elan["tier"] == tier][["begin_s", "end_s", "label"]]
        .dropna(subset=["begin_s", "end_s", "label"])
        .sort_values("begin_s")
        .reset_index(drop=True)
        .rename(
            columns={
                "begin_s": "video_time_s",
                "end_s": "_end",
                "label": "_lab"
            }
        )
    )

    label_col = f"label_{tier}"

    if len(sub) == 0:
        xsens[label_col] = pd.NA
        continue

    m = pd.merge_asof(
        xsens[["video_time_s"]].reset_index(),
        sub,
        on="video_time_s",
        direction="backward"
    )

    valid = (
        (m["video_time_s"] < m["_end"]) &
        (m["_lab"].notna())
    )

    xsens[label_col] = m["_lab"].where(valid, other=pd.NA).values

# keep availability flags from merged file
os.makedirs(os.path.dirname(DRIVE_OUT), exist_ok=True)
xsens.to_csv(LOCAL_OUT, index=False)
shutil.copy2(LOCAL_OUT, DRIVE_OUT)

print("================================================")
print("GROUP 10 XSENS LABELED")
print("================================================")
print("Saved to:")
print(DRIVE_OUT)

print("\nShape:")
print(xsens.shape)

print("\nTiming check:")
print("XSens time_s start       :", xsens["time_s"].min())
print("XSens time_s end         :", xsens["time_s"].max())
print("XSens video_time_s start :", xsens["video_time_s"].min())
print("XSens video_time_s end   :", xsens["video_time_s"].max())

print("\nELAN range:")
print("start:", elan["begin_s"].min())
print("end  :", elan["end_s"].max())

print("\nXSens availability:")
for pre in ["p1", "p2", "p3"]:
    col = f"{pre}_xsens_available"
    if col in xsens.columns:
        print(pre, f"{xsens[col].mean() * 100:.2f}%")

print("\nLabel coverage:")
for tier in TIERS:
    col = f"label_{tier}"
    n = xsens[col].notna().sum()
    print(f"{col}: {n} rows ({n / len(xsens) * 100:.1f}%)")

print("================================================")


# --- CELL 128 (code cell #112) ---
# ============================================================
# GROUP 10 - VISUAL SANITY CHECK
# Full labels + Whole_Group label list for OpenEarable and XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_10"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_10_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_10_xsens_labeled.csv"
)

TIME_COL = "video_time_s"

USERS = [1, 2, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]

SIGNAL_TYPE = "acc_mag"

SMOOTH_WINDOW_OE = 25
SMOOTH_WINDOW_XSENS = 15

FIGSIZE = (24, 12)
LABEL_ALPHA = 0.14
TEXT_EVERY_N_SEGMENTS = 5

ROBUST_YLIM = True
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99


def moving_average(signal, window):
    if window <= 1:
        return signal
    return (
        pd.Series(signal)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def robust_ylim(signal, lower_p=1, upper_p=99):
    clean = signal[np.isfinite(signal)]
    if len(clean) == 0:
        return None

    low = np.percentile(clean, lower_p)
    high = np.percentile(clean, upper_p)
    margin = (high - low) * 0.15

    return low - margin, high + margin


def get_signal(df, user, source, signal_type):
    if source == "oe":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyro_x", f"p{user}_gyro_y", f"p{user}_gyro_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    elif source == "xsens":
        if signal_type == "acc_mag":
            x, y, z = f"p{user}_acc_x", f"p{user}_acc_y", f"p{user}_acc_z"
        elif signal_type == "gyro_mag":
            x, y, z = f"p{user}_gyr_x", f"p{user}_gyr_y", f"p{user}_gyr_z"
        else:
            raise ValueError("Use acc_mag or gyro_mag")

    else:
        raise ValueError("source must be oe or xsens")

    return np.sqrt(
        pd.to_numeric(df[x], errors="coerce")**2 +
        pd.to_numeric(df[y], errors="coerce")**2 +
        pd.to_numeric(df[z], errors="coerce")**2
    )


def get_label_segments(df, time_col, label_col):
    if label_col not in df.columns:
        return []

    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def print_label_summary(df, title):
    print("\n================================================")
    print(title)
    print("LABEL SUMMARY")
    print("================================================")

    print("\nTime range:")
    print("start:", df[TIME_COL].min())
    print("end  :", df[TIME_COL].max())
    print("dur  :", df[TIME_COL].max() - df[TIME_COL].min())

    for col in LABEL_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str).unique().tolist()
        values = [v for v in values if v != ""]

        print(f"\n{col}: {len(values)} unique labels")
        for v in values:
            count = (df[col] == v).sum()
            pct = count / len(df) * 100
            print(f"  - {v}: {count} rows ({pct:.1f}%)")

    if "p2_oe_available" in df.columns:
        print("\nP2 OpenEarable availability:")
        print(df["p2_oe_available"].value_counts(dropna=False))
        print("available %:", df["p2_oe_available"].mean() * 100)

    for pre in ["p1", "p2", "p3"]:
        col = f"{pre}_xsens_available"
        if col in df.columns:
            print(f"\n{pre} XSens availability:")
            print(df[col].value_counts(dropna=False))
            print("available %:", df[col].mean() * 100)

    print("================================================")


def plot_full_all_labels(
    csv_path,
    source,
    title,
    signal_type="acc_mag",
    smooth_window=25,
    label_alpha=0.14,
    text_every_n_segments=5,
    robust_y=True
):
    df = pd.read_csv(csv_path, low_memory=False)

    existing_label_cols = [
        col for col in LABEL_COLS
        if col in df.columns
    ]

    times = df[TIME_COL].values

    print_label_summary(df, title)

    all_labels = []

    for col in existing_label_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        all_labels.extend(vals)

    unique_labels = sorted(set(all_labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    label_colors = {
        label: cmap(i % 20)
        for i, label in enumerate(unique_labels)
    }

    fig, axes = plt.subplots(len(USERS), 1, figsize=FIGSIZE, sharex=True)

    for ax, user in zip(axes, USERS):
        sig = get_signal(df, user, source, signal_type)
        sig = moving_average(sig, smooth_window)

        ax.plot(
            times,
            sig,
            linewidth=0.75,
            label=f"Participant {user} - {signal_type}"
        )

        if robust_y:
            ylim = robust_ylim(sig, LOWER_PERCENTILE, UPPER_PERCENTILE)
            if ylim is not None:
                ax.set_ylim(ylim)

        segment_counter = 0

        for label_col in existing_label_cols:
            segments = get_label_segments(df, TIME_COL, label_col)

            for start, end, label in segments:
                if label == "":
                    continue

                ax.axvspan(
                    start,
                    end,
                    color=label_colors.get(label, "gray"),
                    alpha=label_alpha
                )

                if segment_counter % text_every_n_segments == 0:
                    mid = (start + end) / 2
                    y_top = ax.get_ylim()[1]

                    ax.text(
                        mid,
                        y_top,
                        label,
                        rotation=90,
                        fontsize=5,
                        ha="center",
                        va="top"
                    )

                segment_counter += 1

        ax.set_title(f"{title} - Participant {user} - {signal_type}")
        ax.set_ylabel(f"p{user}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Video time (seconds)")

    legend_patches = [
        Patch(
            facecolor=label_colors[label],
            alpha=label_alpha,
            label=label
        )
        for label in unique_labels
    ]

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Labels",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

    return df


df_g10_oe = plot_full_all_labels(
    csv_path=OE_PATH,
    source="oe",
    title="Group 10 OpenEarable",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_OE,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

df_g10_xsens = plot_full_all_labels(
    csv_path=XSENS_PATH,
    source="xsens",
    title="Group 10 XSens",
    signal_type=SIGNAL_TYPE,
    smooth_window=SMOOTH_WINDOW_XSENS,
    label_alpha=LABEL_ALPHA,
    text_every_n_segments=TEXT_EVERY_N_SEGMENTS,
    robust_y=ROBUST_YLIM
)

print("\nOpenEarable Whole_Group labels:")
print(df_g10_oe["label_Whole_Group"].dropna().unique())

print("\nXSens Whole_Group labels:")
print(df_g10_xsens["label_Whole_Group"].dropna().unique())


# --- CELL 129 (code cell #113) ---
import os
import pandas as pd
import numpy as np

# ============================================================
# GROUP 10 - RAW ROOT FILE LENGTH / DURATION CHECK
# Checks original raw files before any merging
# ============================================================

OE_BASE = "/content/drive/MyDrive/thesis/data/group_10/openearable"
XSENS_BASE = "/content/drive/MyDrive/thesis/data/group_10/xsens"

PARTICIPANTS = ["Participant1", "Participant2", "Participant3"]

OE_STREAMS = [
    "acc",
    "gyro",
    "mgnt",
    "bone_acc",
    "baro",
    "env_temp",
    "skin_temp",
    "ppg"
]

XSENS_SENSOR_COLS = [
    "Euler_X", "Euler_Y", "Euler_Z",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z"
]

MOD = 2**32


# ============================================================
# OpenEarable raw files
# ============================================================

print("\n" + "="*80)
print("GROUP 10 RAW OPENEARAMBLE FILES")
print("="*80)

for p in PARTICIPANTS:
    print("\n" + "-"*80)
    print(p)
    print("-"*80)

    stream_windows = []

    for stream in OE_STREAMS:
        path = os.path.join(OE_BASE, p, f"{p}_{stream}.csv")

        if not os.path.exists(path):
            print(f"{stream:10s} MISSING -> {path}")
            continue

        try:
            df = pd.read_csv(path, header=None, low_memory=False)
            ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(np.int64)

            if len(ts) == 0:
                print(f"{stream:10s} rows={len(df):>8} | no valid timestamps")
                continue

            t0 = int(ts.min())
            t1 = int(ts.max())
            duration_s = (t1 - t0) / 1e6

            stream_windows.append((t0, t1))

            print(
                f"{stream:10s} rows={len(df):>8} | "
                f"{pd.to_datetime(t0, unit='us', utc=True)} -> "
                f"{pd.to_datetime(t1, unit='us', utc=True)} | "
                f"duration={duration_s:8.2f}s = {duration_s/60:6.2f} min"
            )

        except Exception as e:
            print(f"{stream:10s} ERROR: {e}")

    if stream_windows:
        imu_like = []
        for stream in ["acc", "gyro", "mgnt"]:
            path = os.path.join(OE_BASE, p, f"{p}_{stream}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, header=None, low_memory=False)
                ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(np.int64)
                if len(ts) > 0:
                    imu_like.append((int(ts.min()), int(ts.max())))

        if imu_like:
            imu_start = min(x[0] for x in imu_like)
            imu_end = max(x[1] for x in imu_like)
            imu_dur = (imu_end - imu_start) / 1e6

            print("\nIMU combined window:")
            print(
                f"{pd.to_datetime(imu_start, unit='us', utc=True)} -> "
                f"{pd.to_datetime(imu_end, unit='us', utc=True)} | "
                f"duration={imu_dur:.2f}s = {imu_dur/60:.2f} min"
            )


# ============================================================
# XSens raw files
# ============================================================

print("\n" + "="*80)
print("GROUP 10 RAW XSENS FILES")
print("="*80)

for p in PARTICIPANTS:
    print("\n" + "-"*80)
    print(p)
    print("-"*80)

    path = os.path.join(XSENS_BASE, f"{p}.csv")

    if not os.path.exists(path):
        print("MISSING ->", path)
        continue

    df = pd.read_csv(path, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df.columns = [str(c).strip() for c in df.columns]

    print("raw rows:", len(df))
    print("columns:", list(df.columns))

    if "SampleTimeFine" not in df.columns:
        print("No SampleTimeFine column found.")
        continue

    df["SampleTimeFine"] = pd.to_numeric(df["SampleTimeFine"], errors="coerce")
    valid = df.dropna(subset=["SampleTimeFine"]).copy()

    valid["SampleTimeFine"] = valid["SampleTimeFine"].astype(np.int64)

    if len(valid) == 0:
        print("No valid SampleTimeFine values.")
        continue

    first_tick = int(valid["SampleTimeFine"].iloc[0])

    valid["time_s_fixed"] = (
        (valid["SampleTimeFine"].astype(np.int64) - first_tick) % MOD
    ) / 1e6

    valid = valid.sort_values("time_s_fixed").reset_index(drop=True)

    diffs = valid["time_s_fixed"].diff()

    print("valid SampleTimeFine rows:", len(valid))
    print("fixed time start:", valid["time_s_fixed"].min())
    print("fixed time end  :", valid["time_s_fixed"].max())
    print("duration        :", f"{valid['time_s_fixed'].max():.2f}s = {valid['time_s_fixed'].max()/60:.2f} min")
    print("median dt       :", diffs.median())
    print("max gap         :", diffs.max())
    print("gaps > 1s       :", (diffs > 1).sum())

    if (diffs > 1).sum() > 0:
        print("\nBig gaps:")
        gap_rows = valid.loc[diffs > 1, ["PacketCounter", "SampleTimeFine", "time_s_fixed"]].copy()
        gap_rows["dt_s"] = diffs[diffs > 1].values
        print(gap_rows.to_string(index=True))


# --- CELL 130 (code cell #114) ---
# ============================================================
# GROUP 10 - ZOOM AROUND Whole_Group synchronizaiton_move
# OpenEarable + XSens
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_10"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_10_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_10_xsens_labeled_cleaned.csv"
)

TIME_COL = "video_time_s"
SYNC_LABEL_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

USERS = [1, 2, 3]
ZOOM_MARGIN = 25

SMOOTH_OE = 10
SMOOTH_XSENS = 5


def acc_mag(df, user):
    return np.sqrt(
        pd.to_numeric(df[f"p{user}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_z"], errors="coerce")**2
    )


def smooth(x, win):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def combined_acc(df):
    sigs = [acc_mag(df, u) for u in USERS]
    return np.nanmean(np.vstack(sigs), axis=0)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def plot_sync(csv_path, source_name, smooth_win):
    df = pd.read_csv(csv_path, low_memory=False)

    segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_LABEL_COL)
        if seg[2] == SYNC_LABEL
    ]

    print("\n================================================")
    print(f"Group 10 {source_name}")
    print("================================================")

    if len(segments) == 0:
        print(f"No {SYNC_LABEL} found.")
        print("Available Whole_Group labels:")
        print(df[SYNC_LABEL_COL].dropna().unique())
        return None

    times = df[TIME_COL].values

    signals = {
        u: smooth(acc_mag(df, u), smooth_win)
        for u in USERS
    }

    combined = smooth(combined_acc(df), smooth_win)

    results = []

    for event_i, (s, e, lab) in enumerate(segments):
        mid = (s + e) / 2

        print(f"event {event_i}: {s:.3f}s -> {e:.3f}s | mid={mid:.3f}s | dur={e-s:.3f}s")

        mask = (times >= s - ZOOM_MARGIN) & (times <= e + ZOOM_MARGIN)

        local_idx = np.where(mask)[0]
        peak_idx = local_idx[np.nanargmax(combined[mask])]
        peak_time = times[peak_idx]
        peak_value = combined[peak_idx]

        print(f"local peak: {peak_time:.3f}s | peak - middle = {peak_time - mid:.3f}s | value={peak_value:.3f}")

        results.append({
            "event": event_i,
            "start": s,
            "end": e,
            "middle": mid,
            "peak_time": peak_time,
            "delta": peak_time - mid,
            "peak_value": peak_value
        })

        fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

        axes[0].plot(times[mask], combined[mask], linewidth=0.9, label=f"Combined {source_name} acc_mag")
        axes[0].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
        axes[0].axvline(mid, linestyle="--", label="label middle")
        axes[0].axvline(peak_time, linestyle="--", label="local peak")
        axes[0].set_title(f"Group 10 {source_name} | Combined | event {event_i}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        for i, u in enumerate(USERS, start=1):
            axes[i].plot(times[mask], signals[u][mask], linewidth=0.8, label=f"P{u} {source_name} acc_mag")
            axes[i].axvspan(s, e, alpha=0.25, label=SYNC_LABEL)
            axes[i].axvline(mid, linestyle="--", label="label middle")
            axes[i].axvline(peak_time, linestyle="--", label="local peak")
            axes[i].set_title(f"Group 10 {source_name} | Participant {u}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper left")

        axes[-1].set_xlabel("Video time (seconds)")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)


oe_sync = plot_sync(OE_PATH, "OpenEarable", SMOOTH_OE)
xsens_sync = plot_sync(XSENS_PATH, "XSens", SMOOTH_XSENS)

print("\n================================================")
print("GROUP 10 SYNC RESULT TABLES")
print("================================================")

print("\nOpenEarable:")
display(oe_sync)

print("\nXSens:")
display(xsens_sync)


# --- CELL 131 (code cell #115) ---
import pandas as pd
import numpy as np
import os
import shutil

# ============================================================
# GROUP 10 XSENS CLEANING
# Remove impossible numeric artifacts before sync decision
# ============================================================

IN_PATH = "/content/drive/MyDrive/thesis/data/group_10/xsens_labeled/group_10_xsens_labeled.csv"
OUT_PATH = "/content/drive/MyDrive/thesis/data/group_10/xsens_labeled/group_10_xsens_labeled_cleaned.csv"

df = pd.read_csv(IN_PATH, low_memory=False)

sensor_cols = [
    c for c in df.columns
    if (
        c.startswith("p1_") or c.startswith("p2_") or c.startswith("p3_")
    )
    and (
        "_acc_" in c or "_gyr_" in c or "_euler_" in c
    )
]

print("================================================")
print("BEFORE CLEANING - largest absolute values")
print("================================================")
print(df[sensor_cols].abs().max().sort_values(ascending=False).head(25))

# Conservative physical thresholds
# Xsens acceleration can spike, but 200 m/s² is already very high.
# Gyro above 500 deg/s also suspicious for this task.
# Euler should stay within +/-180 or +/-90 depending axis, but use 360 as safety.
artifact_mask = pd.Series(False, index=df.index)

for c in sensor_cols:
    vals = pd.to_numeric(df[c], errors="coerce")

    if "_acc_" in c:
        artifact_mask |= vals.abs() > 200

    elif "_gyr_" in c:
        artifact_mask |= vals.abs() > 700

    elif "_euler_" in c:
        artifact_mask |= vals.abs() > 360

# Also catch extreme numeric explosions
for c in sensor_cols:
    vals = pd.to_numeric(df[c], errors="coerce")
    artifact_mask |= vals.abs() > 1e6

print("\n================================================")
print("CLEANING SUMMARY")
print("================================================")
print("Artifact rows detected:", artifact_mask.sum())
print("Artifact row percentage:", artifact_mask.mean() * 100, "%")

df_clean = df.copy()
df_clean.loc[artifact_mask, sensor_cols] = np.nan

print("\nAfter cleaning - largest absolute values:")
print(df_clean[sensor_cols].abs().max().sort_values(ascending=False).head(25))

df_clean.to_csv(OUT_PATH, index=False)

print("\nSaved cleaned XSens file:")
print(OUT_PATH)


# --- CELL 132 (code cell #116) ---
# ============================================================
# GROUP 10 - FINAL SYNC SHIFT USING LATE EVENT ONLY
# Uses P1 + P3 only because P2 is unavailable late.
# Uses cleaned XSens.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_10"

OE_IN = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_10_openearable_labeled.csv"
)

XSENS_IN = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_10_xsens_labeled_cleaned.csv"
)

OE_OUT = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_10_openearable_labeled_SHIFTED.csv"
)

XSENS_OUT = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_10_xsens_labeled_SHIFTED.csv"
)

TIME_COL = "video_time_s"
SYNC_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

# Use P1 and P3 only for late sync
USERS_FOR_SYNC = [1, 3]

LABEL_COLS = [
    "label_Participant1",
    "label_Participant2",
    "label_Participant3",
    "label_Participant1_Participant2",
    "label_Participant1_Participant3",
    "label_Participant2_Participant3",
    "label_Whole_Group"
]


def acc_mag(df, user):
    return np.sqrt(
        pd.to_numeric(df[f"p{user}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_z"], errors="coerce")**2
    )


def smooth(x, win):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def combined_acc_p1_p3(df, smooth_win):
    sigs = []
    for u in USERS_FOR_SYNC:
        sigs.append(acc_mag(df, u))
    return smooth(np.nanmean(np.vstack(sigs), axis=0), smooth_win)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):
        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def choose_late_sync_segment(df):
    segs = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_COL)
        if seg[2] == SYNC_LABEL
    ]

    if len(segs) == 0:
        raise ValueError("No synchronizaiton_move found.")

    # choose latest sync event
    return sorted(segs, key=lambda x: x[0])[-1]


def shift_label_column(df, label_col, delta_s):
    times = df[TIME_COL].values
    shifted = np.array([pd.NA] * len(df), dtype=object)

    segments = get_label_segments(df, TIME_COL, label_col)

    for s, e, lab in segments:
        new_s = s + delta_s
        new_e = e + delta_s
        mask = (times >= new_s) & (times < new_e)
        shifted[mask] = lab

    return shifted


def shift_all_labels(df, delta_s):
    out = df.copy()

    for col in LABEL_COLS:
        if col in out.columns:
            out[col] = shift_label_column(out, col, delta_s)

    return out


def process_file(in_path, out_path, title, smooth_win):
    df = pd.read_csv(in_path, low_memory=False)

    s, e, lab = choose_late_sync_segment(df)
    mid = (s + e) / 2

    times = df[TIME_COL].values
    sig = combined_acc_p1_p3(df, smooth_win)

    # Search after label middle, because plots show movement occurs after the label
    search_start = mid - 5
    search_end = mid + 35

    mask = (times >= search_start) & (times <= search_end)

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(sig[mask])]

    peak_time = times[peak_idx]
    delta_s = peak_time - mid

    shifted = shift_all_labels(df, delta_s)
    shifted.to_csv(out_path, index=False)

    print("\n================================================")
    print(title)
    print("GROUP 10 LATE SYNC SHIFT")
    print("================================================")
    print(f"Sync label start : {s:.3f}s")
    print(f"Sync label end   : {e:.3f}s")
    print(f"Sync label middle: {mid:.3f}s")
    print(f"Detected peak    : {peak_time:.3f}s")
    print(f"Shift delta      : {delta_s:.3f}s")
    print("Labels move RIGHT / later.")
    print("Saved shifted file:")
    print(out_path)
    print("================================================")

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    view = (times >= s - 20) & (times <= peak_time + 15)

    axes[0].plot(times[view], sig[view], linewidth=0.9, label="P1+P3 combined acc_mag")
    axes[0].axvspan(s, e, alpha=0.25, label="Original sync label")
    axes[0].axvline(mid, linestyle="--", label="Original middle")
    axes[0].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[0].set_title(f"{title} - Before shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(times[view], sig[view], linewidth=0.9, label="P1+P3 combined acc_mag")
    axes[1].axvspan(s + delta_s, e + delta_s, alpha=0.25, label="Shifted sync label")
    axes[1].axvline(mid + delta_s, linestyle="--", label="Shifted middle")
    axes[1].axvline(peak_time, linestyle="--", label="Detected peak")
    axes[1].set_title(f"{title} - After shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("Video time (seconds)")

    plt.tight_layout()
    plt.show()

    return delta_s, peak_time


oe_delta, oe_peak = process_file(
    OE_IN,
    OE_OUT,
    "Group 10 OpenEarable",
    smooth_win=10
)

xs_delta, xs_peak = process_file(
    XSENS_IN,
    XSENS_OUT,
    "Group 10 XSens cleaned",
    smooth_win=5
)

print("\n================================================")
print("GROUP 10 FINAL SHIFT SUMMARY")
print("================================================")
print(f"OpenEarable peak: {oe_peak:.3f}s")
print(f"XSens peak      : {xs_peak:.3f}s")
print(f"OpenEarable shift: {oe_delta:.3f}s")
print(f"XSens shift      : {xs_delta:.3f}s")
print(f"XSens peak - OE peak: {xs_peak - oe_peak:.3f}s")
print("================================================")


# --- CELL 133 (code cell #117) ---
# ============================================================
# GROUP 10 - PRINT FINAL SHIFT SUMMARY ONLY
# Uses late Whole_Group synchronizaiton_move
# Uses P1 + P3 only
# Uses cleaned XSens
# ============================================================

import os
import pandas as pd
import numpy as np

GROUP_DIR = "/content/drive/MyDrive/thesis/data/group_10"

OE_PATH = os.path.join(
    GROUP_DIR,
    "openearable_labeled",
    "group_10_openearable_labeled.csv"
)

XSENS_PATH = os.path.join(
    GROUP_DIR,
    "xsens_labeled",
    "group_10_xsens_labeled_cleaned.csv"
)

TIME_COL = "video_time_s"
SYNC_COL = "label_Whole_Group"
SYNC_LABEL = "synchronizaiton_move"

USERS_FOR_SYNC = [1, 3]

SMOOTH_OE = 10
SMOOTH_XSENS = 5


def acc_mag(df, user):
    return np.sqrt(
        pd.to_numeric(df[f"p{user}_acc_x"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_y"], errors="coerce")**2 +
        pd.to_numeric(df[f"p{user}_acc_z"], errors="coerce")**2
    )


def smooth(x, win):
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .mean()
        .values
    )


def combined_acc_p1_p3(df, smooth_win):
    sigs = [acc_mag(df, u) for u in USERS_FOR_SYNC]
    return smooth(np.nanmean(np.vstack(sigs), axis=0), smooth_win)


def get_label_segments(df, time_col, label_col):
    times = df[time_col].values
    labels = df[label_col].fillna("").astype(str).values

    segments = []
    active = None
    start_idx = None

    for i, lab in enumerate(labels):

        if lab != "" and active is None:
            active = lab
            start_idx = i

        elif lab != "" and active is not None and lab != active:
            segments.append((times[start_idx], times[i - 1], active))
            active = lab
            start_idx = i

        elif lab == "" and active is not None:
            segments.append((times[start_idx], times[i - 1], active))
            active = None
            start_idx = None

    if active is not None:
        segments.append((times[start_idx], times[-1], active))

    return segments


def calculate_shift(path, title, smooth_win):
    df = pd.read_csv(path, low_memory=False)

    sync_segments = [
        seg for seg in get_label_segments(df, TIME_COL, SYNC_COL)
        if seg[2] == SYNC_LABEL
    ]

    if len(sync_segments) == 0:
        raise ValueError(f"No {SYNC_LABEL} found in {title}")

    print("\n================================================")
    print(title)
    print("Available sync events:")
    print("================================================")

    for i, (s, e, lab) in enumerate(sync_segments):
        mid = (s + e) / 2
        print(f"{i}: {s:.3f}s -> {e:.3f}s | mid={mid:.3f}s | dur={e-s:.3f}s")

    # choose latest sync event
    s, e, lab = sorted(sync_segments, key=lambda x: x[0])[-1]
    mid = (s + e) / 2

    times = df[TIME_COL].values
    sig = combined_acc_p1_p3(df, smooth_win)

    search_start = mid - 5
    search_end = mid + 35

    mask = (times >= search_start) & (times <= search_end)

    local_idx = np.where(mask)[0]
    peak_idx = local_idx[np.nanargmax(sig[mask])]

    peak_time = times[peak_idx]
    peak_value = sig[peak_idx]
    delta = peak_time - mid

    print("\nSelected latest sync event:")
    print(f"Sync label start : {s:.3f}s")
    print(f"Sync label end   : {e:.3f}s")
    print(f"Sync label middle: {mid:.3f}s")
    print(f"Detected peak    : {peak_time:.3f}s")
    print(f"Peak value       : {peak_value:.3f}")
    print(f"Shift delta      : {delta:.3f}s")
    print("Direction        : Labels move RIGHT / later")
    print("================================================")

    return {
        "title": title,
        "sync_start": s,
        "sync_end": e,
        "sync_middle": mid,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "shift_delta": delta
    }


oe_result = calculate_shift(
    OE_PATH,
    "Group 10 OpenEarable",
    SMOOTH_OE
)

xs_result = calculate_shift(
    XSENS_PATH,
    "Group 10 XSens cleaned",
    SMOOTH_XSENS
)

print("\n\n================================================")
print("GROUP 10 FINAL SHIFT SUMMARY")
print("================================================")
print(f"OpenEarable peak       : {oe_result['peak_time']:.3f}s")
print(f"XSens cleaned peak     : {xs_result['peak_time']:.3f}s")
print(f"OpenEarable shift      : {oe_result['shift_delta']:.3f}s")
print(f"XSens cleaned shift    : {xs_result['shift_delta']:.3f}s")
print(f"XSens peak - OE peak   : {xs_result['peak_time'] - oe_result['peak_time']:.3f}s")
print("================================================")

