# --- CELL 0 (code cell #1) ---
# ============================================================
# OPTITRACK GROUP 1 - INSPECT MARKER TRACKS
# Reads Motive CSV with multi-row headers.
# Produces:
# - marker track summary
# - active marker count plot
# - X-Z and X-Y trajectory plots
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
}

OUT_DIR = "/mnt/data/optitrack_group1_inspection"
os.makedirs(OUT_DIR, exist_ok=True)


def read_motive_unlabeled_csv(path):
    """
    Reads Motive CSV exported with multiple header rows.
    Returns:
      meta dict
      long dataframe: time_s, frame, marker_name, marker_id, x, y, z
      marker summary
      active marker count per frame
    """

    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for i in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    long_parts = []
    active_count = np.zeros(len(raw), dtype=int)

    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values,
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        long_parts.append(sub)

        idx = np.where(present.values)[0]
        diffs = np.diff(idx)
        segments = int((diffs > 1).sum() + 1) if len(idx) > 1 else 1

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "segments": segments,
            "mean_x": xyz.loc[present, "x"].mean(),
            "mean_y": xyz.loc[present, "y"].mean(),
            "mean_z": xyz.loc[present, "z"].mean(),
            "std_x": xyz.loc[present, "x"].std(),
            "std_y": xyz.loc[present, "y"].std(),
            "std_z": xyz.loc[present, "z"].std(),
        })

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"{take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved ->", out)


def plot_marker_trajectories(long_df, summary, take_name, min_frames=1000):
    """
    Plots only markers with enough frames, to avoid unreadable ghost clutter.
    """
    keep = summary[summary["n_frames"] >= min_frames]["marker_name"].tolist()
    df = long_df[long_df["marker_name"].isin(keep)].copy()

    print(f"{take_name}: plotting {len(keep)} marker tracks with n_frames >= {min_frames}")

    # X-Z top-down-ish view
    plt.figure(figsize=(10, 8))
    for marker, sub in df.groupby("marker_name"):
        plt.plot(sub["x"], sub["z"], linewidth=0.8, label=marker)

    plt.title(f"{take_name} - Marker trajectories X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    out = os.path.join(OUT_DIR, f"{take_name}_trajectories_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved ->", out)

    # X-Y view
    plt.figure(figsize=(10, 8))
    for marker, sub in df.groupby("marker_name"):
        plt.plot(sub["x"], sub["y"], linewidth=0.8, label=marker)

    plt.title(f"{take_name} - Marker trajectories X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    out = os.path.join(OUT_DIR, f"{take_name}_trajectories_xy.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved ->", out)


all_summaries = {}

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name, path)
    print("="*90)

    meta, long_df, summary, active_df = read_motive_unlabeled_csv(path)

    print("Metadata:")
    for k in ["Take Name", "Capture Start Time", "Export Frame Rate", "Total Exported Frames"]:
        print(k, ":", meta.get(k))

    print("\nLong df shape:", long_df.shape)
    print("Marker summary shape:", summary.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(25))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_marker_trajectories(long_df, summary, take_name, min_frames=1000)

    all_summaries[take_name] = {
        "meta": meta,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\nInspection output folder:")
print(OUT_DIR)


# --- CELL 1 (code cell #2) ---
# ============================================================
# OPTITRACK GROUP 1 - FAST FIRST PASS 3-LANDMARK RECONSTRUCTION
# Faster version using groupby instead of filtering every frame.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_LANDMARKS = 3

MAX_ASSIGN_DIST = 0.35
MAX_REASONABLE_JUMP = 0.45
SMOOTH_WINDOW = 5


def read_motive_points_by_frame(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    long_parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        long_parts.append(sub)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, long_df


def initialize_landmarks(pts):
    if len(pts) > EXPECTED_LANDMARKS:
        selected = [0]
        while len(selected) < EXPECTED_LANDMARKS:
            remaining = [i for i in range(len(pts)) if i not in selected]
            dists = []
            for r in remaining:
                min_d = min(np.linalg.norm(pts[r] - pts[s]) for s in selected)
                dists.append(min_d)
            selected.append(remaining[int(np.argmax(dists))])
        pts = pts[selected]

    order = np.argsort(pts[:, 0])
    return pts[order]


def choose_best_three_when_too_many(pts, previous):
    if len(pts) <= EXPECTED_LANDMARKS:
        return pts

    prev_valid = np.isfinite(previous).all(axis=1)

    if prev_valid.sum() == 0:
        return initialize_landmarks(pts)

    prev = previous[prev_valid]

    nearest_dist = np.array([
        np.min(np.linalg.norm(prev - p, axis=1))
        for p in pts
    ])

    selected = np.argsort(nearest_dist)[:EXPECTED_LANDMARKS]
    return pts[selected]


def reconstruct_three_landmarks_fast(long_df):
    output_rows = []
    previous = np.full((EXPECTED_LANDMARKS, 3), np.nan)
    initialized = False

    grouped = long_df.groupby("frame", sort=True)

    total_frames = grouped.ngroups

    for k, (frame, sub) in enumerate(grouped):
        if k % 25000 == 0:
            print(f"Processing frame group {k}/{total_frames}")

        time_s = sub["time_s"].iloc[0]
        pts = sub[["x", "y", "z"]].values

        assigned = np.full((EXPECTED_LANDMARKS, 3), np.nan)

        if len(pts) == 0:
            pass

        elif not initialized:
            if len(pts) >= EXPECTED_LANDMARKS:
                assigned = initialize_landmarks(pts)
                previous = assigned.copy()
                initialized = True

        else:
            pts = choose_best_three_when_too_many(pts, previous)

            prev_valid = np.isfinite(previous).all(axis=1)

            if prev_valid.sum() == 0:
                if len(pts) >= EXPECTED_LANDMARKS:
                    assigned = initialize_landmarks(pts)
                else:
                    assigned[:len(pts)] = pts

            else:
                prev_idx = np.where(prev_valid)[0]
                prev_pts = previous[prev_valid]

                cost = np.zeros((len(prev_pts), len(pts)))

                for i, p_prev in enumerate(prev_pts):
                    for j, p_now in enumerate(pts):
                        cost[i, j] = np.linalg.norm(p_now - p_prev)

                row_ind, col_ind = linear_sum_assignment(cost)

                used_now = set()

                for r, c in zip(row_ind, col_ind):
                    landmark_id = prev_idx[r]
                    dist = cost[r, c]

                    if dist <= MAX_ASSIGN_DIST:
                        assigned[landmark_id] = pts[c]
                        used_now.add(c)

                empty = [
                    i for i in range(EXPECTED_LANDMARKS)
                    if not np.isfinite(assigned[i]).all()
                ]

                unused = [
                    j for j in range(len(pts))
                    if j not in used_now
                ]

                if len(empty) > 0 and len(unused) > 0:
                    unused_pts = pts[unused]
                    unused_order = np.argsort(unused_pts[:, 0])

                    for landmark_id, uidx in zip(empty, unused_order):
                        assigned[landmark_id] = unused_pts[uidx]

            for i in range(EXPECTED_LANDMARKS):
                if np.isfinite(assigned[i]).all() and np.isfinite(previous[i]).all():
                    jump = np.linalg.norm(assigned[i] - previous[i])
                    if jump > MAX_REASONABLE_JUMP:
                        assigned[i] = np.nan

            for i in range(EXPECTED_LANDMARKS):
                if np.isfinite(assigned[i]).all():
                    previous[i] = assigned[i]

        row = {
            "frame": frame,
            "time_s": time_s,
        }

        for i in range(EXPECTED_LANDMARKS):
            row[f"landmark{i+1}_x"] = assigned[i, 0]
            row[f"landmark{i+1}_y"] = assigned[i, 1]
            row[f"landmark{i+1}_z"] = assigned[i, 2]

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def add_diagnostics(clean):
    for i in range(1, EXPECTED_LANDMARKS + 1):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        [f"landmark{i}_available" for i in range(1, EXPECTED_LANDMARKS + 1)]
    ].sum(axis=1)

    return clean


def smooth_landmarks(clean):
    out = clean.copy()

    for i in range(1, EXPECTED_LANDMARKS + 1):
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"

            out[col] = (
                out[col]
                .interpolate(limit=10, limit_direction="both")
                .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
                .mean()
            )

    return out


def plot_clean_result(clean, take_name):
    plt.figure(figsize=(10, 8))

    for i in range(1, EXPECTED_LANDMARKS + 1):
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"{take_name} cleaned landmark trajectories X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"{take_name} cleaned active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("active clean landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name, path)
    print("="*90)

    meta, long_df = read_motive_points_by_frame(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Long detections:", long_df.shape)

    clean = reconstruct_three_landmarks_fast(long_df)
    clean = add_diagnostics(clean)

    clean_smooth = smooth_landmarks(clean)
    clean_smooth = add_diagnostics(clean_smooth)

    out_raw = os.path.join(OUT_DIR, f"group_1_optitrack_{take_name}_clean_raw.csv")
    out_smooth = os.path.join(OUT_DIR, f"group_1_optitrack_{take_name}_clean_smoothed.csv")

    clean.to_csv(out_raw, index=False)
    clean_smooth.to_csv(out_smooth, index=False)

    print("\nSaved raw clean:")
    print(out_raw)

    print("Saved smoothed clean:")
    print(out_smooth)

    print("\nAvailability before smoothing:")
    for i in range(1, EXPECTED_LANDMARKS + 1):
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    plot_clean_result(clean_smooth, take_name)


# --- CELL 2 (code cell #3) ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned"

FILES = {
    "take_1": os.path.join(OUT_DIR, "group_1_optitrack_take_1_clean_raw.csv"),
    "take_2": os.path.join(OUT_DIR, "group_1_optitrack_take_2_clean_raw.csv"),
}

LANDMARKS = [1, 2, 3]
AXES = ["x", "y", "z"]


def plot_landmarks_over_time(path, take_name):
    df = pd.read_csv(path)

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(AXES):
        ax = axs[ax_i]

        for lm in LANDMARKS:
            col = f"landmark{lm}_{axis}"
            ax.plot(
                df["time_s"],
                df[col],
                linewidth=0.8,
                label=f"Landmark {lm}"
            )

        ax.set_title(f"{take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


for take_name, path in FILES.items():
    print("\n" + "="*80)
    print(take_name)
    print(path)
    print("="*80)

    plot_landmarks_over_time(path, take_name)


# --- CELL 3 (code cell #4) ---
# ============================================================
# OPTITRACK GROUP 1 - CONSERVATIVE 3-LANDMARK RECONSTRUCTION
# Goal:
# - fewer identity swaps
# - do NOT force every frame to have 3 landmarks
# - if uncertain, leave NaN
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_conservative"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_LANDMARKS = 3

# More conservative than before
MAX_ASSIGN_DIST = 0.18          # smaller = fewer wrong switches
MAX_REASONABLE_JUMP = 0.25
MAX_MISSING_FRAMES_MEMORY = 240 # 1 second at 240 Hz
SMOOTH_WINDOW = 5


def read_motive_points_by_frame(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    long_parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        long_parts.append(sub)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, long_df


def initialize_landmarks_conservative(pts):
    """
    First stable initialization:
    sort 3 selected landmarks by X.
    If more than 3 points, choose 3 with most spatial spread.
    """
    if len(pts) > EXPECTED_LANDMARKS:
        selected = [0]
        while len(selected) < EXPECTED_LANDMARKS:
            remaining = [i for i in range(len(pts)) if i not in selected]
            dists = []
            for r in remaining:
                min_d = min(np.linalg.norm(pts[r] - pts[s]) for s in selected)
                dists.append(min_d)
            selected.append(remaining[int(np.argmax(dists))])
        pts = pts[selected]

    order = np.argsort(pts[:, 0])
    return pts[order]


def reconstruct_conservative(long_df):
    grouped = long_df.groupby("frame", sort=True)

    previous = np.full((EXPECTED_LANDMARKS, 3), np.nan)
    missing_age = np.full(EXPECTED_LANDMARKS, 10**9, dtype=int)

    initialized = False
    rows = []

    total = grouped.ngroups

    for k, (frame, sub) in enumerate(grouped):
        if k % 25000 == 0:
            print(f"Processing frame group {k}/{total}")

        time_s = sub["time_s"].iloc[0]
        pts = sub[["x", "y", "z"]].values

        assigned = np.full((EXPECTED_LANDMARKS, 3), np.nan)

        # Initialize only when exactly or at least 3 points are visible
        if not initialized:
            if len(pts) >= 3:
                assigned = initialize_landmarks_conservative(pts)
                previous = assigned.copy()
                missing_age[:] = 0
                initialized = True

        else:
            valid_prev = np.isfinite(previous).all(axis=1) & (missing_age <= MAX_MISSING_FRAMES_MEMORY)

            if valid_prev.sum() > 0 and len(pts) > 0:
                prev_idx = np.where(valid_prev)[0]
                prev_pts = previous[prev_idx]

                cost = np.zeros((len(prev_pts), len(pts)))

                for i, p_prev in enumerate(prev_pts):
                    for j, p_now in enumerate(pts):
                        cost[i, j] = np.linalg.norm(p_now - p_prev)

                row_ind, col_ind = linear_sum_assignment(cost)

                for r, c in zip(row_ind, col_ind):
                    landmark_id = prev_idx[r]
                    dist = cost[r, c]

                    if dist <= MAX_ASSIGN_DIST:
                        assigned[landmark_id] = pts[c]

            # Important:
            # No forced filling of empty landmarks.
            # If we cannot confidently assign it, leave it NaN.

            for i in range(EXPECTED_LANDMARKS):
                if np.isfinite(assigned[i]).all() and np.isfinite(previous[i]).all():
                    jump = np.linalg.norm(assigned[i] - previous[i])
                    if jump > MAX_REASONABLE_JUMP:
                        assigned[i] = np.nan

            for i in range(EXPECTED_LANDMARKS):
                if np.isfinite(assigned[i]).all():
                    previous[i] = assigned[i]
                    missing_age[i] = 0
                else:
                    missing_age[i] += 1

            # If all landmarks are lost for a while, allow reinitialization.
            if (missing_age > MAX_MISSING_FRAMES_MEMORY).all() and len(pts) >= 3:
                assigned = initialize_landmarks_conservative(pts)
                previous = assigned.copy()
                missing_age[:] = 0

        row = {
            "frame": frame,
            "time_s": time_s,
        }

        for i in range(EXPECTED_LANDMARKS):
            row[f"landmark{i+1}_x"] = assigned[i, 0]
            row[f"landmark{i+1}_y"] = assigned[i, 1]
            row[f"landmark{i+1}_z"] = assigned[i, 2]

        rows.append(row)

    clean = pd.DataFrame(rows)

    return clean


def add_diagnostics(df):
    out = df.copy()

    for i in range(1, 4):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        [f"landmark{i}_available" for i in range(1, 4)]
    ].sum(axis=1)

    return out


def smooth_short_gaps(df):
    """
    Only interpolate short gaps, not long gaps.
    """
    out = df.copy()

    for i in range(1, 4):
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"

            out[col] = (
                out[col]
                .interpolate(limit=10, limit_direction="both")
                .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
                .mean()
            )

    return out


def plot_xyz_over_time(df, take_name):
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for lm in [1, 2, 3]:
            ax.plot(
                df["time_s"],
                df[f"landmark{lm}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {lm}"
            )

        ax.set_title(f"{take_name} conservative - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_xz(df, take_name):
    plt.figure(figsize=(10, 8))

    for lm in [1, 2, 3]:
        plt.plot(
            df[f"landmark{lm}_x"],
            df[f"landmark{lm}_z"],
            linewidth=0.8,
            label=f"Landmark {lm}"
        )

    plt.title(f"{take_name} conservative cleaned landmark trajectories X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_active(df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(df["time_s"], df["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"{take_name} conservative active clean landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print(path)
    print("="*90)

    meta, long_df = read_motive_points_by_frame(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Long detections:", long_df.shape)

    clean = reconstruct_conservative(long_df)
    clean = add_diagnostics(clean)

    smooth = smooth_short_gaps(clean)
    smooth = add_diagnostics(smooth)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_conservative_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_conservative_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability:")
    for lm in [1, 2, 3]:
        print(f"Landmark {lm}: {clean[f'landmark{lm}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    plot_xz(smooth, take_name)
    plot_active(clean, take_name)
    plot_xyz_over_time(clean, take_name)


# --- CELL 4 (code cell #5) ---
# ============================================================
# OPTITRACK GROUP 1 - BALANCED 3-LANDMARK TRACKER
# Uses:
# - nearest-neighbor + velocity prediction
# - no random forced filling
# - moderate re-acquisition
# Goal:
# - less identity switching than first-pass
# - much higher availability than conservative version
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_balanced"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_LANDMARKS = 3

# Balanced parameters
MAX_ASSIGN_DIST = 0.32          # main continuation distance
MAX_REACQUIRE_DIST = 0.55       # after short gaps
MAX_REASONABLE_JUMP = 0.65
MAX_MISSING_FRAMES_MEMORY = 240 * 5  # remember for 5 seconds at 240 Hz
SMOOTH_WINDOW = 5


def read_motive_points_by_frame(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    long_parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        long_parts.append(sub)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, long_df


def initialize_landmarks(pts):
    """
    Choose 3 spread-out points, then sort by X for deterministic identity.
    """
    if len(pts) > EXPECTED_LANDMARKS:
        selected = [0]
        while len(selected) < EXPECTED_LANDMARKS:
            remaining = [i for i in range(len(pts)) if i not in selected]
            dists = []
            for r in remaining:
                min_d = min(np.linalg.norm(pts[r] - pts[s]) for s in selected)
                dists.append(min_d)
            selected.append(remaining[int(np.argmax(dists))])
        pts = pts[selected]

    order = np.argsort(pts[:, 0])
    return pts[order]


def balanced_reconstruct(long_df):
    grouped = long_df.groupby("frame", sort=True)

    previous = np.full((EXPECTED_LANDMARKS, 3), np.nan)
    velocity = np.zeros((EXPECTED_LANDMARKS, 3))
    missing_age = np.full(EXPECTED_LANDMARKS, 10**9, dtype=int)

    initialized = False
    rows = []

    total = grouped.ngroups

    for k, (frame, sub) in enumerate(grouped):
        if k % 25000 == 0:
            print(f"Processing frame group {k}/{total}")

        time_s = sub["time_s"].iloc[0]
        pts = sub[["x", "y", "z"]].values

        assigned = np.full((EXPECTED_LANDMARKS, 3), np.nan)

        if not initialized:
            if len(pts) >= 3:
                assigned = initialize_landmarks(pts)
                previous = assigned.copy()
                velocity[:] = 0
                missing_age[:] = 0
                initialized = True

        else:
            valid_prev = np.isfinite(previous).all(axis=1) & (missing_age <= MAX_MISSING_FRAMES_MEMORY)

            if valid_prev.sum() > 0 and len(pts) > 0:
                prev_idx = np.where(valid_prev)[0]

                # Predict position with simple constant velocity.
                predicted = previous.copy()
                for i in prev_idx:
                    predicted[i] = previous[i] + velocity[i]

                pred_pts = predicted[prev_idx]

                cost = np.zeros((len(pred_pts), len(pts)))

                for i, p_pred in enumerate(pred_pts):
                    for j, p_now in enumerate(pts):
                        cost[i, j] = np.linalg.norm(p_now - p_pred)

                row_ind, col_ind = linear_sum_assignment(cost)

                used_now = set()

                for r, c in zip(row_ind, col_ind):
                    landmark_id = prev_idx[r]
                    dist = cost[r, c]

                    # Allow larger distance if it was missing recently
                    threshold = MAX_ASSIGN_DIST
                    if missing_age[landmark_id] > 0:
                        threshold = MAX_REACQUIRE_DIST

                    if dist <= threshold:
                        candidate = pts[c]

                        # Check real jump from previous observed point
                        if np.isfinite(previous[landmark_id]).all():
                            jump = np.linalg.norm(candidate - previous[landmark_id])
                            if jump <= MAX_REASONABLE_JUMP:
                                assigned[landmark_id] = candidate
                                used_now.add(c)

                # If all three are missing and at least 3 detections exist,
                # reinitialize whole tracker.
                if np.isnan(assigned).all() and len(pts) >= 3:
                    assigned = initialize_landmarks(pts)

            # Update state
            for i in range(EXPECTED_LANDMARKS):
                if np.isfinite(assigned[i]).all():
                    if np.isfinite(previous[i]).all():
                        velocity[i] = assigned[i] - previous[i]
                    else:
                        velocity[i] = 0

                    previous[i] = assigned[i]
                    missing_age[i] = 0
                else:
                    missing_age[i] += 1

        row = {
            "frame": frame,
            "time_s": time_s,
        }

        for i in range(EXPECTED_LANDMARKS):
            row[f"landmark{i+1}_x"] = assigned[i, 0]
            row[f"landmark{i+1}_y"] = assigned[i, 1]
            row[f"landmark{i+1}_z"] = assigned[i, 2]

        rows.append(row)

    return pd.DataFrame(rows)


def add_diagnostics(df):
    out = df.copy()

    for i in range(1, 4):
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        [f"landmark{i}_available" for i in range(1, 4)]
    ].sum(axis=1)

    return out


def smooth_short_gaps(df):
    out = df.copy()

    for i in range(1, 4):
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"

            out[col] = (
                out[col]
                .interpolate(limit=10, limit_direction="both")
                .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
                .mean()
            )

    return out


def plot_xz(df, take_name):
    plt.figure(figsize=(10, 8))

    for lm in [1, 2, 3]:
        plt.plot(
            df[f"landmark{lm}_x"],
            df[f"landmark{lm}_z"],
            linewidth=0.8,
            label=f"Landmark {lm}"
        )

    plt.title(f"{take_name} balanced cleaned landmark trajectories X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_active(df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(df["time_s"], df["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"{take_name} balanced active clean landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_xyz_over_time(df, take_name):
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for lm in [1, 2, 3]:
            ax.plot(
                df["time_s"],
                df[f"landmark{lm}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {lm}"
            )

        ax.set_title(f"{take_name} balanced - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print(path)
    print("="*90)

    meta, long_df = read_motive_points_by_frame(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Long detections:", long_df.shape)

    clean = balanced_reconstruct(long_df)
    clean = add_diagnostics(clean)

    smooth = smooth_short_gaps(clean)
    smooth = add_diagnostics(smooth)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_balanced_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_balanced_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability:")
    for lm in [1, 2, 3]:
        print(f"Landmark {lm}: {clean[f'landmark{lm}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    plot_xz(smooth, take_name)
    plot_active(clean, take_name)
    plot_xyz_over_time(clean, take_name)


# --- CELL 5 (code cell #6) ---
# ============================================================
# OPTITRACK GROUP 1 - TRACKLET SUCCESSOR INSPECTION
# Goal:
# Identify which Unlabeled marker IDs should be stitched together.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_tracklet_stitching"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_FRAMES = 500       # ignore tiny ghost fragments first
MAX_GAP_S = 120        # candidate successor must start within 120s
MAX_DIST = 1.2         # candidate successor must start reasonably close


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()

    return meta, long_df


def make_tracklet_summary(long_df):
    rows = []

    for marker, sub in long_df.groupby("marker_name"):
        sub = sub.sort_values("time_s")

        if len(sub) < MIN_FRAMES:
            continue

        start = sub.iloc[0]
        end = sub.iloc[-1]

        rows.append({
            "marker_name": marker,
            "n_frames": len(sub),
            "start_time": float(start["time_s"]),
            "end_time": float(end["time_s"]),
            "duration_s": float(end["time_s"] - start["time_s"]),

            "start_x": float(start["x"]),
            "start_y": float(start["y"]),
            "start_z": float(start["z"]),

            "end_x": float(end["x"]),
            "end_y": float(end["y"]),
            "end_z": float(end["z"]),

            "mean_x": float(sub["x"].mean()),
            "mean_y": float(sub["y"].mean()),
            "mean_z": float(sub["z"].mean()),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("start_time").reset_index(drop=True)

    return summary


def find_successors(summary):
    candidates = []

    for _, a in summary.iterrows():
        end_pos = np.array([a["end_x"], a["end_y"], a["end_z"]])

        possible = summary[
            summary["start_time"] > a["end_time"]
        ].copy()

        possible["gap_s"] = possible["start_time"] - a["end_time"]

        possible = possible[
            possible["gap_s"] <= MAX_GAP_S
        ].copy()

        if len(possible) == 0:
            continue

        start_pos = possible[["start_x", "start_y", "start_z"]].values
        possible["dist"] = np.linalg.norm(start_pos - end_pos, axis=1)

        possible = possible[
            possible["dist"] <= MAX_DIST
        ].copy()

        possible["score"] = possible["dist"] + 0.003 * possible["gap_s"]

        possible = possible.sort_values("score").head(5)

        for _, b in possible.iterrows():
            candidates.append({
                "from_marker": a["marker_name"],
                "to_marker": b["marker_name"],
                "from_end": a["end_time"],
                "to_start": b["start_time"],
                "gap_s": b["gap_s"],
                "dist": b["dist"],
                "score": b["score"],
                "from_n_frames": a["n_frames"],
                "to_n_frames": b["n_frames"],
            })

    return pd.DataFrame(candidates).sort_values(
        ["from_marker", "score"]
    ).reset_index(drop=True)


def plot_tracklet_timeline(summary, take_name):
    summary = summary.sort_values("start_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in summary.iterrows():
        plt.hlines(
            y=i,
            xmin=row["start_time"],
            xmax=row["end_time"],
            linewidth=4
        )
        plt.text(
            row["start_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"{take_name} - marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_tracklets_xz(summary, long_df, take_name, top_n=30):
    keep = summary.sort_values("n_frames", ascending=False).head(top_n)["marker_name"]

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")
        plt.plot(sub["x"], sub["z"], linewidth=1.0, label=marker)

        # mark start/end
        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=20)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=20, marker="x")

    plt.title(f"{take_name} - top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z ->", out)


all_outputs = {}

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print(path)
    print("="*90)

    meta, long_df = read_motive_long(path)

    print("Take name:", meta.get("Take Name"))
    print("Long detections:", long_df.shape)

    summary = make_tracklet_summary(long_df)
    successors = find_successors(summary)

    summary_path = os.path.join(OUT_DIR, f"{take_name}_tracklet_summary.csv")
    successors_path = os.path.join(OUT_DIR, f"{take_name}_successor_candidates.csv")

    summary.to_csv(summary_path, index=False)
    successors.to_csv(successors_path, index=False)

    print("\nTracklet summary:")
    display(summary.sort_values("n_frames", ascending=False).head(30))

    print("\nBest successor candidates:")
    display(successors.head(80))

    print("\nSaved:")
    print(summary_path)
    print(successors_path)

    plot_tracklet_timeline(summary, take_name)
    plot_tracklets_xz(summary, long_df, take_name, top_n=35)

    all_outputs[take_name] = {
        "long_df": long_df,
        "summary": summary,
        "successors": successors
    }

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 6 (code cell #7) ---
# ============================================================
# OPTITRACK GROUP 1 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1669", "Unlabeled 1682", "Unlabeled 1726"],
        "landmark2": ["Unlabeled 1670", "Unlabeled 1691", "Unlabeled 1710", "Unlabeled 1718"],
        "landmark3": ["Unlabeled 1668", "Unlabeled 1674", "Unlabeled 1717", "Unlabeled 1732", "Unlabeled 1733"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1820"],
        "landmark2": ["Unlabeled 1823"],
        "landmark3": ["Unlabeled 1835"],
    }
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        # If more than one selected marker appears in same frame, average them.
        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(
            lm_by_frame,
            on="frame",
            how="left"
        )

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z trajectory
    plt.figure(figsize=(10, 8))

    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"{take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # availability
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"{take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{take_name} manually stitched - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        if col in clean.columns:
            print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 7 (code cell #8) ---
# ============================================================
# OPTITRACK GROUP 1 - FIND CANDIDATE FRAGMENTS FOR MISSING GAPS
# Goal:
# Identify unused marker fragments that may fill missing periods
# in manually stitched Landmark1/2/3.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

CLEAN_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_manual_stitched"

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_gap_candidate_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1669", "Unlabeled 1682", "Unlabeled 1726"],
        "landmark2": ["Unlabeled 1670", "Unlabeled 1691", "Unlabeled 1710", "Unlabeled 1718"],
        "landmark3": ["Unlabeled 1668", "Unlabeled 1674", "Unlabeled 1717", "Unlabeled 1732", "Unlabeled 1733"],
    },
    "take_2": {
        "landmark1": ["Unlabeled 1820"],
        "landmark2": ["Unlabeled 1823"],
        "landmark3": ["Unlabeled 1835"],
    }
}

MIN_GAP_S = 2.0
MIN_CANDIDATE_FRAMES = 200
TOP_N = 10


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]
    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]
        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return long_df


def get_missing_gaps(clean, lm):
    avail_col = f"landmark{lm}_available"
    time = clean["time_s"].values
    available = clean[avail_col].fillna(False).astype(bool).values

    gaps = []
    in_gap = False
    start_idx = None

    for i, ok in enumerate(available):
        if not ok and not in_gap:
            in_gap = True
            start_idx = i

        elif ok and in_gap:
            end_idx = i - 1
            s = time[start_idx]
            e = time[end_idx]
            if e - s >= MIN_GAP_S:
                gaps.append({
                    "lm": lm,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_s": e - s
                })
            in_gap = False

    if in_gap:
        end_idx = len(available) - 1
        s = time[start_idx]
        e = time[end_idx]
        if e - s >= MIN_GAP_S:
            gaps.append({
                "lm": lm,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": s,
                "end_time": e,
                "duration_s": e - s
            })

    return gaps


def nearest_valid_position(clean, lm, idx, direction):
    cols = [f"landmark{lm}_x", f"landmark{lm}_y", f"landmark{lm}_z"]

    if direction == "before":
        search = range(idx, -1, -1)
    else:
        search = range(idx, len(clean))

    for j in search:
        vals = clean.loc[j, cols].values.astype(float)
        if np.isfinite(vals).all():
            return vals

    return None


def find_candidates_for_gap(long_df, clean, gap, used_markers):
    s = gap["start_time"]
    e = gap["end_time"]
    lm = gap["lm"]

    before_pos = nearest_valid_position(clean, lm, gap["start_idx"], "before")
    after_pos = nearest_valid_position(clean, lm, gap["end_idx"], "after")

    gap_df = long_df[
        (long_df["time_s"] >= s) &
        (long_df["time_s"] <= e) &
        (~long_df["marker_name"].isin(used_markers))
    ].copy()

    rows = []

    for marker, sub in gap_df.groupby("marker_name"):
        if len(sub) < MIN_CANDIDATE_FRAMES:
            continue

        mean_pos = sub[["x", "y", "z"]].mean().values
        start_pos = sub.sort_values("time_s")[["x", "y", "z"]].iloc[0].values
        end_pos = sub.sort_values("time_s")[["x", "y", "z"]].iloc[-1].values

        dist_before = np.nan
        dist_after = np.nan

        if before_pos is not None:
            dist_before = np.linalg.norm(start_pos - before_pos)

        if after_pos is not None:
            dist_after = np.linalg.norm(end_pos - after_pos)

        score_parts = []
        if np.isfinite(dist_before):
            score_parts.append(dist_before)
        if np.isfinite(dist_after):
            score_parts.append(dist_after)

        continuity_score = np.mean(score_parts) if score_parts else np.nan

        rows.append({
            "landmark": lm,
            "gap_start": s,
            "gap_end": e,
            "gap_duration_s": e - s,
            "candidate_marker": marker,
            "candidate_frames_in_gap": len(sub),
            "candidate_start_time": sub["time_s"].min(),
            "candidate_end_time": sub["time_s"].max(),
            "dist_to_before": dist_before,
            "dist_to_after": dist_after,
            "continuity_score": continuity_score,
            "mean_x": mean_pos[0],
            "mean_y": mean_pos[1],
            "mean_z": mean_pos[2],
        })

    if len(rows) == 0:
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    cand = cand.sort_values(
        ["candidate_frames_in_gap", "continuity_score"],
        ascending=[False, True]
    )

    return cand


all_suggestions = []

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print("="*90)

    clean_path = os.path.join(
        CLEAN_DIR,
        f"group_1_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    clean = pd.read_csv(clean_path)
    long_df = read_motive_long(path)

    used_markers = []
    for lm_name, markers in CURRENT_CHAINS[take_name].items():
        used_markers.extend(markers)

    print("Currently used markers:", used_markers)

    for lm in [1, 2, 3]:
        gaps = get_missing_gaps(clean, lm)

        print(f"\nLandmark {lm} has {len(gaps)} missing gaps >= {MIN_GAP_S}s")

        # Show largest gaps first
        gaps = sorted(gaps, key=lambda g: g["duration_s"], reverse=True)

        for gap_i, gap in enumerate(gaps[:8]):
            print(
                f"  Gap {gap_i}: "
                f"{gap['start_time']:.2f}s -> {gap['end_time']:.2f}s "
                f"dur={gap['duration_s']:.2f}s"
            )

            cand = find_candidates_for_gap(long_df, clean, gap, used_markers)

            if len(cand) > 0:
                print("  Top candidate markers:")
                display(cand.head(TOP_N))

                all_suggestions.append(cand.head(TOP_N))

                # Plot candidate fragments for this gap
                plt.figure(figsize=(10, 8))

                # existing landmark path around gap
                lm_cols = [f"landmark{lm}_x", f"landmark{lm}_z"]
                window = clean[
                    (clean["time_s"] >= gap["start_time"] - 20) &
                    (clean["time_s"] <= gap["end_time"] + 20)
                ]

                plt.plot(
                    window[lm_cols[0]],
                    window[lm_cols[1]],
                    linewidth=2,
                    label=f"Current Landmark {lm}",
                    color="black"
                )

                for marker in cand.head(5)["candidate_marker"]:
                    sub = long_df[
                        (long_df["marker_name"] == marker) &
                        (long_df["time_s"] >= gap["start_time"]) &
                        (long_df["time_s"] <= gap["end_time"])
                    ]
                    plt.plot(sub["x"], sub["z"], linewidth=1.2, label=marker)

                plt.title(
                    f"{take_name} Landmark {lm} gap {gap_i}: "
                    f"{gap['start_time']:.1f}-{gap['end_time']:.1f}s"
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

if len(all_suggestions) > 0:
    suggestions = pd.concat(all_suggestions, ignore_index=True)
    suggestions_path = os.path.join(OUT_DIR, "gap_candidate_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    print("\nSaved all suggestions:")
    print(suggestions_path)

    print("\nMost promising candidates overall:")
    display(
        suggestions
        .sort_values(["landmark", "candidate_frames_in_gap"], ascending=[True, False])
        .head(80)
    )
else:
    print("No candidates found.")


# --- CELL 8 (code cell #9) ---
# ============================================================
# OPTITRACK GROUP 1 - UPDATED MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# Updated after gap-candidate inspection.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_1/optitrack/Arda_Thesis_Group-1_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_manual_stitched_UPDATED"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# UPDATED CHAINS
# ============================================================

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 1669",
            "Unlabeled 1682",
            "Unlabeled 1726",
        ],

        "landmark2": [
            "Unlabeled 1670",
            "Unlabeled 1687",
            "Unlabeled 1691",
            "Unlabeled 1704",
            "Unlabeled 1710",
            "Unlabeled 1718",
            "Unlabeled 1735",
        ],

        "landmark3": [
            "Unlabeled 1668",
            "Unlabeled 1674",
            "Unlabeled 1673",
            "Unlabeled 1678",
            "Unlabeled 1680",
            "Unlabeled 1683",
            "Unlabeled 1696",
            "Unlabeled 1703",
            "Unlabeled 1705",
            "Unlabeled 1711",
            "Unlabeled 1717",
            "Unlabeled 1732",
            "Unlabeled 1733",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 1820",
        ],

        "landmark2": [
            "Unlabeled 1823",
        ],

        "landmark3": [
            "Unlabeled 1826",
            "Unlabeled 1829",
            "Unlabeled 1835",
        ],
    }
}


# ============================================================
# FUNCTIONS
# ============================================================

def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        # If more than one selected marker appears in the same frame, average them.
        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(
            lm_by_frame,
            on="frame",
            how="left"
        )

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"

            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z trajectory
    plt.figure(figsize=(10, 8))

    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"{take_name} updated manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmarks over time
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"{take_name} updated manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{take_name} updated manually stitched - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(take_name)
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_manual_stitched_UPDATED_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_1_optitrack_{take_name}_manual_stitched_UPDATED_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        if col in clean.columns:
            print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 9 (code cell #10) ---
# ============================================================
# OPTITRACK GROUP 1 - COMBINE TAKE 1 + TAKE 2
# Uses real capture-start offset:
# Take 2 starts 1385.836s after Take 1.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_cleaned_manual_stitched_UPDATED"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_1/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE1_PATH = f"{IN_DIR}/group_1_optitrack_take_1_manual_stitched_UPDATED_smoothed.csv"
TAKE2_PATH = f"{IN_DIR}/group_1_optitrack_take_2_manual_stitched_UPDATED_smoothed.csv"

# Real offset from capture start times:
# 10:23:10.716 -> 10:46:16.552
TAKE2_OFFSET_S = 1385.836

take1 = pd.read_csv(TAKE1_PATH)
take2 = pd.read_csv(TAKE2_PATH)

take1 = take1.copy()
take2 = take2.copy()

take1["take"] = "take_1"
take2["take"] = "take_2"

take1["time_s_original"] = take1["time_s"]
take2["time_s_original"] = take2["time_s"]

take1["time_s"] = take1["time_s_original"]
take2["time_s"] = take2["time_s_original"] + TAKE2_OFFSET_S

combined = pd.concat([take1, take2], ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Save
out_path = f"{OUT_DIR}/group_1_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 1 OPTITRACK COMBINED")
print("="*80)

print("Take 1 rows:", len(take1))
print("Take 2 rows:", len(take2))
print("Combined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries:")
print("Take 1 end:", take1["time_s"].max())
print("Take 2 start after offset:", take2["time_s"].min())
print("Gap between take 1 and take 2:", take2["time_s"].min() - take1["time_s"].max())

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")
plt.axvline(take1["time_s"].max(), linestyle="--", label="Take 1 end")
plt.axvline(take2["time_s"].min(), linestyle="--", label="Take 2 start")
plt.title("Group 1 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 1 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    ax.axvline(take1["time_s"].max(), linestyle="--", alpha=0.7)
    ax.axvline(take2["time_s"].min(), linestyle="--", alpha=0.7)

    ax.set_title(f"Group 1 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 11 (code cell #11) ---
# ============================================================
# OPTITRACK GROUP 2 - START INSPECTION
# Automatically finds OptiTrack take files for Group 2.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 2

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

# Automatically find CSV files
TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group2_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group2_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print("GROUP 2 OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 12 (code cell #12) ---
# ============================================================
# OPTITRACK GROUP 2 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 2 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 2

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_2/optitrack/Arda_Group_2_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_2/optitrack/Arda_Group_2_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_2/optitrack/Arda_Group_2_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_2/optitrack/Arda_Group_2_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_2/optitrack/Arda_Group_2_Take_5.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_2/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1163"],
        "landmark2": ["Unlabeled 1164", "Unlabeled 1178"],
        "landmark3": [
            "Unlabeled 1165",
            "Unlabeled 1168",
            "Unlabeled 1169",
            "Unlabeled 1170",
            "Unlabeled 1181",
            "Unlabeled 1182",
        ],
    },

    "take_2": {
        "landmark1": ["Unlabeled 1259"],
        "landmark2": ["Unlabeled 1257"],
        "landmark3": [
            "Unlabeled 1256",
            "Unlabeled 1260",
            "Unlabeled 1265",
            "Unlabeled 1268",
        ],
    },

    "take_3": {
        "landmark1": ["Unlabeled 1307"],
        "landmark2": [
            "Unlabeled 1308",
            "Unlabeled 1312",
            "Unlabeled 1314",
        ],
        "landmark3": [
            "Unlabeled 1309",
            "Unlabeled 1310",
            "Unlabeled 1311",
            "Unlabeled 1313",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 1353",
            "Unlabeled 1356",
            "Unlabeled 1361",
            "Unlabeled 1363",
        ],
        "landmark2": [
            "Unlabeled 1354",
            "Unlabeled 1357",
            "Unlabeled 1362",
        ],
        "landmark3": [
            "Unlabeled 1352",
            "Unlabeled 1358",
        ],
    },

    "take_5": {
        "landmark1": ["Unlabeled 1386"],
        "landmark2": ["Unlabeled 1387"],
        "landmark3": [
            "Unlabeled 1388",
            "Unlabeled 1392",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 13 (code cell #13) ---
# ============================================================
# OPTITRACK GROUP 2 - COMBINE TAKE 1–5
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 2 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 2

IN_DIR = "/content/drive/MyDrive/thesis/data/group_2/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_2/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_2_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_2_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_2_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_2_optitrack_take_4_manual_stitched_smoothed.csv",
    "take_5": f"{IN_DIR}/group_2_optitrack_take_5_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-21 14:18:05.206
# Take 2: 2026-04-21 14:28:08.543
# Take 3: 2026-04-21 14:38:16.803
# Take 4: 2026-04-21 14:48:22.744
# Take 5: 2026-04-21 14:58:28.830
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 603.337,
    "take_3": 1211.597,
    "take_4": 1817.538,
    "take_5": 2423.624,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

out_path = f"{OUT_DIR}/group_2_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 2 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 2 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 2 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 2 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 14 (code cell #14) ---
# ============================================================
# OPTITRACK GROUP 3 - START INSPECTION
# Automatically finds OptiTrack take files for Group 3.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 3

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 15 (code cell #15) ---
# ============================================================
# OPTITRACK GROUP 3 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 3 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 3

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_3/optitrack/Arda_Group_3_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_3/optitrack/Arda_Group_3_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_3/optitrack/Arda_Group_3_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_3/optitrack/Arda_Group_3_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_3/optitrack/Arda_Group_3_Take_5.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_3/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1202"],
        "landmark2": ["Unlabeled 1201", "Unlabeled 1227"],
        "landmark3": [
            "Unlabeled 1203",
            "Unlabeled 1225",
            "Unlabeled 1228",
            "Unlabeled 1230",
        ],
    },

    "take_2": {
        "landmark1": ["Unlabeled 1287"],
        "landmark2": ["Unlabeled 1288"],
        "landmark3": ["Unlabeled 1286", "Unlabeled 1308"],
    },

    "take_3": {
        "landmark1": ["Unlabeled 1352"],
        "landmark2": [
            "Unlabeled 1351",
            "Unlabeled 1360",
            "Unlabeled 1363",
            "Unlabeled 1366",
            "Unlabeled 1367",
            "Unlabeled 1369",
            "Unlabeled 1370",
        ],
        "landmark3": [
            "Unlabeled 1353",
            "Unlabeled 1357",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 1460",
            "Unlabeled 1493",
        ],
        "landmark2": [
            "Unlabeled 1462",
            "Unlabeled 1478",
            "Unlabeled 1487",
            "Unlabeled 1492",
            "Unlabeled 1499",
        ],
        "landmark3": [
            "Unlabeled 1461",
            "Unlabeled 1466",
            "Unlabeled 1468",
            "Unlabeled 1476",
            "Unlabeled 1485",
            "Unlabeled 1486",
        ],
    },

    "take_5": {
        "landmark1": ["Unlabeled 1520"],
        "landmark2": ["Unlabeled 1521"],
        "landmark3": ["Unlabeled 1522"],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmarks
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 16 (code cell #16) ---
# ============================================================
# OPTITRACK GROUP 3 - COMBINE TAKE 1–5
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 3 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 3

IN_DIR = "/content/drive/MyDrive/thesis/data/group_3/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_3/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_3_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_3_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_3_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_3_optitrack_take_4_manual_stitched_smoothed.csv",
    "take_5": f"{IN_DIR}/group_3_optitrack_take_5_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-22 14:33:14.209
# Take 2: 2026-04-22 14:43:30.161
# Take 3: 2026-04-22 14:53:39.149
# Take 4: 2026-04-22 15:03:56.360
# Take 5: 2026-04-22 15:16:01.878
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 615.952,
    "take_3": 1224.940,
    "take_4": 1842.151,
    "take_5": 2567.669,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

out_path = f"{OUT_DIR}/group_3_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 3 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 3 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 3 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 3 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 3 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 18 (code cell #17) ---
# ============================================================
# OPTITRACK GROUP 5 - START INSPECTION
# Automatically finds OptiTrack take files for Group 5.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 5

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 19 (code cell #18) ---
# ============================================================
# OPTITRACK GROUP 5 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 5 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 5

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_5/optitrack/Arda_Group-5_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_5/optitrack/Arda_Group-5_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_5/optitrack/Arda_Group-5_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_5/optitrack/Arda_Group-5_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_5/optitrack/Arda_Group-5_Take_5.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_5/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# FIRST-PASS CHAINS
# ============================================================

CHAINS = {
    "take_1": {
        # Stable/full landmark
        "landmark1": [
            "Unlabeled 1317",
        ],

        # Negative/central trajectory fragments
        "landmark2": [
            "Unlabeled 1318",
            "Unlabeled 1322",
            "Unlabeled 1339",
            "Unlabeled 1340",
            "Unlabeled 1341",
            "Unlabeled 1344",
            "Unlabeled 1345",
            "Unlabeled 1348",
            "Unlabeled 1350",
            "Unlabeled 1351",
        ],

        # Positive-Z / build-area trajectory fragments
        "landmark3": [
            "Unlabeled 1319",
            "Unlabeled 1320",
            "Unlabeled 1323",
            "Unlabeled 1327",
            "Unlabeled 1329",
            "Unlabeled 1331",
            "Unlabeled 1338",
        ],
    },

    "take_2": {
        # Two full landmarks
        "landmark1": [
            "Unlabeled 1412",
        ],

        "landmark2": [
            "Unlabeled 1413",
        ],

        # Fragmented third landmark
        "landmark3": [
            "Unlabeled 1411",
            "Unlabeled 1414",
            "Unlabeled 1415",
            "Unlabeled 1416",
            "Unlabeled 1417",
            "Unlabeled 1418",
            "Unlabeled 1419",
            "Unlabeled 1420",
            "Unlabeled 1425",
            "Unlabeled 1426",
            "Unlabeled 1427",
            "Unlabeled 1428",
            "Unlabeled 1429",
            "Unlabeled 1430",
            "Unlabeled 1431",
            "Unlabeled 1433",
            "Unlabeled 1434",
            "Unlabeled 1435",
            "Unlabeled 1438",
            "Unlabeled 1439",
        ],
    },

    "take_3": {
        # Stable/full landmark
        "landmark1": [
            "Unlabeled 1522",
        ],

        # Positive-Z landmark
        "landmark2": [
            "Unlabeled 1523",
            "Unlabeled 1551",
        ],

        # Fragmented negative/central landmark
        "landmark3": [
            "Unlabeled 1524",
            "Unlabeled 1527",
            "Unlabeled 1530",
            "Unlabeled 1533",
            "Unlabeled 1534",
            "Unlabeled 1535",
            "Unlabeled 1536",
            "Unlabeled 1537",
            "Unlabeled 1538",
            "Unlabeled 1539",
            "Unlabeled 1540",
            "Unlabeled 1541",
            "Unlabeled 1542",
            "Unlabeled 1549",
            "Unlabeled 1554",
            "Unlabeled 1556",
        ],
    },

    "take_4": {
        # This take is the most uncertain. Inspect plots carefully.
        "landmark1": [
            "Unlabeled 1644",
            "Unlabeled 1668",
            "Unlabeled 1672",
            "Unlabeled 1677",
            "Unlabeled 1678",
        ],

        "landmark2": [
            "Unlabeled 1645",
            "Unlabeled 1647",
            "Unlabeled 1654",
            "Unlabeled 1667",
        ],

        "landmark3": [
            "Unlabeled 1646",
            "Unlabeled 1657",
            "Unlabeled 1662",
            "Unlabeled 1666",
            "Unlabeled 1674",
            "Unlabeled 1675",
            "Unlabeled 1680",
        ],
    },

    "take_5": {
        # Two full landmarks
        "landmark1": [
            "Unlabeled 1751",
        ],

        "landmark2": [
            "Unlabeled 1752",
        ],

        # Third landmark appears mainly as 1754 then 1756
        "landmark3": [
            "Unlabeled 1754",
            "Unlabeled 1756",
        ],
    },
}


# ============================================================
# FUNCTIONS
# ============================================================

def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmarks
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 20 (code cell #19) ---
# ============================================================
# OPTITRACK GROUP 5 - COMBINE TAKE 1–5
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 5 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 5

IN_DIR = "/content/drive/MyDrive/thesis/data/group_5/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_5/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_5_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_5_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_5_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_5_optitrack_take_4_manual_stitched_smoothed.csv",
    "take_5": f"{IN_DIR}/group_5_optitrack_take_5_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-23 13:20:45.737
# Take 2: 2026-04-23 13:33:20.518
# Take 3: 2026-04-23 13:43:29.474
# Take 4: 2026-04-23 13:53:41.450
# Take 5: 2026-04-23 14:07:08.483
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 754.781,
    "take_3": 1363.737,
    "take_4": 1975.713,
    "take_5": 2782.746,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

out_path = f"{OUT_DIR}/group_5_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 5 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 5 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 5 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 5 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 5 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 22 (code cell #20) ---
# ============================================================
# OPTITRACK GROUP 6 - FILE INTEGRITY / TIMESTAMP AUDIT
# Checks:
# - file exists / size
# - Motive header readability
# - metadata: take name, capture start, frame rate, total exported frames
# - CSV row count vs metadata frame count
# - malformed row lengths
# - frame/time monotonicity
# - duplicate frames
# - large time gaps
# - active marker count distribution
# - first/last valid time
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np

GROUP = 6

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"
OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_integrity_check"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*90)
print(f"GROUP {GROUP} OPTITRACK FILE INTEGRITY CHECK")
print("="*90)

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

for i, p in enumerate(TAKE_FILES):
    print(f"{i}: {p}")

print("\nFiles found:", len(TAKE_FILES))


def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def inspect_motive_file(path, expected_header_rows=7):
    result = {
        "path": path,
        "file_name": os.path.basename(path),
        "file_size_mb": os.path.getsize(path) / (1024 * 1024),
        "readable_header": False,
        "readable_csv": False,
        "error": "",
    }

    # --------------------------------------------------------
    # 1. Read first 7 header rows safely
    # --------------------------------------------------------
    try:
        header_rows = []
        with open(path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            for _ in range(expected_header_rows):
                header_rows.append(next(reader))

        result["readable_header"] = True

        meta_row = header_rows[0]
        names_row = header_rows[3]
        ids_row = header_rows[4]
        axis_row = header_rows[6]

        meta = {}
        for i in range(0, len(meta_row) - 1, 2):
            if meta_row[i]:
                meta[meta_row[i]] = meta_row[i + 1]

        result["take_name"] = meta.get("Take Name", "")
        result["capture_start"] = meta.get("Capture Start Time", "")
        result["export_frame_rate"] = meta.get("Export Frame Rate", "")
        result["total_exported_frames_meta"] = meta.get("Total Exported Frames", "")

        ncols_expected = len(axis_row)
        result["ncols_expected_from_header"] = ncols_expected

        # count marker columns
        marker_names = []
        for c in range(2, ncols_expected, 3):
            if c < len(names_row) and str(names_row[c]).strip():
                marker_names.append(names_row[c])

        result["marker_column_groups"] = len(marker_names)
        result["unique_marker_names_in_header"] = len(set(marker_names))

    except Exception as e:
        result["error"] = f"Header read failed: {repr(e)}"
        return result, None, None

    # --------------------------------------------------------
    # 2. Check row lengths without loading full data first
    # --------------------------------------------------------
    malformed_rows = 0
    data_rows_seen = 0
    first_bad_examples = []

    try:
        with open(path, newline="", errors="replace") as f:
            reader = csv.reader(f)

            for row_i, row in enumerate(reader):
                if row_i < expected_header_rows:
                    continue

                data_rows_seen += 1

                if len(row) != ncols_expected:
                    malformed_rows += 1
                    if len(first_bad_examples) < 5:
                        first_bad_examples.append({
                            "line_number_1based": row_i + 1,
                            "row_length": len(row),
                            "expected_length": ncols_expected,
                            "first_values": row[:8],
                            "last_values": row[-8:] if len(row) >= 8 else row,
                        })

        result["data_rows_seen_raw_csv"] = data_rows_seen
        result["malformed_rows"] = malformed_rows
        result["malformed_row_pct"] = (
            malformed_rows / data_rows_seen * 100 if data_rows_seen else np.nan
        )
        result["first_bad_examples"] = str(first_bad_examples)

    except Exception as e:
        result["error"] = f"Row-length scan failed: {repr(e)}"
        return result, None, None

    # --------------------------------------------------------
    # 3. Load CSV data
    # --------------------------------------------------------
    try:
        raw = pd.read_csv(
            path,
            header=None,
            skiprows=expected_header_rows,
            low_memory=False,
            on_bad_lines="skip"
        )

        result["readable_csv"] = True
        result["loaded_rows"] = len(raw)
        result["loaded_cols"] = raw.shape[1]

        # Keep only expected columns if extra columns exist
        raw = raw.iloc[:, :ncols_expected].copy()

    except Exception as e:
        result["error"] = f"pandas load failed: {repr(e)}"
        return result, None, None

    # --------------------------------------------------------
    # 4. Frame/time analysis
    # --------------------------------------------------------
    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    valid_frame_time = frame.notna() & time_s.notna()

    result["valid_frame_time_rows"] = int(valid_frame_time.sum())
    result["invalid_frame_time_rows"] = int((~valid_frame_time).sum())

    frame_valid = frame[valid_frame_time].astype(int).reset_index(drop=True)
    time_valid = time_s[valid_frame_time].astype(float).reset_index(drop=True)

    if len(frame_valid) > 0:
        result["frame_start"] = int(frame_valid.iloc[0])
        result["frame_end"] = int(frame_valid.iloc[-1])
        result["time_start"] = float(time_valid.iloc[0])
        result["time_end"] = float(time_valid.iloc[-1])
        result["duration_s"] = float(time_valid.iloc[-1] - time_valid.iloc[0])
        result["duration_min"] = float((time_valid.iloc[-1] - time_valid.iloc[0]) / 60)

        frame_diff = frame_valid.diff().dropna()
        time_diff = time_valid.diff().dropna()

        result["frame_monotonic_increasing"] = bool((frame_diff > 0).all())
        result["time_monotonic_increasing"] = bool((time_diff > 0).all())

        result["duplicate_frame_count"] = int(frame_valid.duplicated().sum())

        result["frame_diff_min"] = float(frame_diff.min()) if len(frame_diff) else np.nan
        result["frame_diff_median"] = float(frame_diff.median()) if len(frame_diff) else np.nan
        result["frame_diff_max"] = float(frame_diff.max()) if len(frame_diff) else np.nan

        result["time_dt_min"] = float(time_diff.min()) if len(time_diff) else np.nan
        result["time_dt_median"] = float(time_diff.median()) if len(time_diff) else np.nan
        result["time_dt_max"] = float(time_diff.max()) if len(time_diff) else np.nan

        result["negative_or_zero_dt_count"] = int((time_diff <= 0).sum())
        result["large_gap_gt_1s_count"] = int((time_diff > 1.0).sum())
        result["large_gap_gt_5s_count"] = int((time_diff > 5.0).sum())

        # Compare metadata total exported frames
        meta_frames = result.get("total_exported_frames_meta", "")
        try:
            meta_frames_int = int(float(meta_frames))
            result["metadata_minus_loaded_rows"] = meta_frames_int - len(raw)
            result["metadata_minus_valid_time_rows"] = meta_frames_int - int(valid_frame_time.sum())
        except:
            result["metadata_minus_loaded_rows"] = np.nan
            result["metadata_minus_valid_time_rows"] = np.nan

    # --------------------------------------------------------
    # 5. Active marker count distribution
    # --------------------------------------------------------
    active_count = np.zeros(len(raw), dtype=int)

    for c in range(2, ncols_expected, 3):
        if c + 2 < raw.shape[1]:
            xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
            present = xyz.notna().all(axis=1)
            active_count += present.values.astype(int)

    active_series = pd.Series(active_count)
    active_dist = active_series.value_counts().sort_index()

    result["active_marker_count_distribution"] = dict(active_dist)
    result["active_marker_mean"] = float(active_series.mean())
    result["active_marker_max"] = int(active_series.max())

    result["rows_with_0_markers"] = int((active_series == 0).sum())
    result["rows_with_1_marker"] = int((active_series == 1).sum())
    result["rows_with_2_markers"] = int((active_series == 2).sum())
    result["rows_with_3_markers"] = int((active_series == 3).sum())
    result["rows_with_more_than_3_markers"] = int((active_series > 3).sum())

    # --------------------------------------------------------
    # 6. Last few rows preview for truncation suspicion
    # --------------------------------------------------------
    result["last_5_frame_values"] = str(frame.tail(5).tolist())
    result["last_5_time_values"] = str(time_s.tail(5).tolist())

    return result, raw, active_series


results = []
raw_cache = {}

for path in TAKE_FILES:
    print("\n" + "="*90)
    print(path)
    print("="*90)

    result, raw, active_series = inspect_motive_file(path)

    results.append(result)

    print("File:", result["file_name"])
    print("Size MB:", round(result["file_size_mb"], 2))
    print("Readable header:", result["readable_header"])
    print("Readable CSV:", result["readable_csv"])
    print("Error:", result.get("error", ""))

    print("\nMetadata:")
    print("Take name:", result.get("take_name"))
    print("Capture start:", result.get("capture_start"))
    print("Frame rate:", result.get("export_frame_rate"))
    print("Total exported frames meta:", result.get("total_exported_frames_meta"))
    print("Expected columns:", result.get("ncols_expected_from_header"))
    print("Marker groups in header:", result.get("marker_column_groups"))

    print("\nRows:")
    print("Raw CSV data rows seen:", result.get("data_rows_seen_raw_csv"))
    print("Malformed rows:", result.get("malformed_rows"))
    print("Malformed row %:", result.get("malformed_row_pct"))
    print("Loaded rows:", result.get("loaded_rows"))
    print("Loaded cols:", result.get("loaded_cols"))
    print("Metadata - loaded rows:", result.get("metadata_minus_loaded_rows"))

    print("\nTime/frame:")
    print("Frame:", result.get("frame_start"), "->", result.get("frame_end"))
    print("Time :", result.get("time_start"), "->", result.get("time_end"))
    print("Duration min:", result.get("duration_min"))
    print("Frame monotonic:", result.get("frame_monotonic_increasing"))
    print("Time monotonic:", result.get("time_monotonic_increasing"))
    print("Duplicate frames:", result.get("duplicate_frame_count"))
    print("Median dt:", result.get("time_dt_median"))
    print("Max dt:", result.get("time_dt_max"))
    print("Gaps >1s:", result.get("large_gap_gt_1s_count"))
    print("Gaps >5s:", result.get("large_gap_gt_5s_count"))

    print("\nActive marker count distribution:")
    print(result.get("active_marker_count_distribution"))

    print("\nLast row check:")
    print("Last 5 frames:", result.get("last_5_frame_values"))
    print("Last 5 times :", result.get("last_5_time_values"))

    if result.get("malformed_rows", 0) > 0:
        print("\nFirst bad row examples:")
        print(result.get("first_bad_examples"))

summary_df = pd.DataFrame(results)

summary_path = os.path.join(
    OUT_DIR,
    f"group_{GROUP}_optitrack_file_integrity_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)

display_cols = [
    "file_name",
    "file_size_mb",
    "readable_header",
    "readable_csv",
    "take_name",
    "capture_start",
    "export_frame_rate",
    "total_exported_frames_meta",
    "data_rows_seen_raw_csv",
    "loaded_rows",
    "metadata_minus_loaded_rows",
    "malformed_rows",
    "duration_min",
    "time_monotonic_increasing",
    "duplicate_frame_count",
    "large_gap_gt_1s_count",
    "active_marker_max",
    "rows_with_more_than_3_markers",
]

display(summary_df[display_cols])

print("\nSaved integrity summary:")
print(summary_path)


# --- CELL 23 (code cell #21) ---
# ============================================================
# GROUP 6 - CHECK IF TAKE 3 IS DUPLICATE PREFIX OF TAKE 2
# Compares frame/time and numeric marker values.
# ============================================================

import pandas as pd
import numpy as np
import csv

TAKE2 = "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_2.csv"
TAKE3 = "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_3.csv"

def get_expected_ncols(path):
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(7)]
    return len(header_rows[6])

def load_raw(path):
    ncols = get_expected_ncols(path)
    df = pd.read_csv(path, header=None, skiprows=7, low_memory=False, on_bad_lines="skip")
    return df.iloc[:, :ncols].copy()

t2 = load_raw(TAKE2)
t3 = load_raw(TAKE3)

n = len(t3)

print("="*80)
print("GROUP 6 TAKE 2 vs TAKE 3 DUPLICATE CHECK")
print("="*80)

print("Take 2 rows:", len(t2))
print("Take 3 rows:", len(t3))
print("Comparing Take 3 with first", n, "rows of Take 2")

t2_prefix = t2.iloc[:n, :].copy()

# Compare frame/time first
frame_time_equal = t2_prefix.iloc[:, :2].astype(str).equals(t3.iloc[:, :2].astype(str))

print("\nFrame/time identical?")
print(frame_time_equal)

# Numeric comparison for all columns
t2_num = t2_prefix.apply(pd.to_numeric, errors="coerce")
t3_num = t3.apply(pd.to_numeric, errors="coerce")

same_shape = t2_num.shape == t3_num.shape
print("\nSame shape?")
print(same_shape)

if same_shape:
    diff = (t2_num - t3_num).abs()
    max_diff = np.nanmax(diff.values)
    mean_diff = np.nanmean(diff.values)

    print("\nNumeric max absolute difference:", max_diff)
    print("Numeric mean absolute difference:", mean_diff)

    identical_numeric = np.nan_to_num(diff.values, nan=0).max() == 0
    print("Numerically identical ignoring NaNs?")
    print(identical_numeric)

    # Check where they differ most, if any
    if max_diff > 0:
        rows, cols = np.where(diff.values == max_diff)
        print("\nLargest difference example:")
        print("row:", rows[0], "col:", cols[0])
        print("Take 2 value:", t2_prefix.iloc[rows[0], cols[0]])
        print("Take 3 value:", t3.iloc[rows[0], cols[0]])

print("\nRecommendation:")
if frame_time_equal and same_shape:
    if np.nanmax((t2_num - t3_num).abs().values) == 0:
        print("Take 3 is an exact duplicate prefix of Take 2. Exclude Take 3.")
    else:
        print("Take 3 has same timing as Take 2 but different values. Treat as suspicious and inspect manually.")
else:
    print("Take 3 is not a simple prefix duplicate. Inspect manually before deciding.")


# --- CELL 24 (code cell #22) ---
# ============================================================
# OPTITRACK GROUP 6 - START INSPECTION
# Take 3 is excluded because it is an exact duplicate prefix of Take 2.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 6

BASE = "/content/drive/MyDrive/thesis/data"
OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

# IMPORTANT:
# Take 3 is excluded because duplicate check showed:
# Take 3 == exact first 63.06s prefix of Take 2.
TAKE_FILES = [
    "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_1.csv",
    "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_2.csv",
]

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES USED")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 25 (code cell #23) ---
# ============================================================
# OPTITRACK GROUP 6 - MANUAL TRACKLET STITCHING
# Take 3 excluded: exact duplicate prefix of Take 2.
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 6

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_6/optitrack/Arda_Group-6_Take_2.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_6/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 1289",
            "Unlabeled 1348",
            "Unlabeled 1369",
        ],

        "landmark2": [
            "Unlabeled 1288",
            "Unlabeled 1290",
            "Unlabeled 1305",
            "Unlabeled 1320",
            "Unlabeled 1321",
            "Unlabeled 1338",
        ],

        "landmark3": [
            "Unlabeled 1293",
            "Unlabeled 1303",
            "Unlabeled 1307",
            "Unlabeled 1334",
            "Unlabeled 1344",
            "Unlabeled 1349",
            "Unlabeled 1360",
            "Unlabeled 1362",
            "Unlabeled 1363",
            "Unlabeled 1374",
            "Unlabeled 1378",
            "Unlabeled 1385",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 2086",
            "Unlabeled 2108",
            "Unlabeled 2130",
            "Unlabeled 2135",
            "Unlabeled 2137",
            "Unlabeled 2139",
            "Unlabeled 2144",
            "Unlabeled 2174",
        ],

        "landmark2": [
            "Unlabeled 2084",
            "Unlabeled 2088",
            "Unlabeled 2090",
            "Unlabeled 2093",
            "Unlabeled 2096",
            "Unlabeled 2098",
            "Unlabeled 2103",
            "Unlabeled 2109",
            "Unlabeled 2111",
            "Unlabeled 2115",
            "Unlabeled 2118",
            "Unlabeled 2120",
            "Unlabeled 2127",
        ],

        "landmark3": [
            "Unlabeled 2085",
            "Unlabeled 2100",
            "Unlabeled 2106",
            "Unlabeled 2110",
            "Unlabeled 2116",
            "Unlabeled 2117",
            "Unlabeled 2123",
            "Unlabeled 2126",
            "Unlabeled 2148",
            "Unlabeled 2177",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmarks
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 26 (code cell #24) ---
# ============================================================
# OPTITRACK GROUP 6 - COMBINE TAKE 1–2
# Take 3 excluded: exact duplicate prefix of Take 2.
# Uses real capture-start offset.
# Input: manually stitched smoothed files.
# Output: one continuous Group 6 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 6

IN_DIR = "/content/drive/MyDrive/thesis/data/group_6/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_6/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_6_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_6_optitrack_take_2_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-23 16:25:43.206
# Take 2: 2026-04-23 16:40:51.075
# Offset = 15 min 7.869 s = 907.869 s
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 907.869,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Add data-quality note column
combined["optitrack_quality_note"] = ""
combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 has weak Landmark 3 availability; large missing interval not force-filled."
)

out_path = f"{OUT_DIR}/group_6_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 6 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 6 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 6 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 6 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 6 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 28 (code cell #25) ---
# ============================================================
# OPTITRACK GROUP 7 - START INSPECTION
# Automatically finds OptiTrack take files for Group 7.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 7

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 29 (code cell #26) ---
# ============================================================
# OPTITRACK GROUP 7 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 7 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 7

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_4.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_7/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 1216",
        ],
        "landmark2": [
            "Unlabeled 1218",
        ],
        "landmark3": [
            "Unlabeled 1220",
            "Unlabeled 1221",
            "Unlabeled 1226",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 1406",
        ],
        "landmark2": [
            "Unlabeled 1404",
            "Unlabeled 1408",
            "Unlabeled 1414",
            "Unlabeled 1416",
            "Unlabeled 1433",
        ],
        "landmark3": [
            "Unlabeled 1405",
            "Unlabeled 1411",
            "Unlabeled 1412",
            "Unlabeled 1417",
            "Unlabeled 1425",
            "Unlabeled 1430",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 1483",
            "Unlabeled 1494",
            "Unlabeled 1496",
        ],
        "landmark2": [
            "Unlabeled 1484",
            "Unlabeled 1491",
            "Unlabeled 1499",
        ],
        "landmark3": [
            "Unlabeled 1482",
            "Unlabeled 1492",
            "Unlabeled 1495",
            "Unlabeled 1497",
            "Unlabeled 1502",
            "Unlabeled 1503",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 1289",
            "Unlabeled 1303",
            "Unlabeled 1304",
            "Unlabeled 1307",
            "Unlabeled 1308",
            "Unlabeled 1320",
            "Unlabeled 1333",
            "Unlabeled 1337",
            "Unlabeled 1338",
        ],
        "landmark2": [
            "Unlabeled 1291",
            "Unlabeled 1293",
            "Unlabeled 1298",
            "Unlabeled 1315",
            "Unlabeled 1324",
            "Unlabeled 1328",
            "Unlabeled 1329",
        ],
        "landmark3": [
            "Unlabeled 1292",
            "Unlabeled 1294",
            "Unlabeled 1295",
            "Unlabeled 1301",
            "Unlabeled 1302",
            "Unlabeled 1314",
            "Unlabeled 1316",
            "Unlabeled 1334",
            "Unlabeled 1339",
            "Unlabeled 1342",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmarks
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 30 (code cell #27) ---
# ============================================================
# GROUP 7 - GAP CANDIDATE INSPECTION FOR CURRENT STITCHING
# Goal:
# Find unused marker fragments that may fill missing periods,
# especially Take 2 Landmark 2 and Take 4 Landmark 2/3.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 7

RAW_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_7/optitrack/Arda_Group_7_Take_4.csv",
}

CLEAN_DIR = "/content/drive/MyDrive/thesis/data/group_7/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_7/optitrack_gap_candidate_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1216"],
        "landmark2": ["Unlabeled 1218"],
        "landmark3": ["Unlabeled 1220", "Unlabeled 1221", "Unlabeled 1226"],
    },

    "take_2": {
        "landmark1": ["Unlabeled 1406"],
        "landmark2": ["Unlabeled 1404", "Unlabeled 1408", "Unlabeled 1416"],
        "landmark3": ["Unlabeled 1405", "Unlabeled 1412", "Unlabeled 1417", "Unlabeled 1425", "Unlabeled 1430"],
    },

    "take_3": {
        "landmark1": ["Unlabeled 1483", "Unlabeled 1494", "Unlabeled 1496"],
        "landmark2": ["Unlabeled 1484", "Unlabeled 1491", "Unlabeled 1499"],
        "landmark3": ["Unlabeled 1482", "Unlabeled 1492", "Unlabeled 1495", "Unlabeled 1497", "Unlabeled 1503"],
    },

    "take_4": {
        "landmark1": ["Unlabeled 1289", "Unlabeled 1308", "Unlabeled 1320", "Unlabeled 1338"],
        "landmark2": ["Unlabeled 1291", "Unlabeled 1293", "Unlabeled 1298", "Unlabeled 1329"],
        "landmark3": ["Unlabeled 1295", "Unlabeled 1302", "Unlabeled 1316", "Unlabeled 1339", "Unlabeled 1342"],
    },
}

MIN_GAP_S = 2.0
MIN_CANDIDATE_FRAMES = 300
TOP_N = 12


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]
    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return long_df


def get_missing_gaps(clean, lm):
    avail_col = f"landmark{lm}_available"
    time = clean["time_s"].values
    available = clean[avail_col].fillna(False).astype(bool).values

    gaps = []
    in_gap = False
    start_idx = None

    for i, ok in enumerate(available):
        if not ok and not in_gap:
            in_gap = True
            start_idx = i

        elif ok and in_gap:
            end_idx = i - 1
            s = time[start_idx]
            e = time[end_idx]

            if e - s >= MIN_GAP_S:
                gaps.append({
                    "lm": lm,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_s": e - s,
                })

            in_gap = False

    if in_gap:
        end_idx = len(available) - 1
        s = time[start_idx]
        e = time[end_idx]

        if e - s >= MIN_GAP_S:
            gaps.append({
                "lm": lm,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": s,
                "end_time": e,
                "duration_s": e - s,
            })

    return gaps


def nearest_valid_position(clean, lm, idx, direction):
    cols = [f"landmark{lm}_x", f"landmark{lm}_y", f"landmark{lm}_z"]

    if direction == "before":
        search = range(idx, -1, -1)
    else:
        search = range(idx, len(clean))

    for j in search:
        vals = clean.loc[j, cols].values.astype(float)
        if np.isfinite(vals).all():
            return vals

    return None


def find_candidates_for_gap(long_df, clean, gap, used_markers):
    s = gap["start_time"]
    e = gap["end_time"]
    lm = gap["lm"]

    before_pos = nearest_valid_position(clean, lm, gap["start_idx"], "before")
    after_pos = nearest_valid_position(clean, lm, gap["end_idx"], "after")

    gap_df = long_df[
        (long_df["time_s"] >= s) &
        (long_df["time_s"] <= e) &
        (~long_df["marker_name"].isin(used_markers))
    ].copy()

    rows = []

    for marker, sub in gap_df.groupby("marker_name"):
        if len(sub) < MIN_CANDIDATE_FRAMES:
            continue

        sub_sorted = sub.sort_values("time_s")

        start_pos = sub_sorted[["x", "y", "z"]].iloc[0].values.astype(float)
        end_pos = sub_sorted[["x", "y", "z"]].iloc[-1].values.astype(float)
        mean_pos = sub_sorted[["x", "y", "z"]].mean().values.astype(float)

        dist_before = np.nan
        dist_after = np.nan

        if before_pos is not None:
            dist_before = np.linalg.norm(start_pos - before_pos)

        if after_pos is not None:
            dist_after = np.linalg.norm(end_pos - after_pos)

        score_parts = []
        if np.isfinite(dist_before):
            score_parts.append(dist_before)
        if np.isfinite(dist_after):
            score_parts.append(dist_after)

        continuity_score = np.mean(score_parts) if score_parts else np.nan

        rows.append({
            "landmark": lm,
            "gap_start": s,
            "gap_end": e,
            "gap_duration_s": e - s,
            "candidate_marker": marker,
            "candidate_frames_in_gap": len(sub),
            "candidate_start_time": sub["time_s"].min(),
            "candidate_end_time": sub["time_s"].max(),
            "dist_to_before": dist_before,
            "dist_to_after": dist_after,
            "continuity_score": continuity_score,
            "mean_x": mean_pos[0],
            "mean_y": mean_pos[1],
            "mean_z": mean_pos[2],
        })

    if len(rows) == 0:
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    cand = cand.sort_values(
        ["candidate_frames_in_gap", "continuity_score"],
        ascending=[False, True]
    )

    return cand


all_suggestions = []

for take_name, raw_path in RAW_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print("="*90)

    clean_path = os.path.join(
        CLEAN_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    clean = pd.read_csv(clean_path, low_memory=False)
    long_df = read_motive_long(raw_path)

    used_markers = []
    for lm_name, markers in CURRENT_CHAINS[take_name].items():
        used_markers.extend(markers)

    print("Currently used markers:")
    print(used_markers)

    for lm in [1, 2, 3]:
        gaps = get_missing_gaps(clean, lm)
        gaps = sorted(gaps, key=lambda g: g["duration_s"], reverse=True)

        print(f"\nLandmark {lm} has {len(gaps)} gaps >= {MIN_GAP_S}s")

        for gap_i, gap in enumerate(gaps[:8]):
            print(
                f"  Gap {gap_i}: "
                f"{gap['start_time']:.2f}s -> {gap['end_time']:.2f}s "
                f"dur={gap['duration_s']:.2f}s"
            )

            cand = find_candidates_for_gap(long_df, clean, gap, used_markers)

            if len(cand) > 0:
                print("  Top candidate markers:")
                display(cand.head(TOP_N))

                all_suggestions.append(cand.head(TOP_N))

                plt.figure(figsize=(10, 8))

                window = clean[
                    (clean["time_s"] >= gap["start_time"] - 20) &
                    (clean["time_s"] <= gap["end_time"] + 20)
                ]

                plt.plot(
                    window[f"landmark{lm}_x"],
                    window[f"landmark{lm}_z"],
                    linewidth=2,
                    color="black",
                    label=f"Current Landmark {lm}"
                )

                for marker in cand.head(6)["candidate_marker"]:
                    sub = long_df[
                        (long_df["marker_name"] == marker) &
                        (long_df["time_s"] >= gap["start_time"]) &
                        (long_df["time_s"] <= gap["end_time"])
                    ]
                    plt.plot(sub["x"], sub["z"], linewidth=1.2, label=marker)

                plt.title(
                    f"Group {GROUP} {take_name} Landmark {lm} gap {gap_i}: "
                    f"{gap['start_time']:.1f}-{gap['end_time']:.1f}s"
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

if len(all_suggestions) > 0:
    suggestions = pd.concat(all_suggestions, ignore_index=True)
    suggestions_path = os.path.join(OUT_DIR, f"group_{GROUP}_gap_candidate_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    print("\nSaved all suggestions:")
    print(suggestions_path)

    print("\nMost promising candidates overall:")
    display(
        suggestions
        .sort_values(["landmark", "candidate_frames_in_gap"], ascending=[True, False])
        .head(120)
    )
else:
    print("No candidates found.")


# --- CELL 31 (code cell #28) ---
# ============================================================
# OPTITRACK GROUP 7 - COMBINE TAKE 1–4
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 7 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 7

IN_DIR = "/content/drive/MyDrive/thesis/data/group_7/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_7/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_7_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_7_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_7_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_7_optitrack_take_4_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-29 12:35:49.872
# Take 2: 2026-04-29 12:46:00.195
# Take 3: 2026-04-29 12:56:17.972
# Take 4: 2026-04-29 13:07:50.337
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 610.323,
    "take_3": 1228.100,
    "take_4": 1920.465,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Add quality note column
combined["optitrack_quality_note"] = ""
combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 has weak Landmark 2 availability; large gap was not force-filled."
)

out_path = f"{OUT_DIR}/group_7_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 7 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 7 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 7 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 7 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 7 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 33 (code cell #29) ---
# ============================================================
# OPTITRACK GROUP 8 - START INSPECTION
# Automatically finds OptiTrack take files for Group 8.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 8

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 34 (code cell #30) ---
# ============================================================
# OPTITRACK GROUP 8 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 8 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 8

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_5.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_8/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 1657",
        ],
        "landmark2": [
            "Unlabeled 1658",
            "Unlabeled 1678",
            "Unlabeled 1692",
        ],
        "landmark3": [
            "Unlabeled 1655",
            "Unlabeled 1662",
            "Unlabeled 1664",
            "Unlabeled 1665",
            "Unlabeled 1672",
            "Unlabeled 1675",
            "Unlabeled 1676",
            "Unlabeled 1679",
            "Unlabeled 1680",
            "Unlabeled 1681",
            "Unlabeled 1682",
            "Unlabeled 1684",
            "Unlabeled 1685",
            "Unlabeled 1687",
            "Unlabeled 1688",
            "Unlabeled 1689",
            "Unlabeled 1690",
            "Unlabeled 1691",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 1782",
        ],
        "landmark2": [
            "Unlabeled 1783",
            "Unlabeled 1790",
            "Unlabeled 1802",
            "Unlabeled 1823",
        ],
        "landmark3": [
            "Unlabeled 1781",
            "Unlabeled 1784",
            "Unlabeled 1785",
            "Unlabeled 1786",
            "Unlabeled 1787",
            "Unlabeled 1789",
            "Unlabeled 1791",
            "Unlabeled 1792",
            "Unlabeled 1793",
            "Unlabeled 1796",
            "Unlabeled 1797",
            "Unlabeled 1798",
            "Unlabeled 1801",
            "Unlabeled 1808",
            "Unlabeled 1809",
            "Unlabeled 1810",
            "Unlabeled 1811",
            "Unlabeled 1813",
            "Unlabeled 1814",
            "Unlabeled 1815",
            "Unlabeled 1817",
            "Unlabeled 1821",
            "Unlabeled 1822",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 1992",
            "Unlabeled 2015",
            "Unlabeled 2053",
        ],
        "landmark2": [
            "Unlabeled 1991",
            "Unlabeled 1994",
            "Unlabeled 1995",
            "Unlabeled 2011",
            "Unlabeled 2043",
            "Unlabeled 2062",
            "Unlabeled 2063",
            "Unlabeled 2074",
        ],
        "landmark3": [
            "Unlabeled 1993",
            "Unlabeled 1996",
            "Unlabeled 1997",
            "Unlabeled 1998",
            "Unlabeled 1999",
            "Unlabeled 2000",
            "Unlabeled 2004",
            "Unlabeled 2007",
            "Unlabeled 2008",
            "Unlabeled 2020",
            "Unlabeled 2021",
            "Unlabeled 2022",
            "Unlabeled 2023",
            "Unlabeled 2024",
            "Unlabeled 2025",
            "Unlabeled 2027",
            "Unlabeled 2030",
            "Unlabeled 2031",
            "Unlabeled 2033",
            "Unlabeled 2034",
            "Unlabeled 2035",
            "Unlabeled 2036",
            "Unlabeled 2037",
            "Unlabeled 2042",
            "Unlabeled 2044",
            "Unlabeled 2045",
            "Unlabeled 2046",
            "Unlabeled 2047",
            "Unlabeled 2048",
            "Unlabeled 2050",
            "Unlabeled 2051",
            "Unlabeled 2055",
            "Unlabeled 2056",
            "Unlabeled 2060",
            "Unlabeled 2064",
            "Unlabeled 2068",
            "Unlabeled 2075",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 2149",
        ],
        "landmark2": [
            "Unlabeled 2142",
            "Unlabeled 2147",
            "Unlabeled 2153",
            "Unlabeled 2155",
            "Unlabeled 2161",
            "Unlabeled 2162",
            "Unlabeled 2173",
            "Unlabeled 2174",
        ],
        "landmark3": [
            "Unlabeled 2143",
            "Unlabeled 2144",
            "Unlabeled 2146",
            "Unlabeled 2148",
            "Unlabeled 2150",
            "Unlabeled 2151",
            "Unlabeled 2158",
            "Unlabeled 2160",
        ],
    },

    "take_5": {
        "landmark1": [
            "Unlabeled 2186",
        ],
        "landmark2": [
            "Unlabeled 2187",
            "Unlabeled 2189",
        ],
        "landmark3": [
            "Unlabeled 2185",
            "Unlabeled 2188",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 35 (code cell #31) ---
# ============================================================
# GROUP 8 - GAP CANDIDATE INSPECTION FOR CURRENT STITCHING
# Goal:
# Find unused marker fragments that may fill missing periods,
# especially Take 2 L3, Take 3 L2, Take 4 L1/L3, Take 5 L2.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 8

RAW_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_8/optitrack/Arda_Group_8_Take_5.csv",
}

CLEAN_DIR = "/content/drive/MyDrive/thesis/data/group_8/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_8/optitrack_gap_candidate_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 1657"],
        "landmark2": ["Unlabeled 1658", "Unlabeled 1678"],
        "landmark3": [
            "Unlabeled 1655", "Unlabeled 1662", "Unlabeled 1664", "Unlabeled 1665",
            "Unlabeled 1672", "Unlabeled 1675", "Unlabeled 1676", "Unlabeled 1679",
            "Unlabeled 1680", "Unlabeled 1681", "Unlabeled 1682", "Unlabeled 1684",
            "Unlabeled 1685", "Unlabeled 1687", "Unlabeled 1688", "Unlabeled 1689",
            "Unlabeled 1690", "Unlabeled 1691",
        ],
    },

    "take_2": {
        "landmark1": ["Unlabeled 1782"],
        "landmark2": ["Unlabeled 1783", "Unlabeled 1790", "Unlabeled 1802", "Unlabeled 1823"],
        "landmark3": [
            "Unlabeled 1781", "Unlabeled 1784", "Unlabeled 1785", "Unlabeled 1786",
            "Unlabeled 1787", "Unlabeled 1791", "Unlabeled 1792", "Unlabeled 1793",
            "Unlabeled 1796", "Unlabeled 1797", "Unlabeled 1798", "Unlabeled 1801",
            "Unlabeled 1808", "Unlabeled 1809", "Unlabeled 1810", "Unlabeled 1811",
            "Unlabeled 1813", "Unlabeled 1814", "Unlabeled 1815", "Unlabeled 1817",
            "Unlabeled 1821", "Unlabeled 1822",
        ],
    },

    "take_3": {
        "landmark1": ["Unlabeled 1992", "Unlabeled 2015", "Unlabeled 2053"],
        "landmark2": ["Unlabeled 1995", "Unlabeled 2011"],
        "landmark3": [
            "Unlabeled 1993", "Unlabeled 1999", "Unlabeled 2000", "Unlabeled 2004",
            "Unlabeled 2007", "Unlabeled 2008", "Unlabeled 2021", "Unlabeled 2022",
            "Unlabeled 2025", "Unlabeled 2027", "Unlabeled 2033", "Unlabeled 2034",
            "Unlabeled 2036", "Unlabeled 2037", "Unlabeled 2042", "Unlabeled 2044",
            "Unlabeled 2045", "Unlabeled 2046", "Unlabeled 2047", "Unlabeled 2048",
            "Unlabeled 2050", "Unlabeled 2051", "Unlabeled 2055", "Unlabeled 2056",
            "Unlabeled 2060", "Unlabeled 2064", "Unlabeled 2068", "Unlabeled 2075",
        ],
    },

    "take_4": {
        "landmark1": ["Unlabeled 2149"],
        "landmark2": [
            "Unlabeled 2142", "Unlabeled 2147", "Unlabeled 2153", "Unlabeled 2155",
            "Unlabeled 2161", "Unlabeled 2162", "Unlabeled 2173", "Unlabeled 2174",
        ],
        "landmark3": [
            "Unlabeled 2143", "Unlabeled 2144", "Unlabeled 2146", "Unlabeled 2148",
            "Unlabeled 2150", "Unlabeled 2151", "Unlabeled 2158", "Unlabeled 2160",
        ],
    },

    "take_5": {
        "landmark1": ["Unlabeled 2186"],
        "landmark2": ["Unlabeled 2187"],
        "landmark3": ["Unlabeled 2185", "Unlabeled 2188"],
    },
}

MIN_GAP_S = 2.0
MIN_CANDIDATE_FRAMES = 300
TOP_N = 12


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]
    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return long_df


def get_missing_gaps(clean, lm):
    avail_col = f"landmark{lm}_available"
    time = clean["time_s"].values
    available = clean[avail_col].fillna(False).astype(bool).values

    gaps = []
    in_gap = False
    start_idx = None

    for i, ok in enumerate(available):
        if not ok and not in_gap:
            in_gap = True
            start_idx = i

        elif ok and in_gap:
            end_idx = i - 1
            s = time[start_idx]
            e = time[end_idx]

            if e - s >= MIN_GAP_S:
                gaps.append({
                    "lm": lm,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_s": e - s,
                })

            in_gap = False

    if in_gap:
        end_idx = len(available) - 1
        s = time[start_idx]
        e = time[end_idx]

        if e - s >= MIN_GAP_S:
            gaps.append({
                "lm": lm,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": s,
                "end_time": e,
                "duration_s": e - s,
            })

    return gaps


def nearest_valid_position(clean, lm, idx, direction):
    cols = [f"landmark{lm}_x", f"landmark{lm}_y", f"landmark{lm}_z"]

    if direction == "before":
        search = range(idx, -1, -1)
    else:
        search = range(idx, len(clean))

    for j in search:
        vals = clean.loc[j, cols].values.astype(float)
        if np.isfinite(vals).all():
            return vals

    return None


def find_candidates_for_gap(long_df, clean, gap, used_markers):
    s = gap["start_time"]
    e = gap["end_time"]
    lm = gap["lm"]

    before_pos = nearest_valid_position(clean, lm, gap["start_idx"], "before")
    after_pos = nearest_valid_position(clean, lm, gap["end_idx"], "after")

    gap_df = long_df[
        (long_df["time_s"] >= s) &
        (long_df["time_s"] <= e) &
        (~long_df["marker_name"].isin(used_markers))
    ].copy()

    rows = []

    for marker, sub in gap_df.groupby("marker_name"):
        if len(sub) < MIN_CANDIDATE_FRAMES:
            continue

        sub_sorted = sub.sort_values("time_s")

        start_pos = sub_sorted[["x", "y", "z"]].iloc[0].values.astype(float)
        end_pos = sub_sorted[["x", "y", "z"]].iloc[-1].values.astype(float)
        mean_pos = sub_sorted[["x", "y", "z"]].mean().values.astype(float)

        dist_before = np.nan
        dist_after = np.nan

        if before_pos is not None:
            dist_before = np.linalg.norm(start_pos - before_pos)

        if after_pos is not None:
            dist_after = np.linalg.norm(end_pos - after_pos)

        score_parts = []
        if np.isfinite(dist_before):
            score_parts.append(dist_before)
        if np.isfinite(dist_after):
            score_parts.append(dist_after)

        continuity_score = np.mean(score_parts) if score_parts else np.nan

        rows.append({
            "landmark": lm,
            "gap_start": s,
            "gap_end": e,
            "gap_duration_s": e - s,
            "candidate_marker": marker,
            "candidate_frames_in_gap": len(sub),
            "candidate_start_time": sub["time_s"].min(),
            "candidate_end_time": sub["time_s"].max(),
            "dist_to_before": dist_before,
            "dist_to_after": dist_after,
            "continuity_score": continuity_score,
            "mean_x": mean_pos[0],
            "mean_y": mean_pos[1],
            "mean_z": mean_pos[2],
        })

    if len(rows) == 0:
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    cand = cand.sort_values(
        ["candidate_frames_in_gap", "continuity_score"],
        ascending=[False, True]
    )

    return cand


all_suggestions = []

for take_name, raw_path in RAW_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print("="*90)

    clean_path = os.path.join(
        CLEAN_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    clean = pd.read_csv(clean_path, low_memory=False)
    long_df = read_motive_long(raw_path)

    used_markers = []
    for lm_name, markers in CURRENT_CHAINS[take_name].items():
        used_markers.extend(markers)

    print("Currently used markers:")
    print(used_markers)

    for lm in [1, 2, 3]:
        gaps = get_missing_gaps(clean, lm)
        gaps = sorted(gaps, key=lambda g: g["duration_s"], reverse=True)

        print(f"\nLandmark {lm} has {len(gaps)} gaps >= {MIN_GAP_S}s")

        for gap_i, gap in enumerate(gaps[:8]):
            print(
                f"  Gap {gap_i}: "
                f"{gap['start_time']:.2f}s -> {gap['end_time']:.2f}s "
                f"dur={gap['duration_s']:.2f}s"
            )

            cand = find_candidates_for_gap(long_df, clean, gap, used_markers)

            if len(cand) > 0:
                print("  Top candidate markers:")
                display(cand.head(TOP_N))

                all_suggestions.append(cand.head(TOP_N))

                plt.figure(figsize=(10, 8))

                window = clean[
                    (clean["time_s"] >= gap["start_time"] - 20) &
                    (clean["time_s"] <= gap["end_time"] + 20)
                ]

                plt.plot(
                    window[f"landmark{lm}_x"],
                    window[f"landmark{lm}_z"],
                    linewidth=2,
                    color="black",
                    label=f"Current Landmark {lm}"
                )

                for marker in cand.head(6)["candidate_marker"]:
                    sub = long_df[
                        (long_df["marker_name"] == marker) &
                        (long_df["time_s"] >= gap["start_time"]) &
                        (long_df["time_s"] <= gap["end_time"])
                    ]
                    plt.plot(sub["x"], sub["z"], linewidth=1.2, label=marker)

                plt.title(
                    f"Group {GROUP} {take_name} Landmark {lm} gap {gap_i}: "
                    f"{gap['start_time']:.1f}-{gap['end_time']:.1f}s"
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

if len(all_suggestions) > 0:
    suggestions = pd.concat(all_suggestions, ignore_index=True)
    suggestions_path = os.path.join(OUT_DIR, f"group_{GROUP}_gap_candidate_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    print("\nSaved all suggestions:")
    print(suggestions_path)

    print("\nMost promising candidates overall:")
    display(
        suggestions
        .sort_values(["landmark", "candidate_frames_in_gap"], ascending=[True, False])
        .head(160)
    )
else:
    print("No candidates found.")


# --- CELL 36 (code cell #32) ---
# ============================================================
# OPTITRACK GROUP 8 - COMBINE TAKE 1–5
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 8 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 8

IN_DIR = "/content/drive/MyDrive/thesis/data/group_8/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_8/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_8_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_8_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_8_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_8_optitrack_take_4_manual_stitched_smoothed.csv",
    "take_5": f"{IN_DIR}/group_8_optitrack_take_5_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-30 13:45:54.738
# Take 2: 2026-04-30 13:56:41.954
# Take 3: 2026-04-30 14:06:57.794
# Take 4: 2026-04-30 14:20:35.288
# Take 5: 2026-04-30 14:28:25.432
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 647.216,
    "take_3": 1263.056,
    "take_4": 2080.550,
    "take_5": 2550.694,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Quality notes
combined["optitrack_quality_note"] = ""

combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 has moderate Landmark 3 availability; weak gaps were not force-filled."
)

combined.loc[combined["take"] == "take_3", "optitrack_quality_note"] = (
    "Take 3 has weak Landmark 2 availability; long gap was not force-filled."
)

combined.loc[combined["take"] == "take_4", "optitrack_quality_note"] = (
    "Take 4 has moderate Landmark 1 and Landmark 3 availability; risky candidates were not added."
)

out_path = f"{OUT_DIR}/group_8_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 8 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 8 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 8 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 8 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 8 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 38 (code cell #33) ---
# ============================================================
# OPTITRACK GROUP 9 - START INSPECTION
# Automatically finds OptiTrack take files for Group 9.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 39 (code cell #34) ---
# ============================================================
# OPTITRACK GROUP 9 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 9 chains based on inspection output.
#
# NOTE:
# Take 5 and Take 6 may need refinement because the printed
# inspection output was truncated around Take 5.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_5.csv",
    "take_6": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_6.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 2310",
            "Unlabeled 2331",
        ],
        "landmark2": [
            "Unlabeled 2311",
            "Unlabeled 2317",
            "Unlabeled 2334",
            "Unlabeled 2338",
            "Unlabeled 2343",
            "Unlabeled 2345",
            "Unlabeled 2358",
            "Unlabeled 2373",
        ],
        "landmark3": [
            "Unlabeled 2335",
            "Unlabeled 2344",
            "Unlabeled 2354",
            "Unlabeled 2359",
            "Unlabeled 2371",
            "Unlabeled 2372",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 2479",
        ],
        "landmark2": [
            "Unlabeled 2481",
            "Unlabeled 2482",
            "Unlabeled 2493",
            "Unlabeled 2495",
            "Unlabeled 2497",
            "Unlabeled 2503",
            "Unlabeled 2525",
        ],
        "landmark3": [
            "Unlabeled 2485",
            "Unlabeled 2498",
            "Unlabeled 2501",
            "Unlabeled 2505",
            "Unlabeled 2507",
            "Unlabeled 2508",
            "Unlabeled 2511",
            "Unlabeled 2512",
            "Unlabeled 2513",
            "Unlabeled 2514",
            "Unlabeled 2516",
            "Unlabeled 2517",
            "Unlabeled 2518",
            "Unlabeled 2519",
            "Unlabeled 2521",
            "Unlabeled 2524",
            "Unlabeled 2528",
            "Unlabeled 2530",
            "Unlabeled 2533",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 2605",
            "Unlabeled 2613",
            "Unlabeled 2635",
        ],
        "landmark2": [
            "Unlabeled 2604",
        ],
        "landmark3": [
            "Unlabeled 2610",
            "Unlabeled 2611",
            "Unlabeled 2612",
            "Unlabeled 2616",
            "Unlabeled 2617",
            "Unlabeled 2618",
            "Unlabeled 2620",
            "Unlabeled 2621",
            "Unlabeled 2623",
            "Unlabeled 2624",
            "Unlabeled 2625",
            "Unlabeled 2626",
            "Unlabeled 2627",
            "Unlabeled 2630",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 2732",
            "Unlabeled 2759",
            "Unlabeled 2768",
            "Unlabeled 2771",
        ],
        "landmark2": [
            "Unlabeled 2731",
            "Unlabeled 2763",
            "Unlabeled 2767",
            "Unlabeled 2770",
        ],
        "landmark3": [
            "Unlabeled 2733",
            "Unlabeled 2743",
            "Unlabeled 2744",
            "Unlabeled 2747",
            "Unlabeled 2752",
            "Unlabeled 2754",
            "Unlabeled 2758",
            "Unlabeled 2769",
            "Unlabeled 2774",
        ],
    },

    # First-pass approximate because Take 5 inspection output was truncated.
    # If availability is bad, we will correct this block using Take 5 marker summary.
    "take_5": {
    "landmark1": [
        "Unlabeled 2904",
        "Unlabeled 2910",
        "Unlabeled 2920",
        "Unlabeled 2925",
        "Unlabeled 2929",
    ],

    "landmark2": [
        "Unlabeled 2903",
        "Unlabeled 2916",
        "Unlabeled 2922",
        "Unlabeled 2924",
    ],

    "landmark3": [
        "Unlabeled 2905",
        "Unlabeled 2906",
        "Unlabeled 2907",
        "Unlabeled 2912",
        "Unlabeled 2914",
        "Unlabeled 2915",
        "Unlabeled 2919",
    ],
},

    "take_6": {
        "landmark1": [
            "Unlabeled 3031",
        ],

        "landmark2": [
            "Unlabeled 3032",
            "Unlabeled 3060",
            "Unlabeled 3062",
        ],

        "landmark3": [
            "Unlabeled 3033",
            "Unlabeled 3061",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z trajectory
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y trajectory
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmark count
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 40 (code cell #35) ---
# ============================================================
# GROUP 9 - FOCUSED MARKER SUMMARY FOR TAKE 5 AND TAKE 6
# Goal:
# Recover correct marker IDs for Take 5 and refine Take 6.
# Reads the already saved inspection summary CSVs.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

GROUP = 9

INSPECTION_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_inspection"

SUMMARY_PATHS = {
    "take_5": f"{INSPECTION_DIR}/take_5_marker_summary.csv",
    "take_6": f"{INSPECTION_DIR}/take_6_marker_summary.csv",
}

for take_name, path in SUMMARY_PATHS.items():
    print("\n" + "="*100)
    print(f"GROUP {GROUP} {take_name} MARKER SUMMARY")
    print(path)
    print("="*100)

    if not os.path.exists(path):
        print("Missing summary file:", path)
        continue

    df = pd.read_csv(path)

    print("\nTop 50 markers by n_frames:")
    display(
        df.sort_values("n_frames", ascending=False)
        .head(50)
        [[
            "marker_name",
            "n_frames",
            "percent",
            "first_time",
            "last_time",
            "duration_s",
            "mean_x",
            "mean_y",
            "mean_z",
            "std_x",
            "std_y",
            "std_z",
            "start_x",
            "start_y",
            "start_z",
            "end_x",
            "end_y",
            "end_z",
        ]]
    )

    # Timeline plot for visible tracklets
    plot_df = df[df["n_frames"] >= 300].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - marker timeline, n_frames >= 300")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # X-Z plot for top markers
    print("\nCandidate marker names only:")
    print(df.sort_values("n_frames", ascending=False).head(50)["marker_name"].tolist())


# --- CELL 41 (code cell #36) ---
# ============================================================
# GROUP 9 - FOCUSED GAP CANDIDATE INSPECTION
# Focus:
# - Take 1 Landmark 3
# - Take 2 Landmark 3
# - Take 5 Landmark 1
# - Take 6 Landmark 1
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

RAW_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_2.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_5.csv",
    "take_6": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_6.csv",
}

CLEAN_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_gap_candidate_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_CHAINS = {
    "take_1": {
        "landmark1": ["Unlabeled 2310", "Unlabeled 2331"],
        "landmark2": [
            "Unlabeled 2311", "Unlabeled 2317", "Unlabeled 2334",
            "Unlabeled 2338", "Unlabeled 2343", "Unlabeled 2345",
            "Unlabeled 2358", "Unlabeled 2373",
        ],
        "landmark3": [
            "Unlabeled 2335", "Unlabeled 2344", "Unlabeled 2354",
            "Unlabeled 2359", "Unlabeled 2371", "Unlabeled 2372",
        ],
    },

    "take_2": {
        "landmark1": ["Unlabeled 2479"],
        "landmark2": [
            "Unlabeled 2481", "Unlabeled 2482", "Unlabeled 2493",
            "Unlabeled 2495", "Unlabeled 2497", "Unlabeled 2503",
            "Unlabeled 2525",
        ],
        "landmark3": [
            "Unlabeled 2485", "Unlabeled 2498", "Unlabeled 2501",
            "Unlabeled 2505", "Unlabeled 2507", "Unlabeled 2508",
            "Unlabeled 2511", "Unlabeled 2512", "Unlabeled 2513",
            "Unlabeled 2514", "Unlabeled 2516", "Unlabeled 2517",
            "Unlabeled 2518", "Unlabeled 2519", "Unlabeled 2521",
            "Unlabeled 2524", "Unlabeled 2528", "Unlabeled 2530",
            "Unlabeled 2533",
        ],
    },

    "take_5": {
        "landmark1": [
            "Unlabeled 2904", "Unlabeled 2910", "Unlabeled 2920",
            "Unlabeled 2925", "Unlabeled 2929",
        ],
        "landmark2": [
            "Unlabeled 2903", "Unlabeled 2916", "Unlabeled 2922",
            "Unlabeled 2924",
        ],
        "landmark3": [
            "Unlabeled 2905", "Unlabeled 2906", "Unlabeled 2907",
            "Unlabeled 2912", "Unlabeled 2914", "Unlabeled 2915",
            "Unlabeled 2919",
        ],
    },

    "take_6": {
        "landmark1": ["Unlabeled 3031"],
        "landmark2": ["Unlabeled 3032", "Unlabeled 3060", "Unlabeled 3062"],
        "landmark3": ["Unlabeled 3033", "Unlabeled 3061"],
    },
}

FOCUS = {
    "take_1": [3],
    "take_2": [3],
    "take_5": [1],
    "take_6": [1],
}

MIN_GAP_S = 2.0
MIN_CANDIDATE_FRAMES = 300
TOP_N = 12


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]
    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return long_df


def get_missing_gaps(clean, lm):
    avail_col = f"landmark{lm}_available"
    time = clean["time_s"].values
    available = clean[avail_col].fillna(False).astype(bool).values

    gaps = []
    in_gap = False
    start_idx = None

    for i, ok in enumerate(available):
        if not ok and not in_gap:
            in_gap = True
            start_idx = i

        elif ok and in_gap:
            end_idx = i - 1
            s = time[start_idx]
            e = time[end_idx]

            if e - s >= MIN_GAP_S:
                gaps.append({
                    "lm": lm,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_s": e - s,
                })

            in_gap = False

    if in_gap:
        end_idx = len(available) - 1
        s = time[start_idx]
        e = time[end_idx]

        if e - s >= MIN_GAP_S:
            gaps.append({
                "lm": lm,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": s,
                "end_time": e,
                "duration_s": e - s,
            })

    return gaps


def nearest_valid_position(clean, lm, idx, direction):
    cols = [f"landmark{lm}_x", f"landmark{lm}_y", f"landmark{lm}_z"]

    if direction == "before":
        search = range(idx, -1, -1)
    else:
        search = range(idx, len(clean))

    for j in search:
        vals = clean.loc[j, cols].values.astype(float)
        if np.isfinite(vals).all():
            return vals

    return None


def find_candidates_for_gap(long_df, clean, gap, used_markers):
    s = gap["start_time"]
    e = gap["end_time"]
    lm = gap["lm"]

    before_pos = nearest_valid_position(clean, lm, gap["start_idx"], "before")
    after_pos = nearest_valid_position(clean, lm, gap["end_idx"], "after")

    gap_df = long_df[
        (long_df["time_s"] >= s) &
        (long_df["time_s"] <= e) &
        (~long_df["marker_name"].isin(used_markers))
    ].copy()

    rows = []

    for marker, sub in gap_df.groupby("marker_name"):
        if len(sub) < MIN_CANDIDATE_FRAMES:
            continue

        sub_sorted = sub.sort_values("time_s")

        start_pos = sub_sorted[["x", "y", "z"]].iloc[0].values.astype(float)
        end_pos = sub_sorted[["x", "y", "z"]].iloc[-1].values.astype(float)
        mean_pos = sub_sorted[["x", "y", "z"]].mean().values.astype(float)

        dist_before = np.nan
        dist_after = np.nan

        if before_pos is not None:
            dist_before = np.linalg.norm(start_pos - before_pos)

        if after_pos is not None:
            dist_after = np.linalg.norm(end_pos - after_pos)

        score_parts = []
        if np.isfinite(dist_before):
            score_parts.append(dist_before)
        if np.isfinite(dist_after):
            score_parts.append(dist_after)

        continuity_score = np.mean(score_parts) if score_parts else np.nan

        rows.append({
            "landmark": lm,
            "gap_start": s,
            "gap_end": e,
            "gap_duration_s": e - s,
            "candidate_marker": marker,
            "candidate_frames_in_gap": len(sub),
            "candidate_start_time": sub["time_s"].min(),
            "candidate_end_time": sub["time_s"].max(),
            "dist_to_before": dist_before,
            "dist_to_after": dist_after,
            "continuity_score": continuity_score,
            "mean_x": mean_pos[0],
            "mean_y": mean_pos[1],
            "mean_z": mean_pos[2],
        })

    if len(rows) == 0:
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    cand = cand.sort_values(
        ["candidate_frames_in_gap", "continuity_score"],
        ascending=[False, True]
    )

    return cand


all_suggestions = []

for take_name, raw_path in RAW_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print("="*90)

    clean_path = os.path.join(
        CLEAN_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    clean = pd.read_csv(clean_path, low_memory=False)
    long_df = read_motive_long(raw_path)

    used_markers = []
    for lm_name, markers in CURRENT_CHAINS[take_name].items():
        used_markers.extend(markers)

    print("Currently used markers:")
    print(used_markers)

    for lm in FOCUS[take_name]:
        gaps = get_missing_gaps(clean, lm)
        gaps = sorted(gaps, key=lambda g: g["duration_s"], reverse=True)

        print(f"\nLandmark {lm} has {len(gaps)} gaps >= {MIN_GAP_S}s")

        for gap_i, gap in enumerate(gaps[:10]):
            print(
                f"  Gap {gap_i}: "
                f"{gap['start_time']:.2f}s -> {gap['end_time']:.2f}s "
                f"dur={gap['duration_s']:.2f}s"
            )

            cand = find_candidates_for_gap(long_df, clean, gap, used_markers)

            if len(cand) > 0:
                print("  Top candidate markers:")
                display(cand.head(TOP_N))

                all_suggestions.append(cand.head(TOP_N))

                plt.figure(figsize=(10, 8))

                window = clean[
                    (clean["time_s"] >= gap["start_time"] - 20) &
                    (clean["time_s"] <= gap["end_time"] + 20)
                ]

                plt.plot(
                    window[f"landmark{lm}_x"],
                    window[f"landmark{lm}_z"],
                    linewidth=2,
                    color="black",
                    label=f"Current Landmark {lm}"
                )

                for marker in cand.head(6)["candidate_marker"]:
                    sub = long_df[
                        (long_df["marker_name"] == marker) &
                        (long_df["time_s"] >= gap["start_time"]) &
                        (long_df["time_s"] <= gap["end_time"])
                    ]
                    plt.plot(sub["x"], sub["z"], linewidth=1.2, label=marker)

                plt.title(
                    f"Group {GROUP} {take_name} Landmark {lm} gap {gap_i}: "
                    f"{gap['start_time']:.1f}-{gap['end_time']:.1f}s"
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

if len(all_suggestions) > 0:
    suggestions = pd.concat(all_suggestions, ignore_index=True)
    suggestions_path = os.path.join(OUT_DIR, f"group_{GROUP}_focused_gap_candidate_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    print("\nSaved focused suggestions:")
    print(suggestions_path)

    print("\nMost promising candidates overall:")
    display(
        suggestions
        .sort_values(["landmark", "candidate_frames_in_gap"], ascending=[True, False])
        .head(120)
    )
else:
    print("No candidates found.")


# --- CELL 42 (code cell #37) ---
# ============================================================
# OPTITRACK GROUP 9 - MANUAL TRACKLET STITCHING UPDATED
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
#
# Updates after focused inspection:
# - Take 2 L3: added Unlabeled 2532
# - Take 5 L1: added Unlabeled 2908
# - Take 6 L1 kept unchanged because no safe successor was found
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_3.csv",
    "take_4": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_4.csv",
    "take_5": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_5.csv",
    "take_6": "/content/drive/MyDrive/thesis/data/group_9/optitrack/Arda_Group_9_Take_6.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 2310",
            "Unlabeled 2331",
        ],

        "landmark2": [
            "Unlabeled 2311",
            "Unlabeled 2317",
            "Unlabeled 2334",
            "Unlabeled 2338",
            "Unlabeled 2343",
            "Unlabeled 2345",
            "Unlabeled 2358",
            "Unlabeled 2373",
        ],

        "landmark3": [
            "Unlabeled 2335",
            "Unlabeled 2344",
            "Unlabeled 2354",
            "Unlabeled 2359",
            "Unlabeled 2371",
            "Unlabeled 2372",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 2479",
        ],

        "landmark2": [
            "Unlabeled 2481",
            "Unlabeled 2482",
            "Unlabeled 2493",
            "Unlabeled 2495",
            "Unlabeled 2497",
            "Unlabeled 2503",
            "Unlabeled 2525",
        ],

        "landmark3": [
            "Unlabeled 2485",
            "Unlabeled 2498",
            "Unlabeled 2501",
            "Unlabeled 2505",
            "Unlabeled 2507",
            "Unlabeled 2508",
            "Unlabeled 2511",
            "Unlabeled 2512",
            "Unlabeled 2513",
            "Unlabeled 2514",
            "Unlabeled 2516",
            "Unlabeled 2517",
            "Unlabeled 2518",
            "Unlabeled 2519",
            "Unlabeled 2521",
            "Unlabeled 2524",
            "Unlabeled 2528",
            "Unlabeled 2530",
            "Unlabeled 2532",
            "Unlabeled 2533",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 2605",
            "Unlabeled 2613",
            "Unlabeled 2635",
        ],

        "landmark2": [
            "Unlabeled 2604",
        ],

        "landmark3": [
            "Unlabeled 2610",
            "Unlabeled 2611",
            "Unlabeled 2612",
            "Unlabeled 2616",
            "Unlabeled 2617",
            "Unlabeled 2618",
            "Unlabeled 2620",
            "Unlabeled 2621",
            "Unlabeled 2623",
            "Unlabeled 2624",
            "Unlabeled 2625",
            "Unlabeled 2626",
            "Unlabeled 2627",
            "Unlabeled 2630",
        ],
    },

    "take_4": {
        "landmark1": [
            "Unlabeled 2732",
            "Unlabeled 2759",
            "Unlabeled 2768",
            "Unlabeled 2771",
        ],

        "landmark2": [
            "Unlabeled 2731",
            "Unlabeled 2763",
            "Unlabeled 2767",
            "Unlabeled 2770",
        ],

        "landmark3": [
            "Unlabeled 2733",
            "Unlabeled 2743",
            "Unlabeled 2744",
            "Unlabeled 2747",
            "Unlabeled 2752",
            "Unlabeled 2754",
            "Unlabeled 2758",
            "Unlabeled 2769",
            "Unlabeled 2774",
        ],
    },

    "take_5": {
        "landmark1": [
            "Unlabeled 2904",
            "Unlabeled 2908",
            "Unlabeled 2910",
            "Unlabeled 2920",
            "Unlabeled 2925",
            "Unlabeled 2929",
        ],

        "landmark2": [
            "Unlabeled 2903",
            "Unlabeled 2916",
            "Unlabeled 2922",
            "Unlabeled 2924",
        ],

        "landmark3": [
            "Unlabeled 2905",
            "Unlabeled 2906",
            "Unlabeled 2907",
            "Unlabeled 2912",
            "Unlabeled 2914",
            "Unlabeled 2915",
            "Unlabeled 2919",
        ],
    },

    "take_6": {
        "landmark1": [
            "Unlabeled 3031",
        ],

        "landmark2": [
            "Unlabeled 3032",
            "Unlabeled 3060",
            "Unlabeled 3062",
        ],

        "landmark3": [
            "Unlabeled 3033",
            "Unlabeled 3061",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    # X-Z trajectory
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X-Y trajectory
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Active landmark count
    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # X/Y/Z over time
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 43 (code cell #38) ---
# ============================================================
# OPTITRACK GROUP 9 - COMBINE TAKE 1–6
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 9 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

IN_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_9/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_9_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_9_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_9_optitrack_take_3_manual_stitched_smoothed.csv",
    "take_4": f"{IN_DIR}/group_9_optitrack_take_4_manual_stitched_smoothed.csv",
    "take_5": f"{IN_DIR}/group_9_optitrack_take_5_manual_stitched_smoothed.csv",
    "take_6": f"{IN_DIR}/group_9_optitrack_take_6_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-30 15:42:50.664
# Take 2: 2026-04-30 15:53:22.735
# Take 3: 2026-04-30 16:04:25.584
# Take 4: 2026-04-30 16:15:00.610
# Take 5: 2026-04-30 16:30:24.858
# Take 6: 2026-04-30 16:41:25.186
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 632.071,
    "take_3": 1294.920,
    "take_4": 1929.946,
    "take_5": 2854.194,
    "take_6": 3514.522,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Quality notes
combined["optitrack_quality_note"] = ""

combined.loc[combined["take"] == "take_1", "optitrack_quality_note"] = (
    "Take 1 has weak Landmark 3 availability; large missing interval was not force-filled."
)

combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 has weak/moderate Landmark 3 availability; only safe short candidate was added."
)

combined.loc[combined["take"] == "take_6", "optitrack_quality_note"] = (
    "Take 6 has weak Landmark 1 availability; no safe successor candidate was found."
)

out_path = f"{OUT_DIR}/group_9_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 9 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 9 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 9 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 9 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 9 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 45 (code cell #39) ---
# ============================================================
# OPTITRACK GROUP 10 - START INSPECTION
# Automatically finds OptiTrack take files for Group 10.
# Produces:
# - active marker count plots
# - marker tracklet summaries
# - top tracklet X-Z plots
# - tracklet timeline plots
# ============================================================

import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

BASE = "/content/drive/MyDrive/thesis/data"
OPTITRACK_DIR = f"{BASE}/group_{GROUP}/optitrack"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_FILES = sorted(glob.glob(os.path.join(OPTITRACK_DIR, "*.csv")))

print("="*80)
print(f"GROUP {GROUP} OPTITRACK FILES FOUND")
print("="*80)

for i, path in enumerate(TAKE_FILES):
    print(f"{i}: {path}")

if len(TAKE_FILES) == 0:
    raise FileNotFoundError(f"No CSV files found in {OPTITRACK_DIR}")

MIN_FRAMES_FOR_TRACKLET_PLOT = 500
TOP_N_TRACKLETS = 40


def read_motive_long_and_active(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    active_count = np.zeros(len(raw), dtype=int)

    parts = []
    summary_rows = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)
        active_count += present.values.astype(int)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

        idx = np.where(present.values)[0]

        summary_rows.append({
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "n_frames": int(present.sum()),
            "percent": present.mean() * 100,
            "first_time": float(time_s.iloc[idx[0]]),
            "last_time": float(time_s.iloc[idx[-1]]),
            "duration_s": float(time_s.iloc[idx[-1]] - time_s.iloc[idx[0]]),
            "start_x": float(xyz.loc[present, "x"].iloc[0]),
            "start_y": float(xyz.loc[present, "y"].iloc[0]),
            "start_z": float(xyz.loc[present, "z"].iloc[0]),
            "end_x": float(xyz.loc[present, "x"].iloc[-1]),
            "end_y": float(xyz.loc[present, "y"].iloc[-1]),
            "end_z": float(xyz.loc[present, "z"].iloc[-1]),
            "mean_x": float(xyz.loc[present, "x"].mean()),
            "mean_y": float(xyz.loc[present, "y"].mean()),
            "mean_z": float(xyz.loc[present, "z"].mean()),
            "std_x": float(xyz.loc[present, "x"].std()),
            "std_y": float(xyz.loc[present, "y"].std()),
            "std_z": float(xyz.loc[present, "z"].std()),
        })

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("n_frames", ascending=False).reset_index(drop=True)

    active_df = pd.DataFrame({
        "frame": frame,
        "time_s": time_s,
        "active_markers": active_count
    })

    return meta, frame_time, long_df, summary, active_df


def plot_active_counts(active_df, take_name):
    plt.figure(figsize=(18, 4))
    plt.plot(active_df["time_s"], active_df["active_markers"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3 markers")
    plt.title(f"Group {GROUP} {take_name} - Active marker count per frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Active markers")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(OUT_DIR, f"{take_name}_active_marker_count.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved active count plot ->", out)


def plot_tracklet_timeline(summary, take_name):
    plot_df = summary[summary["n_frames"] >= MIN_FRAMES_FOR_TRACKLET_PLOT].copy()
    plot_df = plot_df.sort_values("first_time").reset_index(drop=True)

    plt.figure(figsize=(18, 10))

    for i, row in plot_df.iterrows():
        plt.hlines(
            y=i,
            xmin=row["first_time"],
            xmax=row["last_time"],
            linewidth=4
        )
        plt.text(
            row["first_time"],
            i,
            row["marker_name"],
            fontsize=8,
            va="center",
            ha="right"
        )

    plt.title(f"Group {GROUP} {take_name} - Marker tracklets timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Marker fragments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_tracklet_timeline.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved timeline ->", out)


def plot_top_tracklets_xz(summary, long_df, take_name):
    keep = (
        summary
        .sort_values("n_frames", ascending=False)
        .head(TOP_N_TRACKLETS)["marker_name"]
        .tolist()
    )

    plt.figure(figsize=(11, 9))

    for marker in keep:
        sub = long_df[long_df["marker_name"] == marker].sort_values("time_s")

        plt.plot(
            sub["x"],
            sub["z"],
            linewidth=0.9,
            label=marker
        )

        plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], s=18)
        plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], s=18, marker="x")

    plt.title(f"Group {GROUP} {take_name} - Top tracklets X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{take_name}_top_tracklets_xz.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()

    print("Saved X-Z plot ->", out)


# ============================================================
# RUN INSPECTION
# ============================================================

group_outputs = {}

for i, path in enumerate(TAKE_FILES):
    take_name = f"take_{i+1}"

    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df, summary, active_df = read_motive_long_and_active(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Export frame rate:", meta.get("Export Frame Rate"))
    print("Total exported frames:", meta.get("Total Exported Frames"))

    print("\nFrame/time:")
    print("frames:", len(frame_time))
    print("time start:", frame_time["time_s"].min())
    print("time end  :", frame_time["time_s"].max())
    print("duration min:", (frame_time["time_s"].max() - frame_time["time_s"].min()) / 60)

    print("\nLong detections:", long_df.shape)

    print("\nActive marker count distribution:")
    print(active_df["active_markers"].value_counts().sort_index())

    print("\nTop marker tracks:")
    display(summary.head(35))

    summary_path = os.path.join(OUT_DIR, f"{take_name}_marker_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved marker summary ->", summary_path)

    plot_active_counts(active_df, take_name)
    plot_tracklet_timeline(summary, take_name)
    plot_top_tracklets_xz(summary, long_df, take_name)

    group_outputs[take_name] = {
        "path": path,
        "meta": meta,
        "frame_time": frame_time,
        "long_df": long_df,
        "summary": summary,
        "active_df": active_df
    }

print("\n" + "="*80)
print(f"GROUP {GROUP} OPTITRACK INSPECTION COMPLETE")
print("="*80)
print("Output folder:")
print(OUT_DIR)


# --- CELL 46 (code cell #40) ---
# ============================================================
# OPTITRACK GROUP 10 - MANUAL TRACKLET STITCHING
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
# First-pass Group 10 chains based on inspection output.
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_3.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 3113",
            "Unlabeled 3122",
            "Unlabeled 3128",
        ],

        "landmark2": [
            "Unlabeled 3120",
            "Unlabeled 3125",
            "Unlabeled 3126",
            "Unlabeled 3127",
            "Unlabeled 3130",
            "Unlabeled 3132",
            "Unlabeled 3136",
        ],

        "landmark3": [
            "Unlabeled 3118",
            "Unlabeled 3131",
            "Unlabeled 3133",
            "Unlabeled 3134",
            "Unlabeled 3135",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 3176",
            "Unlabeled 3186",
            "Unlabeled 3193",
        ],

        "landmark2": [
            "Unlabeled 3177",
            "Unlabeled 3180",
        ],

        "landmark3": [
            "Unlabeled 3178",
            "Unlabeled 3183",
            "Unlabeled 3184",
            "Unlabeled 3185",
            "Unlabeled 3190",
            "Unlabeled 3192",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 3251",
            "Unlabeled 3256",
        ],

        "landmark2": [
            "Unlabeled 3252",
            "Unlabeled 3254",
            "Unlabeled 3257",
            "Unlabeled 3258",
            "Unlabeled 3260",
        ],

        "landmark3": [
            "Unlabeled 3253",
            "Unlabeled 3259",
            "Unlabeled 3261",
            "Unlabeled 3266",
            "Unlabeled 3267",
            "Unlabeled 3268",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 47 (code cell #41) ---
# ============================================================
# GROUP 10 - FOCUSED GAP CANDIDATE INSPECTION
# Focus:
# - Take 1 Landmark 2
# - Take 2 Landmark 1
# - Take 3 Landmark 1
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

RAW_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_3.csv",
}

CLEAN_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_gap_candidate_inspection"
os.makedirs(OUT_DIR, exist_ok=True)

CURRENT_CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 3113",
            "Unlabeled 3122",
            "Unlabeled 3128",
        ],
        "landmark2": [
            "Unlabeled 3120",
            "Unlabeled 3125",
            "Unlabeled 3126",
            "Unlabeled 3127",
            "Unlabeled 3130",
            "Unlabeled 3132",
            "Unlabeled 3136",
        ],
        "landmark3": [
            "Unlabeled 3118",
            "Unlabeled 3131",
            "Unlabeled 3133",
            "Unlabeled 3134",
            "Unlabeled 3135",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 3176",
            "Unlabeled 3186",
            "Unlabeled 3193",
        ],
        "landmark2": [
            "Unlabeled 3177",
            "Unlabeled 3180",
        ],
        "landmark3": [
            "Unlabeled 3178",
            "Unlabeled 3183",
            "Unlabeled 3184",
            "Unlabeled 3185",
            "Unlabeled 3190",
            "Unlabeled 3192",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 3251",
            "Unlabeled 3256",
        ],
        "landmark2": [
            "Unlabeled 3252",
            "Unlabeled 3254",
            "Unlabeled 3257",
            "Unlabeled 3258",
            "Unlabeled 3260",
        ],
        "landmark3": [
            "Unlabeled 3253",
            "Unlabeled 3259",
            "Unlabeled 3261",
            "Unlabeled 3266",
            "Unlabeled 3267",
            "Unlabeled 3268",
        ],
    },
}

FOCUS = {
    "take_1": [2],
    "take_2": [1],
    "take_3": [1],
}

MIN_GAP_S = 2.0
MIN_CANDIDATE_FRAMES = 300
TOP_N = 12


def read_motive_long(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]
    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return long_df


def get_missing_gaps(clean, lm):
    avail_col = f"landmark{lm}_available"
    time = clean["time_s"].values
    available = clean[avail_col].fillna(False).astype(bool).values

    gaps = []
    in_gap = False
    start_idx = None

    for i, ok in enumerate(available):
        if not ok and not in_gap:
            in_gap = True
            start_idx = i

        elif ok and in_gap:
            end_idx = i - 1
            s = time[start_idx]
            e = time[end_idx]

            if e - s >= MIN_GAP_S:
                gaps.append({
                    "lm": lm,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": s,
                    "end_time": e,
                    "duration_s": e - s,
                })

            in_gap = False

    if in_gap:
        end_idx = len(available) - 1
        s = time[start_idx]
        e = time[end_idx]

        if e - s >= MIN_GAP_S:
            gaps.append({
                "lm": lm,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": s,
                "end_time": e,
                "duration_s": e - s,
            })

    return gaps


def nearest_valid_position(clean, lm, idx, direction):
    cols = [f"landmark{lm}_x", f"landmark{lm}_y", f"landmark{lm}_z"]

    if direction == "before":
        search = range(idx, -1, -1)
    else:
        search = range(idx, len(clean))

    for j in search:
        vals = clean.loc[j, cols].values.astype(float)
        if np.isfinite(vals).all():
            return vals

    return None


def find_candidates_for_gap(long_df, clean, gap, used_markers):
    s = gap["start_time"]
    e = gap["end_time"]
    lm = gap["lm"]

    before_pos = nearest_valid_position(clean, lm, gap["start_idx"], "before")
    after_pos = nearest_valid_position(clean, lm, gap["end_idx"], "after")

    gap_df = long_df[
        (long_df["time_s"] >= s) &
        (long_df["time_s"] <= e) &
        (~long_df["marker_name"].isin(used_markers))
    ].copy()

    rows = []

    for marker, sub in gap_df.groupby("marker_name"):
        if len(sub) < MIN_CANDIDATE_FRAMES:
            continue

        sub_sorted = sub.sort_values("time_s")

        start_pos = sub_sorted[["x", "y", "z"]].iloc[0].values.astype(float)
        end_pos = sub_sorted[["x", "y", "z"]].iloc[-1].values.astype(float)
        mean_pos = sub_sorted[["x", "y", "z"]].mean().values.astype(float)

        dist_before = np.nan
        dist_after = np.nan

        if before_pos is not None:
            dist_before = np.linalg.norm(start_pos - before_pos)

        if after_pos is not None:
            dist_after = np.linalg.norm(end_pos - after_pos)

        score_parts = []
        if np.isfinite(dist_before):
            score_parts.append(dist_before)
        if np.isfinite(dist_after):
            score_parts.append(dist_after)

        continuity_score = np.mean(score_parts) if score_parts else np.nan

        rows.append({
            "landmark": lm,
            "gap_start": s,
            "gap_end": e,
            "gap_duration_s": e - s,
            "candidate_marker": marker,
            "candidate_frames_in_gap": len(sub),
            "candidate_start_time": sub["time_s"].min(),
            "candidate_end_time": sub["time_s"].max(),
            "dist_to_before": dist_before,
            "dist_to_after": dist_after,
            "continuity_score": continuity_score,
            "mean_x": mean_pos[0],
            "mean_y": mean_pos[1],
            "mean_z": mean_pos[2],
        })

    if len(rows) == 0:
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    cand = cand.sort_values(
        ["candidate_frames_in_gap", "continuity_score"],
        ascending=[False, True]
    )

    return cand


all_suggestions = []

for take_name, raw_path in RAW_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print("="*90)

    clean_path = os.path.join(
        CLEAN_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    clean = pd.read_csv(clean_path, low_memory=False)
    long_df = read_motive_long(raw_path)

    used_markers = []
    for lm_name, markers in CURRENT_CHAINS[take_name].items():
        used_markers.extend(markers)

    print("Currently used markers:")
    print(used_markers)

    for lm in FOCUS[take_name]:
        gaps = get_missing_gaps(clean, lm)
        gaps = sorted(gaps, key=lambda g: g["duration_s"], reverse=True)

        print(f"\nLandmark {lm} has {len(gaps)} gaps >= {MIN_GAP_S}s")

        for gap_i, gap in enumerate(gaps[:10]):
            print(
                f"  Gap {gap_i}: "
                f"{gap['start_time']:.2f}s -> {gap['end_time']:.2f}s "
                f"dur={gap['duration_s']:.2f}s"
            )

            cand = find_candidates_for_gap(long_df, clean, gap, used_markers)

            if len(cand) > 0:
                print("  Top candidate markers:")
                display(cand.head(TOP_N))

                all_suggestions.append(cand.head(TOP_N))

                plt.figure(figsize=(10, 8))

                window = clean[
                    (clean["time_s"] >= gap["start_time"] - 20) &
                    (clean["time_s"] <= gap["end_time"] + 20)
                ]

                plt.plot(
                    window[f"landmark{lm}_x"],
                    window[f"landmark{lm}_z"],
                    linewidth=2,
                    color="black",
                    label=f"Current Landmark {lm}"
                )

                for marker in cand.head(6)["candidate_marker"]:
                    sub = long_df[
                        (long_df["marker_name"] == marker) &
                        (long_df["time_s"] >= gap["start_time"]) &
                        (long_df["time_s"] <= gap["end_time"])
                    ]
                    plt.plot(sub["x"], sub["z"], linewidth=1.2, label=marker)

                plt.title(
                    f"Group {GROUP} {take_name} Landmark {lm} gap {gap_i}: "
                    f"{gap['start_time']:.1f}-{gap['end_time']:.1f}s"
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()

if len(all_suggestions) > 0:
    suggestions = pd.concat(all_suggestions, ignore_index=True)
    suggestions_path = os.path.join(OUT_DIR, f"group_{GROUP}_focused_gap_candidate_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    print("\nSaved focused suggestions:")
    print(suggestions_path)

    print("\nMost promising candidates overall:")
    display(
        suggestions
        .sort_values(["landmark", "candidate_frames_in_gap"], ascending=[True, False])
        .head(120)
    )
else:
    print("No candidates found.")


# --- CELL 48 (code cell #42) ---
# ============================================================
# OPTITRACK GROUP 10 - MANUAL TRACKLET STITCHING UPDATED
# Builds clean Landmark1/2/3 from selected Unlabeled marker chains.
#
# Updates after focused inspection:
# - Take 1 L2: added 3115, 3119, 3123, 3124, 3129
# - Take 2 L1: added 3187, 3188 cautiously
# - Take 3 L1 unchanged because no safe candidate was found
# ============================================================

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

TAKE_PATHS = {
    "take_1": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_1.csv",
    "take_2": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_2.csv",
    "take_3": "/content/drive/MyDrive/thesis/data/group_10/optitrack/Arda_Group_10_Take_3.csv",
}

OUT_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_cleaned_manual_stitched"
os.makedirs(OUT_DIR, exist_ok=True)

CHAINS = {
    "take_1": {
        "landmark1": [
            "Unlabeled 3113",
            "Unlabeled 3122",
            "Unlabeled 3128",
        ],

        "landmark2": [
            "Unlabeled 3115",
            "Unlabeled 3119",
            "Unlabeled 3120",
            "Unlabeled 3123",
            "Unlabeled 3124",
            "Unlabeled 3125",
            "Unlabeled 3126",
            "Unlabeled 3127",
            "Unlabeled 3129",
            "Unlabeled 3130",
            "Unlabeled 3132",
            "Unlabeled 3136",
        ],

        "landmark3": [
            "Unlabeled 3118",
            "Unlabeled 3131",
            "Unlabeled 3133",
            "Unlabeled 3134",
            "Unlabeled 3135",
        ],
    },

    "take_2": {
        "landmark1": [
            "Unlabeled 3176",
            "Unlabeled 3186",
            "Unlabeled 3187",
            "Unlabeled 3188",
            "Unlabeled 3193",
        ],

        "landmark2": [
            "Unlabeled 3177",
            "Unlabeled 3180",
        ],

        "landmark3": [
            "Unlabeled 3178",
            "Unlabeled 3183",
            "Unlabeled 3184",
            "Unlabeled 3185",
            "Unlabeled 3190",
            "Unlabeled 3192",
        ],
    },

    "take_3": {
        "landmark1": [
            "Unlabeled 3251",
            "Unlabeled 3256",
        ],

        "landmark2": [
            "Unlabeled 3252",
            "Unlabeled 3254",
            "Unlabeled 3257",
            "Unlabeled 3258",
            "Unlabeled 3260",
        ],

        "landmark3": [
            "Unlabeled 3253",
            "Unlabeled 3259",
            "Unlabeled 3261",
            "Unlabeled 3266",
            "Unlabeled 3267",
            "Unlabeled 3268",
        ],
    },
}


def read_motive_long_and_frame_time(path):
    header_rows = []

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        for _ in range(7):
            header_rows.append(next(reader))

    meta_row = header_rows[0]
    names_row = header_rows[3]
    ids_row = header_rows[4]
    axis_row = header_rows[6]

    meta = {}
    for i in range(0, len(meta_row) - 1, 2):
        if meta_row[i]:
            meta[meta_row[i]] = meta_row[i + 1]

    ncols = len(axis_row)

    marker_infos = []
    for c in range(2, ncols, 3):
        if c + 2 < ncols:
            marker_infos.append({
                "start_col": c,
                "marker_name": names_row[c] if c < len(names_row) else "",
                "marker_id": ids_row[c] if c < len(ids_row) else "",
            })

    raw = pd.read_csv(path, header=None, skiprows=7, low_memory=False)
    raw = raw.iloc[:, :ncols]

    frame = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    time_s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

    frame_time = pd.DataFrame({
        "frame": frame.astype("Int64"),
        "time_s": time_s
    }).dropna().copy()

    frame_time["frame"] = frame_time["frame"].astype(int)

    parts = []

    for info in marker_infos:
        c = info["start_col"]

        xyz = raw.iloc[:, [c, c + 1, c + 2]].apply(pd.to_numeric, errors="coerce")
        xyz.columns = ["x", "y", "z"]

        present = xyz.notna().all(axis=1)

        if present.sum() == 0:
            continue

        sub = pd.DataFrame({
            "frame": frame[present].values.astype(int),
            "time_s": time_s[present].values,
            "marker_name": info["marker_name"],
            "marker_id": info["marker_id"],
            "x": xyz.loc[present, "x"].values,
            "y": xyz.loc[present, "y"].values,
            "z": xyz.loc[present, "z"].values,
        })

        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["frame", "time_s", "x", "y", "z"]).copy()
    long_df["frame"] = long_df["frame"].astype(int)

    return meta, frame_time, long_df


def build_stitched_clean(frame_time, long_df, chains):
    clean = frame_time.copy().sort_values("frame").reset_index(drop=True)

    for landmark_name, marker_list in chains.items():
        lm_df = long_df[long_df["marker_name"].isin(marker_list)].copy()

        lm_by_frame = (
            lm_df
            .groupby("frame", as_index=False)
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                source_markers=("marker_name", lambda x: "+".join(sorted(set(x))))
            )
        )

        clean = clean.merge(lm_by_frame, on="frame", how="left")

        clean = clean.rename(columns={
            "x": f"{landmark_name}_x",
            "y": f"{landmark_name}_y",
            "z": f"{landmark_name}_z",
            "source_markers": f"{landmark_name}_source"
        })

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        clean[f"landmark{i}_available"] = clean[cols].notna().all(axis=1)

    clean["active_clean_landmarks"] = clean[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return clean


def smooth_short_gaps(clean, limit=10, window=5):
    out = clean.copy()

    for i in [1, 2, 3]:
        for axis in ["x", "y", "z"]:
            col = f"landmark{i}_{axis}"
            out[col] = (
                out[col]
                .interpolate(limit=limit, limit_direction="both")
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

    for i in [1, 2, 3]:
        cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
        out[f"landmark{i}_available"] = out[cols].notna().all(axis=1)

    out["active_clean_landmarks"] = out[
        ["landmark1_available", "landmark2_available", "landmark3_available"]
    ].sum(axis=1)

    return out


def plot_stitched(clean, take_name):
    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_z"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Z")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in [1, 2, 3]:
        plt.plot(
            clean[f"landmark{i}_x"],
            clean[f"landmark{i}_y"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    plt.title(f"Group {GROUP} {take_name} manually stitched landmarks X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(clean["time_s"], clean["active_clean_landmarks"], linewidth=0.8)
    plt.axhline(3, linestyle="--", label="Expected 3")
    plt.title(f"Group {GROUP} {take_name} manually stitched active landmarks")
    plt.xlabel("Time (s)")
    plt.ylabel("Active landmarks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                clean["time_s"],
                clean[f"landmark{i}_{axis}"],
                linewidth=0.8,
                label=f"Landmark {i}"
            )

        ax.set_title(f"Group {GROUP} {take_name} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN STITCHING PER TAKE
# ============================================================

for take_name, path in TAKE_PATHS.items():
    print("\n" + "="*90)
    print(f"GROUP {GROUP} {take_name}")
    print(path)
    print("="*90)

    meta, frame_time, long_df = read_motive_long_and_frame_time(path)

    print("Take name:", meta.get("Take Name"))
    print("Capture start:", meta.get("Capture Start Time"))
    print("Total frames:", len(frame_time))
    print("Long detections:", long_df.shape)

    clean = build_stitched_clean(frame_time, long_df, CHAINS[take_name])
    smooth = smooth_short_gaps(clean)

    raw_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_raw.csv"
    )

    smooth_path = os.path.join(
        OUT_DIR,
        f"group_{GROUP}_optitrack_{take_name}_manual_stitched_smoothed.csv"
    )

    clean.to_csv(raw_path, index=False)
    smooth.to_csv(smooth_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(smooth_path)

    print("\nAvailability raw:")
    for i in [1, 2, 3]:
        print(f"Landmark {i}: {clean[f'landmark{i}_available'].mean()*100:.2f}%")

    print("\nActive clean landmark count raw:")
    print(clean["active_clean_landmarks"].value_counts().sort_index())

    print("\nSource markers used:")
    for i in [1, 2, 3]:
        col = f"landmark{i}_source"
        print(f"Landmark {i}:", clean[col].dropna().unique())

    plot_stitched(smooth, take_name)

print("\nOutput folder:")
print(OUT_DIR)


# --- CELL 49 (code cell #43) ---
# ============================================================
# OPTITRACK GROUP 10 - COMBINE TAKE 1–3
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 10 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

IN_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_10_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_10_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_10_optitrack_take_3_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-30 17:43:36.632
# Take 2: 2026-04-30 17:53:46.260
# Take 3: 2026-04-30 18:04:00.673
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 609.628,
    "take_3": 1224.041,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Quality notes
combined["optitrack_quality_note"] = ""

combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 Landmark 1 was improved using short candidates; medium confidence around the stitched gap."
)

combined.loc[combined["take"] == "take_3", "optitrack_quality_note"] = (
    "Take 3 has weak Landmark 1 availability; no safe continuation candidate was found."
)

out_path = f"{OUT_DIR}/group_10_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 10 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 10 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 10 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 10 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 10 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 50 (code cell #44) ---
# ============================================================
# OPTITRACK GROUP 10 - COMBINE TAKE 1–3
# Uses real capture-start offsets.
# Input: manually stitched smoothed files.
# Output: one continuous Group 10 OptiTrack file.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 10

IN_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_cleaned_manual_stitched"
OUT_DIR = "/content/drive/MyDrive/thesis/data/group_10/optitrack_final"
os.makedirs(OUT_DIR, exist_ok=True)

TAKE_PATHS = {
    "take_1": f"{IN_DIR}/group_10_optitrack_take_1_manual_stitched_smoothed.csv",
    "take_2": f"{IN_DIR}/group_10_optitrack_take_2_manual_stitched_smoothed.csv",
    "take_3": f"{IN_DIR}/group_10_optitrack_take_3_manual_stitched_smoothed.csv",
}

# Real offsets from Take 1 capture start:
# Take 1: 2026-04-30 17:43:36.632
# Take 2: 2026-04-30 17:53:46.260
# Take 3: 2026-04-30 18:04:00.673
TAKE_OFFSETS_S = {
    "take_1": 0.000,
    "take_2": 609.628,
    "take_3": 1224.041,
}

parts = []

for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False).copy()

    df["take"] = take_name
    df["time_s_original"] = df["time_s"]
    df["time_s"] = df["time_s_original"] + TAKE_OFFSETS_S[take_name]

    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("time_s").reset_index(drop=True)

# Recompute availability after combining
for i in [1, 2, 3]:
    cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
    combined[f"landmark{i}_available"] = combined[cols].notna().all(axis=1)

combined["active_clean_landmarks"] = combined[
    ["landmark1_available", "landmark2_available", "landmark3_available"]
].sum(axis=1)

# Quality notes
combined["optitrack_quality_note"] = ""

combined.loc[combined["take"] == "take_2", "optitrack_quality_note"] = (
    "Take 2 Landmark 1 was improved using short candidates; medium confidence around the stitched gap."
)

combined.loc[combined["take"] == "take_3", "optitrack_quality_note"] = (
    "Take 3 has weak Landmark 1 availability; no safe continuation candidate was found."
)

out_path = f"{OUT_DIR}/group_10_optitrack_cleaned_combined_240hz.csv"
combined.to_csv(out_path, index=False)

print("="*80)
print("GROUP 10 OPTITRACK COMBINED")
print("="*80)

print("Rows per take:")
for take_name, path in TAKE_PATHS.items():
    df = pd.read_csv(path, low_memory=False)
    print(take_name, len(df), "rows")

print("\nCombined rows:", len(combined))

print("\nTime range:")
print("start:", combined["time_s"].min())
print("end  :", combined["time_s"].max())
print("duration min:", (combined["time_s"].max() - combined["time_s"].min()) / 60)

print("\nTake boundaries and gaps:")
previous_end = None

for take_name, path in TAKE_PATHS.items():
    sub = combined[combined["take"] == take_name]

    start = sub["time_s"].min()
    end = sub["time_s"].max()

    print(f"{take_name}: {start:.3f}s -> {end:.3f}s")

    if previous_end is not None:
        print(f"  gap from previous take: {start - previous_end:.3f}s")

    previous_end = end

print("\nAvailability combined:")
for i in [1, 2, 3]:
    print(f"Landmark {i}: {combined[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nAvailability by take:")
for take_name in TAKE_PATHS.keys():
    sub = combined[combined["take"] == take_name]
    print(f"\n{take_name}:")
    for i in [1, 2, 3]:
        print(f"  Landmark {i}: {sub[f'landmark{i}_available'].mean()*100:.2f}%")

print("\nActive clean landmark count:")
print(combined["active_clean_landmarks"].value_counts().sort_index())

print("\nSaved combined file:")
print(out_path)

# ============================================================
# PLOTS
# ============================================================

take_start_times = {
    take: combined[combined["take"] == take]["time_s"].min()
    for take in TAKE_PATHS.keys()
}

plt.figure(figsize=(18, 4))
plt.plot(combined["time_s"], combined["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")

for take_name, t_start in take_start_times.items():
    plt.axvline(t_start, linestyle="--", alpha=0.7)
    plt.text(t_start, 3.1, take_name, rotation=90, va="bottom", fontsize=8)

plt.title("Group 10 OptiTrack combined - active landmarks")
plt.xlabel("Continuous time (s)")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_z"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 10 OptiTrack combined - X-Z trajectories")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
for i in [1, 2, 3]:
    plt.plot(
        combined[f"landmark{i}_x"],
        combined[f"landmark{i}_y"],
        linewidth=0.8,
        label=f"Landmark {i}"
    )

plt.title("Group 10 OptiTrack combined - X-Y trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            combined["time_s"],
            combined[f"landmark{i}_{axis}"],
            linewidth=0.8,
            label=f"Landmark {i}"
        )

    for take_name, t_start in take_start_times.items():
        ax.axvline(t_start, linestyle="--", alpha=0.7)

    ax.set_title(f"Group 10 OptiTrack combined - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("Continuous time (s)")
plt.tight_layout()
plt.show()


# --- CELL 52 (code cell #45) ---
# ============================================================
# OPTITRACK FINAL SUMMARY TABLE - ALL GROUPS
# Creates one clean overview table for supervisor presentation.
#
# Output:
# /content/drive/MyDrive/thesis/data/optitrack_final_summary/optitrack_all_groups_summary.csv
# ============================================================

import os
import glob
import pandas as pd
import numpy as np

BASE = "/content/drive/MyDrive/thesis/data"
OUT_DIR = f"{BASE}/optitrack_final_summary"
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]

FINAL_PATHS = {
    group: f"{BASE}/group_{group}/optitrack/optitrack_final/group_{group}_optitrack_cleaned_combined_240hz.csv"
    for group in GROUPS
}

summary_rows = []
take_rows = []

for group, path in FINAL_PATHS.items():
    print("\n" + "="*80)
    print(f"GROUP {group}")
    print(path)
    print("="*80)

    if not os.path.exists(path):
        print("Missing file, skipped.")
        continue

    df = pd.read_csv(path, low_memory=False)

    # Basic info
    n_rows = len(df)
    start_s = df["time_s"].min()
    end_s = df["time_s"].max()
    duration_s = end_s - start_s
    duration_min = duration_s / 60

    # Take info
    if "take" in df.columns:
        takes = list(df["take"].dropna().unique())
        n_takes = len(takes)
    else:
        takes = []
        n_takes = np.nan

    # Availability
    availability = {}

    for i in [1, 2, 3]:
        avail_col = f"landmark{i}_available"

        if avail_col in df.columns:
            availability[f"L{i}_availability_pct"] = df[avail_col].mean() * 100
        else:
            xyz_cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
            availability[f"L{i}_availability_pct"] = df[xyz_cols].notna().all(axis=1).mean() * 100

    # Active landmark count
    if "active_clean_landmarks" not in df.columns:
        for i in [1, 2, 3]:
            xyz_cols = [f"landmark{i}_x", f"landmark{i}_y", f"landmark{i}_z"]
            df[f"landmark{i}_available"] = df[xyz_cols].notna().all(axis=1)

        df["active_clean_landmarks"] = df[
            ["landmark1_available", "landmark2_available", "landmark3_available"]
        ].sum(axis=1)

    active_counts = df["active_clean_landmarks"].value_counts().sort_index()

    active_0 = int(active_counts.get(0, 0))
    active_1 = int(active_counts.get(1, 0))
    active_2 = int(active_counts.get(2, 0))
    active_3 = int(active_counts.get(3, 0))

    active_3_pct = active_3 / n_rows * 100
    active_2_or_3_pct = (active_2 + active_3) / n_rows * 100

    # Take boundaries and gaps
    max_gap_s = np.nan
    total_gap_s = 0
    gap_list = []

    if "take" in df.columns:
        previous_end = None

        for take in takes:
            sub = df[df["take"] == take]

            take_start = sub["time_s"].min()
            take_end = sub["time_s"].max()
            take_duration_s = take_end - take_start

            gap_from_previous = np.nan

            if previous_end is not None:
                gap_from_previous = take_start - previous_end
                gap_list.append(gap_from_previous)

            previous_end = take_end

            take_rows.append({
                "group": group,
                "take": take,
                "rows": len(sub),
                "start_s": take_start,
                "end_s": take_end,
                "duration_s": take_duration_s,
                "duration_min": take_duration_s / 60,
                "gap_from_previous_s": gap_from_previous,
                "L1_availability_pct": sub["landmark1_available"].mean() * 100,
                "L2_availability_pct": sub["landmark2_available"].mean() * 100,
                "L3_availability_pct": sub["landmark3_available"].mean() * 100,
                "active_3_pct": (sub["active_clean_landmarks"] == 3).mean() * 100,
            })

        if len(gap_list) > 0:
            max_gap_s = max(gap_list)
            total_gap_s = sum(gap_list)

    # Automatic quality note
    weak_notes = []

    for i in [1, 2, 3]:
        value = availability[f"L{i}_availability_pct"]

        if value < 60:
            weak_notes.append(f"L{i} weak ({value:.1f}%)")
        elif value < 80:
            weak_notes.append(f"L{i} moderate ({value:.1f}%)")

    if max_gap_s is not np.nan and pd.notna(max_gap_s) and max_gap_s > 60:
        weak_notes.append(f"large take gap ({max_gap_s:.1f}s)")

    if len(weak_notes) == 0:
        quality_level = "Good"
        quality_note = "All landmarks have strong availability."
    elif any("weak" in note for note in weak_notes):
        quality_level = "Medium / lower confidence"
        quality_note = "; ".join(weak_notes)
    else:
        quality_level = "Medium"
        quality_note = "; ".join(weak_notes)

    # Existing notes from file, if any
    existing_quality_notes = ""

    if "optitrack_quality_note" in df.columns:
        notes = (
            df["optitrack_quality_note"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        existing_quality_notes = " | ".join(notes)

    summary_rows.append({
        "group": group,
        "file": path,
        "rows": n_rows,
        "duration_s": duration_s,
        "duration_min": duration_min,
        "n_takes": n_takes,
        "start_s": start_s,
        "end_s": end_s,
        "max_gap_between_takes_s": max_gap_s,
        "total_gap_between_takes_s": total_gap_s,
        "L1_availability_pct": availability["L1_availability_pct"],
        "L2_availability_pct": availability["L2_availability_pct"],
        "L3_availability_pct": availability["L3_availability_pct"],
        "active_0_rows": active_0,
        "active_1_rows": active_1,
        "active_2_rows": active_2,
        "active_3_rows": active_3,
        "active_3_pct": active_3_pct,
        "active_2_or_3_pct": active_2_or_3_pct,
        "quality_level": quality_level,
        "auto_quality_note": quality_note,
        "file_quality_notes": existing_quality_notes,
    })

summary = pd.DataFrame(summary_rows)
take_summary = pd.DataFrame(take_rows)

# Round numeric columns for readability
numeric_cols = summary.select_dtypes(include=[np.number]).columns
summary[numeric_cols] = summary[numeric_cols].round(2)

take_numeric_cols = take_summary.select_dtypes(include=[np.number]).columns
take_summary[take_numeric_cols] = take_summary[take_numeric_cols].round(2)

summary_path = f"{OUT_DIR}/optitrack_all_groups_summary.csv"
take_summary_path = f"{OUT_DIR}/optitrack_all_groups_take_level_summary.csv"

summary.to_csv(summary_path, index=False)
take_summary.to_csv(take_summary_path, index=False)

print("\n" + "="*80)
print("OPTITRACK ALL GROUPS SUMMARY")
print("="*80)

display(summary)

print("\nSaved group-level summary:")
print(summary_path)

print("\nSaved take-level summary:")
print(take_summary_path)

print("\n" + "="*80)
print("TAKE-LEVEL SUMMARY")
print("="*80)

display(take_summary)


# --- CELL 53 (code cell #46) ---
# ============================================================
# OPTITRACK SUPERVISOR VISUALIZATION
# Creates clear plots from the summary CSV:
# 1) Landmark availability per group
# 2) Active 3 landmarks percentage per group
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_PATH = "/content/drive/MyDrive/thesis/data/optitrack_final_summary/optitrack_all_groups_summary.csv"
OUT_DIR = "/content/drive/MyDrive/thesis/data/optitrack_final_summary/plots"
os.makedirs(OUT_DIR, exist_ok=True)

summary = pd.read_csv(SUMMARY_PATH)

# ------------------------------------------------------------
# 1) Landmark availability heatmap
# ------------------------------------------------------------

groups = summary["group"].astype(str).tolist()
availability = summary[
    ["L1_availability_pct", "L2_availability_pct", "L3_availability_pct"]
].values

fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(availability, aspect="auto", vmin=0, vmax=100)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Landmark 1", "Landmark 2", "Landmark 3"])
ax.set_yticks(np.arange(len(groups)))
ax.set_yticklabels([f"Group {g}" for g in groups])

for i in range(availability.shape[0]):
    for j in range(availability.shape[1]):
        ax.text(
            j,
            i,
            f"{availability[i, j]:.1f}%",
            ha="center",
            va="center",
            fontsize=9
        )

ax.set_title("OptiTrack landmark availability by group")
fig.colorbar(im, ax=ax, label="Availability (%)")
plt.tight_layout()

heatmap_path = os.path.join(OUT_DIR, "optitrack_landmark_availability_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", heatmap_path)

# ------------------------------------------------------------
# 2) Active 3 landmarks percentage
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(summary["group"].astype(str), summary["active_3_pct"])

plt.axhline(80, linestyle="--", label="80% reference")
plt.title("Percentage of frames with all 3 OptiTrack landmarks available")
plt.xlabel("Group")
plt.ylabel("Frames with 3 landmarks available (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

bar_path = os.path.join(OUT_DIR, "optitrack_active_3_landmarks_by_group.png")
plt.savefig(bar_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", bar_path)

# ------------------------------------------------------------
# 3) Compact supervisor table
# ------------------------------------------------------------

supervisor_table = summary[
    [
        "group",
        "duration_min",
        "n_takes",
        "L1_availability_pct",
        "L2_availability_pct",
        "L3_availability_pct",
        "active_3_pct",
        "quality_level",
        "auto_quality_note",
    ]
].copy()

supervisor_table = supervisor_table.round(2)

display(supervisor_table)

supervisor_table_path = "/content/drive/MyDrive/thesis/data/optitrack_final_summary/optitrack_supervisor_table.csv"
supervisor_table.to_csv(supervisor_table_path, index=False)

print("Saved supervisor table:")
print(supervisor_table_path)


# --- CELL 55 (code cell #47) ---
# ============================================================
# OPTITRACK LABELING FROM ELAN - ONE GROUP
#
# What this cell does:
# 1. Loads cleaned combined OptiTrack data.
# 2. Loads ELAN annotation CSV.
# 3. Finds ELAN synchronization_move interval.
# 4. Aligns OptiTrack time to ELAN/video time using sync move.
# 5. Transfers ELAN labels to OptiTrack rows.
# 6. Saves labeled OptiTrack CSV + label summary.
# ============================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

GROUP = 10

OPTITRACK_PATH = (
    f"/content/drive/MyDrive/thesis/data/group_{GROUP}/optitrack/optitrack_final/"
    f"group_{GROUP}_optitrack_cleaned_combined_240hz.csv"
)

# Change this if your ELAN file is elsewhere in Drive.
# For the uploaded file in this chat / Colab-style environment, use:
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_10/elan/Group_10_individual_build_renamed.csv"

OUT_DIR = f"/content/drive/MyDrive/thesis/data/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# If you visually identified OptiTrack sync time, write it here.
# For Group 10, from the whole-recording plot, this is very likely 35.95s.
OPTITRACK_SYNC_TIME_MANUAL = 35.95

# If you want automatic selection instead, set this to None:
# OPTITRACK_SYNC_TIME_MANUAL = None

# How much around sync to plot for confirmation
SYNC_PLOT_WINDOW_S = 15

# Label columns to create
EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    """
    Normalizes common typo variants but keeps labels mostly as they are.
    Add more replacements if needed.
    """
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
    }

    # Handle combined labels too
    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]
    label = " + ".join(parts)

    return label


def load_elan_annotations(elan_path):
    """
    Expected ELAN export format:
    col 0 = tier / participant
    col 3 = start seconds
    col 5 = end seconds
    col 8 = label
    """
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_elan_sync_interval(elan):
    sync_mask = elan["label"].str.lower().str.contains(
        "sync|synchron", regex=True, na=False
    )

    sync_rows = elan[sync_mask].copy()

    if len(sync_rows) == 0:
        raise ValueError(
            "No synchronization label found in ELAN file. "
            "Check label names manually."
        )

    # Prefer Whole_Group if available, otherwise use earliest sync row
    whole = sync_rows[sync_rows["tier_clean"].str.lower() == "whole_group"]

    if len(whole) > 0:
        chosen = whole.sort_values("start_s").iloc[0]
    else:
        chosen = sync_rows.sort_values("start_s").iloc[0]

    return chosen, sync_rows


def compute_optitrack_movement(df):
    """
    Computes per-landmark and synchronized movement scores from XYZ.
    """
    out = df.copy()

    for i in [1, 2, 3]:
        dx = out[f"landmark{i}_x"].diff()
        dy = out[f"landmark{i}_y"].diff()
        dz = out[f"landmark{i}_z"].diff()

        out[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

        out[f"landmark{i}_movement_smooth"] = (
            out[f"landmark{i}_movement"]
            .rolling(120, center=True, min_periods=1)   # ~0.5 sec at 240 Hz
            .mean()
        )

    movement_cols = [
        "landmark1_movement_smooth",
        "landmark2_movement_smooth",
        "landmark3_movement_smooth",
    ]

    available_movement_count = out[movement_cols].notna().sum(axis=1)

    out["sync_movement_score_tolerant"] = out[movement_cols].mean(axis=1, skipna=True)
    out.loc[available_movement_count < 2, "sync_movement_score_tolerant"] = np.nan

    out["sync_movement_score_strict"] = out[movement_cols].min(axis=1, skipna=False)

    out["sync_movement_score_tolerant_smooth"] = (
        out["sync_movement_score_tolerant"]
        .rolling(240, center=True, min_periods=1)   # ~1 sec
        .mean()
    )

    out["sync_movement_score_strict_smooth"] = (
        out["sync_movement_score_strict"]
        .rolling(240, center=True, min_periods=1)
        .mean()
    )

    return out


def auto_detect_optitrack_sync(df_movement):
    """
    Automatically proposes OptiTrack sync peak.
    It ignores large peaks exactly at take boundaries.
    This is only a helper; visual/manual confirmation is better.
    """
    cand = df_movement[
        df_movement["sync_movement_score_tolerant_smooth"].notna()
    ].copy()

    # Remove small windows around take boundaries because they can create artificial jumps.
    if "take" in cand.columns:
        take_starts = cand.groupby("take")["time_s"].min().values

        for t in take_starts:
            cand = cand[
                ~((cand["time_s"] >= t - 2.0) & (cand["time_s"] <= t + 2.0))
            ]

    cand["time_bin_2s"] = (cand["time_s"] // 2).astype(int)

    peaks = (
        cand
        .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
        .groupby("time_bin_2s", as_index=False)
        .head(1)
        .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
        .head(20)
        .copy()
    )

    return peaks


def plot_sync_confirmation(df_movement, optitrack_sync_time, elan_sync_mid, shift):
    """
    Plots OptiTrack around detected sync time and shows computed alignment.
    """
    start = max(df_movement["time_s"].min(), optitrack_sync_time - SYNC_PLOT_WINDOW_S)
    end = min(df_movement["time_s"].max(), optitrack_sync_time + SYNC_PLOT_WINDOW_S)

    win = df_movement[
        (df_movement["time_s"] >= start) &
        (df_movement["time_s"] <= end)
    ].copy()

    fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.axvline(optitrack_sync_time, linestyle="--", linewidth=1.2)
        ax.set_title(
            f"Group {GROUP} OptiTrack sync at {optitrack_sync_time:.3f}s - {axis.upper()}"
        )
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.0,
            label=f"Landmark {i} movement"
        )

    axs[3].axvline(optitrack_sync_time, linestyle="--", linewidth=1.2)
    axs[3].set_title("Per-landmark movement around OptiTrack sync")
    axs[3].set_xlabel("OptiTrack continuous time_s")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.show()

    print("="*80)
    print("SYNC ALIGNMENT")
    print("="*80)
    print(f"ELAN sync midpoint       : {elan_sync_mid:.3f}s")
    print(f"OptiTrack sync time      : {optitrack_sync_time:.3f}s")
    print(f"SHIFT = Opti - ELAN      : {shift:.3f}s")
    print()
    print("Applied alignment:")
    print("video_time_s = time_s - SHIFT")


def assign_elan_labels_to_optitrack(optitrack, elan):
    """
    Creates one label column per tier:
    label_Participant1, label_Participant2, ...
    A row can receive multiple label columns.
    """
    out = optitrack.copy()

    # Create expected columns first
    for tier in EXPECTED_TIERS:
        out[f"label_{tier}"] = ""

    # Also create any extra tier columns found in ELAN
    for tier in sorted(elan["tier_clean"].unique()):
        col = f"label_{tier}"
        if col not in out.columns:
            out[col] = ""

    # Assign labels interval by interval
    for _, row in elan.iterrows():
        tier = row["tier_clean"]
        col = f"label_{tier}"

        start = row["start_s"]
        end = row["end_s"]
        label = row["label"]

        mask = (out["video_time_s"] >= start) & (out["video_time_s"] <= end)

        # If empty, just assign label
        # If already labeled, append with " + " if different
        existing = out.loc[mask, col].astype(str)

        empty_mask = mask & (out[col].astype(str).str.len() == 0)
        out.loc[empty_mask, col] = label

        non_empty_mask = mask & (out[col].astype(str).str.len() > 0)

        if non_empty_mask.any():
            current_vals = out.loc[non_empty_mask, col].astype(str)

            needs_append = ~current_vals.str.contains(
                re.escape(label), regex=True, na=False
            )

            idx_to_append = current_vals[needs_append].index
            out.loc[idx_to_append, col] = (
                out.loc[idx_to_append, col].astype(str) + " + " + label
            )

    return out


def make_labeling_summary(labeled, elan, shift, optitrack_sync_time, elan_sync_row):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        non_empty = labeled[col].fillna("").astype(str).str.len() > 0

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": non_empty.mean() * 100,
            "unique_labels": " | ".join(
                sorted([x for x in labeled.loc[non_empty, col].dropna().unique()])
            )
        })

    # Add technical summary rows separately as printed output too
    summary = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("LABELING SUMMARY")
    print("="*80)
    print("Rows:", len(labeled))
    print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
    print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
    print("ELAN range:", elan["start_s"].min(), "->", elan["end_s"].max())
    print("Shift:", shift)
    print("OptiTrack sync time:", optitrack_sync_time)
    print("ELAN sync label:", elan_sync_row["label"])
    print("ELAN sync tier:", elan_sync_row["tier"])
    print("ELAN sync interval:", elan_sync_row["start_s"], "->", elan_sync_row["end_s"])

    display(summary)

    return summary


# ============================================================
# RUN
# ============================================================

print("="*80)
print(f"GROUP {GROUP} OPTITRACK LABELING FROM ELAN")
print("="*80)

# Load files
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False)
elan = load_elan_annotations(ELAN_PATH)

print("\nLoaded OptiTrack:")
print("Rows:", len(optitrack))
print("Time:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("\nLoaded ELAN:")
print("Rows:", len(elan))
print("Tiers:", sorted(elan["tier_clean"].unique()))
print("Labels:", sorted(elan["label"].unique())[:50])

# Find ELAN sync
elan_sync_row, all_sync_rows = find_elan_sync_interval(elan)

print("\nELAN sync rows found:")
display(all_sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_SYNC_START = float(elan_sync_row["start_s"])
ELAN_SYNC_END = float(elan_sync_row["end_s"])
ELAN_SYNC_MID = (ELAN_SYNC_START + ELAN_SYNC_END) / 2

# Compute OptiTrack movement + sync candidate
optitrack_movement = compute_optitrack_movement(optitrack)

if OPTITRACK_SYNC_TIME_MANUAL is None:
    peaks = auto_detect_optitrack_sync(optitrack_movement)

    print("\nAutomatic OptiTrack sync candidates:")
    display(
        peaks[
            [
                "time_s",
                "take",
                "active_clean_landmarks",
                "landmark1_movement_smooth",
                "landmark2_movement_smooth",
                "landmark3_movement_smooth",
                "sync_movement_score_tolerant_smooth",
                "sync_movement_score_strict_smooth",
            ]
        ]
    )

    # Use first candidate, but manual confirmation is recommended
    OPTITRACK_SYNC_TIME = float(peaks.iloc[0]["time_s"])
else:
    OPTITRACK_SYNC_TIME = float(OPTITRACK_SYNC_TIME_MANUAL)

# Compute shift
SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID

# Plot confirmation
plot_sync_confirmation(
    optitrack_movement,
    optitrack_sync_time=OPTITRACK_SYNC_TIME,
    elan_sync_mid=ELAN_SYNC_MID,
    shift=SHIFT
)

# Add video time
optitrack_labeled_base = optitrack.copy()
optitrack_labeled_base["video_time_s"] = optitrack_labeled_base["time_s"] - SHIFT
optitrack_labeled_base["optitrack_to_elan_shift_s"] = SHIFT
optitrack_labeled_base["optitrack_sync_time_s"] = OPTITRACK_SYNC_TIME
optitrack_labeled_base["elan_sync_mid_s"] = ELAN_SYNC_MID

# Assign ELAN labels
labeled = assign_elan_labels_to_optitrack(optitrack_labeled_base, elan)

# Make summary
label_summary = make_labeling_summary(
    labeled=labeled,
    elan=elan,
    shift=SHIFT,
    optitrack_sync_time=OPTITRACK_SYNC_TIME,
    elan_sync_row=elan_sync_row
)

# Save
labeled.to_csv(LABELED_OUT_PATH, index=False)
label_summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\nSaved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 56 (code cell #48) ---
# ============================================================
# GROUP 10 - VALIDATE FIRST AND LAST SYNC ALIGNMENT
# ============================================================

first_elan_mid = (27.000 + 29.476) / 2
last_elan_mid = (1495.476 + 1498.476) / 2

first_opti_sync = 35.950
last_opti_sync = 1504.799333  # from candidate table, visually verify if needed

first_shift = first_opti_sync - first_elan_mid
last_shift = last_opti_sync - last_elan_mid

print("First ELAN sync midpoint:", first_elan_mid)
print("First OptiTrack sync:", first_opti_sync)
print("First shift:", first_shift)

print("\nLast ELAN sync midpoint:", last_elan_mid)
print("Last OptiTrack sync:", last_opti_sync)
print("Last shift:", last_shift)

print("\nShift difference:", last_shift - first_shift)
print("Absolute drift estimate:", abs(last_shift - first_shift), "seconds")


# --- CELL 57 (code cell #49) ---
# ============================================================
# GROUP 9 - WHOLE OPTITRACK SYNC MOVE VISUALIZATION
# Goal:
# Find the true synchronized sit-up movement in OptiTrack.
# Ignore peaks at take boundaries.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

PATH = f"/content/drive/MyDrive/thesis/data/group_{GROUP}/optitrack/optitrack_final/group_{GROUP}_optitrack_cleaned_combined_240hz.csv"

df = pd.read_csv(PATH, low_memory=False).copy()

print("="*80)
print(f"GROUP {GROUP} WHOLE OPTITRACK SYNC VISUALIZATION")
print("="*80)
print("Rows:", len(df))
print("Time range:", df["time_s"].min(), "->", df["time_s"].max())
print("Duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)

# ------------------------------------------------------------
# Compute movement score
# ------------------------------------------------------------

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth_05s"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth_05s",
    "landmark2_movement_smooth_05s",
    "landmark3_movement_smooth_05s",
]

df["combined_movement"] = df[movement_cols].mean(axis=1, skipna=True)

available_movement_count = df[movement_cols].notna().sum(axis=1)
df["sync_movement_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_movement_count < 2, "sync_movement_score_tolerant"] = np.nan

df["sync_movement_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

df["combined_movement_smooth"] = (
    df["combined_movement"]
    .rolling(240, center=True, min_periods=1)
    .mean()
)

df["sync_movement_score_tolerant_smooth"] = (
    df["sync_movement_score_tolerant"]
    .rolling(240, center=True, min_periods=1)
    .mean()
)

df["sync_movement_score_strict_smooth"] = (
    df["sync_movement_score_strict"]
    .rolling(240, center=True, min_periods=1)
    .mean()
)

# Downsample for readable plots
DOWNSAMPLE_STEP = 24
plot_df = df.iloc[::DOWNSAMPLE_STEP].copy()

# ------------------------------------------------------------
# Plot whole X/Y/Z
# ------------------------------------------------------------

fig, axs = plt.subplots(3, 1, figsize=(20, 11), sharex=True)

for ax_i, axis in enumerate(["x", "y", "z"]):
    ax = axs[ax_i]

    for i in [1, 2, 3]:
        ax.plot(
            plot_df["time_s"],
            plot_df[f"landmark{i}_{axis}"],
            linewidth=0.7,
            label=f"Landmark {i}"
        )

    ax.set_title(f"Group {GROUP} whole recording - {axis.upper()} over time")
    ax.set_ylabel(axis.upper())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axs[-1].set_xlabel("OptiTrack continuous time_s")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot per-landmark movement
# ------------------------------------------------------------

plt.figure(figsize=(20, 5))

for i in [1, 2, 3]:
    plt.plot(
        plot_df["time_s"],
        plot_df[f"landmark{i}_movement_smooth_05s"],
        linewidth=0.8,
        label=f"Landmark {i} movement"
    )

plt.title(f"Group {GROUP} whole recording - per-landmark movement score")
plt.xlabel("OptiTrack continuous time_s")
plt.ylabel("3D movement score, smoothed")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Plot synchronized movement candidates
# ------------------------------------------------------------

plt.figure(figsize=(20, 5))

plt.plot(
    plot_df["time_s"],
    plot_df["combined_movement_smooth"],
    linewidth=0.9,
    label="Mean movement, available landmarks"
)

plt.plot(
    plot_df["time_s"],
    plot_df["sync_movement_score_tolerant_smooth"],
    linewidth=1.2,
    label="Tolerant sync score, at least 2 landmarks"
)

plt.plot(
    plot_df["time_s"],
    plot_df["sync_movement_score_strict_smooth"],
    linewidth=1.2,
    label="Strict sync score, all 3 landmarks"
)

plt.title(f"Group {GROUP} whole recording - synchronized movement candidates")
plt.xlabel("OptiTrack continuous time_s")
plt.ylabel("Movement score")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Active landmarks
# ------------------------------------------------------------

plt.figure(figsize=(20, 3))
plt.plot(plot_df["time_s"], plot_df["active_clean_landmarks"], linewidth=0.8)
plt.axhline(3, linestyle="--", label="Expected 3")
plt.title(f"Group {GROUP} whole recording - active clean landmarks")
plt.xlabel("OptiTrack continuous time_s")
plt.ylabel("Active landmarks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Candidate peak detection
# ------------------------------------------------------------

candidate_df = df[
    df["sync_movement_score_tolerant_smooth"].notna()
].copy()

# Ignore +/- 2 seconds around take starts to avoid take-boundary artifacts
if "take" in candidate_df.columns:
    take_starts = candidate_df.groupby("take")["time_s"].min().values

    for t in take_starts:
        candidate_df = candidate_df[
            ~((candidate_df["time_s"] >= t - 2.0) & (candidate_df["time_s"] <= t + 2.0))
        ]

candidate_df["time_bin_2s"] = (candidate_df["time_s"] // 2).astype(int)

peaks = (
    candidate_df
    .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
    .groupby("time_bin_2s", as_index=False)
    .head(1)
    .sort_values("sync_movement_score_tolerant_smooth", ascending=False)
    .head(20)
    .copy()
)

peaks = peaks[
    [
        "time_s",
        "take",
        "active_clean_landmarks",
        "landmark1_movement_smooth_05s",
        "landmark2_movement_smooth_05s",
        "landmark3_movement_smooth_05s",
        "combined_movement_smooth",
        "sync_movement_score_tolerant_smooth",
        "sync_movement_score_strict_smooth",
    ]
].copy()

print("\n" + "="*80)
print("TOP 20 SYNCHRONIZED MOVEMENT CANDIDATES")
print("="*80)
display(peaks)

# ------------------------------------------------------------
# Plot zooms around top 8 candidate peaks
# ------------------------------------------------------------

TOP_N_ZOOMS = 8
ZOOM_HALF_WINDOW_S = 10

for idx, row in peaks.head(TOP_N_ZOOMS).iterrows():
    center = row["time_s"]
    start = max(df["time_s"].min(), center - ZOOM_HALF_WINDOW_S)
    end = min(df["time_s"].max(), center + ZOOM_HALF_WINDOW_S)

    zoom = df[(df["time_s"] >= start) & (df["time_s"] <= end)].copy()

    fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                zoom["time_s"],
                zoom[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.axvline(center, linestyle="--", linewidth=1.2)
        ax.set_title(
            f"Group {GROUP} candidate sync peak at {center:.2f}s - {axis.upper()}"
        )
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            zoom["time_s"],
            zoom[f"landmark{i}_movement_smooth_05s"],
            linewidth=1.0,
            label=f"Landmark {i} movement"
        )

    axs[3].axvline(center, linestyle="--", linewidth=1.2)
    axs[3].set_title("Per-landmark movement around candidate")
    axs[3].set_xlabel("OptiTrack continuous time_s")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.show()


# --- CELL 58 (code cell #50) ---
# ============================================================
# CHECK ELAN SYNCHRONIZATION ROWS - HEADERLESS ELAN EXPORT
# Works with the same ELAN format we used for Group 10.
# ============================================================

import pandas as pd

ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"  # <<< change if needed

elan_raw = pd.read_csv(ELAN_PATH, header=None, low_memory=False)

print("Raw ELAN shape:", elan_raw.shape)
print("First rows:")
display(elan_raw.head())

elan = pd.DataFrame({
    "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
    "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
    "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
    "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
    "label": elan_raw.iloc[:, 8].astype(str).str.strip(),
})

elan = elan.dropna(subset=["start_s", "end_s"]).copy()
elan = elan[elan["end_s"] > elan["start_s"]].copy()

# Normalize common typo variants
elan["label_normalized"] = (
    elan["label"]
    .str.replace("synchronizaiton_move", "synchronization_move", regex=False)
    .str.replace("synchronisation_move", "synchronization_move", regex=False)
)

sync_rows = elan[
    elan["label_normalized"].str.contains(
        "sync|synchron",
        case=False,
        regex=True,
        na=False
    )
].copy()

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)

display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label", "label_normalized"]])

print("\nELAN full time range:")
print(elan["start_s"].min(), "->", elan["end_s"].max())

print("\nUnique labels containing possible sync words:")
display(
    elan[elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)]["label"]
    .drop_duplicates()
    .sort_values()
)


# --- CELL 59 (code cell #51) ---
# ============================================================
# GROUP 9 - FOCUSED SYNC WINDOW INSPECTION
# Uses known ELAN sync windows:
# - Beginning sync: Whole_Group 12.180–15.300s
# - End sync: Participant1 3594.727–3598.364s
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 9

OPTITRACK_PATH = f"/content/drive/MyDrive/thesis/data/group_{GROUP}/optitrack/optitrack_final/group_{GROUP}_optitrack_cleaned_combined_240hz.csv"

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

# ELAN sync intervals
ELAN_FIRST_SYNC_START = 12.180
ELAN_FIRST_SYNC_END   = 15.300
ELAN_FIRST_SYNC_MID   = (ELAN_FIRST_SYNC_START + ELAN_FIRST_SYNC_END) / 2

ELAN_LAST_SYNC_START = 3594.727
ELAN_LAST_SYNC_END   = 3598.364
ELAN_LAST_SYNC_MID   = (ELAN_LAST_SYNC_START + ELAN_LAST_SYNC_END) / 2

print("="*80)
print(f"GROUP {GROUP} FOCUSED SYNC INSPECTION")
print("="*80)
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ------------------------------------------------------------
# Movement score per landmark
# ------------------------------------------------------------

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)  # ~0.5 sec
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)
df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------

def plot_sync_window(start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    # Per-landmark movement
    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    # Sync score
    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )
    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Candidate peaks in this window
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(10)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ------------------------------------------------------------
# Windows to inspect
# ------------------------------------------------------------

# Beginning: ELAN first sync is 12.18–15.30s.
# Search wider because OptiTrack may be shifted by several seconds.
plot_sync_window(
    start_s=0,
    end_s=45,
    title="Group 9 beginning sync search"
)

# End: ELAN last sync is around 3596.5s.
# Search wider near end.
plot_sync_window(
    start_s=3560,
    end_s=3705,
    title="Group 9 ending sync search"
)


# --- CELL 60 (code cell #52) ---
# ============================================================
# GROUP 9 - FIND CORRECT OPTITRACK AND ELAN FILES
# This prevents accidentally loading ELAN as OptiTrack.
# ============================================================

import os
import glob
import pandas as pd

GROUP = 9
BASE = "/content/drive/MyDrive/thesis/data"

# Search for cleaned combined OptiTrack file
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

print("="*80)
print("OPTITRACK CANDIDATES")
print("="*80)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        cols = set(test.columns)

        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }

        is_valid = required.issubset(cols)

        print("FOUND:", p)
        print("Valid OptiTrack:", is_valid)
        print("Columns sample:", list(test.columns)[:15])
        print()

        if is_valid:
            valid_opti.append(p)

    except Exception as e:
        print("Could not read:", p)
        print(e)

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

print("="*80)
print("SELECTED OPTITRACK_PATH")
print("="*80)
print(OPTITRACK_PATH)


# Search for ELAN file
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

print("\n" + "="*80)
print("ELAN CANDIDATES")
print("="*80)

for p in elan_candidates:
    print(p)

if len(elan_candidates) == 0:
    raise FileNotFoundError("No Group 9 ELAN file found.")

ELAN_PATH = elan_candidates[0]

print("\n" + "="*80)
print("SELECTED ELAN_PATH")
print("="*80)
print(ELAN_PATH)


# --- CELL 61 (code cell #53) ---
# ============================================================
# GROUP 9 - FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC
#
# Faster version:
# - No heavy plots
# - Uses np.searchsorted for interval labeling
# - Much faster for 880k+ rows
# ============================================================

import os
import re
import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

GROUP = 9
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_9/optitrack/optitrack_final/group_9_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_9/elan/Group_9_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 9 sync inspection
OPTITRACK_FIRST_SYNC_TIME = 23.141667
OPTITRACK_LAST_SYNC_TIME = 3606.117833

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_rows(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()

    first_sync = sync_rows.iloc[0]
    last_sync = sync_rows.iloc[-1]

    return first_sync, last_sync, sync_rows


def compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last):
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b


def fast_assign_labels(optitrack, elan):
    """
    Fast interval labeling.

    Assumes optitrack['video_time_s'] is sorted increasing.
    Uses searchsorted to find row ranges for each ELAN interval.
    """
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    # Create expected label columns
    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    # Work tier by tier
    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            # Find index range directly
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask
            if non_empty_mask.any():
                # Append only where label is not already present
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = (
            values[non_empty]
            .dropna()
            .unique()
            .tolist()
        )

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("="*80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING")
print("="*80)

# Safety checks
if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

# Load data
print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# Find ELAN sync rows
first_sync, last_sync, sync_rows = find_sync_rows(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

first_shift = OPTITRACK_FIRST_SYNC_TIME - ELAN_FIRST_SYNC_MID
last_shift = OPTITRACK_LAST_SYNC_TIME - ELAN_LAST_SYNC_MID
drift = last_shift - first_shift

a, b = compute_two_point_alignment(
    opti_first=OPTITRACK_FIRST_SYNC_TIME,
    elan_first=ELAN_FIRST_SYNC_MID,
    opti_last=OPTITRACK_LAST_SYNC_TIME,
    elan_last=ELAN_LAST_SYNC_MID,
)

print("\n" + "="*80)
print("SYNC ALIGNMENT")
print("="*80)
print(f"ELAN first sync midpoint      : {ELAN_FIRST_SYNC_MID:.6f}s")
print(f"OptiTrack first sync time     : {OPTITRACK_FIRST_SYNC_TIME:.6f}s")
print(f"First shift, Opti - ELAN      : {first_shift:.6f}s")
print()
print(f"ELAN last sync midpoint       : {ELAN_LAST_SYNC_MID:.6f}s")
print(f"OptiTrack last sync time      : {OPTITRACK_LAST_SYNC_TIME:.6f}s")
print(f"Last shift, Opti - ELAN       : {last_shift:.6f}s")
print()
print(f"Shift difference / drift      : {drift:.6f}s")
print()
print("Two-point alignment:")
print(f"video_time_s = {a:.12f} * time_s + ({b:.12f})")

# Apply alignment
print("\nApplying alignment...")
optitrack["video_time_s"] = a * optitrack["time_s"] + b

# Metadata
optitrack["alignment_method"] = "two_point_sync_linear"
optitrack["alignment_a"] = a
optitrack["alignment_b"] = b
optitrack["optitrack_first_sync_time_s"] = OPTITRACK_FIRST_SYNC_TIME
optitrack["optitrack_last_sync_time_s"] = OPTITRACK_LAST_SYNC_TIME
optitrack["elan_first_sync_mid_s"] = ELAN_FIRST_SYNC_MID
optitrack["elan_last_sync_mid_s"] = ELAN_LAST_SYNC_MID
optitrack["first_shift_s"] = first_shift
optitrack["last_shift_s"] = last_shift
optitrack["sync_drift_s"] = drift

# Fast label transfer
print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

# Summary
print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "="*80)
print("LABELING SUMMARY")
print("="*80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

# Save
print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "="*80)
print("DONE")
print("="*80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 63 (code cell #54) ---
# ============================================================
# GROUP 8 - FOCUSED SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 8 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot beginning and ending OptiTrack windows
# - Print top movement candidates in those windows
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 8
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 8 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 8 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )
    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Candidate peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(12)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

# Beginning window: usually first 45–60 seconds
plot_sync_window(
    df=df,
    start_s=0,
    end_s=60,
    title=f"Group {GROUP} beginning sync search"
)

# Ending window:
# Use ELAN last sync midpoint and assume OptiTrack shift around 5–15 seconds.
# Search a wide area near the end.
end_search_start = max(0, ELAN_LAST_SYNC_MID - 20)
end_search_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 40)

plot_sync_window(
    df=df,
    start_s=end_search_start,
    end_s=end_search_end,
    title=f"Group {GROUP} ending sync search"
)


# --- CELL 64 (code cell #55) ---
# ============================================================
# GROUP 8 - CORRECTED SYNC INSPECTION AROUND ELAN SYNC TIMES
#
# First ELAN sync midpoint: 125.4285
# Last ELAN sync midpoint : 2327.091
#
# We inspect wider windows around those expected areas.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 8

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_8/optitrack/optitrack_final/group_8_optitrack_cleaned_combined_240hz.csv"

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

ELAN_FIRST_SYNC_START = 123.381
ELAN_FIRST_SYNC_END   = 127.476
ELAN_FIRST_SYNC_MID   = (ELAN_FIRST_SYNC_START + ELAN_FIRST_SYNC_END) / 2

ELAN_LAST_SYNC_START = 2325.546
ELAN_LAST_SYNC_END   = 2328.636
ELAN_LAST_SYNC_MID   = (ELAN_LAST_SYNC_START + ELAN_LAST_SYNC_END) / 2

print("="*80)
print("GROUP 8 CORRECTED SYNC INSPECTION")
print("="*80)
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ------------------------------------------------------------
# Movement score
# ------------------------------------------------------------

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------

def plot_sync_window(start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ------------------------------------------------------------
# Correct windows
# ------------------------------------------------------------

# Around first ELAN sync. Search wide because OptiTrack may be shifted.
plot_sync_window(
    start_s=100,
    end_s=160,
    title="Group 8 FIRST sync search around ELAN 125s"
)

# Around last ELAN sync. Search wide.
plot_sync_window(
    start_s=2300,
    end_s=2350,
    title="Group 8 LAST sync search around ELAN 2327s"
)


# --- CELL 65 (code cell #56) ---
# ============================================================
# GROUP 8 - FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC
#
# Uses:
# ELAN first sync midpoint  = 125.4285s
# OptiTrack first sync      = 111.125000s
#
# ELAN last sync midpoint   = 2327.091s
# OptiTrack last sync       = 2312.608333s
#
# Applies:
# video_time_s = a * time_s + b
# ============================================================

import os
import re
import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

GROUP = 8
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_8/optitrack/optitrack_final/group_8_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_8/elan/Group_8_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 8 sync inspection
OPTITRACK_FIRST_SYNC_TIME = 111.125000
OPTITRACK_LAST_SYNC_TIME = 2312.608333

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_rows(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    first_sync = sync_rows.iloc[0]
    last_sync = sync_rows.iloc[-1]

    return first_sync, last_sync, sync_rows


def compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last):
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b


def fast_assign_labels(optitrack, elan):
    """
    Fast interval labeling using np.searchsorted.
    Assumes video_time_s is sorted increasing.
    """
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("="*80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING")
print("="*80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# Find sync rows
first_sync, last_sync, sync_rows = find_sync_rows(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

first_shift = OPTITRACK_FIRST_SYNC_TIME - ELAN_FIRST_SYNC_MID
last_shift = OPTITRACK_LAST_SYNC_TIME - ELAN_LAST_SYNC_MID
drift = last_shift - first_shift

a, b = compute_two_point_alignment(
    opti_first=OPTITRACK_FIRST_SYNC_TIME,
    elan_first=ELAN_FIRST_SYNC_MID,
    opti_last=OPTITRACK_LAST_SYNC_TIME,
    elan_last=ELAN_LAST_SYNC_MID,
)

print("\n" + "="*80)
print("SYNC ALIGNMENT")
print("="*80)
print(f"ELAN first sync midpoint      : {ELAN_FIRST_SYNC_MID:.6f}s")
print(f"OptiTrack first sync time     : {OPTITRACK_FIRST_SYNC_TIME:.6f}s")
print(f"First shift, Opti - ELAN      : {first_shift:.6f}s")
print()
print(f"ELAN last sync midpoint       : {ELAN_LAST_SYNC_MID:.6f}s")
print(f"OptiTrack last sync time      : {OPTITRACK_LAST_SYNC_TIME:.6f}s")
print(f"Last shift, Opti - ELAN       : {last_shift:.6f}s")
print()
print(f"Shift difference / drift      : {drift:.6f}s")
print()
print("Two-point alignment:")
print(f"video_time_s = {a:.12f} * time_s + ({b:.12f})")

# Apply alignment
print("\nApplying alignment...")
optitrack["video_time_s"] = a * optitrack["time_s"] + b

# Add alignment metadata
optitrack["alignment_method"] = "two_point_sync_linear"
optitrack["alignment_a"] = a
optitrack["alignment_b"] = b
optitrack["optitrack_first_sync_time_s"] = OPTITRACK_FIRST_SYNC_TIME
optitrack["optitrack_last_sync_time_s"] = OPTITRACK_LAST_SYNC_TIME
optitrack["elan_first_sync_mid_s"] = ELAN_FIRST_SYNC_MID
optitrack["elan_last_sync_mid_s"] = ELAN_LAST_SYNC_MID
optitrack["first_shift_s"] = first_shift
optitrack["last_shift_s"] = last_shift
optitrack["sync_drift_s"] = drift

# Fast label transfer
print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

# Summary
print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "="*80)
print("LABELING SUMMARY")
print("="*80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

# Save
print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "="*80)
print("DONE")
print("="*80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 67 (code cell #57) ---
# ============================================================
# GROUP 7 - SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 7 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot beginning/end sync windows
# - Print top movement candidates
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 7
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 7 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 7 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "cbuilding_subpiece": "co_building_subpiece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

# First sync window:
# Search around ELAN first sync midpoint with wide margin.
first_start = max(0, ELAN_FIRST_SYNC_MID - 40)
first_end = min(df["time_s"].max(), ELAN_FIRST_SYNC_MID + 40)

plot_sync_window(
    df=df,
    start_s=first_start,
    end_s=first_end,
    title=f"Group {GROUP} FIRST sync search around ELAN first sync"
)

# Last sync window:
# Search around ELAN last sync midpoint with wide margin.
last_start = max(0, ELAN_LAST_SYNC_MID - 40)
last_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 40)

plot_sync_window(
    df=df,
    start_s=last_start,
    end_s=last_end,
    title=f"Group {GROUP} LAST sync search around ELAN last sync"
)


# --- CELL 68 (code cell #58) ---
# ============================================================
# GROUP 7 - EXTRA SYNC CHECK
# We need to decide whether the true sync is:
# first: around 66–72s OR around 95–97s
# last : around 2977s OR around 3002s
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 7

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_7/optitrack/optitrack_final/group_7_optitrack_cleaned_combined_240hz.csv"

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

# ELAN sync midpoints
ELAN_FIRST_SYNC_MID = 71.119
ELAN_LAST_SYNC_MID = 2977.0455

# Candidate OptiTrack sync times from previous table
CANDIDATE_FIRST_TIMES = [66.941667, 71.119, 96.475]
CANDIDATE_LAST_TIMES = [2977.631667, 3002.4]

# ------------------------------------------------------------
# Movement score
# ------------------------------------------------------------

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------

def plot_window(start_s, end_s, title, candidate_lines):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        for t in candidate_lines:
            ax.axvline(t, linestyle="--", linewidth=1.2)
            ax.text(t, ax.get_ylim()[1], f"{t:.1f}", rotation=90, verticalalignment="top")

        ax.set_title(f"{title} - {axis.upper()}")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.0,
            label=f"Landmark {i} movement"
        )

    for t in candidate_lines:
        axs[3].axvline(t, linestyle="--", linewidth=1.2)
        axs[3].text(t, axs[3].get_ylim()[1], f"{t:.1f}", rotation=90, verticalalignment="top")

    axs[3].set_title(f"{title} - movement")
    axs[3].set_xlabel("OptiTrack time_s")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.show()


# First sync possibilities
plot_window(
    start_s=50,
    end_s=105,
    title="Group 7 first sync candidate comparison",
    candidate_lines=CANDIDATE_FIRST_TIMES
)

# Last sync possibilities
plot_window(
    start_s=2965,
    end_s=3010,
    title="Group 7 last sync candidate comparison",
    candidate_lines=CANDIDATE_LAST_TIMES
)


# --- CELL 69 (code cell #59) ---
# ============================================================
# GROUP 7 - FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 7
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_7/optitrack/optitrack_final/group_7_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_7/elan/Group_7_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 7 sync inspection
OPTITRACK_FIRST_SYNC_TIME = 71.119
OPTITRACK_LAST_SYNC_TIME = 2977.631667

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "cbuilding_subpiece": "co_building_subpiece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_rows(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    first_sync = sync_rows.iloc[0]
    last_sync = sync_rows.iloc[-1]

    return first_sync, last_sync, sync_rows


def compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last):
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

first_sync, last_sync, sync_rows = find_sync_rows(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

first_shift = OPTITRACK_FIRST_SYNC_TIME - ELAN_FIRST_SYNC_MID
last_shift = OPTITRACK_LAST_SYNC_TIME - ELAN_LAST_SYNC_MID
drift = last_shift - first_shift

a, b = compute_two_point_alignment(
    opti_first=OPTITRACK_FIRST_SYNC_TIME,
    elan_first=ELAN_FIRST_SYNC_MID,
    opti_last=OPTITRACK_LAST_SYNC_TIME,
    elan_last=ELAN_LAST_SYNC_MID,
)

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"ELAN first sync midpoint      : {ELAN_FIRST_SYNC_MID:.6f}s")
print(f"OptiTrack first sync time     : {OPTITRACK_FIRST_SYNC_TIME:.6f}s")
print(f"First shift, Opti - ELAN      : {first_shift:.6f}s")
print()
print(f"ELAN last sync midpoint       : {ELAN_LAST_SYNC_MID:.6f}s")
print(f"OptiTrack last sync time      : {OPTITRACK_LAST_SYNC_TIME:.6f}s")
print(f"Last shift, Opti - ELAN       : {last_shift:.6f}s")
print()
print(f"Shift difference / drift      : {drift:.6f}s")
print()
print("Two-point alignment:")
print(f"video_time_s = {a:.12f} * time_s + ({b:.12f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = a * optitrack["time_s"] + b

optitrack["alignment_method"] = "two_point_sync_linear"
optitrack["alignment_a"] = a
optitrack["alignment_b"] = b
optitrack["optitrack_first_sync_time_s"] = OPTITRACK_FIRST_SYNC_TIME
optitrack["optitrack_last_sync_time_s"] = OPTITRACK_LAST_SYNC_TIME
optitrack["elan_first_sync_mid_s"] = ELAN_FIRST_SYNC_MID
optitrack["elan_last_sync_mid_s"] = ELAN_LAST_SYNC_MID
optitrack["first_shift_s"] = first_shift
optitrack["last_shift_s"] = last_shift
optitrack["sync_drift_s"] = drift

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 71 (code cell #60) ---
# ============================================================
# GROUP 6 - SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 6 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot first/last sync windows
# - Print top movement candidates
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 6
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 6 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 6 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpeing_target_image": "inspecting_target_image",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

first_start = max(0, ELAN_FIRST_SYNC_MID - 50)
first_end = min(df["time_s"].max(), ELAN_FIRST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=first_start,
    end_s=first_end,
    title=f"Group {GROUP} FIRST sync search around ELAN first sync"
)

last_start = max(0, ELAN_LAST_SYNC_MID - 50)
last_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=last_start,
    end_s=last_end,
    title=f"Group {GROUP} LAST sync search around ELAN last sync"
)


# --- CELL 72 (code cell #61) ---
# ============================================================
# GROUP 6 - EXTRA SYNC CANDIDATE COMPARISON
# Need to decide between:
# first: 32.108 vs 38.50
# last : 1367.34 vs 1377.14
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP = 6

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_6/optitrack/optitrack_final/group_6_optitrack_cleaned_combined_240hz.csv"

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

# ELAN sync midpoints
ELAN_FIRST_SYNC_MID = 30.957
ELAN_LAST_SYNC_MID = 1369.273

# Candidate OptiTrack sync times from your tables
CANDIDATE_FIRST_TIMES = [32.108333, 38.495833, 38.587500]
CANDIDATE_LAST_TIMES = [1367.339833, 1376.998167, 1377.139833]

# ------------------------------------------------------------
# Movement score
# ------------------------------------------------------------

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------

def plot_window(start_s, end_s, title, candidate_lines, elan_mid=None):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        for t in candidate_lines:
            ax.axvline(t, linestyle="--", linewidth=1.2)
            ax.text(
                t,
                ax.get_ylim()[1],
                f"{t:.1f}",
                rotation=90,
                verticalalignment="top"
            )

        if elan_mid is not None:
            ax.axvline(elan_mid, linestyle=":", linewidth=1.5)
            ax.text(
                elan_mid,
                ax.get_ylim()[0],
                f"ELAN {elan_mid:.1f}",
                rotation=90,
                verticalalignment="bottom"
            )

        ax.set_title(f"{title} - {axis.upper()}")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.0,
            label=f"Landmark {i} movement"
        )

    for t in candidate_lines:
        axs[3].axvline(t, linestyle="--", linewidth=1.2)
        axs[3].text(
            t,
            axs[3].get_ylim()[1],
            f"{t:.1f}",
            rotation=90,
            verticalalignment="top"
        )

    if elan_mid is not None:
        axs[3].axvline(elan_mid, linestyle=":", linewidth=1.5)
        axs[3].text(
            elan_mid,
            axs[3].get_ylim()[0],
            f"ELAN {elan_mid:.1f}",
            rotation=90,
            verticalalignment="bottom"
        )

    axs[3].set_title(f"{title} - movement")
    axs[3].set_xlabel("OptiTrack time_s")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.show()


# First sync comparison
plot_window(
    start_s=25,
    end_s=45,
    title="Group 6 first sync candidate comparison",
    candidate_lines=CANDIDATE_FIRST_TIMES,
    elan_mid=ELAN_FIRST_SYNC_MID
)

# Last sync comparison
plot_window(
    start_s=1360,
    end_s=1382,
    title="Group 6 last sync candidate comparison",
    candidate_lines=CANDIDATE_LAST_TIMES,
    elan_mid=ELAN_LAST_SYNC_MID
)


# --- CELL 73 (code cell #62) ---
# ============================================================
# GROUP 6 - FAST OPTITRACK LABELING FROM ELAN WITH TWO-POINT SYNC
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 6
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_6/optitrack/optitrack_final/group_6_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_6/elan/Group_6_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 6 visual sync inspection
OPTITRACK_FIRST_SYNC_TIME = 32.108333
OPTITRACK_LAST_SYNC_TIME = 1367.339833

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpeing_target_image": "inspecting_target_image",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_rows(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    first_sync = sync_rows.iloc[0]
    last_sync = sync_rows.iloc[-1]

    return first_sync, last_sync, sync_rows


def compute_two_point_alignment(opti_first, elan_first, opti_last, elan_last):
    a = (elan_last - elan_first) / (opti_last - opti_first)
    b = elan_first - a * opti_first
    return a, b


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

first_sync, last_sync, sync_rows = find_sync_rows(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

first_shift = OPTITRACK_FIRST_SYNC_TIME - ELAN_FIRST_SYNC_MID
last_shift = OPTITRACK_LAST_SYNC_TIME - ELAN_LAST_SYNC_MID
drift = last_shift - first_shift

a, b = compute_two_point_alignment(
    opti_first=OPTITRACK_FIRST_SYNC_TIME,
    elan_first=ELAN_FIRST_SYNC_MID,
    opti_last=OPTITRACK_LAST_SYNC_TIME,
    elan_last=ELAN_LAST_SYNC_MID,
)

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"ELAN first sync midpoint      : {ELAN_FIRST_SYNC_MID:.6f}s")
print(f"OptiTrack first sync time     : {OPTITRACK_FIRST_SYNC_TIME:.6f}s")
print(f"First shift, Opti - ELAN      : {first_shift:.6f}s")
print()
print(f"ELAN last sync midpoint       : {ELAN_LAST_SYNC_MID:.6f}s")
print(f"OptiTrack last sync time      : {OPTITRACK_LAST_SYNC_TIME:.6f}s")
print(f"Last shift, Opti - ELAN       : {last_shift:.6f}s")
print()
print(f"Shift difference / drift      : {drift:.6f}s")
print()
print("Two-point alignment:")
print(f"video_time_s = {a:.12f} * time_s + ({b:.12f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = a * optitrack["time_s"] + b

optitrack["alignment_method"] = "two_point_sync_linear"
optitrack["alignment_a"] = a
optitrack["alignment_b"] = b
optitrack["optitrack_first_sync_time_s"] = OPTITRACK_FIRST_SYNC_TIME
optitrack["optitrack_last_sync_time_s"] = OPTITRACK_LAST_SYNC_TIME
optitrack["elan_first_sync_mid_s"] = ELAN_FIRST_SYNC_MID
optitrack["elan_last_sync_mid_s"] = ELAN_LAST_SYNC_MID
optitrack["first_shift_s"] = first_shift
optitrack["last_shift_s"] = last_shift
optitrack["sync_drift_s"] = drift

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 75 (code cell #63) ---
# ============================================================
# GROUP 5 - SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 5 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot first/last sync windows
# - Print top movement candidates
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 5
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 5 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 5 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpeing_target_image": "inspecting_target_image",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

first_start = max(0, ELAN_FIRST_SYNC_MID - 50)
first_end = min(df["time_s"].max(), ELAN_FIRST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=first_start,
    end_s=first_end,
    title=f"Group {GROUP} FIRST sync search around ELAN first sync"
)

last_start = max(0, ELAN_LAST_SYNC_MID - 50)
last_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=last_start,
    end_s=last_end,
    title=f"Group {GROUP} LAST sync search around ELAN last sync"
)


# --- CELL 76 (code cell #64) ---
# ============================================================
# GROUP 5 - FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC
#
# Reason:
# - OptiTrack ends at ~3449s
# - ELAN ends at ~4090s
# - Final ELAN sync is outside OptiTrack
# - Therefore we use the first synchronization_move only
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 5
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_5/optitrack/optitrack_final/group_5_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_5/elan/Group_5_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 5 sync inspection
OPTITRACK_FIRST_SYNC_TIME = 113.587500

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpeing_target_image": "inspecting_target_image",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_first_sync_row(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    first_sync = sync_rows.iloc[0]

    return first_sync, sync_rows


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING - SINGLE SYNC")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

first_sync, sync_rows = find_first_sync_row(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2

SHIFT = OPTITRACK_FIRST_SYNC_TIME - ELAN_FIRST_SYNC_MID

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"ELAN first sync midpoint      : {ELAN_FIRST_SYNC_MID:.6f}s")
print(f"OptiTrack first sync time     : {OPTITRACK_FIRST_SYNC_TIME:.6f}s")
print(f"SHIFT, Opti - ELAN            : {SHIFT:.6f}s")
print()
print("Single-point alignment:")
print(f"video_time_s = time_s - ({SHIFT:.6f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = optitrack["time_s"] - SHIFT

optitrack["alignment_method"] = "single_point_sync"
optitrack["optitrack_first_sync_time_s"] = OPTITRACK_FIRST_SYNC_TIME
optitrack["elan_first_sync_mid_s"] = ELAN_FIRST_SYNC_MID
optitrack["single_sync_shift_s"] = SHIFT
optitrack["alignment_note"] = (
    "Group 5 used single-point sync because final ELAN sync occurs after OptiTrack recording ends."
)

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 78 (code cell #65) ---
# ============================================================
# GROUP 3 - SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 3 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot first/last sync windows
# - Print top movement candidates
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 3
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except Exception:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 3 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 3 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",
        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",
        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",
        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",
        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",
        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",
        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

first_start = max(0, ELAN_FIRST_SYNC_MID - 50)
first_end = min(df["time_s"].max(), ELAN_FIRST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=first_start,
    end_s=first_end,
    title=f"Group {GROUP} FIRST sync search around ELAN first sync"
)

last_start = max(0, ELAN_LAST_SYNC_MID - 50)
last_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 50)

plot_sync_window(
    df=df,
    start_s=last_start,
    end_s=last_end,
    title=f"Group {GROUP} LAST sync search around ELAN last sync"
)


# --- CELL 79 (code cell #66) ---
# ============================================================
# GROUP 3 - FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC
#
# Reason:
# - Only one ELAN synchronization_move exists
# - Therefore drift validation is not possible
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 3
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_3/optitrack/optitrack_final/group_3_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_3/elan/Group_3_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 3 sync inspection
OPTITRACK_SYNC_TIME = 301.704167

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronaztion_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",

        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",

        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",

        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",

        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",

        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",

        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_row(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    sync_row = sync_rows.iloc[0]

    return sync_row, sync_rows


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING - SINGLE SYNC")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

sync_row, sync_rows = find_sync_row(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_SYNC_MID = (float(sync_row["start_s"]) + float(sync_row["end_s"])) / 2

SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"ELAN sync midpoint          : {ELAN_SYNC_MID:.6f}s")
print(f"OptiTrack sync time         : {OPTITRACK_SYNC_TIME:.6f}s")
print(f"SHIFT, Opti - ELAN          : {SHIFT:.6f}s")
print()
print("Single-point alignment:")
print(f"video_time_s = time_s - ({SHIFT:.6f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = optitrack["time_s"] - SHIFT

optitrack["alignment_method"] = "single_point_sync"
optitrack["optitrack_sync_time_s"] = OPTITRACK_SYNC_TIME
optitrack["elan_sync_mid_s"] = ELAN_SYNC_MID
optitrack["single_sync_shift_s"] = SHIFT
optitrack["alignment_note"] = (
    "Group 3 used single-point sync because only one ELAN synchronization_move was available."
)

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 81 (code cell #67) ---
# ============================================================
# GROUP 2 - SYNC INSPECTION
#
# Important Group 2 note:
# - Not all members did a synchronization move.
# - Sync label may be on a single participant tier, especially Participant1.
# - Therefore we inspect ALL sync-like labels, not only Whole_Group.
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 2
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except Exception:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 2 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 2 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronaztion_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "synch_motion": "synchronization_move",
        "sync_motion": "synchronization_move",
        "synch_move": "synchronization_move",
        "sync_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",

        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",

        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",

        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",

        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",

        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",

        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title, candidate_lines=None):
    if candidate_lines is None:
        candidate_lines = []

    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        for t in candidate_lines:
            ax.axvline(t, linestyle="--", linewidth=1.2)
            ax.text(t, ax.get_ylim()[1], f"{t:.1f}", rotation=90, verticalalignment="top")

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    for t in candidate_lines:
        axs[3].axvline(t, linestyle="--", linewidth=1.2)
        axs[3].text(t, axs[3].get_ylim()[1], f"{t:.1f}", rotation=90, verticalalignment="top")

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["sync_score_tolerant"].index.map(lambda idx: df.loc[idx, "time_s"]),
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["sync_score_strict"].index.map(lambda idx: df.loc[idx, "time_s"]),
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    for t in candidate_lines:
        axs[4].axvline(t, linestyle="--", linewidth=1.2)

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks in this window
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
print("ELAN tiers:", sorted(elan["tier"].unique()))

# ============================================================
# FIND SYNC-LIKE ROWS IN ALL TIERS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron|synch", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ALL ELAN SYNC-LIKE ROWS ACROSS ALL TIERS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No sync-like rows found in ELAN file.")

# Important for Group 2:
# Prefer single-participant tier syncs if present.
participant_sync_rows = sync_rows[
    sync_rows["tier"].astype(str).str.contains("Participant", case=False, na=False)
].copy()

print("\n" + "="*80)
print("PARTICIPANT-TIER SYNC ROWS ONLY")
print("="*80)
display(participant_sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(participant_sync_rows) > 0:
    chosen_sync_rows = participant_sync_rows.copy()
    print("\nUsing participant-tier sync rows for Group 2 inspection.")
else:
    chosen_sync_rows = sync_rows.copy()
    print("\nNo participant-tier sync rows found; using all sync-like rows.")

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan
df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT WINDOWS AROUND EACH CHOSEN SYNC ROW
# ============================================================

for idx, row in chosen_sync_rows.iterrows():
    elan_mid = (float(row["start_s"]) + float(row["end_s"])) / 2

    print("\n" + "#"*80)
    print(f"Inspecting ELAN sync row: tier={row['tier']}, label={row['label']}")
    print(f"ELAN interval: {row['start_s']} -> {row['end_s']}, midpoint={elan_mid}")
    print("#"*80)

    start = max(0, elan_mid - 60)
    end = min(df["time_s"].max(), elan_mid + 80)

    plot_sync_window(
        df=df,
        start_s=start,
        end_s=end,
        title=f"Group {GROUP} sync search around {row['tier']} ELAN sync midpoint {elan_mid:.3f}",
        candidate_lines=[elan_mid]
    )


# --- CELL 82 (code cell #68) ---
# ============================================================
# GROUP 2 - FAST OPTITRACK LABELING FROM ELAN WITH SINGLE PARTICIPANT SYNC
#
# Important Group 2 note:
# - Not all group members did the synchronization move.
# - Sync label was placed on Participant1 action.
# - Therefore we use Participant1 sync label:
#   khalil_synchornizaion_move
# - We ignore the dropping earable sync label for OptiTrack alignment.
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 2
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_2/optitrack/optitrack_final/group_2_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_2/elan/Group_2_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 2 inspection:
# Participant1 sync label: 191.217 -> 194.450, midpoint 192.8335
# OptiTrack strongest Participant1-like movement around 199.883333
OPTITRACK_SYNC_TIME = 199.883333

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronaztion_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "khalil_synchornizaion_move": "participant1_synchronization_move",
        "droping_erarble_syncornaziton_move": "dropping_earable_synchronization_move",
        "synch_motion": "synchronization_move",
        "sync_motion": "synchronization_move",
        "synch_move": "synchronization_move",
        "sync_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",

        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",

        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",

        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",

        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",

        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",

        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_group2_sync_row(elan):
    """
    Group 2 special rule:
    Use Participant1 khalil sync row, not the dropping earable row.
    """
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron|participant1_synchronization", case=False, regex=True, na=False)
    ].copy().sort_values("start_s")

    if len(sync_rows) == 0:
        raise ValueError("No sync-like rows found in ELAN file.")

    preferred = sync_rows[
        (sync_rows["tier_clean"] == "Participant1") &
        (sync_rows["label"] == "participant1_synchronization_move")
    ].copy()

    if len(preferred) == 0:
        raise ValueError(
            "Could not find the preferred Participant1 synchronization row. "
            "Check the ELAN sync labels."
        )

    sync_row = preferred.iloc[0]
    return sync_row, sync_rows


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING - PARTICIPANT1 SINGLE SYNC")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

sync_row, sync_rows = find_group2_sync_row(elan)

print("\nAll sync-like ELAN rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

print("\nChosen Group 2 sync row:")
display(sync_row[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_SYNC_MID = (float(sync_row["start_s"]) + float(sync_row["end_s"])) / 2

SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"Chosen ELAN sync tier       : {sync_row['tier']}")
print(f"Chosen ELAN sync label      : {sync_row['label']}")
print(f"ELAN sync midpoint          : {ELAN_SYNC_MID:.6f}s")
print(f"OptiTrack sync time         : {OPTITRACK_SYNC_TIME:.6f}s")
print(f"SHIFT, Opti - ELAN          : {SHIFT:.6f}s")
print()
print("Single-point alignment:")
print(f"video_time_s = time_s - ({SHIFT:.6f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = optitrack["time_s"] - SHIFT

optitrack["alignment_method"] = "single_point_participant1_sync"
optitrack["optitrack_sync_time_s"] = OPTITRACK_SYNC_TIME
optitrack["elan_sync_mid_s"] = ELAN_SYNC_MID
optitrack["single_sync_shift_s"] = SHIFT
optitrack["alignment_note"] = (
    "Group 2 used Participant1-only synchronization because not all members performed a group sync move. "
    "The dropping earable sync annotation was not used for OptiTrack alignment."
)

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)


# --- CELL 84 (code cell #69) ---
# ============================================================
# GROUP 1 - SYNC WINDOW INSPECTION
#
# Goal:
# - Load Group 1 OptiTrack and ELAN
# - Find ELAN synchronization_move rows
# - Plot first/last sync windows
# - Print top movement candidates
# ============================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

GROUP = 1
BASE = "/content/drive/MyDrive/thesis/data"

# Robust OptiTrack path search
opti_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/group_{GROUP}_optitrack_cleaned_combined_240hz.csv",
    recursive=True
)

valid_opti = []

for p in opti_candidates:
    try:
        test = pd.read_csv(p, nrows=5, low_memory=False)
        required = {
            "time_s",
            "landmark1_x", "landmark1_y", "landmark1_z",
            "landmark2_x", "landmark2_y", "landmark2_z",
            "landmark3_x", "landmark3_y", "landmark3_z",
        }
        if required.issubset(set(test.columns)):
            valid_opti.append(p)
    except Exception:
        pass

if len(valid_opti) == 0:
    raise FileNotFoundError("No valid Group 1 OptiTrack cleaned combined file found.")

OPTITRACK_PATH = valid_opti[0]

# Robust ELAN path search, avoid backup files
elan_candidates = glob.glob(
    f"{BASE}/group_{GROUP}/**/*Group_{GROUP}*individual*build*renamed*.csv",
    recursive=True
)

normal_elan_candidates = [
    p for p in elan_candidates
    if "backup" not in os.path.basename(p).lower()
]

if len(normal_elan_candidates) > 0:
    ELAN_PATH = normal_elan_candidates[0]
elif len(elan_candidates) > 0:
    ELAN_PATH = elan_candidates[0]
else:
    raise FileNotFoundError("No Group 1 ELAN file found.")

print("="*80)
print("SELECTED FILES")
print("="*80)
print("OptiTrack:", OPTITRACK_PATH)
print("ELAN     :", ELAN_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronaztion_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "synch_motion": "synchronization_move",
        "sync_motion": "synchronization_move",
        "synch_move": "synchronization_move",
        "sync_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",

        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",

        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",

        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",

        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",

        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",

        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",

        "carrying_tray_to_centraltable": "carrying_tray_to_central_table",
        "co_building_sub_part": "co_building_subpiece",
        "co_building_sub_piece": "co_building_subpiece",
        "co_building_subpart": "co_building_subpiece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()

    return elan


def plot_sync_window(df, start_s, end_s, title):
    win = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    print("\n" + "="*80)
    print(title)
    print("Window:", start_s, "->", end_s)
    print("Rows:", len(win))
    print("="*80)

    fig, axs = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axs[ax_i]

        for i in [1, 2, 3]:
            ax.plot(
                win["time_s"],
                win[f"landmark{i}_{axis}"],
                linewidth=0.9,
                label=f"Landmark {i}"
            )

        ax.set_title(f"{title} - {axis.upper()} over time")
        ax.set_ylabel(axis.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    for i in [1, 2, 3]:
        axs[3].plot(
            win["time_s"],
            win[f"landmark{i}_movement_smooth"],
            linewidth=1.1,
            label=f"Landmark {i} movement"
        )

    axs[3].set_title(f"{title} - per-landmark movement")
    axs[3].set_ylabel("Movement")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right", ncol=3)

    axs[4].plot(
        win["time_s"],
        win["sync_score_tolerant"],
        linewidth=1.2,
        label="Tolerant sync score, >=2 landmarks"
    )

    axs[4].plot(
        win["time_s"],
        win["sync_score_strict"],
        linewidth=1.2,
        label="Strict sync score, all 3 landmarks"
    )

    axs[4].set_title(f"{title} - sync scores")
    axs[4].set_xlabel("OptiTrack continuous time_s")
    axs[4].set_ylabel("Score")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Top peaks
    peak_df = win[win["sync_score_tolerant"].notna()].copy()

    if len(peak_df) > 0:
        peak_df["time_bin_05s"] = (peak_df["time_s"] // 0.5).astype(int)

        peaks = (
            peak_df
            .sort_values("sync_score_tolerant", ascending=False)
            .groupby("time_bin_05s", as_index=False)
            .head(1)
            .sort_values("sync_score_tolerant", ascending=False)
            .head(15)
        )

        print("\nTop peaks in window:")
        display(
            peaks[
                [
                    "time_s",
                    "take",
                    "active_clean_landmarks",
                    "landmark1_movement_smooth",
                    "landmark2_movement_smooth",
                    "landmark3_movement_smooth",
                    "sync_score_tolerant",
                    "sync_score_strict",
                ]
            ]
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()
elan = load_elan_annotations(ELAN_PATH)

print("\n" + "="*80)
print(f"GROUP {GROUP} DATA INFO")
print("="*80)
print("OptiTrack rows:", len(df))
print("OptiTrack time range:", df["time_s"].min(), "->", df["time_s"].max())
print("OptiTrack duration min:", (df["time_s"].max() - df["time_s"].min()) / 60)
print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
print("ELAN tiers:", sorted(elan["tier"].unique()))

# ============================================================
# FIND ELAN SYNC ROWS
# ============================================================

sync_rows = elan[
    elan["label"].str.contains("sync|synchron|synch", case=False, regex=True, na=False)
].copy().sort_values("start_s")

print("\n" + "="*80)
print("ELAN SYNCHRONIZATION ROWS")
print("="*80)
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

if len(sync_rows) == 0:
    raise ValueError("No synchronization rows found in ELAN file.")

first_sync = sync_rows.iloc[0]
last_sync = sync_rows.iloc[-1]

ELAN_FIRST_SYNC_MID = (float(first_sync["start_s"]) + float(first_sync["end_s"])) / 2
ELAN_LAST_SYNC_MID = (float(last_sync["start_s"]) + float(last_sync["end_s"])) / 2

print("ELAN first sync midpoint:", ELAN_FIRST_SYNC_MID)
print("ELAN last sync midpoint :", ELAN_LAST_SYNC_MID)

# ============================================================
# COMPUTE MOVEMENT SCORES
# ============================================================

for i in [1, 2, 3]:
    dx = df[f"landmark{i}_x"].diff()
    dy = df[f"landmark{i}_y"].diff()
    dz = df[f"landmark{i}_z"].diff()

    df[f"landmark{i}_movement"] = np.sqrt(dx**2 + dy**2 + dz**2)

    df[f"landmark{i}_movement_smooth"] = (
        df[f"landmark{i}_movement"]
        .rolling(120, center=True, min_periods=1)
        .mean()
    )

movement_cols = [
    "landmark1_movement_smooth",
    "landmark2_movement_smooth",
    "landmark3_movement_smooth",
]

available_count = df[movement_cols].notna().sum(axis=1)

df["sync_score_tolerant"] = df[movement_cols].mean(axis=1, skipna=True)
df.loc[available_count < 2, "sync_score_tolerant"] = np.nan

df["sync_score_strict"] = df[movement_cols].min(axis=1, skipna=False)

# ============================================================
# PLOT FOCUSED WINDOWS
# ============================================================

first_start = max(0, ELAN_FIRST_SYNC_MID - 60)
first_end = min(df["time_s"].max(), ELAN_FIRST_SYNC_MID + 80)

plot_sync_window(
    df=df,
    start_s=first_start,
    end_s=first_end,
    title=f"Group {GROUP} FIRST sync search around ELAN first sync"
)

# If there are multiple sync rows, also plot last sync.
# If only one sync exists, this will repeat the same window.
last_start = max(0, ELAN_LAST_SYNC_MID - 60)
last_end = min(df["time_s"].max(), ELAN_LAST_SYNC_MID + 80)

plot_sync_window(
    df=df,
    start_s=last_start,
    end_s=last_end,
    title=f"Group {GROUP} LAST sync search around ELAN last sync"
)


# --- CELL 85 (code cell #70) ---
# ============================================================
# GROUP 1 - FAST OPTITRACK LABELING FROM ELAN WITH SINGLE SYNC
# ============================================================

import os
import re
import pandas as pd
import numpy as np

GROUP = 1
BASE = "/content/drive/MyDrive/thesis/data"

OPTITRACK_PATH = "/content/drive/MyDrive/thesis/data/group_1/optitrack/optitrack_final/group_1_optitrack_cleaned_combined_240hz.csv"
ELAN_PATH = "/content/drive/MyDrive/thesis/data/group_1/elan/Group_1_individual_build_renamed.csv"

OUT_DIR = f"{BASE}/group_{GROUP}/optitrack/optitrack_labeled"
os.makedirs(OUT_DIR, exist_ok=True)

LABELED_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeled.csv"
SUMMARY_OUT_PATH = f"{OUT_DIR}/group_{GROUP}_optitrack_labeling_summary.csv"

# From Group 1 sync inspection
OPTITRACK_SYNC_TIME = 1829.856833

EXPECTED_TIERS = [
    "Participant1",
    "Participant2",
    "Participant3",
    "Participant1_Participant2",
    "Participant1_Participant3",
    "Participant2_Participant3",
    "Whole_Group",
]


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_label(x):
    if pd.isna(x):
        return ""

    label = str(x).strip()

    replacements = {
        "synchronaziton_move": "synchronization_move",
        "synchronaztion_move": "synchronization_move",
        "synchronizaiton_move": "synchronization_move",
        "synchronisation_move": "synchronization_move",
        "synch_motion": "synchronization_move",
        "sync_motion": "synchronization_move",
        "synch_move": "synchronization_move",
        "sync_move": "synchronization_move",
        "clap_synchronizaiton_move": "clap_synchronization_move",

        "task_operaitonal_convo": "task_operational_convo",
        "task_operaitonal_talk": "task_operational_convo",
        "object_handıver": "object_handover",

        "co_builidng_subpiece": "co_building_subpiece",
        "co_bulding_subpiece": "co_building_subpiece",
        "cbuilding_subpiece": "co_building_subpiece",
        "co_buiilding_subpiece": "co_building_subpiece",

        "inspecitng_piece": "inspecting_piece",
        "inspectingpiece": "inspecting_piece",
        "inscpecting_pieces": "inspecting_pieces",
        "inscpecing_other_pieces": "inspecting_other_pieces",
        "inscpeing_target_image": "inspecting_target_image",

        "pickign_up_pice": "picking_up_piece",
        "picking_up_traget_image": "picking_up_target_image",

        "traveling_between_unitsinspecting_other_units": "traveling_between_units + inspecting_other_units",
        ",,nspecting_pieces": "inspecting_pieces",

        "caryying_the_tray": "carrying_the_tray",
        "caryying_tray": "carrying_tray",
        "carrying_tray_to_centraltable": "carrying_tray_to_central_table",

        "actiavely_using_guide_image": "actively_using_guide_image",
        "approcahing_to_piece": "approaching_to_piece",

        "co_building_sub_part": "co_building_subpiece",
        "co_building_sub_piece": "co_building_subpiece",
        "co_building_subpart": "co_building_subpiece",
    }

    parts = [p.strip() for p in label.split("+")]
    parts = [replacements.get(p, p) for p in parts]

    return " + ".join(parts)


def load_elan_annotations(elan_path):
    elan_raw = pd.read_csv(elan_path, header=None, low_memory=False)

    elan = pd.DataFrame({
        "tier": elan_raw.iloc[:, 0].astype(str).str.strip(),
        "start_s": pd.to_numeric(elan_raw.iloc[:, 3], errors="coerce"),
        "end_s": pd.to_numeric(elan_raw.iloc[:, 5], errors="coerce"),
        "duration_s": pd.to_numeric(elan_raw.iloc[:, 7], errors="coerce"),
        "label": elan_raw.iloc[:, 8].apply(normalize_label),
    })

    elan = elan.dropna(subset=["start_s", "end_s"]).copy()
    elan = elan[elan["end_s"] > elan["start_s"]].copy()
    elan["tier_clean"] = elan["tier"].apply(sanitize_name)

    return elan


def find_sync_row(elan):
    sync_rows = elan[
        elan["label"].str.contains("sync|synchron|synch", case=False, regex=True, na=False)
    ].copy()

    if len(sync_rows) < 1:
        raise ValueError("No synchronization rows found in ELAN file.")

    sync_rows = sync_rows.sort_values("start_s").copy()
    sync_row = sync_rows.iloc[0]

    return sync_row, sync_rows


def fast_assign_labels(optitrack, elan):
    out = optitrack.copy()
    times = out["video_time_s"].to_numpy()

    all_tiers = sorted(set(EXPECTED_TIERS).union(set(elan["tier_clean"].unique())))

    for tier in all_tiers:
        out[f"label_{tier}"] = ""

    for tier in all_tiers:
        col = f"label_{tier}"
        tier_rows = elan[elan["tier_clean"] == tier].copy()

        if len(tier_rows) == 0:
            continue

        labels_array = np.full(len(out), "", dtype=object)

        for _, row in tier_rows.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])
            label = str(row["label"])

            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")

            if right <= left:
                continue

            current = labels_array[left:right]

            empty_mask = current == ""
            current[empty_mask] = label

            non_empty_mask = ~empty_mask

            if non_empty_mask.any():
                for local_idx in np.where(non_empty_mask)[0]:
                    existing = current[local_idx]
                    if label not in existing.split(" + "):
                        current[local_idx] = existing + " + " + label

            labels_array[left:right] = current

        out[col] = labels_array

    return out


def make_labeling_summary(labeled):
    rows = []

    label_cols = [c for c in labeled.columns if c.startswith("label_")]

    for col in label_cols:
        values = labeled[col].fillna("").astype(str)
        non_empty = values.str.len() > 0

        unique_labels = values[non_empty].dropna().unique().tolist()

        rows.append({
            "group": GROUP,
            "label_column": col,
            "labeled_rows": int(non_empty.sum()),
            "labeled_pct": float(non_empty.mean() * 100),
            "unique_labels": " | ".join(sorted(unique_labels)),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print(f"GROUP {GROUP} FAST OPTITRACK LABELING - SINGLE SYNC")
print("=" * 80)

if not os.path.exists(OPTITRACK_PATH):
    raise FileNotFoundError(f"OptiTrack file not found:\n{OPTITRACK_PATH}")

if not os.path.exists(ELAN_PATH):
    raise FileNotFoundError(f"ELAN file not found:\n{ELAN_PATH}")

print("Loading OptiTrack...")
optitrack = pd.read_csv(OPTITRACK_PATH, low_memory=False).copy()

print("Loading ELAN...")
elan = load_elan_annotations(ELAN_PATH)

required_cols = [
    "time_s",
    "landmark1_x", "landmark1_y", "landmark1_z",
    "landmark2_x", "landmark2_y", "landmark2_z",
    "landmark3_x", "landmark3_y", "landmark3_z",
]

missing_cols = [c for c in required_cols if c not in optitrack.columns]
if missing_cols:
    raise ValueError(f"This does not look like the OptiTrack file. Missing columns: {missing_cols}")

print("OptiTrack rows:", len(optitrack))
print("OptiTrack time range:", optitrack["time_s"].min(), "->", optitrack["time_s"].max())

print("ELAN rows:", len(elan))
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())

sync_row, sync_rows = find_sync_row(elan)

print("\nELAN sync rows:")
display(sync_rows[["tier", "start_s", "end_s", "duration_s", "label"]])

ELAN_SYNC_MID = (float(sync_row["start_s"]) + float(sync_row["end_s"])) / 2

SHIFT = OPTITRACK_SYNC_TIME - ELAN_SYNC_MID

print("\n" + "=" * 80)
print("SYNC ALIGNMENT")
print("=" * 80)
print(f"ELAN sync midpoint          : {ELAN_SYNC_MID:.6f}s")
print(f"OptiTrack sync time         : {OPTITRACK_SYNC_TIME:.6f}s")
print(f"SHIFT, Opti - ELAN          : {SHIFT:.6f}s")
print()
print("Single-point alignment:")
print(f"video_time_s = time_s - ({SHIFT:.6f})")

print("\nApplying alignment...")
optitrack["video_time_s"] = optitrack["time_s"] - SHIFT

optitrack["alignment_method"] = "single_point_sync"
optitrack["optitrack_sync_time_s"] = OPTITRACK_SYNC_TIME
optitrack["elan_sync_mid_s"] = ELAN_SYNC_MID
optitrack["single_sync_shift_s"] = SHIFT
optitrack["alignment_note"] = (
    "Group 1 used single-point sync because only one ELAN synchronization_move was available."
)

print("\nAssigning labels fast...")
labeled = fast_assign_labels(optitrack, elan)

print("\nCreating summary...")
summary = make_labeling_summary(labeled)

print("\n" + "=" * 80)
print("LABELING SUMMARY")
print("=" * 80)
print("Rows:", len(labeled))
print("OptiTrack time range:", labeled["time_s"].min(), "->", labeled["time_s"].max())
print("Video time range:", labeled["video_time_s"].min(), "->", labeled["video_time_s"].max())
print("ELAN time range:", elan["start_s"].min(), "->", elan["end_s"].max())
display(summary)

print("\nSaving labeled OptiTrack...")
labeled.to_csv(LABELED_OUT_PATH, index=False)

print("Saving summary...")
summary.to_csv(SUMMARY_OUT_PATH, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved labeled OptiTrack:")
print(LABELED_OUT_PATH)

print("\nSaved labeling summary:")
print(SUMMARY_OUT_PATH)

