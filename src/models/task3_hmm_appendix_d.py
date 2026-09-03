"""Task 3, Appendix D — HMM / next-state experiments (Table 8.6).

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "RQ3 — HMM
NEXT-ACTIVITY PREDICTION", "ONE-CELL: 5-class collective-state prediction
+ HMM", and "5-CLASS COLLECTIVE-STATE NEXT-STATE PREDICTION" (source
notebook's own "Appendix D", cells 46-48). See
docs/thesis_reproduction_targets.md §8.8 (Table 8.6).

Table 8.6's 7 rows are assembled from THREE separately self-contained
notebook cells, each with its own data rebuild — not one pipeline. This
module keeps that structure as three functions rather than forcing them
into one:

  - run_three_class_next_window() [cell 46]: 3-class (co_building /
    co_merging / conversation) window-level next-window prediction using
    a supervised Gaussian HMM (hmmlearn) with LOGO-fold-fit transition
    matrix and per-class Gaussian emissions on a robust sensor feature
    set. Rivals: repeat-last, Markov (argmax next), Markov-transition
    (argmax next state that ISN'T the current one — "the fair opponent
    at switch points"). -> Table 8.6 rows 1-3.

  - run_five_class_collective_state_smoothing() [cell 47]: rebuilds a
    30s collective-state timeline directly from the raw
    ALL_MODEL_READY_FILES_IDENTITY_FIXED per-group CSVs (not the
    RQ3_LABEL_NORMALIZATION output the rest of Task 3 uses), trains a
    LogisticRegression on lagged categorical tier-label history, and
    decodes with a custom discrete HMM two ways: a causal forward filter
    (honest prediction) and Viterbi (smoothing — uses future information,
    reported only as an analysis ceiling, never as "the" result). ->
    Table 8.6 rows 4-5 (all-windows only; this cell has no
    transition-only split).

  - run_five_class_collective_state_transition() [cell 48]: the same 30s
    timeline, extended with two emission variants side by side —
    categorical (tier-label history, same as cell 47) and sensor
    (Gaussian-ish LogReg on aggregated OPTI2+OE10 features) — plus the
    repeat/Markov/Markov-transition baselines, reported on both
    all-windows and transition-only scopes. -> Table 8.6 rows 6-7 use
    this cell's transition-only HMM_categorical / HMM_sensor rows
    specifically (cell 47 doesn't compute a transition-only split at
    all, which is why rows 4-5 and 6-7 come from two different cells
    despite describing the "same" 5-class task).

REQUIRES the optional `hmmlearn` dependency (only for
run_three_class_next_window's GaussianHMM) — see requirements.txt.
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-9

# ---- cell 47/48 shared timeline-construction constants ----
GROUPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # group_4 excluded (camera failure) — same as everywhere else in this repo
WIN_SECONDS = 30.0
K_LAG = 3
IND_TIERS = ["label_Participant1", "label_Participant2", "label_Participant3"]
PAIR_TIERS = ["label_Participant1_Participant2", "label_Participant1_Participant3", "label_Participant2_Participant3"]
WHOLE_GROUP_TIER = "label_Whole_Group"
ALL_TIERS = IND_TIERS + PAIR_TIERS + [WHOLE_GROUP_TIER]
COLLECTIVE_STATES = ["individual_phase", "conversation", "co_building", "co_merging", "other_collective"]
_LABEL_TYPO_FIXES = {"object_handiver": "object_handover"}

# ---- cell 46 constants ----
CORE_3CLASS = ["co_building", "co_merging", "conversation"]
CELL46_EMISSION_CANDIDATES = [
    "dist_close_mean", "dist_close_min", "dist_mid_mean", "dist_far_mean", "dist_disp_mean",
    "dist_disp_std", "centroid_speed", "speed_max", "accE_min", "accE_mid", "accE_max",
    "gyrE_min", "gyrE_mid", "gyrE_max", "move_coord", "hand_freq_mean", "hand_freq_max",
    "hand_power_mean", "hand_orient_var", "hand_coord",
]

# Published values this module's output should reproduce (thesis Table
# 8.6 — docs/thesis_reproduction_targets.md §8.8). Keyed by
# (experiment, model). Used only for the optional verification check in
# run_all().
REPORT_REFERENCE = {
    ("three_class_next_window_all", "repeat_last"): {"n": 1773, "accuracy": 0.955, "macro_f1": 0.955},
    ("three_class_next_window_all", "Markov"): {"n": 1773, "accuracy": 0.955, "macro_f1": 0.955},
    ("three_class_next_window_transition", "Markov_transition"): {"n": 79, "accuracy": 0.886, "macro_f1": 0.627},
    ("three_class_next_window_transition", "HMM"): {"n": 79, "accuracy": 0.304, "macro_f1": 0.301},
    ("five_class_collective_state_all", "HMM_viterbi_smoothing"): {"n": 732, "accuracy": 0.643, "macro_f1": 0.599},
    ("five_class_collective_state_all", "HMM_causal_prediction"): {"n": 732, "accuracy": 0.585, "macro_f1": 0.546},
    ("five_class_collective_state_transition", "HMM_categorical"): {"n": 259, "accuracy": 0.282, "macro_f1": 0.265},
    ("five_class_collective_state_transition", "HMM_sensor"): {"n": 259, "accuracy": 0.282, "macro_f1": 0.202},
}


# ================================================================
# Cell 46 — 3-class next-window HMM (Table 8.6 rows 1-3)
# ================================================================

def run_three_class_next_window(data_root: str, out_dir: str):
    """Requires `hmmlearn` (raises ImportError with install instructions
    if missing, rather than the source notebook's own auto-pip-install)."""
    try:
        from hmmlearn import hmm
    except ImportError as exc:
        raise ImportError(
            "run_three_class_next_window needs the optional 'hmmlearn' package "
            "(pip install hmmlearn) — see requirements.txt."
        ) from exc

    path = os.path.join(data_root, "INTERACTION_ENG3", "eng3_recognition_3class_core_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"3-class core feature file not found:\n{path}")

    df = pd.read_csv(path)
    df = df[df["recognition_label"].isin(CORE_3CLASS)].sort_values(["group", "window_start"]).reset_index(drop=True)

    emit_cols = [c for c in CELL46_EMISSION_CANDIDATES if c in df.columns]
    print("emission features:", len(emit_cols), emit_cols)

    classes = CORE_3CLASS
    cls2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    def fit_hmm(train: pd.DataFrame):
        A = np.ones((n, n))
        start = np.ones(n)
        for _, s in train.groupby("group"):
            seq = [cls2i[l] for l in s["recognition_label"].values]
            start[seq[0]] += 1
            for a, b in zip(seq[:-1], seq[1:]):
                A[a, b] += 1
        A = A / A.sum(1, keepdims=True)
        start = start / start.sum()

        X = train[emit_cols].values
        means = np.vstack([X[train["recognition_label"].values == c].mean(0) for c in classes])
        covs = np.vstack([X[train["recognition_label"].values == c].var(0)[None] + 1e-3 for c in classes])

        model = hmm.GaussianHMM(n_components=n, covariance_type="diag", init_params="", params="")
        model.startprob_, model.transmat_, model.means_, model.covars_ = start, A, means, covs
        return model

    y = df["recognition_label"].values
    groups = df["group"].values
    logo = LeaveOneGroupOut()

    preds_all = {k: [] for k in ["HMM", "Markov", "Markov_transition", "repeat_last"]}
    preds_transition = {k: [] for k in preds_all}
    truth_all, truth_transition = [], []

    for tr, te in logo.split(df, y, groups):
        train, test = df.iloc[tr].copy(), df.iloc[te].copy()
        imp = SimpleImputer(strategy="median").fit(train[emit_cols])
        sc = StandardScaler().fit(imp.transform(train[emit_cols]))
        train[emit_cols] = sc.transform(imp.transform(train[emit_cols]))
        test_scaled = sc.transform(imp.transform(test[emit_cols]))

        model = fit_hmm(train)
        A = model.transmat_
        A_off = A.copy()
        np.fill_diagonal(A_off, -1)
        diff_next = np.argmax(A_off, axis=1)

        for g, s in test.groupby("group"):
            loc = [test.index.get_loc(i) for i in s.index]
            x_seq = test_scaled[loc]
            seq = [cls2i[l] for l in s["recognition_label"].values]
            post = model.predict_proba(x_seq)
            hmm_next = np.argmax(post @ A, axis=1)

            for j in range(len(seq) - 1):
                cur, nxt = seq[j], seq[j + 1]
                true = classes[nxt]
                predictions = {
                    "HMM": classes[hmm_next[j]], "Markov": classes[np.argmax(A[cur])],
                    "Markov_transition": classes[diff_next[cur]], "repeat_last": classes[cur],
                }
                for k in preds_all:
                    preds_all[k].append(predictions[k])
                truth_all.append(true)

                if nxt != cur:
                    for k in preds_transition:
                        preds_transition[k].append(predictions[k])
                    truth_transition.append(true)

    def _report(model_name, preds, truth):
        yt, yp = np.array(truth), np.array(preds)
        return {"model": model_name, "n": len(yt), "accuracy": round(accuracy_score(yt, yp), 3),
                "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 3)}

    all_rows = [_report(k, preds_all[k], truth_all) for k in ["repeat_last", "Markov", "Markov_transition", "HMM"]]
    transition_rows = [_report(k, preds_transition[k], truth_transition) for k in ["repeat_last", "Markov", "Markov_transition", "HMM"]]

    all_df = pd.DataFrame(all_rows)
    transition_df = pd.DataFrame(transition_rows)
    print("\n=== 3-class next-window: ALL windows ===")
    print(all_df.to_string(index=False))
    print("\n=== 3-class next-window: TRANSITION-ONLY (the headline) ===")
    print(transition_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    all_df.to_csv(os.path.join(out_dir, "three_class_next_window_all.csv"), index=False)
    transition_df.to_csv(os.path.join(out_dir, "three_class_next_window_transition.csv"), index=False)

    return all_df, transition_df


# ================================================================
# Shared 30s collective-state timeline (cells 47 & 48)
# ================================================================

def _normalize_tier_label(x) -> str:
    x = str(x).strip().lower()
    x = re.split(r"[+|/]", x)[0].strip()
    return _LABEL_TYPO_FIXES.get(x, x)


def _to_collective_state(label) -> str:
    a = _normalize_tier_label(label)
    if a in ("co_building_subpiece", "co_building_piece", "building_subpiece_together", "co_building"):
        return "co_building"
    if a in ("co_merging_subpiece", "merging_subpiece", "co_merging"):
        return "co_merging"
    if a in ("task_operational_convo", "task_social_convo", "task_related_convo", "task_related_social_convo", "non_task_convo"):
        return "conversation"
    return "other_collective"


def _nonempty_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return (s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none")


def _dominant_value(series: pd.Series):
    s = series[_nonempty_mask(series)]
    return s.mode().iloc[0] if len(s) else None


def _find_model_ready_file(data_root: str, group: int):
    matches = glob.glob(os.path.join(data_root, "ALL_MODEL_READY_FILES_IDENTITY_FIXED", f"group_{group}_openearable_model_ready.csv"))
    return matches[0] if matches else None


def build_30s_collective_state_timeline(data_root: str, groups=GROUPS, win_seconds: float = WIN_SECONDS) -> pd.DataFrame:
    """Rebuilds the 30s collective-state timeline directly from the raw
    per-group model-ready CSVs: for each 30s window, the "state" is the
    dominant Whole_Group label if present, else the dominant Pair-tier
    label collapsed to a collective state, else "individual_phase" if
    no group activity was annotated at all. Also carries the dominant
    (normalized) label at every tier, for building lagged categorical
    features downstream."""
    records = []
    for g in groups:
        path = _find_model_ready_file(data_root, g)
        if path is None:
            continue

        d = pd.read_csv(path, low_memory=False)
        d["t"] = pd.to_numeric(d["video_time_s"], errors="coerce")
        d = d.dropna(subset=["t"])
        t0, t1 = d["t"].min(), d["t"].max()

        for win_start in np.arange(t0, t1 - win_seconds + 1e-9, win_seconds):
            mask = (d["t"] >= win_start) & (d["t"] < win_start + win_seconds)
            if mask.sum() == 0:
                continue
            sub = d[mask]

            whole_group = _dominant_value(sub[WHOLE_GROUP_TIER]) if WHOLE_GROUP_TIER in sub else None
            if whole_group is not None:
                state = _to_collective_state(whole_group)
            else:
                pair_label = None
                for pair_tier in PAIR_TIERS:
                    if pair_tier in sub:
                        candidate = _dominant_value(sub[pair_tier])
                        if candidate is not None:
                            pair_label = candidate
                            break
                state = "individual_phase" if pair_label is None else _to_collective_state(pair_label)

            tier_features = {
                tier: (_normalize_tier_label(_dominant_value(sub[tier])) if (tier in sub and _dominant_value(sub[tier]) is not None) else "none")
                for tier in ALL_TIERS
            }
            records.append({"group": g, "win_start": win_start, "win_end": win_start + win_seconds, "state": state, **tier_features})

    timeline = pd.DataFrame(records)
    keep = {"individual_phase", "conversation", "co_building", "co_merging"}
    timeline["state"] = timeline["state"].where(timeline["state"].isin(keep), "other_collective")
    return timeline


def _learn_transition_matrix(label_sequences, classes=COLLECTIVE_STATES):
    cls2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    A = np.ones((n, n))
    start = np.ones(n)
    for seq_labels in label_sequences:
        idx = [cls2i[x] for x in seq_labels]
        start[idx[0]] += 1
        for a, b in zip(idx[:-1], idx[1:]):
            A[a, b] += 1
    return A / A.sum(1, keepdims=True), start / start.sum()


def _causal_filter_smoothing(log_probs, A, start):
    """Cell 47's `causal`: at step t, filters over the CURRENT state given
    observations up to and including t (belief state), argmax'd — used
    for the all-windows "causal prediction" row. Distinct from cell 48's
    `_causal_filter_next_state` below, which projects one step AHEAD
    before taking the argmax (predicting t+1, not t)."""
    T = len(log_probs)
    out = np.zeros(T, dtype=int)
    a = np.log(start + EPS) + log_probs[0]
    out[0] = a.argmax()
    posterior = a - np.logaddexp.reduce(a)
    for t in range(1, T):
        predicted = np.logaddexp.reduce(posterior[:, None] + np.log(A + EPS), axis=0)
        a = predicted + log_probs[t]
        out[t] = a.argmax()
        posterior = a - np.logaddexp.reduce(a)
    return out


def _viterbi_decode(log_probs, A, start):
    T = len(log_probs)
    n = A.shape[0]
    logA = np.log(A + EPS)
    delta = np.log(start + EPS) + log_probs[0]
    backpointer = np.zeros((T, n), dtype=int)
    for t in range(1, T):
        candidates = delta[:, None] + logA
        backpointer[t] = candidates.argmax(0)
        delta = candidates.max(0) + log_probs[t]
    path = np.zeros(T, dtype=int)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]
    return path


# ================================================================
# Cell 47 — 5-class collective-state: LogReg vs. HMM causal/Viterbi
# (Table 8.6 rows 4-5, all-windows only)
# ================================================================

def run_five_class_collective_state_smoothing(data_root: str, out_dir: str, k_lag: int = K_LAG):
    timeline = build_30s_collective_state_timeline(data_root)
    print("windows:", len(timeline), "| target dist:", timeline["state"].value_counts().to_dict())

    classes = COLLECTIVE_STATES
    cls2i = {c: i for i, c in enumerate(classes)}

    rows, targets, groups_out, order = [], [], [], []
    for g, sub in timeline.groupby("group"):
        sub = sub.sort_values("win_start").reset_index(drop=True)
        for i in range(k_lag, len(sub)):
            feat = {f"{tier}__lag{lag}": sub.iloc[i - lag][tier] for lag in range(1, k_lag + 1) for tier in ALL_TIERS}
            feat["prev_state"] = sub.iloc[i - 1]["state"]
            rows.append(feat)
            targets.append(sub.iloc[i]["state"])
            groups_out.append(g)
            order.append((g, i))

    X_df = pd.DataFrame(rows).fillna("none")
    y = np.array(targets)
    groups_arr = np.array(groups_out)
    order_df = pd.DataFrame(order, columns=["group", "pos"])
    print("examples:", len(X_df))

    logo = LeaveOneGroupOut()
    y_true, y_pred_logreg, y_pred_causal, y_pred_viterbi = [], [], [], []

    for tr, te in logo.split(X_df, y, groups_arr):
        model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000, class_weight="balanced"))
        model.fit(X_df.iloc[tr], y[tr])
        proba = model.predict_proba(X_df.iloc[te])
        proba = proba[:, [list(model.classes_).index(c) for c in classes]]
        baseline_pred = np.array(classes)[proba.argmax(1)]

        train_sequences = [y[tr][order_df.iloc[tr]["group"].values == g] for g in np.unique(groups_arr[tr])]
        A, start = _learn_transition_matrix(train_sequences, classes)

        test_order = order_df.iloc[te].copy()
        test_order["r"] = np.arange(len(te))
        test_order["yt"] = y[te]

        for g, s in test_order.groupby("group"):
            s = s.sort_values("pos")
            ri = s["r"].values
            log_probs = np.log(proba[ri] + EPS)
            y_true += list(s["yt"])
            y_pred_logreg += list(baseline_pred[ri])
            y_pred_causal += [classes[i] for i in _causal_filter_smoothing(log_probs, A, start)]
            y_pred_viterbi += [classes[i] for i in _viterbi_decode(log_probs, A, start)]

    y_true = np.array(y_true)

    def _score(name, preds):
        preds = np.array(preds)
        return {"model": name, "n": len(y_true), "accuracy": round(accuracy_score(y_true, preds), 3),
                "macro_f1": round(f1_score(y_true, preds, average="macro", zero_division=0), 3)}

    summary = pd.DataFrame([
        _score("LogReg_before", y_pred_logreg),
        _score("HMM_causal_prediction", y_pred_causal),
        _score("HMM_viterbi_smoothing", y_pred_viterbi),
    ])
    print("\n=== 5-class collective-state: LogReg vs HMM (30s, LOGO, ALL windows) ===")
    print(summary.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "five_class_collective_state_all_windows.csv"), index=False)

    return summary


# ================================================================
# Cell 48 — 5-class collective-state next-state, dual emissions
# (Table 8.6 rows 6-7, transition-only)
# ================================================================

def _load_10s_sensor_features(path: str, prefixes: tuple[str, ...]):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    return df[["group", "window_start", "window_end"] + cols], cols


def _aggregate_sensor_to_30s(sensor_df: pd.DataFrame, sensor_cols: list[str], group, win_start: float, win_end: float) -> pd.Series:
    mask = (sensor_df["group"] == group) & (sensor_df["window_start"] >= win_start - 1e-6) & (sensor_df["window_start"] < win_end)
    matched = sensor_df[mask]
    return matched[sensor_cols].mean() if len(matched) else pd.Series({c: np.nan for c in sensor_cols})


def _causal_filter_next_state(log_probs, A, start):
    """Cell 48's `causal_next`: forward filter that predicts the NEXT
    state at every step (belief over current state, projected one step
    ahead through A) — this is the "honest" causal-prediction decode
    used for both the smoothing-cell's causal row and this cell's
    HMM_categorical / HMM_sensor rows."""
    T = len(log_probs)
    preds = np.zeros(T, dtype=int)
    a = np.log(start + EPS) + log_probs[0]
    for t in range(T):
        belief = a - np.logaddexp.reduce(a)
        next_dist = np.logaddexp.reduce(belief[:, None] + np.log(A + EPS), axis=0)
        preds[t] = next_dist.argmax()
        if t + 1 < T:
            a = np.logaddexp.reduce(belief[:, None] + np.log(A + EPS), axis=0) + log_probs[t + 1]
    return preds


def run_five_class_collective_state_transition(data_root: str, out_dir: str, k_lag: int = K_LAG):
    timeline = build_30s_collective_state_timeline(data_root)

    opti2_path = os.path.join(data_root, "INTERACTION_OPTI2", "interaction_opti2_10s.csv")
    oe10_path = os.path.join(data_root, "INTERACTION_OE10", "interaction_oe10_10s.csv")
    if not os.path.exists(opti2_path) or not os.path.exists(oe10_path):
        raise FileNotFoundError(f"10s sensor feature files not found:\n  {opti2_path}\n  {oe10_path}")

    opti_df, opti_cols = _load_10s_sensor_features(opti2_path, ("opti2_",))
    oe_df, oe_cols = _load_10s_sensor_features(oe10_path, ("oe_", "mag_"))
    sensor_df = pd.merge(opti_df, oe_df, on=["group", "window_start", "window_end"], how="outer")
    sensor_cols = opti_cols + oe_cols

    sensor_feats = timeline.apply(lambda row: _aggregate_sensor_to_30s(sensor_df, sensor_cols, row.group, row.win_start, row.win_end), axis=1)
    timeline = pd.concat([timeline.reset_index(drop=True), sensor_feats.reset_index(drop=True)], axis=1)

    classes = COLLECTIVE_STATES

    rows, targets, groups_out, order = [], [], [], []
    for g, sub in timeline.groupby("group"):
        sub = sub.sort_values("win_start").reset_index(drop=True)
        for i in range(k_lag, len(sub)):
            feat = {f"{tier}__lag{lag}": sub.iloc[i - lag][tier] for lag in range(1, k_lag + 1) for tier in ALL_TIERS}
            feat["prev_state"] = sub.iloc[i - 1]["state"]
            for c in sensor_cols:
                feat[c] = sub.iloc[i - 1][c]  # sensor emission = features at CURRENT window (t-1 -> predict t)
            rows.append(feat)
            targets.append(sub.iloc[i]["state"])
            groups_out.append(g)
            order.append((g, i))

    X_df = pd.DataFrame(rows)
    y = np.array(targets)
    groups_arr = np.array(groups_out)
    order_df = pd.DataFrame(order, columns=["group", "pos"])

    cat_cols = [c for c in X_df.columns if c.endswith(tuple(f"lag{lag}" for lag in range(1, k_lag + 1))) or c == "prev_state"]
    num_cols = [c for c in sensor_cols if c in X_df.columns]
    print("examples:", len(X_df), "| categorical:", len(cat_cols), "| sensor:", len(num_cols))

    logo = LeaveOneGroupOut()
    preds_all = {k: [] for k in ["repeat", "Markov", "Markov_transition", "HMM_categorical", "HMM_sensor"]}
    preds_transition = {k: [] for k in preds_all}
    truth_all, truth_transition = [], []

    for tr, te in logo.split(X_df, y, groups_arr):
        cat_model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X_df.iloc[tr][cat_cols], y[tr])
        sensor_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X_df.iloc[tr][num_cols], y[tr])

        proba_cat = cat_model.predict_proba(X_df.iloc[te][cat_cols])[:, [list(cat_model.classes_).index(c) for c in classes]]
        proba_sensor = sensor_model.predict_proba(X_df.iloc[te][num_cols])[:, [list(sensor_model.classes_).index(c) for c in classes]]

        train_sequences = [y[tr][order_df.iloc[tr]["group"].values == g] for g in np.unique(groups_arr[tr])]
        A, start = _learn_transition_matrix(train_sequences, classes)
        A_off = A.copy()
        np.fill_diagonal(A_off, -1)
        diff_next = np.argmax(A_off, axis=1)

        test_order = order_df.iloc[te].copy()
        test_order["r"] = np.arange(len(te))
        test_order["yt"] = y[te]

        for g, s in test_order.groupby("group"):
            s = s.sort_values("pos")
            ri = s["r"].values
            seq = [{c: i for i, c in enumerate(classes)}[x] for x in s["yt"]]
            hmm_cat = _causal_filter_next_state(np.log(proba_cat[ri] + EPS), A, start)
            hmm_sensor = _causal_filter_next_state(np.log(proba_sensor[ri] + EPS), A, start)

            for j in range(len(seq) - 1):
                cur, nxt = seq[j], seq[j + 1]
                true = classes[nxt]
                predictions = {
                    "repeat": classes[cur], "Markov": classes[A[cur].argmax()], "Markov_transition": classes[diff_next[cur]],
                    "HMM_categorical": classes[hmm_cat[j]], "HMM_sensor": classes[hmm_sensor[j]],
                }
                for k in preds_all:
                    preds_all[k].append(predictions[k])
                truth_all.append(true)

                if nxt != cur:
                    for k in preds_transition:
                        preds_transition[k].append(predictions[k])
                    truth_transition.append(true)

    def _report(name, preds, truth):
        preds, truth = np.array(preds), np.array(truth)
        return {"model": name, "n": len(truth), "accuracy": round(accuracy_score(truth, preds), 3),
                "macro_f1": round(f1_score(truth, preds, average="macro", zero_division=0), 3)}

    order_names = ["repeat", "Markov", "Markov_transition", "HMM_categorical", "HMM_sensor"]
    all_df = pd.DataFrame([_report(k, preds_all[k], truth_all) for k in order_names])
    transition_df = pd.DataFrame([_report(k, preds_transition[k], truth_transition) for k in order_names])

    print("\n=== 5-class collective-state next-state: ALL windows ===")
    print(all_df.to_string(index=False))
    print("\n=== 5-class collective-state next-state: TRANSITION-ONLY (the honest test) ===")
    print(transition_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    all_df.to_csv(os.path.join(out_dir, "five_class_collective_state_next_state_all.csv"), index=False)
    transition_df.to_csv(os.path.join(out_dir, "five_class_collective_state_next_state_transition.csv"), index=False)

    return all_df, transition_df


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_root: str, out_dir: str, skip_three_class: bool = False):
    """Runs all 3 cells and assembles Table 8.6's 7 rows. `skip_three_class`
    lets callers without `hmmlearn` installed still get rows 4-7."""
    os.makedirs(out_dir, exist_ok=True)
    combined_rows = []

    if not skip_three_class:
        three_all, three_transition = run_three_class_next_window(data_root, out_dir)
        for _, row in three_all.iterrows():
            combined_rows.append({"experiment": "three_class_next_window_all", **row.to_dict()})
        for _, row in three_transition.iterrows():
            combined_rows.append({"experiment": "three_class_next_window_transition", **row.to_dict()})

    smoothing_summary = run_five_class_collective_state_smoothing(data_root, out_dir)
    for _, row in smoothing_summary.iterrows():
        combined_rows.append({"experiment": "five_class_collective_state_all", **row.to_dict()})

    five_all, five_transition = run_five_class_collective_state_transition(data_root, out_dir)
    for _, row in five_transition.iterrows():
        combined_rows.append({"experiment": "five_class_collective_state_transition", **row.to_dict()})

    combined = pd.DataFrame(combined_rows)
    combined.to_csv(os.path.join(out_dir, "task3_hmm_appendix_d_combined.csv"), index=False)

    check_rows = []
    for (experiment, model), ref in REPORT_REFERENCE.items():
        row = combined[(combined["experiment"] == experiment) & (combined["model"] == model)]
        if row.empty:
            check_rows.append({"experiment": experiment, "model": model, "match": "SKIPPED" if skip_three_class and experiment.startswith("three_class") else "MISSING"})
            continue
        row = row.iloc[0]
        match = (
            "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0015 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0015 and int(row["n"]) == ref["n"])
            else "DIFFERS"
        )
        check_rows.append({
            "experiment": experiment, "model": model, "n": int(row["n"]), "report_n": ref["n"],
            "accuracy": row["accuracy"], "report_accuracy": ref["accuracy"],
            "macro_f1": row["macro_f1"], "report_macro_f1": ref["macro_f1"],
            "match": match,
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.6")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "task3_hmm_appendix_d_report_reproduction_check.csv"), index=False)

    n_exact = int((check["match"] == "EXACT").sum())
    print(f"\n{n_exact} of {len(check)} referenced Table 8.6 rows reproduce exactly.")

    return combined, check
