"""Task 3, Appendix C — expanding-prefix next-segment prediction.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cell "TASK 3 —
EXPANDING-PREFIX SEGMENT PREDICTION" plus its fold-std follow-up (source
notebook's own "Appendix C", cells 43-44). See
docs/thesis_reproduction_targets.md §8.5 (Table 8.5) and
docs/table_to_source_mapping.md's Task 3 breakdown.

Unlike Appendix A/B (which read the RQ3_LABEL_NORMALIZATION output), this
module reads its own pair of inputs directly:
`INTERACTION_ENG3/recognition_interaction_window_label_inventory.csv`
(window-level `dominant_normalized_label`) and
`INTERACTION_ENG3/interaction_eng3_features.csv` (engineered
Optitrack/OpenEarable/Xsens features) — merged on (group, window_start,
window_end). That is the source notebook's own choice, not this port's;
preserved as-is.

Task: given the full sequence of past activity segments in a session
(the "expanding prefix" [t1] -> t2, [t1, t2] -> t3, ...), predict the
next segment's activity, at two label granularities:
  - fine_13: rare activities (< MIN_WINDOWS windows) collapsed to "other",
    everything else kept as-is (~13 classes).
  - coarse_4: co_merging / co_building / conversation / other, by simple
    substring priority matching on the raw label text.
Compared across two feature modes (activity-sequence only vs. activity +
mean-pooled segment sensor features) and 6 predictors: majority,
Markov-last (argmax next given only the last segment), suffix-backoff
Markov (variable-order, up to MAX_SUFFIX_ORDER), and 3 sequence neural
nets (LSTM / CNN1D / Transformer) trained fresh per LOGO fold.

Table 8.5 reports 4 coarse_4 rows and 3 fine_13 rows — not the full grid —
see REPORT_REFERENCE below.
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
MIN_WINDOWS = 15
LABEL_MODES = ("coarse_4", "fine_13")
FEATURE_MODES = ("activity_only", "activity_plus_segment_features")
DEEP_MODELS = ("lstm", "cnn1d", "transformer")
MAX_EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMB_DIM = 16
HIDDEN_DIM = 64
MAX_SUFFIX_ORDER = 5

OPTITRACK_FEATURES = [
    "dist_close_mean", "dist_close_min", "dist_mid_mean", "dist_far_mean",
    "dist_disp_mean", "dist_disp_std", "speed_min", "speed_mid", "speed_max", "centroid_speed",
]
OPENEAREABLE_FEATURES = [
    "accE_min", "accE_mid", "accE_max", "gyrE_min", "gyrE_mid", "gyrE_max", "move_coord",
]
XSENS_FEATURES = [
    "hand_freq_mean", "hand_freq_max", "hand_power_mean", "hand_orient_var", "hand_coord", "xsens_available",
]
CANDIDATE_FEATURES = OPTITRACK_FEATURES + OPENEAREABLE_FEATURES + XSENS_FEATURES

LABEL_COL = "dominant_normalized_label"

# Published values this module's output should reproduce (thesis Table
# 8.5 — docs/thesis_reproduction_targets.md §8.5). Only 7 of the 24
# (label_mode x feature_mode x predictor) combinations are tabulated —
# the ones the thesis singles out. Used only for the optional
# verification check in run_all().
REPORT_REFERENCE = {
    ("coarse_4", "activity_only", "cnn1d"): {"n": 216, "accuracy": 0.657, "macro_f1": 0.564, "weighted_f1": 0.684},
    ("coarse_4", "activity_only", "suffix_backoff"): {"n": 216, "accuracy": 0.708, "macro_f1": 0.561, "weighted_f1": 0.703},
    ("coarse_4", "activity_only", "transformer"): {"n": 216, "accuracy": 0.653, "macro_f1": 0.557, "weighted_f1": 0.665},
    ("coarse_4", "activity_only", "markov_last"): {"n": 216, "accuracy": 0.722, "macro_f1": 0.409, "weighted_f1": 0.624},
    ("fine_13", "activity_only", "transformer"): {"n": 287, "accuracy": 0.254, "macro_f1": 0.178, "weighted_f1": 0.274},
    ("fine_13", "activity_only", "suffix_backoff"): {"n": 287, "accuracy": 0.355, "macro_f1": 0.169, "weighted_f1": 0.337},
    ("fine_13", "activity_plus_segment_features", "transformer"): {"n": 287, "accuracy": 0.345, "macro_f1": 0.173, "weighted_f1": 0.346},
}


def load_inventory_and_features(data_root: str):
    """Loads the window-level label inventory and the ENG3 sensor feature
    table this module needs (a different pair of inputs than
    task3_persistence.py / task3_segment_forecast.py use)."""
    inv_path = os.path.join(data_root, "INTERACTION_ENG3", "recognition_interaction_window_label_inventory.csv")
    feat_path = os.path.join(data_root, "INTERACTION_ENG3", "interaction_eng3_features.csv")

    if not os.path.exists(inv_path) or not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Expanding-prefix inputs not found:\n  {inv_path}\n  {feat_path}\n"
            "These are feature-engineering stage outputs, not raw data — see "
            "docs/table_to_source_mapping.md for what produces them."
        )

    inv_original = pd.read_csv(inv_path)
    features_df = pd.read_csv(feat_path)

    if LABEL_COL not in inv_original.columns:
        raise ValueError(f"{LABEL_COL} not found. Available columns: {inv_original.columns.tolist()}")

    return inv_original, features_df


def _normalize_text(x) -> str:
    return str(x).strip().lower().replace(" ", "_")


def make_fine_13_labels(inv: pd.DataFrame, min_windows: int = MIN_WINDOWS) -> pd.DataFrame:
    """Keep labels with >= min_windows windows; rare labels -> 'other'."""
    inv = inv.copy()
    counts = inv[LABEL_COL].value_counts()
    keep = set(counts[counts >= min_windows].index)
    inv["activity_for_prediction"] = inv[LABEL_COL].where(inv[LABEL_COL].isin(keep), "other")
    return inv


def _coarse_4_map(raw_label) -> str:
    """co_merging > co_building > conversation > other, by substring
    priority (avoids tiny fine-grained transition classes)."""
    x = _normalize_text(raw_label)
    if "merging" in x or "merge" in x:
        return "co_merging"
    if "building" in x or "build" in x:
        return "co_building"
    if "convo" in x or "conversation" in x:
        return "conversation"
    return "other"


def make_coarse_4_labels(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    inv["activity_for_prediction"] = inv[LABEL_COL].apply(_coarse_4_map)
    return inv


def _add_key_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["group_key"] = d["group"].astype(int)
    d["window_start_key"] = np.round(d["window_start"].astype(float), 6)
    d["window_end_key"] = np.round(d["window_end"].astype(float), 6)
    return d


def detect_sensor_features(features_df: pd.DataFrame) -> list[str]:
    return [c for c in CANDIDATE_FEATURES if c in features_df.columns]


def build_segments_for_label_mode(inv_labelled: pd.DataFrame, label_mode: str, features_df: pd.DataFrame, sensor_features: list[str]) -> pd.DataFrame:
    """Collapse consecutive same-label windows into activity segments;
    each segment gets its member windows' mean sensor feature vector."""
    inv = _add_key_columns(inv_labelled)
    features_keyed = _add_key_columns(features_df)
    merge_cols = ["group_key", "window_start_key", "window_end_key"] + sensor_features

    df = inv.merge(features_keyed[merge_cols], on=["group_key", "window_start_key", "window_end_key"], how="left", validate="one_to_one")
    df = df.sort_values(["group", "window_start"]).reset_index(drop=True)

    df["prev_activity"] = df.groupby("group")["activity_for_prediction"].shift(1)
    df["new_segment"] = (df["activity_for_prediction"] != df["prev_activity"]) | df["prev_activity"].isna()
    df["segment_id_in_group"] = df.groupby("group")["new_segment"].cumsum().astype(int)

    segment_rows = []
    for (g, seg_id), sub in df.groupby(["group", "segment_id_in_group"]):
        sub = sub.sort_values("window_start")
        row = {
            "label_mode": label_mode, "group": int(g), "segment_id_in_group": int(seg_id),
            "segment_key": f"{int(g)}_{int(seg_id)}", "activity": sub["activity_for_prediction"].iloc[0],
            "segment_start": float(sub["window_start"].min()), "segment_end": float(sub["window_end"].max()),
            "segment_duration_s": float(sub["window_end"].max() - sub["window_start"].min()), "n_windows": int(len(sub)),
        }
        numeric = sub[sensor_features].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean(axis=0, skipna=True)
        for c in sensor_features:
            row[f"full__{c}"] = float(means[c]) if pd.notna(means[c]) else np.nan
        segment_rows.append(row)

    return pd.DataFrame(segment_rows).sort_values(["group", "segment_start"]).reset_index(drop=True)


def build_prefix_examples(segments: pd.DataFrame, sensor_features: list[str]):
    """[t1] -> t2, [t1, t2] -> t3, ... per group. Returns (seq_acts,
    seq_nums, y, groups, meta, label_names, act_to_id, id_to_act,
    numeric_cols)."""
    label_names = sorted(segments["activity"].unique())
    act_to_id = {a: i for i, a in enumerate(label_names)}
    id_to_act = {i: a for a, i in act_to_id.items()}

    segments = segments.copy()
    segments["activity_id"] = segments["activity"].map(act_to_id)
    numeric_cols = ["segment_duration_s", "n_windows"] + [f"full__{c}" for c in sensor_features]

    seq_acts, seq_nums, y, groups, meta_rows = [], [], [], [], []
    for g, sub in segments.sort_values(["group", "segment_start"]).groupby("group"):
        sub = sub.reset_index(drop=True)
        for i in range(1, len(sub)):
            prefix, target = sub.iloc[:i], sub.iloc[i]
            seq_acts.append(prefix["activity_id"].values.astype(np.int64))
            seq_nums.append(prefix[numeric_cols].values.astype(np.float32))
            y.append(int(target["activity_id"]))
            groups.append(int(g))
            meta_rows.append({
                "label_mode": target["label_mode"], "group": int(g), "target_segment_key": target["segment_key"],
                "target_activity": target["activity"], "prefix_len": int(len(prefix)),
                "prefix_activities": " -> ".join(prefix["activity"].tolist()),
            })

    return (seq_acts, seq_nums, np.array(y, dtype=np.int64), np.array(groups, dtype=np.int64),
            pd.DataFrame(meta_rows), label_names, act_to_id, id_to_act, numeric_cols)


def _pad_activity_sequences(seq_acts, max_len: int, pad_idx: int):
    X = np.full((len(seq_acts), max_len), pad_idx, dtype=np.int64)
    mask = np.zeros((len(seq_acts), max_len), dtype=bool)
    for i, seq in enumerate(seq_acts):
        X[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    return X, mask


def _pad_numeric_sequences(seq_nums, max_len: int, n_numeric: int):
    X = np.full((len(seq_nums), max_len, n_numeric), np.nan, dtype=np.float32)
    for i, seq in enumerate(seq_nums):
        X[i, :seq.shape[0], :] = seq
    return X


def _preprocess_numeric_fold(X_num_train_raw, X_num_test_raw, mask_train, mask_test, use_numeric: bool):
    """Fits imputer/scaler only on valid training prefix positions; padding stays zero."""
    if not use_numeric:
        n_train, max_len, _ = X_num_train_raw.shape
        n_test = X_num_test_raw.shape[0]
        return np.zeros((n_train, max_len, 0), dtype=np.float32), np.zeros((n_test, max_len, 0), dtype=np.float32)

    n_train, max_len, n_numeric = X_num_train_raw.shape
    n_test = X_num_test_raw.shape[0]
    Xtr = np.zeros((n_train, max_len, n_numeric), dtype=np.float32)
    Xte = np.zeros((n_test, max_len, n_numeric), dtype=np.float32)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_valid_scaled = scaler.fit_transform(imputer.fit_transform(X_num_train_raw[mask_train]))
    test_valid_scaled = scaler.transform(imputer.transform(X_num_test_raw[mask_test]))
    Xtr[mask_train] = train_valid_scaled.astype(np.float32)
    Xte[mask_test] = test_valid_scaled.astype(np.float32)
    return Xtr, Xte


class PrefixDataset(Dataset):
    def __init__(self, X_act, X_num, mask, y):
        self.X_act = torch.tensor(X_act, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_act[idx], self.X_num[idx], self.mask[idx], self.y[idx]


class PrefixLSTM(nn.Module):
    def __init__(self, n_classes, pad_idx, n_numeric, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb_dim, padding_idx=pad_idx)
        self.use_numeric = n_numeric > 0
        if self.use_numeric:
            self.num_proj = nn.Sequential(nn.Linear(n_numeric, emb_dim), nn.ReLU())
            input_dim = emb_dim * 2
        else:
            input_dim = emb_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, x_act, x_num, mask):
        e = self.emb(x_act)
        x = torch.cat([e, self.num_proj(x_num)], dim=-1) if self.use_numeric else e
        out, _ = self.lstm(x)
        lengths = mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        return self.head(out.gather(1, idx).squeeze(1))


class PrefixCNN1D(nn.Module):
    def __init__(self, n_classes, pad_idx, n_numeric, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb_dim, padding_idx=pad_idx)
        self.use_numeric = n_numeric > 0
        if self.use_numeric:
            self.num_proj = nn.Sequential(nn.Linear(n_numeric, emb_dim), nn.ReLU())
            input_dim = emb_dim * 2
        else:
            input_dim = emb_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, x_act, x_num, mask):
        e = self.emb(x_act)
        x = torch.cat([e, self.num_proj(x_num)], dim=-1) if self.use_numeric else e
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = h.masked_fill(~mask.unsqueeze(1), -1e9)
        return self.head(h.max(dim=-1).values)


class PrefixTransformer(nn.Module):
    def __init__(self, n_classes, pad_idx, n_numeric, max_len, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_heads=4, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        self.use_numeric = n_numeric > 0
        if self.use_numeric:
            self.num_proj = nn.Sequential(nn.Linear(n_numeric, emb_dim), nn.ReLU())
            input_dim = emb_dim * 2
        else:
            input_dim = emb_dim
        heads = n_heads if input_dim % n_heads == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=heads, dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, n_classes))

    def forward(self, x_act, x_num, mask):
        e = self.emb(x_act) + self.pos_emb[:, :x_act.shape[1], :]
        x = torch.cat([e, self.num_proj(x_num)], dim=-1) if self.use_numeric else e
        h = self.encoder(x, src_key_padding_mask=~mask)
        lengths = mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
        return self.head(h.gather(1, idx).squeeze(1))


def _build_deep_model(model_name: str, n_classes: int, pad_idx: int, n_numeric: int, max_len: int):
    if model_name == "lstm":
        return PrefixLSTM(n_classes, pad_idx, n_numeric)
    if model_name == "cnn1d":
        return PrefixCNN1D(n_classes, pad_idx, n_numeric)
    if model_name == "transformer":
        return PrefixTransformer(n_classes, pad_idx, n_numeric, max_len)
    raise ValueError(model_name)


def predict_majority(y_train, n_test: int) -> np.ndarray:
    maj = Counter(y_train.tolist()).most_common(1)[0][0]
    return np.array([maj] * n_test, dtype=np.int64)


def predict_markov_last(X_act_train, y_train, X_act_test, pad_idx: int) -> np.ndarray:
    table = defaultdict(Counter)
    for x, target in zip(X_act_train, y_train):
        valid = x[x != pad_idx]
        if len(valid) == 0:
            continue
        table[int(valid[-1])][int(target)] += 1

    fallback = Counter(y_train.tolist()).most_common(1)[0][0]
    preds = []
    for x in X_act_test:
        valid = x[x != pad_idx]
        if len(valid) == 0:
            preds.append(fallback)
            continue
        last = int(valid[-1])
        preds.append(table[last].most_common(1)[0][0] if last in table and len(table[last]) > 0 else fallback)
    return np.array(preds, dtype=np.int64)


def train_suffix_tables(X_act_train, y_train, pad_idx: int, max_order: int = MAX_SUFFIX_ORDER):
    """Variable-order Markov / suffix-backoff: stores last-1..last-max_order -> next; prediction uses the longest suffix seen in training."""
    tables = {k: defaultdict(Counter) for k in range(1, max_order + 1)}
    for x, target in zip(X_act_train, y_train):
        valid = tuple(int(v) for v in x if int(v) != pad_idx)
        for k in range(1, min(max_order, len(valid)) + 1):
            tables[k][valid[-k:]][int(target)] += 1
    fallback = Counter(y_train.tolist()).most_common(1)[0][0]
    return tables, fallback


def predict_suffix_backoff(X_act_test, tables: dict, fallback, pad_idx: int, max_order: int = MAX_SUFFIX_ORDER) -> np.ndarray:
    preds = []
    for x in X_act_test:
        valid = tuple(int(v) for v in x if int(v) != pad_idx)
        pred = fallback
        for k in range(min(max_order, len(valid)), 0, -1):
            key = valid[-k:]
            if key in tables[k] and len(tables[k][key]) > 0:
                pred = tables[k][key].most_common(1)[0][0]
                break
        preds.append(pred)
    return np.array(preds, dtype=np.int64)


def train_eval_deep(model_name, X_act_train, X_num_train, mask_train, y_train,
                     X_act_test, X_num_test, mask_test, y_test, n_classes, pad_idx, max_len,
                     device, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    n_numeric = X_num_train.shape[-1]
    model = _build_deep_model(model_name, n_classes, pad_idx, n_numeric, max_len).to(device)

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = class_counts > 0
    class_weights[nonzero] = class_counts.sum() / class_counts[nonzero]
    if class_weights[nonzero].mean() > 0:
        class_weights[nonzero] = class_weights[nonzero] / class_weights[nonzero].mean()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(PrefixDataset(X_act_train, X_num_train, mask_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(max_epochs):
        for xb_act, xb_num, xb_mask, yb in train_loader:
            xb_act, xb_num, xb_mask, yb = xb_act.to(device), xb_num.to(device), xb_mask.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb_act, xb_num, xb_mask), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loader = DataLoader(PrefixDataset(X_act_test, X_num_test, mask_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    preds = []
    with torch.no_grad():
        for xb_act, xb_num, xb_mask, _ in test_loader:
            xb_act, xb_num, xb_mask = xb_act.to(device), xb_num.to(device), xb_mask.to(device)
            preds.extend(model(xb_act, xb_num, xb_mask).argmax(dim=1).cpu().numpy().tolist())
    return np.array(preds, dtype=np.int64)


def score_prefix_predictions(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def run_prefix_experiment(segments: pd.DataFrame, label_mode: str, sensor_features: list[str], feature_modes, deep_models,
                           max_epochs: int, device, seed: int = SEED):
    """Runs every (feature_mode, predictor) combination with LOGO CV for
    one already-built segment table. Returns (summary_rows, prediction_rows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    seq_acts, seq_nums, y, groups, meta, label_names, act_to_id, id_to_act, numeric_cols = build_prefix_examples(segments, sensor_features)

    n_classes = len(label_names)
    labels_ids = list(range(n_classes))
    pad_idx = n_classes
    max_len = max(len(s) for s in seq_acts)
    n_numeric = len(numeric_cols)

    print(f"\nlabel_mode={label_mode}: {len(y)} expanding-prefix examples, max_prefix_len={max_len}, n_classes={n_classes}")

    X_act_all, mask_all = _pad_activity_sequences(seq_acts, max_len, pad_idx)
    X_num_all_raw = _pad_numeric_sequences(seq_nums, max_len, n_numeric)

    summary_rows, prediction_rows = [], []
    logo = LeaveOneGroupOut()

    for feature_mode in feature_modes:
        use_numeric = feature_mode == "activity_plus_segment_features"
        predictors = ["majority", "markov_last", "suffix_backoff"] + list(deep_models)

        for predictor in predictors:
            y_true_all, y_pred_all, held_group_all, row_meta_all = [], [], [], []

            for tr, te in logo.split(X_act_all, y, groups):
                held_group = int(groups[te][0])
                X_act_train, X_act_test = X_act_all[tr], X_act_all[te]
                mask_train, mask_test = mask_all[tr], mask_all[te]
                y_train, y_test = y[tr], y[te]
                X_num_train, X_num_test = _preprocess_numeric_fold(X_num_all_raw[tr], X_num_all_raw[te], mask_train, mask_test, use_numeric)

                if predictor == "majority":
                    y_pred = predict_majority(y_train, len(y_test))
                elif predictor == "markov_last":
                    y_pred = predict_markov_last(X_act_train, y_train, X_act_test, pad_idx)
                elif predictor == "suffix_backoff":
                    tables, fallback = train_suffix_tables(X_act_train, y_train, pad_idx, MAX_SUFFIX_ORDER)
                    y_pred = predict_suffix_backoff(X_act_test, tables, fallback, pad_idx, MAX_SUFFIX_ORDER)
                elif predictor in deep_models:
                    y_pred = train_eval_deep(predictor, X_act_train, X_num_train, mask_train, y_train,
                                              X_act_test, X_num_test, mask_test, y_test, n_classes, pad_idx, max_len,
                                              device, max_epochs=max_epochs)
                else:
                    raise ValueError(predictor)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred.tolist())
                held_group_all.extend([held_group] * len(y_test))
                row_meta_all.extend(meta.iloc[te].to_dict("records"))

            y_true_all, y_pred_all = np.array(y_true_all, dtype=np.int64), np.array(y_pred_all, dtype=np.int64)
            pooled = score_prefix_predictions(y_true_all, y_pred_all, labels_ids)

            summary_rows.append({
                "label_mode": label_mode, "feature_mode": feature_mode, "predictor": predictor,
                "n_samples": int(len(y_true_all)), "n_groups": int(len(np.unique(groups))), "n_classes": n_classes,
                "accuracy": pooled["accuracy"], "macro_f1": pooled["macro_f1"], "weighted_f1": pooled["weighted_f1"],
                "max_prefix_len": int(max_len),
            })

            for m, yt, yp, hg in zip(row_meta_all, y_true_all, y_pred_all, held_group_all):
                prediction_rows.append({
                    "label_mode": label_mode, "feature_mode": feature_mode, "predictor": predictor, "held_group": int(hg),
                    "prefix_len": int(m["prefix_len"]), "prefix_activities": m["prefix_activities"],
                    "true_activity": id_to_act[int(yt)], "predicted_activity": id_to_act[int(yp)], "correct": bool(int(yt) == int(yp)),
                })

            print(f"  feature_mode={feature_mode} predictor={predictor}: acc={pooled['accuracy']:.3f} macroF1={pooled['macro_f1']:.3f}")

    return summary_rows, prediction_rows


def compute_prefix_fold_std(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Cell 44. Held-out-group mean +/- SD per (label_mode, feature_mode, predictor)."""
    rows = []
    for (label_mode, feature_mode, predictor), subset in pred_df.groupby(["label_mode", "feature_mode", "predictor"]):
        labels = sorted(subset["true_activity"].astype(str).unique())
        for held_group, group_df in subset.groupby("held_group"):
            rows.append({
                "label_mode": label_mode, "feature_mode": feature_mode, "predictor": predictor, "held_group": held_group,
                "n": len(group_df),
                "accuracy": accuracy_score(group_df["true_activity"], group_df["predicted_activity"]),
                "macro_f1": f1_score(group_df["true_activity"], group_df["predicted_activity"], labels=labels, average="macro", zero_division=0),
            })

    fold_metrics = pd.DataFrame(rows)
    return fold_metrics.groupby(["label_mode", "feature_mode", "predictor"], as_index=False).agg(
        fold_accuracy_mean=("accuracy", "mean"), fold_accuracy_std=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"), fold_macro_f1_std=("macro_f1", "std"),
        n_folds=("held_group", "nunique"),
    )


def run_all(data_root: str, out_dir: str, label_modes=LABEL_MODES, feature_modes=FEATURE_MODES,
            deep_models=DEEP_MODELS, max_epochs: int = MAX_EPOCHS, min_windows: int = MIN_WINDOWS, seed: int = SEED):
    """Orchestrates cells 43-44. Returns (summary_df, pred_df, fold_std, check)."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    inv_original, features_df = load_inventory_and_features(data_root)
    sensor_features = detect_sensor_features(features_df)
    print("Sensor features found:", len(sensor_features))

    all_summary, all_predictions = [], []
    for label_mode in label_modes:
        print(f"\n{'#' * 100}\nLABEL MODE: {label_mode}\n{'#' * 100}")
        inv_labelled = make_fine_13_labels(inv_original, min_windows) if label_mode == "fine_13" else make_coarse_4_labels(inv_original)
        segments = build_segments_for_label_mode(inv_labelled, label_mode, features_df, sensor_features)
        print("Total segments:", len(segments))

        summary_rows, prediction_rows = run_prefix_experiment(segments, label_mode, sensor_features, feature_modes, deep_models, max_epochs, device, seed=seed)
        all_summary.extend(summary_rows)
        all_predictions.extend(prediction_rows)

    summary_df = pd.DataFrame(all_summary).sort_values(["label_mode", "macro_f1"], ascending=[True, False]).reset_index(drop=True)
    pred_df = pd.DataFrame(all_predictions)

    summary_df.to_csv(os.path.join(out_dir, "task3_expanding_prefix_segment_prediction_summary.csv"), index=False)
    pred_df.to_csv(os.path.join(out_dir, "task3_expanding_prefix_segment_prediction_predictions.csv"), index=False)

    pivot = summary_df.pivot_table(index=["label_mode", "feature_mode"], columns="predictor", values="macro_f1")
    pivot.to_csv(os.path.join(out_dir, "task3_expanding_prefix_segment_prediction_macro_f1_pivot.csv"))

    fold_std = compute_prefix_fold_std(pred_df)
    fold_std.to_csv(os.path.join(out_dir, "task3_expanding_prefix_segment_prediction_fold_std.csv"), index=False)

    print("\n" + "=" * 100)
    print("EXPANDING-PREFIX SEGMENT PREDICTION SUMMARY")
    print("=" * 100)
    print(summary_df.round(3).to_string(index=False))

    check_rows = []
    for (label_mode, feature_mode, predictor), ref in REPORT_REFERENCE.items():
        row = summary_df[(summary_df["label_mode"] == label_mode) & (summary_df["feature_mode"] == feature_mode) & (summary_df["predictor"] == predictor)]
        if row.empty:
            check_rows.append({"label_mode": label_mode, "feature_mode": feature_mode, "predictor": predictor, "match": "MISSING"})
            continue
        row = row.iloc[0]
        match = (
            "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0015 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0015
                        and abs(row["weighted_f1"] - ref["weighted_f1"]) < 0.0015 and int(row["n_samples"]) == ref["n"])
            else "DIFFERS"
        )
        check_rows.append({
            "label_mode": label_mode, "feature_mode": feature_mode, "predictor": predictor,
            "n": int(row["n_samples"]), "report_n": ref["n"],
            "accuracy": round(row["accuracy"], 3), "report_accuracy": ref["accuracy"],
            "macro_f1": round(row["macro_f1"], 3), "report_macro_f1": ref["macro_f1"],
            "match": match,
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.5")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "task3_expanding_prefix_report_reproduction_check.csv"), index=False)

    n_exact = int((check["match"] == "EXACT").sum())
    print(f"\n{n_exact} of {len(check)} referenced Table 8.5 rows reproduce exactly.")
    if len(summary_df) > 0:
        print("\nNote: neural predictors (lstm/cnn1d/transformer) are stochastic even with a fixed seed "
              "(fresh model + optimizer per LOGO fold) — a DIFFERS verdict for those specifically is expected "
              "even on the real token table, not just on synthetic smoke-test data.")

    return summary_df, pred_df, fold_std, check
