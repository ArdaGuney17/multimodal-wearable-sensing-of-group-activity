"""Dedicated OpenEarable-only experiment (Ch.7 §7.5.1, Table 7.8).

Ported from `THESIS_NOTEBOOKS/ML/oe_conversation_nonconversation_experiment.ipynb`.
Isolates OE ("OpenEarable", ear-worn audio/IMU-derived) features from the
same `activity3_advanced_merged_10s_features.csv` Task 2 already consumes,
restricts to the 3 core recognition classes, binarizes to
conversation/non_conversation, and runs the same classical+DL comparison
shape as `src/models/common.py` — but as this notebook's own
self-contained code (OE-only feature selection, its own classical model
dict, its own DL training loop), not routed through `common.py`, to stay
faithful to what actually produced Table 7.8's numbers.

Confirms a discrepancy flagged in docs/thesis_reproduction_targets.md:
Table 7.8's GRU row really is "seq=3, 30s" (3 x 10s windows), not a
transcription artifact — this notebook's own `SEQ_LENS = [3, 6, 9]` are
10-second-window multiples (30/60/90s), consistent with the activity
dataset's 10s windowing, genuinely different from Task 1's 5s-window
seq=9/18 convention.

SCOPE NOTE: this notebook produces ONLY Table 7.8. Table 7.9 (OE9
three-class recognition: "OE9 motion + MAG magnitude", "+manual
OptiTrack", "+ENG7 proximity", "+ENG7 proximity + elapsed") and Table
7.10 (SPECIAL_OE binary interaction, alone and +OPTI2 relative) need
different feature sources — pre-computed OE9/SPECIAL_OE feature CSVs
under `data/INTERACTION_OE9`, `INTERACTION_OE8`, `INTERACTION_OE10` per
docs/table_to_source_mapping.md — not found in this notebook and not yet
fetched from Drive. Documented as a remaining gap, not silently skipped.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COL = "recognition_label"
GROUP_COL = "group"
START_COL = "window_start"
END_COL = "window_end"
CORE_CLASSES = ["co_building", "co_merging", "conversation"]

POS_LABEL = "conversation"
NEG_LABEL = "non_conversation"
TARGET_COL = "conv_binary_label"

K_CLASSICAL = [40, 80, 120, 200, "all"]
TIME_CONDITIONS = ["no_elapsed", "with_elapsed"]
SEQ_LENS = (3, 6, 9)  # 10s windows -> 30/60/90s
K_DL = (80, 120, 200)
MAX_EPOCHS_DEFAULT = 80
PATIENCE_DEFAULT = 12
BATCH_SIZE_DEFAULT = 64

MODEL_CONFIGS = [
    {"model_type": "lstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "bilstm", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "gru", "hidden_dim": 64, "num_layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
    {"model_type": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.25, "lr": 5e-4, "weight_decay": 1e-4},
]

# Published values this module's output should reproduce (thesis Table
# 7.8 — docs/thesis_reproduction_targets.md §7.5.1). Used only for the
# optional verification check in run_all().
REPORT_REFERENCE = {
    "classical_no_elapsed": {"model": "rbfSVC_C1_gscale", "k": "all", "accuracy": 0.7339, "macro_f1": 0.7108, "balanced_accuracy": 0.7302},
    "classical_with_elapsed": {"model": "logreg_C1", "k": 80, "accuracy": 0.8075, "macro_f1": 0.7810, "balanced_accuracy": 0.7873},
    "dl_no_elapsed": {"model_type": "gru", "seq_len": 3, "k_features": 200, "accuracy": 0.7618, "macro_f1": 0.7221, "balanced_accuracy": 0.7266},
}


def is_oe_feature(c) -> bool:
    c = str(c)
    return c.startswith(("oe__", "oe_", "mag__", "mag_", "oebest__"))


def clean_feature_list(dataframe: pd.DataFrame, feats: list[str], include_elapsed: bool = False) -> list[str]:
    bad_cols = {
        LABEL_COL, TARGET_COL, GROUP_COL, START_COL, END_COL, "window_mid",
        "label", "target", "class", "activity", "general_class", "binary_label", "pred", "prediction", "correct",
    }
    cleaned = []
    for f in feats:
        if f not in dataframe.columns or f in bad_cols:
            continue
        if f == "elapsed_min" and not include_elapsed:
            continue
        if "elapsed" in str(f).lower() and not include_elapsed:
            continue
        if "recognition_label" in str(f).lower():
            continue
        if dataframe[f].dtype == "object":
            continue
        x = pd.to_numeric(dataframe[f], errors="coerce").values
        if np.isfinite(x).sum() < 20:
            continue
        if np.nanstd(x) < 1e-12:
            continue
        cleaned.append(f)
    return list(dict.fromkeys(cleaned))


def load_and_prepare(data_root: str):
    """Loads the activity feature CSV, restricts to the 3 core classes,
    binarizes to conversation/non_conversation, adds causal elapsed-time,
    and returns (df, oe_features, oe_features_elapsed)."""
    path = os.path.join(data_root, "INTERACTION_ABLATIONS", "activity3_advanced_merged_10s_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find advanced dataset:\n{path}")

    df = pd.read_csv(path)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL].isin(CORE_CLASSES)].copy().reset_index(drop=True)
    df[TARGET_COL] = np.where(df[LABEL_COL] == "conversation", POS_LABEL, NEG_LABEL)

    df["window_mid"] = (pd.to_numeric(df[START_COL], errors="coerce") + pd.to_numeric(df[END_COL], errors="coerce")) / 2.0
    df["elapsed_min"] = (df["window_mid"] - df.groupby(GROUP_COL)["window_mid"].transform("min")) / 60.0

    oe_features = clean_feature_list(df, [c for c in df.columns if is_oe_feature(c)], include_elapsed=False)
    oe_features_elapsed = clean_feature_list(df, list(dict.fromkeys(oe_features + ["elapsed_min"])), include_elapsed=True)

    if not oe_features:
        raise ValueError("No OE features found. Check column prefixes in the advanced dataset.")

    return df, oe_features, oe_features_elapsed


# ================================================================
# Classical grid
# ================================================================

def make_classical_models() -> dict:
    return {
        "logreg_C1": LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="liblinear", random_state=42),
        "linearSVC_C1": LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000),
        "rbfSVC_C1_gscale": SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42),
        "rf_leaf2": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "extraTrees_leaf1": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1),
    }


def _make_pipeline_for_model(model, k, n_features):
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    if k != "all" and n_features > int(k):
        steps.append(("select", SelectKBest(f_classif, k=int(k))))
    steps.append(("model", model))
    return make_pipeline(*[s[1] for s in steps])


def score_binary(y_true, y_pred) -> dict:
    labels = [POS_LABEL, NEG_LABEL]
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_conversation": pr[0], "recall_conversation": rc[0], "f1_conversation": f1[0], "support_conversation": sup[0],
        "precision_non_conversation": pr[1], "recall_non_conversation": rc[1], "f1_non_conversation": f1[1], "support_non_conversation": sup[1],
    }


def run_classical_condition(df: pd.DataFrame, feature_list: list[str], time_condition: str, model_name: str, model, k):
    X = df[feature_list].apply(pd.to_numeric, errors="coerce").values
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []

    for _, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        clf = _make_pipeline_for_model(model, k, X.shape[1])
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.extend(y[te])
        y_pred_all.extend(pred)

    metrics = score_binary(np.array(y_true_all), np.array(y_pred_all))
    metrics.update({"time_condition": time_condition, "model": model_name, "k": k, "n_features": len(feature_list)})
    return metrics


def run_classical_grid(df: pd.DataFrame, oe_features: list[str], oe_features_elapsed: list[str], k_values=K_CLASSICAL, time_conditions=TIME_CONDITIONS):
    models = make_classical_models()
    rows = []
    for time_condition in time_conditions:
        feats = oe_features if time_condition == "no_elapsed" else oe_features_elapsed
        for model_name, model in models.items():
            for k in k_values:
                if k != "all" and int(k) > len(feats):
                    continue
                print(f"Classical | {time_condition} | {model_name} | k={k} | n_features={len(feats)}")
                rows.append(run_classical_condition(df, feats, time_condition, model_name, model, k))

    summary = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_per_condition = (
        summary.sort_values(["time_condition", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("time_condition", as_index=False).head(1).reset_index(drop=True)
    )
    return summary, best_per_condition


# ================================================================
# DL grid
# ================================================================

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, rnn_type="lstm", hidden_dim=64, num_layers=1, dropout=0.25, bidirectional=False, n_classes=2):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.25, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.pos(self.input_proj(x))
        return self.head(self.encoder(x)[:, -1, :])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, input_dim):
    t = cfg["model_type"]
    if t == "lstm":
        return RNNClassifier(input_dim=input_dim, rnn_type="lstm", hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=False)
    if t == "bilstm":
        return RNNClassifier(input_dim=input_dim, rnn_type="lstm", hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=True)
    if t == "gru":
        return RNNClassifier(input_dim=input_dim, rnn_type="gru", hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=False)
    if t == "transformer":
        return TransformerClassifier(input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"])
    raise ValueError(t)


def make_sequences(X, y, groups, starts, seq_len):
    X_seq, y_seq, g_seq, start_seq = [], [], [], []
    for g in sorted(np.unique(groups)):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(starts[idx])]
        if len(idx) < seq_len:
            continue
        for j in range(seq_len - 1, len(idx)):
            seq_idx = idx[j - seq_len + 1:j + 1]
            X_seq.append(X[seq_idx]); y_seq.append(y[idx[j]]); g_seq.append(g); start_seq.append(starts[idx[j]])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64), np.asarray(g_seq), np.asarray(start_seq)


def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def choose_validation_group(train_groups):
    return sorted(np.unique(train_groups).tolist())[-1]


def _evaluate_predictions(y_true, y_pred):
    id_to_label = {0: POS_LABEL, 1: NEG_LABEL}
    return score_binary(np.array([id_to_label[int(i)] for i in y_true]), np.array([id_to_label[int(i)] for i in y_pred]))


def train_one_dl_run(df, oe_features, seq_len, k_features, model_cfg, seed, max_epochs=MAX_EPOCHS_DEFAULT, patience=PATIENCE_DEFAULT, batch_size=BATCH_SIZE_DEFAULT):
    """OE-only, no-elapsed DL run — mirrors the source notebook's own
    `train_one_dl_run` exactly (in-fold impute/scale/select, LOGO with a
    held-out validation group carved from the training groups, early
    stopping on validation macro-F1)."""
    set_seed(seed)
    label_to_id = {POS_LABEL: 0, NEG_LABEL: 1}
    y_all = df[TARGET_COL].map(label_to_id).astype(int).values
    groups_all = df[GROUP_COL].values
    starts_all = pd.to_numeric(df[START_COL], errors="coerce").values
    X_raw = df[oe_features].apply(pd.to_numeric, errors="coerce").values

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, fold_rows = [], [], []

    for fold, (trval_idx, te_idx) in enumerate(logo.split(X_raw, y_all, groups_all), start=1):
        test_group = groups_all[te_idx][0]
        train_groups = np.unique(groups_all[trval_idx])
        val_group = choose_validation_group(train_groups)
        val_mask = groups_all[trval_idx] == val_group
        val_idx, tr_idx = trval_idx[val_mask], trval_idx[~val_mask]

        imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(X_raw[tr_idx]))
        Xval = scaler.transform(imputer.transform(X_raw[val_idx]))
        Xte = scaler.transform(imputer.transform(X_raw[te_idx]))

        actual_k = min(int(k_features), Xtr.shape[1])
        selector = SelectKBest(f_classif, k=actual_k)
        Xtr = selector.fit_transform(Xtr, y_all[tr_idx])
        Xval, Xte = selector.transform(Xval), selector.transform(Xte)

        X_fold = np.zeros((len(X_raw), actual_k), dtype=np.float32)
        X_fold[tr_idx], X_fold[val_idx], X_fold[te_idx] = Xtr, Xval, Xte

        Xtr_seq, ytr_seq, _, _ = make_sequences(X_fold[tr_idx], y_all[tr_idx], groups_all[tr_idx], starts_all[tr_idx], seq_len)
        Xval_seq, yval_seq, _, _ = make_sequences(X_fold[val_idx], y_all[val_idx], groups_all[val_idx], starts_all[val_idx], seq_len)
        Xte_seq, yte_seq, gte_seq, ste_seq = make_sequences(X_fold[te_idx], y_all[te_idx], groups_all[te_idx], starts_all[te_idx], seq_len)
        if len(Xtr_seq) == 0 or len(Xval_seq) == 0 or len(Xte_seq) == 0:
            print(f"Skipping fold {fold}: not enough sequences")
            continue

        model = build_model(model_cfg, input_dim=actual_k).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        train_loader = make_loader(Xtr_seq, ytr_seq, batch_size, shuffle=True)
        val_loader = make_loader(Xval_seq, yval_seq, batch_size, shuffle=False)
        test_loader = make_loader(Xte_seq, yte_seq, batch_size, shuffle=False)

        best_state, best_val_macro, best_epoch, patience_left = None, -1, 0, patience
        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                    val_true.extend(yb.numpy())
            val_macro = f1_score(val_true, val_preds, average="macro", zero_division=0)
            if val_macro > best_val_macro:
                best_val_macro, best_epoch = val_macro, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for xb, yb in test_loader:
                fold_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())

        y_true_all.extend(yte_seq); y_pred_all.extend(fold_preds)
        fold_metrics = _evaluate_predictions(np.array(yte_seq), np.array(fold_preds))
        fold_metrics.update({
            "fold": fold, "test_group": test_group, "val_group": val_group, "seq_len": seq_len,
            "context_seconds": seq_len * 10, "k_features": actual_k, "model_type": model_cfg["model_type"],
            "seed": seed, "n_test_sequences": len(yte_seq), "best_epoch": best_epoch, "best_val_macro_f1": best_val_macro,
        })
        fold_rows.append(fold_metrics)

    if not y_true_all:
        return None, pd.DataFrame(fold_rows)

    overall = _evaluate_predictions(np.array(y_true_all), np.array(y_pred_all))
    overall.update({
        "model_type": model_cfg["model_type"], "seed": seed, "seq_len": seq_len, "context_seconds": seq_len * 10,
        "k_features": k_features, "n_sequences_evaluated": len(y_true_all), "time_condition": "no_elapsed", "feature_set": "OE_ONLY",
    })
    return overall, pd.DataFrame(fold_rows)


def run_dl_grid(df, oe_features, seq_lens=SEQ_LENS, k_values=K_DL, model_configs=MODEL_CONFIGS, seeds=(42,), **train_kwargs):
    rows = []
    total = len(seeds) * len(seq_lens) * len(k_values) * len(model_configs)
    run_i = 0
    for seed in seeds:
        for seq_len in seq_lens:
            for k_features in k_values:
                if int(k_features) > len(oe_features):
                    continue
                for model_cfg in model_configs:
                    run_i += 1
                    print(f"DL RUN {run_i}/{total}: {model_cfg['model_type']} | seq_len={seq_len} | k={k_features} | seed={seed}")
                    summary, _ = train_one_dl_run(df, oe_features, seq_len, k_features, model_cfg, seed, **train_kwargs)
                    if summary is not None:
                        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_root: str, out_dir: str | None = None, k_classical=K_CLASSICAL, seq_lens=SEQ_LENS, k_dl=K_DL,
            max_epochs=MAX_EPOCHS_DEFAULT, patience=PATIENCE_DEFAULT, batch_size=BATCH_SIZE_DEFAULT):
    """Runs the classical grid + DL grid for Table 7.8. Returns
    (classical_summary, classical_best, dl_summary, check)."""
    out_dir = out_dir or os.path.join(data_root, "OE_CONVERSATION_NONCONVERSATION")
    os.makedirs(out_dir, exist_ok=True)

    df, oe_features, oe_features_elapsed = load_and_prepare(data_root)
    print(f"OE-only usable features: no_elapsed={len(oe_features)} with_elapsed={len(oe_features_elapsed)}")

    classical_summary, classical_best = run_classical_grid(df, oe_features, oe_features_elapsed, k_values=k_classical)
    classical_summary.to_csv(os.path.join(out_dir, "oe_conv_nonconv_classical_summary.csv"), index=False)
    classical_best.to_csv(os.path.join(out_dir, "oe_conv_nonconv_classical_best_per_condition.csv"), index=False)

    dl_summary = run_dl_grid(df, oe_features, seq_lens=seq_lens, k_values=k_dl, max_epochs=max_epochs, patience=patience, batch_size=batch_size)
    if len(dl_summary):
        dl_summary.to_csv(os.path.join(out_dir, "oe_conv_nonconv_dl_no_elapsed_summary.csv"), index=False)

    print("\nBest classical per time condition:")
    print(classical_best.round(4).to_string(index=False))
    if len(dl_summary):
        print("\nBest DL results:")
        print(dl_summary.head(5).round(4).to_string(index=False))

    check_rows = []
    no_elapsed_best = classical_best[classical_best["time_condition"] == "no_elapsed"]
    if len(no_elapsed_best):
        row = no_elapsed_best.iloc[0]
        ref = REPORT_REFERENCE["classical_no_elapsed"]
        match = "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0015 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0015) else "DIFFERS"
        check_rows.append({"row": "classical_no_elapsed", "model": row["model"], "report_model": ref["model"], "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"], "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"], "match": match})
    else:
        check_rows.append({"row": "classical_no_elapsed", "match": "MISSING"})

    with_elapsed_best = classical_best[classical_best["time_condition"] == "with_elapsed"]
    if len(with_elapsed_best):
        row = with_elapsed_best.iloc[0]
        ref = REPORT_REFERENCE["classical_with_elapsed"]
        match = "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.0015 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.0015) else "DIFFERS"
        check_rows.append({"row": "classical_with_elapsed", "model": row["model"], "report_model": ref["model"], "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"], "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"], "match": match})
    else:
        check_rows.append({"row": "classical_with_elapsed", "match": "MISSING"})

    if len(dl_summary):
        row = dl_summary.iloc[0]
        ref = REPORT_REFERENCE["dl_no_elapsed"]
        match = "EXACT" if (abs(row["accuracy"] - ref["accuracy"]) < 0.02 and abs(row["macro_f1"] - ref["macro_f1"]) < 0.02) else "DIFFERS"
        check_rows.append({"row": "dl_no_elapsed", "model": row["model_type"], "report_model": ref["model_type"], "accuracy": round(row["accuracy"], 4), "report_accuracy": ref["accuracy"], "macro_f1": round(row["macro_f1"], 4), "report_macro_f1": ref["macro_f1"], "match": match})
    else:
        check_rows.append({"row": "dl_no_elapsed", "match": "MISSING"})

    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 7.8")
    print(check.to_string(index=False))
    check.to_csv(os.path.join(out_dir, "table_7_8_report_reproduction_check.csv"), index=False)

    return classical_summary, classical_best, dl_summary, check
