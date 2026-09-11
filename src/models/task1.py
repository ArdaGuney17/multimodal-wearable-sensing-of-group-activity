"""Task 1 — Interaction Detection (binary: interaction vs. non_interaction).

Ported from the Colab notebook `github notebooks/actual ones/
task1_full_comparison_classical_elapsed_dl_with_std.ipynb` (see
docs/table_to_source_mapping.md). Shared modelling code (classical LOGO
pipeline, DL sequence pipeline, feature classification, publication tables)
lives in src/models/common.py — this file has just Task 1's own config
default and its two extra "exact reproduction" paths (cells 16-17 in the
source notebook), which don't exist in Task 2.

Reproduces (once fed the right input feature CSVs — see --help):
  - Table 7.1 / 7.2 (thesis Ch.7 §7.3): classical vs. DL comparison across all
    7 sensor combinations, with/without elapsed-time, LOGO evaluation.
  - The "exact optimized Transformer" headline number (OPTI2_RELATIVE_ONLY,
    seq=18, k=120, seed=42, pooled macro-F1 ≈ 0.8062) via --run-exact.
  - The naive-5-cohort sensitivity rerun of that same config via --run-naive5
    (feeds Table 7.11-7.13 / Table 10.1).

INPUT DEPENDENCY: this script consumes pre-computed per-window feature CSVs
(e.g. binary_5s_specialized_oe_merged_all_features.csv). It does NOT compute
features from raw sensor data — that's a separate, not-yet-ported stage (see
src/features/, still empty). Until that stage exists, point --data-root at a
copy of the original Drive feature-CSV folders to validate this script.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler

from src.models.common import (
    DEVICE,
    NAIVE_GROUPS,
    REPO_ROOT,
    Config,
    TransformerClassifier,
    build_publication_tables,
    build_task_specs,
    make_loader,
    make_sequences,
    run_all_classical,
    run_all_dl,
    set_seed,
)


def _default_config(**overrides) -> Config:
    overrides.setdefault("run_tasks", ["interaction_vs_noninteraction"])
    cfg = Config(**overrides)
    if not cfg.out_dir:
        cfg.out_dir = os.path.join(cfg.data_root, "PUBLICATION_TASK1_FULL_COMPARISON")
    return cfg


# =====================================================================
# Exact optimized Transformer reproduction (thesis headline number)
# =====================================================================

def _exact_repro_shared(data_root):
    """Shared setup for --run-exact / --run-naive5: load the specialized-OE
    feature CSV and recover the exact OPTI2_RELATIVE_ONLY feature set."""
    data_path = os.path.join(data_root, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "binary_5s_specialized_oe_merged_all_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found:\n{data_path}")

    df = pd.read_csv(data_path)
    tokens = ["dist", "spread", "area", "speed", "active_speed", "pair", "nearest", "farthest", "triangle"]

    def is_relative_opti2(column):
        name = str(column)
        lower = name.lower()
        return (name.startswith("opti2_") or name.startswith("opti2__")) and any(t in lower for t in tokens)

    features = []
    for column in df.columns:
        if not is_relative_opti2(column) or column == "elapsed_min":
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() < 20 or np.nanstd(values) < 1e-12:
            continue
        features.append(column)

    print("Dataset shape:", df.shape, "| OPTI2_RELATIVE_ONLY features:", len(features))
    if len(features) != 211:
        raise RuntimeError(
            f"The exact historical feature set was not recovered (expected 211, found {len(features)}). "
            "This assertion is pinned to the thesis's exact feature-engineering output — if your "
            "regenerated feature CSV differs, that's a real signal something upstream changed, not "
            "a bug to silently relax."
        )
    return df, features


def _impute_from_training(X_train, X_test):
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, medians)
    X_test = np.where(np.isfinite(X_test), X_test, medians)
    return X_train, X_test


def _predict_loader_exact(model, loader):
    model.eval()
    true_values, predictions = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            predictions.extend(pred.tolist())
            true_values.extend(yb.numpy().tolist())
    return np.asarray(true_values), np.asarray(predictions)


def _train_exact_fold(X_train_sequences, y_train_sequences, group_train_sequences, X_test_sequences, y_test_sequences, seed, batch_size):
    set_seed(seed)  # exact old behavior: reset seed 42 for every LOGO fold

    validation_group = sorted(np.unique(group_train_sequences))[-1]  # exact old rule: largest remaining group ID
    fit_mask = group_train_sequences != validation_group
    validation_mask = group_train_sequences == validation_group

    if fit_mask.sum() < 50 or validation_mask.sum() < 50:  # historical fallback, normally not triggered
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y_train_sequences))
        rng.shuffle(indices)
        n_validation = max(50, int(0.15 * len(indices)))
        fit_mask = np.zeros(len(indices), dtype=bool)
        validation_mask = np.zeros(len(indices), dtype=bool)
        validation_mask[indices[:n_validation]] = True
        fit_mask[indices[n_validation:]] = True

    X_fit, y_fit = X_train_sequences[fit_mask], y_train_sequences[fit_mask]
    X_validation, y_validation = X_train_sequences[validation_mask], y_train_sequences[validation_mask]

    train_loader = make_loader(X_fit, y_fit, batch_size=batch_size, shuffle=True)
    validation_loader = make_loader(X_validation, y_validation, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(X_test_sequences, y_test_sequences, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(input_dim=X_train_sequences.shape[-1], n_classes=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.25).to(DEVICE)

    class_counts = np.bincount(y_fit, minlength=2).astype(float)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-5)

    best_state, best_validation_macro_f1, best_epoch, bad_epochs = None, -np.inf, 0, 0
    for epoch in range(1, 80 + 1):  # MAX_EPOCHS
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_function(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        validation_true, validation_pred = _predict_loader_exact(model, validation_loader)
        validation_macro_f1 = f1_score(validation_true, validation_pred, average="macro", zero_division=0)
        scheduler.step(validation_macro_f1)

        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1, best_epoch, bad_epochs = validation_macro_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= 12:  # PATIENCE
            break

    if best_state is None:
        raise RuntimeError("No valid model state was saved.")
    model.load_state_dict(best_state)
    test_true, test_pred = _predict_loader_exact(model, test_loader)
    return {"y_true": test_true, "y_pred": test_pred, "validation_group": validation_group, "best_epoch": best_epoch, "best_validation_macro_f1": best_validation_macro_f1}


def run_exact_reproduction(cfg: Config):
    """Reproduces the thesis's headline Task 1 number: OPTI2_RELATIVE_ONLY
    feature set, Transformer, seq_len=18 (90s context), k=120, seed=42.
    Target: pooled accuracy/macro-F1/balanced-accuracy ≈ 0.8064/0.8062/0.8065."""
    seed, seq_len, k, batch_size = 42, 18, 120, 64
    df, features = _exact_repro_shared(cfg.data_root)

    out_dir = os.path.join(cfg.data_root, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "PUBLICATION_TASK1_OPTIMIZED_0806")
    os.makedirs(out_dir, exist_ok=True)

    label_to_id = {"non_interaction": 0, "interaction": 1}
    y = df["binary_label"].map(label_to_id).to_numpy(dtype=int)
    groups = df["group"].to_numpy()
    starts = pd.to_numeric(df["window_start"], errors="raise").to_numpy(dtype=float)
    X_raw = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    logo = LeaveOneGroupOut()
    all_true, all_pred, fold_rows = [], [], []

    for fold, (train_indices, test_indices) in enumerate(logo.split(X_raw, y, groups), start=1):
        test_group = groups[test_indices][0]
        X_train, X_test = _impute_from_training(X_raw[train_indices], X_raw[test_indices])
        scaler = RobustScaler()
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

        actual_k = min(k, X_train.shape[1] - 1)
        selector = SelectKBest(score_func=f_classif, k=actual_k)
        X_train = selector.fit_transform(X_train, y[train_indices])
        X_test = selector.transform(X_test)

        X_train_sequences, y_train_sequences, group_train_sequences, _ = make_sequences(X_train, y[train_indices], groups[train_indices], starts[train_indices], seq_len)
        X_test_sequences, y_test_sequences, _, _ = make_sequences(X_test, y[test_indices], groups[test_indices], starts[test_indices], seq_len)

        print(f"Fold {fold}/9 | test group={test_group} | train sequences={len(X_train_sequences)} | test sequences={len(X_test_sequences)}")
        result = _train_exact_fold(X_train_sequences, y_train_sequences, group_train_sequences, X_test_sequences, y_test_sequences, seed, batch_size)

        y_true_fold, y_pred_fold = result["y_true"], result["y_pred"]
        fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
        fold_macro_f1 = f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
        fold_balanced_accuracy = balanced_accuracy_score(y_true_fold, y_pred_fold)
        fold_rows.append({
            "fold": fold, "test_group": test_group, "validation_group": result["validation_group"],
            "n_test_sequences": len(y_true_fold), "accuracy": fold_accuracy, "macro_f1": fold_macro_f1,
            "balanced_accuracy": fold_balanced_accuracy, "best_epoch": result["best_epoch"],
            "best_validation_macro_f1": result["best_validation_macro_f1"],
        })
        all_true.extend(y_true_fold.tolist()); all_pred.extend(y_pred_fold.tolist())
        print(f"  accuracy={fold_accuracy:.4f} | macro-F1={fold_macro_f1:.4f} | balanced={fold_balanced_accuracy:.4f} | best_epoch={result['best_epoch']}")

    all_true, all_pred = np.asarray(all_true), np.asarray(all_pred)
    fold_metrics_0806 = pd.DataFrame(fold_rows)

    if len(all_true) != 4426:
        raise RuntimeError(
            f"Historical sequence count mismatch (expected 4426, found {len(all_true)}). "
            "Pinned to the thesis's exact dataset — a real difference upstream, not to be relaxed silently."
        )

    summary_0806 = pd.DataFrame([{
        "run_name": "OPTI2_REL_seq18_k120_transformer_seed42", "feature_set": "OPTI2_RELATIVE_ONLY",
        "model": "Transformer", "seed": seed, "sequence_length": seq_len, "context_seconds": 90,
        "available_features": len(features), "selected_features": min(k, X_raw.shape[1] - 1),
        "n_sequences_evaluated": len(all_true),
        "pooled_accuracy": accuracy_score(all_true, all_pred),
        "pooled_macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "pooled_balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "fold_accuracy_mean": fold_metrics_0806["accuracy"].mean(), "fold_accuracy_std": fold_metrics_0806["accuracy"].std(ddof=1),
        "fold_macro_f1_mean": fold_metrics_0806["macro_f1"].mean(), "fold_macro_f1_std": fold_metrics_0806["macro_f1"].std(ddof=1),
        "fold_balanced_accuracy_mean": fold_metrics_0806["balanced_accuracy"].mean(), "fold_balanced_accuracy_std": fold_metrics_0806["balanced_accuracy"].std(ddof=1),
        "n_folds": fold_metrics_0806["test_group"].nunique(),
    }])

    print("\nOPTIMIZED HISTORICAL TASK 1 RESULT")
    print(summary_0806.round(4).to_string(index=False))
    summary_0806.to_csv(os.path.join(out_dir, "task1_optimized_0806_summary_with_std.csv"), index=False)
    fold_metrics_0806.to_csv(os.path.join(out_dir, "task1_optimized_0806_fold_metrics.csv"), index=False)
    pd.DataFrame({"feature_name": features}).to_csv(os.path.join(out_dir, "task1_optimized_0806_exact_features.csv"), index=False)

    print("\nHistorical target: pooled accuracy ~0.8064, macro-F1 ~0.8062, balanced acc. ~0.8065")
    return df, features, summary_0806, fold_metrics_0806, out_dir


def run_naive5_reproduction(cfg: Config, df=None, features=None):
    """Reruns the exact-reproduction config restricted to the 5 fully-naive
    groups (no researcher participation): {2, 3, 5, 6, 10}. Feeds Table 7.11-
    7.13 / Table 10.1's naive-cohort comparison. If df/features aren't passed
    (i.e. --run-naive5 without --run-exact), recomputes them from scratch."""
    seed, seq_len, k, batch_size = 42, 18, 120, 64
    if df is None or features is None:
        df, features = _exact_repro_shared(cfg.data_root)

    def _gid(g):
        digits = "".join(ch for ch in str(g) if ch.isdigit())
        return int(digits) if digits else -1

    mask_n = df["group"].map(_gid).isin(NAIVE_GROUPS).to_numpy()
    dfn = df.loc[mask_n].copy()
    print(f"rows: {len(df)} -> {len(dfn)}   naive groups: {sorted(set(dfn['group'].map(_gid)))}")

    label_to_id = {"non_interaction": 0, "interaction": 1}
    yn = dfn["binary_label"].map(label_to_id).to_numpy(dtype=int)
    groupsn = dfn["group"].to_numpy()
    startsn = pd.to_numeric(dfn["window_start"], errors="raise").to_numpy(dtype=float)
    Xn_raw = dfn[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    logo = LeaveOneGroupOut()
    all_true_n, all_pred_n, fold_rows_n = [], [], []

    for fold, (tr, te) in enumerate(logo.split(Xn_raw, yn, groupsn), start=1):
        test_group = groupsn[te][0]
        X_train, X_test = _impute_from_training(Xn_raw[tr], Xn_raw[te])
        scaler = RobustScaler()
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

        actual_k = min(k, X_train.shape[1] - 1)
        selector = SelectKBest(score_func=f_classif, k=actual_k)
        X_train = selector.fit_transform(X_train, yn[tr])
        X_test = selector.transform(X_test)

        Xtr_seq, ytr_seq, gtr_seq, _ = make_sequences(X_train, yn[tr], groupsn[tr], startsn[tr], seq_len)
        Xte_seq, yte_seq, _, _ = make_sequences(X_test, yn[te], groupsn[te], startsn[te], seq_len)
        if len(Xtr_seq) == 0 or len(Xte_seq) == 0:
            print(f"Fold {fold}: empty sequence, skipping")
            continue

        print(f"Fold {fold}/5 | test group={test_group} | train seq={len(Xtr_seq)} | test seq={len(Xte_seq)}")
        res = _train_exact_fold(Xtr_seq, ytr_seq, gtr_seq, Xte_seq, yte_seq, seed, batch_size)
        yt, yp = res["y_true"], res["y_pred"]
        all_true_n.extend(yt.tolist()); all_pred_n.extend(yp.tolist())
        fold_rows_n.append({
            "fold": fold, "test_group": _gid(test_group), "validation_group": _gid(res["validation_group"]),
            "n_test_sequences": len(yt), "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "best_epoch": res["best_epoch"], "best_validation_macro_f1": res["best_validation_macro_f1"],
        })

    fold_n = pd.DataFrame(fold_rows_n).sort_values("test_group").reset_index(drop=True)
    at, ap = np.array(all_true_n), np.array(all_pred_n)
    pooled_acc, pooled_mf1, pooled_bal = accuracy_score(at, ap), f1_score(at, ap, average="macro", zero_division=0), balanced_accuracy_score(at, ap)

    v = fold_n["macro_f1"].to_numpy(dtype=float)
    m, sd = v.mean(), v.std(ddof=1)
    t4 = 2.776  # 5-fold, 4 df, per docs/thesis_reproduction_targets.md §4
    lo, hi = m - t4 * sd / np.sqrt(len(v)), m + t4 * sd / np.sqrt(len(v))

    print("\n" + "=" * 78)
    print("TASK 1 OPTIMIZED (OPTI2_RELATIVE_ONLY, Transformer, seq=18, k=120) — NAIVE 5")
    print("=" * 78)
    print(fold_n[["test_group", "n_test_sequences", "accuracy", "macro_f1", "balanced_accuracy"]].to_string(index=False))
    print(f"\npooled  A={pooled_acc:.4f}  M={pooled_mf1:.4f}  B={pooled_bal:.4f}")
    print(f"fold    mean={m:.4f} +/- {sd:.4f}   95% CI=[{max(lo, 0):.4f}, {min(hi, 1):.4f}]  (n={len(v)})")
    print("\nfull-9 reference:  A=0.8064  M=0.8062  B=0.8065  fold 0.781+/-0.086 [0.716, 0.847]")
    print(f"dM = {pooled_mf1 - 0.8062:+.4f}")

    out_dir = os.path.join(cfg.data_root, "INTERACTION_BINARY_5S_SPECIALIZED_OE", "PUBLICATION_TASK1_OPTIMIZED_0806")
    os.makedirs(out_dir, exist_ok=True)
    fold_n.to_csv(os.path.join(out_dir, "task1_optimized_0806_fold_metrics_NAIVE5.csv"), index=False)
    print(f"\nsaved -> {os.path.join(out_dir, 'task1_optimized_0806_fold_metrics_NAIVE5.csv')}")
    return fold_n


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "processed"), help="Folder containing the feature CSVs (see module docstring for exact expected filenames).")
    parser.add_argument("--out-dir", default=None, help="Output folder for results (default: <data-root>/PUBLICATION_TASK1_FULL_COMPARISON).")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--run-exact", action="store_true", help="Run the exact-reproduction Transformer config (thesis headline number).")
    parser.add_argument("--run-naive5", action="store_true", help="Run the naive-5-cohort sensitivity rerun (implies --run-exact's feature recovery).")
    parser.add_argument("--full-grid", action="store_true", help="Use the wider exploratory classical grid instead of the report-reproduction one.")
    parser.add_argument("--resume", action="store_true", help="Skip re-running classical/DL configurations whose output CSVs already exist.")
    parser.add_argument("--max-logo-folds", type=int, default=None, help="Cap LOGO folds for a quick smoke test (e.g. 2).")
    args = parser.parse_args()

    cfg = _default_config(
        data_root=args.data_root,
        out_dir=args.out_dir or "",
        run_classical=not args.skip_classical,
        run_dl=not args.skip_dl,
        report_reproduction_mode=not args.full_grid,
        resume_existing=args.resume,
        max_logo_folds=args.max_logo_folds,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Device:", DEVICE)
    print("Data root:", cfg.data_root)
    print("Output dir:", cfg.out_dir)

    classical_result, dl_result = {}, {}
    if cfg.run_classical or cfg.run_dl:
        task_specs = build_task_specs(cfg)
        if cfg.run_classical:
            classical_result = run_all_classical(task_specs, cfg)
        if cfg.run_dl:
            dl_result = run_all_dl(task_specs, cfg)
        if cfg.run_classical and cfg.run_dl:
            build_publication_tables(classical_result, dl_result, cfg.out_dir)

    exact_df, exact_features = None, None
    if args.run_exact:
        exact_df, exact_features, _, _, _ = run_exact_reproduction(cfg)
    if args.run_naive5:
        run_naive5_reproduction(cfg, df=exact_df, features=exact_features)


if __name__ == "__main__":
    main()
