"""Task 3, Part III — leakage-free neural (Transformer/LSTM) token sequence
models on the same common targets as task3_common_targets.py's primary panel.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "TOKEN SEQUENCE
MODELS" / "RESUMABLE THREE-SEED TOKEN MODEL RUNNER" (source notebook's own
"Part III"). Produces **Table 8.4**'s three neural rows: Transformer
labels-only, Transformer labels+full-stat sensors (the best tokenized
neural result, M=0.424±0.020), and LSTM labels+sensors.

Model: a small causal Transformer or LSTM over the token sequence, predicting
each next label autoregressively from previous labels (teacher-forced, label
input shifted by one and prefixed with a learned start token) optionally
combined with the current token's own sensor-statistics features (in-fold
SelectKBest k=40, exactly matching common_targets/grammar.py's k). Evaluated
under the same 9-fold LOGO + "predict positions 3 onward" protocol as
task3_common_targets.py's primary panel, so results are directly comparable
— this module's own cross-check enforces identical target counts.

Averaged over 3 seeds (42, 1, 7), with two kinds of uncertainty reported
separately (as the thesis does): seed-level SD (population SD, ddof=0 — the
report's convention) and held-out-group SD (sample SD, ddof=1, computed by
first averaging seeds within each group, then taking SD across the 9 group
means).
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKEN_SEEDS = [42, 1, 7]
SEED_STD_DDOF = 0   # population SD across the 3 seeds — the thesis's own convention
FOLD_STD_DDOF = 1   # sample SD across the 9 held-out groups
K_SELECT = 40
PRIMARY_START = 3   # must match task3_common_targets.PRIMARY_START — same protocol


class TokenSeqNet(nn.Module):
    def __init__(self, n_classes, n_features, kind="transformer", d_model=96, n_heads=8, n_layers=3, dropout=0.3, max_len=128):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes + 1, d_model)  # +1 for the start token
        self.feature_projection = nn.Linear(n_features, d_model) if n_features > 0 else None
        self.position = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.kind = kind
        self.dropout = nn.Dropout(dropout)

        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, activation="gelu")
            self.core = nn.TransformerEncoder(layer, n_layers)
        elif kind == "lstm":
            self.core = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        else:
            raise ValueError(kind)

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, label_input, feature_input=None):
        hidden = self.label_embedding(label_input)
        if self.feature_projection is not None and feature_input is not None:
            hidden = hidden + self.feature_projection(feature_input)

        length = hidden.size(1)
        hidden = hidden + self.position[:, :length, :]
        hidden = self.dropout(hidden)

        if self.kind == "transformer":
            causal_mask = torch.triu(torch.ones(length, length, device=hidden.device) * float("-inf"), diagonal=1)
            hidden = self.core(hidden, mask=causal_mask)
        else:
            hidden, _ = self.core(hidden)

        return self.head(hidden)


def run_token_seed(tokens: pd.DataFrame, feature_cols: list, model_kind: str, use_features: bool, seed: int, epochs: int = 100, learning_rate: float = 8e-4, k_select: int = K_SELECT, primary_start: int = PRIMARY_START):
    """One (model_kind, use_features, seed) run: 9-fold LOGO, causal
    next-label prediction, evaluated on token positions primary_start onward.
    Returns (y_true, y_pred, held_group_per_prediction, classes)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    classes = sorted(tokens["label"].astype(str).unique())
    class_to_id = {label: i for i, label in enumerate(classes)}
    n_classes = len(classes)
    start_id = n_classes

    groups = tokens["group"].astype(str).values
    logo = LeaveOneGroupOut()

    all_true, all_pred, all_group = [], [], []

    for train_index, test_index in logo.split(tokens, tokens["label"], groups):
        train_groups = np.unique(groups[train_index])
        validation_group = sorted(train_groups)[-1]  # select validation group before fitting any preprocessing
        fitting_groups = [g for g in train_groups if g != validation_group]
        fitting_index = np.flatnonzero(np.isin(groups, fitting_groups))

        imputer = selector = scaler = None
        selected_feature_count = 0

        if use_features and feature_cols:
            # Imputation/selection/scaling fit ONLY on fitting groups — validation
            # and outer test groups stay completely unseen by preprocessing.
            imputer = SimpleImputer(strategy="median").fit(tokens.iloc[fitting_index][feature_cols])
            transformed_train = imputer.transform(tokens.iloc[fitting_index][feature_cols])

            if k_select and len(feature_cols) > k_select:
                selector = SelectKBest(f_classif, k=k_select).fit(transformed_train, tokens.iloc[fitting_index]["label"].astype(str).values)
                transformed_train = selector.transform(transformed_train)

            scaler = StandardScaler().fit(transformed_train)
            selected_feature_count = transformed_train.shape[1]

        def sequences_for(group_ids):
            sequence_list = []
            for group in group_ids:
                sub = tokens[tokens["group"].astype(str) == str(group)].sort_values("start_time")
                labels = np.asarray([class_to_id[label] for label in sub["label"].astype(str)], dtype=int)

                if use_features and feature_cols:
                    feature_values = imputer.transform(sub[feature_cols])
                    if selector is not None:
                        feature_values = selector.transform(feature_values)
                    feature_values = scaler.transform(feature_values).astype(np.float32)
                else:
                    feature_values = None

                sequence_list.append((str(group), labels, feature_values))
            return sequence_list

        fitting_sequences = sequences_for(fitting_groups)
        validation_sequences = sequences_for([validation_group])
        testing_sequences = sequences_for(np.unique(groups[test_index]))

        model = TokenSeqNet(n_classes=n_classes, n_features=selected_feature_count if use_features else 0, kind=model_kind).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
        loss_function = nn.CrossEntropyLoss()

        def sequence_iterator(sequence_list):
            for group, labels, features in sequence_list:
                if len(labels) < 3:
                    continue
                label_input = np.concatenate([[start_id], labels[:-1]])
                yield group, label_input, labels, features

        best_state, best_validation_loss, bad_epochs = None, np.inf, 0

        for _ in range(epochs):
            model.train()
            for _, label_input, target, features in sequence_iterator(fitting_sequences):
                label_tensor = torch.tensor(label_input[None], dtype=torch.long, device=DEVICE)
                target_tensor = torch.tensor(target, dtype=torch.long, device=DEVICE)
                feature_tensor = torch.tensor(features[None], dtype=torch.float32, device=DEVICE) if features is not None else None

                output = model(label_tensor, feature_tensor)
                loss = loss_function(output[0], target_tensor)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            validation_loss, validation_batches = 0.0, 0
            with torch.no_grad():
                for _, label_input, target, features in sequence_iterator(validation_sequences):
                    label_tensor = torch.tensor(label_input[None], dtype=torch.long, device=DEVICE)
                    target_tensor = torch.tensor(target, dtype=torch.long, device=DEVICE)
                    feature_tensor = torch.tensor(features[None], dtype=torch.float32, device=DEVICE) if features is not None else None
                    validation_loss += loss_function(model(label_tensor, feature_tensor)[0], target_tensor).item()
                    validation_batches += 1
            validation_loss /= max(validation_batches, 1)

            if validation_loss < best_validation_loss - 1e-4:
                best_validation_loss = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= 12:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(DEVICE)

        model.eval()
        with torch.no_grad():
            for group, label_input, target, features in sequence_iterator(testing_sequences):
                label_tensor = torch.tensor(label_input[None], dtype=torch.long, device=DEVICE)
                feature_tensor = torch.tensor(features[None], dtype=torch.float32, device=DEVICE) if features is not None else None
                prediction = model(label_tensor, feature_tensor)[0].argmax(-1).cpu().numpy()

                # Match the primary grammar panel exactly: predict only positions primary_start onward.
                for token_index in range(primary_start, len(target)):
                    all_true.append(int(target[token_index]))
                    all_pred.append(int(prediction[token_index]))
                    all_group.append(str(group))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.asarray(all_true, dtype=int), np.asarray(all_pred, dtype=int), np.asarray(all_group, dtype=str), classes


def run_all_seeds(T: pd.DataFrame, fcols: list, out_dir: str, expected_core_targets: int | None = None, resume_existing: bool = True, seeds=tuple(TOKEN_SEEDS), model_kinds=("transformer", "lstm"), epochs: int = 100, k_select: int = K_SELECT, pipeline_version: str = "final_v2_common_targets"):
    """Sweeps model_kind x {labels_only, labels_sensors} x seed, with
    per-run checkpointing (skips a run whose prediction+summary CSVs already
    exist). Returns (neural_summary, neural_seed_results, neural_fold_metrics,
    group_means_over_seeds) — neural_summary is what maps to Table 8.4.

    If expected_core_targets is given (pass task3_common_targets.run_primary_
    panel's summary['n'].iloc[0]), asserts every seed/model saw the identical
    target count — matches the source notebook's own cross-check that the
    neural and grammar panels are evaluated on the same 217 targets.
    """
    checkpoint_dir = os.path.join(out_dir, "TOKEN_NEURAL_CHECKPOINTS_FINAL_V2_COMMON_TARGETS")
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_result_rows, fold_metric_rows = [], []
    n_classes = T["label"].astype(str).nunique()

    for model_kind in model_kinds:
        for use_features, feature_tag in [(False, "labels_only"), (True, "labels_sensors")]:
            for seed in seeds:
                run_key = f"{pipeline_version}__{model_kind}__{feature_tag}__seed{seed}"
                prediction_path = os.path.join(checkpoint_dir, f"{run_key}__predictions.csv")
                summary_path = os.path.join(checkpoint_dir, f"{run_key}__summary.csv")

                if resume_existing and os.path.exists(prediction_path) and os.path.exists(summary_path):
                    print("Reloading corrected token run:", run_key)
                    prediction_df = pd.read_csv(prediction_path)
                    seed_summary = pd.read_csv(summary_path).iloc[0].to_dict()
                else:
                    print("Running corrected token model:", run_key)
                    y_true, y_pred, held_groups, token_classes = run_token_seed(
                        tokens=T, feature_cols=fcols, model_kind=model_kind, use_features=use_features,
                        seed=seed, epochs=epochs, k_select=k_select,
                    )
                    prediction_df = pd.DataFrame({
                        "model_kind": model_kind, "feature_setting": feature_tag, "seed": seed,
                        "held_group": held_groups, "y_true": y_true, "y_pred": y_pred,
                    })
                    seed_summary = {
                        "model_kind": model_kind, "feature_setting": feature_tag, "seed": seed, "n": len(y_true),
                        "accuracy": accuracy_score(y_true, y_pred),
                        "macro_f1": f1_score(y_true, y_pred, labels=np.arange(len(token_classes)), average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    }
                    prediction_df.to_csv(prediction_path, index=False)
                    pd.DataFrame([seed_summary]).to_csv(summary_path, index=False)

                seed_result_rows.append(seed_summary)

                for held_group, fold_df in prediction_df.groupby("held_group"):
                    fold_metric_rows.append({
                        "model_kind": model_kind, "feature_setting": feature_tag, "seed": seed, "held_group": str(held_group),
                        "n": len(fold_df),
                        "accuracy": accuracy_score(fold_df["y_true"], fold_df["y_pred"]),
                        "macro_f1": f1_score(fold_df["y_true"], fold_df["y_pred"], labels=np.arange(n_classes), average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(fold_df["y_true"], fold_df["y_pred"]),
                    })

    neural_seed_results = pd.DataFrame(seed_result_rows)
    neural_fold_metrics = pd.DataFrame(fold_metric_rows)

    if expected_core_targets is not None:
        target_counts = neural_seed_results.groupby(["model_kind", "feature_setting", "seed"])["n"].first()
        if not (target_counts == expected_core_targets).all():
            raise RuntimeError(
                f"Core neural target counts do not match the deterministic panel. "
                f"Expected {expected_core_targets}; found {sorted(target_counts.unique())}."
            )
        print("Verified identical core target count for every model/seed:", expected_core_targets)

    seed_summary = neural_seed_results.groupby(["model_kind", "feature_setting"], as_index=False).agg(
        seed_accuracy_mean=("accuracy", "mean"),
        seed_accuracy_std=("accuracy", lambda x: np.std(x, ddof=SEED_STD_DDOF)),
        seed_accuracy_sample_std=("accuracy", lambda x: np.std(x, ddof=1)),
        seed_macro_f1_mean=("macro_f1", "mean"),
        seed_macro_f1_std=("macro_f1", lambda x: np.std(x, ddof=SEED_STD_DDOF)),
        seed_macro_f1_sample_std=("macro_f1", lambda x: np.std(x, ddof=1)),
        seed_balanced_accuracy_mean=("balanced_accuracy", "mean"),
        seed_balanced_accuracy_std=("balanced_accuracy", lambda x: np.std(x, ddof=SEED_STD_DDOF)),
        n_seeds=("seed", "nunique"),
    )

    # Held-out-group variability: average seeds within each group first, then SD across the 9 group means.
    group_means_over_seeds = neural_fold_metrics.groupby(["model_kind", "feature_setting", "held_group"], as_index=False).agg(
        group_accuracy=("accuracy", "mean"), group_macro_f1=("macro_f1", "mean"), group_balanced_accuracy=("balanced_accuracy", "mean"),
        n_seeds=("seed", "nunique"),
    )
    group_summary = group_means_over_seeds.groupby(["model_kind", "feature_setting"], as_index=False).agg(
        group_accuracy_mean=("group_accuracy", "mean"), group_accuracy_std=("group_accuracy", lambda x: np.std(x, ddof=FOLD_STD_DDOF)),
        group_macro_f1_mean=("group_macro_f1", "mean"), group_macro_f1_std=("group_macro_f1", lambda x: np.std(x, ddof=FOLD_STD_DDOF)),
        group_balanced_accuracy_mean=("group_balanced_accuracy", "mean"), group_balanced_accuracy_std=("group_balanced_accuracy", lambda x: np.std(x, ddof=FOLD_STD_DDOF)),
        n_groups=("held_group", "nunique"),
    )

    neural_summary = seed_summary.merge(group_summary, on=["model_kind", "feature_setting"], how="left", validate="one_to_one")
    for col, mean_col, std_col in [
        ("seed_accuracy_mean_pm_std", "seed_accuracy_mean", "seed_accuracy_std"),
        ("seed_macro_f1_mean_pm_std", "seed_macro_f1_mean", "seed_macro_f1_std"),
        ("group_accuracy_mean_pm_std", "group_accuracy_mean", "group_accuracy_std"),
        ("group_macro_f1_mean_pm_std", "group_macro_f1_mean", "group_macro_f1_std"),
    ]:
        neural_summary[col] = neural_summary.apply(lambda row, m=mean_col, s=std_col: f"{row[m]:.3f} ± {row[s]:.3f}", axis=1)
    neural_summary = neural_summary.sort_values("seed_macro_f1_mean", ascending=False).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    neural_seed_results.to_csv(os.path.join(out_dir, "task3_token_neural_FINAL_V2_COMMON_TARGETS_seed_results.csv"), index=False)
    neural_fold_metrics.to_csv(os.path.join(out_dir, "task3_token_neural_FINAL_V2_COMMON_TARGETS_seed_group_metrics.csv"), index=False)
    group_means_over_seeds.to_csv(os.path.join(out_dir, "task3_token_neural_FINAL_V2_COMMON_TARGETS_group_means.csv"), index=False)
    neural_summary.to_csv(os.path.join(out_dir, "task3_token_neural_FINAL_V2_COMMON_TARGETS_summary_with_std.csv"), index=False)

    print("\nCORRECTED NEURAL SUMMARY (-> Table 8.4)")
    print(neural_summary.round(4).to_string(index=False))

    return neural_summary, neural_seed_results, neural_fold_metrics, group_means_over_seeds
