"""Task 3, Part II — "corrected fair" grammar/sensor/hybrid comparison on
common targets.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cell "GRAMMAR, SECOND-
ORDER, SENSOR AND HYBRID MODELS" (source notebook's own "Part II").

Every model here predicts the SAME token positions — either "primary"
(position 3 onward, 217 shared targets across the 9 held-out-group folds) or
"long-history" (position 10 onward, sensitivity check) — so accuracy/macro-F1
are directly comparable across models, unlike task3_grammar.py's models
(which each use however much history they can from position 1).

IMPORTANT CROSS-REFERENCE (verified against actual numbers, not assumed):
this module's `segment_markov_h1` (order=1, primary/position-3 panel) is the
SAME computation that appears as both:
  - Table 8.4's "Segment Markov" baseline (0.481/0.285)
  - Table 8.7's "First-order segment Markov" row (same 0.481/0.285)
It is NOT the same as task3_grammar.py's `ngram_backoff_h1` (0.515/0.304),
which uses all 244 tokens from position 1 with text-typed groups. Two
genuinely different first-order-Markov numbers exist in this thesis — don't
conflate them.

The rest of this module's output (ngram_backoff_h2/h3, second_order_
categorical_markov, sensor_next_logreg, hybrid_ngram3_sensor — all on the
217-target common-target protocol) is NOT directly published in any Ch.8
table as far as we've confirmed. The thesis's own text (§8.9) narratively
refers to this as "the earlier comparison" that motivated the task3_grammar.py
follow-up ("tested whether the earlier comparison had handicapped classical
temporal models by restricting their history. It had.") — i.e. this module's
role is mostly to reproduce that narrative point and to supply the
segment_markov_h1 baseline; task3_grammar.py's numbers are what's headline.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRIMARY_START = 3
LONG_START = 10
FOLD_STD_DDOF = 1
HYBRID_ORDER = 3
HYBRID_LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 2)
K_SELECT = 40


def token_sequences(T: pd.DataFrame, fcols: list) -> dict:
    """group (as string) -> {'labels': int array, 'features': float array}."""
    class_to_id = {label: i for i, label in enumerate(sorted(T["label"].astype(str).unique()))}
    sequences = {}
    for group, sub in T.groupby("group"):
        sub = sub.sort_values("start_time").reset_index(drop=True)
        sequences[str(group)] = {
            "labels": np.asarray([class_to_id[label] for label in sub["label"].astype(str)], dtype=int),
            "features": sub[fcols].apply(pd.to_numeric, errors="coerce").values.astype(float),
        }
    return sequences, class_to_id


def fit_backoff_tables(sequences, train_groups, max_order, n_classes, alpha=1.0):
    tables = {order: defaultdict(lambda: np.full(n_classes, alpha, dtype=float)) for order in range(1, max_order + 1)}
    global_counts = np.full(n_classes, alpha, dtype=float)
    for group in train_groups:
        labels = sequences[group]["labels"]
        for target_index in range(1, len(labels)):
            target = labels[target_index]
            global_counts[target] += 1
            for order in range(1, min(max_order, target_index) + 1):
                context = tuple(labels[target_index - order:target_index])
                tables[order][context][target] += 1
    return tables, global_counts


def backoff_probabilities(history, tables, global_counts, max_order):
    history = list(history)
    for order in range(min(max_order, len(history)), 0, -1):
        context = tuple(history[-order:])
        if context in tables[order]:
            counts = np.asarray(tables[order][context], dtype=float)
            return counts / counts.sum()
    return global_counts / global_counts.sum()


def fit_second_order_transition(sequences, train_groups, n_classes, alpha=1.0):
    table = defaultdict(lambda: np.full(n_classes, alpha, dtype=float))
    global_counts = np.full(n_classes, alpha, dtype=float)
    for group in train_groups:
        labels = sequences[group]["labels"]
        for target_index in range(2, len(labels)):
            context = (labels[target_index - 2], labels[target_index - 1])
            target = labels[target_index]
            table[context][target] += 1
            global_counts[target] += 1
    return table, global_counts


def fit_sensor_pipeline(sequences, train_groups, k_select=K_SELECT):
    X_rows, y_rows = [], []
    for group in train_groups:
        labels = sequences[group]["labels"]
        features = sequences[group]["features"]
        for target_index in range(1, len(labels)):
            X_rows.append(features[target_index - 1])
            y_rows.append(labels[target_index])
    X_rows, y_rows = np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)

    k = min(k_select, X_rows.shape[1])
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_classif, k=k)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_rows, y_rows)
    return pipeline


def aligned_sensor_probabilities(model, feature_row, n_classes):
    probabilities = model.predict_proba(np.asarray(feature_row, dtype=float).reshape(1, -1))[0]
    aligned = np.full(n_classes, 1e-12, dtype=float)
    for class_id, probability in zip(model.classes_, probabilities):
        aligned[int(class_id)] = float(probability)
    aligned /= aligned.sum()
    return aligned


def metric_record(y_true, y_pred, n_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def choose_hybrid_lambda(sequences, train_groups, order, n_classes):
    if len(train_groups) < 2:
        return 1.0
    validation_group = sorted(train_groups)[-1]
    fitting_groups = [g for g in train_groups if g != validation_group]

    tables, global_counts = fit_backoff_tables(sequences, fitting_groups, order, n_classes)
    sensor_model = fit_sensor_pipeline(sequences, fitting_groups)

    labels = sequences[validation_group]["labels"]
    features = sequences[validation_group]["features"]

    y_true, ngram_probs, sensor_probs = [], [], []
    for target_index in range(order, len(labels)):
        y_true.append(labels[target_index])
        ngram_probs.append(backoff_probabilities(labels[:target_index], tables, global_counts, order))
        sensor_probs.append(aligned_sensor_probabilities(sensor_model, features[target_index - 1], n_classes))

    if not y_true:
        return 1.0
    y_true = np.asarray(y_true, dtype=int)
    ngram_probs, sensor_probs = np.asarray(ngram_probs), np.asarray(sensor_probs)

    best_lambda, best_score = 1.0, -np.inf
    for lam in HYBRID_LAMBDAS:
        combined = lam * ngram_probs + (1.0 - lam) * sensor_probs
        pred = combined.argmax(axis=1)
        score = f1_score(y_true, pred, labels=np.arange(n_classes), average="macro", zero_division=0)
        if score > best_score + 1e-12:
            best_score, best_lambda = score, float(lam)
    return best_lambda


def _save_fold(rows, predictions, model, held_group, start, y_true, y_pred, n_classes, selected_lambda=np.nan):
    scores = metric_record(y_true, y_pred, n_classes)
    rows.append({"model": model, "held_group": held_group, "evaluation_start_index": start, "n": len(y_true), "selected_lambda": selected_lambda, **scores})
    predictions.extend({
        "model": model, "held_group": held_group, "evaluation_start_index": start,
        "y_true": int(t), "y_pred": int(p), "selected_lambda": selected_lambda,
    } for t, p in zip(y_true, y_pred))


def _ngram_predictions_on_common_targets(labels, sequences, train_groups, order, start, n_classes):
    tables, global_counts = fit_backoff_tables(sequences, train_groups, order, n_classes)
    y_true, y_pred = [], []
    for index in range(start, len(labels)):
        probability = backoff_probabilities(labels[:index], tables, global_counts, order)
        y_true.append(labels[index])
        y_pred.append(int(probability.argmax()))
    return y_true, y_pred


def summarize_common_targets(predictions: pd.DataFrame, folds: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    output = []
    for model, model_predictions in predictions.groupby("model"):
        pooled = metric_record(
            model_predictions["y_true"].astype(int).values,
            model_predictions["y_pred"].astype(int).values,
            n_classes=n_classes,
        )
        model_folds = folds[folds["model"] == model]
        output.append({
            "model": model, "evaluation_start_index": int(model_predictions["evaluation_start_index"].iloc[0]),
            "n": len(model_predictions),
            "pooled_accuracy": pooled["accuracy"], "pooled_macro_f1": pooled["macro_f1"], "pooled_balanced_accuracy": pooled["balanced_accuracy"],
            "fold_accuracy_mean": model_folds["accuracy"].mean(), "fold_accuracy_std": model_folds["accuracy"].std(ddof=FOLD_STD_DDOF),
            "fold_macro_f1_mean": model_folds["macro_f1"].mean(), "fold_macro_f1_std": model_folds["macro_f1"].std(ddof=FOLD_STD_DDOF),
            "fold_balanced_accuracy_mean": model_folds["balanced_accuracy"].mean(), "fold_balanced_accuracy_std": model_folds["balanced_accuracy"].std(ddof=FOLD_STD_DDOF),
            "n_folds": model_folds["held_group"].nunique(),
            "mean_selected_lambda": model_folds["selected_lambda"].mean() if model_folds["selected_lambda"].notna().any() else np.nan,
        })
    output = pd.DataFrame(output).sort_values(["pooled_macro_f1", "pooled_accuracy"], ascending=False).reset_index(drop=True)
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
        output[f"fold_{metric}_mean_pm_std"] = output.apply(lambda row, m=metric: f"{row[f'fold_{m}_mean']:.3f} ± {row[f'fold_{m}_std']:.3f}", axis=1)
    return output


def run_primary_panel(T: pd.DataFrame, fcols: list, out_dir: str):
    """Every model predicts token positions 3 onward (217 shared targets).
    Produces segment_markov_h1 (-> Table 8.4 "Segment Markov" / Table 8.7
    "First-order segment Markov"), plus ngram_backoff_h2/h3,
    second_order_categorical_markov, sensor_next_logreg, hybrid_ngram3_sensor
    (context/narrative only, see module docstring)."""
    sequences, class_to_id = token_sequences(T, fcols)
    n_classes = len(class_to_id)
    all_groups = sorted(sequences)

    fold_rows, prediction_rows = [], []
    for held_group in all_groups:
        train_groups = [g for g in all_groups if g != held_group]
        labels = sequences[held_group]["labels"]
        features = sequences[held_group]["features"]

        for order in [1, 2, 3]:
            y_true, y_pred = _ngram_predictions_on_common_targets(labels, sequences, train_groups, order, PRIMARY_START, n_classes)
            model_name = "segment_markov_h1" if order == 1 else f"ngram_backoff_h{order}"
            _save_fold(fold_rows, prediction_rows, model_name, held_group, PRIMARY_START, y_true, y_pred, n_classes)

        second_order_table, second_global = fit_second_order_transition(sequences, train_groups, n_classes)
        y_true, y_pred = [], []
        for index in range(PRIMARY_START, len(labels)):
            context = (labels[index - 2], labels[index - 1])
            counts = second_order_table.get(context, second_global)
            y_true.append(labels[index]); y_pred.append(int(np.argmax(counts)))
        _save_fold(fold_rows, prediction_rows, "second_order_categorical_markov", held_group, PRIMARY_START, y_true, y_pred, n_classes)

        sensor_model = fit_sensor_pipeline(sequences, train_groups)
        y_true, y_pred = [], []
        for index in range(PRIMARY_START, len(labels)):
            probability = aligned_sensor_probabilities(sensor_model, features[index - 1], n_classes)
            y_true.append(labels[index]); y_pred.append(int(probability.argmax()))
        _save_fold(fold_rows, prediction_rows, "sensor_next_logreg", held_group, PRIMARY_START, y_true, y_pred, n_classes)

        selected_lambda = choose_hybrid_lambda(sequences, train_groups, HYBRID_ORDER, n_classes)
        grammar_tables, grammar_global = fit_backoff_tables(sequences, train_groups, HYBRID_ORDER, n_classes)
        sensor_model = fit_sensor_pipeline(sequences, train_groups)
        y_true, y_pred = [], []
        for index in range(PRIMARY_START, len(labels)):
            grammar_probability = backoff_probabilities(labels[:index], grammar_tables, grammar_global, HYBRID_ORDER)
            sensor_probability = aligned_sensor_probabilities(sensor_model, features[index - 1], n_classes)
            combined = selected_lambda * grammar_probability + (1.0 - selected_lambda) * sensor_probability
            y_true.append(labels[index]); y_pred.append(int(combined.argmax()))
        _save_fold(fold_rows, prediction_rows, "hybrid_ngram3_sensor", held_group, PRIMARY_START, y_true, y_pred, n_classes, selected_lambda)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_common_targets(predictions, folds, n_classes)

    if summary["n"].nunique() != 1:
        raise RuntimeError("Primary grammar models do not have identical target counts.")
    print("\nPRIMARY FAIR COMPARISON (every model predicts positions 3 onward)")
    print(summary.round(4).to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "task3_grammar_primary_common_targets_summary_with_std.csv"), index=False)
    folds.to_csv(os.path.join(out_dir, "task3_grammar_primary_common_targets_fold_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(out_dir, "task3_grammar_primary_common_targets_predictions.csv"), index=False)
    return summary, folds, predictions


def run_long_history_panel(T: pd.DataFrame, fcols: list, out_dir: str, orders=(1, 2, 3, 5, 10)):
    """Sensitivity check: every n-gram order predicts positions 10 onward
    (a much smaller common-target set), to see whether longer-history orders
    do better once everyone is judged on the same, later-starting targets."""
    sequences, class_to_id = token_sequences(T, fcols)
    n_classes = len(class_to_id)
    all_groups = sorted(sequences)

    fold_rows, prediction_rows = [], []
    for held_group in all_groups:
        train_groups = [g for g in all_groups if g != held_group]
        labels = sequences[held_group]["labels"]
        for order in orders:
            y_true, y_pred = _ngram_predictions_on_common_targets(labels, sequences, train_groups, order, LONG_START, n_classes)
            _save_fold(fold_rows, prediction_rows, f"ngram_backoff_h{order}", held_group, LONG_START, y_true, y_pred, n_classes)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_common_targets(predictions, folds, n_classes)

    if summary["n"].nunique() != 1:
        raise RuntimeError("Long-history models do not have identical target counts.")
    print("\nLONG-HISTORY SENSITIVITY (every order predicts positions 10 onward)")
    print(summary.round(4).to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "task3_grammar_long_history_common_targets_summary_with_std.csv"), index=False)
    folds.to_csv(os.path.join(out_dir, "task3_grammar_long_history_common_targets_fold_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(out_dir, "task3_grammar_long_history_common_targets_predictions.csv"), index=False)
    return summary, folds, predictions
