"""Task 3 — the short-order activity grammar over tokens: n-gram Markov
back-off, second-order HMM, and the hybrid grammar×sensor model.

Ported from `github notebooks/actual ones/
task3_FINAL_V2_with_exact_report_reproduction.ipynb`, cells "APPENDIX R —
..." (the source notebook's own naming; see the IMPORTANT note below for why
that name is misleading here).

Produces Table 8.7 — **the thesis's headline Task 3 result**
(docs/thesis_reproduction_targets.md §6.9, section title "THE WINNING
RESULT"): a simple back-off n-gram Markov model over the last 2-3 activity
tokens, beating the tokenized Transformer. Verified against the source code,
not just the thesis text (see docs/table_to_source_mapping.md's Task 3 entry
for the full reasoning) — this is genuinely what produced those numbers.

IMPORTANT — why this isn't really "Appendix R"/"legacy": the source notebook
labels this code "Appendix R — legacy reproduction", framing it as verifying
an old, methodologically-superseded protocol against the corrected Parts
II-IV pipeline elsewhere in the same notebook. That framing is misleading for
reproduction purposes: the thesis's own text (§8.9) describes this exact
n-gram/HMM/hybrid computation — evaluated over all 244 tokens starting from
position 1, not the "common-target" 217-target restriction Parts II-IV use —
as the actual follow-up analysis that produced the published headline number.
Do not "clean this up" to match the corrected common-target protocol; that
would silently produce different (and wrong) numbers.

CRITICAL REPRODUCIBILITY DETAIL (from the source notebook's own code
comment, cell 50): the historical token table stored `group` as **text**, so
training groups were iterated in the order 1, 10, 2, 3, 5, ... (string sort,
not numeric). Tie-breaking in `Counter.most_common()` depends on this
insertion order. The source notebook explicitly verified both variants:
int-typed group order gives h=2 acc/macro-F1 = 0.596/0.460; text-typed group
order gives 0.604/0.499 — the published number. This module therefore casts
`group` to string before running these models, deliberately, and this must
not be "fixed" to int for tidiness.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

EPS = 1e-9
FOLD_STD_DDOF = 1  # group-level SD uses the conventional sample SD (matches source notebook)

# Published numbers this module's output should reproduce (thesis Table 8.7 /
# source notebook's REPORT_REFERENCE, cell 54). Used only for the optional
# verification check below — never to adjust behavior.
REPORT_REFERENCE = {
    "ngram_backoff_h1": (0.515, 0.304),
    "ngram_backoff_h2": (0.604, 0.499),
    "ngram_backoff_h3": (0.583, 0.513),
    "ngram_backoff_h5": (0.519, 0.442),
    "HMM2_categorical": (0.606, 0.443),
    "HMM2_sensor": (0.389, 0.194),
    "hybrid_ngram3_sensor": (0.583, 0.495),
}


def _fold_table(rows_out: list, fold_rows_out: list, model_name: str, yt, yp, fold_ids):
    """Records pooled + per-fold metrics for one model into the shared
    accumulator lists (mirrors the source notebook's module-level LEGACY_ROWS
    / LEGACY_FOLD_ROWS globals, made explicit here instead)."""
    yt, yp = np.asarray(yt), np.asarray(yp)
    fold_ids = np.asarray(fold_ids, dtype=object)

    per_fold = []
    for g in pd.unique(fold_ids):
        m = fold_ids == g
        per_fold.append({
            "held_group": g, "n": int(m.sum()),
            "accuracy": accuracy_score(yt[m], yp[m]),
            "macro_f1": f1_score(yt[m], yp[m], average="macro", zero_division=0),
        })
    fold_df = pd.DataFrame(per_fold)
    fold_rows_out.extend({"model": model_name, **r} for r in per_fold)
    rows_out.append({
        "model": model_name, "n": int(len(yt)), "n_folds": int(len(fold_df)),
        "pooled_accuracy": accuracy_score(yt, yp),
        "pooled_macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "fold_accuracy_mean": fold_df["accuracy"].mean(), "fold_accuracy_std": fold_df["accuracy"].std(ddof=FOLD_STD_DDOF),
        "fold_macro_f1_mean": fold_df["macro_f1"].mean(), "fold_macro_f1_std": fold_df["macro_f1"].std(ddof=FOLD_STD_DDOF),
    })


def _prepare_text_group_tokens(T: pd.DataFrame) -> pd.DataFrame:
    """The critical group-as-text cast — see module docstring."""
    return T.assign(group=T["group"].astype(str))


def run_ngram_backoff(T: pd.DataFrame, rows_out: list, fold_rows_out: list, hist_lens=(1, 2, 3, 5)):
    """N-gram Markov back-off over token history, orders in `hist_lens`. Uses
    the last h token labels (not raw windows), with back-off to shorter
    contexts when the exact context was unseen in training."""
    T = _prepare_text_group_tokens(T)
    classes = sorted(T["label"].unique())
    cls2i = {c: i for i, c in enumerate(classes)}
    groups = T["group"].values
    logo = LeaveOneGroupOut()

    print("\n===== 6label: n-gram Markov over token history =====")
    for H in hist_lens:
        yt, yp, fold_ids = [], [], []
        for tr, te in logo.split(T, T["label"].values, groups):
            tables = [defaultdict(Counter) for _ in range(H)]  # tables[o][context][next]
            for g in np.unique(groups[tr]):
                s = [cls2i[l] for l in T[T["group"] == g]["label"].values]
                for t in range(1, len(s)):
                    for o in range(1, H + 1):
                        if t - o >= 0:
                            tables[o - 1][tuple(s[t - o:t])][s[t]] += 1

            for g in np.unique(groups[te]):
                s = [cls2i[l] for l in T[T["group"] == g]["label"].values]
                for t in range(1, len(s)):
                    pred = None
                    for o in range(min(H, t), 0, -1):  # try longest context first
                        ctx = tuple(s[t - o:t])
                        if ctx in tables[o - 1]:
                            pred = tables[o - 1][ctx].most_common(1)[0][0]
                            break
                    if pred is None:  # global fallback
                        pred = Counter([cls2i[l] for l in T.iloc[tr]["label"]]).most_common(1)[0][0]
                    yt.append(s[t]); yp.append(pred); fold_ids.append(g)

        yt, yp = np.array(yt), np.array(yp)
        acc = accuracy_score(yt, yp)
        mf = f1_score(yt, yp, average="macro", zero_division=0)
        per = {classes[ci]: round(f1_score(yt == ci, yp == ci, zero_division=0), 2) for ci in range(len(classes))}
        print(f"  h={H}: acc={acc:.3f} macroF1={mf:.3f}  per-class={per}")
        _fold_table(rows_out, fold_rows_out, f"ngram_backoff_h{H}", yt, yp, fold_ids)


def run_second_order_hmm(T: pd.DataFrame, fcols: list, rows_out: list, fold_rows_out: list, k_select: int = 40):
    """Second-order HMM over token labels: state = (prev_label, current_label)
    composite (n*n states). Causal forward filter predicting the NEXT label.
    Two emission variants: categorical (label itself, no sensor info) and
    sensor (LogReg probability of the current label given sensor features).
    """
    T = _prepare_text_group_tokens(T)
    classes = sorted(T["label"].unique())
    cls2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    N2 = n * n

    def pid(a, b):
        return a * n + b

    groups = T["group"].values
    logo = LeaveOneGroupOut()
    res = {"HMM2_sensor": ([], []), "HMM2_categorical": ([], [])}
    res_folds = {"HMM2_sensor": [], "HMM2_categorical": []}

    for tr, te in logo.split(T, T["label"].values, groups):
        A2 = np.ones((N2, n))  # Laplace-smoothed transition (a,b) -> c
        st2 = np.ones(N2)
        for g in np.unique(groups[tr]):
            s = [cls2i[l] for l in T[T["group"] == g]["label"].values]
            if len(s) >= 2:
                st2[pid(s[0], s[1])] += 1
            for t in range(2, len(s)):
                A2[pid(s[t - 2], s[t - 1]), s[t]] += 1
        A2 = A2 / A2.sum(1, keepdims=True)
        st2 = st2 / st2.sum()

        imp = SimpleImputer(strategy="median").fit(T.iloc[tr][fcols])
        Xtr = imp.transform(T.iloc[tr][fcols])
        sel = SelectKBest(f_classif, k=min(k_select, Xtr.shape[1])).fit(Xtr, T.iloc[tr]["label"].values)
        sc = StandardScaler().fit(sel.transform(Xtr))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(sel.transform(Xtr)), T.iloc[tr]["label"].values)
        col = [list(clf.classes_).index(c) for c in classes]

        for g in np.unique(groups[te]):
            sub = T[T["group"] == g]
            s = [cls2i[l] for l in sub["label"].values]
            if len(s) < 4:
                continue
            proba = clf.predict_proba(sc.transform(sel.transform(imp.transform(sub[fcols]))))[:, col]
            lp_sen = np.log(proba + EPS)  # (T, n): P(features_t | label b)

            for key in ["HMM2_sensor", "HMM2_categorical"]:
                a = np.log(st2 + EPS).copy()

                def emis_vec(t, key=key):
                    if key == "HMM2_sensor":
                        return lp_sen[t]
                    e = np.full(n, np.log(EPS))
                    e[s[t]] = 0.0
                    return e

                a = a + np.repeat(emis_vec(0), n) + np.tile(emis_vec(1), n)
                for t in range(1, len(s) - 1):
                    bel = a - np.logaddexp.reduce(a)  # belief over (prev, cur)
                    nxt = np.logaddexp.reduce(bel[:, None] + np.log(A2 + EPS), axis=0)
                    res[key][0].append(s[t + 1]); res[key][1].append(int(nxt.argmax())); res_folds[key].append(g)

                    belm = bel.reshape(n, n)  # [a, b]
                    new = np.full(N2, -np.inf)
                    for b in range(n):
                        col_ab = belm[:, b] + np.log(A2[[pid(aa, b) for aa in range(n)], :] + EPS).T  # noqa: F841 (kept for fidelity with source)
                        for c in range(n):
                            new[pid(b, c)] = np.logaddexp.reduce(belm[:, b] + np.log(A2[[pid(aa, b) for aa in range(n)], c] + EPS))
                    a = new + np.tile(emis_vec(t + 1), n)

    print("\n===== 6label: SECOND-ORDER HMM over tokens =====")
    for k, (yt, yp) in res.items():
        yt, yp = np.array(yt), np.array(yp)
        acc = accuracy_score(yt, yp)
        mf = f1_score(yt, yp, average="macro", zero_division=0)
        per = {classes[ci]: round(f1_score(yt == ci, yp == ci, zero_division=0), 2) for ci in range(n)}
        print(f"  {k:18s} acc={acc:.3f} macroF1={mf:.3f}  per-class={per}")
        _fold_table(rows_out, fold_rows_out, k, yt, yp, res_folds[k])


def run_hybrid_ngram_sensor(T: pd.DataFrame, fcols: list, rows_out: list, fold_rows_out: list, h: int = 3, k_select: int = 40):
    """Hybrid model: P_hybrid(next) = lambda * P_ngram(next | last h labels)
    + (1 - lambda) * P_sensor(next | current token's features). Lambda is
    chosen per LOGO fold on a held-out TRAIN group (never the test group),
    swept over {0.0, 0.1, ..., 1.0}."""
    T = _prepare_text_group_tokens(T)
    classes = sorted(T["label"].unique())
    cls2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    groups = T["group"].values
    logo = LeaveOneGroupOut()
    lambdas = np.linspace(0.0, 1.0, 11)
    yt_all, yp_all, lam_used, fold_all = [], [], [], []

    def ngram_tables(gids):
        tabs = [defaultdict(Counter) for _ in range(h)]
        glob = Counter()
        for g in gids:
            s = [cls2i[l] for l in T[T["group"] == g]["label"].values]
            for t in range(1, len(s)):
                glob[s[t]] += 1
                for o in range(1, h + 1):
                    if t - o >= 0:
                        tabs[o - 1][tuple(s[t - o:t])][s[t]] += 1
        return tabs, glob

    def p_ngram(tabs, glob, s, t):
        for o in range(min(h, t), 0, -1):
            ctx = tuple(s[t - o:t])
            if ctx in tabs[o - 1]:
                c = tabs[o - 1][ctx]
                v = np.full(n, 1.0)
                for k_, cnt in c.items():
                    v[k_] += cnt
                return v / v.sum()
        v = np.full(n, 1.0)
        for k_, cnt in glob.items():
            v[k_] += cnt
        return v / v.sum()

    for tr, te in logo.split(T, T["label"].values, groups):
        tr_groups = np.unique(groups[tr])
        te_groups = np.unique(groups[te])
        val_g = tr_groups[-1]  # held-out TRAIN group used to pick lambda
        fit_groups = [g for g in tr_groups if g != val_g]

        def make_xy(gids):
            X, y = [], []
            for g in gids:
                sub = T[T["group"] == g]
                F = sub[fcols].values
                s = [cls2i[l] for l in sub["label"].values]
                for t in range(len(s) - 1):
                    X.append(F[t]); y.append(s[t + 1])  # current features -> next label
            return np.array(X, dtype=float), np.array(y)

        Xf, yf = make_xy(fit_groups)
        imp = SimpleImputer(strategy="median").fit(Xf)
        sel = SelectKBest(f_classif, k=min(k_select, Xf.shape[1])).fit(imp.transform(Xf), yf)
        sc = StandardScaler().fit(sel.transform(imp.transform(Xf)))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(sel.transform(imp.transform(Xf))), yf)
        present = list(clf.classes_)

        def p_sensor(feat_row):
            pr = clf.predict_proba(sc.transform(sel.transform(imp.transform(feat_row[None, :]))))[0]
            v = np.full(n, EPS)
            for ci, cl in enumerate(present):
                v[cl] = pr[ci]
            return v / v.sum()

        tabs_fit, glob_fit = ngram_tables(fit_groups)

        best_lam, best_f1 = 1.0, -1
        sub = T[T["group"] == val_g]
        sv = [cls2i[l] for l in sub["label"].values]
        Fv = sub[fcols].values
        if len(sv) >= 3:
            for lam in lambdas:
                yv_t, yv_p = [], []
                for t in range(1, len(sv)):
                    pg = p_ngram(tabs_fit, glob_fit, sv, t)
                    ps = p_sensor(Fv[t - 1])  # current token's features predict s[t]
                    yv_t.append(sv[t]); yv_p.append(int((lam * pg + (1 - lam) * ps).argmax()))
                f1v = f1_score(yv_t, yv_p, average="macro", zero_division=0)
                if f1v > best_f1:
                    best_f1, best_lam = f1v, lam

        tabs_all, glob_all = ngram_tables(tr_groups)
        Xa, ya = make_xy(list(tr_groups))
        imp = SimpleImputer(strategy="median").fit(Xa)
        sel = SelectKBest(f_classif, k=min(k_select, Xa.shape[1])).fit(imp.transform(Xa), ya)
        sc = StandardScaler().fit(sel.transform(imp.transform(Xa)))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(sel.transform(imp.transform(Xa))), ya)
        present = list(clf.classes_)

        for g in te_groups:
            sub = T[T["group"] == g]
            s = [cls2i[l] for l in sub["label"].values]
            F = sub[fcols].values
            for t in range(1, len(s)):
                pg = p_ngram(tabs_all, glob_all, s, t)
                ps = p_sensor(F[t - 1])
                yp_all.append(int((best_lam * pg + (1 - best_lam) * ps).argmax()))
                yt_all.append(s[t]); lam_used.append(best_lam); fold_all.append(g)

    yt_all, yp_all = np.array(yt_all), np.array(yp_all)
    acc = accuracy_score(yt_all, yp_all)
    mf = f1_score(yt_all, yp_all, average="macro", zero_division=0)
    print(f"\n===== 6label: HYBRID n-gram(h<={h}) x sensor->next =====")
    print(f"  acc={acc:.3f} macroF1={mf:.3f}")
    print(f"  lambda per fold (1.0 = pure grammar): {sorted(set(lam_used))}")
    _fold_table(rows_out, fold_rows_out, "hybrid_ngram3_sensor", yt_all, yp_all, fold_all)


def run_all(T: pd.DataFrame, fcols: list, out_dir: str, hist_lens=(1, 2, 3, 5), hybrid_h: int = 3, k_select: int = 40):
    """Runs the n-gram sweep, second-order HMM (both emission variants), and
    the hybrid model; writes the combined summary/fold-metrics CSVs and a
    verification check against the published Table 8.7 numbers. Returns
    (summary_df, fold_df, check_df)."""
    rows, fold_rows = [], []

    print(f"tokens: {len(T)} | groups: {T['group'].nunique()} | classes: {T['label'].nunique()}")
    print("expected for exact reproduction: 244 tokens, 9 groups, 6 classes")

    run_ngram_backoff(T, rows, fold_rows, hist_lens=hist_lens)
    run_second_order_hmm(T, fcols, rows, fold_rows, k_select=k_select)
    run_hybrid_ngram_sensor(T, fcols, rows, fold_rows, h=hybrid_h, k_select=k_select)

    summary = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    for metric in ["accuracy", "macro_f1"]:
        summary[f"{metric}_mean_pm_std"] = summary.apply(
            lambda row, m=metric: f"{row[f'fold_{m}_mean']:.3f} ± {row[f'fold_{m}_std']:.3f}", axis=1
        )

    check_rows = []
    for _, row in summary.iterrows():
        ref = REPORT_REFERENCE.get(row["model"])
        check_rows.append({
            "model": row["model"], "n": row["n"],
            "pooled_acc": round(row["pooled_accuracy"], 3), "report_acc": ref[0] if ref else None,
            "pooled_macro_f1": round(row["pooled_macro_f1"], 3), "report_macro_f1": ref[1] if ref else None,
            "match": (
                "—" if ref is None else
                "EXACT" if (abs(row["pooled_accuracy"] - ref[0]) < 0.0006 and abs(row["pooled_macro_f1"] - ref[1]) < 0.0006)
                else "DIFFERS"
            ),
        })
    check = pd.DataFrame(check_rows)
    print("\nREPRODUCTION CHECK AGAINST docs/thesis_reproduction_targets.md Table 8.7")
    print(check.to_string(index=False))

    n_exact = int((check["match"] == "EXACT").sum())
    n_ref = int(check["match"].isin(["EXACT", "DIFFERS"]).sum())
    print(f"\n{n_exact} of {n_ref} referenced rows reproduce the published table exactly.")
    if n_exact < n_ref:
        print(
            "Any DIFFERS row means the token table here is not identical to the historical one "
            "(check token count/ordering above) — this is expected if tokens were rebuilt "
            "from scratch rather than loaded from a verified table (see task3_tokens.py)."
        )

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "task3_grammar_summary_with_std.csv"), index=False)
    folds.to_csv(os.path.join(out_dir, "task3_grammar_fold_metrics.csv"), index=False)
    check.to_csv(os.path.join(out_dir, "task3_grammar_report_reproduction_check.csv"), index=False)

    return summary, folds, check
