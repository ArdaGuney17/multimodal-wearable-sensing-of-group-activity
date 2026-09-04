"""Shared signal-processing / feature-list helpers for the OpenEarable
feature-engineering family (ENG3 recognition labels, ENG7 proximity, OE9,
OE10/MAG) that produce Table 7.9 (Ch.7 §7.5.2).

The source notebooks (`master_feature_generator_task1_task2_task3_
CORRECTED_V4.ipynb`'s CELL 11, and `07_feature_engineering_ENG7_
activity_invariant.ipynb`'s CELLs 6/10/15/17/19/20/22) each redefine
near-identical copies of these helpers (`unique_feats`, `clean_feature_
list`, `safe_mean/std/min/max/range/iqr/percentile`, `spectral_entropy`,
`band_ratio`, `corr_safe`, `max_lag_corr`, `sinterp`, `aggregate_values`,
`event_count`, `mad_diff`, `circular_diff`) — see notebooks_reference/
07_feature_engineering_ENG7_activity_invariant_EXECUTED_ANALYSIS.md's
"296 near-duplicate helper functions" note. Consolidated here once,
imported by every module in src/features/ that needs a subset of them,
rather than re-porting each cell's own copy verbatim. Each function below
is checked against every notebook copy it replaces; where two copies
differ (rare), the docstring says so and the more general version is kept.

`discover_model_ready_groups`/`load_sensor_csv` are NOT duplicated here —
those already live in `src/features/eng2_features.py` with compatible
semantics (see that module + docs/table_to_source_mapping.md) and are
imported directly by the modules that need the "all 3 sensors present"
group-discovery rule (ENG3, ENG7 proximity). OE9/OE10 need a different,
single-sensor discovery rule (openearable file only) — see
`discover_single_sensor_files` below.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd


def unique_feats(feats: list[str]) -> list[str]:
    """Order-preserving de-duplication, used everywhere a feature list is
    assembled from multiple `[c for c in ... if ...]` blocks that may
    overlap."""
    return list(dict.fromkeys(feats))


def clean_feature_list(df: pd.DataFrame, feats: list[str], min_finite: int = 20, min_std: float = 1e-12) -> list[str]:
    """Drops feature names that aren't columns of `df`, or whose numeric
    values have fewer than `min_finite` finite entries or near-zero
    variance. Identical logic in every notebook cell that builds a
    feature set for LOGO evaluation (CELLs 17/19/20/22)."""
    cleaned = []
    for f in feats:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        if np.isfinite(x.values).sum() < min_finite:
            continue
        if np.nanstd(x.values) < min_std:
            continue
        cleaned.append(f)
    return cleaned


def discover_single_sensor_files(input_dir: str, sensor: str) -> dict:
    """Globs `group_*_{sensor}_model_ready.csv` under input_dir and
    returns {group_id: path} — no multi-sensor availability filter, unlike
    `eng2_features.discover_model_ready_groups`. Used by the OE9/OE10
    builders (CELLs 10/15), which only ever read the openearable file."""
    found = {}
    pattern = os.path.join(input_dir, f"group_*_{sensor}_model_ready.csv")
    for path in glob.glob(pattern):
        m = re.search(rf"group_(\d+)_{sensor}_model_ready", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def sinterp(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linear-interpolates `v(t)` onto `grid`, NaN-filling when fewer than
    2 finite samples are available. Identical across every notebook cell
    that resamples a signal onto a fixed-rate window grid."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    if m.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, t[m], v[m])


def safe_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def safe_std(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def safe_min(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan


def safe_max(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def safe_range(x) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanmax(x) - np.nanmin(x))


def safe_iqr(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def safe_percentile(x, q) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, q))


def mad_diff(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return float(np.nanmedian(np.abs(np.diff(x))))


def event_count(mask) -> int:
    """Counts rising edges (False->True transitions) in a boolean mask."""
    mask = np.asarray(mask).astype(bool)
    if len(mask) < 2:
        return 0
    return int(np.sum(np.diff(mask.astype(int)) == 1))


def spectral_entropy(sig, fs: float) -> float:
    """0 = tonal, 1 = broadband. NaN if fewer than 8 finite samples or
    zero AC power."""
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 8:
        return np.nan
    s = s - np.mean(s)
    p = np.abs(np.fft.rfft(s)) ** 2
    p = p[1:]
    if p.sum() <= 0:
        return np.nan
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def band_ratio(sig, fs: float, lo: float, hi: float) -> float:
    """Fraction of AC spectral power (excluding DC) inside [lo, hi) Hz."""
    s = np.asarray(sig, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 8:
        return np.nan
    s = s - np.mean(s)
    f = np.fft.rfftfreq(len(s), 1 / fs)
    p = np.abs(np.fft.rfft(s)) ** 2
    total = p[1:].sum()
    if total <= 0:
        return np.nan
    return float(p[(f >= lo) & (f < hi)].sum() / total)


def corr_safe(a, b) -> float:
    """Pearson correlation, NaN if fewer than 8 overlapping finite
    samples or either signal has ~zero variance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    a, b = a[m], b[m]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def max_lag_corr(a, b, max_lag_steps: int) -> float:
    """Best-magnitude `corr_safe` over integer lags in
    [-max_lag_steps, +max_lag_steps]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    best = np.nan
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa, bb = a, b
        c = corr_safe(aa, bb)
        if np.isfinite(c) and (not np.isfinite(best) or abs(c) > abs(best)):
            best = c
    return best


def circular_diff(a, b):
    """Angle difference (radians), wrapped to [-pi, pi]."""
    return np.angle(np.exp(1j * (a - b)))


def aggregate_values(row: dict, prefix: str, vals) -> None:
    """Writes `{prefix}_mean/_std/_min/_max/_range` into `row` from a
    small array of per-person (or per-pair) scalar values."""
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_std"] = np.nan
        row[f"{prefix}_min"] = np.nan
        row[f"{prefix}_max"] = np.nan
        row[f"{prefix}_range"] = np.nan
        return
    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_std"] = float(np.std(finite))
    row[f"{prefix}_min"] = float(np.min(finite))
    row[f"{prefix}_max"] = float(np.max(finite))
    row[f"{prefix}_range"] = float(np.max(finite) - np.min(finite))
