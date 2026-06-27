"""
utils/compare_utils.py — Helper functions for Phase 2 analysis.

Pure CPU utilities for aggregating results across variants and seeds.
No PyTorch dependency — can run locally without GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_all_results(eval_results_dir: str | Path, holdout: bool = False) -> pd.DataFrame:
    """
    Scan eval_results_dir recursively and load all result JSONs into a DataFrame.

    Supports two directory layouts:

    1) Colab / 01_train_and_eval.ipynb output:
        eval_results_dir/
            {variant}/
                seed_{seed}/
                    results.json          ← test
                    holdout/
                        results.json      ← holdout

    2) Local batch-eval output:
        eval_results_dir/
            {variant}/
                {timestamp}/
                    results_seed_{seed}.json

    Args:
        eval_results_dir: Root directory containing variant subdirectories.
        holdout: If True, return holdout metrics (holdout_* keys with prefix stripped).

    Returns:
        DataFrame with columns: variant, seed, plus requested metrics.
    """
    eval_results_dir = Path(eval_results_dir)
    records = []

    # ------------------------------------------------------------------
    # Pattern 1: Colab structure — results.json under seed_{seed}/ dirs
    # ------------------------------------------------------------------
    for path in eval_results_dir.rglob("results.json"):
        parts = path.parts
        is_holdout_path = "holdout" in parts

        # Only pick paths that match the expected holdout flag
        if holdout != is_holdout_path:
            continue

        # Skip stray results.json (e.g. root-level or timestamp-level aggregates)
        variant = None
        seed = None
        for i, part in enumerate(parts):
            if part.startswith("seed_"):
                seed_str = part.replace("seed_", "")
                try:
                    seed = int(seed_str)
                except ValueError:
                    seed = seed_str
                if i > 0:
                    variant = parts[i - 1]
                break

        if variant is None:
            continue

        with open(path) as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            record = dict(entry)
            if "variant" not in record:
                record["variant"] = variant
            if "seed" not in record and seed is not None:
                record["seed"] = seed
            records.append(record)

    # ------------------------------------------------------------------
    # Pattern 2: Local structure — results_seed_{seed}.json under timestamp dirs
    # ------------------------------------------------------------------
    for path in eval_results_dir.rglob("results_seed_*.json"):
        # Path: .../{variant}/{timestamp}/results_seed_{seed}.json
        variant = path.parent.parent.name
        timestamp = path.parent.name

        with open(path) as f:
            data = json.load(f)

        record = dict(data)
        record["variant"] = variant
        record["_timestamp"] = timestamp
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Deduplicate Pattern 2 by keeping latest timestamp per (variant, seed)
    if "_timestamp" in df.columns:
        df = df.sort_values("_timestamp").drop_duplicates(
            subset=["variant", "seed"], keep="last"
        )
        df = df.drop(columns=["_timestamp"])

    if holdout:
        holdout_cols = [c for c in df.columns if c.startswith("holdout_")]
        if holdout_cols:
            # Pattern 2 (local batch-eval): single file with holdout_* prefix
            rename_map = {c: c.replace("holdout_", "") for c in holdout_cols}
            keep_cols = ["variant", "seed"] + holdout_cols
            df = df[[c for c in keep_cols if c in df.columns]].copy()
            df = df.rename(columns=rename_map)
        # else: Pattern 1 (Colab) — data came from holdout/ dirs, keep as-is
    else:
        holdout_cols = [c for c in df.columns if c.startswith("holdout_")]
        df = df.drop(columns=[c for c in holdout_cols if c in df.columns])

    return df


def aggregate_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by variant: mean ± std across seeds.

    Hanya meng-aggregate kolom numerik (int/float). Kolom non-numerik
    seperti dict (confusion_matrix) di-skip otomatis.

    Returns:
        DataFrame with MultiIndex columns (metric, mean/std/min/max).
    """
    # Hanya ambil kolom numerik (int/float) — skip dict, list, str, object, dll.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [c for c in numeric_cols if c not in ("variant", "seed", "model")]

    if not metrics:
        return pd.DataFrame()

    grouped = df.groupby("variant")
    agg = grouped.agg({m: ["mean", "std", "min", "max"] for m in metrics})
    return agg


def format_mean_std(df: pd.DataFrame, metric: str, decimals: int = 4) -> pd.Series:
    """
    Format a metric as 'mean ± std' string per variant.

    Args:
        df: DataFrame from load_all_results()
        metric: Column name to format
        decimals: Number of decimal places

    Returns:
        Series indexed by variant
    """
    grouped = df.groupby("variant")[metric]
    mean = grouped.mean()
    std = grouped.std()
    return (mean.round(decimals).astype(str) + " ± " + std.round(decimals).astype(str))


def paired_ttest(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    metric: str = "eer",
) -> tuple[float, float, float]:
    """
    Paired t-test between two variants across seeds.

    Args:
        df: DataFrame with variant, seed, and metric columns
        variant_a: First variant name
        variant_b: Second variant name
        metric: Metric to compare

    Returns:
        (t_statistic, p_value, cohens_d)
    """
    a = df[df["variant"] == variant_a].set_index("seed")[metric]
    b = df[df["variant"] == variant_b].set_index("seed")[metric]

    # Align by seed (paired)
    common = a.index.intersection(b.index)
    a = a.loc[common].values
    b = b.loc[common].values

    if len(a) < 2:
        return np.nan, np.nan, np.nan

    t_stat, p_val = stats.ttest_rel(a, b)
    pooled_std = np.std(a - b, ddof=1)
    cohens_d = np.mean(a - b) / pooled_std if pooled_std > 0 else np.nan

    return t_stat, p_val, cohens_d


def export_latex_table(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: str | Path,
    caption: str = "Comparison of Variants",
) -> str:
    """
    Export aggregated results to a LaTeX table snippet.

    Args:
        df: DataFrame from load_all_results()
        metrics: List of metrics to include
        save_path: Where to save the .tex file
        caption: Table caption

    Returns:
        LaTeX string
    """
    rows = []
    rows.append("\\begin{table}[h]")
    rows.append("\\centering")
    rows.append("\\caption{" + caption + "}")
    rows.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    rows.append("\\hline")
    rows.append("Variant & " + " & ".join(metrics) + " \\\\")
    rows.append("\\hline")

    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        cells = [variant.replace("_", "\\_")]
        for m in metrics:
            mean = sub[m].mean()
            std = sub[m].std()
            cells.append(f"${mean:.4f} \\pm {std:.4f}$")
        rows.append(" & ".join(cells) + " \\\\")

    rows.append("\\hline")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")

    latex = "\n".join(rows)
    Path(save_path).write_text(latex)
    return latex
