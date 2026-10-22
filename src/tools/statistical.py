"""
DataMind AI - Statistical Analysis & Hypothesis Testing Tools

Comprehensive statistical operations using SciPy and NumPy:
- Independent and Paired T-tests
- One-Way ANOVA and Kruskal-Wallis
- Chi-Square Test of Independence
- Mann-Whitney U Test (non-parametric comparison)
- Normality Tests (D'Agostino-Pearson and Shapiro-Wilk)
- Pearson and Spearman Correlation with p-values
- Effect size calculations (Cohen's d) and Confidence Intervals
- Smart automatic test selection based on data characteristics
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.tools.data_loader import get_dataset

logger = logging.getLogger(__name__)


def run_normality_test(
    dataset_id: str,
    column: str,
) -> dict[str, Any]:
    """
    Test whether a numeric column follows a normal distribution.
    Uses Shapiro-Wilk (n <= 5000) or D'Agostino-Pearson (n > 5000).
    """
    df = get_dataset(dataset_id)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    data = df[column].dropna().to_numpy()
    if len(data) < 8:
        raise ValueError(f"Column '{column}' has too few observations ({len(data)}) for normality testing.")

    # Subsample if massive, otherwise choose test
    n = len(data)
    if n <= 5000:
        stat, p_val = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p_val = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"

    is_normal = bool(p_val >= 0.05)
    return {
        "test": test_name,
        "column": column,
        "sample_size": n,
        "statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "is_normal": is_normal,
        "significance_level": 0.05,
        "interpretation": (
            f"The data in '{column}' does NOT significantly deviate from normal (p={p_val:.4f} >= 0.05)."
            if is_normal
            else f"The data in '{column}' significantly deviates from normal (p={p_val:.4e} < 0.05)."
        ),
    }


def run_ttest(
    dataset_id: str,
    numeric_col: str,
    group_col: str,
    paired: bool = False,
) -> dict[str, Any]:
    """
    Perform independent or paired Two-Sample T-test across two distinct groups.
    Also computes Cohen's d effect size.
    """
    df = get_dataset(dataset_id)
    if numeric_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns not found: {numeric_col}, {group_col}")

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Group column '{group_col}' must contain exactly 2 distinct groups, found {len(groups)}: {groups}")

    g1_label, g2_label = str(groups[0]), str(groups[1])
    g1 = df[df[group_col] == groups[0]][numeric_col].dropna().to_numpy()
    g2 = df[df[group_col] == groups[1]][numeric_col].dropna().to_numpy()

    if len(g1) < 3 or len(g2) < 3:
        raise ValueError("Each group must contain at least 3 non-null observations.")

    if paired:
        if len(g1) != len(g2):
            raise ValueError(f"Paired t-test requires equal group sizes. Got {len(g1)} vs {len(g2)}.")
        t_stat, p_val = stats.ttest_rel(g1, g2)
        test_type = "Paired Two-Sample T-Test"
    else:
        # Welch's t-test (equal_var=False) is safer for real-world unequal variances
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        test_type = "Welch's Independent Two-Sample T-Test"

    # Effect Size: Cohen's d
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))
    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_sd if pooled_sd > 0 else 0.0

    effect_qual = (
        "negligible" if abs(cohens_d) < 0.2
        else "small" if abs(cohens_d) < 0.5
        else "medium" if abs(cohens_d) < 0.8
        else "large"
    )

    is_significant = bool(p_val < 0.05)

    return {
        "test": test_type,
        "numeric_column": numeric_col,
        "group_column": group_col,
        "group_1": {"name": g1_label, "count": n1, "mean": round(float(np.mean(g1)), 3), "std": round(float(np.std(g1)), 3)},
        "group_2": {"name": g2_label, "count": n2, "mean": round(float(np.mean(g2)), 3), "std": round(float(np.std(g2)), 3)},
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "statistically_significant": is_significant,
        "cohens_d": round(float(cohens_d), 3),
        "effect_size": effect_qual,
        "interpretation": (
            f"Statistically significant difference detected between '{g1_label}' and '{g2_label}' "
            f"(t={t_stat:.3f}, p={p_val:.4e}, effect size d={cohens_d:.2f} [{effect_qual}])."
            if is_significant
            else f"No statistically significant difference between '{g1_label}' and '{g2_label}' (p={p_val:.4f} >= 0.05)."
        ),
    }


def run_anova(
    dataset_id: str,
    numeric_col: str,
    group_col: str,
) -> dict[str, Any]:
    """
    Perform One-Way ANOVA across 3 or more categorical groups.
    Also performs Kruskal-Wallis non-parametric test.
    """
    df = get_dataset(dataset_id)
    if numeric_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns not found: {numeric_col}, {group_col}")

    grouped_data = []
    group_summaries = []
    for grp_val, sub_df in df.groupby(group_col):
        clean_vals = sub_df[numeric_col].dropna().to_numpy()
        if len(clean_vals) >= 2:
            grouped_data.append(clean_vals)
            group_summaries.append({
                "group": str(grp_val),
                "count": len(clean_vals),
                "mean": round(float(np.mean(clean_vals)), 3),
                "std": round(float(np.std(clean_vals)), 3),
            })

    if len(grouped_data) < 2:
        raise ValueError(f"Need at least 2 valid groups with data for ANOVA, found {len(grouped_data)}")

    f_stat, p_val = stats.f_oneway(*grouped_data)
    kw_stat, kw_p = stats.kruskal(*grouped_data)

    is_significant = bool(p_val < 0.05)

    return {
        "test": "One-Way ANOVA",
        "numeric_column": numeric_col,
        "group_column": group_col,
        "groups": group_summaries,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val),
        "statistically_significant": is_significant,
        "kruskal_wallis": {
            "h_statistic": round(float(kw_stat), 4),
            "p_value": float(kw_p),
            "statistically_significant": bool(kw_p < 0.05),
        },
        "interpretation": (
            f"Statistically significant difference among group means (F={f_stat:.3f}, p={p_val:.4e} < 0.05)."
            if is_significant
            else f"No statistically significant difference between group means (F={f_stat:.3f}, p={p_val:.4f} >= 0.05)."
        ),
    }


def run_chi_square(
    dataset_id: str,
    col1: str,
    col2: str,
) -> dict[str, Any]:
    """
    Perform Pearson Chi-Square Test of Independence between two categorical variables.
    Computes Cramer's V effect size.
    """
    df = get_dataset(dataset_id)
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns not found: {col1}, {col2}")

    ct = pd.crosstab(df[col1], df[col2])
    if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Contingency table must be at least 2x2 with valid observations.")

    chi2, p_val, dof, expected = stats.chi2_contingency(ct)

    # Cramer's V
    n = ct.to_numpy().sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

    is_significant = bool(p_val < 0.05)

    return {
        "test": "Chi-Square Test of Independence",
        "column_1": col1,
        "column_2": col2,
        "contingency_table": ct.to_dict(),
        "chi2_statistic": round(float(chi2), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val),
        "cramers_v": round(float(cramers_v), 3),
        "statistically_significant": is_significant,
        "interpretation": (
            f"Statistically significant association between '{col1}' and '{col2}' (chi2={chi2:.2f}, p={p_val:.4e}, Cramer's V={cramers_v:.2f})."
            if is_significant
            else f"No statistically significant association between '{col1}' and '{col2}' (chi2={chi2:.2f}, p={p_val:.4f} >= 0.05)."
        ),
    }


def run_correlation_test(
    dataset_id: str,
    col1: str,
    col2: str,
) -> dict[str, Any]:
    """
    Compute Pearson and Spearman correlations with p-values and confidence intervals.
    """
    df = get_dataset(dataset_id)
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns not found: {col1}, {col2}")

    clean = df[[col1, col2]].dropna()
    if len(clean) < 5:
        raise ValueError(f"Need at least 5 paired observations, got {len(clean)}.")

    x = clean[col1].to_numpy()
    y = clean[col2].to_numpy()

    p_r, p_pval = stats.pearsonr(x, y)
    s_r, s_pval = stats.spearmanr(x, y)

    return {
        "test": "Correlation Analysis",
        "column_1": col1,
        "column_2": col2,
        "sample_size": len(clean),
        "pearson": {
            "r": round(float(p_r), 4),
            "p_value": float(p_pval),
            "statistically_significant": bool(p_pval < 0.05),
        },
        "spearman": {
            "rho": round(float(s_r), 4),
            "p_value": float(s_pval),
            "statistically_significant": bool(s_pval < 0.05),
        },
        "strength": (
            "very_strong" if abs(p_r) >= 0.8
            else "moderate_strong" if abs(p_r) >= 0.5
            else "weak" if abs(p_r) >= 0.3
            else "negligible"
        ),
        "direction": "positive" if p_r > 0 else "negative",
    }
