"""
DataMind AI - Data Profiler & Exploratory Data Analysis (EDA) Tools

Comprehensive profiling of datasets:
- Descriptive statistics for numerical, categorical, and datetime columns
- Correlation matrix and strong relationship detection
- Data health & anomalies (constants, high cardinality, skewness, outliers)
- Automated data insights generation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.tools.data_loader import get_dataset

logger = logging.getLogger(__name__)


def profile_dataset(dataset_id: str, sample_size: int = 100_000) -> dict[str, Any]:
    """
    Generate a deep exploratory profile of a dataset.

    Args:
        dataset_id: Identifier of the loaded dataset.
        sample_size: Maximum rows to sample for heavy statistical calculations.

    Returns:
        dict: Complete profiling metrics, distribution stats, and detected anomalies.
    """
    df = get_dataset(dataset_id)
    n_rows, n_cols = df.shape

    # Downsample if dataset is very large for quick responsiveness
    is_sampled = n_rows > sample_size
    work_df = df.sample(sample_size, random_state=42) if is_sampled else df

    # 1. Overview
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    duplicates_count = int(df.duplicated().sum())

    # 2. Column Profiling
    numeric_profiles = {}
    categorical_profiles = {}
    datetime_profiles = {}
    anomalies = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / n_rows) * 100, 2)
        n_unique = int(series.nunique(dropna=True))

        # Check constant
        if n_unique <= 1:
            anomalies.append({
                "type": "constant_column",
                "column": col,
                "message": f"Column '{col}' has {n_unique} unique value(s) and provides no variance.",
            })

        # Check potential ID
        if n_unique == n_rows and n_rows > 10:
            anomalies.append({
                "type": "high_cardinality_id",
                "column": col,
                "message": f"Column '{col}' is 100% unique and likely acts as an identifier/primary key.",
            })

        if pd.api.types.is_numeric_dtype(series):
            clean_s = work_df[col].dropna()
            if not clean_s.empty:
                mean_val = float(clean_s.mean())
                std_val = float(clean_s.std()) if len(clean_s) > 1 else 0.0
                q25 = float(clean_s.quantile(0.25))
                median_val = float(clean_s.median())
                q75 = float(clean_s.quantile(0.75))
                iqr = q75 - q25
                skewness = float(clean_s.skew()) if len(clean_s) > 2 else 0.0

                # IQR outliers
                outlier_count = int(((clean_s < (q25 - 1.5 * iqr)) | (clean_s > (q75 + 1.5 * iqr))).sum()) if iqr > 0 else 0

                if abs(skewness) > 2.0:
                    anomalies.append({
                        "type": "high_skewness",
                        "column": col,
                        "skewness": round(skewness, 2),
                        "message": f"Column '{col}' has severe skewness ({skewness:.2f}).",
                    })

                numeric_profiles[col] = {
                    "dtype": str(series.dtype),
                    "null_count": null_count,
                    "null_pct": null_pct,
                    "unique_count": n_unique,
                    "min": round(float(clean_s.min()), 3),
                    "q25": round(q25, 3),
                    "median": round(median_val, 3),
                    "mean": round(mean_val, 3),
                    "q75": round(q75, 3),
                    "max": round(float(clean_s.max()), 3),
                    "std": round(std_val, 3),
                    "skewness": round(skewness, 2),
                    "zero_count": int((clean_s == 0).sum()),
                    "outliers_iqr_count": outlier_count,
                }
            else:
                numeric_profiles[col] = {"dtype": str(series.dtype), "null_count": null_count, "all_null": True}

        elif pd.api.types.is_datetime64_any_dtype(series):
            clean_dt = series.dropna()
            if not clean_dt.empty:
                min_date = clean_dt.min()
                max_date = clean_dt.max()
                datetime_profiles[col] = {
                    "dtype": str(series.dtype),
                    "null_count": null_count,
                    "null_pct": null_pct,
                    "unique_count": n_unique,
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                    "duration_days": int((max_date - min_date).days) if hasattr(max_date - min_date, "days") else None,
                }
            else:
                datetime_profiles[col] = {"dtype": str(series.dtype), "null_count": null_count, "all_null": True}

        else:
            # Categorical / string
            top_vals = series.value_counts(dropna=True).head(5).to_dict()
            top_vals_formatted = [{"value": str(k), "count": int(v), "pct": round((v / n_rows) * 100, 1)} for k, v in top_vals.items()]
            categorical_profiles[col] = {
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": n_unique,
                "top_categories": top_vals_formatted,
            }

    # 3. Correlation Matrix (Numeric)
    num_cols = list(numeric_profiles.keys())
    correlations: list[dict[str, Any]] = []
    correlation_matrix: dict[str, dict[str, float]] = {}

    if len(num_cols) >= 2:
        num_df = work_df[num_cols].dropna()
        if len(num_df) > 5:
            corr_df = num_df.corr(method="pearson").round(3)
            correlation_matrix = corr_df.to_dict()

            # Find top correlated pairs
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_df.loc[c1, c2]
                    if not pd.isna(val):
                        val_float = float(val)
                        if abs(val_float) >= 0.5:
                            correlations.append({
                                "column_1": c1,
                                "column_2": c2,
                                "correlation": val_float,
                                "strength": "very_strong" if abs(val_float) >= 0.8 else "moderate_strong",
                                "direction": "positive" if val_float > 0 else "negative",
                            })

            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "dataset_id": dataset_id,
        "overview": {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "memory_mb": memory_mb,
            "duplicate_rows": duplicates_count,
            "duplicate_pct": round((duplicates_count / max(n_rows, 1)) * 100, 2),
            "numeric_column_count": len(numeric_profiles),
            "categorical_column_count": len(categorical_profiles),
            "datetime_column_count": len(datetime_profiles),
            "is_sampled": is_sampled,
            "sample_size": sample_size if is_sampled else n_rows,
        },
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
        "datetime_profiles": datetime_profiles,
        "correlations": correlations[:10],
        "correlation_matrix": correlation_matrix,
        "anomalies": anomalies,
    }


def compute_group_aggregations(
    dataset_id: str,
    group_col: str,
    agg_col: str,
    agg_funcs: list[str] | None = None,
) -> dict[str, Any]:
    """
    Perform group-by aggregation on a dataset.
    """
    df = get_dataset(dataset_id)
    if group_col not in df.columns or agg_col not in df.columns:
        raise ValueError(f"Columns {group_col} or {agg_col} not found in dataset.")

    funcs = agg_funcs or ["count", "mean", "median", "std", "min", "max"]
    grouped = df.groupby(group_col)[agg_col].agg(funcs).reset_index()

    # Sort by count or mean descending
    sort_col = "count" if "count" in funcs else funcs[0]
    grouped = grouped.sort_values(by=sort_col, ascending=False).head(50)

    # Convert to json serializable records
    records = grouped.round(3).to_dict(orient="records")
    return {
        "group_column": group_col,
        "aggregated_column": agg_col,
        "results": records,
    }
