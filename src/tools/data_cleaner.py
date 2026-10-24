"""
DataMind AI - Data Cleaner Tools

Comprehensive data cleaning and preprocessing utilities using Pandas.
Supports:
- Missing value imputation and removal
- Duplicate detection and removal
- Type inference and conversion
- Outlier detection (IQR and Z-score) and handling
- Column name normalization
- Comprehensive before/after audit reporting
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from src.tools.data_loader import get_dataset, update_dataset, get_dataset_metadata

logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Standardize column names to lower_snake_case and remove special characters.
    """
    renamed = []
    new_cols = []
    for col in df.columns:
        original = str(col)
        # Strip leading/trailing whitespace
        clean = original.strip().lower()
        # Replace non-alphanumeric chars with underscore
        clean = re.sub(r"[^\w\s]", "", clean)
        clean = re.sub(r"[\s\-]+", "_", clean)
        # Remove consecutive underscores
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean:
            clean = "col"

        if clean != original:
            renamed.append({"original": original, "cleaned": clean})
        new_cols.append(clean)

    # Handle collisions if any
    seen: dict[str, int] = {}
    final_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)

    df_cleaned = df.copy()
    df_cleaned.columns = final_cols
    return df_cleaned, renamed


def handle_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """
    Detect and remove duplicate rows.

    Returns:
        tuple: (cleaned DataFrame, count of duplicates removed)
    """
    dup_count = int(df.duplicated(subset=subset).sum())
    if dup_count > 0:
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    else:
        cleaned_df = df.copy()
    return cleaned_df, dup_count


def infer_and_convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Automatically inspect and cast object/string columns to numeric or datetime where appropriate.
    """
    df_converted = df.copy()
    changes = []

    for col in df_converted.columns:
        orig_dtype = str(df_converted[col].dtype)
        if orig_dtype == "object":
            series = df_converted[col].dropna()
            if series.empty:
                continue

            # Try numeric conversion (handle currencies, commas, percentages)
            if series.astype(str).str.match(r"^[\$€£]?\s*[-+]?[0-9,]+(\.[0-9]+)?%?$").all():
                cleaned_series = (
                    df_converted[col]
                    .astype(str)
                    .str.replace(r"[\$€£,]", "", regex=True)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                try:
                    num_series = pd.to_numeric(cleaned_series)
                    df_converted[col] = num_series
                    changes.append({
                        "column": col,
                        "from_dtype": orig_dtype,
                        "to_dtype": str(df_converted[col].dtype),
                        "reason": "String formatted numbers converted to numeric",
                    })
                    continue
                except Exception:
                    pass

            # Try datetime conversion
            if series.astype(str).str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}").any() or \
               series.astype(str).str.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}").any():
                try:
                    dt_series = pd.to_datetime(df_converted[col], errors="coerce")
                    # Only accept if conversion succeeded for > 80% of non-nulls
                    if dt_series.notna().sum() / len(series) >= 0.8:
                        df_converted[col] = dt_series
                        changes.append({
                            "column": col,
                            "from_dtype": orig_dtype,
                            "to_dtype": "datetime64[ns]",
                            "reason": "Detected date/datetime strings",
                        })
                        continue
                except Exception:
                    pass

            # Low cardinality strings -> category
            unique_ratio = df_converted[col].nunique() / len(df_converted)
            if unique_ratio < 0.05 and df_converted[col].nunique() < 50:
                df_converted[col] = df_converted[col].astype("category")
                changes.append({
                    "column": col,
                    "from_dtype": orig_dtype,
                    "to_dtype": "category",
                    "reason": f"Low cardinality ({df_converted[col].nunique()} distinct values)",
                })

    return df_converted, changes


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "smart",
    threshold: float = 0.5,
    custom_rules: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Handle missing values across columns.

    Args:
        df: Input DataFrame.
        strategy: 'smart' (mean/median for numeric, mode for categorical),
                  'drop_rows', 'drop_cols', or 'custom'.
        threshold: Ratio above which a column is dropped entirely if mostly null (default 50%).
        custom_rules: Per-column strategy dict (e.g. {'age': 'median', 'salary': 0}).

    Returns:
        tuple: (cleaned DataFrame, list of changes made)
    """
    df_clean = df.copy()
    changes = []
    initial_rows = len(df_clean)

    # 1. Drop columns exceeding null threshold
    null_ratios = df_clean.isnull().mean()
    high_null_cols = null_ratios[null_ratios > threshold].index.tolist()
    if high_null_cols and strategy == "smart":
        df_clean = df_clean.drop(columns=high_null_cols)
        for c in high_null_cols:
            changes.append({
                "column": c,
                "action": "dropped_column",
                "missing_count": int(df[c].isnull().sum()),
                "missing_ratio": round(float(null_ratios[c]), 3),
                "reason": f"Exceeded missingness threshold ({threshold*100:.0f}%)",
            })

    # 2. Impute or handle remaining missing columns
    for col in df_clean.columns:
        null_count = int(df_clean[col].isnull().sum())
        if null_count == 0:
            continue

        rule = (custom_rules or {}).get(col, strategy)

        if rule == "drop_rows" or strategy == "drop_rows":
            df_clean = df_clean.dropna(subset=[col])
            changes.append({
                "column": col,
                "action": "dropped_rows",
                "missing_count": null_count,
                "reason": "Dropped rows with null values",
            })
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            # If skewed, median is safer; otherwise mean
            skew = df_clean[col].dropna().skew() if len(df_clean[col].dropna()) > 2 else 0
            if abs(skew) > 1.0 or rule == "median":
                fill_val = df_clean[col].median()
                fill_type = "median"
            else:
                fill_val = df_clean[col].mean()
                fill_type = "mean"

            df_clean[col] = df_clean[col].fillna(fill_val)
            changes.append({
                "column": col,
                "action": f"imputed_{fill_type}",
                "missing_count": null_count,
                "fill_value": round(float(fill_val), 3) if isinstance(fill_val, (int, float)) else str(fill_val),
                "reason": f"Imputed with {fill_type} (skew={skew:.2f})",
            })
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].ffill().bfill()
            changes.append({
                "column": col,
                "action": "forward_backward_fill",
                "missing_count": null_count,
                "reason": "Temporal fill for datetime",
            })
        else:
            # Categorical or string -> mode or 'Unknown'
            mode_series = df_clean[col].mode()
            if not mode_series.empty:
                fill_val = mode_series.iloc[0]
                action_name = "imputed_mode"
            else:
                fill_val = "Unknown"
                action_name = "imputed_unknown"

            df_clean[col] = df_clean[col].fillna(fill_val)
            changes.append({
                "column": col,
                "action": action_name,
                "missing_count": null_count,
                "fill_value": str(fill_val),
                "reason": "Imputed categorical missing with mode/unknown",
            })

    return df_clean, changes


def detect_and_handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    factor: float = 1.5,
    action: str = "clip",  # 'clip', 'drop', or 'flag'
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Detect and handle numeric outliers via IQR or Z-Score.

    Args:
        df: Input DataFrame.
        method: 'iqr' (Interquartile Range) or 'zscore'.
        factor: Multiplier for IQR (default 1.5) or Z threshold (default 3.0).
        action: 'clip' (winsorize to boundary), 'drop' (remove rows), or 'flag' (create indicator).

    Returns:
        tuple: (DataFrame with outliers handled, list of outlier summaries)
    """
    df_out = df.copy()
    outlier_reports = []

    numeric_cols = df_out.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df_out[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            if iqr == 0:
                continue
            lower_bound = q25 - (factor * iqr)
            upper_bound = q75 + (factor * iqr)
        else:
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            lower_bound = mean - (factor * std)
            upper_bound = mean + (factor * std)

        outlier_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        count = int(outlier_mask.sum())

        if count > 0:
            if action == "clip":
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "drop":
                df_out = df_out[~outlier_mask].reset_index(drop=True)
            elif action == "flag":
                df_out[f"{col}_is_outlier"] = outlier_mask

            outlier_reports.append({
                "column": col,
                "outlier_count": count,
                "percentage": round((count / len(df)) * 100, 2),
                "lower_bound": round(float(lower_bound), 3),
                "upper_bound": round(float(upper_bound), 3),
                "action_taken": action,
            })

    return df_out, outlier_reports


def auto_clean_dataset(
    dataset_id: str,
    normalize_cols: bool = True,
    remove_dups: bool = True,
    convert_types: bool = True,
    impute_missing: bool = True,
    handle_outliers: bool = True,
    outlier_action: str = "clip",
) -> dict[str, Any]:
    """
    Execute an automated, end-to-end cleaning pipeline on a loaded dataset.
    Updates the dataset in memory and DuckDB, and returns an audit trail.

    Returns:
        dict: Complete cleaning report with before/after stats and changes made.
    """
    df = get_dataset(dataset_id)
    initial_shape = df.shape
    initial_missing = int(df.isnull().sum().sum())
    initial_dups = int(df.duplicated().sum())

    audit_log: dict[str, Any] = {
        "dataset_id": dataset_id,
        "initial_rows": initial_shape[0],
        "initial_cols": initial_shape[1],
        "initial_missing_cells": initial_missing,
        "initial_duplicates": initial_dups,
        "actions_performed": [],
    }

    cleaned_df = df.copy()

    # 1. Normalize Column Names
    if normalize_cols:
        cleaned_df, renamed = clean_column_names(cleaned_df)
        if renamed:
            audit_log["actions_performed"].append({
                "step": "normalize_columns",
                "count": len(renamed),
                "details": renamed,
            })

    # 2. Remove Duplicates
    if remove_dups:
        cleaned_df, dups_removed = handle_duplicates(cleaned_df)
        if dups_removed > 0:
            audit_log["actions_performed"].append({
                "step": "remove_duplicates",
                "count": dups_removed,
                "details": f"Removed {dups_removed} exact duplicate rows",
            })

    # 3. Infer & Convert Data Types
    if convert_types:
        cleaned_df, type_changes = infer_and_convert_types(cleaned_df)
        if type_changes:
            audit_log["actions_performed"].append({
                "step": "convert_types",
                "count": len(type_changes),
                "details": type_changes,
            })

    # 4. Handle Missing Values
    if impute_missing:
        cleaned_df, missing_changes = handle_missing_values(cleaned_df, strategy="smart")
        if missing_changes:
            audit_log["actions_performed"].append({
                "step": "handle_missing",
                "count": len(missing_changes),
                "details": missing_changes,
            })

    # 5. Outliers
    if handle_outliers:
        cleaned_df, outliers = detect_and_handle_outliers(cleaned_df, action=outlier_action)
        if outliers:
            audit_log["actions_performed"].append({
                "step": "handle_outliers",
                "count": len(outliers),
                "details": outliers,
            })

    audit_log["final_rows"] = cleaned_df.shape[0]
    audit_log["final_cols"] = cleaned_df.shape[1]
    audit_log["final_missing_cells"] = int(cleaned_df.isnull().sum().sum())
    audit_log["final_duplicates"] = int(cleaned_df.duplicated().sum())
    audit_log["quality_score_improvement"] = _compute_quality_delta(
        initial_shape[0], initial_missing, initial_dups,
        cleaned_df.shape[0], audit_log["final_missing_cells"], audit_log["final_duplicates"],
    )

    # Update in memory storage & sync with DuckDB with explicit stage & audit
    update_dataset(dataset_id, cleaned_df, stage="cleaned", audit_log=audit_log)

    logger.info(f"Dataset {dataset_id} cleaned successfully. New shape: {cleaned_df.shape}")
    return audit_log


def _compute_quality_delta(
    r_before: int, m_before: int, d_before: int,
    r_after: int, m_after: int, d_after: int,
) -> dict[str, Any]:
    """Calculate simple data quality score out of 100 before and after."""
    total_cells_before = max(r_before * 1, 1)
    score_before = max(0.0, 100.0 - (m_before / total_cells_before * 50) - (d_before / max(r_before, 1) * 50))
    total_cells_after = max(r_after * 1, 1)
    score_after = max(0.0, 100.0 - (m_after / total_cells_after * 50) - (d_after / max(r_after, 1) * 50))
    return {
        "before_score": round(min(score_before, 100.0), 1),
        "after_score": round(min(score_after, 100.0), 1),
        "improvement": round(min(score_after, 100.0) - min(score_before, 100.0), 1),
    }
