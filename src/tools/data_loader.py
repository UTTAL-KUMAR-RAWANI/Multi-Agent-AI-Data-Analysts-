"""
DataMind AI - Data Loader Tool

Handles loading CSV, Excel, and JSON files into pandas DataFrames,
with automatic metadata extraction and validation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.exceptions import DatasetLoadError, DatasetNotFoundError
from src.core.state import DatasetMetadata

logger = logging.getLogger(__name__)

# In-memory dataset registry: dataset_id -> DataFrame
_datasets: dict[str, pd.DataFrame] = {}
_metadata: dict[str, DatasetMetadata] = {}


def _generate_dataset_id(file_path: str) -> str:
    """Generate a short, unique dataset ID from the file path."""
    name = Path(file_path).stem
    hash_suffix = hashlib.md5(file_path.encode()).hexdigest()[:6]
    return f"{name}_{hash_suffix}"


def load_dataset(
    file_path: str,
    encoding: str = "utf-8",
    delimiter: str | None = None,
) -> tuple[str, DatasetMetadata]:
    """
    Load a dataset from a file into the workspace.

    Supports CSV, Excel (.xlsx/.xls), and JSON files.
    Automatically detects format from file extension.

    Args:
        file_path: Path to the data file.
        encoding: Text encoding (default utf-8).
        delimiter: CSV delimiter (auto-detected if None).

    Returns:
        Tuple of (dataset_id, metadata).

    Raises:
        DatasetLoadError: If the file can't be loaded.
    """
    path = Path(file_path)

    if not path.exists():
        raise DatasetLoadError(file_path, "File does not exist.")

    suffix = path.suffix.lower()
    logger.info(f"Loading dataset from: {path} (format: {suffix})")

    try:
        if suffix == ".csv":
            df = pd.read_csv(path, encoding=encoding, delimiter=delimiter)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".json":
            df = pd.read_json(path, encoding=encoding)
        else:
            raise DatasetLoadError(
                file_path,
                f"Unsupported file format: '{suffix}'. Supported: .csv, .xlsx, .xls, .json"
            )
    except DatasetLoadError:
        raise
    except Exception as e:
        raise DatasetLoadError(file_path, str(e))

    # Generate ID and store
    dataset_id = _generate_dataset_id(file_path)
    _datasets[dataset_id] = df

    # Extract metadata
    missing_summary = df.isnull().sum()
    missing_dict = {
        col: int(count)
        for col, count in missing_summary.items()
        if count > 0
    }

    metadata: DatasetMetadata = {
        "dataset_id": dataset_id,
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ],
        "sample_rows": df.head(5).to_dict(orient="records"),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "has_missing": bool(missing_dict),
        "missing_summary": missing_dict,
    }
    _metadata[dataset_id] = metadata

    logger.info(
        f"Dataset loaded: id={dataset_id}, shape=({metadata['num_rows']}, {metadata['num_columns']}), "
        f"memory={metadata['memory_usage_mb']}MB"
    )

    return dataset_id, metadata


def _find_dataset_on_disk(dataset_id: str) -> Path | None:
    """Find a matching data file in uploads by filename, stem, or prefix."""
    try:
        settings = get_settings()
        dirs = [settings.get_upload_path(), Path("data/uploads")]
    except Exception:
        dirs = [Path("data/uploads")]

    clean_id = dataset_id.lower().strip()
    prefix = clean_id.rsplit("_", 1)[0] if "_" in clean_id else clean_id

    for d in dirs:
        if not d.exists():
            continue
        # Direct file check with extensions
        for ext in [".csv", ".xlsx", ".xls", ".json"]:
            p1 = d / f"{dataset_id}{ext}"
            if p1.exists():
                return p1
            p2 = d / f"{prefix}{ext}"
            if p2.exists():
                return p2
        # Check files in directory
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in [".csv", ".xlsx", ".xls", ".json"]:
                stem_low = f.stem.lower()
                if stem_low == clean_id or stem_low == prefix or clean_id.startswith(stem_low) or stem_low.startswith(prefix):
                    return f
    return None


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Get a loaded DataFrame by its ID. Auto-resolves from memory or disk if needed.

    Raises:
        DatasetNotFoundError: If the dataset ID doesn't exist.
    """
    if dataset_id in _datasets:
        return _datasets[dataset_id]

    # Check prefix or partial match in currently loaded datasets
    clean_id = dataset_id.lower().strip()
    prefix = clean_id.rsplit("_", 1)[0] if "_" in clean_id else clean_id
    for k, df in _datasets.items():
        k_low = k.lower()
        if k_low == clean_id or k_low.startswith(prefix) or clean_id.startswith(k_low.rsplit("_", 1)[0]):
            _datasets[dataset_id] = df
            if k in _metadata:
                _metadata[dataset_id] = _metadata[k]
            return df

    # Auto-load from uploads on disk
    matched_file = _find_dataset_on_disk(dataset_id)
    if matched_file:
        try:
            gen_id, meta = load_dataset(str(matched_file))
            _datasets[dataset_id] = _datasets[gen_id]
            _metadata[dataset_id] = meta
            return _datasets[dataset_id]
        except Exception as e:
            logger.warning(f"Found file {matched_file} for {dataset_id} but loading failed: {e}")

    raise DatasetNotFoundError(dataset_id)


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    """
    Get metadata for a loaded dataset. Auto-resolves from memory or disk if needed.

    Raises:
        DatasetNotFoundError: If the dataset ID doesn't exist.
    """
    if dataset_id in _metadata:
        return _metadata[dataset_id]

    # Calling get_dataset will load/alias if available on disk
    try:
        get_dataset(dataset_id)
        if dataset_id in _metadata:
            return _metadata[dataset_id]
    except Exception:
        pass

    raise DatasetNotFoundError(dataset_id)


def update_dataset(dataset_id: str, df: pd.DataFrame) -> DatasetMetadata:
    """
    Update a dataset in the workspace (e.g., after cleaning).

    Returns updated metadata.
    """
    if dataset_id not in _datasets:
        raise DatasetNotFoundError(dataset_id)

    _datasets[dataset_id] = df

    # Refresh metadata
    old_meta = _metadata[dataset_id]
    missing_summary = df.isnull().sum()
    missing_dict = {
        col: int(count)
        for col, count in missing_summary.items()
        if count > 0
    }

    _metadata[dataset_id] = {
        **old_meta,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ],
        "sample_rows": df.head(5).to_dict(orient="records"),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "has_missing": bool(missing_dict),
        "missing_summary": missing_dict,
    }

    return _metadata[dataset_id]


def list_datasets() -> list[DatasetMetadata]:
    """List all loaded datasets with their metadata."""
    return list(_metadata.values())


def delete_dataset(dataset_id: str) -> None:
    """Remove a dataset from the workspace."""
    _datasets.pop(dataset_id, None)
    _metadata.pop(dataset_id, None)


def describe_dataset(dataset_id: str) -> str:
    """
    Get a human-readable description of a dataset.

    This is what gets passed to LLM agents as context about the data.
    """
    meta = get_dataset_metadata(dataset_id)
    df = get_dataset(dataset_id)

    # Build a concise but informative description
    lines = [
        f"Dataset: {meta['file_name']}",
        f"Shape: {meta['num_rows']} rows × {meta['num_columns']} columns",
        f"Memory: {meta['memory_usage_mb']} MB",
        "",
        "Columns:",
    ]

    for col_info in meta["columns"]:
        col = col_info["name"]
        dtype = col_info["dtype"]
        missing = meta["missing_summary"].get(col, 0)
        nunique = df[col].nunique()

        line = f"  - {col} ({dtype}): {nunique} unique values"
        if missing > 0:
            line += f", {missing} missing ({missing/meta['num_rows']*100:.1f}%)"
        lines.append(line)

    # Add statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("")
        lines.append("Numeric Summary:")
        desc = df[numeric_cols].describe().round(2)
        lines.append(desc.to_string())

    return "\n".join(lines)
