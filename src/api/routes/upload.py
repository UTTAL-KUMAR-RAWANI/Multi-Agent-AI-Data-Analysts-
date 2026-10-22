"""
DataMind AI - Upload Routes

Handles file upload, dataset listing, and dataset management.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.core.config import get_settings
from src.core.exceptions import DatasetLoadError, DatasetNotFoundError, FileTooLargeError
from src.tools.data_loader import (
    load_dataset,
    list_datasets,
    get_dataset_metadata,
    delete_dataset,
    describe_dataset,
)
from src.api.models.schemas import DatasetResponse, DatasetListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV, Excel, or JSON file as a dataset.

    The file is saved to the uploads directory and loaded into memory.
    Returns metadata about the loaded dataset.
    """
    settings = get_settings()

    # Validate file extension
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json"}
    suffix = Path(file.filename or "unknown").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{suffix}'. Allowed: {allowed_extensions}",
        )

    # Save uploaded file
    upload_dir = settings.get_upload_path()
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.max_upload_size_mb:
        file_path.unlink()  # Clean up
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f} MB). Max: {settings.max_upload_size_mb} MB.",
        )

    # Load into memory
    try:
        dataset_id, metadata = load_dataset(str(file_path))
    except DatasetLoadError as e:
        file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Dataset uploaded: {file.filename} → {dataset_id}")

    return DatasetResponse(
        dataset_id=metadata["dataset_id"],
        file_name=metadata["file_name"],
        num_rows=metadata["num_rows"],
        num_columns=metadata["num_columns"],
        columns=metadata["columns"],
        sample_rows=metadata["sample_rows"],
        memory_usage_mb=metadata["memory_usage_mb"],
        has_missing=metadata["has_missing"],
        missing_summary=metadata["missing_summary"],
    )


@router.get("/", response_model=DatasetListResponse)
async def list_all_datasets():
    """List all loaded datasets."""
    datasets = list_datasets()
    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                dataset_id=m["dataset_id"],
                file_name=m["file_name"],
                num_rows=m["num_rows"],
                num_columns=m["num_columns"],
                columns=m["columns"],
                sample_rows=m["sample_rows"],
                memory_usage_mb=m["memory_usage_mb"],
                has_missing=m["has_missing"],
                missing_summary=m["missing_summary"],
            )
            for m in datasets
        ],
        count=len(datasets),
    )


@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed information about a specific dataset."""
    try:
        metadata = get_dataset_metadata(dataset_id)
        description = describe_dataset(dataset_id)
        return {
            "metadata": metadata,
            "description": description,
        }
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@router.delete("/{dataset_id}")
async def remove_dataset(dataset_id: str):
    """Remove a dataset from the workspace."""
    try:
        get_dataset_metadata(dataset_id)  # Verify it exists
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    delete_dataset(dataset_id)
    return {"message": f"Dataset '{dataset_id}' removed."}
