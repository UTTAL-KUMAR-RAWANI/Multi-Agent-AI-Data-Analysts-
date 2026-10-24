"""
DataWhale AI - Analysis Store & State Persistence

Handles saving and retrieving completed multi-agent analysis results and reports
so that returning to a previously analyzed dataset maintains its completed state.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/reports")


def get_analysis_state_path(dataset_id: str) -> Path:
    """Return the JSON state file path for a dataset's analysis."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clean_id = dataset_id.strip()
    return REPORT_DIR / f"analysis_{clean_id}.json"


def save_analysis_result(dataset_id: str, result_dict: dict[str, Any]) -> str:
    """
    Save the completed analysis result to disk for permanent recall.
    """
    try:
        fpath = get_analysis_state_path(dataset_id)
        # Ensure payload is JSON serializable
        clean_data = jsonable_encoder(result_dict)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)
        logger.info(f"Persisted analysis state for dataset '{dataset_id}' to {fpath}")
        return str(fpath)
    except Exception as e:
        logger.warning(f"Failed to persist analysis state for '{dataset_id}': {e}")
        return ""


def get_existing_analysis(dataset_id: str) -> dict[str, Any] | None:
    """
    Retrieve existing analysis results for a dataset if it has already been analyzed.
    Checks:
    1. Dedicated analysis state JSON (data/reports/analysis_{dataset_id}.json).
    2. HTML reports generated for this dataset (data/reports/report_{dataset_id}_*.html).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clean_id = dataset_id.strip()

    # 1. Check dedicated JSON file
    json_path = get_analysis_state_path(clean_id)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    # Verify report file exists or update link if needed
                    report_id = data.get("report_id")
                    if report_id:
                        html_path = REPORT_DIR / f"{report_id}.html"
                        if html_path.exists():
                            data["report_url"] = f"/reports/view/{report_id}"
                            data["download_url"] = f"/reports/download/{report_id}"
                    return data
        except Exception as e:
            logger.warning(f"Error loading analysis JSON {json_path}: {e}")

    # 2. Check for matching report HTML files on disk
    prefix = clean_id.rsplit("_", 1)[0] if "_" in clean_id else clean_id
    matching_reports: list[Path] = []

    for p in REPORT_DIR.glob("*.html"):
        p_name = p.stem.lower()
        if p_name.startswith(f"report_{clean_id.lower()}") or (
            f"_{clean_id.lower()}_" in p_name
        ) or (
            prefix and prefix.lower() in p_name and p_name.startswith("report_")
        ):
            matching_reports.append(p)

    if matching_reports:
        # Sort newest first by modification time
        matching_reports.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_report = matching_reports[0]
        report_id = latest_report.stem

        # Extract executive summary from HTML if possible
        exec_summary = "Autonomous multi-agent pipeline completed. Full executive dossier generated and ready for review."
        try:
            content = latest_report.read_text(encoding="utf-8", errors="ignore")
            # Extract header description paragraph
            match = re.search(r'<header[^>]*>.*?<p[^>]*>(.*?)</p>', content, re.DOTALL)
            if match:
                clean_text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if clean_text:
                    exec_summary = clean_text
        except Exception as read_err:
            logger.debug(f"Could not parse HTML report content: {read_err}")

        recovered_state = {
            "dataset_id": dataset_id,
            "report_id": report_id,
            "report_url": f"/reports/view/{report_id}",
            "download_url": f"/reports/download/{report_id}",
            "status": "completed",
            "success": True,
            "message": "Multi-agent analysis process stopped successfully. All tasks complete.",
            "executive_summary": exec_summary,
            "visualizations": [],
            "plan": [
                {"step": 1, "agent": "Supervisor", "status": "completed"},
                {"step": 2, "agent": "Cleaner", "status": "completed"},
                {"step": 3, "agent": "EDA Explorer", "status": "completed"},
                {"step": 4, "agent": "Statistician", "status": "completed"},
                {"step": 5, "agent": "ML Engineer", "status": "completed"},
                {"step": 6, "agent": "Visualizer", "status": "completed"},
                {"step": 7, "agent": "QA Agent", "status": "completed"},
                {"step": 8, "agent": "Report", "status": "completed"},
            ],
            "cleaning_report": {},
            "eda_results": {},
            "ml_results": {},
            "qa_feedback": [],
        }

        # Cache this state as JSON so future loads are fast
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(recovered_state, f, indent=2)
        except Exception:
            pass

        return recovered_state

    return None
