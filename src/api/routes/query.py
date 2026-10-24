"""
DataMind AI - Query Routes

Handles natural language queries and raw SQL execution against datasets.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.core.exceptions import DatasetNotFoundError, QueryExecutionError
from src.agents.nl_query import NLQueryAgent
from src.tools.sql_engine import execute_sql
from src.api.models.schemas import QueryRequest, SQLRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])

# Shared agent instance (created on first use)
_agent: NLQueryAgent | None = None


def _get_agent() -> NLQueryAgent:
    """Get or create the NL Query Agent singleton."""
    global _agent
    if _agent is None:
        _agent = NLQueryAgent()
    return _agent


@router.post("/nl", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """
    Query a dataset using natural language.

    The AI translates your question to SQL, executes it against the verified data source,
    and returns both the raw results and a human-readable interpretation.
    """
    agent = _get_agent()

    try:
        result = await agent.query(
            question=request.question,
            dataset_id=request.dataset_id,
            history=request.history,
            preference=request.preference,
        )
    except DatasetNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found. Upload a dataset first.",
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

    return QueryResponse(
        question=request.question,
        sql=result["sql"],
        explanation=result.get("explanation", ""),
        interpretation=result["interpretation"],
        results=result.get("results"),
        error=result.get("error"),
        data_source=result.get("data_source", "raw"),
        data_source_label=result.get("data_source_label", "Raw Upload (Uncleaned)"),
        processing_status=result.get("processing_status", "raw"),
        suggested_followups=result.get("suggested_followups", []),
    )


@router.get("/suggestions/{dataset_id}")
async def get_query_suggestions(dataset_id: str):
    """Get dynamic suggested questions based on dataset schema."""
    from src.tools.data_loader import get_dataset_metadata
    try:
        meta = get_dataset_metadata(dataset_id)
        cols = meta.get("columns", [])
        num_cols = [c["name"] for c in cols if "int" in c["dtype"] or "float" in c["dtype"]]
        cat_cols = [c["name"] for c in cols if c["name"] not in num_cols]

        suggestions = []
        if num_cols:
            c = num_cols[0]
            suggestions.append(f"What is the average and highest {c}?")
            suggestions.append(f"Show the top 10 records by {c}")
        if cat_cols:
            c = cat_cols[0]
            suggestions.append(f"What is the distribution of {c}?")
        suggestions.append("Are there any missing values or quality anomalies?")
        if len(num_cols) >= 2:
            suggestions.append(f"What is the correlation between {num_cols[0]} and {num_cols[1]}?")

        return {"dataset_id": dataset_id, "suggestions": suggestions[:4]}
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "suggestions": [
                "What are the top 10 records?",
                "Show summary statistics for numeric columns",
                "Are there any missing values?",
                "What is the distribution of the primary categories?",
            ]
        }


@router.get("/status/{dataset_id}")
async def get_dataset_processing_status(dataset_id: str):
    """Get the current data lifecycle stage (raw, cleaned, validated) and audit info."""
    from src.tools.data_loader import get_dataset_stage, get_cleaning_audit, get_dataset_metadata
    try:
        meta = get_dataset_metadata(dataset_id)
        stage = get_dataset_stage(dataset_id)
        audit = get_cleaning_audit(dataset_id)
        return {
            "success": True,
            "dataset_id": dataset_id,
            "stage": stage,
            "stage_label": "Raw Upload (Uncleaned)" if stage == "raw" else ("Cleaned Dataset" if stage == "cleaned" else "Validated Analysis Dataset"),
            "num_rows": meta.get("num_rows"),
            "num_columns": meta.get("num_columns"),
            "has_missing": meta.get("has_missing", False),
            "cleaning_audit": audit,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sql")
async def raw_sql_query(request: SQLRequest):
    """
    Execute a raw SQL query against a dataset.

    Use the dataset_id as the table name in your SQL query.
    The alias "data" is also available.
    """
    try:
        results = execute_sql(request.dataset_id, request.sql, target_stage=request.target_stage)
        return results
    except DatasetNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found.",
        )
    except QueryExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
