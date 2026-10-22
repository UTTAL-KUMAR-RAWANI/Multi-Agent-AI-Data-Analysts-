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

    The AI translates your question to SQL, executes it,
    and returns both the raw results and a human-readable interpretation.
    """
    agent = _get_agent()

    try:
        result = await agent.query(request.question, request.dataset_id)
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
    )


@router.post("/sql")
async def raw_sql_query(request: SQLRequest):
    """
    Execute a raw SQL query against a dataset.

    Use the dataset_id as the table name in your SQL query.
    The alias "data" is also available.
    """
    try:
        results = execute_sql(request.dataset_id, request.sql)
        return results
    except DatasetNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found.",
        )
    except QueryExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
