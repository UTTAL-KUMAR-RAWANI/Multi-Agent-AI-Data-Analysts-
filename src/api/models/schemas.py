"""
DataMind AI - Pydantic Request/Response Models
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Requests ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request to query a dataset with natural language."""
    question: str = Field(..., description="Natural language question about the data")
    dataset_id: str = Field(..., description="ID of the dataset to query")


class SQLRequest(BaseModel):
    """Request to execute raw SQL on a dataset."""
    sql: str = Field(..., description="SQL query to execute")
    dataset_id: str = Field(..., description="ID of the dataset to query")


# ─── Responses ──────────────────────────────────────────────

class DatasetResponse(BaseModel):
    """Response after uploading/loading a dataset."""
    dataset_id: str
    file_name: str
    num_rows: int
    num_columns: int
    columns: list[dict[str, str]]
    sample_rows: list[dict[str, Any]]
    memory_usage_mb: float
    has_missing: bool
    missing_summary: dict[str, int]


class QueryResponse(BaseModel):
    """Response from a natural language query."""
    question: str
    sql: str
    explanation: str
    interpretation: str
    results: dict[str, Any] | None = None
    error: str | None = None


class DatasetListResponse(BaseModel):
    """Response listing all loaded datasets."""
    datasets: list[DatasetResponse]
    count: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None


# ─── Dashboard Chatbot ──────────────────────────────────────

class DashboardChatMessage(BaseModel):
    """A single chat message in the dashboard assistant conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class DashboardChatRequest(BaseModel):
    """Request to ask the dashboard assistant a question."""
    message: str = Field(..., description="User question about the dashboard, datasets, or agents")
    history: list[DashboardChatMessage] = Field(default_factory=list, description="Conversation history")


class DashboardActionLink(BaseModel):
    """A direct navigation action suggested by the assistant."""
    label: str = Field(..., description="Action button title")
    url: str = Field(..., description="Destination URL")
    icon: str = Field(default="arrow-right", description="Lucide icon name")


class DashboardChatResponse(BaseModel):
    """Response from the dashboard assistant."""
    reply: str = Field(..., description="Markdown-formatted assistant response")
    suggested_actions: list[DashboardActionLink] = Field(default_factory=list, description="Interactive action links")
    quick_replies: list[str] = Field(default_factory=list, description="Follow-up question prompts")

