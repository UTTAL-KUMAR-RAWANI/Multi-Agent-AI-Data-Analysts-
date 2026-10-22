"""
DataMind AI - LangGraph State Definitions

Defines the shared state that flows through the multi-agent LangGraph workflow.
Each agent reads from and writes to specific fields in this state.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class DatasetMetadata(TypedDict, total=False):
    """Metadata about a loaded dataset."""
    dataset_id: str
    file_name: str
    file_path: str
    num_rows: int
    num_columns: int
    columns: list[dict[str, str]]  # [{"name": "col", "dtype": "int64"}, ...]
    sample_rows: list[dict]         # First 5 rows as dicts
    memory_usage_mb: float
    has_missing: bool
    missing_summary: dict[str, int]  # column -> missing count


class TaskStep(TypedDict, total=False):
    """A single step in the supervisor's execution plan."""
    step_id: int
    agent: str          # Which agent to invoke
    action: str         # What to do
    description: str    # Human-readable description
    status: str         # pending | running | completed | failed
    result_summary: str # Brief summary of what was accomplished


class QAFeedback(TypedDict, total=False):
    """Feedback from the QA agent on an output."""
    agent: str          # Which agent's output was reviewed
    passed: bool
    issues: list[str]
    suggestions: list[str]
    severity: str       # info | warning | error


class AnalysisState(TypedDict, total=False):
    """
    The shared state for the multi-agent analysis workflow.

    This TypedDict defines all fields that agents can read from and write to
    as data flows through the LangGraph workflow.
    """

    # --- User Input ---
    user_request: str                    # The user's natural language request
    dataset_id: str                      # ID of the active dataset
    dataset_metadata: DatasetMetadata    # Schema and stats about the dataset

    # --- Task Planning ---
    task_plan: list[TaskStep]            # Supervisor's execution plan
    current_step: int                    # Index of current step being executed
    task_type: str                       # Classified type: query, clean, eda, stats, ml, viz, report, full

    # --- Agent Outputs ---
    cleaning_report: dict[str, Any]      # Data Cleaner output
    eda_results: dict[str, Any]          # EDA Explorer output
    statistical_results: dict[str, Any]  # Statistician output
    ml_results: dict[str, Any]           # ML Engineer output
    visualizations: list[dict[str, Any]] # Visualizer output (Plotly JSON specs)
    report_path: str                     # Generated report file path
    query_result: dict[str, Any]         # NL Query agent output (SQL + result)

    # --- Quality Assurance ---
    qa_feedback: list[QAFeedback]        # QA reviews
    revision_count: int                  # Track revisions to prevent loops

    # --- Conversation ---
    messages: Annotated[list, add_messages]  # Chat history with LangGraph message handling
    final_response: str                  # Synthesized response to return to user
    error: str                           # Error message if something went wrong
