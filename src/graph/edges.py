"""
DataMind AI - LangGraph Conditional Edges

Routing logic based on task_type classified by the Supervisor Agent.
"""

from __future__ import annotations

from typing import Literal
from src.core.state import AnalysisState


def route_supervisor_intent(
    state: AnalysisState,
) -> Literal["nl_query", "cleaner", "eda", "statistician", "ml_engineer", "visualizer", "pipeline"]:
    """
    Determine the next node to execute after the supervisor has planned the task.
    """
    tt = state.get("task_type", "query").lower()

    if tt == "query":
        return "nl_query"
    elif tt == "clean":
        return "cleaner"
    elif tt == "eda":
        return "eda"
    elif tt == "stats":
        return "statistician"
    elif tt == "ml":
        return "ml_engineer"
    elif tt == "viz":
        return "visualizer"
    else:
        # Full multi-agent pipeline
        return "pipeline"
