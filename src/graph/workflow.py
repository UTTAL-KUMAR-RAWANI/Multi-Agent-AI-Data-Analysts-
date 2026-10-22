"""
DataMind AI - LangGraph Analysis Workflow Graph

Builds and compiles the end-to-end multi-agent LangGraph workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from src.core.state import AnalysisState
from src.graph.nodes import (
    supervisor_plan_node,
    cleaner_node,
    eda_node,
    statistician_node,
    ml_engineer_node,
    visualizer_node,
    nl_query_node,
    synthesize_node,
)
from src.graph.edges import route_supervisor_intent

logger = logging.getLogger(__name__)


def build_analysis_graph() -> StateGraph:
    """
    Construct the StateGraph connecting all specialist agents.
    """
    workflow = StateGraph(AnalysisState)

    # 1. Add all agent nodes
    workflow.add_node("supervisor_plan", supervisor_plan_node)
    workflow.add_node("nl_query", nl_query_node)
    workflow.add_node("cleaner", cleaner_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("statistician", statistician_node)
    workflow.add_node("ml_engineer", ml_engineer_node)
    workflow.add_node("visualizer", visualizer_node)
    workflow.add_node("synthesize", synthesize_node)

    # 2. Set entry point
    workflow.set_entry_point("supervisor_plan")

    # 3. Add conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor_plan",
        route_supervisor_intent,
        {
            "nl_query": "nl_query",
            "cleaner": "cleaner",
            "eda": "eda",
            "statistician": "statistician",
            "ml_engineer": "ml_engineer",
            "visualizer": "visualizer",
            "pipeline": "cleaner",
        },
    )

    # 4. Single-agent direct routes to synthesis
    workflow.add_edge("nl_query", END)
    workflow.add_edge("cleaner", "eda")  # In pipeline continues to eda
    workflow.add_edge("eda", "statistician")
    workflow.add_edge("statistician", "ml_engineer")
    workflow.add_edge("ml_engineer", "visualizer")
    workflow.add_edge("visualizer", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow


# Compile the workflow graph
analysis_graph = build_analysis_graph().compile()


async def run_analysis_workflow(
    user_request: str,
    dataset_id: str,
) -> AnalysisState:
    """
    Execute the multi-agent graph for a user request against a dataset.
    """
    initial_state: AnalysisState = {
        "user_request": user_request,
        "dataset_id": dataset_id,
        "current_step": 0,
        "visualizations": [],
        "qa_feedback": [],
    }

    final_state = await analysis_graph.ainvoke(initial_state)
    return final_state
