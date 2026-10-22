"""
DataMind AI - LangGraph Workflow Nodes

Defines the individual execution nodes for the multi-agent graph:
Each node receives AnalysisState, invokes the corresponding agent,
updates the state, and passes it forward.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import AnalysisState, QAFeedback
from src.agents.supervisor import SupervisorAgent
from src.agents.cleaner import DataCleanerAgent
from src.agents.eda import EDAExplorerAgent
from src.agents.statistician import StatisticianAgent
from src.agents.ml_engineer import MLEngineerAgent
from src.agents.visualizer import VisualizerAgent
from src.agents.nl_query import NLQueryAgent
from src.agents.qa import QAAgent
from src.tools.data_loader import get_dataset_metadata

logger = logging.getLogger(__name__)

# Agent Singletons
_supervisor = SupervisorAgent()
_cleaner = DataCleanerAgent()
_eda = EDAExplorerAgent()
_statistician = StatisticianAgent()
_ml = MLEngineerAgent()
_visualizer = VisualizerAgent()
_nl_query = NLQueryAgent()
_qa = QAAgent()


async def supervisor_plan_node(state: AnalysisState) -> dict[str, Any]:
    """Classifies task and forms execution plan."""
    dataset_id = state.get("dataset_id")
    meta = state.get("dataset_metadata") or (get_dataset_metadata(dataset_id) if dataset_id else {})

    plan_info = await _supervisor.plan_task(
        user_request=state.get("user_request", ""),
        dataset_metadata=meta,
    )

    return {
        "task_type": plan_info.get("task_type", "query"),
        "task_plan": plan_info.get("plan", []),
        "current_step": 0,
        "dataset_metadata": meta,
    }


async def cleaner_node(state: AnalysisState) -> dict[str, Any]:
    """Data Cleaner Agent node."""
    dataset_id = state["dataset_id"]
    res = await _cleaner.clean(dataset_id=dataset_id)

    # QA review
    feedback = await _qa.review_output("cleaner", res, context={"dataset_id": dataset_id})

    qa_list = list(state.get("qa_feedback", []))
    qa_list.append(feedback)

    return {
        "cleaning_report": res,
        "dataset_metadata": res.get("updated_metadata"),
        "qa_feedback": qa_list,
    }


async def eda_node(state: AnalysisState) -> dict[str, Any]:
    """EDA Explorer Agent node."""
    dataset_id = state["dataset_id"]
    res = await _eda.explore(dataset_id=dataset_id)

    feedback = await _qa.review_output("eda", res, context={"dataset_id": dataset_id})

    qa_list = list(state.get("qa_feedback", []))
    qa_list.append(feedback)

    return {
        "eda_results": res,
        "qa_feedback": qa_list,
    }


async def statistician_node(state: AnalysisState) -> dict[str, Any]:
    """Statistician Agent node."""
    dataset_id = state["dataset_id"]
    # Infer test or default to correlation/normality if unspecified
    req = state.get("user_request", "").lower()
    meta = state.get("dataset_metadata", {})
    num_cols = [c["name"] for c in meta.get("columns", []) if "int" in c["dtype"] or "float" in c["dtype"]]

    if len(num_cols) >= 2:
        res = await _statistician.test_hypothesis(
            dataset_id,
            test_type="correlation",
            params={"col1": num_cols[0], "col2": num_cols[1]},
        )
    elif len(num_cols) == 1:
        res = await _statistician.test_hypothesis(
            dataset_id,
            test_type="normality",
            params={"column": num_cols[0]},
        )
    else:
        res = {"success": False, "message": "No numeric columns for statistical testing."}

    feedback = await _qa.review_output("statistician", res, context={"dataset_id": dataset_id})
    qa_list = list(state.get("qa_feedback", []))
    qa_list.append(feedback)

    return {
        "statistical_results": res,
        "qa_feedback": qa_list,
    }


async def ml_engineer_node(state: AnalysisState) -> dict[str, Any]:
    """ML Engineer Agent node."""
    dataset_id = state["dataset_id"]
    meta = state.get("dataset_metadata", {})
    cols = [c["name"] for c in meta.get("columns", [])]

    # Pick candidate target column (last column by convention or based on request)
    target_col = cols[-1] if cols else ""
    res = await _ml.train_predictive_model(
        dataset_id=dataset_id,
        target_column=target_col,
    )

    feedback = await _qa.review_output("ml_engineer", res, context={"dataset_id": dataset_id, "target": target_col})
    qa_list = list(state.get("qa_feedback", []))
    qa_list.append(feedback)

    return {
        "ml_results": res,
        "qa_feedback": qa_list,
    }


async def visualizer_node(state: AnalysisState) -> dict[str, Any]:
    """Visualizer Agent node."""
    dataset_id = state["dataset_id"]
    res = await _visualizer.get_recommended_visualizations(dataset_id=dataset_id)

    charts = res.get("recommended_charts", [])
    viz_list = [c["spec"] for c in charts]

    return {
        "visualizations": viz_list,
    }


async def nl_query_node(state: AnalysisState) -> dict[str, Any]:
    """NL Query Agent node."""
    dataset_id = state["dataset_id"]
    question = state["user_request"]

    res = await _nl_query.query(question=question, dataset_id=dataset_id)

    return {
        "query_result": res,
        "final_response": res.get("answer", "Query executed."),
    }


async def synthesize_node(state: AnalysisState) -> dict[str, Any]:
    """Supervisor Synthesis node."""
    summary = await _supervisor.synthesize_results(state)
    return {
        "final_response": summary,
    }
