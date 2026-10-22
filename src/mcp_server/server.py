"""
DataWhale AI - Model Context Protocol (MCP) Server

Exposes DataWhale AI's data analytics, machine learning, and multi-agent
workflow as standardized MCP tools. Compatible with Claude Desktop, Cursor,
VS Code Copilot, and any MCP client.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.tools.data_loader import (
    load_dataset as _load_dataset,
    describe_dataset as _describe_dataset,
    list_datasets as _list_datasets,
)
from src.tools.data_cleaner import auto_clean_dataset as _auto_clean_dataset
from src.tools.data_profiler import profile_dataset as _profile_dataset
from src.tools.sql_engine import execute_sql as _execute_sql
from src.tools.statistical import (
    run_ttest as _run_ttest,
    run_anova as _run_anova,
    run_chi_square as _run_chi_square,
    run_correlation_test as _run_correlation_test,
    run_normality_test as _run_normality_test,
)
from src.tools.ml_tools import train_and_evaluate_model as _train_ml
from src.tools.chart_builder import (
    create_bar_chart as _bar_chart,
    create_line_chart as _line_chart,
    create_scatter_plot as _scatter_chart,
    create_histogram as _hist_chart,
)
from src.graph.workflow import run_analysis_workflow

logger = logging.getLogger(__name__)

# Initialize MCP Server instance
mcp_server = MCPServer("datawhale-analytics")


@mcp_server.tool()
def load_dataset(file_path: str) -> str:
    """
    Load a CSV, Excel, or JSON dataset into the DataWhale workspace.

    Args:
        file_path: Absolute or relative path to the data file.
    """
    try:
        ds_id, meta = _load_dataset(file_path)
        return json.dumps({
            "status": "success",
            "dataset_id": ds_id,
            "file_name": meta.get("file_name"),
            "num_rows": meta.get("num_rows"),
            "num_columns": meta.get("num_columns"),
            "columns": meta.get("columns"),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def describe_dataset(dataset_id: str) -> str:
    """
    Get detailed schema, data types, null counts, and summary statistics for a dataset.
    """
    try:
        return _describe_dataset(dataset_id)
    except Exception as e:
        return f"Error: {e}"


@mcp_server.tool()
def list_available_datasets() -> str:
    """
    List all currently loaded datasets in the DataWhale workspace.
    """
    datasets = _list_datasets()
    return json.dumps([
        {
            "dataset_id": d.get("dataset_id"),
            "file_name": d.get("file_name"),
            "rows": d.get("num_rows"),
            "columns": d.get("num_columns"),
        }
        for d in datasets
    ], indent=2)


@mcp_server.tool()
def clean_dataset(
    dataset_id: str,
    normalize_columns: bool = True,
    remove_duplicates: bool = True,
    impute_missing: bool = True,
    handle_outliers: bool = True,
) -> str:
    """
    Automatically clean a dataset (impute missing values, drop duplicates, fix types, clip outliers).
    """
    try:
        res = _auto_clean_dataset(
            dataset_id,
            normalize_cols=normalize_columns,
            remove_dups=remove_duplicates,
            impute_missing=impute_missing,
            handle_outliers=handle_outliers,
        )
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def profile_data(dataset_id: str) -> str:
    """
    Run exploratory data analysis (EDA) profiling, calculating skewness, quantiles, and correlations.
    """
    try:
        res = _profile_dataset(dataset_id)
        return json.dumps(res, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a DuckDB SQL query against loaded datasets.
    Tables are accessible by their dataset_id or alias 'data'.
    """
    try:
        cols, rows = _execute_sql(sql_query)
        return json.dumps({"columns": cols, "rows": rows, "count": len(rows)}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def run_statistical_test(
    dataset_id: str,
    test_type: str,
    col1: str,
    col2: str,
) -> str:
    """
    Execute statistical hypothesis test: 'ttest', 'anova', 'chi_square', 'correlation', or 'normality'.
    """
    try:
        t = test_type.lower()
        if t in ("ttest", "t_test"):
            res = _run_ttest(dataset_id, numeric_col=col1, group_col=col2)
        elif t == "anova":
            res = _run_anova(dataset_id, numeric_col=col1, group_col=col2)
        elif t in ("chi_square", "chi2"):
            res = _run_chi_square(dataset_id, col1=col1, col2=col2)
        elif t in ("correlation", "corr"):
            res = _run_correlation_test(dataset_id, col1=col1, col2=col2)
        elif t == "normality":
            res = _run_normality_test(dataset_id, column=col1)
        else:
            return f"Unsupported test type: {test_type}"
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def train_ml_model(
    dataset_id: str,
    target_column: str,
) -> str:
    """
    Auto-train and benchmark machine learning models (Random Forest, Gradient Boosting, Linear/Logistic).
    """
    try:
        res = _train_ml(dataset_id, target_column=target_column)
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp_server.tool()
def full_multi_agent_analysis(
    dataset_id: str,
    objective: str,
) -> str:
    """
    Execute the autonomous multi-agent LangGraph pipeline (Supervisor -> Cleaner -> EDA -> Stats -> ML -> Visualizer -> QA).
    """
    import asyncio
    try:
        final_state = asyncio.run(run_analysis_workflow(
            user_request=objective,
            dataset_id=dataset_id,
        ))
        return json.dumps({
            "status": "success",
            "task_type": final_state.get("task_type"),
            "executive_summary": final_state.get("final_response"),
            "qa_feedback": final_state.get("qa_feedback"),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def run():
    """Run the MCP server over stdio."""
    mcp_server.run()


if __name__ == "__main__":
    run()
