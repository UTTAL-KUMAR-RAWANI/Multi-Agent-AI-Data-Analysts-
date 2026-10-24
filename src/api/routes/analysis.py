"""
DataMind AI - Analysis Routes

Endpoints for triggering specialist agents and the full multi-agent LangGraph workflow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.graph.workflow import run_analysis_workflow
from src.graph.nodes import (
    supervisor_plan_node,
    cleaner_node,
    eda_node,
    statistician_node,
    ml_engineer_node,
    visualizer_node,
    synthesize_node,
)
from src.agents.cleaner import DataCleanerAgent
from src.agents.eda import EDAExplorerAgent
from src.agents.statistician import StatisticianAgent
from src.agents.ml_engineer import MLEngineerAgent
from src.agents.visualizer import VisualizerAgent
from src.agents.report_writer import ReportWriterAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class MultiAgentAnalysisRequest(BaseModel):
    dataset_id: str
    objective: str = Field(..., description="Goal or question for the multi-agent system")
    mode: str = Field("full", description="Pipeline mode: full, clean, or eda")


@router.get("/stream/{dataset_id}")
async def stream_multi_agent_pipeline(
    dataset_id: str,
    mode: str = "full",
    objective: str | None = None,
):
    """
    Stream multi-agent analysis execution in strict sequence via Server-Sent Events (SSE).
    Notifies client of step starts, progress, step completions, report creation,
    and cleanly stops the analysis process.
    """
    user_request = objective or "Perform comprehensive end-to-end data analysis, clean issues, detect patterns, train models, and generate publication-quality visual insights."

    async def event_generator():
        state = {
            "dataset_id": dataset_id,
            "user_request": user_request,
            "current_step": 0,
            "visualizations": [],
            "qa_feedback": [],
        }

        def format_sse(data_dict: dict) -> str:
            return f"data: {json.dumps(jsonable_encoder(data_dict))}\n\n"

        try:
            # ── 1. Supervisor Planning ──
            yield format_sse({
                "event": "step_start",
                "step_id": "chip-supervisor",
                "step_index": 1,
                "agent": "Supervisor",
                "title": "1. Supervisor",
                "status": "running",
                "message": "Supervisor is analyzing dataset context & constructing specialist execution plan...",
            })
            plan_res = await supervisor_plan_node(state)
            state.update(plan_res)
            yield format_sse({
                "event": "step_done",
                "step_id": "chip-supervisor",
                "step_index": 1,
                "agent": "Supervisor",
                "status": "completed",
                "message": "Supervisor finalized strategy & delegated specialist agent tasks.",
            })
            await asyncio.sleep(0.2)

            run_cleaner = mode in ("full", "clean")
            run_eda = mode in ("full", "eda")
            run_stats = mode == "full"
            run_ml = mode == "full"
            run_viz = mode in ("full", "eda")

            # ── 2. Data Cleaner ──
            if run_cleaner:
                yield format_sse({
                    "event": "step_start",
                    "step_id": "chip-cleaner",
                    "step_index": 2,
                    "agent": "Data Cleaner",
                    "title": "2. Cleaner",
                    "status": "running",
                    "message": "Data Cleaner is auditing missing values, types, duplicates, and outliers...",
                })
                clean_res = await cleaner_node(state)
                state.update(clean_res)
                audit = clean_res.get("cleaning_report", {}).get("audit_report", {})
                final_rows = audit.get("final_rows", state.get("dataset_metadata", {}).get("num_rows", 0))
                yield format_sse({
                    "event": "step_done",
                    "step_id": "chip-cleaner",
                    "step_index": 2,
                    "agent": "Data Cleaner",
                    "status": "completed",
                    "message": f"Data hygiene audit complete ({final_rows} rows verified).",
                })
                await asyncio.sleep(0.2)

            # ── 3. EDA Explorer ──
            if run_eda:
                yield format_sse({
                    "event": "step_start",
                    "step_id": "chip-eda",
                    "step_index": 3,
                    "agent": "EDA Explorer",
                    "title": "3. EDA Explorer",
                    "status": "running",
                    "message": "EDA Explorer is profiling distributions, summary statistics, and correlations...",
                })
                eda_res = await eda_node(state)
                state.update(eda_res)
                yield format_sse({
                    "event": "step_done",
                    "step_id": "chip-eda",
                    "step_index": 3,
                    "agent": "EDA Explorer",
                    "status": "completed",
                    "message": "Exploratory data analysis profile compiled.",
                })
                await asyncio.sleep(0.2)

            # ── 4. Statistician ──
            if run_stats:
                yield format_sse({
                    "event": "step_start",
                    "step_id": "chip-stats",
                    "step_index": 4,
                    "agent": "Statistician",
                    "title": "4. Statistician",
                    "status": "running",
                    "message": "Statistician is testing hypothesis significance and p-values...",
                })
                stat_res = await statistician_node(state)
                state.update(stat_res)
                yield format_sse({
                    "event": "step_done",
                    "step_id": "chip-stats",
                    "step_index": 4,
                    "agent": "Statistician",
                    "status": "completed",
                    "message": "Hypothesis significance testing complete.",
                })
                await asyncio.sleep(0.2)

            # ── 5. ML Engineer ──
            if run_ml:
                yield format_sse({
                    "event": "step_start",
                    "step_id": "chip-ml",
                    "step_index": 5,
                    "agent": "ML Engineer",
                    "title": "5. ML Engineer",
                    "status": "running",
                    "message": "ML Engineer is benchmarking predictive models (Random Forest, GBDT, Ridge)...",
                })
                ml_res = await ml_engineer_node(state)
                state.update(ml_res)
                best_model = ml_res.get("ml_results", {}).get("ml_results", {}).get("best_model", "AutoML Model")
                yield format_sse({
                    "event": "step_done",
                    "step_id": "chip-ml",
                    "step_index": 5,
                    "agent": "ML Engineer",
                    "status": "completed",
                    "message": f"AutoML benchmark completed (Champion model: {best_model}).",
                })
                await asyncio.sleep(0.2)

            # ── 6. Visualizer ──
            if run_viz:
                yield format_sse({
                    "event": "step_start",
                    "step_id": "chip-viz",
                    "step_index": 6,
                    "agent": "Visualizer",
                    "title": "6. Visualizer",
                    "status": "running",
                    "message": "Visualizer is composing publication-grade interactive BI charts...",
                })
                viz_res = await visualizer_node(state)
                state.update(viz_res)
                n_charts = len(viz_res.get("visualizations", []))
                yield format_sse({
                    "event": "step_done",
                    "step_id": "chip-viz",
                    "step_index": 6,
                    "agent": "Visualizer",
                    "status": "completed",
                    "message": f"Generated {n_charts} interactive Power BI tiles.",
                })
                await asyncio.sleep(0.2)

            # ── 7. QA Agent ──
            yield format_sse({
                "event": "step_start",
                "step_id": "chip-qa",
                "step_index": 7,
                "agent": "QA Agent",
                "title": "7. QA Agent",
                "status": "running",
                "message": "QA Agent is auditing findings for statistical validity and data integrity...",
            })
            qa_items = state.get("qa_feedback", [])
            yield format_sse({
                "event": "step_done",
                "step_id": "chip-qa",
                "step_index": 7,
                "agent": "QA Agent",
                "status": "completed",
                "message": f"Quality verification passed ({len(qa_items)} audits reviewed).",
            })
            await asyncio.sleep(0.2)

            # ── Synthesis ──
            synth_res = await synthesize_node(state)
            state.update(synth_res)

            # ── 8. Report Writer ──
            yield format_sse({
                "event": "step_start",
                "step_id": "chip-report",
                "step_index": 8,
                "agent": "Report Writer",
                "title": "8. Report",
                "status": "running",
                "message": "Report Writer is compiling publication-ready executive HTML report...",
            })
            report_writer = ReportWriterAgent()
            report_res = await report_writer.write_report(state)
            yield format_sse({
                "event": "step_done",
                "step_id": "chip-report",
                "step_index": 8,
                "agent": "Report Writer",
                "status": "completed",
                "message": "Executive intelligence report compiled and ready!",
                "report_url": report_res.get("report_url"),
                "report_id": report_res.get("report_id"),
            })
            await asyncio.sleep(0.2)

            # ── 9. Final Complete & STOP ──
            from src.tools.data_loader import set_dataset_stage, get_dataset
            try:
                set_dataset_stage(dataset_id, "validated", get_dataset(dataset_id))
            except Exception as stg_err:
                logger.warning(f"Could not set validated stage: {stg_err}")

            yield format_sse({
                "event": "pipeline_complete",
                "status": "stopped",
                "success": True,
                "message": "Multi-agent analysis process stopped successfully. All tasks complete.",
                "dataset_id": dataset_id,
                "task_type": state.get("task_type", mode),
                "plan": state.get("task_plan", []),
                "executive_summary": state.get("final_response"),
                "visualizations": state.get("visualizations", []),
                "cleaning_report": state.get("cleaning_report"),
                "eda_results": state.get("eda_results"),
                "ml_results": state.get("ml_results"),
                "statistical_results": state.get("statistical_results"),
                "qa_feedback": state.get("qa_feedback", []),
                "report_url": report_res.get("report_url"),
                "report_id": report_res.get("report_id"),
            })
        except Exception as err:
            logger.error(f"Stream pipeline failed: {err}", exc_info=True)
            yield format_sse({
                "event": "pipeline_error",
                "status": "stopped",
                "error": str(err),
                "message": f"Analysis stopped due to error: {err}",
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run")
async def run_full_pipeline(req: MultiAgentAnalysisRequest):
    """
    Run end-to-end multi-agent LangGraph pipeline on a dataset.
    """
    try:
        final_state = await run_analysis_workflow(
            user_request=req.objective,
            dataset_id=req.dataset_id,
        )

        # Generate report as part of full pipeline
        report_writer = ReportWriterAgent()
        report_res = await report_writer.write_report(final_state)

        result = {
            "success": True,
            "status": "stopped",
            "dataset_id": req.dataset_id,
            "task_type": final_state.get("task_type"),
            "plan": final_state.get("task_plan"),
            "executive_summary": final_state.get("final_response"),
            "visualizations": final_state.get("visualizations", []),
            "cleaning_report": final_state.get("cleaning_report"),
            "eda_results": final_state.get("eda_results"),
            "ml_results": final_state.get("ml_results"),
            "statistical_results": final_state.get("statistical_results"),
            "qa_feedback": final_state.get("qa_feedback", []),
            "report_url": report_res.get("report_url"),
            "report_id": report_res.get("report_id"),
        }
        return jsonable_encoder(result)
    except Exception as e:
        logger.error(f"Multi-agent execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/clean")
async def clean_data(req: CleanRequest):
    """Trigger Data Cleaner Agent."""
    try:
        agent = DataCleanerAgent()
        res = await agent.clean(
            dataset_id=req.dataset_id,
            normalize_cols=req.normalize_cols,
            remove_dups=req.remove_dups,
            convert_types=req.convert_types,
            impute_missing=req.impute_missing,
            handle_outliers=req.handle_outliers,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eda")
async def explore_data(req: EDARequest):
    """Trigger EDA Explorer Agent."""
    try:
        agent = EDAExplorerAgent()
        res = await agent.explore(
            dataset_id=req.dataset_id,
            focus_columns=req.focus_columns,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-model")
async def train_model(req: MLTrainRequest):
    """Trigger ML Engineer Agent."""
    try:
        agent = MLEngineerAgent()
        res = await agent.train_predictive_model(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            task_type=req.task_type,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/statistical-test")
async def run_stat_test(req: StatTestRequest):
    """Trigger Statistician Agent."""
    try:
        agent = StatisticianAgent()
        res = await agent.test_hypothesis(
            dataset_id=req.dataset_id,
            test_type=req.test_type,
            params=req.params,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chart")
async def create_chart(req: ChartRequest):
    """Trigger Visualizer Agent."""
    try:
        agent = VisualizerAgent()
        res = await agent.create_chart(
            dataset_id=req.dataset_id,
            chart_type=req.chart_type,
            params=req.params,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
