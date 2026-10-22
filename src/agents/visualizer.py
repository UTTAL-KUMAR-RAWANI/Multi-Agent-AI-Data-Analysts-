"""
DataMind AI - Visualizer Agent

Designs publication-quality interactive data visualizations, selects optimal visual encodings,
and translates complex trends into intuitive visual stories.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.tools.chart_builder import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_box_plot,
    create_correlation_heatmap,
    create_pie_chart,
    recommend_charts_for_dataset,
)

logger = logging.getLogger(__name__)

VIZ_SYSTEM_PROMPT = """You are an Information Designer and Senior Data Visualization Specialist.
Your job is to recommend the single most effective visual representation for the user's data question
and interpret what the resulting visualization communicates.

When provided with dataset column metadata or chart options:
1. Explain why this chart type was chosen (e.g., comparing discrete categories vs displaying continuous density).
2. Highlight key visual cues (peaks, clusters, outliers, trends).
3. Provide a concise, clear takeaway for the user.
"""


class VisualizerAgent:
    """
    Visualizer Agent for intelligent chart creation and visual storytelling.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("visualizer")
        else:
            self.llm = llm

    async def create_chart(
        self,
        dataset_id: str,
        chart_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a chart spec and provide an analytical commentary.
        """
        chart_spec = self._build_spec(dataset_id, chart_type, params)
        narrative = await self._interpret_chart(chart_type, params)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "chart_spec": chart_spec,
            "narrative": narrative,
        }

    async def get_recommended_visualizations(self, dataset_id: str) -> dict[str, Any]:
        """
        Suggest a suite of recommended charts for this dataset.
        """
        recs = recommend_charts_for_dataset(dataset_id)
        charts = []
        for rec in recs[:4]:  # Top 4 recommendations
            try:
                spec = self._build_spec(dataset_id, rec["chart_type"], rec)
                charts.append({
                    "recommendation": rec,
                    "spec": spec,
                })
            except Exception as e:
                logger.warning(f"Failed to generate recommended chart {rec}: {e}")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "recommended_charts": charts,
        }

    def _build_spec(self, dataset_id: str, chart_type: str, params: dict[str, Any]) -> dict[str, Any]:
        ct = chart_type.lower()
        if ct == "bar":
            return create_bar_chart(
                dataset_id,
                x=params["x"],
                y=params["y"],
                title=params.get("title"),
                orientation=params.get("orientation", "v"),
            )
        elif ct == "line":
            return create_line_chart(
                dataset_id,
                x=params["x"],
                y=params["y"],
                color=params.get("color"),
                title=params.get("title"),
            )
        elif ct == "scatter":
            return create_scatter_plot(
                dataset_id,
                x=params["x"],
                y=params["y"],
                color=params.get("color"),
                size=params.get("size"),
                title=params.get("title"),
            )
        elif ct == "histogram":
            return create_histogram(
                dataset_id,
                x=params["x"],
                nbins=params.get("nbins", 30),
                title=params.get("title"),
            )
        elif ct == "box":
            return create_box_plot(
                dataset_id,
                y=params["y"],
                x=params.get("x"),
                title=params.get("title"),
            )
        elif ct in ("correlation_heatmap", "heatmap"):
            return create_correlation_heatmap(
                dataset_id,
                columns=params.get("columns"),
                title=params.get("title"),
            )
        elif ct in ("pie", "donut"):
            return create_pie_chart(
                dataset_id,
                names=params["names"],
                values=params.get("values"),
                title=params.get("title"),
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    async def _interpret_chart(self, chart_type: str, params: dict[str, Any]) -> str:
        prompt = (
            f"Chart Type: {chart_type}\n"
            f"Parameters: {json.dumps(params)}\n\n"
            f"Provide a brief 1-2 sentence visual insight describing what this chart illustrates."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=VIZ_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception:
            return f"Interactive {chart_type} chart rendered."
