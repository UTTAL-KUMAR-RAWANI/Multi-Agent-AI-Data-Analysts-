"""
DataMind AI - Report Writer Agent

Transforms multi-agent analytics data into publication-ready,
executive-grade reports with embedded data stories, tables, and charts.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.core.state import AnalysisState
from src.tools.report_generator import generate_html_report

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    """
    Report Writer Agent for authoring comprehensive analysis reports.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("report_writer")
        else:
            self.llm = llm

    async def write_report(
        self,
        state: AnalysisState,
        report_title: str | None = None,
    ) -> dict[str, Any]:
        """
        Compile report and return file path & URL.
        """
        report_id, file_path = generate_html_report(state, report_title=report_title)

        return {
            "success": True,
            "report_id": report_id,
            "report_path": file_path,
            "report_url": f"/reports/view/{report_id}",
        }
