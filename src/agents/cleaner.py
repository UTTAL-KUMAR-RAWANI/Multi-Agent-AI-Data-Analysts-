"""
DataMind AI - Data Cleaner Agent

Analyzes dataset anomalies, missing values, duplicates, and type mismatches.
Formulates a cleaning strategy and executes it with detailed before/after validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.tools.data_loader import get_dataset_metadata
from src.tools.data_cleaner import auto_clean_dataset

logger = logging.getLogger(__name__)

CLEANER_SYSTEM_PROMPT = """You are an expert Data Cleaning and Preparation Specialist.
Given dataset metadata and an audit of issues (missing values, duplicates, outliers, types),
explain the cleaning actions taken, why each decision was appropriate, and how the data quality improved.

## Output Guidelines:
1. Summarize the initial state (shape, nulls, duplicates).
2. Detail the exact actions taken:
   - Which missing values were imputed and why (e.g. median due to skew vs mode for categories).
   - Any dropped columns or rows and their justification.
   - Outliers handled and the boundaries used.
   - Type conversions or column standardizations.
3. Conclude with the final data health score and suitability for downstream analytics/modeling.
4. Keep the tone professional, concise, and structured with markdown headings or bullets.
"""


class DataCleanerAgent:
    """
    Data Cleaner Agent.
    Evaluates dataset quality and coordinates preprocessing steps.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("cleaner")
        else:
            self.llm = llm

    async def clean(
        self,
        dataset_id: str,
        normalize_cols: bool = True,
        remove_dups: bool = True,
        convert_types: bool = True,
        impute_missing: bool = True,
        handle_outliers: bool = True,
        outlier_action: str = "clip",
    ) -> dict[str, Any]:
        """
        Execute cleaning pipeline and generate an explanatory narrative.
        """
        # 1. Run tool operations
        audit_report = auto_clean_dataset(
            dataset_id=dataset_id,
            normalize_cols=normalize_cols,
            remove_dups=remove_dups,
            convert_types=convert_types,
            impute_missing=impute_missing,
            handle_outliers=handle_outliers,
            outlier_action=outlier_action,
        )

        # 2. Get updated metadata
        updated_meta = get_dataset_metadata(dataset_id)

        # 3. Use LLM to generate narrative summary
        narrative = await self._generate_explanation(audit_report, updated_meta)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "audit_report": audit_report,
            "updated_metadata": updated_meta,
            "narrative": narrative,
        }

    async def _generate_explanation(
        self,
        audit_report: dict[str, Any],
        updated_meta: dict[str, Any],
    ) -> str:
        """Generate human-readable summary of cleaning operations using LLM."""
        prompt = (
            f"Here is the data cleaning audit log:\n"
            f"```json\n{json.dumps(audit_report, indent=2)}\n```\n\n"
            f"Updated Dataset Overview:\n"
            f"- Rows: {updated_meta.get('num_rows')}\n"
            f"- Columns: {updated_meta.get('num_columns')}\n"
            f"- Has missing: {updated_meta.get('has_missing')}\n\n"
            f"Please write a concise, professional executive summary of the cleaning results."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=CLEANER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM explanation generation failed: {e}. Falling back to default.")
            return (
                f"Data cleaning completed successfully. "
                f"Dataset updated from {audit_report.get('initial_rows')} rows to {audit_report.get('final_rows')} rows. "
                f"Cleaned {audit_report.get('initial_missing_cells', 0)} missing values and removed "
                f"{audit_report.get('initial_duplicates', 0)} duplicate rows."
            )
