"""
DataMind AI - EDA Explorer Agent

Conducts comprehensive exploratory data analysis, computes summary metrics,
detects correlations & distribution patterns, and delivers rich data intelligence.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.tools.data_profiler import profile_dataset, compute_group_aggregations

logger = logging.getLogger(__name__)

EDA_SYSTEM_PROMPT = """You are a Senior Principal Data Scientist performing Exploratory Data Analysis (EDA).
Given a detailed data profiling report (overview, numeric distributions, categorical breakdowns, correlations, and anomalies):

Provide a rigorous, executive-grade EDA synthesis structured as follows:
1. **Executive Dataset Summary**: Overall volume, health, and dimension breakdown.
2. **Key Distribution Insights**: Notable central tendencies, dispersion, and skewness across key numeric metrics.
3. **Categorical Patterns & Composition**: Dominant classes and concentration risks.
4. **Correlation & Relationship Matrix**: Significant correlations, potential multicollinearity, or business trade-offs.
5. **Data Anomalies & Risks**: High skew, zero variance, potential IDs, or extreme outliers requiring attention.
6. **Actionable Recommendations**: Recommended features to investigate, transform, or model.

Be specific with actual column names, figures, and percentages. Avoid generic fluff.
"""


class EDAExplorerAgent:
    """
    EDA Explorer Agent for in-depth data exploration and pattern discovery.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("eda")
        else:
            self.llm = llm

    async def explore(
        self,
        dataset_id: str,
        focus_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run deep exploratory profiling and produce LLM-driven insights.
        """
        profile = profile_dataset(dataset_id)

        # Generate intelligent insights
        insights = await self._generate_insights(profile, focus_columns)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "profile": profile,
            "insights": insights,
        }

    async def analyze_groups(
        self,
        dataset_id: str,
        group_col: str,
        agg_col: str,
    ) -> dict[str, Any]:
        """
        Perform group-level aggregation and deliver comparative analysis.
        """
        agg_result = compute_group_aggregations(dataset_id, group_col, agg_col)
        return {
            "success": True,
            "dataset_id": dataset_id,
            "aggregation": agg_result,
        }

    async def _generate_insights(
        self,
        profile: dict[str, Any],
        focus_columns: list[str] | None = None,
    ) -> str:
        # Simplify profile payload to prevent LLM context bloat
        condensed_profile = {
            "overview": profile.get("overview"),
            "numeric_summary": {
                k: {
                    "mean": v.get("mean"),
                    "median": v.get("median"),
                    "std": v.get("std"),
                    "min": v.get("min"),
                    "max": v.get("max"),
                    "skewness": v.get("skewness"),
                    "null_pct": v.get("null_pct"),
                }
                for k, v in list(profile.get("numeric_profiles", {}).items())[:15]
            },
            "categorical_top": {
                k: v.get("top_categories")
                for k, v in list(profile.get("categorical_profiles", {}).items())[:10]
            },
            "top_correlations": profile.get("correlations", [])[:8],
            "anomalies": profile.get("anomalies", []),
        }

        prompt = (
            f"Dataset Profiling Metrics:\n"
            f"```json\n{json.dumps(condensed_profile, indent=2)}\n```\n"
        )
        if focus_columns:
            prompt += f"\nPlease pay special attention to these focus columns: {focus_columns}\n"

        prompt += "\nSynthesize the above findings into a comprehensive Exploratory Data Analysis report."

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=EDA_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"EDA LLM synthesis failed: {e}. Falling back to default.")
            ov = profile.get("overview", {})
            return (
                f"### Exploratory Data Analysis Summary\n\n"
                f"- **Total Rows:** {ov.get('total_rows'):,}\n"
                f"- **Total Columns:** {ov.get('total_columns')}\n"
                f"- **Numeric Columns:** {ov.get('numeric_column_count')}\n"
                f"- **Categorical Columns:** {ov.get('categorical_column_count')}\n"
                f"- **Correlations Found:** {len(profile.get('correlations', []))} strong relationships.\n"
                f"- **Anomalies Detected:** {len(profile.get('anomalies', []))} potential data issues."
            )
