"""
DataMind AI - Supervisor Agent (Orchestrator)

The central brain of the multi-agent system:
- Understands user intent and dataset context
- Decomposes requests into structured step-by-step execution plans
- Delegates subtasks to specialized agents (Cleaner, EDA, Statistician, ML, Visualizer, NL Query)
- Synthesizes all intermediate findings into a cohesive, executive-ready response
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.core.state import AnalysisState, TaskStep

logger = logging.getLogger(__name__)

SUPERVISOR_CLASSIFY_PROMPT = """You are the Lead Data Analysis Supervisor orchestrating a team of specialized AI agents:
1. "query": For direct natural language SQL questions (e.g. "What was highest revenue in Q3?")
2. "clean": For cleaning, handling missing values, duplicates, outliers, types.
3. "eda": For general exploration, summary statistics, correlation matrices, distributions.
4. "stats": For hypothesis testing, t-tests, ANOVA, Chi-Square, p-values.
5. "ml": For predictive modeling, training classifiers/regressors, feature importance, clustering.
6. "viz": For generating interactive charts, plots, histograms, heatmaps.
7. "full": For comprehensive, end-to-end analysis requiring cleaning, profiling, statistical testing, modeling, and visualization.

Given the user request and dataset metadata, return a JSON object:
```json
{
  "task_type": "query | clean | eda | stats | ml | viz | full",
  "plan": [
    {
      "step_id": 1,
      "agent": "cleaner | eda | statistician | ml_engineer | visualizer | nl_query",
      "action": "short action descriptor",
      "description": "What this step will do"
    }
  ]
}
```
Respond ONLY with the JSON object.
"""

SUPERVISOR_SYNTHESIZE_PROMPT = """You are the Chief Data Officer presenting the final synthesized findings of a multi-agent analysis.
Your team of specialist agents (Data Cleaner, EDA Explorer, Statistician, ML Engineer, Visualizer) has completed the work.

Synthesize all findings into a polished, executive-ready briefing:
1. **Executive Summary**: Direct answer to user request with key takeaways.
2. **Key Quantitative Findings**: Grounded metrics, distributions, and empirical numbers.
3. **Statistical & Predictive Highlights**: Significant relationships, hypothesis results, model accuracy.
4. **Visualizations**: Mention what charts were generated and what they demonstrate.
5. **Strategic Recommendations**: 2-3 clear next steps or business actions based on data evidence.

Format with clean Markdown. Be precise, authoritative, and concise.
"""


class SupervisorAgent:
    """
    Supervisor / Orchestrator Agent.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("supervisor")
        else:
            self.llm = llm

    async def plan_task(self, user_request: str, dataset_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Classify task and formulate an execution plan.
        """
        prompt = (
            f"User Request: {user_request}\n\n"
            f"Dataset Metadata:\n"
            f"- File: {dataset_metadata.get('file_name')}\n"
            f"- Rows: {dataset_metadata.get('num_rows')}\n"
            f"- Columns: {[c.get('name') for c in dataset_metadata.get('columns', [])]}\n"
            f"- Has Missing: {dataset_metadata.get('has_missing')}\n\n"
            f"Classify the task and produce the agent execution plan."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=SUPERVISOR_CLASSIFY_PROMPT),
                HumanMessage(content=prompt),
            ])
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
            plan_data = json.loads(text.strip())
            return plan_data
        except Exception as e:
            logger.warning(f"Supervisor plan generation failed: {e}. Defaulting based on keywords.")
            return self._heuristic_fallback_plan(user_request)

    async def synthesize_results(self, state: AnalysisState) -> str:
        """
        Synthesize intermediate agent outputs into a unified user-facing report.
        """
        collected = {
            "user_request": state.get("user_request"),
            "dataset_id": state.get("dataset_id"),
            "cleaning_report": state.get("cleaning_report"),
            "eda_results": state.get("eda_results"),
            "statistical_results": state.get("statistical_results"),
            "ml_results": state.get("ml_results"),
            "visualizations_count": len(state.get("visualizations", [])),
            "query_result": state.get("query_result"),
        }

        prompt = (
            f"Consolidated Agent Outputs:\n"
            f"```json\n{json.dumps(collected, indent=2, default=str)[:6000]}\n```\n\n"
            f"Deliver the final synthesized briefing to the user."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=SUPERVISOR_SYNTHESIZE_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Supervisor synthesis failed: {e}. Falling back to default.")
            return "Analysis completed successfully across all assigned agents."

    def _heuristic_fallback_plan(self, user_request: str) -> dict[str, Any]:
        """Heuristic plan when LLM is unavailable."""
        req = user_request.lower()
        if any(w in req for w in ["full", "complete analysis", "everything", "deep dive", "comprehensive", "end-to-end", "pipeline"]):
            return {
                "task_type": "full",
                "plan": [
                    {"step_id": 1, "agent": "supervisor", "action": "plan_task", "description": "Orchestrate multi-agent analysis"},
                    {"step_id": 2, "agent": "cleaner", "action": "auto_clean", "description": "Clean dataset anomalies"},
                    {"step_id": 3, "agent": "eda", "action": "profile", "description": "Exploratory data analysis"},
                    {"step_id": 4, "agent": "statistician", "action": "test_hypothesis", "description": "Statistical significance testing"},
                    {"step_id": 5, "agent": "ml_engineer", "action": "train_model", "description": "Train predictive AutoML models"},
                    {"step_id": 6, "agent": "visualizer", "action": "recommend_charts", "description": "Generate interactive visual analytics"},
                    {"step_id": 7, "agent": "qa", "action": "audit_pipeline", "description": "Validate analysis integrity & checks"},
                    {"step_id": 8, "agent": "report_writer", "action": "compile_report", "description": "Generate executive intelligence report"},
                ],
            }
        elif any(w in req for w in ["clean", "null", "missing", "duplicate", "outlier"]):
            return {
                "task_type": "clean",
                "plan": [
                    {"step_id": 1, "agent": "cleaner", "action": "auto_clean", "description": "Clean and preprocess dataset"}
                ],
            }
        elif any(w in req for w in ["model", "predict", "train", "ml", "classify", "cluster"]):
            return {
                "task_type": "ml",
                "plan": [
                    {"step_id": 1, "agent": "ml_engineer", "action": "train_model", "description": "Train predictive model"}
                ],
            }
        elif any(w in req for w in ["chart", "plot", "graph", "visualiz", "histogram", "heatmap"]):
            return {
                "task_type": "viz",
                "plan": [
                    {"step_id": 1, "agent": "visualizer", "action": "create_visuals", "description": "Generate charts"}
                ],
            }
        elif any(w in req for w in ["test", "p-value", "hypothesis", "anova", "t-test", "significant"]):
            return {
                "task_type": "stats",
                "plan": [
                    {"step_id": 1, "agent": "statistician", "action": "test_hypothesis", "description": "Statistical testing"}
                ],
            }
        else:
            return {
                "task_type": "query",
                "plan": [
                    {"step_id": 1, "agent": "nl_query", "action": "query_sql", "description": "Translate to SQL and query"}
                ],
            }
