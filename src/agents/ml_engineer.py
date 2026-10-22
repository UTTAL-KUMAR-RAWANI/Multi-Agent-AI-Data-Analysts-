"""
DataMind AI - ML Engineer Agent

Automates predictive modeling, model benchmarking, feature importance extraction,
and model interpretability for tabular datasets.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.tools.ml_tools import train_and_evaluate_model, train_clustering

logger = logging.getLogger(__name__)

ML_SYSTEM_PROMPT = """You are a Lead Machine Learning Engineer.
You have trained and benchmarked multiple machine learning algorithms on a dataset.
Review the benchmark results, metrics, and feature importance.

Provide a comprehensive, production-ready modeling summary:
1. **Model Performance Summary**: Compare the evaluated algorithms (e.g., Random Forest vs Gradient Boosting vs Linear baseline) and declare the champion model.
2. **Evaluation Metrics Interpretation**: Explain R2 / RMSE (for regression) or Accuracy / F1 / Precision / Recall (for classification) in concrete business terms.
3. **Key Drivers (Feature Importance)**: Highlight top predictive signals and what business intuition they confirm or challenge.
4. **Overfitting & Generalization Assessment**: Note train/test splits, variance/bias observations.
5. **Production Deployment Recommendations**: Next steps for monitoring, data drift considerations, or potential feature engineering improvements.
"""


class MLEngineerAgent:
    """
    ML Engineer Agent for automated machine learning model building and evaluation.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("ml_engineer")
        else:
            self.llm = llm

    async def train_predictive_model(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: list[str] | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Train and evaluate predictive models for a given target.
        """
        ml_results = train_and_evaluate_model(
            dataset_id=dataset_id,
            target_column=target_column,
            feature_columns=feature_columns,
            task_type=task_type,
        )

        narrative = await self._generate_ml_summary(ml_results)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "ml_results": ml_results,
            "narrative": narrative,
        }

    async def cluster_data(
        self,
        dataset_id: str,
        feature_columns: list[str],
        n_clusters: int = 3,
    ) -> dict[str, Any]:
        """
        Execute unsupervised clustering and interpret segment characteristics.
        """
        cluster_results = train_clustering(
            dataset_id=dataset_id,
            feature_columns=feature_columns,
            n_clusters=n_clusters,
        )
        return {
            "success": True,
            "dataset_id": dataset_id,
            "cluster_results": cluster_results,
        }

    async def _generate_ml_summary(self, ml_results: dict[str, Any]) -> str:
        prompt = (
            f"Machine Learning Experiment Results:\n"
            f"```json\n{json.dumps(ml_results, indent=2)}\n```\n\n"
            f"Generate a rigorous machine learning engineering report based on these results."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=ML_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"ML LLM summary failed: {e}. Falling back to default.")
            bm = ml_results.get("best_model", "Unknown")
            metrics = ml_results.get("best_metrics", {})
            return (
                f"### Machine Learning Benchmark Summary\n\n"
                f"- **Champion Model:** `{bm}`\n"
                f"- **Key Metrics:** {metrics}\n"
                f"- **Evaluated Models:** {len(ml_results.get('model_benchmarks', []))} models trained and validated."
            )
