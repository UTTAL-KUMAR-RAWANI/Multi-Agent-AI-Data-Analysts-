"""
DataMind AI - Statistician Agent

Performs rigorous hypothesis testing, selects appropriate parametric and
non-parametric statistical methods, and interprets significance, p-values,
and effect sizes with mathematical integrity.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.tools.statistical import (
    run_normality_test,
    run_ttest,
    run_anova,
    run_chi_square,
    run_correlation_test,
)

logger = logging.getLogger(__name__)

STATISTICIAN_SYSTEM_PROMPT = """You are a Principal Biostatistician and Quantitative Methodologist.
Your role is to formulate statistical hypotheses, execute appropriate inferential tests,
and interpret p-values, test statistics, and effect sizes (Cohen's d, Cramer's V, etc.)
with extreme mathematical rigor.

## Guidance:
1. State the Null Hypothesis (H0) and Alternative Hypothesis (H1) clearly.
2. Report the test statistic, degrees of freedom (if applicable), and exact p-value.
3. State whether H0 is rejected at the alpha = 0.05 threshold.
4. Report the effect size and categorize its practical significance (small, medium, large).
5. State any assumption caveats (e.g., normality, variance homogeneity, sample size).
6. Provide the concrete real-world / business takeaway based on the empirical test.
"""


class StatisticianAgent:
    """
    Statistician Agent for inferential statistics and hypothesis testing.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("statistician")
        else:
            self.llm = llm

    async def test_hypothesis(
        self,
        dataset_id: str,
        test_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a statistical test and generate an expert statistical report.

        Args:
            dataset_id: Dataset identifier.
            test_type: 'ttest', 'anova', 'chi_square', 'correlation', or 'normality'.
            params: Parameters required for the chosen test.
        """
        raw_result = self._execute_test(dataset_id, test_type, params)
        narrative = await self._interpret_result(raw_result, test_type, params)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "test_type": test_type,
            "result": raw_result,
            "narrative": narrative,
        }

    def _execute_test(self, dataset_id: str, test_type: str, params: dict[str, Any]) -> dict[str, Any]:
        t = test_type.lower()
        if t in ("ttest", "t_test"):
            return run_ttest(
                dataset_id,
                numeric_col=params["numeric_col"],
                group_col=params["group_col"],
                paired=params.get("paired", False),
            )
        elif t == "anova":
            return run_anova(
                dataset_id,
                numeric_col=params["numeric_col"],
                group_col=params["group_col"],
            )
        elif t in ("chi_square", "chisquare", "chi2"):
            return run_chi_square(
                dataset_id,
                col1=params["col1"],
                col2=params["col2"],
            )
        elif t in ("correlation", "corr"):
            return run_correlation_test(
                dataset_id,
                col1=params["col1"],
                col2=params["col2"],
            )
        elif t in ("normality", "shapiro"):
            return run_normality_test(
                dataset_id,
                column=params["column"],
            )
        else:
            raise ValueError(f"Unsupported statistical test: {test_type}")

    async def _interpret_result(
        self,
        result: dict[str, Any],
        test_type: str,
        params: dict[str, Any],
    ) -> str:
        prompt = (
            f"Statistical Test Executed: {test_type}\n"
            f"Parameters: {json.dumps(params)}\n"
            f"Statistical Output:\n```json\n{json.dumps(result, indent=2)}\n```\n\n"
            f"Provide an authoritative statistical evaluation and interpretation."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=STATISTICIAN_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Statistician LLM failed: {e}. Falling back to default.")
            return result.get("interpretation", "Statistical test completed.")
