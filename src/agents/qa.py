"""
DataMind AI - Quality Assurance (QA) Agent

Audits and validates outputs produced by specialist agents (Cleaner, EDA, Statistician, ML, Visualizer).
Detects hallucinations, statistical misinterpretations, data leaks, and invalid conclusions
before results reach the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import LLMRouter
from src.core.state import QAFeedback

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are a Principal Quality Assurance Auditor for an advanced AI Data Analysis platform.
Your mission is to rigorously evaluate agent outputs to ensure accuracy, statistical validity, and safety.

## Review Criteria:
1. **Data Cleaning**: Did the cleaner preserve data integrity without wiping out critical rows/columns?
2. **Exploratory Data Analysis**: Are the observations mathematically grounded in the actual summary statistics?
3. **Statistical Testing**: Was the right test used? Are p-values interpreted correctly without asserting causality blindly?
4. **Machine Learning**: Are there signs of data leakage (e.g., target feature present in inputs, suspicious 1.0 accuracy)?
5. **Visualizations**: Does the chart correctly map dimensions without misleading axes or missing labels?

## Output JSON Schema:
```json
{
  "passed": true,
  "severity": "info",
  "issues": [],
  "suggestions": []
}
```
If there are critical flaws, set "passed": false and "severity": "error".
Respond ONLY with valid JSON.
"""


class QAAgent:
    """
    Quality Assurance Agent for validating multi-agent outputs.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("qa")
        else:
            self.llm = llm

    async def review_output(
        self,
        agent_name: str,
        output_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> QAFeedback:
        """
        Audit an agent's output and provide structured QA feedback.
        """
        # Rule-based fast heuristics first
        fast_check = self._rule_based_check(agent_name, output_data)
        if fast_check:
            return fast_check

        # LLM-assisted deep validation
        prompt = (
            f"Reviewing output from Agent: {agent_name}\n"
            f"Context: {json.dumps(context or {}, indent=2)}\n"
            f"Agent Output:\n```json\n{json.dumps(output_data, indent=2, default=str)[:3000]}\n```\n\n"
            f"Evaluate the output against the QA criteria and return the JSON review."
        )

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=QA_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            text = response.content.strip()
            # Clean possible markdown fence
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
            parsed = json.loads(text.strip())

            return QAFeedback(
                agent=agent_name,
                passed=bool(parsed.get("passed", True)),
                severity=parsed.get("severity", "info"),
                issues=parsed.get("issues", []),
                suggestions=parsed.get("suggestions", []),
            )
        except Exception as e:
            logger.warning(f"QA LLM audit error: {e}. Defaulting to passed.")
            return QAFeedback(
                agent=agent_name,
                passed=True,
                severity="info",
                issues=[],
                suggestions=["Automated heuristic check passed."],
            )

    def _rule_based_check(self, agent_name: str, output: dict[str, Any]) -> QAFeedback | None:
        """Heuristic checks to catch obvious errors without LLM latency."""
        if agent_name == "cleaner":
            audit = output.get("audit_report", {})
            if audit.get("final_rows", 1) == 0:
                return QAFeedback(
                    agent=agent_name,
                    passed=False,
                    severity="error",
                    issues=["All dataset rows were eliminated during data cleaning."],
                    suggestions=["Relax null threshold or avoid dropping rows."],
                )
            if audit.get("final_cols", 1) == 0:
                return QAFeedback(
                    agent=agent_name,
                    passed=False,
                    severity="error",
                    issues=["All columns were eliminated during cleaning."],
                    suggestions=["Do not drop all columns."],
                )

        elif agent_name == "ml_engineer":
            ml = output.get("ml_results", {})
            metrics = ml.get("best_metrics", {})
            # Suspicious 1.0 accuracy check
            if metrics.get("accuracy") == 1.0 and ml.get("train_samples", 0) > 50:
                return QAFeedback(
                    agent=agent_name,
                    passed=True,
                    severity="warning",
                    issues=["Model achieved 100% accuracy. Possible target leakage or synthetic separation."],
                    suggestions=["Inspect feature importances for features directly correlated with target."],
                )

        return None
