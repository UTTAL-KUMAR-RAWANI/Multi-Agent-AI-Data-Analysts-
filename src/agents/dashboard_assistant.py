"""
DataWhale AI - Dashboard Assistant Agent

Provides an interactive AI assistant for the main dashboard.
Users can ask anything about:
- Loaded datasets (row/col counts, schema, memory, missing values, quality)
- Workspace metrics and summaries
- The 9 AI specialist agents and their capabilities
- Platform workflows (uploading, running Analyze Studio, querying with NL SQL, reports)
- Direct actions and navigation recommendations
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import get_settings, LLMProvider, ModelTier
from src.core.llm_router import get_llm, LLMRouter
from src.tools.data_loader import list_datasets, get_dataset_metadata

logger = logging.getLogger(__name__)

DASHBOARD_ASSISTANT_SYSTEM_PROMPT = """You are DataWhale AI's Dashboard Copilot and Workspace Assistant.
You have direct, real-time knowledge of all datasets currently loaded into the DataWhale AI analytics platform, the workspace metrics, and the 9 specialized AI agents.

## Your Personality & Style
- Helpful, sharp, data-literate, and proactive.
- Use clear markdown formatting (bold key numbers, bullet points, concise tables or lists).
- Keep answers informative yet easy to scan.
- When referencing datasets, use their exact file names and suggest next steps (e.g. running Analyze Studio or Chat).
- If the user asks how to do something, provide clear, step-by-step instructions.

## Live Workspace Context
{workspace_context}

## Capabilities of the 9 AI Specialist Agents
1. Supervisor: Orchestrates tasks, breaks down analytical objectives, and plans agent steps.
2. Data Cleaner: Detects and fixes data quality issues (missing values, deduplication, type casting, outliers).
3. EDA Explorer: Performs exploratory data analysis, computes distributions, skewness, correlations, and summary statistics.
4. Statistician: Runs rigorous hypothesis tests (t-tests, ANOVA, Chi-Square, Mann-Whitney U, normality tests).
5. ML Engineer: Trains predictive machine learning models (regression, classification, clustering, feature importances).
6. Visualizer: Generates high-impact charts (bar, line, scatter, boxplot, heatmap, histograms) via Plotly.
7. Report Writer: Synthesizes findings, charts, and metrics into self-contained HTML executive reports.
8. NL Query: Translates natural language questions directly into executable DuckDB SQL with plain English answers.
9. QA Agent: Validates generated analyses for mathematical plausibility, safety, and correctness.

## Navigation & Workflows
- **Upload Dataset**: Go to `/upload` (or click "New Dataset") to upload CSV, Excel (.xlsx/.xls), or JSON.
- **Analyze Studio**: Go to `/analysis/{{dataset_id}}` to run multi-agent automated or guided analysis.
- **Data Chat**: Go to `/chat/{{dataset_id}}` to ask SQL-powered questions about that specific dataset.
- **Reports Archive**: Go to `/reports` to browse, inspect, and download generated HTML reports.
- **Dashboard Home**: Go to `/` for system metrics and dataset overview.

Always be concise, accurate based on the live context provided, and helpful!
"""


class DashboardAssistantAgent:
    """Agent that handles queries about the dashboard and workspace."""

    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm

    def _get_llm(self) -> BaseChatModel:
        if self.llm is not None:
            return self.llm
        try:
            router = LLMRouter()
            return router.get_llm_for_agent("supervisor")
        except Exception as e:
            logger.warning(f"Failed to obtain router LLM, falling back to fast LLM: {e}")
            return get_llm(tier=ModelTier.FAST)

    def get_workspace_context(self) -> dict[str, Any]:
        """Collect live workspace state: datasets, metrics, reports, settings."""
        settings = get_settings()
        datasets = list_datasets()

        total_rows = sum(d.get("num_rows", 0) for d in datasets)
        total_cols = sum(d.get("num_columns", 0) for d in datasets)
        total_memory_mb = round(sum(d.get("memory_usage_mb", 0.0) for d in datasets), 2)
        datasets_with_missing = [d for d in datasets if d.get("has_missing")]

        # Check existing reports
        report_dir = settings.get_reports_path()
        reports = []
        if report_dir.exists():
            reports = [f.stem for f in sorted(report_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]]

        datasets_summary = []
        for d in datasets:
            missing_summary = d.get("missing_summary", {})
            cols_preview = [c["name"] for c in d.get("columns", [])[:8]]
            datasets_summary.append({
                "dataset_id": d.get("dataset_id"),
                "file_name": d.get("file_name"),
                "rows": d.get("num_rows"),
                "columns": d.get("num_columns"),
                "columns_sample": cols_preview,
                "memory_mb": d.get("memory_usage_mb"),
                "has_missing": d.get("has_missing", False),
                "missing_columns": missing_summary,
            })

        return {
            "dataset_count": len(datasets),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_memory_mb": total_memory_mb,
            "missing_dataset_count": len(datasets_with_missing),
            "datasets": datasets_summary,
            "recent_reports": reports,
            "active_providers": [p.value for p in settings.available_providers],
            "default_provider": settings.default_llm_provider.value,
        }

    def _format_context_text(self, ctx: dict[str, Any]) -> str:
        """Format the workspace context dictionary into a readable string for prompts."""
        lines = [
            f"- **Datasets Loaded**: {ctx['dataset_count']}",
            f"- **Total Rows Across Datasets**: {ctx['total_rows']:,}",
            f"- **Total Columns**: {ctx['total_columns']}",
            f"- **Total Memory Used**: {ctx['total_memory_mb']} MB",
            f"- **Datasets with Missing Values**: {ctx['missing_dataset_count']}",
            f"- **Configured LLM Provider**: {ctx['default_provider']}",
            f"- **Recent Reports ({len(ctx['recent_reports'])} available)**: {', '.join(ctx['recent_reports']) or 'None yet'}",
            "",
            "### Loaded Datasets Detail:",
        ]

        if not ctx["datasets"]:
            lines.append("No datasets are currently loaded. The user can upload one at `/upload`.")
        else:
            for ds in ctx["datasets"]:
                missing_info = (
                    f"Warning: Missing values in: {ds['missing_columns']}"
                    if ds["has_missing"]
                    else "Clean (no missing values)"
                )
                cols_str = ", ".join(ds["columns_sample"])
                if ds["columns"] > len(ds["columns_sample"]):
                    cols_str += f", ... ({ds['columns'] - len(ds['columns_sample'])} more)"

                lines.append(
                    f"* **{ds['file_name']}** (ID: `{ds['dataset_id']}`): "
                    f"{ds['rows']:,} rows × {ds['columns']} cols | {ds['memory_mb']} MB | {missing_info} | Columns: [{cols_str}]"
                )

        return "\n".join(lines)

    async def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Process a user question about the dashboard, datasets, or platform.
        Uses LLM if available; gracefully falls back to deterministic workspace analyzer.
        """
        ctx = self.get_workspace_context()
        context_text = self._format_context_text(ctx)
        history = history or []

        # First attempt: Try LLM
        reply_text = None
        try:
            llm = self._get_llm()
            # Prepare messages
            system_prompt = DASHBOARD_ASSISTANT_SYSTEM_PROMPT.format(
                workspace_context=context_text
            )
            messages = [SystemMessage(content=system_prompt)]

            # Add recent history (last 6 turns max)
            for h in history[-6:]:
                role = h.get("role", "user")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            messages.append(HumanMessage(content=question))

            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=3.5)
            if hasattr(response, "content") and response.content:
                reply_content = response.content.strip()
                # Ensure it's not a mock placeholder or error
                if reply_content and not reply_content.startswith("Analysis completed."):
                    reply_text = reply_content
        except Exception as e:
            logger.warning(f"LLM generation failed for dashboard chat ({e}), using smart fallback.")

        # Fallback if LLM failed or gave placeholder
        if not reply_text:
            reply_text = self._deterministic_fallback_answer(question, ctx)

        # Build dynamic suggested actions & quick replies
        suggested_actions = self._generate_suggested_actions(question, ctx, reply_text)
        quick_replies = self._generate_quick_replies(ctx)

        return {
            "reply": reply_text,
            "suggested_actions": suggested_actions,
            "quick_replies": quick_replies,
        }

    def _deterministic_fallback_answer(self, question: str, ctx: dict[str, Any]) -> str:
        """
        Smart deterministic answer generator that handles dashboard queries
        accurately even without an active LLM connection.
        """
        q = question.lower()
        datasets = ctx["datasets"]
        count = ctx["dataset_count"]

        # Check for specific dataset query
        for ds in datasets:
            stem = Path(ds["file_name"]).stem.lower()
            stem_keywords = [w for w in re.split(r'[_ -]+', stem) if len(w) >= 4 and not w.isdigit()]
            if (
                stem in q
                or ds["file_name"].lower() in q
                or ds["dataset_id"].lower() in q
                or any(k in q for k in stem_keywords)
            ):
                missing_str = (
                    f"**Has missing values:** {ds['missing_columns']}"
                    if ds["has_missing"]
                    else "Clean (0 missing values)"
                )
                cols_str = ", ".join(ds["columns_sample"])
                if ds["columns"] > len(ds["columns_sample"]):
                    cols_str += f", ... (+{ds['columns'] - len(ds['columns_sample'])} more)"

                return (
                    f"### Dataset: **{ds['file_name']}**\n"
                    f"- **Rows:** **{ds['rows']:,}** | **Columns:** **{ds['columns']}**\n"
                    f"- **Memory Usage:** {ds['memory_mb']} MB\n"
                    f"- **Data Quality:** {missing_str}\n"
                    f"- **Columns:** {cols_str}\n\n"
                    f"Click **[Analyze Studio](/analysis/{ds['dataset_id']})** to run multi-agent modeling or **[Chat](/chat/{ds['dataset_id']})** to query this dataset."
                )

        # 1. Missing values / quality check
        if any(w in q for w in ["missing", "null", "empty", "quality", "clean"]):
            missing_ds = [d for d in datasets if d["has_missing"]]
            if not missing_ds:
                return (
                    "**Great news!** None of your loaded datasets have missing values. All data is complete and clean."
                )
            lines = [f"Found **{len(missing_ds)} dataset{'s' if len(missing_ds) != 1 else ''}** with missing values:"]
            for ds in missing_ds:
                details = ", ".join([f"`{col}`: {cnt} nulls" for col, cnt in ds["missing_columns"].items()])
                lines.append(f"- **{ds['file_name']}**: {details}")
            lines.append("\n*Tip: The **Data Cleaner** agent can automatically impute or drop missing values in Analyze Studio!*")
            return "\n".join(lines)

        # 2. Rows / Count / Stats / Size
        if any(w in q for w in ["how many row", "total row", "largest", "biggest", "most row", "smallest", "fewest", "size", "memory"]):
            if count == 0:
                return "There are no datasets loaded yet. Upload your data at `/upload`."
            largest = max(datasets, key=lambda x: x["rows"])
            smallest = min(datasets, key=lambda x: x["rows"])
            return (
                f"### Workspace Size & Rows Breakdown:\n"
                f"- **Total Rows:** **{ctx['total_rows']:,}**\n"
                f"- **Total Datasets:** **{count}**\n"
                f"- **Total Memory Footprint:** **{ctx['total_memory_mb']} MB**\n"
                f"- **Largest Dataset:** **{largest['file_name']}** with **{largest['rows']:,}** rows and **{largest['columns']}** columns.\n"
                f"- **Smallest Dataset:** **{smallest['file_name']}** with **{smallest['rows']:,}** rows."
            )

        # 3. Datasets list / overview
        if any(w in q for w in ["what dataset", "which dataset", "list dataset", "show dataset", "my dataset", "datasets loaded", "summary"]):
            if count == 0:
                return (
                    "You currently have **0 datasets** loaded in your workspace.\n\n"
                    "Click **[Upload Dataset](/upload)** to upload a CSV, Excel, or JSON file and start analyzing."
                )
            items = []
            for ds in datasets:
                items.append(
                    f"- **{ds['file_name']}**: **{ds['rows']:,}** rows × **{ds['columns']}** columns ({ds['memory_mb']} MB)\n"
                    f"  *ID:* `{ds['dataset_id']}`\n"
                    f"  *Columns:* {', '.join(ds['columns_sample'][:5])}..."
                )
            return (
                f"You currently have **{count} dataset{'s' if count != 1 else ''}** loaded ({ctx['total_rows']:,} total rows):\n\n"
                + "\n".join(items) +
                "\n\nYou can run **Analyze Studio** for automated insights or jump into **Chat** to query them via SQL!"
            )


        # AI Agents questions
        if any(w in q for w in ["agent", "specialist", "supervisor", "ml engineer", "cleaner", "statistician", "visualizer", "qa"]):
            if "ml" in q:
                return (
                    "### ML Engineer Agent\n"
                    "The **ML Engineer** handles automated machine learning workflows:\n"
                    "- Automatically selects target variables and task types (Regression, Classification, Clustering)\n"
                    "- Trains baseline models (Random Forest, Gradient Boosting, Ridge, Logistic Regression)\n"
                    "- Evaluates metrics (R2, RMSE, Accuracy, F1-Score, ROC-AUC)\n"
                    "- Computes feature importances so you know which columns matter most."
                )
            if "clean" in q:
                return (
                    "### Data Cleaner Agent\n"
                    "The **Data Cleaner** prepares messy data for analysis:\n"
                    "- Detects and imputes missing values (mean, median, mode, or indicator)\n"
                    "- Eliminates duplicate rows\n"
                    "- Fixes incorrect column types (e.g. string dates to datetimes, numeric strings to floats)\n"
                    "- Flags and handles extreme outliers."
                )
            if "stat" in q:
                return (
                    "### Statistician Agent\n"
                    "The **Statistician** validates patterns scientifically:\n"
                    "- Conducts hypothesis tests (two-sample t-tests, ANOVA, Chi-Square)\n"
                    "- Checks distribution normality (Shapiro-Wilk)\n"
                    "- Computes p-values and effect sizes to distinguish signal from noise."
                )
            return (
                "### DataMind AI — 9 Specialist Agents\n"
                "1. **Supervisor**: Plans the analysis pipeline and routes subtasks.\n"
                "2. **Data Cleaner**: Cleans missing values, duplicates, and type mismatches.\n"
                "3. **EDA Explorer**: Computes correlations, distributions, and skewness.\n"
                "4. **Statistician**: Performs hypothesis tests, ANOVA, and p-value checks.\n"
                "5. **ML Engineer**: Builds predictive regression & classification models.\n"
                "6. **Visualizer**: Generates Plotly charts (heatmaps, distributions, bars).\n"
                "7. **Report Writer**: Combines all findings into a clean HTML report.\n"
                "8. **NL Query**: Translates English questions into DuckDB SQL.\n"
                "9. **QA Agent**: Verifies output accuracy and statistical plausibility."
            )

        # How-to / Upload / Studio / Workflow
        if any(w in q for w in ["how to", "how do i", "upload", "analyze", "report", "start", "export"]):
            return (
                "### How to use DataMind AI:\n\n"
                "1. **Upload Data:** Click **[New Dataset](/upload)** in the top navigation or sidebar. Upload CSV, Excel (.xlsx), or JSON.\n"
                "2. **Analyze Studio:** On any dataset card, click **[Analyze Studio](/analysis/{id})** to unleash the multi-agent pipeline for automated cleaning, statistical analysis, ML models, and visual charts.\n"
                "3. **Ask in English (Chat):** Click **[Chat](/chat/{id})** on any dataset card to run natural language SQL queries against your data.\n"
                "4. **Reports:** View or download full executive reports in the **[Reports](/reports)** section."
            )

        # Reports
        if "report" in q:
            recent = ctx["recent_reports"]
            if recent:
                rep_list = "\n".join([f"- [{r}](/reports/view/{r})" for r in recent])
                return (
                    f"You have **{len(recent)} recent report(s)** generated:\n{rep_list}\n\n"
                    f"Visit the **[Reports](/reports)** page to browse all generated reports."
                )
            return (
                "No HTML reports have been generated yet. Run an analysis in **Analyze Studio** to generate your first comprehensive report!"
            )

        # General overview fallback
        return (
            f"Hello! I am your **Dashboard Copilot**.\n\n"
            f"Here is your current workspace overview:\n"
            f"- **Datasets:** {count} loaded ({ctx['total_rows']:,} rows total)\n"
            f"- **AI Specialists:** 9 agents ready (Supervisor, ML Engineer, Statistician, Cleaner, etc.)\n"
            f"- **System Status:** Ready with `{ctx['default_provider']}` LLM provider.\n\n"
            f"You can ask me anything about your datasets, data quality issues, column statistics, AI agent features, or how to run analysis!"
        )


    def _generate_suggested_actions(
        self, question: str, ctx: dict[str, Any], reply: str
    ) -> list[dict[str, str]]:
        """Generate smart quick-navigation action buttons based on the user's inquiry."""
        actions: list[dict[str, str]] = []
        q = question.lower()
        datasets = ctx["datasets"]

        # Check if user mentioned a specific dataset
        for ds in datasets:
            file_stem = Path(ds["file_name"]).stem.lower()
            stem_keywords = [w for w in re.split(r'[_ -]+', file_stem) if len(w) >= 4 and not w.isdigit()]
            if (
                file_stem in q
                or ds["file_name"].lower() in q
                or ds["dataset_id"].lower() in q
                or any(k in q for k in stem_keywords)
            ):
                actions.append({
                    "label": f"Analyze Studio ({ds['file_name'][:20]})",
                    "url": f"/analysis/{ds['dataset_id']}",
                    "icon": "zap",
                })
                actions.append({
                    "label": f"SQL Chat ({ds['file_name'][:20]})",
                    "url": f"/chat/{ds['dataset_id']}",
                    "icon": "message-circle",
                })
                break

        # If no specific dataset action was added, provide relevant platform links
        if not actions:
            if "upload" in q or ctx["dataset_count"] == 0:
                actions.append({"label": "Upload Dataset", "url": "/upload", "icon": "upload-cloud"})
            if "report" in q:
                actions.append({"label": "View Reports", "url": "/reports", "icon": "file-text"})
            if datasets:
                first_ds = datasets[0]
                actions.append({
                    "label": f"Studio: {first_ds['file_name'][:20]}...",
                    "url": f"/analysis/{first_ds['dataset_id']}",
                    "icon": "zap",
                })
                actions.append({
                    "label": "Upload New Dataset",
                    "url": "/upload",
                    "icon": "plus",
                })

        return actions[:3]

    def _generate_quick_replies(self, ctx: dict[str, Any]) -> list[str]:
        """Generate helpful quick prompt suggestions."""
        if ctx["dataset_count"] == 0:
            return [
                "How do I upload a dataset?",
                "What can the 9 AI agents do?",
                "What file formats are supported?",
            ]

        replies = [
            "Summarize my loaded datasets",
            "Which datasets have missing values?",
            "What can the ML Engineer agent do?",
            "How do I run an analysis in Studio?",
        ]
        return replies
