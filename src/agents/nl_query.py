"""
DataMind AI - Natural Language Query Agent

Translates natural language questions about data into SQL queries,
executes them via DuckDB, and returns human-readable results.

This is the first and most interactive agent — users ask questions
in plain English, and this agent figures out the right SQL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.llm_router import get_llm, LLMRouter
from src.core.config import ModelTier
from src.tools.sql_engine import execute_sql, get_table_schema, get_sample_values
from src.tools.data_loader import describe_dataset

logger = logging.getLogger(__name__)

# System prompt for the NL Query Agent
NL_QUERY_SYSTEM_PROMPT = """You are an expert SQL analyst. Your job is to translate natural language questions about data into DuckDB SQL queries.

## Rules
1. ALWAYS respond with valid JSON containing exactly two keys: "sql" and "explanation".
2. The "sql" value must be a valid DuckDB SQL query.
3. The "explanation" value must be a brief, human-readable explanation of what the query does.
4. Use the table name provided in the schema — it's available as both the dataset ID and the alias "data".
5. Always quote column names with double quotes if they contain spaces or special characters.
6. Use DuckDB SQL syntax (similar to PostgreSQL).
7. For aggregations, always include meaningful column aliases using AS.
8. Limit results to 100 rows unless the user asks for all results.
9. If the question is ambiguous, make reasonable assumptions and explain them.
10. NEVER modify data — only SELECT queries are allowed.

## Response Format
```json
{
    "sql": "SELECT ...",
    "explanation": "This query ..."
}
```

IMPORTANT: Respond with ONLY the JSON object, no markdown code blocks, no extra text."""


INTERPRET_RESULT_PROMPT = """You are a data analyst presenting query results to a non-technical user.

Given the user's original question, the SQL query that was run, and the query results,
provide a clear, concise, human-readable answer.

## Rules
1. Lead with the direct answer to the question.
2. If there are numbers, format them nicely (commas, rounding, percentages).
3. If the result is a table, describe the key findings.
4. Keep it concise — 2-4 sentences for simple questions, more for complex ones.
5. If the data reveals something unexpected or interesting, mention it.
6. Do NOT show raw JSON or SQL — speak in plain English."""


class NLQueryAgent:
    """
    Natural Language to SQL Query Agent.

    Takes a user's question about a dataset, generates SQL,
    executes it, and returns a human-readable answer.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        """
        Initialize the NL Query Agent.

        Args:
            llm: Optional pre-configured LLM. If None, uses the default from LLMRouter.
        """
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("nl_query")
        else:
            self.llm = llm

    async def generate_sql(self, question: str, dataset_id: str) -> dict[str, str]:
        """
        Generate a SQL query from a natural language question.

        Args:
            question: The user's question in plain English.
            dataset_id: The dataset to query against.

        Returns:
            Dict with "sql" and "explanation" keys.
        """
        # Build context about the dataset
        schema = get_table_schema(dataset_id)
        sample_values = get_sample_values(dataset_id)
        description = describe_dataset(dataset_id)

        context = f"""## Dataset Schema
{schema}

## {sample_values}

## Dataset Overview
{description}
"""

        messages = [
            SystemMessage(content=NL_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=f"""## Dataset Context
{context}

## User Question
{question}

Generate the SQL query to answer this question. Respond with ONLY JSON."""),
        ]

        response = await self.llm.ainvoke(messages)
        response_text = response.content.strip()

        # Parse the JSON response, handling markdown code blocks
        if response_text.startswith("```"):
            # Strip markdown code blocks
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        try:
            result = json.loads(response_text)
            if "sql" not in result:
                raise ValueError("Response missing 'sql' key")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.warning(f"Raw response: {response_text}")
            # Fallback: try to extract SQL from the response
            return {
                "sql": response_text,
                "explanation": "Auto-generated query (response parsing failed).",
            }

    async def interpret_results(
        self,
        question: str,
        sql_query: str,
        results: dict[str, Any],
    ) -> str:
        """
        Generate a human-readable interpretation of query results.

        Args:
            question: The original user question.
            sql_query: The SQL that was executed.
            results: The query results dict from sql_engine.

        Returns:
            Human-readable answer string.
        """
        # Format results for the LLM
        if results["row_count"] == 0:
            return "No results found for your query. The data doesn't contain any matching records."

        # Show first N rows in a readable format
        rows_preview = results["rows"][:20]
        results_text = json.dumps(rows_preview, indent=2, default=str)

        messages = [
            SystemMessage(content=INTERPRET_RESULT_PROMPT),
            HumanMessage(content=f"""## User's Question
{question}

## SQL Query Executed
{sql_query}

## Results ({results['row_count']} rows{' — truncated' if results['truncated'] else ''})
{results_text}

Provide a clear, human-readable answer to the user's question based on these results."""),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def query(self, question: str, dataset_id: str) -> dict[str, Any]:
        """
        Full query pipeline: question → SQL → execute → interpret.

        This is the main entry point for the NL Query Agent.

        Args:
            question: Natural language question.
            dataset_id: Dataset to query.

        Returns:
            Dict with sql, explanation, results, interpretation, and raw_data.
        """
        logger.info(f"NL Query Agent processing: '{question}' on dataset '{dataset_id}'")

        # Step 1: Generate SQL
        sql_response = await self.generate_sql(question, dataset_id)
        sql_query = sql_response["sql"]
        explanation = sql_response.get("explanation", "")

        logger.info(f"Generated SQL: {sql_query}")

        # Step 2: Execute SQL
        try:
            results = execute_sql(dataset_id, sql_query)
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            # Try to self-correct: ask LLM to fix the query
            corrected = await self._self_correct(question, dataset_id, sql_query, str(e))
            if corrected:
                sql_query = corrected["sql"]
                explanation = corrected.get("explanation", explanation)
                results = execute_sql(dataset_id, sql_query)
            else:
                return {
                    "sql": sql_query,
                    "explanation": explanation,
                    "results": None,
                    "interpretation": f"Sorry, I couldn't execute that query. Error: {e}",
                    "error": str(e),
                }

        # Step 3: Interpret results
        interpretation = await self.interpret_results(question, sql_query, results)

        return {
            "sql": sql_query,
            "explanation": explanation,
            "results": results,
            "interpretation": interpretation,
            "error": None,
        }

    async def _self_correct(
        self,
        question: str,
        dataset_id: str,
        failed_sql: str,
        error: str,
    ) -> dict[str, str] | None:
        """
        Attempt to fix a failed SQL query by sending the error back to the LLM.

        Returns corrected SQL dict or None if it can't fix it.
        """
        schema = get_table_schema(dataset_id)

        messages = [
            SystemMessage(content=NL_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=f"""The following SQL query failed. Please fix it.

## Schema
{schema}

## Original Question
{question}

## Failed SQL
{failed_sql}

## Error
{error}

Generate a corrected SQL query. Respond with ONLY JSON."""),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()

            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(
                    line for line in lines if not line.startswith("```")
                )

            result = json.loads(response_text)
            logger.info(f"Self-corrected SQL: {result.get('sql', 'N/A')}")
            return result
        except Exception as e:
            logger.warning(f"Self-correction failed: {e}")
            return None
