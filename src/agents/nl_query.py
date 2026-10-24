"""
DataWhale AI - Natural Language Query Agent

Translates natural language questions about data into SQL queries,
executes them via DuckDB, and returns human-readable results.

Enforces intelligent data-source routing between raw upload,
explicitly cleaned data, and validated analysis data.
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
from src.tools.data_loader import describe_dataset, get_dataset_metadata, get_dataset_stage, get_cleaned_dataset, get_analysis_dataset, get_cleaning_audit

logger = logging.getLogger(__name__)

# System prompt for the NL Query Agent
NL_QUERY_SYSTEM_PROMPT = """You are an expert SQL analyst for DataWhale AI. Your job is to translate natural language questions about data into DuckDB SQL queries.

## Rules
1. ALWAYS respond with valid JSON containing exactly two keys: "sql" and "explanation".
2. The "sql" value must be a valid DuckDB SQL query.
3. The "explanation" value must be a brief, human-readable explanation of what the query does.
4. Use the table name provided in the schema — it's available as both the dataset ID and the alias "data".
   You can also reference "raw" for original uncleaned values, or "cleaned" for cleaned values if relevant.
5. Always quote column names with double quotes if they contain spaces or special characters.
6. Use DuckDB SQL syntax (similar to PostgreSQL).
7. For aggregations, always include meaningful column aliases using AS.
8. Limit results to 100 rows unless the user explicitly asks for all results.
9. If the user asks a follow-up question (e.g. "show only the top 5 of those", "compare with the previous result"), use the provided conversation history to understand the target entity, metric, and filters.
10. NEVER modify data — only SELECT queries are allowed.

## Response Format
```json
{
    "sql": "SELECT ...",
    "explanation": "This query ..."
}
```

IMPORTANT: Respond with ONLY the JSON object, no markdown code blocks, no extra text."""


INTERPRET_RESULT_PROMPT = """You are an expert AI data analyst assistant, like ChatGPT, communicating with a decision maker.

Given the user's question, conversation context, the executed SQL query, and the verified query results:
Provide a clear, conversational, articulate, and intelligent answer—just like ChatGPT.

## Guidelines:
1. Speak naturally, warmly, and authoritatively like ChatGPT.
2. Directly answer the question first, highlighting exact figures, totals, averages, or top entities with bold formatting (e.g. **$4,520.50**, **76.1%**, **120 districts**).
3. If the result contains multiple rows or categories, summarize the key patterns, top performers, or notable trends.
4. Keep it easy to read with clean Markdown formatting (bold key numbers, bullet points where helpful).
5. Never output raw JSON or code in the interpretation text—speak in fluent, human English.
6. If answering from a raw uncleaned dataset, transparently acknowledge that cleaning has not yet been executed."""


class NLQueryAgent:
    """
    Natural Language to SQL Query Agent with intelligent data-source routing.

    Takes a user's question, inspects dataset state and intent, selects the
    appropriate data source (raw vs cleaned vs analysis), executes via DuckDB,
    and returns verified human-readable answers.
    """

    def __init__(self, llm: BaseChatModel | None = None):
        if llm is None:
            router = LLMRouter()
            self.llm = router.get_llm_for_agent("nl_query")
        else:
            self.llm = llm

    def determine_target_source(
        self,
        question: str,
        dataset_id: str,
        preference: str | None = None,
    ) -> tuple[str, str, str]:
        """
        Determine whether to route to 'raw', 'cleaned', or 'analysis' data source.

        Returns:
            tuple: (target_stage, data_source_label, current_lifecycle_status)
        """
        current_status = get_dataset_stage(dataset_id)
        has_cleaned = get_cleaned_dataset(dataset_id) is not None
        has_analysis = get_analysis_dataset(dataset_id) is not None

        if preference in ("raw", "cleaned", "analysis"):
            target = preference
        else:
            q_low = question.lower()
            raw_keywords = [
                "original", "raw", "uncleaned", "source file", "uploaded file",
                "before clean", "missing values in original", "initial rows", "original count",
                "nulls in raw", "unprocessed", "uploaded"
            ]
            audit_keywords = [
                "what was cleaned", "cleaning audit", "how many duplicates removed",
                "what was imputed", "cleaning actions", "transformations applied", "cleaning report"
            ]

            if any(k in q_low for k in raw_keywords):
                target = "raw"
            elif any(k in q_low for k in audit_keywords) and has_cleaned:
                target = "cleaned"
            elif has_analysis:
                target = "analysis"
            elif has_cleaned:
                target = "cleaned"
            else:
                target = "raw"

        if target == "raw":
            if current_status == "raw":
                label = "Raw Upload (Uncleaned)"
            else:
                label = "Raw Source Dataset (Original Values)"
        elif target == "cleaned":
            label = "Cleaned Dataset (Hygiene Applied)"
        else:
            label = "Validated Analysis Dataset"

        return target, label, current_status

    async def generate_sql(
        self,
        question: str,
        dataset_id: str,
        target_stage: str = "active",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """
        Generate a SQL query from a natural language question with conversation history.
        """
        schema = get_table_schema(dataset_id, target_stage=target_stage)
        sample_values = get_sample_values(dataset_id, target_stage=target_stage)

        history_context = ""
        if history and len(history) > 0:
            history_context = "## Prior Conversation Context:\n"
            for turn in history[-4:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = str(turn.get("content", ""))[:200]
                history_context += f"- **{role}**: {content}\n"
            history_context += "\n"

        context = f"""## Dataset Schema ({target_stage.upper()} STAGE)
{schema}

## Sample Values:
{sample_values}
"""

        messages = [
            SystemMessage(content=NL_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=f"""## Dataset Context
{context}

{history_context}## Current Question
{question}

Generate the SQL query to answer this question. Respond with ONLY JSON."""),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()

            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )

            result = json.loads(response_text)
            if "sql" not in result:
                raise ValueError("Response missing 'sql' key")
            return result
        except Exception as e:
            logger.warning(f"LLM SQL generation fallback triggered: {e}")
            return self._heuristic_sql(question, dataset_id, target_stage)

    def _heuristic_sql(
        self,
        question: str,
        dataset_id: str,
        target_stage: str = "active",
    ) -> dict[str, str]:
        """Deterministic heuristic fallback if LLM times out or is unreachable."""
        import re
        q_low = question.lower()
        try:
            meta = get_dataset_metadata(dataset_id)
            cols = meta.get("columns", [])
        except Exception:
            cols = []
        num_cols = [c["name"] for c in cols if any(t in c["dtype"].lower() for t in ("int", "float", "double", "num"))]

        # Check for count / total
        if any(w in q_low for w in ("how many", "row count", "total row", "total record")):
            return {
                "sql": "SELECT COUNT(*) AS total_rows FROM data;",
                "explanation": "Calculates the total number of records."
            }

        # Check for missing values
        if any(w in q_low for w in ("missing", "null", "blank", "quality", "clean")):
            if cols:
                sub = ", ".join([f'COUNT(*) FILTER (WHERE "{c["name"]}" IS NULL) AS "{c["name"]}_missing"' for c in cols[:6]])
                return {
                    "sql": f"SELECT {sub} FROM data;",
                    "explanation": "Audits missing and null values across primary columns."
                }
            return {
                "sql": "SELECT COUNT(*) AS total_rows FROM data;",
                "explanation": "Checks dataset record count."
            }

        # Check for average / mean
        if any(w in q_low for w in ("average", "mean", "avg")) and num_cols:
            avgs = ", ".join([f'ROUND(AVG("{c}"), 2) AS "avg_{c}"' for c in num_cols[:4]])
            return {
                "sql": f"SELECT {avgs} FROM data;",
                "explanation": f"Computes arithmetic averages for numeric metrics."
            }

        # Check for top N
        limit_match = re.search(r'\b(?:top|first)\s+(\d+)\b', q_low)
        limit_num = int(limit_match.group(1)) if limit_match else (10 if "top" in q_low else 15)

        order_clause = ""
        # Match any column mentioned in question
        for c in num_cols:
            if c.lower() in q_low:
                order_clause = f' ORDER BY "{c}" DESC'
                break
        if not order_clause and num_cols and any(w in q_low for w in ("highest", "top", "max", "desc", "most", "descending")):
            order_clause = f' ORDER BY "{num_cols[0]}" DESC'
        elif not order_clause and num_cols and any(w in q_low for w in ("lowest", "min", "least", "asc", "ascending")):
            order_clause = f' ORDER BY "{num_cols[0]}" ASC'

        return {
            "sql": f"SELECT * FROM data{order_clause} LIMIT {limit_num};",
            "explanation": f"Retrieves {limit_num} records with verified schema execution."
        }

    async def interpret_results(
        self,
        question: str,
        sql_query: str,
        results: dict[str, Any],
        target_stage: str = "active",
        is_raw_uncleaned: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate human-readable interpretation of query results.
        """
        if results["row_count"] == 0:
            ans = "No matching records were found for your query."
            if is_raw_uncleaned:
                ans += " Note: This query was evaluated against the raw uncleaned dataset."
            return ans

        rows_preview = results["rows"][:5]
        results_text = json.dumps(rows_preview, indent=2, default=str)

        history_context = ""
        if history and len(history) > 0:
            history_context = "## Prior Conversation Context:\n"
            for turn in history[-4:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = str(turn.get("content", ""))[:200]
                history_context += f"- **{role}**: {content}\n"
            history_context += "\n"

        prompt_content = f"""## User's Question
{question}

{history_context}## Data Source Stage
{target_stage.upper()} ({'Uncleaned raw data' if is_raw_uncleaned else 'Cleaned / validated data'})

## SQL Query Executed
{sql_query}

## Results ({results['row_count']} rows{' — truncated' if results['truncated'] else ''})
{results_text}

Provide a clear, human-readable answer to the user's question based on these results."""

        messages = [
            SystemMessage(content=INTERPRET_RESULT_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            answer = response.content.strip()
        except Exception as e:
            logger.warning(f"LLM interpretation fallback triggered: {e}")
            cols = results.get("columns", [])
            rows = results.get("rows", [])
            cnt = results.get("row_count", 0)
            if cnt == 1 and len(cols) == 1:
                val = rows[0][cols[0]]
                answer = f"The computed {cols[0].replace('_', ' ')} is {val}."
            elif cnt == 1:
                pairs = [f"{c.replace('_', ' ')}: {rows[0][c]}" for c in cols]
                answer = f"Computed verified results: {', '.join(pairs)}."
            else:
                answer = f"Retrieved {cnt} record{'s' if cnt != 1 else ''} matching your inquiry."

        if is_raw_uncleaned and "raw" not in answer.lower():
            answer += "\n\n*(Data Source: Raw Upload. Data hygiene cleaning has not been executed on this dataset yet.)*"

        return answer

    def generate_suggested_followups(
        self,
        dataset_id: str,
        question: str,
        results: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Generate relevant follow-up questions tailored to dataset columns and query outcome.
        """
        try:
            meta = get_dataset_metadata(dataset_id)
            cols = meta.get("columns", [])
            num_cols = [c["name"] for c in cols if "int" in c["dtype"] or "float" in c["dtype"]]
            cat_cols = [c["name"] for c in cols if c["name"] not in num_cols]
        except Exception:
            num_cols, cat_cols = [], []

        followups = []
        if num_cols:
            c = num_cols[0]
            followups.append(f"Show the top 5 records by {c}")
            followups.append(f"What is the average and distribution of {c}?")
        if cat_cols:
            c = cat_cols[0]
            followups.append(f"Break down totals grouped by {c}")
        followups.append("Are there any missing values or data quality issues?")

        return followups[:4]

    def is_conversational_question(self, question: str) -> bool:
        """
        Check if a question is conversational, conceptual, advisory, or greetings,
        which should be answered conversationally like ChatGPT without generating SQL tables.
        """
        import re
        q = question.strip().lower()

        # Explicit data calculation keywords that definitely require SQL
        sql_indicators = [
            "how many", "count", "average", "avg", "mean", "median", "sum", "total",
            "highest", "lowest", "maximum", "minimum", "max", "min", "top ", "first ",
            "bottom ", "filter", "where", "group by", "order by", "sort by",
            "null", "missing", "blank"
        ]
        has_sql_calc = any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in sql_indicators)

        # Greetings
        greetings = [
            "hi", "hello", "hey", "good morning", "good evening", "howdy",
            "who are you", "what are you", "what can you do", "help", "help me"
        ]
        if any(q == g or q.startswith(g + " ") or q.startswith(g + "!") or q.startswith(g + ",") for g in greetings):
            if not has_sql_calc:
                return True

        # Conversational / Conceptual / Advisory phrases
        chat_phrases = [
            "what do you think", "what should i", "how should i", "recommend", "suggestion",
            "suggest", "explain what", "what is machine learning", "what is data",
            "what does this mean", "how do i interpret", "how to interpret", "write python",
            "write code", "how does", "why is", "can you explain", "tell me about",
            "what is this dataset", "what does this dataset", "key columns", "what columns",
            "which columns", "what are the columns", "about this data", "describe the data",
            "summarize the data", "understand this data", "overview of this", "what is this data"
        ]
        if any(p in q for p in chat_phrases) and not has_sql_calc:
            return True

        # Dataset overview inquiries without specific column calculations
        if ("dataset about" in q or "describe this dataset" in q or "summarize this dataset" in q) and not has_sql_calc:
            return True

        return False

    async def answer_conversational(
        self,
        question: str,
        dataset_id: str,
        target_stage: str = "active",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Provide a conversational, ChatGPT-like answer using dataset schema and analytics context.
        """
        schema = get_table_schema(dataset_id, target_stage=target_stage)
        try:
            meta = get_dataset_metadata(dataset_id)
            meta_summary = (
                f"- **Dataset File**: {meta.get('file_name', dataset_id)}\n"
                f"- **Total Rows**: {meta.get('num_rows', 0):,}\n"
                f"- **Total Attributes / Columns**: {meta.get('num_columns', 0)}\n"
                f"- **Data Quality Status**: {'Contains missing values' if meta.get('has_missing') else 'Clean (no missing values)'}"
            )
        except Exception:
            meta_summary = f"- **Dataset**: {dataset_id}"

        history_context = ""
        if history and len(history) > 0:
            history_context = "## Prior Conversation History:\n"
            for turn in history[-4:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = str(turn.get("content", ""))[:200]
                history_context += f"- **{role}**: {content}\n"
            history_context += "\n"

        system_prompt = f"""You are DataWhale AI's expert Data Science Copilot, an AI assistant like ChatGPT specialized in data analysis, machine learning, and business intelligence.

You have full context of the active dataset: `{dataset_id}`.

## Dataset Profile:
{meta_summary}

## Dataset Schema & Column Types:
{schema}

## Your Persona & Guidelines:
- Speak conversationally, warmly, smartly, and articulately—just like ChatGPT.
- Directly and helpfully answer the user's question.
- If asked about the dataset, explain columns, recommend models, suggest analysis angles, or provide guidance based on the schema above.
- If asked a general question or greeting, converse naturally and offer specific analytical questions they can ask.
- Format responses cleanly with Markdown (bold text, bullet points, headers, backticks).
"""
        messages = [
            SystemMessage(content=system_prompt),
        ]
        if history_context:
            messages.append(SystemMessage(content=history_context))
        messages.append(HumanMessage(content=question))

        try:
            res = await self.llm.ainvoke(messages)
            return res.content.strip()
        except Exception as e:
            logger.warning(f"Conversational answer fallback triggered: {e}")
            return f"Hello! I am your AI Data Analyst for `{dataset_id}`. You can ask me to calculate statistics, filter data, analyze correlations, or suggest predictive models!"

    async def query(
        self,
        question: str,
        dataset_id: str,
        history: list[dict[str, str]] | None = None,
        preference: str | None = None,
    ) -> dict[str, Any]:
        """
        Full query pipeline: intent routing → SQL → execute → interpret → follow-ups.
        """
        logger.info(f"NL Query Agent processing: '{question}' on dataset '{dataset_id}'")

        # 1. Determine target data source
        target_stage, source_label, current_status = self.determine_target_source(
            question=question,
            dataset_id=dataset_id,
            preference=preference,
        )
        is_raw_uncleaned = (target_stage == "raw" and current_status == "raw")

        # 1b. Check if conversational / conceptual / advisory
        if self.is_conversational_question(question):
            reply = await self.answer_conversational(
                question=question,
                dataset_id=dataset_id,
                target_stage=target_stage,
                history=history,
            )
            return {
                "sql": None,
                "explanation": "Conversational data intelligence & assistance",
                "results": None,
                "interpretation": reply,
                "error": None,
                "data_source": target_stage,
                "data_source_label": source_label,
                "processing_status": current_status,
                "suggested_followups": self.generate_suggested_followups(dataset_id, question),
            }

        # 2. Generate SQL for selected stage with conversation context
        sql_response = await self.generate_sql(
            question=question,
            dataset_id=dataset_id,
            target_stage=target_stage,
            history=history,
        )
        sql_query = sql_response["sql"]
        explanation = sql_response.get("explanation", "")

        logger.info(f"Generated SQL ({target_stage}): {sql_query}")

        # 3. Execute SQL via DuckDB
        try:
            results = execute_sql(dataset_id, sql_query, target_stage=target_stage)
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            corrected = await self._self_correct(question, dataset_id, sql_query, str(e), target_stage=target_stage)
            if corrected:
                sql_query = corrected["sql"]
                explanation = corrected.get("explanation", explanation)
                results = execute_sql(dataset_id, sql_query, target_stage=target_stage)
            else:
                return {
                    "sql": sql_query,
                    "explanation": explanation,
                    "results": None,
                    "interpretation": f"Unable to execute query: {e}. Please try rephrasing or asking about a specific column.",
                    "error": str(e),
                    "data_source": target_stage,
                    "data_source_label": source_label,
                    "processing_status": current_status,
                    "suggested_followups": self.generate_suggested_followups(dataset_id, question),
                }

        # 4. Interpret Results
        interpretation = await self.interpret_results(
            question=question,
            sql_query=sql_query,
            results=results,
            target_stage=target_stage,
            is_raw_uncleaned=is_raw_uncleaned,
            history=history,
        )

        # 5. Suggested follow-ups
        followups = self.generate_suggested_followups(dataset_id, question, results)

        return {
            "sql": sql_query,
            "explanation": explanation,
            "results": results,
            "interpretation": interpretation,
            "error": None,
            "data_source": target_stage,
            "data_source_label": source_label,
            "processing_status": current_status,
            "suggested_followups": followups,
        }

    async def _self_correct(
        self,
        question: str,
        dataset_id: str,
        failed_sql: str,
        error: str,
        target_stage: str = "active",
    ) -> dict[str, str] | None:
        """Attempt to fix a failed SQL query using LLM reflection."""
        schema = get_table_schema(dataset_id, target_stage=target_stage)

        messages = [
            SystemMessage(content=NL_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=f"""The following SQL query failed in DuckDB. Please fix it.

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
