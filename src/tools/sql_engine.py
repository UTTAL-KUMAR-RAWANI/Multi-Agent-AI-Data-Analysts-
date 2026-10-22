"""
DataMind AI - DuckDB SQL Engine

Provides fast SQL query execution on pandas DataFrames using DuckDB.
This is the backbone of the NL Query Agent — natural language gets
translated to SQL, which runs here.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb
import pandas as pd

from src.core.exceptions import QueryExecutionError, DatasetNotFoundError
from src.tools.data_loader import get_dataset, get_dataset_metadata

logger = logging.getLogger(__name__)


def execute_sql(
    dataset_id: str,
    sql_query: str,
    max_rows: int = 1000,
) -> dict[str, Any]:
    """
    Execute a SQL query against a loaded dataset using DuckDB.

    The dataset is registered as a table with its dataset_id as the name.
    Column names with spaces or special characters are automatically quoted.

    Args:
        dataset_id: The dataset to query.
        sql_query: SQL query string. Use the dataset_id as the table name.
        max_rows: Maximum rows to return (prevents massive results).

    Returns:
        Dict with keys: columns, rows, row_count, truncated, query.

    Raises:
        QueryExecutionError: If the SQL query fails.
        DatasetNotFoundError: If the dataset doesn't exist.
    """
    df = get_dataset(dataset_id)

    logger.info(f"Executing SQL on dataset '{dataset_id}': {sql_query[:200]}")

    try:
        # Create a DuckDB connection and register the DataFrame
        conn = duckdb.connect()
        conn.register(dataset_id, df)

        # Also register with a clean alias "data" for convenience
        conn.register("data", df)

        # Execute the query
        result = conn.execute(sql_query)
        result_df = result.fetchdf()

        # Check if we need to truncate
        truncated = len(result_df) > max_rows
        if truncated:
            result_df = result_df.head(max_rows)

        # Convert to serializable format
        columns = result_df.columns.tolist()
        rows = result_df.to_dict(orient="records")

        # Clean up NaN/None for JSON serialization
        for row in rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None

        conn.close()

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "total_count": len(result_df) if not truncated else f">{max_rows}",
            "truncated": truncated,
            "query": sql_query,
        }

    except DatasetNotFoundError:
        raise
    except Exception as e:
        raise QueryExecutionError(sql_query, str(e))


def get_table_schema(dataset_id: str) -> str:
    """
    Get a SQL-friendly schema description for the LLM.

    Returns a CREATE TABLE-like statement that tells the LLM
    what columns exist and their types.
    """
    meta = get_dataset_metadata(dataset_id)

    # Map pandas dtypes to SQL types for LLM understanding
    dtype_map = {
        "int64": "INTEGER",
        "int32": "INTEGER",
        "float64": "DOUBLE",
        "float32": "FLOAT",
        "object": "VARCHAR",
        "string": "VARCHAR",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "category": "VARCHAR",
    }

    columns_sql = []
    for col in meta["columns"]:
        sql_type = dtype_map.get(col["dtype"], "VARCHAR")
        # Quote column names to handle spaces and special characters
        columns_sql.append(f'    "{col["name"]}" {sql_type}')

    schema = f'-- Table: "{dataset_id}" (also aliased as "data")\n'
    schema += f'-- {meta["num_rows"]} rows, {meta["num_columns"]} columns\n'
    schema += f'CREATE TABLE "{dataset_id}" (\n'
    schema += ",\n".join(columns_sql)
    schema += "\n);"

    return schema


def get_sample_values(dataset_id: str, max_unique: int = 10) -> str:
    """
    Get sample unique values for each column.

    Helps the LLM understand what kind of data is in each column,
    so it can write better SQL (e.g., knowing exact category names).
    """
    df = get_dataset(dataset_id)
    lines = []

    for col in df.columns:
        nunique = df[col].nunique()
        if nunique <= max_unique:
            values = df[col].dropna().unique().tolist()
            values_str = ", ".join(repr(v) for v in values[:max_unique])
            lines.append(f'"{col}": [{values_str}]')
        else:
            sample = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
            values_str = ", ".join(repr(v) for v in sample)
            lines.append(f'"{col}": [{values_str}, ...] ({nunique} unique values)')

    return "Sample values per column:\n" + "\n".join(lines)
