# 🔌 DataMind AI — Model Context Protocol (MCP) Integration Guide

DataMind AI natively implements the **Model Context Protocol (MCP)** standard (v2.x). Any MCP-compliant client (including **Claude Desktop**, **Cursor**, and **VS Code Copilot**) can directly access DataMind's suite of specialized data analyst tools and multi-agent workflows.

---

## 🛠️ Available MCP Tools

| MCP Tool Name | Description | Key Parameters |
|---|---|---|
| `load_dataset` | Load CSV, Excel, or JSON into workspace | `file_path: str` |
| `describe_dataset` | Schema, dtypes, nulls, summary statistics | `dataset_id: str` |
| `list_available_datasets` | List all currently active datasets | *None* |
| `clean_dataset` | Auto-impute nulls, drop duplicates, clip outliers | `dataset_id: str`, `normalize_columns: bool` |
| `profile_data` | Deep exploratory profiling & correlation matrix | `dataset_id: str` |
| `execute_sql_query` | Run DuckDB SQL query directly on loaded tables | `sql_query: str` |
| `run_statistical_test` | Inferential tests (t-test, ANOVA, chi-square, corr) | `dataset_id: str`, `test_type: str`, `col1`, `col2` |
| `train_ml_model` | Auto-train and benchmark ML models | `dataset_id: str`, `target_column: str` |
| `full_multi_agent_analysis` | Autonomous LangGraph multi-agent pipeline | `dataset_id: str`, `objective: str` |

---

## ⚙️ Connecting to Claude Desktop

Add the following configuration to your Claude Desktop config file:

- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "datamind": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp_server.server"
      ],
      "cwd": "C:\\ai data analysts"
    }
  }
}
```

Restart Claude Desktop. You will see the hammer icon 🔨 with all DataMind analytics tools ready to call.

---

## 💻 Connecting to Cursor

In Cursor:
1. Open **Settings** → **Features** → **MCP Servers**.
2. Click **Add New MCP Server**.
3. Name: `datamind`
4. Type: `command`
5. Command: `python -m src.mcp_server.server`
6. Working Directory: `C:\ai data analysts`

---

## 🚀 Testing the MCP Server Directly

Run the MCP server via CLI:
```bash
python -m src.mcp_server.server
```
The server communicates via standard JSON-RPC over `stdio`.
