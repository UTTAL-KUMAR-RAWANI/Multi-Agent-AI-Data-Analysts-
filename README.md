# 🐋 DataWhale AI — Multi-Agent AI Data Analysis Platform with MCP Integration

> **DataWhale AI** is an intelligent, autonomous multi-agent data analytics platform that performs the full workflow of a professional data analytics and data science team: data hygiene, exploratory profiling, inferential hypothesis testing, machine learning modeling, interactive visual storytelling, and executive report publishing.
>
> Built with **LangGraph**, **FastAPI**, **DuckDB**, **Plotly**, and official **Model Context Protocol (MCP)** server integration.

---

## 🌟 Key Capabilities

- **🤖 8 Specialized AI Agents:**
  - **Supervisor Agent:** Decomposes complex user requests, schedules task steps, and synthesizes executive briefings.
  - **Data Cleaner Agent:** Standardizes schemas, imputes missing values, eliminates duplicates, and mitigates outliers.
  - **EDA Explorer Agent:** Computes central tendencies, dispersion, skewness, and discovers correlation networks.
  - **Statistician Agent:** Runs Welch's t-tests, ANOVA, Chi-Square tests of independence, and normality checks.
  - **ML Engineer Agent:** Trains, benchmarks, and tunes Random Forest, Gradient Boosting, and Linear/Logistic baselines with automated feature importance.
  - **Visualizer Agent:** Recommends and renders dark-mode interactive Plotly visualizations.
  - **Report Writer Agent:** Generates publication-ready executive HTML and PDF reports with embedded charts and audit logs.
  - **QA Agent:** Audits agent outputs for statistical validity, target leakage, and numerical integrity before results reach the user.

- **🔌 Native MCP Server Platform:** Exposes all 9 core analysis tools via the Model Context Protocol, allowing **Claude Desktop**, **Cursor**, or **VS Code** to directly invoke DataWhale on local datasets.
- **⚡ Blazing Fast SQL Engine:** Powered by DuckDB for lightning-fast SQL aggregations over CSV, Excel, and JSON files without database overhead.
- **🎨 Glassmorphic Dark UI:** Beautiful browser-based dashboard with responsive charts, natural language chat studio, and report repository.

---

## 📁 Project Architecture

```
c:\ai data analysts\
├── 📄 pyproject.toml              # Dependencies & package configuration
├── 📄 README.md                   # Platform documentation
├── 📁 src/
│   ├── 📁 core/                   # Config, LLM router (Gemini/OpenAI/Ollama), state
│   ├── 📁 agents/                 # All 8 specialist AI agents
│   ├── 📁 tools/                  # Computation engine (pandas, duckdb, scipy, sklearn, plotly)
│   ├── 📁 graph/                  # LangGraph multi-agent state graph workflow
│   ├── 📁 mcp_server/             # Official MCP server implementation
│   ├── 📁 api/                    # FastAPI web server & REST endpoints
│   └── 📁 web/                    # Templates (Jinja2) & static assets (CSS, JS)
├── 📁 data/                       # Workspace (uploads, processed, reports, models)
└── 📁 docs/                       # Guides and MCP integration instructions
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone <repo-url>
cd "ai data analysts"
pip install -e .
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your preferred LLM provider:

```bash
cp .env.example .env
```

```env
DEFAULT_LLM_PROVIDER=llamafile     # llamafile | gemini | openai | ollama
LLAMAFILE_BASE_URL=http://127.0.0.1:8080/v1
```

If using **Llamafile** (local):
```bash
# Start the local model server (uses llamafile-0.10.6.exe and models/qwen2.5-3b-instruct-q4_k_m.gguf)
.\start_llamafile.bat
```

### 3. Launching the Web Dashboard

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**.

### 4. Running the MCP Server for Claude / Cursor

```bash
python -m src.mcp_server.server
```

*(See [docs/mcp_integration.md](docs/mcp_integration.md) for full Claude Desktop and Cursor setup).*

---

## 🛡️ License

MIT License. Designed for rapid, reliable, and privacy-preserving data intelligence.
