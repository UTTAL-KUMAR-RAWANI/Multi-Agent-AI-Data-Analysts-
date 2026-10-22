"""
DataMind AI - FastAPI Application

Main entry point for the web application.
Serves both the API endpoints and the web dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from src.core.config import get_settings
from src.api.routes import upload, query, analysis, reports, dashboard_chat
from src.tools.data_loader import list_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# FastAPI app
app = FastAPI(
    title="DataMind AI",
    description="Multi-Agent AI Data Analysis Platform",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register API routes
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(dashboard_chat.router)


@app.on_event("startup")
async def preload_existing_datasets():
    """Ensure any datasets in data/uploads are loaded into memory on server start."""
    from src.tools.data_loader import load_dataset
    settings = get_settings()
    upload_path = settings.get_upload_path()
    if upload_path.exists():
        for file in upload_path.iterdir():
            if file.suffix.lower() in [".csv", ".xlsx", ".xls", ".json"]:
                try:
                    load_dataset(str(file))
                    logger.info(f"Preloaded dataset on startup: {file.name}")
                except Exception as e:
                    logger.warning(f"Could not preload {file.name}: {e}")


# ─── Web Dashboard Routes ──────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    datasets = list_datasets()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "datasets": datasets,
            "active_page": "dashboard",
        },
    )


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Dataset upload page."""
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "active_page": "upload",
        },
    )


@app.get("/chat/{dataset_id}", response_class=HTMLResponse)
async def chat_page(request: Request, dataset_id: str):
    """Chat interface for querying a dataset."""
    from src.tools.data_loader import get_dataset_metadata
    from src.core.exceptions import DatasetNotFoundError

    try:
        metadata = get_dataset_metadata(dataset_id)
    except DatasetNotFoundError:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error": f"Dataset '{dataset_id}' not found.",
                "active_page": "chat",
            },
            status_code=404,
        )

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "dataset_id": dataset_id,
            "metadata": metadata,
            "active_page": "chat",
        },
    )


@app.get("/analysis/{dataset_id}", response_class=HTMLResponse)
async def analysis_page(request: Request, dataset_id: str):
    """Multi-Agent Studio page for in-depth analysis and automated modeling."""
    from src.tools.data_loader import get_dataset_metadata
    from src.core.exceptions import DatasetNotFoundError

    try:
        metadata = get_dataset_metadata(dataset_id)
    except DatasetNotFoundError:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error": f"Dataset '{dataset_id}' not found.",
                "active_page": "analysis",
            },
            status_code=404,
        )

    return templates.TemplateResponse(
        request=request,
        name="analysis.html",
        context={
            "dataset_id": dataset_id,
            "metadata": metadata,
            "active_page": "analysis",
        },
    )


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports archive view."""
    import datetime
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_list = []
    for p in sorted(report_dir.glob("*.html"), key=lambda f: f.stat().st_mtime, reverse=True):
        stat = p.stat()
        report_list.append({
            "report_id": p.stem,
            "filename": p.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "view_url": f"/reports/view/{p.stem}",
            "download_url": f"/reports/download/{p.stem}",
        })

    return templates.TemplateResponse(
        request=request,
        name="reports.html",
        context={
            "reports": report_list,
            "active_page": "reports",
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "providers": [p.value for p in settings.available_providers],
        "datasets_loaded": len(list_datasets()),
    }


# ─── Startup / Shutdown ────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Application startup — create data directories."""
    settings = get_settings()
    settings.get_upload_path()
    settings.get_processed_path()
    settings.get_reports_path()
    logger.info("=" * 60)
    logger.info("  DataMind AI — Multi-Agent Data Analysis Platform")
    logger.info("=" * 60)
    logger.info(f"  Available LLM providers: {[p.value for p in settings.available_providers]}")
    logger.info(f"  Default provider: {settings.default_llm_provider.value}")
    logger.info(f"  Dashboard: http://{settings.host}:{settings.port}")
    logger.info("=" * 60)
