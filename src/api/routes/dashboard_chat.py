"""
DataMind AI - Dashboard Chat API Route

Provides endpoints for the Dashboard Copilot chatbot.
Allows users to ask anything about datasets, workspace statistics,
AI agents, and analysis workflows.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from src.agents.dashboard_assistant import DashboardAssistantAgent
from src.api.models.schemas import DashboardChatRequest, DashboardChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_assistant: DashboardAssistantAgent | None = None


def _get_assistant() -> DashboardAssistantAgent:
    """Singleton getter for DashboardAssistantAgent."""
    global _assistant
    if _assistant is None:
        _assistant = DashboardAssistantAgent()
    return _assistant


@router.post("/chat", response_model=DashboardChatResponse)
async def chat_dashboard(request: DashboardChatRequest):
    """
    Handle natural language inquiries about the dashboard and workspace.
    """
    assistant = _get_assistant()
    try:
        history_list = [h.model_dump() for h in request.history]
        result = await assistant.ask(request.message, history=history_list)
        return DashboardChatResponse(
            reply=result["reply"],
            suggested_actions=result.get("suggested_actions", []),
            quick_replies=result.get("quick_replies", []),
        )
    except Exception as e:
        logger.error(f"Dashboard chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


@router.get("/context")
async def get_dashboard_context():
    """
    Get live context summary for initial chatbot hydration.
    """
    assistant = _get_assistant()
    try:
        ctx = assistant.get_workspace_context()
        quick_replies = assistant._generate_quick_replies(ctx)
        return {
            "status": "ok",
            "context": ctx,
            "quick_replies": quick_replies,
        }
    except Exception as e:
        logger.error(f"Failed to fetch dashboard context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
