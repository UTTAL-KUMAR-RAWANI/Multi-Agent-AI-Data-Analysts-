"""
DataWhale AI - Configuration Management

Centralized settings using pydantic-settings. Loads from .env file
and supports multiple LLM providers with model tiering.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LLAMAFILE = "llamafile"


class ModelTier(str, Enum):
    """Model capability tiers for cost optimization."""
    STRONG = "strong"  # Complex reasoning, supervisor, QA
    FAST = "fast"      # Routine tasks, data cleaning, simple queries


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM API Keys & Endpoints ---
    gemini_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llamafile_base_url: str = "http://127.0.0.1:8080/v1"
    llamafile_timeout: float = 90.0

    # --- Default Provider ---
    default_llm_provider: LLMProvider = LLMProvider.LLAMAFILE

    # --- Model Names per Provider ---
    gemini_strong_model: str = "gemini-2.5-pro-preview-05-06"
    gemini_fast_model: str = "gemini-2.0-flash"
    openai_strong_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    ollama_strong_model: str = "llama3.3:70b"
    ollama_fast_model: str = "qwen2.5:7b"
    llamafile_strong_model: str = "LLaMA_CPP"
    llamafile_fast_model: str = "LLaMA_CPP"

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    # --- Data Directories ---
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    reports_dir: str = "data/reports"
    max_upload_size_mb: int = 500

    def get_model_name(
        self,
        provider: LLMProvider | None = None,
        tier: ModelTier = ModelTier.FAST,
    ) -> str:
        """Get the model name for a given provider and tier."""
        provider = provider or self.default_llm_provider
        key = f"{provider.value}_{tier.value}_model"
        return getattr(self, key)

    def get_upload_path(self) -> Path:
        """Get absolute upload directory path, creating it if needed."""
        path = PROJECT_ROOT / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_processed_path(self) -> Path:
        """Get absolute processed data directory path."""
        path = PROJECT_ROOT / self.processed_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_reports_path(self) -> Path:
        """Get absolute reports directory path."""
        path = PROJECT_ROOT / self.reports_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def available_providers(self) -> list[LLMProvider]:
        """List providers that have valid configuration."""
        providers = []
        if self.gemini_api_key:
            providers.append(LLMProvider.GEMINI)
        if self.openai_api_key:
            providers.append(LLMProvider.OPENAI)
        # Ollama and Llamafile are always "available" (local, no API key needed)
        providers.append(LLMProvider.OLLAMA)
        providers.append(LLMProvider.LLAMAFILE)
        return providers


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
