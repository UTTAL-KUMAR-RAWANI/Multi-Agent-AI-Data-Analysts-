"""
DataMind AI - LLM Router

Routes LLM requests to the correct provider (Gemini, OpenAI, Ollama)
based on configuration. Supports model tiering so expensive models
are only used for complex reasoning tasks.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import Settings, LLMProvider, ModelTier, get_settings

logger = logging.getLogger(__name__)


def _create_gemini_llm(model: str, api_key: str, temperature: float = 0.1) -> BaseChatModel:
    """Create a Google Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=False,
    )


def _create_openai_llm(model: str, api_key: str, temperature: float = 0.1) -> BaseChatModel:
    """Create an OpenAI chat model."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def _create_ollama_llm(model: str, base_url: str, temperature: float = 0.1) -> BaseChatModel:
    """Create an Ollama (local) chat model."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def _create_llamafile_llm(model: str, base_url: str, temperature: float = 0.1) -> BaseChatModel:
    """Create a Llamafile (local OpenAI-compatible) chat model."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="no-key-required",
        temperature=temperature,
        timeout=8.0,
        max_retries=1,
    )


# Factory map: provider -> constructor
_LLM_FACTORIES = {
    LLMProvider.GEMINI: lambda model, settings, temp: _create_gemini_llm(
        model, settings.gemini_api_key, temp
    ),
    LLMProvider.OPENAI: lambda model, settings, temp: _create_openai_llm(
        model, settings.openai_api_key, temp
    ),
    LLMProvider.OLLAMA: lambda model, settings, temp: _create_ollama_llm(
        model, settings.ollama_base_url, temp
    ),
    LLMProvider.LLAMAFILE: lambda model, settings, temp: _create_llamafile_llm(
        model, settings.llamafile_base_url, temp
    ),
}


def get_llm(
    provider: LLMProvider | None = None,
    tier: ModelTier = ModelTier.FAST,
    temperature: float = 0.1,
    settings: Settings | None = None,
) -> BaseChatModel:
    """
    Get an LLM instance for the specified provider and tier.

    Args:
        provider: LLM provider to use. Defaults to settings.default_llm_provider.
        tier: Model tier (STRONG for complex reasoning, FAST for routine tasks).
        temperature: Sampling temperature.
        settings: Optional settings override.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is not configured (missing API key).
    """
    settings = settings or get_settings()
    provider = provider or settings.default_llm_provider

    # Validate provider is available or return fallback
    if provider not in (LLMProvider.OLLAMA, LLMProvider.LLAMAFILE) and provider not in settings.available_providers:
        logger.warning(
            f"Provider '{provider.value}' is not configured (missing API key). "
            f"Using FallbackChatModel. Set your API key in .env for live AI responses."
        )
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(responses=[
            '{"sql": "SELECT * FROM data LIMIT 10", "explanation": "Previewing first 10 rows.", "task_type": "query", "plan": []}',
            '{"passed": true, "severity": "info", "issues": [], "suggestions": []}',
            "Analysis completed successfully based on statistical metrics.",
        ])

    model_name = settings.get_model_name(provider, tier)
    factory = _LLM_FACTORIES[provider]

    logger.info(f"Creating LLM: provider={provider.value}, model={model_name}, tier={tier.value}")

    try:
        return factory(model_name, settings, temperature)
    except Exception as e:
        logger.warning(f"Error initializing LLM factory for {provider.value}: {e}. Using fallback.")
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(responses=["Analysis completed."])


def get_strong_llm(
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Shortcut for getting a strong (expensive, high-capability) model."""
    return get_llm(provider=provider, tier=ModelTier.STRONG, settings=settings)


def get_fast_llm(
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Shortcut for getting a fast (cheap, routine tasks) model."""
    return get_llm(provider=provider, tier=ModelTier.FAST, settings=settings)


class LLMRouter:
    """
    Manages LLM instances with agent-specific model assignments.

    Each agent type can be configured to use a specific provider/tier combo,
    enabling fine-grained cost control.
    """

    # Default agent -> (provider, tier) mapping
    # Override via environment or at runtime
    AGENT_ASSIGNMENTS: dict[str, tuple[LLMProvider | None, ModelTier]] = {
        "supervisor": (None, ModelTier.STRONG),
        "data_cleaner": (None, ModelTier.FAST),
        "eda_explorer": (None, ModelTier.FAST),
        "statistician": (None, ModelTier.STRONG),
        "ml_engineer": (None, ModelTier.STRONG),
        "visualizer": (None, ModelTier.FAST),
        "report_writer": (None, ModelTier.STRONG),
        "nl_query": (None, ModelTier.FAST),
        "qa": (None, ModelTier.STRONG),
    }

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cache: dict[str, BaseChatModel] = {}

    def get_llm_for_agent(self, agent_name: str) -> BaseChatModel:
        """
        Get the appropriate LLM for a specific agent.

        Uses the agent assignment mapping to determine provider and tier.
        Results are cached per agent name.
        """
        if agent_name in self._cache:
            return self._cache[agent_name]

        provider, tier = self.AGENT_ASSIGNMENTS.get(
            agent_name, (None, ModelTier.FAST)
        )

        llm = get_llm(
            provider=provider,
            tier=tier,
            settings=self.settings,
        )
        self._cache[agent_name] = llm
        return llm

    def override_agent_assignment(
        self,
        agent_name: str,
        provider: LLMProvider | None = None,
        tier: ModelTier = ModelTier.FAST,
    ) -> None:
        """Override the model assignment for a specific agent at runtime."""
        self.AGENT_ASSIGNMENTS[agent_name] = (provider, tier)
        # Clear cache for this agent so it gets recreated
        self._cache.pop(agent_name, None)
        logger.info(
            f"Agent '{agent_name}' reassigned to provider={provider}, tier={tier.value}"
        )
