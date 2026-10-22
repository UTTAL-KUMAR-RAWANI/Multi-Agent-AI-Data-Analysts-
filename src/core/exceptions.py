"""
DataMind AI - Custom Exceptions
"""

from __future__ import annotations


class DataMindError(Exception):
    """Base exception for DataMind AI."""
    pass


class DatasetNotFoundError(DataMindError):
    """Raised when a dataset ID doesn't exist in the workspace."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' not found in workspace.")


class DatasetLoadError(DataMindError):
    """Raised when a file can't be loaded as a dataset."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to load '{file_path}': {reason}")


class LLMProviderError(DataMindError):
    """Raised when an LLM provider is unavailable or misconfigured."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM provider '{provider}' error: {reason}")


class QueryExecutionError(DataMindError):
    """Raised when a SQL query fails to execute."""

    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Query execution failed: {reason}")


class AgentError(DataMindError):
    """Raised when an agent encounters an error during execution."""

    def __init__(self, agent_name: str, reason: str):
        self.agent_name = agent_name
        self.reason = reason
        super().__init__(f"Agent '{agent_name}' error: {reason}")


class FileTooLargeError(DataMindError):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, file_name: str, size_mb: float, max_mb: int):
        self.file_name = file_name
        self.size_mb = size_mb
        self.max_mb = max_mb
        super().__init__(
            f"File '{file_name}' ({size_mb:.1f} MB) exceeds maximum size ({max_mb} MB)."
        )
