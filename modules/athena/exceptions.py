"""
modules/athena/exceptions.py
"""

class AthenaError(Exception):
    """Base exception for all Athena errors."""


class QueryError(AthenaError):
    """Raised when the full query pipeline fails."""


class LLMError(AthenaError):
    """Raised when LLM generation fails."""


class RAGError(AthenaError):
    """Raised when retrieval fails."""
