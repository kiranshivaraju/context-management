"""Custom exceptions for Context Management."""

from __future__ import annotations


class MemoryManagerError(Exception):
    """Base exception for all Context Management errors."""


class SourceNotFoundError(MemoryManagerError):
    """Raised when a source_id/thread_id has no state (never received a message)."""


class CompactionError(MemoryManagerError):
    """Raised when compaction fails (LLM or DB error during compaction flow)."""


class LLMProviderError(MemoryManagerError):
    """Raised when an LLM provider call fails. Wraps provider-specific errors."""


class TokenCounterError(MemoryManagerError):
    """Raised when token counting fails."""


class ValidationError(MemoryManagerError):
    """Raised when input validation fails (empty source_id, invalid role, etc.)."""
