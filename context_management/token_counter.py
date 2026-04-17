"""Token counting abstraction for Context Management."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken

from context_management.exceptions import TokenCounterError


class TokenCounter(ABC):
    """Abstract base class for token counting."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text. Raises TokenCounterError on failure."""


class TiktokenCounter(TokenCounter):
    """Token counter using tiktoken (offline, fast)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise TokenCounterError(
                f"Failed to load tiktoken encoding '{encoding_name}': {e}"
            ) from e

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            raise TokenCounterError(f"Token counting failed: {e}") from e


def create_token_counter(provider: str) -> TokenCounter:
    """Factory for creating token counters."""
    if provider == "tiktoken":
        return TiktokenCounter()
    raise ValueError(f"Unknown token counter provider: {provider}")
