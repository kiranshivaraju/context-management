"""Shared test fixtures for Context Management test suite."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from context_management.config import MemoryConfig
from context_management.token_counter import TiktokenCounter


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM provider returning canned responses."""
    provider = AsyncMock()
    provider.generate.return_value = "[]"
    return provider


@pytest.fixture
def token_counter() -> TiktokenCounter:
    """Real tiktoken counter."""
    return TiktokenCounter()


@pytest.fixture
def config() -> MemoryConfig:
    """Test-friendly config with small values."""
    return MemoryConfig(
        database_url="postgresql+asyncpg://localhost/test",
        max_context_tokens=1000,
        compaction_trigger_ratio=0.75,
        compaction_target_ratio=0.50,
        protected_message_count=3,
        max_memories_per_source=5,
        memory_budget=200,
        summary_budget=200,
        output_reserve=100,
    )
