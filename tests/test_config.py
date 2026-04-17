"""Tests for enums, exceptions, and MemoryConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from context_management.enums import MessageRole
from context_management.exceptions import (
    CompactionError,
    LLMProviderError,
    MemoryManagerError,
    SourceNotFoundError,
    TokenCounterError,
    ValidationError,
)
from context_management.config import MemoryConfig


class TestMessageRole:
    def test_values(self) -> None:
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_is_str(self) -> None:
        assert isinstance(MessageRole.USER, str)

    def test_all_roles(self) -> None:
        assert set(MessageRole) == {
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.SYSTEM,
        }


class TestExceptions:
    def test_base_exception(self) -> None:
        with pytest.raises(MemoryManagerError):
            raise MemoryManagerError("test")

    def test_all_inherit_from_base(self) -> None:
        exceptions = [
            SourceNotFoundError,
            CompactionError,
            LLMProviderError,
            TokenCounterError,
            ValidationError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, MemoryManagerError)

    def test_exceptions_carry_message(self) -> None:
        err = SourceNotFoundError("source xyz not found")
        assert str(err) == "source xyz not found"

    def test_catch_specific_via_base(self) -> None:
        with pytest.raises(MemoryManagerError):
            raise CompactionError("compaction failed")


class TestMemoryConfig:
    def test_defaults(self) -> None:
        config = MemoryConfig(database_url="postgresql+asyncpg://localhost/test")
        assert config.max_context_tokens == 180_000
        assert config.compaction_trigger_ratio == 0.75
        assert config.compaction_target_ratio == 0.50
        assert config.protected_message_count == 10
        assert config.memory_budget == 5_000
        assert config.summary_budget == 10_000
        assert config.output_reserve == 8_000
        assert config.extract_memories_on_compaction is True
        assert config.max_memories_per_source == 100
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.compaction_max_output_tokens == 2000
        assert config.extraction_max_output_tokens == 1000
        assert config.token_counter_provider == "tiktoken"

    def test_custom_values(self) -> None:
        config = MemoryConfig(
            database_url="postgresql+asyncpg://localhost/test",
            max_context_tokens=100_000,
            compaction_trigger_ratio=0.80,
            compaction_target_ratio=0.40,
            protected_message_count=5,
            memory_budget=3_000,
            max_memories_per_source=50,
            llm_provider="openai",
            llm_model="gpt-4o",
        )
        assert config.max_context_tokens == 100_000
        assert config.compaction_trigger_ratio == 0.80
        assert config.compaction_target_ratio == 0.40
        assert config.protected_message_count == 5
        assert config.llm_provider == "openai"

    def test_database_url_required(self) -> None:
        with pytest.raises(PydanticValidationError):
            MemoryConfig()  # type: ignore[call-arg]

    def test_trigger_ratio_too_low(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_trigger_ratio"):
            MemoryConfig(database_url="x", compaction_trigger_ratio=0)

    def test_trigger_ratio_too_high(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_trigger_ratio"):
            MemoryConfig(database_url="x", compaction_trigger_ratio=1.0)

    def test_trigger_ratio_negative(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_trigger_ratio"):
            MemoryConfig(database_url="x", compaction_trigger_ratio=-0.5)

    def test_trigger_ratio_above_one(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_trigger_ratio"):
            MemoryConfig(database_url="x", compaction_trigger_ratio=1.5)

    def test_target_ratio_gte_trigger(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_target_ratio"):
            MemoryConfig(
                database_url="x",
                compaction_trigger_ratio=0.75,
                compaction_target_ratio=0.75,
            )

    def test_target_ratio_above_trigger(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_target_ratio"):
            MemoryConfig(
                database_url="x",
                compaction_trigger_ratio=0.75,
                compaction_target_ratio=0.80,
            )

    def test_target_ratio_zero(self) -> None:
        with pytest.raises(PydanticValidationError, match="compaction_target_ratio"):
            MemoryConfig(database_url="x", compaction_target_ratio=0)

    def test_protected_message_count_zero(self) -> None:
        with pytest.raises(PydanticValidationError, match="protected_message_count"):
            MemoryConfig(database_url="x", protected_message_count=0)

    def test_protected_message_count_negative(self) -> None:
        with pytest.raises(PydanticValidationError, match="protected_message_count"):
            MemoryConfig(database_url="x", protected_message_count=-1)

    def test_max_memories_zero(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_memories_per_source"):
            MemoryConfig(database_url="x", max_memories_per_source=0)

    def test_max_memories_negative(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_memories_per_source"):
            MemoryConfig(database_url="x", max_memories_per_source=-1)

    def test_max_context_tokens_too_small(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_context_tokens"):
            MemoryConfig(database_url="x", max_context_tokens=500)
