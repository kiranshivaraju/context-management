"""Tests for MemoryManager facade."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_management.config import MemoryConfig
from context_management.exceptions import ValidationError


def _make_config() -> MemoryConfig:
    return MemoryConfig(database_url="postgresql+asyncpg://localhost/test")


def _create_mm():
    """Create MemoryManager with mocked LLM provider."""
    from context_management import MemoryManager

    with patch("context_management.create_llm_provider") as mock_llm_factory:
        mock_llm_factory.return_value = MagicMock()
        mm = MemoryManager(_make_config())
    return mm


class TestValidation:
    def test_validate_source_id_empty_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="source_id"):
            mm._validate_source_id("")

    def test_validate_source_id_too_long_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="source_id"):
            mm._validate_source_id("x" * 513)

    def test_validate_source_id_valid(self) -> None:
        mm = _create_mm()
        mm._validate_source_id("valid-source")  # Should not raise

    def test_validate_thread_id_none_allowed(self) -> None:
        mm = _create_mm()
        mm._validate_thread_id(None)  # Should not raise

    def test_validate_thread_id_empty_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="thread_id"):
            mm._validate_thread_id("")

    def test_validate_thread_id_too_long_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="thread_id"):
            mm._validate_thread_id("x" * 513)

    def test_validate_user_id_empty_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="user_id"):
            mm._validate_user_id("")

    def test_validate_user_id_too_long_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="user_id"):
            mm._validate_user_id("x" * 256)

    def test_validate_content_empty_raises(self) -> None:
        mm = _create_mm()
        with pytest.raises(ValidationError, match="content"):
            mm._validate_content("")


class TestCheckInitialized:
    def test_raises_before_init(self) -> None:
        mm = _create_mm()
        with pytest.raises(RuntimeError, match="not initialized"):
            mm._check_initialized()


class TestConstructor:
    def test_creates_all_components(self) -> None:
        mm = _create_mm()
        assert mm._db is not None
        assert mm._token_counter is not None
        assert mm._memory_store is not None
        assert mm._compaction_engine is not None
        assert mm._context_assembler is not None


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_calls_db_init(self) -> None:
        mm = _create_mm()
        mm._db = MagicMock()
        mm._db.initialize = AsyncMock()

        await mm.initialize()

        mm._db.initialize.assert_called_once()
        assert mm._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_calls_db_shutdown(self) -> None:
        mm = _create_mm()
        mm._db = MagicMock()
        mm._db.initialize = AsyncMock()
        mm._db.shutdown = AsyncMock()

        await mm.initialize()
        await mm.shutdown()

        mm._db.shutdown.assert_called_once()
        assert mm._initialized is False


class TestMemoryMethods:
    @pytest.mark.asyncio
    async def test_store_memory_delegates(self) -> None:
        mm = _create_mm()
        mm._initialized = True
        fake_mem = MagicMock()
        mm._memory_store = MagicMock()
        mm._memory_store.store = AsyncMock(return_value=fake_mem)

        result = await mm.store_memory("src-1", "PostgreSQL is the DB", "alice")

        mm._memory_store.store.assert_called_once_with(
            "src-1", "PostgreSQL is the DB", "alice"
        )
        assert result is fake_mem

    @pytest.mark.asyncio
    async def test_get_memories_delegates(self) -> None:
        mm = _create_mm()
        mm._initialized = True
        mm._memory_store = MagicMock()
        mm._memory_store.get_active = AsyncMock(return_value=[])

        result = await mm.get_memories("src-1")

        mm._memory_store.get_active.assert_called_once_with("src-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_memory_delegates(self) -> None:
        mm = _create_mm()
        mm._initialized = True
        mm._memory_store = MagicMock()
        mm._memory_store.delete = AsyncMock()

        mid = uuid.uuid4()
        await mm.delete_memory(mid)

        mm._memory_store.delete.assert_called_once_with(mid)


class TestExports:
    def test_all_exports_available(self) -> None:
        import context_management

        expected = [
            "MemoryManager", "MemoryConfig", "AssembledContext",
            "MessageRole", "MemoryManagerError", "SourceNotFoundError",
            "CompactionError", "LLMProviderError", "TokenCounterError",
            "ValidationError",
        ]
        for name in expected:
            assert hasattr(context_management, name), f"Missing export: {name}"
