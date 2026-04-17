"""Tests for memory compaction (consolidation)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from context_management.memory import MemoryStore


def _make_config(**overrides) -> MagicMock:
    cfg = MagicMock()
    cfg.enable_memory_compaction = overrides.get("enable_memory_compaction", True)
    cfg.memory_compaction_count_threshold = overrides.get("memory_compaction_count_threshold", 5)
    cfg.memory_compaction_token_threshold = overrides.get("memory_compaction_token_threshold", 100)
    cfg.max_memories_per_source = overrides.get("max_memories_per_source", 50)
    cfg.extraction_max_output_tokens = 1000
    return cfg


def _make_memory(content: str = "fact", tokens: int = 10,
                 user_id: str = "alice") -> MagicMock:
    mem = MagicMock()
    mem.id = uuid.uuid4()
    mem.content = content
    mem.token_count = tokens
    mem.attributed_user_id = user_id
    return mem


class TestShouldCompactMemories:
    @pytest.mark.asyncio
    async def test_below_both_thresholds_returns_false(self) -> None:
        cfg = _make_config(memory_compaction_count_threshold=10,
                           memory_compaction_token_threshold=200)
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        # 3 memories, 30 tokens — both below thresholds
        store.get_active = AsyncMock(
            return_value=[_make_memory(tokens=10) for _ in range(3)]
        )
        assert await store.should_compact_memories("src-1") is False

    @pytest.mark.asyncio
    async def test_above_count_threshold_returns_true(self) -> None:
        cfg = _make_config(memory_compaction_count_threshold=5,
                           memory_compaction_token_threshold=99999)
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        store.get_active = AsyncMock(
            return_value=[_make_memory() for _ in range(6)]
        )
        assert await store.should_compact_memories("src-1") is True

    @pytest.mark.asyncio
    async def test_above_token_threshold_returns_true(self) -> None:
        cfg = _make_config(memory_compaction_count_threshold=99999,
                           memory_compaction_token_threshold=50)
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        # 3 memories × 20 tokens = 60, above 50 threshold
        store.get_active = AsyncMock(
            return_value=[_make_memory(tokens=20) for _ in range(3)]
        )
        assert await store.should_compact_memories("src-1") is True

    @pytest.mark.asyncio
    async def test_disabled_returns_false(self) -> None:
        cfg = _make_config(enable_memory_compaction=False)
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        store.get_active = AsyncMock(
            return_value=[_make_memory() for _ in range(100)]
        )
        assert await store.should_compact_memories("src-1") is False

    @pytest.mark.asyncio
    async def test_no_memories_returns_false(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        store.get_active = AsyncMock(return_value=[])
        assert await store.should_compact_memories("src-1") is False


class TestCompactMemories:
    @pytest.mark.asyncio
    async def test_single_memory_skipped(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)
        single = [_make_memory("only fact")]
        store.get_active = AsyncMock(return_value=single)
        llm = AsyncMock()

        result = await store.compact_memories("src-1", llm)

        assert result == single
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidates_and_replaces(self) -> None:
        cfg = _make_config()
        tc = MagicMock()
        tc.count_tokens.return_value = 8
        store = MemoryStore(MagicMock(), tc, cfg)

        old_memories = [_make_memory(f"fact-{i}") for i in range(5)]
        store.get_active = AsyncMock(return_value=old_memories)

        # LLM returns consolidated set (fewer than original)
        consolidated = [
            {"content": "merged fact A", "attributed_user_id": "alice"},
            {"content": "merged fact B", "attributed_user_id": "bob"},
        ]
        llm = AsyncMock()
        llm.generate.return_value = json.dumps(consolidated)

        store._execute_memory_compaction = AsyncMock(return_value=[MagicMock(), MagicMock()])

        result = await store.compact_memories("src-1", llm)

        store._execute_memory_compaction.assert_called_once_with(
            "src-1", old_memories, consolidated
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_llm_response_leaves_unchanged(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)

        old = [_make_memory(f"fact-{i}") for i in range(5)]
        store.get_active = AsyncMock(return_value=old)

        llm = AsyncMock()
        llm.generate.return_value = "[]"

        result = await store.compact_memories("src-1", llm)

        assert result == old  # unchanged

    @pytest.mark.asyncio
    async def test_more_consolidated_than_original_skipped(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)

        old = [_make_memory(f"fact-{i}") for i in range(3)]
        store.get_active = AsyncMock(return_value=old)

        # LLM returns MORE items than original (bad consolidation)
        expanded = [{"content": f"expanded-{i}", "attributed_user_id": "x"} for i in range(5)]
        llm = AsyncMock()
        llm.generate.return_value = json.dumps(expanded)

        result = await store.compact_memories("src-1", llm)

        assert result == old  # unchanged

    @pytest.mark.asyncio
    async def test_equal_count_skipped(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)

        old = [_make_memory(f"fact-{i}") for i in range(3)]
        store.get_active = AsyncMock(return_value=old)

        # LLM returns same count (no reduction)
        same = [{"content": f"same-{i}", "attributed_user_id": "x"} for i in range(3)]
        llm = AsyncMock()
        llm.generate.return_value = json.dumps(same)

        result = await store.compact_memories("src-1", llm)

        assert result == old  # unchanged

    @pytest.mark.asyncio
    async def test_malformed_json_leaves_unchanged(self) -> None:
        cfg = _make_config()
        store = MemoryStore(MagicMock(), MagicMock(), cfg)

        old = [_make_memory(f"fact-{i}") for i in range(5)]
        store.get_active = AsyncMock(return_value=old)

        llm = AsyncMock()
        llm.generate.return_value = "not valid json at all"

        result = await store.compact_memories("src-1", llm)

        assert result == old  # unchanged


class TestConfigValidation:
    def test_memory_compaction_count_threshold_too_low(self) -> None:
        from context_management.config import MemoryConfig
        with pytest.raises(Exception, match="memory_compaction_count_threshold must be >= 2"):
            MemoryConfig(
                database_url="postgresql+asyncpg://localhost/test",
                memory_compaction_count_threshold=1,
            )

    def test_memory_compaction_token_threshold_too_low(self) -> None:
        from context_management.config import MemoryConfig
        with pytest.raises(Exception, match="memory_compaction_token_threshold must be >= 100"):
            MemoryConfig(
                database_url="postgresql+asyncpg://localhost/test",
                memory_compaction_token_threshold=50,
            )

    def test_valid_thresholds_accepted(self) -> None:
        from context_management.config import MemoryConfig
        cfg = MemoryConfig(
            database_url="postgresql+asyncpg://localhost/test",
            memory_compaction_count_threshold=10,
            memory_compaction_token_threshold=500,
        )
        assert cfg.memory_compaction_count_threshold == 10
        assert cfg.memory_compaction_token_threshold == 500
        assert cfg.enable_memory_compaction is True
