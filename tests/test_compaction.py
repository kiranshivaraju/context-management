"""Tests for CompactionEngine."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_management.compaction import CompactionEngine


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.max_context_tokens = 180_000
    cfg.compaction_trigger_ratio = 0.75
    cfg.compaction_target_ratio = 0.50
    cfg.protected_message_count = 10
    cfg.extract_memories_on_compaction = True
    cfg.compaction_max_output_tokens = 2000
    return cfg


def _make_message(seq: int, tokens: int = 100, content: str = "msg") -> MagicMock:
    msg = MagicMock()
    msg.id = uuid.uuid4()
    msg.source_id = "src-1"
    msg.thread_id = None
    msg.role = "user"
    msg.user_id = "alice"
    msg.content = content
    msg.token_count = tokens
    msg.sequence_num = seq
    msg.is_compacted = False
    return msg


def _make_source_state(total_tokens: int = 0, compaction_count: int = 0) -> MagicMock:
    state = MagicMock()
    state.total_token_count = total_tokens
    state.compaction_count = compaction_count
    return state


class TestShouldCompact:
    @pytest.mark.asyncio
    async def test_below_threshold_returns_false(self) -> None:
        cfg = _make_config()
        engine = CompactionEngine(MagicMock(), MagicMock(), MagicMock(), MagicMock(), cfg)

        state = _make_source_state(total_tokens=100_000)  # below 135K
        engine._get_source_state = AsyncMock(return_value=state)

        result = await engine.should_compact("src-1", None)
        assert result is False

    @pytest.mark.asyncio
    async def test_above_threshold_returns_true(self) -> None:
        cfg = _make_config()
        engine = CompactionEngine(MagicMock(), MagicMock(), MagicMock(), MagicMock(), cfg)

        state = _make_source_state(total_tokens=140_000)  # above 135K
        engine._get_source_state = AsyncMock(return_value=state)

        result = await engine.should_compact("src-1", None)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_source_state_returns_false(self) -> None:
        cfg = _make_config()
        engine = CompactionEngine(MagicMock(), MagicMock(), MagicMock(), MagicMock(), cfg)

        engine._get_source_state = AsyncMock(return_value=None)

        result = await engine.should_compact("src-1", None)
        assert result is False


class TestRunCompaction:
    @pytest.mark.asyncio
    async def test_no_compactable_messages_returns_early(self) -> None:
        cfg = _make_config()
        cfg.protected_message_count = 10
        db = MagicMock()
        llm = AsyncMock()
        tc = MagicMock()
        ms = MagicMock()

        engine = CompactionEngine(db, llm, tc, ms, cfg)

        # Only 5 messages total — all protected
        messages = [_make_message(i) for i in range(5)]
        state = _make_source_state(total_tokens=140_000, compaction_count=0)
        engine._get_active_messages = AsyncMock(return_value=messages)
        engine._get_source_state = AsyncMock(return_value=state)

        await engine.run_compaction("src-1", None)

        # LLM should not have been called
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_compacts_oldest_messages(self) -> None:
        cfg = _make_config()
        cfg.protected_message_count = 3
        db = MagicMock()
        llm = AsyncMock()
        llm.generate.return_value = "Summary of discussion."
        tc = MagicMock()
        tc.count_tokens.return_value = 50
        ms = MagicMock()
        ms.extract_and_dedup = AsyncMock(return_value=[])

        engine = CompactionEngine(db, llm, tc, ms, cfg)

        # 8 messages: first 5 compactable, last 3 protected
        messages = [_make_message(i, tokens=100) for i in range(8)]
        state = _make_source_state(total_tokens=140_000, compaction_count=0)
        engine._get_active_messages = AsyncMock(return_value=messages)
        engine._get_source_state = AsyncMock(return_value=state)
        engine._execute_compaction = AsyncMock()

        await engine.run_compaction("src-1", None)

        engine._execute_compaction.assert_called_once()
        call_args = engine._execute_compaction.call_args
        compactable = call_args[0][2]  # 3rd positional arg
        assert len(compactable) == 5
        assert compactable[0].sequence_num == 0
        assert compactable[-1].sequence_num == 4

    @pytest.mark.asyncio
    async def test_force_all_compacts_everything(self) -> None:
        cfg = _make_config()
        cfg.protected_message_count = 3
        db = MagicMock()
        llm = AsyncMock()
        llm.generate.return_value = "Summary."
        tc = MagicMock()
        tc.count_tokens.return_value = 30
        ms = MagicMock()
        ms.extract_and_dedup = AsyncMock(return_value=[])

        engine = CompactionEngine(db, llm, tc, ms, cfg)

        messages = [_make_message(i, tokens=100) for i in range(8)]
        state = _make_source_state(total_tokens=140_000, compaction_count=0)
        engine._get_active_messages = AsyncMock(return_value=messages)
        engine._get_source_state = AsyncMock(return_value=state)
        engine._execute_compaction = AsyncMock()

        await engine.run_compaction("src-1", None, force_all=True)

        call_args = engine._execute_compaction.call_args
        compactable = call_args[0][2]
        assert len(compactable) == 8  # ALL messages

    @pytest.mark.asyncio
    async def test_extraction_runs_before_summarization(self) -> None:
        """Verify memory extraction is called before LLM summarization."""
        cfg = _make_config()
        cfg.protected_message_count = 2

        db = MagicMock()
        llm = AsyncMock()
        tc = MagicMock()
        tc.count_tokens.return_value = 50
        ms = MagicMock()

        call_order: list[str] = []

        async def mock_extract(*args, **kwargs):
            call_order.append("extract")
            return []

        async def mock_generate(*args, **kwargs):
            call_order.append("summarize")
            return "Summary."

        ms.extract_and_dedup = mock_extract
        llm.generate = mock_generate

        engine = CompactionEngine(db, llm, tc, ms, cfg)

        messages = [_make_message(i) for i in range(5)]
        state = _make_source_state(total_tokens=140_000, compaction_count=0)
        engine._get_active_messages = AsyncMock(return_value=messages)
        engine._get_source_state = AsyncMock(return_value=state)
        engine._execute_compaction = AsyncMock()

        await engine.run_compaction("src-1", None)

        assert call_order == ["extract", "summarize"]

    @pytest.mark.asyncio
    async def test_extraction_skipped_when_disabled(self) -> None:
        cfg = _make_config()
        cfg.protected_message_count = 2
        cfg.extract_memories_on_compaction = False

        db = MagicMock()
        llm = AsyncMock()
        llm.generate.return_value = "Summary."
        tc = MagicMock()
        tc.count_tokens.return_value = 50
        ms = MagicMock()
        ms.extract_and_dedup = AsyncMock()

        engine = CompactionEngine(db, llm, tc, ms, cfg)

        messages = [_make_message(i) for i in range(5)]
        state = _make_source_state(total_tokens=140_000, compaction_count=0)
        engine._get_active_messages = AsyncMock(return_value=messages)
        engine._get_source_state = AsyncMock(return_value=state)
        engine._execute_compaction = AsyncMock()

        await engine.run_compaction("src-1", None)

        ms.extract_and_dedup.assert_not_called()
