"""Tests for ContextAssembler."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from context_management.context import AssembledContext, ContextAssembler


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.max_context_tokens = 1000
    cfg.output_reserve = 100
    cfg.protected_message_count = 3
    cfg.memory_budget = 200
    cfg.summary_budget = 200
    return cfg


def _make_token_counter(per_call: int = 10) -> MagicMock:
    tc = MagicMock()
    tc.count_tokens.return_value = per_call
    return tc


def _make_message(seq: int, tokens: int = 10, role: str = "user",
                  user_id: str = "alice", content: str = "msg") -> MagicMock:
    msg = MagicMock()
    msg.id = uuid.uuid4()
    msg.role = role
    msg.user_id = user_id if role == "user" else None
    msg.content = content
    msg.token_count = tokens
    msg.sequence_num = seq
    return msg


def _make_memory(content: str = "fact", tokens: int = 10,
                 user_id: str = "alice") -> MagicMock:
    mem = MagicMock()
    mem.id = uuid.uuid4()
    mem.content = content
    mem.token_count = tokens
    mem.attributed_user_id = user_id
    return mem


def _make_summary(batch: int = 1, text: str = "Summary.", tokens: int = 20) -> MagicMock:
    s = MagicMock()
    s.batch_number = batch
    s.summary_text = text
    s.token_count = tokens
    return s


class TestFormatHelpers:
    def test_format_memories_block(self) -> None:
        mem = _make_memory("team uses PostgreSQL")
        result = ContextAssembler._format_memories_block([mem])
        assert "team uses PostgreSQL" in result
        assert "remembered" in result.lower() or "memor" in result.lower()

    def test_format_summaries_block(self) -> None:
        s = _make_summary(text="They discussed databases.")
        result = ContextAssembler._format_summaries_block([s])
        assert "They discussed databases." in result

    def test_format_message_user_with_prefix(self) -> None:
        msg = _make_message(1, content="Hello there")
        result = ContextAssembler._format_message(msg)
        assert result["role"] == "user"
        assert "[alice]:" in result["content"]

    def test_format_message_assistant_no_prefix(self) -> None:
        msg = _make_message(1, role="assistant", content="Hi!")
        result = ContextAssembler._format_message(msg)
        assert result["role"] == "assistant"
        assert "[" not in result["content"]

    def test_format_message_system(self) -> None:
        msg = _make_message(1, role="system", content="System info")
        result = ContextAssembler._format_message(msg)
        assert result["role"] == "system"


class TestAssemble:
    @pytest.mark.asyncio
    async def test_returns_assembled_context(self) -> None:
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Hello")

        assert isinstance(result, AssembledContext)
        assert result.system_prompt == "System"
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_current_message_included_via_recent(self) -> None:
        """Current message is already persisted in DB, so it appears in recent messages."""
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        # Simulate the current message already in DB as the last message
        current = _make_message(1, content="Current msg")
        assembler._get_active_messages = AsyncMock(return_value=[current])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current msg")

        assert len(result.messages) == 1
        assert "Current msg" in result.messages[-1]["content"]
        assert result.messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_recent_messages_included(self) -> None:
        cfg = _make_config()
        cfg.protected_message_count = 2
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        msgs = [_make_message(i, content=f"msg-{i}") for i in range(5)]
        assembler._get_active_messages = AsyncMock(return_value=msgs)
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current")

        # Should include recent messages before the current one
        contents = [m["content"] for m in result.messages]
        assert any("msg-4" in c for c in contents)
        assert any("msg-3" in c for c in contents)

    @pytest.mark.asyncio
    async def test_memories_included_when_present(self) -> None:
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        memories = [_make_memory("PostgreSQL is the DB")]
        ms.get_active = AsyncMock(return_value=memories)

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current")

        # Memories are folded into the enriched system_prompt, not the messages list
        assert "PostgreSQL" in result.system_prompt
        assert all(m["role"] != "system" for m in result.messages)

    @pytest.mark.asyncio
    async def test_summaries_included_when_present(self) -> None:
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        summaries = [_make_summary(text="Earlier they discussed compaction.")]
        assembler._get_summaries = AsyncMock(return_value=summaries)

        result = await assembler.assemble("src-1", None, "System", "Current")

        # Summaries are folded into the enriched system_prompt
        assert "compaction" in result.system_prompt
        assert all(m["role"] != "system" for m in result.messages)

    @pytest.mark.asyncio
    async def test_memory_budget_respected(self) -> None:
        cfg = _make_config()
        cfg.memory_budget = 25  # Only room for ~2 memories at 10 tokens each
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        memories = [_make_memory(f"fact-{i}", tokens=10) for i in range(10)]
        ms.get_active = AsyncMock(return_value=memories)

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current")

        assert result.token_breakdown["memories"] <= 25

    @pytest.mark.asyncio
    async def test_empty_memories_omitted(self) -> None:
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current")

        # No system messages for memories
        assert result.token_breakdown["memories"] == 0

    @pytest.mark.asyncio
    async def test_token_breakdown_populated(self) -> None:
        cfg = _make_config()
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        ms.get_active = AsyncMock(return_value=[])

        assembler = ContextAssembler(db, tc, ms, cfg)
        assembler._get_active_messages = AsyncMock(return_value=[])
        assembler._get_summaries = AsyncMock(return_value=[])

        result = await assembler.assemble("src-1", None, "System", "Current")

        assert "system_prompt" in result.token_breakdown
        assert "current_message" in result.token_breakdown
        assert "memories" in result.token_breakdown
        assert "summaries" in result.token_breakdown
        assert "recent_messages" in result.token_breakdown
        assert "older_messages" in result.token_breakdown

    @pytest.mark.asyncio
    async def test_message_order(self) -> None:
        """system_prompt carries [base] [memories] [summaries]; messages are [older] [recent]."""
        cfg = _make_config()
        cfg.protected_message_count = 2
        tc = _make_token_counter(10)
        db = MagicMock()
        ms = MagicMock()
        memories = [_make_memory("DB is PostgreSQL")]
        ms.get_active = AsyncMock(return_value=memories)

        assembler = ContextAssembler(db, tc, ms, cfg)
        msgs = [_make_message(i, content=f"msg-{i}") for i in range(5)]
        assembler._get_active_messages = AsyncMock(return_value=msgs)
        summaries = [_make_summary(text="Earlier discussion.")]
        assembler._get_summaries = AsyncMock(return_value=summaries)

        result = await assembler.assemble("src-1", None, "System", "Current")

        # Last message should be the last recent message (msg-4)
        assert "msg-4" in result.messages[-1]["content"]
        # System prompt should carry base -> memories -> summaries in that order
        sp = result.system_prompt
        base_idx = sp.index("System")
        memory_idx = sp.index("PostgreSQL")
        summary_idx = sp.index("Earlier discussion")
        assert base_idx < memory_idx < summary_idx
        # No system-role entries leak into messages (Anthropic-incompatible)
        assert all(m["role"] != "system" for m in result.messages)
