"""Tests for MemoryStore."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_management.memory import MemoryStore


def _make_config(max_memories: int = 100) -> MagicMock:
    cfg = MagicMock()
    cfg.max_memories_per_source = max_memories
    cfg.extraction_max_output_tokens = 1000
    return cfg


def _make_token_counter(count: int = 5) -> MagicMock:
    tc = MagicMock()
    tc.count_tokens.return_value = count
    return tc


def _make_db() -> MagicMock:
    return MagicMock()


class TestParseExtractionJson:
    def test_valid_json_array(self) -> None:
        data = [{"content": "fact", "attributed_user_id": "alice"}]
        result = MemoryStore._parse_extraction_json(json.dumps(data))
        assert result == data

    def test_markdown_fenced_json(self) -> None:
        raw = '```json\n[{"content": "fact", "attributed_user_id": "alice"}]\n```'
        result = MemoryStore._parse_extraction_json(raw)
        assert len(result) == 1
        assert result[0]["content"] == "fact"

    def test_invalid_json_returns_empty(self) -> None:
        result = MemoryStore._parse_extraction_json("not json at all")
        assert result == []

    def test_empty_array(self) -> None:
        result = MemoryStore._parse_extraction_json("[]")
        assert result == []

    def test_non_list_json_returns_empty(self) -> None:
        result = MemoryStore._parse_extraction_json('{"key": "value"}')
        assert result == []

    def test_strips_whitespace(self) -> None:
        raw = '  \n [{"content": "fact", "attributed_user_id": null}]  \n '
        result = MemoryStore._parse_extraction_json(raw)
        assert len(result) == 1

    def test_missing_content_field_skipped(self) -> None:
        raw = json.dumps([
            {"content": "valid", "attributed_user_id": "alice"},
            {"attributed_user_id": "bob"},  # missing content
        ])
        result = MemoryStore._parse_extraction_json(raw)
        assert len(result) == 1
        assert result[0]["content"] == "valid"


class TestParseDedupJson:
    def test_valid_add_action(self) -> None:
        data = [{"action": "ADD", "content": "fact", "attributed_user_id": "alice"}]
        result = MemoryStore._parse_dedup_json(json.dumps(data))
        assert result == data

    def test_valid_update_action(self) -> None:
        mid = str(uuid.uuid4())
        data = [{"action": "UPDATE", "memory_id": mid, "content": "updated"}]
        result = MemoryStore._parse_dedup_json(json.dumps(data))
        assert result[0]["action"] == "UPDATE"

    def test_valid_skip_action(self) -> None:
        data = [{"action": "SKIP", "reason": "duplicate"}]
        result = MemoryStore._parse_dedup_json(json.dumps(data))
        assert result[0]["action"] == "SKIP"

    def test_invalid_json_returns_empty(self) -> None:
        result = MemoryStore._parse_dedup_json("broken")
        assert result == []

    def test_markdown_fenced(self) -> None:
        data = [{"action": "ADD", "content": "fact"}]
        raw = f"```json\n{json.dumps(data)}\n```"
        result = MemoryStore._parse_dedup_json(raw)
        assert len(result) == 1

    def test_missing_action_field_skipped(self) -> None:
        raw = json.dumps([
            {"action": "ADD", "content": "valid"},
            {"content": "no action"},  # missing action
        ])
        result = MemoryStore._parse_dedup_json(raw)
        assert len(result) == 1


class TestExtractAndDedup:
    @pytest.mark.asyncio
    async def test_full_pipeline_add(self) -> None:
        """Test extraction + dedup resulting in ADD action."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)

        # Mock get_active to return no existing memories
        store.get_active = AsyncMock(return_value=[])

        # Mock store to return a fake memory
        fake_memory = MagicMock()
        fake_memory.id = uuid.uuid4()
        fake_memory.content = "team uses PostgreSQL"
        store.store = AsyncMock(return_value=fake_memory)

        # Mock LLM
        llm = AsyncMock()
        # First call: extraction
        extraction_response = json.dumps([
            {"content": "team uses PostgreSQL", "attributed_user_id": "alice"}
        ])
        # Second call: dedup
        dedup_response = json.dumps([
            {"action": "ADD", "content": "team uses PostgreSQL", "attributed_user_id": "alice"}
        ])
        llm.generate.side_effect = [extraction_response, dedup_response]

        messages = [MagicMock(role="user", user_id="alice", content="We use PostgreSQL")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        assert len(result) == 1
        store.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_pipeline_update(self) -> None:
        """Test extraction + dedup resulting in UPDATE action."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)

        existing_memory = MagicMock()
        existing_memory.id = uuid.uuid4()
        existing_memory.content = "team uses MySQL"
        existing_memory.attributed_user_id = "alice"
        store.get_active = AsyncMock(return_value=[existing_memory])
        store.update_content = AsyncMock()

        llm = AsyncMock()
        extraction_response = json.dumps([
            {"content": "team uses PostgreSQL now", "attributed_user_id": "alice"}
        ])
        dedup_response = json.dumps([
            {"action": "UPDATE", "memory_id": str(existing_memory.id),
             "content": "team uses PostgreSQL now", "attributed_user_id": "alice"}
        ])
        llm.generate.side_effect = [extraction_response, dedup_response]

        messages = [MagicMock(role="user", user_id="alice", content="We switched to PostgreSQL")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        store.update_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_returns_empty(self) -> None:
        """If LLM returns empty array, no dedup call needed."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)
        store.get_active = AsyncMock(return_value=[])

        llm = AsyncMock()
        llm.generate.return_value = "[]"

        messages = [MagicMock(role="user", user_id="alice", content="Hello")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        assert result == []
        assert llm.generate.call_count == 1  # only extraction, no dedup

    @pytest.mark.asyncio
    async def test_malformed_extraction_json_returns_empty(self) -> None:
        """Graceful fallback on bad extraction JSON."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)
        store.get_active = AsyncMock(return_value=[])

        llm = AsyncMock()
        llm.generate.return_value = "not json"

        messages = [MagicMock(role="user", user_id="alice", content="Hello")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_dedup_json_falls_back_to_add_all(self) -> None:
        """If dedup JSON fails, treat all candidates as ADD."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)
        store.get_active = AsyncMock(return_value=[])

        fake_memory = MagicMock()
        store.store = AsyncMock(return_value=fake_memory)

        llm = AsyncMock()
        extraction_response = json.dumps([
            {"content": "fact1", "attributed_user_id": "alice"},
            {"content": "fact2", "attributed_user_id": "bob"},
        ])
        llm.generate.side_effect = [extraction_response, "broken json"]

        messages = [MagicMock(role="user", user_id="alice", content="Stuff")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        # Both candidates should be added as fallback
        assert store.store.call_count == 2

    @pytest.mark.asyncio
    async def test_skip_action_does_nothing(self) -> None:
        """SKIP actions should not create or update memories."""
        db = _make_db()
        tc = _make_token_counter()
        cfg = _make_config()

        store = MemoryStore(db, tc, cfg)
        store.get_active = AsyncMock(return_value=[])
        store.store = AsyncMock()

        llm = AsyncMock()
        extraction_response = json.dumps([
            {"content": "known fact", "attributed_user_id": "alice"}
        ])
        dedup_response = json.dumps([
            {"action": "SKIP", "reason": "already stored"}
        ])
        llm.generate.side_effect = [extraction_response, dedup_response]

        messages = [MagicMock(role="user", user_id="alice", content="Stuff")]
        result = await store.extract_and_dedup("src-1", messages, llm)

        store.store.assert_not_called()
        assert result == []
