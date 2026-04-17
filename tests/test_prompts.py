"""Tests for prompt templates and helper functions."""

from __future__ import annotations

from context_management.prompts import (
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_USER_PROMPT,
    MEMORY_UPDATE_SYSTEM_PROMPT,
    MEMORY_UPDATE_USER_PROMPT,
    THREAD_SPAWN_SYSTEM_PROMPT,
    THREAD_SPAWN_USER_PROMPT,
    format_messages_for_prompt,
    format_memories_for_prompt,
)


class TestPromptConstants:
    def test_all_prompts_non_empty(self) -> None:
        prompts = [
            COMPACTION_SYSTEM_PROMPT,
            COMPACTION_USER_PROMPT,
            MEMORY_EXTRACTION_SYSTEM_PROMPT,
            MEMORY_EXTRACTION_USER_PROMPT,
            MEMORY_UPDATE_SYSTEM_PROMPT,
            MEMORY_UPDATE_USER_PROMPT,
            THREAD_SPAWN_SYSTEM_PROMPT,
            THREAD_SPAWN_USER_PROMPT,
        ]
        for p in prompts:
            assert len(p.strip()) > 0

    def test_compaction_prompt_placeholders(self) -> None:
        assert "{conversation}" in COMPACTION_USER_PROMPT

    def test_extraction_prompt_placeholders(self) -> None:
        assert "{existing_memories}" in MEMORY_EXTRACTION_USER_PROMPT
        assert "{conversation}" in MEMORY_EXTRACTION_USER_PROMPT

    def test_update_prompt_placeholders(self) -> None:
        assert "{existing_memories}" in MEMORY_UPDATE_USER_PROMPT
        assert "{new_facts}" in MEMORY_UPDATE_USER_PROMPT

    def test_thread_spawn_prompt_placeholders(self) -> None:
        assert "{thread_message}" in THREAD_SPAWN_USER_PROMPT
        assert "{parent_context}" in THREAD_SPAWN_USER_PROMPT

    def test_compaction_format(self) -> None:
        result = COMPACTION_USER_PROMPT.format(conversation="Hello world")
        assert "Hello world" in result

    def test_extraction_format(self) -> None:
        result = MEMORY_EXTRACTION_USER_PROMPT.format(
            existing_memories="- fact 1",
            conversation="Alice: we use PostgreSQL",
        )
        assert "fact 1" in result
        assert "PostgreSQL" in result

    def test_update_format(self) -> None:
        result = MEMORY_UPDATE_USER_PROMPT.format(
            existing_memories="1. old fact",
            new_facts="new fact",
        )
        assert "old fact" in result
        assert "new fact" in result

    def test_thread_spawn_format(self) -> None:
        result = THREAD_SPAWN_USER_PROMPT.format(
            thread_message="database design",
            parent_context="We discussed the project",
        )
        assert "database design" in result
        assert "discussed the project" in result


class TestFormatMessagesForPrompt:
    def test_user_message(self) -> None:
        messages = [_make_msg("user", "Hello!", user_id="alice")]
        result = format_messages_for_prompt(messages)
        assert "[alice]: Hello!" in result

    def test_assistant_message(self) -> None:
        messages = [_make_msg("assistant", "Hi there!")]
        result = format_messages_for_prompt(messages)
        assert "[assistant]: Hi there!" in result

    def test_system_message(self) -> None:
        messages = [_make_msg("system", "Context info")]
        result = format_messages_for_prompt(messages)
        assert "[system]: Context info" in result

    def test_multiple_messages(self) -> None:
        messages = [
            _make_msg("user", "Question?", user_id="bob"),
            _make_msg("assistant", "Answer."),
            _make_msg("user", "Thanks!", user_id="bob"),
        ]
        result = format_messages_for_prompt(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "[bob]: Question?" in lines[0]
        assert "[assistant]: Answer." in lines[1]

    def test_empty_list(self) -> None:
        result = format_messages_for_prompt([])
        assert result == ""

    def test_user_message_no_user_id(self) -> None:
        messages = [_make_msg("user", "Hello!")]
        result = format_messages_for_prompt(messages)
        assert "[user]: Hello!" in result


class TestFormatMemoriesForPrompt:
    def test_single_memory(self) -> None:
        memories = [_make_mem("mem-1", "Team uses PostgreSQL", "alice")]
        result = format_memories_for_prompt(memories)
        assert "mem-1" in result
        assert "Team uses PostgreSQL" in result
        assert "alice" in result

    def test_multiple_memories(self) -> None:
        memories = [
            _make_mem("mem-1", "Fact one", "alice"),
            _make_mem("mem-2", "Fact two", None),
        ]
        result = format_memories_for_prompt(memories)
        assert "1." in result
        assert "2." in result
        assert "mem-1" in result
        assert "mem-2" in result

    def test_empty_list(self) -> None:
        result = format_memories_for_prompt([])
        assert result == ""

    def test_memory_without_attribution(self) -> None:
        memories = [_make_mem("mem-1", "Some fact", None)]
        result = format_memories_for_prompt(memories)
        assert "mem-1" in result
        assert "Some fact" in result


# --- Helpers to create stub objects for testing ---


class _StubMessage:
    """Minimal stub matching MessageModel interface for prompt formatting."""

    def __init__(self, role: str, content: str, user_id: str | None = None) -> None:
        self.role = role
        self.content = content
        self.user_id = user_id


class _StubMemory:
    """Minimal stub matching MemoryModel interface for prompt formatting."""

    def __init__(self, id: str, content: str, attributed_user_id: str | None = None) -> None:
        self.id = id
        self.content = content
        self.attributed_user_id = attributed_user_id


def _make_msg(role: str, content: str, user_id: str | None = None) -> _StubMessage:
    return _StubMessage(role=role, content=content, user_id=user_id)


def _make_mem(id: str, content: str, attributed_user_id: str | None = None) -> _StubMemory:
    return _StubMemory(id=id, content=content, attributed_user_id=attributed_user_id)
