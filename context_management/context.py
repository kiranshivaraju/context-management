"""Context assembly with budget-aware allocation for Context Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from context_management.config import MemoryConfig
from context_management.db import DatabaseManager
from context_management.memory import MemoryStore
from context_management.models import CompactionSummaryModel, MessageModel
from context_management.token_counter import TokenCounter


@dataclass
class AssembledContext:
    """Result of context assembly, ready for LLM submission."""

    system_prompt: str
    messages: list[dict[str, str]]
    total_tokens: int
    token_breakdown: dict[str, int] = field(default_factory=dict)


class ContextAssembler:
    """Budget-aware context assembly following strict priority order."""

    def __init__(
        self,
        db: DatabaseManager,
        token_counter: TokenCounter,
        memory_store: MemoryStore,
        config: MemoryConfig,
    ) -> None:
        self._db = db
        self._token_counter = token_counter
        self._memory_store = memory_store
        self._config = config

    async def assemble(
        self,
        source_id: str,
        thread_id: str | None,
        system_prompt: str,
        current_message: str,
    ) -> AssembledContext:
        """Assemble context within token budget.

        Priority order:
        1. System prompt + current message (non-negotiable)
        2. Recent messages (protected)
        3. Memories (capped at memory_budget)
        4. Compaction summaries (capped at summary_budget)
        5. Older messages (fill remaining budget)
        """
        budget = self._config.max_context_tokens - self._config.output_reserve
        breakdown: dict[str, int] = {}

        # 1. Non-negotiable: system prompt
        system_tokens = self._token_counter.count_tokens(system_prompt)
        breakdown["system_prompt"] = system_tokens
        breakdown["current_message"] = 0
        budget -= system_tokens

        # 2. Recent messages (last N) — includes the current message if already persisted
        all_messages = await self._get_active_messages(source_id, thread_id)
        protect_count = self._config.protected_message_count
        recent_msgs = all_messages[-protect_count:] if all_messages else []
        older_candidates = all_messages[:-protect_count] if len(all_messages) > protect_count else []

        recent_tokens = sum(m.token_count for m in recent_msgs)
        breakdown["recent_messages"] = recent_tokens
        budget -= recent_tokens

        # 3. Memories (capped)
        memories = await self._memory_store.get_active(source_id)
        memory_block: list[Any] = []
        memory_tokens = 0
        memory_cap = min(self._config.memory_budget, max(budget, 0))
        for m in memories:
            if memory_tokens + m.token_count > memory_cap:
                break
            memory_block.append(m)
            memory_tokens += m.token_count
        breakdown["memories"] = memory_tokens
        budget -= memory_tokens

        # 4. Compaction summaries (capped)
        summaries = await self._get_summaries(source_id, thread_id)
        summary_block: list[Any] = []
        summary_tokens = 0
        summary_cap = min(self._config.summary_budget, max(budget, 0))
        for s in summaries:
            if summary_tokens + s.token_count > summary_cap:
                break
            summary_block.append(s)
            summary_tokens += s.token_count
        breakdown["summaries"] = summary_tokens
        budget -= summary_tokens

        # 5. Older messages (fill remaining budget, newest first)
        older_block: list[Any] = []
        older_tokens = 0
        for m in reversed(older_candidates):  # newest first
            if budget - m.token_count < 0:
                break
            older_block.insert(0, m)  # maintain chronological order
            older_tokens += m.token_count
            budget -= m.token_count
        breakdown["older_messages"] = older_tokens

        # 6. Assemble final message list
        messages: list[dict[str, str]] = []
        if memory_block:
            messages.append({
                "role": "system",
                "content": self._format_memories_block(memory_block),
            })
        if summary_block:
            messages.append({
                "role": "system",
                "content": self._format_summaries_block(summary_block),
            })
        for m in older_block:
            messages.append(self._format_message(m))
        for m in recent_msgs:
            messages.append(self._format_message(m))

        total_tokens = sum(breakdown.values())

        return AssembledContext(
            system_prompt=system_prompt,
            messages=messages,
            total_tokens=total_tokens,
            token_breakdown=breakdown,
        )

    async def _get_active_messages(
        self, source_id: str, thread_id: str | None
    ) -> list[Any]:
        """Load non-compacted messages ordered by sequence_num ASC."""
        async with self._db.get_session() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.source_id == source_id)
                .where(MessageModel.is_compacted.is_(False))
            )
            if thread_id is not None:
                stmt = stmt.where(MessageModel.thread_id == thread_id)
            else:
                stmt = stmt.where(MessageModel.thread_id.is_(None))
            stmt = stmt.order_by(MessageModel.sequence_num.asc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def _get_summaries(
        self, source_id: str, thread_id: str | None
    ) -> list[Any]:
        """Load compaction summaries ordered by batch_number ASC."""
        async with self._db.get_session() as session:
            stmt = (
                select(CompactionSummaryModel)
                .where(CompactionSummaryModel.source_id == source_id)
            )
            if thread_id is not None:
                stmt = stmt.where(CompactionSummaryModel.thread_id == thread_id)
            else:
                stmt = stmt.where(CompactionSummaryModel.thread_id.is_(None))
            stmt = stmt.order_by(CompactionSummaryModel.batch_number.asc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    def _format_memories_block(memories: list[Any]) -> str:
        lines = ["The following facts have been remembered from previous conversations:"]
        for m in memories:
            attribution = f" (from {m.attributed_user_id})" if m.attributed_user_id else ""
            lines.append(f"- {m.content}{attribution}")
        return "\n".join(lines)

    @staticmethod
    def _format_summaries_block(summaries: list[Any]) -> str:
        parts = ["Summary of earlier conversation:"]
        for s in summaries:
            parts.append(s.summary_text)
        return "\n\n".join(parts)

    @staticmethod
    def _format_message(message: Any) -> dict[str, str]:
        if message.role == "user" and message.user_id:
            content = f"[{message.user_id}]: {message.content}"
        else:
            content = message.content
        return {"role": message.role, "content": content}
