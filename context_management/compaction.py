"""Compaction engine for Context Management."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text, update

from context_management.config import MemoryConfig
from context_management.db import DatabaseManager
from context_management.memory import MemoryStore
from context_management.models import CompactionSummaryModel, MessageModel, SourceStateModel
from context_management.prompts import (
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    format_messages_for_prompt,
)
from context_management.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class CompactionEngine:
    """Orchestrates memory extraction -> summarization -> marking."""

    def __init__(
        self,
        db: DatabaseManager,
        llm: Any,
        token_counter: TokenCounter,
        memory_store: MemoryStore,
        config: MemoryConfig,
    ) -> None:
        self._db = db
        self._llm = llm
        self._token_counter = token_counter
        self._memory_store = memory_store
        self._config = config

    async def should_compact(self, source_id: str, thread_id: str | None) -> bool:
        """Trigger on the active (non-compacted) message buffer only.

        Summaries sit in separate storage and are budgeted separately during
        context assembly; counting them toward the trigger would make
        compaction fire every turn once accumulated summaries exceed the
        threshold (the summary created by each compaction feeds the next).
        """
        messages = await self._get_active_messages(source_id, thread_id)
        if not messages:
            return False
        active_tokens = sum(m.token_count for m in messages)
        threshold = self._config.max_context_tokens * self._config.compaction_trigger_ratio
        return bool(active_tokens > threshold)

    async def run_compaction(
        self,
        source_id: str,
        thread_id: str | None,
        force_all: bool = False,
    ) -> None:
        """Full compaction flow: extract memories -> summarize -> mark compacted."""
        messages = await self._get_active_messages(source_id, thread_id)
        state = await self._get_source_state(source_id, thread_id)
        if state is None or not messages:
            return

        # Determine compactable vs protected
        if force_all:
            compactable = messages
        else:
            protect_count = self._config.protected_message_count
            if len(messages) <= protect_count:
                return  # nothing to compact
            compactable = messages[:-protect_count]

        if not compactable:
            return

        # Step 1: Extract memories before compacting
        if self._config.extract_memories_on_compaction:
            await self._memory_store.extract_and_dedup(
                source_id, compactable, self._llm
            )

        # Step 2: Summarize
        conversation_text = format_messages_for_prompt(compactable)
        summary_text = await self._llm.generate(
            COMPACTION_SYSTEM_PROMPT,
            COMPACTION_USER_PROMPT.format(conversation=conversation_text),
            self._config.compaction_max_output_tokens,
        )
        summary_tokens = self._token_counter.count_tokens(summary_text)

        # Step 3: Execute in transaction
        await self._execute_compaction(
            source_id, thread_id, compactable, state,
            summary_text, summary_tokens,
        )

    async def _execute_compaction(
        self,
        source_id: str,
        thread_id: str | None,
        compactable: list[Any],
        state: Any,
        summary_text: str,
        summary_tokens: int,
    ) -> None:
        """Atomic: insert summary, mark messages, update source_state."""
        batch_number = state.compaction_count + 1
        compacted_tokens = sum(m.token_count for m in compactable)
        compactable_ids = [m.id for m in compactable]
        start_seq = compactable[0].sequence_num
        end_seq = compactable[-1].sequence_num

        async with self._db.get_session() as session:
            # Insert summary
            summary = CompactionSummaryModel(
                source_id=source_id,
                thread_id=thread_id,
                batch_number=batch_number,
                summary_text=summary_text,
                token_count=summary_tokens,
                messages_start_seq=start_seq,
                messages_end_seq=end_seq,
                original_token_count=compacted_tokens,
            )
            session.add(summary)

            # Mark messages as compacted
            await session.execute(
                update(MessageModel)
                .where(MessageModel.id.in_(compactable_ids))
                .values(is_compacted=True, compaction_batch=batch_number)
            )

            # Update source_state
            new_total = state.total_token_count - compacted_tokens + summary_tokens
            await session.execute(
                update(SourceStateModel)
                .where(SourceStateModel.source_id == source_id)
                .where(
                    SourceStateModel.thread_id == thread_id
                    if thread_id is not None
                    else SourceStateModel.thread_id.is_(None)
                )
                .values(
                    total_token_count=new_total,
                    compaction_count=batch_number,
                )
            )

    async def _get_source_state(
        self, source_id: str, thread_id: str | None
    ) -> Any | None:
        """Load source_state for (source_id, thread_id)."""
        async with self._db.get_session() as session:
            stmt = select(SourceStateModel).where(
                SourceStateModel.source_id == source_id
            )
            if thread_id is not None:
                stmt = stmt.where(SourceStateModel.thread_id == thread_id)
            else:
                stmt = stmt.where(SourceStateModel.thread_id.is_(None))
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

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
