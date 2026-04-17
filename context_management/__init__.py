"""Context Management — Public API facade."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import func, select, update

from context_management.compaction import CompactionEngine
from context_management.config import MemoryConfig
from context_management.context import AssembledContext, ContextAssembler
from context_management.db import DatabaseManager
from context_management.enums import MessageRole
from context_management.exceptions import (
    CompactionError,
    LLMProviderError,
    MemoryManagerError,
    SourceNotFoundError,
    TokenCounterError,
    ValidationError,
)
from context_management.llm import create_llm_provider
from context_management.memory import MemoryStore
from context_management.models import MessageModel, SourceStateModel
from context_management.prompts import (
    THREAD_SPAWN_SYSTEM_PROMPT,
    THREAD_SPAWN_USER_PROMPT,
)
from context_management.token_counter import create_token_counter

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "AssembledContext",
    "MessageRole",
    "MemoryManagerError",
    "SourceNotFoundError",
    "CompactionError",
    "LLMProviderError",
    "TokenCounterError",
    "ValidationError",
]


class MemoryManager:
    """Main facade for the Context Management module."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._initialized = False

        # Create components
        self._token_counter = create_token_counter(config.token_counter_provider)
        self._llm = create_llm_provider(config.llm_provider, config.llm_model)
        self._db = DatabaseManager(config.database_url)
        self._memory_store = MemoryStore(self._db, self._token_counter, config)
        self._compaction_engine = CompactionEngine(
            self._db, self._llm, self._token_counter, self._memory_store, config
        )
        self._context_assembler = ContextAssembler(
            self._db, self._token_counter, self._memory_store, config
        )

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Initialize database connection."""
        await self._db.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Close database connections."""
        await self._db.shutdown()
        self._initialized = False

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

    # --- Validation ---

    @staticmethod
    def _validate_source_id(source_id: str) -> None:
        if not source_id or len(source_id) > 512:
            raise ValidationError("source_id must be non-empty and <= 512 characters")

    @staticmethod
    def _validate_thread_id(thread_id: str | None) -> None:
        if thread_id is not None and (not thread_id or len(thread_id) > 512):
            raise ValidationError("thread_id must be non-empty and <= 512 characters (or None)")

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not user_id or len(user_id) > 255:
            raise ValidationError("user_id must be non-empty and <= 255 characters")

    @staticmethod
    def _validate_content(content: str) -> None:
        if not content:
            raise ValidationError("content must be non-empty")

    # --- Public Methods ---

    async def on_message(
        self,
        source_id: str,
        user_id: str,
        message: str,
        system_prompt: str = "",
        thread_id: str | None = None,
    ) -> AssembledContext:
        """Process an incoming user message and return assembled context."""
        self._check_initialized()
        self._validate_source_id(source_id)
        self._validate_user_id(user_id)
        self._validate_content(message)
        self._validate_thread_id(thread_id)

        token_count = self._token_counter.count_tokens(message)

        async with self._db.get_session() as session:
            # Get or create source_state
            state = await self._get_or_create_source_state(
                session, source_id, thread_id
            )
            if state.closed_at is not None:
                raise ValidationError("Cannot send messages to a closed thread")

            # Assign sequence_num
            seq = await self._next_sequence_num(session, source_id, thread_id)

            # Insert message
            msg = MessageModel(
                source_id=source_id,
                thread_id=thread_id,
                role=MessageRole.USER,
                user_id=user_id,
                content=message,
                token_count=token_count,
                sequence_num=seq,
            )
            session.add(msg)

            # Update token count
            state.total_token_count += token_count
            session.add(state)

        # Check compaction (outside transaction for isolation)
        try:
            if await self._compaction_engine.should_compact(source_id, thread_id):
                await self._compaction_engine.run_compaction(source_id, thread_id)
        except Exception:
            logger.warning(
                "Compaction failed for source=%s thread=%s",
                source_id, thread_id, exc_info=True,
            )

        # Check memory compaction
        await self._maybe_compact_memories(source_id)

        # Assemble context
        return await self._context_assembler.assemble(
            source_id, thread_id, system_prompt, message
        )

    async def on_response(
        self,
        source_id: str,
        response: str,
        thread_id: str | None = None,
    ) -> None:
        """Store an assistant response. Does NOT trigger compaction."""
        self._check_initialized()
        self._validate_source_id(source_id)
        self._validate_content(response)
        self._validate_thread_id(thread_id)

        token_count = self._token_counter.count_tokens(response)

        async with self._db.get_session() as session:
            state = await self._get_source_state(session, source_id, thread_id)
            if state is None:
                raise SourceNotFoundError(
                    f"No source_state for source_id={source_id}, thread_id={thread_id}"
                )
            if state.closed_at is not None:
                raise ValidationError("Cannot send responses to a closed thread")

            seq = await self._next_sequence_num(session, source_id, thread_id)

            msg = MessageModel(
                source_id=source_id,
                thread_id=thread_id,
                role=MessageRole.ASSISTANT,
                content=response,
                token_count=token_count,
                sequence_num=seq,
            )
            session.add(msg)

            state.total_token_count += token_count
            session.add(state)

    async def start_thread(
        self,
        source_id: str,
        thread_id: str,
        user_id: str,
        message: str,
        system_prompt: str = "",
    ) -> AssembledContext:
        """Spawn a sub-thread from the main conversation."""
        self._check_initialized()
        self._validate_source_id(source_id)
        self._validate_thread_id(thread_id)
        self._validate_user_id(user_id)
        self._validate_content(message)

        if thread_id is None:
            raise ValidationError("thread_id is required for start_thread")

        async with self._db.get_session() as session:
            # Check parent (main thread) exists
            parent = await self._get_source_state(session, source_id, None)
            if parent is None:
                raise SourceNotFoundError(
                    f"No main thread for source_id={source_id}"
                )

            # Check thread doesn't exist
            existing = await self._get_source_state(session, source_id, thread_id)
            if existing is not None:
                raise ValidationError(f"Thread {thread_id} already exists")

        # Assemble parent context for LLM summary
        parent_context = await self._context_assembler.assemble(
            source_id, None, system_prompt, message
        )
        parent_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in parent_context.messages
        )

        # Generate thread summary
        summary_text = await self._llm.generate(
            THREAD_SPAWN_SYSTEM_PROMPT,
            THREAD_SPAWN_USER_PROMPT.format(
                thread_message=message,
                parent_context=parent_text,
            ),
            self._config.compaction_max_output_tokens,
        )
        summary_tokens = self._token_counter.count_tokens(summary_text)
        message_tokens = self._token_counter.count_tokens(message)

        # Create thread atomically
        async with self._db.get_session() as session:
            state = SourceStateModel(
                source_id=source_id,
                thread_id=thread_id,
                total_token_count=summary_tokens + message_tokens,
            )
            session.add(state)

            # System message with summary
            sys_msg = MessageModel(
                source_id=source_id,
                thread_id=thread_id,
                role=MessageRole.SYSTEM,
                content=summary_text,
                token_count=summary_tokens,
                sequence_num=1,
            )
            session.add(sys_msg)

            # User message
            user_msg = MessageModel(
                source_id=source_id,
                thread_id=thread_id,
                role=MessageRole.USER,
                user_id=user_id,
                content=message,
                token_count=message_tokens,
                sequence_num=2,
            )
            session.add(user_msg)

        return await self._context_assembler.assemble(
            source_id, thread_id, system_prompt, message
        )

    async def close_thread(
        self,
        source_id: str,
        thread_id: str,
    ) -> None:
        """Close a thread: compact all messages, mark closed."""
        self._check_initialized()
        self._validate_source_id(source_id)
        if thread_id is None:
            raise ValidationError("thread_id is required for close_thread")
        self._validate_thread_id(thread_id)

        async with self._db.get_session() as session:
            state = await self._get_source_state(session, source_id, thread_id)
            if state is None:
                raise SourceNotFoundError(
                    f"No thread {thread_id} for source_id={source_id}"
                )
            if state.closed_at is not None:
                raise ValidationError(f"Thread {thread_id} is already closed")

        # Compact all messages
        await self._compaction_engine.run_compaction(
            source_id, thread_id, force_all=True
        )

        # Mark closed
        async with self._db.get_session() as session:
            await session.execute(
                update(SourceStateModel)
                .where(SourceStateModel.source_id == source_id)
                .where(SourceStateModel.thread_id == thread_id)
                .values(closed_at=func.now())
            )

    async def store_memory(
        self,
        source_id: str,
        content: str,
        attributed_user_id: str | None = None,
    ) -> Any:
        """Manually store a memory."""
        self._check_initialized()
        self._validate_source_id(source_id)
        self._validate_content(content)
        result = await self._memory_store.store(
            source_id, content, attributed_user_id
        )
        await self._maybe_compact_memories(source_id)
        return result

    async def get_memories(self, source_id: str) -> list[Any]:
        """Get all active memories for a source."""
        self._check_initialized()
        self._validate_source_id(source_id)
        return await self._memory_store.get_active(source_id)

    async def delete_memory(self, memory_id: uuid.UUID) -> None:
        """Soft-delete a memory."""
        self._check_initialized()
        await self._memory_store.delete(memory_id)

    async def compact_memories(self, source_id: str) -> list[Any]:
        """Manually trigger memory consolidation for a source."""
        self._check_initialized()
        self._validate_source_id(source_id)
        return await self._memory_store.compact_memories(source_id, self._llm)

    # --- Internal Helpers ---

    async def _maybe_compact_memories(self, source_id: str) -> None:
        """Trigger memory compaction if thresholds exceeded."""
        try:
            if await self._memory_store.should_compact_memories(source_id):
                await self._memory_store.compact_memories(source_id, self._llm)
        except Exception:
            logger.warning(
                "Memory compaction failed for source=%s",
                source_id, exc_info=True,
            )

    async def _get_or_create_source_state(
        self, session: Any, source_id: str, thread_id: str | None
    ) -> SourceStateModel:
        """Get existing or create new source_state."""
        state = await self._get_source_state(session, source_id, thread_id)
        if state is None:
            state = SourceStateModel(
                source_id=source_id,
                thread_id=thread_id,
            )
            session.add(state)
            await session.flush()
        return state

    @staticmethod
    async def _get_source_state(
        session: Any, source_id: str, thread_id: str | None
    ) -> SourceStateModel | None:
        """Load source_state."""
        stmt = select(SourceStateModel).where(
            SourceStateModel.source_id == source_id
        )
        if thread_id is not None:
            stmt = stmt.where(SourceStateModel.thread_id == thread_id)
        else:
            stmt = stmt.where(SourceStateModel.thread_id.is_(None))
        result = await session.execute(stmt)
        state: SourceStateModel | None = result.scalar_one_or_none()
        return state

    @staticmethod
    async def _next_sequence_num(
        session: Any, source_id: str, thread_id: str | None
    ) -> int:
        """Get next sequence number for messages."""
        stmt = select(func.max(MessageModel.sequence_num)).where(
            MessageModel.source_id == source_id
        )
        if thread_id is not None:
            stmt = stmt.where(MessageModel.thread_id == thread_id)
        else:
            stmt = stmt.where(MessageModel.thread_id.is_(None))
        result = await session.execute(stmt)
        max_seq = result.scalar_one_or_none()
        return (max_seq or 0) + 1
