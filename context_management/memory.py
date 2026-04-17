"""Memory CRUD operations and extraction pipeline for Context Management."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from context_management.config import MemoryConfig
from context_management.db import DatabaseManager
from context_management.models import MemoryModel
from context_management.prompts import (
    MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
    MEMORY_CONSOLIDATION_USER_PROMPT,
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_USER_PROMPT,
    MEMORY_UPDATE_SYSTEM_PROMPT,
    MEMORY_UPDATE_USER_PROMPT,
    format_memories_for_prompt,
    format_messages_for_prompt,
)
from context_management.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class MemoryStore:
    """Handles memory CRUD, eviction, and LLM-based extraction + dedup."""

    def __init__(
        self,
        db: DatabaseManager,
        token_counter: TokenCounter,
        config: MemoryConfig,
    ) -> None:
        self._db = db
        self._token_counter = token_counter
        self._config = config

    async def store(
        self,
        source_id: str,
        content: str,
        attributed_user_id: str | None = None,
        source_message_id: uuid.UUID | None = None,
    ) -> MemoryModel:
        """Store a new memory. Evicts oldest if at cap."""
        token_count = self._token_counter.count_tokens(content)
        async with self._db.get_session() as session:
            await self._evict_oldest_if_at_cap(source_id, session)
            memory = MemoryModel(
                source_id=source_id,
                content=content,
                token_count=token_count,
                attributed_user_id=attributed_user_id,
                source_message_id=source_message_id,
            )
            session.add(memory)
            await session.flush()
            return memory

    async def get_active(self, source_id: str) -> list[MemoryModel]:
        """Return active memories for source_id, ordered by created_at DESC."""
        async with self._db.get_session() as session:
            result = await session.execute(
                select(MemoryModel)
                .where(MemoryModel.source_id == source_id)
                .where(MemoryModel.is_active.is_(True))
                .order_by(MemoryModel.created_at.desc())
            )
            return list(result.scalars().all())

    async def delete(self, memory_id: uuid.UUID) -> None:
        """Soft-delete a memory (set is_active=False)."""
        async with self._db.get_session() as session:
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.id == memory_id)
                .values(is_active=False)
            )

    async def update_content(
        self,
        memory_id: uuid.UUID,
        content: str,
        attributed_user_id: str | None = None,
    ) -> None:
        """Update content for a memory (used by dedup UPDATE action)."""
        token_count = self._token_counter.count_tokens(content)
        async with self._db.get_session() as session:
            values: dict[str, Any] = {
                "content": content,
                "token_count": token_count,
            }
            if attributed_user_id is not None:
                values["attributed_user_id"] = attributed_user_id
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.id == memory_id)
                .values(**values)
            )

    async def extract_and_dedup(
        self,
        source_id: str,
        messages: list[Any],
        llm: Any,
    ) -> list[MemoryModel]:
        """Run the full extraction → dedup pipeline on a batch of messages.

        1. Load existing memories
        2. Ask LLM to extract facts from messages
        3. Ask LLM to dedup against existing memories
        4. Execute ADD/UPDATE/SKIP actions
        """
        existing = await self.get_active(source_id)
        conversation_text = format_messages_for_prompt(messages)
        existing_text = format_memories_for_prompt(existing) if existing else "None"

        # Step 1: Extract candidate facts
        extraction_prompt = MEMORY_EXTRACTION_USER_PROMPT.format(
            existing_memories=existing_text,
            conversation=conversation_text,
        )
        extraction_response = await llm.generate(
            MEMORY_EXTRACTION_SYSTEM_PROMPT,
            extraction_prompt,
            self._config.extraction_max_output_tokens,
        )
        candidates = self._parse_extraction_json(extraction_response)
        if not candidates:
            return []

        # Step 2: Dedup against existing
        new_facts_text = "\n".join(
            f"- {c['content']} (by {c.get('attributed_user_id', 'unknown')})"
            for c in candidates
        )
        dedup_prompt = MEMORY_UPDATE_USER_PROMPT.format(
            existing_memories=existing_text,
            new_facts=new_facts_text,
        )
        dedup_response = await llm.generate(
            MEMORY_UPDATE_SYSTEM_PROMPT,
            dedup_prompt,
            self._config.extraction_max_output_tokens,
        )
        actions = self._parse_dedup_json(dedup_response)

        # Fallback: if dedup parsing fails, treat all candidates as ADD
        if not actions and candidates:
            actions = [
                {"action": "ADD", "content": c["content"],
                 "attributed_user_id": c.get("attributed_user_id")}
                for c in candidates
            ]

        # Step 3: Execute actions
        results: list[MemoryModel] = []
        for action in actions:
            act = action.get("action", "").upper()
            if act == "ADD":
                mem = await self.store(
                    source_id,
                    action["content"],
                    action.get("attributed_user_id"),
                )
                results.append(mem)
            elif act == "UPDATE":
                memory_id = uuid.UUID(action["memory_id"])
                await self.update_content(
                    memory_id,
                    action["content"],
                    action.get("attributed_user_id"),
                )
            # SKIP: do nothing

        return results

    async def should_compact_memories(self, source_id: str) -> bool:
        """Check if memories for source_id exceed compaction thresholds."""
        if not self._config.enable_memory_compaction:
            return False
        active = await self.get_active(source_id)
        if not active:
            return False
        count = len(active)
        total_tokens = sum(m.token_count for m in active)
        return (
            count > self._config.memory_compaction_count_threshold
            or total_tokens > self._config.memory_compaction_token_threshold
        )

    async def compact_memories(
        self, source_id: str, llm: Any
    ) -> list[MemoryModel]:
        """Consolidate active memories via LLM. Returns new consolidated set."""
        active = await self.get_active(source_id)
        if len(active) < 2:
            return active

        memories_text = format_memories_for_prompt(active)
        consolidation_prompt = MEMORY_CONSOLIDATION_USER_PROMPT.format(
            memories=memories_text,
        )
        response = await llm.generate(
            MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
            consolidation_prompt,
            self._config.extraction_max_output_tokens,
        )
        consolidated = self._parse_extraction_json(response)
        if not consolidated:
            logger.warning("Memory consolidation returned empty; leaving unchanged")
            return active

        if len(consolidated) >= len(active):
            logger.warning(
                "Consolidation produced %d memories (>= %d existing); skipping",
                len(consolidated), len(active),
            )
            return active

        return await self._execute_memory_compaction(source_id, active, consolidated)

    async def _execute_memory_compaction(
        self,
        source_id: str,
        old_memories: list[MemoryModel],
        consolidated: list[dict],
    ) -> list[MemoryModel]:
        """Atomic: deactivate old memories, insert consolidated set."""
        old_ids = [m.id for m in old_memories]
        async with self._db.get_session() as session:
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.id.in_(old_ids))
                .values(is_active=False)
            )
            new_memories: list[MemoryModel] = []
            for item in consolidated:
                token_count = self._token_counter.count_tokens(item["content"])
                mem = MemoryModel(
                    source_id=source_id,
                    content=item["content"],
                    token_count=token_count,
                    attributed_user_id=item.get("attributed_user_id"),
                )
                session.add(mem)
                new_memories.append(mem)
            await session.flush()
            return new_memories

    async def _evict_oldest_if_at_cap(
        self, source_id: str, session: Any
    ) -> None:
        """Soft-delete the oldest active memory if at max_memories_per_source."""
        result = await session.execute(
            select(MemoryModel.id)
            .where(MemoryModel.source_id == source_id)
            .where(MemoryModel.is_active.is_(True))
            .order_by(MemoryModel.created_at.asc())
        )
        active_ids = list(result.scalars().all())
        if len(active_ids) >= self._config.max_memories_per_source:
            oldest_id = active_ids[0]
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.id == oldest_id)
                .values(is_active=False)
            )

    @staticmethod
    def _parse_extraction_json(text: str) -> list[dict]:
        """Parse LLM extraction response. Returns [] on failure."""
        cleaned = _strip_markdown_fences(text.strip())
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse extraction JSON: %s", text[:200])
            return []
        if not isinstance(data, list):
            logger.warning("Extraction JSON is not a list: %s", type(data))
            return []
        return [item for item in data if isinstance(item, dict) and "content" in item]

    @staticmethod
    def _parse_dedup_json(text: str) -> list[dict]:
        """Parse LLM dedup response. Returns [] on failure."""
        cleaned = _strip_markdown_fences(text.strip())
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse dedup JSON: %s", text[:200])
            return []
        if not isinstance(data, list):
            logger.warning("Dedup JSON is not a list: %s", type(data))
            return []
        return [item for item in data if isinstance(item, dict) and "action" in item]


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines)
    return text
