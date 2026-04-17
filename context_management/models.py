"""SQLAlchemy ORM models for Context Management."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class SourceStateModel(Base):
    """Tracking row per (source_id, thread_id) pair."""

    __tablename__ = "source_state"
    __table_args__ = (
        # COALESCE-based unique index created in Alembic migration
        Index("idx_source_state_source_id", "source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    total_token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    compaction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class MessageModel(Base):
    """Every message received by the module."""

    __tablename__ = "messages"
    __table_args__ = (
        Index(
            "idx_messages_source_thread_seq",
            "source_id",
            "thread_id",
            "sequence_num",
        ),
        Index(
            "idx_messages_active",
            "source_id",
            "thread_id",
            postgresql_where=text("is_compacted = false"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    is_compacted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    compaction_batch: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    sequence_num: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class CompactionSummaryModel(Base):
    """LLM-generated summary replacing a batch of compacted messages."""

    __tablename__ = "compaction_summaries"
    __table_args__ = (
        # COALESCE-based unique index created in Alembic migration
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    batch_number: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    messages_start_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    messages_end_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    original_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryModel(Base):
    """Discrete facts extracted or manually stored. Scoped to source_id."""

    __tablename__ = "memories"
    __table_args__ = (
        Index(
            "idx_memories_active",
            "source_id",
            postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    attributed_user_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
