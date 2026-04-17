"""Tests for SQLAlchemy ORM models."""

from __future__ import annotations

import uuid

from context_management.models import (
    Base,
    CompactionSummaryModel,
    MemoryModel,
    MessageModel,
    SourceStateModel,
)


class TestSourceStateModel:
    def test_tablename(self) -> None:
        assert SourceStateModel.__tablename__ == "source_state"

    def test_instantiation_with_explicit_values(self) -> None:
        state = SourceStateModel(
            source_id="src-1",
            thread_id=None,
            total_token_count=0,
            compaction_count=0,
        )
        assert state.source_id == "src-1"
        assert state.thread_id is None
        assert state.total_token_count == 0
        assert state.compaction_count == 0
        assert state.closed_at is None

    def test_with_thread_id(self) -> None:
        state = SourceStateModel(
            source_id="src-1",
            thread_id="thread-1",
        )
        assert state.thread_id == "thread-1"

    def test_columns_exist(self) -> None:
        table = SourceStateModel.__table__
        column_names = {c.name for c in table.columns}
        expected = {
            "id", "source_id", "thread_id", "total_token_count",
            "compaction_count", "metadata_json", "closed_at",
            "created_at", "updated_at",
        }
        assert expected.issubset(column_names)

    def test_source_id_not_nullable(self) -> None:
        col = SourceStateModel.__table__.c.source_id
        assert col.nullable is False

    def test_thread_id_nullable(self) -> None:
        col = SourceStateModel.__table__.c.thread_id
        assert col.nullable is True


class TestMessageModel:
    def test_tablename(self) -> None:
        assert MessageModel.__tablename__ == "messages"

    def test_instantiation(self) -> None:
        msg = MessageModel(
            source_id="src-1",
            role="user",
            user_id="alice",
            content="Hello",
            token_count=5,
            sequence_num=1,
        )
        assert msg.source_id == "src-1"
        assert msg.role == "user"
        assert msg.user_id == "alice"
        assert msg.content == "Hello"
        assert msg.token_count == 5
        assert msg.sequence_num == 1

    def test_assistant_message(self) -> None:
        msg = MessageModel(
            source_id="src-1",
            role="assistant",
            content="Hi!",
            token_count=3,
            sequence_num=2,
        )
        assert msg.role == "assistant"
        assert msg.user_id is None

    def test_columns_exist(self) -> None:
        table = MessageModel.__table__
        column_names = {c.name for c in table.columns}
        expected = {
            "id", "source_id", "thread_id", "role", "user_id",
            "content", "token_count", "metadata_json", "is_compacted",
            "compaction_batch", "sequence_num", "created_at",
        }
        assert expected.issubset(column_names)

    def test_is_compacted_default(self) -> None:
        col = MessageModel.__table__.c.is_compacted
        assert col.default is not None
        assert col.default.arg is False


class TestCompactionSummaryModel:
    def test_tablename(self) -> None:
        assert CompactionSummaryModel.__tablename__ == "compaction_summaries"

    def test_instantiation(self) -> None:
        summary = CompactionSummaryModel(
            source_id="src-1",
            batch_number=1,
            summary_text="Discussion about databases.",
            token_count=10,
            messages_start_seq=1,
            messages_end_seq=5,
            original_token_count=500,
        )
        assert summary.source_id == "src-1"
        assert summary.batch_number == 1
        assert summary.summary_text == "Discussion about databases."
        assert summary.messages_start_seq == 1
        assert summary.messages_end_seq == 5
        assert summary.original_token_count == 500

    def test_columns_exist(self) -> None:
        table = CompactionSummaryModel.__table__
        column_names = {c.name for c in table.columns}
        expected = {
            "id", "source_id", "thread_id", "batch_number",
            "summary_text", "token_count", "messages_start_seq",
            "messages_end_seq", "original_token_count", "created_at",
        }
        assert expected.issubset(column_names)


class TestMemoryModel:
    def test_tablename(self) -> None:
        assert MemoryModel.__tablename__ == "memories"

    def test_instantiation(self) -> None:
        mem = MemoryModel(
            source_id="src-1",
            content="Team uses PostgreSQL",
            token_count=5,
            attributed_user_id="alice",
        )
        assert mem.source_id == "src-1"
        assert mem.content == "Team uses PostgreSQL"
        assert mem.token_count == 5
        assert mem.attributed_user_id == "alice"
        assert mem.source_message_id is None

    def test_without_attribution(self) -> None:
        mem = MemoryModel(
            source_id="src-1",
            content="A fact",
            token_count=3,
        )
        assert mem.attributed_user_id is None

    def test_is_active_default(self) -> None:
        col = MemoryModel.__table__.c.is_active
        assert col.default is not None
        assert col.default.arg is True

    def test_columns_exist(self) -> None:
        table = MemoryModel.__table__
        column_names = {c.name for c in table.columns}
        expected = {
            "id", "source_id", "content", "token_count",
            "attributed_user_id", "source_message_id", "is_active",
            "created_at", "updated_at",
        }
        assert expected.issubset(column_names)


class TestBase:
    def test_base_is_declarative(self) -> None:
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")

    def test_all_tables_registered(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        assert "source_state" in table_names
        assert "messages" in table_names
        assert "compaction_summaries" in table_names
        assert "memories" in table_names
