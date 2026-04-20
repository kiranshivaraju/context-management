"""Initial schema — all 4 tables with indexes.

Revision ID: 001
Revises:
Create Date: 2026-03-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- source_state ---
    op.create_table(
        "source_state",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", sa.String(512), nullable=False),
        sa.Column("thread_id", sa.String(512), nullable=True),
        sa.Column("total_token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("compaction_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("metadata_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_source_state_source_id", "source_state", ["source_id"])
    op.execute(
        "CREATE UNIQUE INDEX uq_source_state_source_thread "
        "ON source_state (source_id, COALESCE(thread_id, '__main__'))"
    )

    # --- messages ---
    op.create_table(
        "messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", sa.String(512), nullable=False),
        sa.Column("thread_id", sa.String(512), nullable=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("metadata_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("is_compacted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("compaction_batch", sa.Integer, nullable=True),
        sa.Column("sequence_num", sa.Integer, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "idx_messages_source_thread_seq",
        "messages",
        ["source_id", "thread_id", "sequence_num"],
    )
    op.execute(
        "CREATE INDEX idx_messages_active "
        "ON messages (source_id, thread_id) WHERE is_compacted = false"
    )

    # --- compaction_summaries ---
    op.create_table(
        "compaction_summaries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", sa.String(512), nullable=False),
        sa.Column("thread_id", sa.String(512), nullable=True),
        sa.Column("batch_number", sa.Integer, nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("messages_start_seq", sa.Integer, nullable=False),
        sa.Column("messages_end_seq", sa.Integer, nullable=False),
        sa.Column("original_token_count", sa.Integer, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_compaction_source_thread_batch "
        "ON compaction_summaries (source_id, COALESCE(thread_id, '__main__'), batch_number)"
    )

    # --- memories ---
    op.create_table(
        "memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", sa.String(512), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("attributed_user_id", sa.String(255), nullable=True),
        sa.Column("source_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.execute(
        "CREATE INDEX idx_memories_active "
        "ON memories (source_id) WHERE is_active = true"
    )


def downgrade() -> None:
    op.drop_table("memories")
    op.drop_table("compaction_summaries")
    op.drop_table("messages")
    op.drop_table("source_state")
