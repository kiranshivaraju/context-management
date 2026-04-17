"""Tests for Alembic initial migration script structure."""

from __future__ import annotations

import importlib
import inspect


def _load_migration():
    """Import the migration module."""
    spec = importlib.util.spec_from_file_location(
        "migration_001",
        "alembic/versions/001_initial_schema.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMigrationMetadata:
    def test_revision_id(self) -> None:
        mod = _load_migration()
        assert mod.revision == "001"

    def test_down_revision_is_none(self) -> None:
        mod = _load_migration()
        assert mod.down_revision is None

    def test_has_upgrade_function(self) -> None:
        mod = _load_migration()
        assert callable(mod.upgrade)

    def test_has_downgrade_function(self) -> None:
        mod = _load_migration()
        assert callable(mod.downgrade)


class TestUpgradeContents:
    """Verify the upgrade function creates all expected tables and indexes."""

    def _get_source(self) -> str:
        mod = _load_migration()
        return inspect.getsource(mod.upgrade)

    def test_creates_source_state_table(self) -> None:
        src = self._get_source()
        assert '"source_state"' in src

    def test_creates_messages_table(self) -> None:
        src = self._get_source()
        assert '"messages"' in src

    def test_creates_compaction_summaries_table(self) -> None:
        src = self._get_source()
        assert '"compaction_summaries"' in src

    def test_creates_memories_table(self) -> None:
        src = self._get_source()
        assert '"memories"' in src

    def test_coalesce_unique_index_source_state(self) -> None:
        src = self._get_source()
        assert "uq_source_state_source_thread" in src
        assert "COALESCE(thread_id, '__main__')" in src

    def test_coalesce_unique_index_compaction(self) -> None:
        src = self._get_source()
        assert "uq_compaction_source_thread_batch" in src

    def test_partial_index_messages_active(self) -> None:
        src = self._get_source()
        assert "idx_messages_active" in src
        assert "is_compacted = false" in src

    def test_partial_index_memories_active(self) -> None:
        src = self._get_source()
        assert "idx_memories_active" in src
        assert "is_active = true" in src

    def test_source_state_source_id_index(self) -> None:
        src = self._get_source()
        assert "idx_source_state_source_id" in src

    def test_messages_source_thread_seq_index(self) -> None:
        src = self._get_source()
        assert "idx_messages_source_thread_seq" in src


class TestDowngradeContents:
    """Verify downgrade drops tables in correct order."""

    def _get_source(self) -> str:
        mod = _load_migration()
        return inspect.getsource(mod.downgrade)

    def test_drops_all_tables(self) -> None:
        src = self._get_source()
        assert '"memories"' in src
        assert '"compaction_summaries"' in src
        assert '"messages"' in src
        assert '"source_state"' in src

    def test_drops_in_reverse_order(self) -> None:
        src = self._get_source()
        # memories before compaction_summaries before messages before source_state
        assert src.index('"memories"') < src.index('"compaction_summaries"')
        assert src.index('"compaction_summaries"') < src.index('"messages"')
        assert src.index('"messages"') < src.index('"source_state"')
