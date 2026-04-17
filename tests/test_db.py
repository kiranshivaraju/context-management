"""Tests for DatabaseManager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_management.db import DatabaseManager


def _make_mock_engine() -> MagicMock:
    """Create a mock async engine with proper async context manager for begin()."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()

    @asynccontextmanager
    async def fake_begin():
        yield mock_conn

    mock_engine.begin = fake_begin
    mock_engine.dispose = AsyncMock()
    mock_engine._mock_conn = mock_conn  # expose for assertions
    return mock_engine


class TestDatabaseManagerInit:
    def test_stores_url(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        assert dm._database_url == "postgresql+asyncpg://localhost/test"

    def test_engine_is_none_before_init(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        assert dm._engine is None

    def test_session_factory_is_none_before_init(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        assert dm._session_factory is None


class TestGetSessionBeforeInitialize:
    @pytest.mark.asyncio
    async def test_raises_runtime_error(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            async with dm.get_session():
                pass


class TestInitialize:
    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_creates_engine(self, mock_create_engine: MagicMock) -> None:
        mock_engine = _make_mock_engine()
        mock_create_engine.return_value = mock_engine

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()

        mock_create_engine.assert_called_once()
        call_kwargs = mock_create_engine.call_args
        assert call_kwargs[0][0] == "postgresql+asyncpg://localhost/test"
        assert call_kwargs[1]["pool_size"] == 5
        assert call_kwargs[1]["max_overflow"] == 10

    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_sets_session_factory(self, mock_create_engine: MagicMock) -> None:
        mock_create_engine.return_value = _make_mock_engine()

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()

        assert dm._engine is not None
        assert dm._session_factory is not None

    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_verifies_connection(self, mock_create_engine: MagicMock) -> None:
        mock_engine = _make_mock_engine()
        mock_create_engine.return_value = mock_engine

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()

        mock_engine._mock_conn.execute.assert_called_once()


class TestShutdown:
    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_disposes_engine(self, mock_create_engine: MagicMock) -> None:
        mock_engine = _make_mock_engine()
        mock_create_engine.return_value = mock_engine

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()
        await dm.shutdown()

        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_before_init_is_noop(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.shutdown()  # Should not raise

    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_resets_state(self, mock_create_engine: MagicMock) -> None:
        mock_create_engine.return_value = _make_mock_engine()

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()
        await dm.shutdown()

        assert dm._engine is None
        assert dm._session_factory is None


class TestGetSession:
    @pytest.mark.asyncio
    @patch("context_management.db.create_async_engine")
    async def test_yields_session(self, mock_create_engine: MagicMock) -> None:
        mock_create_engine.return_value = _make_mock_engine()

        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        await dm.initialize()

        async with dm.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self) -> None:
        dm = DatabaseManager("postgresql+asyncpg://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            async with dm.get_session():
                pass
