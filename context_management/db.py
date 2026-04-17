"""Async database engine and session management for Context Management."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class DatabaseManager:
    """Manages async SQLAlchemy engine and session lifecycle."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Create async engine, verify connectivity, set up session factory."""
        self._engine = create_async_engine(
            self._database_url,
            pool_size=5,
            max_overflow=10,
        )
        async with self._engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def shutdown(self) -> None:
        """Dispose engine and reset state."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a session. Auto-commits on success, rolls back on exception."""
        if self._session_factory is None:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
