"""Tests for the context-management CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from context_management import cli


class TestAlembicConfig:
    def test_raises_when_database_url_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            cli._alembic_config()

    def test_uses_database_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        cfg = cli._alembic_config()
        assert cfg.get_main_option("sqlalchemy.url") == "postgresql+asyncpg://u:p@h/db"

    def test_script_location_points_at_packaged_alembic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        cfg = cli._alembic_config()
        script_location = cfg.get_main_option("script_location")
        assert script_location is not None
        # Must live inside the installed package, not at repo root
        assert "context_management/alembic" in script_location
        # And must actually exist with env.py + versions/
        loc = Path(script_location)
        assert (loc / "env.py").is_file()
        assert (loc / "versions").is_dir()

    def test_raises_when_alembic_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        fake_pkg_file = tmp_path / "fake_init.py"
        fake_pkg_file.touch()
        with patch.object(cli, "__file__", str(fake_pkg_file)):
            with pytest.raises(RuntimeError, match="alembic directory not found"):
                cli._alembic_config()


class TestMainRouting:
    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0

    def test_missing_subcommand_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code != 0

    def test_migrate_dispatches_to_upgrade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        upgrade = MagicMock()
        with patch.object(cli.command, "upgrade", upgrade):
            rc = cli.main(["migrate"])
        assert rc == 0
        assert upgrade.call_count == 1
        assert upgrade.call_args.args[1] == "head"

    def test_migrate_accepts_revision_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        upgrade = MagicMock()
        with patch.object(cli.command, "upgrade", upgrade):
            rc = cli.main(["migrate", "--revision", "001"])
        assert rc == 0
        assert upgrade.call_args.args[1] == "001"

    def test_downgrade_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        downgrade = MagicMock()
        with patch.object(cli.command, "downgrade", downgrade):
            rc = cli.main(["downgrade", "-1"])
        assert rc == 0
        assert downgrade.call_args.args[1] == "-1"

    def test_current_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        current = MagicMock()
        with patch.object(cli.command, "current", current):
            rc = cli.main(["current", "-v"])
        assert rc == 0
        assert current.call_args.kwargs["verbose"] is True

    def test_history_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        history = MagicMock()
        with patch.object(cli.command, "history", history):
            rc = cli.main(["history"])
        assert rc == 0
        assert history.call_args.kwargs["verbose"] is False

    def test_missing_database_url_returns_1_and_prints_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        rc = cli.main(["migrate"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "DATABASE_URL" in err
