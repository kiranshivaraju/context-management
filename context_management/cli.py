"""Command-line entry point for context-management.

Exposes database migrations so consumers never have to clone this repo.
After `pip install context-management`, the `context-management` command is
available with subcommands:

    context-management migrate              # upgrade to head (the usual case)
    context-management migrate -r <rev>     # upgrade to a specific revision
    context-management current              # show current revision
    context-management history              # show migration history
    context-management downgrade <rev>      # downgrade to a revision

All commands read `DATABASE_URL` from the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_config() -> Config:
    """Build an alembic Config pointing at the alembic dir shipped with the package."""
    pkg_root = Path(__file__).resolve().parent
    alembic_dir = pkg_root / "alembic"
    if not alembic_dir.is_dir():
        raise RuntimeError(
            f"alembic directory not found at {alembic_dir}. "
            "The context-management package appears to be misinstalled — "
            "migration scripts should ship with the package."
        )
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it to your Postgres async URL, e.g. "
            "postgresql+asyncpg://user:pass@host:5432/db"
        )
    cfg = Config()
    cfg.set_main_option("script_location", str(alembic_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _cmd_migrate(args: argparse.Namespace) -> None:
    command.upgrade(_alembic_config(), args.revision)


def _cmd_downgrade(args: argparse.Namespace) -> None:
    command.downgrade(_alembic_config(), args.revision)


def _cmd_current(args: argparse.Namespace) -> None:
    command.current(_alembic_config(), verbose=args.verbose)


def _cmd_history(args: argparse.Namespace) -> None:
    command.history(_alembic_config(), verbose=args.verbose)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="context-management",
        description="context-management CLI — database migrations and utilities.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser(
        "migrate", help="Upgrade the database schema (default: to head)"
    )
    p_up.add_argument(
        "-r", "--revision", default="head",
        help="Target revision (default: head)",
    )
    p_up.set_defaults(func=_cmd_migrate)

    p_down = sub.add_parser("downgrade", help="Downgrade the database schema")
    p_down.add_argument("revision", help="Target revision (e.g. -1 or a revision id)")
    p_down.set_defaults(func=_cmd_downgrade)

    p_cur = sub.add_parser("current", help="Show the current database revision")
    p_cur.add_argument("-v", "--verbose", action="store_true")
    p_cur.set_defaults(func=_cmd_current)

    p_hist = sub.add_parser("history", help="Show the migration history")
    p_hist.add_argument("-v", "--verbose", action="store_true")
    p_hist.set_defaults(func=_cmd_history)

    ns = parser.parse_args(argv)
    try:
        ns.func(ns)
        return 0
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
