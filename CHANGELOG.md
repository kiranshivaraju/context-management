# Changelog

All notable changes to this project are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-20

First tagged release. The library is functionally complete and
verified end-to-end against a real Claude conversation (60-turn tests
across two unrelated domains; both runs kept context bounded and
correctly recalled user-specified constraints from early turns).

### Features

- `MemoryManager` facade with 10 public async methods covering
  the conversation lifecycle, thread spawning, and manual memory CRUD.
- Automatic **message compaction** — oldest non-protected messages are
  summarized and marked compacted when the active buffer exceeds
  `max_context_tokens × compaction_trigger_ratio`.
- **Memory extraction** — durable facts are pulled from messages before
  compaction discards them.
- **Memory consolidation** — when stored memories exceed count or token
  thresholds, an LLM pass merges duplicates and stale entries.
- **Budget-aware context assembly** — `on_message()` returns an
  `AssembledContext` with `system_prompt` (base + memories + summaries)
  and `messages` (user/assistant only, Anthropic-compatible).
- `MemoryConfig.from_env()` reads `DATABASE_URL` so the library and
  alembic migrations share a single env var.
- LLM provider abstraction with built-in support for Anthropic and OpenAI.
- PostgreSQL storage via async SQLAlchemy + asyncpg, with an Alembic
  migration that creates all 4 tables.
- `context-management` CLI (migrate / current / history / downgrade)
  shipped inside the installed wheel — consumers no longer need to
  clone this repo to run migrations.
- 193 tests, 74% line coverage, mypy clean.

### Known limitations

- Not yet published to PyPI — install from the git URL.
- Recall of specific examples mentioned only in deep conversation
  history is best-effort; the library is designed to preserve durable
  constraints and compress exploratory discussion. This is a design
  trade-off, not a bug.
- No metrics/observability instrumentation yet (no built-in Prometheus
  exporter, no latency histograms). Application logs via the standard
  `logging` module.

[0.1.0]: https://github.com/kiranshivaraju/context-management/releases/tag/v0.1.0
