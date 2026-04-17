# Context Management

Platform-agnostic context management module for enterprise agentic systems. Import one module, call `on_message()` and `on_response()` — context management handled.

## What It Does

- **Infinite conversations**: Automatic compaction means conversations never hit context limits
- **Smart memory**: Important facts are automatically extracted before compaction and persist as searchable, discrete memories
- **Thread-aware**: Sub-threads get parent context as a summary, run independently, and share memories back
- **Platform-agnostic**: Works with any chat source via `source_id` + `thread_id`
- **LLM-agnostic**: Abstract provider interface supports Anthropic, OpenAI, and any future provider

## Tech Stack

- **Language:** Python 3.11+
- **Database:** PostgreSQL (via asyncpg + SQLAlchemy 2.0 async)
- **Migrations:** Alembic
- **Token counting:** tiktoken (offline, cl100k_base)
- **Package manager:** uv

## Quick Start

> **Before running this:** set `DATABASE_URL`, set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`), and run migrations against your DB. See [Using This Library in Another Service](#using-this-library-in-another-service).

```python
from context_management import MemoryManager, MemoryConfig

# reads DATABASE_URL from environment
config = MemoryConfig.from_env(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
)

mm = MemoryManager(config)
await mm.initialize()

# Process a user message — returns assembled context ready for LLM
context = await mm.on_message(
    source_id="group-chat-123",
    user_id="alice",
    message="Let's use PostgreSQL for the project",
    system_prompt="You are a helpful assistant.",
)

# Send context.messages to your LLM, get response, then persist it
await mm.on_response(source_id="group-chat-123", response="Sounds good! PostgreSQL is a great choice.")
```

## Using This Library in Another Service

If you're integrating `context-management` into an existing service (not working on the library itself), this is the path.

### 1. Install the package

Not on PyPI yet — install from git:

```bash
uv add "context-management @ git+https://github.com/kiranshivaraju/contenxt-management.git"
# or
pip install "context-management @ git+https://github.com/kiranshivaraju/contenxt-management.git"
```

Pin to a tag for production: `...@v0.1.0`.

### 2. Set the required environment variables

```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/your_db
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY if using OpenAI
```

### 3. Create the database tables (required, one-time + on every library upgrade)

The library needs 4 tables (`messages`, `memories`, `summaries`, `source_state`) in your Postgres. They don't create themselves — you run the migration against your DB.

**⚠️ Do not skip this step. Without it, `mm.initialize()` will fail with `relation "messages" does not exist`.**

Because the alembic config lives inside this repo (not shipped with the installed Python package yet), the current path is to clone this repo once and run alembic from it:

```bash
# one-time setup in your deploy env / CI
git clone https://github.com/kiranshivaraju/contenxt-management.git
cd contenxt-management
export DATABASE_URL=<your service's Postgres URL>
uv sync
uv run alembic upgrade head
```

This creates the tables in **your** service's database (whatever `DATABASE_URL` points at). Re-run after every library upgrade that bumps the schema.

> A `context-management migrate` CLI command is planned so consumers won't need to clone the repo — until then, the clone-and-run approach above is required.

### 4. Use it in code

```python
from context_management import MemoryManager, MemoryConfig

config = MemoryConfig.from_env(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
    max_context_tokens=180_000,
)

mm = MemoryManager(config)
await mm.initialize()

# ... later, on shutdown:
await mm.shutdown()
```

---

## Local Development (working on the library itself)

```bash
git clone https://github.com/kiranshivaraju/contenxt-management.git
cd contenxt-management

uv sync --extra dev
uv sync --extra anthropic   # or --extra openai

export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/context_management
uv run alembic upgrade head
```

## Running Tests

```bash
# Run all tests (requires Docker for testcontainers)
uv run pytest

# Run with coverage report
uv run pytest --cov=context_management --cov-report=html

# Run specific test file
uv run pytest tests/test_compaction.py
```

## Project Structure

```
.
├── context_management/       # Source code (the module)
│   ├── __init__.py       # Public API: MemoryManager + exports
│   ├── config.py         # MemoryConfig (Pydantic)
│   ├── enums.py          # MessageRole enum
│   ├── exceptions.py     # Custom exceptions
│   ├── models.py         # SQLAlchemy ORM models
│   ├── db.py             # Async database engine
│   ├── token_counter.py  # Token counting (tiktoken)
│   ├── llm.py            # LLM provider abstraction
│   ├── prompts.py        # All prompt templates
│   ├── compaction.py     # Compaction engine
│   ├── context.py        # Context assembly
│   └── memory.py         # Memory CRUD + extraction
├── alembic/              # Database migrations
├── tests/                # Test suite
├── product/              # Product documentation
└── sprints/              # Sprint documentation
```

## Contributing

All code changes require:
1. Tests (unit + integration)
2. Passing CI
3. 80% minimum test coverage
4. Type hints on all public methods
