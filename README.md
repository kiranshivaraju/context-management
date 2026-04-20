# context-management

Drop-in conversation memory for LLM apps. Keeps long-running chats within a token budget by compacting old turns, extracting durable facts into memory, and reassembling a budget-bounded context on every turn.

---

## Why you'd use this

Long conversations break LLM apps in two ways: context windows overflow, and the model starts forgetting constraints the user set early on. `context-management` solves both with four coordinated mechanisms running off a single Postgres database:

| Problem | Mechanism |
|---|---|
| Active message buffer growing past the model's context window | **Message compaction** — oldest non-protected messages get summarized and marked compacted when the active buffer exceeds `max_context_tokens × compaction_trigger_ratio` |
| Losing durable facts (decisions, constraints, preferences) when old turns are dropped | **Memory extraction** — before compaction discards messages, an LLM pass pulls out the important facts and stores them as discrete memories |
| Memory itself growing unbounded over weeks of use | **Memory consolidation** — when stored memories exceed count or token thresholds, an LLM pass merges and de-duplicates them |
| Deciding what to include in the next prompt | **Context assembly** — allocates the token budget across system prompt, memories, summaries, older messages, and recent messages according to a priority order, and returns a ready-to-send payload |

You call two methods: `on_message()` to process user input and get back an assembled context, and `on_response()` to store the assistant's reply. The library handles everything else.

## What's in the box

- **`MemoryManager`** — the public facade. Full method list:
  - Lifecycle: `initialize()`, `shutdown()`
  - Conversation: `on_message()`, `on_response()`
  - Threads: `start_thread()`, `close_thread()`
  - Memory CRUD: `store_memory()`, `get_memories()`, `delete_memory()`, `compact_memories()`
- **`MemoryConfig`** — Pydantic config with 20+ tunable fields and an `from_env()` classmethod that reads `DATABASE_URL`
- **`AssembledContext`** — dataclass returned by `on_message()`: `.system_prompt` (with memories + summaries folded in), `.messages` (user/assistant turns only), `.total_tokens`, `.token_breakdown`
- **LLM providers** — Anthropic and OpenAI ship built-in; add your own by subclassing `LLMProvider`
- **Alembic migration** — creates 4 tables: `messages`, `memories`, `compaction_summaries`, `source_state`

## Requirements

- Python 3.11+
- PostgreSQL 14+
- An LLM provider API key (Anthropic or OpenAI)

---

## Installing in your service

### 1. Install the package

Not on PyPI yet — install from git:

```bash
uv add "context-management @ git+https://github.com/kiranshivaraju/context-management.git"
# or
pip install "context-management @ git+https://github.com/kiranshivaraju/context-management.git"
```

Pin to a release tag in production: `...@v0.1.0`.

### 2. Set environment variables

```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/your_db
export ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY if you use OpenAI
```

`DATABASE_URL` is read by both the library (`MemoryConfig.from_env()`) and alembic migrations. Use the same URL for both.

### 3. Run the database migration

The library needs 4 tables in your Postgres. They don't create themselves.

> **⚠️ Required step.** Without it, `mm.initialize()` will fail with `relation "messages" does not exist`.

Installing the package provides a `context-management` CLI. Run the migration against the same `DATABASE_URL` your service will use:

```bash
export DATABASE_URL=<your service's Postgres URL>
uv run context-management migrate
# or, with plain pip:
context-management migrate
```

Re-run after every library upgrade that bumps the schema.

Other CLI subcommands:

```bash
context-management current          # show the current revision
context-management history          # show migration history
context-management downgrade -1     # step back one migration
context-management --help           # full reference
```

### 4. Use it in code

```python
import asyncio
from anthropic import AsyncAnthropic
from context_management import MemoryManager, MemoryConfig

async def main():
    config = MemoryConfig.from_env(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        max_context_tokens=180_000,
    )
    mm = MemoryManager(config)
    await mm.initialize()
    claude = AsyncAnthropic()

    # On every inbound user message:
    ctx = await mm.on_message(
        source_id="chat-abc",          # any stable identifier for the conversation
        user_id="alice",               # the human speaking; enables per-user attribution
        message="I want to use Postgres for this project.",
        system_prompt="You are a helpful engineering assistant.",
    )

    # ctx.system_prompt carries your system prompt + active memories + summaries.
    # ctx.messages is user/assistant turns only (Anthropic-compatible).
    resp = await claude.messages.create(
        model="claude-sonnet-4-20250514",
        system=ctx.system_prompt,
        messages=ctx.messages,
        max_tokens=1024,
    )
    answer = resp.content[0].text

    # Persist the assistant reply:
    await mm.on_response(source_id="chat-abc", response=answer)

    await mm.shutdown()

asyncio.run(main())
```

---

## Scoping: `source_id` and `thread_id`

Every piece of state is scoped by `source_id` (required) and optionally `thread_id`. Conversations with different `source_id` values are fully isolated — messages, memories, and summaries never leak between them. Use `source_id` for the top-level conversation (a chat room, a user-agent pair, a ticket ID), and `thread_id` for sub-threads that share the parent's memory pool but have their own message history.

---

## Configuration reference

Every field on `MemoryConfig` with its default:

| Field | Default | What it does |
|---|---|---|
| `database_url` | *required* | SQLAlchemy async URL, e.g. `postgresql+asyncpg://...` |
| `max_context_tokens` | 180_000 | Total tokens you want assembled context (excluding response) to stay under |
| `output_reserve` | 8_000 | Tokens held back for the model's response |
| `compaction_trigger_ratio` | 0.75 | Compact when active messages exceed `max_context_tokens × this` |
| `compaction_target_ratio` | 0.50 | Compact *down to* this ratio of the budget |
| `protected_message_count` | 10 | Most recent N messages never get compacted |
| `memory_budget` | 5_000 | Max tokens of memories to include in any assembled context |
| `summary_budget` | 10_000 | Max tokens of compaction summaries to include |
| `extract_memories_on_compaction` | True | If False, compaction summarizes but doesn't extract discrete memories |
| `max_memories_per_source` | 100 | Oldest memories are evicted past this |
| `enable_memory_compaction` | True | Run LLM-driven memory consolidation when thresholds exceed |
| `memory_compaction_count_threshold` | 50 | Consolidate when active memories exceed this count |
| `memory_compaction_token_threshold` | 4_000 | Consolidate when memories exceed this token total |
| `llm_provider` | `"anthropic"` | `"anthropic"` or `"openai"` |
| `llm_model` | `"claude-sonnet-4-20250514"` | Model used for compaction, extraction, and consolidation calls |
| `compaction_max_output_tokens` | 2000 | Cap on summary length |
| `extraction_max_output_tokens` | 1000 | Cap on memory-extraction output |
| `token_counter_provider` | `"tiktoken"` | Offline tokenizer (`cl100k_base`) |

`MemoryConfig.from_env()` reads `DATABASE_URL` from the environment. Any keyword argument overrides the env-derived value:

```python
config = MemoryConfig.from_env(
    max_context_tokens=40_000,
    protected_message_count=6,
)
```

## How compaction and memory extraction decide to run

On every `on_message()` call the library:

1. Writes the user message to the DB.
2. Checks if the **active (non-compacted) message buffer** exceeds `max_context_tokens × compaction_trigger_ratio`. If so, it:
   - Extracts memories from the messages about to be compacted (if `extract_memories_on_compaction=True`)
   - Summarizes those messages via the LLM
   - Marks them `is_compacted=True`
3. Checks if stored memories exceed `memory_compaction_count_threshold` or `memory_compaction_token_threshold`. If so, consolidates them via LLM.
4. Assembles and returns the context for your next LLM call.

`on_response()` just stores the assistant reply; it does not trigger compaction.

## What the assembled context looks like

`ctx.system_prompt` gets three things concatenated with blank lines between them:

```
[your system prompt]

The following facts have been remembered from previous conversations:
- fact 1 (from alice)
- fact 2 (from bob)
...

Summary of earlier conversation:
[compaction summary 1]

[compaction summary 2]
```

`ctx.messages` contains only user/assistant turns (no system-role entries — Anthropic rejects those inside the messages array). Budget allocation priority:

1. System prompt (non-negotiable)
2. Protected recent messages (last `protected_message_count`)
3. Memories, capped at `memory_budget`
4. Compaction summaries, capped at `summary_budget`, newest-first
5. Older messages fill any remaining budget

---

## Local development

```bash
git clone https://github.com/kiranshivaraju/context-management.git
cd context-management

uv sync --extra dev
uv sync --extra anthropic   # or --extra openai

export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/context_management

# Either the packaged CLI:
uv run context-management migrate

# ...or the raw alembic CLI (uses alembic.ini at the repo root):
uv run alembic upgrade head
```

### Running tests

```bash
uv run pytest                                             # 181 tests, ~0.2s
uv run pytest --cov=context_management --cov-report=html
uv run pytest tests/test_compaction.py                    # one file
```

### Type checking

```bash
uv run mypy context_management/
```

### CI

GitHub Actions runs on every push and PR: mypy, pytest against a Postgres service container, and `uv audit` for dependency security.

## Project layout

```
.
├── context_management/       # library source
│   ├── __init__.py           # MemoryManager facade + public exports
│   ├── cli.py                # context-management CLI (migrations)
│   ├── config.py             # MemoryConfig (Pydantic)
│   ├── enums.py              # MessageRole
│   ├── exceptions.py         # MemoryManagerError and subclasses
│   ├── models.py             # SQLAlchemy ORM (4 tables)
│   ├── db.py                 # async engine + session factory
│   ├── token_counter.py      # tiktoken wrapper
│   ├── llm.py                # LLMProvider ABC + Anthropic/OpenAI impls
│   ├── prompts.py            # all prompt templates
│   ├── compaction.py         # CompactionEngine
│   ├── context.py            # ContextAssembler
│   ├── memory.py             # MemoryStore (CRUD + extraction + consolidation)
│   └── alembic/              # schema migrations (shipped in the wheel)
└── tests/                    # 181 tests, 70% coverage floor
```

## Contributing

All changes need:
1. Tests (unit + integration where relevant)
2. Passing CI (mypy + pytest + audit)
3. ≥70% coverage (enforced by pytest)
4. Type hints on all public methods (enforced by `disallow_untyped_defs`)

## License

Licensed under the [Apache License 2.0](./LICENSE). See [NOTICE](./NOTICE) for attribution.
