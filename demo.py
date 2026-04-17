"""Demo script for Context Management module.

Requires DATABASE_URL env var, e.g.:
    export DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/context_management
"""

import asyncio

from context_management import MemoryManager, MemoryConfig


async def main() -> None:
    config = MemoryConfig.from_env(max_context_tokens=10_000)

    mm = MemoryManager(config)
    await mm.initialize()
    print("✓ Connected to database\n")

    # --- 1. Conversation flow ---
    print("━" * 50)
    print("  MESSAGE FLOW")
    print("━" * 50)

    ctx = await mm.on_message(
        source_id="demo-app",
        user_id="alice",
        message="Hey team, I think we should use PostgreSQL for our main database.",
        system_prompt="You are a helpful engineering assistant.",
    )
    print(f"[alice]: Hey team, I think we should use PostgreSQL...")
    print(f"  → Context: {ctx.total_tokens} tokens, {len(ctx.messages)} messages")
    print(f"  → Breakdown: {ctx.token_breakdown}\n")

    await mm.on_response(
        source_id="demo-app",
        response="Great choice! PostgreSQL offers strong ACID compliance and excellent JSON support.",
    )
    print("[assistant]: Great choice! PostgreSQL offers strong ACID compliance...\n")

    ctx2 = await mm.on_message(
        source_id="demo-app",
        user_id="bob",
        message="Agreed. Let's also use Redis for caching. Alice, can you set up the schema?",
        system_prompt="You are a helpful engineering assistant.",
    )
    print(f"[bob]: Agreed. Let's also use Redis for caching...")
    print(f"  → Context: {ctx2.total_tokens} tokens, {len(ctx2.messages)} messages\n")

    await mm.on_response(
        source_id="demo-app",
        response="Sounds like a solid plan. Alice on schema, Redis for caching.",
    )
    print("[assistant]: Sounds like a solid plan...\n")

    # --- 2. Manual memory storage ---
    print("━" * 50)
    print("  MANUAL MEMORY STORAGE")
    print("━" * 50)

    mem1 = await mm.store_memory(
        source_id="demo-app",
        content="Team decided to use PostgreSQL as the main database",
        attributed_user_id="alice",
    )
    print(f"Stored: {mem1.content}")

    mem2 = await mm.store_memory(
        source_id="demo-app",
        content="Redis will be used for caching",
        attributed_user_id="bob",
    )
    print(f"Stored: {mem2.content}")

    mem3 = await mm.store_memory(
        source_id="demo-app",
        content="Alice is responsible for database schema setup",
        attributed_user_id="bob",
    )
    print(f"Stored: {mem3.content}\n")

    # --- 3. Retrieve memories ---
    print("━" * 50)
    print("  ACTIVE MEMORIES")
    print("━" * 50)

    memories = await mm.get_memories("demo-app")
    print(f"Found {len(memories)} memories:")
    for m in memories:
        attr = f" (from {m.attributed_user_id})" if m.attributed_user_id else ""
        print(f"  • {m.content}{attr}")
    print()

    # --- 4. New message picks up memories in context ---
    print("━" * 50)
    print("  CONTEXT WITH MEMORIES")
    print("━" * 50)

    ctx3 = await mm.on_message(
        source_id="demo-app",
        user_id="charlie",
        message="I'm new to the project. What database are we using?",
        system_prompt="You are a helpful engineering assistant.",
    )
    print("[charlie]: I'm new to the project. What database are we using?")
    print(f"  → Context: {ctx3.total_tokens} tokens, {len(ctx3.messages)} messages")
    print(f"\nAssembled messages sent to LLM:")
    for i, msg in enumerate(ctx3.messages):
        content = msg["content"]
        if len(content) > 120:
            content = content[:120] + "..."
        print(f"  {i+1}. [{msg['role']}]: {content}")
    print()

    # --- 5. Delete a memory ---
    print("━" * 50)
    print("  DELETE MEMORY")
    print("━" * 50)

    await mm.delete_memory(mem2.id)
    print(f"Deleted: {mem2.content}")

    remaining = await mm.get_memories("demo-app")
    print(f"Remaining memories: {len(remaining)}")
    for m in remaining:
        print(f"  • {m.content}")
    print()

    # --- 6. Source isolation ---
    print("━" * 50)
    print("  SOURCE ISOLATION")
    print("━" * 50)

    await mm.store_memory(
        source_id="other-app",
        content="This memory belongs to a different source",
    )

    demo_memories = await mm.get_memories("demo-app")
    other_memories = await mm.get_memories("other-app")
    print(f"demo-app memories:  {len(demo_memories)}")
    print(f"other-app memories: {len(other_memories)}")
    print("Sources are isolated ✓\n")

    # --- Cleanup ---
    await mm.shutdown()
    print("✓ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
