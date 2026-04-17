"""Test runner — loads test_data.json and plays conversations through MemoryManager.

Shows compaction, memory extraction, summaries, and context assembly in action.
"""

import asyncio
import json
import sys
from pathlib import Path

from context_management import MemoryManager, MemoryConfig

DB_URL = "postgresql+asyncpg://postgres@localhost:5432/context_management"


def load_test_data(path: str = "test_data.json") -> dict:
    with open(Path(__file__).parent / path) as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def print_context(ctx, label: str = "") -> None:
    if label:
        print(f"\n  📦 Context ({label}):")
    else:
        print(f"\n  📦 Context:")
    print(f"     Tokens: {ctx.total_tokens} | Messages: {len(ctx.messages)}")
    print(f"     Breakdown: {ctx.token_breakdown}")


def print_messages(ctx, max_content: int = 100) -> None:
    print(f"  📨 Assembled messages:")
    for i, msg in enumerate(ctx.messages):
        content = msg["content"]
        if len(content) > max_content:
            content = content[:max_content] + "..."
        print(f"     {i+1}. [{msg['role']}] {content}")


async def run_scenario_conversation(mm: MemoryManager, scenario: dict) -> None:
    """Play a conversation from a scenario."""
    source_id = scenario["source_id"]
    system_prompt = "You are a helpful engineering assistant."
    prev_memory_count = 0

    for i, turn in enumerate(scenario["conversation"]):
        if turn["role"] == "user":
            ctx = await mm.on_message(
                source_id=source_id,
                user_id=turn["user_id"],
                message=turn["content"],
                system_prompt=system_prompt,
                thread_id=scenario.get("_active_thread_id"),
            )
            preview = turn["content"][:80]
            print(f"\n  💬 [{turn['user_id']}]: {preview}...")
            print(f"     → {ctx.total_tokens} tokens, {len(ctx.messages)} msgs")

            # Check if compaction happened (token count dropped)
            if ctx.token_breakdown.get("summaries", 0) > 0:
                print(f"     ⚡ Summaries present in context ({ctx.token_breakdown['summaries']} tokens)")
            if ctx.token_breakdown.get("memories", 0) > 0:
                print(f"     🧠 Memories present in context ({ctx.token_breakdown['memories']} tokens)")

            # Check for memory compaction (memory count dropped)
            current_memories = await mm.get_memories(source_id)
            current_memory_count = len(current_memories)
            if prev_memory_count > 0 and current_memory_count < prev_memory_count:
                print(f"     🔄 MEMORY COMPACTION detected! {prev_memory_count} → {current_memory_count} memories")
            elif current_memory_count > prev_memory_count:
                print(f"     📝 Memories: {prev_memory_count} → {current_memory_count} (+{current_memory_count - prev_memory_count} new)")
            prev_memory_count = current_memory_count

        elif turn["role"] == "assistant":
            await mm.on_response(
                source_id=source_id,
                response=turn["content"],
                thread_id=scenario.get("_active_thread_id"),
            )
            preview = turn["content"][:80]
            print(f"  🤖 [assistant]: {preview}...")


async def run_scenario(mm: MemoryManager, scenario: dict) -> None:
    """Run a single test scenario."""
    source_id = scenario["source_id"]

    print_header(f"SCENARIO: {scenario['name']}")
    print(f"  Source: {source_id}")
    print(f"  {scenario['description']}")

    # Inject manual memories first
    if scenario.get("manual_memories"):
        print(f"\n  📝 Storing {len(scenario['manual_memories'])} manual memories...")
        for mem_data in scenario["manual_memories"]:
            mem = await mm.store_memory(
                source_id=source_id,
                content=mem_data["content"],
                attributed_user_id=mem_data.get("attributed_user_id"),
            )
            attr = f" (from {mem.attributed_user_id})" if mem.attributed_user_id else ""
            print(f"     ✓ {mem.content}{attr}")

    # Handle thread scenario
    if "thread" in scenario:
        thread_info = scenario["thread"]
        print(f"\n  🔀 Spawning thread: {thread_info['thread_id']}")
        ctx = await mm.start_thread(
            source_id=source_id,
            thread_id=thread_info["thread_id"],
            user_id=thread_info["spawn_message"]["user_id"],
            message=thread_info["spawn_message"]["content"],
            system_prompt="You are a helpful engineering assistant.",
        )
        print(f"     → Thread context: {ctx.total_tokens} tokens, {len(ctx.messages)} msgs")
        scenario["_active_thread_id"] = thread_info["thread_id"]

    # Play conversation
    await run_scenario_conversation(mm, scenario)

    # Show final state
    print(f"\n  {'─' * 50}")
    print(f"  FINAL STATE for source '{source_id}':")

    # Memories
    memories = await mm.get_memories(source_id)
    print(f"\n  🧠 Active memories ({len(memories)}):")
    for m in memories:
        attr = f" (from {m.attributed_user_id})" if m.attributed_user_id else ""
        print(f"     • {m.content}{attr}")

    # Final context assembly
    ctx = await mm.on_message(
        source_id=source_id,
        user_id="inspector",
        message="Summarize everything we've discussed so far.",
        system_prompt="You are a helpful engineering assistant.",
    )
    print_context(ctx, "final assembly")
    print_messages(ctx)


async def main() -> None:
    data = load_test_data()
    cfg_data = data["config"]

    config = MemoryConfig(
        database_url=DB_URL,
        max_context_tokens=cfg_data["max_context_tokens"],
        compaction_trigger_ratio=cfg_data["compaction_trigger_ratio"],
        compaction_target_ratio=cfg_data["compaction_target_ratio"],
        protected_message_count=cfg_data["protected_message_count"],
        memory_budget=cfg_data["memory_budget"],
        summary_budget=cfg_data["summary_budget"],
        output_reserve=cfg_data["output_reserve"],
        memory_compaction_count_threshold=cfg_data.get("memory_compaction_count_threshold", 50),
        memory_compaction_token_threshold=cfg_data.get("memory_compaction_token_threshold", 4000),
    )

    mm = MemoryManager(config)
    await mm.initialize()
    print("✓ Connected to database")
    print(f"  Config: max_tokens={config.max_context_tokens}, "
          f"trigger={config.compaction_trigger_ratio}, "
          f"target={config.compaction_target_ratio}, "
          f"protected={config.protected_message_count}")
    print(f"  Memory compaction: count_threshold={config.memory_compaction_count_threshold}, "
          f"token_threshold={config.memory_compaction_token_threshold}")

    try:
        for scenario in data["scenarios"]:
            await run_scenario(mm, scenario)

        # Cross-source isolation check
        print_header("SOURCE ISOLATION CHECK")
        for scenario in data["scenarios"]:
            sid = scenario["source_id"]
            mems = await mm.get_memories(sid)
            print(f"  {sid}: {len(mems)} memories")

    finally:
        await mm.shutdown()
        print(f"\n✓ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
