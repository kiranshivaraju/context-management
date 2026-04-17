"""Prompt templates and formatting helpers for Context Management."""

from __future__ import annotations

from typing import Any

# --- Compaction ---

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your job is to create a concise summary of a \
conversation segment that preserves all information needed to continue the conversation \
without losing context."""

COMPACTION_USER_PROMPT = """\
Summarize the following conversation segment.

RULES:
- Attribute statements to specific users by their ID (e.g., "alice asked about...", \
"bob decided...")
- Preserve all decisions made and their rationale
- Preserve action items and who owns them
- Preserve important facts, numbers, and technical details shared
- Preserve any unresolved questions or open threads
- Use past tense
- Be concise but do NOT drop information that would be needed to continue this conversation

STRUCTURE YOUR SUMMARY AS:
1. **Discussion**: What was discussed and by whom
2. **Decisions**: What was decided (if anything)
3. **Action items**: What needs to be done and by whom (if any)
4. **Open questions**: Unresolved topics (if any)
5. **Current state**: Where things stand at the end of this segment

CONVERSATION:
{conversation}"""

# --- Memory Extraction ---

MEMORY_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extractor. Your job is to identify facts from a conversation \
segment that would be useful to remember for future conversations with this group/user."""

MEMORY_EXTRACTION_USER_PROMPT = """\
The following conversation segment is about to be summarized and compressed. \
Extract any facts worth remembering as discrete, standalone memories.

Only extract from USER messages, not from assistant responses.

STORE if the information:
- Will likely be true for months or years (preferences, decisions, facts about the team/project)
- Is a decision that was made ("We'll use PostgreSQL", "Ship date is March 15")
- Is a stated preference ("Always format SQL with uppercase keywords")
- Is a fact about the project/team ("API endpoint is api.example.com", "Alice owns the frontend")
- Is a role or responsibility ("Bob handles deployments")

DO NOT store:
- Temporary states ("I'm busy today", "Running late")
- Opinions without decisions ("Maybe we should try Redis")
- Greetings, small talk, or filler
- Information from assistant messages
- Anything vague or speculative

EXISTING MEMORIES (do not duplicate these):
{existing_memories}

CONVERSATION SEGMENT TO EXTRACT FROM:
{conversation}

Return a JSON array of objects. Each object has:
- "content": the fact to remember (concise, standalone sentence)
- "attributed_user_id": who stated this

Return an empty array [] if nothing is worth storing.

RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT."""

# --- Memory Update / Dedup ---

MEMORY_UPDATE_SYSTEM_PROMPT = """\
You are a memory manager. Compare new facts against existing stored memories and \
decide what action to take for each."""

MEMORY_UPDATE_USER_PROMPT = """\
Compare these new facts against existing memories and decide what to do.

EXISTING MEMORIES:
{existing_memories}

NEW FACTS:
{new_facts}

For each new fact, decide:
- ADD: New information not already stored
- UPDATE: Replaces or refines an existing memory (provide the memory ID to update)
- SKIP: Already stored or not worth keeping

Return a JSON array of objects:
[
  {{"action": "ADD", "content": "the fact", "attributed_user_id": "user_id"}},
  {{"action": "UPDATE", "memory_id": "uuid", "content": "updated fact", "attributed_user_id": "user_id"}},
  {{"action": "SKIP", "reason": "already stored"}}
]

Return an empty array [] if no changes needed.

RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT."""

# --- Memory Consolidation ---

MEMORY_CONSOLIDATION_SYSTEM_PROMPT = """\
You are a memory consolidator. Your job is to take a set of stored facts and \
produce a smaller, cleaner set that preserves all important information while \
merging related facts and removing redundancy."""

MEMORY_CONSOLIDATION_USER_PROMPT = """\
Consolidate the following stored memories into a smaller set.

RULES:
- Merge related facts into single, comprehensive statements
- Drop facts that are redundant (already covered by another fact)
- Drop facts that appear stale or superseded by newer facts
- Preserve all important decisions, preferences, roles, and project facts
- Keep attributed_user_id when facts come from a single user
- Use "multiple" as attributed_user_id when merging facts from different users
- Each consolidated fact must be a concise, standalone sentence
- Aim to reduce the total count meaningfully (at least 30% reduction)

CURRENT MEMORIES:
{memories}

Return a JSON array of objects. Each object has:
- "content": the consolidated fact (concise, standalone sentence)
- "attributed_user_id": who stated this (or "multiple" if merged from different users)

RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT."""

# --- Thread Spawn ---

THREAD_SPAWN_SYSTEM_PROMPT = """\
You are a conversation summarizer. Create a brief context summary that will serve as \
the starting point for a focused sub-conversation branching off from a main thread."""

THREAD_SPAWN_USER_PROMPT = """\
A user is starting a focused sub-thread from the main conversation below. \
Summarize the relevant context they'll need.

The user's sub-thread topic: {thread_message}

MAIN CONVERSATION:
{parent_context}

Write a concise summary (max 500 words) covering:
1. Key context relevant to the sub-thread topic
2. Decisions already made that are relevant
3. Any constraints or requirements mentioned

Focus on what's relevant to the sub-thread topic. Skip unrelated parts of the conversation."""


# --- Formatting Helpers ---


def format_messages_for_prompt(messages: list[Any]) -> str:
    """Format message records into a conversation string for LLM prompts.

    Each message object must have: role, content, user_id (optional).

    Output format per line:
      [user_id]: content     (for user messages with user_id)
      [user]: content        (for user messages without user_id)
      [assistant]: content   (for assistant messages)
      [system]: content      (for system messages)
    """
    if not messages:
        return ""
    lines = []
    for msg in messages:
        if msg.role == "user" and msg.user_id:
            lines.append(f"[{msg.user_id}]: {msg.content}")
        else:
            lines.append(f"[{msg.role}]: {msg.content}")
    return "\n".join(lines)


def format_memories_for_prompt(memories: list[Any]) -> str:
    """Format memory records into a numbered list for LLM prompts.

    Each memory object must have: id, content, attributed_user_id (optional).

    Output format:
      1. [memory_id] content (attributed to user_id)
      2. [memory_id] content
    """
    if not memories:
        return ""
    lines = []
    for i, mem in enumerate(memories, 1):
        entry = f"{i}. [{mem.id}] {mem.content}"
        if mem.attributed_user_id:
            entry += f" (attributed to {mem.attributed_user_id})"
        lines.append(entry)
    return "\n".join(lines)
