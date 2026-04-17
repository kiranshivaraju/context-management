"""Configuration for Context Management."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, field_validator


class MemoryConfig(BaseModel):
    """All configuration and thresholds for the Context Management module."""

    # --- Database ---
    database_url: str

    # --- Context window ---
    max_context_tokens: int = 180_000

    # --- Compaction thresholds ---
    compaction_trigger_ratio: float = 0.75
    compaction_target_ratio: float = 0.50
    protected_message_count: int = 10

    # --- Token budget allocation ---
    memory_budget: int = 5_000
    summary_budget: int = 10_000
    output_reserve: int = 8_000

    # --- Memory ---
    extract_memories_on_compaction: bool = True
    max_memories_per_source: int = 100

    # --- Memory compaction ---
    enable_memory_compaction: bool = True
    memory_compaction_count_threshold: int = 50
    memory_compaction_token_threshold: int = 4_000

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    compaction_max_output_tokens: int = 2000
    extraction_max_output_tokens: int = 1000

    # --- Token counter ---
    token_counter_provider: str = "tiktoken"

    @field_validator("compaction_trigger_ratio")
    @classmethod
    def validate_trigger_ratio(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("compaction_trigger_ratio must be between 0 and 1 (exclusive)")
        return v

    @field_validator("compaction_target_ratio")
    @classmethod
    def validate_target_ratio(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        if not 0 < v < 1:
            raise ValueError("compaction_target_ratio must be between 0 and 1 (exclusive)")
        trigger = info.data.get("compaction_trigger_ratio", 0.75)
        if v >= trigger:
            raise ValueError(
                "compaction_target_ratio must be less than compaction_trigger_ratio"
            )
        return v

    @field_validator("protected_message_count")
    @classmethod
    def validate_protected_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("protected_message_count must be >= 1")
        return v

    @field_validator("max_memories_per_source")
    @classmethod
    def validate_max_memories(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_memories_per_source must be >= 1")
        return v

    @field_validator("memory_compaction_count_threshold")
    @classmethod
    def validate_memory_compaction_count(cls, v: int) -> int:
        if v < 2:
            raise ValueError("memory_compaction_count_threshold must be >= 2")
        return v

    @field_validator("memory_compaction_token_threshold")
    @classmethod
    def validate_memory_compaction_tokens(cls, v: int) -> int:
        if v < 100:
            raise ValueError("memory_compaction_token_threshold must be >= 100")
        return v

    @field_validator("max_context_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1000:
            raise ValueError("max_context_tokens must be >= 1000")
        return v

    @classmethod
    def from_env(cls, **overrides: Any) -> "MemoryConfig":
        """Build config from environment. Reads DATABASE_URL (required).

        Any keyword argument overrides the env-derived value.
        """
        if "database_url" not in overrides:
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise ValueError(
                    "DATABASE_URL environment variable is not set. "
                    "Either export DATABASE_URL or pass database_url=... explicitly."
                )
            overrides["database_url"] = url
        return cls(**overrides)
