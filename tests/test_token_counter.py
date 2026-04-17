"""Tests for token counting."""

from __future__ import annotations

import pytest

from context_management.exceptions import TokenCounterError
from context_management.token_counter import (
    TokenCounter,
    TiktokenCounter,
    create_token_counter,
)


class TestTiktokenCounter:
    def test_counts_tokens(self) -> None:
        counter = TiktokenCounter()
        count = counter.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_empty_string_returns_zero(self) -> None:
        counter = TiktokenCounter()
        assert counter.count_tokens("") == 0

    def test_known_token_count(self) -> None:
        counter = TiktokenCounter()
        # "hello" is a single token in cl100k_base
        count = counter.count_tokens("hello")
        assert count >= 1

    def test_long_string(self) -> None:
        counter = TiktokenCounter()
        text = "word " * 10000
        count = counter.count_tokens(text)
        assert count > 1000

    def test_is_token_counter(self) -> None:
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_custom_encoding(self) -> None:
        counter = TiktokenCounter(encoding_name="cl100k_base")
        count = counter.count_tokens("test")
        assert count > 0

    def test_invalid_encoding_raises_error(self) -> None:
        with pytest.raises(TokenCounterError, match="Failed to load"):
            TiktokenCounter(encoding_name="nonexistent_encoding")

    def test_consistent_results(self) -> None:
        counter = TiktokenCounter()
        text = "The quick brown fox jumps over the lazy dog"
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        assert count1 == count2


class TestCreateTokenCounter:
    def test_tiktoken_provider(self) -> None:
        counter = create_token_counter("tiktoken")
        assert isinstance(counter, TiktokenCounter)

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown token counter provider"):
            create_token_counter("unknown")
