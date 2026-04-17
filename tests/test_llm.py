"""Tests for LLM provider abstraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_management.exceptions import LLMProviderError
from context_management.llm import (
    AnthropicProvider,
    LLMProvider,
    OpenAIProvider,
    create_llm_provider,
)


class TestAnthropicProvider:
    @patch("context_management.llm.anthropic")
    async def test_generate_returns_text(self, mock_anthropic: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated response")]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider("claude-sonnet-4-20250514")
        result = await provider.generate("system", "user prompt", 1000)

        assert result == "Generated response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            system="system",
            messages=[{"role": "user", "content": "user prompt"}],
            max_tokens=1000,
        )

    @patch("context_management.llm.anthropic")
    async def test_generate_wraps_errors(self, mock_anthropic: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        provider = AnthropicProvider("claude-sonnet-4-20250514")
        with pytest.raises(LLMProviderError, match="Anthropic API error"):
            await provider.generate("system", "user prompt", 1000)

    @patch("context_management.llm.anthropic")
    def test_is_llm_provider(self, mock_anthropic: MagicMock) -> None:
        provider = AnthropicProvider("model")
        assert isinstance(provider, LLMProvider)


class TestOpenAIProvider:
    @patch("context_management.llm.openai")
    async def test_generate_returns_text(self, mock_openai: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="GPT response"))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider("gpt-4o")
        result = await provider.generate("system", "user prompt", 1000)

        assert result == "GPT response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user prompt"},
            ],
            max_tokens=1000,
        )

    @patch("context_management.llm.openai")
    async def test_generate_wraps_errors(self, mock_openai: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Rate limited")

        provider = OpenAIProvider("gpt-4o")
        with pytest.raises(LLMProviderError, match="OpenAI API error"):
            await provider.generate("system", "user prompt", 1000)

    @patch("context_management.llm.openai")
    def test_is_llm_provider(self, mock_openai: MagicMock) -> None:
        provider = OpenAIProvider("model")
        assert isinstance(provider, LLMProvider)


class TestCreateLLMProvider:
    @patch("context_management.llm.anthropic")
    def test_anthropic_provider(self, mock_anthropic: MagicMock) -> None:
        provider = create_llm_provider("anthropic", "claude-sonnet-4-20250514")
        assert isinstance(provider, AnthropicProvider)

    @patch("context_management.llm.openai")
    def test_openai_provider(self, mock_openai: MagicMock) -> None:
        provider = create_llm_provider("openai", "gpt-4o")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider("unknown", "model")
