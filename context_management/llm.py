"""LLM provider abstraction for Context Management."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_management.exceptions import LLMProviderError

# Lazy imports — these are set on first use by each provider
anthropic = None
openai = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> str:
        """Send a prompt to the LLM and return the text response.

        Raises LLMProviderError on any failure.
        """


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic SDK."""

    def __init__(self, model: str) -> None:
        global anthropic
        if anthropic is None:
            try:
                import anthropic as _anthropic

                anthropic = _anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: uv add anthropic"
                )
        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def generate(
        self, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_output_tokens,
            )
            return str(response.content[0].text)
        except Exception as e:
            raise LLMProviderError(f"Anthropic API error: {e}") from e


class OpenAIProvider(LLMProvider):
    """LLM provider using the OpenAI SDK."""

    def __init__(self, model: str) -> None:
        global openai
        if openai is None:
            try:
                import openai as _openai

                openai = _openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: uv add openai"
                )
        self._client = openai.AsyncOpenAI()
        self._model = model

    async def generate(
        self, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_output_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("OpenAI returned no content")
            return str(content)
        except Exception as e:
            raise LLMProviderError(f"OpenAI API error: {e}") from e


def create_llm_provider(provider: str, model: str) -> LLMProvider:
    """Factory for creating LLM providers."""
    if provider == "anthropic":
        return AnthropicProvider(model)
    elif provider == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"Unknown LLM provider: {provider}")
