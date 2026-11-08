"""LLM Provider abstraction layer."""

from localai.providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatRole,
    GenerationChunk,
    GenerationOptions,
    GenerationResult,
    ModelInfo,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
)
from localai.providers.ollama import OllamaProvider

__all__ = [
    "BaseLLMProvider",
    "ChatMessage",
    "ChatRole",
    "GenerationChunk",
    "GenerationOptions",
    "GenerationResult",
    "ModelInfo",
    "ProviderError",
    "ProviderConnectionError",
    "ModelNotFoundError",
    "OllamaProvider",
]
