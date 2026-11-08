"""Abstract base class and contract definitions for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class GenerationOptions(BaseModel):
    temperature: float = 0.7
    max_tokens: Optional[int] = 2048
    top_p: float = 0.9
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    format: Optional[str] = None  # e.g., "json"


class ModelInfo(BaseModel):
    name: str
    model: str
    size_bytes: int = 0
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    context_length: Optional[int] = None
    family: Optional[str] = None


class GenerationChunk(BaseModel):
    delta: str
    is_done: bool = False
    model: Optional[str] = None
    # Metrics populated on the final chunk (is_done=True)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None
    eval_duration_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


class GenerationResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    raw_response: Optional[Dict[str, Any]] = Field(default=None, repr=False)


class ProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class ProviderConnectionError(ProviderError):
    """Raised when unable to connect to the model runtime daemon."""

    pass


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not found in the runtime."""

    pass


class BaseLLMProvider(ABC):
    """Abstract interface for local LLM providers."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the provider daemon is reachable and responsive."""
        pass

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List all models currently installed and available in the provider runtime."""
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Fetch metadata and specs for a specific model."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
    ) -> GenerationResult:
        """Perform a synchronous chat completion."""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
    ) -> Iterator[GenerationChunk]:
        """Stream a chat completion yielding incremental chunks."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        """Perform a synchronous single-turn prompt generation."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        system: Optional[str] = None,
    ) -> Iterator[GenerationChunk]:
        """Stream a single-turn prompt generation yielding incremental chunks."""
        pass
