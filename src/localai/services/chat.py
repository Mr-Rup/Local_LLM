"""Chat service managing multi-turn conversational state and streaming responses."""

from __future__ import annotations

from typing import Iterator, List, Optional

from pydantic import BaseModel

from localai.config import get_settings
from localai.providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatRole,
    GenerationChunk,
    GenerationOptions,
    GenerationResult,
)
from localai.providers.ollama import OllamaProvider


class ConversationTurn(BaseModel):
    role: ChatRole
    content: str
    tokens: Optional[int] = None
    duration_ms: Optional[float] = None


class ChatService:
    """Manages chat history and coordinates inference with the LLM provider."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider or OllamaProvider()
        self.settings = get_settings()
        self.model = self.settings.ollama.default_model
        self.system_prompt: str = (
            "You are LocalAI, a privacy-first AI assistant running entirely on local hardware. "
            "Be clear, precise, and helpful. Format your responses using clean Markdown."
        )
        self.temperature: float = self.settings.ollama.default_temperature
        self.max_tokens: int = self.settings.ollama.default_max_tokens
        self.messages: List[ChatMessage] = []

    def set_model(self, model: str) -> None:
        """Update active conversation model."""
        self.model = model

    def set_system_prompt(self, prompt: str) -> None:
        """Update active system prompt."""
        self.system_prompt = prompt

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def get_messages(self) -> List[ChatMessage]:
        """Return full history including system prompt."""
        full: List[ChatMessage] = []
        if self.system_prompt.strip():
            full.append(ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt))
        full.extend(self.messages)
        return full

    def add_user_message(self, content: str) -> None:
        """Append user input to conversation history."""
        self.messages.append(ChatMessage(role=ChatRole.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        """Append assistant response to conversation history."""
        self.messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=content))

    def stream_response(self) -> Iterator[GenerationChunk]:
        """Stream model response for the current conversation history."""
        options = GenerationOptions(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.settings.ollama.default_top_p,
        )
        full_history = self.get_messages()

        collected_chunks: List[str] = []
        for chunk in self.provider.chat_stream(full_history, model=self.model, options=options):
            if chunk.delta:
                collected_chunks.append(chunk.delta)
            if chunk.is_done:
                full_text = "".join(collected_chunks)
                if full_text.strip():
                    self.add_assistant_message(full_text)
            yield chunk

    def send_message(self, content: str) -> GenerationResult:
        """Non-streaming roundtrip convenience method."""
        self.add_user_message(content)
        options = GenerationOptions(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.settings.ollama.default_top_p,
        )
        result = self.provider.chat(self.get_messages(), model=self.model, options=options)
        self.add_assistant_message(result.text)
        return result
