"""Ollama HTTP API provider implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx

from localai.config import get_settings
from localai.providers.base import (
    BaseLLMProvider,
    ChatMessage,
    GenerationChunk,
    GenerationOptions,
    GenerationResult,
    ModelInfo,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Direct HTTP client provider for local Ollama daemon."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        default_model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.ollama.timeout_seconds
        self.default_model = default_model or settings.ollama.default_model
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-loaded reusable httpx Client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def is_healthy(self) -> bool:
        """Check if the Ollama daemon is reachable and responding."""
        try:
            resp = self.client.get("/api/version")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
            return False

    def list_models(self) -> List[ModelInfo]:
        """Fetch list of all installed models from Ollama."""
        try:
            resp = self.client.get("/api/tags")
            if resp.status_code != 200:
                raise ProviderError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            models: List[ModelInfo] = []
            for item in data.get("models", []):
                details = item.get("details", {})
                models.append(
                    ModelInfo(
                        name=item.get("name", ""),
                        model=item.get("model", item.get("name", "")),
                        size_bytes=item.get("size", 0),
                        parameter_size=details.get("parameter_size"),
                        quantization_level=details.get("quantization_level"),
                        context_length=details.get("context_length"),
                        family=details.get("family"),
                    )
                )
            return models
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama daemon at {self.base_url}. Ensure Ollama is running."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP error communicating with Ollama: {exc}") from exc

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Retrieve details for a specific model."""
        models = self.list_models()
        for m in models:
            if m.name == model or m.model == model:
                return m
        return None

    def _build_options_payload(
        self, options: Optional[GenerationOptions | Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Translate GenerationOptions or dict into Ollama's API options dict."""
        settings = get_settings()
        if isinstance(options, dict):
            opts = dict(options)
            opts.setdefault("temperature", settings.ollama.default_temperature)
            opts.setdefault("top_p", settings.ollama.default_top_p)
            return opts

        opts: Dict[str, Any] = {
            "temperature": options.temperature if options else settings.ollama.default_temperature,
            "top_p": options.top_p if options else settings.ollama.default_top_p,
        }
        if options and options.max_tokens is not None:
            opts["num_predict"] = options.max_tokens
        if options and options.seed is not None:
            opts["seed"] = options.seed
        if options and options.stop:
            opts["stop"] = options.stop
        return opts

    def _extract_format(self, options: Optional[GenerationOptions | Dict[str, Any]]) -> Optional[str]:
        """Safely extract format parameter whether options is dict or GenerationOptions."""
        if isinstance(options, dict):
            return options.get("format")
        if isinstance(options, GenerationOptions):
            return options.format
        return None

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[GenerationOptions | Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Synchronous chat completion."""
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": self._build_options_payload(options),
        }
        fmt = self._extract_format(options)
        if fmt:
            payload["format"] = fmt

        start_time = time.perf_counter()
        try:
            resp = self.client.post("/api/chat", json=payload)
            elapsed_app_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 404:
                raise ModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
            if resp.status_code != 200:
                raise ProviderError(f"Ollama error {resp.status_code}: {resp.text}")

            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "")

            # Parse Ollama's native nanosecond metrics
            eval_count = data.get("eval_count", 0)
            eval_dur_ns = data.get("eval_duration", 0)
            eval_dur_ms = eval_dur_ns / 1e6 if eval_dur_ns else 0.0
            prompt_eval_dur_ns = data.get("prompt_eval_duration", 0)
            prompt_eval_dur_ms = prompt_eval_dur_ns / 1e6 if prompt_eval_dur_ns else 0.0
            total_dur_ns = data.get("total_duration", 0)
            total_dur_ms = total_dur_ns / 1e6 if total_dur_ns else elapsed_app_ms
            load_dur_ns = data.get("load_duration", 0)
            load_dur_ms = load_dur_ns / 1e6 if load_dur_ns else 0.0

            tok_per_sec = (
                (eval_count / (eval_dur_ns / 1e9)) if (eval_dur_ns and eval_dur_ns > 0) else 0.0
            )

            return GenerationResult(
                text=content,
                model=target_model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=eval_count,
                total_duration_ms=total_dur_ms,
                load_duration_ms=load_dur_ms,
                prompt_eval_duration_ms=prompt_eval_dur_ms,
                eval_duration_ms=eval_dur_ms,
                tokens_per_second=tok_per_sec,
                raw_response=data,
            )

        except httpx.ConnectError as exc:
            raise ProviderConnectionError(f"Cannot connect to Ollama at {self.base_url}.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP communication error: {exc}") from exc

    def chat_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        options: Optional[GenerationOptions | Dict[str, Any]] = None,
    ) -> Iterator[GenerationChunk]:
        """Streaming chat completion yielding incremental chunks."""
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": True,
            "options": self._build_options_payload(options),
        }
        fmt = self._extract_format(options)
        if fmt:
            payload["format"] = fmt

        try:
            with self.client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    raise ProviderError(f"Ollama stream error {response.status_code}: {body}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    done = chunk_data.get("done", False)
                    delta = chunk_data.get("message", {}).get("content", "")

                    if not done:
                        yield GenerationChunk(delta=delta, is_done=False, model=target_model)
                    else:
                        eval_count = chunk_data.get("eval_count", 0)
                        eval_dur_ns = chunk_data.get("eval_duration", 0)
                        eval_dur_ms = eval_dur_ns / 1e6 if eval_dur_ns else 0.0
                        prompt_eval_dur_ns = chunk_data.get("prompt_eval_duration", 0)
                        prompt_eval_dur_ms = prompt_eval_dur_ns / 1e6 if prompt_eval_dur_ns else 0.0
                        total_dur_ns = chunk_data.get("total_duration", 0)
                        total_dur_ms = total_dur_ns / 1e6 if total_dur_ns else 0.0
                        tok_per_sec = (
                            (eval_count / (eval_dur_ns / 1e9))
                            if (eval_dur_ns and eval_dur_ns > 0)
                            else 0.0
                        )

                        yield GenerationChunk(
                            delta=delta,
                            is_done=True,
                            model=target_model,
                            prompt_tokens=chunk_data.get("prompt_eval_count"),
                            completion_tokens=eval_count,
                            total_duration_ms=total_dur_ms,
                            prompt_eval_duration_ms=prompt_eval_dur_ms,
                            eval_duration_ms=eval_dur_ms,
                            tokens_per_second=tok_per_sec,
                        )

        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Connection failed to Ollama at {self.base_url}."
            ) from exc

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerationOptions | Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> GenerationResult:
        """Synchronous prompt generation."""
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options_payload(options),
        }
        if system:
            payload["system"] = system
        fmt = self._extract_format(options)
        if fmt:
            payload["format"] = fmt

        start_time = time.perf_counter()
        try:
            resp = self.client.post("/api/generate", json=payload)
            elapsed_app_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 404:
                raise ModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
            if resp.status_code != 200:
                raise ProviderError(f"Ollama error {resp.status_code}: {resp.text}")

            data = resp.json()
            eval_count = data.get("eval_count", 0)
            eval_dur_ns = data.get("eval_duration", 0)
            eval_dur_ms = eval_dur_ns / 1e6 if eval_dur_ns else 0.0
            prompt_eval_dur_ns = data.get("prompt_eval_duration", 0)
            prompt_eval_dur_ms = prompt_eval_dur_ns / 1e6 if prompt_eval_dur_ns else 0.0
            total_dur_ns = data.get("total_duration", 0)
            total_dur_ms = total_dur_ns / 1e6 if total_dur_ns else elapsed_app_ms
            load_dur_ns = data.get("load_duration", 0)
            load_dur_ms = load_dur_ns / 1e6 if load_dur_ns else 0.0
            tok_per_sec = (
                (eval_count / (eval_dur_ns / 1e9)) if (eval_dur_ns and eval_dur_ns > 0) else 0.0
            )

            return GenerationResult(
                text=data.get("response", ""),
                model=target_model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=eval_count,
                total_duration_ms=total_dur_ms,
                load_duration_ms=load_dur_ms,
                prompt_eval_duration_ms=prompt_eval_dur_ms,
                eval_duration_ms=eval_dur_ms,
                tokens_per_second=tok_per_sec,
                raw_response=data,
            )
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Connection failed to Ollama at {self.base_url}."
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerationOptions | Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> Iterator[GenerationChunk]:
        """Streaming prompt generation."""
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": self._build_options_payload(options),
        }
        if system:
            payload["system"] = system
        fmt = self._extract_format(options)
        if fmt:
            payload["format"] = fmt

        try:
            with self.client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    raise ProviderError(f"Ollama stream error {response.status_code}: {body}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    done = chunk_data.get("done", False)
                    delta = chunk_data.get("response", "")

                    if not done:
                        yield GenerationChunk(delta=delta, is_done=False, model=target_model)
                    else:
                        eval_count = chunk_data.get("eval_count", 0)
                        eval_dur_ns = chunk_data.get("eval_duration", 0)
                        eval_dur_ms = eval_dur_ns / 1e6 if eval_dur_ns else 0.0
                        prompt_eval_dur_ns = chunk_data.get("prompt_eval_duration", 0)
                        prompt_eval_dur_ms = prompt_eval_dur_ns / 1e6 if prompt_eval_dur_ns else 0.0
                        total_dur_ns = chunk_data.get("total_duration", 0)
                        total_dur_ms = total_dur_ns / 1e6 if total_dur_ns else 0.0
                        tok_per_sec = (
                            (eval_count / (eval_dur_ns / 1e9))
                            if (eval_dur_ns and eval_dur_ns > 0)
                            else 0.0
                        )

                        yield GenerationChunk(
                            delta=delta,
                            is_done=True,
                            model=target_model,
                            prompt_tokens=chunk_data.get("prompt_eval_count"),
                            completion_tokens=eval_count,
                            total_duration_ms=total_dur_ms,
                            prompt_eval_duration_ms=prompt_eval_dur_ms,
                            eval_duration_ms=eval_dur_ms,
                            tokens_per_second=tok_per_sec,
                        )
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Connection failed to Ollama at {self.base_url}."
            ) from exc
