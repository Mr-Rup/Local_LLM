"""Unit tests for the OllamaProvider adapter."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from localai.providers.base import (
    ChatMessage,
    ChatRole,
    ModelNotFoundError,
    ProviderConnectionError,
)
from localai.providers.ollama import OllamaProvider


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://127.0.0.1:11434", default_model="llama3.2:3b")


def test_is_healthy_success(provider: OllamaProvider) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch.object(provider.client, "get", return_value=mock_resp) as mock_get:
        assert provider.is_healthy() is True
        mock_get.assert_called_once_with("/api/version")


def test_is_healthy_failure(provider: OllamaProvider) -> None:
    with patch.object(provider.client, "get", side_effect=httpx.ConnectError("Failed")):
        assert provider.is_healthy() is False


def test_list_models_success(provider: OllamaProvider) -> None:
    sample_tags_response = {
        "models": [
            {
                "name": "llama3.2:3b",
                "model": "llama3.2:3b",
                "size": 2019393189,
                "details": {
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                    "context_length": 131072,
                    "family": "llama",
                },
            },
            {
                "name": "qwen2.5:7b",
                "model": "qwen2.5:7b",
                "size": 4683087332,
                "details": {
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                    "context_length": 32768,
                    "family": "qwen2",
                },
            },
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_tags_response

    with patch.object(provider.client, "get", return_value=mock_resp):
        models = provider.list_models()
        assert len(models) == 2
        assert models[0].name == "llama3.2:3b"
        assert models[0].parameter_size == "3.2B"
        assert models[0].quantization_level == "Q4_K_M"
        assert models[1].name == "qwen2.5:7b"


def test_list_models_connection_error(provider: OllamaProvider) -> None:
    with patch.object(provider.client, "get", side_effect=httpx.ConnectError("Unreachable")):
        with pytest.raises(ProviderConnectionError):
            provider.list_models()


def test_chat_success(provider: OllamaProvider) -> None:
    sample_chat_response = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "Hello there!"},
        "done": True,
        "total_duration": 500000000,  # 500 ms in ns
        "load_duration": 50000000,  # 50 ms in ns
        "prompt_eval_count": 10,
        "prompt_eval_duration": 100000000,  # 100 ms in ns
        "eval_count": 20,
        "eval_duration": 350000000,  # 350 ms in ns
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_chat_response

    messages = [ChatMessage(role=ChatRole.USER, content="Hi")]
    with patch.object(provider.client, "post", return_value=mock_resp):
        result = provider.chat(messages, model="llama3.2:3b")

        assert result.text == "Hello there!"
        assert result.model == "llama3.2:3b"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.eval_duration_ms == 350.0
        assert result.prompt_eval_duration_ms == 100.0
        # Check tokens per second calculation: 20 tokens / 0.35s = ~57.14 tok/s
        assert result.tokens_per_second > 50.0


def test_chat_model_not_found(provider: OllamaProvider) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model 'unknown:latest' not found"

    messages = [ChatMessage(role=ChatRole.USER, content="Hi")]
    with patch.object(provider.client, "post", return_value=mock_resp):
        with pytest.raises(ModelNotFoundError):
            provider.chat(messages, model="unknown:latest")


def test_chat_stream(provider: OllamaProvider) -> None:
    chunks = [
        '{"message": {"role": "assistant", "content": "Part 1"}, "done": false}',
        '{"message": {"role": "assistant", "content": " Part 2"}, "done": false}',
        '{"message": {"role": "assistant", "content": ""}, "done": true, "eval_count": 15, "eval_duration": 300000000}',
    ]

    mock_stream_ctx = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = chunks
    mock_stream_ctx.__enter__.return_value = mock_response

    with patch.object(provider.client, "stream", return_value=mock_stream_ctx):
        messages = [ChatMessage(role=ChatRole.USER, content="Hello")]
        stream_results = list(provider.chat_stream(messages, model="llama3.2:3b"))

        assert len(stream_results) == 3
        assert stream_results[0].delta == "Part 1"
        assert not stream_results[0].is_done
        assert stream_results[1].delta == " Part 2"
        assert not stream_results[1].is_done
        assert stream_results[2].is_done
        assert stream_results[2].completion_tokens == 15
        assert stream_results[2].eval_duration_ms == 300.0
