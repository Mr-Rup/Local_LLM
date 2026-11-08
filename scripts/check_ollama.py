"""Diagnostic script to verify Ollama daemon connectivity and installed models."""

from __future__ import annotations

import sys, time
from pathlib import Path

# Add project root to sys.path so script can run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localai.config import get_settings
from localai.providers.base import (
    ChatMessage,
    ChatRole,
    GenerationOptions,
    ProviderConnectionError,
    ProviderError,
)
from localai.providers.ollama import OllamaProvider

# diagonostic function to verify connectivity, list models and test inference
def main() -> int:
    settings = get_settings()
    print("=" * 60)
    print(" LocalAI: Ollama Runtime Diagnostics")
    print("=" * 60)
    print(f"Target Base URL: {settings.ollama.base_url}")
    print(f"Default Model  : {settings.ollama.default_model}")
    print("-" * 60)

    provider = OllamaProvider()

    # Healthcheck
    # --------------------------------------------------------------------------
    print("[1/3] Checking Ollama daemon connectivity...")
    healthy = provider.is_healthy()
    if not healthy:
        print("[-] FAILED: Ollama daemon is unreachable!")
        print("    Ensure Ollama is running (`ollama serve`).")
        return 1
    print("[+] SUCCESS: Ollama daemon is active and responsive.")

    # List Models
    # --------------------------------------------------------------------------
    print("\n[2/3] Querying installed models...")
    try:
        models = provider.list_models()
    except ProviderConnectionError as e:
        print(f"[-] Connection error: {e}")
        return 1
    except ProviderError as e:
        print(f"[-] Provider error: {e}")
        return 1

    if not models:
        print("[-] WARNING: No models detected in Ollama!")
        print("    Run `ollama pull llama3.2:3b` to install the base model.")
        return 1

    print(f"[+] Found {len(models)} model(s):")
    print(f"{'Model Name':<28} {'Params':<10} {'Quant':<10} {'Size (GB)':<10}")
    print("-" * 60)
    for m in models:
        size_gb = f"{m.size_bytes / (1024**3):.2f}" if m.size_bytes else "N/A"
        param = m.parameter_size or "N/A"
        quant = m.quantization_level or "N/A"
        print(f"{m.name:<28} {param:<10} {quant:<10} {size_gb:<10}")

    # Test Inference
    # --------------------------------------------------------------------------
    test_model = settings.ollama.default_model
    # Check if default model exists, else pick first available
    available_names = [m.name for m in models]
    if test_model not in available_names:
        # Check prefix matching (e.g., llama3.2:3b vs llama3.2:latest)
        matching = [n for n in available_names if test_model.split(":")[0] in n]
        if matching:
            test_model = matching[0]
        else:
            test_model = available_names[0]

    print(f"\n[3/3] Testing inference on '{test_model}'...")
    try:
        t0 = time.perf_counter()
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant. Keep answers brief."),
            ChatMessage(role=ChatRole.USER, content="Reply with only the single word: READY"),
        ]
        result = provider.chat(
            messages=messages,
            model=test_model,
            options=GenerationOptions(temperature=0.0, max_tokens=10),
        )
        total_wall_ms = (time.perf_counter() - t0) * 1000.0

        print("[+] Inference Successful!")
        print(f"    Response          : {result.text.strip()}")
        print(f"    Application Wall  : {total_wall_ms:.1f} ms")
        print(f"    Ollama Eval Time  : {result.eval_duration_ms:.1f} ms")
        print(f"    Prompt Eval Time  : {result.prompt_eval_duration_ms:.1f} ms")
        print(f"    Tokens Generated  : {result.completion_tokens}")
        print(f"    Throughput        : {result.tokens_per_second:.1f} tokens/sec")

    except Exception as exc:
        print(f"[-] Inference failed: {exc}")
        return 1

    print("-" * 60)
    print("ALL CHECKS PASSED: LocalAI is connected and ready for Phase 2!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
