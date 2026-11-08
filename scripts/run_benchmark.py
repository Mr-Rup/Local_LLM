"""Headless CLI runner for executing reproducible model benchmarks."""

from __future__ import annotations

import argparse,sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localai.monitoring.resources import ResourceMonitor
from localai.providers.ollama import OllamaProvider
from localai.services.benchmark import BenchmarkService

# command line entry point for running headless benchmarks
def main() -> int:
    parser = argparse.ArgumentParser(description="Run headless comparative local LLM benchmarks.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:3b", "qwen2.5:7b"],
        help="Model names to benchmark (default: llama3.2:3b qwen2.5:7b)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Repetitions per prompt category (default: 2)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip initial warmup iteration",
    )

    args = parser.parse_args()

    provider = OllamaProvider()
    if not provider.is_healthy():
        print("[-] Error: Ollama daemon is not running!")
        return 1

    installed = [m.name for m in provider.list_models()]
    target_models = []
    for m in args.models:
        if m in installed:
            target_models.append(m)
        else:
            print(f"[-] Warning: Model '{m}' is not installed in Ollama. Skipping.")

    if not target_models:
        print("[-] Error: No valid target models found to benchmark.")
        return 1

    print("=" * 70)
    print(" LocalAI: Headless Model Benchmarking")
    print("=" * 70)
    print(f"Models to evaluate : {', '.join(target_models)}")
    print(f"Repetitions        : {args.reps}")
    print(f"Warmup enabled     : {not args.no_warmup}")
    print("-" * 70)

    service = BenchmarkService(provider=provider, resource_monitor=ResourceMonitor())

    def progress(cur: int, total: int, msg: str) -> None:
        print(f"[{cur}/{total}] {msg}")

    report = service.run_benchmark(
        models=target_models,
        repetitions=args.reps,
        include_warmup=not args.no_warmup,
        progress_callback=progress,
    )

    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Model':<18} {'Success':<10} {'Avg TTFT (ms)':<15} {'Throughput':<15} {'Peak VRAM':<12}"
    print(header)
    print("-" * 70)

    for s in report.summaries:
        vram = f"{s.max_vram_mb:.0f} MB" if s.max_vram_mb else "N/A"
        tps = f"{s.avg_tokens_per_second:.1f} tok/s"
        ttft = f"{s.avg_ttft_ms:.1f} ms"
        rate = f"{s.success_rate:.0f}%"
        print(f"{s.model:<18} {rate:<10} {ttft:<15} {tps:<15} {vram:<12}")

    print("=" * 70)
    print(f"[+] Detailed reports exported to: {service.settings.benchmarking.output_dir.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
