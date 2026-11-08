"""Model benchmarking service with controlled evaluation suite, timing, and resource tracking."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from localai.config import get_settings
from localai.monitoring.resources import ResourceMonitor
from localai.providers.base import BaseLLMProvider, GenerationOptions
from localai.providers.ollama import OllamaProvider
from localai.schemas.benchmark import (
    BenchmarkIteration,
    BenchmarkPrompt,
    BenchmarkReport,
    ModelBenchmarkSummary,
)

logger = logging.getLogger(__name__)

# Standardized benchmark prompt suite
DEFAULT_BENCHMARK_PROMPTS: List[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="qa_short",
        category="Short QA",
        prompt="Explain the core difference between CPU and GPU architectures in exactly 3 bullet points.",
        system_prompt="You are an expert computer systems engineer. Answer concisely.",
        max_tokens=150,
        temperature=0.0,
    ),
    BenchmarkPrompt(
        id="cot_reasoning",
        category="Reasoning / Math",
        prompt=(
            "A server cluster has 4 nodes. Node A processes 120 req/s, Node B processes 150 req/s, "
            "Node C processes 80 req/s, and Node D processes 200 req/s. "
            "If incoming traffic increases by 35% and is distributed proportionally, "
            "what is the new throughput requirement for each node and total throughput? "
            "Show your calculations step by step."
        ),
        system_prompt="You are a precise mathematical reasoning assistant. Show full calculation steps.",
        max_tokens=350,
        temperature=0.0,
    ),
    BenchmarkPrompt(
        id="structured_json",
        category="Extraction / JSON",
        prompt=(
            "Convert the following system alert into valid JSON with keys "
            "'timestamp', 'host', 'severity', 'metric', 'current_value', 'threshold', 'action_required':\n"
            "ALERT [2026-11-08 14:22:01 UTC] node-prod-04.internal HIGH: Memory usage reached 14.8GB (Threshold: 14.0GB). "
            "Action: Trigger garbage collection and notify on-call engineer."
        ),
        system_prompt="You are a JSON converter. Return ONLY valid JSON.",
        max_tokens=200,
        temperature=0.0,
    ),
]


class BenchmarkService:
    """Orchestrates controlled, reproducible local LLM benchmarking."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
    ) -> None:
        self.provider = provider or OllamaProvider()
        self.monitor = resource_monitor or ResourceMonitor()
        self.settings = get_settings()

    def run_benchmark(
        self,
        models: List[str],
        prompts: Optional[List[BenchmarkPrompt]] = None,
        repetitions: Optional[int] = None,
        include_warmup: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BenchmarkReport:
        """Execute benchmark suite across selected models under identical parameters."""
        eval_prompts = prompts or DEFAULT_BENCHMARK_PROMPTS
        reps = repetitions or self.settings.benchmarking.default_repetitions

        # Total iterations calculation
        total_steps = len(models) * len(eval_prompts) * (reps + (1 if include_warmup else 0))
        current_step = 0

        iterations: List[BenchmarkIteration] = []

        for model in models:
            logger.info("Benchmarking model: %s", model)

            for prompt_spec in eval_prompts:
                # 1. Warm-up run (if enabled)
                if include_warmup:
                    current_step += 1
                    if progress_callback:
                        progress_callback(
                            current_step,
                            total_steps,
                            f"Warmup: {model} on {prompt_spec.id}...",
                        )
                    _ = self._execute_single_run(
                        model=model,
                        prompt_spec=prompt_spec,
                        iteration_idx=0,
                        is_warmup=True,
                    )

                # 2. Measured evaluation repetitions
                for rep in range(1, reps + 1):
                    current_step += 1
                    if progress_callback:
                        progress_callback(
                            current_step,
                            total_steps,
                            f"Benchmarking: {model} on {prompt_spec.id} (Run {rep}/{reps})...",
                        )

                    iter_result = self._execute_single_run(
                        model=model,
                        prompt_spec=prompt_spec,
                        iteration_idx=rep,
                        is_warmup=False,
                    )
                    iterations.append(iter_result)

        # Compute summary aggregations
        summaries = self._compute_summaries(models, iterations)

        # Snapshot device profile
        sys_snap = self.monitor.get_snapshot()
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            device_gpu=sys_snap.gpu_name or "CPU Only",
            device_ram_gb=round(sys_snap.ram_total_mb / 1024.0, 1)
            if sys_snap.ram_total_mb
            else None,
            iterations=iterations,
            summaries=summaries,
        )

        # Persist report
        self.save_report(report)
        return report

    def _execute_single_run(
        self,
        model: str,
        prompt_spec: BenchmarkPrompt,
        iteration_idx: int,
        is_warmup: bool,
    ) -> BenchmarkIteration:
        """Execute a single run measuring TTFT, throughput, and hardware memory peak."""
        options = GenerationOptions(
            temperature=prompt_spec.temperature,
            max_tokens=prompt_spec.max_tokens,
            seed=42,  # Deterministic seed for reproducible evaluation
        )

        self.monitor.start_tracking(interval_sec=0.04)
        t_start = time.perf_counter()
        first_token_time: Optional[float] = None
        eval_dur_ms = 0.0
        prompt_eval_dur_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        tok_per_sec = 0.0
        success = True
        error_msg: Optional[str] = None

        try:
            stream = self.provider.generate_stream(
                prompt=prompt_spec.prompt,
                model=model,
                options=options,
                system=prompt_spec.system_prompt,
            )

            for chunk in stream:
                if first_token_time is None and chunk.delta.strip():
                    first_token_time = (time.perf_counter() - t_start) * 1000.0

                if chunk.is_done:
                    eval_dur_ms = chunk.eval_duration_ms or 0.0
                    prompt_eval_dur_ms = chunk.prompt_eval_duration_ms or 0.0
                    prompt_tokens = chunk.prompt_tokens or 0
                    completion_tokens = chunk.completion_tokens or 0
                    tok_per_sec = chunk.tokens_per_second or 0.0

        except Exception as exc:
            success = False
            error_msg = str(exc)
            logger.error("Benchmark iteration failed for %s: %s", model, exc)

        t_end = time.perf_counter()
        peak_ram, peak_vram = self.monitor.stop_tracking()
        total_wall_ms = (t_end - t_start) * 1000.0

        # If model returned no tokens or stream failed to register TTFT
        if first_token_time is None:
            first_token_time = total_wall_ms

        return BenchmarkIteration(
            iteration_idx=iteration_idx,
            is_warmup=is_warmup,
            model=model,
            prompt_id=prompt_spec.id,
            prompt_category=prompt_spec.category,
            success=success,
            error_message=error_msg,
            time_to_first_token_ms=round(first_token_time, 2),
            total_wall_duration_ms=round(total_wall_ms, 2),
            ollama_eval_duration_ms=round(eval_dur_ms, 2),
            ollama_prompt_eval_duration_ms=round(prompt_eval_dur_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tok_per_sec, 2),
            peak_ram_mb=round(peak_ram, 1),
            peak_vram_mb=round(peak_vram, 1) if peak_vram is not None else None,
        )

    def _compute_summaries(
        self,
        models: List[str],
        iterations: List[BenchmarkIteration],
    ) -> List[ModelBenchmarkSummary]:
        """Aggregate iteration statistics per model excluding warmup runs."""
        summaries: List[ModelBenchmarkSummary] = []

        for model in models:
            model_iters = [it for it in iterations if it.model == model and not it.is_warmup]
            if not model_iters:
                continue

            successful = [it for it in model_iters if it.success]
            success_rate = (len(successful) / len(model_iters)) * 100.0

            if successful:
                ttfts = [it.time_to_first_token_ms for it in successful]
                tpss = [it.tokens_per_second for it in successful if it.tokens_per_second > 0]
                durations = [it.total_wall_duration_ms for it in successful]
                rams = [it.peak_ram_mb for it in successful]
                vrams = [it.peak_vram_mb for it in successful if it.peak_vram_mb is not None]

                summary = ModelBenchmarkSummary(
                    model=model,
                    total_runs=len(model_iters),
                    successful_runs=len(successful),
                    success_rate=round(success_rate, 1),
                    avg_ttft_ms=round(sum(ttfts) / len(ttfts), 2),
                    min_ttft_ms=round(min(ttfts), 2),
                    max_ttft_ms=round(max(ttfts), 2),
                    avg_tokens_per_second=round(sum(tpss) / len(tpss), 2) if tpss else 0.0,
                    min_tokens_per_second=round(min(tpss), 2) if tpss else 0.0,
                    max_tokens_per_second=round(max(tpss), 2) if tpss else 0.0,
                    avg_total_duration_ms=round(sum(durations) / len(durations), 2),
                    max_ram_mb=round(max(rams), 1) if rams else 0.0,
                    max_vram_mb=round(max(vrams), 1) if vrams else None,
                )
            else:
                summary = ModelBenchmarkSummary(
                    model=model,
                    total_runs=len(model_iters),
                    successful_runs=0,
                    success_rate=0.0,
                )

            summaries.append(summary)

        return summaries

    def save_report(self, report: BenchmarkReport) -> Path:
        """Export benchmark report to JSON and CSV in output directory."""
        out_dir = self.settings.benchmarking.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"benchmark_{stamp}.json"
        csv_path = out_dir / f"benchmark_{stamp}.csv"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # Save CSV (flattened iterations)
        if report.iterations:
            df = pd.DataFrame([it.model_dump() for it in report.iterations])
            df.to_csv(csv_path, index=False)

        logger.info("Saved benchmark report to %s and %s", json_path, csv_path)
        return json_path
