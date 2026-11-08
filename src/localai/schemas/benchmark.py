"""Pydantic schemas for model benchmarking experiments and statistics."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkPrompt(BaseModel):
    """Specification of an evaluation prompt in the benchmark test suite."""

    id: str
    category: str  # e.g., "Short QA", "Reasoning / CoT", "Extraction"
    prompt: str
    system_prompt: Optional[str] = None
    expected_tokens_min: int = 10
    max_tokens: int = 512
    temperature: float = 0.0


class BenchmarkIteration(BaseModel):
    """Result of a single prompt execution on a given model."""

    iteration_idx: int
    is_warmup: bool = False
    model: str
    prompt_id: str
    prompt_category: str
    success: bool = True
    error_message: Optional[str] = None

    # Application-observed timings
    time_to_first_token_ms: float = 0.0
    total_wall_duration_ms: float = 0.0

    # Ollama native metrics
    ollama_eval_duration_ms: float = 0.0
    ollama_prompt_eval_duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0

    # Resource measurements
    peak_ram_mb: float = 0.0
    peak_vram_mb: Optional[float] = None


class ModelBenchmarkSummary(BaseModel):
    """Aggregated benchmark statistics for a model across all evaluated runs."""

    model: str
    total_runs: int = 0
    successful_runs: int = 0
    success_rate: float = 100.0

    # Latency & throughput
    avg_ttft_ms: float = 0.0
    min_ttft_ms: float = 0.0
    max_ttft_ms: float = 0.0

    avg_tokens_per_second: float = 0.0
    min_tokens_per_second: float = 0.0
    max_tokens_per_second: float = 0.0

    avg_total_duration_ms: float = 0.0

    # Resources
    max_ram_mb: float = 0.0
    max_vram_mb: Optional[float] = None


class BenchmarkReport(BaseModel):
    """Complete serialized benchmark report containing all iterations and summaries."""

    timestamp: str
    device_gpu: Optional[str] = None
    device_ram_gb: Optional[float] = None
    iterations: List[BenchmarkIteration] = Field(default_factory=list)
    summaries: List[ModelBenchmarkSummary] = Field(default_factory=list)
