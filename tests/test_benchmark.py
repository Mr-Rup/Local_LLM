"""Unit tests for the model benchmarking service and metric aggregation."""

from localai.schemas.benchmark import BenchmarkIteration, BenchmarkReport
from localai.services.benchmark import BenchmarkService


def test_compute_summaries_excludes_warmup() -> None:
    service = BenchmarkService()

    iterations = [
        # Warmup iteration
        BenchmarkIteration(
            iteration_idx=0,
            is_warmup=True,
            model="llama3.2:3b",
            prompt_id="test_qa",
            prompt_category="QA",
            success=True,
            time_to_first_token_ms=5000.0,  # Cold cache spike
            tokens_per_second=10.0,
            total_wall_duration_ms=6000.0,
        ),
        # Measured iteration 1
        BenchmarkIteration(
            iteration_idx=1,
            is_warmup=False,
            model="llama3.2:3b",
            prompt_id="test_qa",
            prompt_category="QA",
            success=True,
            time_to_first_token_ms=200.0,
            tokens_per_second=40.0,
            total_wall_duration_ms=1000.0,
        ),
        # Measured iteration 2
        BenchmarkIteration(
            iteration_idx=2,
            is_warmup=False,
            model="llama3.2:3b",
            prompt_id="test_qa",
            prompt_category="QA",
            success=True,
            time_to_first_token_ms=300.0,
            tokens_per_second=60.0,
            total_wall_duration_ms=1200.0,
        ),
    ]

    summaries = service._compute_summaries(["llama3.2:3b"], iterations)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.total_runs == 2
    assert s.successful_runs == 2
    assert s.success_rate == 100.0
    # Mean of 200 and 300 = 250 (warmup 5000 excluded!)
    assert s.avg_ttft_ms == 250.0
    # Mean of 40 and 60 = 50 (warmup 10 excluded!)
    assert s.avg_tokens_per_second == 50.0


def test_save_report(tmp_path) -> None:
    service = BenchmarkService()
    service.settings.benchmarking.output_dir = tmp_path

    report = BenchmarkReport(
        timestamp="2026-11-08T18:30:00",
        device_gpu="NVIDIA GeForce GTX 1660",
        device_ram_gb=16.0,
        iterations=[],
        summaries=[],
    )

    saved_path = service.save_report(report)
    assert saved_path.exists()
    assert saved_path.suffix == ".json"
