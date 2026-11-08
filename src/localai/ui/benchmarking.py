"""Model benchmarking view for LocalAI workbench."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from localai.config import get_settings
from localai.monitoring.resources import ResourceMonitor
from localai.providers.ollama import OllamaProvider
from localai.services.benchmark import DEFAULT_BENCHMARK_PROMPTS, BenchmarkService


def render_benchmarking() -> None:
    st.title("📊 Model Benchmarking")
    st.caption(
        "Measure and compare latency (TTFT), generation throughput (tok/s), and memory utilization across installed models."
    )

    provider = OllamaProvider()
    settings = get_settings()
    service = BenchmarkService(provider=provider, resource_monitor=ResourceMonitor())

    # Sidebar controls
    with st.sidebar:
        st.subheader("Benchmark Settings")
        try:
            available_models = [m.name for m in provider.list_models()]
        except Exception:
            available_models = [settings.ollama.default_model]

        selected_models = st.multiselect(
            "Select Models to Compare",
            options=available_models,
            default=[available_models[0]] if available_models else [],
            help="Select two or more models to generate side-by-side comparative charts.",
        )

        repetitions = st.slider("Repetitions per Prompt", min_value=1, max_value=5, value=2)
        include_warmup = st.checkbox(
            "Include Warm-up Iteration",
            value=True,
            help="Discards first execution from final stats.",
        )

    # Benchmark Suite Description
    st.markdown("### Evaluation Suite")
    st.caption(
        "Controlled evaluation suite tested with deterministic parameters (temp=0.0, seed=42):"
    )

    for prompt in DEFAULT_BENCHMARK_PROMPTS:
        with st.expander(f"📌 {prompt.category} (`{prompt.id}`)", expanded=False):
            st.markdown(f"**Prompt:**\n`{prompt.prompt}`")
            st.markdown(f"**Max Tokens:** `{prompt.max_tokens}`")

    # Run Trigger
    if st.button("🚀 Start Comparative Benchmark", type="primary", use_container_width=True):
        if not selected_models:
            st.warning("Please select at least one model from the sidebar.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int, msg: str) -> None:
            pct = min(1.0, current / max(1, total))
            progress_bar.progress(pct)
            status_text.text(f"[{current}/{total}] {msg}")

        with st.spinner("Executing controlled benchmark suite..."):
            report = service.run_benchmark(
                models=selected_models,
                prompts=DEFAULT_BENCHMARK_PROMPTS,
                repetitions=repetitions,
                include_warmup=include_warmup,
                progress_callback=update_progress,
            )

        status_text.success("✅ Benchmark suite completed successfully!")
        st.session_state.latest_benchmark_report = report

    # Display Results
    if "latest_benchmark_report" in st.session_state:
        report = st.session_state.latest_benchmark_report

        st.markdown("---")
        st.subheader("Benchmark Comparison Results")

        if report.summaries:
            summary_data = []
            for s in report.summaries:
                vram_display = f"{s.max_vram_mb:.0f} MB" if s.max_vram_mb else "N/A"
                summary_data.append(
                    {
                        "Model": s.model,
                        "Success Rate": f"{s.success_rate:.0f}%",
                        "Avg TTFT (ms)": s.avg_ttft_ms,
                        "Throughput (tok/s)": s.avg_tokens_per_second,
                        "Avg Duration (ms)": s.avg_total_duration_ms,
                        "Peak RAM (MB)": s.max_ram_mb,
                        "Peak VRAM": vram_display,
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Comparative Charts
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Throughput (Tokens / Sec) — Higher is Better")
                st.bar_chart(data=summary_df, x="Model", y="Throughput (tok/s)")

            with col_chart2:
                st.markdown("#### Time to First Token (TTFT ms) — Lower is Better")
                st.bar_chart(data=summary_df, x="Model", y="Avg TTFT (ms)")

            # Detailed Iterations
            with st.expander("🔬 Detailed Iteration Records", expanded=False):
                iter_df = pd.DataFrame([it.model_dump() for it in report.iterations])
                st.dataframe(iter_df, use_container_width=True)

            # Export Button
            csv_data = pd.DataFrame([it.model_dump() for it in report.iterations]).to_csv(
                index=False
            )
            st.download_button(
                label="📥 Export Benchmark Data (CSV)",
                data=csv_data,
                file_name=f"localai_benchmark_{report.timestamp.replace(':', '-')}.csv",
                mime="text/csv",
            )
