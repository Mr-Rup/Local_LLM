"""Overview dashboard view for LocalAI workbench."""

from __future__ import annotations

import streamlit as st

from localai.config import get_settings
from localai.monitoring.resources import ResourceMonitor
from localai.providers.ollama import OllamaProvider


def render_overview() -> None:
    st.title("🖥️ LocalAI — Private LLM Workbench")
    st.caption(
        "A lightweight, privacy-first workspace for local LLM inference, document intelligence, and performance benchmarking."
    )

    settings = get_settings()
    provider = OllamaProvider()
    monitor = ResourceMonitor()

    # System & GPU Telemetry
    st.subheader("Hardware & Resource Telemetry")
    snap = monitor.get_snapshot()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System RAM Used", f"{snap.ram_used_mb / 1024:.1f} GB", f"{snap.ram_percent}%")
    with col2:
        st.metric("Total RAM", f"{snap.ram_total_mb / 1024:.1f} GB")
    with col3:
        if snap.vram_used_mb is not None:
            st.metric(
                "GPU VRAM Used", f"{snap.vram_used_mb / 1024:.2f} GB", f"{snap.vram_percent}%"
            )
        else:
            st.metric("GPU VRAM Used", "N/A")
    with col4:
        st.metric("GPU Device", snap.gpu_name or "CPU / Integrated")

    st.markdown("---")

    # Ollama Runtime Status
    st.subheader("Ollama Runtime Status")
    col_status, col_url, col_default = st.columns(3)

    is_healthy = provider.is_healthy()
    with col_status:
        if is_healthy:
            st.success("🟢 Ollama Daemon Active")
        else:
            st.error("🔴 Ollama Daemon Unreachable")

    with col_url:
        st.info(f"🌐 Base URL: `{settings.ollama.base_url}`")

    with col_default:
        st.info(f"🎯 Default Model: `{settings.ollama.default_model}`")

    # Installed Models Inventory
    st.subheader("Installed Local Models")
    if not is_healthy:
        st.warning("Cannot retrieve models. Please start Ollama (`ollama serve`).")
        return

    try:
        models = provider.list_models()
        if not models:
            st.warning("No models installed yet. Run `ollama pull llama3.2:3b` in your terminal.")
            return

        cols = st.columns(min(len(models), 3))
        for idx, model in enumerate(models):
            with cols[idx % 3]:
                size_gb = (
                    f"{model.size_bytes / (1024**3):.2f} GB" if model.size_bytes else "Unknown"
                )
                params = model.parameter_size or "N/A"
                quant = model.quantization_level or "N/A"

                with st.container(border=True):
                    st.markdown(f"**`{model.name}`**")
                    st.caption(f"Family: {model.family or 'Unknown'}")
                    st.markdown(f"- **Parameters:** {params}")
                    st.markdown(f"- **Quantization:** {quant}")
                    st.markdown(f"- **Disk Size:** {size_gb}")

    except Exception as exc:
        st.error(f"Error listing models: {exc}")

    st.markdown("---")
    st.subheader("Quick Navigation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**💬 Chat Playground**\nInteractive streaming conversation with local models.")
    with c2:
        st.markdown(
            "**📄 Document Intelligence**\nUpload local PDFs/DOCXs for grounded Q&A with source citations."
        )
    with c3:
        st.markdown(
            "**🔍 Structured Extraction**\nTransform raw text into validated Pydantic JSON schemas."
        )
    with c4:
        st.markdown("**📊 Model Benchmarking**\nCompare TTFT, throughput, and VRAM across models.")
