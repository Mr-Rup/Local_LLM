"""Settings and configuration view for LocalAI workbench."""

from __future__ import annotations

import streamlit as st

from localai.config import get_settings


def render_settings() -> None:
    st.title("⚙️ Configuration & Storage")
    st.caption("Inspect active runtime settings, model defaults, and local storage state.")

    settings = get_settings()

    st.subheader("Runtime Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ollama Settings**")
        st.json(
            {
                "base_url": settings.ollama.base_url,
                "default_model": settings.ollama.default_model,
                "timeout_seconds": settings.ollama.timeout_seconds,
                "default_temperature": settings.ollama.default_temperature,
                "default_max_tokens": settings.ollama.default_max_tokens,
            }
        )

    with col2:
        st.markdown("**Document Intelligence Settings**")
        st.json(
            {
                "upload_dir": str(settings.documents.upload_dir),
                "max_file_size_mb": settings.documents.max_file_size_mb,
                "supported_extensions": settings.documents.supported_extensions,
                "default_chunk_size": settings.documents.default_chunk_size,
                "default_chunk_overlap": settings.documents.default_chunk_overlap,
                "default_top_k": settings.documents.default_top_k,
            }
        )

    st.markdown("---")
    st.subheader("Local Storage Footprint")

    doc_files = (
        list(settings.documents.upload_dir.glob("*"))
        if settings.documents.upload_dir.exists()
        else []
    )
    doc_files = [f for f in doc_files if f.is_file() and not f.name.startswith(".")]

    bench_files = (
        list(settings.benchmarking.output_dir.glob("*"))
        if settings.benchmarking.output_dir.exists()
        else []
    )
    bench_files = [f for f in bench_files if f.is_file() and not f.name.startswith(".")]

    c1, c2 = st.columns(2)
    with c1:
        total_doc_size = sum(f.stat().st_size for f in doc_files)
        st.metric("Uploaded Documents", len(doc_files), f"{total_doc_size / 1024:.1f} KB")
        if doc_files:
            with st.expander("Stored Document Files"):
                for f in doc_files:
                    st.text(f"{f.name} ({f.stat().st_size / 1024:.1f} KB)")

    with c2:
        total_bench_size = sum(f.stat().st_size for f in bench_files)
        st.metric("Benchmark Reports", len(bench_files), f"{total_bench_size / 1024:.1f} KB")
        if bench_files:
            with st.expander("Saved Benchmark Reports"):
                for f in bench_files:
                    st.text(f"{f.name} ({f.stat().st_size / 1024:.1f} KB)")
