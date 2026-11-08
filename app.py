"""Main Streamlit application entrypoint for LocalAI — Private LLM Workbench."""

from __future__ import annotations

import sys, streamlit as st
from pathlib import Path

# Ensure src/ directory is on sys.path
root_dir = Path(__file__).parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from localai.ui.benchmarking import render_benchmarking
from localai.ui.chat import render_chat
from localai.ui.documents import render_documents
from localai.ui.extraction import render_extraction
from localai.ui.overview import render_overview
from localai.ui.settings import render_settings
from localai.ui.styles import THEME_CSS


def main() -> None:
    st.set_page_config(
        page_title="LocalAI — Private LLM Workbench",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject dark theme workbench styling
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## ⚡ **LocalAI**")
        st.caption("Private LLM Workbench v0.1.0")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Chat Playground",
                "Document Intelligence",
                "Structured Extraction",
                "Model Benchmarking",
                "Settings",
            ],
            index=0,
        )

        st.markdown("---")
        st.caption("🔒 **Privacy Guarantee:** All models, embeddings, and uploaded data run 100% on local hardware.")

    # Page Routing
    if page == "Overview":
        render_overview()
    elif page == "Chat Playground":
        render_chat()
    elif page == "Document Intelligence":
        render_documents()
    elif page == "Structured Extraction":
        render_extraction()
    elif page == "Model Benchmarking":
        render_benchmarking()
    elif page == "Settings":
        render_settings()

if __name__ == "__main__":
    main()
