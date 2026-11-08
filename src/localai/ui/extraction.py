"""Structured extraction view for LocalAI workbench."""

from __future__ import annotations

import json

import streamlit as st

from localai.config import get_settings
from localai.providers.ollama import OllamaProvider
from localai.services.extraction import ExtractionService

PRESET_EXAMPLES = {
    "Invoice / Receipt": (
        "Vendor: ABC Technologies\n"
        "Invoice Number: INV-2026-104\n"
        "Amount: INR 45,000\n"
        "Invoice Date: 04 November 2026\n"
        "Due Date: 04 December 2026\n"
        "GST Number: 19ABCDE1234F1Z5\n"
        "Description: Monthly software engineering consulting - 1 unit @ INR 45,000\n"
        "Terms: Net 30 days"
    ),
    "Candidate Resume": (
        "Alex Rivera\n"
        "Email: alex.rivera@example.com | Phone: +1-555-0182 | Seattle, WA\n"
        "Summary: Senior ML Engineer with 5+ years of experience deploying LLMs and RAG pipelines.\n"
        "Skills: Python, PyTorch, Ollama, Docker, Pydantic, Scikit-Learn, Streamlit, Linux, CUDA.\n"
        "Experience: Senior AI Engineer at CloudScale Inc (3 years), ML Engineer at DataWorks (2.5 years).\n"
        "Education: B.S. in Computer Science, University of Washington."
    ),
    "Meeting Notes": (
        "Project Sync: LocalAI Architecture\n"
        "Date: 2026-11-08\n"
        "Attendees: Priya, Rahul, Samantha\n"
        "Key Decisions:\n"
        "- Adopt Ollama as external runtime daemon rather than embedding weights in process.\n"
        "- Build TF-IDF retrieval for lightweight local grounding.\n"
        "Action Items:\n"
        "- Rahul to write benchmark runner script by Monday.\n"
        "- Priya to test 3B vs 7B VRAM consumption on GTX 1660.\n"
        "Next Steps: Weekly demo scheduled for Thursday."
    ),
}


def render_extraction() -> None:
    st.title("🔍 Structured Information Extraction")
    st.caption(
        "Transform unstructured business text into type-safe, validated JSON adhering to strict Pydantic schemas."
    )

    provider = OllamaProvider()
    settings = get_settings()
    service = ExtractionService(provider=provider)

    with st.sidebar:
        st.subheader("Extraction Settings")
        try:
            models = [m.name for m in provider.list_models()]
        except Exception:
            models = [settings.ollama.default_model]

        selected_model = st.selectbox("Model", options=models)
        schema_name = st.selectbox("Target Schema", options=service.get_available_schemas())

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("📋 Load Preset Example"):
            st.session_state.raw_extract_text = PRESET_EXAMPLES.get(schema_name, "")

    initial_text = st.session_state.get("raw_extract_text", PRESET_EXAMPLES.get(schema_name, ""))
    user_text = st.text_area(
        "Source Text to Extract:",
        value=initial_text,
        height=180,
    )

    if st.button("⚡ Extract & Validate JSON", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please provide text to extract from.")
            return

        with st.spinner("Executing schema-guided extraction and Pydantic validation..."):
            envelope = service.extract(
                text=user_text,
                schema_name=schema_name,
                model=selected_model,
            )

        st.markdown("---")
        st.subheader("Extraction Results")

        # Validation status banner
        if envelope.is_valid:
            st.success("✅ **Validation Passed:** Result strictly conforms to Pydantic schema.")
        else:
            st.error("⚠️ **Validation Failed:** Extraction produced schema or formatting errors.")
            if envelope.validation_errors:
                with st.expander("Inspection Errors", expanded=True):
                    for err in envelope.validation_errors:
                        st.markdown(f"- `{err}`")

        # Telemetry
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model Used", envelope.model_used)
        with c2:
            st.metric("Total Latency", f"{envelope.latency_ms} ms")
        with c3:
            st.metric("Schema Target", envelope.schema_name)

        # JSON Data Display
        st.markdown("#### Validated JSON Output")
        if envelope.data:
            st.json(envelope.data)
            json_str = json.dumps(envelope.data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{schema_name.lower().replace(' ', '_')}_extracted.json",
                mime="application/json",
            )
        else:
            st.code(envelope.raw_response, language="text")
