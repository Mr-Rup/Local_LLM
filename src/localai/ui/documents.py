"""Document Intelligence view for LocalAI workbench."""

from __future__ import annotations

import streamlit as st

from localai.config import get_settings
from localai.providers.ollama import OllamaProvider
from localai.services.document import DocumentService


def render_documents() -> None:
    st.title("📄 Document Intelligence")
    st.caption(
        "Upload local documents for private, TF-IDF indexed question answering with source citations."
    )

    provider = OllamaProvider()
    settings = get_settings()

    if "doc_service" not in st.session_state:
        st.session_state.doc_service = DocumentService(provider=provider)

    doc_svc: DocumentService = st.session_state.doc_service

    # Sidebar: Model and retrieval settings
    with st.sidebar:
        st.subheader("Document QA Settings")
        try:
            models = [m.name for m in provider.list_models()]
        except Exception:
            models = [settings.ollama.default_model]

        selected_model = st.selectbox("Inference Model", options=models)
        top_k = st.slider(
            "Top Chunks (k)", min_value=1, max_value=6, value=settings.documents.default_top_k
        )

        if st.button("🗑️ Reset Indexed Documents", use_container_width=True):
            doc_svc.clear()
            st.success("Indexed documents cleared.")
            st.rerun()

    # Document Upload
    st.subheader("1. Ingest Local Document")
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt", "md"],
        help="All processing occurs strictly on your local machine. Nothing leaves your workstation.",
    )

    if uploaded_file is not None:
        # Check if already ingested in current session
        already_ingested = any(d.file_name == uploaded_file.name for d in doc_svc.loaded_documents)

        if not already_ingested:
            with st.spinner(f"Parsing and chunking '{uploaded_file.name}'..."):
                save_dir = settings.documents.upload_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / uploaded_file.name

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    doc = doc_svc.ingest_file(save_path)
                    st.success(
                        f"Successfully indexed '{doc.file_name}' ({doc.total_pages} page(s), {len(doc.full_text)} characters)."
                    )
                except Exception as exc:
                    st.error(f"Failed to ingest file: {exc}")

    # Display index summary
    if doc_svc.loaded_documents:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Loaded Files", len(doc_svc.loaded_documents))
        with col2:
            st.metric("Total Chunks Indexed", doc_svc.retriever.total_chunks)
        with col3:
            total_chars = sum(len(d.full_text) for d in doc_svc.loaded_documents)
            st.metric("Total Characters", f"{total_chars:,}")

        with st.expander("🔍 Inspect Indexed Chunks", expanded=False):
            for chunk in doc_svc.all_chunks:
                page_str = f"Page {chunk.page_number} | " if chunk.page_number else ""
                st.markdown(
                    f"**[{chunk.chunk_id}]** `{chunk.document_name}` ({page_str}{chunk.approx_tokens} tokens)"
                )
                st.caption(chunk.text)
                st.divider()

    st.markdown("---")

    # Grounded Question Answering
    st.subheader("2. Grounded Question Answering")
    if not doc_svc.loaded_documents:
        st.info("Upload a document above to begin asking grounded questions.")
        return

    question = st.text_input(
        "Ask a question based on your uploaded document(s):",
        placeholder="e.g., What are the payment terms mentioned in this agreement?",
    )

    if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving relevant context and generating grounded answer..."):
            result = doc_svc.query(
                question=question,
                model=selected_model,
                top_k=top_k,
            )

        if not result.has_sufficient_context:
            st.warning(result.answer)
        else:
            st.markdown("### Answer")
            st.markdown(result.answer)
            st.caption(
                f"⏱️ Generation Latency: **{result.latency_ms} ms** | Model: `{result.model_used}`"
            )

            st.markdown("### Source Citations")
            for cit in result.citations:
                page_str = f", Page {cit.page_number}" if cit.page_number else ""
                with st.expander(
                    f"📌 {cit.document_name}{page_str} (Relevance Score: {cit.score:.3f})"
                ):
                    st.markdown(f"**Chunk ID:** `{cit.chunk_id}`")
                    st.markdown(f"```text\n{cit.snippet}\n```")
