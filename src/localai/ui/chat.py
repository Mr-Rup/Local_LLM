"""Chat playground view for LocalAI workbench."""

from __future__ import annotations

import streamlit as st

from localai.config import get_settings
from localai.providers.base import ChatRole
from localai.providers.ollama import OllamaProvider
from localai.services.chat import ChatService


def render_chat() -> None:
    st.title("💬 Local Chat Playground")
    st.caption("Zero-cloud conversational inference with locally hosted LLMs.")

    provider = OllamaProvider()
    settings = get_settings()

    # Initialize chat service in session state
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = ChatService(provider=provider)

    chat_svc: ChatService = st.session_state.chat_service

    # Sidebar parameters
    with st.sidebar:
        st.subheader("Model & Generation Parameters")

        try:
            models = provider.list_models()
            model_names = [m.name for m in models]
        except Exception:
            model_names = [settings.ollama.default_model]

        default_idx = 0
        if chat_svc.model in model_names:
            default_idx = model_names.index(chat_svc.model)

        selected_model = st.selectbox(
            "Selected Model",
            options=model_names,
            index=default_idx,
            help="Select the local model hosted in Ollama to run inference.",
        )
        chat_svc.set_model(selected_model)

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=chat_svc.temperature,
            step=0.05,
            help="Lower values produce deterministic output; higher values increase variety.",
        )
        chat_svc.temperature = temperature

        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=128,
            max_value=4096,
            value=chat_svc.max_tokens,
            step=128,
            help="Maximum completion tokens allowed per generation.",
        )
        chat_svc.max_tokens = max_tokens

        with st.expander("System Prompt", expanded=False):
            system_prompt = st.text_area(
                "Instructions",
                value=chat_svc.system_prompt,
                height=120,
            )
            chat_svc.set_system_prompt(system_prompt)

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            chat_svc.clear()
            st.session_state.chat_metrics = []
            st.rerun()

    if "chat_metrics" not in st.session_state:
        st.session_state.chat_metrics = []

    # Display conversation messages
    for idx, msg in enumerate(chat_svc.messages):
        with st.chat_message(msg.role.value):
            st.markdown(msg.content)
            # Display metrics if available for assistant turn
            if msg.role == ChatRole.ASSISTANT and idx // 2 < len(st.session_state.chat_metrics):
                m = st.session_state.chat_metrics[idx // 2]
                if m:
                    st.caption(
                        f"⚡ **{m.get('tps', 0):.1f} tok/s** | ⏱️ **{m.get('dur_ms', 0):.0f} ms** | 🎯 **{m.get('tokens', 0)} tokens**"
                    )

    # Chat input
    if prompt := st.chat_input("Ask a question or enter a prompt..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_svc.add_user_message(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            metrics_captured = {}

            try:
                for chunk in chat_svc.stream_response():
                    if chunk.delta:
                        full_response += chunk.delta
                        response_container.markdown(full_response + "▌")

                    if chunk.is_done:
                        metrics_captured = {
                            "tps": chunk.tokens_per_second or 0.0,
                            "dur_ms": chunk.total_duration_ms or chunk.eval_duration_ms or 0.0,
                            "tokens": chunk.completion_tokens or 0,
                        }

                response_container.markdown(full_response)
                st.session_state.chat_metrics.append(metrics_captured)

                if metrics_captured:
                    st.caption(
                        f"⚡ **{metrics_captured.get('tps', 0):.1f} tok/s** | "
                        f"⏱️ **{metrics_captured.get('dur_ms', 0):.0f} ms** | "
                        f"🎯 **{metrics_captured.get('tokens', 0)} tokens**"
                    )

            except Exception as exc:
                st.error(f"Inference error: {exc}")
