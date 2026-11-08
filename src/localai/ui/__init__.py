"""UI views module for LocalAI."""

from localai.ui.benchmarking import render_benchmarking
from localai.ui.chat import render_chat
from localai.ui.documents import render_documents
from localai.ui.extraction import render_extraction
from localai.ui.overview import render_overview
from localai.ui.settings import render_settings

__all__ = [
    "render_overview",
    "render_chat",
    "render_documents",
    "render_extraction",
    "render_benchmarking",
    "render_settings",
]
