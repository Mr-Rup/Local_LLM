"""Unit tests for local TF-IDF document retrieval."""

import pytest

from localai.documents.chunking import TextChunk
from localai.documents.retrieval import TFIDFRetriever


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="chunk_0",
            document_name="architecture.txt",
            page_number=1,
            text="LocalAI uses Ollama as an external runtime for local LLM inference.",
        ),
        TextChunk(
            chunk_id="chunk_1",
            document_name="hardware.txt",
            page_number=1,
            text="The Nvidia GeForce GTX 1660 Ti has 6GB of GDDR6 VRAM and handles quantized 3B models easily.",
        ),
        TextChunk(
            chunk_id="chunk_2",
            document_name="benchmarking.txt",
            page_number=2,
            text="Latency and Time to First Token (TTFT) are critical metrics for measuring model throughput.",
        ),
    ]


def test_tfidf_retrieval_ranking(sample_chunks: list[TextChunk]) -> None:
    retriever = TFIDFRetriever()
    retriever.index_chunks(sample_chunks)

    # Search for hardware query
    results = retriever.retrieve("How much VRAM does the GTX 1660 have?", top_k=2)

    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.chunk.document_name == "hardware.txt"
    assert "GTX 1660" in top_hit.chunk.text
    assert top_hit.score > 0.1
    assert top_hit.citation == "[hardware.txt, Page 1]"


def test_tfidf_retrieval_empty_query(sample_chunks: list[TextChunk]) -> None:
    retriever = TFIDFRetriever()
    retriever.index_chunks(sample_chunks)

    results = retriever.retrieve("   ", top_k=3)
    assert results == []


def test_tfidf_retrieval_unrelated_query(sample_chunks: list[TextChunk]) -> None:
    retriever = TFIDFRetriever()
    retriever.index_chunks(sample_chunks)

    # Completely unrelated query with no token overlap
    results = retriever.retrieve("astrophysics planetary orbits", top_k=3, min_score=0.1)
    assert results == []
