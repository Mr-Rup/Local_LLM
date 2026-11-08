"""Unit tests for sliding-window document chunking."""

from localai.documents.chunking import SlidingWindowChunker, TextChunk
from localai.documents.parsers import DocumentContent, PageContent


def test_sliding_window_chunking() -> None:
    text = (
        "Alpha beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron. "
        "Pi rho sigma tau upsilon. "
        "Phi chi psi omega."
    )
    doc = DocumentContent(
        file_name="test.txt",
        file_type="txt",
        file_size_bytes=len(text),
        total_pages=1,
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
    )

    chunker = SlidingWindowChunker(chunk_size=60, chunk_overlap=15)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.document_name == "test.txt"
        assert chunk.page_number == 1
        assert len(chunk.text) > 0


def test_chunking_multi_page_attribution() -> None:
    doc = DocumentContent(
        file_name="report.pdf",
        file_type="pdf",
        file_size_bytes=200,
        total_pages=2,
        pages=[
            PageContent(page_number=1, text="Page one introduction and overview."),
            PageContent(page_number=2, text="Page two results and conclusions."),
        ],
        full_text="Page one introduction and overview.\n\nPage two results and conclusions.",
    )

    chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Page one" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "Page two" in chunks[1].text


def test_empty_document_chunking() -> None:
    doc = DocumentContent(
        file_name="empty.txt",
        file_type="txt",
        file_size_bytes=0,
        total_pages=0,
        pages=[],
        full_text="",
    )
    chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)
    assert chunks == []
