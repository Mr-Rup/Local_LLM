"""Sliding-window text chunking with page tracking and configurable overlap."""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel

from localai.documents.parsers import DocumentContent


class TextChunk(BaseModel):
    """A discrete chunk of document text ready for indexing."""

    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    text: str
    char_start: int = 0
    char_end: int = 0
    approx_tokens: int = 0


class SlidingWindowChunker:
    """Configurable sliding window text chunker."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: DocumentContent) -> List[TextChunk]:
        """Split a DocumentContent into structured TextChunks with page-level attribution."""
        chunks: List[TextChunk] = []
        global_idx = 0

        # If pages are available, chunk each page independently to preserve exact page attribution
        if doc.pages:
            for page in doc.pages:
                page_chunks = self._chunk_text(
                    text=page.text,
                    doc_name=doc.file_name,
                    page_number=page.page_number,
                    start_idx=global_idx,
                )
                chunks.extend(page_chunks)
                global_idx += len(page_chunks)
        else:
            chunks = self._chunk_text(
                text=doc.full_text,
                doc_name=doc.file_name,
                page_number=1,
                start_idx=0,
            )

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_name: str,
        page_number: Optional[int],
        start_idx: int,
    ) -> List[TextChunk]:
        """Sliding window chunking over a contiguous block of text."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        text_len = len(cleaned)
        pos = 0
        local_idx = start_idx

        while pos < text_len:
            end_pos = min(pos + self.chunk_size, text_len)

            # Try to snap to sentence or word boundary if not at the very end
            if end_pos < text_len:
                # Look for sentence boundary (.!?) within the last 15% of the window
                boundary_search_start = max(pos, end_pos - int(self.chunk_size * 0.15))
                match = re.search(r"[.!?]\s", cleaned[boundary_search_start:end_pos])
                if match:
                    end_pos = boundary_search_start + match.end()
                else:
                    # Otherwise snap to last space
                    last_space = cleaned.rfind(" ", boundary_search_start, end_pos)
                    if last_space != -1:
                        end_pos = last_space + 1

            chunk_str = cleaned[pos:end_pos].strip()
            if chunk_str:
                approx_tokens = max(1, len(chunk_str) // 4)
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk_{local_idx}",
                        document_name=doc_name,
                        page_number=page_number,
                        text=chunk_str,
                        char_start=pos,
                        char_end=end_pos,
                        approx_tokens=approx_tokens,
                    )
                )
                local_idx += 1

            if end_pos >= text_len:
                break

            pos += step

        return chunks
