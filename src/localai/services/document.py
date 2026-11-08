"""Document Intelligence service orchestrating parsing, indexing, and grounded QA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from localai.config import get_settings
from localai.documents.chunking import SlidingWindowChunker, TextChunk
from localai.documents.parsers import DocumentContent, load_document
from localai.documents.retrieval import RetrievedChunk, TFIDFRetriever
from localai.providers.base import BaseLLMProvider, GenerationOptions
from localai.providers.ollama import OllamaProvider


class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    score: float
    snippet: str


class DocumentQAResult(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    model_used: str = ""
    latency_ms: float = 0.0


class DocumentService:
    """Manages document ingestion, chunking, TF-IDF search, and grounded QA."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider or OllamaProvider()
        self.settings = get_settings()
        self.chunker = SlidingWindowChunker(
            chunk_size=self.settings.documents.default_chunk_size,
            chunk_overlap=self.settings.documents.default_chunk_overlap,
        )
        self.retriever = TFIDFRetriever()
        self.loaded_documents: List[DocumentContent] = []
        self.all_chunks: List[TextChunk] = []

    def clear(self) -> None:
        """Clear all loaded documents and reset retrieval index."""
        self.loaded_documents.clear()
        self.all_chunks.clear()
        self.retriever.clear()

    def ingest_file(self, file_path: Path | str) -> DocumentContent:
        """Parse file, split into chunks, and update the TF-IDF search index."""
        doc = load_document(file_path)
        self.loaded_documents.append(doc)

        new_chunks = self.chunker.chunk_document(doc)
        self.all_chunks.extend(new_chunks)

        # Re-index all accumulated chunks
        self.retriever.index_chunks(self.all_chunks)
        return doc

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.1,
    ) -> DocumentQAResult:
        """Retrieve relevant context and generate a grounded, cited answer."""
        k = top_k or self.settings.documents.default_top_k
        target_model = model or self.settings.ollama.default_model

        t0 = time.perf_counter()
        retrieved_chunks: List[RetrievedChunk] = self.retriever.retrieve(
            query=question,
            top_k=k,
            min_score=0.04,
        )

        # Explicit fallback if no relevant context is found
        if not retrieved_chunks:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DocumentQAResult(
                question=question,
                answer=(
                    "Insufficient context: The provided document(s) do not contain relevant information "
                    "to answer this question accurately."
                ),
                citations=[],
                has_sufficient_context=False,
                model_used=target_model,
                latency_ms=round(elapsed_ms, 1),
            )

        # Format context blocks with exact citations
        context_blocks: List[str] = []
        citations: List[Citation] = []

        for item in retrieved_chunks:
            c = item.chunk
            page_info = f", Page {c.page_number}" if c.page_number else ""
            context_blocks.append(
                f"--- SOURCE: [{c.document_name}{page_info} | {c.chunk_id}] ---\n{c.text}\n"
            )
            citations.append(
                Citation(
                    document_name=c.document_name,
                    page_number=c.page_number,
                    chunk_id=c.chunk_id,
                    score=item.score,
                    snippet=c.text[:200] + "..." if len(c.text) > 200 else c.text,
                )
            )

        context_str = "\n".join(context_blocks)

        system_prompt = (
            "You are a factual document analysis assistant. Answer the user's question strictly using "
            "the provided excerpts. Attribute facts to their sources where applicable.\n"
            "Rules:\n"
            "1. Base your answer ONLY on the provided context.\n"
            "2. If the context does not contain enough information to answer, state clearly that the "
            "provided documents do not contain the answer.\n"
            "3. Do not make up facts or extrapolate beyond what is stated."
        )

        user_prompt = (
            f"Context Excerpts:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer (grounded in context):"
        )

        options = GenerationOptions(
            temperature=temperature,
            max_tokens=self.settings.ollama.default_max_tokens,
        )

        gen_result = self.provider.generate(
            prompt=user_prompt,
            model=target_model,
            options=options,
            system=system_prompt,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DocumentQAResult(
            question=question,
            answer=gen_result.text.strip(),
            citations=citations,
            has_sufficient_context=True,
            model_used=target_model,
            latency_ms=round(elapsed_ms, 1),
        )
