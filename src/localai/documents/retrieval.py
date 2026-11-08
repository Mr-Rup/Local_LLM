"""Local TF-IDF retrieval index using Scikit-Learn for privacy-first, zero-dependency search."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from localai.documents.chunking import TextChunk


class RetrievedChunk(BaseModel):
    """A retrieved chunk with relevance scoring and source citation metadata."""

    chunk: TextChunk
    score: float

    @property
    def citation(self) -> str:
        """Formatted human-readable citation string."""
        page_str = f", Page {self.chunk.page_number}" if self.chunk.page_number else ""
        return f"[{self.chunk.document_name}{page_str}]"


class TFIDFRetriever:
    """Local, in-memory TF-IDF index for fast document chunk retrieval."""

    def __init__(self, min_df: int = 1, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        self.chunks: List[TextChunk] = []
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        """Reset index and clear stored chunks."""
        self.chunks.clear()
        self._tfidf_matrix = None
        self._is_fitted = False

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Build TF-IDF vocabulary and matrix over the provided text chunks."""
        if not chunks:
            self.clear()
            return

        self.chunks = list(chunks)
        corpus = [c.text for c in self.chunks]

        # Fit vectorizer and compute TF-IDF matrix
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._is_fitted = True

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> List[RetrievedChunk]:
        """Query the index and return top-k most relevant chunks."""
        if not self._is_fitted or not self.chunks or self._tfidf_matrix is None:
            return []

        query_str = query.strip()
        if not query_str:
            return []

        try:
            query_vec = self.vectorizer.transform([query_str])
        except Exception:
            return []

        # Compute cosine similarity between query and all indexed chunks
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get indices sorted by similarity score descending
        top_indices = np.argsort(similarities)[::-1]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=round(score, 4)))
            if len(results) >= top_k:
                break

        return results
