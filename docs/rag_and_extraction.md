# Document Intelligence (RAG) & Structured Information Extraction

This document details the local document ingestion pipeline, TF-IDF retrieval mechanics, and Pydantic-validated JSON extraction engine.

---

## 1. Document Intelligence Pipeline

The document pipeline processes local files into searchable, attributable knowledge chunks without sending any data over the network:

```
[Uploaded Document: PDF / DOCX / TXT]
                    │
                    ▼
          [Document Parsers]
 (PyMuPDF for PDF | python-docx for DOCX | UTF-8 TextParser)
                    │
                    ▼
          [DocumentContent Model]
 (Preserves individual pages, total page count, file metadata)
                    │
                    ▼
       [SlidingWindowChunker]
 (Configurable size & overlap, sentence-boundary snapping, page attribution)
                    │
                    ▼
         [TFIDFRetriever Index]
 (In-memory scikit-learn TfidfVectorizer + Cosine Similarity Matrix)
                    │
      User Query ───┴───► Top-k Ranked Chunks (min_score >= 0.04)
                                  │
                                  ├─► If No Matches: Explicit Fallback
                                  │
                                  └─► If Matches Found: Grounded Prompt Assembly
                                             │
                                             ▼
                                  [Ollama Local Generation]
                                             │
                                             ▼
                                  [Answer + Source Citations]
```

### 1.1 Text Extraction & Page Tracking
* **PDFs (`PyMuPDF` / `fitz`):** Extracted page-by-page. Each page's text is tagged with its 1-indexed page number (`PageContent(page_number=idx + 1)`).
* **DOCXs (`python-docx`):** Extracted by iterating through document paragraphs and embedded table cells.
* **Plain Text (`TextParser`):** Extracted with UTF-8 decoding and Latin-1 fallback to handle legacy enterprise document encodings.

### 1.2 Sentence-Boundary Aware Chunking
Naively splitting text strictly by character length cuts words or sentences in half, degrading context retrieval. `SlidingWindowChunker` implements boundary snapping:

```python
# Sliding Window Chunker Boundary Snapping (src/localai/documents/chunking.py)
if end_pos < text_len:
    # Look for sentence boundary (.!?) within the last 15% of the window
    boundary_search_start = max(pos, end_pos - int(self.chunk_size * 0.15))
    match = re.search(r"[.!?]\s", cleaned[boundary_search_start:end_pos])
    if match:
        end_pos = boundary_search_start + match.end()
    else:
        # Fall back to word boundary (space)
        last_space = cleaned.rfind(" ", boundary_search_start, end_pos)
        if last_space != -1:
            end_pos = last_space + 1
```

### 1.3 Local TF-IDF Search Index
Rather than requiring heavy vector databases or running secondary embedding models on a memory-constrained GPU, LocalAI uses **Scikit-Learn's in-memory TF-IDF vectorizer**:
* **Sublinear TF Scaling:** Uses $ 1 + \log(\text{tf}) $ instead of raw frequency to prevent repeated terms from overwhelming the score.
* **N-gram Range (1, 2):** Indexes unigrams and bigrams to capture technical phrases (e.g., "high availability", "payment terms").
* **Stop Words:** Built-in English stop-word filtering.
* **Cosine Similarity Ranking:** Computes dot product between normalized query vector and chunk matrix.

### 1.4 Source Citations & Fallback Handling
* Each retrieved chunk exposes a formatted citation property:
  ```python
  @property
  def citation(self) -> str:
      page_str = f", Page {self.chunk.page_number}" if self.chunk.page_number else ""
      return f"[{self.chunk.document_name}{page_str}]"
  ```
* **Score Filtering & Grounding Gate:** If all similarity scores fall below the minimum threshold (`min_score = 0.04`), `DocumentService` immediately returns:
  > *"Insufficient context: The provided document(s) do not contain relevant information to answer this question accurately."*
  This prevents hallucinated answers when user queries are unrelated to the uploaded corpus.

---

## 2. Structured Information Extraction

Extracting strongly typed, reliable JSON from unstructured natural language is one of the most critical enterprise use cases for local LLMs.

### 2.1 Pydantic Schema Contracts
Data contracts are declared as standard Pydantic models:

```python
class InvoiceItem(BaseModel):
    description: str
    quantity: Optional[float] = 1.0
    unit_price: Optional[float] = None
    total_price: Optional[float] = None

class InvoiceSchema(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = "USD"
    tax_or_gst: Optional[str] = None
    items: List[InvoiceItem] = Field(default_factory=list)
```

### 2.2 Schema-Guided Prompt Synthesis
`ExtractionService` dynamically inspects the Pydantic model at runtime via `.model_json_schema()` and injects the JSON Schema directly into the system prompt:

```text
You are a strict data extraction engine. 
Your task is to extract information from the user's text and structure it into a JSON object 
that strictly conforms to the following JSON Schema:
{ ... JSON Schema Definition ... }

Rules:
1. Output ONLY the raw JSON object.
2. If a field cannot be determined, use null (or [] for lists).
3. Do not invent or hallucinate facts not present in the source text.
```

### 2.3 JSON Mode Enforcement & Validation
1. **Ollama Format Enforcement:** The provider passes `format="json"` in the API request, enforcing syntactically valid JSON token generation at the sampler level.
2. **Markdown Fence Stripping:** `_extract_json_substring()` safely extracts the JSON string even if an LLM wraps the response in ````json ... ```` fences.
3. **Pydantic Validation:** The parsed dictionary is validated via `schema_cls.model_validate(parsed_dict)`.
4. **Envelope Packaging:** Returns an `ExtractionEnvelope` containing:
   * `is_valid: bool`
   * `data: Optional[Dict[str, Any]]`
   * `validation_errors: List[str]`
   * `latency_ms: float`
   * `model_used: str`

