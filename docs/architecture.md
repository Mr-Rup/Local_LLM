# System Architecture & Design Philosophy

`LocalAI — Private LLM Workbench` is engineered around four core principles:
1. **Zero External Dependencies / 100% Local Privacy:** Models, embeddings, and uploaded document data remain strictly on local workstation storage.
2. **Strict Layer Decoupling:** The Streamlit presentation layer has zero knowledge of model protocols or parsing logic; all business logic lives in clean Python services.
3. **Transparent Provider Abstraction:** Application code communicates with an abstract interface (`BaseLLMProvider`), allowing model runtimes to be swapped or extended without rewriting application services.
4. **No Heavy Framework Bloat:** Deliberately built without LangChain or LangGraph to provide microsecond-accurate benchmark timing, predictable behavior, and minimal runtime overhead.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph UI ["Presentation Layer (Streamlit)"]
        Nav["Sidebar Navigation"]
        OverviewView["Overview Dashboard"]
        ChatView["Chat Playground"]
        DocView["Document Intelligence"]
        ExtractView["Structured Extraction"]
        BenchView["Model Benchmarking"]
        SettingsView["Settings & Storage"]
    end

    subgraph Service ["Application Services Layer (src/localai/services/)"]
        ChatSvc["ChatService<br/><i>Multi-turn history & streaming</i>"]
        DocSvc["DocumentService<br/><i>Ingestion, search & context assembly</i>"]
        ExtractSvc["ExtractionService<br/><i>Schema-guided prompt & Pydantic validation</i>"]
        BenchSvc["BenchmarkService<br/><i>Standardized suite & metric engine</i>"]
        ResourceMon["ResourceMonitor<br/><i>psutil & NVML GPU tracker</i>"]
    end

    subgraph Ingestion ["Document Processing & Retrieval (src/localai/documents/)"]
        Parsers["Parsers (PDF / DOCX / TXT)<br/><i>PyMuPDF & python-docx</i>"]
        Chunker["SlidingWindowChunker<br/><i>Sentence-aware boundary snapping</i>"]
        Retriever["TFIDFRetriever<br/><i>In-memory scikit-learn cosine similarity</i>"]
    end

    subgraph Provider ["Provider Abstraction (src/localai/providers/)"]
        BaseProvider["BaseLLMProvider (Protocol / ABC)"]
        OllamaProvider["OllamaProvider Adapter<br/><i>Direct httpx HTTP client</i>"]
    end

    subgraph Runtime ["External Model Runtime Daemon"]
        OllamaDaemon["Ollama Daemon (127.0.0.1:11434)"]
        LocalModels["GGUF Models in VRAM<br/><i>llama3.2:3b | qwen2.5:7b | mistral:7b</i>"]
    end

    %% UI to Service
    ChatView --> ChatSvc
    DocView --> DocSvc
    ExtractView --> ExtractSvc
    BenchView --> BenchSvc
    OverviewView --> ResourceMon
    OverviewView --> OllamaProvider

    %% Service to Ingestion
    DocSvc --> Parsers
    DocSvc --> Chunker
    DocSvc --> Retriever

    %% Service to Monitoring
    BenchSvc --> ResourceMon

    %% Service to Provider
    ChatSvc --> BaseProvider
    DocSvc --> BaseProvider
    ExtractSvc --> BaseProvider
    BenchSvc --> BaseProvider

    %% Provider to Runtime
    BaseProvider --> OllamaProvider
    OllamaProvider -->|REST /api/chat & /api/generate| OllamaDaemon
    OllamaDaemon --> LocalModels
```

---

## 2. Component Breakdown

### 2.1 Presentation Layer (`src/localai/ui/`)
* **Role:** Pure rendering and user interaction.
* **Guarantees:**
  * No database, HTTP, or parsing logic is written inside Streamlit views.
  * Pages consume typed service responses (`GenerationResult`, `DocumentQAResult`, `ExtractionEnvelope`, `BenchmarkReport`).
  * Custom dark workbench CSS (`src/localai/ui/styles.py`) enforces a developer-first aesthetic.

### 2.2 Application Services Layer (`src/localai/services/`)
* **`ChatService`:** Manages conversation history (`ChatMessage` objects), applies system prompts, handles streaming generation chunks, and updates session state.
* **`DocumentService`:** Coordinates the document lifecycle: parsing files, splitting into chunks, building the TF-IDF inverted index, running similarity searches, assembling grounded context, and formatting source citations.
* **`ExtractionService`:** Inspects Pydantic model schemas, generates JSON-enforcing system prompts, calls the LLM with `format="json"`, extracts raw JSON, and validates the dictionary through Pydantic.
* **`BenchmarkService`:** Executes a controlled suite of evaluation prompts across selected models, discards warm-up runs, and calculates statistical metrics (mean TTFT, throughput, and hardware memory peak).

### 2.3 Provider Abstraction Layer (`src/localai/providers/`)
* **`BaseLLMProvider`:** Defines an abstract contract:
  * `is_healthy() -> bool`
  * `list_models() -> List[ModelInfo]`
  * `chat(...) -> GenerationResult` & `chat_stream(...) -> Iterator[GenerationChunk]`
  * `generate(...) -> GenerationResult` & `generate_stream(...) -> Iterator[GenerationChunk]`
* **`OllamaProvider`:** Implements `BaseLLMProvider` using persistent `httpx.Client` sessions. Communicates directly with:
  * `/api/tags`: Model catalog and quantization details.
  * `/api/version`: Daemon health check.
  * `/api/chat`: Multi-turn chat completions.
  * `/api/generate`: Single-turn prompt completions.

---

## 3. Data Flow Sequences

### 3.1 Streaming Chat Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant ChatUI as Chat Playground (UI)
    participant ChatSvc as ChatService
    participant Provider as OllamaProvider (httpx)
    participant Daemon as Ollama Daemon (127.0.0.1)

    User->>ChatUI: Submits prompt message
    ChatUI->>ChatSvc: add_user_message(prompt)
    ChatUI->>ChatSvc: stream_response()
    ChatSvc->>Provider: chat_stream(messages, model, options)
    Provider->>Daemon: POST /api/chat {"stream": true, "model": "llama3.2:3b"}
    
    loop Stream JSON Lines
        Daemon-->>Provider: Line chunk {"message": {"content": "token"}, "done": false}
        Provider-->>ChatSvc: GenerationChunk(delta="token", is_done=false)
        ChatSvc-->>ChatUI: Yield chunk (st.write_stream renders in real time)
    end

    Daemon-->>Provider: Final chunk {"done": true, "eval_count": 150, "eval_duration": ...}
    Provider-->>ChatSvc: GenerationChunk(is_done=true, tokens_per_second=94.2, ...)
    ChatSvc->>ChatSvc: add_assistant_message(full_text)
    ChatSvc-->>ChatUI: Displays throughput pill (94.2 tok/s | 1.6s)
```

### 3.2 Document Intelligence (RAG) Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant DocUI as Document UI
    participant DocSvc as DocumentService
    participant Chunker as SlidingWindowChunker
    participant Index as TFIDFRetriever
    participant Provider as OllamaProvider

    User->>DocUI: Uploads agreement.pdf
    DocUI->>DocSvc: ingest_file("agreement.pdf")
    DocSvc->>DocSvc: PDFParser extracts pages & text
    DocSvc->>Chunker: chunk_document(doc)
    Chunker-->>DocSvc: Returns [TextChunk(page=1), TextChunk(page=2), ...]
    DocSvc->>Index: index_chunks(all_chunks)
    Index-->>DocUI: Index ready (Total Chunks: 18)

    User->>DocUI: Asks: "What is the termination period?"
    DocUI->>DocSvc: query("What is the termination period?")
    DocSvc->>Index: retrieve(query, top_k=3, min_score=0.04)
    Index-->>DocSvc: [RetrievedChunk(score=0.48, page=2), ...]
    
    alt Sufficient Context Found
        DocSvc->>DocSvc: Formats context blocks with [agreement.pdf, Page 2]
        DocSvc->>Provider: generate(grounded_prompt)
        Provider-->>DocSvc: Answer text
        DocSvc-->>DocUI: DocumentQAResult(answer, citations)
        DocUI-->>User: Renders answer with collapsible source citations
    else Insufficient Context (Scores < 0.04)
        DocSvc-->>DocUI: DocumentQAResult(has_sufficient_context=False)
        DocUI-->>User: Warning: "Insufficient context in provided document(s)"
    end
```

---

## 4. Key Engineering Decisions & Rationale

| Decision | Rationale | Alternatives Considered |
| :--- | :--- | :--- |
| **Direct `httpx` vs. LangChain** | Eliminates framework latency, avoids breaking upstream changes, and allows exact timestamping for TTFT and tokens/sec calculation. | LangChain / LlamaIndex (rejected: excessive abstraction, hidden timing overhead). |
| **Ollama Daemon vs. `llama-cpp-python`** | Treating Ollama as an external runtime mirrors real-world production environments where model serving is decoupled from application servers. | Embedding weights directly in Python (rejected: complicates Python memory management and multiprocessing). |
| **In-Memory TF-IDF vs. Vector DB** | Scikit-learn TF-IDF runs with 0ms startup, requires no external Docker services or vector databases (Chroma/Qdrant), and is 100% deterministic. | Vector DBs (Chroma/FAISS) (rejected: unnecessary complexity for single-workstation local demo). |
| **Pydantic Validation** | Provides strict type safety, automatic JSON Schema generation, and structured error reporting. | Manual regex or unvalidated `json.loads` (rejected: fragile and prone to hallucinated keys). |

