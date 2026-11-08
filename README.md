# LocalAI — Private LLM Workbench

> **A lightweight, private workspace for local LLM inference, document intelligence, and performance benchmarking.**

Built specifically as a demonstration of local model deployment, low-latency inference, document grounding (RAG), structured JSON extraction, and comparative model benchmarking on consumer hardware.

---

## 🚀 Key Capabilities

| Module | Functionality | Technical Highlights |
| :--- | :--- | :--- |
| **💬 Local Chat** | Multi-turn conversational interface with local LLMs | Direct streaming via `httpx`, real-time token/s speed meter, configurable temperature and max tokens. |
| **📄 Document Intelligence** | Local document grounding & question answering | C-level text parsing (`PyMuPDF`, `python-docx`), sliding-window chunker, in-memory TF-IDF cosine similarity, exact source citations `[Document, Page]`, explicit fallback for missing context. |
| **🔍 Structured Extraction** | Extract strongly typed JSON from unstructured text | Schema-guided prompt synthesis, JSON mode enforcement, Pydantic model validation, missing field handling, and exportable JSON envelopes. |
| **📊 Model Benchmarking** | Reproducible comparative evaluation | Fixed prompt evaluation suite, automated warmup discard, dual timing (application TTFT vs. Ollama native eval), real-time RAM & VRAM tracking, side-by-side charts, and CSV export. |

---

## 📚 In-Depth Technical Documentation

For complete engineering deep-dives, design patterns, and empirical data, refer to the documentation in `docs/`:

* 🏛️ **[System Architecture & Design Philosophy](docs/architecture.md)** — Architectural layering, component sequence diagrams, provider abstraction, and design rationale.
* 📊 **[Benchmarking Methodology & Empirical Analysis](docs/benchmarking_methodology.md)** — Experimental controls, TTFT vs. eval timing models, NVML VRAM sampling, and per-prompt iteration records.
* 📄 **[Document Intelligence (RAG) & Extraction](docs/rag_and_extraction.md)** — C-level text parsers, sentence boundary chunking, TF-IDF cosine similarity mathematics, and Pydantic validation envelopes.
* ⚙️ **[Hardware Sizing & VRAM Budgeting Guide](docs/hardware_sizing_guide.md)** — Quantization mathematics (GGUF Q4_K_M vs FP16), KV cache sizing, layer offloading bottlenecks, and consumer GPU matrix.

---

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Streamlit Presentation Layer           │
                  │  (Overview | Chat | Documents | Extraction | Benchmark)│
                  └───────────────────────────┬────────────────────────────┘
                                              │ Calls decoupled Python services
                  ┌───────────────────────────▼────────────────────────────┐
                  │              Application Services Layer                │
                  │   ChatService │ DocumentService │ ExtractionService    │
                  │   BenchmarkService │ ResourceMonitor (psutil/NVML)     │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Uses BaseLLMProvider protocol
                  ┌───────────────────────────▼────────────────────────────┐
                  │                 OllamaProvider Adapter                 │
                  │   Direct HTTP streaming client via httpx (no wrappers) │
                  └───────────────────────────┬────────────────────────────┘
                                              │ HTTP REST (127.0.0.1:11434pytest tests/ -v)
                  ┌───────────────────────────▼────────────────────────────┐
                  │               Local Ollama Runtime Daemon              │
                  │   llama3.2:3b (Q4_K_M) │ qwen2.5:7b │ mistral:7b       │
                  └────────────────────────────────────────────────────────┘
```

### Why No LangChain or LangGraph?

1. **Zero Abstraction Overhead:** Direct HTTP streaming via `httpx` allows microsecond-accurate measurement of **Time to First Token (TTFT)** and generation throughput without framework interference.
2. **Deterministic & Lightweight:** The entire application installs cleanly in minutes, boots in seconds, and has zero fragile dependencies or breaking API migrations.
3. **Demonstrates Foundational Engineering:** Implements core LLM concepts—chunking, inverted indexing, token streaming, schema validation, and resource sampling—directly in clean Python rather than hiding behind framework wrappers.

---

## 🖥️ Hardware Specifications & Model Sizing

* **Host GPU:** NVIDIA GeForce GTX 1660 (6,144 MB VRAM, CUDA 13.4)
* **Host CPU & RAM:** Intel/AMD x64 | 16 GB System RAM
* **OS:** Windows 10/11 x64

### Model Inventory & Memory Allocation

| Model | Parameters | Quantization | Disk / VRAM Size | Execution Profile on GTX 1660 |
| :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:3b`** | 3.2B | Q4_K_M | ~1.9 GB | **100% VRAM residence.** Outstanding throughput (~94 tok/s), low latency TTFT (42ms), zero CPU offloading. |
| **`qwen2.5:7b`** | 7.6B | Q4_K_M | ~4.4 GB | Fits in 6GB VRAM (4.8 GB peak). Solid reasoning throughput (~21 tok/s) with 100% success rate. |
| **`mistral:7b`** | 7.2B | Q4_K_M | ~4.1 GB | Fits in VRAM with high reasoning accuracy; used for comparative architectural benchmarking. |

### 📈 Empirical Benchmark Results (GTX 1660 SUPER 6GB)

*Captured via `scripts/run_benchmark.py` across standardized prompts (QA, Math/CoT, Structured JSON) with warmups discarded:*

| Model | Success Rate | Avg TTFT (ms) | Throughput (tok/s) | Avg Wall Time | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:3b`** | **100%** | **42.0 ms** | **94.3 tok/s** | **2.3 s** | **3,230 MB** (~3.2 GB) |
| **`qwen2.5:7b`** | **100%** | **190.0 ms** | **21.1 tok/s** | **8.2 s** | **4,789 MB** (~4.8 GB) |

*Key Findings:*
1. **Throughput Scaling:** `llama3.2:3b` achieves **4.5x higher throughput** (94.3 vs 21.1 tok/s) and **4.5x lower TTFT** (42ms vs 190ms) compared to `qwen2.5:7b` on identical consumer GPU hardware.
2. **VRAM Headroom:** `llama3.2:3b` leaves ~2.9 GB of free VRAM on a 6GB card for long-context KV caches, while `qwen2.5:7b` uses ~4.8 GB, approaching the card's physical boundary without spilling to system RAM.

---

## 📦 Project Layout

```
localai/
├── app.py                      # Main Streamlit UI entry point
├── pyproject.toml              # Project metadata, dependencies, and tooling configs
├── uv.lock                     # Deterministic, reproducible dependency lockfile
├── .python-version             # Pinned to Python 3.12
├── README.md                   # Complete architectural and usage guide
├── config/
│   └── settings.toml           # Application defaults (model presets, retrieval settings)
│
├── docs/                       # In-depth technical documentation
│   ├── architecture.md         # Component diagrams, data flows, and design decisions
│   ├── benchmarking_methodology.md # Benchmark controls, timing model, and findings
│   ├── rag_and_extraction.md   # Local RAG, TF-IDF ranking, and Pydantic schemas
│   └── hardware_sizing_guide.md# VRAM budgeting, quantization, and layer offloading
│
├── src/
│   └── localai/
│       ├── config.py           # Configuration loader (TOML + env vars via Pydantic)
│       ├── providers/
│       │   ├── base.py         # BaseLLMProvider protocol and data models
│       │   └── ollama.py       # Direct Ollama HTTP client adapter
│       ├── services/
│       │   ├── chat.py         # Multi-turn streaming chat service
│       │   ├── document.py     # Document QA & context assembly service
│       │   ├── extraction.py   # Pydantic schema extraction service
│       │   └── benchmark.py    # Controlled model benchmarking engine
│       ├── documents/
│       │   ├── parsers.py      # PDF, DOCX, and TXT extractors with page tracking
│       │   ├── chunking.py     # Sliding-window sentence-aware text chunker
│       │   └── retrieval.py    # Local TF-IDF search index using Scikit-Learn
│       ├── schemas/
│       │   ├── extraction.py   # InvoiceSchema, ResumeSchema, MeetingNotesSchema
│       │   └── benchmark.py    # BenchmarkIteration, ModelBenchmarkSummary
│       ├── monitoring/
│       │   └── resources.py    # Real-time CPU, RAM, and GPU VRAM profiler
│       └── ui/
│           ├── styles.py       # Dark developer workbench CSS
│           ├── overview.py     # Hardware telemetry and model cards
│           ├── chat.py         # Streaming conversation playground
│           ├── documents.py    # Local RAG with citations
│           ├── extraction.py   # Schema-guided JSON extraction
│           ├── benchmarking.py # Side-by-side comparative benchmarking
│           └── settings.py     # Storage footprints and config viewer
│
├── tests/                      # Automated pytest suite (19 tests)
│   ├── test_provider.py        # Ollama adapter unit tests (mocked offline)
│   ├── test_chunking.py        # Sliding window & page attribution tests
│   ├── test_retrieval.py       # TF-IDF search and threshold tests
│   ├── test_extraction.py      # Schema validation and JSON repair tests
│   └── test_benchmark.py       # Warmup exclusion & metric calculation tests
│
├── data/
│   ├── documents/              # Local storage for uploaded files
│   └── benchmarks/             # Persisted benchmark reports (JSON / CSV)
│
└── scripts/
    ├── check_ollama.py         # Terminal diagnostic utility
    └── run_benchmark.py        # Headless CLI benchmark runner
```

---

## ⚡ Quick Start

### 1. Prerequisites

Ensure [Ollama](https://ollama.com/) is running and you have pulled at least one model:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

### 2. Installation

Using `uv` (recommended) or standard `pip`:

```bash
# Clone the repository
git clone <repo-url>
cd "Local llm"

# Create virtual environment with Python 3.12
uv venv --python 3.12
.venv\Scripts\activate

# Install dependencies in editable mode with dev tools
uv pip install -e ".[dev]"
```

### 3. Verify Local Runtime

Run the diagnostic utility to verify connectivity and test inference:

```bash
python scripts/check_ollama.py
```

### 4. Launch Workbench UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 5. Headless Model Benchmarking (CLI)

You can also run benchmarks directly from the terminal without opening the UI:

```bash
python scripts/run_benchmark.py --models llama3.2:3b qwen2.5:7b --reps 2
```

---

## 🧪 Automated Testing & Code Quality

Run the test suite:

```bash
pytest tests/ -v
```

Run linter and code formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```
