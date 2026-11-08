# Model Benchmarking Methodology & Empirical Analysis

A critical requirement of local LLM engineering is the ability to conduct **reproducible, controlled performance evaluations**. This document explains the benchmark architecture, timing model, resource sampling techniques, and empirical findings on consumer hardware.

---

## 1. Benchmarking Principles

To ensure scientific validity when comparing local models:
1. **Deterministic Parameters:** All models are evaluated with `temperature = 0.0` and a fixed random `seed = 42` to eliminate stochastic variance.
2. **Standardized Prompt Dataset:** Models are evaluated across three distinct task categories:
   * **Short QA (`qa_short`):** Low prompt tokens, short completion tokens (evaluates raw latency and Time to First Token).
   * **Math Reasoning / CoT (`cot_reasoning`):** Multi-step arithmetic problem (evaluates chain-of-thought sustained generation speed).
   * **Structured Extraction (`structured_json`):** Strict JSON formatting (evaluates schema-following capability and syntax generation).
3. **Warm-Up Run Discarding:** The first inference pass on a model loads model weights and allocates the initial KV cache in GPU memory. This cold-start spike is executed as an explicit warmup pass and discarded from statistical aggregations.
4. **Dual Timing Model:** Distinguishes application-observed latency from Ollama's internal native metrics.

---

## 2. Timing & Metrics Architecture

```
Timeline of a Single Generation Request:

Request Start (t0)
   │
   ├─► HTTP connection & payload serialization
   ├─► Ollama receives prompt & computes prompt tokens (Prompt Eval)
   │
First Token Emitted (t_first)
   │   └─► Time to First Token (TTFT) = t_first - t0
   │
   ├─► Autoregressive generation of completion tokens (Eval Duration)
   │
Last Token Emitted (t_end)
   │   └─► Total Wall Duration = t_end - t0
   │
Response Completion
```

### Measured Metrics

| Metric | Measurement Source | Meaning |
| :--- | :--- | :--- |
| **Time to First Token (TTFT)** | Application (`time.perf_counter()`) | Elapsed milliseconds from request dispatch until the first non-empty token chunk arrives in the client stream. |
| **Total Wall Duration** | Application (`time.perf_counter()`) | Total elapsed milliseconds for the entire round-trip request. |
| **Prompt Eval Duration** | Ollama Native (`prompt_eval_duration`) | Nanoseconds spent by Ollama tokenizing and processing the prompt context. |
| **Eval Duration** | Ollama Native (`eval_duration`) | Nanoseconds spent actively generating new completion tokens. |
| **Generation Throughput** | Ollama Native (`eval_count` / `eval_duration`) | Tokens generated per second ($ \text{tokens/sec} = \frac{\text{eval\_count}}{\text{eval\_duration\_ns}} \times 10^9 $). |
| **Peak RAM** | `psutil` background tracker | Maximum system memory consumed by host OS during generation. |
| **Peak VRAM** | NVML / `nvidia-smi` tracker | Maximum dedicated GPU memory allocated during inference. |

---

## 3. Real-Time Resource Sampling

To measure hardware memory consumption without blocking generation, `ResourceMonitor` spawns a background thread polling every 40 milliseconds:

```python
# Resource Monitor Background Sampling Pattern
def start_tracking(self, interval_sec: float = 0.04) -> None:
    self._peak_ram_mb = self.get_snapshot().ram_used_mb
    self._peak_vram_mb = self.get_snapshot().vram_used_mb
    self._tracking = True

    def _track() -> None:
        while self._tracking:
            snap = self.get_snapshot()
            if snap.ram_used_mb > self._peak_ram_mb:
                self._peak_ram_mb = snap.ram_used_mb
            if snap.vram_used_mb and snap.vram_used_mb > self._peak_vram_mb:
                self._peak_vram_mb = snap.vram_used_mb
            time.sleep(interval_sec)

    self._track_thread = threading.Thread(target=_track, daemon=True)
    self._track_thread.start()
```

---

## 4. Empirical Benchmark Results (GTX 1660 SUPER 6GB)

*Hardware Profile: NVIDIA GeForce GTX 1660 SUPER (6,144 MB VRAM, Driver 630.20, CUDA 13.4) | 16 GB System RAM | Windows x64*

### Aggregated Comparison

| Model | Evaluated Runs | Success Rate | Avg TTFT | Min TTFT | Max TTFT | Avg Throughput | Avg Wall Time | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:3b`** | 3 | **100%** | **42.0 ms** | 37.6 ms | 46.4 ms | **94.3 tok/s** | **2.3 s** | **3,230 MB** |
| **`qwen2.5:7b`** | 3 | **100%** | **190.0 ms** | 60.7 ms | 439.1 ms | **21.1 tok/s** | **8.2 s** | **4,789 MB** |

### Per-Prompt Iteration Deep Dive

#### 1. `llama3.2:3b` (Q4_K_M, 3.2B parameters)
* **Short QA (`qa_short`):**
  * TTFT: `37.6 ms` | Throughput: `95.2 tok/s` | Prompt Eval: `11.3 ms` (55 tokens) | Tokens: `150` | VRAM: `3,230 MB`
* **Math Reasoning / CoT (`cot_reasoning`):**
  * TTFT: `46.4 ms` | Throughput: `92.4 tok/s` | Prompt Eval: `11.9 ms` (113 tokens) | Tokens: `350` | VRAM: `3,218 MB`
* **Structured JSON (`structured_json`):**
  * TTFT: `42.0 ms` | Throughput: `95.2 tok/s` | Prompt Eval: `11.1 ms` (122 tokens) | Tokens: `136` | VRAM: `3,215 MB`

#### 2. `qwen2.5:7b` (Q4_K_M, 7.6B parameters)
* **Short QA (`qa_short`):**
  * TTFT: `439.2 ms` | Throughput: `20.9 tok/s` | Prompt Eval: `50.1 ms` (43 tokens) | Tokens: `57` | VRAM: `4,786 MB`
* **Math Reasoning / CoT (`cot_reasoning`):**
  * TTFT: `70.0 ms` | Throughput: `20.9 tok/s` | Prompt Eval: `48.4 ms` (109 tokens) | Tokens: `350` | VRAM: `4,789 MB`
* **Structured JSON (`structured_json`):**
  * TTFT: `60.7 ms` | Throughput: `21.5 tok/s` | Prompt Eval: `48.9 ms` (120 tokens) | Tokens: `99` | VRAM: `4,783 MB`

---

## 5. Key Engineering Insights

1. **Throughput Disparity (4.5x Advantage):**
   * `llama3.2:3b` produced **94.3 tokens/second**, whereas `qwen2.5:7b` produced **21.1 tokens/second**.
   * On consumer GPUs with memory bandwidth constraints (GTX 1660 has 336 GB/s memory bandwidth), reading the 4.7 GB weights of a 7B model takes significantly longer per autoregressive step than reading the 1.9 GB weights of a 3B model.
2. **TTFT and Prompt Evaluation:**
   * `llama3.2:3b` processed prompt tokens in ~11 ms (~10,000 tokens/sec prompt ingestion rate).
   * `qwen2.5:7b` required ~49 ms for prompt evaluation, reflecting the quadratic scaling of attention computation across higher hidden dimensions.
3. **VRAM Boundary Safety:**
   * Both models executed with 100% layer residence on GPU without spilling into system RAM across PCI-Express, avoiding catastrophic throughput drops.

