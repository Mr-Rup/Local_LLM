# Hardware Sizing & VRAM Budgeting Guide

Deploying local LLMs successfully on consumer hardware requires precise understanding of **quantization**, **weight memory footprint**, **KV cache overhead**, and **GPU layer offloading**.

---

## 1. The VRAM Memory Equation

The total GPU memory required to run an LLM is calculated as:

$$\text{Total VRAM} = \text{Model Weights Memory} + \text{KV Cache Memory} + \text{CUDA Context Overhead}$$

### 1.1 Model Weights Memory
Model weights size depends on parameter count and quantization precision:

$$\text{Weight Size (GB)} \approx \frac{\text{Parameters (Billions)} \times \text{Bits Per Weight}}{8}$$

| Precision | Bits/Weight | 3B Model Weights | 7B Model Weights | 14B Model Weights |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 / BF16** | 16 bits | ~6.0 GB | ~14.0 GB | ~28.0 GB |
| **Q8_0 (8-bit)** | 8.5 bits | ~3.2 GB | ~7.5 GB | ~15.0 GB |
| **Q4_K_M (4-bit)** | **4.5 bits** | **~1.9 GB** | **~4.5 GB** | **~9.0 GB** |

### 1.2 KV Cache Memory
During autoregressive generation, keys and values for preceding tokens must be cached in memory. The KV cache size scales linearly with context length:

$$\text{KV Cache Size (Bytes)} = 2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision} \times \text{context\_tokens}$$

For typical models:
* **`llama3.2:3b` at 4,096 context tokens:** ~250 MB
* **`llama3.2:3b` at 16,384 context tokens:** ~1.0 GB
* **`qwen2.5:7b` at 4,096 context tokens:** ~500 MB
* **`qwen2.5:7b` at 32,768 context tokens:** ~4.0 GB

### 1.3 CUDA Context Overhead
The NVIDIA CUDA runtime driver and framework allocate ~300 MB to 600 MB of VRAM simply initializing CUDA buffers on Windows.

---

## 2. Layer Offloading & Split Execution

When a model's total memory exceeds physical VRAM, Ollama automatically splits transformer layers between GPU VRAM and System RAM:

```
Full GPU Residence (High Speed):
┌──────────────────────────────────────────────┐
│  GPU VRAM (GTX 1660 - 6GB)                   │
│  [All 28 Layers of llama3.2:3b] ──► ~94 tok/s │
└──────────────────────────────────────────────┘

Partial Offloading / Split Execution (Reduced Speed):
┌────────────────────────┐  PCIe Bus (16 GB/s)  ┌────────────────────────┐
│ GPU VRAM (6GB)         │◄════════════════════►│ System RAM (16GB)       │
│ [20 Layers on GPU]     │                      │ [8 Layers on CPU]      │
└────────────────────────┘                      └────────────────────────┘
          ▼                                                 ▼
 Fast GPU Compute                               Slow CPU Tensor Math
                      Bottleneck: Layer weights & activations
                      transferred across PCIe bus on every token!
```

> [!WARNING]
> Even offloading 2–4 layers to CPU can reduce inference throughput by **60% to 80%** due to PCI-Express bandwidth saturation. For low latency, choose a model that fits 100% in VRAM!

---

## 3. Consumer Hardware Recommendation Matrix

| GPU Available | Dedicated VRAM | Recommended Primary Model | Secondary Comparison Model | Max Recommended Context |
| :--- | :--- | :--- | :--- | :--- |
| **GTX 1650 / 1050 Ti** | **4 GB** | `qwen2.5:1.5b` (Q4_K_M) | `llama3.2:1b` (Q4_K_M) | 4,096 tokens |
| **GTX 1660 / 1660 SUPER** | **6 GB** | **`llama3.2:3b` (Q4_K_M)** | **`qwen2.5:7b` (Q4_K_M)** | **4,096 to 8,192 tokens** |
| **RTX 3060 / 4060 Laptop** | **8 GB** | `llama3.1:8b` (Q4_K_M) | `mistral:7b` (Q4_K_M) | 8,192 to 16,384 tokens |
| **RTX 3060 Desktop** | **12 GB** | `qwen2.5:14b` (Q4_K_M) | `llama3.1:8b` (Q8_0) | 16,384 to 32,768 tokens |
| **RTX 3090 / 4090** | **24 GB** | `qwen2.5:32b` (Q4_K_M) | `command-r:35b` (Q4_K_M) | 32,768+ tokens |

---

## 4. Case Study: Your GTX 1660 SUPER (6GB VRAM)

On our tested testbed:
* **`llama3.2:3b`:**
  * Model Weights: `1.9 GB`
  * KV Cache (2k tokens): `~150 MB`
  * CUDA Base: `~600 MB`
  * **Total Peak VRAM:** `3,230 MB` (~3.2 GB out of 6.1 GB)
  * **Result:** 100% GPU residence, producing **94.3 tokens/sec**.
* **`qwen2.5:7b`:**
  * Model Weights: `4.4 GB`
  * KV Cache (2k tokens): `~250 MB`
  * CUDA Base: `~600 MB`
  * **Total Peak VRAM:** `4,789 MB` (~4.8 GB out of 6.1 GB)
  * **Result:** 100% GPU residence, producing **21.1 tokens/sec**.

