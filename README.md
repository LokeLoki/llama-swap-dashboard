# llama-swap Dashboard

Live GPU and inference monitor for llama-swap with optional Ollama auxiliary model.

A clean terminal dashboard that monitors GPU stats and inference performance in real time. Auto-detects NVIDIA or AMD GPUs at startup and applies a matching color theme — no separate scripts needed.

## Requirements

- **Python 3.8** (no extra packages — stdlib only)
- **NVIDIA** or **AMD GPU** with drivers installed
- **llama-swap** running on localhost (default port 8080)

## Quick Start

Double-click `dashboard.py` or run from a terminal:

```bash
python dashboard.py
```

Place the dashboard script in the same folder as your llama-swap `config.yaml` (optional — required for model name, quantization, and VRAM calculation features).

## Auto-Detection

The dashboard detects your GPU backend automatically at startup:

| GPU | Detection Method | Color Theme |
|-----|-----------------|-------------|
| NVIDIA | `nvidia-smi` available | Green / Dark green |
| AMD | `amd-smi` available | Orange / Red |

## Supported Models

Accurate VRAM estimation with built-in architecture tables for: **Gemma family**, **Qwen family**, **Llama family**, **GLM 5.2**, **Kimi K2**, **Laguna 2.1**, **DeepSeek**, **Ornith**, **Bonsai**, **Mixtral**, **Mistral**, **Codestral**, **Mistral Nemo 2**, **Command Aura**, **Nemotron-5/H**, **Llama 4**, **Mistral Large 3**, and more.

Models not in the table fall back to safe default estimates — no crashes, no wrong numbers.

> **Note:** The architecture tables will need updates as new models are released. Edit `MODEL_ARCHITECTURES`, `QWEN_HYBRID_LAYERS`, and `NEMOTRON_ATTENTION_LAYERS` directly in the script to add or correct entries.

The dashboard creates its own `dashboard.conf` file automatically on first run. You don't need to edit or touch your `config.yaml` — the dashboard just reads it.

The dashboard auto-detects llama-swap on first run. If it can't reach `localhost:8080`, it will prompt you to enter the correct host. Your setting is saved to `dashboard.conf` and reused on future runs.

## Configuration

`dashboard.conf` is created automatically. You can edit it directly or delete it to reset:

```ini
# llama-swap Dashboard configuration
#
# host            - llama-swap API URL (required)
# config_yaml     - path to llama-swap config.yaml (optional, for model name + quant parsing)
#                   Leave blank to skip quant parsing
# aux_port        - Ollama auxiliary model port (default: 11434)

host=http://localhost:8080
config_yaml=config.yaml
aux_port=11434
```

### Command-line options

```bash
# Use a custom host
python dashboard.py --host http://localhost:9090

# Change refresh interval (default: 2 seconds)
python dashboard.py --refresh 5

# See help
python dashboard.py --help
```

## What It Shows

- **GPU Status** — Real-time temp, VRAM usage, utilization, power draw, and fan speed for every GPU detected. Works with 1 GPU or 8+ — scales automatically to whatever hardware you have.
- **System RAM** — Host memory usage from llama-swap
- **Model VRAM** — Additive estimate: model weights + KV cache. Breakdown shown as **Static** (weights + KV) and **Runtime** (context buffers + compute + flash attention + tensor sync). Estimated values — live `nvidia-smi` / `amd-smi` remains the ground truth.
- **Decode t/s by Context Length** — Shows decode speed across input token ranges. Reveals where throughput degrades as context grows. Solid bar shows median (p50), dim tail shows p90 spread, with total output tokens. Only requests with ≥512 input tokens are charted.
- **Last Prompts** — Rolling log of recent inference requests with decode speed, prompt speed, input/output tokens, and cache hit count.
- **Session Tokens** — Cumulative input, output, and request count for the active model.
- **Speculative Decoding** — Acceptance rate displayed in real time when `--spec-type` is active.

## Model VRAM Calculation (EXPERIMENTAL)

The dashboard estimates total VRAM usage by combining model weights with a calculated KV cache size. When multimodal (`--mmproj`) or speculative decoding (`--model-draft` / MTP) are used, the exact file sizes of those models are included too. The `--cache-ram` flag is respected — KV cache is capped to the specified limit when set.

The core formula starts with a model's architecture (layers, KV heads, head dimension) from a built-in table, then multiplies:

```
cache = 2 × layers × kv_heads × head_dim × cache_bytes × tokens
```

### Partial Offload (`-ngl`)

When `-ngl N` is set (or `--n-gpu-layers` / `--gpu-layers`), only the first N layers and their KV cache reside on GPU. The dashboard scales weight and KV cache estimates by `min(ngl, layers) / layers`. mmproj, draft models, and runtime overhead are not scaled — they stay fully on GPU.

When offloading is active, the Static line shows `(ngl N/L)` to indicate the partial offload ratio.

### Model-Specific Adjustments

| Model / Family | Behavior | Dashboard handling |
|---------------|----------|-------------------|
| **Qwen 3.5 / 3.6** | Only Gated-Attention layers hold KV (DeltaNet layers use linear attention) | `effective_layers` from `QWEN_HYBRID_LAYERS`; fixed DeltaNet state added for non-KV layers |
| **Nemotron-5 / H** | ~8% of layers are attention; rest are Mamba-2 (fixed SSM state) | `effective_layers` from `NEMOTRON_ATTENTION_LAYERS`; Mamba state added for non-attention layers |
| **Llama 4** | All layers hold KV; iRoPE only changes attention mask / chunking | Full layer count — no reduction |
| **DeepSeek V3/R1, Kimi K2, Mistral Large 3** | MLA (Multi-head Latent Attention) — compressed KV cache | Flat ~70 KB/token estimate, scaled by quantization. Distill models use standard GQA. |
| **Gemma** | Sliding window attention reduces cache for local layers | Window size from `GEMMA_ISWA_WINDOW`; Gemma 4 halves global layer cache when E2B/E4B heads are active |
| **MTP / Speculative Decoding** | Bundled MTP adds minimal overhead; separate draft models use full formula | Bundled MTP: single-layer per head. Separate draft: full KV × `spec_draft_n`. MLA MTP shares main cache. |

### Runtime Overhead

Context buffers, compute buffers, flash attention scratch, and tensor-parallel sync are estimated separately and shown under **Runtime**. These stay on GPU regardless of `-ngl`.

### Caveats

- Estimates assume full context; hybrid sliding window models may use less once context exceeds the window limit.
- Layer sizes vary (especially MoE/hybrid models) — the per-layer ratio is an approximation.
- `nvidia-smi` / `amd-smi` values are the authoritative measurement; the dashboard provides a useful planning estimate.

## Keyboard

| Key | Action |
|-----|--------|
| **Ctrl+C** | Exit |
| **Ctrl+R**, **c**, **r** | Reset chart history |
| **+**, **=** | Show more prompts in log |
| **-**, **_** | Show fewer prompts in log |

## License

MIT
