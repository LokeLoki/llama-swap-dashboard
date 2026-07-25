# llama-swap Dashboard

Live GPU and inference monitor for llama-swap with optional Ollama auxiliary model.

A clean terminal dashboard that monitors GPU stats and inference performance in real time. Auto-detects NVIDIA or AMD GPUs at startup and applies a matching color theme — no separate scripts needed.

## Requirements

- **Python 3** (no extra packages — stdlib only)
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

Accurate VRAM estimation with built-in architecture tables for: **Gemma family**, **Qwen family**, **Llama family**, **GLM 5.2**, **Kimi K2**, **Laguna 2.1**, **DeepSeek**, **Ornith**, **Bonsai**, **Mixtral**, and more.

Models not in the table fall back to safe default estimates — no crashes, no wrong numbers. Add your own by editing `MODEL_ARCHITECTURES` in the script.

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
- **Model VRAM** — Clean additive estimate: model weights + KV cache with left-aligned activity spinners
- **Decode t/s by Context Bucket** — Shows decode speed across context length ranges. Reveals where throughput degrades as context grows. Solid bar shows median (p50), dim tail shows p90 spread, with total output tokens
- **Last Prompts** — Rolling log of the 3 most recent inference requests with decode speed, prompt speed, input/output tokens, and cache hit count
- **Session Tokens** — Cumulative input, output, and request count for the active model

## Model VRAM Calculation (EXPERIMENTAL)

The dashboard estimates total VRAM usage by combining model weights with a calculated KV cache size. When multimodal (`--mmproj`) or speculative decoding (`--model-draft` / MTP) are used, the exact file sizes of those models are included too. The `--cache-ram` flag is respected — KV cache is capped to the specified limit when set.

The core formula starts with a model's architecture (layers, KV heads, head dimension) from a built-in table, then multiplies:

```
cache = 2 × layers × kv_heads × head_dim × cache_bytes × tokens
```

This estimate is adjusted for model-specific behaviors:

- **Qwen 3.x / DeltaNet** — Effective layers are reduced by 4× to account for DeltaNet's recurrent state design
- **Gemma** — Sliding window attention reduces cache for local layers; Gemma 4 halves global layer cache when E2B/E4B heads are active
- **DeepSeek / Kimi (MLA)** — Uses a flat ~70 KB/token estimate at FP16/BF16, scaled by quantization (q8_0 halves it). Distill models use standard GQA instead
- **MTP / Speculative Decoding** — Bundled MTP adds minimal overhead (single-layer per head). Separate draft models use the full formula. MLA MTP shares the main cache

**Caveats:** The formula estimates maximum cache at full context. Models with hybrid sliding window attention may use less once context exceeds the sliding window limit.

## Keyboard

| Key | Action |
|-----|--------|
| **Ctrl+C** | Exit |
| **Ctrl+R** | Reset chart |
| **+** | Show more prompts in log |
| **-** | Show fewer prompts in log |

## License

MIT
