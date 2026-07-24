# GPU Dashboard

Live GPU and inference monitor for llama-swap with optional Ollama auxiliary model.

A clean terminal dashboard that monitors GPU stats and inference performance in real time. Available in two versions — NVIDIA and AMD — that share the same design and feature set.

## Versions

| Version | Script | GPU Backend | Color Theme |
|---------|--------|-------------|-------------|
| NVIDIA | `dashboardnv.py` | `nvidia-smi` | Cyan / Green |
| AMD | `dashboardamd.py` | `amd-smi` (ROCm) | Orange |

## Requirements

- **Python 3** (no extra packages — stdlib only)
- **NVIDIA** or **AMD GPU** with drivers installed
- **llama-swap** running on localhost (default port 8080)

## Quick Start

Double-click the script you need, or run from a terminal:

### Windows

- Double-click `dashboardnv.py` — NVIDIA
- Double-click `dashboardamd.py` — AMD

### Linux / macOS

```bash
# NVIDIA
python dashboardnv.py

# AMD
python dashboardamd.py
```

Place the dashboard scripts in the same folder as your llama-swap `config.yaml` (optional — required for model name, quantization, and VRAM calculation features).

The dashboard creates its own `dashboard.conf` file automatically on first run. You don't need to edit or touch your `config.yaml` — the dashboard just reads it.

The dashboard auto-detects llama-swap on first run. If it can't reach `localhost:8080`, it will prompt you to enter the correct host. Your setting is saved to `dashboard.conf` and reused on future runs.

## Configuration

`dashboard.conf` is created automatically. You can edit it directly or delete it to reset:

```ini
# GPU Dashboard configuration
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
python dashboardnv.py --host http://localhost:9090

# Change refresh interval (default: 2 seconds)
python dashboardnv.py --refresh 5

# See help
python dashboardnv.py --help
```

## What It Shows

- **GPU Status** — Real-time temp, VRAM usage, utilization, power draw, and fan speed for every GPU detected. Works with 1 GPU or 8+ — scales automatically to whatever hardware you have.
- **System RAM** — Host memory usage from llama-swap
- **Model VRAM** — Clean additive estimate: model weights + KV cache
- **t/s by Context Size** — Average prompt and decode speed grouped by input token range
- **Last Prompts** — Rolling log of the 3 most recent inference requests with decode speed, prompt speed, input/output tokens, and cache hit count
- **Session Tokens** — Cumulative input, output, and request count for the active model

## Model VRAM Calculation

The dashboard estimates total VRAM usage by combining model weights with a calculated KV cache size — no guessing.

It works by looking up each model's architecture (layers, KV heads, head dimension) in a built-in table covering Qwen, Llama, Gemma, DeepSeek, and Ornith families. Then it multiplies:

```
cache = 2 × layers × kv_heads × head_dim × cache_bytes × tokens
```

The cache bytes come from the model's quantization level (e.g. Q4_K_M = 0.5, Q8_0 = 1.0) or an explicit `-ctk` flag. This gives you a real-time breakdown of how much VRAM the model weights and active context are actually consuming.

## Keyboard

Press **Ctrl+C** to exit.

## License

MIT
