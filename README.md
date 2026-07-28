# llama-swap Dashboard

Live GPU and inference monitor for llama-swap (with optional Ollama auxiliary model).

A clean terminal dashboard that shows real-time GPU stats, model VRAM estimates, decode performance, and recent prompts. Auto-detects NVIDIA, AMD, or mixed systems and applies a matching theme.

> Live memory from `nvidia-smi` / `amd-smi` is always authoritative. The built-in VRAM estimate is a planning aid.

## Requirements

- Python 3.8+ (stdlib only — no extra packages)
- NVIDIA and/or AMD GPU with working drivers
- llama-swap (or plain llama-server) running

## Quick Start

```bash
python dashboard.py
```

Optional: place the script next to your llama-swap `config.yaml` for richer model name and quant detection.

## What It Shows

- **GPU panel** — Temperature, VRAM, utilization, power, and fan for every detected card (scales to any number of GPUs)
- **System RAM** — Host memory usage, with CPU-offloaded model portion when partial offload is active
- **Model VRAM estimate** — Weights + KV cache + runtime overhead, broken into Static and Runtime
- **Tensor split** — Per-GPU share when `-ts` / `--tensor-split` is used
- **Decode performance** — Speed by context length (p50 / p90)
- **Recent prompts** — Rolling log with decode/prompt speeds, tokens, and cache hits
- **Speculative decoding** — Live acceptance rate when active (MTP, draft models, DFlash, etc.)
- **Session totals** — Cumulative input / output tokens and request count

<img width="519" height="1209" alt="image" src="https://github.com/user-attachments/assets/6c41e8f6-ac79-4f92-be94-42d7a234c56e" />

## Auto-Detection

| Situation              | Behavior                          |
|------------------------|-----------------------------------|
| NVIDIA only            | Green theme, `nvidia-smi`         |
| AMD only               | Orange theme, `amd-smi`           |
| Both present (mixed)   | Neutral theme, polls both SMIs    |
| Vulkan / mixed compute | Lower runtime overhead path used  |

## Model Fit View

Press **F** to switch to a focused Model Fit Calculator screen:

- Full breakdown (weights, KV, mmproj, draft, runtime)
- Offload summary (GB on CPU + ngl ratio)
- Live System RAM
- Live GPU VRAM bars with tensor-split percentages

Press **F** again to return to the main dashboard.

<img width="527" height="1284" alt="image" src="https://github.com/user-attachments/assets/db4ef238-74bd-4db1-a808-50fc853ca622" />


## Configuration

On first run the dashboard creates `dashboard.conf`:

```ini
host=http://localhost:8080
config_yaml=config.yaml
aux_port=11434
```

Command-line options:

```bash
python dashboard.py --host http://localhost:9090
python dashboard.py --refresh 2
python dashboard.py --help
```

## VRAM Estimation

The dashboard builds an estimate from:

- Actual GGUF file size(s) (main model, mmproj, draft)
- Architecture-aware KV cache calculation
- Runtime overhead (context, compute buffers, flash-attn, multi-GPU sync)
- Partial offload (`-ngl`) scaling
- `--cache-ram` cap when present

Architecture tables cover the common families (Qwen, Llama, Gemma, DeepSeek, Laguna, Nemotron, Mixtral, and others). Unknown models fall back to live SMI readings instead of a calculated estimate.

The estimate is useful for planning and for understanding how flags affect memory. **Always treat live `nvidia-smi` / `amd-smi` numbers as truth.**

## Keyboard

| Key            | Action                        |
|----------------|-------------------------------|
| **F**          | Toggle Model Fit Calculator   |
| **+** | Show more recent prompts     |
| **-** | Show fewer recent prompts    |
| **Ctrl+R**     | Reset chart history |
| **Ctrl+C**     | Exit                          |

## Notes

- Works with both llama-swap and plain llama-server
- Multi-GPU and heterogeneous (NVIDIA + AMD) setups are supported
- Future model architectures may need table entries for best estimate accuracy.

## License

MIT
