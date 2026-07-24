#!/usr/bin/env python3
"""
GPU Inference Dashboard for llama-swap (ROCm/AMD).
Polls amd-smi + /api/performance + /api/metrics for a clean terminal view.
Press Ctrl+C to exit.

Usage:
    python dashboard.py [OPTIONS]

Options:
    --host HOST       llama-swap proxy URL (default: auto-detect from dashboard.conf, then localhost:8080)
    --refresh SECS    Refresh interval in seconds (default: 2)
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

@dataclasses.dataclass
class GpuStats:
    id: int
    name: str
    temp_c: int
    gpu_util_pct: int
    mem_used_mb: int
    mem_total_mb: int
    fan_pct: int
    power_w: float

@dataclasses.dataclass
class SystemInfo:
    mem_used_mb: int
    mem_total_mb: int

@dataclasses.dataclass
class AuxiliaryModel:
    name: str
    size_vram_mb: float
    context_length: int
    decode_tps: float

@dataclasses.dataclass
class MainModelVram:
    total_mb: float
    weight_mb: float
    mmproj_mb: float
    draft_mb: float
    cache_mb: float
    cache_type: str

@dataclasses.dataclass
class ModelIdentity:
    model_id: str
    quant: str

DEFAULT_HOST = "http://localhost:8080"
DEFAULT_REFRESH = 2
DEFAULT_AUX_PORT = "11434"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.conf")

# Quantization pattern regex — matches common GGUF quant labels
# Handles: Q4_K_M, Q5_K_XL, Q6_K, IQ4_XS, F16, BF16, etc.
QUANT_PATTERN = re.compile(
    r"(Q\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*|"
    r"IQ\d+_[A-Z]+|"
    r"F16|BF16)",
    re.IGNORECASE,
)

TOKEN_BUCKETS = [
    ("0-10k", 0, 9999),
    ("10-20k", 10000, 19999),
    ("20-30k", 20000, 29999),
    ("30-40k", 30000, 39999),
    ("40-50k", 40000, 49999),
    ("50-60k", 50000, 59999),
    ("60-70k", 60000, 69999),
    ("70-80k", 70000, 79999),
    ("80k+", 80000, 999999),
]

# ── Model architecture lookup ──────────────────────────
# Exact params for KV cache calculation per model family.
# Format: (layers, kv_heads, head_dim)
# Cache per token = 2 * layers * kv_heads * head_dim * cache_bytes
#
# Key = family substring to match against model name/cmd (case-insensitive).
# Add more models here as needed — each entry is 3 numbers.
MODEL_ARCHITECTURES = {
    # Qwen 3.6 / 3.5 (hybrid Gated DeltaNet + Gated Attention)
    "qwen3.6-27b":   (64, 4, 256),
    "qwen3.5-27b":   (64, 4, 256),
    "qwen3.5-9b":    (48, 4, 256),
    "qwen3.5-8b":    (48, 4, 256),
    # Qwen 3.5 MoE
    "qwen3.5-35b-a3b":    (48, 4, 256),
    "qwen3.5-122b-a10b":  (64, 8, 256),
    "qwen3.5-397b-a17b":  (72, 8, 256),
    # Ornith (Qwen3.5-based)
    "ornith":        (48, 4, 256),
    # Qwen 3 (dense)
    "qwen3-32b":     (64, 8, 128),
    "qwen3-14b":     (40, 8, 128),
    "qwen3-8b":      (36, 8, 128),
    "qwen3-4b":      (36, 8, 128),
    # Qwen 3 MoE
    "qwen3-30b-a3b": (48, 4, 128),
    "qwen3-235b-a22b": (94, 4, 128),
    # Qwen 2.5
    "qwen2.5-72b":   (80, 8, 128),
    "qwen2.5-32b":   (64, 8, 128),
    "qwen2.5-14b":   (48, 8, 128),
    "qwen2.5-7b":    (28, 4, 128),
    "qwen2.5-3b":    (36, 2, 128),
    "qwen2.5-1.5b":  (28, 2, 128),
    "qwen2.5-0.5b":  (24, 2, 128),
    # Llama 3.1
    "llama3.1-405b": (126, 8, 128),
    "llama3.1-70b":  (80, 8, 128),
    "llama3.1-8b":   (32, 8, 128),
    # Gemma 2
    "gemma2-27b":    (46, 16, 128),
    "gemma2-9b":     (42, 8, 256),
    "gemma2-2b":     (26, 4, 256),
    # Gemma 4 (hybrid sliding/global attention — use sliding layer values for KV cache)
    "gemma4-e4b":    (42, 8, 256),
    "gemma4-e2b":    (35, 8, 256),
    "gemma4-12b":    (48, 8, 256),
    "gemma4-31b":    (60, 8, 256),
    "gemma4-26b-a4b": (30, 8, 256),
    # Bonsai 27B (binary/ternary quantization of Qwen3.6-27B — architecture unchanged)
    "bonsai":        (64, 4, 256),
    # DeepSeek
    "deepseek-v3":   (61, 128, 128),
    "deepseek-r1":   (61, 128, 128),
    # Laguna family (GQA)
    "laguna-s-2.1":      (48, 8, 128),
    "laguna-xs-2.1":     (40, 8, 128),
    # GLM 5.2 (MHA)
    "glm-5.2":           (78, 64, 64),
    # Kimi K2 (MLA — follows DeepSeek pattern)
    "kimi-k2":           (61, 128, 128),
    # Mistral family
    "mistral-7b":        (32, 8, 128),
    "mixtral-8x7b":      (32, 8, 128),
    "mixtral-8x22b":     (56, 8, 128),
    "codestral":         (40, 8, 128),
    # Llama 3.2 / 3.3
    "llama3.3-70b":      (80, 8, 128),
    "llama3.3-8b":       (32, 8, 128),
    "llama3.2-3b":       (28, 8, 128),
    "llama3.2-1b":       (16, 4, 64),
    # Phi family
    "phi-4":             (40, 40, 128),
    "phi-3.5":           (32, 32, 96),
    "phi-3":             (32, 32, 96),
    # Command-R family
    "command-r-plus":    (64, 8, 128),
    "command-r":         (32, 8, 128),
    # Yi family
    "yi-34b":            (60, 8, 128),
    "yi-9b":             (32, 8, 128),
    # Exaone
    "exaone-3.5":        (64, 8, 128),
    # SmollM2
    "smollm2":           (24, 8, 64),
}

# Quantization bytes-per-element for KV cache.
# Maps quant label → cache bytes. When --cache-type is set, that overrides.
QUANT_CACHE_BYTES = {
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0,
    "q6_k":   0.75,
    "q5_k_m": 0.5,
    "q5_k_s": 0.5,
    "q5_0":   0.5,
    "q4_k_m": 0.5,
    "q4_k_s": 0.5,
    "q4_0":   0.5,
    "iq4_xxs": 0.25,
    "iq4_xs":  0.5,
    "q3_k_m":  0.375,
    "q2_k":    0.25,
    # Bonsai 27B quantizations (1-bit and 1.58-bit ternary)
    "q1_0":    0.5,
    "q2_0":    0.5,
}

# Gemma iSWA (Interleaved Sliding Window Attention): every other layer only
# caches a fixed window. Window sizes per model family.
GEMMA_ISWA_WINDOW = {
    "gemma2-27b": 4096,
    "gemma2-9b":  4096,
    "gemma2-2b":  4096,
    "gemma4-e4b": 512,
    "gemma4-e2b": 512,
    "gemma4-12b": 1024,
    "gemma4-31b": 1024,
    "gemma4-26b-a4b": 1024,
}

# Qwen 3.5/3.6 hybrid attention: 3:1 DeltaNet:GatedAttn ratio.
# Only 25% of layers carry KV cache (DeltaNet is linear attention, no KV).
QWEN_HYBRID_LAYERS = {
    "qwen3.6-27b":   16,   # 64 total → 16 GatedAttn
    "qwen3.5-27b":   16,
    "qwen3.5-9b":    12,   # 48 total → 12 GatedAttn
    "qwen3.5-8b":    12,
    # Qwen 3.5 MoE
    "qwen3.5-35b-a3b":    12,   # 48 total → 12 GatedAttn
    "qwen3.5-122b-a10b":  16,   # 64 total → 16 GatedAttn
    "qwen3.5-397b-a17b":  18,   # 72 total → 18 GatedAttn
    # Bonsai 27B (same Qwen3.6-27B architecture)
    "bonsai":        16,
}

RESET = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
CYAN = "\033[33m"
GREEN = "\033[92m"
ORANGE = "\033[33m"
LIGHT_GREEN = "\033[1;92m"  # Bold bright green for decode values
LIGHT_ORANGE = "\033[1;33m"  # Bold bright orange for decode values
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[90m"
WHITE = "\033[97m"
SOFT_WHITE = "\033[37m"


# ── Config ─────────────────────────────────────────────

def load_config():
    """Load settings from dashboard.conf (simple key=value format)."""
    config = {}
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        config[key.strip()] = value.strip()
        except Exception:
            pass
    return config


def save_config(host, config_yaml="", aux_port=DEFAULT_AUX_PORT):
    """Save host to dashboard.conf."""
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write("# GPU Dashboard configuration\n")
            f.write("#\n")
            f.write("# host            - llama-swap API URL (required)\n")
            f.write("# config_yaml     - path to llama-swap config.yaml (optional, for model name + quant parsing)\n")
            f.write("#                   Leave blank to skip quant parsing\n")
            f.write("# aux_port        - Ollama auxiliary model port (default: 11434)\n")
            f.write("\n")
            f.write(f"host={host}\n")
            f.write(f"config_yaml={config_yaml}\n")
            f.write(f"aux_port={aux_port}\n")
    except Exception:
        pass


def get_config_yaml(config):
    """Get the config.yaml path from dashboard.conf, resolving relative paths."""
    cfg_path = config.get("config_yaml", "").strip().strip("\"'")
    if not cfg_path:
        return None
    # Resolve relative paths against the dashboard script directory
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_path)
    return cfg_path if os.path.isfile(cfg_path) else None


def get_aux_port(config):
    """Get the auxiliary (Ollama) port from config."""
    try:
        port = int(config.get("aux_port", DEFAULT_AUX_PORT))
        return port
    except (ValueError, TypeError):
        return int(DEFAULT_AUX_PORT)


def check_host(host):
    """Check if llama-swap API is reachable at the given host."""
    try:
        url = f"{host}/api/performance"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def parse_cli():
    """Parse --host and --refresh from command line args."""
    host = None
    refresh = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--refresh" and i + 1 < len(args):
            try:
                refresh = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1
    return host, refresh


def resolve_host(cli_host, config=None):
    """Determine the llama-swap host: CLI > config file > interactive > default."""
    if config is None:
        config = load_config()

    if cli_host:
        # CLI override
        if check_host(cli_host):
            save_config(cli_host)
            print(f"Connected to {cli_host} ✓")
            return cli_host
        print(f"llama-swap not reachable at {cli_host}")
        sys.exit(1)

    if config.get("host"):
        # Saved config
        saved_host = config["host"]
        if check_host(saved_host):
            return saved_host
        # Stale config — ask user
        print(f"llama-swap not reachable at saved host: {saved_host}")
        print(f"Enter new host (e.g., http://localhost:9090) or press Enter for default [{DEFAULT_HOST}]: ", end="")
        user_input = sys.stdin.readline().strip()
        new_host = user_input if user_input else DEFAULT_HOST
        if check_host(new_host):
            save_config(new_host)
            print(f"Connected to {new_host} ✓")
            return new_host
        print(f"Could not connect to {new_host}. Using default.")

    # No config file yet — try default first
    if check_host(DEFAULT_HOST):
        save_config(DEFAULT_HOST)
        return DEFAULT_HOST

    # Default didn't work — interactive prompt
    print(f"llama-swap not reachable at {DEFAULT_HOST}")
    print(f"Enter host (e.g., http://localhost:9090) or press Enter to continue anyway: ", end="")
    user_input = sys.stdin.readline().strip()
    new_host = user_input if user_input else DEFAULT_HOST
    if check_host(new_host):
        save_config(new_host)
        print(f"Connected to {new_host} ✓")
    return new_host


def build_urls(host):
    """Build API URLs from host."""
    base = host.rstrip("/")
    return f"{base}/api/performance", f"{base}/api/metrics"


# ── GPU stats (ROCm/AMD) ───────────────────────────────

def short_gpu_name(name):
    """Extract a short GPU name from the full amd-smi name.
    E.g. 'AMD Radeon RX 7900 XTX' -> 'RX 7900 XTX'
         'AMD Instinct MI300X' -> 'MI300X'
    Falls back to the full name if no Radeon/Instinct found."""
    if "Radeon RX" in name:
        return name.split("Radeon RX", 1)[-1].strip()
    if "Instinct" in name:
        return name.split("Instinct", 1)[-1].strip()
    return name


def get_amd_gpu_names():
    """Query amd-smi list once at startup to cache GPU names."""
    try:
        list_result = subprocess.run(
            ["amd-smi", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        list_data = json.loads(list_result.stdout)
        gpu_names = {}
        for gpu in list_data.get("gpu_data", []):
            idx = gpu.get("gpu", 0)
            name = (
                gpu.get("name")
                or gpu.get("gpu_name")
                or gpu.get("part_number", "AMD GPU")
            )
            gpu_names[idx] = name
        return gpu_names
    except Exception:
        return {}


def get_amd_smi(gpu_names=None):
    """Query amd-smi for current GPU stats.
    Uses JSON output for reliable parsing:
      amd-smi metric --usage --power --temperature --mem-usage --fan --json
    GPU names are cached from a startup call to amd-smi list --json to avoid
    redundant subprocess overhead on every loop iteration.
    """
    if gpu_names is None:
        gpu_names = {}
    try:
        # Get metrics (usage, power, temperature, memory, fan)
        metric_result = subprocess.run(
            [
                "amd-smi", "metric",
                "--usage", "--power", "--temperature",
                "--mem-usage", "--fan",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        metric_data = json.loads(metric_result.stdout)
        gpus = []
        for gpu in metric_data.get("gpu_data", []):
            idx = gpu.get("gpu", 0)

            # GFX activity (GPU utilization)
            usage = gpu.get("usage", {})
            gfx_activity = 0
            gfx_val = usage.get("GFX_ACTIVITY")
            if gfx_val:
                gfx_activity = int(str(gfx_val).rstrip("%"))

            # Temperature (Edge)
            temp = 0
            temp_data = gpu.get("temperature", {})
            edge_val = temp_data.get("EDGE")
            if edge_val and str(edge_val).upper() != "N/A":
                temp = int(str(edge_val).replace("°C", "").strip())

            # Memory usage
            mem_usage = gpu.get("mem_usage") or {}
            mem_used_mb = int(str(mem_usage.get("USED_VRAM", 0)).replace("MB", "").strip())
            mem_total_mb = int(str(mem_usage.get("TOTAL_VRAM", 0)).replace("MB", "").strip())

            # Power
            power = 0.0
            power_data = gpu.get("power", {})
            sock_power = power_data.get("SOCKET_POWER")
            if sock_power:
                power = float(str(sock_power).replace("W", "").strip())

            # Fan speed
            fan = 0
            fan_data = gpu.get("fan", {})
            fan_val = fan_data.get("SPEED")
            if fan_val and str(fan_val).upper() != "N/A":
                fan = int(str(fan_val).rstrip("%"))

            gpus.append(GpuStats(
                id=idx,
                name=short_gpu_name(gpu_names.get(idx, f"AMD GPU {idx}")),
                temp_c=temp,
                gpu_util_pct=gfx_activity,
                mem_used_mb=mem_used_mb,
                mem_total_mb=mem_total_mb,
                fan_pct=fan,
                power_w=power,
            ))
        return gpus
    except Exception:
        return None


def util_bar(pct, width=16):
    """Draw a simple ASCII bar."""
    filled = round(pct / 100 * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return bar


def color_temp(temp):
    """Color-code temperature."""
    if temp <= 67:
        return GREEN
    elif temp <= 75:
        return YELLOW
    return RED


def format_duration(ms):
    """Format milliseconds to human-readable."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms}ms"


def format_time(ts_str):
    """Format ISO timestamp to short time string."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts_str[-8:]


# ── llama-swap API ─────────────────────────────────────

def get_llama_swap_stats(api_url):
    """Get latest system stats from /api/performance."""
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        sys_stats = data.get("sys_stats", [])
        if sys_stats:
            latest = sys_stats[-1]
            return SystemInfo(
                mem_used_mb=latest.get("mem_used_mb", 0),
                mem_total_mb=latest.get("mem_total_mb", 0),
            )
    except Exception:
        pass
    return None


def get_auxiliary_model(aux_port=DEFAULT_AUX_PORT):
    """Get auxiliary model info from Ollama /api/ps and /api/generate (timing probe)."""
    aux_host = f"http://127.0.0.1:{aux_port}"
    try:
        # Get loaded model
        req = urllib.request.Request(f"{aux_host}/api/ps")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            return None
        m = models[0]

        # Quick timing probe — only if not recently probed (cached in get_auxiliary_model._cache)
        now = time.time()
        if not hasattr(get_auxiliary_model, "_cache") or (now - get_auxiliary_model._cache["time"]) > 30:
            probe_data = json.dumps({
                "model": m.get("name", "Ollama Aux"),
                "prompt": "What color is the sky?",
                "stream": False,
            }).encode()
            try:
                probe = urllib.request.Request(f"{aux_host}/api/generate", data=probe_data, method="POST")
                with urllib.request.urlopen(probe, timeout=30) as resp:
                    probe_resp = json.loads(resp.read())
                decode_tps = probe_resp.get("eval_count", 0) / (probe_resp.get("eval_duration", 1) / 1e9)
                get_auxiliary_model._cache = {"time": now, "decode_tps": decode_tps}
            except Exception:
                decode_tps = 0
        else:
            decode_tps = get_auxiliary_model._cache.get("decode_tps", 0)

        return AuxiliaryModel(
            name=m.get("name", "—"),
            size_vram_mb=m.get("size_vram", 0) / (1024 * 1024),
            context_length=m.get("context_length", 0),
            decode_tps=decode_tps,
        )
    except Exception:
        pass
    return None


def fetch_running_models(host):
    """Fetch running models from llama-swap /running endpoint.
    Returns a list of dicts with model info and parsed cmd flags."""
    try:
        url = f"{host.rstrip('/')}/running"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        running = []
        for item in data.get("running", []):
            cmd = item.get("cmd", "")
            # Parse model path from -m "path/to/model.gguf"
            model_path = ""
            m_match = re.search(r'-m\s+"([^"]+\.gguf)"', cmd)
            if m_match:
                model_path = m_match.group(1)
            # Parse quant from model path
            model_quant = parse_quant_from_path(model_path)
            # Parse --model file size from amd-smi or gguf header
            # We'll get file size from the path
            model_file_mb = 0
            if model_path:
                try:
                    model_file_mb = os.path.getsize(model_path) / (1024 * 1024)
                except OSError:
                    pass
            # Parse cache type from -ctk flag
            cache_type = None
            ctk_match = re.search(r'-ctk\s+(\S+)', cmd)
            if ctk_match:
                cache_type = ctk_match.group(1).lower()
            # Parse max context from -c flag
            max_context = 0
            ctx_match = re.search(r'\s-c\s+(\d+)', cmd)
            if ctx_match:
                try:
                    max_context = int(ctx_match.group(1))
                except ValueError:
                    pass
            # Parse --mmproj path from cmd
            mmproj_path = ""
            mmproj_match = re.search(r'--mmproj\s+"([^"]+\.gguf)"', cmd)
            if mmproj_match:
                mmproj_path = mmproj_match.group(1)
            # Get mmproj file size
            mmproj_file_mb = 0
            if mmproj_path:
                try:
                    mmproj_file_mb = os.path.getsize(mmproj_path) / (1024 * 1024)
                except OSError:
                    pass
            # Parse --model-draft path from cmd
            draft_path = ""
            draft_match = re.search(r'--model-draft\s+"([^"]+\.gguf)"', cmd)
            if draft_match:
                draft_path = draft_match.group(1)
            # Get draft file size
            draft_file_mb = 0
            if draft_path:
                try:
                    draft_file_mb = os.path.getsize(draft_path) / (1024 * 1024)
                except OSError:
                    pass
            # Parse --cache-ram cap (in MB)
            cache_ram_mb = -1  # -1 = not set (unlimited on GPU)
            cram_match = re.search(r'--cache-ram\s+(\d+)', cmd)
            if cram_match:
                try:
                    cache_ram_mb = int(cram_match.group(1))
                except ValueError:
                    pass
            # Parse --parallel (number of server slots)
            parallel = 1
            np_match = re.search(r'(?<!\w)-np\s+(\d+)', cmd)
            if not np_match:
                np_match = re.search(r'--parallel\s+(\d+)', cmd)
            if np_match:
                try:
                    parallel = int(np_match.group(1))
                except ValueError:
                    pass
            # Parse ALL flags generically from cmd
            all_flags = {}
            for match in re.finditer(r'(?:^|\s)((?:--|-[a-zA-Z])[a-zA-Z0-9_-]*\s+(?:"[^"]*"|[^\s]+)|(?:--|-[a-zA-Z])[a-zA-Z0-9_-]*(?:\s|$))', cmd):
                flag_str = match.group(1).strip()
                parts = flag_str.split(None, 1)
                flag_name = parts[0]
                flag_value = parts[1] if len(parts) > 1 else None
                if flag_name in ('llama-server.exe', 'llama-server'):
                    continue
                if flag_value and flag_value.startswith('"') and flag_value.endswith('"'):
                    flag_value = flag_value.strip('"')
                if flag_value and flag_value.endswith('.gguf'):
                    flag_value = os.path.basename(flag_value)
                all_flags[flag_name] = flag_value
            running.append({
                "model_id": item.get("model", ""),
                "state": item.get("state", ""),
                "cmd": cmd,
                "model_path": model_path,
                "model_quant": model_quant,
                "model_file_mb": model_file_mb,
                "cache_type": cache_type,
                "max_context": max_context,
                "has_spec": "--spec-type" in cmd,
                "spec_draft_n_max": _parse_spec_draft_n_max(cmd),
                "mmproj_path": mmproj_path,
                "mmproj_file_mb": mmproj_file_mb,
                "draft_path": draft_path,
                "draft_file_mb": draft_file_mb,
                "cache_ram_mb": cache_ram_mb,
                "parallel": parallel,
                "all_flags": all_flags,
            })
        return running
    except Exception:
        return None


def short_model_name(model_path_or_id):
    """Generate a short model alias from a model path or ID.
    Dynamic rules — no hardcoding:
    - Family first letter(s) + param count (e.g., q27, b27, ge4)
    - MoE: append MoE params (e.g., q35a3, g26a4, o35a3)
    - Gemma E-variants: ge4, ge2, g12, g31
    - DeepSeek R: dr14, dr33
    Returns the short alias or the original if nothing matches."""
    text = model_path_or_id.lower()
    # Strip .gguf extension first
    text = re.sub(r'\\.gguf$', '', text)
    # Remove common suffixes that don't affect the name
    text = re.sub(r'[-_](it|chat|instruct|ud|abliterated|heretic|uncensored|qat|code|mt)', '', text)
    # Remove quantization tags (e.g., -Q4_K_M, -Q8_0, -F16)
    text = re.sub(r'[-_](q\\d+_?k_?[a-z]*|q\\d+_?\\d*|iq\\d+_?[a-z]*|f16|bf16)', '', text)

    # Detect family (first meaningful word or known prefix)
    family = ""
    if 'gemma' in text or 'gemma4' in text or 'gemma2' in text:
        family = 'g'
    elif 'qwen' in text:
        family = 'q'
    elif 'bonsai' in text:
        family = 'b'
    elif 'ornith' in text:
        family = 'o'
    elif 'deepseek' in text:
        family = 'd'
    elif 'llama' in text:
        family = 'll'
    elif 'laguna' in text:
        family = 'la'
    elif 'glm' in text:
        family = 'gl'
    elif 'kimi' in text:
        family = 'k'
    elif 'mixtral' in text:
        family = 'mx'
    elif 'yi' in text:
        family = 'yi'
    elif 'commandr' in text or 'command-r' in text:
        family = 'c'
    elif 'phi' in text:
        family = 'phi'
    elif 'mistral' in text:
        family = 'm'
    elif 'nemotron' in text:
        family = 'n'
    elif 'internlm' in text:
        family = 'i'
    elif 'commandaura' in text or 'command-aura' in text:
        family = 'ca'
    elif 'aya' in text:
        family = 'aya'

    if not family:
        # Fallback: first letter of first word
        match = re.match(r'^([a-z]+)', text.split('/')[-1])
        if match:
            family = match.group(1)[:2]

    # Detect Gemma E-variant (E4B, E2B) — these use 'e' prefix
    e_var = re.search(r'e(\d+)b', text)
    if e_var and family == 'g':
        return f"Ge{e_var.group(1)}"

    # Detect total params XB or XT pattern (e.g., 27B, 35B, 1T)
    params = re.search(r'(\d+)b', text)
    tparams = re.search(r'(\d+)t', text)
    if params:
        param_str = params.group(1)

        # Detect MoE AxB pattern (e.g., A3B, A10B, A17B)
        moe = re.search(r'a(\d+)b', text)
        if moe:
            return f"{family[0].upper()}{family[1:]}{param_str}a{moe.group(1)}"

        # DeepSeek R variants
        if family == 'd' and 'r' in text[:15]:
            return f"Dr{param_str}"

        return f"{family[0].upper()}{family[1:]}{param_str}"
    if tparams:
        # Trillion-param models get "T" denotation
        t_str = tparams.group(1)

        moe = re.search(r'a(\d+)b', text)
        if moe:
            return f"{family[0].upper()}{family[1:]}{t_str}Ta{moe.group(1)}"

        return f"{family[0].upper()}{family[1:]}{t_str}T"

    # Handle models without "XB" or "XT" param suffix in filename
    # (e.g., Kimi-K2-Instruct-Q4_K_M.gguf, GLM-5.2-Instruct-Q4_K_M.gguf)
    if family == 'k':
        return "K1Ta32"  # Kimi K2 = 1T total, 32B activated
    if family == 'gl':
        return "Gl744a40"  # GLM-5.2 = 744B total, 40B activated
    if family == 'la':
        # Distinguish S-2.1 (118B-A8B) vs XS-2.1 (33B-A3B)
        if 'xs' in text:
            return "La33a3"
        return "La118a8"

    # Fallback: return original if pattern didn't match
    return model_path_or_id


def find_model_arch(model_path, model_quant):
    """Find the architecture params for a model by matching its path.
    Returns (layers, kv_heads, head_dim) or None."""
    if not model_path:
        return None
    # Normalize: strip hyphens, dots, underscores so "llama-3.1-70b" matches "llama3.1-70b"
    clean_path = re.sub(r'[-._]', '', model_path.lower())
    sorted_keys = sorted(MODEL_ARCHITECTURES.keys(), key=len, reverse=True)
    for family in sorted_keys:
        clean_family = re.sub(r'[-._]', '', family.lower())
        if clean_family in clean_path:
            return MODEL_ARCHITECTURES[family]
    return None


def calc_kv_cache_mb(layers, kv_heads, head_dim, cache_bytes, num_tokens, iswa_window=None, effective_layers=None, gemma4_kv=False):
    """Calculate KV cache size in MB.
    Formula: 2 * layers * kv_heads * head_dim * cache_bytes * tokens / 1MB
    With iSWA: effective layers = layers/2 + layers/2 * min(ctx/window, 1)
    With effective_layers: overrides 'layers' for KV-bearing layers (e.g. Qwen 3.5/3.6 DeltaNet).
    With gemma4_kv: Gemma 4 global layers reuse keys as values → 50% reduction on global cache.
    Gemma 2: 1:1 global/sliding ratio (50% global).
    Gemma 3/4: 5:1 local/global ratio (~17% global). E2B uses 4:1."""
    # Determine which layers to use
    kv_layers = effective_layers if effective_layers is not None else layers
    if iswa_window is not None and iswa_window > 0:
        # Interleaved sliding window: most layers cache only the window, few cache full context
        # Gemma 2: 1:1 ratio → half global, half sliding
        # Gemma 3/4: 5:1 ratio → ~1/6 global, ~5/6 sliding (E2B uses 4:1)
        global_ratio = 6 if iswa_window <= 1024 else 2  # Gemma 3/4=6, Gemma 2=2
        global_layers = kv_layers // global_ratio
        sliding_layers = kv_layers - global_layers
        # Gemma 4 global layers use K=V (keys reused as values) → 50% reduction on global cache
        global_cache_ratio = 0.5 if gemma4_kv else 1.0
        effective_tokens = global_layers * num_tokens * global_cache_ratio + sliding_layers * min(num_tokens, iswa_window)
        bytes_total = 2 * kv_heads * head_dim * cache_bytes * effective_tokens
    else:
        bytes_total = 2 * kv_layers * kv_heads * head_dim * cache_bytes * num_tokens
    return bytes_total / (1024 * 1024)


def _parse_spec_draft_n_max(cmd):
    """Parse --spec-draft-n-max from command string. Returns 0 if not set."""
    m = re.search(r'--spec-draft-n-max\s+(\d+)', cmd)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0


def get_cache_bytes(cache_type, model_quant):
    """Determine bytes per element for KV cache.
    Uses explicit cache type if set, otherwise defaults to F16 (llama.cpp default)."""
    if cache_type and cache_type in QUANT_CACHE_BYTES:
        return QUANT_CACHE_BYTES[cache_type]
    # Default to F16 (2.0) — llama.cpp's default when -ctk is not set
    return 2.0


# Active state labels that rotate randomly when the model is working
ACTIVE_STATES = ["processing", "computing", "synthesizing", "generating", "reasoning"]


def get_inference_state(valid_metrics, gpus):
    """Detect if the model is currently active.
    Returns 'active' or 'idle'."""
    if not valid_metrics or not gpus:
        return "idle"
    active_gpus = [g for g in gpus if g.gpu_util_pct > 5]
    return "active" if active_gpus else "idle"


def get_aux_state(aux_info, aux_port):
    """Detect if the auxiliary model is currently active.
    Returns 'active' or 'idle'."""
    if not aux_info:
        return "idle"
    try:
        aux_host = f"http://127.0.0.1:{aux_port}"
        req = urllib.request.Request(f"{aux_host}/api/ps")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            return "idle"
        load = models[0].get("load", 0)
        return "active" if load > 0 else "idle"
    except Exception:
        pass
    return "idle"


def get_main_model_vram(running_models, valid_metrics):
    """Calculate main model VRAM: weights + mmproj + draft + KV cache (capped by --cache-ram).
    Returns MainModelVram or None."""
    if not running_models:
        return None
    # Find active model
    active = running_models[0]
    model_id = active.get("model_id", "")
    if not model_id:
        return None
    # Get architecture
    arch = find_model_arch(active["model_path"], active["model_quant"])
    if not arch:
        return None
    layers, kv_heads, head_dim = arch
    # DeepSeek R1/V3 use MLA (Multi-head Latent Attention) — compressed KV cache.
    # Standard formula wildly overestimates. Use flat ~70 KB/token instead.
    is_mla = ("deepseek" in active["model_path"].lower() or "kimi" in active["model_path"].lower()) and "distill" not in active["model_path"].lower()
    # Gemma iSWA: find matching window size
    path_lower = active["model_path"].lower()
    iswa_window = None
    for key, window in GEMMA_ISWA_WINDOW.items():
        clean_key = re.sub(r'[-._]', '', key.lower())
        clean_path = re.sub(r'[-._]', '', path_lower)
        if clean_key in clean_path:
            iswa_window = window
            break
    # Qwen 3.5/3.6 hybrid: only 25% of layers have KV cache (DeltaNet = linear attention)
    effective_layers = None
    for key, el in QWEN_HYBRID_LAYERS.items():
        clean_key = re.sub(r'[-._]', '', key.lower())
        if clean_key in clean_path:
            effective_layers = el
            break
    # Gemma 4: global layers reuse keys as values → 50% KV reduction on global cache
    gemma4_kv = iswa_window is not None and "gemma4" in path_lower
    # Get weights size
    weight_mb = active.get("model_file_mb", 0)
    if weight_mb == 0:
        return None
    # Get mmproj size (if --mmproj is set)
    mmproj_mb = active.get("mmproj_file_mb", 0)
    # Get draft model weight size (if --model-draft is set)
    draft_mb = active.get("draft_file_mb", 0)
    # Get reserved context size from --ctx-size (-c flag)
    ctx_size = active.get("max_context", 0)
    if ctx_size == 0:
        # Fallback: use active tokens from metrics if ctx-size not in cmd
        if valid_metrics:
            latest = valid_metrics[-1]
            input_tokens = latest.get("tokens", {}).get("input_tokens", 0)
            cache_tokens = latest.get("tokens", {}).get("cache_tokens", 0)
            ctx_size = cache_tokens + input_tokens
    # Get cache bytes
    cache_bytes = get_cache_bytes(active["cache_type"], active["model_quant"])
    # Calculate reserved KV cache (full --ctx-size budget)
    if is_mla:
        # MLA: ~70 KB/token at FP16/BF16 (compressed key/value + RoPE keys).
        # Scales linearly with cache quantization — q8_0 halves the cache, etc.
        mla_base_mb = 70.0 * ctx_size / (1024)
        cache_mb = mla_base_mb * (cache_bytes / 2.0)
    else:
        cache_mb = calc_kv_cache_mb(layers, kv_heads, head_dim, cache_bytes, ctx_size, iswa_window, effective_layers, gemma4_kv)
    # Apply --cache-ram cap if set (limits KV cache on GPU, rest spills to DRAM)
    cache_ram_cap = active.get("cache_ram_mb", -1)
    if cache_ram_cap > 0:
        cache_mb = min(cache_mb, cache_ram_cap)
    # MTP / draft KV cache: depends on whether this is bundled MTP (lightweight)
    # or a separate draft model (full architecture).
    spec_draft_n = active.get("spec_draft_n_max", 0)
    draft_cache_mb = 0.0
    if spec_draft_n > 0:
        if is_mla:
            # MLA MTP (DeepSeek-V3/R1): MTP heads share the main MLA KV cache.
            # MTP Eagle reuses the same KV slots; MTP Vanilla adds minimal overhead.
            # Keep draft_cache_mb near zero — the main cache already accounts for it.
            draft_cache_mb = 0.0
        elif draft_mb > 0:
            draft_cache_mb = calc_kv_cache_mb(layers, kv_heads, head_dim, cache_bytes, ctx_size, iswa_window, effective_layers, gemma4_kv) * spec_draft_n
        else:
            mtp_layers = min(spec_draft_n, 3)
            draft_cache_mb = calc_kv_cache_mb(mtp_layers, kv_heads, head_dim, cache_bytes, ctx_size, iswa_window, mtp_layers, gemma4_kv)
    # Build cache type string for display
    ct_display = active["cache_type"] or "f16"
    total_vram_mb = weight_mb + mmproj_mb + draft_mb + cache_mb + draft_cache_mb
    return MainModelVram(
        total_mb=total_vram_mb,
        weight_mb=weight_mb,
        mmproj_mb=mmproj_mb,
        draft_mb=draft_mb,
        cache_mb=cache_mb,
        cache_type=ct_display,
    )


def get_aux_vram(aux_info, aux_port):
    """Calculate auxiliary model VRAM: weights + KV cache estimate.
    Ollama exposes model details via /api/show. We parse architecture params
    directly from the response instead of using the lookup table.
    Returns total_vram_mb or falls back to size_vram_mb if details unavailable.
    Caches result to avoid repeated API calls."""
    if not aux_info:
        return None
    weight_mb = aux_info.size_vram_mb
    if weight_mb == 0:
        return None
    # Check cache
    cache = getattr(get_aux_vram, "_cache", None)
    if cache and cache["name"] == aux_info.name:
        return cache["total_mb"]
    # Try to get architecture from Ollama /api/show
    aux_host = f"http://127.0.0.1:{aux_port}"
    try:
        show_data = json.dumps({"name": aux_info.name}).encode()
        show_req = urllib.request.Request(f"{aux_host}/api/show", data=show_data, method="POST")
        with urllib.request.urlopen(show_req, timeout=2) as resp:
            show = json.loads(resp.read())
        info = show.get("model_info", {})
        # Try multiple architecture keys (qwen35, llama, gemma2, qwen2)
        arch_keys = ["qwen35", "llama", "gemma2", "qwen2"]
        layers, kv_heads, head_dim = 0, 0, 0
        for arch in arch_keys:
            l = info.get(f"{arch}.block_count")
            k = info.get(f"{arch}.attention.head_count_kv")
            h = info.get(f"{arch}.attention.key_length")
            if l and k and h:
                layers, kv_heads, head_dim = l, k, h
                break
        if layers and kv_heads and head_dim:
            # Ollama KV cache defaults to q8_0; user can set OLLAMA_KV_CACHE_TYPE
            cache_bytes = 1.0
            ctx = aux_info.context_length
            # DeepSeek/Kimi use MLA — flat ~70 KB/token
            aux_is_mla = ("deepseek" in aux_info.name.lower() or "kimi" in aux_info.name.lower()) and "distill" not in aux_info.name.lower()
            if aux_is_mla:
                cache_mb = 70.0 * ctx / (1024)
            else:
                cache_mb = calc_kv_cache_mb(layers, kv_heads, head_dim, cache_bytes, ctx)
            total = weight_mb + cache_mb
            get_aux_vram._cache = {"name": aux_info.name, "total_mb": total}
            return total
    except Exception:
        pass
    # Fallback: just return weight size
    return weight_mb


def fetch_metrics(metrics_url):
    """Fetch all metrics from /api/metrics."""
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


def filter_valid(metrics):
    """Filter to valid requests: status 200, >= 5 output tokens."""
    return [
        m for m in metrics
        if m.get("resp_status_code") == 200
        and m.get("tokens", {}).get("output_tokens", 0) >= 5
    ]


# ── Model identity ─────────────────────────────────────

def parse_quant_from_path(filepath):
    """Extract quantization level from a GGUF filepath or command string.
    Works on any standard GGUF filename convention.
    E.g. 'Qwen3.6-27B-Q5_K_M.gguf' -> 'q5_k_m'
         'ornith-1.0-35b-Q4_K_M.gguf' -> 'q4_k_m'
         'Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf' -> 'iq4_xs'
    Returns None if no quant found."""
    if not filepath:
        return None
    match = QUANT_PATTERN.search(filepath)
    if match:
        quant = match.group(1).rstrip("_")
        return quant.lower()
    return None


def _parse_yaml_models_simple(yaml_path):
    """Parse model IDs and their GGUF paths from llama-swap config.yaml.
    Simple parser — no YAML dependency needed. Reads -m flag from cmd lines."""
    if not yaml_path or not os.path.isfile(yaml_path):
        return {}
    try:
        model_map = {}
        current_model = None
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                # Model ID: indented key with quotes, e.g. '  "27":'
                model_match = re.match(r'^\s+"([^"]+)"\s*:', line)
                if model_match:
                    current_model = model_match.group(1)
                    continue
                # Extract -m "path/to/model.gguf"
                if current_model and "-m" in stripped:
                    m_match = re.search(r'-m\s+"([^"]+\.gguf)"', stripped)
                    if m_match:
                        model_map[current_model] = m_match.group(1)
                        current_model = None
        return model_map
    except Exception:
        return {}


def get_active_model_identity(valid_metrics, config_yaml_path=None):
    """Get the active model name and quantization level.
    Returns a ModelIdentity or None."""
    if not valid_metrics:
        return None
    active_model = valid_metrics[-1].get("model")
    if not active_model:
        return None
    # Check cache
    cache = getattr(get_active_model_identity, "_cache", None)
    if cache and cache.model_id == active_model:
        return cache
    # Resolve quant from config.yaml
    quant = None
    if config_yaml_path:
        model_map = _parse_yaml_models_simple(config_yaml_path)
        gguf_path = model_map.get(active_model)
        if gguf_path:
            quant = parse_quant_from_path(gguf_path)
    result = ModelIdentity(model_id=active_model, quant=quant)
    get_active_model_identity._cache = result
    return result


def get_last_metrics(valid_metrics, count=1):
    """Get the last N successful request metrics."""
    return valid_metrics[-count:] if valid_metrics else []


LOOKBACK_CAP = 500


def get_metrics_by_bucket(valid_metrics):
    """Get average t/s per token bucket.

    Always scans the last LOOKBACK_CAP metrics — sliding window so recent
    performance dominates. Active model only, uncached prompts.
    """
    scan = valid_metrics[-LOOKBACK_CAP:] if len(valid_metrics) > LOOKBACK_CAP else valid_metrics

    active_model = scan[-1].get("model") if scan else None
    uncached = [
        m for m in scan
        if (active_model is None or m.get("model") == active_model)
        and m.get("tokens", {}).get("cache_tokens", 0) == 0
    ]

    bucket_vals = {}
    for m in uncached:
        input_tok = m.get("tokens", {}).get("input_tokens", 0)
        output_tok = m.get("tokens", {}).get("output_tokens", 0)
        seq_len = input_tok + output_tok  # final sequence length (includes long reasoning)
        pps = m.get("tokens", {}).get("prompt_per_second", 0)
        dps = m.get("tokens", {}).get("tokens_per_second", 0)
        for label, mn, mx in TOKEN_BUCKETS:
            if mn <= seq_len <= mx:
                existing = bucket_vals.get(label)
                if existing is None or dps > existing["best_d"]:
                    bucket_vals[label] = {
                        "bucket_key": mn,
                        "label": seq_len,
                        "prompt_per_second": pps,
                        "tokens_per_second": dps,
                        "best_d": dps,
                    }
                break

    # Sort by bucket boundary, keep actual token count as display label
    sorted_buckets = sorted(bucket_vals.items(), key=lambda x: x[1]["bucket_key"])
    return {
        s["label"]: {
            "bucket_key": s["bucket_key"],
            "prompt_per_second": s["prompt_per_second"],
            "tokens_per_second": s["tokens_per_second"],
        }
        for label, s in sorted_buckets
    }


# ── Rendering ──────────────────────────────────────────

def render_prompt_log(valid_metrics, running_models=None):
    """Render a rolling log of the last 3 prompts."""
    lines = []
    recent = get_last_metrics(valid_metrics, 3)
    if not recent:
        return lines

    # Build model_id -> model_path mapping from running_models
    path_map = {}
    if running_models:
        for rm in running_models:
            mid = rm.get("model_id", "")
            mp = rm.get("model_path", "")
            if mid and mp:
                path_map[mid] = mp
        # Also build a fallback: config key -> short name from actual running model
        running_short = {}
        for rm in running_models:
            mid = rm.get("model_id", "")
            mp = rm.get("model_path", "")
            if mid and mp:
                running_short[mid] = short_model_name(mp)

    lines.append(f"  {BOLD}Last Prompts{RESET}")
    lines.append(f"  {BOLD}{DIM}{'─' * 56}{RESET}")
    for req in reversed(recent):
        t = req.get("tokens", {})
        raw_model = req.get("model", "—")
        # Map config key to GGUF path, then shorten
        display_path = path_map.get(raw_model, raw_model)
        model = short_model_name(display_path) if raw_model != "—" else "—"
        # Fallback: use cached short name if config key wasn't resolved to a path
        if model == short_model_name(raw_model) and raw_model in running_short:
            model = running_short[raw_model]
        prompt_tps = t.get("prompt_per_second", 0)
        decode_tps = t.get("tokens_per_second", 0)
        input_tok = t.get("input_tokens", 0)
        output_tok = t.get("output_tokens", 0)
        cached_tok = t.get("cache_tokens", 0)
        duration = req.get("duration_ms", 0)
        req_time = format_time(req.get("timestamp", ""))

        lines.append(
            f"  {DIM}[{req_time}]{RESET} {BOLD}{model}{RESET} "
            f"{DIM}│{RESET} {DIM}decode:{RESET} {LIGHT_ORANGE}{decode_tps:.0f}{RESET}{WHITE}t/s{RESET} "
            f"{DIM}│{RESET} {DIM}prompt:{RESET} {ORANGE}{prompt_tps:.0f}{RESET}{WHITE}pp{RESET} "
            f"{DIM}│{RESET} {DIM}{format_duration(duration)}{RESET}"
        )
        lines.append(
            f"  {DIM}     {RESET}{DIM}in:{RESET}{WHITE}{input_tok}{RESET} "
            f"{DIM}│ {RESET}{DIM}out:{RESET}{WHITE}{output_tok}{RESET} "
            f"{DIM}│ {RESET}{DIM}cache:{RESET}{WHITE}{cached_tok}{RESET}"
        )
        lines.append(f"  {DIM}{'─' * 56}{RESET}")

    return lines


def render_chart(buckets):
    """Render a unified chart showing prompt bar + decode value by context size."""
    lines = []
    max_prompt = 0
    max_decode = 0
    populated = []
    # Sort by bucket_key (boundary), not by display label
    for ctx, data in sorted(buckets.items(), key=lambda x: x[1].get("bucket_key", 0)):
        pps = data.get("prompt_per_second", 0)
        dps = data.get("tokens_per_second", 0)
        bucket_key = data.get("bucket_key", ctx)
        populated.append((ctx, bucket_key, pps, dps))
        if pps > max_prompt:
            max_prompt = pps
        if dps > max_decode:
            max_decode = dps

    if not populated:
        return lines

    bar_width = 20
    lines.append(f"  {BOLD}{CYAN}{'═' * 56}{RESET}")
    lines.append(f"  {BOLD}  t/s by context size{RESET}")
    lines.append(f"  {BOLD}{DIM}{'─' * 56}{RESET}")

    # Open-ended bucket start value (last in TOKEN_BUCKETS)
    open_ended = TOKEN_BUCKETS[-1][1] if TOKEN_BUCKETS else None

    for ctx, bucket_key, pps, dps in populated:
        if pps <= 0 and dps <= 0:
            continue
        # Context label — format real token count with k/M suffix
        ctx_str = _fmt_num(ctx)
        if open_ended and bucket_key == open_ended:
            ctx_str += "+"
        ctx_display = f"{ctx_str:>5}"

        # Prompt bar — pad visible chars, then color
        if pps > 0:
            p_len = max(1, round((pps / max_prompt) * bar_width)) if max_prompt > 0 else 1
            p_bar_raw = "\u2588" * p_len + " " * (bar_width - p_len)
            p_color = WHITE
            p_num = f"{ORANGE}{pps:.0f}{RESET}"
            p_unit = f"{WHITE}pp{RESET}"
        else:
            p_bar_raw = "\u2591" * bar_width
            p_color = DIM
            p_num = f"{DIM}---{RESET}"
            p_unit = f"{DIM}pp{RESET}"

        # Decode value — pad visible chars, then color
        if dps > 0:
            d_num = f"{LIGHT_GREEN}{dps:.0f}{RESET}"
            d_unit = f"{WHITE}t/s{RESET}"
        else:
            d_num = f"{DIM}      {RESET}"
            d_unit = f"{DIM}t/s{RESET}"

        # Pad plain text first, then wrap with color — prevents ANSI from breaking alignment
        ctx_cell = f"{DIM}{ctx_display}{RESET}"
        bar_cell = f"{p_color}{p_bar_raw}{RESET}"
        lines.append(f"  {ctx_cell} │ {bar_cell} {d_num}{d_unit} │ {p_num}{p_unit}")

    lines.append(f"  {BOLD}{CYAN}{'═' * 56}{RESET}")
    return lines


def _visible_len(s):
    """Count visible characters in a string (excluding ANSI escape codes)."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _fmt_num(n):
    """Format a number with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _format_metric_line(label, vram_str, decode_tps, align_visible=40):
    """Format a single metric line with aligned decode values."""
    if decode_tps > 0:
        decode_str = f"{DIM}decode: {RESET}{LIGHT_ORANGE}{decode_tps:.0f}{RESET}{WHITE}t/s{RESET}"
    else:
        decode_str = f"{DIM}decode: 0t/s{RESET}"
    if vram_str:
        prefix = f"{BOLD}{label}{RESET} {SOFT_WHITE}{vram_str}{RESET}"
    else:
        prefix = f"{BOLD}{label}{RESET}"
    pad = max(1, align_visible - _visible_len(prefix))
    return f"  {prefix}{' ' * pad}{decode_str}"


def render_main_model_decode(valid_metrics, sys_info):
    """Render system RAM and return latest decode tps from valid metrics."""
    lines = []

    # Get latest decode speed from valid metrics
    latest = valid_metrics[-1] if valid_metrics else None
    decode_tps = latest.get("tokens", {}).get("tokens_per_second", 0) if latest else 0

    if sys_info:
        sys_mem_used = sys_info.mem_used_mb
        sys_mem_total = sys_info.mem_total_mb
        sys_mem_pct = (sys_mem_used / sys_mem_total * 100) if sys_mem_total else 0
        sys_bar = util_bar(sys_mem_pct, 16)
        sys_mem_str = f"{sys_mem_used / 1024:.1f} / {sys_mem_total / 1024:.0f} GB ({sys_mem_pct:.0f}%)"
        lines.append(f"  {BOLD}System RAM{RESET}: {sys_bar} {sys_mem_str}")

    return lines, decode_tps


def render(gpus, sys_info, buckets, valid_metrics, refresh_interval, aux_info, session_totals, identity=None, host=None, aux_port=None, running_models=None):
    """Render the dashboard."""
    sys.stdout.write("\033[H\033[0J")
    now = time.strftime("%H:%M:%S")
    lines = []

    lines.append(f" {BOLD}{CYAN}{'═' * 56}{RESET}")
    lines.append(f" {BOLD}  llama-swap Dashboard{RESET}  {now}")
    lines.append(f" {BOLD}{CYAN}{'═' * 56}{RESET}")
    lines.append("")

    if not gpus:
        lines.append(f"  {RED}amd-smi not available{RESET}")
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
        return

    for i, gpu in enumerate(gpus):
        temp = gpu.temp_c
        mem_used = gpu.mem_used_mb
        mem_total = gpu.mem_total_mb
        util = gpu.gpu_util_pct
        power = gpu.power_w
        fan = gpu.fan_pct
        mem_pct = (mem_used / mem_total * 100) if mem_total else 0

        if util >= 5:
            status = f"{ORANGE}● ACTIVE{RESET}"
            status_color = ORANGE
        else:
            status = f"{DIM}● IDLE{RESET}"
            status_color = DIM

        vram_bar = util_bar(mem_pct, 14)
        util_bar_str = util_bar(util, 14)
        mem_str = f"{mem_used / 1024:.1f} / {mem_total / 1024:.0f} GB"

        lines.append(f"  {BOLD}{WHITE}[GPU {gpu.id}] {gpu.name}{RESET}")
        lines.append(f"  {status}  {DIM}{color_temp(temp)}{temp}°C{RESET}")
        lines.append(f"  {DIM}VRAM:{RESET} {vram_bar} {mem_str}")
        lines.append(f"  {DIM}UTIL:{RESET} {status_color}{util_bar_str}{RESET} {util}%")
        lines.append(f"  {DIM}PWR:{RESET}  {power:.0f}W  {DIM}|{RESET} {DIM}FAN:{RESET} {fan}%")

        if i < len(gpus) - 1:
            lines.append(f"  {DIM}{'─' * 48}{RESET}")
            lines.append("")

    lines.append("")

    # System memory + model decode speeds
    sys_lines, decode_tps = render_main_model_decode(valid_metrics, sys_info)
    lines.extend(sys_lines)
    lines.append(f"  {DIM}{'─' * 56}{RESET}")
    # Calculate main model VRAM: weights + KV cache (additive estimate)
    main_vram_info = get_main_model_vram(running_models, valid_metrics) if running_models else None
    if main_vram_info:
        main_vram_str = f"{main_vram_info.total_mb / 1024:.1f} GB"
    else:
        main_vram_mb = sum(gpu.mem_used_mb for gpu in gpus if gpu.gpu_util_pct >= 5) if gpus else 0
        main_vram_str = f"{main_vram_mb / 1024:.1f} GB" if main_vram_mb > 0 else None
    # Build model label from actual model path (not config key)
    actual_model_path = None
    if running_models:
        actual_model_path = running_models[0].get("model_path")
    if actual_model_path:
        short_name = short_model_name(actual_model_path)
        display_label = short_name
        if identity and identity.quant:
            display_label = f"{display_label} {identity.quant}"
        model_label = f"{display_label} ({host.split(':')[-1] if ':' in host else '8080'})"
    else:
        model_label = f"— ({host.split(':')[-1] if ':' in host else '8080'})"
    # Inference state
    main_state = get_inference_state(valid_metrics, gpus) if valid_metrics else None
    lines.append(_format_metric_line(model_label, main_vram_str, decode_tps))
    if aux_info:
        aux_name = aux_info.name
        aux_short = aux_name.split(":")[0]
        aux_total_mb = get_aux_vram(aux_info, aux_port)
        aux_tps = aux_info.decode_tps
        aux_vram_str = f"{aux_total_mb / 1024:.1f} GB"
        aux_state = get_aux_state(aux_info, aux_port)
        lines.append(_format_metric_line(f"Ollama Aux ({aux_port})", aux_vram_str, aux_tps))
        if aux_total_mb is not None:
            lines.append(f"  {DIM}     Size from Ollama API, not amd-smi{RESET}")
    else:
        lines.append(f"  {BOLD}Ollama Aux ({aux_port}){RESET}  {DIM}offline{RESET}")
    lines.append(f"  {DIM}{'─' * 56}{RESET}")
    lines.append("")

    # Unified chart: prompt & decode side by side
    chart = render_chart(buckets)
    if chart:
        lines.extend(chart)

    lines.append("")

    # Last 3 prompts rolling log
    lines.extend(render_prompt_log(valid_metrics, running_models))
    lines.append("")

    # Session token totals — passed in, no O(n) scan
    total_in = session_totals["in"]
    total_out = session_totals["out"]
    total_reqs = session_totals["reqs"]

    token_line = (
        f" {DIM}Session Tokens  "
        f"in: {_fmt_num(total_in)}  "
        f"out: {_fmt_num(total_out)}  "
        f"reqs: {total_reqs}{RESET}"
    )

    lines.append(f" {BOLD}{CYAN}{'═' * 56}{RESET}")
    lines.append(token_line)

    # Subtle footer: full model path
    if running_models and running_models[0].get("model_path"):
        rm = running_models[0]
        gguf = os.path.basename(rm["model_path"])
        lines.append(f"  {DIM}└─ {gguf}{RESET}")

    lines.append(f" {DIM}Refresh: {refresh_interval}s | Ctrl+C to quit")
    lines.append(f" {DIM}GPU UTIL >5% = active")
    lines.append("")

    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


def main():
    cli_host, cli_refresh = parse_cli()
    config = load_config()
    host = resolve_host(cli_host, config)
    refresh = cli_refresh if cli_refresh else DEFAULT_REFRESH
    api_url, metrics_url = build_urls(host)

    config_yaml = get_config_yaml(config)
    aux_port = get_aux_port(config)

    # Cache GPU names once at startup (names never change at runtime)
    gpu_names = get_amd_gpu_names()

    # Incremental state
    session_totals = {"in": 0, "out": 0, "reqs": 0}
    prev_count = 0
    prev_model = None

    if config_yaml:
        print(f"Model config loaded: {config_yaml}")
    print("GPU Dashboard starting...")
    print("Press Ctrl+C to exit.\n")

    while True:
        loop_start = time.time()
        gpus = get_amd_smi(gpu_names)
        sys_info = get_llama_swap_stats(api_url)
        aux_info = get_auxiliary_model(aux_port)
        running_models = fetch_running_models(host)
        metrics = fetch_metrics(metrics_url)
        valid = filter_valid(metrics)

        # Detect new metrics since last render
        new_valid = valid[prev_count:]
        current_model = valid[-1].get("model") if valid else None

        # Reset on model switch
        if current_model != prev_model:
            session_totals = {"in": 0, "out": 0, "reqs": 0}
            prev_count = 0
            new_valid = valid

        # Incrementally update session totals
        for m in new_valid:
            session_totals["in"] += m.get("tokens", {}).get("input_tokens", 0)
            session_totals["out"] += m.get("tokens", {}).get("output_tokens", 0)
            session_totals["reqs"] += 1

        prev_count = len(valid)
        prev_model = current_model

        buckets = get_metrics_by_bucket(valid)
        identity = get_active_model_identity(valid, config_yaml)
        render(gpus, sys_info, buckets, valid, refresh, aux_info, session_totals, identity, host=host, aux_port=aux_port, running_models=running_models)

        # Fixed refresh interval — subtract work time to prevent drift
        elapsed = time.time() - loop_start
        time.sleep(max(0.1, refresh - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.exit(0)
