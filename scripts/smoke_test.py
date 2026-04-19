"""
scripts/smoke_test.py
=====================
Week 1 KPI: Validate compute environment by running inference
and measuring baseline latency.

TWO MODES:
    --mode fast     TinyLlama-1.1B  (CPU, ~30s)   ← run this locally
    --mode full     Llama-3-8B      (GPU, ~2min)  ← run on A100 server

What this tests:
    1. HF_TOKEN loads correctly from .env
    2. Model downloads / loads without error
    3. Inference runs and produces output
    4. Baseline latency is recorded (needed for overhead comparison later)
    5. Hidden states are accessible via hooks (needed for LID in Week 4)

Usage:
    python scripts/smoke_test.py --mode fast
    python scripts/smoke_test.py --mode full
    python scripts/smoke_test.py --mode full --model meta-llama/Meta-Llama-3-8B

Author: MIT LID Research Team
Week  : 1 — Compute Validation KPI
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Load .env before anything else ───────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# ── Rich for pretty terminal output ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    def header(text): console.print(Panel(f"[bold cyan]{text}[/bold cyan]"))
    def ok(text):     console.print(f"  [green]✅[/green] {text}")
    def warn(text):   console.print(f"  [yellow]⚠️ [/yellow] {text}")
    def err(text):    console.print(f"  [red]❌[/red] {text}")
    def info(text):   console.print(f"  [blue]→[/blue]  {text}")
except ImportError:
    def header(text): print(f"\n{'='*60}\n{text}\n{'='*60}")
    def ok(text):     print(f"  ✅ {text}")
    def warn(text):   print(f"  ⚠️  {text}")
    def err(text):    print(f"  ❌ {text}")
    def info(text):   print(f"  →  {text}")


# ── Model configs ─────────────────────────────────────────────────────────────
MODELS = {
    "fast": {
        "hf_id":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama-1.1B (CPU-friendly, ~30s)",
        "n_layers":    22,
        "d_model":     2048,
        "needs_token": False,
        "device":      "cpu",
        "dtype":       "float32",
    },
    "full": {
        "hf_id":       "meta-llama/Meta-Llama-3-8B",
        "description": "Llama-3-8B (requires A100 GPU)",
        "n_layers":    32,
        "d_model":     4096,
        "needs_token": True,
        "device":      "cuda",
        "dtype":       "float16",
    },
}

# ── 5 test prompts (same ones we'll use in real experiments) ──────────────────
TEST_PROMPTS = [
    "What is the capital of France?",
    "Who wrote the theory of relativity?",
    "What is machine learning in one sentence?",
    "How many planets are in the solar system?",
    "What year did World War II end?",
]


def check_environment(cfg: dict) -> bool:
    """Pre-flight checks before loading any model."""
    header("Pre-flight Environment Check")
    all_ok = True

    # Python version
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        warn(f"Python {v.major}.{v.minor} — recommend 3.10+")

    # HuggingFace token
    if cfg["needs_token"]:
        if HF_TOKEN:
            ok(f"HF_TOKEN loaded from .env ({HF_TOKEN[:8]}...)")
        else:
            err("HF_TOKEN not found in .env — required for this model")
            err("Add HF_TOKEN=hf_xxxx to your .env file")
            all_ok = False
    else:
        ok("HF_TOKEN not required for this model")

    # PyTorch
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")

        if cfg["device"] == "cuda":
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                ok(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
            else:
                err("CUDA not available — this mode requires a GPU")
                err("Run with --mode fast for CPU testing")
                all_ok = False
        else:
            warn("Running on CPU — inference will be slow (~30-60s per example)")
            info("This is fine for smoke testing. Use GPU server for real experiments.")

    except ImportError:
        err("PyTorch not installed — run: pip install torch")
        all_ok = False

    # Transformers
    try:
        import transformers
        ok(f"Transformers {transformers.__version__}")
    except ImportError:
        err("transformers not installed — run: pip install transformers")
        all_ok = False

    return all_ok


def load_model(cfg: dict):
    """Load tokenizer and model with appropriate settings."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_id = cfg["hf_id"]
    header(f"Loading Model: {cfg['description']}")
    info(f"HuggingFace ID: {hf_id}")

    token = HF_TOKEN if cfg["needs_token"] else None

    # Load tokenizer
    info("Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ok(f"Tokenizer loaded in {time.time()-t0:.1f}s")

    # Load model
    info("Loading model weights (this may take a few minutes on first run)...")
    t0 = time.time()

    load_kwargs = {
        "token": token,
        "low_cpu_mem_usage": True,
    }

    if cfg["device"] == "cuda":
        import torch
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()

    load_time = time.time() - t0
    ok(f"Model loaded in {load_time:.1f}s")

    # Report memory usage
    try:
        import torch
        if cfg["device"] == "cuda" and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            info(f"GPU memory used: {mem:.2f} GB")
        else:
            import psutil
            mem = psutil.Process().memory_info().rss / 1e9
            info(f"RAM used by process: {mem:.2f} GB")
    except Exception:
        pass

    return model, tokenizer


def run_inference_examples(model, tokenizer, cfg: dict) -> dict:
    """
    Run 5 inference examples and record latency.
    Also validates hidden state hook access (critical for LID Week 4).
    """
    import torch

    header("Running 5 Inference Examples")
    device = next(model.parameters()).device

    results = []
    latencies = []

    # ── Hidden state hook (validates LID instrumentation will work) ──────────
    hidden_states_captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states_captured[layer_idx] = output[0].detach().cpu()
            else:
                hidden_states_captured[layer_idx] = output.detach().cpu()
        return hook

    # Register hook on middle layer (most important for LID)
    mid_layer = cfg["n_layers"] // 2
    try:
        hook_handle = model.model.layers[mid_layer].register_forward_hook(
            make_hook(mid_layer)
        )
        hook_ok = True
    except Exception as e:
        warn(f"Hook registration failed: {e} — will test inference only")
        hook_ok = False
        hook_handle = None

    # ── Run inference ─────────────────────────────────────────────────────────
    for i, prompt in enumerate(TEST_PROMPTS):
        info(f"Example {i+1}/5: \"{prompt}\"")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,         # greedy — deterministic
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.time() - t0
        latencies.append(latency)

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        ok(f"  Response: \"{response[:80]}\"")
        info(f"  Latency: {latency:.2f}s | New tokens: {len(new_tokens)}")
        results.append({"prompt": prompt, "response": response, "latency": latency})

    # Cleanup hook
    if hook_handle:
        hook_handle.remove()

    # ── Report latency baseline ───────────────────────────────────────────────
    header("Latency Baseline (Critical for Overhead Measurement)")

    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = 30 / avg_latency  # ~30 new tokens per example

    import rich.table
    try:
        table = Table(title="Smoke Test Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Model", cfg["description"])
        table.add_row("Device", str(device))
        table.add_row("Examples run", str(len(results)))
        table.add_row("Avg latency / example", f"{avg_latency:.2f}s")
        table.add_row("Tokens / second", f"{tokens_per_sec:.1f}")
        table.add_row("Hidden state hooks", "✅ Working" if hook_ok else "⚠️ Failed")
        if hook_ok and hidden_states_captured:
            h = hidden_states_captured[mid_layer]
            table.add_row(
                f"Hidden state shape (layer {mid_layer})",
                str(tuple(h.shape))
            )
        console.print(table)
    except Exception:
        print(f"\nAvg latency: {avg_latency:.2f}s | Tokens/sec: {tokens_per_sec:.1f}")

    # ── Save baseline to file ─────────────────────────────────────────────────
    baseline_path = Path("outputs") / "smoke_test_baseline.txt"
    baseline_path.parent.mkdir(exist_ok=True)
    with open(baseline_path, "w") as f:
        f.write(f"model={cfg['hf_id']}\n")
        f.write(f"device={device}\n")
        f.write(f"avg_latency_s={avg_latency:.4f}\n")
        f.write(f"tokens_per_sec={tokens_per_sec:.2f}\n")
        f.write(f"hook_working={hook_ok}\n")
        if hook_ok and hidden_states_captured:
            h = hidden_states_captured[mid_layer]
            f.write(f"hidden_state_shape={tuple(h.shape)}\n")
            f.write(f"n_layers={cfg['n_layers']}\n")
            f.write(f"d_model={cfg['d_model']}\n")

    ok(f"Baseline saved to {baseline_path}")
    info("This file will be used in Week 5 to measure LID overhead ratio.")

    return {
        "latencies": latencies,
        "avg_latency": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "hook_ok": hook_ok,
        "hidden_shape": tuple(hidden_states_captured[mid_layer].shape)
                        if hook_ok and hidden_states_captured else None,
    }


def main():
    parser = argparse.ArgumentParser(description="LID Smoke Test — Week 1 Compute Validation")
    parser.add_argument(
        "--mode", choices=["fast", "full"], default="fast",
        help="fast=TinyLlama CPU (~30s)  |  full=Llama-3-8B GPU (~2min)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override HuggingFace model ID"
    )
    args = parser.parse_args()

    cfg = MODELS[args.mode].copy()
    if args.model:
        cfg["hf_id"] = args.model

    # ── Banner ────────────────────────────────────────────────────────────────
    header("LID Research — Week 1 Smoke Test")
    info(f"Mode   : {args.mode.upper()}")
    info(f"Model  : {cfg['hf_id']}")
    info(f"Device : {cfg['device']}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not check_environment(cfg):
        err("Pre-flight failed. Fix the issues above before continuing.")
        sys.exit(1)

    # ── Load + Run ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(cfg)
    results = run_inference_examples(model, tokenizer, cfg)

    # ── Final verdict ─────────────────────────────────────────────────────────
    header("Week 1 KPI — Compute Validation")
    ok("Model loads without error")
    ok("5 inference examples completed")

    if results["hook_ok"]:
        ok(f"Hidden state hooks working — shape: {results['hidden_shape']}")
        ok("LID instrumentation is READY for Week 4")
    else:
        warn("Hidden state hooks failed — investigate before Week 4")

    ok(f"Baseline latency recorded: {results['avg_latency']:.2f}s/example")
    ok(f"Baseline saved to outputs/smoke_test_baseline.txt")

    print()
    info("Week 1 KPI #3 (compute validation): PASS ✅")
    info("Ready to proceed to Week 2: Baseline implementation")


if __name__ == "__main__":
    main()
