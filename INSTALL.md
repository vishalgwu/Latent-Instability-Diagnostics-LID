# INSTALL.md — LID Research Environment Setup

Reproducible environment for Latent Instability Diagnostics research.  
Verified on: MIT SuperCloud A100 nodes, Ubuntu 22.04.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| GPU | A100 40GB | A100 80GB (140GB for 70B models) |
| CUDA | 11.8 | 12.1 |
| RAM | 64GB | 128GB |
| Disk | 500GB | 2TB (for model weights + datasets) |

---

## Step 1 — Clone Repository

```bash
git clone https://github.com/<org>/lid-research.git
cd lid-research
git checkout main
```

---

## Step 2 — Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Step 3 — Install Dependencies

**For development (CPU-only, unit tests):**
```bash
pip install numpy scipy scikit-learn pytest pytest-cov
pytest tests/test_metrics_np.py -v   # must be 31/31 green
```

**For full GPU experiments:**
```bash
pip install -r requirements.txt
pip install -e ".[dev,full]"
```

---

## Step 4 — Verify GPU Environment

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

Expected output on A100 80GB:
```
PyTorch: 2.2.x
CUDA available: True
GPU: NVIDIA A100 80GB PCIe
VRAM: 79.9 GB
```

---

## Step 5 — Smoke Test (Llama-7B Inference)

```bash
python scripts/smoke_test.py --model meta-llama/Llama-2-7b-hf --n-examples 5
```

Expected: 5 examples generated in <30s. Overhead vs clean baseline recorded.

---

## Step 6 — Configure Experiment Tracking

```bash
wandb login
# Enter API key from wandb.ai/authorize
python scripts/test_wandb.py   # creates dummy run in W&B
```

---

## Step 7 — Configure DVC

```bash
dvc init
dvc remote add -d myremote s3://your-bucket/lid-research-data
```

---

## Environment Variables

Create `.env` file (never commit this):
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=lid-research
CUDA_VISIBLE_DEVICES=0,1,2,3    # adjust for your node
```

---

## Model Download

Models are downloaded automatically by HuggingFace `transformers`. Ensure `HF_TOKEN` is set for gated models (Llama-2).

| Model | Size | VRAM |
|---|---|---|
| meta-llama/Llama-2-7b-hf | 13GB | 16GB min |
| meta-llama/Llama-2-70b-hf | 130GB | 2x A100 80GB |
| mistralai/Mistral-7B-v0.1 | 14GB | 16GB min |

---

## Troubleshooting

**`CUDA out of memory` on Llama-70B:**  
Use `load_in_4bit=True` via bitsandbytes. See `configs/model/llama-70b.yaml`.

**`torch.manual_seed` not making runs deterministic:**  
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in environment before running.

**CI tests failing on import errors:**  
Run `pytest tests/test_metrics_np.py` only (no torch required for these).
