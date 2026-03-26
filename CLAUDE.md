# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FlowTok** is a flow matching-based generative model framework originally designed for text-to-image (T2I) generation, extended here for **satellite-to-radar translation** tasks:
- **v2v** (video-to-video): multi-frame satellite IR → radar reflectivity sequences
- **i2i** (image-to-image): single-frame satellite IR → radar reflectivity

The codebase is a PyTorch research project intended for HPC cluster deployment (NCI GADI, PBS job scheduler).

## Setup & Installation

```bash
# Python with CUDA 12.1
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install -U --pre triton
pip3 install -r requirements.txt
```

## Common Commands

### Training
```bash
# Single GPU (local)
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config=configs/Sat2Radar-v2v-satlight-tokenfusion-FlowTiTok-XL_gadi.py

# Multi-GPU
NUM_PROCESSES=4 accelerate launch --num_processes 4 scripts/train_sat2radar_v2v.py --config=configs/<config>.py

# GADI HPC cluster (PBS)
qsub train_sat2radar_v2v_satlight_tokenfusion_0_gadi.sh
```

### Validation (during/after training)
```bash
python scripts/validate_sat2radar_v2v.py --ckpt /path/to/checkpoint.pth --config configs/<config>.py
```

### Testing / Inference
```bash
# Single checkpoint
python scripts/test_sat2radar_v2v.py --ckpt /path/to/checkpoint.pth --config configs/<config>.py

# Multiple checkpoints
python scripts/test_sat2radar_v2v.py --ckpts "ckpt1.pth,ckpt2.pth" --config configs/<config>.py
```

## Architecture

### Core Data Flow (Training)
```
Raw Input [B, T, C, H, W]
  → [Optional] AdapterIn  (channel/resolution mismatch handling)
  → FlowTiTok Encoder     → Tokens [B, T×L, C_tok]
  → Flow Matching DiT     (denoising on token space)
  → MSE Loss on tokens
  → AdamW optimization
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| FlowTiTok Autoencoder | `libs/flowtitok.py` | Tokenizes images/frames to latent tokens; also loads pretrained weights with channel adaptation |
| Flow Matching Solver | `diffusion/flow_matching.py` | `FlowMatching` (training loss), `ODEEulerFlowMatchingSolver` (inference) |
| DiT Model | `libs/model/flowtok_t2i.py` | Transformer-based denoising model (S/B/L/XL/H sizes) |
| Channel Adapters | `libs/adapters.py` | `AdapterIn`/`AdapterOut` for mismatched input/output channels (satellite=3ch, radar=1ch) |
| Dataset | `data/dataset.py` | `SatelliteRadarNpyDataset` — loads `.npy` files with 12 channels (IR×10 + lightning + radar) |
| Main Training Script | `scripts/train_sat2radar_v2v.py` | Orchestrates full training loop for v2v/i2i |

### Data Channels
- **Channels 0–9**: Satellite IR bands (normalized to [200K, 320K])
- **Channel 10**: Lightning (normalized to [0.1, 50.0])
- **Channel 11**: Radar reflectivity (normalized to [0, 60 dBZ]) — this is the prediction target

### Configuration System
Configs are Python files in `configs/` returning `ml_collections.ConfigDict` via `get_config()`. Naming convention:
- `Sat2Radar-{v2v|i2i}-{variant}-FlowTiTok-XL[_gadi].py`
- `_gadi` suffix = NCI GADI HPC cluster version with PBS-specific paths

Key config sections: `config.train`, `config.dataset`, `config.optimizer`, `config.model`, `config.vq_model`

### Model Variants (configs)
- `satlight_tokenfusion`: Fuses satellite lightning channel at token level — the primary research variant
- `adapter`: Uses AdapterIn/AdapterOut for channel adaptation
- `contrastive`: Adds contrastive loss for modality alignment
- `newposemb`: Custom positional embeddings

## HPC Cluster Details (NCI GADI)

PBS scripts use:
```
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=48:00:00
```

Data paths are under `/g/data/kl02/yh0308/Data/`. Pretrained tokenizer checkpoint: `/g/data/kl02/yh0308/Data/flowtok_ckpts/FlowTiTok_512.bin`.

Environment variables needed for offline/cached model loading:
- `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME` — model caches
- `OPENCLIP_LOCAL_CKPT` — local CLIP checkpoint path

## Evaluation Metrics
- **FSS** (Fractions Skill Score): via `pysteps`, primary spatial structure metric for radar
- **SSIM**, **PSNR**, **MAE**: standard reconstruction metrics
- Computed in `scripts/validate_sat2radar_v2v.py` and `scripts/test_sat2radar_v2v.py`
