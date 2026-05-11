#!/bin/bash
#PBS -P kl02
#PBS -q gpuvolta
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=60GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N per_ch_eval_lgt_v

set -euo pipefail

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok    # flowtok env has older torch with sm_70 kernels

cd /scratch/kl02/$USER/Projv2v/FlowTok
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONWARNINGS="ignore::FutureWarning"

python3 -u -c "import torch; print('torch=', torch.__version__, 'cu=', torch.version.cuda); print('arch list:', torch.cuda.get_arch_list()); print('dev cap:', torch.cuda.get_device_capability(0))"

python3 -u scripts/_per_ch_eval_overfit_lgt.py 2>&1 | tee /scratch/kl02/$USER/Projv2v/Experiments/_per_ch_eval_overfit_lgt_volta.log
