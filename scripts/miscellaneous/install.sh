#!/bin/bash
# Set up the MirrorMoE environment (same software stack as FLAME-MoE).
#
#   1. Download apex and TransformerEngine at the commits FLAME-MoE pins.
#   2. Create (or reuse) a Python 3.10 conda environment.
#   3. Install PyTorch 2.6.0 (CUDA 12.4 build).
#   4. Run install_local.sh: Megatron-LM requirements, apex, TransformerEngine.
#
# Usage (from the repository root):
#   bash scripts/miscellaneous/install.sh                     # environment "mirrormoe"
#   ENV_NAME=<name> bash scripts/miscellaneous/install.sh

set -e

ENV_NAME=${ENV_NAME:-mirrormoe}
APEX_COMMIT=c02c6c891eedfabf91f0de8127d7636d4292356d  # NVIDIA/apex
TE_COMMIT=fc034785f5e3a5bc5600a88766d9a1d75137ce77    # NVIDIA/TransformerEngine 1.11.0

if [ ! -d Megatron-LM ]; then
    echo "ERROR: run this script from the repository root."
    exit 1
fi

if ! command -v conda > /dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    exit 1
fi

# ── apex and TransformerEngine (skipped if the folder already exists) ─────────
if [ ! -d apex ]; then
    git clone https://github.com/NVIDIA/apex.git
    git -C apex -c advice.detachedHead=false checkout $APEX_COMMIT
fi

if [ ! -d TransformerEngine ]; then
    git clone https://github.com/NVIDIA/TransformerEngine.git
    git -C TransformerEngine -c advice.detachedHead=false checkout $TE_COMMIT
    git -C TransformerEngine submodule update --init --recursive
fi

# ── Conda environment (an existing one with the same name is reused) ──────────
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Reusing existing conda environment: $ENV_NAME"
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

if ! python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'; then
    echo "ERROR: environment $ENV_NAME must use Python 3.10."
    exit 1
fi

# ── PyTorch 2.6.0, CUDA 12.4 build (install_local.sh keeps this version) ──────
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Megatron-LM requirements, apex and TransformerEngine ──────────────────────
bash scripts/miscellaneous/install_local.sh

echo "Done. Activate the environment with: conda activate $ENV_NAME"
