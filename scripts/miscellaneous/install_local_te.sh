#!/bin/bash
# Verify and (re)build TransformerEngine and flash-attn.
#
# Usage:
#   conda activate <your-env>
#   bash scripts/miscellaneous/install_local_te.sh

set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment is active. Run: conda activate <your-env>"
    exit 1
fi

# ── Locate CUDA_HOME ──────────────────────────────────────────────────────────
if [ -z "$CUDA_HOME" ]; then
    for candidate in \
        /usr/local/cuda-13.0 \
        /usr/local/cuda-12.5 \
        /usr/local/cuda-12.4 \
        /usr/local/cuda-12.1 \
        /usr/local/cuda-11.8 \
        /usr/local/cuda; do
        if [ -f "$candidate/bin/nvcc" ]; then
            export CUDA_HOME=$candidate
            break
        fi
    done
fi

if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: Could not find nvcc. Set CUDA_HOME manually:"
    echo "  CUDA_HOME=/usr/local/cuda-<version> bash scripts/miscellaneous/install_local_te.sh"
    exit 1
fi

export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "Using CUDA_HOME=$CUDA_HOME"

# ── Detect GPU architecture for flash-attn ────────────────────────────────────
ARCH=$(python -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
    print(f'{cap[0]}.{cap[1]}')
else:
    print('8.0')
" 2>/dev/null || echo "8.0")
echo "GPU compute capability: $ARCH"

# ── TransformerEngine ─────────────────────────────────────────────────────────
if python -c "import transformer_engine" 2>/dev/null; then
    echo "TransformerEngine already installed — skipping."
else
    echo "Building TransformerEngine..."
    pushd TransformerEngine
    rm -rf build
    export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvtx/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/include
    export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
    export CUDNN_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn
    export NVTE_FRAMEWORK=pytorch
    export MAX_JOBS=64
    python -m pip install --no-build-isolation .
    popd
    echo "TransformerEngine done."
fi

# ── flash-attn ────────────────────────────────────────────────────────────────
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed — skipping."
else
    echo "Building flash-attn (this takes 10-60 min)..."
    export TORCH_CUDA_ARCH_LIST="$ARCH"
    export MAX_JOBS=64
    python -m pip install flash-attn --no-build-isolation
    echo "flash-attn done."
fi

echo "All done."

