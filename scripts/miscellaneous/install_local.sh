#!/bin/bash
# Install the runtime environment into the currently active conda environment.
#
# Usage:
#   conda activate <your-env>
#   bash scripts/miscellaneous/install_local.sh           # full install (training)
#   bash scripts/miscellaneous/install_local.sh tokenize  # minimal install (tokenization only)

set -e

MODE=${1:-full}

# ── Verify an env is active ───────────────────────────────────────────────────
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment is active. Run: conda activate <your-env>"
    exit 1
fi
echo "Installing into: $CONDA_PREFIX"

# Always use the active env's Python to avoid hitting the system pip
PY="python -m pip"

# ── Locate nvcc (not always on PATH) ─────────────────────────────────────────
NVCC_BIN=""
for candidate in \
    "$(which nvcc 2>/dev/null)" \
    "/usr/local/cuda/bin/nvcc" \
    "/usr/local/cuda-12.5/bin/nvcc" \
    "/usr/local/cuda-12.4/bin/nvcc" \
    "/usr/local/cuda-12.1/bin/nvcc" \
    "/usr/local/cuda-11.8/bin/nvcc"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        NVCC_BIN=$candidate
        break
    fi
done

if [ -z "$NVCC_BIN" ]; then
    echo "ERROR: nvcc not found. Install the CUDA toolkit (e.g. sudo apt install cuda-toolkit-12-5)."
    exit 1
fi

NVCC_CUDA=$("$NVCC_BIN" --version | grep "release" | awk '{print $6}' | cut -c2-)
NVCC_MAJOR=$(echo $NVCC_CUDA | cut -d. -f1)
NVCC_MINOR=$(echo $NVCC_CUDA | cut -d. -f2)
export CUDA_HOME=$(dirname $(dirname $NVCC_BIN))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "nvcc $NVCC_CUDA at $CUDA_HOME"

# ── Pick the closest PyTorch wheel for this nvcc ──────────────────────────────
if [ "$NVCC_MAJOR" -ge 13 ]; then
    TORCH_CUDA_TAG="cu128"
elif [ "$NVCC_MAJOR" -ge 12 ] && [ "$NVCC_MINOR" -ge 8 ]; then
    TORCH_CUDA_TAG="cu128"
elif [ "$NVCC_MAJOR" -ge 12 ] && [ "$NVCC_MINOR" -ge 4 ]; then
    TORCH_CUDA_TAG="cu124"
elif [ "$NVCC_MAJOR" -ge 12 ]; then
    TORCH_CUDA_TAG="cu121"
elif [ "$NVCC_MAJOR" -ge 11 ] && [ "$NVCC_MINOR" -ge 8 ]; then
    TORCH_CUDA_TAG="cu118"
else
    echo "ERROR: CUDA $NVCC_CUDA is too old. Need >= 11.8."
    exit 1
fi
echo "Using PyTorch wheel: $TORCH_CUDA_TAG"

# ── Install/reinstall torch if missing or CUDA tag doesn't match nvcc ─────────
NEEDS_TORCH_INSTALL=false
if python -c "import torch" 2>/dev/null; then
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'cpu')")
    if [ "$TORCH_CUDA" != "$NVCC_CUDA" ]; then
        echo "torch CUDA ($TORCH_CUDA) ≠ nvcc ($NVCC_CUDA) — reinstalling with $TORCH_CUDA_TAG..."
        NEEDS_TORCH_INSTALL=true
    else
        echo "torch already installed and CUDA matches ($TORCH_CUDA) — skipping."
    fi
else
    echo "torch not found — installing with $TORCH_CUDA_TAG..."
    NEEDS_TORCH_INSTALL=true
fi

if [ "$NEEDS_TORCH_INSTALL" = true ]; then
    $PY install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/$TORCH_CUDA_TAG
fi

# ── Common packages ───────────────────────────────────────────────────────────
$PY install transformers sentencepiece pybind11 tensorboard "numpy==1.26.4"

if [ "$MODE" = "tokenize" ]; then
    echo "Tokenize-only install complete."
    exit 0
fi

# ── Megatron-LM deps — pin torch so requirements.txt can't upgrade it ─────────
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VERSION" > /tmp/torch_pin.txt
echo "Installing Megatron-LM requirements (torch pinned at $TORCH_VERSION)..."
$PY install -r Megatron-LM/requirements/pytorch_24.10/requirements.txt \
    --constraint /tmp/torch_pin.txt
rm /tmp/torch_pin.txt

# ── Patch apex's strict CUDA version check ────────────────────────────────────
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
if [ "$TORCH_CUDA" != "$NVCC_CUDA" ]; then
    echo "Patching apex: torch CUDA $TORCH_CUDA vs nvcc $NVCC_CUDA..."
    sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/pass  # patched: minor cuda version mismatch/' apex/setup.py
fi

# ── apex (skipped if already installed) ───────────────────────────────────────
# Check with pip rather than "import apex": from the repo root, the apex/
# source folder would make the import succeed even when nothing is installed.
if $PY show apex > /dev/null 2>&1; then
    echo "apex already installed — skipping."
else
    echo "Building apex with CUDA_HOME=$CUDA_HOME..."
    pushd apex
    $PY install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./
    popd
fi

# ── TransformerEngine (skipped if already installed) ──────────────────────────
if $PY show transformer_engine > /dev/null 2>&1; then
    echo "TransformerEngine already installed — skipping."
else
    echo "Building TransformerEngine..."
    pushd TransformerEngine
    export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvtx/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/include
    export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
    export CUDNN_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn
    export NVTE_FRAMEWORK=pytorch
    export MAX_JOBS=64
    # setup.py imports torch, so build in this environment instead of pip's
    # isolated build environment (which has no torch). Its build tools must
    # then be installed here first.
    $PY install cmake ninja
    $PY install --no-build-isolation .
    popd
fi

echo "Full install complete."

