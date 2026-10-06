#!/bin/bash

source devconfig.sh

export JOB_NAME=${JOB_NAME:-flame-moe}
export JOB_ID=$(date +%s)

# Derive WandB project from model size in JOB_NAME (e.g. flame-moe-38m → moe-38m).
_MODEL_SIZE=$(echo "$JOB_NAME" | grep -oE '[0-9]+m')
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:8000}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-moe-${_MODEL_SIZE}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$JOB_NAME}"
export WANDB_NAME="${WANDB_NAME:-flame-moe}"
export TRAIN_DATASET="${TRAIN_DATASET:-$TOKENIZED_DIR}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-/local/home/bingli/moe/$JOB_NAME}"

mkdir -p "$TRAIN_WEIGHTS"

# Data is already local — no download step needed.
bash scripts/training/modules/flame-moe_step2.sh
