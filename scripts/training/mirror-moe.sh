#!/bin/bash

source devconfig.sh

export JOB_NAME=${JOB_NAME:-mirror-moe}
export JOB_ID=$(date +%s)


_MODEL_SIZE=$(echo "$JOB_NAME" | grep -oE '[0-9]+m')

export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:8000}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-moe-${_MODEL_SIZE}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$JOB_NAME}"
export WANDB_NAME="${WANDB_NAME:-mirror-rho_${MOE_ROUTER_RHO}}"
export TRAIN_DATASET="${TRAIN_DATASET:-$TOKENIZED_DIR}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-/local/home/bingli/moe/$JOB_NAME}"

mkdir -p "$TRAIN_WEIGHTS"

bash scripts/training/modules/mirror-moe_step2.sh
