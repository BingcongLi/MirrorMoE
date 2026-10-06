#!/bin/bash
# Shared-vs-routed energy split per layer for an SE checkpoint.
#   frac_shared = mean_t ||h_s||^2/||h||^2 ,  frac_routed = mean_t ||h_r||^2/||h||^2
#
# Usage:
#   JOB_NAME=flame-moe-150m ITER=3636 CHECKPOINT_DIR=/path bash scripts/analysis/shared_split.sh

set -e
source devconfig.sh 2>/dev/null || true

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/flame-moe.sh}"

# 150m SE arch by default (override inline for other sizes)
if [[ -z "$HIDDEN_SIZE" ]]; then
    export NUM_LAYERS=13 HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=2736 MOE_FFN_HIDDEN_SIZE=352
    export MOE_LAYER_FREQ="[0]*1+[1]*12"
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/local/home/bingli/moe/$JOB_NAME}"
OUT="${OUT:-logs-eval/shared-split/${JOB_NAME}}"
mkdir -p "$(dirname "$OUT")"

echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
source "$MODEL_CONFIG"

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH="$PWD/Megatron-LM"

DATA_PREFIX="${SPLIT_DATA_PREFIX:-}"
if [[ -z "$DATA_PREFIX" && -n "$TRAIN_DATASET" ]]; then
    DATA_PREFIX=$(find "$TRAIN_DATASET" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
if [[ -z "$DATA_PREFIX" && -n "$TOKENIZED_DIR" ]]; then
    DATA_PREFIX=$(find "$TOKENIZED_DIR" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
echo "Split data prefix: ${DATA_PREFIX:-<none, will use embedded text>}"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port "${MASTER_PORT:-12396}" \
    scripts/analysis/shared_split.py \
    "${MODEL_ARGS[@]}" \
    --bf16 --seq-length 2048 --micro-batch-size 1 \
    --expert-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --max-tokens-to-oom 100000000 \
    --load "$CHECKPOINT_DIR" \
    --split-output "$OUT" \
    --split-data-prefix "$DATA_PREFIX" \
    --split-num-seqs "${SPLIT_NUM_SEQS:-128}" \
    --split-seq-len "${SPLIT_SEQ_LEN:-1024}" \
    --split-tok-chunk "${SPLIT_TOK_CHUNK:-8192}"

echo "Done. Results: $OUT.npz / $OUT.png"
