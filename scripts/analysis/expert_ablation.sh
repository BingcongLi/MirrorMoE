#!/bin/bash
# Expert-ablation heatmap for a 150m shared or no-share checkpoint.
#
# Usage:
#   JOB_NAME=flame-moe-150m            ITER=3636 CHECKPOINT_DIR=/path bash scripts/analysis/expert_ablation.sh
#   JOB_NAME=flame-moe-150m-no-shared  ITER=3636 CHECKPOINT_DIR=/path bash scripts/analysis/expert_ablation.sh

set -e

# pick up TOKENIZED_DIR / TRAIN_DATASET if defined there
source devconfig.sh 2>/dev/null || true

if [[ -z "$MODEL_CONFIG" ]]; then
    if [[ "$JOB_NAME" == *no-shared* || "$JOB_NAME" == *no-share* ]]; then
        MODEL_CONFIG=configs/model/flame-moe-no-shared.sh
    else
        MODEL_CONFIG=configs/model/flame-moe.sh
    fi
fi

# 150m arch (override inline for other sizes)
if [[ -z "$HIDDEN_SIZE" ]]; then
    if [[ "$JOB_NAME" == *no-shar* ]]; then
        export NUM_LAYERS=12 HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=2736 MOE_FFN_HIDDEN_SIZE=352
        export MOE_LAYER_FREQ="[1]*12"
    else
        export NUM_LAYERS=13 HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=2736 MOE_FFN_HIDDEN_SIZE=352
        export MOE_LAYER_FREQ="[0]*1+[1]*12"
    fi
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/local/home/bingli/moe/$JOB_NAME}"
_SHARE=$( [[ "$JOB_NAME" == *no-shar* ]] && echo "noshare" || echo "share" )
OUT="${OUT:-logs-eval/ablation/${JOB_NAME}}"
mkdir -p "$(dirname "$OUT")"

echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
source "$MODEL_CONFIG"   # sets MODEL_ARGS (needs MOE_ROUTER_TOPK etc. exported)

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH="$PWD/Megatron-LM"

# find one tokenized .bin and use its prefix (path without .bin) as the val slice
DATA_PREFIX="${ABL_DATA_PREFIX:-}"
if [[ -z "$DATA_PREFIX" && -n "$TRAIN_DATASET" ]]; then
    DATA_PREFIX=$(find "$TRAIN_DATASET" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
if [[ -z "$DATA_PREFIX" && -n "$TOKENIZED_DIR" ]]; then
    DATA_PREFIX=$(find "$TOKENIZED_DIR" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
echo "Ablation data prefix: ${DATA_PREFIX:-<none, will use embedded text>}"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port "${MASTER_PORT:-12399}" \
    scripts/analysis/expert_ablation.py \
    "${MODEL_ARGS[@]}" \
    --bf16 --seq-length 2048 --micro-batch-size 1 \
    --expert-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --max-tokens-to-oom 100000000 \
    --load "$CHECKPOINT_DIR" \
    --ablation-output "$OUT" \
    --ablation-data-prefix "$DATA_PREFIX" \
    --ablation-num-seqs "${ABL_NUM_SEQS:-32}" \
    --ablation-seq-len "${ABL_SEQ_LEN:-1024}"

echo "Done. Results: $OUT.npy / $OUT.png"
