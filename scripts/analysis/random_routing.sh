#!/bin/bash
# Random-routing PPL ablation for a 150m shared or no-share checkpoint.
#
# Usage:
#   JOB_NAME=flame-moe-150m            ITER=3636 CHECKPOINT_DIR=/path bash scripts/analysis/random_routing.sh
#   JOB_NAME=flame-moe-150m-no-shared  ITER=3636 CHECKPOINT_DIR=/path bash scripts/analysis/random_routing.sh
#
# RR_COMBINE=affinity (default) or uniform

set -e
source devconfig.sh 2>/dev/null || true

if [[ -z "$MODEL_CONFIG" ]]; then
    if [[ "$JOB_NAME" == *no-shar* ]]; then
        MODEL_CONFIG=configs/model/flame-moe-no-shared.sh
    else
        MODEL_CONFIG=configs/model/flame-moe.sh
    fi
fi

if [[ -z "$HIDDEN_SIZE" ]]; then
    NOSHARE=$([[ "$JOB_NAME" == *no-shar* ]] && echo 1 || echo 0)
    if [[ "$JOB_NAME" == *270m* ]]; then
        export NUM_LAYERS=16 HIDDEN_SIZE=768 FFN_HIDDEN_SIZE=3648 MOE_FFN_HIDDEN_SIZE=512 NUM_EXPERTS=128
        [[ $NOSHARE == 1 ]] && export MOE_LAYER_FREQ="[1]*16" || export MOE_LAYER_FREQ="[0]*1+[1]*15"
    elif [[ "$JOB_NAME" == *38m* ]]; then
        export NUM_LAYERS=$([[ $NOSHARE == 1 ]] && echo 8 || echo 9)
        export HIDDEN_SIZE=256 FFN_HIDDEN_SIZE=1368 MOE_FFN_HIDDEN_SIZE=176
        [[ $NOSHARE == 1 ]] && export MOE_LAYER_FREQ="[1]*8" || export MOE_LAYER_FREQ="[0]*1+[1]*8"
    else  # 150m
        export HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=2736 MOE_FFN_HIDDEN_SIZE=352
        if [[ $NOSHARE == 1 ]]; then
            export NUM_LAYERS=12 MOE_LAYER_FREQ="[1]*12"
        else
            export NUM_LAYERS=13 MOE_LAYER_FREQ="[0]*1+[1]*12"
        fi
    fi
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/local/home/bingli/moe/$JOB_NAME}"
OUT="${OUT:-logs-eval/random-routing/${JOB_NAME}}"
mkdir -p "$(dirname "$OUT")"

echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
source "$MODEL_CONFIG"

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH="$PWD/Megatron-LM"

DATA_PREFIX="${RR_DATA_PREFIX:-}"
if [[ -z "$DATA_PREFIX" && -n "$TRAIN_DATASET" ]]; then
    DATA_PREFIX=$(find "$TRAIN_DATASET" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
if [[ -z "$DATA_PREFIX" && -n "$TOKENIZED_DIR" ]]; then
    DATA_PREFIX=$(find "$TOKENIZED_DIR" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
echo "Data prefix: ${DATA_PREFIX:-<none, embedded text>}"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port "${MASTER_PORT:-12398}" \
    scripts/analysis/random_routing.py \
    "${MODEL_ARGS[@]}" \
    --bf16 --seq-length 2048 --micro-batch-size 1 \
    --expert-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --max-tokens-to-oom 100000000 \
    --load "$CHECKPOINT_DIR" \
    --rr-output "$OUT" \
    --rr-data-prefix "$DATA_PREFIX" \
    --rr-num-seqs "${RR_NUM_SEQS:-32}" \
    --rr-seq-len "${RR_SEQ_LEN:-1024}" \
    --rr-repeats "${RR_REPEATS:-3}" \
    ${RR_GUMBEL_BETA:+--rr-gumbel-beta $RR_GUMBEL_BETA}

echo "Done. Results: $OUT.npz / $OUT.png"
