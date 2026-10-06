#!/bin/bash
# Routing-induced variance for a shared or no-share flame checkpoint.
#
#   JOB_NAME=flame-moe-270m           ITER=6100 CHECKPOINT_DIR=/path bash scripts/analysis/routing_variance.sh
#   JOB_NAME=flame-moe-270m-no-shared ITER=6100 CHECKPOINT_DIR=/path bash scripts/analysis/routing_variance.sh

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
    else  # 150m
        export HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=2736 MOE_FFN_HIDDEN_SIZE=352
        if [[ $NOSHARE == 1 ]]; then export NUM_LAYERS=12 MOE_LAYER_FREQ="[1]*12"
        else export NUM_LAYERS=13 MOE_LAYER_FREQ="[0]*1+[1]*12"; fi
    fi
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/local/home/bingli/moe/$JOB_NAME}"
OUT="${OUT:-logs-eval/routing-variance/${JOB_NAME}}"
mkdir -p "$(dirname "$OUT")"

echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
source "$MODEL_CONFIG"

export OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH="$PWD/Megatron-LM"

DATA_PREFIX="${RV_DATA_PREFIX:-}"
if [[ -z "$DATA_PREFIX" && -n "$TRAIN_DATASET" ]]; then
    DATA_PREFIX=$(find "$TRAIN_DATASET" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
if [[ -z "$DATA_PREFIX" && -n "$TOKENIZED_DIR" ]]; then
    DATA_PREFIX=$(find "$TOKENIZED_DIR" -name '*.bin' 2>/dev/null | head -1 | sed 's/\.bin$//')
fi
echo "Data prefix: ${DATA_PREFIX:-<none>}"

GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port "${MASTER_PORT:-12394}" \
    scripts/analysis/routing_variance.py \
    "${MODEL_ARGS[@]}" \
    --bf16 --seq-length 2048 --micro-batch-size 1 \
    --expert-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --max-tokens-to-oom 100000000 \
    --load "$CHECKPOINT_DIR" \
    --rv-output "$OUT" \
    --rv-data-prefix "$DATA_PREFIX" \
    --rv-num-seqs "${RV_NUM_SEQS:-16}" \
    --rv-seq-len "${RV_SEQ_LEN:-512}" \
    --rv-draws "${RV_DRAWS:-24}" \
    --rv-micro "${RV_MICRO:-4}" \
    ${RV_GUMBEL_BETA:+--rv-gumbel-beta $RV_GUMBEL_BETA}

echo "Done. Results: $OUT.npz / $OUT.png"
