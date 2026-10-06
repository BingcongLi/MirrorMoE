#!/bin/bash
# Training launcher script for MirrorMoE.

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

source ${MODEL_CONFIG:-configs/model/mirror-moe.sh}
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $TRAIN_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 90,5,5
)

SAVE_ARGS=(
    --log-interval 5
    --log-throughput
    --save $TRAIN_WEIGHTS
    --save-interval $SAVE_INTERVAL
    --load $TRAIN_WEIGHTS
    --eval-interval $EVAL_INTERVAL
    --wandb-save-dir $TRAIN_WEIGHTS
    --wandb-project $WANDB_PROJECT
    --wandb-exp-name $JOB_ID
    --tensorboard-dir $TRAIN_WEIGHTS
)

cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"
