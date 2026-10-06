#!/bin/bash

export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MOE_ROUTER_TOPK=6
export MICRO_BATCH_SIZE=32
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212
export LR=3e-4
export MIN_LR=3e-5
JOB_NAME=mirror-moe-38m bash scripts/training/mirror-moe.sh
