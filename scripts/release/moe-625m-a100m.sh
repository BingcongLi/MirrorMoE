#!/bin/bash

export NUM_LAYERS=9
export HIDDEN_SIZE=512
export FFN_HIDDEN_SIZE=2736
export MOE_FFN_HIDDEN_SIZE=352
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MOE_ROUTER_TOPK=6
export MICRO_BATCH_SIZE=32
export NUM_EXPERTS=128
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=4
export TRAIN_ITERS=3636
export SAVE_INTERVAL=726
export EVAL_INTERVAL=726
export LR=3e-4
export MIN_LR=3e-5
JOB_NAME=flame-moe-150m-wide bash scripts/training/flame-moe.sh
