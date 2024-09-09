#!/usr/bin/env bash

CONFIG=cfgs/vic/vimi_960x540_12e_bs2_lidar.py
CHECKPOINT=work_dirs/0907_EMIFF_960x540_12e_bs1x4_lr3e-05/epoch_1.pth
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --out work_dirs/0907_EMIFF_960x540_12e_bs1x4_lr3e-05/results.pkl \
    --show-dir ./show_dir/0907_EMIFF_960x540_12e_bs1x4_lr3e-05/
    ${@:4}
