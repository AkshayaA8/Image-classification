#!/bin/sh

python -u train.py \
    --model convnet \
    --kernel-size 1 \
    --hidden-dim 12 \
    --epochs 1 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --no-cuda \
    --batch-size 64 \
    --lr 0.01 | tee convnet.log

