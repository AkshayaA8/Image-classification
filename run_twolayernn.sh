#!/bin/sh

python -u train.py \
    --model twolayernn \
    --hidden-dim 15 \
    --epochs 8 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --no-cuda \
    --batch-size 1024 \
    --lr 0.01 | tee twolayernn.log

