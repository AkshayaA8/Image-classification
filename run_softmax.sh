#!/bin/sh

python -u train.py \
    --model softmax \
    --epochs 1 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --no-cuda \
    --batch-size 1024 \
    --lr 0.01 | tee softmax.log

