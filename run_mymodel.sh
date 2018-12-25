#!/bin/sh
python -u train.py \
    --model mymodel \
    --kernel-size 2 \
    --hidden-dim 12 \
    --epochs 5 \
    --weight-decay 0. \
    --momentum 0.0 \
    --no-cuda \
    --batch-size 64 \
    --lr 0.01 | tee mymodel.log
