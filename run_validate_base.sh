#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode=validate \
    --data_dir=/workspace/EchoNet \
    --ckpt_path=lightning_logs/ultraswin/version_18/checkpoints/epoch=19-step=8960.ckpt \
    --batch_size=2 \
    --num_workers=4 \
    --accelerator=gpu 