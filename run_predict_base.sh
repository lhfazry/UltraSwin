#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode=predict \
    --data_dir=/workspace/EchoNet \
    --ckpt_path=lightning_logs/ultraswin/version_62/checkpoints/epoch\=1-step\=190.ckpt \
    --batch_size=8 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=base