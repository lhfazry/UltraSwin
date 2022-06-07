#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode=validate \
    --data_dir=/workspace/EchoNet \
    --ckpt_path=lightning_logs/ultraswin/version_26/checkpoints/epoch\=4-step\=8955.ckpt \
    --batch_size=16 \
    --num_workers=4 \
    --accelerator=gpu 