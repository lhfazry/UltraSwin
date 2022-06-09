#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode=test \
    --data_dir=/workspace/EchoNet \
    --ckpt_path=lightning_logs/ultraswin/version_86/checkpoints/epoch\=19-step\=4480.ckpt \
    --batch_size=8 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=base