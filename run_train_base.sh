#!/bin/bash

nohup python main.py \
    --data_dir=/workspace/EchoNet \
    --pretrained=pretrained/swin_base_patch244_window877_kinetics400_22k.pth \
    --batch_size=8 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=base \
    --max_epoch=100