#!/bin/bash

#LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth"
LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth"
# mkdir -p $FILE >> /dev/null
#FILE="pretrained/swin_small_patch4_window7_224.pth"
FILE="pretrained/swin_base_patch4_window7_224_22k.pth"
FILE="pretrained/swin_base_patch244_window877_kinetics400_22k.pth"
FILE="pretrained/swin_small_patch4_window7_224_22k.pth"

curl -L $LINK --output $FILE

exit 1