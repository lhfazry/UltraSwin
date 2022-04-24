#!/bin/bash

LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
# mkdir -p $FILE >> /dev/null
FILE="pretrained/swin_small_patch4_window7_224.pth"
curl -L $LINK --output $FILE

exit 1