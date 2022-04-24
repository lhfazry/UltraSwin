#!/bin/bash

LINK="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth"
# mkdir -p $FILE >> /dev/null
FILE="pretrained/swin_base_patch4_window12_384_22k.pth"
curl -L $LINK --output $FILE

exit 1