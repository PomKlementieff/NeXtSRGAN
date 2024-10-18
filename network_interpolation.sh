#!/bin/bash

python3 net_interp.py \
    --cfg_path1="./configs/psnr.yaml" \
    --cfg_path2="./configs/nextsrgan.yaml" \
    --img_path="./data/PIPRM_3_crop.png" \
    --save_image=True \
    --save_ckpt=True