#!/bin/bash

# NeXtSRGAN model
python3 test.py --cfg_path="./configs/nextsrgan.yaml" --img_path="./data/baboon.png"

# PSNR pretrain model
# python3 test.py --cfg_path="./configs/psnr.yaml" --img_path="./data/baboon.png"