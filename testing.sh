#!/bin/bash

# Test NeXtSRGAN model
python3 test.py --cfg_path="./configs/nextsrgan.yaml"

# PSNR pretrain model
python3 test.py --cfg_path="./configs/psnr.yaml"