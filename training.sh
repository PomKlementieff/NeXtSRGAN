#!/bin/bash

python3 train_psnr.py --cfg_path="./configs/psnr.yaml" --gpu=0
python3 train_nextsrgan.py --cfg_path="./configs/nextsrgan.yaml" --gpu=0