# [NextSRGAN-tf2](https://github.com/your-github-username/nextsrgan-tf2)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/your-github-username/nextsrgan-tf2.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/your-github-username/nextsrgan-tf2/context:python)
![Star](https://img.shields.io/github/stars/your-github-username/nextsrgan-tf2)
![Fork](https://img.shields.io/github/forks/your-github-username/nextsrgan-tf2)
![License](https://img.shields.io/github/license/your-github-username/nextsrgan-tf2)

:fire: NextSRGAN (Next Super-Resolution Generative Adversarial Network) implemented in TensorFlow 2.0+. This implementation enhances ESRGAN with a ConvNeXt-based discriminator for superior realism. :fire:

> NextSRGAN introduces a novel discriminator design inspired by the ConvNeXt architecture to improve upon the structure of ESRGAN. This new approach leads to more effective enhancement of image quality, particularly in facial image super-resolution.

Original Paper: [NeXtSRGAN: Enhancing Super-Resolution GAN with ConvNeXt Discriminator for Superior Realism](link-to-your-paper)

## Contents
:bookmark_tabs:

* [Installation](#Installation)
* [Data Preparing](#Data-Preparing)
* [Training and Testing](#Training-and-Testing)
* [References](#References)

***

## Installation
:pizza:

Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) or just use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone https://github.com/your-github-username/nextsrgan-tf2.git
cd nextsrgan-tf2
```

### Conda
```bash
conda env create -f environment.yml
conda activate nextsrgan-tf2
```

### Pip

```bash
pip install -r requirements.txt
```

****

## Data Preparing
:beer:

This implementation uses DIV2K and KID-F datasets for training, and various benchmark datasets for testing.

### Training Dataset

**Step 1**: Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [KID-F](https://github.com/KU-CVLAB/KID-F) datasets.

**Step 2**: Extract them into `./data/DIV2K/` and `./data/KID-F/` respectively.

**Step 3**: Use the provided scripts to preprocess the data:
```bash
python data/extract_subimages.py
python data/convert_train_tfrecord.py
```

### Testing Dataset

Download the common image SR datasets (Set5, Set14, BSD100, Urban100, Manga109, Historical) and place them in the `./data/` directory.

****

## Training and Testing
:lollipop:

### Config File
Modify the config files in `./configs/*.yaml` for training and testing settings.

### Training

#### Pretrain PSNR
```bash
python train_psnr.py --cfg_path="./configs/psnr.yaml" --gpu=0
```

#### NextSRGAN
```bash
python train_nextsrgan.py --cfg_path="./configs/nextsrgan.yaml" --gpu=0
```

### Testing

Evaluate the models on the testing dataset:

```bash
# Test NextSRGAN model
python test.py --cfg_path="./configs/nextsrgan.yaml"
# or
# PSNR pretrain model
python test.py --cfg_path="./configs/psnr.yaml"
```

### SR Input Image

Upsample a single image:

```bash
python test.py --cfg_path="./configs/nextsrgan.yaml" --img_path="./data/your_image.png"
```

### Network Interpolation

Produce interpolation results between PSNR-oriented and GAN-based methods:

```bash
python net_interp.py --cfg_path1="./configs/psnr.yaml" --cfg_path2="./configs/nextsrgan.yaml" --img_path="./data/your_image.png" --save_image=True --save_ckpt=True
```

****

## References
:hamburger:

1. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., et al., "Esrgan: Enhanced super-resolution generative adversarial networks," In Proceedings of the European conference on computer vision workshops, 2018.
2. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., and Xie, S., "A convnet for the 2020s," In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976-11986, 2022.
3. Agustsson, E., and Timofte, R., "Ntire 2017 challenge on single image super-resolution: Dataset and study," In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 126-135, 2017.
4. Kim, D. K., Han, D. G., Kwon, H. W., Jeong, D. I., and Jeong, C. H., "K-pop Idol Dataset - Female: High Quality K-pop Female Idol Face Image Dataset with Identity Labels," 2022.
