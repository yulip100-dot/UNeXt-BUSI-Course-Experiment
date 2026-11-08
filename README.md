# UNeXt：BUSI 图像分割课程实验

本仓库用于“图像处理与分析”课程学习，基于 UNeXt 作者公开代码进行复现与少量修改。原模型及其核心实现由原作者提出和开发，本仓库主要记录输入分辨率调整、预测阈值设置及相关实验分析，不属于独立原创模型，也不是作者官方仓库。

## 原始来源

- 原作者仓库：https://github.com/jeya-maria-jose/UNeXt-pytorch
- 论文：UNeXt: MLP-based Rapid Medical Image Segmentation Network
- 论文作者：Jeya Maria Jose Valanarasu、Vishal M. Patel
- 论文链接：https://arxiv.org/abs/2203.04967
- 发表会议：MICCAI 2022

## 本次修改

本次修改未改变 UNeXt 网络结构，主要涉及以下三个文件：

1. `train.py`：将默认输入分辨率由 256×256 调整为 128×128，仍可通过 `--input_h` 和 `--input_w` 指定其他尺寸。
2. `metrics.py`：为 `iou_score` 增加可配置的 `threshold` 参数，默认值为 0.5。
3. `val.py`：增加 `--threshold` 命令行参数，使指标计算与预测掩膜保存统一采用“概率大于阈值”的判断规则；不同阈值的结果分别保存，并使用 PNG 格式保存二值掩膜。

原代码的评分阈值已经是 0.5。本次阈值相关修改的重点是将其参数化，并统一评分与保存结果的规则，便于开展不同阈值的对照实验。

## 数据来源

课程实验使用 BUSI 乳腺超声图像数据集，进行病灶区域的二值分割。

数据下载入口：
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

本仓库不重新分发数据集。数据集的使用与再分发应遵守其自身的授权和使用条款。

## 版权与致谢

原始项目采用 MIT 许可证，本仓库保留原始 `LICENSE` 文件及其中的版权声明：

Copyright (c) 2022 Jeya Maria Jose

课程修改部分沿用 MIT 许可证。原始代码、模型设计及相关研究成果的署名归原作者，本仓库仅对上述课程实验修改作出说明。

感谢 UNeXt 作者公开论文和代码。原仓库中对 UNet++、SegFormer、AS-MLP 等项目的来源说明、致谢和论文引用一并保留。


## 原作者项目说明

以下保留原仓库 README 内容，其中的模型介绍和论文结果属于原作者工作，不代表本课程实验的结果。

Official Pytorch Code base for [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967), MICCAI 2022

[Paper](https://arxiv.org/abs/2203.04967) | [Project](https://jeya-maria-jose.github.io/UNext-web/)

## Introduction

UNet and its latest extensions like TransUNet have been the leading medical image segmentation methods in recent years. However, these networks cannot be effectively adopted for rapid image segmentation in point-of-care applications as they are parameter-heavy, computationally complex and slow to use.  To this end, we propose UNeXt which is a Convolutional multilayer perceptron (MLP) based network for image segmentation. We design UNeXt in an effective way with an early convolutional stage and a MLP stage in the latent stage. We propose a tokenized MLP block where we efficiently tokenize and project the convolutional features and use MLPs to model the representation. To further boost the performance, we propose shifting the channels of the inputs while feeding in to MLPs so as to focus on learning local dependencies. Using tokenized MLPs in latent space reduces the number of parameters and computational complexity while being able to result in a better representation to help segmentation. The network also consists of skip connections between various levels of encoder and decoder.   We test UNeXt on multiple medical image segmentation datasets and show that we reduce the number of parameters by 72x, decrease the computational complexity by 68x, and improve the inference speed by 10x while also obtaining better segmentation performance over the  state-of-the-art medical image segmentation architectures.

<p align="center">
  <img src="imgs/unext.png" width="800"/>
</p>


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
git clone https://github.com/jeya-maria-jose/UNeXt-pytorch
cd UNeXt-pytorch
```

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate unext
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```

## Datasets

1) ISIC 2018 - [Link](https://challenge.isic-archive.com/data/)
2) BUSI - [Link](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)

## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
python train.py --dataset <dataset name> --arch UNext --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 500 --input_w 512 --input_h 512 --b 8
```
2. Evaluate.
```
python val.py --name <exp name>
```

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet), [Segformer](https://github.com/NVlabs/SegFormer), and [AS-MLP](https://github.com/svip-lab/AS-MLP). Naming credits to [Poojan](https://scholar.google.co.in/citations?user=9dhBHuAAAAAJ&hl=en).

### Citation:
```
@article{valanarasu2022unext,
  title={UNeXt: MLP-based Rapid Medical Image Segmentation Network},
  author={Valanarasu, Jeya Maria Jose and Patel, Vishal M},
  journal={arXiv preprint arXiv:2203.04967},
  year={2022}
}
```
