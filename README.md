# [CASS: Cross Architectural Self-Supervision for Medical Image Analysis](https://arxiv.org/abs/2206.04170v2)

Official PyTorch implementation of **CASS**, from the following paper:

[CASS: Cross Architectural Self-Supervision for Medical Image Analysis](https://arxiv.org/abs/2206.04170v2). arXiv 2022.\
[Pranav Singh](https://pranavsinghps1.github.io/), [Elena Sizikova](https://esizikova.github.io/), [Jacopo Cirrone](https://scholar.google.com/citations?user=DF9nXUYAAAAJ&hl=en) \
New York University.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/partial-label-learning-on-autoimmune-dataset)](https://paperswithcode.com/sota/partial-label-learning-on-autoimmune-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-autoimmune-dataset)](https://paperswithcode.com/sota/classification-on-autoimmune-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-brain-tumor-mri-dataset)](https://paperswithcode.com/sota/classification-on-brain-tumor-mri-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-isic-2019)](https://paperswithcode.com/sota/classification-on-isic-2019?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/partial-label-learning-on-isic-2019)](https://paperswithcode.com/sota/partial-label-learning-on-isic-2019?p=cass-cross-architectural-self-supervision-for)


## Installation
Kindly use the requirements.txt for installing all the required dependencies.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Evaluation
For evaluation simply use the saved weight at:
```
model_vit=torch.load('/scratch/ps4364/BTMRI/code/1p-data/Cov-t/covt-r50-label-bMRI-100p-es-0.pt')

```


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [Phil Wang](https://github.com/lucidrains/vit-pytorch), [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)

## Citation
If you find this repository helpful, please consider citing:
```
@article{singh2022cass,
  title={CASS: Cross Architectural Self-Supervision for Medical Image Analysis},
  author={Singh, Pranav and Sizikova, Elena and Cirrone, Jacopo},
  journal={arXiv preprint arXiv:2206.04170},
  year={2022}
}
```
