# Official PyTorch implementation of **CASS**, from the following paper:

[CASS: Cross Architectural Self-Supervision for Medical Image Analysis](https://arxiv.org/abs/2206.04170v3). arXiv 2022.\
[Pranav Singh](https://pranavsinghps1.github.io/), [Elena Sizikova](https://esizikova.github.io/), [Jacopo Cirrone](https://scholar.google.com/citations?user=DF9nXUYAAAAJ&hl=en) \
New York University.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/partial-label-learning-on-autoimmune-dataset)](https://paperswithcode.com/sota/partial-label-learning-on-autoimmune-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-autoimmune-dataset)](https://paperswithcode.com/sota/classification-on-autoimmune-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-brain-tumor-mri-dataset)](https://paperswithcode.com/sota/classification-on-brain-tumor-mri-dataset?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/classification-on-isic-2019)](https://paperswithcode.com/sota/classification-on-isic-2019?p=cass-cross-architectural-self-supervision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cass-cross-architectural-self-supervision-for/partial-label-learning-on-isic-2019)](https://paperswithcode.com/sota/partial-label-learning-on-isic-2019?p=cass-cross-architectural-self-supervision-for)


# Description and Assumptions

CASS stands for Cross-Architectural Self-Supervised Learning with its primary aim to work robustly with small batch size and limited computational resources, to make self-supervised learning accessible. We have tested CASS for various label fractions (1%, 10% and 100%), with three modalities in medical imaging (Brain MRI classification, Autoimmune biopsy cell classification and Skin Lesion Classification), various dataset sizes (198 samples, 7k samples and 25k samples) as well as with Multi-class and multi-label classification. The pipeline is also compatible with binary classification. 

For a detailed description of the datasets and tasks refer to the [CASS: Cross Architectural Self-Supervision for Medical Image Analysis](https://arxiv.org/abs/2206.04170v3) paper.

We have not accounted for any meta-data and the pipeline purely functions on image-label mapping. Labels are not required during the self-supervised training but are required for the downstream supervised training.

## Analysis of complexity (time & space)
As compared to existing state-of-the-art methods CASS is twice as computationally efficient. The size of CASS-trained models is the same as that trained by other state-of-the-art self-supervised methods.

## Datasets

### Dermatomyositis Autoimmunity Dataset
This is a private dataset  [Van Buren et al.] [1] of autoimmunity biopsies of 198 samples. This is a multi-label class classification. For train/validation/test splits we follow an 80/10/10 split.

### Brain MRI Classification
Courtesy of [Cheng, Jun] [2] This dataset contains 7k samples of brain MRI with different tumour-related diseases. We perform multi-class classification in this context.  Train/validation/test splits have already been provided by the dataset curator.

### SIIM-ISIC 2019 Dataset
 This is a collection of skin lesions images contributed to the [2019 SIIM-ISIC challenge] [3]. This contains 25k samples and is a multi-class classification problem.  For train/validation/test splits we follow an 80/10/10 split.

## Specification of dependencies
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

```
pip install torchmetrics
pip install torchcontrib
pip install pytorch-lightning
pip install timm
```
Note: The code has been tested with Pytorch version 1.11. From PyTorch version 1.12, the GELU function has an added parameter, which might not work with older versions of Timm models and may raise errors.

We provide an extensive list of dependencies used on our development system in full_reqs.txt.


## Explore files
-- CASS/[CASS.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/CASS.ipynb "CASS.ipynb") : contains the code for self-supervised and downstream supervised fine-tuning. For the supervised fine-tuning we use Focal loss to address the class imbalance problem so the class-wise distribution of the dataset is required.
Sequentially running the notebooks should render the required result. See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

-- CASS/[eval.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/eval.ipynb "eval.ipynb") contains the evaluation code for the trained and saved model.

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

## References

-   `[1]: https://www.sciencedirect.com/science/article/abs/pii/S0022175922000205`
- `[2]:https://figshare.com/articles/dataset/brain_tumor_dataset/1512427` & `https://www.hindawi.com/journals/cin/2022/3236305/`
- `[3]:https://challenge.isic-archive.com/data/#2019`