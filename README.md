
# Official PyTorch implementation of **CASS**, from the following paper:


[CASS: Cross Architectural Self-Supervision for Medical Image Analysis](https://arxiv.org/abs/2206.04170). arXiv 2022.

[Pranav Singh](https://pranavsinghps1.github.io/), [Elena Sizikova](https://esizikova.github.io/), [Jacopo Cirrone](https://scholar.google.com/citations?user=DF9nXUYAAAAJ&hl=en)

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

This is a private dataset [Van Buren et al.] [1] of autoimmunity biopsies of 198 samples. This is a multi-label class classification. For train/validation/test splits we follow an 80/10/10 split.

### DERMOFIT Datset
This is a paid dataset sourced by the University of Edinburgh, it contains 1,300 samples of high quality skin lesions. [2]

### Brain MRI Classification

Courtesy of [Cheng, Jun] [3] This dataset contains 7k samples of brain MRI with different tumour-related diseases. We perform multi-class classification in this context. Train/validation/test splits have already been provided by the dataset curator.

  

### SIIM-ISIC 2019 Dataset

This is a collection of skin lesions images contributed to the [2019 SIIM-ISIC challenge] [4]. This contains 25k samples and is a multi-class classification problem. For train/validation/test splits we follow an 80/10/10 split.

  

## Specification of dependencies


```

pip install torchmetrics

pip install torchcontrib

pip install pytorch-lightning

pip install timm

```

Note: The code has been tested with Pytorch version 1.11. From PyTorch version 1.12, the GELU function has an added parameter, which might not work with older versions of Timm models and may raise errors.

We provide the list of dependencies in requirements.txt and an extensive list used on our development system in full_reqs.txt.

# Pre-processing


We assume that a train.csv containing the image's address and corresponding labels is present for each dataset. Similarly, a test.csv containing the image addresses and labels for the test set is also present. We split the dataset in a ratio of 70/10/20 for training, validation, and testing except for brain MRI classification. The curators had already split the dataset into training and testing sets for brain tumor MRI classification.

Furthermore, for creating class weights for Focal loss used during downstream fine-tuning, we use the `normalize` function in EDA.ipynb. Since it is a jupyter notebook, sequentially executing the cells is recommended.

If the accompanying CSV is not present, we can create it using the EDA.ipynb; we assume that the dataset is stored at `/scratch/Dermofit/`.

## Explore files

-- [CASS](https://github.com/pranavsinghps1/CASS)/[CASS.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/CASS.ipynb  "CASS.ipynb") : contains the code for self-supervised and downstream supervised fine-tuning. For the supervised fine-tuning we use Focal loss to address the class imbalance problem so the class-wise distribution of the dataset is required.

Sequentially running the notebooks should render the required result. See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.


-- [CASS](https://github.com/pranavsinghps1/CASS)/[eval.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/eval.ipynb  "eval.ipynb") contains the evaluation code for the trained and saved model.

--  [CASS](https://github.com/pranavsinghps1/CASS)/[Examples](https://github.com/pranavsinghps1/CASS/tree/master/Examples)/[MedMNIST](https://github.com/pranavsinghps1/CASS/tree/master/Examples/MedMNIST)/[MNIST Get-started-CASS.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/Examples/MedMNIST/MNIST%20Get-started-CASS.ipynb "MNIST Get-started-CASS.ipynb") contains the required preprocessing to get started with CASS downstream labelled training. 

--  [CASS](https://github.com/pranavsinghps1/CASS)/[Examples](https://github.com/pranavsinghps1/CASS/tree/master/Examples)/[MedMNIST](https://github.com/pranavsinghps1/CASS/tree/master/Examples/MedMNIST)/[CASS.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/Examples/MedMNIST/CASS.ipynb "CASS.ipynb") once we have the preprocessed data from [MNIST Get-started-CASS.ipynb](https://github.com/pranavsinghps1/CASS/blob/master/Examples/MedMNIST/MNIST%20Get-started-CASS.ipynb "MNIST Get-started-CASS.ipynb") we can get started with self-supervised training and supervised downstream fine-tuning.  

## Updates
### July 24, 2022
* Added example and support for [MedMNIST](https://github.com/MedMNIST/MedMNIST) datasets. This includes 12 datasets for 2D of different sizes (range: 780 to 236,386 samples) and modalities.

### June 23, 2022
* Initial Code release for CASS


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


-  `[1]: https://www.sciencedirect.com/science/article/abs/pii/S0022175922000205`

-  `[2]:https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library`

-  `[3]:https://figshare.com/articles/dataset/brain_tumor_dataset/1512427` & `https://www.hindawi.com/journals/cin/2022/3236305/`

-  `[4]:https://challenge.isic-archive.com/data/#2019`
