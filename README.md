<div align="center">

# DEM

## Dual-extraction modeling: A multi-modal deep-learning architecture for phenotypic prediction and functional gene mining of complex traits

[![pypi-badge](https://img.shields.io/pypi/v/biodem)](https://pypi.org/project/biodem)
[![pypi-badge](https://img.shields.io/pypi/dm/biodem.svg?label=Pypi%20downloads)](https://pypi.org/project/biodem)
![License](https://img.shields.io/github/license/cma2015/DEM)
[![docs-badge](https://cma2015.github.io/DEM)](https://cma2015.github.io/DEM)

</div>

> # Latest news
>
> v0.9.1 is released with a lot of improvements!
>
> Please checkout the tutorials and documentations at [cma2015.github.io/DEM](https://cma2015.github.io/DEM).

+ The **DEM** is implemented in the Python package [**`biodem`**](https://pypi.org/project/biodem), which comprises 4 modules: data preprocessing, dual-extraction modeling, phenotypic prediction, and functional gene mining.
+ For more details, please check out our [publication](https://doi.org/10.1016/j.xplc.2024.101002). [🖱️Click to copy citation](#citation)

<table style="border-collapse: collapse; border: 1px solid black;">
  <tr>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/cma2015/DEM/blob/main/docs/images/fig_1.png?raw=true" alt="DEM architecture"   /></td>
    <td style="padding: 5px;background-color:#fff;"><img src= "https://github.com/cma2015/DEM/blob/main/docs/images/fig_7.png?raw=true" alt="Modules of biodem"   /></td>
  </tr>
</table>

## Installation

### System requirements
+ Python 3.10 / 3.11 / 3.12.
+ Optional: Hardware accelerator supporting [PyTorch](https://pytorch.org).
> Recommended: NVIDIA graphics card with 12GB memory or larger.

### Install `biodem`
> [Conda](https://conda.io/projects/conda/en/latest/index.html) / [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) is recommended for installation.

1. Create a conda environment:
    ```sh
    mamba create -n dem python=3.11
    mamba activate dem

    # Install PyTorch with CUDA support
    mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

2. Install biodem package from [PyPI](https://pypi.org/project/biodem)
    ```sh
    pip install biodem
    ```

## Usage

Please checkout the documentations at [cma2015.github.io/DEM](https://cma2015.github.io/DEM).

<br></br>

## [`biodem`](https://pypi.org/project/biodem) comprises 4 functional modules:

### 1. Data preprocessing

> **Nested cross-validation** is recommended for data preprocessing.

+ _Steps:_
    1. Split data into nested cross-validation sets.
    2. Imputation & standardization.
    2. Feature selection using the variance threshold filter and Random Forests.
    3. SNP2Gene transformation.

### 2. Dual-extraction modeling

+ It takes preprocessed multi-omics data and phenotypic data as inputs. DEM is capable of performing both classification and regression tasks.

### 3. Phenotypic prediction

+ It loads the trained DEM model checkpoint and performs phenotypic prediction.

### 4. Functional gene mining

+ It performs functional gene mining based on the trained DEM model through _feature ranking by permutation_.


<br></br>

# Citation

Please cite our paper if you use this package:

```bibtex
@article{renDualextractionModelingMultimodal2024a,
  title = {Dual-Extraction Modeling: {{A}} Multi-Modal Deep-Learning Architecture for Phenotypic Prediction and Functional Gene Mining of Complex Traits},
  shorttitle = {Dual-Extraction Modeling},
  author = {Ren, Yanlin and Wu, Chenhua and Zhou, He and Hu, Xiaona and Miao, Zhenyan},
  year = {2024},
  month = sep,
  journal = {Plant Communications},
  volume = {5},
  number = {9},
  pages = {101002},
  issn = {25903462},
  doi = {10.1016/j.xplc.2024.101002},
  langid = {english}
}
```

# Asking for help

If you have any questions, please contact us via [GitHub issues](https://github.com/cma2015/dem/issues) or [email](mailto:ryl1999@126.com) us.
