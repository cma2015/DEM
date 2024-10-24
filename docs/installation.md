# Installation

> [Conda](https://conda.io/projects/conda/en/latest/index.html) / [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) is recommended for installation.

## Create a conda environment:
```sh
mamba create -n dem python=3.11
mamba activate dem

# Install PyTorch with CUDA support
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Install biodem from [PyPI](https://pypi.org/project/biodem)
```sh
pip install biodem
```

<br></br>

---

We also provide a lightweight tool **`pregv`** for VCF & GFF file processing and SNP encoding, which is implemented in Rust.
[Click to download](https://github.com/cma2015/DEM/blob/main/bin) or **install with**:
``` sh title="Install pregv"
# mamba activate dem
dem-install-pregv
```
You may need to deactivate and reactivate the conda environment after installation.

Test your installation with:
``` sh title="Test pregv installation"
pregv --help
```
