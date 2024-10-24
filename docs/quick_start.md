# Quick start

## Input and output data formats

These are data formats that users can prepare for `biodem` modules.

### Tabular data

The Modules [`DEMDataset`](reference/biodem.utils.data_ncv.md#biodem.utils.data_ncv.DEMDataset) can read tabular data in the CSV and Parquet formats.

+ The first column contains sample IDs that will be read as string type.
+ The first column's name must be `"ID"`.

Please see the detailed requirements in the [Modules > Utilities > Preprocessing Data > `DEMDataset`](reference/biodem.utils.data_ncv.md#biodem.utils.data_ncv.DEMDataset).

Example:

ID | gene_1 | gene_2
:----------- |:-------------:|:-----------:
id_1         | 0.87        | 0.03
id_2         | 0.34        | 0.65

### VCF
Genotype data in Variant Call Format (VCF).

### GFF
Genomic annotations in Generic Feature Format Version 3 (GFF3).

## Running tests

A simple usage example is provided in [`./tests/test_biodem.py`](https://github.com/cma2015/DEM/blob/main/tests/test_biodem.py). Please refer to the script and "Modules" documentation for more details.

``` sh title="Run tests"
git clone https://github.com/cma2015/DEM.git
cd DEM/tests
# Activate your conda environment
# mamba activate dem
python test_biodem.py
```
