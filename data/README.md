# Description of data

## Directory structure

---

+ The preprocessed datasets for regression and classification tasks.
    The two datasets are generated from *1001 Arabidopsis Genomes* and *The Cancer Genome Atlas Program (TCGA)* that were used in our research respectively.
```sh
./preprocessed
├── 01_arabidopsis
│   ├── FT.zip                # The Arabidopsis flowering time (FT) dataset has 3 omics and total 5000 features for 600 samples
│   └── RLN.zip               # The Arabidopsis rosette leaf number (RLN) dataset has 3 omics and total 5000 features for 543 samples
└── 02_human
    ├── BRCA.zip              # The breast invasive carcinoma (BRCA) dataset has 3 omics and total 2502 features for 875 samples
    ├── KIPAN.zip             # The pan-kidney cohort (KIPAN) dataset has 3 omics and total 3887 features for 572 samples
    ├── LGG.zip               # The brain lower grade glioma (LGG) dataset has 3 omics and total 5861 features for 523 samples
    └── ROSMAP.zip            # The religious orders study and memory and aging project (ROSMAP) dataset has total 599 features for 351 samples
```

---

+ Two small datasets are provided for `biodem` modeling test, for classification and regression tasks respectively.
    The partial data are stored in the following directory structure:
```sh
./for_biodem_model_test
├── classification            # The partial KIPAN dataset has 3 omics and total 300 features for 172 samples
│   ├── labels_172.csv        # Cancer type labels of 172 samples of the KIPAN dataset (encoded in integer)
│   ├── meth_100.csv          # 100 DNA methylation features of 172 KIPAN samples
│   ├── mirna_100.csv         # 100 miRNA expression features of 172 KIPAN samples
│   └── mrna_100.csv          # 100 mRNA expression features of 172 KIPAN samples
└── regression                # The partial Arabidopsis flowering time dataset has 3 omics and total 250 features for 90 samples
    ├── pheno_90.csv          # Flowering time phenotypes of 90 Arabidopsis samples (z-score treated)
    ├── meth_150.csv          # 150 DNA methylation features of 90 Arabidopsis samples (50 mCHG features, 50 mCHH features and 50 mCG features)
    ├── mrna_50.csv           # 50 mRNA expression features of 90 Arabidopsis samples
    └── snp_50.csv            # 50 single nucleotide polymorphism (SNP) features of 90 Arabidopsis samples
```

---

+ The unprocessed data examples for feature selection and SNP processing are stored in the following directory structure:
```sh
./unprocessed_examples
├── omics_exp_n23498.csv      # A partial unprocessed mRNA expression matrix of 50 samples and 23498 features
├── phenotypes_ft16_n1123.csv # Flowering time phenotypes of 1123 Arabidopsis samples
├── gtf_tair10.1_.gtf         # A partial TAIR10.1 GTF file, utilized for extracting SNP-related gene coordinates essential for SNP transformation
└── snp_1001genomes.csv       # A partial unprocessed SNP matrix of 1001 Genomes
```
