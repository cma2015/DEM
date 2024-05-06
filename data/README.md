# Description of data

## Directory structure

+ The preprocessed datasets for the testing of DEM regression and classification tasks are stored in the following directory structure (please decompress the zip files before using them):
```sh
# Phenotypes and multi-omics data are included.
./test_dem
├── BR.zip           # For classification task    # Osa    # Blast Resistance
├── DTT.zip          # For regression task        # Zma    # Days to tasseling
├── FT.zip           # For regression task        # Ath    # Flowering time
├── KNPE.zip         # For regression task        # Zma    # Kernel number per ear
├── KWPE.zip         # For regression task        # Zma    # Kernel weight per ear
├── PH.zip           # For regression task        # Zma    # Plant height
├── RLN.zip          # For regression task        # Ath    # Rosette leaf number
└── SS.zip           # For classification task    # Osa    # Straighthead susceptibility
```

+ The preprocessed datasets for the testing of SNP2Gene transformation are stored in the following directory structure (please decompress the 7z files before using them):
```sh
# Phenotypes and genotypes data are included with nested cross-validation splits.
./test_s2g
├── ath.7z
├── osa.7z
└── zma.7z
```
