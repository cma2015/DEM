# Test biodem
from biodem import dem_impute, dem_select_varpca, dem_select_rf, dem_model, dem_predict, dem_rank
import os

try:
    # Make a directory to store testing results
    os.mkdir('tests')
except FileExistsError:
    pass


# Define the data paths
path_phenotypes = "../data/for_biodem_model_test/regression/pheno_90.csv"
path_omics_1 = "../data/for_biodem_model_test/regression/mrna_50.csv"
path_omics_2 = "../data/for_biodem_model_test/regression/snp_50.csv"


# Filter and impute missing values, then scale the data
path_treated_omics = 'tests/test_biodem_treated_omics.csv'
path_treated_pheno = 'tests/test_biodem_treated_pheno.csv'
dem_impute(path_omics_1, path_treated_omics,
           path_phenotypes, path_treated_pheno,
           0.1, True, True)

# Feature selection
dem_select_varpca(path_treated_omics, path_treated_pheno,
                  path_treated_omics,
                  0.01, 0.5)
dem_select_rf(path_treated_omics, path_treated_pheno,
              path_treated_omics,
              20, 0.2, 500, [1,2,3], [4,5,6])


# Construct DEM model
path_save_model = 'tests/test_biodem_model_output.pth'
dem_model([path_omics_1, path_omics_2],
          path_phenotypes,
          path_save_model,
          True, 0.3, [1,2,3], 8, 0.01, 5, 2, 0.1, 10)


# Predict phenotypes
path_model_i = path_save_model.replace(".pth", "_seedsp1.pth")
path_predicted_pheno = 'tests/test_biodem_predict_output.csv'
dem_predict(path_model_i,
            [path_omics_1, path_omics_2],
            path_predicted_pheno)


# Functional gene mining
path_output_gene_rank = 'tests/test_biodem_gene_rank_output.csv'
dem_rank(path_model_i,
         [path_omics_1, path_omics_2],
         path_phenotypes,
         path_output_gene_rank,
         True, [1,2,3])
