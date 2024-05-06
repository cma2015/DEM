# The test cases for biodem.

import biodem
import os
import time


time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
dir_work = os.path.join(os.getcwd(), 'tests_' + time_str)

try:
    # Make a directory to store testing results
    os.mkdir(dir_work)
except FileExistsError:
    exit("The directory already exists.")

# Unzip the test data into the working directory
os.system("unzip ../data/test_dem/KWPE.zip -d " + dir_work)

print("Test data unzipped to " + dir_work)
print(os.listdir(os.path.join(dir_work, "KWPE")))


# Define the data paths

path_tr_pheno = os.path.join(dir_work, "KWPE", "outer_zscore_labels_tr.csv")
path_te_pheno = os.path.join(dir_work, "KWPE", "outer_zscore_labels_te.csv")
path_tr_snp = os.path.join(dir_work, "KWPE", "snp_outer_tr.csv")
path_te_snp = os.path.join(dir_work, "KWPE", "snp_outer_te.csv")
path_tr_exp = os.path.join(dir_work, "KWPE", "exp_outer_tr.csv")
path_te_exp = os.path.join(dir_work, "KWPE", "exp_outer_te.csv")


# Simple test: data preprocessing with nested cross-validation

dir_ncv_out = os.path.join(dir_work, "try_ncv")
os.mkdir(dir_ncv_out)
biodem.data_prep_ncv_regre(
    outer_loop=3,
    inner_loop=3,
    input_path=path_tr_snp,
    output_dir=dir_ncv_out,
    raw_labels_path=path_tr_pheno,
    which_trait="KWPE",
    Threshold_Variance=0.01,
    RF_selected_num=50,
    number_of_trees=200,
    na_rate=0.1,
)


# Note: biodem.DEMTrainPipeline is provided for nested cross-validation
# This is a simple usage example for DEM model training with hyperparameter optimization.

dir_dem = os.path.join(dir_work, "try_dem")
os.mkdir(dir_dem)
dem_train_x = biodem.DEMTrain(
    log_dir=dir_dem,
    log_name='try_dem',
    trait_name="KWPE",
    n_label_class=1,
    paths_trn_omics=[path_tr_snp, path_tr_exp],
    paths_val_omics=[path_te_snp, path_te_exp],
    path_trn_label=path_tr_pheno,
    path_val_label=path_te_pheno,
    devices=1,
    n_heads=2,
    hidden_dim=512,
    patience=9,
    max_epochs=20,
    min_epochs=10,
)
dem_train_x.optimize(n_trials=3, storage='sqlite:///try_dem_' + time_str + '.db')


# Predict phenotypes

# TODO: add the code for predicting phenotypes
# biodem.predict_pheno(
#     model_path=,
#     omics_paths=,
#     result_dir=,
# )


# Functional gene mining (feature ranking)

# TODO: add the code for feature ranking
# biodem.rank_feat(
#     path_model_i=,
#     paths_omics_i=,
#     path_pheno_i=,
#     trait_name=,
#     n_label_class=,
#     result_dir=,
#     random_seeds=,
#     batch_size=,
# )


#-----------------------------------------------------------
# Test SNP2Gene on nested cross-validation

# Define the data paths
dir_s2g = os.path.join(dir_work, "snp2gene")
dir_save_converted = os.path.join(dir_s2g, "converted")

# Decompress the example data zma.7z
os.system("7z x -o" + dir_s2g + " ../data/test_s2g/zma.7z")

dir_pheno_inner = os.path.join(dir_s2g, "zma", "labels", "inner_zscore")
dir_pheno_outer = os.path.join(dir_s2g, "zma", "labels", "outer_zscore")
path_snp_h5 = os.path.join(dir_s2g, "zma", "processed_data_snp.h5")
path_g2s_json = os.path.join(dir_s2g, "zma", "processed_data_gene2snp.json")

# Transform SNP to gene feautres
# This is a simple usage example

s2g_pipe = biodem.SNP2GenePipeline(
    trait_name="KWPE",
    n_label_class=1,
    dir_label_inner=dir_pheno_inner,
    path_h5_processed=path_snp_h5,
    path_json_genes_snps=path_g2s_json,
    log_folder=dir_s2g,
    dir_to_save_converted=dir_save_converted,
    dir_label_outer=dir_pheno_outer,
    dense_layers_hidden_dims=[512, 64],
    n_trials=2,
    outer_end_idx=1,
    inner_end_idx=1,
)
s2g_pipe.train_pipeline()
s2g_pipe.convert_snp(batch_size=16)

#-----------------------------------------------------------


# Done

print("All tests passed.")
