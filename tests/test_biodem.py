# The test cases for biodem.
import biodem
import os
import sys


if __name__ == "__main__":
    try:
        dir_work = sys.argv[1]
        continue_test = True
    except:
        continue_test = False
    
    if not continue_test:
        time_start = biodem.utils.uni.time_string()
        dir_work = os.path.join(os.getcwd(), 'tests_' + time_start)
        # Make a directory to store testing results
        os.makedirs(dir_work, exist_ok=False)

    trait_name = "KWPE"

    # Define the data paths
    path_tr_pheno = os.path.join(dir_work, trait_name, "outer_zscore_labels_tr.csv")
    path_te_pheno = os.path.join(dir_work, trait_name, "outer_zscore_labels_te.csv")
    path_tr_snp = os.path.join(dir_work, trait_name, "snp_outer_tr.csv")
    path_te_snp = os.path.join(dir_work, trait_name, "snp_outer_te.csv")
    path_tr_exp = os.path.join(dir_work, trait_name, "exp_outer_tr.csv")
    path_te_exp = os.path.join(dir_work, trait_name, "exp_outer_te.csv")

    dir_ncv_litdata = os.path.join(dir_work, "litdata_ncv")
    os.makedirs(dir_ncv_litdata, exist_ok=True)

    dir_dem_fit_log = os.path.join(dir_work, "fit_logs")
    os.makedirs(dir_dem_fit_log, exist_ok=True)

    dir_dem_pred = os.path.join(dir_work, "predictions")

    path_dem_rank = os.path.join(dir_work, "feature_rank")

    which_outer_inner = [0, 0]

    # Unzip the test data into the working directory
    if not continue_test:
        os.system(f"unzip ../data/test_dem/{trait_name}.zip -d " + dir_work)
        print("Testing Data have been released to " + dir_work)
        print(os.listdir(os.path.join(dir_work, trait_name)))


    # Data preprocessing with nested cross-validation and litdata optimization.
    if not continue_test:
        dem_opt_data = biodem.OptimizeLitdataNCV(
            paths_omics={"genotype": path_tr_snp, "expression": path_tr_exp},
            path_label=path_tr_pheno,
            output_dir=dir_ncv_litdata,
            k_outer=4,
            k_inner=3,
            which_outer_inner=which_outer_inner,
            col2use_in_labels=[trait_name],
            compression=None,
            save_n_feat=100,
            n_estimators=300,
            random_states=[42, 43],
        )
        dem_opt_data.run_optimization()

    # This is a simple usage example for DEM model training with hyperparameter optimization.    
    if not continue_test or len(os.listdir(dir_dem_fit_log)) == 0:
        dem_fit = biodem.DEMFitPipe(
            litdata_dir=dir_ncv_litdata,
            list_ncv=[which_outer_inner],
            log_dir=dir_dem_fit_log,
            regression=True,
            n_trials=2,
        )
        dem_fit.train_pipeline()

    # Predict phenotypes
    dem_predict = biodem.DEMPredict()
    dem_predict.collect_models(
        dir_fit_logs=dir_dem_fit_log,
        dir_output=dir_dem_pred,
        overwrite_collected_log=True,
    )
    dem_predict.run_xo_xi(
        x_outer=which_outer_inner[0],
        x_inner=which_outer_inner[1],
        litdata_dir=dir_ncv_litdata,
        dir_output=dir_dem_pred,
        batch_size=biodem.constants.default.batch_size,
    )

    # Functional gene mining (feature ranking)
    dem_feat_rank = biodem.DEMFeatureRanking()
    dem_feat_rank.run_a_outer(
        ncv_litdata_dir=dir_ncv_litdata,
        fit_log_dir=dir_dem_fit_log,
        which_outer=which_outer_inner[0],
        output_path=os.path.join(path_dem_rank, f"rank_features_{trait_name}_outer+{which_outer_inner[0]}.csv"),
        random_states=[55, 66, 77],
    )
    dem_feat_rank.collect_ranks(path_dem_rank, os.path.join(path_dem_rank, "rank_merged_sorted.csv"))

    # Done
    print("\n------ Done ------\n")
