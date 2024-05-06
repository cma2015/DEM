import argparse
from .utils import rm_ckpt
from .module_data_prep import filter_na_pheno, impute_omics, select_varpca, select_rf
from .module_data_prep_ncv import data_prep_ncv_regre, data_prep_ncv_class
from .module_snp import SNP2GenePipeline
from .module_dem import DEMTrainPipeline, predict_pheno, rank_feat


def hello():
    parser = argparse.ArgumentParser(description="Hello, DEM!")
    parser.add_argument("-n", "--name", type=str, default="MaLab@NWAFU", help="Any string", required=False)
    args = parser.parse_args()
    return f"Hello from {args.name}."

def cli_rm_ckpt():
    parser = argparse.ArgumentParser(description="Remove ckpt files")
    parser.add_argument("-p", "--path", type=str, help="Path to ckpt files", required=True)
    parser.add_argument("-a", "--all", action="store_true", default=False, help="Remove all ckpt files")
    args = parser.parse_args()
    rm_ckpt(args.path, args.all)

def cli_filter_pheno():
    parser = argparse.ArgumentParser(description="Filter pheno file. Remove samples with no phenotype data and traits with NA ratio excceding a certain threshold.")
    parser.add_argument("-p", "--pheno", type=str, help="Path to pheno file (tsv format)", required=True)
    parser.add_argument("-m", "--max-na", type=float, help="Max proportion of missing values allowed", required=True)
    args = parser.parse_args()
    filter_na_pheno(args.pheno, args.max_na)


def cli_ncv_prep_regre():
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for regression task, on nested cross-validation.")
    parser.add_argument("-K", "--loop-outer", type=int, default=5, help="Number of outer loops for nested cross-validation.")
    parser.add_argument("-k", "--loop-inner", type=int, default=5, help="Number of inner loops for nested cross-validation.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input omics/phenotypes data.")
    parser.add_argument("-o", "--dir-out", type=str, help="Path to the output directory.")
    parser.add_argument("-t", "--threshold-var", type=float, default=0.01, help="Threshold for variance selection.")
    parser.add_argument("-n", "--n-rf-selected", type=int, default=1000, help="Number of selected features by random forest.")
    parser.add_argument("-x", "--trait", type=str, help="Name of the phenotype column in the input data.")
    parser.add_argument("--raw-label", type=str, help="Path to the raw labels data.")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of trees in random forest.")
    parser.add_argument("--na", type=float, default=0.25, help="Threshold for missing value.")
    args = parser.parse_args()
    data_prep_ncv_regre(
        args.loop_outer,
        args.loop_inner,
        args.input,
        args.dir_out,
        args.raw_label,
        args.trait,
        args.threshold_var,
        args.n_rf_selected,
        args.n_trees,
        args.na,
    )


def cli_ncv_prep_class():
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for classification task, on nested cross-validation.")
    parser.add_argument("-K", "--loop-outer", type=int, default=5, help="Number of outer loops for nested cross-validation.")
    parser.add_argument("-k", "--loop-inner", type=int, default=5, help="Number of inner loops for nested cross-validation.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input omics/phenotypes data.")
    parser.add_argument("-o", "--dir-out", type=str, help="Path to the output directory.")
    parser.add_argument("-t", "--threshold-var", type=float, default=0.01, help="Threshold for variance selection.")
    parser.add_argument("-r", "--target-var-ratio", type=float, default=0.9, help="Target variance ratio for PCA.")
    parser.add_argument("-x", "--trait", type=str, help="Name of the phenotype column in the input data.")
    parser.add_argument("--raw-label", type=str, help="Path to the raw labels data.")
    parser.add_argument("--na", type=float, default=0.25, help="Threshold for missing value.")
    args = parser.parse_args()
    data_prep_ncv_class(
        args.loop_outer,
        args.loop_inner,
        args.input,
        args.dir_out,
        args.raw_label,
        args.trait,
        args.threshold_var,
        args.target_var_ratio,
        args.na,
    )


def cli_dem_impute():
    parser = argparse.ArgumentParser(
        description="Remove features which has too many missing values, impute and min-max scale. Apply z-score to phenotypes."
        )
    parser.add_argument("-I", "--inom", type=str, help="(OPTIONAL) Input a path to an omics file")
    parser.add_argument("-O", "--outom", type=str, help="(OPTIONAL) Define your output omics file path")
    parser.add_argument("-i", "--inph", type=str, help="(OPTIONAL) Input a path to a trait's phenotypes")
    parser.add_argument("-o", "--outph", type=str, help="(OPTIONAL) Define your output phenotypes path")
    parser.add_argument("-p", "--propna", type=float, default=0.25, help="(OPTIONAL) The allowed max proportion of missing values in a feature (DEFAULT: 0.25)")
    parser.add_argument("-m", "--minmax", type=int, choices=[0,1], default=1, help="(OPTIONAL) Whether min-max scaling for omics is required (0 denotes False, 1 denotes True)")
    parser.add_argument("-z", "--zscore", type=int, choices=[0,1], default=1, help="(OPTIONAL) Whether z-score for phenotypes is required (0 denotes False, 1 denotes True)")
    args = parser.parse_args()
    is_minmax = bool(args.minmax)
    is_zscore = bool(args.zscore)
    impute_omics(args.inom, args.outom, args.inph, args.outph, args.propna, is_minmax, is_zscore)


def cli_dem_select_varpca():
    parser = argparse.ArgumentParser(description="Apply a variance threshold and PCA for feature selection.")
    parser.add_argument("-I", "--inom", type=str, required=True, help="*Input a path to an omics file")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-O", "--outom", type=str, required=True, help="*Define your output omics file path")
    parser.add_argument("-V", "--minvar", type=float, default=0.0, help="(OPTIONAL) The allowed minimum variance of a feature (DEFAULT: 0.0)")
    parser.add_argument("-P", "--varpc", type=float, default=0.5, help="(OPTIONAL) Target variance of PC1 (DEFAULT: 0.5)")
    args = parser.parse_args()
    select_varpca(args.inom, args.inph, args.outom, args.minvar, args.varpc)


def cli_dem_select_rf():
    parser = argparse.ArgumentParser(description="RF is employed to screen out representative omics features.")
    parser.add_argument("-I", "--inom", type=str, required=True, help="*Input a path to an omics file")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-O", "--outom", type=str, required=True, help="*Output omics path")
    parser.add_argument("-n", "--nfeat", type=int, required=True, help="*Number of features to save")
    parser.add_argument("-N", "--ntree", type=int, default=2500, help="(OPTIONAL) Number of trees in RF (DEFAULT: 2500)")
    parser.add_argument("-S", "--seedrf", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seeds for RF (DEFAULT: 1000, 1001, ..., 1009)")
    args = parser.parse_args()
    if args.seedrf is None:
        seeds_rf = [i+1000 for i in range(10)]
    else:
        seeds_rf = args.seedrf
    select_rf(args.inom, args.inph, args.outom, args.nfeat, args.ntree, seeds_rf)


def cli_s2g_pipe():
    parser = argparse.ArgumentParser(description="SNP2Gene Pipeline")
    parser.add_argument("-t", "--trait", type=str, help="Trait name", required=True)
    parser.add_argument("-l", "--lbl-class", type=int, help="Number of classes for trait", required=True)
    parser.add_argument("--inner-lbl", type=str, help="Directory to inner label files", required=True)
    parser.add_argument("--outer-lbl", type=str, help="Directory to outer label files", required=False, default=None)
    parser.add_argument("--h5", type=str, help="Path to h5 file containing genotype data", required=True)
    parser.add_argument("--json", type=str, help="Path to json file containing SNP-gene mapping information", required=True)
    parser.add_argument("--log-dir", type=str, help="Directory to save log files and models", required=True)
    parser.add_argument("--o-s2g-dir", type=str, help="Directory to save SNP2Gene results", required=True)
    args = parser.parse_args()
    s2g_pipeline = SNP2GenePipeline(
        trait_name=args.trait,
        n_label_class=args.lbl_class,
        dir_label_inner=args.inner_lbl,
        dir_label_outer=args.outer_lbl,
        path_h5_processed=args.h5,
        path_json_genes_snps=args.json,
        log_folder=args.log_dir,
        dir_to_save_converted=args.o_s2g_dir,
    )
    s2g_pipeline.train_pipeline()
    s2g_pipeline.convert_snp()


def cli_dem_train_pipe():
    parser = argparse.ArgumentParser(description="DEM Training Pipeline")
    parser.add_argument("-o", "--log-dir", type=str, help="Directory to save log files and models", required=True)
    parser.add_argument("-t", "--trait", type=str, help="Trait name", required=True)
    parser.add_argument("-l", "--lbl-class", type=int, help="Number of classes for trait", required=True)
    parser.add_argument("--inner-lbl", type=str, help="Directory to inner label files", required=True)
    parser.add_argument("--outer-lbl", type=str, help="Directory to outer label files", required=True)
    parser.add_argument("--inner-om", type=str, help="Directory to inner omics files", required=True)
    parser.add_argument("--outer-om", type=str, help="Directory to outer omics files", required=True)
    args = parser.parse_args()
    dem_pipeline = DEMTrainPipeline(
        log_folder=args.log_dir,
        trait_name=args.trait,
        n_label_class=args.lbl_class,
        dir_label_inner=args.inner_lbl,
        dir_label_outer=args.outer_lbl,
        dir_omics_inner=args.inner_om,
        dir_omics_outer=args.outer_om,
    )
    dem_pipeline.train_pipeline()


def cli_dem_predict():
    parser = argparse.ArgumentParser(description="Predict phenotypes from the given omics data files using a trained DEM model.")
    parser.add_argument("-I", "--inom", action="extend", nargs="+", type=str, required=True, help="*Input path(s) to omics file(s)")
    parser.add_argument("-m", "--inmd", type=str, required=True, help="*The path to a pretrained DEM model")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="*The directory to save predicted phenotypes")
    args = parser.parse_args()
    predict_pheno(args.inmd, args.inom, args.outdir)


def cli_dem_rank():
    parser = argparse.ArgumentParser(description="Rank features based on their importance in predicting the given trait.")
    parser.add_argument("-I", "--inom", action="extend", nargs="+", type=str, required=True, help="*Input path(s) to omics file(s)")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-t", "--trait", type=str, required=True, help="*The name of the trait to be predicted")
    parser.add_argument("-l", "--lbl-class", type=int, required=True, help="*Number of classes for the trait (1 for regression)")
    parser.add_argument("-m", "--inmd", type=str, required=True, help="*The path to a pretrained DEM model")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="*The directory to save ranked features")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="(OPTIONAL) Batch size for feature ranking (default: 16)")
    parser.add_argument("-s", "--seeds", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seeds for ranking repeats (default: 0-9)")
    args = parser.parse_args()
    if args.seeds is None:
        seeds_rk = [i for i in range(10)]
    else:
        seeds_rk = args.seedrk
    rank_feat(args.inmd, args.inom, args.inph, args.trait, args.lbl_class, args.outdir, seeds_rk, args.batch_size)
