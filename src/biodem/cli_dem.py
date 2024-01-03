import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from .module1_predata import na_imputed_scaler, variance_pca, rf_feat_importance
from .module2_model import model_dem
from .module3_predict import predict_pheno
from .module4_mining import rank_feat


# dem-hello

def hello():
    parser = argparse.ArgumentParser(description="Hello test")
    parser.add_argument("-n", "--name", type=str, default="MaLab@NWAFU", help="Your name", required=True)
    args = parser.parse_args()

    return f"hello {args.name} ."


# dem-impute

def dem_impute(
        path_omics_i: str | None = None,
        path_omics_o: str | None = None,
        path_pheno_i: str | None = None,
        path_pheno_o: str | None = None,
        max_prop_na: float = 0.25,
        is_minmaxscale_omics: bool = True,
        is_zscore_pheno: bool = True,
) -> (tuple[pd.DataFrame, pd.DataFrame] | None):
    if path_omics_i is not None and path_omics_o is not None:
        omics_i = pd.read_csv(path_omics_i, index_col=0)
        treated_omics = na_imputed_scaler(omics_i, max_prop_na, is_minmaxscale_omics)
        treated_omics.to_csv(path_omics_o)
        
    if path_pheno_i is not None and path_pheno_o is not None:
        pheno_i = pd.read_csv(path_pheno_i, index_col=0)
        treated_pheno = na_imputed_scaler(pheno_i, 0.05, False, is_zscore_pheno)
        treated_pheno.to_csv(path_pheno_o)

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

    dem_impute(args.inom, args.outom, args.inph, args.outph, args.propna, is_minmax, is_zscore)


# dem-select-varpca

def dem_select_varpca(
        path_omics_i: str,
        path_pheno_i: str,
        path_omics_o: str,
        min_var: float = 0.0,
        target_var_pc1: float = 0.5,
        # is_return: bool = False,
) -> (pd.DataFrame | None):    
    omics_i = pd.read_csv(path_omics_i, index_col=0)
    pheno_i = pd.read_csv(path_pheno_i, index_col=0)

    treated_omics = variance_pca(omics_i, pheno_i, min_var, target_var_pc1)
    treated_omics.to_csv(path_omics_o)

    # if is_return:
    #     return treated_omics

def cli_dem_select_varpca():
    parser = argparse.ArgumentParser(description="Apply a variance threshold and PCA for feature selection.")
    parser.add_argument("-I", "--inom", type=str, required=True, help="*Input a path to an omics file")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-O", "--outom", type=str, required=True, help="*Define your output omics file path")
    parser.add_argument("-V", "--minvar", type=float, default=0.0, help="(OPTIONAL) The allowed minimum variance of a feature (DEFAULT: 0.0)")
    parser.add_argument("-P", "--varpc", type=float, default=0.5, help="(OPTIONAL) Target variance of PC1 (DEFAULT: 0.5)")
    args = parser.parse_args()

    dem_select_varpca(args.inom, args.inph, args.outom, args.minvar, args.varpc)


# dem-select-rf

def dem_select_rf(
        path_omics_i: str,
        path_pheno_i: str,
        path_omics_o_prefix: str,
        n_feat_save: int,
        prop_val: float = 0.2,
        n_trees: int = 2500,
        rand_seeds_rf: list[int] = [i+1000 for i in range(10)],
        rand_seeds_split: list[int] = [i for i in range(5)],
):
    omics_i = pd.read_csv(path_omics_i, index_col=0)
    pheno_i = pd.read_csv(path_pheno_i, index_col=0)

    # Split the data into training and testing sets
    for seed_sp in rand_seeds_split:
        print(f"\nSeed for spliting {seed_sp}...")
        x_trn, x_val, y_trn, y_val = train_test_split(omics_i, pheno_i, test_size=prop_val, random_state=seed_sp)

        list_feat_importance = []
        for seed_rf in rand_seeds_rf:
            list_feat_importance.append(rf_feat_importance(x_trn, y_trn, n_trees, seed_rf))
        # Calculate mean importance scores among these forests
        feat_importance = np.mean(list_feat_importance, axis=0)
        idx = np.argsort(feat_importance)[::-1]
        # Save the first n_save important features
        x_selected = omics_i.iloc[:, idx[:n_feat_save]]
        save_path = path_omics_o_prefix.replace(".csv", f"_rf_n{n_feat_save}_sp{seed_sp}.csv")
        x_selected.to_csv(save_path)
        print(save_path)

def cli_dem_select_rf():
    parser = argparse.ArgumentParser(description="RF is employed to screen out representative omics features.")
    parser.add_argument("-I", "--inom", type=str, required=True, help="*Input a path to an omics file")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-O", "--outom", type=str, required=True, help="*Tag/Prefix of output omics path")
    parser.add_argument("-n", "--nfeat", type=int, required=True, help="*Number of features to save")
    parser.add_argument("-p", "--propv", type=float, default=0.2, help="(OPTIONAL) Proportion of validation set (DEFAULT: 0.2)")
    parser.add_argument("-N", "--ntree", type=int, default=2500, help="(OPTIONAL) Number of trees in RF (DEFAULT: 2500)")
    parser.add_argument("-S", "--seedrf", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seeds for RF (DEFAULT: 1000, 1001, ..., 1009)")
    parser.add_argument("-s", "--seedsp", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seeds for splitting (DEFAULT: 0, 1, ..., 4)")
    args = parser.parse_args()

    if args.seedrf is None:
        seeds_rf = [i+1000 for i in range(10)]
    else:
        seeds_rf = args.seedrf
    
    if args.seedsp is None:
        seeds_sp = [i for i in range(5)]
    else:
        seeds_sp = args.seedsp
    
    dem_select_rf(args.inom, args.inph, args.outom, args.nfeat, args.propv, args.ntree, seeds_rf, seeds_sp)


# dem-model

def dem_model(
        paths_omics_i: list[str],
        path_pheno_i: str,
        path_model_o: str,
        regr_or_clas: bool,
        prop_val: float = 0.2,
        rand_seeds_split: list[int] = [i for i in range(5)],
        batch_size: int = 32,
        lr: float = 0.0001,
        es_patience: int = 15,
        n_dem_enc: int = 4,
        dropout: float = 0.1,
        max_epochs: int = 1000,
):
    path_save_model = path_model_o
    if not path_save_model.endswith(".pth"):
        path_save_model = path_model_o + ".pth"
    for seed_sp in rand_seeds_split:
        path_save_model_x = path_save_model.replace(".pth", f"_seedsp{seed_sp}.pth")
        model_dem(paths_omics_i, path_pheno_i, path_save_model_x, regr_or_clas, prop_val, seed_sp, batch_size, lr, dropout, max_epochs, es_patience, n_dem_enc)

def cli_dem_model():
    parser = argparse.ArgumentParser(description="Construct a DEM model based on the provided omics data and phenotypes, employing cross-validation or random sampling.")
    parser.add_argument("-I", "--inom", action="extend", nargs="+", type=str, required=True, help="*Input path(s) to omics file(s)")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-o", "--outmd", type=str, required=True, help="*The path to save your trained DEM model")
    parser.add_argument("-r", "--regrclas", type=int, choices=[0, 1], required=True, help="*Regression or Classification task (1 denotes regression, 0 denotes classification)")
    parser.add_argument("-p", "--propv", type=float, default=0.2, help="(OPTIONAL) Proportion of validation set (DEFAULT: 0.2)")
    parser.add_argument("-s", "--seedsp", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seed(s) for data partition(s) (DEFAULT: 0...4)")
    parser.add_argument("-b", "--batchsize", type=int, default=32, help="(OPTIONAL) Batch size (DEFAULT: 32)")
    parser.add_argument("-l", "--learningrate", type=float, default=0.0001, help="(OPTIONAL) Learning rate (DEFAULT: 0.0001)")
    parser.add_argument("-e", "--patience", type=int, default=10, help="(OPTIONAL) Early stopping patience (DEFAULT: 10)")
    parser.add_argument("-N", "--nenc", type=int, default=4, help="(OPTIONAL) Number of DEM encoders (DEFAULT: 4)")
    parser.add_argument("-D", "--dropout", type=float, default=0.1, help="(OPTIONAL) Dropout rate (DEFAULT: 0.1)")
    args = parser.parse_args()

    if args.seedsp is None:
        seeds_sp = [i for i in range(5)]
    else:
        seeds_sp = args.seedsp
    
    regr_clas = bool(args.regrclas)

    dem_model(args.inom, args.inph, args.outmd, regr_clas, args.propv, seeds_sp, args.batchsize, args.learningrate, args.patience, args.nenc, args.dropout)


# dem-predict

def dem_predict(
        path_model_i: str,
        paths_omics_i: list[str],
        path_pheno_o: str,
):
    predict_pheno(path_pheno_o, path_model_i, paths_omics_i)

def cli_dem_predict():
    parser = argparse.ArgumentParser(description="Predict phenotypes from the given omics data files using a trained DEM model.")
    parser.add_argument("-I", "--inom", action="extend", nargs="+", type=str, required=True, help="*Input path(s) to omics file(s)")
    parser.add_argument("-m", "--inmd", type=str, required=True, help="*The path to your trained DEM model")
    parser.add_argument("-o", "--outph", type=str, required=True, help="*The path to save predicted phenotypes")
    args = parser.parse_args()

    dem_predict(args.inmd, args.inom, args.outph)


# dem-rank

def dem_rank(
        path_model_i: str,
        paths_omics_i: list[str],
        path_pheno_i: str,
        path_rank_o: str,
        regr_or_clas: bool,
        rand_seeds: list[int] = [i for i in range(10)],
):
    rank_feat(path_model_i, path_pheno_i, paths_omics_i, path_rank_o, rand_seeds, regr_or_clas)

def cli_dem_rank():
    parser = argparse.ArgumentParser(description="Rank omics features by their importance scores using a trained DEM model.")
    parser.add_argument("-I", "--inom", action="extend", nargs="+", type=str, required=True, help="*Input path(s) to omics file(s)")
    parser.add_argument("-i", "--inph", type=str, required=True, help="*Input a path to a trait's phenotypes")
    parser.add_argument("-m", "--inmd", type=str, required=True, help="*The path to your trained DEM model")
    parser.add_argument("-o", "--outrank", type=str, required=True, help="*The path to save importance scores and the order")
    parser.add_argument("-r", "--regrclas", type=int, choices=[0, 1], required=True, help="*Regression or Classification task (1 denotes regression, 0 denotes classification)")
    parser.add_argument("-s", "--seedrk", action="extend", nargs="*", type=int, help="(OPTIONAL) Random seeds for ranking repeats (default: 0-9)")
    args = parser.parse_args()

    if args.seedrk is None:
        seeds_rk = [i for i in range(10)]
    else:
        seeds_rk = args.seedrk

    regr_clas = bool(args.regrclas)
    
    dem_rank(args.inmd, args.inom, args.inph, args.outrank, regr_clas, seeds_rk)
