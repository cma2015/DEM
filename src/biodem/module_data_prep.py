from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



def filter_na_pheno(path_pheno_tsv:str, min_nonna_ratio:float):
    """
    Filter out samples with no phenotype data and traits with NA ratio excceding a certain threshold.

    Args:
        `path_pheno_tsv` (str): path to the pheno file
        `min_nonna_ratio` (float): minimum non-NA ratio for a trait to be included in the filtered pheno file
    
    Example:
        ```python
        filter_na_pheno('path/to/pheno_file/GSTP014.pheno', 0.85)
        ```
    """
    # read pheno file, the first column is sample ID, the first row is trait name
    pheno_df = pd.read_csv(path_pheno_tsv, sep='\t', header=0, index_col=0)

    # # filter samples with NA ratio excceding a certain threshold
    # pheno_df = pheno_df.dropna(axis=0, thresh=int(len(pheno_df.columns)*min_nonna_ratio))
    # print(pheno_df.shape)

    # filter out samples with no phenotype data
    pheno_df = pheno_df.dropna(axis=0, how='all')

    # filter out traits with NA ratio excceding a certain threshold
    pheno_df = pheno_df.dropna(axis=1, thresh=int(len(pheno_df)*min_nonna_ratio))

    # filter out samples with no phenotype data
    pheno_df = pheno_df.dropna(axis=0, how='all')

    # save filtered pheno file with sample ID as index and trait name as column
    str2replace = '_filtered_nonna_' + str(min_nonna_ratio) + '_.pheno'
    
    pheno_df.to_csv(path_pheno_tsv.replace('.pheno', str2replace), sep='\t', header=True, index=True)


def na_imputed_scaler(
        omics_data: pd.DataFrame,
        na_ratio: float = 0.15,
        is_minmax: bool = True,
        is_zscore: bool = False,
    ) -> pd.DataFrame:
    '''
    input: omics_data is csv format
    output: omics_data_scaled is csv format
    '''

    # Delete omics features with missing values exceeding 25%
    omics_data_filtered = omics_data.loc[:, omics_data.isna().mean() < na_ratio]

    # Check if df is empty
    if omics_data_filtered.empty:
        # Throw an error
        raise Exception('All omics features have missing values.')

    # Missing values are imputed with mean values.
    imputer = SimpleImputer(strategy='mean')
    omics_data_imputed = pd.DataFrame(imputer.fit_transform(omics_data_filtered), columns=omics_data_filtered.columns)

    omics_o = omics_data_imputed.copy()

    # [0, 1]
    if is_minmax:
        scaler = MinMaxScaler()
        omics_o = pd.DataFrame(scaler.fit_transform(omics_o), columns=omics_o.columns)
    
    # Z-score
    if is_zscore and not is_minmax:
        scaler_z = StandardScaler()
        omics_o = pd.DataFrame(scaler_z.fit_transform(omics_o), columns=omics_o.columns)
    
    return omics_o


def impute_omics(
        path_omics_i: str | None = None,
        path_omics_o: str | None = None,
        path_pheno_i: str | None = None,
        path_pheno_o: str | None = None,
        max_prop_na: float = 0.25,
        is_minmaxscale_omics: bool = True,
        is_zscore_pheno: bool = True,
    ) -> (tuple[pd.DataFrame, pd.DataFrame] | None):
    """
    Impute missing values in omics data and scale the data.
    """
    if path_omics_i is not None and path_omics_o is not None:
        omics_i = pd.read_csv(path_omics_i, index_col=0)
        treated_omics = na_imputed_scaler(omics_i, max_prop_na, is_minmaxscale_omics)
        treated_omics.to_csv(path_omics_o)
        
    if path_pheno_i is not None and path_pheno_o is not None:
        pheno_i = pd.read_csv(path_pheno_i, index_col=0)
        treated_pheno = na_imputed_scaler(pheno_i, 0.05, False, is_zscore_pheno)
        treated_pheno.to_csv(path_pheno_o)


def variance_pca(
        X: pd.DataFrame,
        y: pd.DataFrame,
        variance_threshold: float = 0.0,
        target_variance_ratio : float = 0.5,
    ) -> pd.DataFrame:
    """
    Perform feature selection using variance threshold with PCA.
    """
    # Set a variance threshold for feature selection
    var_selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = var_selector.fit_transform(X)

    # Calculate the ANOVA F value for each feature
    f_values, _ = f_classif(X_selected, y)

    sorted_indices = np.argsort(f_values)

    pca = PCA()
    remove_index = []

    for i in range(X_selected.shape[1]):
        # Remove features with small variances
        print("Fvalue is :{}".format(i))
        removed_feature_index = sorted_indices[i]
        remove_index.append(removed_feature_index)
        X_selected1 = np.delete(X_selected, remove_index, axis=1)

        pca.fit(X_selected1)
        n_components_list = [pca.explained_variance_ratio_]
        first_component_explained = n_components_list[0][0]

        if float(first_component_explained) < target_variance_ratio:
            class_input = X_selected1
            break
        else:
            X_selected1 = deepcopy(X_selected)

    return pd.DataFrame(class_input)


def select_varpca(
        path_omics_i: str,
        path_pheno_i: str,
        path_omics_o: str,
        min_var: float = 0.0,
        target_var_pc1: float = 0.5,
    ) -> (pd.DataFrame | None):
    omics_i = pd.read_csv(path_omics_i, index_col=0)
    pheno_i = pd.read_csv(path_pheno_i, index_col=0)
    treated_omics = variance_pca(omics_i, pheno_i, min_var, target_var_pc1)
    treated_omics.to_csv(path_omics_o)


def rf_feat_importance(
        x: pd.DataFrame,
        y: pd.DataFrame,
        n_trees: int,
        rand_state: int,
        n_th: int = -1,
    ):
    x_np = x.to_numpy()
    # Single-trait
    y_np = y.to_numpy().reshape(-1,)

    regressor = RandomForestRegressor(n_estimators=n_trees, random_state=rand_state, n_jobs=n_th)
    print("Training RF......", "\nRandom state: ", rand_state)
    regressor.fit(x_np, y_np)
    print("Training RF done......")

    return regressor.feature_importances_


def select_rf_df(
        x: pd.DataFrame,
        y: pd.DataFrame,
        n_feat_save: int,
        n_trees: int,
        rand_states: list[int],
        n_th: int = -1,
    ) -> pd.DataFrame:

    list_feat_importance = []
    for seed_rf in rand_states:
        list_feat_importance.append(rf_feat_importance(x, y, n_trees, seed_rf, n_th))
    
    # Calculate mean importance scores among these forests
    feat_importance = np.mean(list_feat_importance, axis=0)
    idx = np.argsort(feat_importance)[::-1]
    
    # Save the first n_save important features
    x_selected = x.iloc[:, idx[:n_feat_save]]

    return x_selected

def select_rf(
        path_omics_i: str,
        path_pheno_i: str,
        path_omics_o: str,
        n_feat_save: int,
        n_trees: int,
        rand_seeds: list[int],
        n_th: int = -1,
    ):
    omics_i = pd.read_csv(path_omics_i, index_col=0)
    pheno_i = pd.read_csv(path_pheno_i, index_col=0)
    x_selected = select_rf_df(omics_i, pheno_i, n_feat_save, n_trees, rand_seeds, n_th)
    x_selected.to_csv(path_omics_o)
