# -*- coding: utf8 -*-
from copy import deepcopy
from math import sqrt as math_sqrt
# import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


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


def variance_pca(X: pd.DataFrame, y: pd.DataFrame,
                 variance_threshold: float = 0.0,
                 target_variance_ratio : float = 0.5,
                 ) -> pd.DataFrame:
    '''
    input: X, y is csv format
    X_selected1 is numpy.array format
    output: pd.Dataframe
    '''

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


def rf_feat_importance(x: pd.DataFrame, y: pd.DataFrame,
                       n_trees: int, rand_state: int, n_th: int = -1):
    x_np = x.to_numpy()
    y_np = y.to_numpy().reshape(-1,)

    regressor = RandomForestRegressor(n_estimators=n_trees, random_state=rand_state, n_jobs=n_th)
    print("training RF......", "random state: ", rand_state)
    regressor.fit(x_np, y_np)
    print("training RF done......")

    return regressor.feature_importances_
