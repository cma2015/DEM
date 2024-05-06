import os
import numpy as np
import pandas as pd
from typing import List, Union
from .module_data_prep import select_rf_df, variance_pca


class KFoldSplitter:
    """
    Split the input data into nested k-folds.

    Example usage:

    - Split the input data into 5-folds and save the test and train sets in separate files:
    
    ```python
    splitter = KFoldSplitter(n_fold=5, input_path='path/to/input.csv', output_dir='path/to/output_dir')
    splitter.split()
    ```

    - Nested 10-fold cross-validation:
    
    ```python
    KFoldSplitter.nested_split(outer_loop=10, inner_loop=5, input_path='path/to/input.csv', output_dir='path/to/output_dir')
    ```

    """
    def __init__(self, n_fold: int, input_path: str, output_dir: str):
        self.n_fold = n_fold
        self.input_path = input_path
        self.output_dir = output_dir
        self.all_samples = pd.read_csv(input_path, index_col=0)
        self.all_sample_ids = list(self.all_samples.index)

    def split(self) -> None:
        fold_num = len(self.all_sample_ids) // self.n_fold
        for i in range(self.n_fold):
            test_start = i * fold_num
            test_end = (i + 1) * fold_num if i < self.n_fold - 1 else None
            test_indices = self.all_sample_ids[test_start:test_end]
            train_indices = [idx for idx in self.all_sample_ids if idx not in test_indices]
            
            test = self.all_samples.loc[test_indices]
            train = self.all_samples.loc[train_indices]
            
            test.to_csv(os.path.join(self.output_dir, f"{i}_test.csv"))
            train.to_csv(os.path.join(self.output_dir, f"{i}_train.csv"))

    @staticmethod
    def nested_split(outer_loop: int, inner_loop: int, input_path: str, output_dir: str) -> None:
        os.makedirs(os.path.join(output_dir, "outer_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "inner_data"), exist_ok=True)

        outer_splitter = KFoldSplitter(outer_loop, input_path, os.path.join(output_dir, "outer_data"))
        outer_splitter.split()

        for n in range(outer_loop):
            outer_train_path = os.path.join(output_dir, "outer_data", f"{n}_train.csv")
            inner_output_path = os.path.join(output_dir, "inner_data", f"{n}fold")
            os.makedirs(inner_output_path, exist_ok=True)
            inner_splitter = KFoldSplitter(inner_loop, outer_train_path, inner_output_path)
            # Assuming the inner split uses the first column as IDs
            inner_splitter.split()


def label_zscore4regression(raw_labels_path: str, train_labels_id_path: str, test_labels_id_path: str):
    """
    Normalize the labels by z-score.
    """
    # Read the data
    raw_labels = pd.read_csv(raw_labels_path, index_col=0)
    train_labels = pd.read_csv(train_labels_id_path, index_col=0)
    test_labels = pd.read_csv(test_labels_id_path, index_col=0)

    # Extract the labels based on the ID presence in train and test sets
    train_raw_labels = raw_labels.loc[train_labels.index].copy()
    test_raw_labels = raw_labels.loc[test_labels.index].copy()

    # Calculate z-scores based on the training set only, for each TRAIT (column)
    train_mean = train_raw_labels.mean()
    train_std = train_raw_labels.std()
    train_zscore_labels = (train_raw_labels - train_mean) / train_std
    test_zscore_labels = (test_raw_labels - train_mean) / train_std

    return train_zscore_labels, test_zscore_labels

def label_split4class(raw_labels_path: str, train_labels_id_path: str, test_labels_id_path: str):
    """
    Split the raw labels into two parts: train_labels and test_labels.
    """
    # Read the data
    raw_labels = pd.read_csv(raw_labels_path, index_col=0)
    train_labels = pd.read_csv(train_labels_id_path, index_col=0)
    test_labels = pd.read_csv(test_labels_id_path, index_col=0)

    # Extract the labels based on the ID presence in train and test sets
    train_raw_labels = raw_labels.loc[train_labels.index].copy()
    test_raw_labels = raw_labels.loc[test_labels.index].copy()

    return train_raw_labels, test_raw_labels


def na_del_na_imp(df2process: pd.DataFrame, na_rate: float = 0.25):
    """
    Delete the columns with missing values more than na_rate, and impute the missing values with mean.
    """
    missing_percentage = df2process.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > na_rate].index
    data = df2process.drop(columns=columns_to_drop)
    return data


def max_min_scale(
        omics_data: pd.DataFrame,
        train_labels_id_path: str,
        test_labels_id_path: str,
    ):
    """
    Normalize the omics data by max-min scale.
    """
    # Read the omics data and labels
    train_labels = pd.read_csv(train_labels_id_path, index_col=0)
    test_labels = pd.read_csv(test_labels_id_path, index_col=0)

    # Extract the omics data based on the ID presence in train and test sets
    train_omics = omics_data.loc[train_labels.index].copy()
    test_omics = omics_data.loc[test_labels.index].copy()

    # Calculate the max and min values from the training data
    train_max = train_omics.max()
    train_min = train_omics.min()

    train_omics0_1 = (train_omics - train_min) / (train_max - train_min)
    test_omics0_1 = (test_omics - train_min) / (train_max - train_min)

    # Clip the test data to ensure it's within the range [0, 1]
    test_omics0_1 = test_omics0_1.clip(lower=0, upper=1)

    return train_omics0_1, test_omics0_1


def feature_select_var(pd_dataframe: pd.DataFrame, Threshold_Variance: float):
    """
    Delete the columns with small variance.
    """
    var = pd_dataframe.var()
    feature_drop = []
    for i in range(len(var)):
        if var[i] < Threshold_Variance:
            feature_drop.append(var.index[i])
    return pd_dataframe.drop(feature_drop, axis=1)


def use_trfeats_filter_var_tefeats(omics_df: pd.DataFrame, test_omics0_1: pd.DataFrame):
    train_gene_list = list(omics_df.columns)
    filtered_te = test_omics0_1[train_gene_list]

    return filtered_te

def omics_regression_data_preproc(
        omics_dataframe: pd.DataFrame,
        train_labels_id_path: str,
        test_labels_id_path: str,
        which_trait: Union[str, int],
        Threshold_Variance: float,
        RF_selected_num: int,
        raw_labels_path: str,
        number_of_trees: int,
        na_rate: float,
        rf_rand_states: list[int] = [0, 1, 2, 3, 4],
    ):
    """
    Preprocess the omics data for regression.
    """
    train_zscore_labels, test_zscore_labels = label_zscore4regression(raw_labels_path, train_labels_id_path,
                                                           test_labels_id_path)

    train_zscore_labels = train_zscore_labels[[which_trait]]
    test_zscore_labels = test_zscore_labels[[which_trait]]

    train_omics0_1, test_omics0_1 = max_min_scale(omics_dataframe, train_labels_id_path, test_labels_id_path)
    train_omics0_1 = na_del_na_imp(train_omics0_1, na_rate)

    train_omics0_1 = feature_select_var(train_omics0_1, Threshold_Variance)
    train_omics0_1 = select_rf_df(train_omics0_1, train_zscore_labels, RF_selected_num, number_of_trees, rf_rand_states)

    test_omics0_1 = use_trfeats_filter_var_tefeats(train_omics0_1, test_omics0_1)

    return train_omics0_1, test_omics0_1, train_zscore_labels, test_zscore_labels

def omics_classification_data_preproc(
        omics_dataframe: pd.DataFrame,
        train_labels_id_path: str,
        test_labels_id_path: str,
        which_trait: Union[str, int],
        Threshold_Variance: float,
        raw_labels_path: str,
        target_variance_ratio: float,
        na_rate: float,
    ):
    """
    Preprocess the omics data for classification.
    """
    train_labels, test_labels = label_split4class(raw_labels_path, train_labels_id_path, test_labels_id_path)

    train_labels = train_labels[[which_trait]]
    test_labels = test_labels[[which_trait]]

    train_omics0_1, test_omics0_1 = max_min_scale(omics_dataframe, train_labels_id_path, test_labels_id_path)
    train_omics0_1 = na_del_na_imp(train_omics0_1, na_rate)

    train_omics0_1 = variance_pca(train_omics0_1, train_labels, Threshold_Variance, target_variance_ratio)

    test_omics0_1 = use_trfeats_filter_var_tefeats(train_omics0_1, test_omics0_1)

    return train_omics0_1, test_omics0_1, train_labels, test_labels



def create_output_directories(output_dir, outer, inner=None):
    """
    Create output directories for the given outer and inner loop indices.
    """
    dir_ncv_outer = os.path.join(output_dir, "nested_CV_data", "outer")
    dir_ncv_outer_zscore_labels = os.path.join(output_dir, "nested_CV_data", "outer_zscore_labels")
    os.makedirs(dir_ncv_outer, exist_ok=True)
    os.makedirs(dir_ncv_outer_zscore_labels, exist_ok=True)

    if inner is not None:
        dir_out = os.path.join(output_dir, "nested_CV_data", "inner", "{}fold".format(outer))
        dir_out_zscore_labels = os.path.join(output_dir, "nested_CV_data", "inner_zscore_labels", "{}fold".format(outer))
        os.makedirs(dir_out, exist_ok=True)
        os.makedirs(dir_out_zscore_labels, exist_ok=True)
        return dir_out, dir_out_zscore_labels
    return dir_ncv_outer, dir_ncv_outer_zscore_labels

def process_omics_labels_data(
        outer_loop: int,
        inner_loop: int,
        output_dir: str,
        omics: pd.DataFrame,
        data_preproc_function,
        **kwargs,
    ):
    """
    Process omics and labels data for either regression or classification.
    """
    for out in range(outer_loop):
        train_labels_id_path = os.path.join(output_dir, "outer_data", "{}_train.csv".format(out))
        test_labels_id_path = os.path.join(output_dir, "outer_data", "{}_test.csv".format(out))
        
        dir_ncv_outer, dir_ncv_outer_zscore_labels = create_output_directories(output_dir, out)

        train_omics0_1, test_omics0_1, train_zscore_labels, test_zscore_labels = data_preproc_function(
            omics, train_labels_id_path, test_labels_id_path, **kwargs)

        train_omics0_1.to_csv(os.path.join(dir_ncv_outer, "{}_outer_loop_tr.csv".format(out)))
        test_omics0_1.to_csv(os.path.join(dir_ncv_outer, "{}_outer_loop_te.csv".format(out)))
        train_zscore_labels.to_csv(os.path.join(dir_ncv_outer_zscore_labels, "{}_outer_zscore_labels_tr.csv".format(out)))
        test_zscore_labels.to_csv(os.path.join(dir_ncv_outer_zscore_labels, "{}_outer_zscore_labels_te.csv".format(out)))

        for inner in range(inner_loop):
            train_labels_id_path = os.path.join(output_dir, "inner_data", "{}fold".format(out), "{}_train.csv".format(inner))
            test_labels_id_path = os.path.join(output_dir, "inner_data", "{}fold".format(out), "{}_test.csv".format(inner))
            
            dir_out, dir_out_zscore_labels = create_output_directories(output_dir, out, inner)

            train_omics0_1, test_omics0_1, train_zscore_labels, test_zscore_labels = data_preproc_function(
                omics, train_labels_id_path, test_labels_id_path, **kwargs)

            path_tr = os.path.join(dir_out, "{}_inner_loop_tr.csv".format(inner))
            path_te = os.path.join(dir_out, "{}_inner_loop_te.csv".format(inner))
            train_omics0_1.to_csv(path_tr)
            test_omics0_1.to_csv(path_te)

            path_tr = os.path.join(dir_out_zscore_labels, "{}_inner_zscore_labels_tr.csv".format(inner))
            path_te = os.path.join(dir_out_zscore_labels, "{}_inner_zscore_labels_te.csv".format(inner))
            train_zscore_labels.to_csv(path_tr)
            test_zscore_labels.to_csv(path_te)


def data_prep_ncv_regre(
        outer_loop: int,
        inner_loop: int,
        input_path: str,
        output_dir: str,
        raw_labels_path: str,
        which_trait: Union[str, int],
        Threshold_Variance: float,
        RF_selected_num: int,
        number_of_trees: int,
        na_rate: float,
        rf_rand_states: list[int] = [0, 1, 2, 3, 4],
    ):
    """
    Data preprocessing pipeline for regression task, on nested cross-validation.
    """
    KFoldSplitter.nested_split(outer_loop, inner_loop, input_path, output_dir)
    omics = pd.read_csv(input_path, index_col=0)

    process_omics_labels_data(
        outer_loop,
        inner_loop,
        output_dir,
        omics,
        omics_regression_data_preproc,
        which_trait=which_trait,
        Threshold_Variance=Threshold_Variance,
        RF_selected_num=RF_selected_num,
        raw_labels_path=raw_labels_path,
        number_of_trees=number_of_trees,
        na_rate=na_rate,
        rf_rand_states=rf_rand_states,
    )


def data_prep_ncv_class(
        outer_loop: int,
        inner_loop: int,
        input_path: str,
        output_dir: str,
        raw_labels_path: str,
        which_trait: Union[str, int],
        Threshold_Variance: float,
        target_variance_ratio: float,
        na_rate: float,
    ):
    """
    Data preprocessing pipeline for classification task, on nested cross-validation.
    """
    KFoldSplitter.nested_split(outer_loop, inner_loop, input_path, output_dir)
    omics = pd.read_csv(input_path, index_col=0)
    process_omics_labels_data(
        outer_loop,
        inner_loop,
        output_dir,
        omics,
        omics_classification_data_preproc,
        which_trait=which_trait,
        Threshold_Variance=Threshold_Variance,
        raw_labels_path=raw_labels_path,
        target_variance_ratio=target_variance_ratio,
        na_rate=na_rate,
    )
