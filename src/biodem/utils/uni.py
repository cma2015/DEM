r"""
Some universal functions.
"""
import os
import time
import random
import string
import numpy as np
import polars as pl
import pickle
import gzip
import optuna
from typing import Any, List, Dict, Optional, Union
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from lightning import Trainer, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from multiprocessing import cpu_count
import biodem.constants as const


def get_avail_cpu_count(target_n: int) -> int:
    total_n = cpu_count()
    n_cpu = target_n
    if target_n <= 0:
        n_cpu = total_n
    else:
        n_cpu = min(target_n, total_n)
    return n_cpu

def get_avail_nvgpu(devices: Union[list[int], str, int] = const.default.devices):
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices
    return avail_dev

def get_map_location(map_loc: Optional[str] = None):
    if map_loc is None:
        if device_count() > 0:
            which_dev = find_usable_cuda_devices(1)
            if len(which_dev) == 0:
                return 'cpu'
            else:
                return f'cuda:{which_dev[0]}'
        else:
            return 'cpu'
    else:
        return map_loc


def time_string() -> str:
    _time_str = time.strftime(const.default.time_format, time.localtime())
    return _time_str

def random_string(length: int = 7) -> str:
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def idx_convert(indices: List[int], onehot_bits: int = const.default.snp_onehot_bits) -> List[int]:
    r"""Convert the indices to the corresponding indices in the one-hot vector.
    """
    converted_indices = np.array(indices)[:, np.newaxis] * onehot_bits + np.arange(onehot_bits)
    converted_indices = np.sort(converted_indices.flatten()).tolist()
    return converted_indices


def intersect_lists(lists: List[List[Any]], get_indices: bool = True, to_sorted: bool = True):
    r"""Find the shared elements between multiple lists.

    Args:
        lists: A list of lists.

        get_indices: Whether to return the indices of the shared elements in each list.

        to_sorted: Whether to sort the shared elements.
    
    """
    if len(lists) == 0:
        raise ValueError("The list of lists is empty.")
    elif len(lists) == 1:
        shared = lists[0]
    else:
        shared = list(set.intersection(*map(set, lists)))
    
    assert len(shared) > 0, "No intersecting elements!"
    if to_sorted:
        shared = sorted(shared)
    if get_indices:
        indices = []
        # for xl in range(len(lists)):
        #     # Accelerate the search by NumPy
        #     indices.append(np.where(np.isin(lists[xl], shared))[0].tolist())
        # !!!!!!!!!!! The following can keep the order of `shared` !!!!!!!!!!!!!!
        for xl in range(len(lists)):
            indices.append([])
        for i in shared:
            for xl in range(len(lists)):
                indices[xl].append(np.where(np.isin(lists[xl], i))[0].tolist()[0])
        return shared, indices
    else:
        return shared

def read_labels(path_label: str, col2use: Optional[List[Any]] = None):
    r"""Read labels from a csv file.
    
    Args:
        path_label: Path to the labels file.

        col2use: A list of column names or indices.
            If `col2use` is `List[int]`, its numbers are the indices **(1-based)** of the columns to be used.
    
    """
    label_df = read_omics(path_label)
    if label_df.columns[0] != const.dkey.id:
        # Check the element data type of the first column
        if label_df.schema.dtypes()[0] == pl.String():
            label_df = label_df.rename({label_df.columns[0]: const.dkey.id})
        else:
            raise ValueError("The first column of the label file must be the sample IDs.")
    
    sample_ids = label_df.select(const.dkey.id).to_series().to_list()

    if col2use is not None:
        if type(col2use[0]) == str:
            label_df = label_df.select([const.dkey.id] + col2use)
        elif type(col2use[0]) == int:
            label_df = label_df[[0]+col2use,:]
        else:
            raise ValueError("col2use must be either a list of strings or a list of integers.")
    dim_model_output = len(label_df.columns) - 1
    return label_df, dim_model_output, sample_ids


def read_omics(data_path: str):
    r"""Read omics data from various formats.
    """
    # Check if the path is a file or a folder
    if os.path.isdir(data_path):
        raise ValueError(f"The path {data_path} is a folder. Please provide a path to a single file.")
    
    # Check the file extension
    file_ext = os.path.splitext(data_path)[-1]
    if len(file_ext) < 2:
        raise ValueError("The file extension is empty.")
    else:
        file_ext = file_ext.lower()
    
    match file_ext:
        case '.csv':
            _data = pl.read_csv(data_path, schema_overrides={const.dkey.id: pl.Utf8})
            if _data.columns[0] == '':
                # Rename the first column to const.dkey.id
                _data = _data.rename({_data.columns[0]: const.dkey.id})
        case '.parquet':
            _data = pl.read_parquet(data_path)
            if _data.columns[0] == '':
                # Rename the first column to const.dkey.id
                _data = _data.rename({_data.columns[0]: const.dkey.id})
        case '.gz':
            if data_path.endswith('.pkl.gz'):
                snp_data_dict = read_pkl_gv(data_path)
                snp_matrix = snp_data_dict[const.dkey.genotype_matrix]
                snp_sample_ids = snp_data_dict[const.dkey.sample_ids]
                snp_ids = snp_data_dict[const.dkey.snp_ids]
                # snp_block_ids = snp_data_dict['block_ids']
                _tmp_snp_df = pl.DataFrame(data=snp_matrix, schema=snp_ids)
                _tmp_id = pl.DataFrame({const.dkey.id: snp_sample_ids})
                _data = _tmp_id.hstack(_tmp_snp_df)
            else:
                raise ValueError("The file extension is not supported.")
        case _:
            raise ValueError("The file extension is not supported.")
    
    return _data


def read_omics_xoxi(
        data_path: str,
        which_outer_test: int,
        which_inner_val: int,
        trnvaltst: str = const.abbr_train,
        file_ext: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
    r"""Read processed data from a directory.

    Args:
        data_path: Path to the directory containing the data.

        which_outer_test: Which outer test set to read.

        which_inner_val: Which inner validation set to read.

        trnvaltst: The abbreviation of the training/validation/test set.

        file_ext: The file extension of the data files. If None, the default extension will be used.

        prefix: The prefix of the file name. (Optional)
    
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"The path {data_path} is not a directory.")
    if file_ext is None:
        fname_ext = const.fname.data_ext
    else:
        fname_ext = file_ext
    
    # Walk through the directory and find all files with the specified pattern
    if prefix is None:
        files_found = [os.path.join(dir_path, f) for dir_path, _, files in os.walk(data_path) for f in files if f.endswith(fname_ext)]
    else:
        files_found = [os.path.join(dir_path, f) for dir_path, _, files in os.walk(data_path) for f in files if f.startswith(prefix) and f.endswith(fname_ext)]
    
    if len(files_found) == 0:
        raise ValueError(f"No files found with the specified pattern in {data_path}.")

    # Search for the specific file name
    for file_path in files_found:
        _tmp_name = os.path.basename(file_path)
        _tmp_name = os.path.splitext(_tmp_name)[0]
        _tmp_name_parts = _tmp_name.split('_')
        if _tmp_name_parts[-1] == trnvaltst:
            if _tmp_name_parts[-2] == str(which_inner_val) and _tmp_name_parts[-3] == str(which_outer_test):
                return read_omics(file_path)
    
    raise ValueError(f"No file found with the specified pattern in {data_path}.")


class ProcOnTrainSet:
    def __init__(self, df_in: pl.DataFrame, ind_for_fit: Optional[List[Any]], n_feat2save: Optional[int] = None, df_labels: Optional[pl.DataFrame] = None):
        r"""Process all data points based on the training set.

        Args:
            df_in: Input dataframe.
            
            ind_for_fit: Sample indices for fitting the preprocessors.
            
            n_feat2save: Number of features to save.
            
            df_labels: Labels dataframe.
        
        How to use:
            - Initialize the class.
            - Call the method `pr_xxxxx` to process the data.
            - Call the method `save_processors` to save the processors (as a dict) to a pickle file.

        """
        self.ind_for_fit = ind_for_fit
        self.n_feat2save = n_feat2save
        self._df = df_in
        if df_labels is not None:
            self._labels = df_labels

        if ind_for_fit is not None:
            self._df_part = self._df[ind_for_fit,:]
            if df_labels is not None:
                self._labels_part = self._labels[ind_for_fit,:]
        else:
            self._df_part = self._df
            if df_labels is not None:
                self._labels_part = df_labels
                
        self.preprocessors = {}
    
    def keep_preprocessors(self, x_value: Any):
        r"""The key (int, ***0-based***) is automatically generated by the order of the data processor,
        for the reproduction of data processing steps.
        """
        x_order = len(self.preprocessors)
        self.preprocessors[x_order] = x_value

    def save_preprocessors(self, dir_save_processors: str, file_name: Optional[str] = None):
        if file_name is None:
            fname_preprocessors = const.fname.preprocessors
        else:
            fname_preprocessors = file_name
        os.makedirs(dir_save_processors, exist_ok=True)
        path_save_processors = os.path.join(dir_save_processors, fname_preprocessors)
        if os.path.exists(path_save_processors):
            raise FileExistsError(f"The file {path_save_processors} already exists.")
        
        if len(self.preprocessors) < 1:
            raise Warning("No data processor is saved.")
        else:
            with open(path_save_processors, "wb") as f:
                pickle.dump(self.preprocessors, f)
            print(f"The data processors have been saved to: {path_save_processors}")
    
    def load_run_preprocessors(self, dir_save_processors: str, file_name: str):
        path_processors = os.path.join(dir_save_processors, file_name)
        with open(path_processors, "rb") as f:
            processors_dict = pickle.load(f)
        # Run the processors
        _tmp_df = self._df.drop(const.dkey.id).to_numpy()
        for i in range(len(processors_dict.keys())):
            _tmp_df = processors_dict[i].transform(_tmp_df)
        _tmp_df = pl.DataFrame(_tmp_df, schema=self._df.columns[1:])
        _tmp_df = pl.DataFrame({const.dkey.id: self._df[const.dkey.id]}).hstack(_tmp_df)
        self._df = _tmp_df
    
    def general_preprocessor(self, _processor: Any):
        try:
            _processor.fit(self._df_part.drop(const.dkey.id).to_numpy())
        except:
            try:
                _processor.fit(self._df_part)
            except:
                _processor.fit(self._df_part, self._labels_part)
        
        try:
            df_o = _processor.transform(self._df.drop(const.dkey.id).to_numpy())
        except:
            df_o = _processor.transform(self._df)
        
        if type(df_o) != pl.DataFrame:
            df_o = pl.DataFrame(df_o, schema=self._df.columns[1:])
            df_o = pl.DataFrame({const.dkey.id: self._df[const.dkey.id]}).hstack(df_o)
        
        self._df = df_o
        self.keep_preprocessors(_processor)

        # Refresh the part data for next processing steps.
        if self.ind_for_fit is not None:
            self._df_part = self._df[self.ind_for_fit,:]
        else:
            self._df_part = self._df
    
    def pr_var(self, threshold: float = const.default.variance_threshold):
        _selector = VarThreSelector(threshold=threshold)
        self.general_preprocessor(_selector)

    def pr_minmax(self):
        _processor = MinMaxScaler()
        self.general_preprocessor(_processor)
    
    def pr_zscore(self):
        _processor = StandardScaler()
        self.general_preprocessor(_processor)
        
    def pr_impute(self, strategy: str = "mean"):
        _imputer = SimpleImputer(strategy=strategy)
        self.general_preprocessor(_imputer)
    
    def pr_rf(self, random_states: List[int], n_estimators: int = const.default.n_estimators, n_jobs_rf=const.default.n_jobs_rf):
        if not hasattr(self, "_labels") or self.n_feat2save is None:
            raise ValueError("The labels are not provided.")
        
        _selector = RFSelector(self.n_feat2save, random_states, n_estimators, n_jobs=n_jobs_rf)
        self.general_preprocessor(_selector)


class VarThreSelector:
    def __init__(self, threshold: float):
        r"""Select features based on variance.
        After `fit`, the selector recgonizes the colnames of the input dataframe.

        Args:
            threshold: The threshold for variance.
        
        """
        self.threshold = threshold
        self.processors = {}
    
    def fit(self, omics_df: pl.DataFrame):
        _omics_np = omics_df.drop(const.dkey.id).to_numpy()
        _selector = VarianceThreshold(threshold=self.threshold)
        _selector.fit(_omics_np)
        _feat_to_save = _selector.get_support()
        _colnames = omics_df.drop(const.dkey.id).columns
        self.colname_to_save = [_colnames[i] for i in range(len(_colnames)) if _feat_to_save[i]]
    
    def transform(self, X_df: pl.DataFrame):
        _selected = X_df.select(self.colname_to_save)
        df_o = pl.DataFrame({const.dkey.id: X_df[const.dkey.id]}).hstack(_selected)
        return df_o


class RFSelector:
    def __init__(self, n_feat2save: int, random_states: List[int], n_estimators: int, n_jobs: int, save_processors: bool = False):
        r"""Select features based on random forest.
        After `fit`, the selector recgonizes the colnames of the input dataframe.

        Args:
            n_feat2save: The number of features to save.
            
            random_states: The random states for the random forest.
            
            n_estimators: The number of trees in the random forest.
            
            n_jobs: The number of jobs to run in parallel.
            
            save_processors: Whether to save the random forest models.
        
        """
        self.n_feat2save = n_feat2save
        self.random_states = random_states
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.processors = {}
        self.save_processors = save_processors
    
    def fit(self, omics_df: pl.DataFrame, labels_df: pl.DataFrame):
        _omics_np = omics_df.drop(const.dkey.id).to_numpy()
        _labels_np = labels_df.drop(const.dkey.id).to_numpy()
        if _labels_np.shape[1] == 1:
            _labels_np = _labels_np.ravel()
        _feat_imp = np.zeros(shape=(len(self.random_states), _omics_np.shape[1]))
        print(f"Starting to fit RF for {len(self.random_states)} random states...")
        for i in range(len(self.random_states)):
            _feat_imp[i,:] = self.fit_1(_omics_np, _labels_np, self.random_states[i])
            print(f"Finished {i+1}/{len(self.random_states)}")
        _feat_imp_mean = np.mean(_feat_imp, axis=0)
        _feat_imp_mean_sorted = np.argsort(_feat_imp_mean)[::-1]
        if self.n_feat2save <= _omics_np.shape[1]:
            _feat_to_save = _feat_imp_mean_sorted[:self.n_feat2save]
        else:
            _feat_to_save = _feat_imp_mean_sorted
        _colnames = omics_df.drop(const.dkey.id).columns
        self.colname_to_save = [_colnames[i] for i in _feat_to_save]
    
    def transform(self, X_df: pl.DataFrame):
        _selected = X_df.select(self.colname_to_save)
        df_o = pl.DataFrame({const.dkey.id: X_df[const.dkey.id]}).hstack(_selected)
        return df_o

    def keep_preprocessor(self, x_processor: Any):
        r"""The key (int, ***0-based***) is automatically generated by the order of the data processor,
        for the reproduction of data processing steps.
        
        Args:
            x_processor: the processor to be kept.
        
        """
        x_order = len(self.processors)
        self.processors[x_order] = x_processor

    def fit_1(self, X: np.ndarray, y: np.ndarray, random_state: int):
        _processor = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=self.n_jobs, random_state=random_state)
        _processor.fit(X, y)
        if self.save_processors:
            self.keep_preprocessor(_processor)
        return _processor.feature_importances_


def get_indices_ncv(
        k_outer: int,
        k_inner: int,
        which_outer_test: int,
        which_inner_val: int,
    ):
    r"""Get indices of fragments for NCV.
    """
    # Init fragment indices for test dataset
    n_fragments = int(k_outer * k_inner)
    n_f_test = int(n_fragments / k_outer)
    indices_test_dataset = [i for i in range(int(which_outer_test * n_f_test), int((which_outer_test + 1) * n_f_test))]
    # Indices excluding test dataset
    indices_train_dataset = [i for i in range(n_fragments) if i not in indices_test_dataset]
    # Indices for validation dataset
    parts = np.array_split(indices_train_dataset, k_inner)
    indices_val_dataset = parts[which_inner_val].tolist()
    indices_trn_dataset = [i for i in indices_train_dataset if i not in indices_val_dataset]
    
    return indices_trn_dataset, indices_val_dataset, indices_test_dataset


def onehot_encode_snp_mat(
        snp_matrix: np.ndarray,
        onehot_bits: Optional[int] = None,
        genes_snps: Optional[List[List[int]]] = None,
    ):
    r"""One-hot encode the SNP matrix.
    """
    if onehot_bits is None:
        len_onehot = const.default.snp_onehot_bits
    else:
        len_onehot = onehot_bits
    
    if genes_snps is not None:
        num_genes = len(genes_snps)
        indices_snp = []
        for i_gene in range(num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_onehot))
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_onehot + 1)[snp_vec][:, 1:].reshape(-1)
            snp_vec_genes = [snp_vec[indices_snp[i_gene]].astype(np.float32) for i_gene in range(num_genes)]
            snp_data.append(snp_vec_genes)
    else:
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_onehot + 1)[snp_vec][:, 1:].reshape(-1).astype(np.float32)
            snp_data.append(snp_vec)
    snp_data_np = np.array(snp_data)
    return snp_data_np


def read_pkl_gv(path_pkl: str) -> Dict[str, Any]:
    r"""Read processed VCF data from a pickle file.
    """
    with gzip.open(path_pkl, 'rb') as file:
        # Initialize an empty list to hold all the deserialized vectors
        _vectors = []

        # While there is data in the file, load it
        while True:
            try:
                # Load the next pickled object from the file
                _data = pickle.load(file)
                # Append the loaded data to the list
                _vectors.append(_data)
            except EOFError:
                # An EOFError is raised when there is no more data to read
                break

    _sample_ids = _vectors[0]
    _snp_ids = _vectors[1]
    _block_ids = _vectors[2]
    _block2gtype = _vectors[3]
    _mat_vec = _vectors[4]
    _mat_shape = (len(_snp_ids), len(_sample_ids))

    # Reshape the matrix to the correct shape
    vcf_mat = np.reshape(_mat_vec, _mat_shape).transpose()

    return {
        const.dkey.genotype_matrix: vcf_mat,
        const.dkey.gblock2gtype: _block2gtype,
        const.dkey.sample_ids: _sample_ids,
        const.dkey.snp_ids: _snp_ids,
        const.dkey.gblock_ids: _block_ids,
    }


def train_model(
        model: Any,
        datamodule: LightningDataModule,
        es_patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        devices: Union[list[int], str, int] = const.default.devices,
        accelerator: str = const.default.accelerator,
        in_dev: bool = False,
    ):
    r"""Fit the model.
    """
    avail_dev = get_avail_nvgpu(devices)

    callback_es = EarlyStopping(
        monitor=const.title_val_loss,
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=const.default.ckpt_fname_format,
        monitor=const.title_val_loss,
    )

    logger_tr = TensorBoardLogger(
        save_dir=log_dir,
        name='',
    )

    trainer = Trainer(
        fast_dev_run=in_dev,
        logger=logger_tr,
        log_every_n_steps=1,
        precision="16-mixed",
        devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
    )
    
    trainer.fit(model=model, datamodule=datamodule)

    if callback_ckpt.best_model_score is not None:
        best_score = callback_ckpt.best_model_score.item()
    else:
        best_score = None

    trainer.test(ckpt_path=callback_ckpt.best_model_path, dataloaders=datamodule)

    print(f"\nBest validation score: {best_score}")
    print(f"Best model path: {callback_ckpt.best_model_path}\n")

    return best_score


class CollectFitLog:
    def __init__(self, dir_log: str):
        r"""Collect training logs from optuna db files and ckpt files.

        Args:
            dir_log: Directory containing the model fitting logs.
        
        """
        self.dir_log = dir_log
        if not os.path.exists(self.dir_log):
            raise ValueError(f'Directory {self.dir_log} does not exist.')
    
    def get_df_csv(self, dir_output: str, overwrite_collected_log: bool = False):
        r"""Collect trained models for each fold in nested cross-validation.

        Args:
            dir_output: Directory to save the collected logs.

            overwrite_collected_log: Whether to overwrite existing collected logs.
        
        """
        collected_logs = self.collect()

        models_bv = collected_logs[const.dkey.best_trials]
        path_log_best_trials = os.path.join(dir_output, const.fname.log_best_trials)
        if os.path.exists(path_log_best_trials) and not overwrite_collected_log:
            models_bv = pl.read_csv(path_log_best_trials)
        else:
            models_bv.write_csv(path_log_best_trials)

        models_bi = collected_logs[const.dkey.best_inner_folds]
        path_log_best_inners = os.path.join(dir_output, const.fname.log_best_inners)
        if os.path.exists(path_log_best_inners) and not overwrite_collected_log:
            models_bi = pl.read_csv(path_log_best_inners)
        else:
            models_bi.write_csv(path_log_best_inners)
        
        return models_bv, models_bi
    
    def collect(self) -> Dict[str, pl.DataFrame]:
        r"""Collect training logs from optuna db files and ckpt files.
        """
        best_trials_df, all_ckpt = self.collect_ckpt()
        optuna_best_inners_df = self.collect_optuna_db()

        # Merge the two dataframes on the const.dkey.which_outer and const.dkey.which_inner columns
        logs_df = optuna_best_inners_df.join(best_trials_df, on=[const.dkey.which_outer, const.dkey.which_inner], how='left')
        # Remove the const.title_val_loss column from the merged dataframe
        logs_df = logs_df.drop(const.title_val_loss)
        # Rename 'min_loss' column to const.title_val_loss
        logs_df = logs_df.rename({const.dkey.min_loss: const.title_val_loss})
        # Sort the dataframe by const.dkey.which_outer and const.dkey.which_inner
        logs_df = logs_df.sort([const.dkey.which_outer, const.dkey.which_inner])

        best_inners_df = logs_df.group_by(const.dkey.which_outer).agg(pl.col(const.title_tst_loss).min()).join(logs_df, on=[const.dkey.which_outer, const.title_tst_loss], how='left')
        best_inners_df = best_inners_df.sort([const.dkey.which_outer, const.title_tst_loss])

        print("\nFound model logs:")
        print(logs_df)
        print("\nBest inner folds:")
        print(best_inners_df)

        return {const.dkey.best_trials: logs_df, const.dkey.best_inner_folds: best_inners_df}
    
    def collect_ckpt(self):
        r"""Collect info from ckpt files and tensorboard events.
        """
        paths_ckpt = self.search_ckpt()

        # Pick ids of outer and inner folds, val_loss and version from ckpt file paths
        test_loss_values = [self.read_tensorboard_events(os.path.join(os.path.dirname(path_x), "version_0")) for path_x in paths_ckpt]
        val_loss_values = [float(os.path.basename(path_x).split('-')[3].split('=')[1].split('.ckpt')[0]) for path_x in paths_ckpt]
        trial_tags = [path_x.split(os.path.sep)[-2] for path_x in paths_ckpt]
        study_tags = [path_x.split(os.path.sep)[-4].split('_')[-1] for path_x in paths_ckpt]
        ncv_inner_x = [int(path_x.split(os.path.sep)[-3].split('_')[-1]) for path_x in paths_ckpt]
        ncv_outer_x = [int(path_x.split(os.path.sep)[-3].split('_')[-2]) for path_x in paths_ckpt]

        # Create a dataframe with the above values
        ckpt_df = pl.DataFrame({const.dkey.which_outer: ncv_outer_x, const.dkey.which_inner: ncv_inner_x, const.title_tst_loss: test_loss_values, const.title_val_loss: val_loss_values, const.dkey.trial_tag: trial_tags, const.dkey.study_tag: study_tags, const.dkey.ckpt_path: paths_ckpt})

        # Pick the best model based on val_loss between the trials of the same outer and inner fold
        best_trials_df = ckpt_df.group_by([const.dkey.which_outer, const.dkey.which_inner]).agg([pl.col(const.title_val_loss).min()]).join(ckpt_df, on=[const.dkey.which_outer, const.dkey.which_inner, const.title_val_loss], how='left')

        return best_trials_df, ckpt_df

    def collect_optuna_db(self):
        r"""Collect info of optuna db files.
        """
        # Find all optuna db files in the directory `dir_log` and its subdirectories
        paths_optuna_db = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(self.dir_log)
                    for f in files if f.endswith('.db')
        ]
        if len(paths_optuna_db) == 0:
            raise FileNotFoundError('No optuna db files found.')
        paths_optuna_db.sort()
        print(f'Found {len(paths_optuna_db)} optuna db files\n')
        
        # Read optuna db files and store the results in a dataframe
        studies_dicts = [self.read_optuna_db(path_optuna_db) for path_optuna_db in paths_optuna_db]
        studies_df = pl.DataFrame(studies_dicts)
        
        return studies_df

    def read_optuna_db(self, path_optuna_db: str) -> Dict[str, Any]:
        loaded_study = optuna.load_study(study_name=None, storage=f"sqlite:///{path_optuna_db}")
        study_name = loaded_study.study_name
        min_loss = loaded_study.best_value
        # trials_df = loaded_study.trials_dataframe()

        frag_name = study_name.split('_')
        assert len(frag_name) > 4, 'Study name is not in the expected format.'
        x_outer = int(frag_name[2])
        x_inner = int(frag_name[3])
        x_time = frag_name[4]
        return {const.dkey.study_name: study_name, const.dkey.which_outer: x_outer, const.dkey.which_inner: x_inner, const.dkey.min_loss: min_loss, const.dkey.time_str: x_time}
    
    def read_tensorboard_events(self, dir_events: str, get_test_loss: bool = True):
        r"""Read tensorboard events from the directory.
        """
        event_acc = EventAccumulator(dir_events)
        event_acc.Reload()
        scalar_tags = event_acc.Tags()["scalars"]
        scalar_data = {tag: [] for tag in scalar_tags}

        for tag in scalar_tags:
            _events = event_acc.Scalars(tag)
            for _event in _events:
                scalar_data[tag].append((_event.step, _event.value))
        
        test_loss: float = scalar_data[const.title_tst_loss][0][1]
        
        if get_test_loss:
            return test_loss
        else:
            return scalar_data

    def search_ckpt(self):
        r"""Search checkpoints in the directory and its subdirectories.
        """
        paths_ckpt = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(self.dir_log)
                    for f in files if f.endswith('.ckpt')]
        if len(paths_ckpt) == 0:
            raise FileNotFoundError("No checkpoint files found.")
        paths_ckpt.sort()
        print(f'Found {len(paths_ckpt)} checkpoints.\n')
        return paths_ckpt
    
    def remove_inferior_models(self):
        r"""Remove inferior models based on the collected result table.
        """
        best_trials, all_trials = self.collect_ckpt()
        n_all_ckpt = len(all_trials)
        n_removed_models = 0
        for _x in range(n_all_ckpt):
            # Check if all_trials[const.dkey.trial_tag][_x] is in best_trials[const.dkey.trial_tag]
            if best_trials[const.dkey.trial_tag].str.contains(all_trials[const.dkey.trial_tag][_x]).any():
                continue
            else:
                os.remove(all_trials[const.dkey.ckpt_path][_x])
                print(f"Removed {all_trials[const.dkey.ckpt_path][_x]}")
                n_removed_models += 1
        print(f"Removed {n_removed_models} inferior models.")
        return None
