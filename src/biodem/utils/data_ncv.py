r"""
Read multi-omics data (and labels) and split into nested train/val/test sets.

The processed data will be stored in litdata's format.

"""

import os
import shutil
from typing import Optional, Union, List, Dict, Any
from copy import deepcopy
import itertools
import numpy as np
import polars as pl
from litdata import optimize, StreamingDataLoader, StreamingDataset
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import biodem.constants as const
from biodem.utils.uni import intersect_lists, read_labels, read_omics, read_omics_xoxi, get_indices_ncv, ProcOnTrainSet, onehot_encode_snp_mat, random_string


class DEMDataset:
    def __init__(
            self,
            reproduction_mode: bool,
            paths_omics: Dict[str, str],
            path_label: Optional[str] = None,
            col2use_in_label: Optional[Union[List[str], List[int]]] = None,
            sample_ind_for_preproc: Optional[List[int]] = None,
            dir_preprocessors: Optional[str] = None,
            target_n_samples: Optional[int] = None,
            seed_resample: int = const.default.seed_1,
            n_fragments: int = 1,
            prepr_labels: bool = True,
            prepr_omics: bool = True,
            snp_onehot_bits: int = const.default.snp_onehot_bits,
            which_outer_test: Optional[int] = None,
            which_inner_val: Optional[int] = None,
            variance_threshold: float = const.default.variance_threshold,
            save_n_feat: int = const.default.n_feat2save,
            n_estimators: int = const.default.n_estimators,
            random_states: List[int] = const.default.random_states,
            n_jobs_rf: int = const.default.n_jobs_rf,
        ) -> None:
        r"""Read data for litdata optimization. Preprocessing is optional.

        If you have labels (phenotypes) and want to use them,
        the input labels (phenotypes) are expected as follows:

        + For **REGRESSION** task
            + Please keep original values that are not preprocessed.
            + If you have MULTIPLE traits, please set different columns names in CSV file.
            + The pipeline ***standardize/normalize data AFTER splitting*** to ***avoid data leakage.***
            + The method and parameters of standardization/normalization are kept the same as those in training data.
        + For **CLASSIFICATION** task
            + Please transform labels by one-hot encoder ***MANUALLY BEFORE input***.
            + The length of one-hot vectors is recommended to be **n_categories + 1** for **UNPRECEDENTED labels**.
            + If you have MULTIPLE traits, please **concatenate** one-hot encoded matrix along the horizontal axis before input.

        Args:
            reproduction_mode: Whether to use existing processors.
                Please provide ``dir_preprocessors`` if ``reproduction_mode`` is ``True``.
            
            paths_omics: The ``Dict`` ``{name: path}`` of paths to multiple ``.csv`` or ``.parquet`` files or directories. For genotypes, it is a path to a ``.pkl.gz`` file.
                If some paths are directories, the files inside will be read as existing data, and ``target_n_samples`` will be ignored.

            path_label: The path to label data (a ``.csv`` or ``.parquet`` file).
                If it is `None`, no labels(phenotypes) are provided.

            col2use_in_label: The columns to use in label data.
                If it is `List[int]`, its numbers are the indices **(1-based)** of the columns to be used.
                If it is `None`, all columns are used.
            
            sample_ind_for_preproc: The indices for selecting samples for preprocessors fitting.
                If it is `None`, all samples are used.
                It is ignored when existing data are used.

            dir_preprocessors: The directory used to save data processors that fitted on training data.
                If it is `None`, preprocessing is not performed.
            
            target_n_samples: The target sample size for expanding the dataset through random sampling.
                Resampling is not performed when existing data are used.

            seed_resample: The random seed for sampling new samples.
                Default: ``const.default.seed_1``.

            n_fragments: The number of fragments (= k_outer * k_inner).
                Default: ``1``.

            prepr_labels: Whether to preprocess labels or not.
                Default: ``True``.

            prepr_omics: Whether to preprocess omics data or not.
                Default: ``True``.

            snp_onehot_bits: The number of bits for one-hot encoding SNPs.
                Default: ``const.default.snp_onehot_bits``.

            which_outer_test: The index of outer test set. It is used for reading existing data.

            which_inner_val: The index of inner validation set. It is used for reading existing data.

            variance_threshold: The threshold for variance selection.

            save_n_feat: The number of features to save when random forest selection is performed.
                Default: ``const.default.n_feat2save``.
            
            n_estimators: The number of trees in the random forest.
                Default: ``const.default.n_estimators``.
            
            random_states: The random states for random forest selection.
                Default: ``const.default.random_states``.

            n_jobs_rf: The number of jobs for parallelized random forest fitting.
                Default: ``const.default.n_jobs_rf``.

        Usage:

            >>> from biodem.utils.data_ncv import DEMDataset
            >>> _dataset = DEMDataset(...)
            >>> _dataset._setup()
            
        """
        super().__init__()
        self.reproduction_mode = reproduction_mode
        self.paths_omics = paths_omics
        self.path_label = path_label
        self.col2use_in_label = col2use_in_label

        self.sample_ind_for_preproc = sample_ind_for_preproc
        self.dir_preprocessors = dir_preprocessors
        
        self.prepr_labels = prepr_labels
        self.prepr_omics = prepr_omics
        self.variance_threshold = variance_threshold
        self.save_n_feat = save_n_feat
        self.n_estimators = n_estimators
        self.random_states = random_states
        self.n_jobs_rf = n_jobs_rf
        
        self.snp_onehot_bits = snp_onehot_bits
        
        self.target_n_samples = target_n_samples
        self.seed_resample = seed_resample
        self.n_fragments = n_fragments

        self.which_outer_test = which_outer_test
        self.which_inner_val = which_inner_val

        self.existing_omics_sample_id: Dict[str, Dict[str, List[str]]] = {}
        self.existing_omics: Dict[str, Dict[str, pl.DataFrame]] = {}
        
        self.omics_name = sorted(list(paths_omics.keys()))
        self.omics_name_new: List[str] = []
        self.omics_name_existing: List[str] = []
        self.indices_trn: List[int] | None = None
        self.indices_val: List[int] | None = None
        self.indices_tst: List[int] | None = None

        self.key_gv = None
        self.n_omics = len(self.omics_name)
        self.omics_dfs: Dict[str, pl.DataFrame] = {}
        
        self.sample_ids: List[str] = []
        self.labels_df = None
        self.dim_model_output = None

        self.omics_data: List[np.ndarray] = []
        self.omics_features: List[List[str]] = []
        self.omics_dims: List[int] = []

    def _setup(self):
        self._pick_shared_samples_in_omics()

        if self.path_label is not None:
            self._pick_shared_samples_in_omics_and_labels(self.path_label, self.col2use_in_label)
        
        self.n_samples = len(self.sample_ids)

        # Try reading existing (treated) omics data
        self._read_existing_omics(self.paths_omics)

        # ! Pick samples that are shared in all omics, especially for the case of existing data
        self.recommend_index_by_existing_omics()
        
        # !!! Resampling is not necessary when using existing data !!!
        if len(self.omics_name_existing) < 1:
            self._calc_n_samples2sample(self.target_n_samples, self.n_fragments)

            if self.n_samples_to_add > 0:
                self._sample_new2add(self.seed_resample)
        
        # Preprocess omics data
        if self.dir_preprocessors is not None:
            if self.prepr_labels:
                self._proc_labels(self.sample_ind_for_preproc, self.dir_preprocessors, self.reproduction_mode)
            if self.prepr_omics:
                self._proc_omics(self.sample_ind_for_preproc, self.dir_preprocessors, self.reproduction_mode)
        
        # Preprocess genotype data
        if self.key_gv is not None:
            gv_np = self.omics_dfs[self.key_gv].drop(const.dkey.id).to_numpy()
            gv_np_onehot = onehot_encode_snp_mat(gv_np, self.snp_onehot_bits)
            self.omics_dfs[self.key_gv] = pl.DataFrame({const.dkey.id: self.sample_ids}).hstack(pl.DataFrame(data=gv_np_onehot))

        # Get omics data properties
        for i in range(self.n_omics):
            self.omics_data.append(self.omics_dfs[self.omics_name[i]].drop(const.dkey.id).to_numpy().astype(np.float32))
            self.omics_features.append(self.omics_dfs[self.omics_name[i]].drop(const.dkey.id).columns)
            self.omics_dims.append(self.omics_data[i].shape[1])
        
        if self.dir_preprocessors is not None:
            # Write omics' dimensions for the initialization of the model
            pl.DataFrame(data={const.dkey.omics_dim: self.omics_dims}).write_csv(os.path.join(self.dir_preprocessors, const.fname.predata_omics_dims))
            # Write omics' features
            for i in range(self.n_omics):
                pl.DataFrame(data={const.dkey.omics_feature: self.omics_features[i]}).write_csv(os.path.join(self.dir_preprocessors, f"{const.fname.predata_omics_features_prefix}_{self.omics_name[i]}.csv"))
            # Write labels' names
            if self.labels_df is not None:
                pl.DataFrame(data={const.dkey.label: self.labels_df.columns[1:]}).write_csv(os.path.join(self.dir_preprocessors, const.fname.predata_label_names))
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        omics_data_i = [omics_x[index, :] for omics_x in self.omics_data]
        sample_id_i = self.sample_ids[index]
        if hasattr(self, "label_data"):
            label_data_i = self.label_data[index, :]
            data_o = {const.dkey.litdata_index: index, const.dkey.litdata_omics: deepcopy(omics_data_i), const.dkey.litdata_id: deepcopy(sample_id_i), const.dkey.litdata_label: deepcopy(label_data_i)}
        else:
            data_o = {const.dkey.litdata_index: index, const.dkey.litdata_omics: deepcopy(omics_data_i), const.dkey.litdata_id: deepcopy(sample_id_i)}
        return data_o
    
    def _read_existing_omics(self, paths_omics: Dict[str, str], default_file_ext: str = const.fname.data_ext):
        r"""Read existing omics data from the given paths.
        If the path is a directory, then the data will be read from the files that names could be recognized by the function ``read_omics_xoxi``.
        """
        for ikey in self.omics_name:
            _tmp_path = paths_omics[ikey]

            if os.path.isdir(_tmp_path):
                if self.which_inner_val is not None and self.which_outer_test is not None:
                    _tmp_trn = read_omics_xoxi(_tmp_path, self.which_outer_test, self.which_inner_val, const.abbr_train, default_file_ext)
                    _tmp_val = read_omics_xoxi(_tmp_path, self.which_outer_test, self.which_inner_val, const.abbr_val, default_file_ext)
                    _tmp_tst = read_omics_xoxi(_tmp_path, self.which_outer_test, self.which_inner_val, const.abbr_test, default_file_ext)
                    _tmp_trn_sample_id = _tmp_trn.select(const.dkey.id).to_series().to_list()
                    _tmp_val_sample_id = _tmp_val.select(const.dkey.id).to_series().to_list()
                    _tmp_tst_sample_id = _tmp_tst.select(const.dkey.id).to_series().to_list()
                    self.existing_omics_sample_id[ikey] = {const.abbr_train: _tmp_trn_sample_id, const.abbr_val: _tmp_val_sample_id, const.abbr_test: _tmp_tst_sample_id}
                    self.existing_omics[ikey] = {const.abbr_train: _tmp_trn, const.abbr_val: _tmp_val, const.abbr_test: _tmp_tst}
                    self.omics_name_existing.append(ikey)
                else:
                    raise NotImplementedError
            else:
                continue
        return None
    
    def recommend_index_by_existing_omics(self):
        r"""Recommend indices for the samples that are shared in all omics if existing data is used.
        """
        if len(self.omics_name_existing) < 1:
            return None
        inters_trn, indices_trn = intersect_lists([self.sample_ids, *[self.existing_omics_sample_id[ikey][const.abbr_train] for ikey in self.omics_name_existing]])
        inters_val, indices_val = intersect_lists([self.sample_ids, *[self.existing_omics_sample_id[ikey][const.abbr_val] for ikey in self.omics_name_existing]])
        inters_tst, indices_tst = intersect_lists([self.sample_ids, *[self.existing_omics_sample_id[ikey][const.abbr_test] for ikey in self.omics_name_existing]])
        
        for i in range(len(self.omics_name_existing)):
            _tmp_part_trn = self.existing_omics[self.omics_name_existing[i]][const.abbr_train][indices_trn[i+1], :]
            _tmp_part_val = self.existing_omics[self.omics_name_existing[i]][const.abbr_val][indices_val[i+1], :]
            _tmp_part_tst = self.existing_omics[self.omics_name_existing[i]][const.abbr_test][indices_tst[i+1], :]
            self.omics_dfs[self.omics_name_existing[i]] = _tmp_part_trn.vstack(_tmp_part_val).vstack(_tmp_part_tst)

        for i in range(len(self.omics_name_new)):
            _tmp_part_trn = self.omics_dfs[self.omics_name_new[i]][indices_trn[0],:]
            _tmp_part_val = self.omics_dfs[self.omics_name_new[i]][indices_val[0],:]
            _tmp_part_tst = self.omics_dfs[self.omics_name_new[i]][indices_tst[0],:]
            self.omics_dfs[self.omics_name_new[i]] = _tmp_part_trn.vstack(_tmp_part_val).vstack(_tmp_part_tst)
        
        if self.labels_df is not None:
            self.labels_df = self.labels_df[indices_trn[0]+indices_val[0]+indices_tst[0],:]

        n_id_trn = len(inters_trn)
        n_id_val = len(inters_val)
        n_id_tst = len(inters_tst)
        self.indices_trn = [i for i in range(n_id_trn)]
        self.indices_val = [i+n_id_trn for i in range(n_id_val)]
        self.indices_tst = [i+n_id_trn+n_id_val for i in range(n_id_tst)]

        self.sample_ids = inters_trn + inters_val + inters_tst
        self.sample_ids_trn = inters_trn
        self.sample_ids_val = inters_val
        self.sample_ids_tst = inters_tst
        
        self.n_samples = len(self.sample_ids)
        self.sample_ind_for_preproc = self.indices_trn

    def _pick_shared_samples_in_omics(self):
        r"""Pick samples that are shared between omics.
        """
        original_omics_IDs = []
        for ikey in self.omics_name:
            _tmp_path = self.paths_omics[ikey]

            if os.path.isdir(_tmp_path):
                continue
            else:
                self.omics_dfs[ikey] = read_omics(_tmp_path)
                if _tmp_path.lower().endswith(".pkl.gz"):
                    self.key_gv = ikey

                original_omics_IDs.append(self.omics_dfs[ikey].select(const.dkey.id).to_series().to_list())
                self.omics_name_new.append(ikey)
        
        intersect_ids_in_omics, _indices = intersect_lists(original_omics_IDs)
        for i in range(len(self.omics_name_new)):
            self.omics_dfs[self.omics_name_new[i]] = self.omics_dfs[self.omics_name_new[i]][_indices[i],:].sort(const.dkey.id)
        
        self.sample_ids = intersect_ids_in_omics
    
    def _pick_shared_samples_in_omics_and_labels(self, path_label: str, col2use_in_label: Optional[Union[List[str], List[int]]]):
        r"""Pick samples that are shared between omics and labels.
        """
        labels_df, dim_model_output, sample_ids_in_labels = read_labels(path_label, col2use_in_label)
        intersect_ids, _indices = intersect_lists([self.sample_ids, sample_ids_in_labels])
        
        for i in range(len(self.omics_name_new)):
            self.omics_dfs[self.omics_name_new[i]] = self.omics_dfs[self.omics_name_new[i]][_indices[0],:].sort(const.dkey.id)
        labels_df = labels_df[_indices[1],:].sort(const.dkey.id)

        self.sample_ids = intersect_ids
        self.labels_df = labels_df
        self.model_output_dim = dim_model_output
    
    def _calc_n_samples2sample(self, target_n_samples: Optional[int], n_fragments: int):
        r"""Calculate the number of samples to add to the dataset.
        """
        n_samples_to_add = 0
        match target_n_samples:
            case None:
                if n_fragments > 1:
                    if self.n_samples % n_fragments != 0:
                        n_samples_to_add = n_fragments - (self.n_samples % n_fragments)
            case x if x > self.n_samples:
                n_samples_to_add = x - self.n_samples
            case _:
                raise Warning("target_n_samples must be larger than the number of samples")
        self.n_samples_to_add = n_samples_to_add
        self.n_samples_target = self.n_samples + n_samples_to_add
    
    def _sample_new2add(self, seed_resample: int):
        r"""Generate new samples by resampling existing samples.
        """
        np.random.seed(seed_resample)
        new_indices: list[int] = np.random.choice(self.n_samples, self.n_samples_to_add, replace=True).tolist()
        self.n_samples = self.n_samples_target
        self.sample_ids = self.sample_ids + [self.sample_ids[i] for i in new_indices]
        
        if self.labels_df is not None:
            self.labels_df = self.labels_df.vstack(self.labels_df[new_indices,:])

        for ikey in self.omics_name:
            self.omics_dfs[ikey] = self.omics_dfs[ikey].vstack(self.omics_dfs[ikey][new_indices,:])
    
    def _proc_omics(self, sample_ind_for_proc: Optional[List[int]], dir_preprocessors: str, reproduction_mode: bool):
        r"""Preprocess omics data.
        """
        if reproduction_mode:
            if os.path.exists(dir_preprocessors):
                for ikey in self.omics_name:
                    if self.key_gv is not None and self.key_gv == ikey:
                        continue
                    _loaded_proc = ProcOnTrainSet(self.omics_dfs[ikey], None)
                    _loaded_proc.load_run_preprocessors(dir_preprocessors, f'preprocessors_for_omics_{ikey}.pkl')
                    self.omics_dfs[ikey] = _loaded_proc._df
            else:
                raise FileNotFoundError(f"Processor files for omics data are not found in {dir_preprocessors}")
        else:
            for ikey in self.omics_name:
                if self.key_gv is not None and self.key_gv == ikey:
                    continue
                _tmp_proc = ProcOnTrainSet(self.omics_dfs[ikey], sample_ind_for_proc, self.save_n_feat, self.labels_df)
                _tmp_proc.pr_impute(strategy="mean")
                print(f"\nDone imputation for {ikey}")
                _tmp_proc.pr_var(threshold=self.variance_threshold)
                print(f"\nDone variance filtering for {ikey}")
                _tmp_proc.pr_minmax()
                print(f"\nDone min-max scaling for {ikey}")
                _tmp_proc.pr_rf(self.random_states, self.n_estimators, self.n_jobs_rf)
                print(f"\nDone random forest filtering for {ikey}")
                _tmp_proc.save_preprocessors(dir_preprocessors, f'preprocessors_for_omics_{ikey}.pkl')
                self.omics_dfs[ikey] = _tmp_proc._df
    
    def _proc_labels(self, sample_ind_for_proc: Optional[List[int]], dir_preprocessors: str, reproduction_mode: bool):
        r"""Preprocess labels.
        """
        if self.labels_df is not None:
            if reproduction_mode:
                if os.path.exists(dir_preprocessors):
                    labels_processor = ProcOnTrainSet(self.labels_df, None)
                    labels_processor.load_run_preprocessors(dir_preprocessors, 'preprocessors_for_labels.pkl')
                    self.labels_df = labels_processor._df
                else:
                    raise FileNotFoundError(f"Processor files for labels are not found in {dir_preprocessors}")
            else:
                labels_processor = ProcOnTrainSet(self.labels_df, sample_ind_for_proc)
                labels_processor.pr_impute(strategy="mean")
                # labels_processor.pr_var(threshold=const.default.variance_threshold)
                # labels_processor.pr_minmax()
                labels_processor.pr_zscore()
                labels_processor.save_preprocessors(dir_preprocessors, 'preprocessors_for_labels.pkl')
                self.labels_df = labels_processor._df
                self.label_data = self.labels_df.drop(const.dkey.id).to_numpy().astype(np.float32)


class OptimizeLitdataNCV:
    def __init__(
            self,
            paths_omics: Dict[str, str],
            path_label: Optional[str],
            output_dir: str,
            k_outer: int,
            k_inner: int,
            fragment_elem_ids: Optional[List[List[int]]] = None,
            which_outer_inner: Optional[List[int]] = None,
            col2use_in_labels: Optional[Union[List[str], List[int]]] = None,
            prepr_labels: bool = True,
            prepr_omics: bool = True,
            seed_permut: int = const.default.seed_1,
            seed_resample: int = const.default.seed_2,
            compression: Optional[str] = const.default.compression_alg,
            n_workers: int = const.default.n_workers_litdata,
            variance_threshold: float = const.default.variance_threshold,
            save_n_feat: int = const.default.n_feat2save,
            n_estimators: int = const.default.n_estimators,
            random_states: List[int] = const.default.random_states,
            n_jobs_rf: int = const.default.n_jobs_rf,
        ):
        r"""Optimize the data for nested cross validation.

        Args:
            paths_omics: The ``Dict`` ``{name: path}`` of paths to multiple ``.csv`` or ``.parquet`` files or directories. For genotypes, it is a path to a ``.pkl.gz`` file.
                If some paths are directories, the files inside will be read as existing data.
            
            path_label: The path to label data (a ``.csv`` or ``.parquet`` file).
                If it is `None`, no labels(phenotypes) are provided.
            
            output_dir: The directory to save optimized data.

            k_outer: Number of outer folds.

            k_inner: Number of inner folds.

            fragment_elem_ids: List of list of indices of elements in each fragment.
                It is optional when the data is already split into fragments.
                It overrides ``seed_permut` and disables random permutation.
                For nested cross validation with 10 outer folds and 5 inner folds, it is a list of 50 lists of indices.
            
            which_outer_inner: If specified, only the specified outer-inner fold will be optimized.

            col2use_in_labels: The columns to use in label data.
                If it is `List[int]`, its numbers are the indices **(1-based)** of the columns to be used.
                If it is `None`, all columns are used.

            prepr_labels: Whether to preprocess labels.
                Default: ``True``.

            prepr_omics: Whether to preprocess omics.
                Default: ``True``.
            
            seed_permut: Seed for permutation.
                Default: ``const.default.seed_1``.
            
            seed_resample: The random seed for sampling new samples for the target number of samples.
                Default: ``const.default.seed_2``.
            
            compression: Compression method.
                Default: ``const.default.compression_alg``.
            
            n_workers: Number of workers.
                Default: ``const.default.n_workers_litdata``.

            variance_threshold: The threshold for variance selection.
                Default: ``const.default.variance_threshold``.

            save_n_feat: The number of features to save when random forest selection is performed.
                Default: ``const.default.n_feat2save``.
            
            n_estimators: The number of trees in the random forest.
                Default: ``const.default.n_estimators``.
            
            random_states: The random states for random forest selection.
                Default: ``const.default.random_states``.
            
            n_jobs_rf: The number of jobs for parallelized random forest fitting.
                Default: ``const.default.n_jobs_rf``.
        
        Usage:

            >>> from biodem.utils.data_ncv import OptimizeLitdataNCV
            >>> prep_data_ncv = OptimizeLitdataNCV(...)
            >>> prep_data_ncv.run_optimization()
        
        """
        self.paths_omics = paths_omics
        self.path_label = path_label
        self.output_dir = output_dir
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.col2use_in_labels = col2use_in_labels
        self.prepr_labels = prepr_labels
        self.prepr_omics = prepr_omics

        self.variance_threshold = variance_threshold
        self.n_feat2save = save_n_feat
        self.n_estimators = n_estimators
        self.random_states = random_states
        self.n_jobs_rf = n_jobs_rf

        self.seed_resample = seed_resample
        self.compression = compression
        self.n_workers = n_workers
        self.which_outer_inner = which_outer_inner
        if which_outer_inner is not None:
            if len(which_outer_inner) != 2:
                raise ValueError("which_outer_inner must be a list of two elements")
            self.which_outer = which_outer_inner[0]
            self.which_inner = which_outer_inner[1]
        else:
            self.which_outer = None
            self.which_inner = None
        self.fragments = self._check(seed_permut, fragment_elem_ids)
        self.n_fragments = len(self.fragments)

        self.litdata_cache_dir = os.path.join(output_dir, f".cache_{random_string(9)}")
        os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = self.litdata_cache_dir
        # self.run_optimization()

    def run_optimization(self):
        if self.which_outer_inner is None:
            combn_outer_inner = list(itertools.product(range(self.k_outer), range(self.k_inner)))
            for xo, xi in combn_outer_inner:
                self.optimize_xoxi(xo, xi)
        else:
            self.optimize_xoxi(*self.which_outer_inner)
        
        if self.path_label is not None:
            _, dim_model_output, _ = read_labels(self.path_label, self.col2use_in_labels)
            df_output_dim = pl.DataFrame(data={const.dkey.model_output_dim: [dim_model_output]})
            path_output_dim = os.path.join(self.output_dir, const.fname.output_dim)
            if not os.path.exists(path_output_dim):
                df_output_dim.write_csv(path_output_dim)
        
        # Check if Genomic Variants are available in paths_omics
        for px in self.paths_omics.values():
            if px.endswith(".pkl.gz"):
                path_cp_pklgz = os.path.join(self.output_dir, const.fname.genotypes)
                if not os.path.exists(path_cp_pklgz):
                    shutil.copy(px, path_cp_pklgz)

        # Remove cache dir
        shutil.rmtree(self.litdata_cache_dir)
        return None

    def _check(self, seed_permut: int, fragment_elem_ids: Optional[List[List[int]]]):
        """
        Check if fragment_elem_ids is provided or not. If not, generate random fragments.
        """
        if fragment_elem_ids is None:
            n_fragments = int(self.k_outer * self.k_inner)
            tmp_init = DEMDataset(
                reproduction_mode=False,
                paths_omics=self.paths_omics,
                path_label=self.path_label,
                col2use_in_label=self.col2use_in_labels,
                sample_ind_for_preproc=None,
                dir_preprocessors=None,
                target_n_samples=None,
                seed_resample=self.seed_resample,
                n_fragments=n_fragments,
                prepr_labels=False,
                prepr_omics=False,
                which_outer_test=self.which_outer,
                which_inner_val=self.which_inner,
                variance_threshold=self.variance_threshold,
                save_n_feat=self.n_feat2save,
                n_estimators=self.n_estimators,
                random_states=self.random_states,
                n_jobs_rf=self.n_jobs_rf,
            )
            tmp_init._setup()

            np.random.seed(seed_permut)
            _indices = np.random.permutation(len(tmp_init))
            fragments = np.array_split(_indices, n_fragments)
            fragments = [i.tolist() for i in fragments]
        else:
            fragments = fragment_elem_ids
            n_fragments = len(fragments)
            assert n_fragments == self.k_outer * self.k_inner
        
        return fragments
    
    def optimize_xoxi(self, which_outer_test: int, which_inner_val: int):
        dir_xoxi = os.path.join(self.output_dir, f"ncv_test_{which_outer_test}_val_{which_inner_val}")
        os.makedirs(dir_xoxi, exist_ok=True)
        fr_indices_trn, fr_indices_val, fr_indices_test = get_indices_ncv(self.k_outer, self.k_inner, which_outer_test, which_inner_val)
        ind_trn = np.concatenate([self.fragments[i] for i in fr_indices_trn]).tolist()
        ind_val = np.concatenate([self.fragments[i] for i in fr_indices_val]).tolist()
        ind_tst = np.concatenate([self.fragments[i] for i in fr_indices_test]).tolist()

        dataset_xoxi = DEMDataset(
            reproduction_mode=False,
            paths_omics=self.paths_omics,
            path_label=self.path_label,
            col2use_in_label=self.col2use_in_labels,
            sample_ind_for_preproc=ind_trn,
            dir_preprocessors=dir_xoxi,
            target_n_samples=None,
            seed_resample=self.seed_resample,
            n_fragments=self.n_fragments,
            prepr_labels=self.prepr_labels,
            prepr_omics=self.prepr_omics,
            which_outer_test=which_outer_test,
            which_inner_val=which_inner_val,
            variance_threshold=self.variance_threshold,
            save_n_feat=self.n_feat2save,
            n_estimators=self.n_estimators,
            random_states=self.random_states,
            n_jobs_rf=self.n_jobs_rf,
        )
        dataset_xoxi._setup()
        sample_ids = dataset_xoxi.sample_ids
        if dataset_xoxi.indices_trn is not None:
            ind_trn = dataset_xoxi.indices_trn
        if dataset_xoxi.indices_val is not None:
            ind_val = dataset_xoxi.indices_val
        if dataset_xoxi.indices_tst is not None:
            ind_tst = dataset_xoxi.indices_tst

        # Write sample IDs
        _df_ids = pl.DataFrame(sample_ids, schema=[const.dkey.id])
        _df_ids.write_csv(os.path.join(dir_xoxi, const.fname.predata_ids))
        _df_ids_trn = pl.DataFrame([sample_ids[i] for i in ind_trn], schema=[const.dkey.id])
        _df_ids_trn.write_csv(os.path.join(dir_xoxi, const.fname.predata_ids_trn))
        _df_ids_val = pl.DataFrame([sample_ids[i] for i in ind_val], schema=[const.dkey.id])
        _df_ids_val.write_csv(os.path.join(dir_xoxi, const.fname.predata_ids_val))
        _df_ids_tst = pl.DataFrame([sample_ids[i] for i in ind_tst], schema=[const.dkey.id])
        _df_ids_tst.write_csv(os.path.join(dir_xoxi, const.fname.predata_ids_tst))
        
        # Start optimizing
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_trn,
            output_dir = os.path.join(dir_xoxi, const.title_train),
            chunk_bytes = const.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_val,
            output_dir = os.path.join(dir_xoxi, const.title_val),
            chunk_bytes = const.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_tst,
            output_dir = os.path.join(dir_xoxi, const.title_test),
            chunk_bytes = const.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )


def optimize_data_external(
        output_dir: str,
        paths_omics: Dict[str, str],
        path_label: Optional[str] = None,
        col2use_in_labels: Optional[Union[List[str], List[int]]] = None,
        prepr_labels: bool = True,
        prepr_omics: bool = True,
        reproduction_mode: bool = False,
        dir_preprocessors: Optional[str] = None,
        compression: Optional[str] = const.default.compression_alg,
        n_workers: int = const.default.n_workers_litdata,
        chunk_bytes: str = const.default.chunk_bytes,
    ):
    r"""Optimize data for external data.

    Args:
        output_dir: The directory to save optimized data.
        
        paths_omics: The ``Dict`` ``{name: path}`` of paths to multiple ``.csv`` or ``.parquet`` files or directories. For genotypes, it is a path to a ``.pkl.gz`` file.
            If some paths are directories, the files inside will be read as existing data.
        
        path_label: The path to label data (a ``.csv`` or ``.parquet`` file).
            If it is `None`, no labels(phenotypes) are provided.
        
        col2use_in_labels: The columns to use in label data.
            If it is `List[int]`, its numbers are the indices **(1-based)** of the columns to be used.
            If it is `None`, all columns are used.
        
        prepr_labels: Whether to preprocess labels.
            Default: ``True``.

        prepr_omics: Whether to preprocess omics.
            Default: ``True``.

        reproduction_mode: Whether to use existing processors.
            Please provide ``dir_preprocessors`` if ``reproduction_mode`` is ``True``.

        dir_preprocessors: The directory used to save data processors that fitted on training data.
            If it is `None`, preprocessing is not performed.

        compression: Compression method.
            Default: ``const.default.compression_alg``.
        
        n_workers: Number of workers.
            Default: ``const.default.n_workers_litdata``.
        
        chunk_bytes: Chunk size.
            Default: ``const.default.chunk_bytes``.
    
    """
    dataset_ext = DEMDataset(
        reproduction_mode=reproduction_mode,
        paths_omics=paths_omics,
        path_label=path_label,
        col2use_in_label=col2use_in_labels,
        dir_preprocessors=dir_preprocessors,
        prepr_labels=prepr_labels,
        prepr_omics=prepr_omics,
    )
    dataset_ext._setup()
    optimize(
        fn = dataset_ext.__getitem__,
        inputs = range(len(dataset_ext)),
        output_dir = output_dir,
        chunk_bytes = chunk_bytes,
        compression = compression,
        num_workers = n_workers,
    )


class DEMDataModule4Train(LightningDataModule):
    def __init__(
            self,
            litdata_dir: str,
            which_outer_testset: int,
            which_inner_valset: int,
            batch_size: int,
            n_workers: int = const.default.n_workers,
        ):
        r"""LightningDataModule for training.

        Args:
            litdata_dir: Directory containing the LitData for nested cross-validation.

            which_outer_testset: Index of the outer test set fold.

            which_inner_valset: Index of the inner validation set fold.

            batch_size: Batch size for dataloader.

            n_workers: Number of workers for dataloader.
        
        """
        super().__init__()
        self.litdata_dir = litdata_dir
        self.which_outer_testset = which_outer_testset
        self.which_inner_valset = which_inner_valset
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage=None):
        self.dataloder_trn, self.dataloader_val, self.dataloader_test = self.read_litdata_ncv()
    
    def train_dataloader(self):
        return self.dataloder_trn
    def val_dataloader(self):
        return self.dataloader_val
    def test_dataloader(self):
        return self.dataloader_test
    
    def read_litdata_ncv(self):
        """
        Read litdata from directories and return dataloaders for NCV.
        """
        dir_train, dir_valid, dir_test = self.get_dir_ncv_litdata()
        dataloader_train = StreamingDataLoader(StreamingDataset(dir_train), batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
        dataloader_valid = StreamingDataLoader(StreamingDataset(dir_valid), batch_size=self.batch_size, num_workers=self.n_workers)
        dataloader_test = StreamingDataLoader(StreamingDataset(dir_test), batch_size=self.batch_size, num_workers=self.n_workers)
        return dataloader_train, dataloader_valid, dataloader_test
    
    def get_dir_ncv_litdata(self):
        self.dir_xoi = os.path.join(self.litdata_dir, f"ncv_test_{self.which_outer_testset}_val_{self.which_inner_valset}")
        dir_trn = os.path.join(self.dir_xoi, const.title_train)
        dir_val = os.path.join(self.dir_xoi, const.title_val)
        dir_tst = os.path.join(self.dir_xoi, const.title_test)
        return dir_trn, dir_val, dir_tst
    
    def read_omics_dims(self):
        return pl.read_csv(os.path.join(self.dir_xoi, const.fname.predata_omics_dims)).select(const.dkey.omics_dim).to_series().to_list()


class Dict2Dataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any]):
        r"""Transform a dict of data to a Dataset for shuffled omics feature values.

        Args:
            data_dict: dict of data
        
        """
        super().__init__()
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict[const.dkey.litdata_index])
    
    def __getitem__(self, idx: int):
        _omics = [om_x[idx, :] for om_x in self.data_dict[const.dkey.litdata_omics]]
        _id = self.data_dict[const.dkey.litdata_id][idx]
        _idx = idx
        if const.dkey.litdata_label in self.data_dict:
            _labels = self.data_dict[const.dkey.litdata_label][idx]
            return {const.dkey.litdata_index: _idx, const.dkey.litdata_omics: _omics, const.dkey.litdata_id: _id, const.dkey.litdata_label: _labels}
        else:
            return {const.dkey.litdata_index: _idx, const.dkey.litdata_omics: _omics, const.dkey.litdata_id: _id}


class DEMDataModule4Uni(LightningDataModule):
    def __init__(
            self,
            litdata_dir: str,
            batch_size: int = const.default.batch_size,
            n_workers: int = const.default.n_workers,
        ):
        r"""LightningDataModule for prediction.

        Args:
            litdata_dir: Directory containing the LitData for prediction.

            batch_size: Batch size for prediction.

            n_workers: Number of workers for dataloader.
        
        """
        super().__init__()
        self.litdata_dir = litdata_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
    
    def setup(self, stage=None):
        self._dataset = StreamingDataset(self.litdata_dir)
        print(f"Length of dataset: {len(self._dataset)}")
        self._dataloader_pred = StreamingDataLoader(self._dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        self._dataloader_test = StreamingDataLoader(self._dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
    def predict_dataloader(self):
        return self._dataloader_pred

    def test_dataloader(self):
        return self._dataloader_test
    
    def shuffle_a_feat(
            self,
            which_omics: Union[int, str],
            which_feature: int,
            random_state: int,
            save_litdata: bool = False,
            chunk_bytes: str = const.default.chunk_bytes,
            compression: str = const.default.compression_alg,
        ) -> str | DataLoader:
        r"""
        Shuffle one feature in one omics data.

        Args:
            which_omics: The index or name of the omics data to shuffle.

            which_feature: The index of the feature to shuffle.

            random_state: The random seed for reproducibility.
        
        """
        if not hasattr(self, 'data_all'):
            self.data_all = self.read_dataloader()
        
        if not hasattr(self, 'omics_names'):
            self.omics_names = read_omics_names(self.litdata_dir)
        
        if isinstance(which_omics, str):
            which_om = self.omics_names.index(which_omics)
        else:
            if (which_omics + 1) > len(self.omics_names):
                raise ValueError("The specified omics index is out of range.")
            which_om = which_omics
        
        """
        Shuffle a feature's values
        """
        data_all = deepcopy(self.data_all)
        _tmp: np.ndarray = data_all[const.dkey.litdata_omics][which_om]

        np.random.seed(random_state)
        _tmp[:, which_feature] = np.random.permutation(_tmp[:, which_feature])
        data_all[const.dkey.litdata_omics][which_om] = _tmp
        
        """
        Create new dataset and save as litdata
        """
        _dataset = Dict2Dataset(data_all)
        if save_litdata:
            output_dir = self.litdata_dir + f"_shuffle_om+{which_om}_feat+{which_feature}_rand+{random_state}"
            optimize(
                fn = _dataset.__getitem__,
                inputs = range(len(_dataset)),
                output_dir = output_dir,
                chunk_bytes = chunk_bytes,
                compression = compression,
            )
            return output_dir
        else:
            return DataLoader(_dataset, self.batch_size, num_workers=self.n_workers)
    
    def read_dataloader(self) -> Dict[str, Any]:
        if not hasattr(self, "_dataloader_pred"):
            self.setup()
        
        """
        Read data from dataloader.
        """
        data_all = {}
        for batch in self.predict_dataloader():
            for xkey in batch.keys():
                if xkey not in data_all:
                    data_all[xkey] = batch[xkey]
                else:
                    if xkey == const.dkey.litdata_omics:
                        data_all[xkey] = [np.concatenate([data_all[xkey][i], batch[xkey][i]], axis=0) for i in range(len(data_all[xkey]))]
                        # for i in range(len(data_all[xkey])):
                        #     print(f"shape of {xkey}: {data_all[xkey][i].shape}, type: {type(data_all[xkey][i])}")
                    else:
                        data_all[xkey] = np.concatenate((data_all[xkey], batch[xkey]), axis=0)
                        # print(f"shape of {xkey}: {data_all[xkey].shape}, type: {type(data_all[xkey])}")
        return data_all


def read_omics_names(litdata_dir: str, get_path: bool = False):
    r"""Read omics names from the parent directory of the litdata.
    
    Args:
        litdata_dir: Path to the litdata directory.
    
    Output:
        A sorted list of omics names.
    
    """
    _dir_trnvaltst = os.path.dirname(litdata_dir)
    omics_names: List[str] = []
    omics_paths: List[str] = []
    for _fname in os.listdir(_dir_trnvaltst):
        if _fname.startswith(const.fname.predata_omics_features_prefix):
            _tmp = os.path.splitext(_fname)[0]
            _tmp = _tmp.removeprefix(const.fname.predata_omics_features_prefix)
            _tmp = _tmp.removeprefix("_")
            omics_names.append(_tmp)
            omics_paths.append(os.path.join(_dir_trnvaltst, _fname))
    if len(omics_names) == 0:
        raise ValueError("No omics data found in the specified directory.")

    sortperm = np.argsort(omics_names)
    omics_names = np.take(omics_names, sortperm).tolist()
    omics_paths = np.take(omics_paths, sortperm).tolist()
    
    if get_path:
        return omics_names, omics_paths
    else:
        return omics_names
