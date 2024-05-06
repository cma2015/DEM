r"""
Convert SNPs to genes using SNP2Gene model.
"""

from .model_snp2gene import SNPDataModule, train_snp2gene, snp_to_gene
from .utils import read_processed_data, gen_ncv_pheno_filenames, collect_models_paths, random_string
import os
import time
import pandas as pd
import optuna


class SNP2GeneTrain:
    r"""
    SNP2Gene model training with hyperparameter optimization.
    """
    def __init__(
            self,
            log_dir: str,
            log_name: str,
            trait_name: str,
            n_label_class: int,
            path_trn_label: str,
            path_val_label: str,
            path_h5_processed: str,
            path_json_genes_snps: str,
            dense_layers_hidden_dims: list[int],
            len_one_hot_vec: int = 10,
            devices: list[int] | str | int = 'auto',
            accelerator: str = 'auto',
            n_jobs: int = 1,
            learning_rate: float = 1e-3,
            patience: int = 20,
            max_epochs: int = 1000,
            min_epochs: int = 20,
            batch_size: int = 16,
        ):
        """
        Initialize SNP2GeneTrain class.
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.len_one_hot_vec = len_one_hot_vec

        snp_data_dict = read_processed_data(
            path_h5_processed=path_h5_processed,
            path_json_genes_snps=path_json_genes_snps,
        )

        self.io_dict = self.train_io_dict(
            trait_name=trait_name,
            n_label_class=n_label_class,
            path_trn_label=path_trn_label,
            path_val_label=path_val_label,
            data_dict=snp_data_dict,
            len_one_hot_vec=len_one_hot_vec,
        )

        self.hparams = self.train_hparams(
            learning_rate=learning_rate,
            patience=patience,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            batch_size=batch_size,
            dense_layers_hidden_dims=dense_layers_hidden_dims,
        )

    def train_io_dict(
            self,
            trait_name: str,
            n_label_class: int,
            path_trn_label: str,
            path_val_label: str,
            data_dict: dict,
            len_one_hot_vec: int,
        ):
        """
        Generate a dictionary of input/output data for SNP2Gene model training.
        """
        io_dict = {
            'trait_name': trait_name,
            'n_label_class': n_label_class,
            'path_trn_label': path_trn_label,
            'path_val_label': path_val_label,
            'data_dict': data_dict,
            'len_one_hot_vec': len_one_hot_vec,
        }
        return io_dict
    
    def train_hparams(
            self,
            learning_rate: float,
            patience: int,
            max_epochs: int,
            min_epochs: int,
            batch_size: int,
            dense_layers_hidden_dims: list[int],
        ):
        """
        Generate a dictionary of hyperparameters for SNP2Gene model training.
        """
        hparams = {
            'learning_rate': learning_rate,
            'patience': patience,
            'max_epochs': max_epochs,
            'min_epochs': min_epochs,
            'batch_size': batch_size,
            'dense_layers_hidden_dims': dense_layers_hidden_dims,
        }
        return hparams

    def snp2gene_train(
            self,
            hparams: dict,
            io_dict: dict,
            devices: list[int] | str | int,
            accelerator: str,
        ):
        """
        Train SNP2Gene model with given hyperparameters and input/output data.
        """
        datamodule = SNPDataModule(
            snp_matrix=io_dict['data_dict']['snp_matrix'],
            sample_ids_in_mat=io_dict['data_dict']['sample_ids'],
            path_csv_pheno_trn=io_dict['path_trn_label'],
            path_csv_pheno_val=io_dict['path_val_label'],
            batch_size=hparams['batch_size'],
            len_one_hot_vec=io_dict['len_one_hot_vec'],
            which_trait=io_dict['trait_name'],
            n_pheno_categories=io_dict['n_label_class'],
        )
        datamodule.setup()

        val_loss_min = train_snp2gene(
            data_module=datamodule,
            genes_snps=io_dict['data_dict']['genes_snps'],
            dense_layers_hidden_dims=hparams['dense_layers_hidden_dims'],
            es_patience=hparams['patience'],
            learning_rate=hparams['learning_rate'],
            max_epochs=hparams['max_epochs'],
            min_epochs=hparams['min_epochs'],
            log_dir=self.log_dir,
            log_name=self.log_name,
            devices=devices,
            accelerator=accelerator,
        )
        return val_loss_min
    
    def manual_train(self):
        """
        Train SNP2Gene model with manually set hyperparameters.
        """
        val_loss_min = self.snp2gene_train(
            hparams=self.hparams,
            io_dict=self.io_dict,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        return val_loss_min
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for SNP2Gene model hyperparameter optimization.
        """
        print('Trial number:', trial.number)
        if self.n_jobs > 1:
            time_delay = (trial.number + self.n_jobs) % self.n_jobs * 11.7
            time.sleep(time_delay)
        
        lr = trial.suggest_categorical('lr', [1e-3, 1e-4, 1e-5, 1e-6])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        hparams_trial = self.hparams.copy()
        hparams_trial['learning_rate'] = lr
        hparams_trial['batch_size'] = batch_size
                
        val_loss_min = self.snp2gene_train(
            hparams = hparams_trial,
            io_dict = self.io_dict,
            devices = self.devices,
            accelerator=self.accelerator,
        )
        
        return val_loss_min
    
    def optimize(
            self,
            n_trials: int | None = None,
            storage: str = 'sqlite:///optuna_snp2gene.db',
        ):
        """
        Optimize hyperparameters of SNP2Gene model.
        """
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())

        study = optuna.create_study(
            storage = storage,
            study_name = self.log_name + '_' + time_str,
            load_if_exists = True,
            direction = 'minimize',
        )
        study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=n_trials, gc_after_trial=True)


class SNP2GenePipeline:
    r"""
    SNP2Gene model pipeline.
    Hyperparameters are optimized for each fold in nested cross-validation.
    The best SNP2Gene model for each fold is used to convert SNPs to genes.
    """
    def __init__(
            self,
            trait_name: str,
            n_label_class: int,
            dir_label_inner: str,
            path_h5_processed: str,
            path_json_genes_snps: str,
            log_folder: str,
            dir_to_save_converted: str,
            dir_label_outer: str | None = None,
            len_onehot_snp: int = 10,
            dense_layers_hidden_dims: list[int] = [1024, 256, 64],
            devices: list[int] | str | int = 'auto',
            accelerator: str = 'auto',
            n_jobs: int = 1,
            n_trials: int | None = 16,
            outer_sta_idx: int | None = None,
            outer_end_idx: int | None = None,
            inner_sta_idx: int | None = None,
            inner_end_idx: int | None = None,
        ):
        """
        Initialize SNP2Gene pipeline.

        Parameters:

        - `path_h5_processed`: Path to the processed data in h5 format.
        - `path_json_genes_snps`: Path to the JSON file that defines the SNP-gene relation.
        - `len_onehot_snp`: Length of the one-hot vector for each SNP.
        - `dense_layers_hidden_dims`: List of hidden dimensions for the dense layers.
        """
        self.log_dir = os.path.join(log_folder, 'models_' + trait_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.dir_to_save_converted = dir_to_save_converted
        if not os.path.exists(self.dir_to_save_converted):
            os.makedirs(self.dir_to_save_converted)

        self.trait_name = trait_name
        self.n_label_class = n_label_class
        self.dir_label_inner = dir_label_inner
        self.path_h5_processed = path_h5_processed
        self.path_json_genes_snps = path_json_genes_snps
        
        self.len_onehot_snp = len_onehot_snp
        self.dense_layers_hidden_dims = dense_layers_hidden_dims
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.n_trials = n_trials

        if dir_label_outer is not None:
            # The outer test data is provided.
            self.dir_label_outer = dir_label_outer
        
        # Scan number of outer folds for cross-validation according to the folders in `dir_label_inner`
        n_outer_folds = len(os.listdir(self.dir_label_inner))
        # Scan number of inner folds for cross-validation according to the CSV files in each folder in `dir_label_inner`
        n_inner_folds = len(os.listdir(os.path.join(self.dir_label_inner, os.listdir(self.dir_label_inner)[0]))) // 2

        if inner_sta_idx is not None:
            if inner_end_idx is not None:
                self.inner_fold_indices = list(range(inner_sta_idx, inner_end_idx))
            else:
                self.inner_fold_indices = list(range(inner_sta_idx, n_inner_folds))
        else:
            if inner_end_idx is not None:
                self.inner_fold_indices = list(range(min(n_inner_folds, inner_end_idx)))
            else:
                self.inner_fold_indices = list(range(n_inner_folds))

        if outer_sta_idx is not None:
            if outer_end_idx is not None:
                self.outer_fold_indices = list(range(outer_sta_idx, outer_end_idx))
            else:
                self.outer_fold_indices = list(range(outer_sta_idx, n_outer_folds))
        else:
            if outer_end_idx is not None:
                self.outer_fold_indices = list(range(min(n_outer_folds, outer_end_idx)))
            else:
                self.outer_fold_indices = list(range(n_outer_folds))
        
        self.random_str = random_string()
    
    def train_pipeline(self):
        """
        Train SNP2Gene model for each fold in nested cross-validation.
        """
        # Storage for optuna trials in self.log_dir
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        path_storage = 'sqlite:///' + self.log_dir + '/optuna_s2g_' + time_str + '_rs' + self.random_str + '.db'
        
        log_names = []
        paths_csv_pheno_trn = []
        paths_csv_pheno_val = []
        for i in self.outer_fold_indices:
            for j in self.inner_fold_indices:
                log_names.append(f'run_ncv_{i}_{j}')
                tmp_path_trn_label, tmp_path_val_label = gen_ncv_pheno_filenames(i, j, self.dir_label_inner)
                paths_csv_pheno_trn.append(tmp_path_trn_label)
                paths_csv_pheno_val.append(tmp_path_val_label)

        # Train SNP2Gene model for each fold in nested cross-validation
        for xfold in range(len(log_names)):
            snp2gene_train_x = SNP2GeneTrain(
                log_dir = self.log_dir,
                log_name = log_names[xfold],
                trait_name = self.trait_name,
                n_label_class = self.n_label_class,
                path_trn_label = paths_csv_pheno_trn[xfold],
                path_val_label = paths_csv_pheno_val[xfold],
                path_h5_processed = self.path_h5_processed,
                path_json_genes_snps = self.path_json_genes_snps,
                dense_layers_hidden_dims = self.dense_layers_hidden_dims,
                len_one_hot_vec = self.len_onehot_snp,
                devices = self.devices,
                accelerator=self.accelerator,
                n_jobs=self.n_jobs,
            )

            snp2gene_train_x.optimize(n_trials=self.n_trials, storage=path_storage)

    def collect_trained_models(self):
        """
        Collect trained SNP2Gene models for each fold in nested cross-validation.
        """
        models_dict = collect_models_paths(self.log_dir)

        key_best_versions = 'best_versions'
        self.models_bv = models_dict[key_best_versions]
        self.models_bv.to_csv(os.path.join(self.dir_to_save_converted, 'models_best_versions' + '_rs' + self.random_str + '.csv'))

        key_best_inner_folds = 'best_inner_folds'
        self.models_bi = models_dict[key_best_inner_folds]
        self.models_bi.to_csv(os.path.join(self.dir_to_save_converted, 'models_best_inner_folds' + '_rs' + self.random_str + '.csv'))

    def convert_snp(self, batch_size: int = 32):
        """
        Convert SNPs to genes using the best SNP2Gene model for each fold in nested cross-validation.
        """
        if not hasattr(self,'models_bv'):
            self.collect_trained_models()

        # For each inner fold
        dir_to_save_converted_inner = os.path.join(self.dir_to_save_converted, 'inner')
        if not os.path.exists(dir_to_save_converted_inner):
            os.makedirs(dir_to_save_converted_inner)

        for wh_mdl in range(self.models_bv.shape[0]):
            path_mdl = self.models_bv.iloc[wh_mdl]['path_ckpt']
            which_outer_fold = self.models_bv.iloc[wh_mdl]['ncv_outer_x']
            which_inner_fold = self.models_bv.iloc[wh_mdl]['ncv_inner_x']

            if which_outer_fold not in self.outer_fold_indices or which_inner_fold not in self.inner_fold_indices:
                continue

            path_trn_label, path_val_label = gen_ncv_pheno_filenames(which_outer_fold, which_inner_fold, self.dir_label_inner)
            dir_to_save_converted_inner_fold = os.path.join(dir_to_save_converted_inner, str(which_outer_fold) + 'fold')
            if not os.path.exists(dir_to_save_converted_inner_fold):
                os.makedirs(dir_to_save_converted_inner_fold)

            # For training data
            genes_trn = snp_to_gene(
                path_pretrained_model=path_mdl,
                path_h5_processed=self.path_h5_processed,
                path_json_genes_snps=self.path_json_genes_snps,
                path_csv_pheno_test=path_trn_label,
                dir4predictions=dir_to_save_converted_inner,
                len_one_hot_vec=self.len_onehot_snp,
                batch_size=batch_size,
                which_trait=self.trait_name,
                accelerator=self.accelerator,
            )
            path_genes_trn = os.path.join(
                dir_to_save_converted_inner_fold,
                str(which_inner_fold) + '_snp_inner_tr.csv',
            )
            genes_trn.to_csv(path_genes_trn)

            # For validation data
            genes_val = snp_to_gene(
                path_pretrained_model=path_mdl,
                path_h5_processed=self.path_h5_processed,
                path_json_genes_snps=self.path_json_genes_snps,
                path_csv_pheno_test=path_val_label,
                dir4predictions=dir_to_save_converted_inner,
                len_one_hot_vec=self.len_onehot_snp,
                batch_size=batch_size,
                which_trait=self.trait_name,
                accelerator=self.accelerator,
            )
            path_genes_val = os.path.join(
                dir_to_save_converted_inner_fold,
                str(which_inner_fold) + '_snp_inner_te.csv',
            )
            genes_val.to_csv(path_genes_val)

        # For each outer fold
        if hasattr(self, 'dir_label_outer'):
            dir_to_save_converted_outer = os.path.join(self.dir_to_save_converted, 'outer')
            if not os.path.exists(dir_to_save_converted_outer):
                os.makedirs(dir_to_save_converted_outer)

            for wh_mdl in range(self.models_bi.shape[0]):
                path_mdl = self.models_bi.iloc[wh_mdl]['path_ckpt']
                which_outer_fold = self.models_bi.iloc[wh_mdl]['ncv_outer_x']

                if which_outer_fold not in self.outer_fold_indices:
                    continue

                path_tst_label = os.path.join(self.dir_label_outer, str(which_outer_fold) + '_outer_zscore_labels_te.csv')
                path_trn_label = os.path.join(self.dir_label_outer, str(which_outer_fold) + '_outer_zscore_labels_tr.csv')

                # For training data
                genes_trn = snp_to_gene(
                    path_pretrained_model=path_mdl,
                    path_h5_processed=self.path_h5_processed,
                    path_json_genes_snps=self.path_json_genes_snps,
                    path_csv_pheno_test=path_trn_label,
                    dir4predictions=dir_to_save_converted_outer,
                    len_one_hot_vec=self.len_onehot_snp,
                    batch_size=batch_size,
                    which_trait=self.trait_name,
                    accelerator=self.accelerator,
                )
                path_genes_trn = os.path.join(
                    dir_to_save_converted_outer,
                    str(which_outer_fold) + '_snp_outer_tr.csv',
                )
                genes_trn.to_csv(path_genes_trn)

                # For test data
                genes_tst = snp_to_gene(
                    path_pretrained_model=path_mdl,
                    path_h5_processed=self.path_h5_processed,
                    path_json_genes_snps=self.path_json_genes_snps,
                    path_csv_pheno_test=path_tst_label,
                    dir4predictions=dir_to_save_converted_outer,
                    len_one_hot_vec=self.len_onehot_snp,
                    batch_size=batch_size,
                    which_trait=self.trait_name,
                    accelerator=self.accelerator,
                )
                path_genes_tst = os.path.join(
                    dir_to_save_converted_outer,
                    str(which_outer_fold) + '_snp_outer_te.csv',
                )
                genes_tst.to_csv(path_genes_tst)
