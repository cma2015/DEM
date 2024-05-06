r"""
DEM model training and hyperparameter optimization functions.
"""
# import torch
import lightning as ltn
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
import numpy as np
import pandas as pd
from torch.cuda import device_count
from .model_dem import DEMLTNDataModule, train_dem, DEMLTN
from .utils import gen_ncv_omics_filepaths, gen_ncv_pheno_filenames, collect_models_paths, random_string
import os
import time
import optuna


class DEMTrain:
    r"""
    DEM model training with hyperparameter optimization using Optuna.
    """
    def __init__(
            self,
            log_dir: str,
            log_name: str,
            trait_name: str,
            n_label_class: int,
            paths_trn_omics: list[str],
            paths_val_omics: list[str],
            path_trn_label: str,
            path_val_label: str,
            devices: list[int] | str | int = 'auto',
            n_jobs: int = 1,
            n_heads: int = 4,
            n_encoders: int = 2,
            hidden_dim: int = 1024,
            learning_rate: float = 1e-5,
            dropout: float = 0.25,
            patience: int = 20,
            max_epochs: int = 1000,
            min_epochs: int = 20,
            batch_size: int = 16,
        ):
        """
        Initialize DEM model training with given hyperparameters and input/output paths.
        """
        self.devices = devices
        self.n_jobs = n_jobs
        self.log_dir = log_dir
        self.log_name = log_name

        self.io_dict = self.train_io_dict(
            trait_name=trait_name,
            n_label_class=n_label_class,
            paths_trn_omics=paths_trn_omics,
            paths_val_omics=paths_val_omics,
            path_trn_label=path_trn_label,
            path_val_label=path_val_label,
        )

        self.hparams = self.train_hparams(
            n_heads=n_heads,
            n_encoders=n_encoders,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout=dropout,
            patience=patience,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            batch_size=batch_size,
        )

    def train_io_dict(
            self,
            trait_name: str,
            n_label_class: int,
            paths_trn_omics: list[str],
            paths_val_omics: list[str],
            path_trn_label: str,
            path_val_label: str,
        ):
        """
        Generate a dictionary of input/output paths for DEM model training.
        """
        io_dict = {
            'trait_name': trait_name,
            'n_label_class': n_label_class,
            'paths_trn_omics': paths_trn_omics,
            'paths_val_omics': paths_val_omics,
            'path_trn_label': path_trn_label,
            'path_val_label': path_val_label,
        }
        return io_dict
        
    def train_hparams(
            self,
            n_heads: int,
            n_encoders: int,
            hidden_dim: int,
            learning_rate: float,
            dropout: float,
            patience: int,
            max_epochs: int,
            min_epochs: int,
            batch_size: int,
        ):
        """
        Generate a dictionary of hyperparameters for DEM model training.
        """
        hparams = {
            'n_heads': n_heads,
            'n_encoders': n_encoders,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'patience': patience,
            'max_epochs': max_epochs,
            'min_epochs': min_epochs,
            'batch_size': batch_size,
        }
        return hparams

    def dem_train(
            self,
            hparams: dict,
            io_dict: dict,
            devices: list[int] | str | int,
        ):
        """
        Train DEM model with given hyperparameters and input/output paths.
        """
        datamodule = DEMLTNDataModule(
            batch_size=hparams['batch_size'],
            trait_name=io_dict['trait_name'],
            n_label_classes=io_dict['n_label_class'],
            paths_omics_trn=io_dict['paths_trn_omics'],
            paths_omics_val=io_dict['paths_val_omics'],
            path_label_trn=io_dict['path_trn_label'],
            path_label_val=io_dict['path_val_label'],
        )
        datamodule.setup()

        val_loss_min = train_dem(
            data_module=datamodule,
            n_heads=hparams['n_heads'],
            n_encoders=hparams['n_encoders'],
            hidden_dim=hparams['hidden_dim'],
            learning_rate=hparams['learning_rate'],
            dropout=hparams['dropout'],
            patience=hparams['patience'],
            max_epochs=hparams['max_epochs'],
            min_epochs=hparams['min_epochs'],
            log_dir=self.log_dir,
            log_name=self.log_name,
            devices=devices,
        )
        return val_loss_min

    def manual_train(self):
        """
        Train DEM model with manually specified hyperparameters.
        """
        val_loss_min = self.dem_train(
            hparams = self.hparams,
            io_dict = self.io_dict,
            devices=self.devices,
        )
        return val_loss_min

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for DEM model training with Optuna.
        """
        print("Trial number:", trial.number)
        if self.n_jobs > 1:
            time_delay = (trial.number + self.n_jobs) % self.n_jobs * 11.7
            time.sleep(time_delay)

        # Generate hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        n_encoders = trial.suggest_categorical("n_encoders", [1, 2, 4])
        hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024])
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        lr = trial.suggest_categorical("lr", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])

        # Update hyperparameters in DEMTrain object based on manual parameters in initialization
        hparams_tmp = self.hparams.copy()
        hparams_tmp['batch_size'] = batch_size
        hparams_tmp['n_heads'] = n_heads
        hparams_tmp['n_encoders'] = n_encoders
        hparams_tmp['hidden_dim'] = hidden_dim
        hparams_tmp['learning_rate'] = lr
        hparams_tmp['dropout'] = dropout
        
        val_loss_min = self.dem_train(
            hparams = hparams_tmp,
            io_dict = self.io_dict,
            devices=self.devices,
        )
        
        return val_loss_min

    def optimize(
            self,
            n_trials: int | None = None,
            storage: str = "sqlite:///optuna_dem.db",
        ):
        """
        Optimize hyperparameters of DEM model with Optuna.
        """
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())

        study = optuna.create_study(
            storage = storage,
            study_name = self.log_name + "_" + time_str,
            direction = "minimize",
            load_if_exists = True,
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs, gc_after_trial=True)


class DEMTrainPipeline:
    r"""
    DEM model training pipeline with hyperparameter optimization using Optuna.
    """
    def __init__(
            self,
            log_folder: str,
            trait_name: str,
            n_label_class: int,
            dir_label_inner: str,
            dir_omics_inner: str,
            dir_label_outer: str,
            dir_omics_outer: str,
            devices: list[int] | str | int = 'auto',
            n_jobs: int = 1,
            n_trials: int | None = 16,
            outer_sta_idx: int | None = None,
            outer_end_idx: int | None = None,
            inner_sta_idx: int | None = None,
            inner_end_idx: int | None = None,
        ):
        """
        Initialize DEM model training pipeline.
        """
        self.log_dir = os.path.join(log_folder, 'models_' + trait_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.trait_name = trait_name
        self.n_label_class = n_label_class
        self.dir_label_inner = dir_label_inner
        self.dir_omics_inner = dir_omics_inner
        self.dir_label_outer = dir_label_outer
        self.dir_omics_outer = dir_omics_outer
        self.devices = devices
        self.n_jobs = n_jobs
        self.n_trials = n_trials

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
            
    def train_pipeline(self):
        """
        Train DEM model for each fold in nested cross-validation.
        """
        # Storage for optuna trials in self.log_dir
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        random_str = random_string()
        path_storage = 'sqlite:///' + self.log_dir + '/optuna_dem_' + time_str + '_rs' + random_str + '.db'
        
        log_names = []
        list_paths_trn_omics = []
        list_paths_val_omics = []
        paths_trn_label = []
        paths_val_label = []
        for i in self.outer_fold_indices:
            for j in self.inner_fold_indices:
                log_names.append(f'run_ncv_{i}_{j}')
                tmp_paths_trn_omics, tmp_paths_val_omics, title_omics = gen_ncv_omics_filepaths(self.dir_omics_inner, i, j)
                tmp_path_trn_label, tmp_path_val_label = gen_ncv_pheno_filenames(i, j, self.dir_label_inner)
                list_paths_trn_omics.append(tmp_paths_trn_omics)
                list_paths_val_omics.append(tmp_paths_val_omics)
                paths_trn_label.append(tmp_path_trn_label)
                paths_val_label.append(tmp_path_val_label)
        
        # Train DEM model for each fold in nested cross-validation
        for xfold in range(len(log_names)):
            dem_train_x = DEMTrain(
                log_dir = self.log_dir,
                log_name = log_names[xfold],
                trait_name = self.trait_name,
                n_label_class = self.n_label_class,
                paths_trn_omics = list_paths_trn_omics[xfold],
                paths_val_omics = list_paths_val_omics[xfold],
                path_trn_label = paths_trn_label[xfold],
                path_val_label = paths_val_label[xfold],
                devices = self.devices,
                n_jobs=self.n_jobs,
            )

            dem_train_x.optimize(n_trials=self.n_trials, storage=path_storage)
    
    def collect_models(self):
        """
        Collect trained DEM models for each fold in nested cross-validation.
        """
        models_dict = collect_models_paths(self.log_dir)

        key_best_versions = 'best_versions'
        models_bv = models_dict[key_best_versions]
        models_bv.to_csv(os.path.join(self.log_dir, 'models_best_versions.csv'))

        key_best_inner_folds = 'best_inner_folds'
        models_bi = models_dict[key_best_inner_folds]
        models_bi.to_csv(os.path.join(self.log_dir, 'models_best_inner_folds.csv'))

        return models_dict


class DEMPredict:
    r"""
    DEM model prediction.
    """
    def __init__(
            self,
            model_path: str,
            batch_size: int = 16,
            map_location: str | None = None,
        ):
        """
        Initialize DEM model prediction.
        - map_location: `cuda:0` for GPU, `cpu` for CPU.
        """
        self.batch_size = batch_size

        if map_location is None:
            if device_count() > 0:
                which_dev = find_usable_cuda_devices(1)
                if len(which_dev) > 0:
                    map_location = 'cuda:' + str(which_dev[0])
                else:
                    map_location = 'cpu'
            else:
                map_location = 'cpu'

        # Load DEM model
        self.dem_model = DEMLTN.load_from_checkpoint(
            model_path,
            map_location,
            # os.path.join(os.path.dirname(os.path.dirname(model_path)), 'hparams.yaml'),
        )
        self.dem_model.eval()
        self.dem_model.freeze()

    def predict(self, omics_paths: list[str], result_dir: str):
        """
        Predict phenotypes from omics data using a trained DEM model.
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        datamodule = DEMLTNDataModule(
            batch_size=self.batch_size,
            paths_omics_pred=omics_paths,
        )
        datamodule.setup()

        trainer = ltn.Trainer(default_root_dir=result_dir, logger=False)
        predictions = trainer.predict(self.dem_model, datamodule=datamodule)
        pred_array = np.concatenate(predictions, axis=0)
        
        indices_ = pd.read_csv(omics_paths[0], index_col=0).index
        pred_df = pd.DataFrame(pred_array, index=indices_)
        pred_df.to_csv(os.path.join(result_dir, 'predictions.csv'), index=False)


def predict_pheno(
        model_path: str,
        omics_paths: list[str],
        result_dir: str,
        batch_size: int = 16,
        map_location: str | None = None,
    ):
    """
    Predict phenotypes from omics data using a trained DEM model.
    """
    forpred = DEMPredict(model_path, batch_size, map_location)
    forpred.predict(omics_paths, result_dir)


class DEMFeatureRanking:
    r"""
    DEM model feature ranking.
    """
    def __init__(
            self,
            model_path: str,
            batch_size: int = 16,
            map_location: str | None = None,
        ):
        """
        Initialize DEM model feature ranking.
        - map_location: `cuda:0` for GPU, `cpu` for CPU.
        """
        self.batch_size = batch_size

        if map_location is None:
            if device_count() > 0:
                which_dev = find_usable_cuda_devices(1)
                if len(which_dev) > 0:
                    map_location = 'cuda:' + str(which_dev[0])
                else:
                    map_location = 'cpu'
            else:
                map_location = 'cpu'

        # Load DEM model
        self.dem_model = DEMLTN.load_from_checkpoint(
            model_path,
            map_location,
            # os.path.join(os.path.dirname(os.path.dirname(model_path)), 'hparams.yaml'),
        )
        self.dem_model.eval()
        self.dem_model.freeze()

    def shuffle_features(
            self,
            omics_paths: list[str],
            result_dir: str,
            which_omics2shuffle: int,
            which_feature2shuffle: int,
            random_state: int = 42,
        ):
        """
        Shuffle one feature in one omics data and save the shuffled data.
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        tmp_omics = pd.read_csv(omics_paths[which_omics2shuffle], index_col=0)

        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Shuffle one feature
        tmp_omics.iloc[:, which_feature2shuffle] = np.random.permutation(tmp_omics.iloc[:, which_feature2shuffle])
        # Save shuffled omics data
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        random_str = random_string()
        shuffled_omics_path = os.path.join(result_dir, f'shuffled_omics_{which_omics2shuffle}_{which_feature2shuffle}_{time_str}_rs{random_str}.csv')
        tmp_omics.to_csv(shuffled_omics_path)

        shuffled_omics_paths = omics_paths.copy()
        shuffled_omics_paths[which_omics2shuffle] = shuffled_omics_path
        return shuffled_omics_paths, shuffled_omics_path

    def rank_features(
            self,
            omics_paths: list[str],
            pheno_path: str,
            trait_name: str,
            n_label_class: int,
            result_dir: str,
            random_states: list[int],
        ):
        """
        Rank features by their importance in predicting phenotypes using a trained DEM model.
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # Get original prediction loss
        data_module_orig = DEMLTNDataModule(
            batch_size=self.batch_size,
            trait_name=trait_name,
            n_label_classes=n_label_class,
            path_label_tst=pheno_path,
            paths_omics_tst=omics_paths,
        )
        data_module_orig.setup()
        trainer_orig = ltn.Trainer(default_root_dir=result_dir, logger=False)
        losses_orig = trainer_orig.test(self.dem_model, datamodule=data_module_orig)
        loss_orig = np.mean(np.concatenate(losses_orig, axis=0))

        # Shuffle features and get shuffled prediction loss
        importance_scores = []
        feat_names_all = []
        for i in range(len(omics_paths)):
            # n_features = pd.read_csv(omics_paths[i], index_col=0).shape[1]
            feat_names = pd.read_csv(omics_paths[i], index_col=0).columns
            feat_names_all.append(feat_names)
            n_features = len(feat_names)
            for j in range(n_features):
                importance_scores_tmp = []
                for random_state in random_states:
                    shuffled_omics_paths, shuffled_omics_path = self.shuffle_features(omics_paths, result_dir, i, j, random_state)
                    data_module_shuffled = DEMLTNDataModule(
                        batch_size=self.batch_size,
                        trait_name=trait_name,
                        n_label_classes=n_label_class,
                        path_label_tst=pheno_path,
                        paths_omics_tst=shuffled_omics_paths,
                    )
                    data_module_shuffled.setup()
                    os.remove(shuffled_omics_path)
                    trainer_shuffled = ltn.Trainer(default_root_dir=result_dir, logger=False)
                    losses_shuffled = trainer_shuffled.test(self.dem_model, datamodule=data_module_shuffled)
                    loss_shuffled = np.mean(np.concatenate(losses_shuffled, axis=0))
                    importance_scores_tmp.append(loss_orig - loss_shuffled)

                importance_scores.append(np.mean(importance_scores_tmp))

        # Rank features by their importance scores
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        random_str = random_string()
        filename_suffix = f'_{time_str}_rs{random_str}.csv'

        feat_importance = pd.DataFrame(importance_scores, index=feat_names_all, columns=['importance_score'])
        feat_importance.to_csv(os.path.join(result_dir, 'feature_importance' + filename_suffix))
        feat_importance_ranked = feat_importance.sort_values(by='importance_score', ascending=False)
        feat_importance_ranked.to_csv(os.path.join(result_dir, 'feature_importance_ranked' + filename_suffix))


def rank_feat(
        path_model_i: str,
        paths_omics_i: list[str],
        path_pheno_i: str,
        trait_name: str,
        n_label_class: int,
        result_dir: str,
        random_seeds: list[int],
        batch_size: int,
    ):
    """
    Rank features.
    """
    forranking = DEMFeatureRanking(path_model_i, batch_size)
    forranking.rank_features(paths_omics_i, path_pheno_i, trait_name, n_label_class, result_dir, random_seeds)
