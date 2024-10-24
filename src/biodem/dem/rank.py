r"""
Feature ranking.
"""
import os
from typing import Optional, Union, List
import numpy as np
import polars as pl
from lightning import Trainer
from biodem.dem.model import DEMLTN
import biodem.constants as const
from biodem.utils.data_ncv import DEMDataModule4Uni, read_omics_names
from biodem.utils.uni import get_map_location, get_avail_nvgpu, CollectFitLog, time_string


class DEMFeatureRanking:
    def __init__(
            self,
            batch_size: int = const.default.batch_size,
            n_workers: int = const.default.n_workers,
            accelerator: str = const.default.accelerator,
            map_location: Optional[str] = None,
        ):
        r"""Feature ranking using a trained DEM model.
        
        Args:
            batch_size: Batch size for prediction.

            n_workers: Number of workers for data loading.

            accelerator: Accelerator for inference.

            map_location: Map location for loading the model.
        
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.accelerator = accelerator
        self.map_location = map_location
    
    def run_a_outer(self, ncv_litdata_dir: str, fit_log_dir: str, which_outer: int, output_path: str, random_states: list[int]):
        r"""Run feature ranking for a single outer fold in nested cross-validation.
        This function searches the best inner fold for the specified outer fold and runs feature ranking on the test set.

        Args:
            ncv_litdata_dir: Path to the directory containing the nested cross-validation litdata.

            fit_log_dir: Path to the directory containing the training logs.

            which_outer: Which outer fold to run.

            output_path: Path to the file to save the results.

            random_states: List of random states for shuffling values of each feature.
        
        """
        # Collect fit logs
        collector = CollectFitLog(fit_log_dir)
        fit_logs = collector.collect()
        log_best_inner_foreach_outer = fit_logs[const.dkey.best_inner_folds]
        
        row_x_outer = log_best_inner_foreach_outer.filter(pl.col(const.dkey.which_outer)==which_outer)
        _model_path = row_x_outer[const.dkey.ckpt_path][0]
        _which_inner = row_x_outer[const.dkey.which_inner][0]
        _litdata_dir = os.path.join(ncv_litdata_dir, f"ncv_test_{which_outer}_val_{_which_inner}", const.title_test)

        self.run(_model_path, _litdata_dir, output_path, random_states)

    def run(self, model_path: str, litdata_dir: str, output_path: str, random_states: list[int]):
        r"""Rank features.

        Args:
            model_path: Path to the model's checkpoint.
            
            litdata_dir: Path to the directory containing litdata.

            output_path: Path to the file to save the results.

            random_states: List of random states for shuffling features.
        
        """
        _filename = os.path.splitext(output_path)[0]
        dir_save_pred = _filename + "_pred"
        os.makedirs(dir_save_pred, exist_ok=True)

        # Load model
        self.load_model(model_path)

        # Load data
        self._datamodule = DEMDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)
        self._datamodule.setup()

        # Get original prediction loss
        prediction_and_loss_values = self.trainer.predict(model=self._model, datamodule=self._datamodule)
        assert prediction_and_loss_values is not None
        predictions, loss = self.prep_pred_and_loss(prediction_and_loss_values)
        print(f"\nOriginal loss: {loss:.4f}\n")
        # Write original prediction and loss
        _path_npz = os.path.join(dir_save_pred, "_original.npz")
        np.savez(_path_npz, predictions, loss)
        
        # Shuffle features and get shuffled prediction loss.
        omics_names, omics_feat_paths = read_omics_names(litdata_dir, True)
        importance_scores: List[float] = []
        feature_names: List[str] = []
        which_omics = []
        for x_om in range(len(omics_names)):
            _feat_names: List[str] = pl.read_csv(omics_feat_paths[x_om]).to_series().to_list()
            n_features = len(_feat_names)
            for x_feat in range(n_features):
                feature_names.append(_feat_names[x_feat])
                which_omics.append(omics_names[x_om])
                _loss, _pred = self.run_a_feat(x_om, x_feat, random_states)
                _score = np.mean([np.abs(_iv - loss) for _iv in _loss]).item() / loss
                print(f"\nAverage impact of {_feat_names[x_feat]} on {omics_names[x_om]} : {_score:.4f}\n")
                importance_scores.append(_score.item())

                # Write predicted labels `_pred` (List[np.ndarray]) to npz file
                _path_npz = os.path.join(dir_save_pred, f"om+{x_om}_feat+{x_feat}.npz")
                np.savez(_path_npz, _pred, _loss)

                # Save/Append omics name and feature name to a text file
                with open(os.path.join(dir_save_pred, "_log.txt"), "a") as f:
                    f.write(f"{x_om}\t{x_feat}\t{omics_names[x_om]}\t{_feat_names[x_feat]}\t{_score}\t{time_string()}\n")

        # Rank features by their importance scores
        _sortperm = np.argsort(importance_scores)[::-1].tolist()

        importance_scores = [importance_scores[i] for i in _sortperm]
        feature_names = [feature_names[i] for i in _sortperm]
        which_omics = [which_omics[i] for i in _sortperm]

        # Save feature ranking results as CSV and Parquet
        df_o = pl.DataFrame({const.dkey.omics: which_omics, const.dkey.feature: feature_names, const.dkey.feat_importance: importance_scores})
        df_o.write_csv(_filename + ".csv")
        df_o.write_parquet(_filename + ".parquet")
    
    def run_a_feat(self, which_omics: Union[int, str], which_feature: int, random_states: List[int], litdata_dir: Optional[str]=None, model_path: Optional[str]=None):
        r""" Get average loss for a single shuffled feature.

        Args:
            which_omics: Which omics to shuffle.

            which_feature: Which feature to shuffle.

            random_states: List of random states for shuffling values of the specified feature.

            litdata_dir: Path to the directory containing litdata.

            model_path: Path to the model's checkpoint.

        """
        if not hasattr(self, '_datamodule'):
            if litdata_dir is not None:
                self._datamodule = DEMDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)
            else:
                raise ValueError("Please specify litdata_dir.")
        if not hasattr(self, "trainer"):
            if model_path is not None:
                self.load_model(model_path)
            else:
                raise ValueError("Please specify model_path.")
        
        losses: List[float] = []
        predictions: List[np.ndarray] = []
        
        for random_state in random_states:
            _dataloader = self._datamodule.shuffle_a_feat(which_omics, which_feature, random_state)
            
            prediction_and_loss_shuffled = self.trainer.predict(self._model, _dataloader)
            assert prediction_and_loss_shuffled is not None
            predictions_shuffled, losses_shuffled = self.prep_pred_and_loss(prediction_and_loss_shuffled)
            print(f"\nShuffled loss: {losses_shuffled:.4f}\n")
            losses.append(losses_shuffled.item())
            predictions.append(predictions_shuffled)
        
        return losses, predictions
    
    def collect_ranks(self, log_dir: str, output_path: str, overwrite: bool = True):
        r"""Collect feature ranking results from log files of multiple outer test sets.

        Args:
            log_dir: Path to the directory containing ranking log files.

            output_path: Path to the output file.

            overwrite: Whether to overwrite the output file if it already exists.
        
        """
        # Read log files in parquet format for outer test sets
        fpaths = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".parquet")])
        dfs = {i: pl.read_parquet(fpaths[i]).rename({"importance": f"importance_{str(i+1)}"}) for i in range(len(fpaths))}

        # Merge dataframes and save shared features
        df_merged = dfs[0]
        for i in range(1, len(dfs)):
            df_merged = df_merged.join(dfs[i], on=["omics", "feature"])
        
        # Normalize importance scores for each outer test set
        np_merged_norm = df_merged.drop(["omics", "feature"]).to_numpy()
        min_vals = np_merged_norm.min(axis=0)
        max_vals = np_merged_norm.max(axis=0)
        np_merged_norm = (np_merged_norm - min_vals) / (max_vals - min_vals)
        df_merged_norm = df_merged.select(["omics", "feature"]).hstack(pl.DataFrame(np_merged_norm, schema=["test_"+str(i+1) for i in range(np_merged_norm.shape[1])]))

        # Average normalized importance scores
        np_mean = np_merged_norm.mean(axis=1)
        df_merged_norm = df_merged_norm.hstack(pl.DataFrame({"average": np_mean})).sort(["average"], descending=True)
        df_merged_norm = df_merged_norm.select(["omics", "feature", "average"]).hstack(df_merged_norm.drop(["omics", "feature", "average"]))
        
        # Write results to files
        _output_path = os.path.splitext(output_path)[0]
        _output_path_1 = _output_path + ".parquet"
        _output_path_2 = _output_path + ".csv"
        if not os.path.exists(_output_path_1) or overwrite:
            df_merged_norm.write_parquet(_output_path_1)
            df_merged_norm.write_csv(_output_path_2)
        
    def load_model(self, model_path: str):
        r"""Load a model's checkpoint to specified device and define a trainer.

        Args:
            model_path: Path to the model's checkpoint.
        
        """
        self._model = DEMLTN.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=get_map_location(self.map_location),
        )
        self._model.eval()
        self._model.freeze()
        self.available_devices = get_avail_nvgpu()
        self.trainer = Trainer(accelerator=self.accelerator, devices=self.available_devices, default_root_dir=None, logger=False)

    def prep_pred_and_loss(self, _pred):
        r"""Prepare the output of "predict" step.
        """
        _predicted_each_batch = [np.array(i[0]) for i in _pred]
        _predicted = np.concatenate(_predicted_each_batch, axis=0)
        print(f"Shape of predicted: {_predicted.shape}")
        
        # Get actual batch sizes (The last batch may be smaller than others)
        _batch_sizes = [len(i) for i in _predicted_each_batch]

        _loss_each_batch = np.concatenate([np.array(i[1], ndmin=1) for i in _pred])

        # Weight the loss by batch size
        _loss = np.average(_loss_each_batch, weights=_batch_sizes)

        return _predicted, _loss
