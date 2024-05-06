r"""
This file contains the implementation of the DEM model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as ltn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing import cpu_count
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from .utils import one_hot_encode_phen


torch.set_float32_matmul_precision('medium')

dim_1st_linear_o_xomics = 512
dim_1st_linear_o_cat = 2048
dim_1st_linear_o_integrated = 1024


class TransformerXOmics(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super(TransformerXOmics, self).__init__()
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dim_1st_linear_o_xomics),
            nn.LayerNorm(dim_1st_linear_o_xomics),
            nn.Linear(dim_1st_linear_o_xomics, 128),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        x = self.linears(x)
        # if self.output_dim > 1:
        #     x = F.softmax(x, dim=1)
        return x, h_out


class TransformerAllOmics(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super(TransformerAllOmics, self).__init__()
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dim_1st_linear_o_cat),
            nn.LayerNorm(dim_1st_linear_o_cat),
            nn.Linear(dim_1st_linear_o_cat, 512),
            nn.Linear(512, 64),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        x = self.linears(x)
        # if self.output_dim > 1:
        #     x = F.softmax(x, dim=1)
        return x, h_out


class TransformerIntegrated(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super(TransformerIntegrated, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dim_1st_linear_o_integrated),
            nn.LayerNorm(dim_1st_linear_o_integrated),
            # nn.Mish(),
            nn.Linear(dim_1st_linear_o_integrated, 256),
            nn.Linear(256, 64),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linears(x)
        # if self.output_dim > 1:
        #     x = F.softmax(x, dim=1)
        return x


class DEMLTN(ltn.LightningModule):
    """
    DEM model.

    Args:
        `omics_dim`: list of input dimensions for each omics data.
        `n_heads`: number of heads in the multi-head attention.
        `n_encoders`: number of encoders in the transformer.
        `hidden_dim`: dimension of the feedforward network in the transformer.
        `output_dim`: number of output classes. If it is 1, the model will be a regression model. Otherwise, it should be at least 3 (3 for binary classification) for classification tasks.
        `dropout`: dropout rate in the transformer.
        `learning_rate`: learning rate for the optimizer.
    """
    def __init__(
            self,
            omics_dim: list[int],
            n_heads: int,
            n_encoders: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float,
            learning_rate: float,
        ):
        super(DEMLTN, self).__init__()
        self.save_hyperparameters()

        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.net_cat = TransformerAllOmics(n_heads, n_encoders, sum(omics_dim), hidden_dim, output_dim, dropout)
        self.nets_omics = nn.ModuleList([
            TransformerXOmics(n_heads, n_encoders, omics_dim[i], hidden_dim, output_dim, dropout)
            for i in range(len(omics_dim))
        ])

        input_dim_integrated = dim_1st_linear_o_cat + dim_1st_linear_o_xomics * len(omics_dim)

        self.net_integrated = TransformerIntegrated(n_heads, n_encoders, input_dim_integrated, hidden_dim, output_dim, dropout)

        if output_dim > 1:
            self.loss_fn = nn.CrossEntropyLoss()

            self.recall_micro = MulticlassRecall(average='micro', num_classes=output_dim)
            self.recall_macro = MulticlassRecall(average='macro', num_classes=output_dim)
            self.recall_weighted = MulticlassRecall(average='weighted', num_classes=output_dim)

            self.precision_micro = MulticlassPrecision(average='micro', num_classes=output_dim)
            self.precision_macro = MulticlassPrecision(average='macro', num_classes=output_dim)
            self.precision_weighted = MulticlassPrecision(average='weighted', num_classes=output_dim)

            self.f1_micro = MulticlassF1Score(average='micro', num_classes=output_dim)
            self.f1_macro = MulticlassF1Score(average='macro', num_classes=output_dim)
            self.f1_weighted = MulticlassF1Score(average='weighted', num_classes=output_dim)

            self.accuracy_micro = MulticlassAccuracy(average='micro', num_classes=output_dim)
            self.accuracy_macro = MulticlassAccuracy(average='macro', num_classes=output_dim)
            self.accuracy_weighted = MulticlassAccuracy(average='weighted', num_classes=output_dim)

            self.auroc_macro = MulticlassAUROC(average='macro', num_classes=output_dim)
            self.auroc_weighted = MulticlassAUROC(average='weighted', num_classes=output_dim)

        else:
            self.loss_fn = nn.MSELoss()

            self.mae = MeanAbsoluteError()
            self.rmse = MeanSquaredError()
            self.r2 = R2Score()
            self.pcc = PearsonCorrCoef()

    def forward(self, x_omics: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x_cat = torch.cat(x_omics, dim=1)
        y_hat_cat, h_cat = self.net_cat(x_cat)
        
        list_h_omics = []
        y_hat_omics = []
        for i in range(len(self.nets_omics)):
            x_omics_i, h_omics_i = self.nets_omics[i](x_omics[i])
            list_h_omics.append(h_omics_i)
            y_hat_omics.append(x_omics_i)
        
        h_omics = torch.cat(list_h_omics, dim=1)
        x_integrated = torch.cat([h_cat, h_omics], dim=1)
        y_hat_integrated = self.net_integrated(x_integrated)

        return y_hat_cat, y_hat_omics, y_hat_integrated
    
    def training_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], batch_idx):
        x_omics, y = batch
        y_hat_cat, y_hat_omics, y_hat_integrated = self(x_omics)

        if self.output_dim > 1:
            # y and y_hat_integrated are one-hot vectors, so we need to convert them to integers for calculating metrics
            y = y.argmax(dim=1)
            # y = y.softmax(dim=1)
            # y_hat_integrated = y_hat_integrated.softmax(dim=1)
            # y_hat_cat = y_hat_cat.softmax(dim=1)
            # y_hat_omics = [y_hat_omics[i].softmax(dim=1) for i in range(len(y_hat_omics))]
        
        loss_cat = self.loss_fn(y_hat_cat, y)
        loss_omics = sum([self.loss_fn(y_hat_omics[i], y) for i in range(len(y_hat_omics))]) / len(y_hat_omics)
        loss_integrated = self.loss_fn(y_hat_integrated, y)
        loss_total = (loss_cat + loss_omics + loss_integrated) / 3

        loss2log = loss_integrated
        # loss2log = loss_total

        self.log('train_loss', loss2log, sync_dist=True)
        self.log('train_loss_total_avg', loss_total, sync_dist=True)
        self.log('train_loss_cat', loss_cat, sync_dist=True)
        self.log('train_loss_omics', loss_omics, sync_dist=True)
        self.log('train_loss_integrated', loss_integrated, sync_dist=True)

        if self.output_dim > 1:
            self.log('train_f1_micro', self.f1_micro(y_hat_integrated, y), sync_dist=True)
            self.log('train_f1_macro', self.f1_macro(y_hat_integrated, y), sync_dist=True)
            self.log('train_f1_weighted', self.f1_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('train_accuracy_micro', self.accuracy_micro(y_hat_integrated, y), sync_dist=True)
            self.log('train_accuracy_macro', self.accuracy_macro(y_hat_integrated, y), sync_dist=True)
            self.log('train_accuracy_weighted', self.accuracy_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('train_precision_micro', self.precision_micro(y_hat_integrated, y), sync_dist=True)
            self.log('train_precision_macro', self.precision_macro(y_hat_integrated, y), sync_dist=True)
            self.log('train_precision_weighted', self.precision_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('train_recall_micro', self.recall_micro(y_hat_integrated, y), sync_dist=True)
            self.log('train_recall_macro', self.recall_macro(y_hat_integrated, y), sync_dist=True)
            self.log('train_recall_weighted', self.recall_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('train_auroc_macro', self.auroc_macro(y_hat_integrated, y), sync_dist=True)
            self.log('train_auroc_weighted', self.auroc_weighted(y_hat_integrated, y), sync_dist=True)
            
        else:
            self.log('train_mae', self.mae(y_hat_integrated, y), sync_dist=True)
            self.log('train_rmse', self.rmse(y_hat_integrated, y), sync_dist=True)
            self.log('train_pcc', self.pcc(y_hat_integrated, y), sync_dist=True)
            self.log('train_r2', self.r2(y_hat_integrated, y), sync_dist=True)

        return loss2log
    
    def validation_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], batch_idx):
        x_omics, y = batch
        y_hat_cat, y_hat_omics, y_hat_integrated = self(x_omics)

        if self.output_dim > 1:
            y = y.argmax(dim=1)
            # y = y.softmax(dim=1)
            # y_hat_integrated = y_hat_integrated.softmax(dim=1)
            # y_hat_cat = y_hat_cat.softmax(dim=1)
            # y_hat_omics = [y_hat_omics[i].softmax(dim=1) for i in range(len(y_hat_omics))]

        loss_cat = self.loss_fn(y_hat_cat, y)
        loss_omics = sum([self.loss_fn(y_hat_omics[i], y) for i in range(len(y_hat_omics))]) / len(y_hat_omics)
        loss_integrated = self.loss_fn(y_hat_integrated, y)
        loss_total = (loss_cat + loss_omics + loss_integrated) / 3

        loss2log = loss_integrated

        self.log('val_loss', loss2log, sync_dist=True)
        self.log('val_loss_total_avg', loss_total, sync_dist=True)
        self.log('val_loss_cat', loss_cat, sync_dist=True)
        self.log('val_loss_omics', loss_omics, sync_dist=True)
        self.log('val_loss_integrated', loss_integrated, sync_dist=True)

        if self.output_dim > 1:
            self.log('val_f1_micro', self.f1_micro(y_hat_integrated, y), sync_dist=True)
            self.log('val_f1_macro', self.f1_macro(y_hat_integrated, y), sync_dist=True)
            self.log('val_f1_weighted', self.f1_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('val_accuracy_micro', self.accuracy_micro(y_hat_integrated, y), sync_dist=True)
            self.log('val_accuracy_macro', self.accuracy_macro(y_hat_integrated, y), sync_dist=True)
            self.log('val_accuracy_weighted', self.accuracy_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('val_precision_micro', self.precision_micro(y_hat_integrated, y), sync_dist=True)
            self.log('val_precision_macro', self.precision_macro(y_hat_integrated, y), sync_dist=True)
            self.log('val_precision_weighted', self.precision_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('val_recall_micro', self.recall_micro(y_hat_integrated, y), sync_dist=True)
            self.log('val_recall_macro', self.recall_macro(y_hat_integrated, y), sync_dist=True)
            self.log('val_recall_weighted', self.recall_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('val_auroc_macro', self.auroc_macro(y_hat_integrated, y), sync_dist=True)
            self.log('val_auroc_weighted', self.auroc_weighted(y_hat_integrated, y), sync_dist=True)
        
        else:
            self.log('val_mae', self.mae(y_hat_integrated, y), sync_dist=True)
            self.log('val_rmse', self.rmse(y_hat_integrated, y), sync_dist=True)
            self.log('val_pcc', self.pcc(y_hat_integrated, y), sync_dist=True)
            self.log('val_r2', self.r2(y_hat_integrated, y), sync_dist=True)

        return loss2log
    
    def test_step(self, batch, batch_idx):
        x_omics, y = batch
        y_hat_cat, y_hat_omics, y_hat_integrated = self(x_omics)

        if self.output_dim > 1:
            y = y.argmax(dim=1)
            # y = y.softmax(dim=1)
            # y_hat_integrated = y_hat_integrated.softmax(dim=1)
            # y_hat_cat = y_hat_cat.softmax(dim=1)
            # y_hat_omics = [y_hat_omics[i].softmax(dim=1) for i in range(len(y_hat_omics))]

        loss_cat = self.loss_fn(y_hat_cat, y)
        loss_omics = sum([self.loss_fn(y_hat_omics[i], y) for i in range(len(y_hat_omics))]) / len(y_hat_omics)
        loss_integrated = self.loss_fn(y_hat_integrated, y)
        loss_total = (loss_cat + loss_omics + loss_integrated) / 3

        loss2log = loss_integrated

        self.log('test_loss', loss2log, sync_dist=True)
        self.log('test_loss_total_avg', loss_total, sync_dist=True)
        self.log('test_loss_cat', loss_cat, sync_dist=True)
        self.log('test_loss_omics', loss_omics, sync_dist=True)
        self.log('test_loss_integrated', loss_integrated, sync_dist=True)

        if self.output_dim > 1:
            self.log('test_f1_micro', self.f1_micro(y_hat_integrated, y), sync_dist=True)
            self.log('test_f1_macro', self.f1_macro(y_hat_integrated, y), sync_dist=True)
            self.log('test_f1_weighted', self.f1_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('test_accuracy_micro', self.accuracy_micro(y_hat_integrated, y), sync_dist=True)
            self.log('test_accuracy_macro', self.accuracy_macro(y_hat_integrated, y), sync_dist=True)
            self.log('test_accuracy_weighted', self.accuracy_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('test_precision_micro', self.precision_micro(y_hat_integrated, y), sync_dist=True)
            self.log('test_precision_macro', self.precision_macro(y_hat_integrated, y), sync_dist=True)
            self.log('test_precision_weighted', self.precision_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('test_recall_micro', self.recall_micro(y_hat_integrated, y), sync_dist=True)
            self.log('test_recall_macro', self.recall_macro(y_hat_integrated, y), sync_dist=True)
            self.log('test_recall_weighted', self.recall_weighted(y_hat_integrated, y), sync_dist=True)

            self.log('test_auroc_macro', self.auroc_macro(y_hat_integrated, y), sync_dist=True)
            self.log('test_auroc_weighted', self.auroc_weighted(y_hat_integrated, y), sync_dist=True)
        
        else:
            self.log('test_mae', self.mae(y_hat_integrated, y), sync_dist=True)
            self.log('test_rmse', self.rmse(y_hat_integrated, y), sync_dist=True)
            self.log('test_pcc', self.pcc(y_hat_integrated, y), sync_dist=True)
            self.log('test_r2', self.r2(y_hat_integrated, y), sync_dist=True)

        return loss2log

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x_omics = batch
        y_hat_cat, y_hat_omics, y_hat_integrated = self(x_omics)

        # if self.output_dim > 1:
        #     y_hat_integrated = y_hat_integrated.softmax(dim=1)
        
        return y_hat_integrated
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': 'val_loss',
                    }
                }
        # return optimizer


class DEMDataset(Dataset):
    def __init__(
            self,
            paths_omics: list[str],
            path_label: str | None = None,
            trait_name: str | None = None,
            output_dim: int | None = None,
        ):
        super().__init__()
        self.omics_dfs = [pd.read_csv(path, index_col=0) for path in paths_omics]
        if path_label is not None and trait_name is not None and output_dim is not None:
            label_df = pd.read_csv(path_label, index_col=0)
            self.label_np = label_df[[trait_name]].values.astype(np.float32)

            self.label_np = one_hot_encode_phen(self.label_np, output_dim)

    def __len__(self):
        return len(self.omics_dfs[0])

    def __getitem__(self, idx):
        x_omics = [df.iloc[idx, :].values.astype(np.float32) for df in self.omics_dfs]
        if hasattr(self, 'label_np'):
            y = self.label_np[idx]
            return x_omics, y.astype(np.float32)
        else:
            return x_omics


class DEMLTNDataModule(ltn.LightningDataModule):
    """
    Data module for DEM.

    Args:
        `paths_omics_trn`: list of paths to training omics data.
        `paths_omics_val`: list of paths to validation omics data.
        `path_label_trn`: path to training label data.
        `path_label_val`: path to validation label data.
        `n_label_classes`: number of label classes. If it is 1, the model will be a regression model. Otherwise, it should be at least 3 (3 for binary classification) for classification tasks.
        `trait_name`: name of the trait to be predicted.
        `batch_size`: batch size.
        `paths_omics_tst`: list of paths to test omics data.
        `path_label_tst`: path to test label data.
    """
    def __init__(
            self,
            batch_size: int = 16,
            trait_name: str | None = None,
            n_label_classes: int = 1,
            paths_omics_trn: list[str] | None = None,
            paths_omics_val: list[str] | None = None,
            path_label_trn: str | None = None,
            path_label_val: str | None = None,
            paths_omics_tst: list[str] | None = None,
            path_label_tst: str | None = None,
            paths_omics_pred: list[str] | None = None,
            n_threads: int | None = None,
        ):
        super().__init__()

        if n_threads is None:
            self.n_threads = round(cpu_count() * 0.9)
        else:
            self.n_threads = n_threads
        
        self.paths_omics_trn = paths_omics_trn
        self.paths_omics_val = paths_omics_val
        self.path_label_trn = path_label_trn
        self.path_label_val = path_label_val
        self.batch_size = batch_size
        self.paths_omics_tst = paths_omics_tst
        self.path_label_tst = path_label_tst
        self.paths_omics_pred = paths_omics_pred

        if n_label_classes == 1:
            self.output_dim = 1
        else:
            self.output_dim = n_label_classes + 1
        self.trait_name = trait_name

    def setup(self, stage=None):
        if self.trait_name is not None and self.paths_omics_trn is not None and self.paths_omics_val is not None:
            if self.path_label_trn is not None and self.path_label_val is not None:
                self.dataset_trn = DEMDataset(self.paths_omics_trn, self.path_label_trn, self.trait_name, self.output_dim)
                self.dataset_val = DEMDataset(self.paths_omics_val, self.path_label_val, self.trait_name, self.output_dim)
        if self.paths_omics_tst is not None and self.path_label_tst is not None:
            self.dataset_tst = DEMDataset(self.paths_omics_tst, self.path_label_tst, self.trait_name, self.output_dim)
        if self.paths_omics_pred is not None:
            self.dataset_pred = DEMDataset(self.paths_omics_pred)

    def train_dataloader(self):
        if hasattr(self, 'dataset_trn'):
            return DataLoader(self.dataset_trn, batch_size=self.batch_size, num_workers=self.n_threads, shuffle=True)
        else:
            return None

    def val_dataloader(self):
        if hasattr(self, 'dataset_val'):
            return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.n_threads, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, 'dataset_tst'):
            return DataLoader(self.dataset_tst, batch_size=self.batch_size, num_workers=self.n_threads, shuffle=False)
        else:
            return None
    
    def predict_dataloader(self):
        if hasattr(self, 'dataset_pred'):
            return DataLoader(self.dataset_pred, batch_size=self.batch_size, num_workers=self.n_threads, shuffle=False)
        else:
            return None


def train_dem(
        data_module: DEMLTNDataModule,
        n_heads: int,
        n_encoders: int,
        hidden_dim: int,
        learning_rate: float,
        dropout: float,
        patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        log_name: str,
        devices: list[int] | str | int = 'auto',
    ):
    """
    Construct and train DEM model.
    """
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices

    omics_dims = [dfx.shape[1] for dfx in data_module.dataset_trn.omics_dfs]

    model_DEM = DEMLTN(
        omics_dim=omics_dims,
        n_heads=n_heads,
        n_encoders=n_encoders,
        hidden_dim=hidden_dim,
        output_dim=data_module.output_dim,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callback_es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=False,
    )
    callback_ckpt = ModelCheckpoint(
        monitor='val_loss',
        filename="best_model-{epoch:02d}-{val_loss:.2f}"
    )

    logger_dem = TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
    )

    trainer_dem = ltn.Trainer(
        fast_dev_run=False,
        devices=avail_dev,
        precision='16-mixed',
        logger=logger_dem,
        callbacks=[callback_es, callback_ckpt],
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        log_every_n_steps=1,
        default_root_dir=log_dir,
    )

    trainer_dem.fit(model_DEM, datamodule=data_module)

    # return the best validation loss
    return callback_ckpt.best_model_score.item()
