r"""
The implementation of the DEM model.
"""

import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning as ltn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
import biodem.constants as const


torch.set_float32_matmul_precision(const.default.float32_matmul_precision)


class Extract1Omics(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, const.hparam_candidates.linear_dims_single_omics[0][0]),
            nn.LayerNorm(const.hparam_candidates.linear_dims_single_omics[0][0]),
            nn.Linear(const.hparam_candidates.linear_dims_single_omics[0][0], const.hparam_candidates.linear_dims_single_omics[0][1]),
            nn.Mish(),
            nn.Linear(const.hparam_candidates.linear_dims_single_omics[0][1], output_dim),
        )

    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        x = self.linears(x)
        return x, h_out


class ExtractConcOmics(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, const.hparam_candidates.linear_dims_conc_omics[0][0]),
            nn.Mish(),
            nn.LayerNorm(const.hparam_candidates.linear_dims_conc_omics[0][0]),
            nn.Linear(const.hparam_candidates.linear_dims_conc_omics[0][0], const.hparam_candidates.linear_dims_conc_omics[0][1]),
            nn.Linear(const.hparam_candidates.linear_dims_conc_omics[0][1], output_dim),
        )
    
    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        x = self.linears(x)
        return x, h_out


class IntegrateExtractions(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout),
            n_encoders,
        )
        self.linears = nn.Sequential(
            nn.Linear(input_dim, const.hparam_candidates.linear_dims_integrated[0][0]),
            nn.LayerNorm(const.hparam_candidates.linear_dims_integrated[0][0]),
            nn.Linear(const.hparam_candidates.linear_dims_integrated[0][0], const.hparam_candidates.linear_dims_integrated[0][1]),
            nn.Mish(),
            nn.Linear(const.hparam_candidates.linear_dims_integrated[0][1], const.hparam_candidates.linear_dims_integrated[0][2]),
            nn.Linear(const.hparam_candidates.linear_dims_integrated[0][2], output_dim),
        )
    
    def forward(self, x):
        x = self.encoders(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linears(x)
        return x


class DEM(nn.Module):
    def __init__(
            self,
            omics_dim: list[int],
            n_heads: int,
            n_encoders: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float,
        ):
        r"""The DEM model.
        """
        super().__init__()
        self.omics_dim = omics_dim
        self.extract_conc = ExtractConcOmics(n_heads, n_encoders, sum(omics_dim), hidden_dim, output_dim, dropout)
        self.extract_each_omics = nn.ModuleList([
            Extract1Omics(n_heads, n_encoders, omics_dim[i], hidden_dim, output_dim, dropout)
            for i in range(len(omics_dim))
        ])

        integrated_input_dim = const.hparam_candidates.linear_dims_conc_omics[0][0] + const.hparam_candidates.linear_dims_single_omics[0][0] * len(omics_dim)

        self.integrate_extractions = IntegrateExtractions(n_heads, n_encoders, integrated_input_dim, hidden_dim, output_dim, dropout)

        # Initialize learnable weights for every omics in y_pred_each_omics
        self.weights_each_omics = nn.ParameterList([
            nn.Parameter(torch.ones(1) / len(omics_dim))
            for _ in range(len(omics_dim))
        ])
        # Initialize learnable weights for y_pred_conc
        self.weight_conc = nn.Parameter(torch.ones(1))
        # Initialize learnable weights for y_pred_integrated
        self.weight_integrated = nn.Parameter(torch.ones(1))
    
    def forward(self, x: list[torch.Tensor]):
        # Extract concatenated omics
        y_pred_conc, h_conc = self.extract_conc(torch.cat(x, dim=1))

        # Extract each omics
        y_pred_each_omics = []
        h_each_omics = []
        for i in range(len(self.omics_dim)):
            y_pred_omics_i, h_xomics_i = self.extract_each_omics[i](x[i])
            y_pred_each_omics.append(y_pred_omics_i)
            h_each_omics.append(h_xomics_i)
        # Calc weighted y_pred_each_omics
        y_pred_each_omics = [self.weights_each_omics[i] * y_pred_each_omics[i] for i in range(len(y_pred_each_omics))]
        # Sum weighted y_pred_each_omics
        y_pred_each_omics = torch.sum(torch.stack(y_pred_each_omics), dim=0)
        
        h_conc_each_omics = torch.cat(h_each_omics, dim=1)
        
        h_integrated = torch.cat([h_conc, h_conc_each_omics], dim=1)
        
        y_pred_integrated = self.integrate_extractions(h_integrated)

        y_pred = self.weight_conc * y_pred_conc + self.weight_integrated * y_pred_integrated + y_pred_each_omics
        
        return y_pred


class DEMLTN(ltn.LightningModule):
    def __init__(
            self,
            omics_dim: list[int],
            n_heads: int,
            n_encoders: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float,
            learning_rate: float,
            is_regression: bool,
        ):
        r"""DEM model in lightning.

        Args:
            omics_dim: list of input dimensions for each omics data.
            n_heads: number of heads in the multi-head attention.
            n_encoders: number of encoders.
            hidden_dim: dimension of the feedforward network.
            output_dim: number of output classes. If it is 1, the model will be a regression model. Otherwise, it should be at least 3 (3 for binary classification) for classification tasks.
            dropout: dropout rate.
            learning_rate: learning rate for the optimizer.
            is_regression: whether the task is a regression task or not.
        
        """
        super().__init__()
        self.save_hyperparameters()

        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.is_regression = is_regression

        self._define_metrics(output_dim, is_regression)

        self.DEM_model = DEM(
            omics_dim=omics_dim,
            n_heads=n_heads,
            n_encoders=n_encoders,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x_omics: list[torch.Tensor]):
        return self.DEM_model(x_omics)
    
    def training_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics]
        y = batch[const.dkey.litdata_label]
        y_pred = self.forward(x)
        loss = self._loss(const.title_train, y, y_pred)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics]
        y = batch[const.dkey.litdata_label]
        y_pred = self.forward(x)
        loss = self._loss(const.title_val, y, y_pred)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics]
        y = batch[const.dkey.litdata_label]
        y_pred = self.forward(x)
        loss = self._loss(const.title_test, y, y_pred)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[const.dkey.litdata_omics]
        y_pred = self.forward(x)

        y = batch[const.dkey.litdata_label]
        loss = self._loss(const.title_predict, y, y_pred)
        return y_pred, loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': const.title_val_loss,
                    }
                }
    
    def _define_metrics(self, output_dim: int, regression: bool):
        r"""Define the loss function and the metrics.
        """
        if output_dim == 1:
            self.loss_fn = nn.MSELoss()
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()
            self.pcc = PearsonCorrCoef()
        else:
            if regression:
                # Multi-label regression
                self.loss_fn = nn.MSELoss(reduction='none')
                # self.mae = MeanAbsoluteError()
                # self.r2 = R2Score()
                # self.pcc = PearsonCorrCoef()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                self.mcc = MatthewsCorrCoef(task='multiclass', num_classes=output_dim)
                self.recall_micro = MulticlassRecall(average="micro", num_classes=output_dim)
                self.recall_macro = MulticlassRecall(average="macro", num_classes=output_dim)
                self.recall_weighted = MulticlassRecall(average="weighted", num_classes=output_dim)
                #
                self.precision_micro = MulticlassPrecision(average="micro", num_classes=output_dim)
                self.precision_macro = MulticlassPrecision(average="macro", num_classes=output_dim)
                self.precision_weighted = MulticlassPrecision(average="weighted", num_classes=output_dim)
                #
                self.f1_micro = MulticlassF1Score(average="micro", num_classes=output_dim)
                self.f1_macro = MulticlassF1Score(average="macro", num_classes=output_dim)
                self.f1_weighted = MulticlassF1Score(average="weighted", num_classes=output_dim)
                #
                self.accuracy_micro = MulticlassAccuracy(average="micro", num_classes=output_dim)
                self.accuracy_macro = MulticlassAccuracy(average="macro", num_classes=output_dim)
                self.accuracy_weighted = MulticlassAccuracy(average="weighted", num_classes=output_dim)
                #
                self.auroc_macro = MulticlassAUROC(average="macro", num_classes=output_dim)
                self.auroc_weighted = MulticlassAUROC(average="weighted", num_classes=output_dim)
    
    def _loss(self, which_step: str, y: torch.Tensor, y_pred: torch.Tensor):
        #!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.output_dim > 1 and not self.is_regression:
            y = y.argmax(dim=-1)

        if self.output_dim == 1:
            loss = self.loss_fn(y_pred, y)
            if which_step == const.title_predict:
                return loss
            self.log(f"{which_step}_loss", loss, sync_dist=True)
            self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
            self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
        else:
            if self.is_regression:
                loss = self.loss_fn(y_pred, y).mean(dim=0).sum()
                if which_step == const.title_predict:
                    return loss
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                # self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
                # if y.shape[0] < 2:
                #     return loss
                # self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
                # self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
            else:
                loss = self.loss_fn(y_pred, y)
                if which_step == const.title_predict:
                    return loss
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                
                self.log(f"{which_step}_mcc", self.mcc(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_micro", self.f1_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_macro", self.f1_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_f1_weighted", self.f1_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_recall_micro", self.recall_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_recall_macro", self.recall_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_recall_weighted", self.recall_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_precision_micro", self.precision_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_precision_macro", self.precision_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_precision_weighted", self.precision_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_accuracy_micro", self.accuracy_micro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_macro", self.accuracy_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_accuracy_weighted", self.accuracy_weighted(y_pred, y), sync_dist=True)
                #
                self.log(f"{which_step}_auroc_macro", self.auroc_macro(y_pred, y), sync_dist=True)
                self.log(f"{which_step}_auroc_weighted", self.auroc_weighted(y_pred, y), sync_dist=True)
        
        return loss

