r"""
This code aims to reduce the dimensionality of SNPs, assumming that SNPs are located in genome regions.
The SNP-genome block relation is pre-defined.
The input is the one-hot SNPs, and the first layer of the network is a sparse linear layer that maps the SNPs to a low-dimensional space with features representing genome regions.
The following layers are dense layers, that could be trained to predict phenotypes based on the low-dimensional features.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning as ltn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, R2Score, PearsonCorrCoef
from biodem.utils.uni import idx_convert, get_map_location
import biodem.constants as const


torch.set_float32_matmul_precision(const.default.float32_matmul_precision)


class SNPReductionNetModel(nn.Module):
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            snp_onehot_bits: int,
            dense_layer_dims: List[int],
        ):
        super().__init__()
        self.n_blocks = len(blocks_gt)
        
        # Define the sparse linear layers that maps SNPs to genome blocks
        self.sparse_layers = nn.ModuleList([
            nn.Linear(len(block) * snp_onehot_bits, 1, bias=False) for block in blocks_gt
        ])

        self.indices_gt = [idx_convert(block, snp_onehot_bits) for block in blocks_gt]
        
        # Define the dense layers for predicting the phenotype
        # + Apply LayerNorm to the input features.
        # + First dense layer takes the genome blocks' features as input.
        dense_layers = [
            nn.LayerNorm(self.n_blocks),
            nn.Linear(self.n_blocks, dense_layer_dims[0]),
        ]
        for i in range(len(dense_layer_dims) - 1):
            dense_layers.extend([
                nn.Linear(dense_layer_dims[i], dense_layer_dims[i + 1]),
                nn.Sigmoid(),
                # nn.Dropout(p=0.1),
            ])
        dense_layers.append(nn.Linear(dense_layer_dims[-1], output_dim))
        self.dense_layers = nn.Sequential(*dense_layers)
    
    def forward(self, x):
        # Map SNPs to genome features
        # Predict phenotype based on the low-dimensional features
        g_features = [layer(x[:, indices]) for layer, indices in zip(self.sparse_layers, self.indices_gt)]
        gblocks = torch.cat(g_features, dim=1)
        return self.dense_layers(gblocks)


class SNPReductionNet(ltn.LightningModule):
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            snp_onehot_bits: int,
            dense_layer_dims: List[int],
            learning_rate: float,
            regression: bool,
        ):
        r"""A PyTorch Lightning module for SNP reduction and phenotype prediction.
        """
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.regression = regression
        
        self._define_metrics()

        self.model = SNPReductionNetModel(
            output_dim=output_dim,
            blocks_gt=blocks_gt,
            snp_onehot_bits=snp_onehot_bits,
            dense_layer_dims=dense_layer_dims,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics][0]
        y = batch[const.dkey.litdata_label]

        y_pred = self.forward(x)
        loss = self._loss(y_pred, y, const.title_train)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics][0]
        y = batch[const.dkey.litdata_label]
        
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y, const.title_val)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[const.dkey.litdata_omics][0]
        y = batch[const.dkey.litdata_label]
        
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y, const.title_test)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[const.dkey.litdata_omics][0]
        y_pred = self.forward(x)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _define_metrics(self):
        """
        Define the loss function and the metrics.
        """
        if self.output_dim == 1:
            self.loss_fn = nn.MSELoss()
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()
            self.pcc = PearsonCorrCoef()
        else:
            if self.regression:
                # Multi-label regression
                self.loss_fn = nn.MSELoss(reduction='none')
                # self.mae = MeanAbsoluteError()
                # self.r2 = R2Score()
                # self.pcc = PearsonCorrCoef()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                self.recall_micro = MulticlassRecall(average="micro", num_classes=self.output_dim)
                self.recall_macro = MulticlassRecall(average="macro", num_classes=self.output_dim)
                self.recall_weighted = MulticlassRecall(average="weighted", num_classes=self.output_dim)
                #
                self.precision_micro = MulticlassPrecision(average="micro", num_classes=self.output_dim)
                self.precision_macro = MulticlassPrecision(average="macro", num_classes=self.output_dim)
                self.precision_weighted = MulticlassPrecision(average="weighted", num_classes=self.output_dim)
                #
                self.f1_micro = MulticlassF1Score(average="micro", num_classes=self.output_dim)
                self.f1_macro = MulticlassF1Score(average="macro", num_classes=self.output_dim)
                self.f1_weighted = MulticlassF1Score(average="weighted", num_classes=self.output_dim)
                #
                self.accuracy_micro = MulticlassAccuracy(average="micro", num_classes=self.output_dim)
                self.accuracy_macro = MulticlassAccuracy(average="macro", num_classes=self.output_dim)
                self.accuracy_weighted = MulticlassAccuracy(average="weighted", num_classes=self.output_dim)
                #
                self.auroc_macro = MulticlassAUROC(average="macro", num_classes=self.output_dim)
                self.auroc_weighted = MulticlassAUROC(average="weighted", num_classes=self.output_dim)
    
    def _loss(self, y_pred: torch.Tensor, y: torch.Tensor, which_step: str):
        #!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.output_dim > 1 and not self.regression:
            y = y.argmax(dim=-1)

        if self.output_dim == 1:
            loss = self.loss_fn(y_pred, y)
            self.log(f"{which_step}_loss", loss, sync_dist=True)
            self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
            self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
        else:
            if self.regression:
                loss = self.loss_fn(y_pred, y)
                loss = loss.mean(dim=0)
                loss = loss.sum()
                self.log(f"{which_step}_loss", loss, sync_dist=True)
                # self.log(f"{which_step}_mae", self.mae(y_pred, y), sync_dist=True)
                # if y.shape[0] < 2:
                #     return loss
                # self.log(f"{which_step}_pcc", self.pcc(y_pred, y), sync_dist=True)
                # self.log(f"{which_step}_r2", self.r2(y_pred, y), sync_dist=True)
            else:
                loss = self.loss_fn(y_pred, y)
                self.log(f"{which_step}_loss", loss, sync_dist=True)

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


class SNP2GB(ltn.LightningModule):
    def __init__(
            self,
            path_pretrained_model: str,
            blocks_gt: List[List[int]],
            snp_onehot_bits: int,
            map_location: Optional[str] = None,
        ):
        r"""Transform SNPs to genome blocks using a pre-trained model.
        """
        super().__init__()
        self.n_blocks = len(blocks_gt)

        self.indices_gt = [idx_convert(block, snp_onehot_bits) for block in blocks_gt]

        # Load the pre-trained model
        pretrained_model = SNPReductionNet.load_from_checkpoint(
            checkpoint_path = path_pretrained_model,
            map_location = get_map_location(map_location),
        )
        pretrained_model.eval()
        pretrained_model.freeze()
        
        # Extract the sparse layer
        self.sparse_layers = pretrained_model.model.sparse_layers
        
        # Freeze the sparse layer
        self.sparse_layers.requires_grad_(False)

    def forward(self, x):
        # Map SNPs to genome blocks
        g_features = [layer(x[:, indices]) for layer, indices in zip(self.sparse_layers, self.indices_gt)]
        gblocks = torch.cat(g_features, dim=1)
        return gblocks
    
    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        return self.forward(batch[const.dkey.litdata_omics][0])

