r"""
This code aims to reduce the dimensionality of SNPs, assumming that SNPs are located in gene-specific regions.
The SNP-gene relation is pre-defined and stored in a list[int], where each element is the number of SNPs in a gene.
The input is the SNPs, and the first layer of the network is a sparse linear layer that maps the SNPs to a low-dimensional space, which features are genomic regions.
The next layers of the network are dense layers, which are trained to predict the phenotype based on the low-dimensional features.
"""

import os
import torch
import torch.nn as nn
import lightning as ltn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from .utils import one_hot_encode_phen, one_hot_encode_snp_matrix, read_into_trnval, read_into_test, read_processed_data


torch.set_float32_matmul_precision('medium')


def idx_convert(indices:list[int], len_one_hot_vec:int=10) -> list[int]:
    """
    Convert the indices to the corresponding indices in the one-hot vector.
    """
    converted_indices = [(i * len_one_hot_vec + nx) for nx in range(len_one_hot_vec) for i in indices]
    return sorted(converted_indices)


class SNPReductionNet(ltn.LightningModule):
    """
    A PyTorch Lightning module for SNP reduction and phenotype prediction.
    """
    def __init__(
            self,
            output_dim: int,
            genes_snps: list[list[int]],
            len_one_hot_vec: int,
            dense_layers_hidden_dims: list[int],
            learning_rate: float,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        num_genes = len(genes_snps)
        self.num_genes = num_genes
        
        self.sparse_layers = nn.ModuleList()
        # Define the sparse linear layers that maps SNPs to gene-specific regions
        for i_gene in range(num_genes):
            self.sparse_layers.append(nn.Linear(len(genes_snps[i_gene]) * len_one_hot_vec, 1, bias=False))
        
        indices_snp = []
        for i_gene in range(num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_one_hot_vec))
        self.indices_snp = indices_snp

        # Define the dense layers for predicting the phenotype
        self.dense_layers = nn.ModuleList()
        
        # Apply LayerNorm to the input features.
        self.dense_layers.append(nn.LayerNorm(num_genes))
        
        # First dense layer takes the gene-specific features as input.
        self.dense_layers.append(nn.Linear(num_genes, dense_layers_hidden_dims[0]))
        for i_dim in range(len(dense_layers_hidden_dims) - 1):
            self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[i_dim], dense_layers_hidden_dims[i_dim + 1]))
            # self.dense_layers.append(nn.Sigmoid())
            self.dense_layers.append(nn.Dropout(p=0.1))
        self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[-1], output_dim))
        # if output_dim > 1:
        #     self.dense_layers.append(nn.Softmax(dim=1))

        # Define the loss function and the metrics
        if output_dim == 1:
            self.loss_fn = nn.MSELoss()

            self.mae = MeanAbsoluteError()
            self.rmse = MeanSquaredError()
            self.r2 = R2Score()
            self.pcc = PearsonCorrCoef()

        else:
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
        
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map SNPs to gene-specific regions
        gene_features: list[torch.Tensor] = []
        for i_gene in range(self.num_genes):
            gene_features.append(self.sparse_layers[i_gene](x[:, self.indices_snp[i_gene]].float()))
        
        genes = torch.cat(gene_features, dim=1)
        
        # Predict phenotype based on the low-dimensional features
        for layer in self.dense_layers:
            genes = layer(genes)
        
        # Return predicted phenotype(s)
        return genes#.type(torch.float32)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        if self.output_dim > 1:
            # y and y_pred are one-hot vectors, so we need to convert them to integers for calculating metrics
            y = y.argmax(dim=1)
            # y_pred = y_pred.softmax(dim=1)
            # y = y.softmax(dim=1)
        
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, sync_dist=True)

        if self.output_dim == 1:
            self.log('train_mae', self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log('train_rmse', self.rmse(y_pred, y), sync_dist=True)
            self.log('train_pcc', self.pcc(y_pred, y), sync_dist=True)
            self.log('train_r2', self.r2(y_pred, y), sync_dist=True)

        else:
            self.log('train_f1_micro', self.f1_micro(y_pred, y), sync_dist=True)
            self.log('train_f1_macro', self.f1_macro(y_pred, y), sync_dist=True)
            self.log('train_f1_weighted', self.f1_weighted(y_pred, y), sync_dist=True)

            self.log('train_recall_micro', self.recall_micro(y_pred, y), sync_dist=True)
            self.log('train_recall_macro', self.recall_macro(y_pred, y), sync_dist=True)
            self.log('train_recall_weighted', self.recall_weighted(y_pred, y), sync_dist=True)

            self.log('train_precision_micro', self.precision_micro(y_pred, y), sync_dist=True)
            self.log('train_precision_macro', self.precision_macro(y_pred, y), sync_dist=True)
            self.log('train_precision_weighted', self.precision_weighted(y_pred, y), sync_dist=True)

            self.log('train_accuracy_micro', self.accuracy_micro(y_pred, y), sync_dist=True)
            self.log('train_accuracy_macro', self.accuracy_macro(y_pred, y), sync_dist=True)
            self.log('train_accuracy_weighted', self.accuracy_weighted(y_pred, y), sync_dist=True)

            self.log('train_auroc_macro', self.auroc_macro(y_pred, y), sync_dist=True)
            self.log('train_auroc_weighted', self.auroc_weighted(y_pred, y), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        if self.output_dim > 1:
            y = y.argmax(dim=1)
            # y_pred = y_pred.softmax(dim=1)
            # y = y.softmax(dim=1)

        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, sync_dist=True)

        if self.output_dim == 1:
            self.log('val_mae', self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log('val_rmse', self.rmse(y_pred, y), sync_dist=True)
            self.log('val_pcc', self.pcc(y_pred, y), sync_dist=True)
            self.log('val_r2', self.r2(y_pred, y), sync_dist=True)

        else:
            self.log('val_f1_micro', self.f1_micro(y_pred, y), sync_dist=True)
            self.log('val_f1_macro', self.f1_macro(y_pred, y), sync_dist=True)
            self.log('val_f1_weighted', self.f1_weighted(y_pred, y), sync_dist=True)

            self.log('val_recall_micro', self.recall_micro(y_pred, y), sync_dist=True)
            self.log('val_recall_macro', self.recall_macro(y_pred, y), sync_dist=True)
            self.log('val_recall_weighted', self.recall_weighted(y_pred, y), sync_dist=True)

            self.log('val_precision_micro', self.precision_micro(y_pred, y), sync_dist=True)
            self.log('val_precision_macro', self.precision_macro(y_pred, y), sync_dist=True)
            self.log('val_precision_weighted', self.precision_weighted(y_pred, y), sync_dist=True)

            self.log('val_accuracy_micro', self.accuracy_micro(y_pred, y), sync_dist=True)
            self.log('val_accuracy_macro', self.accuracy_macro(y_pred, y), sync_dist=True)
            self.log('val_accuracy_weighted', self.accuracy_weighted(y_pred, y), sync_dist=True)

            self.log('val_auroc_macro', self.auroc_macro(y_pred, y), sync_dist=True)
            self.log('val_auroc_weighted', self.auroc_weighted(y_pred, y), sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        if self.output_dim > 1:
            y = y.argmax(dim=1)
            # y_pred = y_pred.softmax(dim=1)
            # y = y.softmax(dim=1)

        loss = self.loss_fn(y_pred, y)
        self.log('test_loss', loss, sync_dist=True)

        if self.output_dim == 1:
            self.log('test_mae', self.mae(y_pred, y), sync_dist=True)
            if y.shape[0] < 2:
                return loss
            self.log('test_rmse', self.rmse(y_pred, y), sync_dist=True)
            self.log('test_pcc', self.pcc(y_pred, y), sync_dist=True)
            self.log('test_r2', self.r2(y_pred, y), sync_dist=True)

        else:
            self.log('test_f1_micro', self.f1_micro(y_pred, y), sync_dist=True)
            self.log('test_f1_macro', self.f1_macro(y_pred, y), sync_dist=True)
            self.log('test_f1_weighted', self.f1_weighted(y_pred, y), sync_dist=True)

            self.log('test_recall_micro', self.recall_micro(y_pred, y), sync_dist=True)
            self.log('test_recall_macro', self.recall_macro(y_pred, y), sync_dist=True)
            self.log('test_recall_weighted', self.recall_weighted(y_pred, y), sync_dist=True)

            self.log('test_precision_micro', self.precision_micro(y_pred, y), sync_dist=True)
            self.log('test_precision_macro', self.precision_macro(y_pred, y), sync_dist=True)
            self.log('test_precision_weighted', self.precision_weighted(y_pred, y), sync_dist=True)

            self.log('test_accuracy_micro', self.accuracy_micro(y_pred, y), sync_dist=True)
            self.log('test_accuracy_macro', self.accuracy_macro(y_pred, y), sync_dist=True)
            self.log('test_accuracy_weighted', self.accuracy_weighted(y_pred, y), sync_dist=True)

            self.log('test_auroc_macro', self.auroc_macro(y_pred, y), sync_dist=True)
            self.log('test_auroc_weighted', self.auroc_weighted(y_pred, y), sync_dist=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        y_pred = self(x)
        # if self.output_dim > 1:
        #     y_pred = y_pred.softmax(dim=1)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SNP2Gene(ltn.LightningModule):
    """
    Transform SNPs to gene-specific regions using a pre-trained model.
    """
    def __init__(
            self,
            path_pretrained_model: str,
            genes_snps: list[list[int]],
            len_one_hot_vec: int,
            map_location: str | None = None,
        ):
        super().__init__()
        self.num_genes = len(genes_snps)

        indices_snp = []
        for i_gene in range(self.num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_one_hot_vec))
        self.indices_snp = indices_snp

        # Load the pre-trained model
        if map_location is None:
            if device_count() > 0:
                which_dev = find_usable_cuda_devices(1)
                if len(which_dev) == 0:
                    map_location = 'cpu'
                else:
                    map_location = f'cuda:{which_dev[0]}'
            else:
                map_location = 'cpu'

        pretrained_model = SNPReductionNet.load_from_checkpoint(
            checkpoint_path = path_pretrained_model,
            map_location = map_location,
            # hparams_file = os.path.join(os.path.dirname(os.path.dirname(path_pretrained_model)), 'hparams.yaml'),
        )
        pretrained_model.eval()
        pretrained_model.freeze()
        
        # Extract the sparse layer
        self.sparse_layer = list(pretrained_model.children())[0]
        
        # Freeze the sparse layer
        self.sparse_layer.requires_grad_(False)

    def forward(self, x):
        # Map SNPs to gene-specific regions
        gene_features: list[torch.Tensor] = []
        for i_gene in range(self.num_genes):
            gene_features.append(self.sparse_layer[i_gene](x[:, self.indices_snp[i_gene]].float()))
        genes = torch.cat(gene_features, dim=1)
        return genes
    
    # def test_step(self, batch, batch_idx) -> torch.Tensor:
    #     x, y = batch
    #     genes = self(x)
    #     return genes

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch)


class SNPDataset(Dataset):
    """
    A PyTorch dataset for SNP data and phenotype data.
    - `phenotype_data` is a numpy array of phenotype data for single-trait or multi-trait analysis.
    - `snp_matrix` is a numpy array of SNP data.
        - Each element is a int that needs to be converted to a one-hot vector.
    - `len_one_hot_vec` is the length of the one-hot vector for each SNP.
        - Default is 10, which means 10 genotypes.
        - If all elements of the vector are 0, it means the SNP is missing.
    - `output_dim` is the number of output dimensions for the phenotype category.
        - Default is 1 for REGRESSION tasks.
        - For CLASSIFICATION tasks, `output_dim` should be larger than 1.
    """
    def __init__(
            self,
            snp_matrix: np.ndarray,
            phenotype_data: np.ndarray | None = None,
            len_one_hot_vec: int = 10,
            output_dim: int = 1,
        ):
        super().__init__()
        self.snp_data = one_hot_encode_snp_matrix(snp_matrix, len_one_hot_vec)
        if phenotype_data is not None:
            self.phenotype_data = one_hot_encode_phen(phenotype_data, output_dim)

    def __len__(self):
        return len(self.snp_data)

    def __getitem__(self, idx):
        x = self.snp_data[idx]
        if hasattr(self, 'phenotype_data'):
            y = self.phenotype_data[idx]
            return x.astype(np.float32), y.astype(np.float32)
        else:
            return x.astype(np.float32)


class SNPDataModule(ltn.LightningDataModule):
    def __init__(
            self,
            snp_matrix: np.ndarray,
            sample_ids_in_mat: list[str],
            path_csv_pheno_trn: str | None = None,
            path_csv_pheno_val: str | None = None,
            path_csv_pheno_tst: str | None = None,
            batch_size: int = 16,
            len_one_hot_vec: int = 10,
            which_trait: str | None = None,
            n_pheno_categories: int = 1,
            n_threads: int | None = None,
        ):
        super().__init__()
        if n_threads is None:
            self.n_threads = round(cpu_count() * 0.9)
        else:
            self.n_threads = n_threads
        
        self.batch_size = batch_size
        self.len_one_hot_vec = len_one_hot_vec

        if path_csv_pheno_trn is not None and path_csv_pheno_val is not None:
            self.phenotypes_trn, self.phenotypes_val, self.snp_matrix_trn, self.snp_matrix_val = read_into_trnval(
                snp_matrix,
                sample_ids_in_mat,
                path_csv_pheno_trn,
                path_csv_pheno_val,
                which_trait,
            )
        if path_csv_pheno_tst is not None:
            self.phenotypes_tst, self.snp_matrix_tst, self.sample_ids_tst = read_into_test(
                snp_matrix,
                sample_ids_in_mat,
                path_csv_pheno_tst,
                which_trait,
            )

            self.snp_matrix_pred = self.snp_matrix_tst
            self.sample_ids_pred = self.sample_ids_tst
        
        if path_csv_pheno_trn is None and path_csv_pheno_val is None and path_csv_pheno_tst is None:
            
            # This is for prediction
            self.snp_matrix_pred = snp_matrix
            self.sample_ids_pred = sample_ids_in_mat

        if n_pheno_categories == 1:
            self.output_dim = 1
        else:
            self.output_dim = n_pheno_categories + 1

    def setup(self, stage = None):
        if hasattr(self, 'phenotypes_trn') and hasattr(self, 'phenotypes_val'):
            self.train_dataset = SNPDataset(
                self.snp_matrix_trn,
                self.phenotypes_trn,
                self.len_one_hot_vec,
                self.output_dim,
            )
            self.val_dataset = SNPDataset(
                self.snp_matrix_val,
                self.phenotypes_val,
                self.len_one_hot_vec,
                self.output_dim,
            )
        if hasattr(self, 'phenotypes_tst'):
            self.test_dataset = SNPDataset(
                self.snp_matrix_tst,
                self.phenotypes_tst,
                self.len_one_hot_vec,
                self.output_dim,
            )
        if hasattr(self,'snp_matrix_pred'):
            self.pred_dataset = SNPDataset(
                snp_matrix=self.snp_matrix_pred,
                len_one_hot_vec=self.len_one_hot_vec,
            )

    def train_dataloader(self):
        if hasattr(self, 'train_dataset'):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.n_threads,
                pin_memory=True,
                shuffle=True,
            )
        else:
            return None

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.n_threads,
                pin_memory=True,
            )
        else:
            return None
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.n_threads,
                pin_memory=True,
            )
        else:
            return None
    
    def predict_dataloader(self):
        if hasattr(self, 'pred_dataset'):
            return DataLoader(
                self.pred_dataset,
                batch_size=self.batch_size,
                num_workers=self.n_threads,
                pin_memory=True,
            )
        else:
            return None


def train_snp2gene(
        data_module: SNPDataModule,
        genes_snps: list[list[int]],
        dense_layers_hidden_dims: list[int],
        es_patience: int,
        learning_rate: float,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        log_name: str | None = None,
        devices: list[int] | str | int = 'auto',
        accelerator: str = 'auto',
    ):
    """
    Train a SNP-gene relation-based sparse neural network for SNP2Gene transformation.
    
    Note:
    - `n_categories` is not the number of traits, but the categorical number of a trait.
    - For classification tasks, `n_categories` should be larger than 1.
    """
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices

    model = SNPReductionNet(
        output_dim=data_module.output_dim,
        genes_snps=genes_snps,
        len_one_hot_vec=data_module.len_one_hot_vec,
        dense_layers_hidden_dims=dense_layers_hidden_dims,
        learning_rate=learning_rate,
    )

    callback_es = EarlyStopping(
        monitor='val_loss',
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
    )

    logger_tr = TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
    )

    trainer = ltn.Trainer(
        fast_dev_run=False,
        logger=logger_tr,
        log_every_n_steps=1,
        precision='16-mixed',
        devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
    )

    trainer.fit(model=model, datamodule=data_module)

    return callback_ckpt.best_model_score.item()


def snp_to_gene(
        path_pretrained_model: str,
        path_h5_processed: str,
        path_json_genes_snps: str | None = None,
        path_csv_pheno_test: str | None = None,
        dir4predictions: str = os.getcwd(),
        len_one_hot_vec: int = 10,
        batch_size: int = 32,
        which_trait: str | None = None,
        accelerator: str = 'auto',
    ):
    """
    Run the SNP2Gene model.
    - If `path_csv_pheno_test` is `None`, it means we are doing prediction.
    """
    data_dict = read_processed_data(path_h5_processed, path_json_genes_snps)

    snp2gene_data_module = SNPDataModule(
        snp_matrix=data_dict['snp_matrix'],
        sample_ids_in_mat=data_dict['sample_ids'],
        path_csv_pheno_tst=path_csv_pheno_test,
        batch_size=batch_size,
        len_one_hot_vec=len_one_hot_vec,
        which_trait=which_trait,
    )

    model4gene = SNP2Gene(
        path_pretrained_model=path_pretrained_model,
        genes_snps=data_dict['genes_snps'],
        len_one_hot_vec=len_one_hot_vec,
    )

    if device_count() > 0:
        avail_dev = find_usable_cuda_devices(1)
    else:
        avail_dev = 1

    trainer = ltn.Trainer(accelerator=accelerator, devices=avail_dev, default_root_dir=dir4predictions, logger=False)
    
    # if path_csv_pheno_test is None:
    #     predictions = trainer.predict(model4gene, snp2gene_data_module)
    # else:
    #     predictions = trainer.test(model4gene, snp2gene_data_module)
    predictions = trainer.predict(model4gene, snp2gene_data_module)

    pred_array = np.concatenate(predictions, axis=0)

    # Rename column names to gene_ids
    pred_df = pd.DataFrame(pred_array, columns=data_dict['gene_ids'])

    # Rename index to sample_ids
    # if path_csv_pheno_test is None:
    #     pred_df.index = snp2gene_data_module.sample_ids_pred
    # else:
    #     pred_df.index = snp2gene_data_module.sample_ids_tst
    pred_df.index = snp2gene_data_module.sample_ids_pred

    # Evaluate the predictions of test if `path_csv_pheno_test` is not `None`

    if path_csv_pheno_test is not None:

        # Load the pre-trained model
        
        pretrained_model = SNPReductionNet.load_from_checkpoint(
            checkpoint_path = path_pretrained_model,
            hparams_file = os.path.join(os.path.dirname(os.path.dirname(path_pretrained_model)), 'hparams.yaml'),
            # devices = avail_dev,
        )
        pretrained_model.eval()
        pretrained_model.freeze()

        log_name = '-'.join(path_csv_pheno_test.split(os.sep)[-2:]).replace('.csv', '')
        log_dir = os.path.join(os.path.dirname(dir4predictions), 's2g_eval')
        logger_te = TensorBoardLogger(save_dir=log_dir, name=log_name)

        if device_count() > 0:
            avail_dev = find_usable_cuda_devices(1)
        else:
            avail_dev = 1

        evaluator = ltn.Trainer(devices=avail_dev, default_root_dir=dir4predictions, logger=logger_te)
        evaluator.test(
            pretrained_model,
            snp2gene_data_module,
        )

    return pred_df
